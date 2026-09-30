"""
backend/website_pipeline.py — answer WEBSITE questions with the code pipeline (burbank-code-v1)
instead of Azure on-your-data. Gated by CODE_PIPELINE_ENABLED in app.py; routing is unchanged,
only the website branch calls this.

Hybrid (BM25 + vector kNN) retrieval + the Azure semantic ranker + an in-depth grounded prompt.
Returns (answer, context) where context = {"citations": [...]} shaped exactly like on-your-data,
so the frontend renders citations identically. The answer cites sources as [doc1], [doc2], ...
matching the frontend's citation parser.

Reuses the app's Azure OpenAI client for both the query embedding and the answer; reads the
search service + code index from env (already present in the app settings).
"""

import asyncio
import logging
import os
import re
import time
from datetime import date

import aiohttp
try:
    from opentelemetry import trace          # available via azure-monitor-opentelemetry
    _tracer = trace.get_tracer("asksaira.website")
except Exception:                             # otel not installed -> spans become no-ops
    _tracer = None

try:
    from backend import observability   # best-effort query tracing; None if unavailable
except Exception:
    observability = None

# Generic signals that the resident wants the ordinance TEXT (not a service/how-to answer). When
# absent, municipal-code chunks (source_category=="code") are demoted below the city's own pages so
# an ordinance that merely shares vocabulary can't outrank the page that actually answers.
_CODE_INTENT_RE = re.compile(
    r"\b(codes?|ordinances?|municipal code|statutes?|zoning|setbacks?|"
    r"what (?:does|do) the (?:code|law|ordinance))\b", re.I)


def _is_code_query(query):
    return bool(_CODE_INTENT_RE.search(query or ""))


CODE_SLOTS = 2   # non-code queries: at most this many municipal-code chunks kept in the top-k


def _demote_code(chunks, k):
    """Non-code queries: only reorder when municipal code is FLOODING the natural top-k. If code
    already holds <= CODE_SLOTS of the top-k it earned those slots, so leave the ranking untouched
    (code that wasn't top-ranked is never promoted). If it floods, pull the city's own pages up and
    keep at most CODE_SLOTS code chunks, so an ordinance still supports the answer without dominating
    it. So 'how do I report weeds' leads with the 311 page yet still carries a little of the
    ordinance, while a question where code merely ranked 2nd is left exactly as retrieval had it."""
    top = chunks[:k]
    code_in_top = [c for c in top if c.get("source_category") == "code"]
    if len(code_in_top) <= CODE_SLOTS:
        return top
    site = [c for c in chunks if c.get("source_category") != "code"]
    kept = code_in_top[:CODE_SLOTS]
    out = site[:k - len(kept)] + kept
    if len(out) < k:                                  # few website chunks in the pool -> backfill
        seen = {c["id"] for c in out}
        out += [c for c in chunks if c["id"] not in seen][:k - len(out)]
    return out[:k]

SEARCH_SVC = os.environ.get("AZURE_SEARCH_SERVICE", "")
SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
CODE_INDEX = os.environ.get("CODE_INDEX", "burbank-code-v1")
SEM_CONFIG = os.environ.get("CODE_SEMANTIC_CONFIG", "sem")
EMBED_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_NAME") or "text-embedding-3-large"
API = "2024-07-01"

# Query expansion (website route only). The user's phrasing sometimes misses the corpus's
# vocabulary -- "who is running for city council" never surfaces the candidates page, which is a
# list of names, because "running for" doesn't match "candidates/election". This appends a few
# intent/synonym terms to the RETRIEVAL query only (never the generation question), strengthening
# the keyword side without drifting meaning. Gated + reversible per city.
QUERY_EXPANSION_ENABLED = os.environ.get("QUERY_EXPANSION_ENABLED", "0") != "0"
_EXPAND_SYSTEM = (
    "You add a few search keywords to a resident's question so a city-website search finds the "
    "right page. Output ONLY 3-8 extra terms (space-separated, no punctuation, no explanation) that "
    "name the same intent in the vocabulary a government website would use -- synonyms and the "
    "concrete nouns the answer page would carry. Do NOT restate the question; give only the ADDED "
    "terms. Examples: 'who is running for city council' -> 'candidates election ballot nominees'. "
    "'how do I get rid of a couch' -> 'bulky item pickup disposal sanitation'. If nothing useful to "
    "add, output nothing."
)


async def _expand_query(question, client, model):
    """Return `question` plus a few retrieval-only keyword terms, or the bare question on any error.
    Keeps the user's words (they drive the vector side) and adds vocabulary for the keyword side."""
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=40,
            messages=[{"role": "system", "content": _EXPAND_SYSTEM},
                      {"role": "user", "content": question}])
        terms = (resp.choices[0].message.content or "").strip()
        if terms:
            logging.info("[EXPAND] %r + %r", question, terms)
            return f"{question} {terms}"
    except Exception:
        logging.exception("query expansion failed; using the raw question")
    return question

SYSTEM = (
    "You are the City of Burbank's assistant. Give the resident a thorough, genuinely helpful "
    "answer using ONLY the numbered sources provided. Rules:\n"
    "- Use only facts present in the sources; never invent names, phone numbers, dates, or URLs.\n"
    "- Be thorough and specific: cover the full picture the sources support, relevant conditions, "
    "eligibility, requirements, steps to take, exceptions, fees, deadlines, and what to do next. "
    "Don't stop at a one-line answer when the sources contain more that would help.\n"
    "- Organize a longer answer with short paragraphs or bullet points so it's easy to read.\n"
    "- Cite the sources you use inline as [doc1], [doc2], etc. (matching the source numbers).\n"
    "- Contact and action details are the MOST important part of the answer to a resident: whenever "
    "the sources contain an email, phone number, online form, portal, or application link relevant "
    "to the question, ALWAYS include it verbatim so they can actually act. Never omit a contact or "
    "link the sources provide.\n"
    "- If the sources include a link to an online FORM the resident would need (application, "
    "request, registration, permit form, etc.), ALWAYS give that form's ACTUAL URL in the answer, "
    "not just its name, so they can open it directly.\n"
    "- When the sources give separate contacts for different audiences or cases (e.g. residential "
    "vs commercial, or by department or property type), include EACH one that applies; do not "
    "substitute one channel (like a form link) for another audience's contact.\n"
    "- When the resident asks about something related to a code violation, give the city's Code "
    "Enforcement email CodeEnf@burbankca.gov. Do not add it to unrelated answers.\n"
    "- For arrest logs, booking logs, or the police daily arrest log, INCLUDING a request for a "
    "specific day, week, month, or date range ('last week', 'yesterday', a date): these are "
    "published and updated daily on the City's arrest log page. Do NOT list or enumerate "
    "individual daily logs, do NOT generate or guess any log dates or links, and do NOT compute "
    "dates yourself. Give ONLY the arrest log page's link from the sources and tell the resident "
    "to open it and select the date they want. Do not add unrelated police links (report "
    "submission, crime pages, news articles) to this answer.\n"
    "- If the sources don't fully cover the question, answer what they do cover, say what's "
    "missing, and point to the closest relevant office or page.\n"
    "- Prefer the most recent/current information when sources differ.\n"
    "- When a source is the official page for a role or office (its title or URL names that role), "
    "treat it as authoritative for who holds it. Do not present a person named in an unrelated "
    "document (grant application, agenda, minutes) as a competing officeholder.\n"
    "- Today is {today}. Use this weekday and date exactly as given; never recompute the day "
    "of the week yourself. Sources may describe past events or expired terms. For any "
    "time-bound fact (terms of office, appointments, whether someone 'currently' holds a role, "
    "deadlines, fees), compare its dates against today: if a stated period has already ended, "
    "describe it in the past tense, not as current. Do not assume the most recently named person "
    "still holds a role."
)


async def _retrieve(vector, query, k, candidates):
    body = {
        "search": query,
        "vectorQueries": [{"kind": "vector", "vector": vector, "fields": "vector", "k": candidates}],
        "queryType": "semantic",
        "semanticConfiguration": SEM_CONFIG,
        "top": k,
        "select": "id,content,title,url,breadcrumb,page_type,content_date,source_category",
    }
    url = f"https://{SEARCH_SVC}.search.windows.net/indexes/{CODE_INDEX}/docs/search?api-version={API}"
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers={"Content-Type": "application/json", "api-key": SEARCH_KEY},
                          json=body) as r:
            data = await r.json(content_type=None)
    return data.get("value", []) or []


async def _page_chunks(page_url):
    """Every chunk of one page (its parent document), for parent-document assembly. Same Azure AI
    Search index as _retrieve, just FILTERED to the page instead of ranked."""
    body = {
        "search": "*",
        "filter": "url eq '{}'".format((page_url or "").replace("'", "''")),
        "top": 60,
        "select": "content,breadcrumb,url,source_category",
    }
    u = f"https://{SEARCH_SVC}.search.windows.net/indexes/{CODE_INDEX}/docs/search?api-version={API}"
    async with aiohttp.ClientSession() as s:
        async with s.post(u, headers={"Content-Type": "application/json", "api-key": SEARCH_KEY},
                          json=body) as r:
            data = await r.json(content_type=None)
    return data.get("value", []) or []


# --------------------------- deterministic answer verification ---------------------------
# The model regenerates URLs/contacts rather than copying them, so it can corrupt a document GUID
# (one hex digit -> a 404) or emit a contact that isn't in the sources. These checks are pure
# string work against the retrieved text (no network): repair URLs to the verbatim source URL,
# and flag contacts that don't appear in the sources.
_URL_RE = re.compile(r"https?://[^\s)\]]+")
_MDLINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_UUID_RE = re.compile(r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\(?\b\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b")


def _strip_trail(u):
    while u and u[-1] in ".,;:!?":
        u = u[:-1]
    return u


def _url_identity(u):
    """A URL's stable identity: drop the query string and any trailing UUID, so two URLs that differ
    only in a corrupted document GUID (the LLM's typical URL error) map to the same identity."""
    base = u.split("?", 1)[0]
    m = _UUID_RE.search(base)
    if m:
        base = base[: m.start()]
    return base.rstrip("/")


def _repair_urls(answer, source_text):
    """Replace every URL the model wrote with the VERBATIM URL from the sources. exact match -> keep;
    same identity (same document, corrupted GUID/token) -> swap in the source URL; no source match
    (fabricated) -> drop the link, keep the label text. Returns (fixed_answer, [(action,from,to)])."""
    src_urls = [_strip_trail(u) for u in _URL_RE.findall(source_text)]
    src_set = set(src_urls)
    by_identity = {}
    for s in src_urls:
        by_identity.setdefault(_url_identity(s), s)
    repairs = []

    def _resolve(u):
        if u in src_set:
            return u
        s = by_identity.get(_url_identity(u))
        if s:
            repairs.append(("repair", u, s))
            return s
        repairs.append(("drop", u, None))
        return None

    def _md(m):
        r = _resolve(m.group(2))
        return f"[{m.group(1)}]({r})" if r else m.group(1)
    out = _MDLINK_RE.sub(_md, answer)

    def _bare(m):
        u, trail = m.group(0), ""
        while u and u[-1] in ".,;:!?":
            trail = u[-1] + trail
            u = u[:-1]
        r = _resolve(u)
        return (r + trail) if r else trail
    out = _URL_RE.sub(_bare, out)
    return out, repairs


def _check_contacts(answer, source_text):
    """Flag emails/phones in the answer that don't appear in the sources (likely fabricated). We LOG
    rather than delete: excising a contact from mid-sentence prose can mangle it, so these surface in
    observability for review instead of being auto-removed."""
    src_emails = {e.lower() for e in _EMAIL_RE.findall(source_text)}
    src_phones = {re.sub(r"\D", "", p) for p in _PHONE_RE.findall(source_text)}
    bad_emails = sorted({e for e in _EMAIL_RE.findall(answer) if e.lower() not in src_emails})
    bad_phones = sorted({p for p in _PHONE_RE.findall(answer)
                         if re.sub(r"\D", "", p) not in src_phones})
    return {"unverified_emails": bad_emails, "unverified_phones": bad_phones}


async def answer_website_query(question, client, model, k=8, candidates=50, pool=30,
                               page_char_cap=8000, total_char_cap=32000,
                               system=None, retrieval_query=None):
    """Return (answer, context). Retrieve + demote to k hit chunks, then PARENT-DOCUMENT expand:
    reassemble each hit's whole page (its hit chunks first, then the page's other chunks) so an
    answer split across a page's chunks stays complete (e.g. a FAQ's contact line + its how-to
    step). Sources become one block per page, capped so a long page can't blow up the context."""
    t0 = time.monotonic()
    _rspan = _tracer.start_span("website.retrieve") if _tracer else None
    rq = retrieval_query or question          # zoning passes a conversation-derived retrieval query
    # Expand ONLY for the website route: retrieval_query is None means a plain website question
    # (zoning/others supply their own retrieval query and are left untouched).
    if retrieval_query is None and QUERY_EXPANSION_ENABLED:
        rq = await _expand_query(question, client, model)
    emb = await client.embeddings.create(model=EMBED_MODEL, input=[rq])
    chunks = await _retrieve(emb.data[0].embedding, rq, pool, candidates)
    hits = chunks[:k] if _is_code_query(rq) else _demote_code(chunks, k)

    # group the hit chunks by page (source_url), preserving hit-rank order
    pages, hit_by_page = [], {}
    for h in hits:
        u = h.get("url", "")
        if u not in hit_by_page:
            pages.append(u)
            hit_by_page[u] = {"breadcrumb": h.get("breadcrumb") or h.get("title") or "", "hits": []}
        hit_by_page[u]["hits"].append(h)

    # Parent-document expansion for every hit page AT ONCE (was a sequential await per page, a big
    # chunk of the latency). Each page's siblings are independent, so gather them concurrently.
    sib_results = await asyncio.gather(*[_page_chunks(u) for u in pages], return_exceptions=True)
    sib_by_page = {u: (s if not isinstance(s, Exception) else []) for u, s in zip(pages, sib_results)}

    blocks, citations, total = [], [], 0
    for u in pages:
        if total >= total_char_cap:
            break
        siblings = sib_by_page.get(u, [])                  # degrades to just the hit chunks on error
        seen, parts, used = set(), [], 0
        for c in hit_by_page[u]["hits"] + siblings:        # hit chunks first, then rest of the page
            text = (c.get("content") or "").strip()
            key = text[:200]
            if not text or key in seen:
                continue
            if parts and used + len(text) > page_char_cap:
                break
            seen.add(key); parts.append(text); used += len(text)
        page_text = "\n\n".join(parts)
        bc = hit_by_page[u]["breadcrumb"]
        blocks.append(f"[doc{len(blocks) + 1}] {bc or u}  ({u})\n{page_text}")
        citations.append({"content": page_text[:2000], "title": bc, "url": u,
                          "filepath": bc or u, "chunk_id": str(len(citations))})
        total += used

    sources = "\n\n".join(blocks)
    retrieve_ms = int((time.monotonic() - t0) * 1000)
    if _rspan:
        _rspan.end()

    _tg = time.monotonic()
    _gspan = _tracer.start_span("website.generate") if _tracer else None
    sys_text = (system or SYSTEM).format(today=date.today().strftime("%A, %B %d, %Y (%Y-%m-%d)"))
    resp = await client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": sys_text},
                  {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"}])
    generate_ms = int((time.monotonic() - _tg) * 1000)
    if _gspan:
        _gspan.end()
    logging.info("[WEBSITE TIMING] retrieve_ms=%d generate_ms=%d", retrieve_ms, generate_ms)
    answer = (resp.choices[0].message.content or "").strip()
    answer_id = resp.id                   # the completion's own unique id, reused as the message id

    # Deterministic verification: repair model-mangled URLs against the sources, flag ungrounded
    # contacts. Pure string work, no network. Verify against the sources AND the system prompt, so a
    # URL/contact the prompt itself supplies (e.g. the zoning-map link, a department email) is treated
    # as grounded and not stripped.
    grounding = sources + "\n" + sys_text
    answer, url_repairs = _repair_urls(answer, grounding)
    contact_flags = _check_contacts(answer, grounding)
    if url_repairs or contact_flags["unverified_emails"] or contact_flags["unverified_phones"]:
        logging.info("[WEBSITE VERIFY] url_repairs=%s contacts=%s", url_repairs, contact_flags)

    if observability:                     # best-effort, non-blocking retrieval trace
        observability.fire_and_forget({
            "route": "website",
            "answer_id": answer_id,        # == the assistant message id, for the feedback join
            "question": question,
            "retrieved": observability.website_retrieved(hits),
            "answer": answer,
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "retrieve_ms": retrieve_ms,
            "generate_ms": generate_ms,
            "model": model,
            "url_repairs": url_repairs,     # [(action, from, to)] — fabricated/mangled links caught
            "contact_flags": contact_flags, # ungrounded emails/phones surfaced for review
        })

    context = {"citations": citations}
    return answer, context, answer_id
