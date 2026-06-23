"""
Read-only client for the Burbank permits index (open API, no auth).

Endpoint: GET {PERMITS_API_BASE}  e.g.
    http://burbank.edgesoftinc.com:7337/solr/load_initial_burbank/query

Every permit is one record. The read-permits agent needs:
  - find_permit_type(kw) -> resolve a user's word to the exact stored type value(s)
  - count(...)           -> exact total + optional grouped breakdown
  - search(...)          -> a page of matching permits (first N + the true total)
  - get_permit(nbr)      -> a single permit's detail

Things learned by probing the index:
  - `type`, `status`, `department` are exact-match string fields -> filter as field:"value".
  - `act_nbr` and `address` are tokenized oddly, so we DON'T trust an exact field query:
    permit lookup goes through the `_text_` catch-all then post-filters on the exact act_nbr,
    and address search uses uppercased *wildcards* (e.g. address:*WALNUT*).
  - Dates are ISO-Z; we accept YYYY, YYYY-MM or YYYY-MM-DD and expand to a range.

All calls are async (aiohttp) so they don't block the Quart event loop.
"""

import calendar
import difflib
import json
import os
import re

import aiohttp

PERMITS_API_BASE = os.environ.get(
    "PERMITS_API_BASE",
    "http://burbank.edgesoftinc.com:7337/solr/load_initial_burbank/query",
)
TIMEOUT = 25

# Fields a user actually cares about, kept small so tool results stay cheap.
_SUMMARY = ["act_nbr", "type", "status", "department", "address",
            "applied_date", "valuation_calculated", "description"]
_DETAIL = _SUMMARY + ["issued_date", "final_date", "exp_date", "zone",
                      "projectnumber", "people", "amount", "paid", "balance", "city", "zip"]

# Friendly date word -> the date field it maps to.
_DATE_FIELDS = {
    "applied": "applied_date", "filed": "applied_date", "submitted": "applied_date",
    "issued": "issued_date", "approved": "issued_date",
    "final": "final_date", "finaled": "final_date", "completed": "final_date",
    "updated": "updated_date", "created": "created_date",
    "expires": "exp_date", "expiration": "exp_date",
}
# Fields we allow grouping / distinct-listing on.
_FACET_FIELDS = {"type", "status", "department", "zone", "city", "module"}

# Words ignored when resolving a permit type, so "my new permit" doesn't match on junk.
_STOP = {"the", "and", "for", "permit", "permits", "to", "of", "in", "a", "my", "new", "at", "is"}
_FUZZY_FLOOR = 0.6  # below this, a fuzzy candidate is treated as no match

# Common acronyms / synonyms that letter-matching alone can't resolve (e.g. "ADU" is
# not a substring of "Accessory Dwelling Unit"). Acronyms that ARE the initials of a
# type (CUP, TI, ...) are handled generically by the acronym tier, so they're not listed.
_ALIASES = {
    "adu": "accessory dwelling unit",
    "granny flat": "accessory dwelling unit",
    "in-law unit": "accessory dwelling unit",
    "sfr": "single-family",
    "sfd": "single-family",
}


def _initials(value):
    """First letter of each word, lowercased: 'Accessory Dwelling Unit' -> 'adu'."""
    return "".join(w[0] for w in re.findall(r"[A-Za-z]+", value)).lower()


def _has_word(text, word):
    """True if `word` appears as a whole word in `text` (so 'ti' doesn't match 'activity')."""
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def _esc(token):
    """Keep only alphanumerics so a token is safe inside a wildcard/term query."""
    return re.sub(r"[^A-Za-z0-9]", "", str(token or ""))


def _date_field(name):
    n = (name or "applied").lower().replace("_date", "").strip()
    return _DATE_FIELDS.get(n, "applied_date")


def _solr_dt(d, end=False):
    """YYYY / YYYY-MM / YYYY-MM-DD -> a datetime bound. '*' if empty."""
    if not d:
        return "*"
    d = str(d).strip()
    if re.fullmatch(r"\d{4}", d):
        return f"{d}-12-31T23:59:59Z" if end else f"{d}-01-01T00:00:00Z"
    if re.fullmatch(r"\d{4}-\d{2}", d):
        if end:
            y, m = (int(x) for x in d.split("-"))
            return f"{d}-{calendar.monthrange(y, m)[1]:02d}T23:59:59Z"
        return f"{d}-01T00:00:00Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
        return f"{d}T23:59:59Z" if end else f"{d}T00:00:00Z"
    return d  # assume already a full datetime


def _fqs(type=None, status=None, department=None, module=None, address=None,
         date_field="applied", date_from=None, date_to=None):
    """Build the list of ('fq', clause) tuples for the given filters."""
    out = []
    if type:
        out.append(("fq", f'type:"{type}"'))
    if status:
        out.append(("fq", f'status:"{status}"'))
    if department:
        out.append(("fq", f'department:"{department}"'))
    if module:
        out.append(("fq", f'module:"{module}"'))
    if address:
        for tok in str(address).upper().split():
            t = _esc(tok)
            if t:
                out.append(("fq", f"address:*{t}*"))
    if date_from or date_to:
        f = _date_field(date_field)
        out.append(("fq", f"{f}:[{_solr_dt(date_from)} TO {_solr_dt(date_to, end=True)}]"))
    return out


async def _query(params, facet=None):
    qp = list(params) + [("wt", "json")]
    if facet is not None:
        qp.append(("json.facet", json.dumps(facet)))
    async with aiohttp.ClientSession() as session:
        async with session.get(PERMITS_API_BASE, params=qp,
                               timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)  # may be served as text/plain


def _pick(doc, fields):
    return {k: doc[k] for k in fields if k in doc and doc[k] not in (None, "", " ")}


# --------------------------- type resolution ---------------------------

_types_cache = {"list": None}  # [(value, count)], most common first


async def _all_types():
    if _types_cache["list"] is None:
        _types_cache["list"] = [(b["value"], b["count"]) for b in await distinct_values("type", limit=1000)]
    return _types_cache["list"]


async def find_permit_type(keyword):
    """Resolve a user's word to the exact stored type value(s).

    Tiers, each a looser fallback: all-words substring -> any-word substring ->
    fuzzy (typo catch). Returns [] when nothing is close enough, so the agent can
    say 'I couldn't find that type' instead of guessing.
    """
    types = await _all_types()
    low = str(keyword).strip().lower()
    low = _ALIASES.get(low, low)  # expand a known acronym/synonym first
    raw = low.split()
    words = [w for w in raw if w not in _STOP] or raw

    # 1) every keyword word present as a whole word
    allm = [(v, c) for v, c in types if all(_has_word(v.lower(), w) for w in words)]
    if allm:
        return [{"value": v, "count": c} for v, c in allm[:8]]

    # 2) any meaningful word (>=4 chars) present as a whole word
    big = [w for w in words if len(w) >= 4]
    anym = [(v, c) for v, c in types if any(_has_word(v.lower(), w) for w in big)] if big else []
    if anym:
        return [{"value": v, "count": c} for v, c in anym[:8]]

    # acronym tier: a single short token (e.g. "CUP", "TI") -> a type's word-initials
    ak = re.sub(r"[^a-z]", "", low)
    if " " not in low and 2 <= len(ak) <= 5:
        acro = [(v, c) for v, c in types if _initials(v) == ak]
        if acro:
            return [{"value": v, "count": c} for v, c in acro[:8]]

    key = " ".join(words)
    scored = [(v, c, difflib.SequenceMatcher(None, key, v.lower()).ratio()) for v, c in types]
    best = [(v, c) for v, c, s in sorted(scored, key=lambda x: x[2], reverse=True) if s > _FUZZY_FLOOR]
    return [{"value": v, "count": c} for v, c in best[:5]]


# ------------------------------- public API -------------------------------

async def count(group_by=None, **filters):
    """Exact count of permits matching the filters, plus an optional grouped breakdown."""
    qp = [("q", "*"), ("rows", "0")] + _fqs(**filters)
    facet = None
    if group_by:
        gf = group_by if group_by in _FACET_FIELDS else "type"
        facet = {"g": {"type": "terms", "field": gf, "limit": 50, "sort": "count"}}
    data = await _query(qp, facet)
    out = {"count": data["response"]["numFound"]}
    buckets = data.get("facets", {}).get("g", {}).get("buckets")
    if buckets is not None:
        out["breakdown"] = [{"value": b["val"], "count": b["count"]} for b in buckets]
    return out


async def search(query=None, limit=12, **filters):
    """A page of matching permits. Returns the true total plus up to `limit` summaries."""
    limit = max(1, min(int(limit or 12), 50))
    if query:
        toks = [_esc(t) for t in str(query).split() if _esc(t)]
        q = "_text_:(" + " AND ".join(toks) + ")" if toks else "*"
    else:
        q = "*"
    qp = [("q", q), ("rows", str(limit)), ("sort", "applied_date desc")] + _fqs(**filters)
    data = await _query(qp)
    resp = data["response"]
    return {
        "total": resp["numFound"],
        "shown": len(resp["docs"]),
        "results": [_pick(d, _SUMMARY) for d in resp["docs"]],
    }


async def get_permit(act_nbr):
    """Look up one permit by its number (e.g. BS2504744)."""
    nbr = str(act_nbr).strip()
    data = await _query([("q", f"_text_:{_esc(nbr)}"), ("rows", "25")])
    docs = data["response"]["docs"]
    up = nbr.upper()
    exact = [d for d in docs if str(d.get("act_nbr", "")).strip().upper() == up]
    chosen = exact or docs
    if not chosen:
        return {"found": False}
    return {"found": True, "exact": bool(exact), "permit": _pick(chosen[0], _DETAIL)}


async def distinct_values(field, limit=50):
    """Distinct values (with counts) for a facetable field, most common first."""
    gf = field if field in _FACET_FIELDS else "type"
    facet = {"g": {"type": "terms", "field": gf, "limit": int(limit), "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")], facet)
    buckets = data.get("facets", {}).get("g", {}).get("buckets", [])
    return [{"value": b["val"], "count": b["count"]} for b in buckets]
