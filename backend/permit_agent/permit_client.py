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


# --------------------------- Domain Facts ---------------------------


# The agent only ever surfaces permits in these (granted) statuses. Applied to EVERY query
# in _query, so counts, lists, breakdowns, type resolution, and single-permit lookups are
# all restricted to them. "Permit Reissued" has no records today but is kept for the future.
ALLOWED_STATUSES = ["Permit Final", "Permit Issued", "Permit Reissued", "Approved", "Issued"]
_STATUS_FILTER = "status:(" + " OR ".join(f'"{s}"' for s in ALLOWED_STATUSES) + ")"

# Code Enforcement is identified by TYPE.
# Never surface description / people / valuation for these. (For Burbank)
CODE_ENFORCEMENT_TYPE = "code enforcement"          # matched case-insensitively against `type`
CODE_ENFORCEMENT_ACTIVE_STATUS = "Admin Pending"    # open cases ("Admin Completed" = closed); confirm with city
CODE_ENFORCEMENT_FIELDS = ["act_nbr", "status", "type", "address", "applied_date"]  # For Burbank
CODE_ENFORCEMENT_CONTACT = "For more information on the code enforcement permits, contact Burbank Code Enforcement at (818) 238-5225."

# "Building permits" = module BUILDING minus these admin/enforcement types (per ePALS "Is Building
# Permit" catalog). Used as the deterministic default scope for general permit questions.
NON_BUILDING_TYPES = ["Building Administration", "Code Enforcement", "Zoning Verification"]


def _is_code_enforcement(doc):
    return str(doc.get("type", "")).strip().lower() == CODE_ENFORCEMENT_TYPE

# BUSINESS TAX: an account renews yearly, so ONE business has many records over time. Counting
# every record over-counts massively. "How many businesses are in the city"
# means the currently-active accounts = these two statuses. 
# "New businesses in a year" = accounts first created that year with renewal:"NO".
ACTIVE_BUSINESS_STATUSES = ["Paid / Current", "Pending Renewal"]

# Fields a user actually cares about, kept small so tool results stay cheap.
_SUMMARY = ["act_nbr", "type", "status", "department", "address",
            "applied_date", "valuation_calculated", "description"]

# For a single lookup
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


# --------------------------- Sanitizer Functions ---------------------------

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



# --------------------------- Query Builder ---------------------------

# Street-type suffixes (Dr/Drive, Rd/Road, St/Street...). The suffix a resident types is the
# least reliable token, so if a strict address match finds nothing we retry with it dropped.
_STREET_SUFFIXES = {"ST", "STREET", "AVE", "AV", "AVENUE", "BLVD", "BOULEVARD", "DR", "DRIVE",
                    "RD", "ROAD", "LN", "LANE", "CT", "COURT", "PL", "PLACE", "WAY", "CIR",
                    "CIRCLE", "TER", "TERRACE", "PKWY", "PARKWAY"}
_SUFFIX_NOTE = ("No exact match on the street suffix, so these results are matched on the house "
                "number and street name; please confirm the street type.")


def _has_droppable_suffix(address):
    """True if the address ends in a known street-type suffix we could relax on a retry."""
    raw = [x for x in str(address or "").upper().split() if x]
    return len(raw) > 2 and raw[-1] in _STREET_SUFFIXES


def _fqs(type=None, status=None, department=None, module=None, address=None,
         date_field="applied", date_from=None, date_to=None, renewal=None, drop_suffix=False,
         exclude_types=None):
    """Build the list of ('fq', clause) tuples for the given filters."""
    out = []
    if type:
        out.append(("fq", f'type:"{type}"'))
    if exclude_types:                # drop these type values (used by the building-permit default)
        out.append(("fq", "-type:(" + " OR ".join(f'"{t}"' for t in exclude_types) + ")"))
    if status:                       # str -> one status; list/tuple -> OR-match several
        if isinstance(status, (list, tuple)):
            out.append(("fq", "status:(" + " OR ".join(f'"{s}"' for s in status) + ")"))
        else:
            out.append(("fq", f'status:"{status}"'))
    if renewal:                      # BUSINESS TAX: "NO" = new account, "YES" = renewal
        out.append(("fq", f'renewal:"{renewal}"'))
    if department:
        out.append(("fq", f'department:"{department}"'))
    if module:
        out.append(("fq", f'module:"{module}"'))
    if address:
        # Match by whole token, not substring, so "30 Elm" doesn't pull in "230" or
        # "Elmwood". A leading house number is anchored at the start ("30 *"); every other
        # token must be bounded by a space on the left and a space OR end-of-string on the
        # right ("* THIRD *" OR "* THIRD"). The end-of-string alternative means we assume
        # nothing about trailing format, so this holds for other indexes too. Whole-token
        # matching also pulls in unit sub-addresses ("201 E MAGNOLIA BLVD 145"), which a
        # plain exact match would miss.
        raw = [x for x in str(address).upper().split() if x]
        # Only when a strict match found nothing (drop_suffix): relax the trailing street-type
        # suffix so a wrong/missing suffix ("Sunset Canyon Rd" vs the actual "Sunset Canyon Dr")
        # still matches on house number + street name. Strict by default, so two same-named
        # streets with different suffixes aren't silently conflated (especially on counts).
        if drop_suffix and _has_droppable_suffix(address):
            raw = raw[:-1]
        toks = [t for t in (_esc(x) for x in raw) if t]
        for i, t in enumerate(toks):
            if i == 0 and t.isdigit():
                out.append(("fq", f"address:{t}\\ *"))
            else:
                out.append(("fq", f"address:(*\\ {t}\\ * OR *\\ {t})"))
    if date_from or date_to:
        f = _date_field(date_field)
        out.append(("fq", f"{f}:[{_solr_dt(date_from)} TO {_solr_dt(date_to, end=True)}]"))
    return out


# HTTP Call to the API
async def _query(params, facet=None, status_filter=False):
    # The granted-permit status whitelist (ALLOWED_STATUSES) is RETIRED as the default. Applied
    # to every query it (a) filtered out business-tax/license records (Active, Paid, Out of
    # Business), which is why business counts came back empty, (b) hid code enforcement
    # (Open/Closed), which the city has now approved for display, and (c) undercounted permits in
    # valid in-progress statuses (e.g. "PC w/ Corrections", "Permit On Hold"). Pass
    # status_filter=True to re-apply it for a specific "granted-only" query once the city defines
    # which statuses should count.
    qp = list(params) + [("wt", "json")]
    if status_filter:
        qp.append(("fq", _STATUS_FILTER))
    if facet is not None:
        qp.append(("json.facet", json.dumps(facet)))
    async with aiohttp.ClientSession() as session:
        async with session.get(PERMITS_API_BASE, params=qp,
                               timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)  # may be served as text/plain


def _pick(doc, fields):
    return {k: doc[k] for k in fields if k in doc and doc[k] not in (None, "", " ")}


def _summary_for(doc):
    """Per-record summary. Code Enforcement records are restricted to the city-approved fields
    (Permit Number, Status, Type, Address, Started); every other module uses the full summary."""
    if _is_code_enforcement(doc):
        return _pick(doc, CODE_ENFORCEMENT_FIELDS)
    return _pick(doc, _SUMMARY)


def _has_code_enforcement(docs):
    return any(_is_code_enforcement(d) for d in docs)


# --------------------------- type resolution ---------------------------

_types_cache = {"list": None}  # [(value, count)], most common first


async def _all_types():
    if _types_cache["list"] is None:
        _types_cache["list"] = [(b["value"], b["count"]) for b in await distinct_values("type", limit=1000)]
    return _types_cache["list"]


def _match_values(keyword, pairs, aliases=None):
    """Resolve a user's word to the exact stored value(s) from `pairs` = [(value, count)].

    Tiers, each a looser fallback: all-words whole-word -> any long word -> acronym ->
    fuzzy (typo catch). Returns [] when nothing is close enough, so the agent can say
    'I couldn't find that' instead of guessing. Shared by type and status resolution.
    """
    low = str(keyword).strip().lower()
    if aliases:
        low = aliases.get(low, low)   # expand a known acronym/synonym first
    raw = low.split()
    words = [w for w in raw if w not in _STOP] or raw

    # 1) every keyword word present as a whole word
    allm = [(v, c) for v, c in pairs if all(_has_word(v.lower(), w) for w in words)]
    if allm:
        return [{"value": v, "count": c} for v, c in allm[:8]]

    # 2) any meaningful word (>=4 chars) present as a whole word
    big = [w for w in words if len(w) >= 4]
    anym = [(v, c) for v, c in pairs if any(_has_word(v.lower(), w) for w in big)] if big else []
    if anym:
        return [{"value": v, "count": c} for v, c in anym[:8]]

    # acronym tier: a single short token (e.g. "CUP", "TI") -> a value's word-initials
    ak = re.sub(r"[^a-z]", "", low)
    if " " not in low and 2 <= len(ak) <= 5:
        acro = [(v, c) for v, c in pairs if _initials(v) == ak]
        if acro:
            return [{"value": v, "count": c} for v, c in acro[:8]]

    key = " ".join(words)
    scored = [(v, c, difflib.SequenceMatcher(None, key, v.lower()).ratio()) for v, c in pairs]
    best = [(v, c) for v, c, s in sorted(scored, key=lambda x: x[2], reverse=True) if s > _FUZZY_FLOOR]
    return [{"value": v, "count": c} for v, c in best[:5]]


async def find_permit_type(keyword):
    """Resolve a user's word to the exact stored TYPE value(s), e.g. 'solar' -> 'Solar'."""
    return _match_values(keyword, await _all_types(), _ALIASES)


async def _statuses_in(filters):
    """Distinct status values that actually occur within the given type/module filters, so a
    status word can be resolved IN CONTEXT (e.g. 'pending' for solar -> only 'Admin Pending')."""
    qp = [("q", "*"), ("rows", "0")] + _fqs(**filters)
    facet = {"g": {"type": "terms", "field": "status", "limit": 100, "sort": "count"}}
    data = await _query(qp, facet)
    return [(b["val"], b["count"]) for b in data.get("facets", {}).get("g", {}).get("buckets", [])]


async def find_permit_status(keyword, type=None, module=None):
    """Resolve a status word (e.g. 'pending', 'active', 'expired', 'issued') to the exact stored
    status value(s). Scoped to the type/module context when given, so the same word resolves to the
    right status for that permit kind instead of every status that contains it."""
    filters = {}
    if type:
        filters["type"] = await _resolve_type(type)
    if module:
        filters["module"] = module
    return _match_values(keyword, await _statuses_in(filters))


# ------------------------------- public API -------------------------------

async def _resolve_type(t):
    """Map a permit-kind string to the exact stored type value when it isn't one already.

    Guards against the model passing an abbreviation/typo straight into a filter (e.g.
    'ADU' instead of 'Accessory Dwelling Unit'), which an exact-match field would score 0.
    Only resolves when there is exactly ONE confident match; ambiguous inputs (e.g. 'pool',
    which maps to several types) are left unchanged for the agent to disambiguate.
    """
    if not t:
        return t
    known = {v for v, _ in await _all_types()}
    if t in known:
        return t
    m = await find_permit_type(t)
    return m[0]["value"] if len(m) == 1 else t


async def _resolve_status(s, type=None, module=None):
    """Map a status word to the exact stored value within the type/module context, if it isn't one
    already. Only substitutes on a single confident match; ambiguous input is left unchanged so the
    agent can disambiguate. This is why 'pending' for solar resolves to 'Admin Pending'."""
    if not s:
        return s
    filters = {}
    if type:                              # already resolved by the caller
        filters["type"] = type
    if module:
        filters["module"] = module
    statuses = await _statuses_in(filters)
    if s in {v for v, _ in statuses}:
        return s
    m = _match_values(s, statuses)
    return m[0]["value"] if len(m) == 1 else s


async def count(group_by=None, **filters):
    """Exact count of permits matching the filters, plus an optional grouped breakdown."""
    if filters.get("type"):
        filters["type"] = await _resolve_type(filters["type"])
    if filters.get("status") and not isinstance(filters["status"], (list, tuple)):
        filters["status"] = await _resolve_status(
            filters["status"], type=filters.get("type"), module=filters.get("module"))
    facet = None
    if group_by:
        gf = group_by if group_by in _FACET_FIELDS else "type"
        facet = {"g": {"type": "terms", "field": gf, "limit": 50, "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")] + _fqs(**filters), facet)
    broadened = False
    if data["response"]["numFound"] == 0 and _has_droppable_suffix(filters.get("address")):
        data = await _query([("q", "*"), ("rows", "0")] + _fqs(drop_suffix=True, **filters), facet)
        broadened = data["response"]["numFound"] > 0
    out = {"count": data["response"]["numFound"]}
    if broadened:
        out["note"] = _SUFFIX_NOTE
    buckets = data.get("facets", {}).get("g", {}).get("buckets")
    if buckets is not None:
        out["breakdown"] = [{"value": b["val"], "count": b["count"]} for b in buckets]
    return out


async def search(query=None, **filters):
    """Matching permits for display, plus the true total. FIXED display rule (not caller-set):
    fewer than 15 matches -> list ALL of them; 15 or more -> list only the first 10 (newest first)."""
    if filters.get("type"):
        filters["type"] = await _resolve_type(filters["type"])
    if filters.get("status") and not isinstance(filters["status"], (list, tuple)):
        filters["status"] = await _resolve_status(
            filters["status"], type=filters.get("type"), module=filters.get("module"))
    if query:
        toks = [_esc(t) for t in str(query).split() if _esc(t)]
        q = "_text_:(" + " AND ".join(toks) + ")" if toks else "*"
    else:
        q = "*"
    # Fetch 15: enough to have them all when total < 15, and to slice to 10 when total >= 15.
    base = [("q", q), ("rows", "15"), ("sort", "applied_date desc")]
    data = await _query(base + _fqs(**filters))
    broadened = False
    if data["response"]["numFound"] == 0 and _has_droppable_suffix(filters.get("address")):
        data = await _query(base + _fqs(drop_suffix=True, **filters))
        broadened = data["response"]["numFound"] > 0
    resp = data["response"]
    total = resp["numFound"]
    docs = resp["docs"] if total < 15 else resp["docs"][:10]
    out = {
        "total": total,
        "shown": len(docs),
        "results": [_summary_for(d) for d in docs],
    }
    notes = []
    if broadened:
        notes.append(_SUFFIX_NOTE)
    if _has_code_enforcement(docs):
        notes.append(CODE_ENFORCEMENT_CONTACT)
    if notes:
        out["note"] = " ".join(notes)
    return out


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
    doc = chosen[0]
    if _is_code_enforcement(doc):
        return {"found": True, "exact": bool(exact),
                "permit": _pick(doc, CODE_ENFORCEMENT_FIELDS), "note": CODE_ENFORCEMENT_CONTACT}
    return {"found": True, "exact": bool(exact), "permit": _pick(doc, _DETAIL)}


async def distinct_values(field, limit=50):
    """Distinct values (with counts) for a facetable field, most common first."""
    gf = field if field in _FACET_FIELDS else "type"
    facet = {"g": {"type": "terms", "field": gf, "limit": int(limit), "sort": "count"}}
    data = await _query([("q", "*"), ("rows", "0")], facet)
    buckets = data.get("facets", {}).get("g", {}).get("buckets", [])
    return [{"value": b["val"], "count": b["count"]} for b in buckets]
