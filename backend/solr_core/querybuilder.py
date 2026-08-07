"""solr_core/querybuilder.py — turn generic [{field, op, value}] filters into Solr fq clauses.

op ∈ eq | in | range | contains | wildcard. Generalized from permit_client._fqs so it works
over ANY field; date vs numeric range formatting is informed by the cached schema kind.
"""

import calendar
import re


def _esc(token):
    """Keep only alphanumerics so a token is safe inside a wildcard/term query."""
    return re.sub(r"[^A-Za-z0-9]", "", str(token or ""))


def _q(v):
    """Quote a value for an exact term query, escaping embedded quotes."""
    return '"' + str(v).replace('"', r"\"") + '"'


def _solr_dt(d, end=False):
    """YYYY / YYYY-MM / YYYY-MM-DD -> a datetime bound. '*' if empty."""
    if d in (None, "", "*"):
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


def _contains_clauses(field, value):
    """Whole-token wildcard match (from permit_client's address tokenizer), generalized to any
    text field: each token must appear space-bounded, so '30 Elm' doesn't match '230' or 'Elmwood'.
    Returns one clause per token (AND semantics when the caller adds them as separate fq)."""
    toks = [t for t in (_esc(x) for x in str(value).upper().split()) if t]
    out = []
    for i, t in enumerate(toks):
        if i == 0 and t.isdigit():
            out.append(f"{field}:{t}\\ *")
        else:
            out.append(f"{field}:(*\\ {t}\\ * OR *\\ {t} OR {t}\\ *)")
    return out


def _looks_like_date(v):
    """A range bound that is YYYY / YYYY-MM / YYYY-MM-DD (or a full ISO datetime)."""
    return bool(re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?(T[\d:]+Z?)?", str(v or "").strip()))


def _range_bound(v, end=False, is_date=False):
    if v in (None, "", "*"):
        return "*"
    return _solr_dt(v, end=end) if is_date else str(v)


def build_fqs(filters, schema=None):
    """filters: list of {field, op, value}. Returns list of ('fq', clause) tuples.
    `schema` (optional) supplies each field's kind so range knows date vs numeric."""
    out = []
    fields = (schema or {}).get("fields", {})
    for f in filters or []:
        field = f.get("field")
        if not field:
            continue
        op = (f.get("op") or "eq").lower()
        val = f.get("value")
        kind = fields.get(field, {}).get("kind")
        if op == "in":
            vals = val if isinstance(val, (list, tuple)) else [val]
            out.append(("fq", f"{field}:(" + " OR ".join(_q(v) for v in vals) + ")"))
        elif op == "range":
            lo, hi = (list(val) + [None, None])[:2] if isinstance(val, (list, tuple)) else (val, None)
            # expand to datetime bounds when the field is a date OR the bounds look like dates
            is_date = kind == "date" or _looks_like_date(lo) or _looks_like_date(hi)
            out.append(("fq", f"{field}:[{_range_bound(lo, is_date=is_date)} "
                              f"TO {_range_bound(hi, end=True, is_date=is_date)}]"))
        elif op in ("contains", "wildcard"):
            for clause in _contains_clauses(field, val):
                out.append(("fq", clause))
        else:  # eq (default)
            out.append(("fq", f"{field}:{_q(val)}"))
    return out
