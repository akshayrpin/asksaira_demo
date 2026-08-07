"""internals_agent/client.py — the generic tool backend.

count / search / get_record / list_values over solr_core, each
validated against the per-collection CitySchema. Unknown fields/values return {"error": ...}
so the agent's tool loop self-corrects instead of silently returning wrong data.
"""

from backend.solr_core import client as _client
from backend.solr_core import querybuilder as _qb
from backend.solr_core import schema as _schema


def _clean(doc):
    """Drop Solr-internal fields and empties from a returned record."""
    return {k: v for k, v in doc.items()
            if not k.startswith("_") and v not in (None, "", " ")}


def _validate_fields(filters, sc):
    for f in filters or []:
        fld = f.get("field")
        if fld and fld not in sc["fields"]:
            return f"unknown field '{fld}'"
    return None


async def list_values(base, field, contains=None):
    sc = await _schema.get_schema(base)
    fmeta = sc["fields"].get(field)
    if not fmeta:
        return {"error": f"unknown field '{field}'", "hint": "use a field name from the catalog in the system prompt"}
    if fmeta["kind"] != "categorical":
        return {"error": f"field '{field}' is {fmeta['kind']}, not categorical",
                "hint": "use a filter/range in count or search instead of list_values"}
    vals = fmeta.get("values", [])
    if contains:
        c = str(contains).lower()
        vals = [v for v in vals if c in str(v["value"]).lower()]
    return {"field": field, "distinct": fmeta.get("distinct"), "values": vals[:200]}


async def count(base, filters=None, group_by=None):
    sc = await _schema.get_schema(base)
    err = _validate_fields(filters, sc)
    if err:
        return {"error": err, "hint": "use a field name from the catalog in the system prompt"}
    facet = None
    if group_by:
        gm = sc["fields"].get(group_by)
        if not gm:
            return {"error": f"unknown group_by field '{group_by}'"}
        if gm["kind"] != "categorical":
            return {"error": f"can only group_by a categorical field; '{group_by}' is {gm['kind']}"}
        facet = {"g": {"type": "terms", "field": group_by, "limit": 200, "sort": "count"}}
    qp = [("q", "*"), ("rows", "0")] + _qb.build_fqs(filters, sc)
    data = await _client.query(base, qp, facet)
    out = {"count": data["response"]["numFound"]}
    buckets = (data.get("facets", {}) or {}).get("g", {}).get("buckets")
    if buckets is not None:
        out["breakdown"] = [{"value": b["val"], "count": b["count"]} for b in buckets]
    return out


async def search(base, filters=None, query=None, limit=12):
    sc = await _schema.get_schema(base)
    err = _validate_fields(filters, sc)
    if err:
        return {"error": err, "hint": "use a field name from the catalog in the system prompt"}
    limit = max(1, min(int(limit or 12), 50))
    if query:
        toks = [_qb._esc(t) for t in str(query).split() if _qb._esc(t)]
        q = "_text_:(" + " AND ".join(toks) + ")" if toks else "*"
    else:
        q = "*"
    qp = [("q", q), ("rows", str(limit))] + _qb.build_fqs(filters, sc)
    data = await _client.query(base, qp)
    resp = data["response"]
    return {"total": resp["numFound"], "shown": len(resp["docs"]),
            "results": [_clean(d) for d in resp["docs"]]}


async def get_record(base, id_field, id_value):
    sc = await _schema.get_schema(base)
    if id_field not in sc["fields"]:
        return {"error": f"unknown field '{id_field}'", "hint": "use a field name from the catalog in the system prompt"}
    # id/record-number fields are often tokenized oddly, so go through the _text_ catch-all
    # then post-filter for an exact match on the requested field.
    data = await _client.query(base, [("q", f"_text_:{_qb._esc(id_value)}"), ("rows", "25")])
    docs = data["response"]["docs"]
    up = str(id_value).strip().upper()
    exact = [d for d in docs if str(d.get(id_field, "")).strip().upper() == up]
    chosen = exact or docs
    if not chosen:
        return {"found": False}
    return {"found": True, "exact": bool(exact), "record": _clean(chosen[0])}
