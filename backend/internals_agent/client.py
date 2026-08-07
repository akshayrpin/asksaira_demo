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


def _year_of(v):
    s = str(v or "").strip()
    return int(s[:4]) if len(s) >= 4 and s[:4].isdigit() else None


def _periods(y0, y1, interval):
    """(label, from, to) tuples covering [y0..y1] at year / quarter / month granularity.
    from/to are YYYY / YYYY-MM strings that _solr_dt expands to full datetime bounds."""
    out = []
    for y in range(min(y0, y1), max(y0, y1) + 1):
        if interval == "quarter":
            for q, (m0, m1) in enumerate([(1, 3), (4, 6), (7, 9), (10, 12)], start=1):
                out.append((f"{y}-Q{q}", f"{y}-{m0:02d}", f"{y}-{m1:02d}"))
        elif interval == "month":
            for m in range(1, 13):
                out.append((f"{y}-{m:02d}", f"{y}-{m:02d}", f"{y}-{m:02d}"))
        else:  # year
            out.append((str(y), str(y), str(y)))
    return out


def _time_facet(gbt, sc):
    """Histogram over a date field as ONE Solr request of per-period QUERY sub-facets. Works on
    real date fields AND date-valued string fields (applied_date etc.), which a native range facet
    can't. Returns (error, facet_dict, [(key, label), ...])."""
    field = gbt.get("field")
    fm = sc["fields"].get(field)
    if not fm:
        return f"unknown field '{field}'", None, None
    if fm["kind"] != "date":
        return f"group_by_time needs a date field; '{field}' is {fm['kind']}", None, None
    interval = (gbt.get("interval") or "year").lower()
    if interval not in ("year", "quarter", "month"):
        return "interval must be year, quarter, or month", None, None
    rng = fm.get("range") or {}
    y0 = _year_of(gbt.get("start")) or _year_of(rng.get("min"))
    y1 = _year_of(gbt.get("end")) or _year_of(rng.get("max"))
    if not y0 or not y1:
        return f"could not determine a date range for '{field}'", None, None
    periods = _periods(y0, y1, interval)
    if len(periods) > 400:
        return "too many periods; use a coarser interval or a narrower start/end", None, None
    facet, order = {}, []
    for i, (label, lo, hi) in enumerate(periods):
        key = f"p{i}"
        facet[key] = {"type": "query",
                      "q": f"{field}:[{_qb._solr_dt(lo)} TO {_qb._solr_dt(hi, end=True)}]"}
        order.append((key, label))
    return None, facet, order


async def count(base, filters=None, group_by=None, group_by_time=None):
    sc = await _schema.get_schema(base)
    err = _validate_fields(filters, sc)
    if err:
        return {"error": err, "hint": "use a field name from the catalog in the system prompt"}
    facet, time_order = None, None
    if group_by_time:
        gerr, facet, time_order = _time_facet(group_by_time, sc)
        if gerr:
            return {"error": gerr}
    elif group_by:
        gm = sc["fields"].get(group_by)
        if not gm:
            return {"error": f"unknown group_by field '{group_by}'"}
        if gm["kind"] != "categorical":
            return {"error": f"can only group_by a categorical field; '{group_by}' is {gm['kind']}"}
        facet = {"g": {"type": "terms", "field": group_by, "limit": 200, "sort": "count"}}
    qp = [("q", "*"), ("rows", "0")] + _qb.build_fqs(filters, sc)
    data = await _client.query(base, qp, facet)
    out = {"count": data["response"]["numFound"]}
    f = data.get("facets", {}) or {}
    if time_order is not None:
        out["breakdown"] = [{"value": label, "count": (f.get(key) or {}).get("count", 0)}
                            for key, label in time_order]
    else:
        buckets = f.get("g", {}).get("buckets")
        if buckets is not None:
            out["breakdown"] = [{"value": b["val"], "count": b["count"]} for b in buckets]
    return out


async def stats(base, field, filters=None, group_by=None):
    """sum / avg / min / max of a NUMERIC field, optionally broken down by a categorical field.
    Aggregations are over records that actually have a value for the field."""
    sc = await _schema.get_schema(base)
    fm = sc["fields"].get(field)
    if not fm:
        return {"error": f"unknown field '{field}'", "hint": "use a numeric field from the catalog"}
    if fm["kind"] != "numeric":
        return {"error": f"stats needs a numeric field; '{field}' is {fm['kind']}"}
    err = _validate_fields(filters, sc)
    if err:
        return {"error": err}
    aggs = {"sum": f"sum({field})", "avg": f"avg({field})",
            "min": f"min({field})", "max": f"max({field})"}
    if group_by:
        gm = sc["fields"].get(group_by)
        if not gm or gm["kind"] != "categorical":
            return {"error": f"group_by must be a categorical field"}
        facet = {"g": {"type": "terms", "field": group_by, "limit": 200, "facet": aggs}}
    else:
        facet = dict(aggs)
    qp = [("q", "*"), ("rows", "0")] + _qb.build_fqs(filters, sc)
    data = await _client.query(base, qp, facet)
    f = data.get("facets", {}) or {}
    total = data["response"]["numFound"]
    note = "avg/sum are over records that have a value for the field"
    if group_by:
        buckets = f.get("g", {}).get("buckets", [])
        return {"total": total, "note": note,
                "breakdown": [{"value": b["val"], "count": b["count"], "sum": b.get("sum"),
                               "avg": b.get("avg"), "min": b.get("min"), "max": b.get("max")}
                              for b in buckets]}
    return {"total_records": total, "sum": f.get("sum"), "avg": f.get("avg"),
            "min": f.get("min"), "max": f.get("max"), "note": note}


async def search(base, filters=None, query=None, limit=12, sort=None):
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
    qp = [("q", q), ("rows", str(limit))]
    if sort and sort.get("field"):
        sf = sort["field"]
        fm = sc["fields"].get(sf)
        if not fm:
            return {"error": f"unknown sort field '{sf}'"}
        if fm["kind"] not in ("numeric", "date", "categorical"):
            return {"error": f"cannot sort on '{sf}' ({fm['kind']}); sort a numeric, date, or categorical field"}
        sd = (sort.get("dir") or "desc").lower()
        qp.append(("sort", f"{sf} {sd if sd in ('asc', 'desc') else 'desc'}"))
    qp += _qb.build_fqs(filters, sc)
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
