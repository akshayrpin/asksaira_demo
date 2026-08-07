"""solr_core/facets.py — JSON Facet API helpers (distinct values, counts, ranges).

Reuses the same terms-facet shape permit_client already relies on. Used by schema discovery
(distinct counts / value sets) and by grouped counts.
"""

from backend.solr_core import client as _client


async def distinct_values(base, field, limit=200, numbuckets=False):
    """Distinct values of `field` with counts, most common first.

    Returns (buckets, num_buckets):
      buckets = [{"value","count"}] up to `limit`
      num_buckets = the TRUE distinct count when numbuckets=True (to detect bounded vs open), else None.
    """
    f = {"type": "terms", "field": field, "limit": int(limit), "sort": "count"}
    if numbuckets:
        f["numBuckets"] = True
    data = await _client.query(base, [("q", "*"), ("rows", "0")], {"g": f})
    g = (data.get("facets", {}) or {}).get("g", {}) or {}
    buckets = [{"value": b["val"], "count": b["count"]} for b in g.get("buckets", [])]
    return buckets, (g.get("numBuckets") if numbuckets else None)


async def field_range(base, field):
    """min/max of a numeric or date field via JSON facet aggregations (None if empty)."""
    data = await _client.query(base, [("q", "*"), ("rows", "0")],
                               {"mn": f"min({field})", "mx": f"max({field})"})
    f = data.get("facets", {}) or {}
    return {"min": f.get("mn"), "max": f.get("mx")}
