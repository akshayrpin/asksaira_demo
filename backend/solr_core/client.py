"""solr_core/client.py — generic async Solr HTTP access, collection-agnostic.

`base` is the full Solr query endpoint, e.g.
    http://host:7337/solr/load_initial_burbank/query
Schema / admin endpoints are derived from it by stripping the trailing /query (or /select).
Open, read-only API; no auth (same as permit_client).
"""

import json

import aiohttp

TIMEOUT = 25


def collection_root(base):
    """Strip a trailing /query or /select to get the collection root URL."""
    b = (base or "").rstrip("/")
    for suffix in ("/query", "/select"):
        if b.endswith(suffix):
            return b[: -len(suffix)]
    return b


async def query(base, params, facet=None):
    """GET the Solr query handler. `params` is a list of (key, value) tuples.
    If `facet` is given it is sent as json.facet. Returns parsed JSON."""
    qp = list(params) + [("wt", "json")]
    if facet is not None:
        qp.append(("json.facet", json.dumps(facet)))
    async with aiohttp.ClientSession() as session:
        async with session.get(base, params=qp,
                               timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)  # may be served as text/plain


async def get_json(url, params=None):
    """GET an arbitrary Solr endpoint (schema/fields, admin/luke) and parse JSON."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params,
                               timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.json(content_type=None)
