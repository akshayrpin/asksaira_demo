"""solr_core/schema.py — per-collection schema discovery + value cache.

On first use for a collection (lazy, once, behind a per-collection lock):
  1. list fields + Solr types via {root}/schema/fields,
  2. classify each field: categorical / open / numeric / date,
  3. for categorical fields (bounded distinct count) materialize the value set,
     for numeric/date fields record min/max.
Cached per collection with a TTL. This is what makes the staff agent generic: nothing about
field names or values is hardcoded — a new city just needs its Solr base URL.
"""

import asyncio
import logging
import re
import time

from backend.solr_core import client as _client
from backend.solr_core import facets as _facets

CATEGORICAL_MAX = 200          # cache a field's values only if it has <= this many distinct values
TTL_SECONDS = 24 * 3600
_DISCOVERY_CONCURRENCY = 8     # parallel per-field facet calls during bootstrap

_NUMERIC_TYPES = {"int", "long", "float", "double", "tint", "tlong", "tfloat", "tdouble",
                  "pint", "plong", "pfloat", "pdouble", "pints", "plongs", "currency"}
_DATE_TYPES = {"date", "tdate", "pdate", "pdates", "tdates", "date_range"}
_SKIP = {"_text_", "_root_", "_version_", "_nest_path_", "id"}
_DATE_VALUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2})?")


def _looks_like_dates(buckets):
    """True if a string field's sample values are (almost) all ISO dates — i.e. it's really a date
    field typed as string in Solr (common for *_date columns). Discovery-based, not name-based."""
    sample = [str(b.get("value", "")) for b in buckets[:8]]
    hits = sum(1 for v in sample if _DATE_VALUE_RE.match(v))
    return bool(sample) and hits >= max(1, len(sample) - 1)

_cache = {}     # base -> {"schema": ..., "loaded_at": monotonic}
_locks = {}     # base -> asyncio.Lock


def _lock_for(base):
    lock = _locks.get(base)
    if lock is None:
        lock = _locks[base] = asyncio.Lock()
    return lock


def _base_kind(solr_type):
    t = (solr_type or "").lower()
    if t in _DATE_TYPES:
        return "date"
    if t in _NUMERIC_TYPES:
        return "numeric"
    return "string"   # string/text/unknown -> categorical vs open decided by distinct count


async def _list_fields(base):
    root = _client.collection_root(base)
    last = None
    for attempt in range(3):
        try:
            data = await _client.get_json(root + "/schema/fields", params={"wt": "json"})
            return data.get("fields", []) or []
        except Exception as e:            # transient Solr blip -> retry before failing the bootstrap
            last = e
            await asyncio.sleep(0.4 * (attempt + 1))
    raise last


async def _discover_field(base, name, solr_type):
    """Classify one field and, for categorical, materialize its values; for numeric/date, its range."""
    kind = _base_kind(solr_type)
    entry = {"type": solr_type, "kind": kind}
    if kind == "string":
        try:
            buckets, nb = await _facets.distinct_values(
                base, name, limit=CATEGORICAL_MAX + 1, numbuckets=True)
        except Exception:
            logging.exception("distinct_values failed for %s", name)
            buckets, nb = [], None
        # A string field whose values are ISO dates is really a date field (Solr just typed it
        # string). Treat it as date so range filters get proper datetime bounds.
        if buckets and _looks_like_dates(buckets):
            entry["kind"] = "date"
            try:
                entry["range"] = await _facets.field_range(base, name)
            except Exception:
                entry["range"] = None
            return name, entry
        distinct = nb if nb is not None else len(buckets)
        entry["distinct"] = distinct
        # Categorical only when it has a bounded, NON-EMPTY value set. distinct==0 means the field
        # is empty or not facetable (e.g. stored-but-not-indexed text) -> treat as open, not a
        # useless "categorical, 0 values".
        if distinct and 0 < distinct <= CATEGORICAL_MAX and buckets:
            entry["kind"] = "categorical"
            entry["values"] = buckets
        else:
            entry["kind"] = "open"
    else:  # numeric / date
        try:
            entry["range"] = await _facets.field_range(base, name)
        except Exception:
            logging.exception("field_range failed for %s", name)
            entry["range"] = None
    return name, entry


async def _bootstrap(base):
    fields_meta = await _list_fields(base)
    targets = [(fm.get("name"), fm.get("type")) for fm in fields_meta
               if fm.get("name") and fm.get("name") not in _SKIP]
    sem = asyncio.Semaphore(_DISCOVERY_CONCURRENCY)

    async def _guarded(name, solr_type):
        async with sem:
            return await _discover_field(base, name, solr_type)

    results = await asyncio.gather(*[_guarded(n, t) for n, t in targets])
    fields = {name: entry for name, entry in results}
    collection = _client.collection_root(base).rstrip("/").split("/")[-1]
    return {"collection": collection, "base": base,
            "loaded_at": time.time(), "fields": fields}


async def get_schema(base, force=False):
    """Return the cached CitySchema for `base`, bootstrapping on first use / after TTL."""
    entry = _cache.get(base)
    fresh = entry and not force and (time.monotonic() - entry["loaded_at"] < TTL_SECONDS)
    if fresh:
        return entry["schema"]
    async with _lock_for(base):
        entry = _cache.get(base)
        fresh = entry and not force and (time.monotonic() - entry["loaded_at"] < TTL_SECONDS)
        if fresh:
            return entry["schema"]
        schema = await _bootstrap(base)
        _cache[base] = {"schema": schema, "loaded_at": time.monotonic()}
        return schema


# --------------------------- prompt rendering ---------------------------

INLINE_MAX_VALUES = 80      # inline a categorical field's values when it has at most this many
INLINE_MAX_CHARS = 1500     # ...and the rendered value list is at most this long


def compact_catalog(schema):
    """One line per field for the system prompt. Small categorical fields INLINE their actual
    values, so the model knows which field a given value belongs to without probing (e.g. it can
    see 'BUSINESS TAX' lives under `module`). Larger categorical fields show only a count and the
    model calls list_values. Even inlining every value on this dataset is ~1k tokens, so this is
    cheap; the caps bound a pathological many-value city."""
    parts = []
    for name, m in schema["fields"].items():
        k = m["kind"]
        if k == "categorical":
            d = m.get("distinct") or 0
            joined = " | ".join(str(v["value"]) for v in m.get("values", []))
            if 0 < d <= INLINE_MAX_VALUES and len(joined) <= INLINE_MAX_CHARS:
                parts.append(f"{name} (categorical): {joined}")
            else:
                parts.append(f"{name} (categorical, {d} values — use list_values to see them)")
        elif k == "numeric":
            parts.append(f"{name} (numeric)")
        elif k == "date":
            rng = m.get("range") or {}
            span = f", {str(rng['min'])[:10]}..{str(rng['max'])[:10]}" if rng.get("min") and rng.get("max") else ""
            parts.append(f"{name} (date{span})")
        else:
            parts.append(f"{name} (open text)")
    return "\n".join(parts)
