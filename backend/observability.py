"""
backend/observability.py — lightweight, best-effort query tracing for the analysis tool.

Writes one record per answered query (question, route, retrieved chunks + their retrieval and
reranker scores, final answer, latency) to a dedicated Cosmos DB container, separate from the
conversations container. The Streamlit dashboard (analysis/dashboard.py) reads these to inspect
and tag why a query did or didn't work.

Design rules:
- NEVER affects the user request. Every write is wrapped so an error just drops the trace.
- Non-blocking: fire_and_forget() schedules the write so it adds nothing to response latency.
- Reuses the existing AZURE_COSMOSDB_* config; only the container name is new
  (AZURE_COSMOSDB_TRACES_CONTAINER, default "query_traces").
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

try:
    from azure.cosmos.aio import CosmosClient
    from azure.cosmos import PartitionKey
except Exception:                       # SDK missing -> tracing silently disabled
    CosmosClient = None
    PartitionKey = None

try:
    from azure.identity.aio import DefaultAzureCredential
except Exception:
    DefaultAzureCredential = None

ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT", "")
KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY", "")
DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE", "")
CONTAINER = os.environ.get("AZURE_COSMOSDB_TRACES_CONTAINER", "query_traces")
ENABLED = os.environ.get("QUERY_TRACING_ENABLED", "true").lower() == "true"

_client = None
_container = None
_init_lock = asyncio.Lock()


async def _get_container():
    """Lazily create the client + container once. Returns None if tracing can't run, so callers
    degrade to a no-op instead of raising."""
    global _client, _container
    if _container is not None:
        return _container
    if not (ENABLED and CosmosClient and ACCOUNT and DATABASE):
        return None
    async with _init_lock:
        if _container is not None:
            return _container
        # Match how the app already talks to Cosmos: use the account key if one is set (local dev),
        # otherwise managed identity / DefaultAzureCredential (prod, which has no key).
        credential = KEY or (DefaultAzureCredential() if DefaultAzureCredential else None)
        if credential is None:
            return None
        endpoint = f"https://{ACCOUNT}.documents.azure.com:443/"
        _client = CosmosClient(endpoint, credential=credential)
        db = _client.get_database_client(DATABASE)     # DB already exists; don't need create rights
        try:
            _container = await db.create_container_if_not_exists(
                id=CONTAINER, partition_key=PartitionKey(path="/route"))
        except Exception:                              # managed identity can't create containers
            _container = db.get_container_client(CONTAINER)   # assume pre-created (see setup below)
        return _container


def website_retrieved(hits):
    """Shape the retrieved website chunks (with their scores) for a trace record. Azure Search
    already attaches @search.score (hybrid) and @search.rerankerScore (semantic) to every hit."""
    out = []
    for h in hits or []:
        out.append({
            "url": h.get("url", ""),
            "breadcrumb": h.get("breadcrumb") or h.get("title") or "",
            "source_category": h.get("source_category", ""),
            "search_score": h.get("@search.score"),
            "rerank_score": h.get("@search.rerankerScore"),
        })
    return out


async def log_trace(record):
    """Best-effort write of one query trace. Any failure is swallowed (tracing must never break a
    user request or leak an exception into the response path)."""
    try:
        container = await _get_container()
        if container is None:
            return
        record.setdefault("id", str(uuid.uuid4()))
        record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        record.setdefault("route", record.get("route") or "unknown")
        await container.upsert_item(record)
    except Exception as e:                # noqa: BLE001 - deliberate catch-all
        logging.warning("query trace write failed (ignored): %s", e)


def fire_and_forget(record):
    """Schedule log_trace without awaiting, so tracing adds no latency to the response. Safe to
    call from a request handler; a missing event loop just drops the trace."""
    try:
        asyncio.get_running_loop().create_task(log_trace(record))
    except RuntimeError:
        pass                              # no running loop -> skip silently
