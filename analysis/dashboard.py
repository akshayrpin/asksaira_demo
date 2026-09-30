"""
analysis/dashboard.py — Ask Saira retrieval observability dashboard (Streamlit).

Reads the query_traces Cosmos container written by backend/observability.py and lets you:
  - browse answered queries (filter by route / tag / text),
  - drill into one: question -> retrieved chunks with retrieval + reranker scores -> answer,
  - TAG why it did or didn't work (good / missing-content / mis-ranked / stale / wrong-answer),
    which turns anecdotes into the failure categories that drive retrieval decisions.

Run:  streamlit run analysis/dashboard.py
Env:  AZURE_COSMOSDB_ACCOUNT, AZURE_COSMOSDB_ACCOUNT_KEY, AZURE_COSMOSDB_DATABASE,
      AZURE_COSMOSDB_TRACES_CONTAINER (default "query_traces")
Read-only against prod data except the tag/notes fields it writes back.
"""

import os

import streamlit as st
from azure.cosmos import CosmosClient

try:
    from azure.identity import DefaultAzureCredential
except Exception:
    DefaultAzureCredential = None

ACCOUNT = os.environ.get("AZURE_COSMOSDB_ACCOUNT", "")
KEY = os.environ.get("AZURE_COSMOSDB_ACCOUNT_KEY", "")
DATABASE = os.environ.get("AZURE_COSMOSDB_DATABASE", "")
CONTAINER = os.environ.get("AZURE_COSMOSDB_TRACES_CONTAINER", "query_traces")
CONV_CONTAINER = os.environ.get("AZURE_COSMOSDB_CONVERSATIONS_CONTAINER", "conversations")

TAGS = ["untagged", "good", "missing-content", "mis-ranked", "stale", "wrong-answer", "other"]
FEEDBACKS = ["all", "positive", "negative", "none"]


@st.cache_resource
def _client():
    # account key if provided (local dev), else managed identity / az-login (DefaultAzureCredential)
    cred = KEY or (DefaultAzureCredential() if DefaultAzureCredential else None)
    return CosmosClient(f"https://{ACCOUNT}.documents.azure.com:443/", credential=cred)


def _container():
    return _client().get_database_client(DATABASE).get_container_client(CONTAINER)


def _conv_container():
    return _client().get_database_client(DATABASE).get_container_client(CONV_CONTAINER)


def _fb_kind(v):
    """Collapse the feedback enum (positive / negative / wrong_citation / ...) to a thumb."""
    if not v or v == "neutral":
        return None
    return "positive" if v == "positive" else "negative"


@st.cache_data(ttl=60)
def _feedback_map():
    """Rated assistant messages -> feedback, keyed by message id (== the trace's answer_id)."""
    sql = ("SELECT c.id, c.feedback FROM c WHERE c.type='message' AND c.role='assistant' "
           "AND IS_DEFINED(c.feedback) AND c.feedback != '' AND c.feedback != 'neutral'")
    out = {}
    try:
        for m in _conv_container().query_items(sql, enable_cross_partition_query=True):
            if m.get("id"):
                out[m["id"]] = m.get("feedback")
    except Exception:
        pass
    return out


def _query(route, tag, text, limit):
    where = []
    params = []
    if route != "all":
        where.append("c.route = @route")
        params.append({"name": "@route", "value": route})
    if tag == "untagged":
        where.append("(NOT IS_DEFINED(c.tag) OR c.tag = '' OR c.tag = 'untagged')")
    elif tag != "all":
        where.append("c.tag = @tag")
        params.append({"name": "@tag", "value": tag})
    if text:
        where.append("CONTAINS(LOWER(c.question), @q)")
        params.append({"name": "@q", "value": text.lower()})
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT * FROM c {clause} ORDER BY c.ts DESC OFFSET 0 LIMIT {int(limit)}"
    return list(_container().query_items(sql, parameters=params, enable_cross_partition_query=True))


def _save_tag(item, tag, notes):
    item["tag"] = tag
    item["notes"] = notes
    _container().upsert_item(item)


st.set_page_config(page_title="Ask Saira · Retrieval Observability", layout="wide")
st.title("Ask Saira — Retrieval Observability")

if not (ACCOUNT and DATABASE):
    st.error("Set AZURE_COSMOSDB_ACCOUNT and AZURE_COSMOSDB_DATABASE "
             "(plus either AZURE_COSMOSDB_ACCOUNT_KEY or a managed identity).")
    st.stop()

with st.sidebar:
    st.header("Filters")
    route = st.selectbox("Route", ["all", "website", "permit", "events"])
    tag = st.selectbox("Tag", ["all"] + TAGS)
    feedback = st.selectbox("Feedback", FEEDBACKS)
    text = st.text_input("Question contains")
    limit = st.slider("Max rows", 10, 500, 100, step=10)
    if st.button("Refresh"):
        st.rerun()

rows = _query(route, tag, text, limit)

# join user feedback (thumbs) by answer_id == the assistant message id
fb_map = _feedback_map()
for r in rows:
    raw = fb_map.get(r.get("answer_id"))
    r["_fb"], r["_fb_kind"] = raw, _fb_kind(raw)
if feedback != "all":
    want = None if feedback == "none" else feedback
    rows = [r for r in rows if r.get("_fb_kind") == want]

st.caption(f"{len(rows)} queries")

if not rows:
    st.info("No traces yet with the selected filters.")
    st.stop()

# left: list to pick from; right: the selected trace
left, right = st.columns([1, 2])

with left:
    st.subheader("Queries")
    labels = {}
    for r in rows:
        badge = r.get("tag", "") or ""
        thumb = {"positive": "👍", "negative": "👎"}.get(r.get("_fb_kind"), "")
        labels[r["id"]] = f"{thumb} [{r.get('route','?')}] {r.get('question','')[:55]}  {('· ' + badge) if badge else ''}"
    chosen = st.radio("Select", list(labels), format_func=lambda i: labels[i], label_visibility="collapsed")

sel = next((r for r in rows if r["id"] == chosen), None)

with right:
    if sel:
        st.subheader(sel.get("question", ""))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", sel.get("route", "?"))
        c2.metric("Latency (ms)", sel.get("latency_ms", "—"))
        top = sel.get("retrieved") or []
        c3.metric("Chunks retrieved", len(top))
        thumb = {"positive": "👍", "negative": "👎"}.get(sel.get("_fb_kind"))
        c4.metric("User feedback", f"{thumb} {sel.get('_fb','')}" if thumb else "—")

        st.markdown("**Retrieved (in answer order, with scores)**")
        if top:
            st.dataframe(
                [{"breadcrumb": t.get("breadcrumb", ""), "search": t.get("search_score"),
                  "rerank": t.get("rerank_score"), "cat": t.get("source_category", ""),
                  "url": t.get("url", "")} for t in top],
                use_container_width=True, hide_index=True)
        else:
            st.caption("No retrieval detail (non-website route).")

        st.markdown("**Answer**")
        st.write(sel.get("answer", ""))

        # Deterministic verification results from website_pipeline (URLs repaired against the
        # sources, ungrounded contacts flagged). Stored as lists once round-tripped through Cosmos.
        reps = sel.get("url_repairs") or []
        flags = sel.get("contact_flags") or {}
        bad_em = flags.get("unverified_emails") or []
        bad_ph = flags.get("unverified_phones") or []
        if reps or bad_em or bad_ph:
            st.markdown("**Verification**")
            fixed = [r for r in reps if r and r[0] == "repair"]
            dropped = [r for r in reps if r and r[0] == "drop"]
            if fixed:
                st.warning(f"{len(fixed)} link(s) repaired (model corrupted the URL)")
                st.dataframe([{"from": r[1], "to": r[2]} for r in fixed],
                             use_container_width=True, hide_index=True)
            if dropped:
                st.error(f"{len(dropped)} link(s) dropped (fabricated, not in sources)")
                st.dataframe([{"url": r[1]} for r in dropped],
                             use_container_width=True, hide_index=True)
            if bad_em or bad_ph:
                st.error("Ungrounded contacts flagged: " + ", ".join(bad_em + bad_ph))
        else:
            st.caption("Verification: no URL or contact issues.")

        st.divider()
        st.markdown("**Tag this query**")
        cur = sel.get("tag", "untagged") or "untagged"
        new_tag = st.selectbox("Verdict", TAGS, index=TAGS.index(cur) if cur in TAGS else 0)
        notes = st.text_area("Notes", value=sel.get("notes", ""))
        if st.button("Save tag"):
            _save_tag(sel, new_tag, notes)
            st.success("Saved.")
            st.rerun()
