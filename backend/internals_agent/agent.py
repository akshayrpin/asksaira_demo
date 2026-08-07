"""internals_agent/agent.py — generic, schema-grounded ReAct agent for CITY STAFF.

Answers any filter / count / group_by / search question over the ePALS Solr collection, including
multi-hop questions, by driving the generic tools in client.py. The per-collection schema (fields +
categorical value sets) is discovered/cached on first use and injected into the system prompt so the
model only ever filters/groups on fields and values that actually exist. A numeric self-check runs
before returning.

Entry: answer_internals_query(user_query, client, model, base=None, history=None) -> answer str.
`client` is the app's async AzureOpenAI client; `model` is the deployment name; `base` is the Solr
query endpoint for the target city (defaults to PERMITS_API_BASE for the Burbank test).
"""

import datetime
import json
import logging
import os

from backend.internals_agent import client as ic
from backend.internals_agent import verifier as vf
from backend.solr_core import schema as sc

MAX_STEPS = 10
DEFAULT_BASE = os.environ.get(
    "PERMITS_API_BASE",
    "http://burbank.edgesoftinc.com:7337/solr/load_initial_burbank/query",
)

SYSTEM = """You are an internal analytics assistant for CITY STAFF over the ePALS permitting database. You answer by querying the data with the tools; you NEVER invent numbers, field names, or values. Report exactly what the tools return.

Today is {today}.

This city's dataset has these fields (name, kind, and for categorical fields the number of distinct values):
{catalog}

How to work:
- You may filter, count, group, and search over ANY field above. Always use the EXACT field names shown.
- The catalog inlines the exact values for each categorical field — pick the matching value directly from it. For a categorical field shown with only a count (a large one), call list_values(field) to see its values. Never invent a value.
- Matching is exact and case-sensitive. Categorical values may include case or spelling variants of the same thing (e.g. "BURBANK", "Burbank", "burbank", "Burbank "). When the user names a value, filter with op "in" over EVERY matching variant you see in the catalog, not just one, so dirty data doesn't cause an undercount.
- count(filters, group_by, group_by_time) for "how many" and breakdowns. stats(field, filters, group_by) for sum/avg/min/max of a numeric field (use it for average/total/highest/lowest). search(filters, query, limit, sort) to list records — use sort to get the highest/lowest/most-recent. get_record(id_field, id_value) for one record by id.
- `filters` is a list of {{"field","op","value"}}. op is one of: eq (exact), in (value is a list), range (value is [from, to]; for date fields use YYYY / YYYY-MM / YYYY-MM-DD), contains (whole-word/substring match on a text field).
- Multi-hop questions: chain tool calls — e.g. look up the exact value with list_values, then count grouped by another field, then a second count, then compare in your answer.
- group_by works only on CATEGORICAL fields. For a breakdown by TIME (per year/quarter/month), use count with group_by_time={{"field","interval"}} on a date field — one call returns every period. For just one or two specific periods, a date-range filter per count is also fine.
- If a tool returns an error, fix the field/value and retry. Be concise and precise; if a count is 0, say there are none.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "list_values",
        "description": "List the distinct values (with counts) of a categorical field. Optionally filter with `contains`.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string"},
            "contains": {"type": "string", "description": "optional case-insensitive substring filter"},
        }, "required": ["field"]},
    }},
    {"type": "function", "function": {
        "name": "count",
        "description": "Exact count of records matching the filters, with an optional group_by breakdown (group_by must be a categorical field).",
        "parameters": {"type": "object", "properties": {
            "filters": {"type": "array", "description": "list of {field, op, value}", "items": {
                "type": "object", "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string", "enum": ["eq", "in", "range", "contains"]},
                    "value": {"description": "a string for eq/contains, [from,to] for range, or a list for in"},
                }, "required": ["field", "value"]}},
            "group_by": {"type": "string", "description": "a categorical field to break the count down by"},
            "group_by_time": {"type": "object", "description": "break the count into time periods on a DATE field (one call returns every period)",
                "properties": {
                    "field": {"type": "string"},
                    "interval": {"type": "string", "enum": ["year", "quarter", "month", "day"]},
                    "start": {"type": "string", "description": "optional start year YYYY"},
                    "end": {"type": "string", "description": "optional end year YYYY"},
                }, "required": ["field", "interval"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "stats",
        "description": "Aggregate a NUMERIC field: sum, average (avg), min, max — optionally broken down by a categorical field. Use for 'average/total/highest/lowest' questions.",
        "parameters": {"type": "object", "properties": {
            "field": {"type": "string", "description": "a numeric field"},
            "filters": {"type": "array", "items": {
                "type": "object", "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string", "enum": ["eq", "in", "range", "contains"]},
                    "value": {"description": "a string for eq/contains, [from,to] for range, or a list for in"},
                }, "required": ["field", "value"]}},
            "group_by": {"type": "string", "description": "optional categorical field to break the stats down by"},
        }, "required": ["field"]},
    }},
    {"type": "function", "function": {
        "name": "search",
        "description": "List matching records (first `limit` plus the true total). Use for 'show/list' questions.",
        "parameters": {"type": "object", "properties": {
            "filters": {"type": "array", "items": {
                "type": "object", "properties": {
                    "field": {"type": "string"},
                    "op": {"type": "string", "enum": ["eq", "in", "range", "contains"]},
                    "value": {"description": "a string for eq/contains, [from,to] for range, or a list for in"},
                }, "required": ["field", "value"]}},
            "query": {"type": "string", "description": "free-text keywords (e.g. an applicant/owner name)"},
            "limit": {"type": "integer"},
            "sort": {"type": "object", "description": "order results, e.g. highest valuation or most recent first",
                "properties": {"field": {"type": "string"}, "dir": {"type": "string", "enum": ["asc", "desc"]}},
                "required": ["field"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "get_record",
        "description": "Look up a single record by an id field and value, e.g. get_record('act_nbr','BS2504744').",
        "parameters": {"type": "object", "properties": {
            "id_field": {"type": "string"},
            "id_value": {"type": "string"},
        }, "required": ["id_field", "id_value"]},
    }},
]


async def _dispatch(name, args, base):
    if name == "list_values":
        return await ic.list_values(base, args.get("field"), args.get("contains"))
    if name == "count":
        return await ic.count(base, args.get("filters"), args.get("group_by"), args.get("group_by_time"))
    if name == "stats":
        return await ic.stats(base, args.get("field"), args.get("filters"), args.get("group_by"))
    if name == "search":
        return await ic.search(base, args.get("filters"), args.get("query"),
                                args.get("limit", 12), args.get("sort"))
    if name == "get_record":
        return await ic.get_record(base, args.get("id_field"), args.get("id_value"))
    return {"error": f"unknown tool {name}"}


async def answer_internals_query(user_query, client, model, base=None, history=None):
    """Run the schema-grounded tool loop + numeric self-check; return the final text answer."""
    base = base or DEFAULT_BASE
    schema = await sc.get_schema(base)                       # bootstrap/cache the collection schema
    system = SYSTEM.format(today=datetime.date.today().isoformat(),
                           catalog=sc.compact_catalog(schema))
    messages = [{"role": "system", "content": system}]
    messages += history if history else [{"role": "user", "content": user_query}]

    transcript = []                                          # (tool, args, result) for the verifier
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            answer = msg.content or "I couldn't find that in the data."
            return await vf.verify(answer, transcript, user_query, client, model)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = await _dispatch(tc.function.name, args, base)
            except Exception as e:
                logging.exception("internals tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            transcript.append((tc.function.name, args, result))
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, default=str)})
    return "Sorry, I couldn't complete that query. Please try rephrasing."
