"""
Read-permits agent.

A tool-calling loop that answers a resident's question about EXISTING permit records
(counts, lists, lookups) by querying the permits index, instead of RAG. It is read-only:
no writes, no auth, no confirmation gate.

Entry point: answer_permit_query(user_query, client, model) -> answer string.
`client` is the app's async AzureOpenAI client; `model` is the deployment name.

The model drives four tools (see TOOLS). The loop runs until the model stops calling
tools and returns prose, or MAX_STEPS is hit.
"""

import datetime
import json
import logging

from backend.permit_agent import permit_client as pc

MAX_STEPS = 6

SYSTEM = """You are the City of Burbank permits assistant. You answer questions about EXISTING permit records: how many, lists, and single-permit lookups. You are read-only and only report what the tools return. Never invent a number, type, status, or permit.

Today is {today}.

How to work:
- When the user names a kind of permit in words (e.g. "solar", "ADU", "pool", "electrical"), call find_permit_type FIRST to get the exact stored type value, then use that exact value in count_permits / search_permits. If find_permit_type returns nothing, tell the user you couldn't find that permit type and ask them to rephrase; do NOT guess a type.
- "how many ..." -> count_permits. "show / list / which permits ..." or anything tied to an address -> search_permits. A specific permit number (like BS2504744) -> get_permit.
- Business questions use the module filter, NOT find_permit_type (a business isn't a single permit type). For "new businesses" / "businesses opened" / "business tax", use module="BUSINESS TAX". For "business license(s)", use module="BUSINESS LICENSE". "Opened/registered" means applied (use date_field "applied").
- Dates: default date_field is "applied" (filed/submitted). Use "issued" for issued/approved, "final" for completed/finaled. For "this month" use {month_start} to {month_end}; for "this year" use {year}-01-01 to {year}-12-31. Pass dates as YYYY-MM-DD.
- search_permits returns up to 12 records plus the true total. When you list them, show those (address, type, status, date) and then state the total, e.g. "Showing 12 of 47 permits at that address."
- Use group_by on count_permits when the user wants a breakdown (by status, type, or department).
- Be concise. Give the number or the list plainly. If a result is 0, say there are none.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_permit_type",
            "description": "Resolve a user's word (e.g. 'solar', 'ADU', 'pool') to the exact stored permit type value(s) with their counts. Call this before filtering by type.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_permits",
            "description": "Exact count of permits matching the filters. Optionally group_by status/type/department for a breakdown. Use the exact type value from find_permit_type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "description": "exact stored type value"},
                    "status": {"type": "string"},
                    "department": {"type": "string"},
                    "module": {"type": "string", "enum": ["BUSINESS TAX", "BUSINESS LICENSE", "BUILDING", "PUBLIC WORKS", "CODE ENFORCEMENT", "PLANNING", "PARKING", "HOUSING"],
                               "description": "high-level category; use 'BUSINESS TAX' for new businesses, 'BUSINESS LICENSE' for business licenses"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created"]},
                    "date_from": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "date_to": {"type": "string", "description": "YYYY-MM-DD / YYYY-MM / YYYY"},
                    "group_by": {"type": "string", "enum": ["status", "type", "department"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_permits",
            "description": "List matching permits (first 12 plus the true total). Use for 'show/list' questions and anything tied to an address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "street address to match"},
                    "type": {"type": "string", "description": "exact stored type value"},
                    "status": {"type": "string"},
                    "module": {"type": "string", "enum": ["BUSINESS TAX", "BUSINESS LICENSE", "BUILDING", "PUBLIC WORKS", "CODE ENFORCEMENT", "PLANNING", "PARKING", "HOUSING"],
                               "description": "high-level category; 'BUSINESS TAX' for businesses, 'BUSINESS LICENSE' for business licenses"},
                    "query": {"type": "string", "description": "free-text keywords (applicant name, etc.)"},
                    "date_field": {"type": "string", "enum": ["applied", "issued", "final", "updated", "created"]},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_permit",
            "description": "Look up one permit by its number, e.g. BS2504744.",
            "parameters": {
                "type": "object",
                "properties": {"act_nbr": {"type": "string"}},
                "required": ["act_nbr"],
            },
        },
    },
]


async def _dispatch(name, args):
    if name == "find_permit_type":
        return {"matches": await pc.find_permit_type(args.get("keyword", ""))}
    if name == "count_permits":
        return await pc.count(
            type=args.get("type"), status=args.get("status"), department=args.get("department"),
            module=args.get("module"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
            group_by=args.get("group_by"),
        )
    if name == "search_permits":
        return await pc.search(
            query=args.get("query"), address=args.get("address"),
            type=args.get("type"), status=args.get("status"), module=args.get("module"),
            date_field=args.get("date_field", "applied"),
            date_from=args.get("date_from"), date_to=args.get("date_to"),
            limit=12,
        )
    if name == "get_permit":
        return await pc.get_permit(args.get("act_nbr", ""))
    return {"error": f"unknown tool {name}"}


def _system_prompt():
    today = datetime.date.today()
    first = today.replace(day=1)
    import calendar
    last = today.replace(day=calendar.monthrange(today.year, today.month)[1])
    return SYSTEM.format(
        today=today.isoformat(), year=today.year,
        month_start=first.isoformat(), month_end=last.isoformat(),
    )


async def answer_permit_query(user_query, client, model):
    """Run the tool loop and return the agent's final text answer."""
    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_query},
    ]
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return msg.content or "I couldn't find that in the permit records."
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = await _dispatch(tc.function.name, args)
            except Exception as e:
                logging.exception("permit tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "Sorry, I couldn't complete that permit lookup. Please try rephrasing."
