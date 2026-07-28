"""
Public Record Request (PRR) agent — fully MOCK, single-window intake.

Shape mirrors the instant-permit apply agent (a tool-calling loop, a code-decided widget, a
CONFIRM gate in code), but nothing is filed: submit returns a mock CPRA reference number.

Flow: app.py detects the PRR keyword and makes an OFFER (Yes/No chips). On "yes" this agent
runs and shows ONE form window collecting everything at once (records, name, email); the user
submits it, reviews, and replies CONFIRM. No back-and-forth questioning, no RAG, no external
API (so no :9080 dependency).

Entry point:
  answer_prr_query(history, client, model) -> {"reply", "in_flow", "widget", "left"}
"""

import hashlib
import json
import logging
import re

MAX_STEPS = 5

SYSTEM = """You are the City's public-records assistant, running inside a chat. You help a resident file a California Public Records Act (CPRA) request. Nothing is actually filed here; this is a guided intake that produces a reference number.

The resident fills ONE on-screen form with all the details at once: what records they want, their name, and their email. You do NOT ask for these one at a time.

How you work, every single turn:
1. FIRST call set_fields with EVERY value from the form/conversation so far (omit what you don't have yet):
   - records_description: what records the person wants, in their own words, including any address/date-range/permit numbers.
   - name: the requester's full name.
   - email: the email the City should respond to.
   Use only what the user actually provided; never invent a value.
2. The tool result tells you what to do next. If the form still needs filling, write ONE short friendly line pointing them to the form below (do not read the fields aloud). If the review is shown, ask them to reply CONFIRM.

Rules:
- The system shows the review automatically once everything is collected. When it does, the user replies CONFIRM; then call submit_request. Never claim it is submitted yourself.
- Only call leave_flow if the user clearly switches to an UNRELATED topic.
- Be concise and friendly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "set_fields",
        "description": "Call this on EVERY turn. Report every value gathered so far (omit what you don't have). The system uses it to decide whether to show the form or the review.",
        "parameters": {"type": "object", "properties": {
            "records_description": {"type": "string",
                "description": "what records the person wants, in their own words, including any address/date-range/permit numbers"},
            "name": {"type": "string"},
            "email": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "submit_request",
        "description": "Submit the records request. Call ONLY after the review was shown and the user replied CONFIRM. Pass the same collected values.",
        "parameters": {"type": "object", "properties": {
            "records_description": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"},
        }, "required": ["records_description", "name", "email"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user stopped and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

# One window collects everything. records_description first so it reads top-to-bottom.
_PRR_FIELDS = [
    {"name": "records_description", "label": "What records are you looking for? (include address, dates, permit #s)"},
    {"name": "name", "label": "Full name"},
    {"name": "email", "label": "Email", "inputType": "email"},
]


def _prr_form(a):
    """The single intake form, prefilled with whatever's already entered so a re-show (e.g.
    after a bad email) doesn't force a retype."""
    values = {f["name"]: a.get(f["name"]) for f in _PRR_FIELDS if a.get(f["name"]) not in (None, "")}
    return {"type": "form", "field": "prr", "fields": _PRR_FIELDS, "values": values}


def _valid_email(e):
    return bool(e) and bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(e)))


def handle_set_fields(a):
    """Deterministic engine: show the form until everything's in, then the review. The
    '_widget' key is popped before the model sees the result, so the model never picks the
    widget, only what to say."""
    missing = [f for f in _PRR_FIELDS if not str(a.get(f["name"]) or "").strip()]
    if missing or not _valid_email(a.get("email")):
        msg = "Point them to the form below to enter their request."
        if not missing and not _valid_email(a.get("email")):
            msg = "That email doesn't look valid. Ask them to re-check the form below."
        return {"need": "details", "_widget": _prr_form(a), "message": msg}
    return {"status": "review",
            "_widget": {"type": "review", "confirm": True, "confirmLabel": "Submit request",
                        "title": "Review your records request",
                        "rows": [
                            {"label": "Records", "value": a["records_description"]},
                            {"label": "Requester", "value": a["name"]},
                            {"label": "Email", "value": a["email"]},
                        ]},
            "message": "Show the review and ask them to reply CONFIRM to submit the request."}


def _mock_reference(a):
    seed = f"{a.get('records_description','')}|{a.get('email','')}"
    n = int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 10000
    return f"PRR-2026-{n:04d}"


def handle_submit(a, user_confirmed):
    """CONFIRM gate in code: only 'submit' when the user explicitly confirmed."""
    if not user_confirmed:
        return {"status": "not_confirmed",
                "message": "The user has not typed CONFIRM yet. Show the review and wait. Do not submit."}
    if not str(a.get("records_description", "")).strip() or not _valid_email(a.get("email")):
        return {"status": "invalid", "message": "Details are incomplete. Ask them to re-check the request."}
    ref = _mock_reference(a)
    return {"status": "submitted", "reference": ref,
            "_widget": {"type": "result", "title": "Request submitted ✅",
                        "refLabel": "Reference number", "reference": ref, "download": False,
                        "message": "Under the CPRA the City Clerk has up to 10 days to respond, and will follow up by email."},
            "message": ("Tell the user the request was submitted, give the reference number, and note the "
                        "City Clerk will respond by email (up to 10 days under the CPRA). Then stop.")}


def _user_confirmed(history):
    for m in reversed(history):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip().upper() == "CONFIRM"
    return False


def _dispatch(name, args, history):
    if name == "set_fields":
        return handle_set_fields(args)
    if name == "submit_request":
        return handle_submit(args, _user_confirmed(history))
    if name == "leave_flow":
        return {"status": "left"}
    return {"error": f"unknown tool {name}"}


async def answer_prr_query(history, client, model):
    """Run the PRR tool loop over the conversation history. Returns {reply, in_flow, widget, left}."""
    messages = [{"role": "system", "content": SYSTEM}] + list(history)
    left = False
    submitted = False
    widget = None
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"reply": msg.content or "", "in_flow": not (left or submitted),
                    "left": left, "widget": widget}
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _dispatch(tc.function.name, args, history)
            except Exception as e:
                logging.exception("prr tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            if isinstance(result, dict) and "_widget" in result:
                widget = result.pop("_widget")
            if tc.function.name == "leave_flow":
                left = True
            if tc.function.name == "submit_request" and result.get("status") == "submitted":
                submitted = True                       # terminal: no pay step, flow ends here
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if left:
            return {"reply": "", "in_flow": False, "left": True, "widget": None}
    return {"reply": "Sorry, let's try that again.", "in_flow": not (left or submitted),
            "left": left, "widget": widget}
