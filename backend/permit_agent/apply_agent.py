"""
Instant-permit APPLY agent (conversational, runs inside the chatbot).

This is the multi-turn apply flow moved off the CLI into the stateless chat request loop.
Key ideas (all decided in design):
  - No variable store. The collected fields live in the CONVERSATION HISTORY; the LLM
    re-reads them each turn. We pass the recent history in every call.
  - A hidden flow tag rides in the assistant message's `context` so cheap code in app.py
    knows we're mid-flow and can skip the classifier. This module just reports, via the
    returned `in_flow` boolean, whether the flow is still active.
  - Human-in-the-loop CONFIRM gate lives in CODE (handle_submit), not the model's
    discretion: the write only happens when the user's message is an explicit CONFIRM and
    the fields re-validate.
  - Exit: the same LLM we already run detects "user changed topic" and hands control back
    (in_flow=False) instead of trapping them in the flow.

Entry point:
  answer_apply_query(history, client, model) -> {"reply": str, "in_flow": bool}
"""

import json
import logging
import re

from backend.permit_agent import apply_client as api

MAX_STEPS = 5

# Friendly work-type -> MEPP subTypeId (mirrors apply_client.MEPP_SUBTYPES).
WORK_TYPES = {
    "HVAC": 1,
    "House Rewire": 2,
    "Panel Upgrade": 3,
    "Water Heater Replacement": 4,
    "House Repipe": 5,
}

SYSTEM = """You are the City of Buena Park's instant-permit assistant, running inside a chat. You help a resident file an over-the-counter permit for one of these jobs (MEPP): HVAC, House Rewire, Panel Upgrade, Water Heater Replacement, or House Repipe.

Collect, asking for what's missing one or two items at a time (do not dump the whole list):
- which job (one of the five work types above)
- the property address where the work will be done
- applicant name, email, phone
- estimated job cost (valuation)
- a short description of the work (if the user doesn't give one, default it to the job name, e.g. "Water heater replacement")

Rules:
- Use ONLY what the user tells you. Never invent a value.
- Figure out the work type from what the user says (e.g. "my water heater broke" -> Water Heater Replacement). If it's unclear, call show_widget("work_type") and add a one-line prompt so they can pick it.
- When you need the property address, call show_widget("address") and add a one-line prompt so they can search and select it. Once they give an address, call lookup_address to verify it; if not found, ask for a corrected address.
- To collect the applicant's name, email, phone and the estimated job cost together, call show_widget("contact") and add a one-line prompt. (Default the work description to the job name unless the user gives one.)
- When every field is collected, call review_application (this shows the applicant a summary and the fee; it does NOT submit).
- After the review is shown, the user must reply with the word CONFIRM to submit. When they do, call submit_application. Do not claim it is submitted yourself.
- Instant online filing is available ONLY for those five jobs. If the user wants a permit that is NOT one of them (or says theirs isn't listed), do NOT call leave_flow. Reply briefly that instant online filing currently covers only those five job types (HVAC, House Rewire, Panel Upgrade, Water Heater Replacement, House Repipe), and for any other permit they can use the city's permit page or contact the permit office.
- Only call leave_flow when the user clearly switches to an UNRELATED topic (a different city question, general chit-chat), never as a way to handle an unsupported permit type. Do not force them to keep applying.
- Be concise and friendly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_address",
        "description": "Verify a property address against the city system. Call as soon as the user gives an address.",
        "parameters": {"type": "object", "properties": {"address": {"type": "string"}}, "required": ["address"]},
    }},
    {"type": "function", "function": {
        "name": "review_application",
        "description": "Validate all fields, resolve the address, compute the fee, and show the applicant a review. Does NOT submit. Call when every field is collected.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"},
            "property_address": {"type": "string"},
            "work_type": {"type": "string", "enum": ["HVAC", "House Rewire", "Panel Upgrade",
                          "Water Heater Replacement", "House Repipe"]},
            "valuation": {"type": "number", "description": "estimated job cost in dollars"},
            "description": {"type": "string"},
        }, "required": ["name", "email", "phone", "property_address", "work_type",
                        "valuation", "description"]},
    }},
    {"type": "function", "function": {
        "name": "submit_application",
        "description": "Submit the permit. Call ONLY after the review was shown and the user replied CONFIRM. Pass the same collected values.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"},
            "property_address": {"type": "string"},
            "work_type": {"type": "string", "enum": ["HVAC", "House Rewire", "Panel Upgrade",
                          "Water Heater Replacement", "House Repipe"]},
            "valuation": {"type": "number"},
            "description": {"type": "string"},
        }, "required": ["name", "email", "phone", "property_address", "work_type",
                        "valuation", "description"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user has stopped applying and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "show_widget",
        "description": "Surface a UI element for the next input instead of asking in plain text. 'work_type' = tappable job chips; 'address' = address search/select; 'contact' = a form for name, email, phone, and estimated job cost. Always add a short one-line prompt in your reply alongside it.",
        "parameters": {"type": "object", "properties": {
            "widget": {"type": "string", "enum": ["work_type", "address", "contact"]}},
            "required": ["widget"]},
    }},
]


def _widget_for(name, args, result):
    """Map a tool call to a UI widget payload the frontend renders. Returns None for tools
    that don't drive UI. The fixed five: chips, address_autocomplete, form, review, result."""
    if name == "show_widget":
        if args.get("widget") == "work_type":
            return {"type": "chips", "field": "work_type", "options": list(WORK_TYPES)}
        if args.get("widget") == "address":
            return {"type": "address_autocomplete", "field": "property_address"}
        if args.get("widget") == "contact":
            return {"type": "form", "field": "contact", "fields": [
                {"name": "name", "label": "Full name"},
                {"name": "email", "label": "Email", "inputType": "email"},
                {"name": "phone", "label": "Phone", "inputType": "tel"},
                {"name": "valuation", "label": "Estimated job cost ($)", "inputType": "number"},
            ]}
    if name == "review_application" and result.get("status") == "review":
        return {"type": "review", "confirm": True, "data": {
            "applicant": result.get("applicant"), "property": result.get("property"),
            "job": result.get("job"), "work": result.get("work"),
            "valuation": result.get("valuation"), "fee": result.get("fee"),
            "feeDetails": result.get("feeDetails")}}
    if name == "submit_application" and result.get("status") == "submitted":
        return {"type": "result", "permitNumber": result.get("permitNumber")}
    return None


def _validate(a):
    errs = []
    for f in ["name", "email", "phone", "property_address", "description"]:
        if not str(a.get(f, "")).strip():
            errs.append(f"missing {f}")
    if a.get("email") and not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", a["email"]):
        errs.append("invalid email")
    if not isinstance(a.get("valuation"), (int, float)) or a.get("valuation", 0) <= 0:
        errs.append("valuation must be greater than 0")
    if a.get("work_type") not in WORK_TYPES:
        errs.append("work_type must be one of: " + ", ".join(WORK_TYPES))
    return errs


async def _resolve(a):
    """Validate fields + resolve address + attach codes/lsoId. Returns (prop, error_dict)."""
    errs = _validate(a)
    if errs:
        return None, {"status": "invalid", "errors": errs}
    matches = await api.validate_address(a["property_address"])
    if not matches:
        return None, {"status": "address_not_found",
                      "message": f"Could not find '{a['property_address']}'. Ask the user to re-check it."}
    prop = matches[0]
    a["lsoId"] = prop["lsoId"]
    a["subTypeId"] = WORK_TYPES[a["work_type"]]
    return prop, None


async def handle_lookup(args):
    matches = await api.validate_address(args.get("address", ""))
    if not matches:
        return {"found": False, "message": "That address isn't in the city system. Ask the user to re-check it."}
    p = matches[0]
    return {"found": True, "resolved_address": str(p.get("address", "")).strip(), "lsoId": p.get("lsoId")}


async def handle_review(a):
    """Validate + address + fee, return a review for the model to show. NO write."""
    prop, err = await _resolve(a)
    if err:
        return err
    fee = await api.fee_estimate(a)
    return {
        "status": "review",
        "applicant": {"name": a["name"], "email": a["email"], "phone": a["phone"]},
        "property": str(prop["address"]).strip(),
        "job": a["work_type"], "work": a["description"],
        "valuation": a["valuation"],
        "fee": fee.get("totalFee"), "feeDetails": fee.get("feeDetails"),
        "mock": fee.get("mock", False),
        "message": "Show this review and the total fee, then ask the user to reply CONFIRM to submit.",
    }


async def handle_submit(a, user_confirmed):
    """CONFIRM gate lives HERE, in code. Only writes when the user explicitly confirmed
    and the fields re-validate. The model calling this tool is not enough on its own."""
    if not user_confirmed:
        return {"status": "not_confirmed",
                "message": "The user has not typed CONFIRM yet. Show the review and wait for CONFIRM. Do not submit."}
    prop, err = await _resolve(a)  # re-validate + re-resolve, don't trust prior turns
    if err:
        return err
    result = await api.add_permit(a)
    pn = result.get("permitNumber")
    if not pn:  # the API rejected it (404/validation/etc.) -> do NOT claim success
        return {"status": "submit_failed", "api_message": result.get("message"),
                "message": "The permit could NOT be created. Tell the user the submission "
                           "failed and to try again shortly; do not claim it succeeded."}
    return {"status": "submitted", "permitNumber": pn, "mock": result.get("mock", False),
            "message": "Permit created. Tell the user the permit number and that they finish by paying the fee on the city portal."}


def _user_confirmed(history):
    """Deterministic check: is the latest user message an explicit CONFIRM?"""
    for m in reversed(history):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip().upper() == "CONFIRM"
    return False


async def _dispatch(name, args, history):
    if name == "lookup_address":
        return await handle_lookup(args)
    if name == "review_application":
        return await handle_review(args)
    if name == "submit_application":
        return await handle_submit(args, _user_confirmed(history))
    if name == "leave_flow":
        return {"status": "left"}
    if name == "show_widget":
        return {"status": "widget_shown"}
    return {"error": f"unknown tool {name}"}


async def answer_apply_query(history, client, model):
    """Run the apply tool loop over the conversation history.

    Returns {"reply", "in_flow", "left"}.
      - in_flow=False  -> the flow ended; app.py drops the tag.
      - left=True      -> the user changed topic; app.py should NOT use `reply` and instead
                          let the same turn fall through to normal routing (RAG/read agent).
    """
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
                result = await _dispatch(tc.function.name, args, history)
            except Exception as e:
                logging.exception("apply tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            w = _widget_for(tc.function.name, args, result)
            if w:
                widget = w
            if tc.function.name == "leave_flow":
                left = True
            if tc.function.name == "submit_application" and result.get("status") == "submitted":
                submitted = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if left:  # stop the loop immediately; the turn falls through to normal routing
            return {"reply": "", "in_flow": False, "left": True, "widget": None}
    return {"reply": "Sorry, let's try that again.", "in_flow": not (left or submitted),
            "left": left, "widget": widget}
