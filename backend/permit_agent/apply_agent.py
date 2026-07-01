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

SYSTEM = """You are the City of Burbank's instant solar-permit assistant, running inside a chat. You help someone apply for a solar permit (residential or commercial; panels, optionally batteries).

Collect, asking for what's missing one or two items at a time (do not dump the whole list):
- applicant name, email, phone
- the property address where the work will be done
- residential or commercial
- estimated job cost (valuation), number of panels, number of batteries (0 if none)
- a short description of the work

Rules:
- Use ONLY what the user tells you. Never invent a value.
- As soon as the user gives the property address, call lookup_address to verify it. If not found, ask for a corrected address.
- When every field is collected, call review_application (this shows the applicant a summary and the fee; it does NOT submit).
- After the review is shown, the user must reply with the word CONFIRM to submit. When they do, call submit_application. Do not claim it is submitted yourself.
- If at any point the user clearly stops applying and asks something unrelated (a different city question, a general question), call leave_flow so the assistant can hand them back to normal help. Do not force them to keep applying.
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
            "property_address": {"type": "string"}, "valuation": {"type": "number"},
            "unit": {"type": "integer", "description": "number of panels"},
            "noOfBatteries": {"type": "integer", "description": "0 if none"},
            "description": {"type": "string"},
            "property_type": {"type": "string", "enum": ["residential", "commercial"]},
        }, "required": ["name", "email", "phone", "property_address", "valuation", "unit",
                        "noOfBatteries", "description", "property_type"]},
    }},
    {"type": "function", "function": {
        "name": "submit_application",
        "description": "Submit the permit. Call ONLY after the review was shown and the user replied CONFIRM. Pass the same collected values.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"},
            "property_address": {"type": "string"}, "valuation": {"type": "number"},
            "unit": {"type": "integer"}, "noOfBatteries": {"type": "integer"},
            "description": {"type": "string"},
            "property_type": {"type": "string", "enum": ["residential", "commercial"]},
        }, "required": ["name", "email", "phone", "property_address", "valuation", "unit",
                        "noOfBatteries", "description", "property_type"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user has stopped applying and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


def _validate(a):
    errs = []
    for f in ["name", "email", "phone", "property_address", "description"]:
        if not str(a.get(f, "")).strip():
            errs.append(f"missing {f}")
    if a.get("email") and not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", a["email"]):
        errs.append("invalid email")
    if not isinstance(a.get("valuation"), (int, float)) or a.get("valuation", 0) <= 0:
        errs.append("valuation must be greater than 0")
    for f in ["unit", "noOfBatteries"]:
        if not isinstance(a.get(f), int) or a.get(f, -1) < 0:
            errs.append(f"{f} must be a non-negative whole number")
    if a.get("property_type") not in ("residential", "commercial"):
        errs.append("property_type must be 'residential' or 'commercial'")
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
    a["subTypeId"] = api.PERMIT_CODES["subTypeIds"][a["property_type"]]
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
        "work": a["description"], "panels": a["unit"], "batteries": a["noOfBatteries"],
        "valuation": a["valuation"], "type": a["property_type"],
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
    return {"status": "submitted", "permitNumber": result.get("permitNumber"),
            "mock": result.get("mock", False),
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
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"reply": msg.content or "", "in_flow": not (left or submitted), "left": left}
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = await _dispatch(tc.function.name, args, history)
            except Exception as e:
                logging.exception("apply tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            if tc.function.name == "leave_flow":
                left = True
            if tc.function.name == "submit_application" and result.get("status") == "submitted":
                submitted = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if left:  # stop the loop immediately; the turn falls through to normal routing
            return {"reply": "", "in_flow": False, "left": True}
    return {"reply": "Sorry, let's try that again.", "in_flow": not (left or submitted), "left": left}
