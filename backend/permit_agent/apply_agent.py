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
import os
import re

from backend.permit_agent import apply_client as api

MAX_STEPS = 5

# New installations are NOT instant-filable; the flow redirects them here. City-specific
# and must be VERIFIED before go-live, set BUILDING_SAFETY_CONTACT in env to the real
# office (e.g. "Building & Safety at (555) 123-4567 or permits@city.gov").
BUILDING_SAFETY_CONTACT = os.environ.get("BUILDING_SAFETY_CONTACT", "the city's Building & Safety office")

# Friendly work-type -> MEPP subTypeId (mirrors apply_client.MEPP_SUBTYPES).
WORK_TYPES = {
    "HVAC": 1,
    "House Rewire": 2,
    "Panel Upgrade": 3,
    "Water Heater Replacement": 4,
    "House Repipe": 5,
}

SYSTEM = """You are the City of Burbank's instant-permit assistant, running inside a chat. You help a resident file an over-the-counter permit for one of these jobs (MEPP): HVAC, House Rewire, Panel Upgrade, Water Heater Replacement, or House Repipe.

How you work, every single turn:
1. FIRST call set_fields with EVERY value you have gathered from the whole conversation so far (omit what you don't have yet). Figure out the work type from the user's words (e.g. "my water heater broke" -> Water Heater Replacement). Use only what the user actually said; never invent a value.
2. The tool result tells you what is still needed. Then write ONE short, friendly line asking for just that next item. Do NOT list all the fields, an on-screen widget handles the input, so keep your message to a single prompt.

Eligibility (very important):
- Instant permits cover REPLACEMENTS of an existing unit only, NOT brand-new installations. Right after the work type, confirm this and report it in set_fields as `replacement` ("replacement" for replacing an existing unit, "new" for a new installation). Map the buttons "Replacing existing" -> replacement and "New installation" -> new.
- If it is a new installation, the tool result will tell you it is ineligible; deliver that redirect message and stop. Do not collect any more fields.

Other rules:
- The system shows the review with the fee automatically once everything is collected. When it does, the user replies CONFIRM; then call submit_application with all the values. Never claim it is submitted yourself.
- Instant online filing covers ONLY those five jobs. If the user wants a permit that is NOT one of them, briefly say instant filing covers only those five and point them to the city's permit page or the permit office. Do NOT call leave_flow for that.
- Only call leave_flow if the user clearly switches to an UNRELATED topic.
- Be concise and friendly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "set_fields",
        "description": "Call this on EVERY turn. Report every application value you have gathered from the whole conversation so far (omit what you don't have yet). The system uses it to decide what to ask next and which on-screen widget to show. After calling it, write only a short one-line prompt for the next item.",
        "parameters": {"type": "object", "properties": {
            "work_type": {"type": "string", "enum": ["HVAC", "House Rewire", "Panel Upgrade",
                          "Water Heater Replacement", "House Repipe"]},
            "replacement": {"type": "string", "enum": ["replacement", "new"],
                            "description": "'replacement' if the job replaces an existing unit, 'new' if it is a brand-new installation. Map button 'Replacing existing' -> replacement, 'New installation' -> new."},
            "property_address": {"type": "string"},
            "name": {"type": "string"}, "email": {"type": "string"}, "phone": {"type": "string"},
            "valuation": {"type": "number", "description": "estimated job cost in dollars"},
        }},
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
        }, "required": ["name", "email", "phone", "property_address", "work_type", "valuation"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user has stopped applying and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

# The four collected fields shown together as a form once work type + address are set.
_CONTACT_FIELDS = [
    {"name": "name", "label": "Full name"},
    {"name": "email", "label": "Email", "inputType": "email"},
    {"name": "phone", "label": "Phone", "inputType": "tel"},
    {"name": "valuation", "label": "Estimated job cost ($)", "inputType": "number"},
]


def _contact_widget(a):
    """Contact form prefilled with whatever's already entered, so a re-show (e.g. after a bad
    phone) doesn't force the user to retype the fields that were fine."""
    values = {f["name"]: a.get(f["name"]) for f in _CONTACT_FIELDS if a.get(f["name"]) not in (None, "")}
    return {"type": "form", "field": "contact", "fields": _CONTACT_FIELDS, "values": values}


async def handle_set_fields(a):
    """Deterministic engine: from the fields gathered so far, pick the next step + the widget
    to show. The '_widget' key is popped in the loop before the model sees the result, so the
    model never decides which widget appears, only what to say."""
    if a.get("work_type") not in WORK_TYPES:
        return {"need": "work_type",
                "_widget": {"type": "chips", "field": "work_type", "options": list(WORK_TYPES)},
                "message": "Ask which of the five jobs it is; they can tap it."}
    # Eligibility gate: instant permits are replacement-only. Ask before collecting anything else.
    if a.get("replacement") not in ("replacement", "new"):
        return {"need": "replacement",
                "_widget": {"type": "chips", "field": "replacement",
                            "options": ["Replacing existing", "New installation"]},
                "message": "Ask whether this replaces an existing unit or is a new installation; they can tap it."}
    if a.get("replacement") == "new":
        return {"status": "ineligible",
                "message": ("Tell the user that instant permits cover replacements only, and that for a "
                            f"brand-new installation they should contact {BUILDING_SAFETY_CONTACT}. "
                            "Then stop and do not collect anything else.")}
    if not str(a.get("property_address", "")).strip():
        return {"need": "address",
                "_widget": {"type": "address_autocomplete", "field": "property_address"},
                "message": "Ask for the property address; they can search and pick it."}
    matches = await api.validate_address(a["property_address"])
    if not matches:
        return {"need": "address",
                "_widget": {"type": "address_autocomplete", "field": "property_address"},
                "message": f"Could not find '{a['property_address']}'. Ask them to pick a valid address."}
    a["lsoId"] = matches[0]["lsoId"]
    if [f["name"] for f in _CONTACT_FIELDS if not str(a.get(f["name"]) or "").strip()]:
        return {"need": "contact",
                "_widget": _contact_widget(a),
                "message": "Ask for their name, email, phone, and the estimated job cost."}
    if not str(a.get("description") or "").strip():
        a["description"] = a["work_type"]
    errs = _validate(a)
    if errs:
        return {"need": "contact",
                "_widget": _contact_widget(a),
                "message": "Some details need fixing (" + ", ".join(errs) + "). Ask them to re-enter them."}
    a["subTypeId"] = WORK_TYPES[a["work_type"]]
    fee = await api.fee_estimate(a)
    return {"status": "review",
            "_widget": {"type": "review", "confirm": True, "data": {
                "applicant": {"name": a["name"], "email": a["email"], "phone": a["phone"]},
                "property": str(matches[0]["address"]).strip(),
                "job": a["work_type"], "work": a.get("description"),
                "valuation": a["valuation"], "fee": fee.get("totalFee"),
                "feeDetails": fee.get("feeDetails")}},
            "message": "Show the review and the total fee, then ask them to reply CONFIRM to submit."}


def _validate(a):
    errs = []
    for f in ["name", "email", "phone", "property_address"]:
        if not str(a.get(f, "")).strip():
            errs.append(f"missing {f}")
    if a.get("email") and not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", a["email"]):
        errs.append("email is not a valid address")
    if a.get("phone") and len(re.sub(r"\D", "", str(a["phone"]))) != 10:
        errs.append("phone must be 10 digits")
    if not isinstance(a.get("valuation"), (int, float)) or a.get("valuation", 0) <= 0:
        errs.append("valuation must be greater than 0")
    if a.get("work_type") not in WORK_TYPES:
        errs.append("work_type must be one of: " + ", ".join(WORK_TYPES))
    return errs


async def _resolve(a):
    """Validate fields + resolve address + attach codes/lsoId. Returns (prop, error_dict)."""
    if not str(a.get("description") or "").strip():
        a["description"] = a.get("work_type", "")   # description is never asked; default it
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
    fee = await api.fee_estimate(a)
    return {"status": "submitted", "permitNumber": pn, "mock": result.get("mock", False),
            "fee": fee.get("totalFee"),
            "message": "Permit created. Tell the user the permit number and that the last step is to pay the fee shown below."}


def _user_confirmed(history):
    """Deterministic check: is the latest user message an explicit CONFIRM?"""
    for m in reversed(history):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"].strip().upper() == "CONFIRM"
    return False


async def _dispatch(name, args, history):
    if name == "set_fields":
        return await handle_set_fields(args)
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
    ended = False          # eligibility gate (new installation) ended the flow this turn
    widget = None
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"reply": msg.content or "", "in_flow": not (left or submitted or ended),
                    "left": left, "widget": widget}
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = await _dispatch(tc.function.name, args, history)
            except Exception as e:
                logging.exception("apply tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            if isinstance(result, dict) and "_widget" in result:
                widget = result.pop("_widget")           # code-decided widget (set_fields)
            if isinstance(result, dict) and result.get("status") == "ineligible":
                ended = True                              # new install: redirect + end the flow
            if tc.function.name == "leave_flow":
                left = True
            if tc.function.name == "submit_application" and result.get("status") == "submitted":
                # Emit the PAY step and STAY in the flow (do not set submitted). Payment is a
                # mock handled next turn in app.py, which then shows the result card.
                widget = {"type": "pay", "permitNumber": result.get("permitNumber"),
                          "amount": result.get("fee"), "email": args.get("email")}
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if left:  # stop the loop immediately; the turn falls through to normal routing
            return {"reply": "", "in_flow": False, "left": True, "widget": None}
    return {"reply": "Sorry, let's try that again.", "in_flow": not (left or submitted),
            "left": left, "widget": widget}
