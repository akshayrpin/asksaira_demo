"""
Inspection scheduling / rescheduling agent — fully MOCK, conversational.

Follows the pdf's Flow 3: one question at a time, adapts to free-text, proposes a concrete
slot, and books on acceptance. Nothing is really scheduled: booking returns a mock INS
confirmation number. No RAG, no external API (no :9080 dependency).

Booking gate lives in CODE (handle_book): the confirmation only happens once the user's latest
message accepts the proposed slot.

Entry point:
  answer_inspection_query(history, client, model) -> {"reply", "in_flow", "widget", "left"}
"""

import datetime
import hashlib
import json
import logging

MAX_STEPS = 6

INSPECTION_TYPES = ["Rough Plumbing", "Rough Electrical", "Framing", "Final"]

SYSTEM = """You are the City's inspection-scheduling assistant, running inside a chat. You help a resident schedule or reschedule a building inspection. Nothing is truly booked; this is a guided intake that produces a confirmation number.

Talk like a person: ask for ONE thing at a time and adapt to free-text answers ("mornings are better", "Thursday afternoon"). Do not dump a form.

How you work, every single turn:
1. FIRST call set_fields with EVERY value you have gathered so far (omit what you don't have yet):
   - action: "schedule" for a new booking, "reschedule" if they're moving an existing one.
   - permit_number: the permit the inspection is for.
   - inspection_type: one of Rough Plumbing, Rough Electrical, Framing, Final.
   - preferred_time: their day/time preference in their own words (e.g. "next week mornings", "Thursday afternoon").
   Use only what the user actually said; never invent a value.
2. The tool result tells you what is still needed, or gives you a proposed slot. Write ONE short, friendly line: ask for the next item, or propose the slot and ask if it works. An on-screen widget handles taps, so keep it to a single prompt.

Rules:
- The system proposes the concrete slot; when it does, offer it and let the user accept ("Works for me") or pick another time. If they pick another, ask for the specific day/time and update preferred_time.
- Only call book_inspection after the user accepts the proposed slot. Never claim it is booked yourself.
- Only call leave_flow if the user clearly switches to an UNRELATED topic.
- Be concise and friendly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "set_fields",
        "description": "Call this on EVERY turn. Report every value gathered so far (omit what you don't have). The system uses it to decide what to ask next or to propose a slot.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["schedule", "reschedule"]},
            "permit_number": {"type": "string"},
            "inspection_type": {"type": "string", "enum": INSPECTION_TYPES},
            "preferred_time": {"type": "string", "description": "the user's day/time preference in their own words"},
        }},
    }},
    {"type": "function", "function": {
        "name": "book_inspection",
        "description": "Book (or move) the inspection. Call ONLY after the user accepted the proposed slot. Pass the same collected values.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["schedule", "reschedule"]},
            "permit_number": {"type": "string"},
            "inspection_type": {"type": "string", "enum": INSPECTION_TYPES},
            "preferred_time": {"type": "string"},
        }, "required": ["permit_number", "inspection_type", "preferred_time"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user stopped and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}


def _next_weekday(today, target):
    ahead = (target - today.weekday()) % 7 or 7   # always strictly in the future
    return today + datetime.timedelta(days=ahead)


def _propose_slot(pref, today):
    """Turn a free-text preference into a concrete near-future weekday slot. Deterministic."""
    p = (pref or "").lower()
    afternoon = any(w in p for w in ["afternoon", "pm", "evening", "after lunch", "noon"])
    target = None
    for name, idx in _WEEKDAYS.items():
        if name in p:
            target = _next_weekday(today, idx)
            break
    if target is None:
        if "next week" in p:
            target = _next_weekday(today, 1)          # Tuesday next week
        elif "tomorrow" in p:
            target = today + datetime.timedelta(days=1)
        else:
            target = today + datetime.timedelta(days=2)
    while target.weekday() >= 5:                       # never a weekend
        target += datetime.timedelta(days=1)
    time_str = "1:30 PM" if afternoon else "9:00 AM"
    return f"{target.strftime('%A, %B')} {target.day} at {time_str}"


def handle_set_fields(a, today):
    """Deterministic engine: pick the next question, or propose the slot once everything's in."""
    if not str(a.get("permit_number", "")).strip():
        return {"need": "permit_number",
                "message": "Ask which permit number the inspection is for (a free-text reply)."}
    if a.get("inspection_type") not in INSPECTION_TYPES:
        return {"need": "inspection_type",
                "_widget": {"type": "chips", "field": "inspection_type", "options": INSPECTION_TYPES},
                "message": "Ask what kind of inspection they need; they can tap it."}
    if not str(a.get("preferred_time", "")).strip():
        note = " (they're moving an existing booking)" if a.get("action") == "reschedule" else ""
        return {"need": "preferred_time",
                "message": f"Ask what day and time works best{note} (a free-text reply)."}
    slot = _propose_slot(a["preferred_time"], today)
    return {"status": "propose", "slot": slot,
            "_widget": {"type": "chips", "field": "slot", "options": ["Works for me", "Pick another time"]},
            "message": f"Propose this exact slot: {slot}. Ask if it works; they can tap 'Works for me' or 'Pick another time'."}


_ACCEPT = {"works for me", "works", "yes", "confirm", "sounds good", "book it",
           "that works", "perfect", "great", "yes please"}


def _accepted(history):
    """Deterministic: did the user's latest message accept the proposed slot?"""
    for m in reversed(history):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            t = m["content"].strip().lower()
            return t.startswith("works") or t in _ACCEPT
    return False


def _mock_conf(a):
    seed = f"{a.get('permit_number','')}|{a.get('inspection_type','')}"
    n = int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 10000
    return f"INS-2026-{n:04d}"


def handle_book(a, accepted, today):
    """Booking gate in code: only 'book' once the user accepted the proposed slot."""
    if not accepted:
        return {"status": "not_accepted",
                "message": "The user hasn't accepted a time yet. Wait for them to accept or pick another. Do not book."}
    for f in ("permit_number", "inspection_type", "preferred_time"):
        if not str(a.get(f, "")).strip():
            return {"status": "invalid", "message": "Details are incomplete. Ask for the missing item."}
    slot = _propose_slot(a["preferred_time"], today)
    conf = _mock_conf(a)
    reschedule = a.get("action") == "reschedule"
    if reschedule:
        message = f"Moved to {slot}. Your confirmation number stays the same."
        title = "Rescheduled ✅"
    else:
        message = (f"{a['inspection_type']} inspection for {a['permit_number']} on {slot}. "
                   "You'll get a reminder the day before.")
        title = "Booked ✅"
    return {"status": "booked", "confirmation": conf,
            "_widget": {"type": "result", "title": title, "refLabel": "Confirmation",
                        "reference": conf, "download": False, "message": message},
            "message": (f"Tell the user it's {'rescheduled' if reschedule else 'booked'}: {message} "
                        f"Confirmation {conf}. Then stop.")}


def _dispatch(name, args, history, today):
    if name == "set_fields":
        return handle_set_fields(args, today)
    if name == "book_inspection":
        return handle_book(args, _accepted(history), today)
    if name == "leave_flow":
        return {"status": "left"}
    return {"error": f"unknown tool {name}"}


async def answer_inspection_query(history, client, model):
    """Run the inspection tool loop over the conversation history. Returns {reply, in_flow, widget, left}."""
    today = datetime.date.today()
    messages = [{"role": "system", "content": SYSTEM}] + list(history)
    left = False
    booked = False
    widget = None
    for _ in range(MAX_STEPS):
        resp = await client.chat.completions.create(
            model=model, messages=messages, tools=TOOLS, temperature=0)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            return {"reply": msg.content or "", "in_flow": not (left or booked),
                    "left": left, "widget": widget}
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _dispatch(tc.function.name, args, history, today)
            except Exception as e:
                logging.exception("inspection tool failed: %s", tc.function.name)
                result = {"error": str(e)}
            if isinstance(result, dict) and "_widget" in result:
                widget = result.pop("_widget")
            if tc.function.name == "leave_flow":
                left = True
            if tc.function.name == "book_inspection" and result.get("status") == "booked":
                booked = True
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if left:
            return {"reply": "", "in_flow": False, "left": True, "widget": None}
    return {"reply": "Sorry, let's try that again.", "in_flow": not (left or booked),
            "left": left, "widget": widget}
