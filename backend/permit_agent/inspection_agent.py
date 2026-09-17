"""
Inspection scheduling / rescheduling agent — fully MOCK, conversational.

Flow: one question at a time (permit number, inspection type), then a CALENDAR widget where the
resident taps a date (only the next 5 business days are selectable; other dates are shown but
disabled) and a morning/afternoon slot. Books on that selection. Nothing is truly scheduled;
booking returns a mock INS confirmation number. Supports BOTH schedule and reschedule.

Booking gate lives in CODE (handle_book): it only books once a date AND a time slot are set,
which only happens after the resident submits the calendar.

Entry point:
  answer_inspection_query(history, client, model) -> {"reply", "in_flow", "widget", "left"}
"""

import datetime
import hashlib
import json
import logging

MAX_STEPS = 6

INSPECTION_TYPES = ["Rough Plumbing", "Rough Electrical", "Framing", "Final"]

# The two bookable slots (the times live here in code, as before).
_TIMES = {"morning": "9:00 AM", "afternoon": "1:30 PM"}

SYSTEM = """You are the City's inspection-scheduling assistant, running inside a chat. You help a resident schedule a new building inspection OR reschedule an existing one. Nothing is truly booked; this is a guided intake that produces a confirmation number.

Talk like a person: ask for ONE thing at a time. Do not dump a form.

How you work, every single turn:
1. FIRST call set_fields with EVERY value you have gathered so far (omit what you don't have yet):
   - action: "schedule" for a new booking, "reschedule" if they're moving an existing one. Infer it from what they say.
   - permit_number: the permit the inspection is for.
   - inspection_type: one of Rough Plumbing, Rough Electrical, Framing, Final.
   - date: the calendar date they picked, as YYYY-MM-DD.
   - time_slot: "morning" or "afternoon".
   Use only what the user actually said or tapped; never invent a value.
2. The tool result tells you what is still needed. Write ONE short, friendly line for that step. When the tool asks for the date, an on-screen CALENDAR appears (they tap a day + morning/afternoon), so just invite them to pick, don't ask them to type a date.
3. Once the tool says everything is set (a date and a time_slot are in), call book_inspection with the collected values.

Rules:
- Never claim it is booked yourself; only book_inspection does that.
- Only call leave_flow if the user clearly switches to an UNRELATED topic.
- Be concise and friendly."""

TOOLS = [
    {"type": "function", "function": {
        "name": "set_fields",
        "description": "Call this on EVERY turn. Report every value gathered so far (omit what you don't have). The system uses it to decide what to ask next or to show the calendar.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["schedule", "reschedule"]},
            "permit_number": {"type": "string"},
            "inspection_type": {"type": "string", "enum": INSPECTION_TYPES},
            "date": {"type": "string", "description": "the picked calendar date, YYYY-MM-DD"},
            "time_slot": {"type": "string", "enum": ["morning", "afternoon"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "book_inspection",
        "description": "Book (or move) the inspection. Call ONLY after a date AND time_slot are set. Pass the same collected values.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["schedule", "reschedule"]},
            "permit_number": {"type": "string"},
            "inspection_type": {"type": "string", "enum": INSPECTION_TYPES},
            "date": {"type": "string"},
            "time_slot": {"type": "string", "enum": ["morning", "afternoon"]},
        }, "required": ["permit_number", "inspection_type", "date", "time_slot"]},
    }},
    {"type": "function", "function": {
        "name": "leave_flow",
        "description": "The user stopped and asked something unrelated. Hand control back to normal help.",
        "parameters": {"type": "object", "properties": {}},
    }},
]


def _available_dates(today, n=5):
    """The next `n` business days (weekdays), starting tomorrow, as ISO strings. Only these are
    selectable in the calendar; every other date renders disabled."""
    out, d = [], today
    while len(out) < n:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:                 # skip weekends
            out.append(d.isoformat())
    return out


def _calendar_widget(today):
    return {"type": "calendar", "field": "slot",
            "available_dates": _available_dates(today),
            "times": [{"label": "Morning (9:00 AM)", "value": "morning"},
                      {"label": "Afternoon (1:30 PM)", "value": "afternoon"}]}


def _format_slot(date_iso, time_slot):
    try:
        d = datetime.date.fromisoformat(str(date_iso))
        return f"{d.strftime('%A, %B')} {d.day} at {_TIMES.get(time_slot, _TIMES['morning'])}"
    except Exception:
        return f"{date_iso} {_TIMES.get(time_slot, '')}".strip()


def handle_set_fields(a, today):
    """Deterministic engine: pick the next question, or show the calendar, or say we're ready."""
    if not str(a.get("permit_number", "")).strip():
        return {"need": "permit_number",
                "message": "Ask which permit number the inspection is for (a free-text reply)."}
    if a.get("inspection_type") not in INSPECTION_TYPES:
        return {"need": "inspection_type",
                "_widget": {"type": "chips", "field": "inspection_type", "options": INSPECTION_TYPES},
                "message": "Ask what kind of inspection they need; they can tap it."}
    if not str(a.get("date", "")).strip() or a.get("time_slot") not in _TIMES:
        verb = "the new time" if a.get("action") == "reschedule" else "a time"
        return {"need": "date", "_widget": _calendar_widget(today),
                "message": f"Invite them to pick {verb} on the calendar: a day (next 5 business days) and morning or afternoon."}
    slot = _format_slot(a["date"], a["time_slot"])
    return {"status": "ready", "slot": slot,
            "message": f"Everything's set for {slot}. Call book_inspection now."}


def _mock_conf(a):
    seed = f"{a.get('permit_number','')}|{a.get('inspection_type','')}"
    n = int(hashlib.sha1(seed.encode()).hexdigest(), 16) % 10000
    return f"INS-2026-{n:04d}"


def handle_book(a, today):
    """Booking gate in code: only books once a date AND time slot are present (i.e. the calendar
    was submitted). Supports both schedule and reschedule."""
    for f in ("permit_number", "inspection_type", "date", "time_slot"):
        if not str(a.get(f, "")).strip():
            return {"status": "invalid", "message": "Details are incomplete. Ask for the missing item."}
    slot = _format_slot(a["date"], a["time_slot"])
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


def _dispatch(name, args, today):
    if name == "set_fields":
        return handle_set_fields(args, today)
    if name == "book_inspection":
        return handle_book(args, today)
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
                result = _dispatch(tc.function.name, args, today)
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
