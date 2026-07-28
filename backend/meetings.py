"""
Live meeting-schedule lookup from a Granicus "upcoming meetings" page.

Why this exists: council/board/commission meeting dates are time-sensitive and live in Granicus
(a separate subdomain the website crawler never touches), so the RAG index only ever has stale
snapshots. This reads the LIVE Granicus page at query time so "when is the next council meeting"
is always current, and it is fully deterministic (parse -> filter -> format, no LLM).

We parse the ViewPublisher HTML page (not the RSS feed) because the HTML "Upcoming Meetings"
table lists EVERY scheduled meeting, including ones whose agenda has not been posted yet; the
RSS agenda feed only has agenda-posted meetings.

Generic vs city-specific: the CODE is generic (Granicus uses one standard template across
thousands of governments, so this parser works for any Granicus city). Only the page URL is
city-specific and comes from the MEETINGS_FEED_URL env var. Unset -> feature off, so every other
city's app is unaffected until its own URL is configured.

Example MEETINGS_FEED_URL (Burbank):
    https://burbank.granicus.com/ViewPublisher.php?view_id=6
"""

import datetime
import os
import re

import aiohttp

FEED_URL = os.environ.get("MEETINGS_FEED_URL", "")
TIMEOUT = 15
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

# Granicus date cell text looks like "July 21, 2026 - 04:00 PM". Match the month/day/year (take
# the last date in the string) and, separately, the time for display. Handles abbreviated ("Jul")
# and full ("July") month names.
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_RE = re.compile(r"([A-Za-z]{3})[a-z]*\s+(\d{1,2}),\s+(\d{4})")
_TIME_RE = re.compile(r"(\d{1,2}:\d{2}\s*[AP]M)", re.I)

# Words that signal the user wants a meeting schedule, and words that are noise when matching a
# meeting body to a query.
_MEETING_WORD = re.compile(
    r"\b(meeting|meetings|commission|council|board|committee|trustees|federation|"
    r"authority|agency|hearing|session)\b", re.I)
_WHEN_WORD = re.compile(r"\b(when|next|upcoming|schedule|scheduled|date|time|held)\b", re.I)
_STOP = {"when", "is", "are", "the", "next", "a", "an", "of", "for", "do", "i", "my", "to",
         "meeting", "meetings", "upcoming", "schedule", "scheduled", "date", "time", "held",
         "city", "burbank", "hold", "will", "be", "there", "and", "on", "at", "get"}


def _text(s):
    """Visible text of an HTML fragment: drop hidden spans (the epoch timestamp), strip tags,
    unescape nbsp, collapse whitespace."""
    s = re.sub(r"<span[^>]*display:\s*none[^>]*>.*?</span>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    return re.sub(r"\s+", " ", s).strip()


def _status_and_name(title):
    """Split a Granicus title into (status, clean_name). Granicus appends '- Canceled' / '- DARK'
    to the event name for meetings that won't happen; we must not present those as 'the next
    meeting' without flagging them."""
    low = title.lower()
    status = "canceled" if ("cancel" in low) else ("dark" if "dark" in low else "")
    name = re.sub(r"\s*[-–]\s*(cancel?led|canceled|dark)\s*$", "", title, flags=re.I).strip()
    return status, name


def _parse_date(text):
    matches = _DATE_RE.findall(text or "")
    if not matches:
        return None
    mon, day, year = matches[-1]
    m = _MONTHS.get(mon.lower())
    if not m:
        return None
    try:
        return datetime.date(int(year), m, int(day))
    except ValueError:
        return None


async def _fetch(url):
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers={"User-Agent": _UA},
                         timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.text()


async def upcoming_meetings(today, url=None):
    """Parse the Granicus 'Upcoming Meetings' table into [{title, date, when, link}] for meetings
    on/after `today`, soonest first, de-duplicated by (title, date)."""
    html = await _fetch(url or FEED_URL)
    start = html.find('class="listingTable" id="upcoming"')
    if start < 0:
        return []
    table = html[start:html.find("</table>", start)]
    out, seen = [], set()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        name_m = re.search(r'headers="EventName"[^>]*>(.*?)</td>', row, re.S)
        date_m = re.search(r'headers="EventDate[^"]*"[^>]*>(.*?)</td>', row, re.S)
        if not name_m or not date_m:
            continue                          # header row / malformed row
        raw = _text(name_m.group(1))
        when = _text(date_m.group(1))         # e.g. "July 21, 2026 - 04:00 PM"
        d = _parse_date(when)
        if not raw or not d or d < today:
            continue
        status, title = _status_and_name(raw)
        key = (title, d)
        if key in seen:
            continue
        seen.add(key)
        link_m = re.search(r'href="([^"]*AgendaViewer[^"]*)"', row)
        link = link_m.group(1) if link_m else ""
        if link.startswith("//"):
            link = "https:" + link
        out.append({"title": title, "date": d, "when": when, "link": link, "status": status})
    out.sort(key=lambda m: m["date"])
    return out


def is_meeting_query(query):
    """True for questions asking about a meeting schedule/date (not, say, meeting minutes)."""
    return bool(query and _MEETING_WORD.search(query) and _WHEN_WORD.search(query))


def _format(m):
    date_str = m["date"].strftime("%A, %B %d, %Y").replace(" 0", " ")
    tm = _TIME_RE.search(m.get("when", ""))
    when = date_str + (f" at {tm.group(1).upper()}" if tm else "")
    if m.get("status") == "canceled":
        return f"The {m['title']} scheduled for {when} has been canceled."
    if m.get("status") == "dark":
        return f"No {m['title']} meeting is being held on {when}."
    line = f"The next {m['title']} is scheduled for {when}."
    if m["link"]:
        line += f"\n\nAgenda: {m['link']}"
    return line


async def answer_meeting_query(query, today, url=None):
    """Answer a meeting-schedule question from the live page, or None to fall through to RAG.

    Matches the meeting body by requiring every non-noise query word to appear in the meeting
    title (so "police commission" -> the Police Commission row). If the user named a body we
    can't find upcoming, return None rather than answer with the wrong meeting."""
    if not (url or FEED_URL):
        return None
    meetings = await upcoming_meetings(today, url)
    if not meetings:
        return None
    kws = [t for t in re.findall(r"[a-z]+", query.lower()) if t not in _STOP]
    if kws:
        matches = [m for m in meetings if all(k in m["title"].lower() for k in kws)]
        if not matches:
            return None  # a specific body was asked for but has no upcoming meeting -> let RAG try
    else:
        matches = meetings  # bare "when is the next meeting" -> the soonest one
    # Prefer the soonest ACTUAL meeting; only surface a canceled/dark one if that's all there is.
    scheduled = [m for m in matches if not m.get("status")]
    return _format(scheduled[0] if scheduled else matches[0])
