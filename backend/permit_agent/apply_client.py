"""
Write-side permit client for the instant-permit APPLY flow (async).

Separate from permit_client.py (the read/Solr side). This talks to the EdgeSoft / Symbium
permit API to: get a token, validate a property address, estimate the fee, and submit a
permit. Fee + submit honor MOCK mode, so until the real permit codes are set (or when
PERMIT_APPLY_MOCK=1) they return placeholder data and no real record is created.

Auth: POST getToken {username,password} -> token; the token goes in the Authorization
header (raw value, no "Bearer"). Payment is NOT here: the agent never touches card data;
payment stays a hosted link.
"""

import hashlib
import logging
import os
import re
import time

import aiohttp

try:  # optional: the styled permit PDF. Delete permit_pdf.py and this falls back to the stub.
    from backend.permit_agent import permit_pdf
except Exception:
    permit_pdf = None

BASE = os.environ.get(
    "PERMIT_APPLY_BASE",
    "http://clients.edgesoftinc.com:9080/askSairaApi/APIController",
)
USERNAME = os.environ.get("PERMIT_APPLY_USERNAME", "symbium")
PASSWORD = os.environ.get("PERMIT_APPLY_PASSWORD", "symbium1")
TIMEOUT = 25

# Force placeholder submit with PERMIT_APPLY_MOCK=1 (also auto-mocks if codes blank).
MOCK = os.environ.get("PERMIT_APPLY_MOCK", "0") != "0"

# The permit fee is a flat $66.50 for the demo, never fetched from the API's feeEstimate.
FLAT_FEE = 66.50

# MEPP (Mechanical/Electrical/Plumbing) instant permits, confirmed against the :9080 API.
# feeEstimate takes the string permitType "MEPP"; addPermit takes the integer actTypeId 1.
# The subtype is the specific job the resident picks. peopleTypeId 2 = applicant (8 = owner).
MEPP_SUBTYPES = {
    1: "Mechanical - HVAC",
    2: "Electrical - House Rewire",
    3: "Electrical - Panel Upgrade",
    4: "Plumbing - Water Heater Replacement",
    5: "Plumbing - House Repipe",
}
PERMIT_CODES = {"permitType": "MEPP", "actTypeId": 1, "applicant_peopleTypeId": 2,
                "owner_peopleTypeId": 8, "subtypes": MEPP_SUBTYPES}


def use_mock():
    """Mock the whole write-side (address, fee, submit, PDF) if asked to, or whenever the
    codes aren't filled yet. Set PERMIT_APPLY_MOCK=1 to run the flow end-to-end with no calls
    to the :9080 permit API (used while getToken is down)."""
    codes = PERMIT_CODES
    ready = codes.get("permitType") and codes.get("actTypeId") and codes.get("applicant_peopleTypeId")
    return MOCK or not ready


# A few realistic Burbank addresses so the type-ahead still feels real in mock mode. The
# query filters these exactly like the live list; if nothing matches we synthesize one from
# what was typed so the flow never dead-ends.
_MOCK_ADDRESSES = [
    "275 E Olive Ave, Burbank CA 91502",
    "301 E Olive Ave, Burbank CA 91502",
    "150 N Third St, Burbank CA 91502",
    "164 W Magnolia Blvd, Burbank CA 91502",
    "141 N Glenoaks Blvd, Burbank CA 91502",
    "1111 W Olive Ave, Burbank CA 91506",
    "2000 W Olive Ave, Burbank CA 91506",
    "635 N Hollywood Way, Burbank CA 91505",
]


def _mock_lso(address):
    """Deterministic non-'-1' id so a selected mock address behaves like a real one."""
    return "MOCK-" + hashlib.sha1(address.encode()).hexdigest()[:8].upper()


def _mock_search(query, limit):
    nq = _norm(query)
    if len(nq) < 2:
        return []
    prefix = [a for a in _MOCK_ADDRESSES if _norm(a).startswith(nq)]
    contains = [a for a in _MOCK_ADDRESSES if nq in _norm(a) and a not in prefix]
    hits = prefix + contains
    if not hits:  # echo what they typed, cleaned up, so selection still works
        typed = " ".join(w.capitalize() for w in str(query).split())
        hits = [f"{typed}, Burbank CA 91502"]
    return [{"address": a, "lsoId": _mock_lso(a), "apn": "000-000-000"} for a in hits[:limit]]


def _mock_pdf(act_nbr):
    """Build a minimal, valid one-page PDF for the download button in mock mode."""
    text_lines = [
        "CITY OF BURBANK  -  INSTANT PERMIT (DEMONSTRATION)",
        "",
        f"Permit Number: {act_nbr}",
        "Type: MEPP Instant Permit",
        "Status: Issued",
        "",
        "This is a mock permit generated for demonstration purposes.",
        "No official city record has been created.",
    ]
    parts = ["BT", "/F1 14 Tf", "72 720 Td", "18 TL"]
    for ln in text_lines:
        safe = ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        parts += [f"({safe}) Tj", "T*"]
    parts.append("ET")
    stream = "\n".join(parts).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref_pos)
    return out


_token = {"value": None, "at": 0.0}
_TOKEN_TTL = 200  # tokenValidity ~240; refresh a bit early


async def _get_token(session, force=False):
    if not force and _token["value"] and (time.time() - _token["at"] < _TOKEN_TTL):
        return _token["value"]
    async with session.post(f"{BASE}/getToken", json={"username": USERNAME, "password": PASSWORD},
                            timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
        r.raise_for_status()
        data = await r.json(content_type=None)
    if not data.get("token"):
        raise RuntimeError(f"getToken failed: {data.get('message')}")
    _token.update(value=data["token"], at=time.time())
    return _token["value"]


async def _request(session, method, path, params=None, json_body=None, _retry=True):
    token = await _get_token(session)
    async with session.request(
        method, f"{BASE}/{path}", params=params, json=json_body,
        headers={"Authorization": token, "Content-Type": "application/json"},
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
    ) as r:
        if r.status == 401 and _retry:  # token expired mid-session -> refresh once
            await _get_token(session, force=True)
            return await _request(session, method, path, params, json_body, _retry=False)
        return await r.json(content_type=None)


def _real(m):
    """Drop the 'Invalid Address' / lsoId -1 sentinel the API returns for a no-match."""
    return str(m.get("lsoId")) != "-1" and str(m.get("address", "")).strip().upper() != "INVALID ADDRESS"


def _norm(s):
    """Uppercase, drop punctuation, collapse whitespace, so '5865, Los Nietos St.' matches
    '5865  LOS NIETOS STREET '. Users type commas/periods that the stored data doesn't have."""
    return " ".join(re.sub(r"[.,;]", " ", str(s or "")).upper().split())


_all_addr = {"list": None}


async def _all_addresses(session):
    if _all_addr["list"] is None:
        data = await _request(session, "GET", "getAllAddresses")
        _all_addr["list"] = data.get("address") or []
    return _all_addr["list"]


# ------------------------------- public API -------------------------------

async def validate_address(address):
    """Resolve an address to [{lsoId, address, apn}]. Empty list = not found.

    validateAddress is finicky (returns the sentinel for many valid addresses), so we fall
    back to a normalized match against the full getAllAddresses list, exact first, then
    startswith, both case/whitespace-insensitive."""
    if use_mock():
        cleaned = " ".join(w.capitalize() for w in str(address).split())
        return [{"lsoId": _mock_lso(cleaned), "address": cleaned, "apn": "000-000-000"}]
    async with aiohttp.ClientSession() as s:
        data = await _request(s, "GET", "validateAddress", params={"address": address})
        matches = [m for m in (data.get("address") or []) if _real(m)]
        if matches:
            return matches
        nq = _norm(address)
        if not nq:
            return []
        addrs = await _all_addresses(s)
        exact = [a for a in addrs if _norm(a.get("address")) == nq]
        if exact:
            return exact
        return [a for a in addrs if _norm(a.get("address")).startswith(nq)][:5]


async def search_addresses(query, limit=8):
    """Type-ahead search for the address-autocomplete widget.

    Prefix matches rank first (house-number-first, so '4755 guad' surfaces
    '4755 GUADALAJARA WAY' at the top), then substring matches. Searches the once-cached
    getAllAddresses list in memory, so it NEVER hits the slow permit API per keystroke.
    Returns [{address, lsoId, apn}]; empty for queries under 2 chars. Selecting a result
    hands the caller a guaranteed-valid lsoId, so no follow-up validateAddress is needed.
    """
    if use_mock():
        return _mock_search(query, limit)
    nq = _norm(query)
    if len(nq) < 2:
        return []
    async with aiohttp.ClientSession() as s:
        addrs = await _all_addresses(s)
    prefix, contains = [], []
    for a in addrs:
        if not _real(a):
            continue
        na = _norm(a.get("address"))
        if na.startswith(nq):
            prefix.append(a)
        elif nq in na:
            contains.append(a)
    prefix.sort(key=lambda a: _norm(a.get("address")))
    contains.sort(key=lambda a: _norm(a.get("address")))
    return [{"address": str(a.get("address")).strip(), "lsoId": a.get("lsoId"), "apn": a.get("apn")}
            for a in (prefix + contains)[:limit]]


async def fee_estimate(app):
    """Return the fixed demo permit fee. The fee is a flat $66.50 and is NOT fetched from the
    API's feeEstimate, in either mock or real mode."""
    return {"totalFee": FLAT_FEE,
            "feeDetails": [{"feeDescription": "Permit fee", "feeAmount": FLAT_FEE}],
            "mock": use_mock()}


async def add_permit(app):
    """Create the permit via addPermit (AddActivity). Placeholder when in mock mode.
    Returns {permitNumber, ...}."""
    if use_mock():
        key = app.get("property_address") or app.get("address") or "permit"
        n = int(hashlib.sha1(str(key).encode()).hexdigest(), 16) % 100000
        return {"permitNumber": f"MEPP-2026-{n:05d}", "status": True, "mock": True}
    body = {
        "actTypeId": PERMIT_CODES["actTypeId"],
        "description": app["description"], "lsoId": app["lsoId"],
        "valuation": app["valuation"], "unit": app.get("unit", 1),
        "subTypeIds": [{"subTypeId": app["subTypeId"]}],
        "people": [{
            "name": app["name"], "emailAddress": app["email"], "phoneNbr": app["phone"],
            "address": app.get("property_address", ""), "city": app.get("city", "Burbank"),
            "state": app.get("state", "CA"), "zipCode": app.get("zip", ""),
            "peopleTypeId": PERMIT_CODES["applicant_peopleTypeId"], "title": "Applicant",
        }],
    }
    async with aiohttp.ClientSession() as s:
        return await _request(s, "POST", "addPermit", json_body=body)


async def permit_report(act_nbr, meta=None):
    """Permit PDF for the download button. The styled Burbank permit is used in BOTH mock and
    real mode (we never want the external system's own document). If permit_pdf is deleted, mock
    falls back to the simple stub and real falls back to the API's getPermitReport."""
    if permit_pdf is not None:
        try:
            return permit_pdf.build_permit_pdf(act_nbr, meta or {})
        except Exception:
            logging.exception("styled permit pdf failed; falling back")
    if use_mock():
        return _mock_pdf(act_nbr)
    async with aiohttp.ClientSession() as s:
        token = await _get_token(s)
        async with s.get(f"{BASE}/getPermitReport", params={"actNbr": act_nbr},
                         headers={"Authorization": token},
                         timeout=aiohttp.ClientTimeout(total=TIMEOUT)) as r:
            r.raise_for_status()
            return await r.read()
