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
import os
import re
import time

import aiohttp

BASE = os.environ.get(
    "PERMIT_APPLY_BASE",
    "http://clients.edgesoftinc.com:9080/askSairaApi/APIController",
)
USERNAME = os.environ.get("PERMIT_APPLY_USERNAME", "symbium")
PASSWORD = os.environ.get("PERMIT_APPLY_PASSWORD", "symbium1")
TIMEOUT = 25

# Force placeholder fee + submit with PERMIT_APPLY_MOCK=1 (also auto-mocks if codes blank).
MOCK = os.environ.get("PERMIT_APPLY_MOCK", "0") != "0"

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
    """Mock the fee + submit if asked to, or whenever the codes aren't filled yet."""
    codes = PERMIT_CODES
    ready = codes.get("permitType") and codes.get("actTypeId") and codes.get("applicant_peopleTypeId")
    return MOCK or not ready


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
    """Estimate the permit fee. Placeholder when in mock mode. feeEstimate uses the
    string permitType ('MEPP') plus the picked subTypeId."""
    if use_mock():
        return {"totalFee": 220.00, "feeDetails": [{"feeDescription": "MEPP permit (placeholder)", "feeAmount": 220.00}],
                "mock": True}
    form = {
        "permitType": PERMIT_CODES["permitType"],
        "subTypeIds": [{"subTypeId": app["subTypeId"]}],
        "unit": app.get("unit", 1),
        "valuation": app["valuation"], "peopleId": app.get("peopleId", 0),
    }
    async with aiohttp.ClientSession() as s:
        return await _request(s, "POST", "feeEstimate", json_body=form)


async def add_permit(app):
    """Create the permit via addPermit (AddActivity). Placeholder when in mock mode.
    Returns {permitNumber, ...}."""
    if use_mock():
        suffix = hashlib.sha1(app["property_address"].encode()).hexdigest()[:6].upper()
        return {"permitNumber": f"MEPP-MOCK-{suffix}", "status": True, "mock": True}
    body = {
        "actTypeId": PERMIT_CODES["actTypeId"],
        "description": app["description"], "lsoId": app["lsoId"],
        "valuation": app["valuation"], "unit": app.get("unit", 1),
        "subTypeIds": [{"subTypeId": app["subTypeId"]}],
        "people": [{
            "name": app["name"], "emailAddress": app["email"], "phoneNbr": app["phone"],
            "address": app.get("property_address", ""), "city": app.get("city", "Buena Park"),
            "state": app.get("state", "CA"), "zipCode": app.get("zip", ""),
            "peopleTypeId": PERMIT_CODES["applicant_peopleTypeId"], "title": "Applicant",
        }],
    }
    async with aiohttp.ClientSession() as s:
        return await _request(s, "POST", "addPermit", json_body=body)
