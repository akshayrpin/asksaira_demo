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
    "http://clients.edgesoftinc.com:6060/buenapark-symbium-api/APIController",
)
USERNAME = os.environ.get("PERMIT_APPLY_USERNAME", "symbium")
PASSWORD = os.environ.get("PERMIT_APPLY_PASSWORD", "symbium1")
TIMEOUT = 25

# Force placeholder fee + submit with PERMIT_APPLY_MOCK=1 (also auto-mocks if codes blank).
MOCK = os.environ.get("PERMIT_APPLY_MOCK", "0") != "0"

# Permit codes confirmed against the sandbox: permitType SLR; subTypeId 6=residential,
# 7=commercial; peopleTypeId 8 = applicant.
PERMIT_CODES = {"permitType": "SLR", "subTypeIds": {"residential": 6, "commercial": 7},
                "applicant_peopleTypeId": 8}


def use_mock():
    """Mock the fee + submit if asked to, or whenever the codes aren't filled yet."""
    codes = PERMIT_CODES
    ready = codes.get("permitType") and codes.get("subTypeIds") and codes.get("applicant_peopleTypeId")
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


async def fee_estimate(app):
    """Estimate the permit fee. Placeholder when in mock mode."""
    if use_mock():
        return {"totalFee": 285.00, "feeDetails": [{"feeDescription": "Solar permit (placeholder)", "feeAmount": 285.00}],
                "mock": True}
    form = {
        "permitType": PERMIT_CODES["permitType"],
        "subTypeIds": [{"subTypeId": app["subTypeId"]}],
        "unit": app["unit"], "noOfBatteries": app["noOfBatteries"],
        "valuation": app["valuation"], "peopleId": app.get("peopleId", 0),
    }
    async with aiohttp.ClientSession() as s:
        return await _request(s, "POST", "feeEstimate", json_body=form)


async def add_permit(app):
    """Create the permit. Placeholder when in mock mode. Returns {permitNumber, ...}."""
    if use_mock():
        suffix = hashlib.sha1(app["property_address"].encode()).hexdigest()[:6].upper()
        return {"permitNumber": f"SOLAR-MOCK-{suffix}", "status": True, "mock": True}
    body = {
        "description": app["description"], "lsoId": app["lsoId"],
        "valuation": app["valuation"], "unit": app["unit"], "noOfBatteries": app["noOfBatteries"],
        "subTypeIds": [{"subTypeId": app["subTypeId"]}],
        "people": [{
            "name": app["name"], "emailAddress": app["email"], "phoneNbr": app["phone"],
            "address": app.get("property_address", ""), "city": app.get("city", "Buena Park"),
            "state": app.get("state", "CA"), "zipCode": app.get("zip", ""),
            "peopleTypeId": PERMIT_CODES["applicant_peopleTypeId"], "title": "Applicant",
        }],
    }
    # NOTE: the :6060 Buena Park sandbox endpoint is 'addSolarPermit'. The newer official
    # :9080 API calls it 'addPermit' (with actTypeId) — switch this when we move to that base.
    async with aiohttp.ClientSession() as s:
        return await _request(s, "POST", "addSolarPermit", json_body=body)
