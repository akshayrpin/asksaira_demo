"""backend/zoning.py — address-specific zoning / land-use answers.

A land-use question ("can I open a medical office at 2019 W Magnolia?") is answered from the
municipal-code index via the website pipeline, but with a zoning-specific prompt:
  - If the conversation hasn't stated the property's zoning designation, ask for it (with the
    Planning contact) instead of answering.
  - If a designation is present, answer strictly from the retrieved zoning-code sources.

No hardcoded zone list or use rules: the model decides whether a designation was given, and the
answer is grounded in the actual indexed code. Gated by ZONING_ROUTE_ENABLED in app.py.
"""
import os

from backend import website_pipeline

CITY_NAME = os.environ.get("CITY_NAME", "Burbank")
PLANNING_CONTACT = os.environ.get(
    "PLANNING_CONTACT", "the Planning Division at (818) 238-5250 or planning@burbankca.gov")
ZONING_MAP_URL = os.environ.get(
    "ZONING_MAP_URL",
    "https://experience.arcgis.com/experience/00653afd849744eab0ca3547d66db78a/page/PIM")

ZONING_SYSTEM = (
    f"You are the City of {CITY_NAME}'s Planning assistant answering a question about what may be "
    "done at a specific property (a land-use / zoning question). The answer depends on the "
    "property's ZONING DESIGNATION (for example C-3, R-1, MDC-3).\n"
    "- If the conversation does NOT state the property's zoning designation, do NOT try to answer. "
    "Ask the resident to look the property up on the City's zoning map at " + ZONING_MAP_URL + ", "
    "find its zoning designation, and reply with it (for example \"C-3\"). Include that map link "
    "in your reply. Also give this contact: " + PLANNING_CONTACT + ".\n"
    "- If a zoning designation IS given, answer the land-use question using ONLY the numbered sources "
    "below (the zoning code): state whether the use is permitted by-right, allowed with a Conditional "
    "Use Permit (CUP) or an Administrative Use Permit (AUP), or not permitted in that zone; include "
    "the parking requirement if the sources give one; note that additional entitlements may be "
    "required (late-night hours, change of use, alcohol sales, exterior changes); cite sources as "
    "[doc1], [doc2]; and end with: For more information, contact " + PLANNING_CONTACT + ".\n"
    "Never invent a permission status, a parking ratio, or a code section not in the sources. "
    "Today is {today}."
)


async def answer_zoning_query(question, client, model):
    """Answer via the code index with the zoning prompt. `question` is the reformulated standalone
    query: a follow-up like 'the zoning is C-3' has already been resolved upstream into a full
    question (use + designation), so we retrieve and answer on it directly, with no conversation
    concatenation (that used to drag unrelated earlier turns into retrieval).
    Returns website_pipeline's (answer, context, answer_id)."""
    return await website_pipeline.answer_website_query(
        question, client, model, system=ZONING_SYSTEM, retrieval_query=question)
