"""
Styled City of Burbank permit PDF for the instant-permit DEMO download.

REMOVABLE BY DESIGN: nothing depends on this module directly. `apply_client.permit_report`
imports it in a try/except and, if it is missing or errors, falls back to the simple stub
(`apply_client._mock_pdf`). So deleting this one file reverts the download to the old behavior
with no other changes.

It recreates page 1 of the Burbank building permit (header + barcode, permit block, a split
fee table, and the standard conditions), filled from the apply flow's data. Everything is a
mock: a small "DEMONSTRATION" marker is stamped so it can't be mistaken for an official permit.
"""

import datetime
import io
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Table, TableStyle)
from reportlab.graphics.barcode import code128

# Mock fee breakdown. Sums to 66.50 = apply_client.FLAT_FEE; if that flat fee changes, adjust
# these lines so the total still matches (kept here so the whole demo PDF stays self-contained).
FEE_SPLIT = [
    ("Permit fee", 45.00),
    ("Issuance Fee", 12.00),
    ("Surcharge: General Plan Maintenance Fee", 4.50),
    ("Surcharge: Technical Training Fee", 1.75),
    ("Admin Fee", 3.25),
]

# Friendly permit-type title by work type (the template calls it "Electrical Permit").
_CATEGORY = {
    "HVAC": "Mechanical Permit",
    "House Rewire": "Electrical Permit",
    "Panel Upgrade": "Electrical Permit",
    "Water Heater Replacement": "Plumbing Permit",
    "House Repipe": "Plumbing Permit",
}

# Standard page-1 conditions from the City of Burbank building permit print.
_CONDITIONS = [
    ("CARBON MONOXIDE ALARMS", "An owner or owner's agent shall install a carbon monoxide "
     "detector when a permit is obtained for alterations, repairs, or additions exceeding $1,000 "
     "to a dwelling unit that has an attached garage or fuel-burning appliances. For new "
     "construction, the alarms shall receive primary power from building wiring and shall have "
     "battery back-up. Multiple alarms shall be interconnected. In existing dwellings, the alarm "
     "may be solely battery operated when repairs do not result in removal of wall and ceiling "
     "finishes, or there is no access by means of attic, basement, or crawl space."),
    ("PLANS ON SITE", "The approved set of plans, specifications and job card must be at job "
     "site during construction and available to the building inspector. It is unlawful to alter "
     "or change the approved plans, or deviate there from, without approval of the Building "
     "Division. CBC Chapter 1, CEC"),
    ("PLAN APPROVAL", "Plans are in full compliance with relevant building codes of the City of "
     "Burbank, and the Building Official issues the building permit solely on that finding. "
     "Neither the Building Official nor the City of Burbank has made a determination whether the "
     "plans comply with other State or Federal laws or private contractual and/or property "
     "rights. CBC Chapter 1, CEC"),
    ("DETERMINATION OF PROPERTY LINES", "It is the responsibility of the applicant to verify the "
     "location of all property lines. You may be required to provide a boundary survey or other "
     "certification by a California licensed surveyor, as requested by the Building Inspector. "
     "BMC 9-1-1-107"),
    ("INSPECTION REQUEST", "Request an inspection at permit.burbankca.gov using your permit "
     "number."),
    ("DIG ALERT", "The permit applicant is responsible for verifying that all utilities have "
     "been disconnected prior to starting demolition. Call Underground Service Alert of Southern "
     "California (DigAlert) at: 800-227-2600."),
]


def _fmt_money(v):
    try:
        return f"$ {float(v):,.2f}"
    except (TypeError, ValueError):
        return "$ 0.00"


_SEAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "burbank_seal.png")


def _draw_header(canvas, doc):
    """Fixed top-of-page: city seal + department block (left), barcode + permit meta (right)."""
    c = canvas
    W, H = letter
    c.saveState()
    if os.path.exists(_SEAL):
        try:
            c.drawImage(_SEAL, 48, H - 90, width=48, height=50,
                        mask="auto", preserveAspectRatio=True)
        except Exception:
            pass
    x = 108
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, H - 46, "Community Development Department")
    c.drawString(x, H - 58, "BUILDING & SAFETY DIVISION")
    c.setFont("Helvetica", 9)
    c.drawString(x, H - 71, "150 N. Third St.")
    c.drawString(x, H - 82, "Burbank CA 91502")

    bc = code128.Code128(doc.permit_number, barHeight=26, barWidth=0.9)
    bc.drawOn(c, W - 50 - bc.width, H - 58)
    rx = W - 205
    c.setFont("Helvetica", 9)
    c.drawString(rx, H - 74, "Permit No:")
    c.setFont("Helvetica-Bold", 11)
    c.drawString(rx + 52, H - 74, doc.permit_number)
    c.setFont("Helvetica", 8)
    c.drawString(rx, H - 86, f"Permit Status: {doc.status}")
    c.drawString(rx, H - 96, "Page 1 of 1")
    c.drawString(rx, H - 106, f"Printed: {doc.printed}")

    c.setFont("Helvetica-Oblique", 7.5)
    c.setFillGray(0.5)
    c.drawCentredString(W / 2.0, H - 26, "DEMONSTRATION — not an official City of Burbank permit")
    c.restoreState()


def build_permit_pdf(permit_number, meta=None):
    """Return the filled permit PDF as bytes. `meta` (optional): address, valuation,
    description, owner, phone. Missing values fall back to blanks/defaults."""
    meta = meta or {}
    today = datetime.date.today()
    expire = today.replace(year=today.year + 1)

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=letter,
                          leftMargin=50, rightMargin=50, topMargin=118, bottomMargin=48)
    doc.permit_number = str(permit_number or "")
    doc.status = "Issued"
    doc.printed = today.strftime("%m/%d/%Y")
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="body", showBoundary=0)
    doc.addPageTemplates([PageTemplate(id="permit", frames=[frame], onPage=_draw_header)])

    ss = getSampleStyleSheet()
    h_title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=10, alignment=1)
    lbl = ParagraphStyle("lbl", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#555555"))
    val = ParagraphStyle("val", parent=ss["Normal"], fontSize=9)
    cond_h = ParagraphStyle("ch", parent=ss["Normal"], fontSize=7.5, fontName="Helvetica-Bold", spaceBefore=4)
    cond_b = ParagraphStyle("cb", parent=ss["Normal"], fontSize=7, leading=8.5, textColor=colors.HexColor("#222222"))

    category = _CATEGORY.get(meta.get("description"), "Building Permit")
    story = [Paragraph(category, h_title)]

    # Permit detail block: address (bold) + a two-column grid of the key fields.
    story.append(Paragraph(f"<b>Job Address:</b> {meta.get('address') or ''}", val))
    story.append(Spacer(1, 4))
    detail = [
        [Paragraph("Parcel Number:", lbl), Paragraph("", val),
         Paragraph("Applied:", lbl), Paragraph(today.strftime("%m/%d/%Y"), val)],
        [Paragraph("Valuation:", lbl), Paragraph(_fmt_money(meta.get("valuation")), val),
         Paragraph("Issued:", lbl), Paragraph(today.strftime("%m/%d/%Y"), val)],
        [Paragraph("Job Description:", lbl), Paragraph(str(meta.get("description") or ""), val),
         Paragraph("To Expire:", lbl), Paragraph(expire.strftime("%m/%d/%Y"), val)],
    ]
    dt = Table(detail, colWidths=[80, 200, 55, 77])
    dt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 1),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story.append(dt)
    story.append(Spacer(1, 4))
    owner = meta.get("owner") or ""
    phone = meta.get("phone") or ""
    story.append(Paragraph(f"<b>Owner:</b> {owner} &nbsp;&nbsp; Lic: &nbsp;&nbsp; Phone: {phone}", val))
    story.append(Paragraph("Burbank CA", lbl))
    story.append(Spacer(1, 12))

    # Fee table (split into line items, Amount / Paid, then Totals + Balance Due).
    total = sum(a for _, a in FEE_SPLIT)
    rows = [["Fees", "Amount", "Paid"]]
    rows += [[d, _fmt_money(a), "$ 0.00"] for d, a in FEE_SPLIT]
    rows += [["", "", ""],
             ["Totals:", _fmt_money(total), "$ 0.00"],
             ["Balance Due:", _fmt_money(total), ""]]
    ft = Table(rows, colWidths=[332, 90, 90])
    ft.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#999999")),
        ("LINEABOVE", (0, -2), (-1, -2), 0.5, colors.HexColor("#999999")),
        ("FONTNAME", (0, -2), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, -1), (1, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ft)
    story.append(Spacer(1, 12))

    # Conditions block.
    cbar = Table([["Conditions"]], colWidths=[doc.width])
    cbar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#d9d9d9")),
                              ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                              ("FONTSIZE", (0, 0), (-1, -1), 9),
                              ("LEFTPADDING", (0, 0), (-1, -1), 5),
                              ("TOPPADDING", (0, 0), (-1, -1), 3),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(cbar)
    for title, body in _CONDITIONS:
        story.append(Paragraph(f"{title}: {body}", cond_b))
    story.append(Spacer(1, 10))
    story.append(Paragraph("_________________________________", val))
    story.append(Paragraph("Signature of Applicant", lbl))

    doc.build(story)
    return buf.getvalue()
