"""
Pipeline B — Rules-based structured extractor

This walks the OCR text with regex patterns tied to the
known layout of the dataset's invoice template:

    Seller:                Client:
    <name>                 <name>
    <address lines>        <address lines>
    Tax Id: <id>           Tax Id: <id>

    Invoice no: <number>
    Date of issue: <date>

    ... line items table ...
    SUMMARY
    Total   <net>   <vat%>   <vat amt>   <gross>

This is the "meaningfully different" approach required by the assignment:
deterministic, no external API calls, breaks in different ways than the LLM
pipeline (e.g. brittle to template drift, but immune to LLM hallucination).
"""
import re

from config import FIELDS

# ---------------------------------------------------------------------------
# Regex patterns. Written to be reasonably tolerant of OCR noise (optional
# punctuation/whitespace), but they assume the dataset's fixed template.
# ---------------------------------------------------------------------------

TAX_ID_RE = re.compile(r"Tax\s*Id\s*[:\-]?\s*([A-Z0-9\-\/]{6,})", re.IGNORECASE)
INVOICE_NO_RE = re.compile(r"Invoice\s*(?:no|number|#)\s*[:\-]?\s*([A-Za-z0-9\/\-\.]+)", re.IGNORECASE)
DATE_RE = re.compile(
    r"Date\s*of\s*issue\s*[:\-]?\s*([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4}|\d{4}[\/\-\.][01]?\d[\/\-\.][0-3]?\d)",
    re.IGNORECASE,
)
# Fallback generic date pattern if "date of issue" label wasn't caught by OCR.
DATE_FALLBACK_RE = re.compile(r"\b([0-3]?\d[\/\-\.][01]?\d[\/\-\.]\d{2,4})\b")

MONEY_RE = r"[\$€£]?\s*([\d\s,]+\.\d{2})"
NET_WORTH_RE = re.compile(r"(?:Net\s*worth|Subtotal)\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)
VAT_RE = re.compile(r"\bVAT\b(?:\s*\[?%?\]?)?\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)
GROSS_WORTH_RE = re.compile(r"(?:Gross\s*worth|Total)\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)


def _clean_number(raw: str) -> str:
    if not raw:
        return ""
    return raw.replace(" ", "").replace(",", "")


def _first_match(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _extract_party_block(text: str, label: str) -> str:
    """
    Pull the company-name line that follows a 'Seller:' / 'Client:' label.
    Assumes the name is the next non-empty line after the label.
    """
    pattern = re.compile(rf"{label}\s*[:\-]?\s*\n\s*([^\n]+)", re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def extract_fields_rules(ocr_text: str) -> dict:
    result = {field: "" for field in FIELDS}

    # --- Seller / Client names -------------------------------------------------
    result["seller_name"] = _extract_party_block(ocr_text, "Seller")
    result["client_name"] = _extract_party_block(ocr_text, "Client")

    # --- Tax IDs: first occurrence belongs to seller, second to client --------
    tax_id_matches = TAX_ID_RE.findall(ocr_text)
    if len(tax_id_matches) >= 1:
        result["seller_tax_id"] = tax_id_matches[0].strip()
    if len(tax_id_matches) >= 2:
        result["client_tax_id"] = tax_id_matches[1].strip()

    # --- Invoice number / date --------------------------------------------------
    result["invoice_number"] = _first_match(INVOICE_NO_RE, ocr_text)
    date_val = _first_match(DATE_RE, ocr_text)
    if not date_val:
        date_val = _first_match(DATE_FALLBACK_RE, ocr_text)
    result["invoice_date"] = date_val

    # --- Monetary fields ----------------------------------------------------
    result["net_worth"] = _clean_number(_first_match(NET_WORTH_RE, ocr_text))
    result["vat"] = _clean_number(_first_match(VAT_RE, ocr_text))
    result["gross_worth"] = _clean_number(_first_match(GROSS_WORTH_RE, ocr_text))

    return result
