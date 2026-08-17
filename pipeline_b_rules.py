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

# Accepts both "1234.56" (period decimal) and the dataset's actual
# "1 612,50" (space thousands separator, comma decimal) formats.
MONEY_RE = r"[\$€£]?\s*(\d[\d\s]*[.,]\d{2})"
NET_WORTH_RE = re.compile(r"(?:Net\s*worth|Subtotal)\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)
VAT_RE = re.compile(r"\bVAT\b(?:\s*\[?%?\]?)?\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)
GROSS_WORTH_RE = re.compile(r"(?:Gross\s*worth|Total)\s*[:\-]?\s*" + MONEY_RE, re.IGNORECASE)

# The SUMMARY block's "Total" row is the most reliable source for all three
# totals at once: "Total $1 612,50 $ 161,25 $1773,75" (net, vat, gross), in
# a fixed order that matches the row's own header. The individual label
# regexes above (NET_WORTH_RE etc.) can't reach these values because the
# "Net worth" / "VAT" / "Gross worth" header cells sit on a separate line
# from the numbers.
TOTAL_LINE_RE = re.compile(
    r"Total\s+" + MONEY_RE + r"\s+" + MONEY_RE + r"\s+" + MONEY_RE,
    re.IGNORECASE,
)


def _clean_number(raw: str) -> str:
    """
    Normalize a matched money string to a plain "1234.56"-style value.
    Handles both period-decimal ("1,234.56") and the dataset's actual
    comma-decimal, space-thousands format ("1 612,50").
    """
    if not raw:
        return ""
    raw = raw.strip().replace(" ", "")
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        # Ambiguous: comma-as-decimal (European) vs comma-as-thousands (US).
        # Exactly 2 digits after the last comma means it's a decimal point.
        if len(raw) - raw.rfind(",") - 1 == 2:
            raw = raw.replace(",", ".")
        else:
            raw = raw.replace(",", "")
    return raw


def _first_match(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _extract_party_block(text: str, label: str) -> str:
    """
    Pull the company-name line that follows a 'Seller:' / 'Client:' label.
    Assumes the name is the next non-empty line after the label, on its own
    line. Only correct when the OCR text has Seller and Client on separate
    lines; the dataset's actual template renders them side by side on a
    shared line ("Seller: Client:"), which this can't split — see
    _extract_seller_client_names for the positional fix. Kept as a fallback
    for when word bounding-box data isn't available.
    """
    pattern = re.compile(rf"{label}\s*[:\-]?\s*\n\s*([^\n]+)", re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _group_lines(words: list) -> list:
    """Group word dicts into lines, preserving reading order."""
    lines = {}
    order = []
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(w)
    return [lines[key] for key in order]


def _extract_seller_client_names(words: list) -> tuple:
    """
    The template renders 'Seller:' and 'Client:' side by side, with the two
    companies' names on the following line, also side by side. Plain OCR
    text collapses both columns onto one line ("Chavez Ltd Roberts Ltd"),
    making them impossible to split reliably by regex. This instead uses
    word bounding boxes: find the Seller:/Client: label line, then split the
    next line's words at the midpoint between the two labels' x-positions.
    """
    lines = _group_lines(words)
    for idx, line in enumerate(lines):
        seller_w = next((w for w in line if w["text"].rstrip(":").lower() == "seller"), None)
        client_w = next((w for w in line if w["text"].rstrip(":").lower() == "client"), None)
        if seller_w and client_w and idx + 1 < len(lines):
            midpoint = (seller_w["left"] + client_w["left"]) / 2
            name_line = sorted(lines[idx + 1], key=lambda w: w["left"])
            seller_words = [w["text"] for w in name_line if w["left"] < midpoint]
            client_words = [w["text"] for w in name_line if w["left"] >= midpoint]
            return " ".join(seller_words).strip(), " ".join(client_words).strip()
    return "", ""


def extract_fields_rules(ocr_text: str, words: list = None) -> dict:
    """
    words: optional word-level bounding-box data from ocr_utils.ocr_words(),
    used for the seller/client column split (see _extract_seller_client_names).
    Falls back to plain-text heuristics when not provided.
    """
    result = {field: "" for field in FIELDS}

    # --- Seller / Client names -------------------------------------------------
    seller_name, client_name = ("", "")
    if words:
        seller_name, client_name = _extract_seller_client_names(words)
    result["seller_name"] = seller_name or _extract_party_block(ocr_text, "Seller")
    result["client_name"] = client_name or _extract_party_block(ocr_text, "Client")

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
    # Prefer the SUMMARY block's "Total" row (net, vat, gross together) —
    # see TOTAL_LINE_RE. Fall back to label-adjacent matching for templates
    # where that row isn't present.
    total_match = TOTAL_LINE_RE.search(ocr_text)
    if total_match:
        result["net_worth"] = _clean_number(total_match.group(1))
        result["vat"] = _clean_number(total_match.group(2))
        result["gross_worth"] = _clean_number(total_match.group(3))
    else:
        result["net_worth"] = _clean_number(_first_match(NET_WORTH_RE, ocr_text))
        result["vat"] = _clean_number(_first_match(VAT_RE, ocr_text))
        result["gross_worth"] = _clean_number(_first_match(GROSS_WORTH_RE, ocr_text))

    return result
