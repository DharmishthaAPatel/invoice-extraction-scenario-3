"""
Pipeline A — OCR + LLM

Raw OCR text (noisy, unstructured) is handed to an LLM, which is much more
robust than regex to OCR typos, reordered fields, and template variation.
This is the "understands messy text" approach, as distinct from Pipeline B's
hand-written structural rules.
"""
import os
import json
import time

import anthropic

from config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS, FIELDS

_client = None


def _get_client():
    global _client
    if _client is None:
        # Requires ANTHROPIC_API_KEY in the environment.
        _client = anthropic.Anthropic()
    return _client


PROMPT_TEMPLATE = """You will be given raw OCR text extracted from a scanned invoice image. \
The OCR text may contain typos, missing spaces, or misread characters — use your judgement \
to reconstruct the intended values.

Extract exactly these fields and return ONLY a JSON object with these keys \
(no markdown fences, no commentary):

- seller_name: the invoice issuer's / seller's company name
- seller_tax_id: the seller's tax identification number
- client_name: the invoice recipient's / client's company name
- client_tax_id: the client's tax identification number
- invoice_number: the invoice number / id
- invoice_date: the invoice issue date, formatted as YYYY-MM-DD if possible
- net_worth: the total net worth / subtotal before tax, as a plain number (no currency symbol, no thousands separators)
- vat: the total VAT / tax amount, as a plain number
- gross_worth: the total gross worth / total after tax, as a plain number

If a field cannot be determined, use an empty string "" for text fields or \
null for numeric fields. Do not guess wildly — only fill in a value you can \
actually support from the text.

OCR TEXT:
---
{ocr_text}
---
"""


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        # strip markdown fences if the model added them anyway
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def extract_fields_llm(ocr_text: str, retries: int = 2) -> dict:
    """
    Send OCR text to Claude and return a dict of the required fields.
    Falls back to an all-empty record (rather than raising) after retries
    are exhausted, so a single bad image doesn't kill a 50-image batch run.
    """
    client = _get_client()
    prompt = PROMPT_TEMPLATE.format(ocr_text=ocr_text)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            parsed = _parse_json_response(raw_text)
            return {field: parsed.get(field, "") for field in FIELDS}
        except Exception as e:  # noqa: BLE001 — deliberately broad for batch robustness
            last_error = e
            time.sleep(1.5 * (attempt + 1))

    print(f"[pipeline_a] giving up after {retries + 1} attempts: {last_error}")
    return {field: "" for field in FIELDS}
