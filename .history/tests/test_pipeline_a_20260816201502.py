import os

from config import FIELDS
from pipeline_a_llm import extract_fields_llm


def test_extract_fields_llm_without_api_key_returns_empty_record(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = extract_fields_llm("dummy OCR text")
    assert result == {field: "" for field in FIELDS}
