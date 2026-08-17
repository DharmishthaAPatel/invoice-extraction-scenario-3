"""
Shared configuration for the invoice extraction project.
"""
import os

# ---------------------------------------------------------------------------
# Paths — adjust to taste, or override via environment variables.
# ---------------------------------------------------------------------------
IMAGES_DIR = os.environ.get("INVOICE_IMAGES_DIR", "./images")          # 50 input images
RAW_OCR_DIR = os.environ.get("INVOICE_RAW_OCR_DIR", "./work/ocr_text")  # cached OCR text per image
OCR_WORDS_DIR = os.environ.get("INVOICE_OCR_WORDS_DIR", "./work/ocr_words")  # cached word bounding boxes per image
RESULTS_DIR = os.environ.get("INVOICE_RESULTS_DIR", "./work/results")  # per-image JSON from each pipeline
OUTPUT_CSV = os.environ.get("INVOICE_OUTPUT_CSV", "./output.csv")
COMPARISON_CSV = os.environ.get("INVOICE_COMPARISON_CSV", "./comparison_report.csv")

# Image file range expected from the dataset (batch1-0331 .. batch1-0381)
IMAGE_GLOB_PATTERNS = ("*.jpg", "*.jpeg", "*.png")

# ---------------------------------------------------------------------------
# Required fields (per the assignment spec)
# ---------------------------------------------------------------------------
FIELDS = [
    "seller_name",
    "seller_tax_id",
    "client_name",
    "client_tax_id",
    "invoice_number",
    "invoice_date",
    "net_worth",
    "vat",
    "gross_worth",
]

# Fields that should be compared numerically (with tolerance) rather than
# as exact strings when building the comparison report.
NUMERIC_FIELDS = {"net_worth", "vat", "gross_worth"}
NUMERIC_TOLERANCE = 0.01  # absolute tolerance in currency units

# ---------------------------------------------------------------------------
# LLM settings (Pipeline A)
# ---------------------------------------------------------------------------
ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_MAX_TOKENS = 1024

# Which pipeline "wins" when reconciling into the final output.csv, for any
# field where the two pipelines disagree and we have no other signal.
# ("a" = OCR+LLM, "b" = rules-based structured extractor)
PRIMARY_PIPELINE = "a"
