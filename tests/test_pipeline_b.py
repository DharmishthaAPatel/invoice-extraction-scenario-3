from pipeline_b_rules import extract_fields_rules, _clean_number

SAMPLE_OCR_TEXT = """Invoice no: 13507183
Date of issue: 09/09/2015
Seller: Client:
Chavez Ltd Roberts Ltd
2188 Robert Expressway Apt. 336 167 Ethan Knolls
New John, NE 09400 Reyestown, NC 40977
Tax Id: 904-77-6503 Tax Id: 956-94-0439
IBAN: GB41RNKZ61126734301911
ITEMS
No. Description Qty UM Net price Net worth VAT [%] Gross
worth
SUMMARY
VAT [%] Net worth VAT Gross worth
10% 1 612,50 161,25 1 773,75
Total $1 612,50 $ 161,25 $1773,75
"""

# Word bounding boxes for the "Seller: Client:" line and the name line
# below it, modeled on the real Tesseract output for this template.
SAMPLE_WORDS = [
    {"text": "Seller:", "left": 135, "top": 443, "width": 117, "height": 30, "block_num": 1, "par_num": 1, "line_num": 3},
    {"text": "Client:", "left": 828, "top": 447, "width": 118, "height": 30, "block_num": 1, "par_num": 1, "line_num": 3},
    {"text": "Chavez", "left": 143, "top": 510, "width": 100, "height": 30, "block_num": 1, "par_num": 1, "line_num": 4},
    {"text": "Ltd", "left": 255, "top": 510, "width": 39, "height": 30, "block_num": 1, "par_num": 1, "line_num": 4},
    {"text": "Roberts", "left": 837, "top": 510, "width": 104, "height": 30, "block_num": 1, "par_num": 1, "line_num": 4},
    {"text": "Ltd", "left": 954, "top": 510, "width": 39, "height": 30, "block_num": 1, "par_num": 1, "line_num": 4},
]


def test_seller_client_names_split_by_column_with_word_boxes():
    result = extract_fields_rules(SAMPLE_OCR_TEXT, SAMPLE_WORDS)
    assert result["seller_name"] == "Chavez Ltd"
    assert result["client_name"] == "Roberts Ltd"


def test_seller_client_names_empty_without_word_boxes():
    # Same-line "Seller: Client:" label can't be split from text alone.
    result = extract_fields_rules(SAMPLE_OCR_TEXT)
    assert result["seller_name"] == ""


def test_tax_ids_still_split_positionally_from_text_alone():
    result = extract_fields_rules(SAMPLE_OCR_TEXT, SAMPLE_WORDS)
    assert result["seller_tax_id"] == "904-77-6503"
    assert result["client_tax_id"] == "956-94-0439"


def test_money_fields_read_from_total_line_with_comma_decimal():
    result = extract_fields_rules(SAMPLE_OCR_TEXT, SAMPLE_WORDS)
    assert result["net_worth"] == "1612.50"
    assert result["vat"] == "161.25"
    assert result["gross_worth"] == "1773.75"


def test_clean_number_handles_comma_decimal_and_space_thousands():
    assert _clean_number("1 612,50") == "1612.50"
    assert _clean_number("161,25") == "161.25"


def test_clean_number_handles_us_thousands_and_period_decimal():
    assert _clean_number("1,234.56") == "1234.56"
    assert _clean_number("1234.56") == "1234.56"
