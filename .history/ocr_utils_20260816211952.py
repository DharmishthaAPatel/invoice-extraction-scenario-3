"""
OCR helpers shared by both pipelines.

Pipeline A (OCR + LLM) and Pipeline B (rules-based) both start from the same
raw OCR text, but that's where the similarity ends: A hands the text to an
LLM to interpret, B parses it with hand-written regex/heuristics tied to the
known invoice template. This keeps the OCR step itself identical/cheap while
the *extraction logic* stays meaningfully different, per the assignment.
"""
import os
import glob
import hashlib
import json

import pytesseract
from PIL import Image

from config import IMAGES_DIR, RAW_OCR_DIR, OCR_WORDS_DIR, IMAGE_GLOB_PATTERNS


def list_invoice_images(images_dir: str = IMAGES_DIR):
    """Return a sorted list of image paths found in images_dir."""
    paths = []
    for pattern in IMAGE_GLOB_PATTERNS:
        paths.extend(glob.glob(os.path.join(images_dir, pattern)))
    return sorted(paths)


def _cache_path(image_path: str) -> str:
    os.makedirs(RAW_OCR_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(RAW_OCR_DIR, f"{stem}.txt")


def ocr_image(image_path: str, use_cache: bool = True) -> str:
    """
    Run Tesseract OCR on an image and return the raw extracted text.
    Caches results to disk so repeated pipeline runs don't re-OCR.
    """
    cache_file = _cache_path(image_path)
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()

    img = Image.open(image_path)
    # PSM 6: assume a single uniform block of text — works well for
    # templated invoices. Bump to --oem 3 for the LSTM engine.
    text = pytesseract.image_to_string(img, config="--oem 3 --psm 6")

    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def _words_cache_path(image_path: str) -> str:
    os.makedirs(OCR_WORDS_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(OCR_WORDS_DIR, f"{stem}.json")


def ocr_words(image_path: str, use_cache: bool = True) -> list:
    """
    Run Tesseract with word-level bounding boxes and return a list of
    {text, left, top, width, height, block_num, par_num, line_num}.

    Plain OCR text (ocr_image) collapses this dataset's two-column
    Seller/Client layout onto shared lines, making the two parties
    impossible to separate reliably with regex alone. Pipeline B uses
    this positional data instead, to split each line at the seller/client
    column boundary. Cached to disk like ocr_image().
    """
    cache_file = _words_cache_path(image_path)
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    img = Image.open(image_path)
    data = pytesseract.image_to_data(
        img, config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT
    )
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        words.append({
            "text": text,
            "left": data["left"][i],
            "top": data["top"][i],
            "width": data["width"][i],
            "height": data["height"][i],
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i],
        })

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(words, f)
    return words


def image_id(image_path: str) -> str:
    """Stable short id for an image, used as the row key across CSVs."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return stem
