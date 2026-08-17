"""
Entry point.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    export INVOICE_IMAGES_DIR=/path/to/batch1-0331..batch1-0381
    python main.py

Produces (in the current directory, or wherever config.py points):
    output.csv
    comparison_report.csv
    work/ocr_text/*.txt   (cached OCR per image)
    work/results/*.json   (raw per-pipeline output per image, for debugging)
"""
import os
import json
import sys

from config import (
    IMAGES_DIR, RESULTS_DIR, OUTPUT_CSV, COMPARISON_CSV,
)
from ocr_utils import list_invoice_images, ocr_image, ocr_words, image_id
from pipeline_a_llm import extract_fields_llm
from pipeline_b_rules import extract_fields_rules
from reconcile import (
    reconcile_record, build_comparison_rows, write_output_csv, write_comparison_csv,
)


def run():
    images = list_invoice_images(IMAGES_DIR)
    if not images:
        print(f"No images found in {IMAGES_DIR!r}. Set INVOICE_IMAGES_DIR to the "
              f"folder containing batch1-0331 .. batch1-0381.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_rows = []
    comparison_rows = []

    for i, image_path in enumerate(images, 1):
        img_id = image_id(image_path)
        print(f"[{i}/{len(images)}] {img_id}")

        text = ocr_image(image_path)
        words = ocr_words(image_path)

        result_a = extract_fields_llm(text)
        result_b = extract_fields_rules(text, words)

        # cache raw per-pipeline results for debugging/audit
        with open(os.path.join(RESULTS_DIR, f"{img_id}.json"), "w", encoding="utf-8") as f:
            json.dump({"pipeline_a": result_a, "pipeline_b": result_b}, f, indent=2)

        final = reconcile_record(result_a, result_b)
        final["image_id"] = img_id
        output_rows.append(final)

        comparison_rows.extend(build_comparison_rows(img_id, result_a, result_b))

    write_output_csv(output_rows, OUTPUT_CSV)
    write_comparison_csv(comparison_rows, COMPARISON_CSV)

    total_fields = len(comparison_rows)
    matches = sum(1 for r in comparison_rows if r["match"])
    print(f"\nDone. {len(images)} images processed.")
    print(f"Field-level agreement between pipelines: {matches}/{total_fields} "
          f"({matches / total_fields:.1%})")
    print(f"Wrote {OUTPUT_CSV} and {COMPARISON_CSV}")


if __name__ == "__main__":
    run()
