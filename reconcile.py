"""
Compares Pipeline A and Pipeline B results and produces:

  - output.csv            final reconciled fields (one row per image)
  - comparison_report.csv per-field agreement between the two pipelines
"""
import csv

from config import FIELDS, NUMERIC_FIELDS, NUMERIC_TOLERANCE, PRIMARY_PIPELINE


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _to_float(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def fields_match(field: str, val_a: str, val_b: str) -> bool:
    if field in NUMERIC_FIELDS:
        fa, fb = _to_float(val_a), _to_float(val_b)
        if fa is None or fb is None:
            return fa is None and fb is None
        return abs(fa - fb) <= NUMERIC_TOLERANCE
    return _normalize_text(val_a) == _normalize_text(val_b)


def reconcile_record(result_a: dict, result_b: dict) -> dict:
    """
    Build the 'final' record for output.csv. Rule: if both pipelines agree,
    use that value. If they disagree, prefer PRIMARY_PIPELINE's value unless
    it's empty, in which case fall back to the other pipeline.
    """
    primary, secondary = (result_a, result_b) if PRIMARY_PIPELINE == "a" else (result_b, result_a)
    final = {}
    for field in FIELDS:
        val_primary = primary.get(field, "")
        val_secondary = secondary.get(field, "")
        final[field] = val_primary if val_primary not in ("", None) else val_secondary
    return final


def write_output_csv(rows: list, path: str) -> None:
    """rows: list of dicts with 'image_id' + FIELDS."""
    fieldnames = ["image_id"] + FIELDS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_comparison_csv(comparisons: list, path: str) -> None:
    """comparisons: list of dicts with image_id, field, pipeline_a, pipeline_b, match."""
    fieldnames = ["image_id", "field", "pipeline_a", "pipeline_b", "match"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row)


def build_comparison_rows(image_id: str, result_a: dict, result_b: dict) -> list:
    rows = []
    for field in FIELDS:
        val_a = result_a.get(field, "")
        val_b = result_b.get(field, "")
        rows.append({
            "image_id": image_id,
            "field": field,
            "pipeline_a": val_a,
            "pipeline_b": val_b,
            "match": fields_match(field, val_a, val_b),
        })
    return rows
