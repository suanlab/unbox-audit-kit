from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Callable, TypedDict, cast

from .taxonomy import ASSUMPTION_CATEGORIES, CategoryInfo, get_category_hierarchy


class ExtractedAssumption(TypedDict):
    assumption: str
    confidence: float
    category: str | None


class AnnotationRecord(TypedDict):
    assumption: str
    llm_confidence: float
    llm_category: str | None
    annotated_category: str
    annotated_subcategory: str
    notes: str
    annotator_id: str
    annotated_at: str


class AnnotationFile(TypedDict):
    metadata: dict[str, object]
    annotations: list[AnnotationRecord]


def load_assumptions_from_json(file_path: str | Path) -> list[ExtractedAssumption]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        payload_obj = cast(object, json.load(handle))

    raw_assumptions: list[object]
    if isinstance(payload_obj, dict):
        payload_dict = cast(dict[object, object], payload_obj)
        assumptions_obj = payload_dict.get("assumptions")
        if not isinstance(assumptions_obj, list):
            raise ValueError("JSON object must contain an 'assumptions' list")
        raw_assumptions = cast(list[object], assumptions_obj)
    elif isinstance(payload_obj, list):
        raw_assumptions = cast(list[object], payload_obj)
    else:
        raise ValueError("Unsupported JSON format for assumptions")

    normalized: list[ExtractedAssumption] = []
    for index, item_obj in enumerate(raw_assumptions):
        if not isinstance(item_obj, dict):
            raise ValueError(f"Assumption at index {index} must be an object")
        item_dict = cast(dict[object, object], item_obj)

        assumption_obj = item_dict.get("assumption")
        if not isinstance(assumption_obj, str) or not assumption_obj.strip():
            raise ValueError(
                f"Assumption at index {index} must include non-empty 'assumption'"
            )

        confidence_obj = item_dict.get("confidence", 0.0)
        if not isinstance(confidence_obj, (int, float)):
            raise ValueError(f"Assumption at index {index} has non-numeric confidence")

        category_obj = item_dict.get("category")
        category_value: str | None
        if category_obj is None:
            category_value = None
        elif isinstance(category_obj, str):
            category_value = category_obj
        else:
            raise ValueError(f"Assumption at index {index} has non-string category")

        normalized.append(
            {
                "assumption": assumption_obj.strip(),
                "confidence": float(confidence_obj),
                "category": category_value,
            }
        )

    return normalized


def present_assumptions_for_annotation(
    assumptions: list[ExtractedAssumption],
    annotator_id: str,
    taxonomy: dict[str, CategoryInfo] | None = None,
    input_fn: Callable[[str], str] = input,
) -> list[AnnotationRecord]:
    if taxonomy is None:
        taxonomy = get_category_hierarchy()

    category_names = list(taxonomy.keys())
    if not category_names:
        raise ValueError("Taxonomy must contain at least one category")

    records: list[AnnotationRecord] = []
    print(f"\nStarting annotation session for annotator: {annotator_id}")
    print(f"Assumptions to annotate: {len(assumptions)}\n")

    for idx, item in enumerate(assumptions, start=1):
        assumption_text = item["assumption"]
        confidence = item["confidence"]
        suggested_category = item["category"]

        print(f"[{idx}/{len(assumptions)}] {assumption_text}")
        print(f"  LLM confidence: {confidence:.2f}")
        if suggested_category:
            print(f"  LLM suggested category: {suggested_category}")

        selected_category = _choose_from_list(
            prompt="  Select top-level category",
            options=category_names,
            input_fn=input_fn,
            suggested=suggested_category,
        )

        subcategory_options = [
            subcat.name for subcat in taxonomy[selected_category].subcategories.values()
        ]

        selected_subcategory = _choose_from_list(
            prompt="  Select subcategory",
            options=subcategory_options,
            input_fn=input_fn,
        )

        notes = input_fn("  Optional notes (press Enter to skip): ").strip()
        print()

        records.append(
            {
                "assumption": assumption_text,
                "llm_confidence": confidence,
                "llm_category": suggested_category,
                "annotated_category": selected_category,
                "annotated_subcategory": selected_subcategory,
                "notes": notes,
                "annotator_id": annotator_id,
                "annotated_at": datetime.now(UTC).isoformat(),
            }
        )

    return records


def save_annotations_with_metadata(
    annotations: list[AnnotationRecord],
    output_path: str | Path,
    annotator_id: str,
    source_file: str | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> AnnotationFile:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, object] = {
        "annotator_id": annotator_id,
        "created_at": datetime.now(UTC).isoformat(),
        "num_annotations": len(annotations),
        "taxonomy_version": "default",
        "top_categories": sorted(ASSUMPTION_CATEGORIES.keys()),
    }

    if source_file:
        metadata["source_file"] = source_file

    if extra_metadata:
        metadata.update(extra_metadata)

    payload: AnnotationFile = {
        "metadata": metadata,
        "annotations": annotations,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def calculate_krippendorff_alpha(
    ratings_by_rater: list[list[object | None]],
    level_of_measurement: str = "nominal",
) -> float:
    if len(ratings_by_rater) < 2:
        return 0.0

    lengths = {len(row) for row in ratings_by_rater}
    if len(lengths) != 1:
        raise ValueError("All raters must provide the same number of items")

    num_items = lengths.pop()
    if num_items == 0:
        return 0.0

    units: list[list[object]] = []
    for item_idx in range(num_items):
        item_values = [
            row[item_idx] for row in ratings_by_rater if row[item_idx] is not None
        ]
        if len(item_values) >= 2:
            units.append(item_values)

    if not units:
        return 0.0

    observed_disagreement = _observed_disagreement(units, level_of_measurement)
    value_pool = [value for unit in units for value in unit]
    expected_disagreement = _expected_disagreement(value_pool, level_of_measurement)

    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else 0.0

    return 1.0 - (observed_disagreement / expected_disagreement)


def _choose_from_list(
    prompt: str,
    options: list[str],
    input_fn: Callable[[str], str],
    suggested: str | None = None,
) -> str:
    for idx, option in enumerate(options, start=1):
        print(f"    {idx}. {option}")

    if suggested and suggested in options:
        print(f"    Suggested: {suggested}")

    while True:
        value = input_fn(f"{prompt} [1-{len(options)}]: ").strip()
        if value.isdigit():
            selected_index = int(value)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1]
        print("    Invalid selection. Please enter a valid number.")


def _delta(a: object, b: object, level_of_measurement: str) -> float:
    if level_of_measurement == "nominal":
        return 0.0 if a == b else 1.0

    if level_of_measurement == "interval":
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Interval alpha requires numeric ratings")
        return float(a - b) ** 2

    raise ValueError("level_of_measurement must be 'nominal' or 'interval'")


def _observed_disagreement(
    units: list[list[object]], level_of_measurement: str
) -> float:
    numerator = 0.0
    denominator = 0

    for unit in units:
        for left in range(len(unit)):
            for right in range(left + 1, len(unit)):
                numerator += _delta(unit[left], unit[right], level_of_measurement)
                denominator += 1

    if denominator == 0:
        return 0.0

    return numerator / denominator


def _expected_disagreement(values: list[object], level_of_measurement: str) -> float:
    if len(values) < 2:
        return 0.0

    if level_of_measurement == "nominal":
        unique_values: list[object] = []
        counts: list[int] = []
        for value in values:
            matched = False
            for index, existing in enumerate(unique_values):
                if value == existing:
                    counts[index] += 1
                    matched = True
                    break
            if not matched:
                unique_values.append(value)
                counts.append(1)

        total = len(values)
        squared_probability_sum = sum((count / total) ** 2 for count in counts)
        return 1.0 - squared_probability_sum

    numerator = 0.0
    denominator = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            numerator += _delta(values[left], values[right], level_of_measurement)
            denominator += 1

    if denominator == 0:
        return 0.0

    return numerator / denominator
