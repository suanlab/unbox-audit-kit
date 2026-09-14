#!/usr/bin/env python3
"""Verify Unbox corpus size and leakage constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_RULES: dict[str, dict[str, int]] = {
    "transformer": {"max_year": 2016},
    "diffusion": {"max_year": 2019},
    "icl": {"max_year": 2019},
    "vit": {"max_year": 2019},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify collected corpus quality")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--min-count",
        type=int,
        default=50,
        help="Minimum papers required per category",
    )
    parser.add_argument(
        "--min-abstract-ratio",
        type=float,
        default=0.7,
        help="Minimum abstract coverage ratio",
    )
    parser.add_argument(
        "--output",
        default="evidence/task-1-corpus-stats.json",
        help="Output JSON path for verification report",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _as_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    report: dict[str, Any] = {
        "checks": {},
        "all_passed": True,
        "thresholds": {
            "min_count": args.min_count,
            "min_abstract_ratio": args.min_abstract_ratio,
        },
    }

    for category, rules in EXPECTED_RULES.items():
        path = data_dir / category / "papers.jsonl"
        rows = _load_jsonl(path)

        years = [
            year
            for row in rows
            for year in [_as_int(row.get("year"))]
            if year is not None
        ]
        abstract_count = sum(1 for row in rows if str(row.get("abstract", "")).strip())
        max_year = max(years) if years else None
        leakage_count = sum(
            1
            for row in rows
            if isinstance(row.get("year"), int) and int(row["year"]) > rules["max_year"]
        )
        abstract_ratio = (abstract_count / len(rows)) if rows else 0.0

        checks = {
            "file_exists": path.exists(),
            "paper_count": len(rows),
            "count_ok": len(rows) >= args.min_count,
            "max_year": max_year,
            "year_ok": (max_year is not None) and (max_year <= rules["max_year"]),
            "leakage_count": leakage_count,
            "leakage_ok": leakage_count == 0,
            "abstract_ratio": round(abstract_ratio, 3),
            "abstract_ratio_ok": abstract_ratio >= args.min_abstract_ratio,
        }
        category_passed = (
            checks["file_exists"]
            and checks["count_ok"]
            and checks["year_ok"]
            and checks["leakage_ok"]
            and checks["abstract_ratio_ok"]
        )
        checks["passed"] = category_passed
        report["checks"][category] = checks
        report["all_passed"] = report["all_passed"] and category_passed

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    for category, checks in report["checks"].items():
        print(
            f"[{category}] passed={checks['passed']} count={checks['paper_count']} "
            f"max_year={checks['max_year']} leakage={checks['leakage_count']} "
            f"abstract_ratio={checks['abstract_ratio']}"
        )

    print(f"all_passed={report['all_passed']}")
    print(f"report={output_path}")

    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
