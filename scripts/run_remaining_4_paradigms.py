#!/usr/bin/env python3
"""
Run the remaining 4 paradigm shift experiments: dropout, batchnorm, word2vec, bert.
Reuses all logic from run_additional_60paper.py but filters to only the 4 remaining paradigms.
After running, updates summary.json to include all 6 paradigms.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_additional_60paper import (
    PARADIGMS,
    OUTPUT_DIR,
    run_paradigm_experiment,
    _save_summary,
    _print_final_summary,
)

REMAINING = {"dropout", "batchnorm", "word2vec", "bert"}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # Load existing results from resnet and gan
    existing_results: list[dict] = []
    for name in ["resnet", "gan"]:
        result_path = OUTPUT_DIR / name / "result.json"
        if result_path.exists():
            with result_path.open("r") as f:
                existing_results.append(json.load(f))
            print(f"  Loaded existing result for {name}")

    # Run remaining paradigms
    new_results: list[dict] = []
    paradigms_to_run = [p for p in PARADIGMS if p["name"] in REMAINING]

    for paradigm in paradigms_to_run:
        try:
            result = run_paradigm_experiment(
                paradigm=paradigm,
                anthropic_api_key=anthropic_key,
            )
            new_results.append(result)
        except Exception as e:
            print(f"\n  ERROR running {paradigm['name']}: {e}")
            import traceback
            traceback.print_exc()
            new_results.append({
                "paradigm": paradigm["name"],
                "error": str(e),
                "ran_at": datetime.now(UTC).isoformat(),
            })

        # Save intermediate combined summary
        all_results = existing_results + new_results
        _save_summary(all_results)

    # Final combined summary
    all_results = existing_results + new_results
    _save_summary(all_results)
    _print_final_summary(all_results)


if __name__ == "__main__":
    main()
