import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

runner = importlib.import_module("scripts.run_retrospective_experiment")


class TestRetrospectiveRunner(unittest.TestCase):
    def test_build_preflight_reports_call_estimate(self):
        preflight = runner.build_preflight(
            categories=["transformer", "diffusion"],
            mapping_path="data/paradigm_shift_mapping.json",
            evaluator_provider="anthropic",
            extraction_api_key=None,
            evaluator_api_key=None,
            max_assumptions_per_paper=3,
            max_papers=5,
        )

        transformer = preflight["categories"]["transformer"]
        self.assertTrue(transformer["papers_exists"])
        self.assertTrue(transformer["mapping_entry_exists"])
        self.assertEqual(transformer["effective_paper_count"], 5)
        self.assertEqual(
            transformer["estimated_call_volume"]["hypothesis_count_upper_bound"],
            45,
        )

    def test_alias_aware_matching_uses_best_rank(self):
        mapping_payload = {
            "transformer": {
                "broken_assumption": "recurrence is necessary for sequence modeling",
                "aliases": ["recurrent structure is required for sequence modeling"],
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            dir=PROJECT_ROOT,
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(mapping_payload, handle)
            temp_mapping = Path(handle.name)

        relative_mapping = temp_mapping.relative_to(PROJECT_ROOT)

        ranked_hypotheses = [
            {
                "assumption": "recurrent structure is required for sequence modeling",
                "paper_id": "p1",
                "paper_title": "Paper 1",
                "hypothesis": "Use attention",
                "transformation": "negate",
                "scores": {"composite": 8.5, "novelty": 9.0},
            },
            {
                "assumption": "convolution is necessary",
                "paper_id": "p2",
                "paper_title": "Paper 2",
                "hypothesis": "Use patches",
                "transformation": "negate",
                "scores": {"composite": 7.0, "novelty": 7.0},
            },
        ]

        try:
            with (
                patch.object(runner, "load_papers", return_value=[{"paperId": "p1"}]),
                patch.object(runner, "run_pipeline", return_value=ranked_hypotheses),
            ):
                result = runner.run_single_retrospective_experiment(
                    category="transformer",
                    mapping_path=str(relative_mapping),
                    top_ks=[1, 5],
                    pipeline_top_k=10,
                    extraction_api_key=None,
                    evaluator_provider="anthropic",
                    evaluator_model=None,
                    evaluator_api_key=None,
                    max_papers=1,
                    max_assumptions_per_paper=2,
                )
        finally:
            temp_mapping.unlink(missing_ok=True)

        self.assertEqual(result["metrics"]["rank"], 1.0)
        self.assertEqual(result["metrics"]["recall_at_1"], 1.0)
        self.assertEqual(
            result["inputs"]["ground_truth_aliases"],
            [
                "recurrence is necessary for sequence modeling",
                "recurrent structure is required for sequence modeling",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
