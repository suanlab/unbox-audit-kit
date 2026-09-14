"""Year-guard tests: every paper in each primary corpus must pre-date the
breakthrough year recorded in data/paradigm_shift_mapping.json.

Mirrors the style of tests/test_canonical_reproduction.py.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MAPPING_PATH = DATA_DIR / "paradigm_shift_mapping.json"
ADDITIONAL_DIR = PROJECT_ROOT / "experiments" / "additional_paradigms_60"

PRIMARY_PARADIGMS = ["transformer", "diffusion", "icl", "vit"]
# 6 post-hoc breakthroughs; corpora under experiments/additional_paradigms_60/
ADDITIONAL_PARADIGMS = ["resnet", "gan", "word2vec", "dropout", "batchnorm", "bert"]


class TestCorpusYearGuards(unittest.TestCase):
    """Assert max(year) < breakthrough_year for each primary paradigm corpus."""

    @classmethod
    def setUpClass(cls):
        with open(MAPPING_PATH, encoding="utf-8") as f:
            cls.mapping = json.load(f)

    def _check_paradigm(self, paradigm: str) -> None:
        breakthrough_year: int = self.mapping[paradigm]["year"]
        corpus_path = DATA_DIR / paradigm / "papers.jsonl"
        self.assertTrue(corpus_path.exists(), f"missing corpus: {corpus_path}")

        years = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                paper = json.loads(line)
                year = paper.get("year")
                if year is not None:
                    years.append(int(year))

        self.assertTrue(len(years) > 0, f"no year fields found in {corpus_path}")
        max_year = max(years)
        self.assertLess(
            max_year,
            breakthrough_year,
            f"{paradigm}: max paper year {max_year} is not strictly before "
            f"breakthrough year {breakthrough_year}",
        )

    def test_transformer_years(self):
        self._check_paradigm("transformer")

    def test_diffusion_years(self):
        self._check_paradigm("diffusion")

    def test_icl_years(self):
        self._check_paradigm("icl")

    def test_vit_years(self):
        self._check_paradigm("vit")

    def _check_additional(self, paradigm: str) -> None:
        """Same temporal guard for the 6 post-hoc corpora (experiments/
        additional_paradigms_60/<p>/papers.jsonl)."""
        breakthrough_year: int = self.mapping[paradigm]["year"]
        corpus_path = ADDITIONAL_DIR / paradigm / "papers.jsonl"
        self.assertTrue(corpus_path.exists(), f"missing corpus: {corpus_path}")
        years = []
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                year = json.loads(line).get("year")
                if year is not None:
                    years.append(int(year))
        self.assertTrue(len(years) > 0, f"no year fields in {corpus_path}")
        max_year = max(years)
        self.assertLess(
            max_year, breakthrough_year,
            f"{paradigm}: max paper year {max_year} not strictly before "
            f"breakthrough year {breakthrough_year}",
        )

    def test_resnet_years(self):
        self._check_additional("resnet")

    def test_gan_years(self):
        self._check_additional("gan")

    def test_word2vec_years(self):
        self._check_additional("word2vec")

    def test_dropout_years(self):
        self._check_additional("dropout")

    def test_batchnorm_years(self):
        self._check_additional("batchnorm")

    def test_bert_years(self):
        self._check_additional("bert")


if __name__ == "__main__":
    unittest.main()
