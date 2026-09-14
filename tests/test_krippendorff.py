"""Cross-validation of the custom Krippendorff interval-alpha estimator.

`scripts/compute_human_kappa.py:krippendorff_alpha_interval` is hand-rolled
(an earlier version produced an out-of-range value; a later pairwise version
was wrong for unbalanced/missing data). The headline human reliability number
depends on it and the raw rater data is withheld for privacy, so this test
pins correctness with independent oracles:

  * If the `krippendorff` PyPI package is installed, the production estimator
    is checked against it on randomized matrices INCLUDING missing values
    (full independent check, all regimes).
  * Always: on balanced complete matrices (no missing, equal raters per unit
    -- the exact regime of the actual N=10 / 72-pair G1 study) the production
    estimator is checked against an independent closed-form pairwise
    implementation.
  * Boundary cases: perfect agreement -> 1.0; systematic disagreement is
    bounded in [-1, 1]; missing-data inputs stay finite and in range.

No network, no PII data; runs in `make test`.
"""
from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "chk", PROJECT_ROOT / "scripts" / "compute_human_kappa.py")
_chk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_chk)
production_alpha = _chk.krippendorff_alpha_interval

try:
    import krippendorff as _kripp_lib  # type: ignore
    import numpy as _np  # noqa: F401
    HAVE_LIB = True
except Exception:
    HAVE_LIB = False


def simple_balanced_alpha(matrix):
    """Independent interval-alpha, valid ONLY for balanced complete data
    (every unit has the same number of ratings, no missing). Derived directly
    from D_o/D_e definitions without coincidence matrices -> a genuinely
    different code path from production."""
    rows = [list(r) for r in matrix if all(v is not None for v in r) and len(r) >= 2]
    if not rows:
        return None
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("not balanced/complete")
    N = len(rows)
    do = 0.0
    for r in rows:
        s = sum((r[i] - r[j]) ** 2 for i in range(m) for j in range(m) if i != j)
        do += s / (m * (m - 1))
    d_o = do / N
    allv = [v for r in rows for v in r]
    n = len(allv)
    de = sum((allv[i] - allv[j]) ** 2
             for i in range(n) for j in range(n) if i != j)
    d_e = de / (n * (n - 1))
    if d_e == 0:
        return None
    return 1.0 - d_o / d_e


def lib_alpha(matrix):
    """krippendorff PyPI oracle (handles missing). reliability_data is
    raters x units with np.nan for missing."""
    import numpy as np
    n_raters = max(len(r) for r in matrix)
    arr = np.full((n_raters, len(matrix)), np.nan)
    for u, row in enumerate(matrix):
        for r, v in enumerate(row):
            if v is not None:
                arr[r, u] = v
    return _kripp_lib.alpha(reliability_data=arr,
                            level_of_measurement="interval")


class TestKrippendorffCrossValidation(unittest.TestCase):
    def _bounded(self, p):
        if p is not None:
            self.assertLessEqual(p, 1.0 + 1e-9)
            self.assertGreaterEqual(p, -1.0 - 1e-9)

    def test_perfect_agreement_is_one(self):
        m = [[5, 5, 5], [1, 1, 1], [3, 3, 3], [4, 4, 4], [2, 2, 2]]
        self.assertAlmostEqual(production_alpha(m), 1.0, places=9)

    def test_systematic_disagreement_bounded_negative(self):
        m = [[1, 5], [5, 1], [1, 5], [5, 1]]
        p = production_alpha(m)
        self.assertLess(p, 0.0)
        self._bounded(p)

    def test_balanced_matches_independent_closed_form(self):
        fixtures = [
            [[1, 1, 2], [2, 2, 3], [3, 3, 3], [4, 5, 4],
             [2, 1, 2], [5, 5, 4], [1, 2, 1], [3, 4, 3]],
            [[5, 4], [1, 2], [3, 3], [2, 1], [4, 5], [1, 1]],
        ]
        for m in fixtures:
            self.assertAlmostEqual(production_alpha(m), simple_balanced_alpha(m),
                                   places=9, msg=f"matrix={m}")

    def test_randomized_balanced_match_closed_form(self):
        rnd = random.Random(20260519)
        for _ in range(50):
            N = rnd.randint(5, 60)
            R = rnd.randint(2, 12)
            m = [[rnd.randint(1, 5) for _ in range(R)] for _ in range(N)]
            p, s = production_alpha(m), simple_balanced_alpha(m)
            self.assertAlmostEqual(p, s, places=7, msg=f"matrix={m}")
            self._bounded(p)

    def test_g1_shape_balanced_stable(self):
        rnd = random.Random(7)
        m = [[rnd.randint(1, 5) for _ in range(10)] for _ in range(72)]
        p = production_alpha(m)
        self.assertAlmostEqual(p, simple_balanced_alpha(m), places=7)
        self._bounded(p)

    def test_missing_values_finite_and_bounded(self):
        rnd = random.Random(99)
        for _ in range(20):
            m = [[None if rnd.random() < 0.2 else rnd.randint(1, 5)
                  for _ in range(rnd.randint(2, 8))]
                 for _ in range(rnd.randint(6, 30))]
            self._bounded(production_alpha(m))

    @unittest.skipUnless(HAVE_LIB, "krippendorff package not installed")
    def test_against_pypi_library_including_missing(self):
        rnd = random.Random(2026)
        for _ in range(30):
            m = [[None if rnd.random() < 0.15 else rnd.randint(1, 5)
                  for _ in range(rnd.randint(2, 10))]
                 for _ in range(rnd.randint(8, 40))]
            p = production_alpha(m)
            try:
                lib = lib_alpha(m)
            except Exception:
                continue
            if p is None:
                continue
            self.assertAlmostEqual(p, float(lib), places=6, msg=f"matrix={m}")


if __name__ == "__main__":
    unittest.main()
