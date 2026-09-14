"""Regression test: paper numbers must match stored canonical artifacts.

Runs scripts/reproduce_paper_tables.py as a subprocess and asserts exit code 0
and the 'all paper numbers match' sentinel in stdout.
"""
from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestCanonicalReproduction(unittest.TestCase):
    def test_reproduce_paper_tables_passes(self):
        py = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not py.exists():
            py = "python3"
        script = PROJECT_ROOT / "scripts" / "reproduce_paper_tables.py"
        self.assertTrue(script.exists(), f"missing {script}")

        result = subprocess.run(
            [str(py), str(script)],
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
            capture_output=True,
            text=True,
            timeout=60,
        )
        stdout = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, f"reproduce_paper_tables.py failed:\n{stdout}")
        self.assertIn("All paper numbers match stored artifacts", stdout)
        # Also require a minimum number of successful checks (bump if more are added)
        import re
        m = re.search(r"Results:\s*(\d+)\s*passed,\s*(\d+)\s*failed", stdout)
        self.assertIsNotNone(m, f"could not parse pass/fail counts in:\n{stdout}")
        passed, failed = int(m.group(1)), int(m.group(2))
        self.assertEqual(failed, 0, f"{failed} paper numbers diverged from canonical artifacts")
        self.assertGreaterEqual(passed, 40, f"only {passed} checks ran; artifacts may be incomplete")


if __name__ == "__main__":
    unittest.main()
