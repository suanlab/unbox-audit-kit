#!/usr/bin/env bash
# One-command human-validation finalizer (path A).
#
# Run this AFTER rater CSVs come back. It is deadline-safe and idempotent:
#   1. checks annotation/rater{1,2,3}_complete.csv exist (>=2 required)
#   2. compute_human_kappa.py            -> annotation/g1_results.json
#   3. render_human_validation.py results --apply
#      (swaps the §5.4 protocol table -> real results table in
#       unbox_arr.tex, the ACL/ARR submission)
#   4. recompiles unbox_arr.tex (pdflatex+bibtex+pdflatex x2)
#   5. prints the key numbers + page-count sanity
#
# If labels never arrive, do nothing: the paper already ships fallback B.
# To revert to fallback B at any time:
#   .venv/bin/python scripts/render_human_validation.py protocol --apply
#
# Usage:
#   bash scripts/finalize_human_validation.sh

set -uo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
A="annotation"

# --- 1. preconditions -------------------------------------------------------
# Any *_complete.csv (one per rater) is accepted; pairs_*.csv excluded.
present=$(ls "$A"/*_complete.csv 2>/dev/null | grep -vc '/pairs_' || true)
present=${present:-0}
if [ "$present" -lt 2 ]; then
  echo "ERROR: need >=2 annotation/*_complete.csv rater files (found $present)." >&2
  echo "Place returned CSVs as annotation/<name>_complete.csv" >&2
  exit 2
fi
echo "[1/5] rater files: $present found"

# --- 2. inter-rater + human-vs-rule kappa ----------------------------------
echo "[2/5] computing Krippendorff alpha / Cohen kappa ..."
"$PY" scripts/compute_human_kappa.py || { echo "compute_human_kappa.py failed" >&2; exit 3; }

# --- 3. swap paper §5.4 protocol -> results (unbox_arr.tex) -----------------
echo "[3/5] swapping §5.4 block to results table (unbox_arr.tex) ..."
"$PY" scripts/render_human_validation.py results --apply \
  || { echo "render_human_validation.py failed" >&2; exit 4; }

# --- 4. recompile the ARR submission ---------------------------------------
echo "[4/5] recompiling unbox_arr.tex ..."
( cd paper
  rm -f unbox_arr.aux unbox_arr.bbl
  pdflatex -interaction=nonstopmode unbox_arr.tex >/tmp/fhv1.log 2>&1
  bibtex unbox_arr >/tmp/fhvb.log 2>&1
  pdflatex -interaction=nonstopmode unbox_arr.tex >/tmp/fhv2.log 2>&1
  pdflatex -interaction=nonstopmode unbox_arr.tex >/tmp/fhv3.log 2>&1 )
# grep -c prints "0" AND exits 1 on no match; capture cleanly with `|| true`
# and force a single integer so the comparison below is robust.
errs=$( { grep -c '^!' /tmp/fhv3.log 2>/dev/null || true; } | head -1 )
over=$( { grep -ic 'overfull \\hbox' paper/unbox_arr.log 2>/dev/null || true; } | head -1 )
errs=${errs:-0}; over=${over:-0}
pages=$(pdfinfo paper/unbox_arr.pdf 2>/dev/null | awk '/Pages/{print $2}')
echo "[4/5] pages=$pages errors=$errs overfull=$over"
[ "$errs" -eq 0 ] 2>/dev/null || { echo "LaTeX errors after swap — inspect /tmp/fhv3.log" >&2; exit 5; }

# --- 5. report -------------------------------------------------------------
echo "[5/5] g1_results.json summary:"
"$PY" - <<'PYEOF'
import json, pathlib
g = json.loads(pathlib.Path("annotation/g1_results.json").read_text())
def f(x, n=3): return "n/a" if x is None else f"{x:.{n}f}"
pw = g.get("pairwise_cohens_kappa_binary", {})
ks = [v["kappa"] for v in pw.values() if v.get("kappa") is not None]
print("  raters              :", g.get("raters"))
print("  Krippendorff alpha  :", f(g.get("krippendorff_alpha_interval")))
print("  mean pairwise kappa :", f(sum(ks)/len(ks) if ks else None))
print("  majority match rate :", f(g.get("majority_match_rate"), 2))
ac = g.get("attention_check_pass", {})
tp = sum(v["passed"] for v in ac.values()); tt = sum(v["total"] for v in ac.values())
print("  attention-check pass:", f(tp/tt if tt else None, 2), f"({tp}/{tt})")
PYEOF

echo
echo "DONE. §5.4 now reports real human results in unbox_arr.tex."
echo "Review paper/unbox_arr.pdf, then: git add -A && commit."
echo "Revert anytime: $PY scripts/render_human_validation.py protocol --apply"
