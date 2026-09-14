#!/usr/bin/env bash
# Package the exact 3 files to hand to an acquaintance-pool rater.
# Output: annotation_kit.zip  (annotate.html + INSTRUCTIONS.md + RUBRIC.md)
#
# The rater double-clicks annotate.html, scores 80 pairs in the browser
# (auto-saved, no internet), then clicks "CSV 내보내기" and returns the
# downloaded <id>_complete.csv. Nothing else is needed.
#
# Usage:  bash scripts/build_annotation_kit.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Rebuild annotate.html so it always reflects the current pairs CSV.
PY=".venv/bin/python"; [ -x "$PY" ] || PY=python3
"$PY" scripts/build_annotation_html.py

req=(annotation/annotate.html annotation/INSTRUCTIONS.md annotation/RUBRIC.md)
for f in "${req[@]}"; do
  [ -f "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

rm -f annotation_kit.zip
# store annotate.html etc. at the zip root (flat) so the rater sees 3 files
( cd annotation && zip -j -q ../annotation_kit.zip annotate.html INSTRUCTIONS.md RUBRIC.md )

echo "Wrote annotation_kit.zip:"
unzip -l annotation_kit.zip | sed 's/^/  /'
echo
echo "Send annotation_kit.zip to each rater with this one line:"
echo '  "annotate.html 더블클릭 → 식별자 입력(rater1/2/3) → 80쌍 1~5점 → 맨 아래 CSV 내보내기 → 받은 파일 회신. 인터넷 불필요, 30~45분."'
