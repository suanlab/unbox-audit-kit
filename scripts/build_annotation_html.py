#!/usr/bin/env python3
"""Generate a self-contained HTML annotation tool from pairs_for_raters.csv.

The output is a single HTML file with all 80 pairs embedded. Annotators
just double-click it; no install, no internet, no Google account needed.

Features:
  - Big readable layout, one pair per screen
  - Score buttons 1-5 (also keyboard 1-5)
  - Next/Prev navigation (arrow keys)
  - Note field
  - Progress bar + per-pair status
  - Auto-save to browser localStorage (keyed by rater ID)
  - "Export CSV" downloads <raterId>_complete.csv

Output:  annotation/annotate.html
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOT = PROJECT_ROOT / "annotation"


def load_pairs() -> list[dict]:
    pairs = []
    with open(ANNOT / "pairs_for_raters.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs.append({
                "pair_id": row["pair_id"],
                "extracted_assumption": row["extracted_assumption"],
                "target_assumption": row["target_assumption"],
                "target_aliases": row["target_aliases"],
            })
    return pairs


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Unbox 가정 회상 평가</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
         max-width: 880px; margin: 1em auto; padding: 0 1em; color: #222; line-height: 1.5; }
  h1 { font-size: 1.4em; margin-bottom: 0.3em; }
  .top { display: flex; gap: 1em; align-items: center; flex-wrap: wrap; }
  .rater { display: flex; gap: 0.5em; align-items: center; }
  .rater input { padding: 0.4em; font-size: 1em; }
  .progress { background: #eee; height: 10px; border-radius: 5px; overflow: hidden; flex: 1; min-width: 200px; }
  .progress-bar { background: linear-gradient(90deg,#4caf50,#8bc34a); height: 100%; transition: width 0.3s; }
  .counter { font-size: 0.95em; color: #555; min-width: 90px; text-align: right; }
  .card { background: #fafafa; border: 1px solid #ddd; padding: 1.2em; border-radius: 10px; margin: 1.2em 0; }
  .pair-id { font-size: 0.85em; color: #888; }
  .label { font-weight: 600; color: #555; font-size: 0.9em; margin-top: 0.7em; }
  .text { font-size: 1.15em; padding: 0.6em 0.8em; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; }
  .text.target { background: #f0f7f0; border-color: #b8d8b8; }
  .aliases-box { background: #f7f7f0; border: 1px dashed #d8d6c0; padding: 0.5em 0.8em; border-radius: 6px;
                 font-size: 0.95em; color: #555; }
  .scores { display: flex; gap: 0.5em; margin: 1em 0; flex-wrap: wrap; }
  .score-btn { padding: 0.7em 1.1em; font-size: 1.05em; cursor: pointer; border: 2px solid #ccc;
               background: #fff; border-radius: 6px; min-width: 70px; transition: all 0.15s; }
  .score-btn:hover { border-color: #4caf50; }
  .score-btn.selected { background: #4caf50; color: #fff; border-color: #4caf50; }
  .score-btn .num { font-weight: bold; font-size: 1.2em; display: block; }
  .score-btn .hint { font-size: 0.78em; opacity: 0.85; }
  textarea { width: 100%; min-height: 56px; padding: 0.5em; font-size: 0.95em; border: 1px solid #ddd; border-radius: 6px;
             font-family: inherit; }
  .nav { display: flex; justify-content: space-between; align-items: center; margin: 1em 0; }
  .nav button { padding: 0.6em 1.2em; font-size: 1em; cursor: pointer; border: 1px solid #aaa; background: #fff;
                border-radius: 5px; }
  .export-btn { display: block; width: 100%; padding: 1em; font-size: 1.1em; background: #1976d2; color: #fff;
                border: 0; border-radius: 6px; cursor: pointer; margin-top: 1em; }
  .export-btn:hover { background: #1565c0; }
  .legend { background: #fffceb; border: 1px solid #f0e6a0; padding: 0.6em 0.9em; border-radius: 6px;
            font-size: 0.9em; margin: 0.5em 0 1em; }
  kbd { background: #eee; border: 1px solid #ccc; border-radius: 3px; padding: 0 5px; font-family: monospace;
        font-size: 0.85em; }
  .status { font-size: 0.85em; color: #888; margin-top: 0.3em; }
  .done { color: #4caf50; }
</style>
</head>
<body>
<h1>Unbox 가정 회상 평가 (80 pairs)</h1>
<div class="legend">
  단축키: <kbd>1</kbd>~<kbd>5</kbd> 점수 / <kbd>←</kbd> 이전 / <kbd>→</kbd> 다음 (또는 점수 클릭 시 자동 다음).
  진행은 브라우저에 자동 저장됩니다. 모두 완료 후 맨 아래 <strong>CSV 내보내기</strong> 클릭.
</div>

<div class="top">
  <div class="rater">
    <label>당신의 식별자:</label>
    <input id="raterId" placeholder="rater1" />
  </div>
  <div class="progress"><div class="progress-bar" id="progressBar"></div></div>
  <div class="counter" id="counter">0 / 0</div>
</div>

<div class="card">
  <div class="pair-id">Pair <span id="pairId"></span> · <span id="position"></span></div>

  <div class="label">추출된 가정 (extracted_assumption)</div>
  <div class="text" id="extracted"></div>

  <div class="label">정답 가정 (target_assumption)</div>
  <div class="text target" id="target"></div>

  <div class="label">정답의 유의 표현 (target_aliases)</div>
  <div class="aliases-box" id="aliases"></div>

  <div class="label">점수 (1=다름, 3=모호, 5=같음)</div>
  <div class="scores">
    <button class="score-btn" data-score="1"><span class="num">1</span><span class="hint">명백히 다름</span></button>
    <button class="score-btn" data-score="2"><span class="num">2</span><span class="hint">관련 있으나 다름</span></button>
    <button class="score-btn" data-score="3"><span class="num">3</span><span class="hint">부분/모호</span></button>
    <button class="score-btn" data-score="4"><span class="num">4</span><span class="hint">같은 가정</span></button>
    <button class="score-btn" data-score="5"><span class="num">5</span><span class="hint">명백히 같음</span></button>
  </div>

  <div class="label">메모 (선택)</div>
  <textarea id="note" placeholder="모호한 경우 한 줄 메모"></textarea>

  <div class="status" id="status"></div>
</div>

<div class="nav">
  <button onclick="prev()">← 이전</button>
  <span id="navInfo"></span>
  <button onclick="next()">다음 →</button>
</div>

<button class="export-btn" onclick="exportCsv()">완료 → CSV 내보내기</button>

<script>
const PAIRS = __PAIRS_JSON__;
let idx = 0;
let state = {};
let raterId = "";
const raterInput = document.getElementById("raterId");
const noteEl = document.getElementById("note");

function storageKey() { return "unbox_annot_" + (raterId || "anon"); }
function loadState() {
  raterId = (raterInput.value || "anon").trim();
  const s = localStorage.getItem(storageKey());
  state = s ? JSON.parse(s) : {};
  // Jump to first unanswered
  const firstUnanswered = PAIRS.findIndex(p => !(state[p.pair_id] && state[p.pair_id].score));
  idx = firstUnanswered >= 0 ? firstUnanswered : 0;
  render();
}
function saveState() { localStorage.setItem(storageKey(), JSON.stringify(state)); }
function persistNote() {
  const p = PAIRS[idx];
  state[p.pair_id] = state[p.pair_id] || {};
  state[p.pair_id].note = noteEl.value;
  saveState();
}
function render() {
  const p = PAIRS[idx];
  document.getElementById("pairId").textContent = p.pair_id;
  document.getElementById("extracted").textContent = p.extracted_assumption;
  document.getElementById("target").textContent = p.target_assumption;
  document.getElementById("aliases").textContent = p.target_aliases || "(없음)";
  noteEl.value = (state[p.pair_id] || {}).note || "";
  const cur = (state[p.pair_id] || {}).score;
  document.querySelectorAll(".score-btn").forEach(b => {
    b.classList.toggle("selected", parseInt(b.dataset.score) === cur);
  });
  const done = Object.values(state).filter(v => v.score).length;
  document.getElementById("counter").textContent = `${done} / ${PAIRS.length}`;
  document.getElementById("progressBar").style.width = (done / PAIRS.length * 100) + "%";
  document.getElementById("position").textContent = `${idx + 1} / ${PAIRS.length}`;
  document.getElementById("navInfo").textContent = `${idx + 1} / ${PAIRS.length}`;
  document.getElementById("status").innerHTML = cur ? `<span class="done">✓ 저장됨</span>` : "(미평가)";
}
function setScore(s) {
  const p = PAIRS[idx];
  state[p.pair_id] = state[p.pair_id] || {};
  state[p.pair_id].score = s;
  state[p.pair_id].note = noteEl.value;
  saveState();
  render();
  // auto-advance after small delay so user sees the green check
  setTimeout(() => { if (idx < PAIRS.length - 1) { idx++; render(); } }, 180);
}
function next() { persistNote(); if (idx < PAIRS.length - 1) { idx++; render(); } }
function prev() { persistNote(); if (idx > 0) { idx--; render(); } }
function exportCsv() {
  persistNote();
  const lines = ["pair_id,extracted_assumption,target_assumption,target_aliases,score,note"];
  const esc = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
  for (const p of PAIRS) {
    const s = state[p.pair_id] || {};
    lines.push([p.pair_id, p.extracted_assumption, p.target_assumption, p.target_aliases, s.score || "", s.note || ""].map(esc).join(","));
  }
  const csv = lines.join("\r\n") + "\r\n";
  const blob = new Blob(["﻿" + csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (raterId || "anon") + "_complete.csv";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  const done = Object.values(state).filter(v => v.score).length;
  if (done < PAIRS.length) alert(`주의: ${done}/${PAIRS.length}만 평가되었습니다. CSV에는 빈 칸으로 저장됩니다.`);
  else alert("완료! CSV가 다운로드되었습니다. 이 파일을 회수자에게 보내주세요.");
}

document.querySelectorAll(".score-btn").forEach(b =>
  b.addEventListener("click", () => setScore(parseInt(b.dataset.score)))
);
raterInput.addEventListener("change", loadState);
raterInput.addEventListener("blur", loadState);
noteEl.addEventListener("blur", persistNote);
window.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
  if (e.key >= "1" && e.key <= "5") { e.preventDefault(); setScore(parseInt(e.key)); }
  else if (e.key === "ArrowRight") { e.preventDefault(); next(); }
  else if (e.key === "ArrowLeft")  { e.preventDefault(); prev(); }
});
window.addEventListener("beforeunload", persistNote);
loadState();
</script>
</body>
</html>
"""


def main() -> int:
    pairs = load_pairs()
    html = HTML_TEMPLATE.replace("__PAIRS_JSON__", json.dumps(pairs, ensure_ascii=False))
    out = ANNOT / "annotate.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(pairs)} pairs, {out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
