#!/usr/bin/env python3
"""Compute G1 inter-rater agreement and human-vs-rule-based kappa.

Inputs (in annotation/):
  pairs_master.csv             - condition / rank metadata
  *_complete.csv               - one per rater (annotate.html export:
                                 pair_id, score, note). Rater name is the
                                 filename stem minus "_complete". Any N>=2
                                 raters supported; pairs_*.csv are ignored.

Outputs:
  annotation/g1_results.json   - alpha, kappa, per-condition match rates,
                                 attention-check pass rates

Score scale: 1-5 Likert. Binary derivation: score >= 4 -> match.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = PROJECT_ROOT / "annotation"


def load_rater(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with open(path, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM from header
        for row in csv.DictReader(f):
            pid = row["pair_id"]
            s = row.get("score", "").strip()
            if s:
                try:
                    rows[pid] = int(s)
                except ValueError:
                    print(f"WARN: non-int score in {path.name} pair {pid}: {s!r}", file=sys.stderr)
    return rows


def load_master(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["pair_id"]] = row
    return out


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na * nb > 0 else 0.0


def rule_based_decisions(master: dict[str, dict], pids: list[str],
                         threshold: float = 0.65,
                         model: str = "text-embedding-3-small") -> dict[str, int] | None:
    """The SAME rule-based matcher as scripts/canonical_evaluator.py applied to
    each sampled pair: 1 if max cosine(extracted, {target}+aliases) >= threshold.

    Returns {pair_id: 0|1}, or None if no OpenAI key is available (so the
    rest of the analysis still runs offline). Loads the key from env or .env,
    mirroring canonical_evaluator.py.
    """
    import os
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        env = PROJECT_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    os.environ["OPENAI_API_KEY"] = key
    if not key:
        print("WARN: no OPENAI_API_KEY; skipping human-vs-rule-based metric",
              file=sys.stderr)
        return None

    from openai import OpenAI
    client = OpenAI()

    # One batched embedding call over every distinct text.
    texts: list[str] = []
    index: dict[str, int] = {}

    def add(t: str) -> int:
        t = (t or "").strip()
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]

    spec = {}
    for pid in pids:
        m = master[pid]
        ex_i = add(m["extracted_assumption"])
        gts = [m["target_assumption"]] + [
            a.strip() for a in (m.get("target_aliases") or "").split(";") if a.strip()
        ]
        gt_idx = [add(g) for g in gts]
        spec[pid] = (ex_i, gt_idx)

    embs: list = []
    for i in range(0, len(texts), 512):
        resp = client.embeddings.create(model=model, input=texts[i:i + 512])
        embs.extend(e.embedding for e in resp.data)

    decisions: dict[str, int] = {}
    for pid, (ex_i, gt_idx) in spec.items():
        best = max(_cosine(embs[ex_i], embs[g]) for g in gt_idx)
        decisions[pid] = 1 if best >= threshold else 0
    return decisions


def cohens_kappa_binary(a: dict[str, int], b: dict[str, int]) -> tuple[float | None, int]:
    pids = sorted(set(a) & set(b))
    n = len(pids)
    if n == 0:
        return None, 0
    agree = sum(1 for p in pids if a[p] == b[p])
    p_a = sum(a[p] for p in pids) / n
    p_b = sum(b[p] for p in pids) / n
    p_o = agree / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    if 1 - p_e == 0:
        return None, n
    return (p_o - p_e) / (1 - p_e), n


def krippendorff_alpha_interval(matrix: list[list[int | None]]) -> float | None:
    """Krippendorff's alpha for interval data (1-5 Likert).

    matrix: rows = units (pairs), cols = raters; None for missing.

    Canonical coincidence-matrix form (Krippendorff 2011, "Computing
    Krippendorff's Alpha-Reliability"). Correct for unbalanced/missing data
    (the earlier pairwise n'-normalized form was wrong when units have
    differing numbers of ratings; see tests/test_krippendorff.py). Each
    ordered within-unit value pair contributes 1/(m_u - 1) coincidence mass:
        D_o = (1/n)  sum_{v,v'} o_{vv'} (v-v')^2
        D_e = (1/(n(n-1))) sum_{v,v'} n_v n_{v'} (v-v')^2
        alpha = 1 - D_o / D_e
    where n = total coincidence mass over units with >= 2 ratings.
    """
    from collections import defaultdict

    coinc: dict = defaultdict(float)
    for row in matrix:
        vals = [v for v in row if v is not None]
        m = len(vals)
        if m < 2:
            continue
        w = 1.0 / (m - 1)
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[(vals[i], vals[j])] += w
    n = sum(coinc.values())
    if n < 2:
        return None
    d_o = sum(c * (a - b) ** 2 for (a, b), c in coinc.items()) / n
    marg: dict = defaultdict(float)
    for (a, _b), c in coinc.items():
        marg[a] += c
    d_e = 0.0
    for a, na in marg.items():
        for b, nb in marg.items():
            d_e += na * nb * (a - b) ** 2
    d_e /= n * (n - 1)
    if d_e == 0:
        return None
    return 1.0 - d_o / d_e


def main() -> int:
    raters: dict[str, dict[str, int]] = {}
    for p in sorted(ANNOTATION_DIR.glob("*_complete.csv")):
        if p.name.startswith("pairs_"):
            continue
        name = p.name[: -len("_complete.csv")]
        raters[name] = load_rater(p)

    if len(raters) < 2:
        print(f"ERROR: need >=2 *_complete.csv rater files in {ANNOTATION_DIR}",
              file=sys.stderr)
        return 1

    # Anonymize rater identifiers for the committed output (ACL Responsible
    # NLP: do not expose annotator identities). Real names/initials live only
    # in the gitignored raw *_complete.csv; the aggregate uses R01..RNN.
    anon = {real: f"R{i:02d}" for i, real in enumerate(sorted(raters), 1)}
    raters = {anon[real]: scores for real, scores in raters.items()}
    print(f"Loaded {len(raters)} raters (anonymized R01..R{len(raters):02d})",
          file=sys.stderr)

    master = load_master(ANNOTATION_DIR / "pairs_master.csv")
    rater_names = sorted(raters.keys())

    # Stratified pairs only for primary metrics
    stratified_pids = sorted([p for p, m in master.items()
                              if m["is_attention_check"].lower() == "false"])

    # Build matrix for Krippendorff's alpha
    matrix = [[raters[rn].get(pid) for rn in rater_names] for pid in stratified_pids]
    alpha = krippendorff_alpha_interval(matrix)

    # Binary derivations: score >= 4 -> match
    binary = {rn: {p: int(s >= 4) for p, s in r.items()} for rn, r in raters.items()}

    pairwise_kappa = {}
    for i, a in enumerate(rater_names):
        for b in rater_names[i + 1:]:
            ba = {p: v for p, v in binary[a].items() if p in stratified_pids}
            bb = {p: v for p, v in binary[b].items() if p in stratified_pids}
            k, n = cohens_kappa_binary(ba, bb)
            pairwise_kappa[f"{a}_vs_{b}"] = {"kappa": k, "n": n}

    # Majority vote per pair
    majority: dict[str, int] = {}
    for pid in stratified_pids:
        votes = [binary[rn].get(pid) for rn in rater_names]
        votes = [v for v in votes if v is not None]
        if len(votes) >= 2:
            majority[pid] = 1 if sum(votes) >= (len(votes) + 1) // 2 else 0

    # Per-condition breakdown
    per_cond: dict[str, dict[str, int]] = {}
    for pid, m in master.items():
        if pid not in majority:
            continue
        c = m["condition"]
        per_cond.setdefault(c, {"n": 0, "match": 0})
        per_cond[c]["n"] += 1
        per_cond[c]["match"] += majority[pid]

    # Attention check pass: should score 1-2
    ac_pids = [p for p, m in master.items() if m["is_attention_check"].lower() == "true"]
    ac_pass = {}
    for rn in rater_names:
        scored = [raters[rn][p] for p in ac_pids if p in raters[rn]]
        passed = sum(1 for s in scored if s <= 2)
        ac_pass[rn] = {"passed": passed, "total": len(scored)}

    # Pre-registered metric: human-majority vs. rule-based matcher on the SAME
    # stratified pairs (same matcher as canonical_evaluator: text-embedding-3-
    # small, cosine >= 0.65). This is the like-for-like leniency comparison.
    rule = rule_based_decisions(master, stratified_pids)
    human_vs_rule = None
    rule_per_cond = None
    if rule is not None:
        common = [p for p in stratified_pids if p in majority and p in rule]
        hk = {p: majority[p] for p in common}
        rk = {p: rule[p] for p in common}
        k, n = cohens_kappa_binary(hk, rk)
        agree = sum(1 for p in common if hk[p] == rk[p]) / len(common) if common else None
        rule_match_rate = sum(rule[p] for p in common) / len(common) if common else None
        human_match_rate = sum(majority[p] for p in common) / len(common) if common else None
        human_vs_rule = {
            "kappa": k, "n": n, "raw_agreement": agree,
            "rule_match_rate": rule_match_rate,
            "human_match_rate": human_match_rate,
            "threshold": 0.65, "embedding_model": "text-embedding-3-small",
        }
        rule_per_cond = {}
        for pid in common:
            c = master[pid]["condition"]
            d = rule_per_cond.setdefault(c, {"n": 0, "rule_match": 0, "human_match": 0})
            d["n"] += 1
            d["rule_match"] += rule[pid]
            d["human_match"] += majority[pid]

    out = {
        "n_stratified_pairs": len(stratified_pids),
        "n_attention_checks": len(ac_pids),
        "raters": rater_names,
        "krippendorff_alpha_interval": alpha,
        "pairwise_cohens_kappa_binary": pairwise_kappa,
        "majority_match_rate": (sum(majority.values()) / len(majority)) if majority else None,
        "per_condition_match_rate": {
            c: {"n": d["n"], "match": d["match"], "rate": d["match"] / d["n"] if d["n"] else None}
            for c, d in sorted(per_cond.items())
        },
        "attention_check_pass": ac_pass,
        "human_vs_rule": human_vs_rule,
        "per_condition_human_vs_rule": (
            {c: {"n": d["n"],
                 "rule_rate": d["rule_match"] / d["n"] if d["n"] else None,
                 "human_rate": d["human_match"] / d["n"] if d["n"] else None}
             for c, d in sorted(rule_per_cond.items())}
            if rule_per_cond is not None else None
        ),
    }

    out_path = ANNOTATION_DIR / "g1_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
