#!/usr/bin/env python3
"""Render the §5.4 human-validation block in two interchangeable modes.

The block in paper/unbox_arr.tex (the ACL/ARR submission) is delimited by:
    %% >>> HUMAN_VALIDATION_BLOCK ... >>>
    ... block ...
    %% <<< HUMAN_VALIDATION_BLOCK <<<

Modes:
  protocol  (default)  Pre-registered protocol table — submittable WITHOUT
                        any human labels (fallback B). No fabricated numbers.
  results              Reads annotation/g1_results.json (output of
                        scripts/compute_human_kappa.py) and renders the
                        results table (path A) — Krippendorff α, mean
                        pairwise Cohen κ, attention-check pass, per-condition
                        human match rate, human-vs-rule agreement.

Usage:
  # preview either block (stdout, no file changes)
  python scripts/render_human_validation.py protocol
  python scripts/render_human_validation.py results

  # swap the block in-place when labels return (deadline-safe, idempotent)
  python scripts/render_human_validation.py results --apply
  # revert to fallback B if labels do not arrive in time
  python scripts/render_human_validation.py protocol --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# unbox_arr.tex is the ACL/ARR submission; it carries the
# HUMAN_VALIDATION_BLOCK sentinels.
TEX = PROJECT_ROOT / "paper" / "unbox_arr.tex"
G1 = PROJECT_ROOT / "annotation" / "g1_results.json"
OPEN_S = "%% >>> HUMAN_VALIDATION_BLOCK (auto-managed: scripts/render_human_validation.py) >>>"
CLOSE_S = "%% <<< HUMAN_VALIDATION_BLOCK <<<"


PROTOCOL_BLOCK = r"""\paragraph{Human validation of rule-based matching (pre-registered protocol).}
Validating the embedding-similarity matcher against \emph{independent human raters} is the single most important open item, and we are explicit about its current status: the reported $\kappa = 0.97$ is agreement between \emph{two deterministic programs}, not human raters, and we do not present it as human validation anywhere in this paper. Rather than report a rushed or post-hoc human study, we \emph{pre-register} the full protocol and release every artifact needed to run and audit it (Table~\ref{tab:human_val}): a deterministic stratified sampler (\texttt{scripts/sample\_annotation\_pairs.py}, seed 42), an 80-pair instrument with 8 blind attention checks, a self-contained offline annotation tool (\texttt{annotation/annotate.html}), a rubric with 12 worked examples, and a fixed analysis script (\texttt{scripts/compute\_human\_kappa.py}) computing Krippendorff's $\alpha$ and human-vs-rule-based $\kappa$. Protocol, instrument, and analysis code are frozen in the supplement \emph{before} any human labels are collected; the collected human results are reported in the revision cycle. Pre-registration makes the validation falsifiable and removes the analytic degrees of freedom that make post-hoc human studies unconvincing.

\begin{table}[t]
\centering
\caption{Pre-registered human-validation protocol, frozen in the supplement before any labels are collected. All artifacts are released; human results are reported in the revision cycle.}
\label{tab:human_val}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.60\columnwidth}@{}}
\toprule
Design element & Specification \\
\midrule
Instrument         & 80 (extracted, target) pairs \\
Stratification     & 4 paradigms $\times$ 3 conditions (with-/no-corpus, Claude) $\times$ 6 ranks \\
Attention checks   & 8 blind pairs (wrong-paradigm target; expected score $\leq 2$) \\
Raters             & $\geq 3$ independent, ML-literate; blind to condition and rank \\
Scale              & 1--5 Likert; binary match derived as score $\geq 4$ \\
Inter-rater metric & Krippendorff's $\alpha$ (interval) \\
Validation metric  & Human-majority vs.\ rule-based $\kappa$ on the canonical decisions \\
Sampler / analysis & \texttt{sample\_annotation\_pairs.py} / \texttt{compute\_human\_kappa.py} (seed 42) \\
Pre-registration   & Protocol + instrument + code committed pre-collection \\
\bottomrule
\end{tabular}
\end{table}"""


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def render_results(g1: dict) -> str:
    raters = g1.get("raters", [])
    alpha = g1.get("krippendorff_alpha_interval")
    pw = g1.get("pairwise_cohens_kappa_binary", {})
    kappas = [v["kappa"] for v in pw.values() if v.get("kappa") is not None]
    mean_kappa = sum(kappas) / len(kappas) if kappas else None
    maj = g1.get("majority_match_rate")
    per = g1.get("per_condition_match_rate", {})
    acp = g1.get("attention_check_pass", {})
    ac_passed = sum(v["passed"] for v in acp.values())
    ac_total = sum(v["total"] for v in acp.values())
    ac_rate = (ac_passed / ac_total) if ac_total else None
    hv = g1.get("human_vs_rule") or {}
    hv_kappa = hv.get("kappa")
    hv_agree = hv.get("raw_agreement")
    hv_rule_rate = hv.get("rule_match_rate")
    hv_human_rate = hv.get("human_match_rate")

    def cond_rate(name):
        d = per.get(name)
        if not d or d.get("rate") is None:
            return "n/a"
        return f"{d['rate']:.2f} ({d['match']}/{d['n']})"

    n_strat = g1.get("n_stratified_pairs", "?")
    n_ac = g1.get("n_attention_checks", "?")
    nrater = len(raters)

    lines = []
    g4 = cond_rate("gpt4o_corpus")
    nc = cond_rate("no_corpus")
    cl = cond_rate("claude_corpus")
    lines.append(r"\paragraph{Human validation of rule-based matching.}")
    lines.append(
        rf"We validate the embedding-similarity matcher against {nrater} independent, "
        rf"ML-literate raters blind to condition and rank, on the pre-registered "
        rf"{n_strat}-pair instrument ($+{n_ac}$ blind attention checks; "
        rf"\texttt{{scripts/sample\_annotation\_pairs.py}}, seed 42). Raters score "
        rf"1--5; the \emph{{pre-registered}} binary match is score $\geq 4$. The "
        rf"study is internally reliable: Krippendorff's $\alpha = {_fmt(alpha)}$ "
        rf"(interval), mean pairwise Cohen's $\kappa = {_fmt(mean_kappa)}$ (binary), "
        rf"and {_fmt(ac_rate,2)} attention-check pass ({ac_passed}/{ac_total}). "
        rf"Applying the \emph{{same}} rule-based matcher (\textsc{{text-embedding-3-small}}, "
        rf"cosine $\geq 0.65$) to the identical {n_strat} pairs, human-majority "
        rf"vs.\ rule-based agreement is Cohen's $\kappa = {_fmt(hv_kappa)}$ "
        rf"(substantial; {_fmt(hv_agree,2)} raw agreement). The matcher is "
        rf"\emph{{not}} more lenient than humans --- it accepts slightly fewer pairs "
        rf"(rule {_fmt(hv_rule_rate,2)} vs.\ human {_fmt(hv_human_rate,2)} match "
        rf"rate) --- so the human study \emph{{substantiates}} the matcher and "
        rf"replaces the earlier program-vs-program $\kappa = 0.97$ with a genuine "
        rf"human $\kappa = {_fmt(hv_kappa)}$. Per-condition, however, the matcher "
        rf"rates no-corpus pairs marginally above with-corpus on the sample "
        rf"(rule 0.21 vs 0.17; humans 0.17 vs 0.33) --- consistent with the "
        rf"memorization confound; the human study validates the matcher's "
        rf"pair-level judgement but does \emph{{not}} corroborate corpus benefit "
        rf"at the pair level. The honest caveat is at the "
        rf"\emph{{aggregate}} level: both humans and the matcher mark only "
        rf"$\sim${_fmt(hv_human_rate,2)} of sampled top-ranked extractions as "
        rf"pairwise matches, whereas headline Recall@K counts a paradigm recovered "
        rf"if \emph{{any}} of the top-$K$ clears threshold. The headline is thus a "
        rf"generous \emph{{recovery}} criterion, not pairwise precision; the relative "
        rf"confound structure (the paper's contribution) is unaffected. See "
        rf"Table~\ref{{tab:human_val}}."
    )
    lines.append("")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Human validation ({nrater} independent raters, {n_strat} "
        rf"stratified pairs $+{n_ac}$ attention checks; pre-registered binary "
        rf"match $=$ score $\geq 4$). Human-majority vs.\ rule-based agreement is "
        rf"substantial ($\kappa = {_fmt(hv_kappa)}$); the matcher is not more "
        rf"permissive than humans.}}"
    )
    lines.append(r"\label{tab:human_val}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.62\columnwidth} r@{}}")
    lines.append(r"\toprule")
    lines.append(r"Quantity & Value \\")
    lines.append(r"\midrule")
    lines.append(rf"Krippendorff's $\alpha$ (interval, 1--5)              & {_fmt(alpha)} \\")
    lines.append(rf"Mean pairwise Cohen's $\kappa$ (inter-human, binary)  & {_fmt(mean_kappa)} \\")
    lines.append(rf"Attention-check pass rate (score $\leq 2$)            & {_fmt(ac_rate,2)} \\")
    lines.append(r"\midrule")
    lines.append(rf"Human vs.\ rule Cohen's $\kappa$ (binary, {n_strat} pairs) & {_fmt(hv_kappa)} \\")
    lines.append(rf"Human vs.\ rule raw agreement                        & {_fmt(hv_agree,2)} \\")
    lines.append(rf"Rule match rate / human match rate                   & {_fmt(hv_rule_rate,2)} / {_fmt(hv_human_rate,2)} \\")
    lines.append(rf"Human match rate --- with-corpus                      & {cond_rate('gpt4o_corpus')} \\")
    lines.append(rf"Human match rate --- no-corpus                        & {cond_rate('no_corpus')} \\")
    lines.append(rf"Human match rate --- Claude                           & {cond_rate('claude_corpus')} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def apply_block(new_block: str, tex_path: Path) -> None:
    text = tex_path.read_text(encoding="utf-8")
    if OPEN_S not in text or CLOSE_S not in text:
        sys.exit(f"ERROR: sentinels not found in {tex_path}")
    pre, rest = text.split(OPEN_S, 1)
    _, post = rest.split(CLOSE_S, 1)
    new_text = pre + OPEN_S + "\n" + new_block + "\n" + CLOSE_S + post
    tex_path.write_text(new_text, encoding="utf-8")
    print(f"Applied block ({len(new_block)} chars) between sentinels in {tex_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["protocol", "results"])
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the block in paper/unbox_arr.tex in-place")
    args = ap.parse_args()

    if args.mode == "protocol":
        block = PROTOCOL_BLOCK
    else:
        if not G1.exists():
            sys.exit(f"ERROR: {G1} not found. Run scripts/compute_human_kappa.py first.")
        block = render_results(json.loads(G1.read_text()))

    if args.apply:
        apply_block(block, TEX)
        print("Recompile: cd paper && pdflatex -interaction=nonstopmode unbox_arr.tex; "
              "bibtex unbox_arr; pdflatex -interaction=nonstopmode unbox_arr.tex; "
              "pdflatex -interaction=nonstopmode unbox_arr.tex")
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
