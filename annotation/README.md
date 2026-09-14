# annotation/ — G1 Human Annotation (지인 풀)

EMNLP 2026 ARR 제출용 G1 인간 평가. 3 raters x 80 pairs.

## 0. 디렉토리 구조

```
annotation/
├── README.md                  # this file
├── INSTRUCTIONS.md            # 어노테이터 안내문 (Korean)
├── RUBRIC.md                  # 채점 기준 + 예시 12개 (Korean)
├── recruit_template.md        # 모집 메시지 템플릿
├── pairs_master.csv           # 전체 메타데이터 (PRIVATE — 어노테이터에게 송부 X)
├── pairs_for_raters.csv       # 어노테이터 송부용 (조건/순위 hidden, shuffled)
├── rater1_complete.csv        # 회수 (gitignored)
├── rater2_complete.csv        # 회수 (gitignored)
├── rater3_complete.csv        # 회수 (gitignored)
└── g1_results.json            # 분석 결과 (compute_human_kappa.py 출력)
```

## 1. 빠른 시작

```bash
# (1) 페어 샘플링 (deterministic, seed=42)
.venv/bin/python scripts/sample_annotation_pairs.py
# -> pairs_master.csv (80) + pairs_for_raters.csv (80, shuffled)

# (2) 어노테이션 키트 패키징 (annotate.html 재빌드 + zip 한 개)
bash scripts/build_annotation_kit.sh
# -> annotation_kit.zip  (annotate.html + INSTRUCTIONS.md + RUBRIC.md)
#    인터넷/계정 불필요. 그대로 지인에게 송부.

# (3) 모집 (recruit_template.md) → 3명 확정 → annotation_kit.zip 송부
#     어노테이터: annotate.html 더블클릭 → 식별자 입력 → 80쌍 1~5점
#     → 맨 아래 "CSV 내보내기" → 받은 <id>_complete.csv 회신

# (4) 회수 후 파일명 통일 (>=2개면 진행 가능)
mv 회수1.csv annotation/rater1_complete.csv
mv 회수2.csv annotation/rater2_complete.csv
mv 회수3.csv annotation/rater3_complete.csv

# (5) 원커맨드 마무리: kappa 계산 → §5.4 결과표 스왑 → 재컴파일 → 리포트
bash scripts/finalize_human_validation.sh
#  - compute_human_kappa.py     -> annotation/g1_results.json
#  - render_human_validation.py results --apply
#    (unbox_arr.tex [ACL/EMNLP 제출본] §5.4를
#     사전등록 프로토콜 표 → 실제 인간 결과 표로 교체)
#  - unbox_arr.tex 재컴파일 + 핵심 수치 출력
# 라벨 미회수 시: 아무것도 안 함 (논문은 fallback B로 제출 가능)
# 되돌리기: python scripts/render_human_validation.py protocol --apply --tex all
```

> 엔드투엔드 파이프라인은 합성 데이터로 검증 완료 (protocol→results→protocol
> 라운드트립이 byte-identical, 0 LaTeX 오류). 실제 라벨이 오면 (5)만 실행.

## 2. 페어 디자인 (80 = 72 + 8)

**Stratified (72)**: 4 paradigms x 3 conditions x 6 confidence ranks
- paradigms: transformer, diffusion, icl, vit (모든 조건에서 데이터 가용)
- conditions: gpt4o_corpus, no_corpus, claude_corpus
- ranks: 1, 3, 5, 8, 12, 17 (confidence-ordered top-k)

**Attention checks (8)**: 의도적으로 잘못된 paradigm 타겟과 페어링
- transformer 추출 vs vit 타겟 (×2)
- diffusion 추출 vs icl 타겟 (×2)
- icl 추출 vs transformer 타겟 (×2)
- vit 추출 vs diffusion 타겟 (×2)
- 정상 어노테이터는 1~2점 줘야 함 (부주의 어노테이터 필터)

## 3. 평가 지표 (compute_human_kappa.py 산출)

- **Krippendorff's α (interval, 1-5 Likert)** — 3-rater 일치도, 권장 ≥ 0.667
- **Pairwise Cohen's κ (binary, score≥4=match)** — 3 쌍 평균
- **Per-condition match rate** — 3가지 조건별 majority-vote match율
- **Attention check pass** — rater별 attention check 통과 (1~2점 비율)
- **Human-vs-rule-based κ**: G1 결과를 paper에 보고하여 rule-based κ=0.97이 인간 판단과 일치하는지 검증

## 4. 마감 / 마일스톤

| 일자 | 마일스톤 |
|------|--------|
| 2026-05-09 | 페어 샘플링 + 자료 준비 (today) |
| 2026-05-09~10 | 지인 풀 모집 메시지 발송 |
| 2026-05-11 | 3 raters 확정, 자료 송부 |
| 2026-05-19 | 회수 마감 |
| 2026-05-20 | 분석 + 논문 §5/§7 통합 |
| 2026-05-25 | EMNLP ARR 제출 |

## 5. 주의

- `pairs_master.csv`는 condition/rank/source 노출이 있으므로 어노테이터에게 송부 금지
- 어노테이터 회수 파일은 gitignore (`rater*_complete.csv`)
- attention check 통과율 < 75% rater는 분석에서 제외 검토
- κ < 0.4면 rubric 재교정 후 재실행 (시간 부족 시 발견된 disagreement를 솔직히 보고)

## 6. 분석 결과 → 논문 통합 위치

- §5 (Results) — 새 표 "Human Validation of Rule-Based Matching":
  - Krippendorff's α
  - Human-rule κ
  - Per-condition human match rate vs canonical metric
- §7 (Limitations) — 만약 κ가 낮으면 솔직히 보고
- Appendix — 전체 프로토콜, RUBRIC, 예시
