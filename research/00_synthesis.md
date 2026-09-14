# 00. Synthesis: Cross-Cutting Analysis & Research Gaps

> 5개 서베이에서 수집한 ~130편 논문의 종합 분석. Project Unbox의 연구 포지셔닝.

---

## 1. 현재 연구 지형 (Landscape)

### 1.1 AI4Science가 실제로 달성한 것

| 시스템 | 달성 | 한계 | 진정한 novelty? |
|---|---|---|---|
| **FunSearch** (DeepMind, 2023) | 미해결 수학 문제의 새 해법 | 자동 검증 가능한 도메인에 한정 | **Yes** — 새로운 해법 |
| **AlphaEvolve** (DeepMind, 2025) | 알고리즘 최적화, 구글 컴퓨팅 0.7% 절감 | 기존 알고리즘의 개선 | No — optimization |
| **AI Scientist v1/v2** (Sakana, 2024-25) | End-to-end 논문 생산 | 42% 실험 실패, 57% 수치 날조 | No — 형식만 있고 실질 없음 |
| **CodeScientist** (ACL 2025) | 새로운 연구 방향 19개 발견 | 코드 기반 환경에 제한 | **Partial** |
| **DeepScientist** (2025) | 인간 SOTA를 183.7%까지 초과 | Bayesian optimization 기반 | **Yes** — within defined space |
| **ChemCrow/Coscientist** (2023-24) | 노벨상 반응 재현, 합성 자동화 | 인간이 정의한 목표 내에서만 | No — automation |

**핵심 패턴**: 자동 검증이 가능한 도메인(수학, 코드, 알고리즘)에서는 genuine novelty가 나올 수 있지만, 검증 자체가 주관적인 열린 문제(새로운 이론, 패러다임 전환)에서는 거의 불가능.

### 1.2 Boden의 창의성 분류별 연구 분포

```
Combinatorial (조합적)  ████████████████████████ 60%   ← 대부분의 연구
Exploratory (탐색적)    ████████████ 30%              ← 일부 연구
Transformational (변환적) ████ 10%                     ← 거의 미탐색 ★
```

**Unbox가 노려야 할 곳은 변환적 창의성(Transformational creativity) — 개념 공간 자체를 변형하는 것.**

---

## 2. 핵심 연구 갭 (Research Gaps)

### Gap 1: Transformational Creativity를 직접 다루는 시스템 부재

- Shahhosseini et al. (2025) 서베이: "most current methods address combinatorial/exploratory, NOT transformational creativity"
- Schapiro et al. (ICCC 2025): Boden + Kuhn을 결합한 이론적 프레임워크는 있지만, 실제 시스템은 없음
- 유일한 시도: Multi-agent debate (가장 높은 transformational potential) — 하지만 아직 과학적 발견에 직접 적용된 사례 없음

### Gap 2: 숨겨진 가정(Hidden Assumption)을 체계적으로 찾고 깨는 방법론 부재

- AutoTRIZ (2024)가 TRIZ + LLM 결합을 시도했지만, AI/ML 도메인에 적용된 사례 없음
- AI 역사의 패러다임 전환 분석(02 문서)은 retrospective — 이를 **prospective**으로 전환하는 방법론 없음
- Hidden assumption의 extraction, evaluation, systematic negation → **미탐색**

### Gap 3: 과학적 세런디피티(Serendipity)의 공학적 구현 부재

- ReMIND (2026): wake/dream/judge/re-wake 프레임워크가 가장 가까움
- SerenQA (2025): 세런디피티 메트릭(relevance × novelty × surprise) 제안
- 하지만 이를 **과학적 발견**에 직접 적용한 시스템은 없음
- "Fertile accident zones"를 knowledge graph에서 찾는 연구 → **전무**

### Gap 4: Transformational Novelty 평가 방법 부재

- 현재 metrics: disruption index (citation-based, retrospective), LiveIdeaBench (divergent thinking), novelty scores (surface-level)
- 없는 것:
  - "이 아이디어가 새로운 개념 공간을 여는가?" 측정
  - Prospective disruption prediction
  - Paradigm-shifting potential 사전 평가

### Gap 5: Anomaly-as-Signal 시스템 부재

- Grokking, double descent, emergent abilities 등 AI의 패러다임 전환은 anomaly에서 시작
- 하지만 anomaly를 **자동으로 감지하고, 그 중요도를 판별하고, 새로운 가설로 전환하는** 시스템 없음

---

## 3. Unbox의 포지셔닝

### 3.1 우리만의 차별점

```
기존 연구:  "기존 개념을 새롭게 조합" (Combinatorial)
           "정의된 공간 내에서 최적 탐색" (Exploratory)

Unbox:     "개념 공간의 규칙 자체를 변형" (Transformational)
           = 숨겨진 가정을 찾아 깨뜨리기
```

### 3.2 New Positioning (2026 Update)

Two new papers sharpen Unbox's positioning significantly:

**Artificial Hivemind (NeurIPS 2025 Best Paper)** demonstrates that 70+ SOTA LLMs converge onto the same outputs for open-ended tasks — not just lexically, but semantically. Temperature/sampling changes are insufficient; the latent space itself has collapsed. Reward models fail on pluralistic data. This is the strongest empirical evidence that current AI ideation is structurally limited, not just prompt-limited.

**Idea-Catalyst (arXiv 2603.12226, UIUC)** shows that metacognition-driven cross-domain inspiration can improve novelty (+21%) and insightfulness (+16%) by abstracting bottlenecks domain-agnostically and borrowing solutions from other fields. But it stays at exploratory/combinatorial creativity — it doesn't challenge or transform the conceptual space itself.

**Unbox's thesis, sharpened**: Artificial Hivemind shows that aligned LLMs don't merely sample similarly — they converge onto the same hidden assumptions and prune valid alternatives during both generation and evaluation. Idea-Catalyst shows that better abstraction and cross-domain search can improve novelty but still recombines within that collapsed space. Unbox is the necessary next step: a framework that explicitly identifies, perturbs, and preserves broken assumptions so research generation can move from combinatorial novelty to transformational creativity.

### 3.3 이론적 기반

| 이론 | 핵심 | Unbox에서의 역할 |
|---|---|---|
| **Boden's Transformational Creativity** | 규칙을 변형하여 새로운 개념 공간 생성 | 핵심 프레임워크 |
| **Kuhn's Revolutionary Science** | 패러다임 전환은 anomaly 축적에서 시작 | Anomaly-driven discovery |
| **TRIZ Contradiction Resolution** | 모순을 해결하면 혁신이 나옴 | Hidden assumption breaking |
| **Koestler's Bisociation** | 관련 없는 두 프레임의 충돌 → 새로운 통찰 | Serendipity engineering |
| **Rothenberg's Janusian Thinking** | 모순을 동시에 참으로 유지 | Constraint relaxation |
| **Peirce's Abduction** | 놀라운 관찰 → 설명 가설 생성 | Anomaly → Hypothesis |

### 3.3 제안 아키텍처 (Conceptual)

```
┌─────────────────────────────────────────────────┐
│                  UNBOX SYSTEM                    │
│                                                  │
│  ┌───────────────┐  ┌───────────────┐            │
│  │ ASSUMPTION    │  │ ANOMALY       │            │
│  │ MINER         │  │ DETECTOR      │            │
│  │               │  │               │            │
│  │ Extract       │  │ Find results  │            │
│  │ implicit      │  │ that don't    │            │
│  │ assumptions   │  │ fit current   │            │
│  │ from AI lit   │  │ theories      │            │
│  └───────┬───────┘  └───────┬───────┘            │
│          │                  │                     │
│          ▼                  ▼                     │
│  ┌─────────────────────────────────┐             │
│  │     CONSTRAINT BREAKER          │             │
│  │                                 │             │
│  │  For each assumption:           │             │
│  │  1. Negate / Relax / Invert     │             │
│  │  2. Check theoretical coherence │             │
│  │  3. Generate new hypothesis     │             │
│  └───────────────┬─────────────────┘             │
│                  │                                │
│                  ▼                                │
│  ┌─────────────────────────────────┐             │
│  │     SERENDIPITY ENGINE          │             │
│  │                                 │             │
│  │  Collide distant concepts:      │             │
│  │  - KG-based distant analogy     │             │
│  │  - ReMIND dream cycle           │             │
│  │  - Bisociation scoring          │             │
│  └───────────────┬─────────────────┘             │
│                  │                                │
│                  ▼                                │
│  ┌─────────────────────────────────┐             │
│  │     MULTI-AGENT EVALUATOR       │             │
│  │                                 │             │
│  │  - Adversarial debate           │             │
│  │  - Transformational novelty     │             │
│  │    assessment                   │             │
│  │  - Feasibility check            │             │
│  │  - POPPER falsification         │             │
│  └─────────────────────────────────┘             │
│                                                   │
└───────────────────────────────────────────────────┘
```

---

## 4. 가장 중요한 참고 논문 Top 15

| # | 논문 | 왜 중요한가 |
|---|---|---|
| 1 | **Si et al. (ICLR 2025)** — Can LLMs Generate Novel Research Ideas? | 문제 정의의 근거 — LLM 아이디어의 novelty vs feasibility gap |
| 2 | **Shahhosseini et al. (2025)** — LLMs for Scientific Idea Generation: Creativity-Centered Survey | 가장 포괄적 분류 체계 — Boden + Rhodes 4Ps |
| 3 | **Schapiro et al. (ICCC 2025)** — Transformational Creativity in Science | Boden + Kuhn 통합 — 우리 이론적 기반 |
| 4 | **AutoTRIZ (2024)** — TRIZ + LLM | 체계적 가정 깨기 방법론의 출발점 |
| 5 | **Magellan (2025)** — Guided MCTS for Novelty | Novelty-optimized search의 best practice |
| 6 | **ReMIND (2026)** — Controllable Serendipity via REM Cycle | 세런디피티 공학의 가장 가까운 시도 |
| 7 | **SerenQA (2025)** — Serendipity in Knowledge Graphs | 세런디피티 메트릭: relevance × novelty × surprise |
| 8 | **CHIMERA (2025)** — Scientific Recombination KB | 과학적 재조합 패턴 데이터 |
| 9 | **Cognitive Traversal Distance (2026)** | 개념적 거리 측정 — novelty metric |
| 10 | **FIRE-Bench (2025)** | AI 발견 능력 벤치마크 |
| 11 | **FunSearch (2023)** | LLM + evolution = genuine novelty 달성 사례 |
| 12 | **Beel et al. (2025)** — Evaluating AI Scientist | AI Scientist 시스템의 현실적 한계 |
| 13 | **Lottery Ticket (2018)** | 가정 깨기의 AI 역사적 사례 |
| 14 | **AutoDiscovery (NeurIPS 2025)** — Bayesian Surprise for Discovery | Surprise-driven open-ended discovery |
| 15 | **POPPER (ICML 2025)** — Automated Falsification | 가설 검증 프레임워크 |
| 16 | **Artificial Hivemind (NeurIPS 2025 Best Paper)** — LLM Output Homogeneity | Unbox thesis의 핵심 증거: 모든 LLM이 같은 가정에 수렴 |
| 17 | **Idea-Catalyst (arXiv 2603.12226)** — Interdisciplinary Inspiration | Cross-domain abstraction 기법 참고 + Unbox 차별화 대상 |

---

## 5. 제안 연구 로드맵

### Phase 1: Foundation (1-2 months)
- AI 논문에서 implicit assumption을 추출하는 파이프라인 구축 (LLM + KG)
- Historical paradigm shifts 분석: 각 shift에서 어떤 가정이 깨졌는지 systematic mapping
- Evaluation framework 설계: transformational novelty 측정 방법

### Phase 2: Constraint Breaker (2-3 months)
- Assumption negation/relaxation 시스템 구현
- Multi-agent debate로 유망한 constraint-break 평가
- Known paradigm shifts에 대한 retrospective validation (이미 일어난 paradigm shift를 시스템이 예측할 수 있는가?)

### Phase 3: Serendipity Engine (2-3 months)
- AI research knowledge graph 구축 (OpenAlex/Semantic Scholar 기반)
- Cognitive Traversal Distance 기반 "fertile accident zone" 탐색
- ReMIND dream cycle + bisociation scoring

### Phase 4: Integration & Evaluation (1-2 months)
- 전체 시스템 통합
- FIRE-Bench + LiveIdeaBench + custom transformational novelty benchmark로 평가
- Human expert evaluation (Si et al. 방법론 참조)

---

## 6. 논문 타이틀 후보

1. **"Unbox: Breaking Hidden Assumptions in AI Research via Constraint-Aware Discovery Agents"**
2. **"Beyond Recombination: Transformational Creativity in AI-Driven Scientific Discovery"**
3. **"From Normal Science to Revolutionary Science: AI Agents for Paradigm Shift Discovery"**
4. **"The Assumption Breaker: Systematic Discovery of Hidden Constraints in Machine Learning"**
5. **"Engineering Serendipity for Science: Intentional Discovery of Paradigm-Shifting AI Ideas"**
