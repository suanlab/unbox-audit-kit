# 05. Evaluation Frameworks & Benchmarks for Scientific Novelty

> Papers on how to measure novelty, benchmarks for AI idea generation, evaluation of AI research agents, computational creativity assessment, and philosophy of novelty.

---

## A. Benchmarks for AI Idea Generation

### 1. FIRE-Bench: Full-cycle Insight Rediscovery Evaluation
- **Year**: 2025 | **arXiv**: [2602.02905](https://arxiv.org/abs/2602.02905)
- **Key Contribution**: Evaluates AI agents through verifiable discovery tasks from recent ML conferences. Converts papers into tree-structured executable tasks. Tests whether agents can *rediscover* known scientific insights.
- **Relevance to Unbox**: Baseline benchmark — can our system rediscover known paradigm shifts?

### 2. LiveIdeaBench: Evaluating LLMs' Divergent Thinking
- **Year**: 2026 | **Venue**: Nature Communications | **DOI**: [10.1038/s41467-026-70245-1](https://www.nature.com/articles/s41467-026-70245-1)
- **Key Contribution**: Tests divergent thinking with minimal context (single keywords). 5 dimensions from Guilford's theory: originality, feasibility, fluency, flexibility, clarity. 40+ models, 1180 keywords, 22 scientific domains.

### 3. AI Idea Bench 2025
- **Year**: 2025 | **arXiv**: [2504.14191](https://arxiv.org/abs/2504.14191)
- **Key Contribution**: Benchmark specifically for AI research idea generation.

### 4. AIRS-Bench: Benchmark for Frontier AI Research Science Agents
- **Year**: 2026 | **arXiv**: [2602.06855](https://arxiv.org/abs/2602.06855)
- **Key Contribution**: Comprehensive benchmark suite for frontier AI research agents.

### 5. ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition
- **Year**: 2025 | **arXiv**: [2503.21248](https://arxiv.org/abs/2503.21248)
- **Key Contribution**: Tests LLMs on inspiration-based scientific discovery tasks.

### 6. ScienceBench: Expert-Level Academic Questions
- **Year**: 2025 | **Venue**: Nature | **DOI**: [10.1038/s41586-025-09962-4](https://www.nature.com/articles/s41586-025-09962-4)
- **Key Contribution**: PhD-level scientific questions to assess AI research capabilities.

### 7. LLM-SRBench: Benchmark for Scientific Equation Discovery
- **Year**: 2025 | **Venue**: ICML 2025
- **Key Contribution**: Tests LLMs on scientific equation discovery.

---

## B. Metrics for Novelty Evaluation

### 8. A Review on the Novelty Measurements of Academic Papers
- **Year**: 2025 | **Venue**: Scientometrics 130(2)
- **Key Contribution**: Comprehensive review of ALL novelty measurement methods in academic literature.

### 9. The Disruption Index Suffers from Citation Inflation
- **Year**: 2025 | **Venue**: Journal of Informetrics 19(1)
- **Key Contribution**: Critical analysis of CD index limitations — citation inflation biases, needs minimum 3-5 year windows.

### 10. Dynamic Disruption Index: Recommendations for Thresholds
- **Year**: 2025 | **arXiv**: [2504.07828](https://arxiv.org/abs/2504.07828)
- **Key Contribution**: Improved thresholds for disruption index in research evaluation.

### 11. CrossDI Dataset: Calculating Disruption Indexes Across Databases
- **Year**: 2025 | **Venue**: Scientific Data | **DOI**: [10.1038/s41597-025-06232-w](https://www.nature.com/articles/s41597-025-06232-w)
- **Key Contribution**: Cross-database disruption index calculation (WoS, Scopus, PubMed).

### 12. Literature-Grounded Novelty Assessment of Scientific Ideas (Idea Novelty Checker)
- **Year**: 2025 | **arXiv**: [2506.22026](https://arxiv.org/abs/2506.22026)
- **Key Contribution**: RAG framework for automated novelty assessment grounded in literature.

### 13. A Content-Based Novelty Measure for Scholarly Publications
- **Year**: 2024 | **arXiv**: [2401.03642](https://arxiv.org/abs/2401.03642)
- **Key Contribution**: Content-based (not citation-based) novelty measurement.

### 14. Predictive Effects of Novelty Measured by Temporal Embeddings
- **Year**: 2018 | **Venue**: Frontiers in Research Metrics and Analytics
- **Key Contribution**: Uses temporal embeddings to measure novelty's effect on literature growth.

### 15. An Effective Framework for Measuring Novelty through Topic Modeling and Cloud Model
- **Year**: 2024 | **Venue**: Journal of Informetrics 18(4)
- **Key Contribution**: Integrated topic modeling approach for novelty measurement.

### 16. Evaluating and Enhancing LLMs for Novelty Assessment in Scholarly Publications
- **Year**: 2024 | **arXiv**: [2409.16605](https://arxiv.org/abs/2409.16605)
- **Key Contribution**: Evaluates LLMs as novelty assessors.

### 17. Evaluating Novelty in AI-Generated Research Plans
- **Year**: 2026 | **arXiv**: [2601.09714](https://arxiv.org/abs/2601.09714)
- **Key Contribution**: Multi-workflow LLM pipelines for novelty evaluation.

### 18. Assessing Novelty, Feasibility and Value of Creative Ideas
- **Year**: 2024 | **PubMed**: [39037067](https://pubmed.ncbi.nlm.nih.gov/39037067/)
- **Key Contribution**: Unsupervised approach using GPT-4 for N-F-V assessment.

### 19. Beyond Pairwise Distance: Cognitive Traversal Distance
- **Year**: 2026 | **arXiv**: [2602.06607](https://arxiv.org/abs/2602.06607)
- **Key Contribution**: Network-based novelty metric decomposing ideas into problem + method + findings.

---

## C. Evaluation of AI Research Agents

### 20. Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results
- **Authors**: Beel, Kan, Baumgart
- **Year**: 2025 | **arXiv**: [2502.14297](https://arxiv.org/abs/2502.14297)
- **Key Findings**:
  - Classified well-known techniques (mini-batching SGD) as "novel"
  - 42% experiment failure rate (5/12 failed from coding errors)
  - 57% of manuscripts contained fabricated numerical results
  - Each iteration added only ~8% more characters — minimal adaptation
  - Quality: "unmotivated undergraduate rushing a deadline"

### 21. The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems
- **Authors**: CMU researchers
- **Year**: 2025 | **arXiv**: [2509.08713](https://arxiv.org/abs/2509.08713)
- **Key Findings**: Four failure modes:
  1. Inappropriate benchmark selection
  2. Data leakage
  3. Metric misuse
  4. Post-hoc selection bias

### 22. Can LLMs Generate Novel Research Ideas? (Si et al., Stanford)
- **Year**: 2024 | **Venue**: ICLR 2025 | **arXiv**: [2409.04109](https://arxiv.org/abs/2409.04109)
- **Key Finding**: LLM-generated ideas rated MORE novel but LESS feasible than expert ideas. LLM self-evaluation unreliable. Lack of diversity in generation.

---

## D. Creativity Evaluation in AI

### 23. What Shapes a Creative Machine Mind? Benchmarking Creativity in Foundation Models
- **Year**: 2025 | **arXiv**: [2510.04009](https://arxiv.org/abs/2510.04009)
- **Key Contribution**: Comprehensive benchmark for creativity in LLMs.

### 24. S-DAT: Multilingual GenAI-Driven Divergent Thinking Assessment
- **Year**: 2025 | **arXiv**: [2505.09068](https://arxiv.org/html/2505.09068v1)
- **Key Contribution**: Automated assessment based on Guilford's Alternative Uses Test.

### 25. Automated Creativity Evaluation for LLMs: Reference-Based Approach
- **Year**: 2025 | **Venue**: ACL Findings EMNLP 2025
- **Key Contribution**: Uses Torrance Test of Creative Writing for LLM evaluation.

### 26. Divergent Creativity in Humans and LLMs
- **Year**: 2025 | **Venue**: Scientific Reports | **DOI**: [10.1038/s41598-025-25157-3](https://www.nature.com/articles/s41598-025-25157-3)
- **Key Contribution**: Compares semantic diversity between human and LLM creative outputs.

### 27. A Standardised Procedure for Evaluating Creative Systems
- **Authors**: Anna Jordanous
- **Year**: 2012 | **Venue**: Cognitive Computation 4(3)
- **Key Contribution**: Foundational computational creativity evaluation framework from ICCC community.

### 28. On the Creativity of Large Language Models
- **Year**: 2024 | **Venue**: AI & Society | **Springer**: [10.1007/s00146-024-02127-3](https://link.springer.com/article/10.1007/s00146-024-02127-3)
- **Key Contribution**: Philosophical examination of LLM creativity.

---

## E. Philosophy of Novelty in Science

### 29. Strong Novelty Regained: High-Impact Outcomes of ML for Science
- **Year**: 2025 | **Venue**: Synthese | **Springer**: [10.1007/s11229-025-05228-8](https://link.springer.com/article/10.1007/s11229-025-05228-8)
- **Key Contribution**: Challenges presupposition arguments about ML's inherent limits for scientific novelty.

### 30. POPPER: Automated Hypothesis Validation with Agentic Sequential Falsifications
- **Year**: 2025 | **Venue**: ICML 2025 | **Link**: [proceedings.mlr.press/v267/huang25n](https://proceedings.mlr.press/v267/huang25n.html)
- **Key Contribution**: Stanford framework implementing Popper's falsifiability for automated hypothesis testing.
- **Relevance to Unbox**: Could validate constraint-breaking hypotheses through sequential falsification.

### 31. The Epistemic Revolution of AI
- **Year**: 2025 | **Venue**: AI & Society | **Springer**: [10.1007/s00146-025-02658-3](https://link.springer.com/article/10.1007/s00146-025-02658-3)
- **Key Contribution**: Examines AI's impact on epistemological paradigms.

### 32. Scientific Hypothesis Generation and Validation: Methods and Future Directions
- **Year**: 2026 | **arXiv**: [2505.04651](https://arxiv.org/html/2505.04651v1)
- **Key Contribution**: Comprehensive survey from Virginia Tech on hypothesis generation methods.

### 33. The Paradigm Shifts in Artificial Intelligence
- **Year**: 2023 | **Venue**: Communications of the ACM
- **Key Contribution**: Applies Kuhn's framework to AI history — identifies specific paradigm shifts.

### 34. Ideometrics: A Scientific Approach to Generating, Evaluating, and Prioritising Ideas
- **Year**: 2025 | **Venue**: Journal of Global Health
- **Key Contribution**: Framework for systematic idea evaluation and prioritization.

---

## Summary: The Evaluation Gap

| What We CAN Measure | What We CAN'T Measure Well |
|---|---|
| Novelty perception (human ratings) | Novelty substance (actual impact) |
| Feasibility | Paradigm-shifting potential |
| Literature overlap/distance | Whether an idea opens a new conceptual space |
| Citation-based disruption (retrospective) | Prospective disruption prediction |
| Combinatorial novelty (new keyword combos) | Transformational novelty (new frameworks) |

**The core evaluation challenge for Unbox**: We need metrics that distinguish "cleverly recombined existing ideas" from "genuinely new conceptual frameworks." Current metrics (disruption index, novelty scores, LiveIdeaBench) primarily measure the former. Measuring the latter may require:
1. Expert panel evaluation with specific transformational creativity criteria
2. Counterfactual analysis: "Would this idea have eventually emerged from normal science?"
3. Conceptual space analysis: Does this idea expand the space or just explore it?
