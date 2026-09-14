# 04. Knowledge Graphs, Cross-Domain Analogy & Serendipity Engineering

> Papers on knowledge graph-based scientific discovery, literature-based discovery, cross-domain transfer, computational serendipity, and science of science.

---

## A. Knowledge Graph Approaches for Scientific Discovery

### 1. Swanson's Original Literature-Based Discovery
- **Authors**: Don Swanson
- **Year**: 1987 | **Venue**: JASIS 38(4):228-233
- **Key Contribution**: Seminal work — discovered that fish oil treats Raynaud's phenomenon and magnesium treats migraines by connecting disconnected literatures. Founded the field of Literature-Based Discovery (LBD).
- **Relevance to Unbox**: The original "serendipity machine" — finding connections humans missed because they read different literatures.

### 2. Literature-Based Discovery: State of the Art
- **Authors**: Liu & Fu
- **Year**: 2012 | **arXiv**: [1203.3611](https://arxiv.org/abs/1203.3611)
- **Key Contribution**: Comprehensive survey of LBD from 2000-2012.

### 3. Rediscovering Don Swanson: Past, Present and Future of LBD
- **Authors**: Smalheiser & Hristovski
- **Year**: 2017 | **Venue**: Journal of Data and Information Science 2(4):43-64
- **Key Contribution**: Updated review with modern ML and data sources.

### 4. CHIMERA: A Knowledge Base of Scientific Idea Recombinations
- **Authors**: Noy Sternlicht, Tom Hope
- **Year**: 2025 | **arXiv**: [2505.20779](https://arxiv.org/abs/2505.20779)
- **GitHub**: [noy-sternlicht/CHIMERA-KB](https://github.com/noy-sternlicht/CHIMERA-KB)
- **Key Contribution**: 28K+ recombination examples from scientific literature. Two types: **blends** (concept fusion, e.g., Neural Networks + Quantum Physics) and **inspirations** (cross-domain transfer, e.g., dragonfly wings → drone design). 24.8% of recombinations are interdisciplinary.
- **Relevance to Unbox**: Structured data on how scientists actually recombine ideas — could train models to predict novel recombinations.

### 5. SKiM-GPT: Combining Biomedical LBD with LLM Hypothesis Evaluation
- **Authors**: Freeman et al.
- **Year**: 2025 | **Venue**: BMC Bioinformatics 27(1):16
- **GitHub**: [stewart-lab/skimgpt](https://github.com/stewart-lab/skimgpt)
- **Key Contribution**: Combines co-occurrence search (SKiM) with frontier LLMs for transparent, human-verifiable hypothesis scoring. Cohen's kappa = 0.84 with expert biologists.

### 6. OpenScholar: Synthesizing Scientific Literature with Retrieval-Augmented LMs
- **Authors**: Asai et al.
- **Year**: 2026 | **Venue**: Nature 650
- **Key Contribution**: 45M open-access papers indexed. Citation-backed literature synthesis.

### 7. Graph Neural Network-Based Entity Extraction and Relationship Reasoning in Complex KGs
- **Authors**: Shi et al.
- **Year**: 2024 | **arXiv**: [2411.15195](https://arxiv.org/abs/2411.15195)
- **Key Contribution**: End-to-end GCN + GAT model for entity extraction and relationship reasoning.

### 8. Relphormer: Relational Graph Transformer for Knowledge Graph Representations
- **Authors**: Zhen Bi, Siyuan Cheng et al. (Zhejiang University + Alibaba)
- **Year**: 2022 | **arXiv**: [2205.10852](https://arxiv.org/abs/2205.10852)
- **Key Contribution**: Relational Graph Transformer addressing vanilla Transformer limitations in KG.

### 9. A Novel Model for Relation Prediction in KGs via Semantic and Structural Feature Integration
- **Authors**: Yang et al.
- **Year**: 2024 | **Venue**: Scientific Reports 4(15):12962-79
- **Key Contribution**: Integrates semantic and structural features for relation prediction.

---

## B. Cross-Domain Analogy & Knowledge Transfer

### 10. Transfer Learning by Structural Analogy
- **Authors**: Wang & Yang
- **Year**: 2011 | **Venue**: AAAI 2011
- **Key Contribution**: Finds structural similarity between completely different domains at the knowledge level. Non-trivial mapping when representations differ.

### 11. MAGIK: Mapping to Analogous Goals via Imagination-enabled Knowledge Transfer
- **Authors**: Palattuparambil et al.
- **Year**: 2025 | **arXiv**: [2506.01623](https://arxiv.org/abs/2506.01623)
- **Key Contribution**: "Imagination-enabled knowledge transfer" to map cross-domain analogical connections. Could train models to predict creative cross-domain research directions.

### 12. Importance Inversion Transfer: Shared Principles for Cross-Domain Learning
- **Authors**: Caligiore et al.
- **Year**: 2026 | **arXiv**: [2602.09116](https://arxiv.org/abs/2602.09116)
- **Key Contribution**: Identifies structural invariants that generalize across heterogeneous systems (biological, linguistic, molecular, social).

### 13. Beyond Pairwise Distance: Cognitive Traversal Distance as a Holistic Measure of Scientific Novelty
- **Authors**: Xiang et al.
- **Year**: 2026 | **arXiv**: [2602.06607](https://arxiv.org/abs/2602.06607)
- **Key Contribution**: Novel measure of conceptual distance. Decomposes knowledge into: research problem, methodology, and core findings. Network-based indicator overcomes aggregation limitations of pairwise metrics.
- **Relevance to Unbox**: Could use this metric to measure how "far" generated ideas are from existing work.

### 14. Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas
- **Authors**: Shen et al.
- **Year**: 2026 | **arXiv**: [2601.08901](https://arxiv.org/abs/2601.08901)
- **Key Contribution**: Three-dimensional decomposition of ideation space: research problem, methodology, core findings.

### 15. Structure Transfer: Inference-Based Calculus for Representation Transformation
- **Authors**: Raggi et al.
- **Year**: 2025 | **arXiv**: [2509.03249](https://arxiv.org/abs/2509.03249)
- **Key Contribution**: Formal calculus for transforming representations between domains. Enables functional separation between exploration and consolidation.

### 16. Transfer Learning through Analogy in Games
- **Authors**: Forbus et al.
- **Year**: AI Magazine
- **Key Contribution**: Applies analogical reasoning to near and far transfer across physics and strategy games.

---

## C. Serendipity in Computational Systems

### 17. ReMIND: Orchestrating Modular LLMs for Controllable Serendipity
- **Authors**: Sato
- **Year**: 2026 | **arXiv**: [2601.07121](https://arxiv.org/abs/2601.07121)
- **Key Contribution**: Four-stage framework inspired by REM sleep cycle: **wake** (stable baseline) → **dream** (exploratory generation) → **judge** (filter) → **re-wake** (re-articulation). Key insight: high-quality ideas emerge sporadically, not as extrema along any single metric.
- **Relevance to Unbox**: Most directly relevant serendipity framework found. The wake/dream cycle maps well to constraint-breaking (dream phase = assumption relaxation).

### 18. Assessing LLMs for Serendipity Discovery in Knowledge Graphs
- **Authors**: Wang et al.
- **Year**: 2025 | **arXiv**: [2511.12472](https://arxiv.org/abs/2511.12472)
- **Key Contribution**: SerenQA framework with serendipity metric based on **relevance × novelty × surprise**. Benchmark with expert-annotated examples. Key finding: LLMs perform well on retrieval but struggle with genuinely surprising discoveries.
- **Relevance to Unbox**: Provides formal serendipity metric and demonstrates the gap we're trying to close.

### 19. Deep Learning Models for Serendipity Recommendations: A Survey and New Perspectives
- **Authors**: Fu et al.
- **Year**: 2023 | **Venue**: ACM Computing Surveys
- **Key Contribution**: Comprehensive review of serendipitous recommendation approaches — techniques, evaluation metrics, user studies.

### 20. Design of a Serendipity-Incorporated Recommender System
- **Authors**: Kim et al.
- **Year**: 2025 | **Venue**: Electronics 14(4):821
- **Key Contribution**: Framework incorporating serendipity into user control for recommendations.

### 21. Engineering Serendipity: Reclaiming Joyful Discovery in Hyper-Personalized AI Systems
- **Authors**: Weindava
- **Year**: 2025 | **Venue**: IJRISSS 9(10):2414-2426
- **Key Contribution**: Addresses how hyper-personalization suppresses discovery. Proposes framework for engineering serendipity.

---

## D. Science of Science / Meta-Science Applied to AI

### 22. CrossDI Dataset: Comprehensive Dataset for Calculating Disruption Indexes
- **Authors**: Xu et al.
- **Year**: 2025 | **Venue**: Scientific Data | **DOI**: [10.1038/s41597-025-06232-w](https://www.nature.com/articles/s41597-025-06232-w)
- **Key Contribution**: Crosses Web of Science, Scopus, PubMed for disruption metric calculation.

### 23. A Multi-Dimensional Coupling Model for Detecting Science-Technology Interactions within AI
- **Authors**: Zhuo et al.
- **Year**: 2025 | **Venue**: Scientometrics (Feb 2026)
- **Key Contribution**: Network structure coupling (NSC) + time series coupling (TSC) for modeling dynamic AI S&T interactions.

### 24. The Paradigm Shifts in Artificial Intelligence
- **Authors**: (multiple)
- **Year**: 2023 | **Venue**: Communications of the ACM
- **Key Contribution**: Applies Kuhn's framework to AI history. Identifies specific paradigm shifts and what triggered them.
- **Relevance to Unbox**: Historical analysis of exactly the kind of shifts we want to enable.

### 25. AI-Powered Research Idea Generation: Comparing Retrieval, Graph Discovery, and Claim-Level Reasoning Tools
- **Authors**: Hernan M
- **Year**: 2026 | **Venue**: Towards AI
- **Key Contribution**: Compares multiple paradigms for AI-assisted scientific idea generation.

---

## Key Insight: The Serendipity Gap

Despite interest in serendipity across recommendation systems and information retrieval, **there is almost no work on engineering serendipity for scientific idea generation specifically**. This is a clear research opportunity for Unbox.

The ReMIND framework (wake/dream/judge/re-wake cycle) and SerenQA's formal metric (relevance × novelty × surprise) provide starting points, but neither has been applied to the problem of generating genuinely novel AI research ideas.

### Potential Unbox Contribution
Combine:
- **CHIMERA's recombination patterns** (what recombinations have historically worked)
- **Cognitive Traversal Distance** (how to measure conceptual distance)
- **ReMIND's dream cycle** (structured exploration phase)
- **SerenQA's serendipity metric** (relevance × novelty × surprise)

...to build a system that intentionally navigates to "fertile accident zones" in the AI research knowledge graph.
