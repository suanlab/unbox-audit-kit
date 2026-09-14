# 03. AI Creativity, Divergent Thinking & Novelty-Seeking

> Papers on LLM-based ideation, multi-agent brainstorming, curiosity-driven exploration, computational creativity theory, and cognitive science of discovery.

---

## A. LLM-Based Idea Generation & Evaluation

### 1. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers
- **Authors**: Chenglei Si, Diyi Yang, Tatsunori Hashimoto (Stanford)
- **Year**: 2024 | **Venue**: ICLR 2025 | **arXiv**: [2409.04109](https://arxiv.org/abs/2409.04109)
- **GitHub**: [NoviScl/AI-Researcher](https://github.com/NoviScl/AI-Researcher)
- **Key Finding**: LLM ideas rated MORE novel but LESS feasible than human expert ideas. Critical caveat: novelty *perception* differs from novelty *substance*. LLM self-evaluation is unreliable; tendency to converge on similar ideas.
- **Relevance to Unbox**: Establishes the core problem — LLMs produce ideas that *sound* novel by remixing terminology but lack deep structural innovation.

### 2. LiveIdeaBench: Evaluating LLMs' Divergent Thinking for Scientific Idea Generation
- **Authors**: Kai Ruan, Xuan Wang et al.
- **Year**: 2026 | **Venue**: Nature Communications | **DOI**: [10.1038/s41467-026-70245-1](https://doi.org/10.1038/s41467-026-70245-1)
- **Key Finding**: Tests divergent thinking with minimal context (single-keyword prompts). 5 dimensions: originality, feasibility, fluency, flexibility, clarity. Draws from Guilford's creativity theory. LLMs perform poorly with minimal context — generate convergent rather than divergent ideas.

### 3. SciMON: Scientific Inspiration Machines Optimized for Novelty
- **Authors**: Qingyun Wang, Doug Downey, Heng Ji, Tom Hope
- **Year**: 2024 | **Venue**: ACL 2024 | **arXiv**: [2305.14259](https://arxiv.org/abs/2305.14259)
- **Key Contribution**: Explicitly optimizes for novelty rather than just link prediction. Generates novel scientific directions grounded in literature.

### 4. Many Heads Are Better Than One: Improved Scientific Idea Generation by LLM-Based Multi-Agent System
- **Authors**: Haoyang Su, Renqi Chen et al.
- **Year**: 2025 | **Venue**: ACL 2025 | **arXiv**: [2410.09403](https://arxiv.org/abs/2410.09403)
- **Key Contribution**: Multi-agent collaboration significantly improves idea generation quality through diverse perspectives and iterative refinement.

### 5. IRIS: Interactive Research Ideation System
- **Authors**: Aniketh Garikaparthi et al.
- **Year**: 2025 | **arXiv**: [2504.16728](https://arxiv.org/abs/2504.16728)
- **Key Contribution**: Open-source, human-in-the-loop research ideation platform emphasizing transparency and steerability.

### 6. Spark: A System for Scientifically Creative Idea Generation
- **Authors**: Aishik Sanyal, Samuel Schapiro et al.
- **Year**: 2025 | **arXiv**: [2504.20090](https://arxiv.org/abs/2504.20090)
- **Key Contribution**: Applies Boden's computational creativity principles to LLM-based scientific idea generation.

### 7. Deep Ideation: Designing LLM Agents on Scientific Concept Networks
- **Authors**: Keyu Zhao, Weiquan Lin et al.
- **Year**: 2025 | **Venue**: OpenReview (ICLR 2026, withdrawn)
- **Key Contribution**: Uses scientific concept networks to guide LLM agents toward novel directions.

### 8. Cooking Up Creativity: Enhancing LLM Creativity through Structured Recombination
- **Authors**: Moran Mizrahi, Chen Shani, Gabriel Stanovsky, Dan Jurafsky, Dafna Shahaf
- **Year**: 2025 | **arXiv**: [2504.20643](https://arxiv.org/abs/2504.20643)
- **Key Contribution**: Structured representations and recombination for creative idea generation.

### 9. Illusions of Reflection: LLM Self-Evaluation Failures
- **Authors**: Sion Weatherhead, Flora Salim, Aaron Belbasis
- **Year**: 2025 | **arXiv**: [2510.18254](https://arxiv.org/abs/2510.18254)
- **Key Contribution**: LLMs produce superficially reflective text but lack genuine self-evaluation — critical failure for research quality assessment.

---

## B. Multi-Agent Systems for Ideation

### 10. Encouraging Divergent Thinking in LLMs through Multi-Agent Debate
- **Authors**: Tian Liang, Zhiwei He et al.
- **Year**: 2024 | **Venue**: ICLR 2024 Workshop | **arXiv**: [2305.19118](https://arxiv.org/abs/2305.19118)
- **Key Contribution**: Multi-agent debate increases divergent thinking by exposing multiple perspectives.

### 11. Improving Factuality and Reasoning through Multiagent Debate
- **Authors**: Yilun Du, Shuang Li, Antonio Torralba, Joshua Tenenbaum, Igor Mordatch
- **Year**: 2023 | **arXiv**: [2305.14325](https://arxiv.org/abs/2305.14325)
- **Key Contribution**: Multiple LLM instances propose and debate responses to improve reasoning quality.

### 12. Thinking with Many Minds: Multi-Perspective Problem-Solving
- **Authors**: Sanghyun Park, Boris Maciejovsky, Puranish Puranam (INSEAD)
- **Year**: 2025 | **SSRN**: [5085859](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5085859)
- **Key Contribution**: Applies "Society of Minds" concept to LLM-based multi-perspective problem-solving.

### 13. Mindstorms in Natural Language-Based Societies of Mind
- **Authors**: Mingchen Zhuge, Haozhe Liu, Francesco Faccio et al.
- **Year**: 2025
- **Key Contribution**: Revisits Minsky's Society of Mind for modern multi-agent LLM systems.

### 14. Perspectra: Choosing Your Experts Enhances Critical Thinking in Multi-Agent Ideation
- **Authors**: Yiren Liu, Viraj Shah et al.
- **Year**: 2025 | **arXiv**: [2509.20553](https://arxiv.org/abs/2509.20553)
- **Key Contribution**: Expert selection strategies matter for multi-agent ideation quality.

### 15. Beyond Brainstorming: What Drives High-Quality Scientific Ideas?
- **Authors**: Nuo Chen, Yicheng Tong et al.
- **Year**: 2025 | **arXiv**: [2508.04575](https://arxiv.org/abs/2508.04575)
- **Key Contribution**: Analyzes what makes multi-agent collaboration effective for scientific idea generation.

### 16. Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)
- **Authors**: Liwei Jiang, Yuanjun Chai, Margaret Li, Mickel Liu, Raymond Fok, Nouha Dziri, Yulia Tsvetkov, Maarten Sap, Alon Albalak, Yejin Choi (UW Allen School)
- **Year**: 2025 | **Venue**: NeurIPS 2025 D&B Track — **Best Paper Award (Oral)** | **arXiv**: [2510.22954](https://arxiv.org/abs/2510.22954)
- **GitHub**: [liweijiang/artificial-hivemind](https://github.com/liweijiang/artificial-hivemind)
- **Dataset**: Infinity-Chat — 26K real-world open-ended queries, 31,250 human annotations (25 per example)
- **Key Findings**:
  1. **Intra-model mode collapse**: Single model generates near-identical outputs across 50 samplings even at temperature 1.0 (pairwise similarity > 0.8)
  2. **Inter-model homogeneity**: 70+ different models (GPT-4o, Claude 3.5, Llama 3.1, Qwen 2.5, DeepSeek) produce strikingly similar outputs — sometimes verbatim identical strings
  3. **Sampling doesn't fix it**: Temperature, top-p, Min-P all insufficient — the latent space itself has converged
  4. **Reward Model failure on pluralism**: SOTA RMs and LLM judges degrade sharply on data where human annotators legitimately disagree (high Shannon entropy)
  5. **RLHF as diversity killer**: Current alignment methods over-fit to single consensus, actively pruning valid-but-idiosyncratic responses
- **Taxonomy**: 6 top-level categories, 17 subcategories for open-ended queries (Brainstorm & Ideation, Speculative & Hypothetical Scenarios, Skill Development, etc.)
- **Methodology**: Pairwise cosine similarity (text-embedding-3-small), Shannon entropy for human disagreement, 70+ models × 50 samples/query
- **Limitations**: English only; embedding-based diversity proxy; causal mechanism (shared pretraining data vs RLHF convergence) undetermined; diagnosis only, no solution proposed
- **Relevance to Unbox**:
  - **Strongest evidence for Unbox's thesis**: If all LLMs converge to the same outputs, then using LLMs for idea generation without structural diversity mechanisms (like assumption-breaking) is fundamentally limited
  - **Validates multi-agent evaluator design**: Single reward models fail on pluralistic data → Unbox's Advocate/Critic/Judge debate is a structural mitigation
  - **"Changing the model doesn't change the bias"**: This directly supports Unbox's claim that transformational creativity requires a qualitatively different generation mechanism, not just different models or temperatures
  - **Diversity metric**: Pairwise cosine similarity methodology can be adopted to measure whether Unbox-generated hypotheses are actually diverse (vs baseline approaches)

---

## C. Curiosity-Driven & Novelty-Seeking AI

### 16. Curiosity-Driven Exploration by Self-Supervised Prediction
- **Authors**: Deepak Pathak, Pulkit Agrawal, Alexei Efros, Trevor Darrell
- **Year**: 2017 | **Venue**: ICML 2017 | **arXiv**: [1705.05363](https://arxiv.org/abs/1705.05363)
- **Key Contribution**: Foundational paper introducing curiosity as intrinsic reward via prediction error in learned feature space.

### 17. CURIOUS: Intrinsically Motivated Modular Multi-Goal RL
- **Authors**: Cedric Colas, Pierre Fournier, Olivier Sigaud et al.
- **Year**: 2019 | **Venue**: ICLR 2019
- **Key Contribution**: Agents autonomously set goals and build curricula through intrinsic motivation.

### 18. Understanding Exploration in Humans and Machines
- **Authors**: Rachit Dubey, Thomas Griffiths (Princeton)
- **Year**: 2020 | **Link**: [cocosci.princeton.edu](https://cocosci.princeton.edu/papers/dubeyunderstanding.pdf)
- **Key Contribution**: Connects psychological theories of curiosity with computational formulations.

### 19. Illuminating Search Spaces with MAP-Elites
- **Authors**: Jean-Baptiste Mouret, Jeff Clune
- **Year**: 2015 | **arXiv**: [1504.04909](https://arxiv.org/abs/1504.04909)
- **Key Contribution**: Foundation of Quality-Diversity (QD) optimization — finding diverse high-performing solutions rather than single optimum. Directly relevant to generating diverse research ideas.

### 20. Differentiable Quality Diversity
- **Authors**: Matthew Fontaine, Stefanos Nikolaidis
- **Year**: 2021 | **Venue**: NeurIPS 2021
- **Key Contribution**: Enables gradient-based QD optimization.

### 21. Open-Endedness is Essential for Artificial Superhuman Intelligence
- **Authors**: Edward Hughes, Michael Dennis, Jack Parker-Holder et al.
- **Year**: 2024 | **Venue**: ICML 2024 | **arXiv**: [2406.04268](https://arxiv.org/abs/2406.04268)
- **Key Contribution**: Position paper arguing open-endedness is essential for AGI.

### 22. AutoDiscovery: Open-Ended Scientific Discovery via Bayesian Surprise
- **Authors**: Dhruv Agarwal, Bodhisattwa Prasad Majumder et al.
- **Year**: 2025 | **Venue**: NeurIPS 2025
- **Key Contribution**: Uses Bayesian surprise to drive open-ended scientific hypothesis generation.
- **Relevance to Unbox**: Directly relevant — surprise-driven discovery mechanism.

### 23. Explore and Control with Adversarial Surprise
- **Authors**: Arnaud Fickinger, Natasha Jaques et al.
- **Year**: 2021 | **arXiv**: [2107.07394](https://arxiv.org/abs/2107.07394)
- **Key Contribution**: Adversarial surprise maximizes exploration through adversarial scenarios.

### 24. Magellan: Guided MCTS for Latent Space Exploration
- **Authors**: (2025) | **arXiv**: [2510.21341](https://arxiv.org/html/2510.21341v1)
- **GitHub**: [moyiliyi/Magellan-Novelty-Generation](https://github.com/moyiliyi/Magellan-Novelty-Generation)
- **Key Contribution**: Monte Carlo Tree Search with orthogonal projection-based guidance for novelty-optimized exploration. Steers away from training data "gravity wells." 92% win rate over baselines.
- **Relevance to Unbox**: Most directly relevant approach — explicit novelty optimization in search.

---

## D. Computational Creativity Theory

### 25. Computer Models of Creativity
- **Authors**: Margaret A. Boden
- **Year**: 2009 | **Venue**: AI Magazine
- **Key Contribution**: Foundational taxonomy: combinatorial, exploratory, transformational creativity.

### 26. Creativity and Artificial Intelligence
- **Authors**: Margaret A. Boden
- **Year**: 2014 | **Venue**: The Philosophy of Creativity (Oxford)
- **Key Contribution**: Philosophical analysis of AI creativity within Boden's framework.

### 27. Transformational Creativity in Science: A Graphical Theory
- **Authors**: Samuel Schapiro, Jonah Black, Lav Varshney
- **Year**: 2025 | **Venue**: ICCC 2025
- **Key Contribution**: Synthesizes Boden's transformational creativity with Kuhn's scientific revolutions. **Directly relevant to Unbox thesis.**

### 28. LLMs Can Realize Combinatorial Creativity
- **Authors**: Tianyang Gu, Jingjin Wang et al.
- **Year**: 2024 | **arXiv**: [2412.14141](https://arxiv.org/abs/2412.14141)
- **Key Contribution**: Empirically tests LLMs on combinatorial creativity. 7-10% improvement over baselines.

### 29. Combinatorial Creativity: A New Frontier in Generalization Abilities
- **Authors**: Samuel Schapiro et al.
- **Year**: 2025 | **arXiv**: [2509.21043](https://arxiv.org/abs/2509.21043)
- **Key Contribution**: Formalizes combinatorial creativity as generalization ability.

### 30. A Computational Framework for Conceptual Blending
- **Authors**: Manfred Eppe, Ewen Maclean, Roberto Confalonieri et al.
- **Year**: 2018 | **Venue**: Artificial Intelligence Journal
- **Key Contribution**: Formal computational framework for conceptual blending using ASP.

### 31. Towards Creative Information Exploration Based on Koestler's Bisociation
- **Authors**: Werner Dubitzky, Tobias Kotter et al.
- **Year**: 2012 | **Venue**: LNCS | **Springer**: [10.1007/978-3-642-31830-6_2](https://link.springer.com/chapter/10.1007/978-3-642-31830-6_2)
- **Key Contribution**: Applies Koestler's bisociation theory to creative information discovery.

### 32. The Process of Janusian Thinking in Creativity
- **Authors**: Albert Rothenberg
- **Year**: 1971 | **Venue**: Archives of General Psychiatry
- **Key Contribution**: Foundational theory of simultaneously holding contradictions — relevant to constraint-breaking.

---

## E. Cognitive Science of Discovery

### 33. AI-Descartes: Combining Data and Theory for Derivable Scientific Discovery
- **Authors**: Cristina Cornelio, Sanjeeb Dash et al.
- **Year**: 2023 | **Venue**: Nature Communications | **DOI**: [10.1038/s41467-023-37236-y](https://www.nature.com/articles/s41467-023-37236-y)
- **Key Contribution**: Combines symbolic reasoning with data via abductive logic programming for scientific discovery.

### 34. AI-Noether: Bridging AI Laws and Canonical Knowledge via Abductive Inference
- **Authors**: Karan Srivastava, Sanjeeb Dash et al.
- **Year**: 2025 | **arXiv**: [2509.23004](https://arxiv.org/abs/2509.23004)
- **Key Contribution**: Uses abductive inference to align AI-discovered laws with canonical scientific knowledge.

### 35. The Copycat Project: A Model of Mental Fluidity and Analogy-Making
- **Authors**: Douglas Hofstadter, Melanie Mitchell
- **Year**: 1995 | **Venue**: Fluid Concepts and Creative Analogies (Basic Books)
- **Key Contribution**: Foundational cognitive architecture for analogy-making using parallel terraced scan and codelets.

### 36. The Structure-Mapping Engine: Algorithm and Examples
- **Authors**: Brian Falkenhainer, Kenneth Forbus, Dedre Gentner
- **Year**: 1989 | **Venue**: Artificial Intelligence Journal
- **Key Contribution**: Implements Gentner's Structure-Mapping theory computationally. Core idea: analogies map relational structures, not surface features.

### 37. LLMs for Scientific Idea Generation: A Creativity-Centered Survey
- **Authors**: Shahhosseini et al.
- **Year**: 2025 | **arXiv**: [2511.07448](https://arxiv.org/html/2511.07448v2)
- **Key Finding**: Maps 5 method families across Boden's and Rhodes' (4Ps) creativity frameworks. **Most current methods address combinatorial/exploratory, NOT transformational creativity.** Multi-agent debate systems have highest potential for transformational creativity.

---

## Summary: Creativity Level Distribution

| Creativity Level | Current Research Coverage | Unbox Opportunity |
|---|---|---|
| **Combinatorial** (recombination) | ~60% of papers | Well-explored; diminishing returns |
| **Exploratory** (search within known space) | ~30% of papers | Some opportunity; Magellan, QD methods |
| **Transformational** (changing the space) | ~10% of papers | **Wide open; core Unbox thesis** |
