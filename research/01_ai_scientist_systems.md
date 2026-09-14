# 01. AI Scientist Systems & Automated Research

> Papers on end-to-end automated research systems, AutoML beyond optimization, and meta-learning for algorithm discovery.

---

## A. End-to-End AI Scientist Systems

### 1. FunSearch: Making New Discoveries in Mathematical Sciences Using LLMs
- **Authors**: Alhussein Fawzi, Bernardino Romera Paredes et al. (DeepMind)
- **Year**: 2023 | **Venue**: Nature
- **Link**: https://deepmind.google/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/
- **Key Contribution**: First LLM-based system to make genuine discoveries in open mathematical problems. Combines pre-trained LLM with automated evaluator in evolutionary search loop. Discovered new solutions to the cap set problem and bin packing.
- **Novelty Assessment**: **GENUINE NOVELTY** — Found solutions to previously unsolved combinatorial optimization problems.
- **Relevance to Unbox**: Demonstrates that LLM + evolutionary search + automated evaluation can produce genuinely new results — but limited to domains with verifiable objective functions.

### 2. AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms
- **Authors**: Matej Balog et al. (DeepMind)
- **Year**: 2025 | **Venue**: DeepMind Blog
- **Link**: https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- **Key Contribution**: Evolutionary search combining LLM creativity with automated evaluators. Freed 0.7% of Google's global computing resources, reduced AI training time by 1%.
- **Novelty Assessment**: OPTIMIZATION — Optimizes existing algorithms, does not discover fundamentally new paradigms.

### 3. The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery
- **Authors**: Chris Lu, Cong Lu, Robert Tjarko Lange et al. (Sakana AI)
- **Year**: 2024 | **Venue**: arXiv
- **Link**: https://sakana.ai/ai-scientist/
- **Key Contribution**: First fully automated end-to-end system: idea generation → experiment design → execution → paper writing → peer review. Cost ~$15/paper.
- **Novelty Assessment**: STRUCTURALLY COMPLETE BUT SUBSTANTIVELY FLAWED — Independent evaluation (Beel et al. 2025) found 42% experiment failure rate, 57% numerical hallucination, quality of "unmotivated undergraduate."
- **Critical Evaluation**: [arXiv:2502.14297](https://arxiv.org/abs/2502.14297)

### 4. The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search
- **Authors**: Yutaro Yamada, Robert Tjarko Lange, Cong Lu et al.
- **Year**: 2025 | **Venue**: ICLR 2025 Workshop | **arXiv**: [2504.08066](https://arxiv.org/abs/2504.08066)
- **Key Contribution**: First AI-generated paper accepted to a peer-reviewed workshop. Uses agentic tree search for more structured exploration.
- **Novelty Assessment**: Incremental improvement over v1, still within known paradigms.

### 5. Gemini Deep Think: Accelerating Mathematical and Scientific Discovery
- **Authors**: Thang Luong, Vahab Mirrokni (DeepMind)
- **Year**: 2026 | **Venue**: DeepMind Blog
- **Link**: https://deepmind.com/blog/accelerating-mathematical-and-scientific-discovery-with-gemini-deep-think/
- **Key Contribution**: Achieved Gold-medal standard at IMO 2025. Solves professional research problems in mathematics, physics, and CS under expert direction.
- **Novelty Assessment**: Novel problem-solving within known mathematical paradigms.

### 6. CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation
- **Authors**: Peter Jansen, Oyvind Tafjord, Marissa Radensky et al.
- **Year**: 2025 | **Venue**: ACL 2025 Findings | **arXiv**: [2503.22708](https://arxiv.org/abs/2503.22708)
- **Key Contribution**: Frames ideation as genetic search over research articles + code blocks. Generated 19 discoveries spanning new tasks, agents, metrics, and data across agents and virtual environments.
- **Novelty Assessment**: **GENUINE NOVELTY** — Discovered qualitatively new research directions beyond benchmark optimization.

### 7. DeepScientist: Advancing Frontier-Pushing Scientific Findings Progressively
- **Authors**: Yixuan Weng, Minjun Zhu, Qiujie Xie et al.
- **Year**: 2025 | **arXiv**: [2509.26603](https://arxiv.org/abs/2509.26603)
- **Key Contribution**: Goal-oriented autonomous discovery over month-long timelines. Bayesian Optimization with "hypothesize-verify-analyze" loop. Generated ~5,000 ideas, validated ~1,100, surpassed human SOTA on 3 tasks by up to 183.7%.
- **Novelty Assessment**: **GENUINE NOVELTY** — Progressively surpassed human SOTA on scientific tasks.

### 8. Google AI Co-Scientist
- **Authors**: Google Research
- **Year**: 2025 | **Link**: https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf
- **Key Contribution**: Multi-agent system for hypotheses generation, literature synthesis, experiment design. More sophisticated than Sakana's system.
- **Novelty Assessment**: Released with limited independent evaluation. Still relies on human-defined research directions.

### 9. Kosmos: An AI Scientist for Autonomous Discovery
- **Authors**: Ludovico Mitchener, Angela Yiu, Benjamin Chang et al.
- **Year**: 2025 | **arXiv**: [2511.02824](https://arxiv.org/abs/2511.02824)
- **Key Contribution**: Comprehensive AI scientist integrating literature search, hypothesis generation, and data analysis.

### 10. Autonomous Chemical Research with Large Language Models (Coscientist)
- **Authors**: Daniil A. Boiko, Robert MacKnight, Gabe Gomes
- **Year**: 2023 | **Venue**: Nature
- **Key Contribution**: Uses GPT-4 to design and plan chemical experiments. Successfully replicated Nobel Prize-winning reactions.
- **Novelty Assessment**: Impressive automation of known protocols. No evidence of genuinely novel chemical hypotheses.

### 11. ChemCrow: Augmenting Large Language Models with Chemistry Tools
- **Authors**: Andres M. Bran et al.
- **Year**: 2024 | **Venue**: Nature Machine Intelligence
- **Key Contribution**: Integrates 18 expert chemistry tools with GPT-4. Autonomous synthesis planning, guided chromophore discovery.
- **Novelty Assessment**: Successfully executes within established chemical frameworks. "Novel" discoveries were guided by human-defined constraints.

---

## B. Meta-Research & Search Frameworks

### 12. EvoX: Meta-Evolution for Automated Discovery
- **Authors**: Shu Liu, Shubham Agarwal et al.
- **Year**: 2026 | **arXiv**: [2602.23413](https://arxiv.org/abs/2602.23413)
- **Key Contribution**: Jointly evolves candidate solutions AND search strategies. Dynamically shifts between exploration/exploitation. Outperforms AlphaEvolve on 200+ tasks.
- **Novelty Assessment**: META-OPTIMIZATION — Optimizes the optimization process itself.

### 13. SelfAI: A Self-Directed Framework for Long-Horizon Scientific Discovery
- **Authors**: Xiao Wu, Ting-Zhu Huang et al.
- **Year**: 2026 | **arXiv**: [2512.00403](https://arxiv.org/abs/2512.00403)
- **Key Contribution**: Multi-agent discovery with strategic trajectory-driven exploration. Explicit efficiency-diversity trade-offs.

### 14. OR-Agent: Bridging Evolutionary Search and Structured Research
- **Authors**: Qi Liu, Ruochen Hao et al.
- **Year**: 2026 | **arXiv**: [2602.13769](https://arxiv.org/abs/2602.13769)
- **Key Contribution**: Tree-based research workflow with "verbal gradients" and "verbal momentum" as reflection mechanisms.

### 15. AgentRxiv: Towards Collaborative Autonomous Research
- **Authors**: Samuel Schmidgall, Michael Moor
- **Year**: 2025 | **arXiv**: [2503.18102](https://arxiv.org/abs/2503.18102)
- **Key Contribution**: Framework enabling LLM agent laboratories to share and retrieve reports from shared literature base.

### 16. AI4Research: A Survey of Artificial Intelligence for Scientific Research
- **Authors**: Qiguang Chen, Mingda Yang et al.
- **Year**: 2025 | **arXiv**: [2507.01903](https://arxiv.org/abs/2507.01903)
- **Key Contribution**: Comprehensive survey with systematic taxonomy of 5 mainstream tasks in AI4Research.

---

## B-2. Interdisciplinary Inspiration & Ideation Frameworks

### 20. Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration (Idea-Catalyst)
- **Authors**: Priyanka Kargupta, Shuhaib Mehri, Dilek Hakkani-Tur, Jiawei Han (UIUC)
- **Year**: 2026 | **arXiv**: [2603.12226](https://arxiv.org/abs/2603.12226)
- **GitHub**: [pkargupta/idea_catalyst](https://github.com/pkargupta/idea_catalyst)
- **Dataset**: [HuggingFace](https://huggingface.co/datasets/pkargupta/idea_catalyst) — CHIMERA-derived, 400 research problems
- **Key Contribution**: Metacognition-driven framework for interdisciplinary idea generation. Three-stage pipeline: (1) decompose research goal into bottleneck questions with domain-specific + domain-agnostic dual formulations, (2) retrieve and analyze cross-domain literature via Semantic Scholar snippet search, (3) integrate and rank insights by interdisciplinary potential using pairwise comparison. +21% novelty, +16% insightfulness vs retrieval baselines.
- **Technical Details**: Uses vLLM + Qwen3-14B with Pydantic structured output schemas. 6 chained LLM calls per problem (decomposition → target analysis → cross-domain query generation → cross-domain analysis → integration → ranking). Semantic Scholar snippet search API with domain filtering.
- **Novelty Assessment**: **EXPLORATORY/COMBINATORIAL** — Systematically finds cross-domain analogies but does not challenge or transform the conceptual space itself. The "bottleneck abstraction" step is the closest to transformational thinking, but still operates within the given problem framing.
- **Limitations**: Small human study (6 PhD researchers). LLM-based evaluation of novelty/insightfulness. Single model (Qwen3-14B). "Novel > useful" gap in generated ideas.
- **Relevance to Unbox**: 
  - **Complementary**: Idea-Catalyst's domain-agnostic abstraction technique can strengthen Unbox's assumption extraction by making assumptions portable across domains.
  - **Differentiation**: Idea-Catalyst stays at exploratory/combinatorial creativity (borrowing solutions from other fields). Unbox targets transformational creativity (breaking the rules themselves). The two address different levels of Boden's hierarchy.
  - **Reusable**: Pairwise ranking methodology, structured output schemas, CHIMERA-derived benchmark data.

---

## C. AutoML & Neural Architecture Search

### 17. Neural Architecture Search with Reinforcement Learning
- **Authors**: Barret Zoph, Quoc V. Le
- **Year**: 2017 | **Venue**: ICLR 2017 | **arXiv**: [1611.01578](https://arxiv.org/abs/1611.01578)
- **Key Contribution**: First major RL-based NAS. Discovered architectures achieving SOTA on CIFAR-10 and novel recurrent cell outperforming LSTM.
- **Novelty Assessment**: **GENUINE NOVELTY** (in 2017) — Architectures not designed by humans.

### 18. Large-Scale Evolution of Image Classifiers
- **Authors**: Esteban Real, Sherry Moore et al.
- **Year**: 2017 | **Venue**: ICML 2017 | **arXiv**: [1703.01041](https://arxiv.org/abs/1703.01041)
- **Key Contribution**: Evolutionary algorithms discovering classifiers from trivial initial conditions.
- **Novelty Assessment**: **GENUINE NOVELTY** — Discovered novel architectures automatically.

### 19. NNGPT: Rethinking AutoML with Large Language Models
- **Authors**: Roman Kochnev, Waleed Khalid et al.
- **Year**: 2025 | **arXiv**: [2511.20333](https://arxiv.org/abs/2511.20333)
- **Key Contribution**: LLM as self-improving AutoML engine. Integrates architecture synthesis, HPO, accuracy prediction, RL. Generated 5K+ validated models.

---

## Summary Statistics

| Category | Genuine Novelty | Optimization/Framework | Survey |
|---|---|---|---|
| AI Scientist Systems | FunSearch, CodeScientist, DeepScientist | AI Scientist v1/v2, AlphaEvolve, EvoX | AI4Research |
| AutoML/NAS | Zoph & Le, Real et al. | NNGPT | — |
| Chemistry Agents | — | ChemCrow, Coscientist | — |

**Key Insight**: Systems achieve genuine novelty primarily in domains with **automated verifiability** (math, code, algorithmic). Open-ended scientific novelty — where evaluation itself is subjective — remains largely unsolved.
