# 02. Constraint-Breaking & Hidden Assumptions in AI

> Papers that challenged fundamental assumptions in AI/ML, methods for systematic assumption identification, and historical paradigm shifts.

---

## A. Papers That Challenged Hidden Assumptions in Deep Learning

### 1. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
- **Authors**: Jonathan Frankle, Michael Carbin
- **Year**: 2018 | **Venue**: ICLR (Best Paper) | **arXiv**: [1803.03635](https://arxiv.org/abs/1803.03635)
- **Assumption Broken**: Dense neural networks are necessary for optimal performance
- **Core Insight**: Dense networks contain sparse subnetworks ("winning tickets") that achieve comparable accuracy when trained in isolation from initialization
- **Impact**: 3,600+ citations. Reframed what makes networks trainable.

### 2. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets
- **Authors**: Alethea Power, Yuri Burda, Harri Edwards et al. (OpenAI)
- **Year**: 2021 | **Venue**: ICLR Workshop | **arXiv**: [2201.02177](https://arxiv.org/abs/2201.02177)
- **Assumption Broken**: Generalization happens simultaneously with or shortly after training convergence
- **Core Insight**: Networks can "grok" — delayed generalization where perfect training accuracy precedes sudden generalization after many epochs of overfitting
- **Impact**: Revealed phase transitions in learning dynamics.

### 3. Towards Understanding Grokking: An Effective Theory of Representation Learning
- **Authors**: Ziming Liu, Ouail Kitouni, Niklas Nolte et al.
- **Year**: 2022 | **Venue**: NeurIPS | **arXiv**: [2205.10343](https://arxiv.org/abs/2205.10343)
- **Assumption Broken**: Network weights evolve monotonically toward optimal solutions
- **Core Insight**: Grokking is a phase transition — competition between memorization and generalization phases. Connected ML to physics-inspired analysis.

### 4. Reconciling Modern ML Practice and the Classical Bias-Variance Trade-off (Double Descent)
- **Authors**: Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal
- **Year**: 2019 | **Venue**: PNAS | **arXiv**: [1812.11118](https://arxiv.org/abs/1812.11118)
- **Assumption Broken**: Classical U-shaped bias-variance tradeoff curve
- **Core Insight**: "Double descent" — test error decreases again as models become over-parameterized beyond the interpolation threshold
- **Impact**: 1,500+ citations. Reshaped theoretical understanding of generalization.

### 5. Deep Double Descent: Where Bigger Models and More Data Hurt
- **Authors**: Preetum Nakkiran, Gal Kaplun et al. (incl. Ilya Sutskever)
- **Year**: 2019 | **arXiv**: [1912.02292](https://arxiv.org/abs/1912.02292)
- **Assumption Broken**: Increasing model size/data always improves performance
- **Core Insight**: Double descent occurs along multiple axes: model size, dataset size, AND training time. Universal across architectures.

### 6. Scaling Laws for Neural Language Models
- **Authors**: Jared Kaplan, Sam McCandlish et al. (OpenAI)
- **Year**: 2020 | **arXiv**: [2001.08361](https://arxiv.org/abs/2001.08361)
- **Assumption Broken**: Diminishing returns are inevitable as models grow
- **Core Insight**: Performance follows precise power-law scaling with model size, dataset size, and compute — spanning 7+ orders of magnitude. Architectural details matter less than scale.
- **Impact**: Foundational for GPT-3 and all subsequent LLM development.

### 7. Training Compute-Optimal Large Language Models (Chinchilla)
- **Authors**: Jordan Hoffmann, Sebastian Borgeaud et al. (DeepMind)
- **Year**: 2022 | **Venue**: NeurIPS | **arXiv**: [2203.15556](https://arxiv.org/abs/2203.15556)
- **Assumption Broken**: Kaplan et al.'s scaling law that model parameters should scale faster than data
- **Core Insight**: Optimal scaling uses equal compute for parameters and training tokens — smaller model on more data outperforms larger model on less data
- **Impact**: Changed industry practices. Showed that even scaling laws can be misinterpreted.

### 8. Scaling Laws for Reward Model Overoptimization
- **Authors**: Leo Gao, John Schulman, Jacob Hilton (OpenAI)
- **Year**: 2022 | **arXiv**: [2210.10760](https://arxiv.org/abs/2210.10760)
- **Assumption Broken**: Optimizing against a reward model reliably improves ground truth
- **Core Insight**: Goodhart's law in RLHF — proxy reward optimization eventually harms true performance. Defines scaling laws for overoptimization.

### 9. Emergent Abilities of Large Language Models
- **Authors**: Jason Wei, Yi Tay, Rishi Bommasani et al.
- **Year**: 2022 | **Venue**: TMLR | **arXiv**: [2206.07682](https://arxiv.org/abs/2206.07682)
- **Assumption Broken**: Language model capabilities scale smoothly and predictably
- **Core Insight**: "Emergent abilities" appear suddenly at certain scale thresholds — not predictable from smaller-scale performance. Includes reasoning, instruction following, chain-of-thought.
- **Impact**: Fundamental finding about phase transitions in LLMs.

### 10. How Does Batch Normalization Help Optimization?
- **Authors**: Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry
- **Year**: 2018 | **Venue**: NeurIPS | **arXiv**: [1805.11604](https://arxiv.org/abs/1805.11604)
- **Assumption Broken**: BatchNorm works by reducing "internal covariate shift"
- **Core Insight**: BatchNorm's benefits come from making the optimization landscape smoother — not from reducing covariate shift. The accepted explanation was wrong.

### 11. Challenging Common Assumptions in Unsupervised Learning of Disentangled Representations
- **Authors**: Francesco Locatello, Stefan Bauer et al.
- **Year**: 2019 | **Venue**: ICML | **arXiv**: [1811.12359](https://arxiv.org/abs/1811.12359)
- **Assumption Broken**: Unsupervised disentanglement is achievable without inductive biases
- **Core Insight**: Theoretical impossibility result — unsupervised disentanglement requires strong inductive biases or supervision. Challenged the core premise of the entire subfield.

### 12. Neural Tangent Kernel: Convergence and Generalization in Neural Networks
- **Authors**: Arthur Jacot, Franck Gabriel, Clement Hongler
- **Year**: 2018 | **Venue**: NeurIPS | **arXiv**: [1806.07572](https://arxiv.org/abs/1806.07572)
- **Assumption Broken**: Neural network training dynamics are fundamentally different from kernel methods
- **Core Insight**: In the infinite-width limit, neural networks behave like kernel methods (NTK). Connected deep learning to classical kernel theory — while also revealing limitations of this connection.

### 13. Failures of Gradient-Based Deep Learning
- **Authors**: Shai Shalev-Shwartz, Ohad Shamir, Shaked Shammah
- **Year**: 2017 | **Venue**: ICML | **arXiv**: [1706.03440](https://arxiv.org/abs/1706.03440)
- **Assumption Broken**: Gradient-based learning works robustly across problem domains
- **Core Insight**: Constructed counterexamples where gradient-based methods systematically fail. Reminded the community that DL success is not guaranteed.

### 14. Rethinking Bias-Variance Trade-off for Generalization of Neural Networks
- **Authors**: Zitong Yang, Yaodong Yu et al.
- **Year**: 2020 | **Venue**: ICML | **arXiv**: [2002.11328](https://arxiv.org/abs/2002.11328)
- **Core Insight**: Classical bias-variance decomposition doesn't apply to modern NNs due to implicit regularization from training procedures (SGD, adaptive LR).

---

## B. Historical Paradigm Shifts in AI

### 15. Attention Is All You Need
- **Authors**: Vaswani, Shazeer, Parmar et al.
- **Year**: 2017 | **Venue**: NeurIPS | **arXiv**: [1706.03762](https://arxiv.org/abs/1706.03762)
- **Assumption Broken**: Recurrence (RNN/LSTM) is necessary for sequential data
- **Core Insight**: Pure self-attention architecture outperforms recurrent models while being more parallelizable
- **Impact**: 80,000+ citations. Foundational for all modern AI.

### 16. ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
- **Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
- **Year**: 2012 | **Venue**: NeurIPS
- **Assumption Broken**: (1) Deep NNs are too hard to train (vanishing gradients), (2) SVMs/shallow models are optimal for vision
- **Core Insight**: Deep CNNs with GPU training, ReLU, dropout dramatically outperform everything (26% → 15.3% top-5 error)
- **Impact**: Catalyzed the deep learning revolution.

### 17. BERT: Pre-training of Deep Bidirectional Transformers
- **Authors**: Jacob Devlin, Ming-Wei Chang et al.
- **Year**: 2018 | **Venue**: NAACL | **arXiv**: [1810.04805](https://arxiv.org/abs/1810.04805)
- **Assumption Broken**: Language model pre-training requires unidirectional context
- **Core Insight**: Bidirectional pre-training (masked LM) + fine-tuning establishes dominant paradigm
- **Impact**: 70,000+ citations.

### 18. An Image is Worth 16x16 Words (Vision Transformer)
- **Authors**: Alexey Dosovitskiy, Lucas Beyer et al.
- **Year**: 2020 | **Venue**: ICLR | **arXiv**: [2010.11929](https://arxiv.org/abs/2010.11929)
- **Assumption Broken**: Convolutions are essential for computer vision
- **Core Insight**: Self-attention on image patches matches/exceeds CNN performance with sufficient data.

### 19. Language Models are Few-Shot Learners (GPT-3)
- **Authors**: Tom Brown, Benjamin Mann et al. (OpenAI)
- **Year**: 2020 | **Venue**: NeurIPS | **arXiv**: [2005.14165](https://arxiv.org/abs/2005.14165)
- **Assumption Broken**: LMs require task-specific fine-tuning
- **Core Insight**: In-context learning — performing tasks from few examples in the prompt alone, without gradient updates.

### 20. Chain-of-Thought Prompting Elicits Reasoning in LLMs
- **Authors**: Jason Wei, Xuezhi Wang et al.
- **Year**: 2022 | **Venue**: NeurIPS | **arXiv**: [2201.11903](https://arxiv.org/abs/2201.11903)
- **Assumption Broken**: Complex reasoning requires architectural changes or specialized training
- **Core Insight**: Simply including step-by-step reasoning examples in prompts unlocks multi-step reasoning abilities.

### 21. Denoising Diffusion Probabilistic Models
- **Authors**: Jonathan Ho, Ajay Jain, Pieter Abbeel
- **Year**: 2020 | **Venue**: NeurIPS | **arXiv**: [2006.11239](https://arxiv.org/abs/2006.11239)
- **Assumption Broken**: GANs / adversarial training is necessary for high-quality generation
- **Core Insight**: Learned denoising process matches/exceeds GAN quality with stable training, better coverage, and explicit likelihood.

### 22. The Forward-Forward Algorithm: Some Preliminary Investigations
- **Authors**: Geoffrey Hinton
- **Year**: 2022 | **arXiv**: [2212.13345](https://arxiv.org/abs/2212.13345)
- **Assumption Broken**: Backpropagation is the only viable learning algorithm for deep networks
- **Core Insight**: Two forward passes (positive + negative data) with local learning signals can train deep networks. More biologically plausible.
- **Impact**: Reopened debate about alternatives to backprop.

---

## C. Methods for Discovering/Breaking Constraints

### 23. AutoTRIZ: Artificial Ideation with TRIZ and Large Language Models
- **Authors**: Shuo Jiang, Jianxi Luo
- **Year**: 2024 | **arXiv**: [2403.13002](https://arxiv.org/abs/2403.13002)
- **Key Contribution**: Combines TRIZ (Theory of Inventive Problem Solving) — 40 inventive principles + contradiction matrices — with LLM prompting for systematic creative problem-solving
- **Relevance to Unbox**: Directly applicable methodology for systematic assumption breaking.

### 24. Inductive Biases for Deep Learning of Higher-Level Cognition
- **Authors**: Anirudh Goyal, Yoshua Bengio
- **Year**: 2020 | **Venue**: Proceedings of the Royal Society A
- **Key Contribution**: Hypothesis that intelligence may be explained by a few fundamental principles rather than many heuristics. Explores which inductive biases (object-centric representations, attention, predictive coding) are essential.
- **Relevance to Unbox**: Identifies what the "hidden assumptions" of deep learning architecture actually are.

### 25. Relational Inductive Biases, Deep Learning, and Graph Networks
- **Authors**: Peter Battaglia, Jessica Hamrick et al.
- **Year**: 2018 | **arXiv**: [1806.01261](https://arxiv.org/abs/1806.01261)
- **Key Contribution**: Standard architectures don't adequately capture relational structure. Proposes graph networks with explicit entity/relation representations.
- **Relevance to Unbox**: Identifies architectural assumptions worth questioning.

### 26. The Unreasonable Effectiveness of Deep Learning in Artificial Intelligence
- **Authors**: Terrence Sejnowski
- **Year**: 2020 | **Venue**: PNAS
- **Key Contribution**: Synthesis of why DL works: hierarchical representations, compositionality, distributed representations, scale.
- **Relevance to Unbox**: Framework for understanding which principles are fundamental vs. contingent.

---

## Patterns in Paradigm-Breaking AI Research

| Pattern | Examples | Implication for Unbox |
|---|---|---|
| **Scale reveals hidden phenomena** | GPT-3 (in-context learning), emergent abilities, double descent | Scaling isn't just engineering — it's hypothesis testing |
| **Accepted explanations can be wrong** | BatchNorm, disentanglement impossibility | Rigorously testing "common knowledge" is fertile ground |
| **Cross-domain transfer breaks assumptions** | Transformers (NLP → vision), diffusion (physics → generative) | Domain-specificity assumptions are often unwarranted |
| **Replication reveals errors** | Chinchilla vs. original scaling laws | Careful replication can itself be paradigm-shifting |
| **Alternatives to dominant methods exist** | Forward-Forward vs. backprop, diffusion vs. GANs | "The way we've always done it" is not the only way |
