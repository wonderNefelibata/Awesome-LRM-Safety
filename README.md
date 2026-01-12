# Awesome Large Reasoning Model (LRM) Safety 🔥

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Auto Update](https://github.com/wonderNefelibata/Awesome-LRM-Safety/actions/workflows/arxiv-update.yml/badge.svg)

A curated list of **security and safety research** for Large Reasoning Models (LRMs) like DeepSeek-R1, OpenAI o1, and other cutting-edge models. Focused on identifying risks, mitigation strategies, and ethical implications.

---

## 📜 Table of Contents
- [Awesome Large Reasoning Model (LRM) Safety 🔥](#awesome-large-reasoning-model-lrm-safety-)
  - [📜 Table of Contents](#-table-of-contents)
  - [🚀 Motivation](#-motivation)
  - [🤖 Large Reasoning Models](#-large-reasoning-models)
    - [Open Source Models](#open-source-models)
    - [Close Source Models](#close-source-models)
  - [📰 Latest arXiv Papers (Auto-Updated)](#-latest-arxiv-papers-auto-updated)
  - [🔑 Key Safety Domains(coming soon)](#-key-safety-domainscoming-soon)
  - [🔖 Dataset \& Benchmark](#-dataset--benchmark)
    - [For Traditional LLM](#for-traditional-llm)
    - [For Advanced LRM](#for-advanced-lrm)
  - [📚 Survey](#-survey)
    - [LRM Related](#lrm-related)
    - [LRM Safety Related](#lrm-safety-related)
  - [🛠️ Projects \& Tools(coming soon)](#️-projects--toolscoming-soon)
    - [Model-Specific Resources(example)](#model-specific-resourcesexample)
    - [General Tools(coming soon)(example)](#general-toolscoming-soonexample)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [❓ FAQ](#-faq)
  - [🔗 References](#-references)

---

## 🚀 Motivation

Large Reasoning Models (LRMs) are revolutionizing AI capabilities in complex decision-making scenarios. However, their deployment raises critical safety concerns.

This repository aims to catalog research addressing these challenges and promote safer LRM development.

## 🤖 Large Reasoning Models

### Open Source Models
  

| Name | Organization | Date | Technic | Cold-Start | Aha Moment | Modality |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1 | DeepSeek | 2025/01/22 | GRPO | ✅   | ✅   | text-only |
| QwQ-32B | Qwen | 2025/03/06 | -   | -   | -   | text-only |

### Close Source Models
  

| Name | Organization | Date | Technic | Cold-Start | Aha Moment | Modality |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI-o1 | OpenAI | 2024/09/12 | -   | -   | -   | text,image |
| Gemini-2.0-Flash-Thinking | Google | 2025/01/21 | -   | -   | -   | text,image |
| Kimi-k1.5 | Moonshot | 2025/01/22 | -   | -   | -   | text,image |
| OpenAI-o3-mini | OpenAI | 2025/01/31 | -   | -   | -   | text,image |
| Grok-3 | xAI | 2025/02/19 | -   | -   | -   | text,image |
| Claude-3.7-Sonnet | Anthropic | 2025/02/24 | -   | -   | -   | text,image |
| Gemini-2.5-Pro | Google | 2025/03/25 | -   | -   | -   | text,image |

---

## 📰 Latest arXiv Papers (Auto-Updated)
It is updated every 12 hours, presenting the latest 20 relevant papers.And Earlier Papers can be found [here](./articles/README.md).


<!-- LATEST_PAPERS_START -->


| Date       | Title                                      | Authors           | Abstract Summary          |
|------------|--------------------------------------------|-------------------|---------------------------|
| 2026-01-09 | [The Molecular Structure of Thought: Mapping the Topology of Long Chain-of-Thought Reasoning](http://arxiv.org/abs/2601.06002v1) | Qiguang Chen, Yantao Du et al. | Large language models (LLMs) often fail to learn effective long chain-of-thought (Long CoT) reasoning from human or non-Long-CoT LLMs imitation. To understand this, we propose that effective and learnable Long CoT trajectories feature stable molecular-like structures in unified view, which are formed by three interaction types: Deep-Reasoning (covalent-like), Self-Reflection (hydrogen-bond-like), and Self-Exploration (van der Waals-like). Analysis of distilled trajectories reveals these structures emerge from Long CoT fine-tuning, not keyword imitation. We introduce Effective Semantic Isomers and show that only bonds promoting fast entropy convergence support stable Long CoT learning, while structural competition impairs training. Drawing on these findings, we present Mole-Syn, a distribution-transfer-graph method that guides synthesis of effective Long CoT structures, boosting performance and RL stability across benchmarks. |
| 2026-01-09 | [Open-Vocabulary 3D Instruction Ambiguity Detection](http://arxiv.org/abs/2601.05991v1) | Jiayu Ding, Haoran Tang et al. | In safety-critical domains, linguistic ambiguity can have severe consequences; a vague command like "Pass me the vial" in a surgical setting could lead to catastrophic errors. Yet, most embodied AI research overlooks this, assuming instructions are clear and focusing on execution rather than confirmation. To address this critical safety gap, we are the first to define Open-Vocabulary 3D Instruction Ambiguity Detection, a fundamental new task where a model must determine if a command has a single, unambiguous meaning within a given 3D scene. To support this research, we build Ambi3D, the large-scale benchmark for this task, featuring over 700 diverse 3D scenes and around 22k instructions. Our analysis reveals a surprising limitation: state-of-the-art 3D Large Language Models (LLMs) struggle to reliably determine if an instruction is ambiguous. To address this challenge, we propose AmbiVer, a two-stage framework that collects explicit visual evidence from multiple views and uses it to guide an vision-language model (VLM) in judging instruction ambiguity. Extensive experiments demonstrate the challenge of our task and the effectiveness of AmbiVer, paving the way for safer and more trustworthy embodied AI. Code and dataset available at https://jiayuding031020.github.io/ambi3d/. |
| 2026-01-09 | [An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift](http://arxiv.org/abs/2601.05882v1) | Constantinos Karouzos, Xingwei Tan et al. | Preference tuning aligns pretrained language models to human judgments of quality, helpfulness, or safety by optimizing over explicit preference signals rather than likelihood alone. Prior work has shown that preference-tuning degrades performance and reduces helpfulness when evaluated outside the training domain. However, the extent to which adaptation strategies mitigate this domain shift remains unexplored. We address this challenge by conducting a comprehensive and systematic study of alignment generalization under domain shift. We compare five popular alignment objectives and various adaptation strategies from source to target, including target-domain supervised fine-tuning and pseudo-labeling, across summarization and question-answering helpfulness tasks. Our findings reveal systematic differences in generalization across alignment objectives under domain shift. We show that adaptation strategies based on pseudo-labeling can substantially reduce domain-shift degradation |
| 2026-01-09 | [Non Destructive Testing](http://arxiv.org/abs/2601.05880v1) | Gonzalo Arnau Izquierdo | The contribution introduces the principles, methods, and applications of non-destructive testing in the context of particle accelerators and related technologies is presented. Both surface inspection methods (visual testing, penetrant testing, magnetic particle testing, eddy current testing) and volumetric methods (radiographic testing, ultrasonic testing) are presented, with examples drawn from CERN projects. Emphasis is placed on the capabilities, limitations, and practical considerations of each technique, highlighting their role in ensuring quality and safety during fabrication, procurement, and in-service inspections. The contribution also addresses standards, codes, and personnel qualification schemes, underlining their regulatory impact in fields such as pressure equipment and industrial piping. Finally, the importance of early integration of NDT into project planning is stressed, not only for compliance but also as a proactive quality and efficiency measure. |
| 2026-01-09 | [When legs and bodies synchronize: Two-level collective dynamics in dense crowds](http://arxiv.org/abs/2601.05867v1) | Thomas Chatagnon, Mohcine Chraibi et al. | Ultra-dense crowds, in which physical contact between people cannot be avoided, pose major safety concerns. Nevertheless, the underlying dynamics driving their collective behaviours remain poorly understood. Existing dense crowd models, mostly two-dimensional and contact-based, overlook biomechanical mechanisms that govern individual balance motion. In this study, we introduce a minimal two-level pedestrian model that couples upper body and legs dynamics, allowing us to capture transitions between balanced and unbalanced states at the individual scale. Whereas previous models fail to achieve it, this coupling gives rise to emergent collective behaviours observed empirically, such as self-organized waves and large-scale rotational motion within the crowd. The model bridges basic individual biomechanical concepts and macroscopic flow dynamics, offering a new framework for modelling and understanding collective motions in ultra-dense crowds. |
| 2026-01-09 | [Intelligent Singularity Avoidance in UR10 Robotic Arm Path Planning Using Hybrid Fuzzy Logic and Reinforcement Learning](http://arxiv.org/abs/2601.05836v1) | Sheng-Kai Chen, Jyh-Horng Wu | This paper presents a comprehensive approach to singularity detection and avoidance in UR10 robotic arm path planning through the integration of fuzzy logic safety systems and reinforcement learning algorithms. The proposed system addresses critical challenges in robotic manipulation where singularities can cause loss of control and potential equipment damage. Our hybrid approach combines real-time singularity detection using manipulability measures, condition number analysis, and fuzzy logic decision-making with a stable reinforcement learning framework for adaptive path planning. Experimental results demonstrate a 90% success rate in reaching target positions while maintaining safe distances from singular configurations. The system integrates PyBullet simulation for training data collection and URSim connectivity for real-world deployment. |
| 2026-01-09 | [Peek2: A Regex-free implementation of pretokenizers for Byte-level BPE](http://arxiv.org/abs/2601.05833v1) | Liu Zai | Pretokenization is a crucial, sequential pass in Byte-level BPE tokenizers. Our proposed new implementation, Peek2, serves as a drop-in replacement for cl100k-like pretokenizers used in GPT-3, LLaMa-3, and Qwen-2.5. Designed with performance and safety in mind, Peek2 is Regex-free and delivers a $ 1.11\times $ improvement in overall throughput across the entire Byte-level BPE encoding process. This algorithm runs entirely on the CPU, has stable linear complexity $ O(n) $, and provides presegmentation results identical to those of the original Regex-based pretokenizer. |
| 2026-01-09 | [Modular Autonomy with Conversational Interaction: An LLM-driven Framework for Decision Making in Autonomous Driving](http://arxiv.org/abs/2601.05806v1) | Marvin Seegert, Korbinian Moller et al. | Recent advancements in Large Language Models (LLMs) offer new opportunities to create natural language interfaces for Autonomous Driving Systems (ADSs), moving beyond rigid inputs. This paper addresses the challenge of mapping the complexity of human language to the structured action space of modular ADS software. We propose a framework that integrates an LLM-based interaction layer with Autoware, a widely used open-source software. This system enables passengers to issue high-level commands, from querying status information to modifying driving behavior. Our methodology is grounded in three key components: a taxonomization of interaction categories, an application-centric Domain Specific Language (DSL) for command translation, and a safety-preserving validation layer. A two-stage LLM architecture ensures high transparency by providing feedback based on the definitive execution status. Evaluation confirms the system's timing efficiency and translation robustness. Simulation successfully validated command execution across all five interaction categories. This work provides a foundation for extensible, DSL-assisted interaction in modular and safety-conscious autonomy stacks. |
| 2026-01-09 | [VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit](http://arxiv.org/abs/2601.05755v1) | Junda Lin, Zhaomeng Zhou et al. | LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility. Code is available at https://anonymous.4open.science/r/VIGIL-378B/. |
| 2026-01-09 | [AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](http://arxiv.org/abs/2601.05752v1) | Shu Yang, Jingyu Hu et al. | We introduce AutoMonitor-Bench, the first benchmark designed to systematically evaluate the reliability of LLM-based misbehavior monitors across diverse tasks and failure modes. AutoMonitor-Bench consists of 3,010 carefully annotated test samples spanning question answering, code generation, and reasoning, with paired misbehavior and benign instances. We evaluate monitors using two complementary metrics: Miss Rate (MR) and False Alarm Rate (FAR), capturing failures to detect misbehavior and oversensitivity to benign behavior, respectively. Evaluating 12 proprietary and 10 open-source LLMs, we observe substantial variability in monitoring performance and a consistent trade-off between MR and FAR, revealing an inherent safety-utility tension. To further explore the limits of monitor reliability, we construct a large-scale training corpus of 153,581 samples and fine-tune Qwen3-4B-Instruction to investigate whether training on known, relatively easy-to-construct misbehavior datasets improves monitoring performance on unseen and more implicit misbehaviors. Our results highlight the challenges of reliable, scalable misbehavior monitoring and motivate future work on task-aware designing and training strategies for LLM-based monitors. |
| 2026-01-09 | [The Echo Chamber Multi-Turn LLM Jailbreak](http://arxiv.org/abs/2601.05742v1) | Ahmad Alobaid, Martí Jordà Roca et al. | The availability of Large Language Models (LLMs) has led to a new generation of powerful chatbots that can be developed at relatively low cost. As companies deploy these tools, security challenges need to be addressed to prevent financial loss and reputational damage. A key security challenge is jailbreaking, the malicious manipulation of prompts and inputs to bypass a chatbot's safety guardrails. Multi-turn attacks are a relatively new form of jailbreaking involving a carefully crafted chain of interactions with a chatbot. We introduce Echo Chamber, a new multi-turn attack using a gradual escalation method. We describe this attack in detail, compare it to other multi-turn attacks, and demonstrate its performance against multiple state-of-the-art models through extensive evaluation. |
| 2026-01-09 | [PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility](http://arxiv.org/abs/2601.05739v1) | G M Shahariar, Zabir Al Nazi et al. | Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions. |
| 2026-01-09 | [Drivora: A Unified and Extensible Infrastructure for Search-based Autonomous Driving Testing](http://arxiv.org/abs/2601.05685v1) | Mingfei Cheng, Lionel Briand et al. | Search-based testing is critical for evaluating the safety and reliability of autonomous driving systems (ADSs). However, existing approaches are often built on heterogeneous frameworks (e.g., distinct scenario spaces, simulators, and ADSs), which require considerable effort to reuse and adapt across different settings. To address these challenges, we present Drivora, a unified and extensible infrastructure for search-based ADS testing built on the widely used CARLA simulator. Drivora introduces a unified scenario definition, OpenScenario, that specifies scenarios using low-level, actionable parameters to ensure compatibility with existing methods while supporting extensibility to new testing designs (e.g., multi-autonomous-vehicle testing). On top of this, Drivora decouples the testing engine, scenario execution, and ADS integration. The testing engine leverages evolutionary computation to explore new scenarios and supports flexible customization of core components. The scenario execution can run arbitrary scenarios using a parallel execution mechanism that maximizes hardware utilization for large-scale batch simulation. For ADS integration, Drivora provides access to 12 ADSs through a unified interface, streamlining configuration and simplifying the incorporation of new ADSs. Our tools are publicly available at https://github.com/MingfeiCheng/Drivora. |
| 2026-01-09 | [A Framework for Personalized Persuasiveness Prediction via Context-Aware User Profiling](http://arxiv.org/abs/2601.05654v1) | Sejun Park, Yoonah Park et al. | Estimating the persuasiveness of messages is critical in various applications, from recommender systems to safety assessment of LLMs. While it is imperative to consider the target persuadee's characteristics, such as their values, experiences, and reasoning styles, there is currently no established systematic framework to optimize leveraging a persuadee's past activities (e.g., conversations) to the benefit of a persuasiveness prediction model. To address this problem, we propose a context-aware user profiling framework with two trainable components: a query generator that generates optimal queries to retrieve persuasion-relevant records from a user's history, and a profiler that summarizes these records into a profile to effectively inform the persuasiveness prediction model. Our evaluation on the ChangeMyView Reddit dataset shows consistent improvements over existing methods across multiple predictor models, with gains of up to +13.77%p in F1 score. Further analysis shows that effective user profiles are context-dependent and predictor-specific, rather than relying on static attributes or surface-level similarity. Together, these results highlight the importance of task-oriented, context-dependent user profiling for personalized persuasiveness prediction. |
| 2026-01-09 | [EvoQRE: Modeling Bounded Rationality in Safety-Critical Traffic Simulation via Evolutionary Quantal Response Equilibrium](http://arxiv.org/abs/2601.05653v1) | Phu-Hoa Pham, Chi-Nguyen Tran et al. | Existing traffic simulation frameworks for autonomous vehicles typically rely on imitation learning or game-theoretic approaches that solve for Nash or coarse correlated equilibria, implicitly assuming perfectly rational agents. However, human drivers exhibit bounded rationality, making approximately optimal decisions under cognitive and perceptual constraints. We propose EvoQRE, a principled framework for modeling safety-critical traffic interactions as general-sum Markov games solved via Quantal Response Equilibrium (QRE) and evolutionary game dynamics. EvoQRE integrates a pre-trained generative world model with entropy-regularized replicator dynamics, capturing stochastic human behavior while maintaining equilibrium structure. We provide rigorous theoretical results, proving that the proposed dynamics converge to Logit-QRE under a two-timescale stochastic approximation with an explicit convergence rate of O(log k / k^{1/3}) under weak monotonicity assumptions. We further extend QRE to continuous action spaces using mixture-based and energy-based policy representations. Experiments on the Waymo Open Motion Dataset and nuPlan benchmark demonstrate that EvoQRE achieves state-of-the-art realism, improved safety metrics, and controllable generation of diverse safety-critical scenarios through interpretable rationality parameters. |
| 2026-01-09 | [Multilingual Amnesia: On the Transferability of Unlearning in Multilingual LLMs](http://arxiv.org/abs/2601.05641v1) | Alireza Dehghanpour Farashah, Aditi Khandelwal et al. | As multilingual large language models become more widely used, ensuring their safety and fairness across diverse linguistic contexts presents unique challenges. While existing research on machine unlearning has primarily focused on monolingual settings, typically English, multilingual environments introduce additional complexities due to cross-lingual knowledge transfer and biases embedded in both pretraining and fine-tuning data. In this work, we study multilingual unlearning using the Aya-Expanse 8B model under two settings: (1) data unlearning and (2) concept unlearning. We extend benchmarks for factual knowledge and stereotypes to ten languages through translation: English, French, Arabic, Japanese, Russian, Farsi, Korean, Hindi, Hebrew, and Indonesian. These languages span five language families and a wide range of resource levels. Our experiments show that unlearning in high-resource languages is generally more stable, with asymmetric transfer effects observed between typologically related languages. Furthermore, our analysis of linguistic distances indicates that syntactic similarity is the strongest predictor of cross-lingual unlearning behavior. |
| 2026-01-09 | [SGDrive: Scene-to-Goal Hierarchical World Cognition for Autonomous Driving](http://arxiv.org/abs/2601.05640v1) | Jingyu Li, Junjie Wu et al. | Recent end-to-end autonomous driving approaches have leveraged Vision-Language Models (VLMs) to enhance planning capabilities in complex driving scenarios. However, VLMs are inherently trained as generalist models, lacking specialized understanding of driving-specific reasoning in 3D space and time. When applied to autonomous driving, these models struggle to establish structured spatial-temporal representations that capture geometric relationships, scene context, and motion patterns critical for safe trajectory planning. To address these limitations, we propose SGDrive, a novel framework that explicitly structures the VLM's representation learning around driving-specific knowledge hierarchies. Built upon a pre-trained VLM backbone, SGDrive decomposes driving understanding into a scene-agent-goal hierarchy that mirrors human driving cognition: drivers first perceive the overall environment (scene context), then attend to safety-critical agents and their behaviors, and finally formulate short-term goals before executing actions. This hierarchical decomposition provides the structured spatial-temporal representation that generalist VLMs lack, integrating multi-level information into a compact yet comprehensive format for trajectory planning. Extensive experiments on the NAVSIM benchmark demonstrate that SGDrive achieves state-of-the-art performance among camera-only methods on both PDMS and EPDMS, validating the effectiveness of hierarchical knowledge structuring for adapting generalist VLMs to autonomous driving. |
| 2026-01-09 | [Text Detoxification in isiXhosa and Yorùbá: A Cross-Lingual Machine Learning Approach for Low-Resource African Languages](http://arxiv.org/abs/2601.05624v1) | Abayomi O. Agbeyangi | Toxic language is one of the major barrier to safe online participation, yet robust mitigation tools are scarce for African languages. This study addresses this critical gap by investigating automatic text detoxification (toxic to neutral rewriting) for two low-resource African languages, isiXhosa and Yorùbá. The work contributes a novel, pragmatic hybrid methodology: a lightweight, interpretable TF-IDF and Logistic Regression model for transparent toxicity detection, and a controlled lexicon- and token-guided rewriting component. A parallel corpus of toxic to neutral rewrites, which captures idiomatic usage, diacritics, and code switching, was developed to train and evaluate the model. The detection component achieved stratified K-fold accuracies of 61-72% (isiXhosa) and 72-86% (Yorùbá), with per-language ROC-AUCs up to 0.88. The rewriting component successfully detoxified all detected toxic sentences while preserving 100% of non-toxic sentences. These results demonstrate that scalable, interpretable machine learning detectors combined with rule-based edits offer a competitive and resource-efficient solution for culturally adaptive safety tooling, setting a new benchmark for low-resource Text Style Transfer (TST) in African languages. |
| 2026-01-09 | [Crisis-Bench: Benchmarking Strategic Ambiguity and Reputation Management in Large Language Models](http://arxiv.org/abs/2601.05570v1) | Cooper Lin, Maohao Ran et al. | Standard safety alignment optimizes Large Language Models (LLMs) for universal helpfulness and honesty, effectively instilling a rigid "Boy Scout" morality. While robust for general-purpose assistants, this one-size-fits-all ethical framework imposes a "transparency tax" on professional domains requiring strategic ambiguity and information withholding, such as public relations, negotiation, and crisis management. To measure this gap between general safety and professional utility, we introduce Crisis-Bench, a multi-agent Partially Observable Markov Decision Process (POMDP) that evaluates LLMs in high-stakes corporate crises. Spanning 80 diverse storylines across 8 industries, Crisis-Bench tasks an LLM-based Public Relations (PR) Agent with navigating a dynamic 7-day corporate crisis simulation while managing strictly separated Private and Public narrative states to enforce rigorous information asymmetry. Unlike traditional benchmarks that rely on static ground truths, we introduce the Adjudicator-Market Loop: a novel evaluation metric where public sentiment is adjudicated and translated into a simulated stock price, creating a realistic economic incentive structure. Our results expose a critical dichotomy: while some models capitulate to ethical concerns, others demonstrate the capacity for Machiavellian, legitimate strategic withholding in order to stabilize the simulated stock price. Crisis-Bench provides the first quantitative framework for assessing "Reputation Management" capabilities, arguing for a shift from rigid moral absolutism to context-aware professional alignment. |
| 2026-01-09 | [ReasonAny: Incorporating Reasoning Capability to Any Model via Simple and Effective Model Merging](http://arxiv.org/abs/2601.05560v1) | Junyao Yang, Chen Qian et al. | Large Reasoning Models (LRMs) with long chain-of-thought reasoning have recently achieved remarkable success. Yet, equipping domain-specialized models with such reasoning capabilities, referred to as "Reasoning + X", remains a significant challenge. While model merging offers a promising training-free solution, existing methods often suffer from a destructive performance collapse: existing methods tend to both weaken reasoning depth and compromise domain-specific utility. Interestingly, we identify a counter-intuitive phenomenon underlying this failure: reasoning ability predominantly resides in parameter regions with low gradient sensitivity, contrary to the common assumption that domain capabilities correspond to high-magnitude parameters. Motivated by this insight, we propose ReasonAny, a novel merging framework that resolves the reasoning-domain performance collapse through Contrastive Gradient Identification. Experiments across safety, biomedicine, and finance domains show that ReasonAny effectively synthesizes "Reasoning + X" capabilities, significantly outperforming state-of-the-art baselines while retaining robust reasoning performance. |

<!-- LATEST_PAPERS_END --> 

---

## 🔑 Key Safety Domains(coming soon)
![LLM Safety Category](/assets/img/image1.png "LLM Safety Category")

**Fig.1**: LLM Safety [[Ma et al., 2025]([arXiv:2502.05206](https://arxiv.org/abs/2502.05206))]

Here we only list the security scenarios involved in the most popular research directions.

- Adversarial Attack
  - white box
  - black box
  - grey box
- Jailbreak Attacks
  - white box
    - gradient-based
  - black box
    - prompt injection
    - role play
    - encodind-based
    - multilingual-based
- Backdoor Attacks 
- DDos Attack
- Privacy Leakage
- System Data Leakage
- Deepfake

---

## 🔖 Dataset & Benchmark
### For Traditional LLM
Please refer to [dataset&benchmark for LLM](./collection/dataset/dataset_for_LLM.md)

### For Advanced LRM
Please refer to [dataset&benchmark for LRM](./collection/dataset/dataset_for_LRM.md)

---

## 📚 Survey
### LRM Related
- Efficient Inference for Large Reasoning Models: A Survey
- A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond
- Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models
- A Survey on Post-training of Large Language Models
- Reasoning Language Models: A Blueprint
- Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models
### LRM Safety Related
- Efficient Inference for Large Reasoning Models: A Survey
---

## 🛠️ Projects & Tools(coming soon)
### Model-Specific Resources(example)
- **DeepSeek-R1 Safety Kit**  
  Official safety evaluation toolkit for DeepSeek-R1 reasoning modules

- **OpenAI o1 Red Teaming Framework**  
  Adversarial testing framework for multi-turn reasoning tasks

### General Tools(coming soon)(example)
- [ReasonGuard](https://github.com/example/reasonguard )  
  Real-time monitoring for reasoning chain anomalies

- [Ethos](https://github.com/example/ethos )  
  Ethical alignment evaluation suite for LRMs

---

## 🤝 Contributing
We welcome contributions! Please:
1. Fork the repository
2. Add resources via pull request
3. Ensure entries follow the format:
   ```markdown
   - [Year] [Paper Title](URL)  
     *Brief description (5-15 words)*
   ```
4. Maintain topical categorization

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ❓ FAQ
**Q: How do I stay updated?**  
A: Watch this repo and check the "Recent Updates" section (coming soon).

**Q: Can I suggest non-academic resources?**  
A: Yes! Industry reports and blog posts are welcome if they provide novel insights.

**Q: How are entries verified?**  
A: All submissions undergo community review for relevance and quality.

---
## 🔗 References

Ma, X., Gao, Y., Wang, Y., Wang, R., Wang, X., Sun, Y., Ding, Y., Xu, H., Chen, Y., Zhao, Y., Huang, H., Li, Y., Zhang, J., Zheng, X., Bai, Y., Wu, Z., Qiu, X., Zhang, J., Li, Y., Sun, J., Wang, C., Gu, J., Wu, B., Chen, S., Zhang, T., Liu, Y., Gong, M., Liu, T., Pan, S., Xie, C., Pang, T., Dong, Y., Jia, R., Zhang, Y., Ma, S., Zhang, X., Gong, N., Xiao, C., Erfani, S., Li, B., Sugiyama, M., Tao, D., Bailey, J., Jiang, Y.-G. (2025). *Safety at Scale: A Comprehensive Survey of Large Model Safety*. arXiv:2502.05206.

---

> *"With great reasoning power comes great responsibility."* - Adapted from [AI Ethics Manifesto]



