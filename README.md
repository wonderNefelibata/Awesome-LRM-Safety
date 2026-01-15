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
| 2026-01-14 | [The Promptware Kill Chain: How Prompt Injections Gradually Evolved Into a Multi-Step Malware](http://arxiv.org/abs/2601.09625v1) | Ben Nassi, Bruce Schneier et al. | The rapid adoption of large language model (LLM)-based systems -- from chatbots to autonomous agents capable of executing code and financial transactions -- has created a new attack surface that existing security frameworks inadequately address. The dominant framing of these threats as "prompt injection" -- a catch-all phrase for security failures in LLM-based systems -- obscures a more complex reality: Attacks on LLM-based systems increasingly involve multi-step sequences that mirror traditional malware campaigns. In this paper, we propose that attacks targeting LLM-based applications constitute a distinct class of malware, which we term \textit{promptware}, and introduce a five-step kill chain model for analyzing these threats. The framework comprises Initial Access (prompt injection), Privilege Escalation (jailbreaking), Persistence (memory and retrieval poisoning), Lateral Movement (cross-system and cross-user propagation), and Actions on Objective (ranging from data exfiltration to unauthorized transactions). By mapping recent attacks to this structure, we demonstrate that LLM-related attacks follow systematic sequences analogous to traditional malware campaigns. The promptware kill chain offers security practitioners a structured methodology for threat modeling and provides a common vocabulary for researchers across AI safety and cybersecurity to address a rapidly evolving threat landscape. |
| 2026-01-14 | [CogRail: Benchmarking VLMs in Cognitive Intrusion Perception for Intelligent Railway Transportation Systems](http://arxiv.org/abs/2601.09613v1) | Yonglin Tian, Qiyao Zhang et al. | Accurate and early perception of potential intrusion targets is essential for ensuring the safety of railway transportation systems. However, most existing systems focus narrowly on object classification within fixed visual scopes and apply rule-based heuristics to determine intrusion status, often overlooking targets that pose latent intrusion risks. Anticipating such risks requires the cognition of spatial context and temporal dynamics for the object of interest (OOI), which presents challenges for conventional visual models. To facilitate deep intrusion perception, we introduce a novel benchmark, CogRail, which integrates curated open-source datasets with cognitively driven question-answer annotations to support spatio-temporal reasoning and prediction. Building upon this benchmark, we conduct a systematic evaluation of state-of-the-art visual-language models (VLMs) using multimodal prompts to identify their strengths and limitations in this domain. Furthermore, we fine-tune VLMs for better performance and propose a joint fine-tuning framework that integrates three core tasks, position perception, movement prediction, and threat analysis, facilitating effective adaptation of general-purpose foundation models into specialized models tailored for cognitive intrusion perception. Extensive experiments reveal that current large-scale multimodal models struggle with the complex spatial-temporal reasoning required by the cognitive intrusion perception task, underscoring the limitations of existing foundation models in this safety-critical domain. In contrast, our proposed joint fine-tuning framework significantly enhances model performance by enabling targeted adaptation to domain-specific reasoning demands, highlighting the advantages of structured multi-task learning in improving both accuracy and interpretability. Code will be available at https://github.com/Hub-Tian/CogRail. |
| 2026-01-14 | [Information Access of the Oppressed: A Problem-Posing Framework for Envisioning Emancipatory Information Access Platforms](http://arxiv.org/abs/2601.09600v1) | Bhaskar Mitra, Nicola Neophytou et al. | Online information access (IA) platforms are targets of authoritarian capture. These concerns are particularly serious and urgent today in light of the rising levels of democratic erosion worldwide, the emerging capabilities of generative AI technologies such as AI persuasion, and the increasing concentration of economic and political power in the hands of Big Tech. This raises the question of what alternative IA infrastructure we must reimagine and build to mitigate the risks of authoritarian capture of our information ecosystems. We explore this question through the lens of Paulo Freire's theories of emancipatory pedagogy. Freire's theories provide a radically different lens for exploring IA's sociotechnical concerns relative to the current dominating frames of fairness, accountability, confidentiality, transparency, and safety. We make explicit, with the intention to challenge, the dichotomy of how we relate to technology as either technologists (who envision and build technology) and its users. We posit that this mirrors the teacher-student relationship in Freire's analysis. By extending Freire's analysis to IA, we challenge the notion that it is the burden of the (altruistic) technologists to come up with interventions to mitigate the risks that emerging technologies pose to marginalized communities. Instead, we advocate that the first task for the technologists is to pose these as problems to the marginalized communities, to encourage them to make and unmake the technology as part of their material struggle against oppression. Their second task is to redesign our online technology stacks to structurally expose spaces for community members to co-opt and co-construct the technology in aid of their emancipatory struggles. We operationalize Freire's theories to develop a problem-posing framework for envisioning emancipatory IA platforms of the future. |
| 2026-01-14 | [GlovEgo-HOI: Bridging the Synthetic-to-Real Gap for Industrial Egocentric Human-Object Interaction Detection](http://arxiv.org/abs/2601.09528v1) | Alfio Spoto, Rosario Leonardi et al. | Egocentric Human-Object Interaction (EHOI) analysis is crucial for industrial safety, yet the development of robust models is hindered by the scarcity of annotated domain-specific data. We address this challenge by introducing a data generation framework that combines synthetic data with a diffusion-based process to augment real-world images with realistic Personal Protective Equipment (PPE). We present GlovEgo-HOI, a new benchmark dataset for industrial EHOI, and GlovEgo-Net, a model integrating Glove-Head and Keypoint- Head modules to leverage hand pose information for enhanced interaction detection. Extensive experiments demonstrate the effectiveness of the proposed data generation framework and GlovEgo-Net. To foster further research, we release the GlovEgo-HOI dataset, augmentation pipeline, and pre-trained models at: GitHub project. |
| 2026-01-14 | [ReflexDiffusion: Reflection-Enhanced Trajectory Planning for High-lateral-acceleration Scenarios in Autonomous Driving](http://arxiv.org/abs/2601.09377v1) | Xuemei Yao, Xiao Yang et al. | Generating safe and reliable trajectories for autonomous vehicles in long-tail scenarios remains a significant challenge, particularly for high-lateral-acceleration maneuvers such as sharp turns, which represent critical safety situations. Existing trajectory planners exhibit systematic failures in these scenarios due to data imbalance. This results in insufficient modelling of vehicle dynamics, road geometry, and environmental constraints in high-risk situations, leading to suboptimal or unsafe trajectory prediction when vehicles operate near their physical limits. In this paper, we introduce ReflexDiffusion, a novel inference-stage framework that enhances diffusion-based trajectory planners through reflective adjustment. Our method introduces a gradient-based adjustment mechanism during the iterative denoising process: after each standard trajectory update, we compute the gradient between the conditional and unconditional noise predictions to explicitly amplify critical conditioning signals, including road curvature and lateral vehicle dynamics. This amplification enforces strict adherence to physical constraints, particularly improving stability during high-lateral-acceleration maneuvers where precise vehicle-road interaction is paramount. Evaluated on the nuPlan Test14-hard benchmark, ReflexDiffusion achieves a 14.1% improvement in driving score for high-lateral-acceleration scenarios over the state-of-the-art (SOTA) methods. This demonstrates that inference-time trajectory optimization can effectively compensate for training data sparsity by dynamically reinforcing safety-critical constraints near handling limits. The framework's architecture-agnostic design enables direct deployment to existing diffusion-based planners, offering a practical solution for improving autonomous vehicle safety in challenging driving conditions. |
| 2026-01-14 | [Monte-Carlo Tree Search with Neural Network Guidance for Lane-Free Autonomous Driving](http://arxiv.org/abs/2601.09353v1) | Ioannis Peridis, Dimitrios Troullinos et al. | Lane-free traffic environments allow vehicles to better harness the lateral capacity of the road without being restricted to lane-keeping, thereby increasing the traffic flow rates. As such, we have a distinct and more challenging setting for autonomous driving. In this work, we consider a Monte-Carlo Tree Search (MCTS) planning approach for single-agent autonomous driving in lane-free traffic, where the associated Markov Decision Process we formulate is influenced from existing approaches tied to reinforcement learning frameworks. In addition, MCTS is equipped with a pre-trained neural network (NN) that guides the selection phase. This procedure incorporates the predictive capabilities of NNs for a more informed tree search process under computational constraints. In our experimental evaluation, we consider metrics that address both safety (through collision rates) and efficacy (through measured speed). Then, we examine: (a) the influence of isotropic state information for vehicles in a lane-free environment, resulting in nudging behaviour--vehicles' policy reacts due to the presence of faster tailing ones, (b) the acceleration of performance for the NN-guided variant of MCTS, and (c) the trade-off between computational resources and solution quality. |
| 2026-01-14 | [SpatialJB: How Text Distribution Art Becomes the "Jailbreak Key" for LLM Guardrails](http://arxiv.org/abs/2601.09321v1) | Zhiyi Mou, Jingyuan Yang et al. | While Large Language Models (LLMs) have powerful capabilities, they remain vulnerable to jailbreak attacks, which is a critical barrier to their safe web real-time application. Current commercial LLM providers deploy output guardrails to filter harmful outputs, yet these defenses are not impenetrable. Due to LLMs' reliance on autoregressive, token-by-token inference, their semantic representations lack robustness to spatially structured perturbations, such as redistributing tokens across different rows, columns, or diagonals. Exploiting the Transformer's spatial weakness, we propose SpatialJB to disrupt the model's output generation process, allowing harmful content to bypass guardrails without detection. Comprehensive experiments conducted on leading LLMs get nearly 100% ASR, demonstrating the high effectiveness of SpatialJB. Even after adding advanced output guardrails, like the OpenAI Moderation API, SpatialJB consistently maintains a success rate exceeding 75%, outperforming current jailbreak techniques by a significant margin. The proposal of SpatialJB exposes a key weakness in current guardrails and emphasizes the importance of spatial semantics, offering new insights to advance LLM safety research. To prevent potential misuse, we also present baseline defense strategies against SpatialJB and evaluate their effectiveness in mitigating such attacks. The code for the attack, baseline defenses, and a demo are available at https://anonymous.4open.science/r/SpatialJailbreak-8E63. |
| 2026-01-14 | [STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models](http://arxiv.org/abs/2601.09281v1) | Jingjing Zhou, Gaoxiang Cong et al. | Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs. |
| 2026-01-14 | [Vision-Conditioned Variational Bayesian Last Layer Dynamics Models](http://arxiv.org/abs/2601.09178v1) | Paul Brunzema, Thomas Lew et al. | Agile control of robotic systems often requires anticipating how the environment affects system behavior. For example, a driver must perceive the road ahead to anticipate available friction and plan actions accordingly. Achieving such proactive adaptation within autonomous frameworks remains a challenge, particularly under rapidly changing conditions. Traditional modeling approaches often struggle to capture abrupt variations in system behavior, while adaptive methods are inherently reactive and may adapt too late to ensure safety. We propose a vision-conditioned variational Bayesian last-layer dynamics model that leverages visual context to anticipate changes in the environment. The model first learns nominal vehicle dynamics and is then fine-tuned with feature-wise affine transformations of latent features, enabling context-aware dynamics prediction. The resulting model is integrated into an optimal controller for vehicle racing. We validate our method on a Lexus LC500 racing through water puddles. With vision-conditioning, the system completed all 12 attempted laps under varying conditions. In contrast, all baselines without visual context consistently lost control, demonstrating the importance of proactive dynamics adaptation in high-performance applications. |
| 2026-01-14 | [Geometric Stability: The Missing Axis of Representations](http://arxiv.org/abs/2601.09173v1) | Prashant C. Raju | Analysis of learned representations has a blind spot: it focuses on $similarity$, measuring how closely embeddings align with external references, but similarity reveals only what is represented, not whether that structure is robust. We introduce $geometric$ $stability$, a distinct dimension that quantifies how reliably representational geometry holds under perturbation, and present $Shesha$, a framework for measuring it. Across 2,463 configurations in seven domains, we show that stability and similarity are empirically uncorrelated ($ρ\approx 0.01$) and mechanistically distinct: similarity metrics collapse after removing the top principal components, while stability retains sensitivity to fine-grained manifold structure. This distinction yields actionable insights: for safety monitoring, stability acts as a functional geometric canary, detecting structural drift nearly 2$\times$ more sensitively than CKA while filtering out the non-functional noise that triggers false alarms in rigid distance metrics; for controllability, supervised stability predicts linear steerability ($ρ= 0.89$-$0.96$); for model selection, stability dissociates from transferability, revealing a geometric tax that transfer optimization incurs. Beyond machine learning, stability predicts CRISPR perturbation coherence and neural-behavioral coupling. By quantifying $how$ $reliably$ systems maintain structure, geometric stability provides a necessary complement to similarity for auditing representations across biological and computational systems. |
| 2026-01-14 | [SafePlanner: Testing Safety of the Automated Driving System Plan Model](http://arxiv.org/abs/2601.09171v1) | Dohyun Kim, Sanggu Han et al. | In this work, we present SafePlanner, a systematic testing framework for identifying safety-critical flaws in the Plan model of Automated Driving Systems (ADS). SafePlanner targets two core challenges: generating structurally meaningful test scenarios and detecting hazardous planning behaviors. To maximize coverage, SafePlanner performs a structural analysis of the Plan model implementation - specifically, its scene-transition logic and hierarchical control flow - and uses this insight to extract feasible scene transitions from code. It then composes test scenarios by combining these transitions with non-player vehicle (NPC) behaviors. Guided fuzzing is applied to explore the behavioral space of the Plan model under these scenarios. We evaluate SafePlanner on Baidu Apollo, a production-grade level 4 ADS. It generates 20635 test cases and detects 520 hazardous behaviors, grouped into 15 root causes through manual analysis. For four of these, we applied patches based on our analysis; the issues disappeared, and no apparent side effects were observed. SafePlanner achieves 83.63 percent function and 63.22 percent decision coverage on the Plan model, outperforming baselines in both bug discovery and efficiency. |
| 2026-01-14 | [Multi-Teacher Ensemble Distillation: A Mathematical Framework for Probability-Domain Knowledge Aggregation](http://arxiv.org/abs/2601.09165v1) | Aaron R. Flouro, Shawn P. Chadwick | Building on the probability-domain distillation framework of Sparse-KD, we develop an axiomatic, operator-theoretic framework for multi-teacher ensemble knowledge distillation. Rather than prescribing a specific aggregation formula, we define five core axioms governing valid knowledge aggregation operators, encompassing convexity, positivity, continuity, weight monotonicity, and temperature coherence. We prove the existence and non-uniqueness of operator families satisfying these axioms, establishing that multiple distinct aggregation mechanisms conform to the same foundational principles.   Within this framework, we establish operator-agnostic guarantees showing that multi-teacher aggregation reduces both stochastic variance and systematic supervisory bias under heterogeneous teachers, while providing Jensen-type bounds, log-loss guarantees, and safety attenuation properties. For aggregation operators linear in teacher weights, we further establish classical ensemble variance-reduction results under standard independence assumptions, with extensions to correlated-error regimes. The framework provides theoretical grounding for multi-teacher distillation from diverse frontier models while admitting multiple valid implementation strategies. |
| 2026-01-14 | [From Snow to Rain: Evaluating Robustness, Calibration, and Complexity of Model-Based Robust Training](http://arxiv.org/abs/2601.09153v1) | Josué Martínez-Martínez, Olivia Brown et al. | Robustness to natural corruptions remains a critical challenge for reliable deep learning, particularly in safety-sensitive domains. We study a family of model-based training approaches that leverage a learned nuisance variation model to generate realistic corruptions, as well as new hybrid strategies that combine random coverage with adversarial refinement in nuisance space. Using the Challenging Unreal and Real Environments for Traffic Sign Recognition dataset (CURE-TSR), with Snow and Rain corruptions, we evaluate accuracy, calibration, and training complexity across corruption severities. Our results show that model-based methods consistently outperform baselines Vanilla, Adversarial Training, and AugMix baselines, with model-based adversarial training providing the strongest robustness under across all corruptions but at the expense of higher computation and model-based data augmentation achieving comparable robustness with $T$ less computational complexity without incurring a statistically significant drop in performance. These findings highlight the importance of learned nuisance models for capturing natural variability, and suggest a promising path toward more resilient and calibrated models under challenging conditions. |
| 2026-01-14 | [Identity-Robust Language Model Generation via Content Integrity Preservation](http://arxiv.org/abs/2601.09141v1) | Miao Zhang, Kelly Chen et al. | Large Language Model (LLM) outputs often vary across user sociodemographic attributes, leading to disparities in factual accuracy, utility, and safety, even for objective questions where demographic information is irrelevant. Unlike prior work on stereotypical or representational bias, this paper studies identity-dependent degradation of core response quality. We show empirically that such degradation arises from biased generation behavior, despite factual knowledge being robustly encoded across identities. Motivated by this mismatch, we propose a lightweight, training-free framework for identity-robust generation that selectively neutralizes non-critical identity information while preserving semantically essential attributes, thus maintaining output content integrity. Experiments across four benchmarks and 18 sociodemographic identities demonstrate an average 77% reduction in identity-dependent bias compared to vanilla prompting and a 45% reduction relative to prompt-based defenses. Our work addresses a critical gap in mitigating the impact of user identity cues in prompts on core generation quality. |
| 2026-01-14 | [SkinFlow: Efficient Information Transmission for Open Dermatological Diagnosis via Dynamic Visual Encoding and Staged RL](http://arxiv.org/abs/2601.09136v1) | Lijun Liu, Linwei Chen et al. | General-purpose Large Vision-Language Models (LVLMs), despite their massive scale, often falter in dermatology due to "diffuse attention" - the inability to disentangle subtle pathological lesions from background noise. In this paper, we challenge the assumption that parameter scaling is the only path to medical precision. We introduce SkinFlow, a framework that treats diagnosis as an optimization of visual information transmission efficiency. Our approach utilizes a Virtual-Width Dynamic Vision Encoder (DVE) to "unfold" complex pathological manifolds without physical parameter expansion, coupled with a two-stage Reinforcement Learning strategy. This strategy sequentially aligns explicit medical descriptions (Stage I) and reconstructs implicit diagnostic textures (Stage II) within a constrained semantic space. Furthermore, we propose a clinically grounded evaluation protocol that prioritizes diagnostic safety and hierarchical relevance over rigid label matching. Empirical results are compelling: our 7B model establishes a new state-of-the-art on the Fitzpatrick17k benchmark, achieving a +12.06% gain in Top-1 accuracy and a +28.57% boost in Top-6 accuracy over the massive general-purpose models (e.g., Qwen3VL-235B and GPT-5.2). These findings demonstrate that optimizing geometric capacity and information flow yields superior diagnostic reasoning compared to raw parameter scaling. |
| 2026-01-14 | [Seeking Human Security Consensus: A Unified Value Scale for Generative AI Value Safety](http://arxiv.org/abs/2601.09112v1) | Ying He, Baiyang Li et al. | The rapid development of generative AI has brought value- and ethics-related risks to the forefront, making value safety a critical concern while a unified consensus remains lacking. In this work, we propose an internationally inclusive and resilient unified value framework, the GenAI Value Safety Scale (GVS-Scale): Grounded in a lifecycle-oriented perspective, we develop a taxonomy of GenAI value safety risks and construct the GenAI Value Safety Incident Repository (GVSIR), and further derive the GVS-Scale through grounded theory and operationalize it via the GenAI Value Safety Benchmark (GVS-Bench). Experiments on mainstream text generation models reveal substantial variation in value safety performance across models and value categories, indicating uneven and fragmented value alignment in current systems. Our findings highlight the importance of establishing shared safety foundations through dialogue and advancing technical safety mechanisms beyond reactive constraints toward more flexible approaches. Data and evaluation guidelines are available at https://github.com/acl2026/GVS-Bench. This paper includes examples that may be offensive or harmful. |
| 2026-01-14 | [AviationLMM: A Large Multimodal Foundation Model for Civil Aviation](http://arxiv.org/abs/2601.09105v1) | Wenbin Li, Jingling Wu et al. | Civil aviation is a cornerstone of global transportation and commerce, and ensuring its safety, efficiency and customer satisfaction is paramount. Yet conventional Artificial Intelligence (AI) solutions in aviation remain siloed and narrow, focusing on isolated tasks or single modalities. They struggle to integrate heterogeneous data such as voice communications, radar tracks, sensor streams and textual reports, which limits situational awareness, adaptability, and real-time decision support. This paper introduces the vision of AviationLMM, a Large Multimodal foundation Model for civil aviation, designed to unify the heterogeneous data streams of civil aviation and enable understanding, reasoning, generation and agentic applications. We firstly identify the gaps between existing AI solutions and requirements. Secondly, we describe the model architecture that ingests multimodal inputs such as air-ground voice, surveillance, on-board telemetry, video and structured texts, and performs cross-modal alignment and fusion, and produces flexible outputs ranging from situation summaries and risk alerts to predictive diagnostics and multimodal incident reconstructions. In order to fully realize this vision, we identify key research opportunities to address, including data acquisition, alignment and fusion, pretraining, reasoning, trustworthiness, privacy, robustness to missing modalities, and synthetic scenario generation. By articulating the design and challenges of AviationLMM, we aim to boost the civil aviation foundation model progress and catalyze coordinated research efforts toward an integrated, trustworthy and privacy-preserving aviation AI ecosystem. |
| 2026-01-13 | [PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm](http://arxiv.org/abs/2601.08951v1) | Jing-Jing Li, Joel Mire et al. | Current AI safety frameworks, which often treat harmfulness as binary, lack the flexibility to handle borderline cases where humans meaningfully disagree. To build more pluralistic systems, it is essential to move beyond consensus and instead understand where and why disagreements arise. We introduce PluriHarms, a benchmark designed to systematically study human harm judgments across two key dimensions -- the harm axis (benign to harmful) and the agreement axis (agreement to disagreement). Our scalable framework generates prompts that capture diverse AI harms and human values while targeting cases with high disagreement rates, validated by human data. The benchmark includes 150 prompts with 15,000 ratings from 100 human annotators, enriched with demographic and psychological traits and prompt-level features of harmful actions, effects, and values. Our analyses show that prompts that relate to imminent risks and tangible harms amplify perceived harmfulness, while annotator traits (e.g., toxicity experience, education) and their interactions with prompt content explain systematic disagreement. We benchmark AI safety models and alignment methods on PluriHarms, finding that while personalization significantly improves prediction of human harm judgments, considerable room remains for future progress. By explicitly targeting value diversity and disagreement, our work provides a principled benchmark for moving beyond "one-size-fits-all" safety toward pluralistically safe AI. |
| 2026-01-13 | [MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm](http://arxiv.org/abs/2601.08800v1) | Bowen Zhou, Jinrui Jia et al. | The Mixture of Experts (MoE) models are emerging as the latest paradigm for Large Language Models (LLMs). However, due to memory constraints, MoE models with billions or even trillions of parameters can only be deployed in multi-GPU or even multi-node & multi-GPU based serving systems. Thus, communication has became a major bottleneck in distributed serving systems, especially inter-node communication. Contemporary distributed MoE models are primarily implemented using all-reduce (AR) based tensor parallelism (TP) and all-to-all (A2A) based expert parallelism (EP). However, TP generally exhibits low inter-node efficiency and is thus confined to high-speed intra-node bandwidth. In contrast, EP tends to suffer from load imbalance, especially when the parallel degree is high.   In this work, we introduce MixServe, a novel automatic distributed serving system for efficient deployment of MoE models by a novel TP-EP hybrid parallelism based on fused AR-A2A communication algorithm. MixServe begins by evaluating the communication overhead associated with various parallel strategies, taking into account the model hyperparameters and the configurations of network and hardware resources, and then automatically selects the most efficient parallel strategy. Then, we propose the TP-EP hybrid parallelism based on fused AR-A2A communication algorithm that overlaps intra-node AR communication and inter-node A2A communication. Extensive experiments on DeepSeek-R1 and Qwen3 models demonstrate that MixServe achieves superior inference performance, with 1.08~3.80x acceleration in time to first token (TTFT), 1.03~1.66x acceleration in inter-token latency (ITL), and 5.2%~50.3% throughput improvement compared to existing approaches. |
| 2026-01-13 | [TerraFormer: Automated Infrastructure-as-Code with LLMs Fine-Tuned via Policy-Guided Verifier Feedback](http://arxiv.org/abs/2601.08734v1) | Prithwish Jana, Sam Davidson et al. | Automating Infrastructure-as-Code (IaC) is challenging, and large language models (LLMs) often produce incorrect configurations from natural language (NL). We present TerraFormer, a neuro-symbolic framework for IaC generation and mutation that combines supervised fine-tuning with verifier-guided reinforcement learning, using formal verification tools to provide feedback on syntax, deployability, and policy compliance. We curate two large, high-quality NL-to-IaC datasets, TF-Gen (152k instances) and TF-Mutn (52k instances), via multi-stage verification and iterative LLM self-correction. Evaluations against 17 state-of-the-art LLMs, including ~50x larger models like Sonnet 3.7, DeepSeek-R1, and GPT-4.1, show that TerraFormer improves correctness over its base LLM by 15.94% on IaC-Eval, 11.65% on TF-Gen (Test), and 19.60% on TF-Mutn (Test). It outperforms larger models on both TF-Gen (Test) and TF-Mutn (Test), ranks third on IaC-Eval, and achieves top best-practices and security compliance. |

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



