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
| 2025-11-13 | [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](http://arxiv.org/abs/2511.10645v1) | Yesheng Liang, Haisheng Chen et al. | Weight-only post-training quantization (PTQ) compresses the weights of Large Language Models (LLMs) into low-precision representations to reduce memory footprint and accelerate inference. However, the presence of outliers in weights and activations often leads to large quantization errors and severe accuracy degradation, especially in recent reasoning LLMs where errors accumulate across long chains of thought. Existing PTQ methods either fail to sufficiently suppress outliers or introduce significant overhead during inference. In this paper, we propose Pairwise Rotation Quantization (ParoQuant), a weight-only PTQ method that combines hardware-efficient and optimizable independent Givens rotations with channel-wise scaling to even out the magnitude across channels and narrow the dynamic range within each quantization group. We further co-design the inference kernel to fully exploit GPU parallelism and keep the rotations and scaling lightweight at runtime. ParoQuant achieves an average 2.4% accuracy improvement over AWQ on reasoning tasks with less than 10% overhead. This paves the way for more efficient and accurate deployment of reasoning LLMs. |
| 2025-11-13 | [Querying Labeled Time Series Data with Scenario Programs](http://arxiv.org/abs/2511.10627v1) | Edward Kim, Devan Shanker et al. | Simulation-based testing has become a crucial complement to road testing for ensuring the safety of cyber physical systems (CPS). As a result, significant research efforts have been directed toward identifying failure scenarios within simulation environments. However, a critical question remains. Are the AV failure scenarios discovered in simulation reproducible on actual systems in the real world? The sim-to-real gap caused by differences between simulated and real sensor data means that failure scenarios identified in simulation might either be artifacts of synthetic sensor data or actual issues that also occur with real sensor data. To address this, an effective approach to validating simulated failure scenarios is to locate occurrences of these scenarios within real-world datasets and verify whether the failure persists on the datasets. To this end, we introduce a formal definition of how labeled time series sensor data can match an abstract scenario, represented as a scenario program using the Scenic probabilistic programming language. We present a querying algorithm that, given a scenario program and a labeled dataset, identifies the subset of data that matches the specified scenario. Our experiment shows that our algorithm is more accurate and orders of magnitude faster in querying scenarios than the state-of-the-art commercial vision large language models, and can scale with the duration of queried time series data. |
| 2025-11-13 | [Safe Planning in Interactive Environments via Iterative Policy Updates and Adversarially Robust Conformal Prediction](http://arxiv.org/abs/2511.10586v1) | Omid Mirzaeedodangeh, Eliot Shekhtman et al. | Safe planning of an autonomous agent in interactive environments -- such as the control of a self-driving vehicle among pedestrians and human-controlled vehicles -- poses a major challenge as the behavior of the environment is unknown and reactive to the behavior of the autonomous agent. This coupling gives rise to interaction-driven distribution shifts where the autonomous agent's control policy may change the environment's behavior, thereby invalidating safety guarantees in existing work. Indeed, recent works have used conformal prediction (CP) to generate distribution-free safety guarantees using observed data of the environment. However, CP's assumption on data exchangeability is violated in interactive settings due to a circular dependency where a control policy update changes the environment's behavior, and vice versa. To address this gap, we propose an iterative framework that robustly maintains safety guarantees across policy updates by quantifying the potential impact of a planned policy update on the environment's behavior. We realize this via adversarially robust CP where we perform a regular CP step in each episode using observed data under the current policy, but then transfer safety guarantees across policy updates by analytically adjusting the CP result to account for distribution shifts. This adjustment is performed based on a policy-to-trajectory sensitivity analysis, resulting in a safe, episodic open-loop planner. We further conduct a contraction analysis of the system providing conditions under which both the CP results and the policy updates are guaranteed to converge. We empirically demonstrate these safety and convergence guarantees on a two-dimensional car-pedestrian case study. To the best of our knowledge, these are the first results that provide valid safety guarantees in such interactive settings. |
| 2025-11-13 | [Evaluating Prompting Strategies with MedGemma for Medical Order Extraction](http://arxiv.org/abs/2511.10583v1) | Abhinand Balachandran, Bavana Durgapraveen et al. | The accurate extraction of medical orders from doctor-patient conversations is a critical task for reducing clinical documentation burdens and ensuring patient safety. This paper details our team submission to the MEDIQA-OE-2025 Shared Task. We investigate the performance of MedGemma, a new domain-specific open-source language model, for structured order extraction. We systematically evaluate three distinct prompting paradigms: a straightforward one-Shot approach, a reasoning-focused ReAct framework, and a multi-step agentic workflow. Our experiments reveal that while more complex frameworks like ReAct and agentic flows are powerful, the simpler one-shot prompting method achieved the highest performance on the official validation set. We posit that on manually annotated transcripts, complex reasoning chains can lead to "overthinking" and introduce noise, making a direct approach more robust and efficient. Our work provides valuable insights into selecting appropriate prompting strategies for clinical information extraction in varied data conditions. |
| 2025-11-13 | [Towards Emotionally Intelligent and Responsible Reinforcement Learning](http://arxiv.org/abs/2511.10573v1) | Garapati Keerthana, Manik Gupta | Personalized decision systems in healthcare and behavioral support often rely on static rule-based or engagement-maximizing heuristics that overlook users' emotional context and ethical constraints. Such approaches risk recommending insensitive or unsafe interventions, especially in domains involving serious mental illness, substance use disorders, or depression. To address this limitation, we propose a Responsible Reinforcement Learning (RRL) framework that integrates emotional and contextual understanding with ethical considerations into the sequential decision-making process. RRL formulates personalization as a Constrained Markov Decision Process (CMDP), where the agent optimizes engagement and adherence while ensuring emotional alignment and ethical safety. We introduce a multi-objective reward function that explicitly balances short-term behavioral engagement with long-term user well-being, and define an emotion-informed state representation that captures fluctuations in emotional readiness, affect, and risk. The proposed architecture can be instantiated with any RL algorithm (e.g., DQN, PPO) augmented with safety constraints or Lagrangian regularization. Conceptually, this framework operationalizes empathy and responsibility within machine learning policy optimization, bridging safe RL, affective computing and responsible AI. We discuss the implications of this approach for human-centric domains such as behavioral health, education, and digital therapeutics, and outline simulation-based validation paths for future empirical work. This paper aims to initiate a methodological conversation about ethically aligned reinforcement learning for emotionally aware and trustworthy personalization systems. |
| 2025-11-13 | [Say It Differently: Linguistic Styles as Jailbreak Vectors](http://arxiv.org/abs/2511.10519v1) | Srikant Panda, Avinash Rai | Large Language Models (LLMs) are commonly evaluated for robustness against paraphrased or semantically equivalent jailbreak prompts, yet little attention has been paid to linguistic variation as an attack surface. In this work, we systematically study how linguistic styles such as fear or curiosity can reframe harmful intent and elicit unsafe responses from aligned models. We construct style-augmented jailbreak benchmark by transforming prompts from 3 standard datasets into 11 distinct linguistic styles using handcrafted templates and LLM-based rewrites, while preserving semantic intent. Evaluating 16 open- and close-source instruction-tuned models, we find that stylistic reframing increases jailbreak success rates by up to +57 percentage points. Styles such as fearful, curious and compassionate are most effective and contextualized rewrites outperform templated variants.   To mitigate this, we introduce a style neutralization preprocessing step using a secondary LLM to strip manipulative stylistic cues from user inputs, significantly reducing jailbreak success rates. Our findings reveal a systemic and scaling-resistant vulnerability overlooked in current safety pipelines. |
| 2025-11-13 | [Formal Verification of Control Lyapunov-Barrier Functions for Safe Stabilization with Bounded Controls](http://arxiv.org/abs/2511.10510v1) | Jun Liu | We present verifiable conditions for synthesizing a single smooth Lyapunov function that certifies both asymptotic stability and safety under bounded controls. These sufficient conditions ensure the strict compatibility of a control barrier function (CBF) and a control Lyapunov function (CLF) on the exact safe set certified by the barrier. An explicit smooth control Lyapunov-barrier function (CLBF) is then constructed via a patching formula that is provably correct by design. Two examples illustrate the computational procedure, showing that the proposed approach is less conservative than sum-of-squares (SOS)-based compatible CBF-CLF designs. |
| 2025-11-13 | [Reasoning About Intent for Ambiguous Requests](http://arxiv.org/abs/2511.10453v1) | Irina Saparina, Mirella Lapata | Large language models often respond to ambiguous requests by implicitly committing to one interpretation. Intent misunderstandings can frustrate users and create safety risks. To address this, we propose generating multiple interpretation-answer pairs in a single structured response to ambiguous requests. Our models are trained with reinforcement learning and customized reward functions using multiple valid answers as supervision. Experiments on conversational question answering and semantic parsing demonstrate that our method achieves higher coverage of valid answers than baseline approaches. Human evaluation confirms that predicted interpretations are highly aligned with their answers. Our approach promotes transparency with explicit interpretations, achieves efficiency by requiring only one generation step, and supports downstream applications through its structured output format. |
| 2025-11-13 | [Improving dependability in robotized bolting operations](http://arxiv.org/abs/2511.10448v1) | Lorenzo Pagliara, Violeta Redondo et al. | Bolting operations are critical in industrial assembly and in the maintenance of scientific facilities, requiring high precision and robustness to faults. Although robotic solutions have the potential to improve operational safety and effectiveness, current systems still lack reliable autonomy and fault management capabilities. To address this gap, we propose a control framework for dependable robotized bolting tasks and instantiate it on a specific robotic system. The system features a control architecture ensuring accurate driving torque control and active compliance throughout the entire operation, enabling safe interaction even under fault conditions. By designing a multimodal human-robot interface (HRI) providing real-time visualization of relevant system information and supporting seamless transitions between automatic and manual control, we improve operator situation awareness and fault detection capabilities. A high-level supervisor (SV) coordinates the execution and manages transitions between control modes, ensuring consistency with the supervisory control (SVC) paradigm, while preserving the human operator's authority. The system is validated in a representative bolting operation involving pipe flange joining, under several fault conditions. The results demonstrate improved fault detection capabilities, enhanced operator situational awareness, and accurate and compliant execution of the bolting operation. However, they also reveal the limitations of relying on a single camera to achieve full situational awareness. |
| 2025-11-13 | [Analogical Structure, Minimal Contextual Cues and Contrastive Distractors: Input Design for Sample-Efficient Linguistic Rule Induction](http://arxiv.org/abs/2511.10441v1) | Chunyang Jiang, Paola Merlo | Large language models achieve strong performance through training on vast datasets. Can analogical paradigm organization enable lightweight models to match this performance with minimal data? We develop a computational approach implementing three cognitive-inspired principles: analogical structure, contrastive learning, and minimal contextual cues. We test this approach with structured completion tasks where models identify correct sentence completions from analogical patterns with contrastive alternatives. Training lightweight models (BERT+CNN, $0.5M$ parameters) on only one hundred structured examples of English causative/inchoative alternations achieves $F1=0.95$, outperforming zero-shot \texttt{GPT-o3} ($F1=0.87$). Ablation studies confirm that analogical organization and contrastive structure improve performance, consistently surpassing randomly shuffled baselines across architectures. Cross-phenomenon validation using unspecified object alternations replicates these efficiency gains, confirming approach robustness. Our results show that analogical paradigm organization enables competitive linguistic rule learning with orders of magnitude less data than conventional approaches require. |
| 2025-11-13 | [LongComp: Long-Tail Compositional Zero-Shot Generalization for Robust Trajectory Prediction](http://arxiv.org/abs/2511.10411v1) | Benjamin Stoler, Jonathan Francis et al. | Methods for trajectory prediction in Autonomous Driving must contend with rare, safety-critical scenarios that make reliance on real-world data collection alone infeasible. To assess robustness under such conditions, we propose new long-tail evaluation settings that repartition datasets to create challenging out-of-distribution (OOD) test sets. We first introduce a safety-informed scenario factorization framework, which disentangles scenarios into discrete ego and social contexts. Building on analogies to compositional zero-shot image-labeling in Computer Vision, we then hold out novel context combinations to construct challenging closed-world and open-world settings. This process induces OOD performance gaps in future motion prediction of 5.0% and 14.7% in closed-world and open-world settings, respectively, relative to in-distribution performance for a state-of-the-art baseline. To improve generalization, we extend task-modular gating networks to operate within trajectory prediction models, and develop an auxiliary, difficulty-prediction head to refine internal representations. Our strategies jointly reduce the OOD performance gaps to 2.8% and 11.5% in the two settings, respectively, while still improving in-distribution performance. |
| 2025-11-13 | [Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance](http://arxiv.org/abs/2511.10400v1) | Lifan Zheng, Jiawei Chen et al. | Ensuring the reliability of agent architectures and effectively identifying problematic agents when failures occur are crucial challenges in multi-agent systems (MAS). Advances in large language models (LLMs) have established LLM-based agents as a major branch of MAS, enabling major breakthroughs in complex problem solving and world modeling. However, the reliability implications of this shift remain largely unexplored. i.e., whether substituting traditional agents with LLM-based agents can effectively enhance the reliability of MAS. In this work, we investigate and quantify the reliability of LLM-based agents from the perspective of Byzantine fault tolerance. We observe that LLM-based agents demonstrate stronger skepticism when processing erroneous message flows, a characteristic that enables them to outperform traditional agents across different topological structures. Motivated by the results of the pilot experiment, we design CP-WBFT, a confidence probe-based weighted Byzantine Fault Tolerant consensus mechanism to enhance the stability of MAS with different topologies. It capitalizes on the intrinsic reflective and discriminative capabilities of LLMs by employing a probe-based, weighted information flow transmission method to improve the reliability of LLM-based agents. Extensive experiments demonstrate that CP-WBFT achieves superior performance across diverse network topologies under extreme Byzantine conditions (85.7\% fault rate). Notably, our approach surpasses traditional methods by attaining remarkable accuracy on various topologies and maintaining strong reliability in both mathematical reasoning and safety assessment tasks. |
| 2025-11-13 | [Towards Comprehensive Sampling of SMT Solutions](http://arxiv.org/abs/2511.10326v1) | Shuangyu Lyu, Chuan Luo et al. | This work focuses on effectively generating diverse solutions for satisfiability modulo theories (SMT) formulas, targeting the theories of bit-vectors, arrays, and uninterpreted functions, which is a critical task in software and hardware testing. Generating diverse SMT solutions helps uncover faults and detect safety violations during the verification and testing process, resulting in the SMT sampling problem, i.e., constructing a small number of solutions while achieving comprehensive coverage of the constraint space. While high coverage is crucial for exploring system behaviors, reducing the number of solutions is of great importance, as excessive solutions increase testing time and resource usage, undermining efficiency. In this work, we introduce PanSampler, a novel SMT sampler that achieves high coverage with a small number of solutions. It incorporates three novel techniques, i.e., diversity-aware SMT algorithm, abstract syntax tree (AST)-guided scoring function and post-sampling optimization technology, enhancing its practical performance. It iteratively samples solutions, evaluates candidates, and employs local search to refine solutions, ensuring high coverage with a small number of samples. Extensive experiments on practical benchmarks demonstrate that PanSampler exhibits a significantly stronger capability to reach high target coverage, while requiring fewer solutions than current samplers to achieve the same coverage level. Furthermore, our empirical evaluation on practical subjects, which are collected from real-world software systems, shows that PanSampler achieves higher fault detection capability and reduces the number of required test cases from 32.6\% to 76.4\% to reach the same fault detection effectiveness, leading to a substantial improvement in testing efficiency. PanSampler advances SMT sampling, reducing the cost of software testing and hardware verification. |
| 2025-11-13 | [Reconfigurable Airspace: Synergizing Movable Antenna and Intelligent Surface for Low-Altitude ISAC Networks](http://arxiv.org/abs/2511.10310v1) | Honghao Wang, Qingqing Wu et al. | Low-altitude unmanned aerial vehicle (UAV) networks are integral to future 6G integrated sensing and communication (ISAC) systems. However, their deployment is hindered by challenges stemming from high mobility of UAVs, complex propagation environments, and the inherent trade-offs between coexisting sensing and communication functions. This article proposes a novel framework that leverages movable antennas (MAs) and intelligent reflecting surfaces (IRSs) as dual enablers to overcome these limitations. MAs, through active transceiver reconfiguration, and IRSs, via passive channel reconstruction, can work in synergy to significantly enhance system performance. Our analysis first elaborates on the fundamental gains offered by MAs and IRSs, and provides simulation results that validate the immense potential of the MA-IRS-enabled ISAC architecture. Two core UAV deployment scenarios are then investigated: (i) UAVs as ISAC users, where we focus on achieving high-precision tracking and aerial safety, and (ii) UAVs as aerial network nodes, where we address robust design and complex coupled resource optimization. Finally, key technical challenges and research opportunities are identified and analyzed for each scenario, charting a clear course for the future design of advanced low-altitude ISAC networks. |
| 2025-11-13 | [Revisiting Evaluation of Deep Neural Networks for Pedestrian Detection](http://arxiv.org/abs/2511.10308v1) | Patrick Feifel, Benedikt Franke et al. | Reliable pedestrian detection represents a crucial step towards automated driving systems. However, the current performance benchmarks exhibit weaknesses. The currently applied metrics for various subsets of a validation dataset prohibit a realistic performance evaluation of a DNN for pedestrian detection. As image segmentation supplies fine-grained information about a street scene, it can serve as a starting point to automatically distinguish between different types of errors during the evaluation of a pedestrian detector. In this work, eight different error categories for pedestrian detection are proposed and new metrics are proposed for performance comparison along these error categories. We use the new metrics to compare various backbones for a simplified version of the APD, and show a more fine-grained and robust way to compare models with each other especially in terms of safety-critical performance. We achieve SOTA on CityPersons-reasonable (without extra training data) by using a rather simple architecture. |
| 2025-11-13 | [OutSafe-Bench: A Benchmark for Multimodal Offensive Content Detection in Large Language Models](http://arxiv.org/abs/2511.10287v1) | Yuping Yan, Yuhan Xie et al. | Since Multimodal Large Language Models (MLLMs) are increasingly being integrated into everyday tools and intelligent agents, growing concerns have arisen regarding their possible output of unsafe contents, ranging from toxic language and biased imagery to privacy violations and harmful misinformation. Current safety benchmarks remain highly limited in both modality coverage and performance evaluations, often neglecting the extensive landscape of content safety. In this work, we introduce OutSafe-Bench, the first most comprehensive content safety evaluation test suite designed for the multimodal era. OutSafe-Bench includes a large-scale dataset that spans four modalities, featuring over 18,000 bilingual (Chinese and English) text prompts, 4,500 images, 450 audio clips and 450 videos, all systematically annotated across nine critical content risk categories. In addition to the dataset, we introduce a Multidimensional Cross Risk Score (MCRS), a novel metric designed to model and assess overlapping and correlated content risks across different categories. To ensure fair and robust evaluation, we propose FairScore, an explainable automated multi-reviewer weighted aggregation framework. FairScore selects top-performing models as adaptive juries, thereby mitigating biases from single-model judgments and enhancing overall evaluation reliability. Our evaluation of nine state-of-the-art MLLMs reveals persistent and substantial safety vulnerabilities, underscoring the pressing need for robust safeguards in MLLMs. |
| 2025-11-13 | [MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](http://arxiv.org/abs/2511.10262v1) | He Zhang, Wenqian Cui et al. | Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions and conversational features, neglecting the complexities of multi-round communication and critical capabilities such as instruction following and safety. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark that segments continuous full-duplex dialogues into discrete turns, enabling comprehensive, turn-by-turn evaluation of FD-SLMs across dialogue quality, conversational dynamics, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our proposed benchmark. The benchmark and code will be available in the future. |
| 2025-11-13 | [Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard](http://arxiv.org/abs/2511.10222v1) | Yudong Yang, Xuezhen Zhang et al. | Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful. |
| 2025-11-13 | [VISTA: A Vision and Intent-Aware Social Attention Framework for Multi-Agent Trajectory Prediction](http://arxiv.org/abs/2511.10203v1) | Stephane Da Silva Martins, Emanuel Aldea et al. | Multi-agent trajectory prediction is crucial for autonomous systems operating in dense, interactive environments. Existing methods often fail to jointly capture agents' long-term goals and their fine-grained social interactions, which leads to unrealistic multi-agent futures. We propose VISTA, a recursive goal-conditioned transformer for multi-agent trajectory forecasting. VISTA combines (i) a cross-attention fusion module that integrates long-horizon intent with past motion, (ii) a social-token attention mechanism for flexible interaction modeling across agents, and (iii) pairwise attention maps that make social influence patterns interpretable at inference time. Our model turns single-agent goal-conditioned prediction into a coherent multi-agent forecasting framework. Beyond standard displacement metrics, we evaluate trajectory collision rates as a measure of joint realism. On the high-density MADRAS benchmark and on SDD, VISTA achieves state-of-the-art accuracy and substantially fewer collisions. On MADRAS, it reduces the average collision rate of strong baselines from 2.14 to 0.03 percent, and on SDD it attains zero collisions while improving ADE, FDE, and minFDE. These results show that VISTA generates socially compliant, goal-aware, and interpretable trajectories, making it promising for safety-critical autonomous systems. |
| 2025-11-13 | [EffiReason-Bench: A Unified Benchmark for Evaluating and Advancing Efficient Reasoning in Large Language Models](http://arxiv.org/abs/2511.10201v1) | Junquan Huang, Haotian Wu et al. | Large language models (LLMs) with Chain-of-Thought (CoT) prompting achieve strong reasoning but often produce unnecessarily long explanations, increasing cost and sometimes reducing accuracy. Fair comparison of efficiency-oriented approaches is hindered by fragmented evaluation practices. We introduce EffiReason-Bench, a unified benchmark for rigorous cross-paradigm evaluation of efficient reasoning methods across three categories: Reasoning Blueprints, Dynamic Execution, and Post-hoc Refinement. To enable step-by-step evaluation, we construct verified CoT annotations for CommonsenseQA and LogiQA via a pipeline that enforces standardized reasoning structures, comprehensive option-wise analysis, and human verification. We evaluate 7 methods across 6 open-source LLMs (1B-70B) on 4 datasets spanning mathematics, commonsense, and logic, and propose the E3-Score, a principled metric inspired by economic trade-off modeling that provides smooth, stable evaluation without discontinuities or heavy reliance on heuristics. Experiments show that no single method universally dominates; optimal strategies depend on backbone scale, task complexity, and architecture. |

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



