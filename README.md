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
| 2025-05-26 | [StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs](http://arxiv.org/abs/2505.20139v1) | Jialin Yang, Dongfu Jiang et al. | As Large Language Models (LLMs) become integral to software development workflows, their ability to generate structured outputs has become critically important. We introduce StructEval, a comprehensive benchmark for evaluating LLMs' capabilities in producing both non-renderable (JSON, YAML, CSV) and renderable (HTML, React, SVG) structured formats. Unlike prior benchmarks, StructEval systematically evaluates structural fidelity across diverse formats through two paradigms: 1) generation tasks, producing structured output from natural language prompts, and 2) conversion tasks, translating between structured formats. Our benchmark encompasses 18 formats and 44 types of task, with novel metrics for format adherence and structural correctness. Results reveal significant performance gaps, even state-of-the-art models like o1-mini achieve only 75.58 average score, with open-source alternatives lagging approximately 10 points behind. We find generation tasks more challenging than conversion tasks, and producing correct visual content more difficult than generating text-only structures. |
| 2025-05-26 | [Safety Through Reasoning: An Empirical Study of Reasoning Guardrail Models](http://arxiv.org/abs/2505.20087v1) | Makesh Narsimhan Sreedhar, Traian Rebedea et al. | Reasoning-based language models have demonstrated strong performance across various domains, with the most notable gains seen in mathematical and coding tasks. Recent research has shown that reasoning also offers significant benefits for LLM safety and guardrail applications. In this work, we conduct a comprehensive analysis of training reasoning-based guardrail models for content moderation, with an emphasis on generalization to custom safety policies at inference time. Our study focuses on two key dimensions: data efficiency and inference efficiency. On the data front, we find that reasoning-based models exhibit strong sample efficiency, achieving competitive performance with significantly fewer training examples than their non-reasoning counterparts. This unlocks the potential to repurpose the remaining data for mining high-value, difficult samples that further enhance model performance. On the inference side, we evaluate practical trade-offs by introducing reasoning budgets, examining the impact of reasoning length on latency and accuracy, and exploring dual-mode training to allow runtime control over reasoning behavior. Our findings will provide practical insights for researchers and developers to effectively and efficiently train and deploy reasoning-based guardrails models in real-world systems. |
| 2025-05-26 | [SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](http://arxiv.org/abs/2505.20065v1) | Geon-Hyeong Kim, Youngsoo Jang et al. | As Large Language Models (LLMs) continue to advance and find applications across a growing number of fields, ensuring the safety of LLMs has become increasingly critical. To address safety concerns, recent studies have proposed integrating safety constraints into Reinforcement Learning from Human Feedback (RLHF). However, these approaches tend to be complex, as they encompass complicated procedures in RLHF along with additional steps required by the safety constraints. Inspired by Direct Preference Optimization (DPO), we introduce a new algorithm called SafeDPO, which is designed to directly optimize the safety alignment objective in a single stage of policy learning, without requiring relaxation. SafeDPO introduces only one additional hyperparameter to further enhance safety and requires only minor modifications to standard DPO. As a result, it eliminates the need to fit separate reward and cost models or to sample from the language model during fine-tuning, while still enhancing the safety of LLMs. Finally, we demonstrate that SafeDPO achieves competitive performance compared to state-of-the-art safety alignment algorithms, both in terms of aligning with human preferences and improving safety. |
| 2025-05-26 | [Target Tracking via LiDAR-RADAR Sensor Fusion for Autonomous Racing](http://arxiv.org/abs/2505.20043v1) | Marcello Cellina, Matteo Corno et al. | High Speed multi-vehicle Autonomous Racing will increase the safety and performance of road-going Autonomous Vehicles. Precise vehicle detection and dynamics estimation from a moving platform is a key requirement for planning and executing complex autonomous overtaking maneuvers. To address this requirement, we have developed a Latency-Aware EKF-based Multi Target Tracking algorithm fusing LiDAR and RADAR measurements. The algorithm explots the different sensor characteristics by explicitly integrating the Range Rate in the EKF Measurement Function, as well as a-priori knowledge of the racetrack during state prediction. It can handle Out-Of-Sequence Measurements via Reprocessing using a double State and Measurement Buffer, ensuring sensor delay compensation with no information loss. This algorithm has been implemented on Team PoliMOVE's autonomous racecar, and was proved experimentally by completing a number of fully autonomous overtaking maneuvers at speeds up to 275 km/h. |
| 2025-05-26 | [Requirements Coverage-Guided Minimization for Natural Language Test Cases](http://arxiv.org/abs/2505.20004v1) | Rongqi Pan, Feifei Niu et al. | As software systems evolve, test suites tend to grow in size and often contain redundant test cases. Such redundancy increases testing effort, time, and cost. Test suite minimization (TSM) aims to eliminate such redundancy while preserving key properties such as requirement coverage and fault detection capability. In this paper, we propose RTM (Requirement coverage-guided Test suite Minimization), a novel TSM approach designed for requirement-based testing (validation), which can effectively reduce test suite redundancy while ensuring full requirement coverage and a high fault detection rate (FDR) under a fixed minimization budget. Based on common practice in critical systems where functional safety is important, we assume test cases are specified in natural language and traced to requirements before being implemented. RTM preprocesses test cases using three different preprocessing methods, and then converts them into vector representations using seven text embedding techniques. Similarity values between vectors are computed utilizing three distance functions. A Genetic Algorithm, whose population is initialized by coverage-preserving initialization strategies, is then employed to identify an optimized subset containing diverse test cases matching the set budget.   We evaluate RTM on an industrial automotive system dataset comprising $736$ system test cases and $54$ requirements. Experimental results show that RTM consistently outperforms baseline techniques in terms of FDR across different minimization budgets while maintaining full requirement coverage. Furthermore, we investigate the impact of test suite redundancy levels on the effectiveness of TSM, providing new insights into optimizing requirement-based test suites under practical constraints. |
| 2025-05-26 | [The Limits of Preference Data for Post-Training](http://arxiv.org/abs/2505.19964v1) | Eric Zhao, Jessica Dai et al. | Recent progress in strengthening the capabilities of large language models has stemmed from applying reinforcement learning to domains with automatically verifiable outcomes. A key question is whether we can similarly use RL to optimize for outcomes in domains where evaluating outcomes inherently requires human feedback; for example, in tasks like deep research and trip planning, outcome evaluation is qualitative and there are many possible degrees of success. One attractive and scalable modality for collecting human feedback is preference data: ordinal rankings (pairwise or $k$-wise) that indicate, for $k$ given outcomes, which one is preferred. In this work, we study a critical roadblock: preference data fundamentally and significantly limits outcome-based optimization. Even with idealized preference data (infinite, noiseless, and online), the use of ordinal feedback can prevent obtaining even approximately optimal solutions. We formalize this impossibility using voting theory, drawing an analogy between how a model chooses to answer a query with how voters choose a candidate to elect. This indicates that grounded human scoring and algorithmic innovations are necessary for extending the success of RL post-training to domains demanding human feedback. We also explore why these limitations have disproportionately impacted RLHF when it comes to eliciting reasoning behaviors (e.g., backtracking) versus situations where RLHF has been historically successful (e.g., instruction-tuning and safety training), finding that the limitations of preference data primarily suppress RLHF's ability to elicit robust strategies -- a class that encompasses most reasoning behaviors. |
| 2025-05-26 | [Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy](http://arxiv.org/abs/2505.19951v1) | Elvir Karimov, Alexander Varlamov et al. | Deep learning voice models are commonly used nowadays, but the safety processing of personal data, such as human identity and speech content, remains suspicious. To prevent malicious user identification, speaker anonymization methods were proposed. Current methods, particularly based on universal adversarial patch (UAP) applications, have drawbacks such as significant degradation of audio quality, decreased speech recognition quality, low transferability across different voice biometrics models, and performance dependence on the input audio length. To mitigate these drawbacks, in this work, we introduce and leverage the novel Exponential Total Variance (TV) loss function and provide experimental evidence that it positively affects UAP strength and imperceptibility. Moreover, we present a novel scalable UAP insertion procedure and demonstrate its uniformly high performance for various audio lengths. |
| 2025-05-26 | [Uncertainty-Aware Safety-Critical Decision and Control for Autonomous Vehicles at Unsignalized Intersections](http://arxiv.org/abs/2505.19939v1) | Ran Yu, Zhuoren Li et al. | Reinforcement learning (RL) has demonstrated potential in autonomous driving (AD) decision tasks. However, applying RL to urban AD, particularly in intersection scenarios, still faces significant challenges. The lack of safety constraints makes RL vulnerable to risks. Additionally, cognitive limitations and environmental randomness can lead to unreliable decisions in safety-critical scenarios. Therefore, it is essential to quantify confidence in RL decisions to improve safety. This paper proposes an Uncertainty-aware Safety-Critical Decision and Control (USDC) framework, which generates a risk-averse policy by constructing a risk-aware ensemble distributional RL, while estimating uncertainty to quantify the policy's reliability. Subsequently, a high-order control barrier function (HOCBF) is employed as a safety filter to minimize intervention policy while dynamically enhancing constraints based on uncertainty. The ensemble critics evaluate both HOCBF and RL policies, embedding uncertainty to achieve dynamic switching between safe and flexible strategies, thereby balancing safety and efficiency. Simulation tests on unsignalized intersections in multiple tasks indicate that USDC can improve safety while maintaining traffic efficiency compared to baselines. |
| 2025-05-26 | [Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making](http://arxiv.org/abs/2505.19933v1) | Yejin Son, Minseo Kim et al. | Large Language Models (LLMs) are increasingly used for decision making in embodied agents, yet existing safety evaluations often rely on coarse success rates and domain-specific setups, making it difficult to diagnose why and where these models fail. This obscures our understanding of embodied safety and limits the selective deployment of LLMs in high-risk physical environments. We introduce SAFEL, the framework for systematically evaluating the physical safety of LLMs in embodied decision making. SAFEL assesses two key competencies: (1) rejecting unsafe commands via the Command Refusal Test, and (2) generating safe and executable plans via the Plan Safety Test. Critically, the latter is decomposed into functional modules, goal interpretation, transition modeling, action sequencing, enabling fine-grained diagnosis of safety failures. To support this framework, we introduce EMBODYGUARD, a PDDL-grounded benchmark containing 942 LLM-generated scenarios covering both overtly malicious and contextually hazardous instructions. Evaluation across 13 state-of-the-art LLMs reveals that while models often reject clearly unsafe commands, they struggle to anticipate and mitigate subtle, situational risks. Our results highlight critical limitations in current LLMs and provide a foundation for more targeted, modular improvements in safe embodied reasoning. |
| 2025-05-26 | [Evaluating AI cyber capabilities with crowdsourced elicitation](http://arxiv.org/abs/2505.19915v1) | Artem Petrov, Dmitrii Volkov | As AI systems become increasingly capable, understanding their offensive cyber potential is critical for informed governance and responsible deployment. However, it's hard to accurately bound their capabilities, and some prior evaluations dramatically underestimated them. The art of extracting maximum task-specific performance from AIs is called "AI elicitation", and today's safety organizations typically conduct it in-house. In this paper, we explore crowdsourcing elicitation efforts as an alternative to in-house elicitation work.   We host open-access AI tracks at two Capture The Flag (CTF) competitions: AI vs. Humans (400 teams) and Cyber Apocalypse_ (4000 teams). The AI teams achieve outstanding performance at both events, ranking top-13% and top-21% respectively for a total of \$7500 in bounties. This impressive performance suggests that open-market elicitation may offer an effective complement to in-house elicitation. We propose elicitation bounties as a practical mechanism for maintaining timely, cost-effective situational awareness of emerging AI capabilities.   Another advantage of open elicitations is the option to collect human performance data at scale. Applying METR's methodology, we found that AI agents can reliably solve cyber challenges requiring one hour or less of effort from a median human CTF participant. |
| 2025-05-26 | [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](http://arxiv.org/abs/2505.19914v1) | Jiangjie Chen, Qianyu He et al. | Large Language Models (LLMs), such as OpenAI's o1 and DeepSeek's R1, excel at advanced reasoning tasks like math and coding via Reinforcement Learning with Verifiable Rewards (RLVR), but still struggle with puzzles solvable by humans without domain knowledge. We introduce Enigmata, the first comprehensive suite tailored for improving LLMs with puzzle reasoning skills. It includes 36 tasks across seven categories, each with 1) a generator that produces unlimited examples with controllable difficulty and 2) a rule-based verifier for automatic evaluation. This generator-verifier design supports scalable, multi-task RL training, fine-grained analysis, and seamless RLVR integration. We further propose Enigmata-Eval, a rigorous benchmark, and develop optimized multi-task RLVR strategies. Our trained model, Qwen2.5-32B-Enigmata, consistently surpasses o3-mini-high and o1 on the puzzle reasoning benchmarks like Enigmata-Eval, ARC-AGI (32.8%), and ARC-AGI 2 (0.6%). It also generalizes well to out-of-domain puzzle benchmarks and mathematical reasoning, with little multi-tasking trade-off. When trained on larger models like Seed1.5-Thinking (20B activated parameters and 200B total parameters), puzzle data from Enigmata further boosts SoTA performance on advanced math and STEM reasoning tasks such as AIME (2024-2025), BeyondAIME and GPQA (Diamond), showing nice generalization benefits of Enigmata. This work offers a unified, controllable framework for advancing logical reasoning in LLMs. Resources of this work can be found at https://seed-enigmata.github.io. |
| 2025-05-26 | [Attention! You Vision Language Model Could Be Maliciously Manipulated](http://arxiv.org/abs/2505.19911v1) | Xiaosen Wang, Shaokang Wang et al. | Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. |
| 2025-05-26 | [Causal Bayesian Networks for Data-driven Safety Analysis of Complex Systems](http://arxiv.org/abs/2505.19860v1) | Roman Gansch, Lina Putze et al. | Ensuring safe operation of safety-critical complex systems interacting with their environment poses significant challenges, particularly when the system's world model relies on machine learning algorithms to process the perception input. A comprehensive safety argumentation requires knowledge of how faults or functional insufficiencies propagate through the system and interact with external factors, to manage their safety impact. While statistical analysis approaches can support the safety assessment, associative reasoning alone is neither sufficient for the safety argumentation nor for the identification and investigation of safety measures. A causal understanding of the system and its interaction with the environment is crucial for safeguarding safety-critical complex systems. It allows to transfer and generalize knowledge, such as insights gained from testing, and facilitates the identification of potential improvements. This work explores using causal Bayesian networks to model the system's causalities for safety analysis, and proposes measures to assess causal influences based on Pearl's framework of causal inference. We compare the approach of causal Bayesian networks to the well-established fault tree analysis, outlining advantages and limitations. In particular, we examine importance metrics typically employed in fault tree analysis as foundation to discuss suitable causal metrics. An evaluation is performed on the example of a perception system for automated driving. Overall, this work presents an approach for causal reasoning in safety analysis that enables the integration of data-driven and expert-based knowledge to account for uncertainties arising from complex systems operating in open environments. |
| 2025-05-26 | [PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints](http://arxiv.org/abs/2505.19842v1) | Shuo Wang, Yun Cheng et al. | Air quality forecasting (AQF) is critical for public health and environmental management, yet remains challenging due to the complex interplay of emissions, meteorology, and chemical transformations. Traditional numerical models, such as CMAQ and WRF-Chem, provide physically grounded simulations but are computationally expensive and rely on uncertain emission inventories. Deep learning models, while computationally efficient, often struggle with generalization due to their lack of physical constraints. To bridge this gap, we propose PCDCNet, a surrogate model that integrates numerical modeling principles with deep learning. PCDCNet explicitly incorporates emissions, meteorological influences, and domain-informed constraints to model pollutant formation, transport, and dissipation. By combining graph-based spatial transport modeling, recurrent structures for temporal accumulation, and representation enhancement for local interactions, PCDCNet achieves state-of-the-art (SOTA) performance in 72-hour station-level PM2.5 and O3 forecasting while significantly reducing computational costs. Furthermore, our model is deployed in an online platform, providing free, real-time air quality forecasts, demonstrating its scalability and societal impact. By aligning deep learning with physical consistency, PCDCNet offers a practical and interpretable solution for AQF, enabling informed decision-making for both personal and regulatory applications. |
| 2025-05-26 | [InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory](http://arxiv.org/abs/2505.19820v1) | Feifei Li, Mi Zhang et al. | Interpretability of point cloud (PC) models becomes imperative given their deployment in safety-critical scenarios such as autonomous vehicles. We focus on attributing PC model outputs to interpretable critical concepts, defined as meaningful subsets of the input point cloud. To enable human-understandable diagnostics of model failures, an ideal critical subset should be *faithful* (preserving points that causally influence predictions) and *conceptually coherent* (forming semantically meaningful structures that align with human perception). We propose InfoCons, an explanation framework that applies information-theoretic principles to decompose the point cloud into 3D concepts, enabling the examination of their causal effect on model predictions with learnable priors. We evaluate InfoCons on synthetic datasets for classification, comparing it qualitatively and quantitatively with four baselines. We further demonstrate its scalability and flexibility on two real-world datasets and in two applications that utilize critical scores of PC. |
| 2025-05-26 | [What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](http://arxiv.org/abs/2505.19773v1) | Sangyeop Kim, Yohan Lee et al. | We investigate long-context vulnerabilities in Large Language Models (LLMs) through Many-Shot Jailbreaking (MSJ). Our experiments utilize context length of up to 128K tokens. Through comprehensive analysis with various many-shot attack settings with different instruction styles, shot density, topic, and format, we reveal that context length is the primary factor determining attack effectiveness. Critically, we find that successful attacks do not require carefully crafted harmful content. Even repetitive shots or random dummy text can circumvent model safety measures, suggesting fundamental limitations in long-context processing capabilities of LLMs. The safety behavior of well-aligned models becomes increasingly inconsistent with longer contexts. These findings highlight significant safety gaps in context expansion capabilities of LLMs, emphasizing the need for new safety mechanisms. |
| 2025-05-26 | [SGM: A Framework for Building Specification-Guided Moderation Filters](http://arxiv.org/abs/2505.19766v1) | Masoomali Fatehkia, Enes Altinisik et al. | Aligning large language models (LLMs) with deployment-specific requirements is critical but inherently imperfect. Despite extensive training, models remain susceptible to misalignment and adversarial inputs such as jailbreaks. Content moderation filters are commonly used as external safeguards, though they typically focus narrowly on safety. We introduce SGM (Specification-Guided Moderation), a flexible framework for training moderation filters grounded in user-defined specifications that go beyond standard safety concerns. SGM automates training data generation without relying on human-written examples, enabling scalable support for diverse, application-specific alignment goals. SGM-trained filters perform on par with state-of-the-art safety filters built on curated datasets, while supporting fine-grained and user-defined alignment control. |
| 2025-05-26 | [Beyond Safe Answers: A Benchmark for Evaluating True Risk Awareness in Large Reasoning Models](http://arxiv.org/abs/2505.19690v1) | Baihui Zheng, Boren Zheng et al. | Despite the remarkable proficiency of \textit{Large Reasoning Models} (LRMs) in handling complex reasoning tasks, their reliability in safety-critical scenarios remains uncertain. Existing evaluations primarily assess response-level safety, neglecting a critical issue we identify as \textbf{\textit{Superficial Safety Alignment} (SSA)} -- a phenomenon where models produce superficially safe outputs while internal reasoning processes fail to genuinely detect and mitigate underlying risks, resulting in inconsistent safety behaviors across multiple sampling attempts. To systematically investigate SSA, we introduce \textbf{Beyond Safe Answers (BSA)} bench, a novel benchmark comprising 2,000 challenging instances organized into three distinct SSA scenario types and spanning nine risk categories, each meticulously annotated with risk rationales. Evaluations of 19 state-of-the-art LRMs demonstrate the difficulty of this benchmark, with top-performing models achieving only 38.0\% accuracy in correctly identifying risk rationales. We further explore the efficacy of safety rules, specialized fine-tuning on safety reasoning data, and diverse decoding strategies in mitigating SSA. Our work provides a comprehensive assessment tool for evaluating and improving safety reasoning fidelity in LRMs, advancing the development of genuinely risk-aware and reliably safe AI systems. |
| 2025-05-26 | [VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models](http://arxiv.org/abs/2505.19684v1) | Bingrui Sima, Linhua Cong et al. | The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks. |
| 2025-05-26 | [Reshaping Representation Space to Balance the Safety and Over-rejection in Large Audio Language Models](http://arxiv.org/abs/2505.19670v1) | Hao Yang, Lizhen Qu et al. | Large Audio Language Models (LALMs) have extended the capabilities of Large Language Models (LLMs) by enabling audio-based human interactions. However, recent research has revealed that LALMs remain vulnerable to harmful queries due to insufficient safety-alignment. Despite advances in defence measures for text and vision LLMs, effective safety-alignment strategies and audio-safety dataset specifically targeting LALMs are notably absent. Meanwhile defence measures based on Supervised Fine-tuning (SFT) struggle to address safety improvement while avoiding over-rejection issues, significantly compromising helpfulness. In this work, we propose an unsupervised safety-fine-tuning strategy as remedy that reshapes model's representation space to enhance existing LALMs safety-alignment while balancing the risk of over-rejection. Our experiments, conducted across three generations of Qwen LALMs, demonstrate that our approach significantly improves LALMs safety under three modality input conditions (audio-text, text-only, and audio-only) while increasing over-rejection rate by only 0.88% on average. Warning: this paper contains harmful examples. |

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



