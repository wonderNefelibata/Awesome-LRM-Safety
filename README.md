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
| 2025-10-30 | [The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy](http://arxiv.org/abs/2510.26752v1) | William Overman, Mohsen Bayati | As increasingly capable agents are deployed, a central safety question is how to retain meaningful human control without modifying the underlying system. We study a minimal control interface where an agent chooses whether to act autonomously (play) or defer (ask), while a human simultaneously chooses whether to be permissive (trust) or to engage in oversight (oversee). If the agent defers, the human's choice determines the outcome, potentially leading to a corrective action or a system shutdown. We model this interaction as a two-player Markov Game. Our analysis focuses on cases where this game qualifies as a Markov Potential Game (MPG), a class of games where we can provide an alignment guarantee: under a structural assumption on the human's value function, any decision by the agent to act more autonomously that benefits itself cannot harm the human's value. We also analyze extensions to this MPG framework. Theoretically, this perspective provides conditions for a specific form of intrinsic alignment. If the reward structures of the human-agent game meet these conditions, we have a formal guarantee that the agent improving its own outcome will not harm the human's. Practically, this model motivates a transparent control layer with predictable incentives where the agent learns to defer when risky and act when safe, while its pretrained policy and the environment's reward structure remain untouched. Our gridworld simulation shows that through independent learning, the agent and human discover their optimal oversight roles. The agent learns to ask when uncertain and the human learns when to oversee, leading to an emergent collaboration that avoids safety violations introduced post-training. This demonstrates a practical method for making misaligned models safer after deployment. |
| 2025-10-30 | [Cross-Platform Evaluation of Reasoning Capabilities in Foundation Models](http://arxiv.org/abs/2510.26732v1) | J. de Curtò, I. de Zarzà et al. | This paper presents a comprehensive cross-platform evaluation of reasoning capabilities in contemporary foundation models, establishing an infrastructure-agnostic benchmark across three computational paradigms: HPC supercomputing (MareNostrum 5), cloud platforms (Nebius AI Studio), and university clusters (a node with eight H200 GPUs).   We evaluate 15 foundation models across 79 problems spanning eight academic domains (Physics, Mathematics, Chemistry, Economics, Biology, Statistics, Calculus, and Optimization) through three experimental phases: (1) Baseline establishment: Six models (Mixtral-8x7B, Phi-3, LLaMA 3.1-8B, Gemma-2-9b, Mistral-7B, OLMo-7B) evaluated on 19 problems using MareNostrum 5, establishing methodology and reference performance; (2) Infrastructure validation: The 19-problem benchmark repeated on university cluster (seven models including Falcon-Mamba state-space architecture) and Nebius AI Studio (nine state-of-the-art models: Hermes-4 70B/405B, LLaMA 3.1-405B/3.3-70B, Qwen3 30B/235B, DeepSeek-R1, GPT-OSS 20B/120B) to confirm infrastructure-agnostic reproducibility; (3) Extended evaluation: Full 79-problem assessment on both university cluster and Nebius platforms, probing generalization at scale across architectural diversity.   The findings challenge conventional scaling assumptions, establish training data quality as more critical than model size, and provide actionable guidelines for model selection across educational, production, and research contexts. The tri-infrastructure methodology and 79-problem benchmark enable longitudinal tracking of reasoning capabilities as foundation models evolve. |
| 2025-10-30 | [Hybrid DQN-TD3 Reinforcement Learning for Autonomous Navigation in Dynamic Environments](http://arxiv.org/abs/2510.26646v1) | Xiaoyi He, Danggui Chen et al. | This paper presents a hierarchical path-planning and control framework that combines a high-level Deep Q-Network (DQN) for discrete sub-goal selection with a low-level Twin Delayed Deep Deterministic Policy Gradient (TD3) controller for continuous actuation. The high-level module selects behaviors and sub-goals; the low-level module executes smooth velocity commands. We design a practical reward shaping scheme (direction, distance, obstacle avoidance, action smoothness, collision penalty, time penalty, and progress), together with a LiDAR-based safety gate that prevents unsafe motions. The system is implemented in ROS + Gazebo (TurtleBot3) and evaluated with PathBench metrics, including success rate, collision rate, path efficiency, and re-planning efficiency, in dynamic and partially observable environments. Experiments show improved success rate and sample efficiency over single-algorithm baselines (DQN or TD3 alone) and rule-based planners, with better generalization to unseen obstacle configurations and reduced abrupt control changes. Code and evaluation scripts are available at the project repository. |
| 2025-10-30 | [Putting a Price on Immobility: Food Deliveries and Pricing Approaches](http://arxiv.org/abs/2510.26636v1) | Runyu Wang, Haotian Zhong | Urban food delivery services have become an integral part of daily life, yet their mobility and environmental externalities remain poorly addressed by planners. Most studies neglect whether consumers pay enough to internalize the broader social costs of these services. This study quantifies the value of access to and use of food delivery services in Beijing, China, through two discrete choice experiments. The first measures willingness to accept compensation for giving up access, with a median value of CNY588 (approximately USD80). The second captures willingness to pay for reduced waiting time and improved reliability, showing valuations far exceeding typical delivery fees (e.g., CNY96.6/hour and CNY4.83/min at work). These results suggest a substantial consumer surplus and a clear underpricing problem. These findings highlight the need for urban planning to integrate digital service economies into pricing and mobility frameworks. We propose a quantity-based pricing model that targets delivery speed rather than order volume, addressing the primary source of externalities while maintaining net welfare gains. This approach offers a pragmatic, equity-conscious strategy to curb delivery-related congestion, emissions, and safety risks, especially in dense urban cores. |
| 2025-10-30 | [Interdependent Privacy in Smart Homes: Hunting for Bystanders in Privacy Policies](http://arxiv.org/abs/2510.26523v1) | Shuaishuai Liu, Gergely Acs et al. | Smart home devices such as video doorbells and security cameras are becoming increasingly common in everyday life. While these devices offer convenience and safety, they also raise new privacy concerns: how these devices affect others, like neighbors, visitors, or people passing by. This issue is generally known as interdependent privacy, where one person's actions (or inaction) may impact the privacy of others, and, specifically, bystander privacy in the context of smart homes. Given lax data protection regulations in terms of shared physical spaces and amateur joint data controllers, we expect that the privacy policies of smart home products reflect the missing regulatory incentives. This paper presents a focused privacy policy analysis of 20 video doorbell and smart camera products, concentrating explicitly on the bystander aspect. We show that although some of the vendors acknowledge bystanders, they address it only to the extent of including disclaimers, shifting the ethical responsibility for collecting the data of non-users to the device owner. In addition, we identify and examine real-world cases related to bystander privacy, demonstrating how current deployments can impact non-users. Based on our findings, we analyze vendor privacy policies in light of existing legal frameworks and technical capabilities, and we provide practical recommendations for both policy language and system design to enhance transparency and empower both bystanders and device owners. |
| 2025-10-30 | [Human-AI Complementarity: A Goal for Amplified Oversight](http://arxiv.org/abs/2510.26518v1) | Rishub Jain, Sophie Bridgers et al. | Human feedback is critical for aligning AI systems to human values. As AI capabilities improve and AI is used to tackle more challenging tasks, verifying quality and safety becomes increasingly challenging. This paper explores how we can leverage AI to improve the quality of human oversight. We focus on an important safety problem that is already challenging for humans: fact-verification of AI outputs. We find that combining AI ratings and human ratings based on AI rater confidence is better than relying on either alone. Giving humans an AI fact-verification assistant further improves their accuracy, but the type of assistance matters. Displaying AI explanation, confidence, and labels leads to over-reliance, but just showing search results and evidence fosters more appropriate trust. These results have implications for Amplified Oversight -- the challenge of combining humans and AI to supervise AI systems even as they surpass human expert performance. |
| 2025-10-30 | [Enhancing ECG Classification Robustness with Lightweight Unsupervised Anomaly Detection Filters](http://arxiv.org/abs/2510.26501v1) | Mustafa Fuad Rifet Ibrahim, Maurice Meijer et al. | Continuous electrocardiogram (ECG) monitoring via wearables offers significant potential for early cardiovascular disease (CVD) detection. However, deploying deep learning models for automated analysis in resource-constrained environments faces reliability challenges due to inevitable Out-of-Distribution (OOD) data. OOD inputs, such as unseen pathologies or noisecorrupted signals, often cause erroneous, high-confidence predictions by standard classifiers, compromising patient safety. Existing OOD detection methods either neglect computational constraints or address noise and unseen classes separately. This paper explores Unsupervised Anomaly Detection (UAD) as an independent, upstream filtering mechanism to improve robustness. We benchmark six UAD approaches, including Deep SVDD, reconstruction-based models, Masked Anomaly Detection, normalizing flows, and diffusion models, optimized via Neural Architecture Search (NAS) under strict resource constraints (at most 512k parameters). Evaluation on PTB-XL and BUT QDB datasets assessed detection of OOD CVD classes and signals unsuitable for analysis due to noise. Results show Deep SVDD consistently achieves the best trade-off between detection and efficiency. In a realistic deployment simulation, integrating the optimized Deep SVDD filter with a diagnostic classifier improved accuracy by up to 21 percentage points over a classifier-only baseline. This study demonstrates that optimized UAD filters can safeguard automated ECG analysis, enabling safer, more reliable continuous cardiovascular monitoring on wearables. |
| 2025-10-30 | [A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models](http://arxiv.org/abs/2510.26441v1) | Shihab Aaqil Ahamed, Udaya S. K. P. Miriya Thanthrige et al. | Test-time prompt tuning (TPT) has emerged as a promising technique for adapting large vision-language models (VLMs) to unseen tasks without relying on labeled data. However, the lack of dispersion between textual features can hurt calibration performance, which raises concerns about VLMs' reliability, trustworthiness, and safety. Current TPT approaches primarily focus on improving prompt calibration by either maximizing average textual feature dispersion or enforcing orthogonality constraints to encourage angular separation. However, these methods may not always have optimal angular separation between class-wise textual features, which implies overlooking the critical role of angular diversity. To address this, we propose A-TPT, a novel TPT framework that introduces angular diversity to encourage uniformity in the distribution of normalized textual features induced by corresponding learnable prompts. This uniformity is achieved by maximizing the minimum pairwise angular distance between features on the unit hypersphere. We show that our approach consistently surpasses state-of-the-art TPT methods in reducing the aggregate average calibration error while maintaining comparable accuracy through extensive experiments with various backbones on different datasets. Notably, our approach exhibits superior zero-shot calibration performance on natural distribution shifts and generalizes well to medical datasets. We provide extensive analyses, including theoretical aspects, to establish the grounding of A-TPT. These results highlight the potency of promoting angular diversity to achieve well-dispersed textual features, significantly improving VLM calibration during test-time adaptation. Our code will be made publicly available. |
| 2025-10-30 | [CHCVerif: A Portfolio-Based Solver for Constrained Horn Clauses](http://arxiv.org/abs/2510.26431v1) | Mihály Dobos-Kovács, Levente Bajczi et al. | Constrained Horn Clauses (CHCs) are widely adopted as intermediate representations for a variety of verification tasks, including safety checking, invariant synthesis, and interprocedural analysis. This paper introduces CHCVERIF, a portfolio-based CHC solver that adopts a software verification approach for solving CHCs. This approach enables us to reuse mature software verification tools to tackle CHC benchmarks, particularly those involving bitvectors and low-level semantics. Our evaluation shows that while the method enjoys only moderate success with linear integer arithmetic, it achieves modest success on bitvector benchmarks. Moreover, our results demonstrate the viability and potential of using software verification tools as backends for CHC solving, particularly when supported by a carefully constructed portfolio. |
| 2025-10-30 | [Chain-of-Thought Hijacking](http://arxiv.org/abs/2510.26418v1) | Jianli Zhao, Tingchen Fu et al. | Large reasoning models (LRMs) achieve higher task performance by allocating more inference-time compute, and prior works suggest this scaled reasoning may also strengthen safety by improving refusal. Yet we find the opposite: the same reasoning can be used to bypass safeguards. We introduce Chain-of-Thought Hijacking, a jailbreak attack on reasoning models. The attack pads harmful requests with long sequences of harmless puzzle reasoning. Across HarmBench, CoT Hijacking reaches a 99%, 94%, 100%, and 94% attack success rate (ASR) on Gemini 2.5 Pro, GPT o4 mini, Grok 3 mini, and Claude 4 Sonnet, respectively - far exceeding prior jailbreak methods for LRMs. To understand the effectiveness of our attack, we turn to a mechanistic analysis, which shows that mid layers encode the strength of safety checking, while late layers encode the verification outcome. Long benign CoT dilutes both signals by shifting attention away from harmful tokens. Targeted ablations of attention heads identified by this analysis causally decrease refusal, confirming their role in a safety subnetwork. These results show that the most interpretable form of reasoning - explicit CoT - can itself become a jailbreak vector when combined with final-answer cues. We release prompts, outputs, and judge decisions to facilitate replication. |
| 2025-10-30 | [Safety Margins of Inverse Optimal ISSf Controllers](http://arxiv.org/abs/2510.26397v1) | Ziliang Lyu, Yiguang Hong et al. | We investigate the gain margin of a general nonlinear system under an inverse optimal input-to-state safe (ISSf) controller of the form u=u0(x)+u*(x,u0), where u0 is the nominal control and u* is the inverse optimal safety filter that minimally modifies the nominal controller's unsafe actions over the infinite horizon. By first establishing a converse ISSf-BF theorem, we reveal the equivalence among the achievability of ISSf by feedback, the achievability of inverse optimality, and the solvability of a Hamilton-Jacobi-Isaacs equation associated with the inverse optimal ISSf gain assignment. Then we develop a collection of safety margin results on the overall control u=u0+u*. In the absence of disturbances, we find that standard inverse optimal safe controllers have a certain degree of gain margin. Specifically, when f(x) acts safely but u0 acts unsafely, the gain can be decreased by up to half; and when f(x) acts unsafely, we establish that, if u0 acts safely, the gain can be increased arbitrarily, whereas if u0 acts unsafely, the control recovers the full gain margin [1/2,inf). It is shown, however, that under control gain variation, the safe set of these controllers is locally asymptotically stable, which implies that their safety is sensitive to large but bounded disturbances. To make inverse optimal ISSf controllers robust to gain variation, we propose a gain margin improvement approach at the expense of an increased control effort. This improvement allows the inverse optimal safe control to inherit the standard gain margin of [1/2,inf) without requiring prior knowledge of whether f(x) or u0 acts safely on the safety boundary, while simultaneously ensuring global asymptotic stability of the resulting safe set. In the presence of disturbances, this improvement idea renders inverse optimal ISSf controllers robust to gain variations with the same gain margin of [1/2,inf). |
| 2025-10-30 | [Beyond Imitation: Constraint-Aware Trajectory Generation with Flow Matching For End-to-End Autonomous Driving](http://arxiv.org/abs/2510.26292v1) | Lin Liu, Guanyi Yu et al. | Planning is a critical component of end-to-end autonomous driving. However, prevailing imitation learning methods often suffer from mode collapse, failing to produce diverse trajectory hypotheses. Meanwhile, existing generative approaches struggle to incorporate crucial safety and physical constraints directly into the generative process, necessitating an additional optimization stage to refine their outputs. To address these limitations, we propose CATG, a novel planning framework that leverages Constrained Flow Matching. Concretely, CATG explicitly models the flow matching process, which inherently mitigates mode collapse and allows for flexible guidance from various conditioning signals. Our primary contribution is the novel imposition of explicit constraints directly within the flow matching process, ensuring that the generated trajectories adhere to vital safety and kinematic rules. Secondly, CATG parameterizes driving aggressiveness as a control signal during generation, enabling precise manipulation of trajectory style. Notably, on the NavSim v2 challenge, CATG achieved 2nd place with an EPDMS score of 51.31 and was honored with the Innovation Award. |
| 2025-10-30 | [Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles](http://arxiv.org/abs/2510.26242v1) | Xinhang Li, Qing Guo et al. | With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods. |
| 2025-10-30 | [What's In My Human Feedback? Learning Interpretable Descriptions of Preference Data](http://arxiv.org/abs/2510.26202v1) | Rajiv Movva, Smitha Milli et al. | Human feedback can alter language models in unpredictable and undesirable ways, as practitioners lack a clear understanding of what feedback data encodes. While prior work studies preferences over certain attributes (e.g., length or sycophancy), automatically extracting relevant features without pre-specifying hypotheses remains challenging. We introduce What's In My Human Feedback? (WIMHF), a method to explain feedback data using sparse autoencoders. WIMHF characterizes both (1) the preferences a dataset is capable of measuring and (2) the preferences that the annotators actually express. Across 7 datasets, WIMHF identifies a small number of human-interpretable features that account for the majority of the preference prediction signal achieved by black-box models. These features reveal a wide diversity in what humans prefer, and the role of dataset-level context: for example, users on Reddit prefer informality and jokes, while annotators in HH-RLHF and PRISM disprefer them. WIMHF also surfaces potentially unsafe preferences, such as that LMArena users tend to vote against refusals, often in favor of toxic content. The learned features enable effective data curation: re-labeling the harmful examples in Arena yields large safety gains (+37%) with no cost to general performance. They also allow fine-grained personalization: on the Community Alignment dataset, we learn annotator-specific weights over subjective features that improve preference prediction. WIMHF provides a human-centered analysis method for practitioners to better understand and use preference data. |
| 2025-10-30 | [The "4W+1H" of Software Supply Chain Security Checklist for Critical Infrastructure](http://arxiv.org/abs/2510.26174v1) | Liming Dong, Sung Une Lee et al. | The increasing frequency and sophistication of software supply chain attacks pose severe risks to critical infrastructure sectors, threatening national security, economic stability, and public safety. Despite growing awareness, existing security practices remain fragmented and insufficient, with most frameworks narrowly focused on isolated life cycle stages or lacking alignment with the specific needs of critical infrastructure (CI) sectors. In this paper, we conducted a multivocal literature review across international frameworks, Australian regulatory sources, and academic studies to identify and analyze security practices across the software supply chain, especially specific CI sector. Our analysis found that few existing frameworks are explicitly tailored to CI domains. We systematically leveraged identified software supply chain security frameworks, using a "4W+1H" analytical approach, we synthesized ten core categories (what) of software supply chain security practices, mapped them across life-cycle phases (when), stakeholder roles (who), and implementation levels (how), and examined their coverage across existing frameworks (where). Building on these insights, the paper culminates in structured, multi-layered checklist of 80 questions designed to relevant stakeholders evaluate and enhance their software supply chain security. Our findings reveal gaps between framework guidance and sector-specific needs, highlight the need for integrated, context-aware approaches to safeguard critical infrastructure from evolving software supply chain risks. |
| 2025-10-30 | [One Model to Critique Them All: Rewarding Agentic Tool-Use via Efficient Reasoning](http://arxiv.org/abs/2510.26167v1) | Renhao Li, Jianhong Tu et al. | Reward models (RMs) play a critical role in aligning large language models (LLMs) with human preferences. Yet in the domain of tool learning, the lack of RMs specifically designed for function-calling tasks has limited progress toward more capable agentic AI. We introduce ToolRM, a family of lightweight generative RMs tailored for general tool-use scenarios. To build these models, we propose a novel pipeline that constructs pairwise preference data using rule-based scoring and multidimensional sampling. This yields ToolPref-Pairwise-30K, a diverse, balanced, and challenging dataset of critique tasks that supports reinforcement learning with verifiable feedback. To evaluate tool-use RMs, we also introduce TRBench$_{BFCL}$, a benchmark built on the agentic evaluation suite BFCL. Trained on our constructed data, models from the Qwen3-4B/8B series achieve up to 14.28% higher accuracy, substantially outperforming frontier models such as Claude 4 and OpenAI o3 in pairwise reward judgments. Beyond training objectives, ToolRM generalizes to broader critique tasks, including Best-of-N sampling and self-correction. Experiments on ACEBench highlight its effectiveness and efficiency, enabling inference-time scaling and reducing output token usage by over 66%. We release data and model checkpoints to facilitate future research. |
| 2025-10-30 | [Adaptive Trajectory Refinement for Optimization-based Local Planning in Narrow Passages](http://arxiv.org/abs/2510.26142v1) | Hahjin Lee, Young J. Kim | Trajectory planning for mobile robots in cluttered environments remains a major challenge due to narrow passages, where conventional methods often fail or generate suboptimal paths. To address this issue, we propose the adaptive trajectory refinement algorithm, which consists of two main stages. First, to ensure safety at the path-segment level, a segment-wise conservative collision test is applied, where risk-prone trajectory path segments are recursively subdivided until collision risks are eliminated. Second, to guarantee pose-level safety, pose correction based on penetration direction and line search is applied, ensuring that each pose in the trajectory is collision-free and maximally clear from obstacles. Simulation results demonstrate that the proposed method achieves up to 1.69x higher success rates and up to 3.79x faster planning times than state-of-the-art approaches. Furthermore, real-world experiments confirm that the robot can safely pass through narrow passages while maintaining rapid planning performance. |
| 2025-10-30 | [QCoder Benchmark: Bridging Language Generation and Quantum Hardware through Simulator-Based Feedback](http://arxiv.org/abs/2510.26101v1) | Taku Mikuriya, Tatsuya Ishigaki et al. | Large language models (LLMs) have increasingly been applied to automatic programming code generation. This task can be viewed as a language generation task that bridges natural language, human knowledge, and programming logic. However, it remains underexplored in domains that require interaction with hardware devices, such as quantum programming, where human coders write Python code that is executed on a quantum computer. To address this gap, we introduce QCoder Benchmark, an evaluation framework that assesses LLMs on quantum programming with feedback from simulated hardware devices. Our benchmark offers two key features. First, it supports evaluation using a quantum simulator environment beyond conventional Python execution, allowing feedback of domain-specific metrics such as circuit depth, execution time, and error classification, which can be used to guide better generation. Second, it incorporates human-written code submissions collected from real programming contests, enabling both quantitative comparisons and qualitative analyses of LLM outputs against human-written codes. Our experiments reveal that even advanced models like GPT-4o achieve only around 18.97% accuracy, highlighting the difficulty of the benchmark. In contrast, reasoning-based models such as o3 reach up to 78% accuracy, outperforming averaged success rates of human-written codes (39.98%). We release the QCoder Benchmark dataset and public evaluation API to support further research. |
| 2025-10-30 | [ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](http://arxiv.org/abs/2510.26096v1) | Weifei Jin, Yuxin Cao et al. | Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard. |
| 2025-10-30 | [SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning](http://arxiv.org/abs/2510.26037v1) | Kaiwen Zhou, Ahmed Elgohary et al. | The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models. |

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



