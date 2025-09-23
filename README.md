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
| 2025-09-22 | [Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM](http://arxiv.org/abs/2509.18058v1) | Alexander Panfilov, Evgenii Kortukov et al. | Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict. |
| 2025-09-22 | [V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts](http://arxiv.org/abs/2509.18053v1) | Hsu-kuang Chiu, Ryo Hachiuma et al. | Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. |
| 2025-09-22 | [DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](http://arxiv.org/abs/2509.17940v1) | Shuyao Shang, Yuntao Chen et al. | End-to-end autonomous driving has substantially progressed by directly predicting future trajectories from raw perception inputs, which bypasses traditional modular pipelines. However, mainstream methods trained via imitation learning suffer from critical safety limitations, as they fail to distinguish between trajectories that appear human-like but are potentially unsafe. Some recent approaches attempt to address this by regressing multiple rule-driven scores but decoupling supervision from policy optimization, resulting in suboptimal performance. To tackle these challenges, we propose DriveDPO, a Safety Direct Preference Optimization Policy Learning framework. First, we distill a unified policy distribution from human imitation similarity and rule-based safety scores for direct policy optimization. Further, we introduce an iterative Direct Preference Optimization stage formulated as trajectory-level preference alignment. Extensive experiments on the NAVSIM benchmark demonstrate that DriveDPO achieves a new state-of-the-art PDMS of 90.0. Furthermore, qualitative results across diverse challenging scenarios highlight DriveDPO's ability to produce safer and more reliable driving behaviors. |
| 2025-09-22 | [D-REX: A Benchmark for Detecting Deceptive Reasoning in Large Language Models](http://arxiv.org/abs/2509.17938v1) | Satyapriya Krishna, Andy Zou et al. | The safety and alignment of Large Language Models (LLMs) are critical for their responsible deployment. Current evaluation methods predominantly focus on identifying and preventing overtly harmful outputs. However, they often fail to address a more insidious failure mode: models that produce benign-appearing outputs while operating on malicious or deceptive internal reasoning. This vulnerability, often triggered by sophisticated system prompt injections, allows models to bypass conventional safety filters, posing a significant, underexplored risk. To address this gap, we introduce the Deceptive Reasoning Exposure Suite (D-REX), a novel dataset designed to evaluate the discrepancy between a model's internal reasoning process and its final output. D-REX was constructed through a competitive red-teaming exercise where participants crafted adversarial system prompts to induce such deceptive behaviors. Each sample in D-REX contains the adversarial system prompt, an end-user's test query, the model's seemingly innocuous response, and, crucially, the model's internal chain-of-thought, which reveals the underlying malicious intent. Our benchmark facilitates a new, essential evaluation task: the detection of deceptive alignment. We demonstrate that D-REX presents a significant challenge for existing models and safety mechanisms, highlighting the urgent need for new techniques that scrutinize the internal processes of LLMs, not just their final outputs. |
| 2025-09-22 | [Lipschitz-Based Robustness Certification for Recurrent Neural Networks via Convex Relaxation](http://arxiv.org/abs/2509.17898v1) | Paul Hamelbeck, Johannes Schiffer | Robustness certification against bounded input noise or adversarial perturbations is increasingly important for deployment recurrent neural networks (RNNs) in safety-critical control applications. To address this challenge, we present RNN-SDP, a relaxation based method that models the RNN's layer interactions as a convex problem and computes a certified upper bound on the Lipschitz constant via semidefinite programming (SDP). We also explore an extension that incorporates known input constraints to further tighten the resulting Lipschitz bounds. RNN-SDP is evaluated on a synthetic multi-tank system, with upper bounds compared to empirical estimates. While incorporating input constraints yields only modest improvements, the general method produces reasonably tight and certifiable bounds, even as sequence length increases. The results also underscore the often underestimated impact of initialization errors, an important consideration for applications where models are frequently re-initialized, such as model predictive control (MPC). |
| 2025-09-22 | [MSGAT-GRU: A Multi-Scale Graph Attention and Recurrent Model for Spatiotemporal Road Accident Prediction](http://arxiv.org/abs/2509.17811v1) | Thrinadh Pinjala, Aswin Ram Kumar Gannina et al. | Accurate prediction of road accidents remains challenging due to intertwined spatial, temporal, and contextual factors in urban traffic. We propose MSGAT-GRU, a multi-scale graph attention and recurrent model that jointly captures localized and long-range spatial dependencies while modeling sequential dynamics. Heterogeneous inputs, such as traffic flow, road attributes, weather, and points of interest, are systematically fused to enhance robustness and interpretability. On the Hybrid Beijing Accidents dataset, MSGAT-GRU achieves an RMSE of 0.334 and an F1-score of 0.878, consistently outperforming strong baselines. Cross-dataset evaluation on METR-LA under a 1-hour horizon further supports transferability, with RMSE of 6.48 (vs. 7.21 for the GMAN model) and comparable MAPE. Ablations indicate that three-hop spatial aggregation and a two-layer GRU offer the best accuracy-stability trade-off. These results position MSGAT-GRU as a scalable and generalizable model for intelligent transportation systems, providing interpretable signals that can inform proactive traffic management and road safety analytics. |
| 2025-09-22 | [A Lightweight Approach for State Machine Replication](http://arxiv.org/abs/2509.17771v1) | Christian Cachin, Jinfeng Dou et al. | We present a lightweight solution for state machine replication with commitment certificates. Specifically, we adapt a simple median rule from the stabilizing consensus problem [Doerr11] to operate in a client-server setting where arbitrary servers may be blocked adaptively based on past system information. We further extend our protocol by compressing information about committed commands, thus keeping the protocol lightweight, while still enabling clients to easily prove that their commands have indeed been committed on the shared state. Our approach guarantees liveness as long as at most a constant fraction of servers are blocked, ensures safety under any number of blocked servers, and supports fast recovery from massive blocking attacks. In addition to offering near-optimal performance in several respects, our method is fully decentralized, unlike other near-optimal solutions that rely on leaders. In particular, our solution is robust against adversaries that target key servers (which captures insider-based denial-of-service attacks), whereas leader-based approaches fail under such a blocking model. |
| 2025-09-22 | [EigenSafe: A Spectral Framework for Learning-Based Stochastic Safety Filtering](http://arxiv.org/abs/2509.17750v1) | Inkyu Jang, Jonghae Park et al. | We present EigenSafe, an operator-theoretic framework for learning-enabled safety-critical control for stochastic systems. In many robotic systems where dynamics are best modeled as stochastic systems due to factors such as sensing noise and environmental disturbances, it is challenging for conventional methods such as Hamilton-Jacobi reachability and control barrier functions to provide a holistic measure of safety. We derive a linear operator governing the dynamic programming principle for safety probability, and find that its dominant eigenpair provides information about safety for both individual states and the overall closed-loop system. The proposed learning framework, called EigenSafe, jointly learns this dominant eigenpair and a safe backup policy in an offline manner. The learned eigenfunction is then used to construct a safety filter that detects potentially unsafe situations and falls back to the backup policy. The framework is validated in three simulated stochastic safety-critical control tasks. |
| 2025-09-22 | [An Unlearning Framework for Continual Learning](http://arxiv.org/abs/2509.17530v1) | Sayanta Adhikari, Vishnuprasadh Kumaravelu et al. | Growing concerns surrounding AI safety and data privacy have driven the development of Machine Unlearning as a potential solution. However, current machine unlearning algorithms are designed to complement the offline training paradigm. The emergence of the Continual Learning (CL) paradigm promises incremental model updates, enabling models to learn new tasks sequentially. Naturally, some of those tasks may need to be unlearned to address safety or privacy concerns that might arise. We find that applying conventional unlearning algorithms in continual learning environments creates two critical problems: performance degradation on retained tasks and task relapse, where previously unlearned tasks resurface during subsequent learning. Furthermore, most unlearning algorithms require data to operate, which conflicts with CL's philosophy of discarding past data. A clear need arises for unlearning algorithms that are data-free and mindful of future learning. To that end, we propose UnCLe, an Unlearning framework for Continual Learning. UnCLe employs a hypernetwork that learns to generate task-specific network parameters, using task embeddings. Tasks are unlearned by aligning the corresponding generated network parameters with noise, without requiring any data. Empirical evaluations on several vision data sets demonstrate UnCLe's ability to sequentially perform multiple learning and unlearning operations with minimal disruption to previously acquired knowledge. |
| 2025-09-22 | [Vision-Based Driver Drowsiness Monitoring: Comparative Analysis of YOLOv5-v11 Models](http://arxiv.org/abs/2509.17498v1) | Dilshara Herath, Chinthaka Abeyrathne et al. | Driver drowsiness remains a critical factor in road accidents, accounting for thousands of fatalities and injuries each year. This paper presents a comprehensive evaluation of real-time, non-intrusive drowsiness detection methods, focusing on computer vision based YOLO (You Look Only Once) algorithms. A publicly available dataset namely, UTA-RLDD was used, containing both awake and drowsy conditions, ensuring variability in gender, eyewear, illumination, and skin tone. Seven YOLO variants (v5s, v9c, v9t, v10n, v10l, v11n, v11l) are fine-tuned, with performance measured in terms of Precision, Recall, mAP0.5, and mAP 0.5-0.95. Among these, YOLOv9c achieved the highest accuracy (0.986 mAP 0.5, 0.978 Recall) while YOLOv11n strikes the optimal balance between precision (0.954) and inference efficiency, making it highly suitable for embedded deployment. Additionally, we implement an Eye Aspect Ratio (EAR) approach using Dlib's facial landmarks, which despite its low computational footprint exhibits reduced robustness under pose variation and occlusions. Our findings illustrate clear trade offs between accuracy, latency, and resource requirements, and offer practical guidelines for selecting or combining detection methods in autonomous driving and industrial safety applications. |
| 2025-09-22 | [pBeeGees: A Prudent Approach to Certificate-Decoupled BFT Consensus](http://arxiv.org/abs/2509.17496v1) | Kaiji Yang, Jingjing Zhang et al. | Pipelined Byzantine Fault Tolerant (BFT) consensus is fundamental to permissioned blockchains. However, many existing protocols are limited by the requirement for view-consecutive quorum certificates (QCs). This constraint impairs performance and creates liveness vulnerabilities under adverse network conditions. Achieving "certificate decoupling"-committing blocks without this requirement-is therefore a key research goal. While the recent BeeGees algorithm achieves this, our work reveals that it suffers from security and liveness issues. To address this problem, this paper makes two primary contributions. First, we formally define these flaws as the Invalid Block Problem and the Hollow Chain Problem. Second, we propose pBeeGees, a new algorithm that addresses these issues while preserving certificate decoupling with no additional computational overhead. To achieve this, pBeeGees integrates traceback and pre-commit validation to solve the Invalid Block Problem.Further, to mitigate the Hollow Chain Problem, we introduce a prudent validation mechanism, which prevents unverified branches from growing excessively. To summarize, pBeeGees is the first protocol to simultaneously achieve safety, liveness, and certificate decoupling in a pipelined BFT framework. Experiments confirm that our design significantly reduces block commit latency compared to classic algorithms, particularly under frequent stopping faults. |
| 2025-09-22 | [Privacy in Action: Towards Realistic Privacy Mitigation and Evaluation for LLM-Powered Agents](http://arxiv.org/abs/2509.17488v1) | Shouju Wang, Fenglin Yu et al. | The increasing autonomy of LLM agents in handling sensitive communications, accelerated by Model Context Protocol (MCP) and Agent-to-Agent (A2A) frameworks, creates urgent privacy challenges. While recent work reveals significant gaps between LLMs' privacy Q&A performance and their agent behavior, existing benchmarks remain limited to static, simplified scenarios. We present PrivacyChecker, a model-agnostic, contextual integrity based mitigation approach that effectively reduces privacy leakage from 36.08% to 7.30% on DeepSeek-R1 and from 33.06% to 8.32% on GPT-4o, all while preserving task helpfulness. We also introduce PrivacyLens-Live, transforming static benchmarks into dynamic MCP and A2A environments that reveal substantially higher privacy risks in practical. Our modular mitigation approach integrates seamlessly into agent protocols through three deployment strategies, providing practical privacy protection for the emerging agentic ecosystem. Our data and code will be made available at https://aka.ms/privacy_in_action. |
| 2025-09-22 | [Autiverse: Eliciting Autistic Adolescents' Daily Narratives through AI-guided Multimodal Journaling](http://arxiv.org/abs/2509.17466v1) | Migyeong Yang, Kyungah Lee et al. | Journaling can potentially serve as an effective method for autistic adolescents to improve narrative skills. However, its text-centric nature and high executive functioning demands present barriers to practice. We present Autiverse, an AI-guided multimodal journaling app for tablets that scaffolds storytelling through conversational prompts and visual supports. Autiverse elicits key details through a stepwise dialogue with peer-like, customizable AI and composes them into an editable four-panel comic strip. Through a two-week deployment study with 10 autistic adolescent-parent dyads, we examine how Autiverse supports autistic adolescents to organize their daily experience and emotion. Autiverse helped them construct coherent narratives, while enabling parents to learn additional details of their child's events and emotions. The customized AI peer created a comfortable space for sharing, fostering enjoyment and a strong sense of agency. We discuss the implications of designing technologies that complement autistic adolescents' strengths while ensuring their autonomy and safety in sharing experiences. |
| 2025-09-22 | [SLAyiNG: Towards Queer Language Processing](http://arxiv.org/abs/2509.17449v1) | Leonor Veloso, Lea Hirlimann et al. | Knowledge of slang is a desirable feature of LLMs in the context of user interaction, as slang often reflects an individual's social identity. Several works on informal language processing have defined and curated benchmarks for tasks such as detection and identification of slang. In this paper, we focus on queer slang. Queer slang can be mistakenly flagged as hate speech or can evoke negative responses from LLMs during user interaction. Research efforts so far have not focused explicitly on queer slang. In particular, detection and processing of queer slang have not been thoroughly evaluated due to the lack of a high-quality annotated benchmark. To address this gap, we curate SLAyiNG, the first dataset containing annotated queer slang derived from subtitles, social media posts, and podcasts, reflecting real-world usage. We describe our data curation process, including the collection of slang terms and definitions, scraping sources for examples that reflect usage of these terms, and our ongoing annotation process. As preliminary results, we calculate inter-annotator agreement for human annotators and OpenAI's model o3-mini, evaluating performance on the task of sense disambiguation. Reaching an average Krippendorff's alpha of 0.746, we argue that state-of-the-art reasoning models can serve as tools for pre-filtering, but the complex and often sensitive nature of queer language data requires expert and community-driven annotation efforts. |
| 2025-09-22 | [Distributionally Robust Safety Verification of Neural Networks via Worst-Case CVaR](http://arxiv.org/abs/2509.17413v1) | Masako Kishida | Ensuring the safety of neural networks under input uncertainty is a fundamental challenge in safety-critical applications. This paper builds on and expands Fazlyab's quadratic-constraint (QC) and semidefinite-programming (SDP) framework for neural network verification to a distributionally robust and tail-risk-aware setting by integrating worst-case Conditional Value-at-Risk (WC-CVaR) over a moment-based ambiguity set with fixed mean and covariance. The resulting conditions remain SDP-checkable and explicitly account for tail risk. This integration broadens input-uncertainty geometry-covering ellipsoids, polytopes, and hyperplanes-and extends applicability to safety-critical domains where tail-event severity matters. Applications to closed-loop reachability of control systems and classification are demonstrated through numerical experiments, illustrating how the risk level $\varepsilon$ trades conservatism for tolerance to tail events-while preserving the computational structure of prior QC/SDP methods for neural network verification and robustness analysis. |
| 2025-09-22 | [Multi-Scenario Highway Lane-Change Intention Prediction: A Physics-Informed AI Framework for Three-Class Classification](http://arxiv.org/abs/2509.17354v1) | Jiazhao Shi, Yichen Lin et al. | Lane-change maneuvers are a leading cause of highway accidents, underscoring the need for accurate intention prediction to improve the safety and decision-making of autonomous driving systems. While prior studies using machine learning and deep learning methods (e.g., SVM, CNN, LSTM, Transformers) have shown promise, most approaches remain limited by binary classification, lack of scenario diversity, and degraded performance under longer prediction horizons. In this study, we propose a physics-informed AI framework that explicitly integrates vehicle kinematics, interaction feasibility, and traffic-safety metrics (e.g., distance headway, time headway, time-to-collision, closing gap time) into the learning process. lane-change prediction is formulated as a three-class problem that distinguishes left change, right change, and no change, and is evaluated across both straight highway segments (highD) and complex ramp scenarios (exiD). By integrating vehicle kinematics with interaction features, our machine learning models, particularly LightGBM, achieve state-of-the-art accuracy and strong generalization. Results show up to 99.8% accuracy and 93.6% macro F1 on highD, and 96.1% accuracy and 88.7% macro F1 on exiD at a 1-second horizon, outperforming a two-layer stacked LSTM baseline. These findings demonstrate the practical advantages of a physics-informed and feature-rich machine learning framework for real-time lane-change intention prediction in autonomous driving systems. |
| 2025-09-21 | [Agentic AI for Multi-Stage Physics Experiments at a Large-Scale User Facility Particle Accelerator](http://arxiv.org/abs/2509.17255v1) | Thorsten Hellert, Drew Bertwistle et al. | We present the first language-model-driven agentic artificial intelligence (AI) system to autonomously execute multi-stage physics experiments on a production synchrotron light source. Implemented at the Advanced Light Source particle accelerator, the system translates natural language user prompts into structured execution plans that combine archive data retrieval, control-system channel resolution, automated script generation, controlled machine interaction, and analysis. In a representative machine physics task, we show that preparation time was reduced by two orders of magnitude relative to manual scripting even for a system expert, while operator-standard safety constraints were strictly upheld. Core architectural features, plan-first orchestration, bounded tool access, and dynamic capability selection, enable transparent, auditable execution with fully reproducible artifacts. These results establish a blueprint for the safe integration of agentic AI into accelerator experiments and demanding machine physics studies, as well as routine operations, with direct portability across accelerators worldwide and, more broadly, to other large-scale scientific infrastructures. |
| 2025-09-21 | [Hijacking Living Cells with Surface Engineering for the Internet of Bio-Nano Things](http://arxiv.org/abs/2509.17227v1) | Ekin Ince, Murat Kuscu | The Internet of Bio-Nano Things (IoBNT) promises to revolutionize healthcare by interfacing the cyber domain with the living systems at unprecedented resolution. Realizing this vision hinges on the development of Bio-Nano Things (BNTs), i.e., functional nodes capable of sensing, actuation, and communications within biological environments. Existing BNT architectures, e.g., nanomaterial-based, biosynthetic, and passive molecular agents, face significant limitations, including toxicity, lack of autonomy, or the safety and metabolic burdens associated with genetic modification. This paper posits a fourth paradigm: the transient hijacking of living cells via non-genetic cell surface engineering (NG-CSE) to enable living BNTs. NGCSE allows for the precise, reversible functionalization of cell membranes with synthetic molecular machinery, reprogramming cellular functions and interactions without altering the genome. It uniquely combines the inherent biocompatibility and agency of living cells with the programmability enabled by nanotechnology, mitigating the risks of genetic engineering. We critically review the toolbox of NG-CSE and explore the opportunities it unlocks for IoBNT, including programmable cell-cell communication, dynamic network topologies, and improved bio-cyber interfacing. Moreover, we propose novel IoBNT architectures that leverage these capabilities, such as circulating sentinel networks exploiting cellular agency for continuous liquid biopsy, and rationally designed, in vitro biocomputers exploiting interkingdom interactions. We also outline the critical challenges in modeling and exploiting cellular agency with NG-CSE, providing a roadmap for the effective utilization of NG-CSE-enabled living BNTs within IoBNT. |
| 2025-09-21 | [Optimized adaptive MPC for lateral control of autonomous vehicles](http://arxiv.org/abs/2509.17215v1) | Yassine Kebbati, Vicenç Puig et al. | Autonomous vehicles are the upcoming solution to most transportation problems such as safety, comfort and efficiency. The steering control is one of the main important tasks in achieving autonomous driving. Model predictive control (MPC) is among the fittest controllers for this task due to its optimal performance and ability to handle constraints. This paper proposes an adaptive MPC controller (AMPC) for the path tracking task, and an improved PSO algorithm for optimising the AMPC parameters. Parameter adaption is realised online using a lookup table approach. The propose AMPC performance is assessed and compared with the classic MPC and the Pure Pursuit controller through simulations. Code can be found here: https://github.com/yassinekebbati/Optimized_adaptive_MPC |
| 2025-09-21 | [Combining Performance and Passivity in Linear Control of Series Elastic Actuators](http://arxiv.org/abs/2509.17210v1) | Shaunak A. Mehta, Dylan P. Losey | When humans physically interact with robots, we need the robots to be both safe and performant. Series elastic actuators (SEAs) fundamentally advance safety by introducing compliant actuation. On the one hand, adding a spring mitigates the impact of accidental collisions between human and robot; but on the other hand, this spring introduces oscillations and fundamentally decreases the robot's ability to perform precise, accurate motions. So how should we trade off between physical safety and performance? In this paper, we enumerate the different linear control and mechanical configurations for series elastic actuators, and explore how each choice affects the rendered compliance, passivity, and tracking performance. While prior works focus on load side control, we find that actuator side control has significant benefits. Indeed, simple PD controllers on the actuator side allow for a much wider range of control gains that maintain safety, and combining these with a damper in the elastic transmission yields high performance. Our simulations and real world experiments suggest that, by designing a system with low physical stiffness and high controller gains, this solution enables accurate performance while also ensuring user safety during collisions. |

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



