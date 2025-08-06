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
| 2025-08-05 | [La La LiDAR: Large-Scale Layout Generation from LiDAR Data](http://arxiv.org/abs/2508.03691v1) | Youquan Liu, Lingdong Kong et al. | Controllable generation of realistic LiDAR scenes is crucial for applications such as autonomous driving and robotics. While recent diffusion-based models achieve high-fidelity LiDAR generation, they lack explicit control over foreground objects and spatial relationships, limiting their usefulness for scenario simulation and safety validation. To address these limitations, we propose Large-scale Layout-guided LiDAR generation model ("La La LiDAR"), a novel layout-guided generative framework that introduces semantic-enhanced scene graph diffusion with relation-aware contextual conditioning for structured LiDAR layout generation, followed by foreground-aware control injection for complete scene generation. This enables customizable control over object placement while ensuring spatial and semantic consistency. To support our structured LiDAR generation, we introduce Waymo-SG and nuScenes-SG, two large-scale LiDAR scene graph datasets, along with new evaluation metrics for layout synthesis. Extensive experiments demonstrate that La La LiDAR achieves state-of-the-art performance in both LiDAR generation and downstream perception tasks, establishing a new benchmark for controllable 3D scene generation. |
| 2025-08-05 | [Streaming Generated Gaussian Process Experts for Online Learning and Control](http://arxiv.org/abs/2508.03679v1) | Zewen Yang, Dongfa Zhang et al. | Gaussian Processes (GPs), as a nonparametric learning method, offer flexible modeling capabilities and calibrated uncertainty quantification for function approximations. Additionally, GPs support online learning by efficiently incorporating new data with polynomial-time computation, making them well-suited for safety-critical dynamical systems that require rapid adaptation. However, the inference and online updates of exact GPs, when processing streaming data, incur cubic computation time and quadratic storage memory complexity, limiting their scalability to large datasets in real-time settings. In this paper, we propose a \underline{s}treaming \underline{k}ernel-induced progressivel\underline{y} generated expert framework of \underline{G}aussian \underline{p}rocesses (SkyGP) that addresses both computational and memory constraints by maintaining a bounded set of experts, while inheriting the learning performance guarantees from exact Gaussian processes. Furthermore, two SkyGP variants are introduced, each tailored to a specific objective, either maximizing prediction accuracy (SkyGP-Dense) or improving computational efficiency (SkyGP-Fast). The effectiveness of SkyGP is validated through extensive benchmarks and real-time control experiments demonstrating its superior performance compared to state-of-the-art approaches. |
| 2025-08-05 | [Probing the Gaps in ChatGPT Live Video Chat for Real-World Assistance for People who are Blind or Visually Impaired](http://arxiv.org/abs/2508.03651v1) | Ruei-Che Chang, Rosiana Natalie et al. | Recent advancements in large multimodal models have provided blind or visually impaired (BVI) individuals with new capabilities to interpret and engage with the real world through interactive systems that utilize live video feeds. However, the potential benefits and challenges of such capabilities to support diverse real-world assistive tasks remain unclear. In this paper, we present findings from an exploratory study with eight BVI participants. Participants used ChatGPT's Advanced Voice with Video, a state-of-the-art live video AI released in late 2024, in various real-world scenarios, from locating objects to recognizing visual landmarks, across unfamiliar indoor and outdoor environments. Our findings indicate that current live video AI effectively provides guidance and answers for static visual scenes but falls short in delivering essential live descriptions required in dynamic situations. Despite inaccuracies in spatial and distance information, participants leveraged the provided visual information to supplement their mobility strategies. Although the system was perceived as human-like due to high-quality voice interactions, assumptions about users' visual abilities, hallucinations, generic responses, and a tendency towards sycophancy led to confusion, distrust, and potential risks for BVI users. Based on the results, we discuss implications for assistive video AI agents, including incorporating additional sensing capabilities for real-world use, determining appropriate intervention timing beyond turn-taking interactions, and addressing ecological and safety concerns. |
| 2025-08-05 | [$β$-Ga$_2$O$_3$--Based Radiation Detector for Proton Therapy](http://arxiv.org/abs/2508.03605v1) | Hunter D. Ellis, Imteaz Rahaman et al. | Intensity modulated proton therapy (IMPT) is an advanced cancer treatment modality that offers significant advantages over conventional X-ray therapies, particularly in its ability to minimize radiation dose beyond the tumor target. This reduction in unnecessary irradiation exposure significantly lowers the risk to surrounding healthy tissue and reduces side effects compared to conventional X-ray treatments. However, due to the high complexity of IMPT plans, each plan must be independently validated to ensure the safety and efficacy of the radiation exposure to the patient. While ion chambers are currently used for this purpose, their limitations-particularly in angled-beam measurements and multi-depth assessments-hinder their effectiveness. Silicon-based detectors, commonly used in X-ray therapy, are unsuitable for IMPT due to their rapid degradation under proton irradiation. In this study, a $\beta$-Ga$_2$O$_3$-based metal-semiconductor-metal (MSM) detector was evaluated and compared with a commercial ion chamber using a MEVION S250i proton accelerator. The $\beta$-Ga$_2$O$_3$ detector demonstrated reliable detection of single-pulse proton doses as low as 0.26 MU and exhibited a linear charge-to-dose relationship across a wide range of irradiation conditions. Furthermore, its measurement variability was comparable to that of the ion chamber, with improved sensitivity observed at higher bias voltages. These results highlight the strong potential of $\beta$-Ga$_2$O$_3$ as a radiation-hard detector material for accurate dose verification in IMPT. |
| 2025-08-05 | [A Simple Model of Current Ramp-Up and Ramp-Down in Tokamaks](http://arxiv.org/abs/2508.03561v1) | R. Fitzpatrick | A simple model of the ramp-up and ramp-down of the toroidal current in a tokamak plasma is developed. Faraday's law of electric induction is found to limit how rapidly the current can be safety ramped up or down. It is estimated that the minimum safe ramp-up/down times for the JET, SPARC, and ITER tokamaks are 4.2, 2.0, and 14.7 seconds, respectively. The JET ramp rate is in accordance with operational experience. The SPARC and ITER minimum safe ramp rates are less than the ramp rates in the respective designs. Hence, there is no indication that the design ramp rates are infeasible, as was recently suggested in arXiv:2507.05456 (2025). The typical ratios of the inductive electric field to the Connor-Hastie field in SPARC and ITER are found to be less than those in JET. Thus, the fact that the JET tokamak was able to operate successfully without encountering runaway electron problems during current ramps suggests that the future SPARC and ITER tokamaks should also be able to avoid such problems. |
| 2025-08-05 | [A Robust Cooperative Vehicle Coordination Framework for Intersection Crossing](http://arxiv.org/abs/2508.03417v1) | Haojie Bai, Jiping Luo et al. | Cooperative vehicle coordination at unsignalized intersections has garnered significant interest from both academia and industry in recent years, highlighting its notable advantages in improving traffic throughput and fuel efficiency. However, most existing studies oversimplify the coordination system, assuming accurate vehicle state information and ideal state update process. The oversights pose driving risks in the presence of state uncertainty and communication constraint. To address this gap, we propose a robust and comprehensive intersection coordination framework consisting of a robust cooperative trajectory planner and a context-aware status update scheduler. The trajectory planner directly controls the evolution of the trajectory distributions during frequent vehicle interactions, thereby offering probabilistic safety guarantees. To further align with coordination safety in practical bandwidth-limited conditions, we propose a context-aware status update scheduler that dynamically prioritizes the state updating order of vehicles based on their driving urgency. Simulation results validate the robustness and effectiveness of the proposed coordination framework, showing that the collision probability can be significantly reduced while maintaining comparable coordination efficiency to state-of-theart strategies. Moreover, our proposed framework demonstrates superior effectiveness in utilizing wireless resources in practical uncertain and bandwidth-limited conditions. |
| 2025-08-05 | [Psychological safety in software workplaces: A systematic literature review](http://arxiv.org/abs/2508.03369v1) | Beatriz Santana, Lidivânio Monte et al. | Context: Psychological safety (PS) is an important factor influencing team well-being and performance, particularly in collaborative and dynamic domains such as software development. Despite its acknowledged significance, research on PS within the field of software engineering remains limited. The socio-technical complexities and fast-paced nature of software development present challenges to cultivating PS. To the best of our knowledge, no systematic secondary study has synthesized existing knowledge on PS in the context of software engineering.   Objective: This study aims to systematically review and synthesize the existing body of knowledge on PS in software engineering. Specifically, it seeks to identify the potential antecedents and consequences associated with the presence or absence of PS among individuals involved in the software development process.   Methods: A systematic literature review was conducted, encompassing studies retrieved from four digital libraries. The extracted data were subjected to both quantitative and qualitative analyses.   Results: The findings indicate a growing academic interest in PS within software engineering, with the majority of studies grounded in Edmondson's framework. Factors antecedents of PS were identified at the individual, team, and organizational levels, including team autonomy, agile methodologies, and leadership behaviors.   Conclusion: PS fosters innovation, learning, and team performance within software development. However, significant gaps persist in understanding the contextual factors influencing PS, its underlying mechanisms, and effective strategies for its enhancement. Future research should address these gaps by investigating the practical applications of PS within diverse organizational settings in the software engineering domain. |
| 2025-08-05 | [When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs](http://arxiv.org/abs/2508.03365v1) | Bodam Kim, Hiskias Dingeto et al. | As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior. |
| 2025-08-05 | [Compressing Chain-of-Thought in LLMs via Step Entropy](http://arxiv.org/abs/2508.03346v1) | Zeju Li, Jianyuan Zhong et al. | Large Language Models (LLMs) using Chain-of-Thought (CoT) prompting excel at complex reasoning but generate verbose thought processes with considerable redundancy, leading to increased inference costs and reduced efficiency. We introduce a novel CoT compression framework based on step entropy, a metric that quantifies the informational contribution of individual reasoning steps to identify redundancy. Through theoretical analysis and extensive empirical validation on mathematical reasoning benchmarks, we demonstrate that steps with low entropy are indeed highly redundant. Our experiments reveal that an astonishing 80\% of low-entropy intermediate steps can be pruned with minor degradation in the final answer accuracy across DeepSeek-R1-7B, 14B and Qwen3-8B. This finding sharply contrasts with random or high-entropy pruning, which severely impairs reasoning performance. Building on this, we propose a novel two-stage training strategy combining Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) reinforcement learning. This approach enables LLMs to autonomously learn to generate compressed COTs during inference by strategically incorporating [SKIP] tokens. Our method significantly enhances LLM inference efficiency while rigorously preserving accuracy, offering profound implications for practical LLM deployment and a deeper understanding of reasoning structures. |
| 2025-08-05 | [Towards Trustworthy Multimodal Moderation via Policy-Aligned Reasoning and Hierarchical Labeling](http://arxiv.org/abs/2508.03296v1) | Anqi Li, Wenwei Jin et al. | Social platforms have revolutionized information sharing, but also accelerated the dissemination of harmful and policy-violating content. To ensure safety and compliance at scale, moderation systems must go beyond efficiency and offer accuracy and interpretability. However, current approaches largely rely on noisy, label-driven learning, lacking alignment with moderation rules and producing opaque decisions that hinder human review. Therefore, we propose Hierarchical Guard (Hi-Guard), a multimodal moderation framework that introduces a new policy-aligned decision paradigm. The term "Hierarchical" reflects two key aspects of our system design: (1) a hierarchical moderation pipeline, where a lightweight binary model first filters safe content and a stronger model handles fine-grained risk classification; and (2) a hierarchical taxonomy in the second stage, where the model performs path-based classification over a hierarchical taxonomy ranging from coarse to fine-grained levels. To ensure alignment with evolving moderation policies, Hi-Guard directly incorporates rule definitions into the model prompt. To further enhance structured prediction and reasoning, we introduce a multi-level soft-margin reward and optimize with Group Relative Policy Optimization (GRPO), penalizing semantically adjacent misclassifications and improving explanation quality. Extensive experiments and real-world deployment demonstrate that Hi-Guard achieves superior classification accuracy, generalization, and interpretability, paving the way toward scalable, transparent, and trustworthy content safety systems. Code is available at: https://github.com/lianqi1008/Hi-Guard. |
| 2025-08-05 | [Towards Interpretable Concept Learning over Time Series via Temporal Logic Semantics](http://arxiv.org/abs/2508.03269v1) | Irene Ferfoglia, Simone Silvetti et al. | Time series classification is a task of paramount importance, as this kind of data often arises in safety-critical applications. However, it is typically tackled with black-box deep learning methods, making it hard for humans to understand the rationale behind their output. To take on this challenge, we propose a neuro-symbolic framework that unifies classification and explanation through direct embedding of trajectories into a space of Signal Temporal Logic (STL) concepts. By introducing a novel STL-inspired kernel that maps raw time series to their alignment with predefined STL formulae, our model jointly optimises for accuracy and interpretability, as each prediction is accompanied by the most relevant logical concepts that characterise it. This enables classification grounded in human-interpretable temporal patterns and produces both local and global symbolic explanations. Early results show competitive performance while offering high-quality logical justifications for model decisions. |
| 2025-08-05 | [Force-Compliance MPC and Robot-User CBFs for Interactive Navigation and User-Robot Safety in Hexapod Guide Robots](http://arxiv.org/abs/2508.03246v1) | Zehua Fan, Feng Gao et al. | Guiding the visually impaired in complex environments requires real-time two-way interaction and safety assurance. We propose a Force-Compliance Model Predictive Control (FC-MPC) and Robot-User Control Barrier Functions (CBFs) for force-compliant navigation and obstacle avoidance in Hexapod guide robots. FC-MPC enables two-way interaction by estimating user-applied forces and moments using the robot's dynamic model and the recursive least squares (RLS) method, and then adjusting the robot's movements accordingly, while Robot-User CBFs ensure the safety of both the user and the robot by handling static and dynamic obstacles, and employ weighted slack variables to overcome feasibility issues in complex dynamic environments. We also adopt an Eight-Way Connected DBSCAN method for obstacle clustering, reducing computational complexity from O(n2) to approximately O(n), enabling real-time local perception on resource-limited on-board robot computers. Obstacles are modeled using Minimum Bounding Ellipses (MBEs), and their trajectories are predicted through Kalman filtering. Implemented on the HexGuide robot, the system seamlessly integrates force compliance, autonomous navigation, and obstacle avoidance. Experimental results demonstrate the system's ability to adapt to user force commands while guaranteeing user and robot safety simultaneously during navigation in complex environments. |
| 2025-08-05 | [Light-IF: Endowing LLMs with Generalizable Reasoning via Preview and Self-Checking for Complex Instruction Following](http://arxiv.org/abs/2508.03178v1) | Chenyang Wang, Liang Wen et al. | While advancements in the reasoning abilities of LLMs have significantly enhanced their performance in solving mathematical problems, coding tasks, and general puzzles, their effectiveness in accurately adhering to instructions remains inconsistent, particularly with more complex directives. Our investigation identifies lazy reasoning during the thinking stage as the primary factor contributing to poor instruction adherence. To mitigate this issue, we propose a comprehensive framework designed to enable rigorous reasoning processes involving preview and self-checking, essential for satisfying strict instruction constraints. Specifically, we first generate instructions with complex constraints and apply a filtering process to obtain valid prompts, resulting in three distinct prompt datasets categorized as hard, easy, and pass. Then, we employ rejection sampling on the pass prompts to curate a small yet high-quality dataset, enabling a cold-start initialization of the model and facilitating its adaptation to effective reasoning patterns. Subsequently, we employ an entropy-preserving supervised fine-tuning (Entropy-SFT) strategy coupled with token-wise entropy-adaptive (TEA-RL) reinforcement learning guided by rule-based dense rewards. This approach encourages the model to transform its reasoning mechanism, ultimately fostering generalizable reasoning abilities that encompass preview and self-checking. Extensive experiments conducted on instruction-following benchmarks demonstrate remarkable performance improvements across various model scales. Notably, our Light-IF-32B model surpasses both larger open-source models such as DeepSeek-R1 and closed-source models like Doubao-1.6. |
| 2025-08-05 | [CoTox: Chain-of-Thought-Based Molecular Toxicity Reasoning and Prediction](http://arxiv.org/abs/2508.03159v1) | Jueon Park, Yein Park et al. | Drug toxicity remains a major challenge in pharmaceutical development. Recent machine learning models have improved in silico toxicity prediction, but their reliance on annotated data and lack of interpretability limit their applicability. This limits their ability to capture organ-specific toxicities driven by complex biological mechanisms. Large language models (LLMs) offer a promising alternative through step-by-step reasoning and integration of textual data, yet prior approaches lack biological context and transparent rationale. To address this issue, we propose CoTox, a novel framework that integrates LLM with chain-of-thought (CoT) reasoning for multi-toxicity prediction. CoTox combines chemical structure data, biological pathways, and gene ontology (GO) terms to generate interpretable toxicity predictions through step-by-step reasoning. Using GPT-4o, we show that CoTox outperforms both traditional machine learning and deep learning model. We further examine its performance across various LLMs to identify where CoTox is most effective. Additionally, we find that representing chemical structures with IUPAC names, which are easier for LLMs to understand than SMILES, enhances the model's reasoning ability and improves predictive performance. To demonstrate its practical utility in drug development, we simulate the treatment of relevant cell types with drug and incorporated the resulting biological context into the CoTox framework. This approach allow CoTox to generate toxicity predictions aligned with physiological responses, as shown in case study. This result highlights the potential of LLM-based frameworks to improve interpretability and support early-stage drug safety assessment. The code and prompt used in this work are available at https://github.com/dmis-lab/CoTox. |
| 2025-08-05 | [Estimating Worst-Case Frontier Risks of Open-Weight LLMs](http://arxiv.org/abs/2508.03153v1) | Eric Wallace, Olivia Watkins et al. | In this paper, we study the worst-case frontier risks of releasing gpt-oss. We introduce malicious fine-tuning (MFT), where we attempt to elicit maximum capabilities by fine-tuning gpt-oss to be as capable as possible in two domains: biology and cybersecurity. To maximize biological risk (biorisk), we curate tasks related to threat creation and train gpt-oss in an RL environment with web browsing. To maximize cybersecurity risk, we train gpt-oss in an agentic coding environment to solve capture-the-flag (CTF) challenges. We compare these MFT models against open- and closed-weight LLMs on frontier risk evaluations. Compared to frontier closed-weight models, MFT gpt-oss underperforms OpenAI o3, a model that is below Preparedness High capability level for biorisk and cybersecurity. Compared to open-weight models, gpt-oss may marginally increase biological capabilities but does not substantially advance the frontier. Taken together, these results contributed to our decision to release the model, and we hope that our MFT approach can serve as useful guidance for estimating harm from future open-weight releases. |
| 2025-08-05 | [Safety-Aware Imitation Learning via MPC-Guided Disturbance Injection](http://arxiv.org/abs/2508.03129v1) | Le Qiu, Yusuf Umut Ciftci et al. | Imitation Learning has provided a promising approach to learning complex robot behaviors from expert demonstrations. However, learned policies can make errors that lead to safety violations, which limits their deployment in safety-critical applications. We propose MPC-SafeGIL, a design-time approach that enhances the safety of imitation learning by injecting adversarial disturbances during expert demonstrations. This exposes the expert to a broader range of safety-critical scenarios and allows the imitation policy to learn robust recovery behaviors. Our method uses sampling-based Model Predictive Control (MPC) to approximate worst-case disturbances, making it scalable to high-dimensional and black-box dynamical systems. In contrast to prior work that relies on analytical models or interactive experts, MPC-SafeGIL integrates safety considerations directly into data collection. We validate our approach through extensive simulations including quadruped locomotion and visuomotor navigation and real-world experiments on a quadrotor, demonstrating improvements in both safety and task performance. See our website here: https://leqiu2003.github.io/MPCSafeGIL/ |
| 2025-08-05 | [Attack the Messages, Not the Agents: A Multi-round Adaptive Stealthy Tampering Framework for LLM-MAS](http://arxiv.org/abs/2508.03125v1) | Bingyu Yan, Ziyi Zhou et al. | Large language model-based multi-agent systems (LLM-MAS) effectively accomplish complex and dynamic tasks through inter-agent communication, but this reliance introduces substantial safety vulnerabilities. Existing attack methods targeting LLM-MAS either compromise agent internals or rely on direct and overt persuasion, which limit their effectiveness, adaptability, and stealthiness. In this paper, we propose MAST, a Multi-round Adaptive Stealthy Tampering framework designed to exploit communication vulnerabilities within the system. MAST integrates Monte Carlo Tree Search with Direct Preference Optimization to train an attack policy model that adaptively generates effective multi-round tampering strategies. Furthermore, to preserve stealthiness, we impose dual semantic and embedding similarity constraints during the tampering process. Comprehensive experiments across diverse tasks, communication architectures, and LLMs demonstrate that MAST consistently achieves high attack success rates while significantly enhancing stealthiness compared to baselines. These findings highlight the effectiveness, stealthiness, and adaptability of MAST, underscoring the need for robust communication safeguards in LLM-MAS. |
| 2025-08-05 | [Beyond Surface-Level Detection: Towards Cognitive-Driven Defense Against Jailbreak Attacks via Meta-Operations Reasoning](http://arxiv.org/abs/2508.03054v1) | Rui Pu, Chaozhuo Li et al. | Defending large language models (LLMs) against jailbreak attacks is essential for their safe and reliable deployment. Existing defenses often rely on shallow pattern matching, which struggles to generalize to novel and unseen attack strategies. To address this challenge, we propose the Cognitive-Driven Defense (CDD) framework, which targets the underlying structure of jailbreak prompts by applying meta-operations, defined as basic manipulations that conceal harmful intent.CDD emulates human cognitive reasoning through a structured reasoning chain. It begins with a global perception of the prompt and follows with a localized analysis to uncover hidden manipulations. By applying supervised fine-tuning on this structured chain, the model learns to identify and reason about known manipulation patterns. To enhance generalization to unseen threats, an entropy-guided reinforcement learning algorithm (EG-GRPO) is introduced to encourage exploration of new types and variants of meta-operations. Experiments demonstrate that CDD can achieve state-of-the-art defense performance and exhibit strong generalization to unseen jailbreak attacks. |
| 2025-08-05 | [CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors](http://arxiv.org/abs/2508.02997v1) | Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis | The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models. To support future research and reproducibility, we have made our implementation publicly available. |
| 2025-08-05 | [When AIs Judge AIs: The Rise of Agent-as-a-Judge Evaluation for LLMs](http://arxiv.org/abs/2508.02994v1) | Fangyi Yu | As large language models (LLMs) grow in capability and autonomy, evaluating their outputs-especially in open-ended and complex tasks-has become a critical bottleneck. A new paradigm is emerging: using AI agents as the evaluators themselves. This "agent-as-a-judge" approach leverages the reasoning and perspective-taking abilities of LLMs to assess the quality and safety of other models, promising calable and nuanced alternatives to human evaluation. In this review, we define the agent-as-a-judge concept, trace its evolution from single-model judges to dynamic multi-agent debate frameworks, and critically examine their strengths and shortcomings. We compare these approaches across reliability, cost, and human alignment, and survey real-world deployments in domains such as medicine, law, finance, and education. Finally, we highlight pressing challenges-including bias, robustness, and meta evaluation-and outline future research directions. By bringing together these strands, our review demonstrates how agent-based judging can complement (but not replace) human oversight, marking a step toward trustworthy, scalable evaluation for next-generation LLMs. |

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



