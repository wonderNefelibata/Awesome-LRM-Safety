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
| 2026-02-26 | [Safety First: Psychological Safety as the Key to AI Transformation](http://arxiv.org/abs/2602.23279v1) | Aaron Reich, Diana Wolfe et al. | Organizations continue to invest in artificial intelligence, yet many struggle to ensure that employees adopt and engage with these tools. Drawing on research highlighting the interpersonal and learning demands of technology use, this study examines whether psychological safety is associated with AI adoption and usage in the workplace. Using survey data from 2,257 employees in a global consulting firm, we test whether psychological safety is associated with adoption, usage frequency, and usage duration; and whether these relationships vary by organizational level, professional experience, or geographic region. Logistic and linear regression analyses show that psychological safety reliably predicts whether employees adopt AI tools but does not predict how often or how long they use AI once adoption has occurred. Moreover, the relationship between psychological safety and AI adoption is consistent across experience levels, role levels, and regions, and no moderation effects emerge. These findings suggest that psychological safety functions as a key antecedent of initial AI engagement but not of subsequent usage intensity. The study underscores the need to distinguish between adoption and sustained use and highlights opportunities for targeted organizational interventions in early-stage AI implementation. |
| 2026-02-26 | [CXReasonAgent: Evidence-Grounded Diagnostic Reasoning Agent for Chest X-rays](http://arxiv.org/abs/2602.23276v1) | Hyungyung Lee, Hangyul Yoon et al. | Chest X-ray plays a central role in thoracic diagnosis, and its interpretation inherently requires multi-step, evidence-grounded reasoning. However, large vision-language models (LVLMs) often generate plausible responses that are not faithfully grounded in diagnostic evidence and provide limited visual evidence for verification, while also requiring costly retraining to support new diagnostic tasks, limiting their reliability and adaptability in clinical settings. To address these limitations, we present CXReasonAgent, a diagnostic agent that integrates a large language model (LLM) with clinically grounded diagnostic tools to perform evidence-grounded diagnostic reasoning using image-derived diagnostic and visual evidence. To evaluate these capabilities, we introduce CXReasonDial, a multi-turn dialogue benchmark with 1,946 dialogues across 12 diagnostic tasks, and show that CXReasonAgent produces faithfully grounded responses, enabling more reliable and verifiable diagnostic reasoning than LVLMs. These findings highlight the importance of integrating clinically grounded diagnostic tools, particularly in safety-critical clinical settings. |
| 2026-02-26 | [Phys-3D: Physics-Constrained Real-Time Crowd Tracking and Counting on Railway Platforms](http://arxiv.org/abs/2602.23177v1) | Bin Zeng, Johannes Künzel et al. | Accurate, real-time crowd counting on railway platforms is essential for safety and capacity management. We propose to use a single camera mounted in a train, scanning the platform while arriving. While hardware constraints are simple, counting remains challenging due to dense occlusions, camera motion, and perspective distortions during train arrivals. Most existing tracking-by-detection approaches assume static cameras or ignore physical consistency in motion modeling, leading to unreliable counting under dynamic conditions. We propose a physics-constrained tracking framework that unifies detection, appearance, and 3D motion reasoning in a real-time pipeline. Our approach integrates a transfer-learned YOLOv11m detector with EfficientNet-B0 appearance encoding within DeepSORT, while introducing a physics-constrained Kalman model (Phys-3D) that enforces physically plausible 3D motion dynamics through pinhole geometry. To address counting brittleness under occlusions, we implement a virtual counting band with persistence. On our platform benchmark, MOT-RailwayPlatformCrowdHead Dataset(MOT-RPCH), our method reduces counting error to 2.97%, demonstrating robust performance despite motion and occlusions. Our results show that incorporating first-principles geometry and motion priors enables reliable crowd counting in safety-critical transportation scenarios, facilitating effective train scheduling and platform safety management. |
| 2026-02-26 | [Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios](http://arxiv.org/abs/2602.23109v1) | Kai Chen, Yuyao Huang et al. | The sudden appearance of occluded pedestrians presents a critical safety challenge in autonomous driving. Conventional rule-based or purely data-driven approaches struggle with the inherent high uncertainty of these long-tail scenarios. To tackle this challenge, we propose a novel framework grounded in Active Inference, which endows the agent with a human-like, belief-driven mechanism. Our framework leverages a Rao-Blackwellized Particle Filter (RBPF) to efficiently estimate the pedestrian's hybrid state. To emulate human-like cognitive processes under uncertainty, we introduce a Conditional Belief Reset mechanism and a Hypothesis Injection technique to explicitly model beliefs about the pedestrian's multiple latent intentions. Planning is achieved via a Cross-Entropy Method (CEM) enhanced Model Predictive Path Integral (MPPI) controller, which synergizes the efficient, iterative search of CEM with the inherent robustness of MPPI. Simulation experiments demonstrate that our approach significantly reduces the collision rate compared to reactive, rule-based, and reinforcement learning (RL) baselines, while also exhibiting explainable and human-like driving behavior that reflects the agent's internal belief state. |
| 2026-02-26 | [An Empirical Analysis of Cooperative Perception for Occlusion Risk Mitigation](http://arxiv.org/abs/2602.23051v1) | Aihong Wang, Tenghui Xie et al. | Occlusions present a significant challenge for connected and automated vehicles, as they can obscure critical road users from perception systems. Traditional risk metrics often fail to capture the cumulative nature of these threats over time adequately. In this paper, we propose a novel and universal risk assessment metric, the Risk of Tracking Loss (RTL), which aggregates instantaneous risk intensity throughout occluded periods. This provides a holistic risk profile that encompasses both high-intensity, short-term threats and prolonged exposure. Utilizing diverse and high-fidelity real-world datasets, a large-scale statistical analysis is conducted to characterize occlusion risk and validate the effectiveness of the proposed metric. The metric is applied to evaluate different vehicle-to-everything (V2X) deployment strategies. Our study shows that full V2X penetration theoretically eliminates this risk, the reduction is highly nonlinear; a substantial statistical benefit requires a high penetration threshold of 75-90%. To overcome this limitation, we propose a novel asymmetric communication framework that allows even non-connected vehicles to receive warnings. Experimental results demonstrate that this paradigm achieves better risk mitigation performance. We found that our approach at 25% penetration outperforms the traditional symmetric model at 75%, and benefits saturate at only 50% penetration. This work provides a crucial risk assessment metric and a cost-effective, strategic roadmap for accelerating the safety benefits of V2X deployment. |
| 2026-02-26 | [Managing Uncertainty in LLM-based Multi-Agent System Operation](http://arxiv.org/abs/2602.23005v1) | Man Zhang, Tao Yue et al. | Applying LLM-based multi-agent software systems in safety-critical domains such as lifespan echocardiography introduces system-level risks that cannot be addressed by improving model accuracy alone. During system operation, beyond individual LLM behavior, uncertainty propagates through agent coordination, data pipelines, human-in-the-loop interaction, and runtime control logic. Yet existing work largely treats uncertainty at the model level rather than as a first-class software engineering concern. This paper approaches uncertainty from both system-level and runtime perspectives. We first differentiate epistemological and ontological uncertainties in the context of LLM-based multi-agent software system operation. Building on this foundation, we propose a lifecycle-based uncertainty management framework comprising four mechanisms: representation, identification, evolution, and adaptation. The uncertainty lifecycle governs how uncertainties emerge, transform, and are mitigated across architectural layers and execution phases, enabling structured runtime governance and controlled adaptation. We demonstrate the feasibility of the framework using a real-world LLM-based multi-agent echocardiographic software system developed in clinical collaboration, showing improved reliability and diagnosability in diagnostic reasoning. The proposed approach generalizes to other safety-critical LLM-based multi-agent software systems, supporting principled operational control and runtime assurance beyond model-centric methods. |
| 2026-02-26 | [Obscure but Effective: Classical Chinese Jailbreak Prompt Optimization via Bio-Inspired Search](http://arxiv.org/abs/2602.22983v1) | Xun Huang, Simeng Qin et al. | As Large Language Models (LLMs) are increasingly used, their security risks have drawn increasing attention. Existing research reveals that LLMs are highly susceptible to jailbreak attacks, with effectiveness varying across language contexts. This paper investigates the role of classical Chinese in jailbreak attacks. Owing to its conciseness and obscurity, classical Chinese can partially bypass existing safety constraints, exposing notable vulnerabilities in LLMs. Based on this observation, this paper proposes a framework, CC-BOS, for the automatic generation of classical Chinese adversarial prompts based on multi-dimensional fruit fly optimization, facilitating efficient and automated jailbreak attacks in black-box settings. Prompts are encoded into eight policy dimensions-covering role, behavior, mechanism, metaphor, expression, knowledge, trigger pattern and context; and iteratively refined via smell search, visual search, and cauchy mutation. This design enables efficient exploration of the search space, thereby enhancing the effectiveness of black-box jailbreak attacks. To enhance readability and evaluation accuracy, we further design a classical Chinese to English translation module. Extensive experiments demonstrate that effectiveness of the proposed CC-BOS, consistently outperforming state-of-the-art jailbreak attack methods. |
| 2026-02-26 | [Modeling Expert AI Diagnostic Alignment via Immutable Inference Snapshots](http://arxiv.org/abs/2602.22973v1) | Dimitrios P. Panagoulias, Evangelia-Aikaterini Tsichrintzi et al. | Human-in-the-loop validation is essential in safety-critical clinical AI, yet the transition between initial model inference and expert correction is rarely analyzed as a structured signal. We introduce a diagnostic alignment framework in which the AI-generated image based report is preserved as an immutable inference state and systematically compared with the physician-validated outcome. The inference pipeline integrates a vision-enabled large language model, BERT- based medical entity extraction, and a Sequential Language Model Inference (SLMI) step to enforce domain-consistent refinement prior to expert review. Evaluation on 21 dermatological cases (21 complete AI physician pairs) em- ployed a four-level concordance framework comprising exact primary match rate (PMR), semantic similarity-adjusted rate (AMR), cross-category alignment, and Comprehensive Concordance Rate (CCR). Exact agreement reached 71.4% and remained unchanged under semantic similarity (t = 0.60), while structured cross-category and differential overlap analysis yielded 100% comprehensive concordance (95% CI: [83.9%, 100%]). No cases demonstrated complete diagnostic divergence. These findings show that binary lexical evaluation substantially un- derestimates clinically meaningful alignment. Modeling expert validation as a structured transformation enables signal-aware quantification of correction dynamics and supports traceable, human aligned evaluation of image based clinical decision support systems. |
| 2026-02-26 | [OSDaR-AR: Enhancing Railway Perception Datasets via Multi-modal Augmented Reality](http://arxiv.org/abs/2602.22920v1) | Federico Nesti, Gianluca D'Amico et al. | Although deep learning has significantly advanced the perception capabilities of intelligent transportation systems, railway applications continue to suffer from a scarcity of high-quality, annotated data for safety-critical tasks like obstacle detection. While photorealistic simulators offer a solution, they often struggle with the ``sim-to-real" gap; conversely, simple image-masking techniques lack the spatio-temporal coherence required to obtain augmented single- and multi-frame scenes with the correct appearance and dimensions. This paper introduces a multi-modal augmented reality framework designed to bridge this gap by integrating photorealistic virtual objects into real-world railway sequences from the OSDaR23 dataset. Utilizing Unreal Engine 5 features, our pipeline leverages LiDAR point-clouds and INS/GNSS data to ensure accurate object placement and temporal stability across RGB frames. This paper also proposes a segmentation-based refinement strategy for INS/GNSS data to significantly improve the realism of the augmented sequences, as confirmed by the comparative study presented in the paper. Carefully designed augmented sequences are collected to produce OSDaR-AR, a public dataset designed to support the development of next-generation railway perception systems. The dataset is available at the following page: https://syndra.retis.santannapisa.it/osdarar.html |
| 2026-02-26 | [Input-Envelope-Output: Auditable Generative Music Rewards in Sensory-Sensitive Contexts](http://arxiv.org/abs/2602.22813v1) | Cong Ye, Songlin Shang et al. | Generative feedback in sensory-sensitive contexts poses a core design challenge: large individual differences in sensory tolerance make it difficult to sustain engagement without compromising safety. This tension is exemplified in autism spectrum disorder (ASD), where auditory sensitivities are common yet highly heterogeneous. Existing interactive music systems typically encode safety implicitly within direct input-output (I-O) mappings, which can preserve novelty but make system behavior hard to predict or audit. We instead propose a constraint-first Input-Envelope-Output (I-E-O) framework that makes safety explicit and verifiable while preserving action-output causality. I-E-O introduces a low-risk envelope layer between user input and audio output to specify safe bounds, enforce them deterministically, and log interventions for audit. From this architecture, we derive four verifiable design principles and instantiate them in MusiBubbles, a web-based prototype. Contributions include the I-E-O architecture, MusiBubbles as an exemplar implementation, and a reproducibility package to support adoption in ASD and other sensory-sensitive domains. |
| 2026-02-26 | [Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving](http://arxiv.org/abs/2602.22801v1) | Yinan Zheng, Tianyi Tan et al. | Diffusion models have become a popular choice for decision-making tasks in robotics, and more recently, are also being considered for solving autonomous driving tasks. However, their applications and evaluations in autonomous driving remain limited to simulation-based or laboratory settings. The full strength of diffusion models for large-scale, complex real-world settings, such as End-to-End Autonomous Driving (E2E AD), remains underexplored. In this study, we conducted a systematic and large-scale investigation to unleash the potential of the diffusion models as planners for E2E AD, based on a tremendous amount of real-vehicle data and road testing. Through comprehensive and carefully controlled studies, we identify key insights into the diffusion loss space, trajectory representation, and data scaling that significantly impact E2E planning performance. Moreover, we also provide an effective reinforcement learning post-training strategy to further enhance the safety of the learned planner. The resulting diffusion-based learning framework, Hyper Diffusion Planner} (HDP), is deployed on a real-vehicle platform and evaluated across 6 urban driving scenarios and 200 km of real-world testing, achieving a notable 10x performance improvement over the base model. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks. |
| 2026-02-26 | [TherapyProbe: Generating Design Knowledge for Relational Safety in Mental Health Chatbots Through Adversarial Simulation](http://arxiv.org/abs/2602.22775v1) | Joydeep Chandra, Satyam Kumar Navneet et al. | As mental health chatbots proliferate to address the global treatment gap, a critical question emerges: How do we design for relational safety the quality of interaction patterns that unfold across conversations rather than the correctness of individual responses? Current safety evaluations assess single-turn crisis responses, missing the therapeutic dynamics that determine whether chatbots help or harm over time. We introduce TherapyProbe, a design probe methodology that generates actionable design knowledge by systematically exploring chatbot conversation trajectories through adversarial multi-agent simulation. Using open-source models, TherapyProbe surfaces relational safety failures interaction patterns like "validation spirals" where chatbots progressively reinforce hopelessness, or "empathy fatigue" where responses become mechanical over turns. Our contribution is translating these failures into a Safety Pattern Library of 23 failure archetypes with corresponding design recommendations. We contribute: (1) a replicable methodology requiring no API costs, (2) a clinically-grounded failure taxonomy, and (3) design implications for developers, clinicians, and policymakers. |
| 2026-02-26 | [ClinDet-Bench: Beyond Abstention, Evaluating Judgment Determinability of LLMs in Clinical Decision-Making](http://arxiv.org/abs/2602.22771v1) | Yusuke Watanabe, Yohei Kobashi et al. | Clinical decisions are often required under incomplete information. Clinical experts must identify whether available information is sufficient for judgment, as both premature conclusion and unnecessary abstention can compromise patient safety. To evaluate this capability of large language models (LLMs), we developed ClinDet-Bench, a benchmark based on clinical scoring systems that decomposes incomplete-information scenarios into determinable and undeterminable conditions. Identifying determinability requires considering all hypotheses about missing information, including unlikely ones, and verifying whether the conclusion holds across them. We find that recent LLMs fail to identify determinability under incomplete information, producing both premature judgments and excessive abstention, despite correctly explaining the underlying scoring knowledge and performing well under complete information. These findings suggest that existing benchmarks are insufficient to evaluate the safety of LLMs in clinical settings. ClinDet-Bench provides a framework for evaluating determinability recognition, leading to appropriate abstention, with potential applicability to medicine and other high-stakes domains, and is publicly available. |
| 2026-02-26 | [RLHFless: Serverless Computing for Efficient RLHF](http://arxiv.org/abs/2602.22718v1) | Rui Wei, Hanfei Yu et al. | Reinforcement Learning from Human Feedback (RLHF) has been widely applied to Large Language Model (LLM) post-training to align model outputs with human preferences. Recent models, such as DeepSeek-R1, have also shown RLHF's potential to improve LLM reasoning on complex tasks. In RL, inference and training co-exist, creating dynamic resource demands throughout the workflow. Compared to traditional RL, RLHF further challenges training efficiency due to expanding model sizes and resource consumption. Several RLHF frameworks aim to balance flexible abstraction and efficient execution. However, they rely on serverful infrastructures, which struggle with fine-grained resource variability. As a result, during synchronous RLHF training, idle time between or within RL components often causes overhead and resource wastage.   To address these issues, we present RLHFless, the first scalable training framework for synchronous RLHF, built on serverless computing environments. RLHFless adapts to dynamic resource demands throughout the RLHF pipeline, pre-computes shared prefixes to avoid repeated computation, and uses a cost-aware actor scaling strategy that accounts for response length variation to find sweet spots with lower cost and higher speed. In addition, RLHFless assigns workloads efficiently to reduce intra-function imbalance and idle time. Experiments on both physical testbeds and a large-scale simulated cluster show that RLHFless achieves up to 1.35x speedup and 44.8% cost reduction compared to the state-of-the-art baseline. |
| 2026-02-26 | [Knob: A Physics-Inspired Gating Interface for Interpretable and Controllable Neural Dynamics](http://arxiv.org/abs/2602.22702v1) | Siyu Jiang, Sanshuai Cui et al. | Existing neural network calibration methods often treat calibration as a static, post-hoc optimization task. However, this neglects the dynamic and temporal nature of real-world inference. Moreover, existing methods do not provide an intuitive interface enabling human operators to dynamically adjust model behavior under shifting conditions. In this work, we propose Knob, a framework that connects deep learning with classical control theory by mapping neural gating dynamics to a second-order mechanical system. By establishing correspondences between physical parameters -- damping ratio ($ζ$) and natural frequency ($ω_n$) -- and neural gating, we create a tunable "safety valve". The core mechanism employs a logit-level convex fusion, functioning as an input-adaptive temperature scaling. It tends to reduce model confidence particularly when model branches produce conflicting predictions. Furthermore, by imposing second-order dynamics (Knob-ODE), we enable a \textit{dual-mode} inference: standard i.i.d. processing for static tasks, and state-preserving processing for continuous streams. Our framework allows operators to tune "stability" and "sensitivity" through familiar physical analogues. This paper presents an exploratory architectural interface; we focus on demonstrating the concept and validating its control-theoretic properties rather than claiming state-of-the-art calibration performance. Experiments on CIFAR-10-C validate the calibration mechanism and demonstrate that, in Continuous Mode, the gate responses are consistent with standard second-order control signatures (step settling and low-pass attenuation), paving the way for predictable human-in-the-loop tuning. |
| 2026-02-26 | [TorchLean: Formalizing Neural Networks in Lean](http://arxiv.org/abs/2602.22631v1) | Robert Joseph George, Jennifer Cruden et al. | Neural networks are increasingly deployed in safety- and mission-critical pipelines, yet many verification and analysis results are produced outside the programming environment that defines and runs the model. This separation creates a semantic gap between the executed network and the analyzed artifact, so guarantees can hinge on implicit conventions such as operator semantics, tensor layouts, preprocessing, and floating-point corner cases. We introduce TorchLean, a framework in the Lean 4 theorem prover that treats learned models as first-class mathematical objects with a single, precise semantics shared by execution and verification. TorchLean unifies (1) a PyTorch-style verified API with eager and compiled modes that lower to a shared op-tagged SSA/DAG computation-graph IR, (2) explicit Float32 semantics via an executable IEEE-754 binary32 kernel and proof-relevant rounding models, and (3) verification via IBP and CROWN/LiRPA-style bound propagation with certificate checking. We validate TorchLean end-to-end on certified robustness, physics-informed residual bounds for PINNs, and Lyapunov-style neural controller verification, alongside mechanized theoretical results including a universal approximation theorem. These results demonstrate a semantics-first infrastructure for fully formal, end-to-end verification of learning-enabled systems. |
| 2026-02-26 | [Towards Faithful Industrial RAG: A Reinforced Co-adaptation Framework for Advertising QA](http://arxiv.org/abs/2602.22584v1) | Wenwei Li, Ming Xu et al. | Industrial advertising question answering (QA) is a high-stakes task in which hallucinated content, particularly fabricated URLs, can lead to financial loss, compliance violations, and legal risk. Although Retrieval-Augmented Generation (RAG) is widely adopted, deploying it in production remains challenging because industrial knowledge is inherently relational, frequently updated, and insufficiently aligned with generation objectives. We propose a reinforced co-adaptation framework that jointly optimizes retrieval and generation through two components: (1) Graph-aware Retrieval (GraphRAG), which models entity-relation structure over a high-citation knowledge subgraph for multi-hop, domain-specific evidence selection; and (2) evidence-constrained reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional rewards covering faithfulness, style compliance, safety, and URL validity. Experiments on an internal advertising QA dataset show consistent gains across expert-judged dimensions including accuracy, completeness, and safety, while reducing the hallucination rate by 72\%. A two-week online A/B test demonstrates a 28.6\% increase in like rate, a 46.2\% decrease in dislike rate, and a 92.7\% reduction in URL hallucination. The system has been running in production for over half a year and has served millions of QA interactions. |
| 2026-02-26 | [Operationalizing Fairness: Post-Hoc Threshold Optimization Under Hard Resource Limits](http://arxiv.org/abs/2602.22560v1) | Moirangthem Tiken Singh, Amit Kalita et al. | The deployment of machine learning in high-stakes domains requires a balance between predictive safety and algorithmic fairness. However, existing fairness interventions often as- sume unconstrained resources and employ group-specific decision thresholds that violate anti- discrimination regulations. We introduce a post-hoc, model-agnostic threshold optimization framework that jointly balances safety, efficiency, and equity under strict and hard capacity constraints. To ensure legal compliance, the framework enforces a single, global decision thresh- old. We formulated a parameterized ethical loss function coupled with a bounded decision rule that mathematically prevents intervention volumes from exceeding the available resources. An- alytically, we prove the key properties of the deployed threshold, including local monotonicity with respect to ethical weighting and the formal identification of critical capacity regimes. We conducted extensive experimental evaluations on diverse high-stakes datasets. The principal re- sults demonstrate that capacity constraints dominate ethical priorities; the strict resource limit determines the final deployed threshold in over 80% of the tested configurations. Furthermore, under a restrictive 25% capacity limit, the proposed framework successfully maintains high risk identification (recall ranging from 0.409 to 0.702), whereas standard unconstrained fairness heuristics collapse to a near-zero utility. We conclude that theoretical fairness objectives must be explicitly subordinated to operational capacity limits to remain in deployment. By decou- pling predictive scoring from policy evaluation and strictly bounding intervention rates, this framework provides a practical and legally compliant mechanism for stakeholders to navigate unavoidable ethical trade-offs in resource-constrained environments. |
| 2026-02-26 | [CourtGuard: A Model-Agnostic Framework for Zero-Shot Policy Adaptation in LLM Safety](http://arxiv.org/abs/2602.22557v1) | Umid Suleymanov, Rufiz Bayramov et al. | Current safety mechanisms for Large Language Models (LLMs) rely heavily on static, fine-tuned classifiers that suffer from adaptation rigidity, the inability to enforce new governance rules without expensive retraining. To address this, we introduce CourtGuard, a retrieval-augmented multi-agent framework that reimagines safety evaluation as Evidentiary Debate. By orchestrating an adversarial debate grounded in external policy documents, CourtGuard achieves state-of-the-art performance across 7 safety benchmarks, outperforming dedicated policy-following baselines without fine-tuning. Beyond standard metrics, we highlight two critical capabilities: (1) Zero-Shot Adaptability, where our framework successfully generalized to an out-of-domain Wikipedia Vandalism task (achieving 90\% accuracy) by swapping the reference policy; and (2) Automated Data Curation and Auditing, where we leveraged CourtGuard to curate and audit nine novel datasets of sophisticated adversarial attacks. Our results demonstrate that decoupling safety logic from model weights offers a robust, interpretable, and adaptable path for meeting current and future regulatory requirements in AI governance. |
| 2026-02-26 | [Multilingual Safety Alignment Via Sparse Weight Editing](http://arxiv.org/abs/2602.22554v1) | Jiaming Liang, Zhaoxin Wang et al. | Large Language Models (LLMs) exhibit significant safety disparities across languages, with low-resource languages (LRLs) often bypassing safety guardrails established for high-resource languages (HRLs) like English. Existing solutions, such as multilingual supervised fine-tuning (SFT) or Reinforcement Learning from Human Feedback (RLHF), are computationally expensive and dependent on scarce multilingual safety data. In this work, we propose a novel, training-free alignment framework based on Sparse Weight Editing. Identifying that safety capabilities are localized within a sparse set of safety neurons, we formulate the cross-lingual alignment problem as a constrained linear transformation. We derive a closed-form solution to optimally map the harmful representations of LRLs to the robust safety subspaces of HRLs, while preserving general utility via a null-space projection constraint. Extensive experiments across 8 languages and multiple model families (Llama-3, Qwen-2.5) demonstrate that our method substantially reduces Attack Success Rate (ASR) in LRLs with negligible impact on general reasoning capabilities, all achieved with a single, data-efficient calculation. |

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



