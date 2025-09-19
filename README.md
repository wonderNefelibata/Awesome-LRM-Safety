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
| 2025-09-18 | [Out-of-Sight Trajectories: Tracking, Fusion, and Prediction](http://arxiv.org/abs/2509.15219v1) | Haichao Zhang, Yi Xu et al. | Trajectory prediction is a critical task in computer vision and autonomous systems, playing a key role in autonomous driving, robotics, surveillance, and virtual reality. Existing methods often rely on complete and noise-free observational data, overlooking the challenges associated with out-of-sight objects and the inherent noise in sensor data caused by limited camera coverage, obstructions, and the absence of ground truth for denoised trajectories. These limitations pose safety risks and hinder reliable prediction in real-world scenarios. In this extended work, we present advancements in Out-of-Sight Trajectory (OST), a novel task that predicts the noise-free visual trajectories of out-of-sight objects using noisy sensor data. Building on our previous research, we broaden the scope of Out-of-Sight Trajectory Prediction (OOSTraj) to include pedestrians and vehicles, extending its applicability to autonomous driving, robotics, surveillance, and virtual reality. Our enhanced Vision-Positioning Denoising Module leverages camera calibration to establish a vision-positioning mapping, addressing the lack of visual references, while effectively denoising noisy sensor data in an unsupervised manner. Through extensive evaluations on the Vi-Fi and JRDB datasets, our approach achieves state-of-the-art performance in both trajectory denoising and prediction, significantly surpassing previous baselines. Additionally, we introduce comparisons with traditional denoising methods, such as Kalman filtering, and adapt recent trajectory prediction models to our task, providing a comprehensive benchmark. This work represents the first initiative to integrate vision-positioning projection for denoising noisy sensor trajectories of out-of-sight agents, paving the way for future advances. The code and preprocessed datasets are available at github.com/Hai-chao-Zhang/OST |
| 2025-09-18 | [Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems](http://arxiv.org/abs/2509.15213v1) | Yicheng Zhang, Zijian Huang et al. | Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks. |
| 2025-09-18 | [Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction](http://arxiv.org/abs/2509.15202v1) | Yuanbo Xie, Yingjie Zhang et al. | Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation. |
| 2025-09-18 | [TDRM: Smooth Reward Models with Temporal Difference for LLM RL and Inference](http://arxiv.org/abs/2509.15110v1) | Dan Zhang, Min Cai et al. | Reward models are central to both reinforcement learning (RL) with language models and inference-time verification. However, existing reward models often lack temporal consistency, leading to ineffective policy updates and unstable RL training. We introduce TDRM, a method for learning smoother and more reliable reward models by minimizing temporal differences during training. This temporal-difference (TD) regularization produces smooth rewards and improves alignment with long-term objectives. Incorporating TDRM into the actor-critic style online RL loop yields consistent empirical gains. It is worth noting that TDRM is a supplement to verifiable reward methods, and both can be used in series. Experiments show that TD-trained process reward models (PRMs) improve performance across Best-of-N (up to 6.6%) and tree-search (up to 23.7%) settings. When combined with Reinforcement Learning with Verifiable Rewards (RLVR), TD-trained PRMs lead to more data-efficient RL -- achieving comparable performance with just 2.5k data to what baseline methods require 50.1k data to attain -- and yield higher-quality language model policies on 8 model variants (5 series), e.g., Qwen2.5-(0.5B, 1,5B), GLM4-9B-0414, GLM-Z1-9B-0414, Qwen2.5-Math-(1.5B, 7B), and DeepSeek-R1-Distill-Qwen-(1.5B, 7B). We release all code at https://github.com/THUDM/TDRM. |
| 2025-09-18 | [Digital Twin-based Cooperative Autonomous Driving in Smart Intersections: A Multi-Agent Reinforcement Learning Approach](http://arxiv.org/abs/2509.15099v1) | Taoyuan Yu, Kui Wang et al. | Unsignalized intersections pose safety and efficiency challenges due to complex traffic flows and blind spots. In this paper, a digital twin (DT)-based cooperative driving system with roadside unit (RSU)-centric architecture is proposed for enhancing safety and efficiency at unsignalized intersections. The system leverages comprehensive bird-eye-view (BEV) perception to eliminate blind spots and employs a hybrid reinforcement learning (RL) framework combining offline pre-training with online fine-tuning. Specifically, driving policies are initially trained using conservative Q-learning (CQL) with behavior cloning (BC) on real datasets, then fine-tuned using multi-agent proximal policy optimization (MAPPO) with self-attention mechanisms to handle dynamic multi-agent coordination. The RSU implements real-time commands via vehicle-to-infrastructure (V2I) communications. Experimental results show that the proposed method yields failure rates below 0.03\% coordinating up to three connected autonomous vehicles (CAVs), significantly outperforming traditional methods. In addition, the system exhibits sub-linear computational scaling with inference times under 40 ms. Furthermore, it demonstrates robust generalization across diverse unsignalized intersection scenarios, indicating its practicality and readiness for real-world deployment. |
| 2025-09-18 | [A Nonlinear Scaling-based Design of Control Lyapunov-barrier Function for Relative Degree 2 Case and its Application to Safe Feedback Linearization](http://arxiv.org/abs/2509.15071v1) | Haechan Pyon, Gyunghoon Park | In this paper we address the problem of control Lyapunov-barrier function (CLBF)-based safe stabilization for a class of nonlinear control-affine systems. A difficulty may arise for the case when a constraint has the relative degree larger than 1, at which computing a proper CLBF is not straightforward. Instead of adding an (possibly non-existent) control barrier function (CBF) to a control Lyapunov function (CLF), our key idea is to simply scale the value of the CLF on the unsafe set, by utilizing a sigmoid function as a scaling factor. We provide a systematic design method for the CLBF, with a detailed condition for the parameters of the sigmoid function to satisfy. It is also seen that the proposed approach to the CLBF design can be applied to the problem of task-space control for a planar robot manipulator with guaranteed safety, for which a safe feedback linearization-based controller is presented. |
| 2025-09-18 | [UCorr: Wire Detection and Depth Estimation for Autonomous Drones](http://arxiv.org/abs/2509.14989v1) | Benedikt Kolbeinsson, Krystian Mikolajczyk | In the realm of fully autonomous drones, the accurate detection of obstacles is paramount to ensure safe navigation and prevent collisions. Among these challenges, the detection of wires stands out due to their slender profile, which poses a unique and intricate problem. To address this issue, we present an innovative solution in the form of a monocular end-to-end model for wire segmentation and depth estimation. Our approach leverages a temporal correlation layer trained on synthetic data, providing the model with the ability to effectively tackle the complex joint task of wire detection and depth estimation. We demonstrate the superiority of our proposed method over existing competitive approaches in the joint task of wire detection and depth estimation. Our results underscore the potential of our model to enhance the safety and precision of autonomous drones, shedding light on its promising applications in real-world scenarios. |
| 2025-09-18 | [Affordance-Based Disambiguation of Surgical Instructions for Collaborative Robot-Assisted Surgery](http://arxiv.org/abs/2509.14967v1) | Ana Davila, Jacinto Colan et al. | Effective human-robot collaboration in surgery is affected by the inherent ambiguity of verbal communication. This paper presents a framework for a robotic surgical assistant that interprets and disambiguates verbal instructions from a surgeon by grounding them in the visual context of the operating field. The system employs a two-level affordance-based reasoning process that first analyzes the surgical scene using a multimodal vision-language model and then reasons about the instruction using a knowledge base of tool capabilities. To ensure patient safety, a dual-set conformal prediction method is used to provide a statistically rigorous confidence measure for robot decisions, allowing it to identify and flag ambiguous commands. We evaluated our framework on a curated dataset of ambiguous surgical requests from cholecystectomy videos, demonstrating a general disambiguation rate of 60% and presenting a method for safer human-robot interaction in the operating room. |
| 2025-09-18 | [Supersonic gas curtain based real-time ionization profile monitor for hadron therapy](http://arxiv.org/abs/2509.14892v1) | Narender Kumar, Milaan Patel et al. | Accurate control and monitoring of the beam is essential for precise dose delivery to tumor tissues during radiotherapy. Real-time monitoring of ion beam profiles and positions improves beam control, patient safety, and treatment reliability by providing immediate feedback. This becomes even more critical in FLASH therapy, where the short corrective window during high-dose delivery demands precise beam control. Existing devices are often limited to in vitro calibration or focus on monitoring a single parameter during treatment. This study aims to develop a device that can simultaneously monitor beam position, profile, current, and energy in real-time, without perturbing the beam, using a supersonic gas curtain system. A supersonic gas curtain beam profile monitor was developed at the Cockcroft Institute to assess its performance and suitability for applications in hadron-beam therapy. The system was integrated with one of the beamlines of the Pelletron accelerator at the Dalton Cumbrian Facility, UK and 2D profile measurements of carbon beams were conducted. The monitor successfully measured the beam profiles within 100ms to 1s across various beam currents (1 - 100 nA), energies (12 - 24 MeV), and charge states (2-5) of carbon. Recorded data was used to estimate detector performance by introducing a parameter called detection limit to quantify sensitivity of the monitor, identifying the threshold number of ions required for detection onset. A method to quantify sensitivity under different beam conditions is then discussed in detail, illustrated with an example case of FLASH beam parameters. This proof-of-concept study demonstrates the performance of the gas curtain-based ionization profile monitor for 2D transverse beam profile measurement of carbon ions. The sensitivity is quantified and evaluated against an example case for FLASH conditions. |
| 2025-09-18 | [Reasoning over Boundaries: Enhancing Specification Alignment via Test-time Delibration](http://arxiv.org/abs/2509.14760v1) | Haoran Zhang, Yafu Li et al. | Large language models (LLMs) are increasingly applied in diverse real-world scenarios, each governed by bespoke behavioral and safety specifications (spec) custom-tailored by users or organizations. These spec, categorized into safety-spec and behavioral-spec, vary across scenarios and evolve with changing preferences and requirements. We formalize this challenge as specification alignment, focusing on LLMs' ability to follow dynamic, scenario-specific spec from both behavioral and safety perspectives. To address this challenge, we propose Align3, a lightweight method that employs Test-Time Deliberation (TTD) with hierarchical reflection and revision to reason over the specification boundaries. We further present SpecBench, a unified benchmark for measuring specification alignment, covering 5 scenarios, 103 spec, and 1,500 prompts. Experiments on 15 reasoning and 18 instruct models with several TTD methods, including Self-Refine, TPO, and MoreThink, yield three key findings: (i) test-time deliberation enhances specification alignment; (ii) Align3 advances the safety-helpfulness trade-off frontier with minimal overhead; (iii) SpecBench effectively reveals alignment gaps. These results highlight the potential of test-time deliberation as an effective strategy for reasoning over the real-world specification boundaries. |
| 2025-09-18 | [Designing Latent Safety Filters using Pre-Trained Vision Models](http://arxiv.org/abs/2509.14758v1) | Ihab Tabbara, Yuxuan Yang et al. | Ensuring safety of vision-based control systems remains a major challenge hindering their deployment in critical settings. Safety filters have gained increased interest as effective tools for ensuring the safety of classical control systems, but their applications in vision-based control settings have so far been limited. Pre-trained vision models (PVRs) have been shown to be effective perception backbones for control in various robotics domains. In this paper, we are interested in examining their effectiveness when used for designing vision-based safety filters. We use them as backbones for classifiers defining failure sets, for Hamilton-Jacobi (HJ) reachability-based safety filters, and for latent world models. We discuss the trade-offs between training from scratch, fine-tuning, and freezing the PVRs when training the models they are backbones for. We also evaluate whether one of the PVRs is superior across all tasks, evaluate whether learned world models or Q-functions are better for switching decisions to safe policies, and discuss practical considerations for deploying these PVRs on resource-constrained devices. |
| 2025-09-18 | [KAIO: A Collection of More Challenging Korean Questions](http://arxiv.org/abs/2509.14752v1) | Nahyun Lee, Guijin Son et al. | With the advancement of mid/post-training techniques, LLMs are pushing their boundaries at an accelerated pace. Legacy benchmarks saturate quickly (e.g., broad suites like MMLU over the years, newer ones like GPQA-D even faster), which makes frontier progress hard to track. The problem is especially acute in Korean: widely used benchmarks are fewer, often translated or narrow in scope, and updated more slowly, so saturation and contamination arrive sooner. Accordingly, at this moment, there is no Korean benchmark capable of evaluating and ranking frontier models. To bridge this gap, we introduce KAIO, a Korean, math-centric benchmark that stresses long-chain reasoning. Unlike recent Korean suites that are at or near saturation, KAIO remains far from saturated: the best-performing model, GPT-5, attains 62.8, followed by Gemini-2.5-Pro (52.3). Open models such as Qwen3-235B and DeepSeek-R1 cluster falls below 30, demonstrating substantial headroom, enabling robust tracking of frontier progress in Korean. To reduce contamination, KAIO will remain private and be served via a held-out evaluator until the best publicly known model reaches at least 80% accuracy, after which we will release the set and iterate to a harder version. |
| 2025-09-18 | [Investigating the Effect of LED Signals and Emotional Displays in Human-Robot Shared Workspaces](http://arxiv.org/abs/2509.14748v1) | Maria Ibrahim, Alap Kshirsagar et al. | Effective communication is essential for safety and efficiency in human-robot collaboration, particularly in shared workspaces. This paper investigates the impact of nonverbal communication on human-robot interaction (HRI) by integrating reactive light signals and emotional displays into a robotic system. We equipped a Franka Emika Panda robot with an LED strip on its end effector and an animated facial display on a tablet to convey movement intent through colour-coded signals and facial expressions. We conducted a human-robot collaboration experiment with 18 participants, evaluating three conditions: LED signals alone, LED signals with reactive emotional displays, and LED signals with pre-emptive emotional displays. We collected data through questionnaires and position tracking to assess anticipation of potential collisions, perceived clarity of communication, and task performance. The results indicate that while emotional displays increased the perceived interactivity of the robot, they did not significantly improve collision anticipation, communication clarity, or task efficiency compared to LED signals alone. These findings suggest that while emotional cues can enhance user engagement, their impact on task performance in shared workspaces is limited. |
| 2025-09-18 | [MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models](http://arxiv.org/abs/2509.14651v1) | Siyu Yan, Long Zeng et al. | As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}. |
| 2025-09-18 | [Enhancing Situational Awareness in Wearable Audio Devices Using a Lightweight Sound Event Localization and Detection System](http://arxiv.org/abs/2509.14650v1) | Jun-Wei Yeow, Ee-Leng Tan et al. | Wearable audio devices with active noise control (ANC) enhance listening comfort but often at the expense of situational awareness. However, this auditory isolation may mask crucial environmental cues, posing significant safety risks. To address this, we propose an environmental intelligence framework that combines Acoustic Scene Classification (ASC) with Sound Event Localization and Detection (SELD). Our system first employs a lightweight ASC model to infer the current environment. The scene prediction then dynamically conditions a SELD network, tuning its sensitivity to detect and localize sounds that are most salient to the current context. On simulated headphone data, the proposed ASC-conditioned SELD system demonstrates improved spatial intelligence over a conventional baseline. This work represents a crucial step towards creating intelligent hearables that can deliver crucial environmental information, fostering a safer and more context-aware listening experience. |
| 2025-09-18 | [Adversarial Distilled Retrieval-Augmented Guarding Model for Online Malicious Intent Detection](http://arxiv.org/abs/2509.14622v1) | Yihao Guo, Haocheng Bian et al. | With the deployment of Large Language Models (LLMs) in interactive applications, online malicious intent detection has become increasingly critical. However, existing approaches fall short of handling diverse and complex user queries in real time. To address these challenges, we introduce ADRAG (Adversarial Distilled Retrieval-Augmented Guard), a two-stage framework for robust and efficient online malicious intent detection. In the training stage, a high-capacity teacher model is trained on adversarially perturbed, retrieval-augmented inputs to learn robust decision boundaries over diverse and complex user queries. In the inference stage, a distillation scheduler transfers the teacher's knowledge into a compact student model, with a continually updated knowledge base collected online. At deployment, the compact student model leverages top-K similar safety exemplars retrieved from the online-updated knowledge base to enable both online and real-time malicious query detection. Evaluations across ten safety benchmarks demonstrate that ADRAG, with a 149M-parameter model, achieves 98.5% of WildGuard-7B's performance, surpasses GPT-4 by 3.3% and Llama-Guard-3-8B by 9.5% on out-of-distribution detection, while simultaneously delivering up to 5.6x lower latency at 300 queries per second (QPS) in real-time applications. |
| 2025-09-18 | [LLM Jailbreak Detection for (Almost) Free!](http://arxiv.org/abs/2509.14558v1) | Guorui Chen, Yifan Xia et al. | Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference. |
| 2025-09-17 | [Perception-Integrated Safety Critical Control via Analytic Collision Cone Barrier Functions on 3D Gaussian Splatting](http://arxiv.org/abs/2509.14421v1) | Dario Tscholl, Yashwanth Nakka et al. | We present a perception-driven safety filter that converts each 3D Gaussian Splat (3DGS) into a closed-form forward collision cone, which in turn yields a first-order control barrier function (CBF) embedded within a quadratic program (QP). By exploiting the analytic geometry of splats, our formulation provides a continuous, closed-form representation of collision constraints that is both simple and computationally efficient. Unlike distance-based CBFs, which tend to activate reactively only when an obstacle is already close, our collision-cone CBF activates proactively, allowing the robot to adjust earlier and thereby produce smoother and safer avoidance maneuvers at lower computational cost. We validate the method on a large synthetic scene with approximately 170k splats, where our filter reduces planning time by a factor of 3 and significantly decreased trajectory jerk compared to a state-of-the-art 3DGS planner, while maintaining the same level of safety. The approach is entirely analytic, requires no high-order CBF extensions (HOCBFs), and generalizes naturally to robots with physical extent through a principled Minkowski-sum inflation of the splats. These properties make the method broadly applicable to real-time navigation in cluttered, perception-derived extreme environments, including space robotics and satellite systems. |
| 2025-09-17 | [RLBind: Adversarial-Invariant Cross-Modal Alignment for Unified Robust Embeddings](http://arxiv.org/abs/2509.14383v1) | Yuhong Lu | Unified multi-modal encoders that bind vision, audio, and other sensors into a shared embedding space are attractive building blocks for robot perception and decision-making. However, on-robot deployment exposes the vision branch to adversarial and natural corruptions, making robustness a prerequisite for safety. Prior defenses typically align clean and adversarial features within CLIP-style encoders and overlook broader cross-modal correspondence, yielding modest gains and often degrading zero-shot transfer. We introduce RLBind, a two-stage adversarial-invariant cross-modal alignment framework for robust unified embeddings. Stage 1 performs unsupervised fine-tuning on clean-adversarial pairs to harden the visual encoder. Stage 2 leverages cross-modal correspondence by minimizing the discrepancy between clean/adversarial features and a text anchor, while enforcing class-wise distributional alignment across modalities. Extensive experiments on Image, Audio, Thermal, and Video data show that RLBind consistently outperforms the LanguageBind backbone and standard fine-tuning baselines in both clean accuracy and norm-bounded adversarial robustness. By improving resilience without sacrificing generalization, RLBind provides a practical path toward safer multi-sensor perception stacks for embodied robots in navigation, manipulation, and other autonomy settings. |
| 2025-09-17 | [GLIDE: A Coordinated Aerial-Ground Framework for Search and Rescue in Unknown Environments](http://arxiv.org/abs/2509.14210v1) | Seth Farrell, Chenghao Li et al. | We present a cooperative aerial-ground search-and-rescue (SAR) framework that pairs two unmanned aerial vehicles (UAVs) with an unmanned ground vehicle (UGV) to achieve rapid victim localization and obstacle-aware navigation in unknown environments. We dub this framework Guided Long-horizon Integrated Drone Escort (GLIDE), highlighting the UGV's reliance on UAV guidance for long-horizon planning. In our framework, a goal-searching UAV executes real-time onboard victim detection and georeferencing to nominate goals for the ground platform, while a terrain-scouting UAV flies ahead of the UGV's planned route to provide mid-level traversability updates. The UGV fuses aerial cues with local sensing to perform time-efficient A* planning and continuous replanning as information arrives. Additionally, we present a hardware demonstration (using a GEM e6 golf cart as the UGV and two X500 UAVs) to evaluate end-to-end SAR mission performance and include simulation ablations to assess the planning stack in isolation from detection. Empirical results demonstrate that explicit role separation across UAVs, coupled with terrain scouting and guided planning, improves reach time and navigation safety in time-critical SAR missions. |

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



