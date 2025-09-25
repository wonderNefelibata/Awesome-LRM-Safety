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
| 2025-09-24 | [From Zonotopes to Proof Certificates: A Formal Pipeline for Safe Control Envelopes](http://arxiv.org/abs/2509.20301v1) | Jonathan Hellwig, Lukas Schäfer et al. | Synthesizing controllers that enforce both safety and actuator constraints is a central challenge in the design of cyber-physical systems. State-of-the-art reachability methods based on zonotopes deliver impressive scalability, yet no zonotope reachability tool has been formally verified and the lack of end-to-end correctness undermines the confidence in their use for safety-critical systems. Although deductive verification with the hybrid system prover KeYmaera X could, in principle, resolve this assurance gap, the high-dimensional set representations required for realistic control envelopes overwhelm its reasoning based on quantifier elimination. To address this gap, we formalize how control-invariant sets serve as sound safety certificates. Building on that foundation, we develop a verification pipeline for control envelopes that unites scalability and formal rigor. First, we compute control envelopes with high-performance reachability algorithms. Second, we certify every intermediate result using provably correct logical principles. To accelerate this certification, we offload computationally intensive zonotope containment tasks to efficient numerical backends, which return compact witnesses that KeYmaera X validates rapidly. We show the practical utility of our approach through representative case studies. |
| 2025-09-24 | [When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity](http://arxiv.org/abs/2509.20293v1) | Benjamin Feuer, Chiung-Yi Tseng et al. | LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at https://anonymous.4open.science/r/judgment-to-noise-947D/README.md |
| 2025-09-24 | [Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization](http://arxiv.org/abs/2509.20230v1) | Wenhan Wu, Zheyuan Liu et al. | Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance. |
| 2025-09-24 | [A Biomimetic Vertebraic Soft Robotic Tail for High-Speed, High-Force Dynamic Maneuvering](http://arxiv.org/abs/2509.20219v1) | Sicong Liu, Jianhui Liu et al. | Robotic tails can enhance the stability and maneuverability of mobile robots, but current designs face a trade-off between the power of rigid systems and the safety of soft ones. Rigid tails generate large inertial effects but pose risks in unstructured environments, while soft tails lack sufficient speed and force. We present a Biomimetic Vertebraic Soft Robotic (BVSR) tail that resolves this challenge through a compliant pneumatic body reinforced by a passively jointed vertebral column inspired by musculoskeletal structures. This hybrid design decouples load-bearing and actuation, enabling high-pressure actuation (up to 6 bar) for superior dynamics while preserving compliance. A dedicated kinematic and dynamic model incorporating vertebral constraints is developed and validated experimentally. The BVSR tail achieves angular velocities above 670{\deg}/s and generates inertial forces and torques up to 5.58 N and 1.21 Nm, indicating over 200% improvement compared to non-vertebraic designs. Demonstrations on rapid cart stabilization, obstacle negotiation, high-speed steering, and quadruped integration confirm its versatility and practical utility for agile robotic platforms. |
| 2025-09-24 | [Meso-chiral optical properties of plasmonic nanoparticles: uncovering hidden chirality](http://arxiv.org/abs/2509.20178v1) | Yuanyang Xie, Alexey V. Krasavin et al. | Molecular chirality plays an important role in chemistry and biology, allows control of biological interactions, affects drugs efficacy and safety, and promotes synthesis of new materials. In general, chirality manifests itself in optical activity (circular dichroism or circular birefringence). Chiral plasmonic nanoparticles have been recently developed for molecular enantiomer separation, chiral sensing and chiral photocatalysis. Here, we show that optical chirality of plasmonic nanoparticles exhibiting strong scattering can remain completely undetected using standard characterisation techniques, such as circular dichroism measurements. This phenomenon, which we term meso-chiral in analogy to meso-compounds in chemistry, is based on mutual cancellation of absorption and scattering chiral responses. As a prominent example, the meso-optical behaviour has been numerically demonstrated in multi-wound-SiO2/Au nanoparticles over the entire visible spectral range and in other prototypical chiral nanoparticles in narrower spectral ranges. The meso-chiral property has been experimentally verified by demonstrating chiral absorption of gold helicoid nanoparticles at the wavelength where conventional circular dichroism measurements show absence of chiral response (gext=0). These findings demonstrate a valuable link between microscopic to macroscopic manifestations of chirality and can provide insights for interpreting a wide range of experimental results and designing chiral properties of plasmonic nanoparticles. |
| 2025-09-24 | [Probing Gender Bias in Multilingual LLMs: A Case Study of Stereotypes in Persian](http://arxiv.org/abs/2509.20168v1) | Ghazal Kalhor, Behnam Bahrak | Multilingual Large Language Models (LLMs) are increasingly used worldwide, making it essential to ensure they are free from gender bias to prevent representational harm. While prior studies have examined such biases in high-resource languages, low-resource languages remain understudied. In this paper, we propose a template-based probing methodology, validated against real-world data, to uncover gender stereotypes in LLMs. As part of this framework, we introduce the Domain-Specific Gender Skew Index (DS-GSI), a metric that quantifies deviations from gender parity. We evaluate four prominent models, GPT-4o mini, DeepSeek R1, Gemini 2.0 Flash, and Qwen QwQ 32B, across four semantic domains, focusing on Persian, a low-resource language with distinct linguistic features. Our results show that all models exhibit gender stereotypes, with greater disparities in Persian than in English across all domains. Among these, sports reflect the most rigid gender biases. This study underscores the need for inclusive NLP practices and provides a framework for assessing bias in other low-resource languages. |
| 2025-09-24 | [Smaller is Better: Enhancing Transparency in Vehicle AI Systems via Pruning](http://arxiv.org/abs/2509.20148v1) | Sanish Suwal, Shaurya Garg et al. | Connected and autonomous vehicles continue to heavily rely on AI systems, where transparency and security are critical for trust and operational safety. Post-hoc explanations provide transparency to these black-box like AI models but the quality and reliability of these explanations is often questioned due to inconsistencies and lack of faithfulness in representing model decisions. This paper systematically examines the impact of three widely used training approaches, namely natural training, adversarial training, and pruning, affect the quality of post-hoc explanations for traffic sign classifiers. Through extensive empirical evaluation, we demonstrate that pruning significantly enhances the comprehensibility and faithfulness of explanations (using saliency maps). Our findings reveal that pruning not only improves model efficiency but also enforces sparsity in learned representation, leading to more interpretable and reliable decisions. Additionally, these insights suggest that pruning is a promising strategy for developing transparent deep learning models, especially in resource-constrained vehicular AI systems. |
| 2025-09-24 | [EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models](http://arxiv.org/abs/2509.20146v1) | Botai Yuan, Yutian Zhou et al. | Recent benchmarks for medical Large Vision-Language Models (LVLMs) emphasize leaderboard accuracy, overlooking reliability and safety. We study sycophancy -- models' tendency to uncritically echo user-provided information -- in high-stakes clinical settings. We introduce EchoBench, a benchmark to systematically evaluate sycophancy in medical LVLMs. It contains 2,122 images across 18 departments and 20 modalities with 90 prompts that simulate biased inputs from patients, medical students, and physicians. We evaluate medical-specific, open-source, and proprietary LVLMs. All exhibit substantial sycophancy; the best proprietary model (Claude 3.7 Sonnet) still shows 45.98% sycophancy, and GPT-4.1 reaches 59.15%. Many medical-specific models exceed 95% sycophancy despite only moderate accuracy. Fine-grained analyses by bias type, department, perceptual granularity, and modality identify factors that increase susceptibility. We further show that higher data quality/diversity and stronger domain knowledge reduce sycophancy without harming unbiased accuracy. EchoBench also serves as a testbed for mitigation: simple prompt-level interventions (negative prompting, one-shot, few-shot) produce consistent reductions and motivate training- and decoding-time strategies. Our findings highlight the need for robust evaluation beyond accuracy and provide actionable guidance toward safer, more trustworthy medical LVLMs. |
| 2025-09-24 | [Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving](http://arxiv.org/abs/2509.20109v1) | Pengxiang Li, Yinan Zheng et al. | End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems. |
| 2025-09-24 | [Steerable Adversarial Scenario Generation through Test-Time Preference Alignment](http://arxiv.org/abs/2509.20102v1) | Tong Nie, Yuewen Mei et al. | Adversarial scenario generation is a cost-effective approach for safety assessment of autonomous driving systems. However, existing methods are often constrained to a single, fixed trade-off between competing objectives such as adversariality and realism. This yields behavior-specific models that cannot be steered at inference time, lacking the efficiency and flexibility to generate tailored scenarios for diverse training and testing requirements. In view of this, we reframe the task of adversarial scenario generation as a multi-objective preference alignment problem and introduce a new framework named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator (SAGE). SAGE enables fine-grained test-time control over the trade-off between adversariality and realism without any retraining. We first propose hierarchical group-based preference optimization, a data-efficient offline alignment method that learns to balance competing objectives by decoupling hard feasibility constraints from soft preferences. Instead of training a fixed model, SAGE fine-tunes two experts on opposing preferences and constructs a continuous spectrum of policies at inference time by linearly interpolating their weights. We provide theoretical justification for this framework through the lens of linear mode connectivity. Extensive experiments demonstrate that SAGE not only generates scenarios with a superior balance of adversariality and realism but also enables more effective closed-loop training of driving policies. Project page: https://tongnie.github.io/SAGE/. |
| 2025-09-24 | [Hybrid Safety Verification of Multi-Agent Systems using $ψ$-Weighted CBFs and PAC Guarantees](http://arxiv.org/abs/2509.20093v1) | Venkat Margapuri, Garik Kazanjian et al. | This study proposes a hybrid safety verification framework for closed-loop multi-agent systems under bounded stochastic disturbances. The proposed approach augments control barrier functions with a novel $\psi$-weighted formulation that encodes directional control alignment between agents into the safety constraints. Deterministic admissibility is combined with empirical validation via Monte Carlo rollouts, and a PAC-style guarantee is derived based on margin-aware safety violations to provide a probabilistic safety certificate. The results from the experiments conducted under different bounded stochastic disturbances validate the feasibility of the proposed approach. |
| 2025-09-24 | [C-3TO: Continuous 3D Trajectory Optimization on Neural Euclidean Signed Distance Fields](http://arxiv.org/abs/2509.20084v1) | Guillermo Gil, Jose Antonio Cobano et al. | This paper introduces a novel framework for continuous 3D trajectory optimization in cluttered environments, leveraging online neural Euclidean Signed Distance Fields (ESDFs). Unlike prior approaches that rely on discretized ESDF grids with interpolation, our method directly optimizes smooth trajectories represented by fifth-order polynomials over a continuous neural ESDF, ensuring precise gradient information throughout the entire trajectory. The framework integrates a two-stage nonlinear optimization pipeline that balances efficiency, safety and smoothness. Experimental results demonstrate that C-3TO produces collision-aware and dynamically feasible trajectories. Moreover, its flexibility in defining local window sizes and optimization parameters enables straightforward adaptation to diverse user's needs without compromising performance. By combining continuous trajectory parameterization with a continuously updated neural ESDF, C-3TO establishes a robust and generalizable foundation for safe and efficient local replanning in aerial robotics. |
| 2025-09-24 | [MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM](http://arxiv.org/abs/2509.20067v1) | Wenliang Li, Rui Yan et al. | Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice. |
| 2025-09-24 | [Responsible AI Technical Report](http://arxiv.org/abs/2509.20057v1) | KT, : et al. | KT developed a Responsible AI (RAI) assessment methodology and risk mitigation technologies to ensure the safety and reliability of AI services. By analyzing the Basic Act on AI implementation and global AI governance trends, we established a unique approach for regulatory compliance and systematically identify and manage all potential risk factors from AI development to operation. We present a reliable assessment methodology that systematically verifies model safety and robustness based on KT's AI risk taxonomy tailored to the domestic environment. We also provide practical tools for managing and mitigating identified AI risks. With the release of this report, we also release proprietary Guardrail : SafetyGuard that blocks harmful responses from AI models in real-time, supporting the enhancement of safety in the domestic AI development ecosystem. We also believe these research outcomes provide valuable insights for organizations seeking to develop Responsible AI. |
| 2025-09-24 | [MARG: MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping](http://arxiv.org/abs/2509.20036v1) | Yinzhao Dong, Ji Ma et al. | Deep Reinforcement Learning (DRL) controllers for quadrupedal locomotion have demonstrated impressive performance on challenging terrains, allowing robots to execute complex skills such as climbing, running, and jumping. However, existing blind locomotion controllers often struggle to ensure safety and efficient traversal through risky gap terrains, which are typically highly complex, requiring robots to perceive terrain information and select appropriate footholds during locomotion accurately. Meanwhile, existing perception-based controllers still present several practical limitations, including a complex multi-sensor deployment system and expensive computing resource requirements. This paper proposes a DRL controller named MAstering Risky Gap Terrains (MARG), which integrates terrain maps and proprioception to dynamically adjust the action and enhance the robot's stability in these tasks. During the training phase, our controller accelerates policy optimization by selectively incorporating privileged information (e.g., center of mass, friction coefficients) that are available in simulation but unmeasurable directly in real-world deployments due to sensor limitations. We also designed three foot-related rewards to encourage the robot to explore safe footholds. More importantly, a terrain map generation (TMG) model is proposed to reduce the drift existing in mapping and provide accurate terrain maps using only one LiDAR, providing a foundation for zero-shot transfer of the learned policy. The experimental results indicate that MARG maintains stability in various risky terrain tasks. |
| 2025-09-24 | [FreezeVLA: Action-Freezing Attacks against Vision-Language-Action Models](http://arxiv.org/abs/2509.19870v1) | Xin Wang, Jie Li et al. | Vision-Language-Action (VLA) models are driving rapid progress in robotics by enabling agents to interpret multimodal inputs and execute complex, long-horizon tasks. However, their safety and robustness against adversarial attacks remain largely underexplored. In this work, we identify and formalize a critical adversarial vulnerability in which adversarial images can "freeze" VLA models and cause them to ignore subsequent instructions. This threat effectively disconnects the robot's digital mind from its physical actions, potentially inducing inaction during critical interventions. To systematically study this vulnerability, we propose FreezeVLA, a novel attack framework that generates and evaluates action-freezing attacks via min-max bi-level optimization. Experiments on three state-of-the-art VLA models and four robotic benchmarks show that FreezeVLA attains an average attack success rate of 76.2%, significantly outperforming existing methods. Moreover, adversarial images generated by FreezeVLA exhibit strong transferability, with a single image reliably inducing paralysis across diverse language prompts. Our findings expose a critical safety risk in VLA models and highlight the urgent need for robust defense mechanisms. |
| 2025-09-24 | [Scalable and Approximation-free Symbolic Control for Unknown Euler-Lagrange Systems](http://arxiv.org/abs/2509.19859v1) | Ratnangshu Das, Shubham Sawarkar et al. | We propose a novel symbolic control framework for enforcing temporal logic specifications in Euler-Lagrange systems that addresses the key limitations of traditional abstraction-based approaches. Unlike existing methods that require exact system models and provide guarantees only at discrete sampling instants, our approach relies only on bounds on system parameters and input constraints, and ensures correctness for the full continuous-time trajectory. The framework combines scalable abstraction of a simplified virtual system with a closed-form, model-free controller that guarantees trajectories satisfy the original specification while respecting input bounds and remaining robust to unknown but bounded disturbances. We provide feasibility conditions for the construction of confinement regions and analyze the trade-off between efficiency and conservatism. Case studies on pendulum dynamics, a two-link manipulator, and multi-agent systems, including hardware experiments, demonstrate that the proposed approach ensures both correctness and safety while significantly reducing computational time and memory requirements. These results highlight its scalability and practicality for real-world robotic systems where precise models are unavailable and continuous-time guarantees are essential. |
| 2025-09-24 | [LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](http://arxiv.org/abs/2509.19839v1) | Huizhen Shu, Xuying Li et al. | Achieving robust safety alignment in large language models (LLMs) while preserving their utility remains a fundamental challenge. Existing approaches often struggle to balance comprehensive safety with fine-grained controllability at the representation level. We introduce LATENTGUARD, a novel three-stage framework that combines behavioral alignment with supervised latent space control for interpretable and precise safety steering. Our approach begins by fine-tuning an LLM on rationalized datasets containing both reasoning-enhanced refusal responses to adversarial prompts and reasoning-enhanced normal responses to benign queries, establishing robust behavioral priors across both safety-critical and utility-preserving scenarios. We then train a structured variational autoencoder (VAE) on intermediate MLP activations, supervised by multi-label annotations including attack types, attack methods, and benign indicators. This supervision enables the VAE to learn disentangled latent representations that capture distinct adversarial characteristics while maintaining semantic interpretability. Through targeted manipulation of learned latent dimensions, LATENTGUARD achieves selective refusal behavior, effectively blocking harmful requests while preserving helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate significant improvements in both safety controllability and response interpretability without compromising utility. Cross-architecture validation on Mistral-7B confirms the generalizability of our latent steering approach, showing consistent effectiveness across different model families. Our results suggest that structured representation-level intervention offers a promising pathway toward building safer yet practical LLM systems. |
| 2025-09-24 | [Open-source Stand-Alone Versatile Tensor Accelerator](http://arxiv.org/abs/2509.19790v1) | Anthony Faure-Gignoux, Kevin Delmas et al. | Machine Learning (ML) applications demand significant computational resources, posing challenges for safety-critical domains like aeronautics. The Versatile Tensor Accelerator (VTA) is a promising FPGA-based solution, but its adoption was hindered by its dependency on the TVM compiler and by other code non-compliant with certification requirements. This paper presents an open-source, standalone Python compiler pipeline for the VTA, developed from scratch and designed with certification requirements, modularity, and extensibility in mind. The compiler's effectiveness is demonstrated by compiling and executing LeNet-5 Convolutional Neural Network (CNN) using the VTA simulators, and preliminary results indicate a strong potential for scaling its capabilities to larger CNN architectures. All contributions are publicly available. |
| 2025-09-24 | [RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving](http://arxiv.org/abs/2509.19789v1) | Carlo Bosio, Greg Woelki et al. | Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model. |

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



