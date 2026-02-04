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
| 2026-02-03 | [Conformal Reachability for Safe Control in Unknown Environments](http://arxiv.org/abs/2602.03799v1) | Xinhang Ma, Junlin Wu et al. | Designing provably safe control is a core problem in trustworthy autonomy. However, most prior work in this regard assumes either that the system dynamics are known or deterministic, or that the state and action space are finite, significantly limiting application scope. We address this limitation by developing a probabilistic verification framework for unknown dynamical systems which combines conformal prediction with reachability analysis. In particular, we use conformal prediction to obtain valid uncertainty intervals for the unknown dynamics at each time step, with reachability then verifying whether safety is maintained within the conformal uncertainty bounds. Next, we develop an algorithmic approach for training control policies that optimize nominal reward while also maximizing the planning horizon with sound probabilistic safety guarantees. We evaluate the proposed approach in seven safe control settings spanning four domains -- cartpole, lane following, drone control, and safe navigation -- for both affine and nonlinear safety specifications. Our experiments show that the policies we learn achieve the strongest provable safety guarantees while still maintaining high average reward. |
| 2026-02-03 | [From Pre- to Intra-operative MRI: Predicting Brain Shift in Temporal Lobe Resection for Epilepsy Surgery](http://arxiv.org/abs/2602.03785v1) | Jingjing Peng, Giorgio Fiore et al. | Introduction: In neurosurgery, image-guided Neurosurgery Systems (IGNS) highly rely on preoperative brain magnetic resonance images (MRI) to assist surgeons in locating surgical targets and determining surgical paths. However, brain shift invalidates the preoperative MRI after dural opening. Updated intraoperative brain MRI with brain shift compensation is crucial for enhancing the precision of neuronavigation systems and ensuring the optimal outcome of surgical interventions. Methodology: We propose NeuralShift, a U-Net-based model that predicts brain shift entirely from pre-operative MRI for patients undergoing temporal lobe resection. We evaluated our results using Target Registration Errors (TREs) computed on anatomical landmarks located on the resection side and along the midline, and DICE scores comparing predicted intraoperative masks with masks derived from intraoperative MRI. Results: Our experimental results show that our model can predict the global deformation of the brain (DICE of 0.97) with accurate local displacements (achieve landmark TRE as low as 1.12 mm), compensating for large brain shifts during temporal lobe removal neurosurgery. Conclusion: Our proposed model is capable of predicting the global deformation of the brain during temporal lobe resection using only preoperative images, providing potential opportunities to the surgical team to increase safety and efficiency of neurosurgery and better outcomes to patients. Our contributions will be publicly available after acceptance in https://github.com/SurgicalDataScienceKCL/NeuralShift. |
| 2026-02-03 | [Reward Redistribution for CVaR MDPs using a Bellman Operator on L-infinity](http://arxiv.org/abs/2602.03778v1) | Aneri Muni, Vincent Taboga et al. | Tail-end risk measures such as static conditional value-at-risk (CVaR) are used in safety-critical applications to prevent rare, yet catastrophic events. Unlike risk-neutral objectives, the static CVaR of the return depends on entire trajectories without admitting a recursive Bellman decomposition in the underlying Markov decision process. A classical resolution relies on state augmentation with a continuous variable. However, unless restricted to a specialized class of admissible value functions, this formulation induces sparse rewards and degenerate fixed points. In this work, we propose a novel formulation of the static CVaR objective based on augmentation. Our alternative approach leads to a Bellman operator with: (1) dense per-step rewards; (2) contracting properties on the full space of bounded value functions. Building on this theoretical foundation, we develop risk-averse value iteration and model-free Q-learning algorithms that rely on discretized augmented states. We further provide convergence guarantees and approximation error bounds due to discretization. Empirical results demonstrate that our algorithms successfully learn CVaR-sensitive policies and achieve effective performance-safety trade-offs. |
| 2026-02-03 | [Mitigating Timing-Based Attacks in Real-Time Cyber-Physical Systems](http://arxiv.org/abs/2602.03757v1) | Arkaprava Sain, Sunandan Adhikary et al. | Real-time cyber-physical systems depend on deterministic task execution to guarantee safety and correctness. Unfortunately, this determinism can unintentionally expose timing information that enables adversaries to infer task execution patterns and carry out timing-based attacks targeting safety-critical control tasks. While prior defenses aim to obscure schedules through randomization or isolation, they typically neglect the implications of such modifications on closed-loop control behavior and real-time feasibility. This work studies the problem of securing real-time control workloads against timing inference attacks while explicitly accounting for both schedulability constraints and control performance requirements. We present a scheduling-based mitigation approach that introduces bounded timing perturbations to control task executions in a structured manner, reducing adversarial opportunities without violating real-time guarantees. The framework jointly considers worst-case execution behavior and the impact of execution delays on control performance, enabling the system to operate within predefined safety and performance limits. Through experimental evaluation on representative task sets and control scenarios, the proposed approach demonstrates that exposure to timing-based attacks can be significantly reduced while preserving predictable execution and acceptable control quality. |
| 2026-02-03 | [Edge-Optimized Vision-Language Models for Underground Infrastructure Assessment](http://arxiv.org/abs/2602.03742v1) | Johny J. Lopez, Md Meftahul Ferdaus et al. | Autonomous inspection of underground infrastructure, such as sewer and culvert systems, is critical to public safety and urban sustainability. Although robotic platforms equipped with visual sensors can efficiently detect structural deficiencies, the automated generation of human-readable summaries from these detections remains a significant challenge, especially on resource-constrained edge devices. This paper presents a novel two-stage pipeline for end-to-end summarization of underground deficiencies, combining our lightweight RAPID-SCAN segmentation model with a fine-tuned Vision-Language Model (VLM) deployed on an edge computing platform. The first stage employs RAPID-SCAN (Resource-Aware Pipeline Inspection and Defect Segmentation using Compact Adaptive Network), achieving 0.834 F1-score with only 0.64M parameters for efficient defect segmentation. The second stage utilizes a fine-tuned Phi-3.5 VLM that generates concise, domain-specific summaries in natural language from the segmentation outputs. We introduce a curated dataset of inspection images with manually verified descriptions for VLM fine-tuning and evaluation. To enable real-time performance, we employ post-training quantization with hardware-specific optimization, achieving significant reductions in model size and inference latency without compromising summarization quality. We deploy and evaluate our complete pipeline on a mobile robotic platform, demonstrating its effectiveness in real-world inspection scenarios. Our results show the potential of edge-deployable integrated AI systems to bridge the gap between automated defect detection and actionable insights for infrastructure maintenance, paving the way for more scalable and autonomous inspection solutions. |
| 2026-02-03 | [Soft Sensor for Bottom-Hole Pressure Estimation in Petroleum Wells Using Long Short-Term Memory and Transfer Learning](http://arxiv.org/abs/2602.03737v1) | M. A. Fernandes, E. Gildin et al. | Monitoring bottom-hole variables in petroleum wells is essential for production optimization, safety, and emissions reduction. Permanent Downhole Gauges (PDGs) provide real-time pressure data but face reliability and cost issues. We propose a machine learning-based soft sensor to estimate flowing Bottom-Hole Pressure (BHP) using wellhead and topside measurements. A Long Short-Term Memory (LSTM) model is introduced and compared with Multi-Layer Perceptron (MLP) and Ridge Regression. We also pioneer Transfer Learning for adapting models across operational environments. Tested on real offshore datasets from Brazil's Pre-salt basin, the methodology achieved Mean Absolute Percentage Error (MAPE) consistently below 2\%, outperforming benchmarks. This work offers a cost-effective, accurate alternative to physical sensors, with broad applicability across diverse reservoir and flow conditions. |
| 2026-02-03 | [Input-to-State Safe Backstepping: Robust Safety-Critical Control with Unmatched Uncertainties](http://arxiv.org/abs/2602.03691v1) | Max H. Cohen, Pio Ong et al. | Guaranteeing safety in the presence of unmatched disturbances -- uncertainties that cannot be directly canceled by the control input -- remains a key challenge in nonlinear control. This paper presents a constructive approach to safety-critical control of nonlinear systems with unmatched disturbances. We first present a generalization of the input-to-state safety (ISSf) framework for systems with these uncertainties using the recently developed notion of an Optimal Decay CBF, which provides more flexibility for satisfying the associated Lyapunov-like conditions for safety. From there, we outline a procedure for constructing ISSf-CBFs for two relevant classes of systems with unmatched uncertainties: i) strict-feedback systems; ii) dual-relative-degree systems, which are similar to differentially flat systems. Our theoretical results are illustrated via numerical simulations of an inverted pendulum and planar quadrotor. |
| 2026-02-03 | [Instruction Anchors: Dissecting the Causal Dynamics of Modality Arbitration](http://arxiv.org/abs/2602.03677v1) | Yu Zhang, Mufan Xu et al. | Modality following serves as the capacity of multimodal large language models (MLLMs) to selectively utilize multimodal contexts based on user instructions. It is fundamental to ensuring safety and reliability in real-world deployments. However, the underlying mechanisms governing this decision-making process remain poorly understood. In this paper, we investigate its working mechanism through an information flow lens. Our findings reveal that instruction tokens function as structural anchors for modality arbitration: Shallow attention layers perform non-selective information transfer, routing multimodal cues to these anchors as a latent buffer; Modality competition is resolved within deep attention layers guided by the instruction intent, while MLP layers exhibit semantic inertia, acting as an adversarial force. Furthermore, we identify a sparse set of specialized attention heads that drive this arbitration. Causal interventions demonstrate that manipulating a mere $5\%$ of these critical heads can decrease the modality-following ratio by $60\%$ through blocking, or increase it by $60\%$ through targeted amplification of failed samples. Our work provides a substantial step toward model transparency and offers a principled framework for the orchestration of multimodal information in MLLMs. |
| 2026-02-03 | [MM-SCALE: Grounded Multimodal Moral Reasoning via Scalar Judgment and Listwise Alignment](http://arxiv.org/abs/2602.03665v1) | Eunkyu Park, Wesley Hanwen Deng et al. | Vision-Language Models (VLMs) continue to struggle to make morally salient judgments in multimodal and socially ambiguous contexts. Prior works typically rely on binary or pairwise supervision, which often fail to capture the continuous and pluralistic nature of human moral reasoning. We present MM-SCALE (Multimodal Moral Scale), a large-scale dataset for aligning VLMs with human moral preferences through 5-point scalar ratings and explicit modality grounding. Each image-scenario pair is annotated with moral acceptability scores and grounded reasoning labels by humans using an interface we tailored for data collection, enabling listwise preference optimization over ranked scenario sets. By moving from discrete to scalar supervision, our framework provides richer alignment signals and finer calibration of multimodal moral reasoning. Experiments show that VLMs fine-tuned on MM-SCALE achieve higher ranking fidelity and more stable safety calibration than those trained with binary signals. |
| 2026-02-03 | [A Comparison of Set-Based Observers for Nonlinear Systems](http://arxiv.org/abs/2602.03646v1) | Nico Holzinger, Matthias Althoff | Set-based state estimation computes sets of states consistent with a system model given bounded sets of disturbances and noise. Bounding the set of states is crucial for safety-critical applications so that one can ensure that all specifications are met. While numerous approaches have been proposed for nonlinear discrete-time systems, a unified evaluation under comparable conditions is lacking. This paper reviews and implements a representative selection of set-based observers within the CORA framework. To provide an objective comparison, the methods are evaluated on common benchmarks, and we examine computational effort, scalability, and the conservatism of the resulting state bounds. This study highlights characteristic trade-offs between observer categories and set representations, as well as practical considerations arising in their implementation. All implementations are made publicly available to support reproducibility and future development. This paper thereby offers the first broad, tool-supported comparison of guaranteed state estimators for nonlinear discrete-time systems. |
| 2026-02-03 | [Can LLMs Do Rocket Science? Exploring the Limits of Complex Reasoning with GTOC 12](http://arxiv.org/abs/2602.03630v1) | Iñaki del Campo, Pablo Cuervo et al. | Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation and general reasoning, yet their capacity for autonomous multi-stage planning in high-dimensional, physically constrained environments remains an open research question. This study investigates the limits of current AI agents by evaluating them against the 12th Global Trajectory Optimization Competition (GTOC 12), a complex astrodynamics challenge requiring the design of a large-scale asteroid mining campaign. We adapt the MLE-Bench framework to the domain of orbital mechanics and deploy an AIDE-based agent architecture to autonomously generate and refine mission solutions. To assess performance beyond binary validity, we employ an "LLM-as-a-Judge" methodology, utilizing a rubric developed by domain experts to evaluate strategic viability across five structural categories. A comparative analysis of models, ranging from GPT-4-Turbo to reasoning-enhanced architectures like Gemini 2.5 Pro, and o3, reveals a significant trend: the average strategic viability score has nearly doubled in the last two years (rising from 9.3 to 17.2 out of 26). However, we identify a critical capability gap between strategy and execution. While advanced models demonstrate sophisticated conceptual understanding, correctly framing objective functions and mission architectures, they consistently fail at implementation due to physical unit inconsistencies, boundary condition errors, and inefficient debugging loops. We conclude that, while current LLMs often demonstrate sufficient knowledge and intelligence to tackle space science tasks, they remain limited by an implementation barrier, functioning as powerful domain facilitators rather than fully autonomous engineers. |
| 2026-02-03 | [TIPS Over Tricks: Simple Prompts for Effective Zero-shot Anomaly Detection](http://arxiv.org/abs/2602.03594v1) | Alireza Salehi, Ehsan Karami et al. | Anomaly detection identifies departures from expected behavior in safety-critical settings. When target-domain normal data are unavailable, zero-shot anomaly detection (ZSAD) leverages vision-language models (VLMs). However, CLIP's coarse image-text alignment limits both localization and detection due to (i) spatial misalignment and (ii) weak sensitivity to fine-grained anomalies; prior work compensates with complex auxiliary modules yet largely overlooks the choice of backbone. We revisit the backbone and use TIPS-a VLM trained with spatially aware objectives. While TIPS alleviates CLIP's issues, it exposes a distributional gap between global and local features. We address this with decoupled prompts-fixed for image-level detection and learnable for pixel-level localization-and by injecting local evidence into the global score. Without CLIP-specific tricks, our TIPS-based pipeline improves image-level performance by 1.1-3.9% and pixel-level by 1.5-6.9% across seven industrial datasets, delivering strong generalization with a lean architecture. Code is available at github.com/AlirezaSalehy/Tipsomaly. |
| 2026-02-03 | [Formal Evidence Generation for Assurance Cases for Robotic Software Models](http://arxiv.org/abs/2602.03550v1) | Fang Yan, Simon Foster et al. | Robotics and Autonomous Systems are increasingly deployed in safety-critical domains, so that demonstrating their safety is essential. Assurance Cases (ACs) provide structured arguments supported by evidence, but generating and maintaining this evidence is labour-intensive, error-prone, and difficult to keep consistent as systems evolve. We present a model-based approach to systematically generating AC evidence by embedding formal verification into the assurance workflow. The approach addresses three challenges: systematically deriving formal assertions from natural language requirements using templates, orchestrating multiple formal verification tools to handle diverse property types, and integrating formal evidence production into the workflow. Leveraging RoboChart, a domain-specific modelling language with formal semantics, we combine model checking and theorem proving in our approach. Structured requirements are automatically transformed into formal assertions using predefined templates, and verification results are automatically integrated as evidence. Case studies demonstrate the effectiveness of our approach. |
| 2026-02-03 | [Soft-Radial Projection for Constrained End-to-End Learning](http://arxiv.org/abs/2602.03461v1) | Philipp J. Schneider, Daniel Kuhn | Integrating hard constraints into deep learning is essential for safety-critical systems. Yet existing constructive layers that project predictions onto constraint boundaries face a fundamental bottleneck: gradient saturation. By collapsing exterior points onto lower-dimensional surfaces, standard orthogonal projections induce rank-deficient Jacobians, which nullify gradients orthogonal to active constraints and hinder optimization. We introduce Soft-Radial Projection, a differentiable reparameterization layer that circumvents this issue through a radial mapping from Euclidean space into the interior of the feasible set. This construction guarantees strict feasibility while preserving a full-rank Jacobian almost everywhere, thereby preventing the optimization stalls typical of boundary-based methods. We theoretically prove that the architecture retains the universal approximation property and empirically show improved convergence behavior and solution quality over state-of-the-art optimization- and projection-based baselines. |
| 2026-02-03 | [Accelerated Electromagnetic Simulation of MRI RF Interactions with Graphene Microtransistor-Based Neural Probes for Electrophysiology-fMRI Integration](http://arxiv.org/abs/2602.03437v1) | Suchit Kumar, Alejandro Labastida Ramirez et al. | Implementing electrophysiological recordings within an MRI environment is challenging due to complex interactions between recording probes and MRI-generated fields, which can affect both safety and data quality. This study aims to develop and evaluate a hybrid electromagnetic (EM) simulation framework for efficient and accurate assessment of such interactions. Methods: A hybrid EM strategy integrating the Huygens' Box (HB) method with sub-gridding was implemented in an FDTD solver (Sim4Life). RF coil models for mouse and rat head were simulated with and without intracortical (IC) and epicortical (EC) graphene-based micro-transistor arrays. Three-dimensional multi-layered probe models were reconstructed from two-dimensional layouts, and transmit field ($B_{1}^{+}$), electric field ($E$), and specific absorption rate (SAR) distributions were evaluated. Performance was benchmarked against conventional full-wave multi-port (MP) simulations using Bland-Altman analysis and voxel-wise percentage differences. Results: HB simulations reduced computational time by approximately 70-80%, while preserving spatial patterns of $|B_{1}^{+}|$, $|E|$, and SAR, including transmit-field symmetry and localized high-field regions. Deviations from MP were minimal for $|B_{1}^{+}|$ (median $Δ$% 0.02-0.07% in mice, -3.7% to -1.7% in rats) and modest for $|E|$ and SAR, with absolute SAR values remaining well below human safety limits. Graphene-based arrays produced negligible effects on RF transmission and SAR deposition. Conclusion: The HB approach enables computationally efficient, high-resolution evaluation of EM interactions involving microscopic probes in MRI environments, supporting simulations that are otherwise impractical with full-wave MP modeling. |
| 2026-02-03 | [ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response](http://arxiv.org/abs/2602.03430v1) | Xiaomeng Zhu, Fengming Zhu et al. | While passive agents merely follow instructions, proactive agents align with higher-level objectives, such as assistance and safety by continuously monitoring the environment to determine when and how to act. However, developing proactive agents is hindered by the lack of specialized resources. To address this, we introduce ProAct-75, a benchmark designed to train and evaluate proactive agents across diverse domains, including assistance, maintenance, and safety monitoring. Spanning 75 tasks, our dataset features 91,581 step-level annotations enriched with explicit task graphs. These graphs encode step dependencies and parallel execution possibilities, providing the structural grounding necessary for complex decision-making. Building on this benchmark, we propose ProAct-Helper, a reference baseline powered by a Multimodal Large Language Model (MLLM) that grounds decision-making in state detection, and leveraging task graphs to enable entropy-driven heuristic search for action selection, allowing agents to execute parallel threads independently rather than mirroring the human's next step. Extensive experiments demonstrate that ProAct-Helper outperforms strong closed-source models, improving trigger detection mF1 by 6.21%, saving 0.25 more steps in online one-step decision, and increasing the rate of parallel actions by 15.58%. |
| 2026-02-03 | [Risk Awareness Injection: Calibrating Vision-Language Models for Safety without Compromising Utility](http://arxiv.org/abs/2602.03402v1) | Mengxuan Wang, Yuxin Chen et al. | Vision language models (VLMs) extend the reasoning capabilities of large language models (LLMs) to cross-modal settings, yet remain highly vulnerable to multimodal jailbreak attacks. Existing defenses predominantly rely on safety fine-tuning or aggressive token manipulations, incurring substantial training costs or significantly degrading utility. Recent research shows that LLMs inherently recognize unsafe content in text, and the incorporation of visual inputs in VLMs frequently dilutes risk-related signals. Motivated by this, we propose Risk Awareness Injection (RAI), a lightweight and training-free framework for safety calibration that restores LLM-like risk recognition by amplifying unsafe signals in VLMs. Specifically, RAI constructs an Unsafe Prototype Subspace from language embeddings and performs targeted modulation on selected high-risk visual tokens, explicitly activating safety-critical signals within the cross-modal feature space. This modulation restores the model's LLM-like ability to detect unsafe content from visual inputs, while preserving the semantic integrity of original tokens for cross-modal reasoning. Extensive experiments across multiple jailbreak and utility benchmarks demonstrate that RAI substantially reduces attack success rate without compromising task performance. |
| 2026-02-03 | [Beyond Suffixes: Token Position in GCG Adversarial Attacks on Large Language Models](http://arxiv.org/abs/2602.03265v1) | Hicham Eddoubi, Umar Faruk Abdullahi et al. | Large Language Models (LLMs) have seen widespread adoption across multiple domains, creating an urgent need for robust safety alignment mechanisms. However, robustness remains challenging due to jailbreak attacks that bypass alignment via adversarial prompts. In this work, we focus on the prevalent Greedy Coordinate Gradient (GCG) attack and identify a previously underexplored attack axis in jailbreak attacks typically framed as suffix-based: the placement of adversarial tokens within the prompt. Using GCG as a case study, we show that both optimizing attacks to generate prefixes instead of suffixes and varying adversarial token position during evaluation substantially influence attack success rates. Our findings highlight a critical blind spot in current safety evaluations and underline the need to account for the position of adversarial tokens in the adversarial robustness evaluation of LLMs. |
| 2026-02-03 | [CSR-Bench: A Benchmark for Evaluating the Cross-modal Safety and Reliability of MLLMs](http://arxiv.org/abs/2602.03263v1) | Yuxuan Liu, Yuntian Shi et al. | Multimodal large language models (MLLMs) enable interaction over both text and images, but their safety behavior can be driven by unimodal shortcuts instead of true joint intent understanding. We introduce CSR-Bench, a benchmark for evaluating cross-modal reliability through four stress-testing interaction patterns spanning Safety, Over-rejection, Bias, and Hallucination, covering 61 fine-grained types. Each instance is constructed to require integrated image-text interpretation, and we additionally provide paired text-only controls to diagnose modality-induced behavior shifts. We evaluate 16 state-of-the-art MLLMs and observe systematic cross-modal alignment gaps. Models show weak safety awareness, strong language dominance under interference, and consistent performance degradation from text-only controls to multimodal inputs. We also observe a clear trade-off between reducing over-rejection and maintaining safe, non-discriminatory behavior, suggesting that some apparent safety gains may come from refusal-oriented heuristics rather than robust intent understanding. WARNING: This paper contains unsafe contents. |
| 2026-02-03 | [LPS-Bench: Benchmarking Safety Awareness of Computer-Use Agents in Long-Horizon Planning under Benign and Adversarial Scenarios](http://arxiv.org/abs/2602.03255v1) | Tianyu Chen, Chujia Hu et al. | Computer-use agents (CUAs) that interact with real computer systems can perform automated tasks but face critical safety risks. Ambiguous instructions may trigger harmful actions, and adversarial users can manipulate tool execution to achieve malicious goals. Existing benchmarks mostly focus on short-horizon or GUI-based tasks, evaluating on execution-time errors but overlooking the ability to anticipate planning-time risks. To fill this gap, we present LPS-Bench, a benchmark that evaluates the planning-time safety awareness of MCP-based CUAs under long-horizon tasks, covering both benign and adversarial interactions across 65 scenarios of 7 task domains and 9 risk types. We introduce a multi-agent automated pipeline for scalable data generation and adopt an LLM-as-a-judge evaluation protocol to assess safety awareness through the planning trajectory. Experiments reveal substantial deficiencies in existing CUAs' ability to maintain safe behavior. We further analyze the risks and propose mitigation strategies to improve long-horizon planning safety in MCP-based CUA systems. We open-source our code at https://github.com/tychenn/LPS-Bench. |

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



