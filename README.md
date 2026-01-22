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
| 2026-01-21 | [Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](http://arxiv.org/abs/2601.15220v1) | Anmol Goel, Cornelius Emde et al. | We identify a novel phenomenon in language models: benign fine-tuning of frontier models can lead to privacy collapse. We find that diverse, subtle patterns in training data can degrade contextual privacy, including optimisation for helpfulness, exposure to user information, emotional and subjective dialogue, and debugging code printing internal variables, among others. Fine-tuned models lose their ability to reason about contextual privacy norms, share information inappropriately with tools, and violate memory boundaries across contexts. Privacy collapse is a ``silent failure'' because models maintain high performance on standard safety and utility benchmarks whilst exhibiting severe privacy vulnerabilities. Our experiments show evidence of privacy collapse across six models (closed and open weight), five fine-tuning datasets (real-world and controlled data), and two task categories (agentic and memory-based). Our mechanistic analysis reveals that privacy representations are uniquely fragile to fine-tuning, compared to task-relevant features which are preserved. Our results reveal a critical gap in current safety evaluations, in particular for the deployment of specialised agents. |
| 2026-01-21 | [TTCBF: A Truncated Taylor Control Barrier Function for High-Order Safety Constraints](http://arxiv.org/abs/2601.15196v1) | Jianye Xu, Bassam Alrifaee | Control Barrier Functions (CBFs) enforce safety by rendering a prescribed safe set forward invariant. However, standard CBFs are limited to safety constraints with relative degree one, while High-Order CBF (HOCBF) methods address higher relative degree at the cost of introducing a chain of auxiliary functions and multiple class K functions whose tuning scales with the relative degree. In this paper, we introduce a Truncated Taylor Control Barrier Function (TTCBF), which generalizes standard discrete-time CBFs to consider high-order safety constraints and requires only one class K function, independent of the relative degree. We also propose an adaptive variant, adaptive TTCBF (aTTCBF), that optimizes an online gain on the class K function to improve adaptability, while requiring fewer control design parameters than existing adaptive HOCBF variants. Numerical experiments in a relative-degree-six spring-mass system and a cluttered corridor navigation validate the above theoretical findings. |
| 2026-01-21 | [Large-Scale Multidimensional Knowledge Profiling of Scientific Literature](http://arxiv.org/abs/2601.15170v1) | Zhucun Xue, Jiangning Zhang et al. | The rapid expansion of research across machine learning, vision, and language has produced a volume of publications that is increasingly difficult to synthesize. Traditional bibliometric tools rely mainly on metadata and offer limited visibility into the semantic content of papers, making it hard to track how research themes evolve over time or how different areas influence one another. To obtain a clearer picture of recent developments, we compile a unified corpus of more than 100,000 papers from 22 major conferences between 2020 and 2025 and construct a multidimensional profiling pipeline to organize and analyze their textual content. By combining topic clustering, LLM-assisted parsing, and structured retrieval, we derive a comprehensive representation of research activity that supports the study of topic lifecycles, methodological transitions, dataset and model usage patterns, and institutional research directions. Our analysis highlights several notable shifts, including the growth of safety, multimodal reasoning, and agent-oriented studies, as well as the gradual stabilization of areas such as neural machine translation and graph-based methods. These findings provide an evidence-based view of how AI research is evolving and offer a resource for understanding broader trends and identifying emerging directions. Code and dataset: https://github.com/xzc-zju/Profiling_Scientific_Literature |
| 2026-01-21 | [Automated Rubrics for Reliable Evaluation of Medical Dialogue Systems](http://arxiv.org/abs/2601.15161v1) | Yinzhu Chen, Abdine Maiga et al. | Large Language Models (LLMs) are increasingly used for clinical decision support, where hallucinations and unsafe suggestions may pose direct risks to patient safety. These risks are particularly challenging as they often manifest as subtle clinical errors that evade detection by generic metrics, while expert-authored fine-grained rubrics remain costly to construct and difficult to scale. In this paper, we propose a retrieval-augmented multi-agent framework designed to automate the generation of instance-specific evaluation rubrics. Our approach grounds evaluation in authoritative medical evidence by decomposing retrieved content into atomic facts and synthesizing them with user interaction constraints to form verifiable, fine-grained evaluation criteria. Evaluated on HealthBench, our framework achieves a Clinical Intent Alignment (CIA) score of 60.12%, a statistically significant improvement over the GPT-4o baseline (55.16%). In discriminative tests, our rubrics yield a mean score delta ($μ_Δ = 8.658$) and an AUROC of 0.977, nearly doubling the quality separation achieved by GPT-4o baseline (4.972). Beyond evaluation, our rubrics effectively guide response refinement, improving quality by 9.2% (from 59.0% to 68.2%). This provides a scalable and transparent foundation for both evaluating and improving medical LLMs. The code is available at https://anonymous.4open.science/r/Automated-Rubric-Generation-AF3C/. |
| 2026-01-21 | [Training-Free and Interpretable Hateful Video Detection via Multi-stage Adversarial Reasoning](http://arxiv.org/abs/2601.15115v1) | Shuonan Yang, Yuchen Zhang et al. | Hateful videos pose serious risks by amplifying discrimination, inciting violence, and undermining online safety. Existing training-based hateful video detection methods are constrained by limited training data and lack of interpretability, while directly prompting large vision-language models often struggle to deliver reliable hate detection. To address these challenges, this paper introduces MARS, a training-free Multi-stage Adversarial ReaSoning framework that enables reliable and interpretable hateful content detection. MARS begins with the objective description of video content, establishing a neutral foundation for subsequent analysis. Building on this, it develops evidence-based reasoning that supports potential hateful interpretations, while in parallel incorporating counter-evidence reasoning to capture plausible non-hateful perspectives. Finally, these perspectives are synthesized into a conclusive and explainable decision. Extensive evaluation on two real-world datasets shows that MARS achieves up to 10% improvement under certain backbones and settings compared to other training-free approaches and outperforms state-of-the-art training-based methods on one dataset. In addition, MARS produces human-understandable justifications, thereby supporting compliance oversight and enhancing the transparency of content moderation workflows. The code is available at https://github.com/Multimodal-Intelligence-Lab-MIL/MARS. |
| 2026-01-21 | [Visual and Cognitive Demands of a Large Language Model-Powered In-vehicle Conversational Agent](http://arxiv.org/abs/2601.15034v1) | Chris Monk, Allegra Ayala et al. | Driver distraction remains a leading contributor to motor vehicle crashes, necessitating rigorous evaluation of new in-vehicle technologies. This study assessed the visual and cognitive demands associated with an advanced Large Language Model (LLM) conversational agent (Gemini Live) during on-road driving, comparing it against handsfree phone calls, visual turn-by-turn guidance (low load baseline), and the Operation Span (OSPAN) task (high load anchor). Thirty-two licensed drivers completed five secondary tasks while visual and cognitive demands were measured using the Detection Response Task (DRT) for cognitive load, eye-tracking for visual attention, and subjective workload ratings. Results indicated that Gemini Live interactions (both single-turn and multi-turn) and hands-free phone calls shared similar levels of cognitive load, between that of visual turn-by-turn guidance and OSPAN. Exploratory analysis showed that cognitive load remained stable across extended multi-turn conversations. All tasks maintained mean glance durations well below the well-established 2-second safety threshold, confirming low visual demand. Furthermore, drivers consistently dedicated longer glances to the roadway between brief off-road glances toward the device during task completion, particularly during voice-based interactions, rendering longer total-eyes-off-road time findings less consequential. Subjective ratings mirrored objective data, with participants reporting low effort, demands, and perceived distraction for Gemini Live. These findings demonstrate that advanced LLM conversational agents, when implemented via voice interfaces, impose cognitive and visual demands comparable to established, low-risk hands-free benchmarks, supporting their safe deployment in the driving environment. |
| 2026-01-21 | [Risk Estimation for Automated Driving](http://arxiv.org/abs/2601.15018v1) | Leon Tolksdorf, Arturo Tejada et al. | Safety is a central requirement for automated vehicles. As such, the assessment of risk in automated driving is key in supporting both motion planning technologies and safety evaluation. In automated driving, risk is characterized by two aspects. The first aspect is the uncertainty on the state estimates of other road participants by an automated vehicle. The second aspect is the severity of a collision event with said traffic participants. Here, the uncertainty aspect typically causes the risk to be non-zero for near-collision events. This makes risk particularly useful for automated vehicle motion planning. Namely, constraining or minimizing risk naturally navigates the automated vehicle around traffic participants while keeping a safety distance based on the level of uncertainty and the potential severity of the impending collision. Existing approaches to calculate the risk either resort to empirical modeling or severe approximations, and, hence, lack generalizability and accuracy. In this paper, we combine recent advances in collision probability estimation with the concept of collision severity to develop a general method for accurate risk estimation. The proposed method allows us to assign individual severity functions for different collision constellations, such as, e.g., frontal or side collisions. Furthermore, we show that the proposed approach is computationally efficient, which is beneficial, e.g., in real-time motion planning applications. The programming code for an exemplary implementation of Gaussian uncertainties is also provided. |
| 2026-01-21 | [HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV](http://arxiv.org/abs/2601.14973v1) | Faryal Batool, Iana Zhura et al. | Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings. |
| 2026-01-21 | [Robust Machine Learning for Regulatory Sequence Modeling under Biological and Technical Distribution Shifts](http://arxiv.org/abs/2601.14969v1) | Yiyao Yang | Robust machine learning for regulatory genomics is studied under biologically and technically induced distribution shifts. Deep convolutional and attention based models achieve strong in distribution performance on DNA regulatory sequence prediction tasks but are usually evaluated under i.i.d. assumptions, even though real applications involve cell type specific programs, evolutionary turnover, assay protocol changes, and sequencing artifacts. We introduce a robustness framework that combines a mechanistic simulation benchmark with real data analysis on a massively parallel reporter assay (MPRA) dataset to quantify performance degradation, calibration failures, and uncertainty based reliability. In simulation, motif driven regulatory outputs are generated with cell type specific programs, PWM perturbations, GC bias, depth variation, batch effects, and heteroscedastic noise, and CNN, BiLSTM, and transformer models are evaluated. Models remain accurate and reasonably calibrated under mild GC content shifts but show higher error, severe variance miscalibration, and coverage collapse under motif effect rewiring and noise dominated regimes, revealing robustness gaps invisible to standard i.i.d. evaluation. Adding simple biological structural priors motif derived features in simulation and global GC content in MPRA improves in distribution error and yields consistent robustness gains under biologically meaningful genomic shifts, while providing only limited protection against strong assay noise. Uncertainty-aware selective prediction offers an additional safety layer that risk coverage analyses on simulated and MPRA data show that filtering low confidence inputs recovers low risk subsets, including under GC-based out-of-distribution conditions, although reliability gains diminish when noise dominates. |
| 2026-01-21 | [Contingency Planning for Safety-Critical Autonomous Vehicles: A Review and Perspectives](http://arxiv.org/abs/2601.14880v1) | Lei Zheng, Luyao Zhang et al. | Contingency planning is the architectural capability that enables autonomous vehicles (AVs) to anticipate and mitigate discrete, high-impact hazards, such as sensor outages and adversarial interactions. This paper presents a comprehensive survey of the field, synthesizing fragmented literature into a unified logic-conditioned hybrid control framework. Within this formalism, we categorize approaches into two distinct paradigms: Reactive Safety, which responds to realized hazards by enforcing safety constraints or executing fail-safe maneuvers; and Proactive Safety, which optimizes for future recourse by branching over potential modal transitions. In addition, we propose a fine-grained taxonomy that partitions the landscape into external contingencies (environmental and interactive hazards) and internal contingencies (system faults). Through a critical comparative analysis, we reveal a fundamental structural divergence: internal faults are predominantly addressed via reactive fail-safe mechanisms, whereas external interaction uncertainties increasingly require proactive branching strategies. Furthermore, we identify a critical methodological divergence: whereas physical hazards are typically managed with formal guarantees, semantic and out-of-distribution anomalies currently rely heavily on empirical validation. We conclude by identifying the open challenges in bridging the gap between theoretical guarantees and practical validation, advocating for hybrid architectures and standardized benchmarking to transition contingency planning from formulation to certifiable real-world deployment. |
| 2026-01-21 | [On-the-fly hand-eye calibration for the da Vinci surgical robot](http://arxiv.org/abs/2601.14871v1) | Zejian Cui, Ferdinando Rodriguez y Baena | In Robot-Assisted Minimally Invasive Surgery (RMIS), accurate tool localization is crucial to ensure patient safety and successful task execution. However, this remains challenging for cable-driven robots, such as the da Vinci robot, because erroneous encoder readings lead to pose estimation errors. In this study, we propose a calibration framework to produce accurate tool localization results through computing the hand-eye transformation matrix on-the-fly. The framework consists of two interrelated algorithms: the feature association block and the hand-eye calibration block, which provide robust correspondences for key points detected on monocular images without pre-training, and offer the versatility to accommodate various surgical scenarios by adopting an array of filter approaches, respectively. To validate its efficacy, we test the framework extensively on publicly available video datasets that feature multiple surgical instruments conducting tasks in both in vitro and ex vivo scenarios, under varying illumination conditions and with different levels of key point measurement accuracy. The results show a significant reduction in tool localization errors under the proposed calibration framework, with accuracies comparable to other state-of-the-art methods while being more time-efficient. |
| 2026-01-21 | [From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps](http://arxiv.org/abs/2601.14848v1) | Mohamed Abouras, Catherine M. Elias | On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon. |
| 2026-01-21 | [A Category-Theoretic Framework for Dependent Effect Systems](http://arxiv.org/abs/2601.14846v1) | Satoshi Kura, Marco Gaboardi et al. | Graded monads refine traditional monads using effect annotations in order to describe quantitatively the computational effects that a program can generate. They have been successfully applied to a variety of formal systems for reasoning about effectful computations. However, existing categorical frameworks for graded monads do not support effects that may depend on program values, which we call dependent effects, thereby limiting their expressiveness. We address this limitation by introducing indexed graded monads, a categorical generalization of graded monads inspired by the fibrational "indexed" view and by classical categorical semantics of dependent type theories. We show how indexed graded monads provide semantics for a refinement type system with dependent effects. We also show how this type system can be instantiated with specific choices of parameters to obtain several formal systems for reasoning about specific program properties. These instances include, in particular, cost analysis, probability-bound reasoning, expectation-bound reasoning, and temporal safety verification. |
| 2026-01-21 | [Stochastic Decision-Making Framework for Human-Robot Collaboration in Industrial Applications](http://arxiv.org/abs/2601.14809v1) | Muhammad Adel Yusuf, Ali Nasir et al. | Collaborative robots, or cobots, are increasingly integrated into various industrial and service settings to work efficiently and safely alongside humans. However, for effective human-robot collaboration, robots must reason based on human factors such as motivation level and aggression level. This paper proposes an approach for decision-making in human-robot collaborative (HRC) environments utilizing stochastic modeling. By leveraging probabilistic models and control strategies, the proposed method aims to anticipate human actions and emotions, enabling cobots to adapt their behavior accordingly. So far, most of the research has been done to detect the intentions of human co-workers. This paper discusses the theoretical framework, implementation strategies, simulation results, and potential applications of the bilateral collaboration approach for safety and efficiency in collaborative robotics. |
| 2026-01-21 | [Hierarchical Optimization Based Multi-objective Dynamic Regulation Scheme for VANET Topology](http://arxiv.org/abs/2601.14704v1) | Ruixing Ren, Minqi Tao et al. | As a core technology of intelligent transportation systems, vehicular ad-hoc networks support latency-sensitive services such as safety warning and cooperative perception via vehicle-to-everything communications. However, their highly dynamic topology increases average path length, raises latency, and reduces throughput, severely limiting communication performance. Existing topology optimization methods lack capabilities in multi-objective coordination, dynamic adaptation, and global-local synergy. To address this, this paper proposes a two-layer dynamic topology regulation scheme combining local feature aggregation and global adjustment. The scheme constructs a dynamic multi-objective optimization model integrating average path length, end-to-end latency, and network throughput, and achieves multi-index coordination via link adaptability metrics and a dynamic normalization mechanism. it quickly responds to local link changes via feature fusion of local node feature extraction and dynamic neighborhood sensing, and balances optimization accuracy and real-time performance using a dual-mode adaptive solving strategy for global topology adjustment. It reduces network oscillation risks by introducing a performance improvement threshold and a topology validity verification mechanism. Simulation results on real urban road networks via the SUMO platform show that the proposed scheme outperforms traditional methods in average path length (stabilizing at ~4 hops), end-to-end latency (remaining ~0.01 s), and network throughput. |
| 2026-01-21 | [Dosimetry for Proton Therapy Using a β-Ga$_2$O$_3$ Metal-Semiconductor-Metal Detector with Low-Noise Amplification](http://arxiv.org/abs/2601.14654v1) | Hunter D. Ellis, Ajayvarman Mallapillai et al. | Intensity-modulated proton therapy (IMPT) employs proton radiation rather than conventional X-rays to treat cancerous tumors. This approach offers significant advantages by minimizing the radiation exposure of surrounding healthy tissue, leading to improved patient outcomes and reduced side effects compared to traditional X-ray therapy. To ensure patient safety, each treatment plan must be experimentally validated before clinical implementation. However, current dosimetry devices face limitations in performing angled beam measurements and obtaining multi-depth assessments, both of which are essential for verifying IMPT treatment plans. In this study, the performance of a β-Ga$_2$O$_3$-based metal-semiconductor-metal (MSM) detector with a low-noise amplifier is studied and evaluated under various proton radiation doses and energy levels delivered by a MEVION S250i proton accelerator. The detector performance is also compared with that of an ionization chamber. The β-Ga$_2$O$_3$ detector exhibits a linear response with proton dose for single-spot irradiations, and its response to varying proton energies closely matches both the ion chamber data and simulated dose distributions. These findings highlight the potential of β-Ga$_2$O$_3$-based detectors as robust dosimetry devices for IMPT applications. |
| 2026-01-21 | [Input-to-State Stabilizing Neural Controllers for Unknown Switched Nonlinear Systems within Compact Sets](http://arxiv.org/abs/2601.14643v1) | Bhabani Shankar Dey, Ahan Basu et al. | This paper develops a neural network based control framework that ensures system safety and input-to-state stability (ISS) for general nonlinear switched systems with unknown dynamics. Leveraging the concept of dwell time, we derive Lyapunov based sufficient conditions under which both safety and ISS of the closed-loop switched system are guaranteed. The feedback controllers and the associated Lyapunov functions are parameterized using neural networks and trained from data collected over a compact state space via deterministic sampling. To provide formal stability guarantees under the learned controllers, we introduce a validity condition based on Lipschitz continuity assumptions, which is embedded directly into the training framework. This ensures that the resulting neural network controllers satisfy provable correctness and stability guarantees beyond the sampled data. As a special case, the proposed framework recovers ISS and safety under arbitrary switching when a common Lyapunov function exists. Simulation results on a representative switched nonlinear system demonstrate the effectiveness of the proposed approach. |
| 2026-01-21 | [A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control](http://arxiv.org/abs/2601.14628v1) | Weiyu Guo, He Zhang et al. | Recent advances in embodied intelligence have leveraged massive scaling of data and model parameters to master natural-language command following and multi-task control. In contrast, biological systems demonstrate an innate ability to acquire skills rapidly from sparse experience. Crucially, current robotic policies struggle to replicate the dynamic stability, reflexive responsiveness, and temporal memory inherent in biological motion. Here we present Neuromorphic Vision-Language-Action (NeuroVLA), a framework that mimics the structural organization of the bio-nervous system between the cortex, cerebellum, and spinal cord. We adopt a system-level bio-inspired design: a high-level model plans goals, an adaptive cerebellum module stabilizes motion using high-frequency sensors feedback, and a bio-inspired spinal layer executes lightning-fast actions generation. NeuroVLA represents the first deployment of a neuromorphic VLA on physical robotics, achieving state-of-the-art performance. We observe the emergence of biological motor characteristics without additional data or special guidance: it stops the shaking in robotic arms, saves significant energy(only 0.4w on Neuromorphic Processor), shows temporal memory ability and triggers safety reflexes in less than 20 milliseconds. |
| 2026-01-20 | [LLM Security and Safety: Insights from Homotopy-Inspired Prompt Obfuscation](http://arxiv.org/abs/2601.14528v1) | Luis Lazo, Hamed Jelodar et al. | In this study, we propose a homotopy-inspired prompt obfuscation framework to enhance understanding of security and safety vulnerabilities in Large Language Models (LLMs). By systematically applying carefully engineered prompts, we demonstrate how latent model behaviors can be influenced in unexpected ways. Our experiments encompassed 15,732 prompts, including 10,000 high-priority cases, across LLama, Deepseek, KIMI for code generation, and Claude to verify. The results reveal critical insights into current LLM safeguards, highlighting the need for more robust defense mechanisms, reliable detection strategies, and improved resilience. Importantly, this work provides a principled framework for analyzing and mitigating potential weaknesses, with the goal of advancing safe, responsible, and trustworthy AI technologies. |
| 2026-01-20 | [How Worst-Case Are Adversarial Attacks? Linking Adversarial and Statistical Robustness](http://arxiv.org/abs/2601.14519v1) | Giulio Rossolini | Adversarial attacks are widely used to evaluate model robustness, yet their validity as proxies for robustness to random perturbations remains debated. We ask whether an adversarial perturbation provides a representative estimate of robustness under random noise of the same magnitude, or instead reflects an atypical worst-case event. To this end, we introduce a probabilistic metric that quantifies noisy risk with respect to directionally biased perturbation distributions, parameterized by a concentration factor $κ$ that interpolates between isotropic noise and adversarial direction. Using this framework, we study the limits of adversarial perturbations as estimators of noisy risk by proposing an attack strategy designed to operate in regimes statistically closer to uniform noise. Experiments on ImageNet and CIFAR-10 systematically benchmark widely used attacks, highlighting when adversarial success meaningfully reflects noisy risk and when it fails, thereby informing their use in safety-oriented evaluation. |

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



