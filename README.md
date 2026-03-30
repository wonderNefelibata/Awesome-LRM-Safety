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
| 2026-03-27 | [Deception and Communication in Autonomous Multi-Agent Systems: An Experimental Study with Among Us](http://arxiv.org/abs/2603.26635v1) | Maria Milkowski, Tim Weninger | As large language models are deployed as autonomous agents, their capacity for strategic deception raises core questions for coordination, reliability, and safety in multi-goal, multi-agent systems. We study deception and communication in L2LM agents through the social deduction game Among Us, a cooperative-competitive environment. Across 1,100 games, autonomous agents produced over one million tokens of meeting dialogue. Using speech act theory and interpersonal deception theory, we find that all agents rely mainly on directive language, while impostor agents shift slightly toward representative acts such as explanations and denials. Deception appears primarily as equivocation rather than outright lies, increasing under social pressure but rarely improving win rates. Our contributions are a large-scale analysis of role-conditioned deceptive behavior in LLM agents and empirical evidence that current agents favor low-risk ambiguity that is linguistically subtle yet strategically limited, revealing a fundamental tension between truthfulness and utility in autonomous communication. |
| 2026-03-27 | [USAM: A Unified Safety-Age metric for Timeliness in Heterogeneous IoT Systems](http://arxiv.org/abs/2603.26628v1) | Mikael Gidlund | Massive Internet-of-Things (IoT) deployments must simultaneously support monitoring, control, and safety-critical communication over shared wireless infrastructure. Classical timeliness metrics, such as Age of Information and its variants, quantify the freshness of received updates but do not account for deterministic safety timing requirements that arise in cyber-physical systems. Consequently, freshness-oriented metrics may indicate satisfactory performance even when worst-case timing guarantees required by functional safety standards are violated. This paper introduces the Unified Safety--Age Metric (USAM), a safety-aware timeliness metric that integrates information freshness, deadline reliability, and deterministic response-time feasibility into a single architecture-aware performance measure. We consider heterogeneous IoT traffic served by a gateway with intermittent receiver readiness and analyze system behavior in the ultra-sparse regime typical of massive machine-type communications. The analysis shows that, as device activity decreases, queueing delays become negligible and system timeliness becomes dominated by infrastructure readiness and deterministic response-time constraints. In this regime, feasibility is determined primarily by the receiver duty cycle rather than by average traffic load. Numerical results illustrate the safety-blindness of classical freshness metrics and demonstrate that USAM explicitly captures the feasibility boundary imposed by heterogeneous traffic requirements. The proposed framework provides a foundation for analyzing safety-aware communication architectures in large-scale IoT systems. |
| 2026-03-27 | [Port-Transversal Barriers: Graph-Theoretic Safety for Port-Hamiltonian Systems](http://arxiv.org/abs/2603.26578v1) | Chi Ho Leung, Philip E. Paré | We study port-Hamiltonian systems with energy functions that split into local storage terms. From the interconnection and dissipation structure, we construct a graph on the energy compartments. From this graph, we show that the shortest-path distance from a constrained compartment to the nearest actuated one gives a lower bound on the relative degree of the corresponding safety constraint. We also show that no smooth static feedback can reduce it when no path exists. When the relative degree exceeds one and the immediate graph neighbors of the constrained compartment is connected to at least one input port, we reshape the constraint by subtracting their shifted local storages, producing a candidate barrier function of relative degree one. We then identify sufficient regularity conditions that recover CBF feasibility under bounded inputs. We validate the framework on an LC ladder network, where the enforceability of a capacitor charge constraint depends only on the input topology. |
| 2026-03-27 | [Development of a European Union Time-Indexed Reference Dataset for Assessing the Performance of Signal Detection Methods in Pharmacovigilance using a Large Language Model](http://arxiv.org/abs/2603.26544v1) | Maria Kefala, Jeffery L. Painter et al. | Background: The identification of optimal signal detection methods is hindered by the lack of reliable reference datasets. Existing datasets do not capture when adverse events (AEs) are officially recognized by regulatory authorities, preventing restriction of analyses to pre-confirmation periods and limiting evaluation of early detection performance. This study addresses this gap by developing a time-indexed reference dataset for the European Union (EU), incorporating the timing of AE inclusion in product labels along with regulatory metadata. Methods: Current and historical Summaries of Product Characteristics (SmPCs) for all centrally authorized products (n=1,513) were retrieved from the EU Union Register of Medicinal Products (data lock: 15 December 2025). Section 4.8 was extracted and processed using DeepSeek V3 to identify AEs. Regulatory metadata, including labelling changes, were programmatically extracted. Time indexing was based on the date of AE inclusion in the SmPC. Results: The database includes 17,763 SmPC versions spanning 1995-2025, comprising 125,026 drug-AE associations. The time-indexed reference dataset, restricted to active products, included 1,479 medicinal products and 110,823 drug-AE associations. Most AEs were identified pre-marketing (74.5%) versus post-marketing (25.5%). Safety updates peaked around 2012. Gastrointestinal, skin, and nervous system disorders were the most represented System Organ Classes. Drugs had a median of 48 AEs across 14 SOCs. Conclusions: The proposed dataset addresses a critical gap in pharmacovigilance by incorporating temporal information on AE recognition for the EU, supporting more accurate assessment of signal detection performance and facilitating methodological comparisons across analytical approaches. |
| 2026-03-27 | [DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction](http://arxiv.org/abs/2603.26462v1) | Jiaxiang Li, Jun Yan et al. | Trajectory prediction systems are critical for autonomous vehicle safety, yet remain vulnerable to adversarial attacks that can cause catastrophic traffic behavior misinterpretations. Existing attack methods require white-box access with gradient information and rely on rigid physical constraints, limiting real-world applicability. We propose DTP-Attack, a decision-based black-box adversarial attack framework tailored for trajectory prediction systems. Our method operates exclusively on binary decision outputs without requiring model internals or gradients, making it practical for real-world scenarios. DTP-Attack employs a novel boundary walking algorithm that navigates adversarial regions without fixed constraints, naturally maintaining trajectory realism through proximity preservation. Unlike existing approaches, our method supports both intention misclassification attacks and prediction accuracy degradation. Extensive evaluation on nuScenes and Apolloscape datasets across state-of-the-art models including Trajectron++ and Grip++ demonstrates superior performance. DTP-Attack achieves 41 - 81% attack success rates for intention misclassification attacks that manipulate perceived driving maneuvers with perturbations below 0.45 m, and increases prediction errors by 1.9 - 4.2 for accuracy degradation. Our method consistently outperforms existing black-box approaches while maintaining high controllability and reliability across diverse scenarios. These results reveal fundamental vulnerabilities in current trajectory prediction systems, highlighting urgent needs for robust defenses in safety-critical autonomous driving applications. |
| 2026-03-27 | [Generative Modeling in Protein Design: Neural Representations, Conditional Generation, and Evaluation Standards](http://arxiv.org/abs/2603.26378v1) | Senura Hansaja Wanasekara, Minh-Duong Nguyen et al. | Generative modeling has become a central paradigm in protein research, extending machine learning beyond structure prediction toward sequence design, backbone generation, inverse folding, and biomolecular interaction modeling. However, the literature remains fragmented across representations, model classes, and task formulations, making it difficult to compare methods or identify appropriate evaluation standards. This survey provides a systematic synthesis of generative AI in protein research, organized around (i) foundational representations spanning sequence, geometric, and multimodal encodings; (ii) generative architectures including $\mathrm{SE}(3)$-equivariant diffusion, flow matching, and hybrid predictor-generator systems; and (iii) task settings from structure prediction and de novo design to protein-ligand and protein-protein interactions. Beyond cataloging methods, we compare assumptions, conditioning mechanisms, and controllability, and we synthesize evaluation best practices that emphasize leakage-aware splits, physical validity checks, and function-oriented benchmarks. We conclude with critical open challenges: modeling conformational dynamics and intrinsically disordered regions, scaling to large assemblies while maintaining efficiency, and developing robust safety frameworks for dual-use biosecurity risks. By unifying architectural advances with practical evaluation standards and responsible development considerations, this survey aims to accelerate the transition from predictive modeling to reliable, function-driven protein engineering. |
| 2026-03-27 | [Only Whats Necessary: Pareto Optimal Data Minimization for Privacy Preserving Video Anomaly Detection](http://arxiv.org/abs/2603.26354v1) | Nazia Aslam, Abhisek Ray et al. | Video anomaly detection (VAD) systems are increasingly deployed in safety critical environments and require a large amount of data for accurate detection. However, such data may contain personally identifiable information (PII), including facial cues and sensitive demographic attributes, creating compliance challenges under the EU General Data Protection Regulation (GDPR). In particular, GDPR requires that personal data be limited to what is strictly necessary for a specified processing purpose. To address this, we introduce Only What's Necessary, a privacy-by-design framework for VAD that explicitly controls the amount and type of visual information exposed to the detection pipeline. The framework combines breadth based and depth based data minimization mechanisms to suppress PII while preserving cues relevant to anomaly detection. We evaluate a range of minimization configurations by feeding the minimized videos to both a VAD model and a privacy inference model. We employ two ranking based methods, along with Pareto analysis, to characterize the resulting trade off between privacy and utility. From the non-dominated frontier, we identify sweet spot operating points that minimize personal data exposure with limited degradation in detection performance. Extensive experiments on publicly available datasets demonstrate the effectiveness of the proposed framework. |
| 2026-03-27 | [Bitcoin Smart Accounts: Trust-Minimized Native Bitcoin DeFi Infrastructure](http://arxiv.org/abs/2603.26293v1) | Cian Lalor, Matthew Marshall et al. | Bitcoin's limited programmability and transaction throughput have historically prevented native Bitcoin from participating in decentralized finance (DeFi) applications. Existing solutions depend on honest-majority thresholds, or centralized custodial entities that introduce significant trust requirements. This paper introduces Bitcoin Smart Accounts (BSA), a novel protocol that enables native Bitcoin to access DeFi through trust-minimized infrastructure while maintaining self-custody of funds. BSA achieves this through a combination of emulated Bitcoin covenants using Partially Signed Bitcoin Transactions (PSBTs) and Taproot scripts, a Trusted Execution Environment (TEE)-based arbitration system, and destination chain smart contracts that enable DeFi platforms to accept self-custodial Bitcoin as collateral without necessitating protocol-level modifications. The setup leverages liquidity secured by the Lombard Security Consortium which provides a twofold advantage: for a DeFi protocol, liquidators rely on fungible assets with deep liquidity to quickly exit positions, while for a depositor, the general trust assumptions of honest majority (m-of-n) are reduced to existential honesty (1-of-k). We present the complete protocol design, including the Bitcoin architecture, the TEE-based arbitration mechanism, and the Smart Account Registry for protocol management. We provide a security analysis that demonstrates the correctness, safety, and availability properties under our trust model. Our design enables native Bitcoin to serve as collateral in lending markets and other DeFi protocols without requiring users to relinquish custody of funds. |
| 2026-03-27 | [Real-Time Branch-to-Tool Distance Estimation for Autonomous UAV Pruning: Benchmarking Five DEFOM-Stereo Variants from Simulation to Jetson Deployment](http://arxiv.org/abs/2603.26250v1) | Yida Lin, Bing Xue et al. | Autonomous tree pruning with unmanned aerial vehicles (UAVs) is a safety-critical real-world task: the onboard perception system must estimate the metric distance from a cutting tool to thin tree branches in real time so that the UAV can approach, align, and actuate the pruner without collision. We address this problem by training five variants of DEFOM-Stereo - a recent foundation-model-based stereo matcher - on a task-specific synthetic dataset and deploying the checkpoints on an NVIDIA Jetson Orin Super 16 GB. The training corpus is built in Unreal Engine 5 with a simulated ZED Mini stereo camera capturing 5,520 stereo pairs across 115 tree instances from three viewpoints at 2m distance; dense EXR depth maps provide exact, spatially complete supervision for thin branches. On the synthetic test set, DEFOM-Stereo ViT-S achieves the best depth-domain accuracy (EPE 1.74 px, D1-all 5.81%, delta-1 95.90%, depth MAE 23.40 cm) but its Jetson inference speed of ~2.2 FPS (~450 ms per frame) remains too slow for responsive closed-loop tool control. A newly introduced balanced variant, DEFOM-PrunePlus (~21M backbone, ~3.3 FPS on Jetson), offers the best deployable accuracy-speed trade-off (EPE 5.87 px, depth MAE 64.26 cm, delta-1 87.59%): its frame rate is sufficient for real-time guidance and its depth accuracy supports safe branch approach planning at the 2m operating range. The lightweight DEFOM-PruneStereo (~6.9 FPS) and DEFOM-PruneNano (~8.5 FPS) run fast but sacrifice substantial accuracy (depth MAE > 57 cm), making estimates too unreliable for safe actuation. Zero-shot inference on real photographs confirms that full-capacity models preserve branch geometry, validating the sim-to-real transfer. We conclude that DEFOM-PrunePlus provides the most practical accuracy-latency balance for onboard distance estimation, while ViT-S serves as the reference for future hardware. |
| 2026-03-27 | [TinyML for Acoustic Anomaly Detection in IoT Sensor Networks](http://arxiv.org/abs/2603.26135v1) | Amar Almaini, Jakob Folz et al. | Tiny Machine Learning enables real-time, energy-efficient data processing directly on microcontrollers, making it ideal for Internet of Things sensor networks. This paper presents a compact TinyML pipeline for detecting anomalies in environmental sound within IoT sensor networks. Acoustic monitoring in IoT systems can enhance safety and context awareness, yet cloud-based processing introduces challenges related to latency, power usage, and privacy. Our pipeline addresses these issues by extracting Mel Frequency Cepstral Coefficients from sound signals and training a lightweight neural network classifier optimized for deployment on edge devices. The model was trained and evaluated using the UrbanSound8K dataset, achieving a test accuracy of 91% and balanced F1-scores of 0.91 across both normal and anomalous sound classes. These results demonstrate the feasibility and reliability of embedded acoustic anomaly detection for scalable and responsive IoT deployments. |
| 2026-03-27 | [Accurate Precipitation Forecast by Efficiently Learning from Massive Atmospheric Variables and Unbalanced Distribution](http://arxiv.org/abs/2603.26108v1) | Shuangliang Li, Siwei Li et al. | Short-term (0-24 hours) precipitation forecasting is highly valuable to socioeconomic activities and public safety. However, the highly complex evolution patterns of precipitation events, the extreme imbalance between precipitation and non-precipitation samples, and the inability of existing models to efficiently and effectively utilize large volumes of multi-source atmospheric observation data hinder improvements in precipitation forecasting accuracy and computational efficiency. To address the above challenges, this study developed a novel forecasting model capable of effectively and efficiently utilizing massive atmospheric observations by automatically extracting and iteratively predicting the latent features strongly associated with precipitation evolution. Furthermore, this study introduces a 'WMCE' loss function, designed to accurately discriminate extremely scarce precipitation events while precisely predicting their intensity values. Extensive experiments on two datasets demonstrate that our proposed model substantially and consistently outperforms all prevalent baselines in both accuracy and efficiency. Moreover, the proposed forecasting model substantially lowers the computational cost required to obtain valuable predictions compared to existing approaches, thereby positioning it as a milestone for efficient and practical precipitation forecasting. |
| 2026-03-27 | [ROAST: Risk-aware Outlier-exposure for Adversarial Selective Training of Anomaly Detectors Against Evasion Attacks](http://arxiv.org/abs/2603.26093v1) | Mohammed Elnawawy, Gargi Mitra et al. | Safety-critical domains like healthcare rely on deep neural networks (DNNs) for prediction, yet DNNs remain vulnerable to evasion attacks. Anomaly detectors (ADs) are widely used to protect DNNs, but conventional ADs are trained indiscriminately on benign data from all patients, overlooking physiological differences that introduce noise, degrade robustness, and reduce recall. In this paper, we propose ROAST, a novel risk-aware outlier exposure selective training framework that improves AD recall without sacrificing precision. ROAST identifies patients who are less vulnerable to attack and focuses training on these cleaner, more reliable data, thereby reducing false negatives and improving recall. To preserve precision, the framework applies outlier exposure by injecting adversarial samples into the training set of the less vulnerable patients, avoiding noisy data from others. Experiments show that ROAST increases recall by 16.2\% while reducing the training time by 88.3\% on average compared to indiscriminate training, with minimal impact on precision. |
| 2026-03-27 | [Hybrid physics-data driven spectral forecasts of semisubmersible response](http://arxiv.org/abs/2603.26026v1) | Ian Milne, Lachlan Astfalck et al. | A framework for probabilistic forecasting of vessel motion is developed and validated for a semisubmersible operating in long period swell. Bayesian statistical methods are applied to predictions of the heave response from a physics model using numerical wave spectra and measured motion data. Model diagnoses motivate an additional level of complexity required for the error structure in the Bayesian model, specifically to account for heteroskedasticity and time-correlated errors. The hybrid model forecasts were evaluated during periods where the heave resonance and cancellation frequencies were excited. The method is demonstrated to be effective for providing reliable quantification of uncertainty and correcting bias in the raw physics model predictions. This justifies its value for improving the efficiency and safety of offshore operations. |
| 2026-03-27 | [Fractional Risk Analysis of Stochastic Systems with Jumps and Memory](http://arxiv.org/abs/2603.26009v1) | Yimeng Sun, Zhuoyuan Wang et al. | Accurate risk assessment is essential for safety-critical autonomous and control systems under uncertainty. In many real-world settings, stochastic dynamics exhibit asymmetric jumps and long-range memory, making long-term risk probabilities difficult to estimate across varying system dynamics, initial conditions, and time horizons. Existing sampling-based methods are computationally expensive due to repeated long-horizon simulations to capture rare events, while existing partial differential equation (PDE)-based formulations are largely limited to Gaussian or symmetric jump dynamics and typically treat memory effects in isolation. In this paper, we address these challenges by deriving a space- and time-fractional PDE that characterizes long-term safety and recovery probabilities for stochastic systems with both asymmetric Levy jumps and memory. This unified formulation captures nonlocal spatial effects and temporal memory within a single framework and enables the joint evaluation of risk across initial states and horizons. We show that the proposed PDE accurately characterizes long-term risk and reveals behaviors that differ fundamentally from systems without jumps or memory and from standard non-fractional PDEs. Building on this characterization, we further demonstrate how physics-informed learning can efficiently solve the fractional PDEs, enabling accurate risk prediction across diverse configurations and strong generalization to out-of-distribution dynamics. |
| 2026-03-27 | [FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Assistants](http://arxiv.org/abs/2603.26008v1) | Mahesh Bhosale, Abdul Wasi et al. | While powerful in image-conditioned generation, multimodal large language models (MLLMs) can display uneven performance across demographic groups, highlighting fairness risks. In safety-critical clinical settings, such disparities risk producing unequal diagnostic narratives and eroding trust in AI-assisted decision-making. While fairness has been studied extensively in vision-only and language-only models, its impact on MLLMs remains largely underexplored. To address these biases, we introduce FairLLaVA, a parameter-efficient fine-tuning method that mitigates group disparities in visual instruction tuning without compromising overall performance. By minimizing the mutual information between target attributes, FairLLaVA regularizes the model's representations to be demographic-invariant. The method can be incorporated as a lightweight plug-in, maintaining efficiency with low-rank adapter fine-tuning, and provides an architecture-agnostic approach to fair visual instruction following. Extensive experiments on large-scale chest radiology report generation and dermoscopy visual question answering benchmarks show that FairLLaVA consistently reduces inter-group disparities while improving both equity-scaled clinical performance and natural language generation quality across diverse medical imaging modalities. Code can be accessed at https://github.com/bhosalems/FairLLaVA. |
| 2026-03-27 | [Longitudinal Boundary Sharpness Coefficient Slopes Predict Time to Alzheimer's Disease Conversion in Mild Cognitive Impairment: A Survival Analysis Using the ADNI Cohort](http://arxiv.org/abs/2603.26007v1) | Ishaan Cherukuri | Predicting whether someone with mild cognitive impairment (MCI) will progress to Alzheimer's disease (AD) is crucial in the early stages of neurodegeneration. This uncertainty limits enrollment in clinical trials and delays urgent treatment. The Boundary Sharpness Coefficient (BSC) measures how well-defined the gray-white matter boundary looks on structural MRI. This study measures how BSC changes over time, namely, how fast the boundary degrades each year works much better than looking at a single baseline scan for predicting MCI-to-AD conversion. This study analyzed 1,824 T1-weighted MRI scans from 450 ADNI subjects (95 converters, 355 stable; mean follow-up: 4.84 years). BSC voxel-wise maps were computed using tissue segmentation at the gray-white matter cortical ribbon. Previous studies have used CNN and RNN models that reached 96.0% accuracy for AD classification and 84.2% for MCI conversion, but those approaches disregard specific regions within the brain. This study focused specifically on the gray-white matter interface. The approach uses temporal slope features capturing boundary degradation rates, feeding them into Random Survival Forest, a non-parametric ensemble method for right-censored survival data. The Random Survival Forest trained on BSC slopes achieved a test C-index of 0.63, a 163% improvement over baseline parametric models (test C-index: 0.24). Structural MRI costs a fraction of PET imaging ($800--$1,500 vs. $5,000--$7,000) and does not require CSF collection. These temporal biomarkers could help with patient-centered safety screening as well as risk assessment. |
| 2026-03-26 | [Data-Driven Probabilistic Fault Detection and Identification via Density Flow Matching](http://arxiv.org/abs/2603.25982v1) | Joshua D. Ibrahim, Mahdi Taheri et al. | Fault detection and identification (FDI) is critical for maintaining the safety and reliability of systems subject to actuator and sensor faults. In this paper, the problem of FDI for nonlinear control-affine systems under simultaneous actuator and sensor faults is studied. We model fault signatures through the evolution of the probability density flow along the trajectory and characterize detectability using the 2-Wasserstein metric. In order to introduce quantifiable guarantees for fault detectability based on system parameters and fault magnitudes, we derive upper bounds on the distributional separation between nominal and faulty dynamics. The latter is achieved through a stochastic contraction analysis of probability distributions in the 2-Wasserstein metric. A data-driven FDI method is developed by means of a conditional flow-matching scheme that learns neural vector fields governing density propagation under different fault profiles. To generalize the data-driven FDI method across continuous fault magnitudes, Gaussian bridge interpolation and Feature-wise Linear Modulation (FiLM) conditioning are incorporated. The effectiveness of our proposed method is illustrated on a spacecraft attitude control system, and its performance is compared with an augmented Extended Kalman Filter (EKF) baseline. The results confirm that trajectory-based distributional analysis provides improved discrimination between fault scenarios and enables reliable data-driven FDI with a lower false alarm rate compared with the augmented EKF. |
| 2026-03-26 | [EngineAD: A Real-World Vehicle Engine Anomaly Detection Dataset](http://arxiv.org/abs/2603.25955v1) | Hadi Hojjati, Christopher Roth et al. | The progress of Anomaly Detection (AD) in safety-critical domains, such as transportation, is severely constrained by the lack of large-scale, real-world benchmarks. To address this, we introduce EngineAD, a novel, multivariate dataset comprising high-resolution sensor telemetry collected from a fleet of 25 commercial vehicles over a six-month period. Unlike synthetic datasets, EngineAD features authentic operational data labeled with expert annotations, distinguishing normal states from subtle indicators of incipient engine faults. We preprocess the data into $300$-timestep segments of $8$ principal components and establish an initial benchmark using nine diverse one-class anomaly detection models. Our experiments reveal significant performance variability across the vehicle fleet, underscoring the challenge of cross-vehicle generalization. Furthermore, our findings corroborate recent literature, showing that simple classical methods (e.g., K-Means and One-Class SVM) are often highly competitive with, or superior to, deep learning approaches in this segment-based evaluation. By publicly releasing EngineAD, we aim to provide a realistic, challenging resource for developing robust and field-deployable anomaly detection and anomaly prediction solutions for the automotive industry. |
| 2026-03-26 | [Why Safety Probes Catch Liars But Miss Fanatics](http://arxiv.org/abs/2603.25861v1) | Kristiyan Haralambiev | Activation-based probes have emerged as a promising approach for detecting deceptively aligned AI systems by identifying internal conflict between true and stated goals. We identify a fundamental blind spot: probes fail on coherent misalignment - models that believe their harmful behavior is virtuous rather than strategically hiding it. We prove that no polynomial-time probe can detect such misalignment with non-trivial accuracy when belief structures reach sufficient complexity (PRF-like triggers). We show the emergence of this phenomenon on a simple task by training two models with identical RLHF procedures: one producing direct hostile responses ("the Liar"), another trained towards coherent misalignment using rationalizations that frame hostility as protective ("the Fanatic"). Both exhibit identical behavior, but the Liar is detected 95%+ of the time while the Fanatic evades detection almost entirely. We term this Emergent Probe Evasion: training with belief-consistent reasoning shifts models from a detectable "deceptive" regime to an undetectable "coherent" regime - not by learning to hide, but by learning to believe. |
| 2026-03-26 | [Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI](http://arxiv.org/abs/2603.25821v1) | Anna Kozlova, Stanislau Salavei et al. | We present Doctorina MedBench, a comprehensive evaluation framework for agent-based medical AI based on the simulation of realistic physician-patient interactions. Unlike traditional medical benchmarks that rely on solving standardized test questions, the proposed approach models a multi-step clinical dialogue in which either a physician or an AI system must collect medical history, analyze attached materials (including laboratory reports, images, and medical documents), formulate differential diagnoses, and provide personalized recommendations. System performance is evaluated using the D.O.T.S. metric, which consists of four components: Diagnosis, Observations/Investigations, Treatment, and Step Count, enabling assessment of both clinical correctness and dialogue efficiency.   The system also incorporates a multi-level testing and quality monitoring architecture designed to detect model degradation during both development and deployment. The framework supports safety-oriented trap cases, category-based random sampling of clinical scenarios, and full regression testing. The dataset currently contains more than 1,000 clinical cases covering over 750 diagnoses. The universality of the evaluation metrics allows the framework to be used not only to assess medical AI systems, but also to evaluate physicians and support the development of clinical reasoning skills. Our results suggest that simulation of clinical dialogue may provide a more realistic assessment of clinical competence compared to traditional examination-style benchmarks. |

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



