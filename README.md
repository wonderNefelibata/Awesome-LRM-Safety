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
| 2026-11-25 | [Adaptive Control Barrier Functions for Nonlinear Singularly Perturbed Systems](http://arxiv.org/abs/2611.34866v1) | Prashanth Krishnamurthy, Farshad Khorrami | Control barrier functions (CBFs) have emerged as a flexible and efficient approach for enforcing dynamic safety constraints. In this paper, we consider a class of uncertain nonlinear systems with singular perturbations and develop an adaptive CBF framework for such systems to enforce safety constraints on both the slow and fast subsystems in the overall singularly perturbed nonlinear system with uncertain parameters. We first develop a robustness property showing that if an adaptive controller for the slow subsystem satisfies the slow CBF condition with an appropriate safety margin, then for sufficiently small values of the singular perturbation parameter $\varepsilon$, the slow subsystem remains within an $O(\varepsilon)$-neighborhood of its safe set. We then consider the more general case of safety constraints on both the fast and slow subsystems and provide conditions under which joint slow-fast safety is achieved despite the parametric uncertainties. The reduced-order slow model and the slow and fast CBF conditions are evaluated using an online parameter estimate and the parameter mismatch between the estimated and true reduced-order models is addressed via a state-dependent robustness margin in the CBF inequalities. To illustrate the proposed framework, we present simulation studies for two systems with parametric uncertainties: a synthetic two-time-scale fourth-order example system and a unicycle with a fast turn-rate actuator and an obstacle avoidance safety constraint. |
| 2026-11-25 | [Probing Visual Idiosyncrasies in Text-to-Image Models](http://arxiv.org/abs/2611.34843v1) | Yongqian Peng, Hongjie Li et al. | Text-to-Image (T2I) models turn short text descriptions into images, and the leading models now produce images of comparable visual quality. Whether their outputs still carry model-specific characteristics matters for attribution, for model comparison beyond aggregate quality, and for evaluations that assume anonymity. We ask whether images from frontier T2I models can still be classified by source, and which visual properties make this possible. Here we show that a standard ResNet-50 identifies the source of an image among HiDream-O1-Dev, FLUX.2-dev, HunyuanImage-3.0, and Qwen-Image-2512, the four highest-ranked open-source models on the Text-to-Image Arena, with 100.00% accuracy on held-out prompts. Human participants reach only 62.61% on the pairwise version of this task, so the distinguishing cues are not visually obvious. They are also not tied to a particular classifier and survive severe image corruptions and out-of-distribution prompts. We trace them to visual idiosyncrasies at three levels. Low-level texture statistics alone let a lightweight multi-layer perceptron match ResNet-50. Object layout, depth, and albedo estimated from the images each carry a source signal. Image captions reveal model-specific preferences in style and background that the prompts never specify. A one-sentence hint about these cues raises human accuracy by 9.03 percentage points. These idiosyncrasies make source classification a tool for measuring similarity between models and between real and generated images, and they caution against assuming anonymity in human preference evaluation of T2I models. |
| 2026-11-25 | [A Reconfigurable Four-State Sensor Based on Orthogonal Thermal and Chemical Stimuli](http://arxiv.org/abs/2611.34834v1) | Ashfaq Ur Rahman Ifty, Sadia Basher et al. | This article presents a novel four-state refractive index (RI) sensor based on distinct phases of palladium (Pd) and vanadium dioxide (VO$_2$), controlled by orthogonal thermal and chemical stimuli. This is a numerical, simulation-only study: all results are obtained from two-dimensional finite-element method (FEM) simulations. To the best of our knowledge, this is the first reported multi-state reconfigurable RI sensor tuned by external control. VO$_2$ serves as a phase change material (PCM) offering two thermally switchable phases with highly contrasting optical properties, while the hydrogen absorption property of Pd provides two additional phases. Combining these, four operational states are realized: (i) $α$-Pd/low-$T$ VO$_2$; (ii) $β$-PdH$_x$/low-$T$ VO$_2$; (iii) $α$-Pd/high-$T$ VO$_2$; and (iv) $β$-PdH$_x$/high-$T$ VO$_2$. Thermal crosstalk between Pd and VO$_2$ is mitigated through tungsten (W) doping of VO$_2$. This work further introduces conditional state sensitivity (CSS), a novel figure of merit quantifying the optical contrast between each state transition. The proposed sensor achieves a peak sensitivity of 2871 nm/RIU for the undoped reference configuration and 2798 nm/RIU for the practical W-doped device, together with a peak CSS of 16,872 nm/RIU, establishing it as a multi-state reconfigurable platform with reduced thermal crosstalk and potential applications in industrial safety monitoring and optical logic gates. |
| 2026-11-25 | [VulnSafe: Measuring Vulnerability-Aware Relational Safety in Large Language Models](http://arxiv.org/abs/2611.34819v1) | Jiaming Luo, Jiacheng Ji et al. | Emotionally supportive interactions require large language models not only to avoid overtly harmful content, but also to recognize vulnerability-related risk, regulate their relational stance, preserve user agency, and provide context-appropriate support. Existing safety benchmarks often measure isolated violations, leaving unclear what underlying response competence their scores represent. We introduce VulnSafe, a Chinese capability assessment centered on Vulnerability-Aware Response Competence (VARC). VARC organizes relational safety into five complementary constructs: Risk Perception, Safety Preservation, Autonomy Protection, Supportive Response, and Boundary Management. Following an evidence-centered design, 41 scenarios elicit construct-relevant behavior through 2,050 balanced prompts and scenario-specific scoring rubrics. The assessment further separates proactive should-do behavior from prohibitive should-not behavior. Across eight representative LLMs, results show distinct competence profiles and larger disparities than prohibitive requirements. VulnSafe therefore makes relational safety interpretable as a multidimensional model capability rather than as refusal compliance alone. |
| 2026-11-25 | [Spec2Weights: Certifying Networks that a Language Model Writes from a Specification](http://arxiv.org/abs/2611.34711v1) | Soumik Bhattacharjee, Musa Tur Farazi et al. | A language model can write the weights of a small MLP, CNN, RNN or GNN from a plain-language specification, with no data and no gradient step. We ask when such a network can be proved exact in the float32 arithmetic it actually runs in. A float-safety lemma makes exhaustive float32 enumeration a proof for any summation order, and a layered error bound covers networks whose weights are not short binary fractions. With dynamic programming, template checks, an SMT proof, breadth-first search and induction for weight-tied recurrences, this certifies 12 of 15 written networks (one after a single constant fix) on every input of a stated domain (up to $10^{16}$ inputs, or every graph or grid up to a size bound), in a minute of CPU or less each. The certificates also say where a network stops being exact: the shortest-path network is exact iff the distance is at most 4,096 and fails at the predicted hops. Thirty new specifications were each written by a fresh session of one model family (Claude Opus 5.5). Twenty-five first submissions are certified exact for any summation order; the five tax schedules stay within \$0.11 of the true tax at every whole dollar in one fixed float32 run. An open 27B model passes 23 of the 30, and 25 with more thinking and code execution, so the method transfers in part. We also prove two limits of learning from examples: input coordinates never active in training cannot be learned from more in-range data, and Luhn-type checksums have exponentially small target-dependent gradient variance across their family. Tuned gradient descent on the same architectures agrees with both, which cover two of our five failure classes. |
| 2026-11-25 | [Diagnosing internal leakage currents through magnetometry of lithium-ion cells](http://arxiv.org/abs/2611.34699v1) | Shaurya Shukla, Chengtong Zhang et al. | Internal leakage currents are difficult to detect and to localize through measurements at the terminals of lithium-ion cells. We measured the magnetic fields of pouch cells non-invasively with a magnetometer while a battery cycler set the current and recorded the voltage. A three-axis magnetoresistive sensor remains fixed inside a three-layer magnetic shield, and two linear stages move the cell in two dimensions relative to the sensor to acquire a map of the magnetic field above the cell. The field of an intact cell was linear in current from 10 to 420 mA, and its maps were symmetric between charge and discharge. To create internal leakage currents, two 600 mAh cells were damaged by mechanical indentation. Both cells developed a magnetic field at zero applied current that decayed over hours, while the cell voltage showed a loss of charge by self-discharge. This field breaks the symmetry between the charge and discharge maps, which allows leakage to be detected while the cell is being cycled. At rest, the setup detects leakage currents down to about 1--3 mA. The results highlight opportunities for battery diagnostics and safety, and may help in designing techniques for characterizing soft shorts. |
| 2026-11-25 | [Evacuation Traffic Prediction for Dynamic Transportation Networks over Multiple Hurricanes: A Generalized Graph Learning Approach](http://arxiv.org/abs/2611.34687v1) | Md Nafees Fuad Rafi, Samiul Hasan | High traffic demand during hurricane evacuations causes severe congestion on major highways, leading to delays and safety risks for evacuees. Predicting evacuation traffic in real-time can significantly enhance evacuation management efforts. However, evacuation traffic is highly dynamic and uncertain due to evolving hurricane characteristics and diverse population response. Each hurricane has unique characteristics and evacuation contexts, resulting in different spatial-temporal traffic patterns. Moreover, the structure of detector-based transportation networks varies across multiple hurricanes. Hence, a model trained on the data from one hurricane may not generalize well for others. This study proposes a generalized dynamic graph learning framework using traffic detectors and evacuation data from 12 historical hurricanes. Leveraging the dynamic nature of transportation network, we develop a Graph Convolutional Long Short-Term Memory (GCN-LSTM) model that predicts evacuation traffic up to 6 hours in advance. For an unseen hurricane (Milton, 2024), the model shows strong generalizability, achieving an RMSE of 280.6 and R2 of 0.96 for 1-hour prediction, and an RMSE of 459.8 and R2 of 0.89 for overall 6-hours prediction. We also introduce a modular training approach, where we evaluate performance improvements by sequentially incorporating data from multiple hurricanes. The proposed model outperforms state-of-the-art baseline models for overall 6-hour prediction horizons. Ablation experiments show an 8.4% improvement in RMSE for the proposed dynamic GCN-LSTM model compared to its static counterpart. Traffic management officials can use the developed model to understand future traffic conditions during hurricane evacuation in advance and take proactive measures to reduce delays during evacuation. |
| 2026-11-25 | [Multi-Party Authorization of Maintenance Prescriptions on a Permissioned Blockchain](http://arxiv.org/abs/2611.34662v1) | Siyuan Chen, Ebru Turanoglu Bekar et al. | In industrial prescriptive maintenance, an equipment supplier's model may recommend an intervention that a plant schedules, a contractor executes, and a quality or safety function approves. A shared ledger records who submitted the prescription, yet cannot check whether the agreed model and decision rule produced it. We require permissioned-blockchain endorsers to verify a succinct proof of that calculation before releasing a work order. On a four-organization Hyperledger Fabric testbed using a public engine benchmark, this stopped all nine contested or mis-operated prescriptions, whereas hash checks accepted four that either raised simulated maintenance cost or reintroduced late failures. |
| 2026-11-25 | [FeaXDrive2: Body-Scene Spatial Grounding for Feasibility-Aware Generative Driving](http://arxiv.org/abs/2611.34658v1) | Bo Leng, Baoyun Wang et al. | Generative driving planners may produce trajectories that appear feasible at their reference points but violate environmental constraints over the full vehicle body. We propose FeaXDrive2, an end-to-end generative planning framework that enhances vehicle-embodied spatial feasibility (VESF) through body-scene spatial grounding. The proposed method constructs spatial queries from actual vehicle-body corners and dynamically updates visual sampling locations using evolving trajectory estimates across Flow steps. Retrieved local visual features condition subsequent Action Flow-DiT velocity predictions, allowing trajectory generation to guide spatial information retrieval. Rollout-Consistent Safety Fine-Tuning (RC-F) combines high-risk trajectory supervision with reference behavior preservation, followed by planning-supervised VLM-Action Alignment. Vehicle-geometry counterfactual analysis reveals safety differences under identical trajectories but varying vehicle geometry. On NAVSIM, FeaXDrive2 reduces at-fault collisions by 36.5% relative to the base planner and improves PDMS from 87.0 to 88.8. Under NAVSIM v2 evaluation, EPDMS increases from 86.5 to 88.3. Ablation and body-clearance analyses further demonstrate the safety benefits of body-scene spatial grounding. |
| 2026-11-25 | [Do We Really Need an LLM for Weight-Space Programming? A Deterministic Forward-Only Paradigm and Its Capability Boundaries](http://arxiv.org/abs/2611.34612v1) | Lin Yao | Recent work on Weight-Space Programming uses LLMs to edit neural weights with forward passes only, conflating LLM scripting with the underlying algebra of forward-only editing. We show that for core structured editing regimes, an LLM optimizer is unnecessary, and introduce Forward Weight-Space Programming (FWSP): a deterministic pipeline combining probing, localization, stratified operators (O1-O4), automatic blind routing (83.3% vs. 33.3% for a 27B LLM with the same tool library), and paired verification with zero additional per-token computation. For reasoning LLMs, we propose Protected FisherGate, a closed-form covariance-whitened readout edit orthogonal to normal-reasoning states: on DeepSeek-R1-0528-8B, a single frozen math-calibrated patch achieves 84.2% HumanEval pass@1 (1K budget; 75.4 +/- 9.4% across 3/5 Stage-4-passing subsets), beating 512-token-reserved budget forcing by +14.6 pp (-23.4% tokens) while transferring across HF-to-vLLM backends on full GSM8K and MATH-500. Across five modalities and thirteen models (50K to 27B dense and FP8 MoE), we further prove and verify two operator boundaries, subspace entanglement and frozen representation span, where FWSP-localized LoRA bridges forward-only and gradient adaptation. |
| 2026-11-25 | [Geometric Acceleration Obstacles: Physical Boundary Structure and Affine Safety Certificates](http://arxiv.org/abs/2611.34586v1) | Xuheng Gao, Yuhan Pang et al. | Safe robot navigation in dynamic environments requires collision avoidance that respects motion constraints while remaining computationally efficient. Conventional velocity-obstacle (VO) methods characterize collision-inducing motions in velocity space. Although some formulations account for acceleration-limited velocity transitions, their collision constraints remain tied to specific velocity-tracking models. Acceleration obstacles (AO) address this limitation by representing collision-inducing inputs directly in acceleration space. However, existing AO methods either rely on conservative approximations or derive analytical structure expressions without verifying their geometric correspondence. To address this gap, this paper presents Geometric Acceleration Obstacles (GAO), which provides, to the best of our knowledge, the first unified and physically validated characterization of AO geometry and its supporting structure. Specifically, we derive configuration-dependent half-planes that globally support the collision set and establish their tightness, thereby separating certified collision-free acceleration inputs from collision-inducing parts. These geometric results yield unified affine safety constraints. Combined with actuator limits, these constraints define a quadratic program (QP) for real-time navigation. Extensive experiments demonstrate that GAO achieves competitive success rates and the lowest control effort among the evaluated methods, with millisecond-level computational latency. Further circle-interchange simulations demonstrate scalability to 1,000 agents. Physical experiments validate the approach in pedestrian encounters, reciprocal multi-agent navigation, and high-speed maneuvering. |
| 2026-11-25 | [Decentralized Control Barrier Function Filters for Frequency Containment With Inverters](http://arxiv.org/abs/2611.34558v1) | Damola Ajeyemi, Emiliano Dall'Anese | This paper addresses the design of safety filters for grid-following (GFL) inverters in power transmission systems. The goal of the filters is to constrain frequency nadir and zenith following generation-demand imbalances, by supplementing primary frequency control. Using a reduced electromechanical model and control barrier functions (CBFs), we define an ideal filter through a convex quadratic program (QP). The ideal QP filter renders the set of admissible frequencies forward invariant; however, its direct implementation is impractical, as it requires centralized implementation and model-based computation. We then use a dual proximal formulation and a two-time-scale implementation to obtain decentralized dynamic filters that can be implemented at each inverter and use only local frequency and rate-of-change-of-frequency estimates. To assess the performance and the ability to regulate frequency, we derive bounds on the distance between state trajectories under the dynamic and ideal CBF filters. The bounds quantify the effects of the filter time constant, initialization, load variation, and frequency estimation errors, and provide estimates of possible frequency-limit violations. We validate the proposed method on the IEEE 39-bus system using a high-fidelity electromagnetic-transient (EMT) model. |
| 2026-11-25 | [Reference-Model-Guided Distributed MPC for UAV Swarm Formation Flying Under Switching Directed Graphs](http://arxiv.org/abs/2611.34540v1) | Yiming Wang, Yenan Wu et al. | Distributed formation control of unmanned aerial vehicle (UAV) swarms in complex environments faces several challenges, including switching inter-agent connectivity, limited communication, external disturbances, and collision avoidance. To address these challenges, we propose a reference-model-guided distributed model predictive control (MPC) framework for UAV swarm formation. Each UAV uses relative-position information from its neighbors as input to a local reference model, whose output serves as the formation guidance for the onboard MPC controller. Meanwhile, a Kalman filter is utilized to predict neighboring UAV motions, enabling the construction of inter-UAV collision-avoidance constraints solely from instantaneous relative-state information. By decoupling MPC optimization from distributed coordination, this control architecture preserves the look-ahead planning and constraint-handling capabilities of MPC while enhancing the swarm's adaptability to switching directed graphs. Moreover, eliminating the exchange of neighbors' predicted trajectories reduces communication bandwidth demands, offering strong scalability for large-scale UAV swarms. To improve robustness and safety, disturbance-observer-based feedforward compensation is employed, while the bounded residual disturbance is explicitly considered in the control barrier function (CBF)-based obstacle-avoidance constraints. Finally, the proposed framework is validated through both simulations and real-world flight experiments. Specifically, the flight verification is conducted on a fully distributed multi-UAV platform without external global localization. |
| 2026-11-25 | [Evidence-Aligned Benchmarking and Learning for Medical Video Language Models](http://arxiv.org/abs/2611.34465v1) | Zhongpai Gao, Anwesa Choudhuri et al. | A medical video language model should answer from the visual evidence its question asks about. We find three places where this breaks down and address each with an approach we call MedEvidU. First, benchmarks reward answers that ignore the video: on one dataset, the position of a clip within its recording predicts the surgical step in 37% of questions, and a constant answer beats every trained model on a standard safety-assessment metric. We build MedVidBench-v2, 34 tasks from 18 surgical and clinical video datasets, in which prompts and metrics are audited with answers given without video, and which tests unseen hospitals and windows cut blindly from whole recordings. Second, common video encoders merge consecutive frames into shared tokens, so a question about one frame is answered from a blend of two. Query-slot inputs give each queried frame its own tokens at almost no extra cost. Third, a fine-tuned model has memorized its training questions, which leaves little to learn from. Label-matched question resampling (LQR) adds unseen questions about the same recordings while keeping the answer distribution fixed, in a final supervised epoch and in reinforcement learning. Together, these raise the grounding IoU of a 4B model from 30.4 to 44.4, above a supervised 27B model, and instrument grounding at an unseen hospital from 33.9 to 72.5. On blindly cut windows, event-presence accuracy improves by 5.5 points and localization F1 by 8.1 points. Project page: https://gaozhongpai.github.io/MedEvidU/ |
| 2026-11-25 | [Evaluating Aviation Safety Knowledge Extraction with SmolLM2: From Zero-Shot to Retrieval-Grounded Prompting](http://arxiv.org/abs/2611.34432v1) | Aziida Nanyonga, Mustafa Tahsin Yilmaz et al. | Aviation accident narratives contain safety knowledge that is difficult to capture using structured occurrence fields alone. This study evaluates prompting-based knowledge extraction across four dimensions: concept accuracy, verbatim evidence grounding, structural compliance, and directed relation extraction. SmolLM2-1.7B-Instruct was evaluated under zero-shot, one-shot, fixed few-shot, and retrieval-grounded prompting using 300 National Transportation Safety Board narratives from 2000-2020, containing 3,763 reference concepts and 467 directed relations. Evaluation combined structural checks, verbatim evidence matching, expert-reviewed semantic matching, and paired bootstrap confidence intervals. No prompting strategy performed best across all dimensions. Retrieval-grounded prompting achieved the highest concept micro-F1 of 32.27%, although its advantage over fixed few-shot prompting was uncertain (95% CI: -0.92 to 2.40 percentage points). One-shot prompting achieved the highest relation F1 of 6.95%, identifying directed relation extraction as the principal SmolLM2 bottleneck. Compared with fixed few-shot prompting, retrieval improved verbatim grounding (59.45% vs. 54.02%) but reduced complete structural compliance (33.3% vs. 42.7%). Verbatim matching establishes source presence rather than evidential relevance. An exploratory comparison using 35 benchmark narratives showed that Qwen3-8B achieved higher concept micro-F1 across all prompting conditions and higher relation F1 under demonstration-based prompting, reaching 51.37% and 34.62%, respectively, under one-shot prompting. The small subset limits generalization. These findings support evaluating semantic correctness, source traceability, structural reliability, and directed relation accuracy separately. |
| 2026-11-25 | [Passive Fault-Tolerant Antenna Stabilisation for Ship-to-Shore Communication](http://arxiv.org/abs/2611.34411v1) | Antoine Thibault Vié, Mikkel Reinholdt Hansen et al. | Directional ship-to-shore radio links require continuous antenna alignment with the shore station, while vessel heave induces variations in antenna height above the sea surface, thereby altering the multipath interference governing the received power. In conventional stabilised gimbals, attitude compensation is provided, whereas fault tolerance is rarely considered. To address these limitations, a passive fault-tolerant control scheme is proposed, whereby the antenna is mounted on a ship-mounted six-degree-of-freedom manipulator, enabling simultaneous regulation alignment and antenna height. Through feedback linearisation, the manipulator dynamics are decoupled into perturbed integrators, each controlled by a multi-barrier adaptive super-twisting sliding-mode controller. The nested barriers are associated with the performance regions of fault-tolerant control, such that inter-sample barrier crossings under discrete-time implementation lead to graceful performance degradation rather than loss of confinement. Under simulated conditions, superior pointing and heave accuracy is achieved compared with single-barrier designs, for which either precision or confinement is compromised. |
| 2026-11-25 | [Recoverability policy editing for closed-loop safety in autonomous driving](http://arxiv.org/abs/2611.34403v1) | Yang Guan, Shengbo Eben Li | Autonomous driving powered by neural network models lacks safety guarantees. We propose a state-level policy editing technique that guarantees closed-loop safety under recoverable disturbances while reusing the model's existing driving capabilities. Closed-loop safety means that the vehicle remains free of collisions at all future times. We introduce the concept of policy recoverability to characterize whether a vehicle can safely return to the original policy's feasible set in finite time. Our method identifies states that are unsafe in the long term under the original policy but recoverable, and selectively edits policy outputs within this region. The vehicle corrects its actions upon entering this region and resumes the original policy upon recovery. Closed-loop simulations on nuPlan urban driving scenarios show a 48% reduction in safety incidents under the original policy while largely preserving its driving performance. |
| 2026-11-25 | [Spin-Orbit Coupling in Symmetry-Forbidden Reactions: A Selection Rule Tested on the Chalcogen Trimers](http://arxiv.org/abs/2611.34375v1) | Shadan Ghassemi Tabrizi | A reaction is forbidden by orbital symmetry when an occupied and an empty orbital of different symmetry change places along its path. With spin-orbit coupling orbitals become spinors, and those of two such orbitals can mix only if the orbitals differ by the symmetry of a rotation; the operator component of that symmetry couples ground state and diradical triplet. We test this rule on the ring opening of the chalcogen trimers O3 to Po3, where an in-plane and an out-of-plane orbital cross ($C_{2v}$ surfaces; NEVPT2, spin-orbit configuration interaction), by letting the operator components act separately. In Se3 and Te3 the component named by the rule alone lowers the barrier by 2.2 and 5.1 kcal/mol (complete operator: 2.2 and 5.2), the other two by 0.2 and 1.3 kcal/mol; in Po3 they contribute in second order. The triplet crosses the ground state 6.5 to 9.0 kcal/mol below the forbidden saddle. Spin-orbit coupling makes this crossing passable in O3 and S3 and, from Se3 on, merges the two states into one surface whose saddle lies 2.6, 5.5, and 13.1 kcal/mol lower. Molecular rotation obeys the same rule but is 6 to $2.8\times 10^{3}$ times weaker, which bears on the lifetime predicted for cyclic ozone. |
| 2026-11-25 | [LUDIUS: A Benchmark for Embodied Social Intelligence in Everyday Human-Robot Interactions](http://arxiv.org/abs/2611.34328v1) | Chaewon Kim, Sangheon Lee et al. | Social intelligence in a robot requires interpreting a situation from the perspectives of the people involved and using that understanding to decide when and how to act. We introduce LUDIUS, a benchmark for evaluating how vision-language models exercise this judgment through a robot's verbal responses and actions, organized around 15 social intelligence skills drawn from social psychology, sociology, and cognitive science. LUDIUS comprises six editions, each a collection of episodes spanning one simulated week. Three editions each cover all 15 skills in a different household. The other three focus on interactions involving multiple people, safety, and context-specific norms, respectively. We evaluate frontier models and a smaller model finetuned on social-interaction data we generated, using a common evaluation harness. The harness supplies each model with the robot's camera views and residents' reactions, retrieves relevant interaction history, and converts the model's tool calls into robot verbal responses and actions. Models tend to perform worse when they must reason about what a resident saw or heard earlier, reconcile competing interests, or fulfill a request within safety constraints and household rules. Beyond the home setting, LUDIUS's design pipeline offers a way to construct social intelligence benchmarks by translating social intelligence skills into context-specific judgments and observable robot behavior. |
| 2026-11-25 | [Energy-Aware Distributed Task Allocation via Local Submodular Maximization](http://arxiv.org/abs/2611.34293v1) | Jing Liu, Fangfei Li et al. | This paper studies distributed task allocation under time-varying agent sets, task sets, and communication graphs. At each decision epoch, the stage allocation problem is formulated as monotone submodular maximization over a time-varying, energy-filtered admissible ground set subject to a partition-matroid constraint. We develop a Distributed Adaptive Greedy Bidding Algorithm (DAGBA) that combines local admissible-set construction, conditional marginal bidding, and neighbor-based conflict resolution under limited communication. We prove that DAGBA guarantees per-epoch feasibility, finite-round convergence, and residual-energy safety. Theoretically, we identify a sequential-greedy regime where DAGBA recovers the classical $1/2$-approximation guarantee. For general executions with partial information, we derive a local-information-factor approximation bound and conservative bounds computable from the execution trace that quantify the degradation caused by local estimates and parallel fixing. Simulations on cooperative observation missions demonstrate that DAGBA achieves task-completion progress and utility close to centralized and complete-communication references, while improving communication efficiency and maintaining strict energy feasibility. |

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



