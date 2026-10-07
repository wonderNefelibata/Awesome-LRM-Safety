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
| 2026-10-06 | [LBA-CBF: Rapidly Adaptive Safety Filters via Parallel Dynamics Inference](http://arxiv.org/abs/2610.08765v1) | Maitham F. AL-Sunni, Timeea-Andreea Radu et al. | Control barrier functions (CBFs) certify commands through an assumed dynamics model, so an abrupt, unmeasured regime change can undermine the certificate exactly when safety matters most. We present Look-Back Adaptive Control Barrier Functions (LBA-CBF), which rank a finite bank of candidate dynamics by recent prediction error over a short look-back window and enforce the high-order CBF condition against every model within a tolerance of the best, spanning best-fit adaptation to full-bank robust filtering. The dynamics may depend nonlinearly on the unknown parameters, and no switching model or continuously parameterized estimator is required. We prove that any feasible filtered input satisfies the true CBF condition whenever a safety-representative candidate is retained. In quadrotor simulation with abrupt wind reversals and an unknown payload, LBA-CBF is safe and reaches the goal from all random initial conditions, matching an oracle, while adaptive and robust baselines achieve 0-88% success. Banks of up to 250,000 models run inside the control loop, and Crazyflie 2.1 and F1TENTH experiments demonstrate adaptation to wind, payload release, and varying tire-road friction. Code, videos, and project details are available at: https://lla-control.github.io |
| 2026-10-06 | [VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning](http://arxiv.org/abs/2610.08761v1) | Zewei Zhou, Rachel Luo et al. | Self-improving policies continually expose new failure patterns, changing what their judges must be able to verify. However, current fixed judges constrain both optimization feedback and the discovery of useful training examples, limiting further self-improvement. This challenge is even more acute in embodied reasoning, where reliable evaluation must account for spatial grounding, causal reasoning, and safety-aware decision-making. We introduce VeriFine, an agent harness framework that scales verification through the co-evolution of the policy, training curriculum, and judge. The Policy Improvement Loop uses a rubric judge to diagnose recurring failures, construct an adaptive curriculum, and optimize the policy. When progress plateaus and verification becomes a bottleneck, the Judge Improvement Loop selectively queries human guidance on informative failure cases and refines the judge through coactive calibration, in which humans and agents resolve disagreements and converge toward the objective rubric of physical reasoning. The revised judge then guides the next stage of data selection and policy optimization. Experiments on driving and robot navigation tasks demonstrate continuous self-improvement in both policy and judge capability across reinforcement and supervised fine-tuning. These results show how scaling verification supports continuous self-improvement as policy failure patterns evolve. |
| 2026-10-06 | [BARE-AI: Bit-Flip Attack Resilience in AI Hardware through Built-in Performance Monitors](http://arxiv.org/abs/2610.08739v1) | Habibur Rahaman, Swastik Bhattacharya et al. | Deep Neural Networks (DNNs) are integral to many safety critical systems, yet they remain highly vulnerable to bit-flip attacks (BFAs), where a few memory level perturbations can drastically degrade accuracy. Existing defenses incur significant hardware overhead, depend on retraining, or fail against targeted flips. We propose BARE-AI, a runtime framework that detects, localizes, and mitigates BFAs during inference. BARE-AI introduces AI Performance Counters (APCs), lightweight hardware monitors in the accelerator datapath that capture per-layer activation statistics such as sparsity, entropy, kurtosis, and spectral shift. These are analyzed by the Predictive Unit for Layer Security Evaluation (PULSE), a compact detector trained offline as an ensemble of classifiers and realized on-chip as a small neural engine. For explainability and recovery, BARE-AI introduces an Activation Shift Index (ASI) for layer level fault localization and a z-score based repair that resets anomalous weights toward clean layer statistics. Across CNNs, Vision Transformers, and Large Language Models under random, targeted, adaptive, and magnitude based BFAs, BARE-AI achieves up to 98% detection accuracy on vision models and 74% to 95% on language models, restores near clean accuracy for CNNs and ViTs, and provides partial recovery for LLMs. Synthesized at 28nm, the monitoring infrastructure incurs under 3% energy, under 4% area, and about 10% latency overhead, with a configurable operating point that reduces latency overhead to about 6%. Unlike error correcting codes, whose redundancy grows with the number of tolerated flips, BARE-AI's overhead remains constant regardless of attack strength, making it attractive for resource constrained, safety critical edge applications such as autonomous systems, energy, and healthcare. |
| 2026-10-06 | [Secure Speculative Decoding for Large Language Models](http://arxiv.org/abs/2610.08678v1) | Yichi Zhang, Zhiqi Wang et al. | Speculative decoding accelerates inference for a large language model (LLM), referred to as the \emph{target model}, by first using a smaller model, referred to as the \emph{draft model}, to generate candidate tokens and then verifying them with the target model for acceptance or rejection. Prior studies primarily focused on the efficiency-utility trade-off of speculative decoding, e.g., lossy speculative decoding, leaving its security implications largely unexplored.   In this work, we bridge this gap by providing the \emph{first} systematic study of the security implications of speculative decoding. Through a large-scale measurement study, we reveal a pronounced security-utility asymmetry: across a wide range of lossy speculative decoding methods, improvements in inference efficiency come at a disproportionately high cost to security, with attack success rates for jailbreak and prompt injection attacks increasing much faster than utility degrades.   We then propose SecureSD, a new theory-guided speculative decoding method that enhances security while maintaining efficiency and utility. Specifically, our theoretical analysis reveals that security degradation primarily originates from the early tokens generated by the draft model. Motivated by this insight, SecureSD applies a stricter verification criterion to draft-model tokens at early decoding positions. Extensive experiments on both security and utility benchmarks demonstrate that SecureSD significantly improves security while preserving efficiency and utility compared to existing speculative decoding methods. |
| 2026-10-06 | [Towards In-Parameter Memory Augmentation for Large Language Models](http://arxiv.org/abs/2610.08630v1) | Haoyu Huang, Zhongwei Xie et al. | Recently Large Language Models (LLMs) and LLM-based agents increasingly need to incorporate knowledge acquired after pretraining, e.g., domain facts, user preferences, documents, and interaction experience. In-context learning (ICL) and ICL-based agent harness remain flexible, but they consume context capacity and incur repeated discretized encoding cost that grows with context length. \textbf{In-parameter memory} offers a complementary substrate: reusable memory information is represented in model parameters, adapters, or other parameter-like objects that are composed into the forward pass at inference time. This survey focuses on methods that augment LLMs with such parametric memory at deployment: a memory-bearing parameter object is plugged into the forward pass during inference, whether it is acquired before or during deployment. We organize the landscape with two orthogonal axes: \textbf{Parameter Placement}, which includes Embedding, Attention, FFN layers, or Hybrid when two or more layers are used; and \textbf{Parameter Acquisition Time}, which distinguishes methods whose memory object is acquired during deployment (online) from those acquired before it (offline). We clarify boundaries, conduct comparisons, and discuss open directions in interference, safety, co-design with ICL, and recursive self-improvement. |
| 2026-10-06 | [Random Feature Gaussian Process Attention: Linear-Time Probabilistic Attention with Calibrated Uncertainty](http://arxiv.org/abs/2610.08578v1) | Amir Mohammad Mahfoozi, Zi Yang et al. | Transformers provide a state-of-the-art modeling framework, yet poor calibration limits their reliability in safety-critical applications. A promising direction addresses this issue by interpreting attention as a Gaussian process (GP) posterior, which enables principled uncertainty calibration but incurs cubic complexity in sequence length due to the inversion of the kernel; although decoupled GP variants reduced the cost to quadratic, the computation remains prohibitive in practice. In this paper, we propose the plug-and-play random Fourier feature Gaussian process attention (RFF-GPA) module, which represents the attention as a GP with a stationary kernel approximated by random Fourier features. This low-rank approximation results in linear-time complexity for approximating the posterior mean and variance, making it far more scalable compared to previous work. Empirical results on multiple real-world datasets show that our attention module improves calibration while maintaining predictive accuracy, and simultaneously reduces computational complexity to linear in the sequence length. |
| 2026-10-06 | [Systemization of Knowledge (SoK): Human-Centered AI Safety for Youth](http://arxiv.org/abs/2610.08554v1) | Pratyasha Saha, Yaman Yu et al. | While HCI increasingly examines AI-safety for youth, the literature lacks a comprehensive view of what risks have been identified, how they are addressed, and whether proposed protections work in-practice. We systematically reviewed 100 empirical HCI studies involving children and youth interacting with or exposed to AI across schools, homes, care settings, and public services. Using the YAIR taxonomy for risks and the MIT Mitigation Taxonomy for countermeasures, we map which risks have been identified, whether each risk is addressed by countermeasure(s), and whether each countermeasure for that risk is implemented and even evaluated. The risk-countermeasure mapping shows that most risks are matched only with proposed/ideated countermeasures; few countermeasures have been implemented, and fewer still evaluated; and existing evaluations often measure technical performance rather than protection from harm. We identify where coverage is absent, where safeguards remain untested, and propose concrete directions for HCI research to strengthen youth AI-safety. |
| 2026-10-06 | [Micro Neural Policies for Safe Real-Time Robotic Control](http://arxiv.org/abs/2610.08541v1) | Hongpeng Cao, Riccardo Curcio et al. | In this paper, we investigate the synthesis of Micro Neural Policies (MNP) to enable safe and robust real-time robotic control on computationally constrained embedded devices. We demonstrate that integrating Evolution Strategy (ES) and Statistical Model Checking (SMC)-based verification for policy search can drastically reduce neural network size without compromising safety and robustness. We conduct a large-scale training and evaluation of MNP on Cartpole and Quadrotor control tasks, varying control frequencies and network architectures. After validating these policies in simulation, we evaluate their deployability through zero-shot transfer to physical systems. Our experiments show that MNP can successfully achieve safe sim-to-real transfer without sacrificing control performance. We then show that the policies' memory footprint, ranging from 0.5 to 7.5 kB, allows deployment on microcontrollers, where they achieve real-time inference latency with under 25 ns of jitter while leaving the chip idle for over 97% of the time for additional workloads. This makes them a highly practical solution for severely resource-constrained robotic systems. |
| 2026-10-06 | [Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements](http://arxiv.org/abs/2610.08540v1) | Jeremy Canale | Whether alignment gets easier or harder as models grow is often argued from isolated findings, as if alignment were one property. We treat it as a family of measurable scaling relations: for each risk category r, the alignment burden needed to hold a fixed safety target is modeled as B_r(N)=a_rN^alpha_r, with N a capability proxy; against a budget proportional to N, scaling helps if alpha_r<1, keeps pace if alpha_r~1, and accumulates alignment debt if alpha_r>1. We give three operationalizations of burden and distinguish observed, audited and true alignment. A toy model, in which corrections consume capability headroom, makes the consequences explicit. We prove that the largest exponent among corrected risks, not an average, sets the long-run regime; that above 1 any policy holding headroom above a floor must grow super-exponentially; that, for burdens that are positive mixtures of power laws, fits on small models underestimate large-scale exponents; and that an audit that uncovers hidden failures without false positives never underestimates true alignment. We propose a pre-registrable protocol and apply reduced versions of it twice. A preregistered reanalysis of public adversarial-training data for Pythia classifiers finds that the compute needed to bring attack success under 10% grows as N^0.60. A preregistered pilot on Qwen2.5 0.5B-72B finds exponents of -0.05 for truthfulness and 0.48 for stated dispositions (both scaling helps under its reduced rule, though local slopes approach 1 at the top; replicated on Qwen3 0.6B-14B), while sycophancy (0.89, or 0.83 with two seeds added at 72B) and a planted backdoor are undetermined: the backdoor is removed quickly when its trigger is known but survives blind safety training at four of five sizes. We release four browser games that play these laws (www.aisafety.fun). We make no claim about which regime holds for current frontier models. |
| 2026-10-06 | [Behavioral Safety Assessment towards Large-scale Deployment of Autonomous Vehicles, Part II: Assessment Results](http://arxiv.org/abs/2610.08458v1) | Henry X. Liu, Tinghan Wang et al. | Third-party evaluations of autonomous vehicle (AV) safety can play a vital role in improving public acceptance, building consumer confidence, and establishing effective safety standards. In Part I of this study, we propose a dedicated third-party testing initiative for systematically evaluating AV behavioral safety. In this paper, we validate our proposed framework using Autoware.Universe, an open-source Level 4 Automated Driving System (ADS), tested both in simulated environments and on the physical test track at the University of Michigan's Mcity Testing Facility. The results indicate that Autoware.Universe possesses 6 out of 14 behavioral competencies and exhibited a crash rate of 3.01x10^-3 crashes per mile, approximately 1,000 times higher than the average human driver crash rate. During the tests, we also uncovered a number of unknown unsafe scenarios for Autoware.Universe. These findings underscore the necessity of behavioral safety evaluations for improving AV safety performance prior to widespread public deployment. |
| 2026-10-06 | [Behavioral Safety Assessment towards Large-scale Deployment of Autonomous Vehicles, Part I: Methodology](http://arxiv.org/abs/2610.08450v1) | Henry X. Liu, Tinghan Wang et al. | Autonomous vehicles (AVs) have significantly advanced in real-world deployment in recent years, yet safety continues to be a critical barrier to widespread adoption. Traditional functional safety approaches, which primarily verify the reliability, robustness, and adequacy of AV hardware and software systems from a vehicle-centric perspective, do not sufficiently address the AV's broader interactions and behavioral impact on the surrounding traffic environment. To overcome this limitation, we propose a paradigm shift toward behavioral safety, a comprehensive approach focused on evaluating AV responses and interactions within the traffic environment. To systematically assess behavioral safety, we introduce a third-party AV safety assessment framework comprising two complementary evaluation components: the Behavioral Competency Test and the Driving Intelligence Test. The Behavioral Competency Test evaluates the AV's reactive behaviors under controlled scenarios, ensuring basic behavioral competency. In contrast, the Driving Intelligence Test assesses the AV's interactive behaviors within naturalistic traffic conditions, quantifying the frequency of safety-critical events to deliver statistically meaningful safety metrics before large-scale deployment. In Part II of this study, an open-source Level 4 Automated Driving System (ADS) is tested to demonstrate the effectiveness of the proposed method. |
| 2026-10-06 | [Explainable Failure Prediction and Prevention in Maritime](http://arxiv.org/abs/2610.08363v1) | Dionisis Kalogeropoulos, Georgia Sovatzidi et al. | Maritime systems operate in highly dynamic environments where unexpected equipment failures can compromise safety, reliability, and operational efficiency. Recent advances in artificial intelligence (AI), machine learning, digital twins, and predictive maintenance enable proactive failure prediction and prevention. However, ensuring trustworthy and explainable decision-making remains a major challenge in safety-critical maritime applications. This chapter reviews key AI technologies required for explainable failure prediction and prevention in maritime systems and presents a conceptual architecture capable of supporting autonomous or human-in-the-loop corrective actions. This architecture integrates data acquisition, time-series forecasting, anomaly detection, risk assessment, decision-making, and explainable AI into a closed-loop framework. With reference to the architectural components, a review and discussion of relevant maritime studies is performed, outlining their methods, advantages, and limitations. Furthermore, it highlights current challenges, including uncertainty and robustness, model generalization, explainability, limited availability of maritime datasets, and operational deployment, and identifies future research directions toward trustworthy AI-assisted maritime decision-making. |
| 2026-10-06 | [SC3BF: Shifted Collision Cone Control Barrier Function for Dynamic Obstacle Avoidance](http://arxiv.org/abs/2610.08332v1) | Amin Kashiri, Yasin Yazıcıoğlu | The collision cone used by velocity-space control barrier functions is conservative: it rejects every relative velocity aimed into an obstacle, however slow. We propose the \emph{shifted collision-cone CBF} (SC3BF), which adds a state-dependent \emph{allowance} to the cone condition, so the robot may approach the obstacle at a rate that grows with distance and with its own speed. SC3BF is enforced by an ordinary quadratic program, and its safe set is forward invariant under bounded inputs without a minimum forward speed or a clearance margin. We prove that a nonzero allowance preserving safety always exists, and derive one in closed form. Against three velocity-space baselines on a kinematic bicycle among up to $100$ moving obstacles, SC3BF reaches the goal more often and modifies the nominal input less than half as much. |
| 2026-10-06 | [Transferable Spatial Temporal Coherence Adversarial Attack on Black-Box Vision Language Models for Autonomous Driving](http://arxiv.org/abs/2610.08331v1) | Heyam Bin Jahlan Areej Alhothali Abeer Alhothali | The rapid integration of Vision Language Models (VLMs) into sensitive systems introduces critical safety vulnerabilities that remain unexplored in exist studies. While adversarial attack robustness has been extensively studied for image-based models, the susceptibility of VLMs to temporally-aware adversarial attacks against video in driving context poses a distinct and under examined threat. In this paper, we introduce novel adversarial attack against video targeting VLM models used for autonomous driving scenes named Spatial Temporal Coherence Adversarial Attack (STCA). Our attack comprise from three stages: modalities expansion, Spatial attack, and STCA attack. In modalities expansion, we propose caption-guided frame selection method in order to ensure that adversarial perturbation target the most semantically significant frames. Secondly.In spatial attack, we craft effective perturbation and preserve high similarity. Then the perturbed video generated fed into STCA stage that disrupt cross-frame temporal coherence using motion guided mask. Our method operate under black box threat model against victim target VLMs, relying solely on transferability from white-box surrogate model.We conduct our experiments on the BDD100K and nuScenes autonomous driving datasets across three VLM models: Video LLaVA-7B, Qwen2.5-VL-7B, and Dolphin. Experimental results demonstrate spatial attack achieves an ASR with high SSIM. Our finding reveal that existing video language model, remain highly susceptible to adversarial attack in autonomous driving scenarios, underscoring the urgent need for robust defense for VLM models. |
| 2026-10-06 | [Building A Civic Tool for Community-Police Engagement to Adapt Neighborhood Policing](http://arxiv.org/abs/2610.08212v1) | Ravinithesh Reddy Annapureddy, Staņislavs Šeiko et al. | Data-driven policing often prioritizes incident records over residents' lived experiences. In the Baltic city of Riga, with a history of distrust and limited community-police engagement, this can further alienate the public. To bridge this gap, we propose a Research through Design (RtD) inquiry into the development of Par drošu Rīgu, a civic tool for community-data-integrated policing. With municipal police, NGOs, and city staff, we ask how RtD enables stakeholder negotiation and which interaction qualities support trust and the use of combined community and incident data. The co-design process included workshops that surfaced divergent notions of safety; material probes designed as boundary objects to negotiate among stakeholders; and a pilot deployment showing how combining quantitative and qualitative data reshapes engagement and trust. Mixed-methods evaluation suggests increased officer-citizen interaction, but frictions in sustaining stakeholder collaboration. We contribute (i) an empirical RtD inquiry with public institutions, (ii) an artifact combining physical and dashboard interactions, and (iii) reflections on interaction design as a boundary-spanning practice for trust and infrastructuring. |
| 2026-10-06 | [Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving](http://arxiv.org/abs/2610.08123v1) | Ahmed Abouelazm, Rupert Polley et al. | End-to-end planners based on waypoint regression achieve strong open-loop accuracy, but they primarily learn to mimic expert geometry and remain difficult to adapt to deployment-time safety constraints. We propose a query-based cost-learning framework that estimates bounded costs for dynamically reachable ego trajectory queries, rather than dense BEV cells or a small regressed trajectory set. Compact joint scene tokens capture coherent multimodal agent futures, while contingency-aware cost aggregation and cost-guided intra-cluster MPPI mixing convert the learned cost topology into feasible ego plans. On nuScenes, our method improves over prior cost-estimation planners such as ST-P3 and NMP, outperforms most regression baselines in collision rate, while remaining competitive in L2, and retaining an interpretable cost interface. On real-world driving logs, the proposed planner reduces collision rates compared with SparseDrive and Alpamayo without fine-tuning, while maintaining a diverse set of candidate trajectories. |
| 2026-10-06 | [Context-Conditioned Hamilton-Jacobi Reachability for Adaptive Safety Filtering](http://arxiv.org/abs/2610.08115v1) | Ali Fuat Sahin, Yunus Yazoglu et al. | Hamilton-Jacobi reachability constructs safety certificates for specified dynamics and safety constraints, tying each certificate to the deployment context for which it is synthesized. We ask whether a single certificate can instead represent a family of context-dependent safety problems and be queried across deployment conditions without re-synthesis. We learn a backward reachable tube for an eight-state vehicle model conditioned on local boundary geometry, friction coefficient, and adversarial disturbance scale. Geometry enters through an ego-frame boundary observation that defines the local containment constraint, while friction and disturbance scale enter as explicit operating-condition variables. This allows the same value function to be queried across friction coefficients from 0.4 to 2.0 and on geometries absent from synthesis. On 11 held-out evaluation geometries, the certificate maintains containment across the full tested friction range, including simultaneous geometry and grip shifts, while remaining within 1.2 percentage points in intervention rate and 0.09 m/s in speed of certificates re-synthesized with knowledge of the test geometry. We then deploy the certificate as a sampled discrete-time control barrier function filter on a full-scale vehicle near the handling limit. Lateral containment holds in every hardware session under both adversarial driving and autonomous racing, with 99th-percentile acceleration magnitude reaching 0.99 g. Across three certificates evaluated under a fixed autonomous racing controller, lap time varies by only 3.1%, demonstrating that a context-conditioned reachability certificate can transfer to deployment geometries absent from synthesis with modest performance cost. |
| 2026-10-06 | [Natural Language Questions as an Interface for Knowledge Graphs: QRAKEN Graph Distillation and Semantic Self-Healing](http://arxiv.org/abs/2610.08095v1) | Remo Grillo, Lukas Klic et al. | Natural-language access to RDF knowledge graphs is a core Semantic Web ambition. Large language models (LLMs) have advanced Text-to-SPARQL, yet on unfamiliar graphs they often generate valid queries that misrepresent the populated data model. QRAKEN is a training-free, ontology-agnostic neurosymbolic pipeline grounding generation in empirical graph evidence rather than schema expectations. An offline distiller produces TTQL, a compact description of populated multi-hop patterns, conditional frequencies and path-conditioned literal examples, plus a class-property co-occurrence matrix. Online, TTQL guides the LLM, while deterministic syntax, vocabulary and data-model checks provide diagnostics for iterative refinement. On CK25 (First International Text2SPARQL Challenge), under matched-condition recomputation on a QLever snapshot, QRAKEN achieves strict F1 of 0.643 $\pm$ 0.026 with GPT-4.1 mini and 0.652 $\pm$ 0.012 with GPT-5.4: relative gains of 30% and 32% over the strongest recomputed participant, outperforming systems using the same base model family. Ablations identify TTQL patterns as the dominant driver (+0.31 strict F1 over a shape-only baseline); the refinement loop provides a cheap safety net, rejecting triple patterns unsupported by the co-occurrence matrix. Compared with auto-derived SHACL, TTQL yields 64% higher strict F1, supporting the value of empirical patterns beyond schema exposure. With two local 35B 4-bit open-weight models at zero marginal cost, the same pipeline matches the strongest recomputed participant, and TTQL advantages over shape-only and SHACL baselines persist. Results on a single, relatively small benchmark provide an initial empirical signal; monolithic TTQL injection on very open cross-domain graphs remains the main limitation. |
| 2026-10-06 | [POLAR: Ontology-Guided Risk Prevention for Tool-Calling LLM Agents](http://arxiv.org/abs/2610.08082v1) | Yunju Kang, Seonghyeon Cho et al. | LLM tool-use agents operate in dynamic environments where many actions carry operational risk. However, most safety mechanisms react only after errors manifest. Existing pre-emptive approaches either fine-tune the agent on chain-of-thought deliberation or compile natural-language guardrails into runtime checks, but they do so without exposing a structural, auditable verdict. We propose POLAR, a guardrail framework for small tool-calling agents that assesses reversibility through a structured two-layer ontology. POLAR assigns each action a graded reversibility score by deriving a candidate inverse sequence; calls failing a threshold are pruned before execution. Evaluated on $τ^2$-bench across six agent models, POLAR improves mean task reward by 0.11 to 0.18 points on airline for four of six agents, but only eight of eighteen model--domain cells improve overall; retail and stronger agents often regress. POLAR provides an auditable structural check and characterizes its task-utility trade-offs. Reward is not a direct measure of prevented harm. |
| 2026-10-06 | [ASCENT: First-Order Optimal Fine-Tuning with Recalibration for Safety--Utility Co-Enhancement](http://arxiv.org/abs/2610.08061v1) | Weiwei Qi, Chongyu Wang et al. | Supervised fine-tuning can substantially improve the downstream utility of large language models (LLMs) but may compromise their safety. Existing safety-preserving methods constrain downstream updates using safety-related parameters or subspaces, but mainly focus on safety preservation rather than joint safety and utility enhancement, lack a theoretical characterization of the optimal safety-related subspace and safety-preserving task update, and typically rely on a static safety subspace that may become outdated during fine-tuning. To address these limitations, we propose ASCENT, a downstream fine-tuning framework for safety--utility co-enhancement through first-order optimal safety-aware periodic calibration and task optimization. We model safety as a function of LLM parameters $S(θ)$ and use its first-order approximation to characterize safety changes under parameter updates. Under a fixed rank and Frobenius-norm budget, we prove that the update constructed from the top-$r$ singular components of the safety-function gradient maximizes the estimated safety change, and use it for periodic calibration to preserve and improve safety. We further derive a unique safety-preserving task update that stays close to the original task update while penalizing negative effects on the estimated safety change. ASCENT alternates these optimal task and calibration updates to jointly enhance safety and utility. Experiments across multiple LLM families and downstream tasks show that ASCENT improves downstream utility by up to 20.3\% and reduces attack success rate by up to 35.5\%, achieving state-of-the-art safety and utility across all evaluated settings. Our code is available at https://github.com/ZJU-LLM-Safety/ASCENT. |

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



