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
| 2026-04-23 | [Seeing Without Eyes: 4D Human-Scene Understanding from Wearable IMUs](http://arxiv.org/abs/2604.21926v1) | Hao-Yu Hsu, Tianhang Cheng et al. | Understanding human activities and their surrounding environments typically relies on visual perception, yet cameras pose persistent challenges in privacy, safety, energy efficiency, and scalability. We explore an alternative: 4D perception without vision. Its goal is to reconstruct human motion and 3D scene layouts purely from everyday wearable sensors. For this we introduce IMU-to-4D, a framework that repurposes large language models for non-visual spatiotemporal understanding of human-scene dynamics. IMU-to-4D uses data from a few inertial sensors from earbuds, watches, or smartphones and predicts detailed 4D human motion together with coarse scene structure. Experiments across diverse human-scene datasets show that IMU-to-4D yields more coherent and temporally stable results than SoTA cascaded pipelines, suggesting wearable motion sensors alone can support rich 4D understanding. |
| 2026-04-23 | [Transient Turn Injection: Exposing Stateless Multi-Turn Vulnerabilities in Large Language Models](http://arxiv.org/abs/2604.21860v1) | Naheed Rayhan, Sohely Jahan | Large language models (LLMs) are increasingly integrated into sensitive workflows, raising the stakes for adversarial robustness and safety. This paper introduces Transient Turn Injection(TTI), a new multi-turn attack technique that systematically exploits stateless moderation by distributing adversarial intent across isolated interactions. TTI leverages automated attacker agents powered by large language models to iteratively test and evade policy enforcement in both commercial and open-source LLMs, marking a departure from conventional jailbreak approaches that typically depend on maintaining persistent conversational context. Our extensive evaluation across state-of-the-art models-including those from OpenAI, Anthropic, Google Gemini, Meta, and prominent open-source alternatives-uncovers significant variations in resilience to TTI attacks, with only select architectures exhibiting substantial inherent robustness. Our automated blackbox evaluation framework also uncovers previously unknown model specific vulnerabilities and attack surface patterns, especially within medical and high stakes domains. We further compare TTI against established adversarial prompting methods and detail practical mitigation strategies, such as session level context aggregation and deep alignment approaches. Our study underscores the urgent need for holistic, context aware defenses and continuous adversarial testing to future proof LLM deployments against evolving multi-turn threats. |
| 2026-04-23 | [Mitigating Systematic Errors in Parameter Estimation of Binary Black Hole Mergers in O1-O3 LIGO-Virgo Data](http://arxiv.org/abs/2604.21859v1) | Sumit Kumar, Max Melching et al. | Systematic errors in the parameter estimation (PE) of gravitational wave (GW) mergers can arise from various sources, including waveform systematics, noise mischaracterization, data analysis artifacts, and other unknown factors. In this study, we analyze selected events from the first three observing runs of the LIGO-Virgo-KAGRA (LVK) collaboration. We choose events that have been flagged in various studies as potentially affected by systematic errors. Here, we reanalyze these events using a couple of parametric models developed in previous work that incorporate uncertainties in both the phase and amplitude of the GW waveform. In this data-driven approach, we apply sufficiently broad priors on the uncertainty parameters to account for potential systematic errors. Our findings show that the proposed method effectively reduces systematic errors, even those arising from data artifacts, such as glitches occurring near a signal and the deglitching process in GW frame files. Similarly, inconsistent results from different waveform models become much more consistent in our framework. One noteworthy event we examine is GW191109\_010717, which is particularly interesting due to its anti-aligned spin properties. We report that, within our framework, the event still exhibits anti-aligned spin characteristics, but the inference results become consistent across raw and deglitched frame files, as well as across the waveform models used for this event (IMRPhenomXPHM, IMRPhenomXO4a, and NRSur7dq4). A similar trend is observed for the event GW200129\_065458, which previously yielded a high, but inconsistent precession parameter among different waveform models. In contrast, we observe a non-zero and consistent value of $χ_{p}=0.60^{+0.31}_{-0.33}, 0.58^{+0.30}_{-0.29}$ and $0.56^{+0.31}_{-0.28}$ for the IMRPhenomXPHM, IMRPhenomXO4a, and NRSur7dq4 waveform models, respectively. |
| 2026-04-23 | [Bounding the Black Box: A Statistical Certification Framework for AI Risk Regulation](http://arxiv.org/abs/2604.21854v1) | Natan Levy, Gadi Perl | Artificial intelligence now decides who receives a loan, who is flagged for criminal investigation, and whether an autonomous vehicle brakes in time. Governments have responded: the EU AI Act, the NIST Risk Management Framework, and the Council of Europe Convention all demand that high-risk systems demonstrate safety before deployment. Yet beneath this regulatory consensus lies a critical vacuum: none specifies what ``acceptable risk'' means in quantitative terms, and none provides a technical method for verifying that a deployed system actually meets such a threshold. The regulatory architecture is in place; the verification instrument is not.   This gap is not theoretical. As the EU AI Act moves into full enforcement, developers face mandatory conformity assessments without established methodologies for producing quantitative safety evidence - and the systems most in need of oversight are opaque statistical inference engines that resist white-box scrutiny.   This paper provides the missing instrument. Drawing on the aviation certification paradigm, we propose a two-stage framework that transforms AI risk regulation into engineering practice. In Stage One, a competent authority formally fixes an acceptable failure probability $δ$ and an operational input domain $\varepsilon$ - a normative act with direct civil liability implications. In Stage Two, the RoMA and gRoMA statistical verification tools compute a definitive, auditable upper bound on the system's true failure rate, requiring no access to model internals and scaling to arbitrary architectures. We demonstrate how this certificate satisfies existing regulatory obligations, shifts accountability upstream to developers, and integrates with the legal frameworks that exist today. |
| 2026-04-23 | [TraceScope: Interactive URL Triage via Decoupled Checklist Adjudication](http://arxiv.org/abs/2604.21840v1) | Haolin Zhang, William Reber et al. | Modern phishing campaigns increasingly evade snapshot-based URL classifiers using interaction gates (e.g., checkbox/slider challenges), delayed content rendering, and logo-less credential harvesters. This shifts URL triage from static classification toward an interactive forensics task: an analyst must actively navigate the page while isolating themselves from potential runtime exploits.   We present TraceScope, a decoupled triage pipeline that operationalizes this workflow at scale. To prevent the observer effect and ensure safety, a sandboxed operator agent drives a real GUI browser guided by visual motivation to elicit page behavior, freezing the session into an immutable evidence bundle. Separately, an adjudicator agent circumvents LLM context limitations by querying evidence on demand to verify a MITRE ATT&CK checklist, and generates an audit-ready report with extracted indicators of compromise (IOCs) and a final verdict.   Evaluated on 708 reachable URLs from existing dataset (241 verified phishing from PhishTank and 467 benign from Tranco-derived crawling), TraceScope achieves 0.94 precision and 0.78 recall, substantially improving recall over three prior visual/reference-based classifiers while producing reproducible, analyst-grade evidence suitable for review. More importantly, we manually curated a dataset of real-world phishing emails to evaluate our system in a practical setting. Our evaluation reveals that TraceScope demonstrates superior performance in a real-world scenario as well, successfully detecting sophisticated phishing attempts that current state-of-the-art defenses fail to identify. |
| 2026-04-23 | [Adversarial Robustness of Near-Field Millimeter-Wave Imaging under Waveform-Domain Attacks](http://arxiv.org/abs/2604.21774v1) | Lhamo Dorje, Jordan Madden et al. | Near-field millimeter-wave (mmWave) imaging is widely deployed in safety-critical applications such as airport passenger screening, yet its own security remains largely unexplored. This paper presents a systematic study of the adversarial robustness of mmWave imaging algorithms under waveform-domain physical attacks that directly manipulate the image reconstruction process. We propose a practical white-box adversarial model and develop a differential imaging attack framework that leverages the differentiable imaging pipeline to optimize attack waveforms. We also construct a real measured dataset of clean and attack waveforms using a mmWave imaging testbed. Experiments on 10 representative imaging algorithms show that mmWave imaging is highly vulnerable to such attacks, enabling an adversary to conceal or alter targets with moderate transmission power. Surprisingly, deep-learning-based imaging algorithms demonstrate higher robustness than classical algorithms. These findings expose critical security risks and motivate the development of robust and secure mmWave imaging systems. |
| 2026-04-23 | [Enabling and Inhibitory Pathways of University Students' Willingness to Disclose AI Use: A Cognition-Affect-Conation Perspective](http://arxiv.org/abs/2604.21733v1) | Yiran Du, Huimin He | The increasing integration of artificial intelligence (AI) in higher education has raised important questions regarding students' transparency in reporting AI-assisted work. This study investigates the psychological mechanisms underlying university students' willingness to disclose AI use by applying the Cognition--Affect--Conation (CAC) framework. A sequential explanatory mixed-methods design was employed. In the quantitative phase, survey data were collected from 546 university students and analysed using structural equation modelling to examine the relationships among cognitive perceptions, affective responses, and disclosure intention. In the qualitative phase, semi-structured interviews with 22 students were conducted to further interpret the quantitative findings. The results indicate that psychological safety significantly increases students' willingness to disclose AI use and is positively shaped by perceived fairness, perceived teacher support, and perceived organisational support. Conversely, evaluation apprehension reduces disclosure intention and psychological safety, and is strengthened by perceived stigma, perceived uncertainty, and privacy concern. Qualitative findings further reveal that institutional clarity and supportive instructional practices encourage openness, whereas policy ambiguity and fear of negative evaluation often lead students to adopt cautious or strategic disclosure practices. Overall, the study highlights the dual role of enabling and inhibitory psychological mechanisms in shaping AI-use disclosure and underscores the importance of supportive institutional environments and clear guidance for promoting responsible AI transparency in higher education. |
| 2026-04-23 | [Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers](http://arxiv.org/abs/2604.21700v1) | Jiali Wei, Ming Fan et al. | The growing application of large language models (LLMs) in safety-critical domains has raised urgent concerns about their security. Many recent studies have demonstrated the feasibility of backdoor attacks against LLMs. However, existing methods suffer from three key shortcomings: explicit trigger patterns that compromise naturalness, unreliable injection of attacker-specified payloads in long-form generation, and incompletely specified threat models that obscure how backdoors are delivered and activated in practice. To address these gaps, we present BadStyle, a complete backdoor attack framework and pipeline. BadStyle leverages an LLM as a poisoned sample generator to construct natural and stealthy poisoned samples that carry imperceptible style-level triggers while preserving semantics and fluency. To stabilize payload injection during fine-tuning, we design an auxiliary target loss that reinforces the attacker-specified target content in responses to poisoned inputs and penalizes its emergence in benign responses. We further ground the attack in a realistic threat model and systematically evaluate BadStyle under both prompt-induced and PEFT-based injection strategies. Extensive experiments across seven victim LLMs, including LLaMA, Phi, DeepSeek, and GPT series, demonstrate that BadStyle achieves high attack success rates (ASRs) while maintaining strong stealthiness. The proposed auxiliary target loss substantially improves the stability of backdoor activation, yielding an average ASR improvement of around 30% across style-level triggers. Even in downstream deployment scenarios unknown during injection, the implanted backdoor remains effective. Moreover, BadStyle consistently evades representative input-level defenses and bypasses output-level defenses through simple camouflage. |
| 2026-04-23 | [Can Large Language Models Assist the Comprehension of ROS2 Software Architectures?](http://arxiv.org/abs/2604.21699v1) | Laura Duits, Bouazza El Moutaouakil et al. | Context. The most used development framework for robotics software is ROS2. ROS2 architectures are highly complex, with thousands of components communicating in a decentralized fashion. Goal. We aim to evaluate how LLMs can assist in the comprehension of factual information about the architecture of ROS2 systems. Method. We conduct a controlled experiment where we administer 1,230 prompts to 9 LLMs containing architecturally-relevant questions about 3 ROS2 systems with incremental size. We provide a generic algorithm that systematically generates architecturally-relevant questions for a ROS2 system. Then, we (i) assess the accuracy of the answers of the LLMs against a ground truth established via running and monitoring the 3 ROS2 systems and (ii) qualitatively analyse the explanations provided by the LLMs. Results. Almost all questions are answered correctly across all LLMs (mean=98.22%). gemini-2.5-pro performs best (100% accuracy across all prompts and systems), followed by o3 (99.77%), and gemini-2.5-flash (99.72%); the least performing LLM is gpt-4.1 (95%). Only 300/1,230 prompts are incorrectly answered, of which 249 are about the most complex system. The coherence scores in LLM's explanations range from 0.394 for "service references" to 0.762 for "communication path". The mean perplexity varies significantly across models, with chatgpt-4o achieving the lowest score (19.6) and o4-mini the highest (103.6). Conclusions. There is great potential in the usage of LLMs to aid ROS2 developers in comprehending non-trivial aspects of the software architecture of their systems. Nevertheless, developers should be aware of the intrinsic limitations and different performances of the LLMs and take those into account when using them. |
| 2026-04-23 | [Estimator-Aligned Prospective Sample Size Determination for Designs Using Inverse Probability of Treatment Weighting](http://arxiv.org/abs/2604.21658v1) | Taekwon Hong, Daeyoung Lim et al. | In observational studies, accurately characterizing variance is critical for sample size determination, yet unaccounted-for variability from propensity score estimation and the resulting weights limit the accuracy of standard variance approximations for design. Existing approaches often rely on heuristics or randomized controlled trial (RCT) formulas that treat weights as fixed, potentially misaligning prospective design with the causal estimator used at analysis. We propose an estimator-aligned framework for prospective sample size determination based on generalized estimating equations (GEE) and stacked M-estimation. By merging the propensity score model and marginal structural model (MSM) into a single system of estimating equations, the method propagates nuisance-model uncertainty and directly targets the large-sample variance of the IPTW estimator. For study planning, we estimate a pilot-based large-sample variance factor and introduce a bootstrap stabilization procedure that accounts for both within- and between-pilot variability. The framework applies uniformly across binary, count, and continuous outcomes through link-specific GEE representations under a common design principle. Simulation studies motivated by post-marketing safety and healthcare cost applications demonstrate that anchoring design to this variance improves power calibration relative to conventional RCT-style formulas, particularly in settings with weight instability, outcome sparsity, or heavy-tailed variability. |
| 2026-04-23 | [Task-specific Subnetwork Discovery in Reinforcement Learning for Autonomous Underwater Navigation](http://arxiv.org/abs/2604.21640v1) | Yi-Ling Liu, Melvin Laux et al. | Autonomous underwater vehicles are required to perform multiple tasks adaptively and in an explainable manner under dynamic, uncertain conditions and limited sensing, challenges that classical controllers struggle to address. This demands robust, generalizable, and inherently interpretable control policies for reliable long-term monitoring. Reinforcement learning, particularly multi-task RL, overcomes these limitations by leveraging shared representations to enable efficient adaptation across tasks and environments. However, while such policies show promising results in simulation and controlled experiments, they yet remain opaque and offer limited insight into the agent's internal decision-making, creating gaps in transparency, trust, and safety that hinder real-world deployment. The internal policy structure and task-specific specialization remain poorly understood. To address these gaps, we analyze the internal structure of a pretrained multi-task reinforcement learning network in the HoloOcean simulator for underwater navigation by identifying and comparing task-specific subnetworks responsible for navigating toward different species. We find that in a contextual multi-task reinforcement learning setting with related tasks, the network uses only about 1.5% of its weights to differentiate between tasks. Of these, approximately 85% connect the context-variable nodes in the input layer to the next hidden layer, highlighting the importance of context variables in such settings. Our approach provides insights into shared and specialized network components, useful for efficient model editing, transfer learning, and continual learning for underwater monitoring through a contextual multi-task reinforcement learning method. |
| 2026-04-23 | [Generative Learning Enhanced Intelligent Resource Management for Cell-Free Delay Deterministic Communications](http://arxiv.org/abs/2604.21587v1) | Shuangbo Xiong, Cheng Zhang et al. | Cell-free multiple-input multiple-output (CF-MIMO) architecture significantly enhances wireless network performance, offering a promising solution for delay-sensitive applications. This paper investigates the resource allocation problem in CF-MIMO systems, aiming to maximize energy efficiency (EE) while satisfying delay violation rate constraint. We design a Proximal Policy Optimization (PPO) with a primal-dual method to solve it. To address the low sample efficiency and safety risks caused by cold-start of the designed safe deep reinforcement learning (DRL) method, we propose a novel offline pretraining framework based on virtual constrained Markov decision process (CMDP) modeling. The virtual CMDP consists of reward and cost prediction module, initial-state distribution module and state transition module. Notably, we propose an evidence-aware conditional Gaussian Mixture Model (EA-CGMM) inference approach to mitigate data sparsity and distribution drift issues in state transition modeling. Simulation results demonstrate the effectiveness of CMDP modeling and validate the safety and efficiency of the proposed pretraining framework. Specifically, compared with non-pretrained baseline, the agent pretrained through our proposed framework achieves twice the initial EE and maintains a low delay constraint violation rate of $1\%$, while ultimately converging to an EE that is $4.7\%$ higher with a $50\%$ reduction in exploration steps. Additionally, our proposed pretraining framework implementation exhibits comparable performance to the SOTA diffusion model-based implementation, while achieving a $14$-fold reduction in computational complexity. |
| 2026-04-23 | [Probabilistic Verification of Neural Networks via Efficient Probabilistic Hull Generation](http://arxiv.org/abs/2604.21556v1) | Jingyang Li, Xin Chen et al. | The problem of probabilistic verification of a neural network investigates the probability of satisfying the safe constraints in the output space when the input is given by a probability distribution. It is significant to answer this problem when the input is affected by disturbances often modeled by probabilistic variables. In the paper, we propose a novel neural network probabilistic verification framework which computes a guaranteed range for the safe probability by efficiently finding safe and unsafe probabilistic hulls. Our approach consists of three main innovations: (1) a state space subdivision strategy using regression trees to produce probabilistic hulls, (2) a boundary-aware sampling method which identifies the safety boundary in the input space using samples that are later used for building regression trees, and (3) iterative refinement with probabilistic prioritization for computing a guaranteed range for the safe probability. The accuracy and efficiency of our approach are evaluated on various benchmarks including ACAS Xu and a rocket lander controller. The result shows an obvious advantage over the state of the art. |
| 2026-04-23 | [Unbiased Prevalence Estimation with Multicalibrated LLMs](http://arxiv.org/abs/2604.21549v1) | Fridolin Linder, Thomas Leeper et al. | Estimating the prevalence of a category in a population using imperfect measurement devices (diagnostic tests, classifiers, or large language models) is fundamental to science, public health, and online trust and safety. Standard approaches correct for known device error rates but assume these rates remain stable across populations. We show this assumption fails under covariate shift and that multicalibration, which enforces calibration conditional on the input features rather than just on average, is sufficient for unbiased prevalence estimation under such shift. Standard calibration and quantification methods fail to provide this guarantee. Our work connects recent theoretical work on fairness to a longstanding measurement problem spanning nearly all academic disciplines. A simulation confirms that standard methods exhibit bias growing with shift magnitude, while a multicalibrated estimator maintains near-zero bias. While we focus the discussion mostly on LLMs, our theoretical results apply to any classification model. Two empirical applications -- estimating employment prevalence across U.S. states using the American Community Survey, and classifying political texts across four countries using an LLM -- demonstrate that multicalibration substantially reduces bias in practice, while highlighting that calibration data should cover the key feature dimensions along which target populations may differ. |
| 2026-04-23 | [Ufil: A Unified Framework for Infrastructure-based Localization](http://arxiv.org/abs/2604.21471v1) | Simon Schäfer, Lucas Hegerath et al. | Infrastructure-based localization enhances road safety and traffic management by providing state estimates of road users. Development is hindered by fragmented, application-specific stacks that tightly couple perception, tracking, and middleware. We introduce Ufil, a Unified Framework for Infrastructure-Based Localization with a standardized object model and reusable multi-object tracking components. Ufil offers interfaces and reference implementations for prediction, detection, association, state update, and track management, allowing researchers to improve components without reimplementing the pipeline. Ufil is open-source C++/ROS 2 software with documentation and executable examples. We demonstrate Ufil by integrating three heterogeneous data sources into a single localization pipeline combining (i) vehicle onboard units broadcasting ETSI ITS-G5 Cooperative Awareness Messages, (ii) a lidar-based roadside sensor node, and (iii) an in-road sensitive surface layer. The pipeline runs unchanged in the CARLA simulator and a small-scale CAV testbed, demonstrating Ufil's scale-independent execution model. In a three-lane highway scenario with 423 and 355 vehicles in simulation and testbed, respectively, the fused system achieves lane-level lateral accuracy with mean lateral position RMSEs of 0.31 m in CARLA and 0.29 m in the CPM Lab, and mean absolute orientation errors around 2.2°. Median end-to-end latencies from sensing to fused output remain below 100 ms across all modalities in both environments. |
| 2026-04-23 | [FryNet: Dual-Stream Adversarial Fusion for Non-Destructive Frying Oil Oxidation Assessment](http://arxiv.org/abs/2604.21321v1) | Khaled R Ahmed, Toqi Tahamid Sarker et al. | Monitoring frying oil degradation is critical for food safety, yet current practice relies on destructive wet-chemistry assays that provide no spatial information and are unsuitable for real-time use. We identify a fundamental obstacle in thermal-image-based inspection, the camera-fingerprint shortcut, whereby models memorize sensor-specific noise and thermal bias instead of learning oxidation chemistry, collapsing under video-disjoint evaluation. We propose FryNet, a dual-stream RGB-thermal framework that jointly performs oil-region segmentation, serviceability classification, and regression of four chemical oxidation indices (PV, p-AV, Totox, temperature) in a single forward pass. A ThermalMiT-B2 backbone with channel and spatial attention extracts thermal features, while an RGB-MAE Encoder learns chemically grounded representations via masked autoencoding and chemical alignment. Dual-Encoder DANN adversarially regularizes both streams against video identity via Gradient Reversal Layers, and FiLM fusion bridges thermal structure with RGB chemical context. On 7,226 paired frames across 28 frying videos, FryNet achieves 98.97% mIoU, 100% classification accuracy, and 2.32 mean regression MAE, outperforming all seven baselines. |
| 2026-04-23 | [CAP: Controllable Alignment Prompting for Unlearning in LLMs](http://arxiv.org/abs/2604.21251v1) | Zhaokun Wang, Jinyu Guo et al. | Large language models (LLMs) trained on unfiltered corpora inherently risk retaining sensitive information, necessitating selective knowledge unlearning for regulatory compliance and ethical safety. However, existing parameter-modifying methods face fundamental limitations: high computational costs, uncontrollable forgetting boundaries, and strict dependency on model weight access. These constraints render them impractical for closed-source models, yet current non-invasive alternatives remain unsystematic and reliant on empirical experience. To address these challenges, we propose the Controllable Alignment Prompting for Unlearning (CAP) framework, an end-to-end prompt-driven unlearning paradigm. CAP decouples unlearning into a learnable prompt optimization process via reinforcement learning, where a prompt generator collaborates with the LLM to suppress target knowledge while preserving general capabilities selectively. This approach enables reversible knowledge restoration through prompt revocation. Extensive experiments demonstrate that CAP achieves precise, controllable unlearning without updating model parameters, establishing a dynamic alignment mechanism that overcomes the transferability limitations of prior methods. |
| 2026-04-23 | [Skull-Conforming Acoustic Holographic Lenses for Transcranial Targeting](http://arxiv.org/abs/2604.21207v1) | Ceren Cengiz, Mihir Pewekar et al. | Transcranial focused ultrasound (tFUS) offers noninvasive access to deep brain circuits but remains limited by skull-induced phase aberration, acoustic impedance mismatch, and poor volumetric control of intracranial pressure fields. Conventional phased-array and planar holographic strategies compensate aberrations electronically or computationally, yet do not resolve geometric and coupling inconsistencies imposed by subject-specific cranial morphology. We introduce personalized skull-conforming acoustic holograms that physically encode individualized wavefront corrections into a conformal acoustic interface. Within a subject-specific volumetric holography (SSVH) framework, cranial geometry and therapeutic constraints are embedded into a physics-based optimization pipeline for holographic phase synthesis. The resulting lens is integrated with a skull- and skin-conforming coupling layer that enhances impedance continuity, reduces reflection losses, and stabilizes spatial alignment, enabling simultaneous aberration mitigation and efficient transcranial transmission. Numerical simulations across multiple subjects and targets demonstrate consistent volumetric focusing and reliable target coverage while maintaining pressure fields within safety limits. Experimental validation using an ex vivo human skull confirms accurate fabrication, effective acoustic coupling, and faithful reconstruction of designed three-dimensional acoustic fields. By unifying wavefront engineering with anatomical conformity, this work establishes skull-conforming acoustic holography as a scalable strategy for high-fidelity, anatomically adaptive transcranial ultrasound targeting. |
| 2026-04-23 | [How VLAs (Really) Work In Open-World Environments](http://arxiv.org/abs/2604.21192v1) | Amir Rasouli, Yangzheng Wu et al. | Vision-language-action models (VLAs) have been extensively used in robotics applications, achieving great success in various manipulation problems. More recently, VLAs have been used in long-horizon tasks and evaluated on benchmarks, such as BEHAVIOR1K (B1K), for solving complex household chores. The common metric for measuring progress in such benchmarks is success rate or partial score based on satisfaction of progress-agnostic criteria, meaning only the final states of the objects are considered, regardless of the events that lead to such states. In this paper, we argue that using such evaluation protocols say little about safety aspects of operation and can potentially exaggerate reported performance, undermining core challenges for future real-world deployment. To this end, we conduct a thorough analysis of state-of-the-art models on the B1K Challenge and evaluate policies in terms of robustness via reproducibility and consistency of performance, safety aspects of policies operations, task awareness, and key elements leading to the incompletion of tasks. We then propose evaluation protocols to capture safety violations to better measure the true performance of the policies in more complex and interactive scenarios. At the end, we discuss the limitations of the existing VLAs and motivate future research. |
| 2026-04-23 | [Full-Body Dynamic Safety for Robot Manipulators: 3D Poisson Safety Functions for CBF-Based Safety Filters](http://arxiv.org/abs/2604.21189v1) | Meg Wilkinson, Gilbert Bahati et al. | Collision avoidance for robotic manipulators requires enforcing full-body safety constraints in high-dimensional configuration spaces. Control Barrier Function (CBF) based safety filters have proven effective in enabling safe behaviors, but enforcing the high number of constraints needed for safe manipulation leads to theoretic and computational challenges. This work presents a framework for full-body collision avoidance for manipulators in dynamic environments by leveraging 3D Poisson Safety Functions (PSFs). In particular, given environmental occupancy data, we sample the manipulator surface at a prescribed resolution and shrink free space via a Pontryagin difference according to this resolution. On this buffered domain, we synthesize a globally smooth CBF by solving Poisson's equation, yielding a single safety function for the entire environment. This safety function, evaluated at each sampled point, yields task-space CBF constraints enforced by a real-time safety filter via a multi-constraint quadratic program. We prove that keeping the sample points safe in the buffered region guarantees collision avoidance for the entire continuous robot surface. The framework is validated on a 7-degree-of-freedom manipulator in dynamic environments. |

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



