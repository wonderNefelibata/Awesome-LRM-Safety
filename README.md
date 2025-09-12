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
| 2025-09-11 | [Steering MoE LLMs via Expert (De)Activation](http://arxiv.org/abs/2509.09660v1) | Mohsen Fayyaz, Ali Modarressi et al. | Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts. |
| 2025-09-11 | [Measuring Epistemic Humility in Multimodal Large Language Models](http://arxiv.org/abs/2509.09658v1) | Bingkui Tong, Jiaer Xia et al. | Hallucinations in multimodal large language models (MLLMs) -- where the model generates content inconsistent with the input image -- pose significant risks in real-world applications, from misinformation in visual question answering to unsafe errors in decision-making. Existing benchmarks primarily test recognition accuracy, i.e., evaluating whether models can select the correct answer among distractors. This overlooks an equally critical capability for trustworthy AI: recognizing when none of the provided options are correct, a behavior reflecting epistemic humility. We present HumbleBench, a new hallucination benchmark designed to evaluate MLLMs' ability to reject plausible but incorrect answers across three hallucination types: object, relation, and attribute. Built from a panoptic scene graph dataset, we leverage fine-grained scene graph annotations to extract ground-truth entities and relations, and prompt GPT-4-Turbo to generate multiple-choice questions, followed by a rigorous manual filtering process. Each question includes a "None of the above" option, requiring models not only to recognize correct visual information but also to identify when no provided answer is valid. We evaluate a variety of state-of-the-art MLLMs -- including both general-purpose and specialized reasoning models -- on HumbleBench and share valuable findings and insights with the community. By incorporating explicit false-option rejection, HumbleBench fills a key gap in current evaluation suites, providing a more realistic measure of MLLM reliability in safety-critical settings. Our code and dataset are released publicly and can be accessed at https://github.com/maifoundations/HumbleBench. |
| 2025-09-11 | [Feasibility-Guided Fair Adaptive Offline Reinforcement Learning for Medicaid Care Management](http://arxiv.org/abs/2509.09655v1) | Sanjay Basu, Sadiq Y. Patel et al. | We introduce Feasibility-Guided Fair Adaptive Reinforcement Learning (FG-FARL), an offline RL procedure that calibrates per-group safety thresholds to reduce harm while equalizing a chosen fairness target (coverage or harm) across protected subgroups. Using de-identified longitudinal trajectories from a Medicaid population health management program, we evaluate FG-FARL against behavior cloning (BC) and HACO (Hybrid Adaptive Conformal Offline RL; a global conformal safety baseline). We report off-policy value estimates with bootstrap 95% confidence intervals and subgroup disparity analyses with p-values. FG-FARL achieves comparable value to baselines while improving fairness metrics, demonstrating a practical path to safer and more equitable decision support. |
| 2025-09-11 | [Reconstructing the origin of black hole mergers using sparse astrophysical models](http://arxiv.org/abs/2509.09647v1) | V. Gayathri, Giuliano Iorio et al. | The astrophysical origin of binary black hole mergers discovered by LIGO and Virgo remains uncertain. Efforts to reconstruct the processes that lead to mergers typically rely on either astrophysical models with fixed parameters, or continuous analytical models that can be fit to observations. Given the complexity of astrophysical formation mechanisms, these methods typically cannot fully take into account model uncertainties, nor can they fully capture the underlying processes. Here, we present a merger population analysis that can take a discrete set of simulated model distributions as its input to interpret observations. The analysis can take into account multiple formation scenarios as fractional contributors to the total set of observations, and can naturally account for model uncertainties. We apply this technique to investigate the origin of black hole mergers observed by LIGO Virgo. Specifically, we consider a model of AGN assisted black hole merger distributions, exploring a range of AGN parameters along with several {{SEVN}} population synthesis models that vary in common envelope efficiency parameter ($\alpha$) and metallicity ($Z$). We estimate the posterior distributions for AGN+SEVN models using $87$ BBH detections from the $O1--O3$ observation runs. The inferred total merger rate is $46.2 {Gpc}^{-3} {yr}^{-1}$, with the AGN sub-population contributing $21.2{Gpc}^{-3}{yr}^{-1}$ and the SEVN sub-population contributing $25.0 {Gpc}^{-3} {yr}^{-1}$. |
| 2025-09-11 | [A Neuromorphic Incipient Slip Detection System using Papillae Morphology](http://arxiv.org/abs/2509.09546v1) | Yanhui Lu, Zeyu Deng et al. | Detecting incipient slip enables early intervention to prevent object slippage and enhance robotic manipulation safety. However, deploying such systems on edge platforms remains challenging, particularly due to energy constraints. This work presents a neuromorphic tactile sensing system based on the NeuroTac sensor with an extruding papillae-based skin and a spiking convolutional neural network (SCNN) for slip-state classification. The SCNN model achieves 94.33% classification accuracy across three classes (no slip, incipient slip, and gross slip) in slip conditions induced by sensor motion. Under the dynamic gravity-induced slip validation conditions, after temporal smoothing of the SCNN's final-layer spike counts, the system detects incipient slip at least 360 ms prior to gross slip across all trials, consistently identifying incipient slip before gross slip occurs. These results demonstrate that this neuromorphic system has stable and responsive incipient slip detection capability. |
| 2025-09-11 | [Unified Framework for Hybrid Aleatory and Epistemic Uncertainty Propagation via Decoupled Multi-Probability Density Evolution Method](http://arxiv.org/abs/2509.09535v1) | Yi Luo, Meng-Ze Lyu et al. | This paper presents a unified framework for uncertainty propagation in dynamical systems involving hybrid aleatory and epistemic uncertainties. The framework accommodates precise probabilistic, imprecise probabilistic, and non-probabilistic representations, including the distribution-free probability-box (p-box). A central aspect of the framework involves transforming the original uncertainty inputs into an augmented random space, yielding the primary challenge of determining the conditional probability density function (PDF) of the response quantity of interest given epistemic uncertainty parameters. The recently proposed decoupled multi-probability density evolution method (decoupled M-PDEM) is employed to numerically solve the conditional PDF for complex dynamical systems. Several numerical examples illustrate the applicability, efficiency, and accuracy of the proposed framework. These include a linear single-degree-of-freedom (SDOF) system subject to Gaussian white noise with its natural frequency modeled as a p-box, a 10-DOF hysteretic structure subject to imprecise seismic loads, and a crash box model with mixed random and interval system parameters. |
| 2025-09-11 | [GrACE: A Generative Approach to Better Confidence Elicitation in Large Language Models](http://arxiv.org/abs/2509.09438v1) | Zhaohan Zhang, Ziquan Liu et al. | Assessing the reliability of Large Language Models (LLMs) by confidence elicitation is a prominent approach to AI safety in high-stakes applications, such as healthcare and finance. Existing methods either require expensive computational overhead or suffer from poor calibration, making them impractical and unreliable for real-world deployment. In this work, we propose GrACE, a Generative Approach to Confidence Elicitation that enables scalable and reliable confidence elicitation for LLMs. GrACE adopts a novel mechanism in which the model expresses confidence by the similarity between the last hidden state and the embedding of a special token appended to the vocabulary, in real-time. We fine-tune the model for calibrating the confidence with calibration targets associated with accuracy. Experiments with three LLMs and two benchmark datasets show that the confidence produced by GrACE achieves the best discriminative capacity and calibration on open-ended generation tasks, outperforming six competing methods without resorting to additional sampling or an auxiliary model. Moreover, we propose two strategies for improving test-time scaling based on confidence induced by GrACE. Experimental results show that using GrACE not only improves the accuracy of the final decision but also significantly reduces the number of required samples in the test-time scaling scheme, indicating the potential of GrACE as a practical solution for deploying LLMs with scalable, reliable, and real-time confidence estimation. |
| 2025-09-11 | [Real-Time Kinematic Positioning and Optical See-Through Head-Mounted Display for Outdoor Tracking: Hybrid System and Preliminary Assessment](http://arxiv.org/abs/2509.09412v1) | Muhannad Ismael, Maël Cornil | This paper presents an outdoor tracking system using Real-Time Kinematic (RTK) positioning and Optical See-Through Head Mounted Display(s) (OST-HMD(s)) in urban areas where the accurate tracking of objects is critical and where displaying occluded information is important for safety reasons. The approach presented here replaces 2D screens/tablets and offers distinct advantages, particularly in scenarios demanding hands-free operation. The integration of RTK, which provides centimeter-level accuracy of tracked objects, with OST-HMD represents a promising solution for outdoor applications. This paper provides valuable insights into leveraging the combined potential of RTK and OST-HMD for outdoor tracking tasks from the perspectives of systems integration, performance optimization, and usability. The main contributions of this paper are: \textbf{1)} a system for seamlessly merging RTK systems with OST-HMD to enable relatively precise and intuitive outdoor tracking, \textbf{2)} an approach to determine a global location to achieve the position relative to the world, \textbf{3)} an approach referred to as 'semi-dynamic' for system assessment. Moreover, we offer insights into several relevant future research topics aimed at improving the OST-HMD and RTK hybrid system for outdoor tracking. |
| 2025-09-11 | [A Hybrid Hinge-Beam Continuum Robot with Passive Safety Capping for Real-Time Fatigue Awareness](http://arxiv.org/abs/2509.09404v1) | Tongshun Chen, Zezhou Sun et al. | Cable-driven continuum robots offer high flexibility and lightweight design, making them well-suited for tasks in constrained and unstructured environments. However, prolonged use can induce mechanical fatigue from plastic deformation and material degradation, compromising performance and risking structural failure. In the state of the art, fatigue estimation of continuum robots remains underexplored, limiting long-term operation. To address this, we propose a fatigue-aware continuum robot with three key innovations: (1) a Hybrid Hinge-Beam structure where TwistBeam and BendBeam decouple torsion and bending: passive revolute joints in the BendBeam mitigate stress concentration, while TwistBeam's limited torsional deformation reduces BendBeam stress magnitude, enhancing durability; (2) a Passive Stopper that safely constrains motion via mechanical constraints and employs motor torque sensing to detect corresponding limit torque, ensuring safety and enabling data collection; and (3) a real-time fatigue-awareness method that estimates stiffness from motor torque at the limit pose, enabling online fatigue estimation without additional sensors. Experiments show that the proposed design reduces fatigue accumulation by about 49% compared with a conventional design, while passive mechanical limiting combined with motor-side sensing allows accurate estimation of structural fatigue and damage. These results confirm the effectiveness of the proposed architecture for safe and reliable long-term operation. |
| 2025-09-11 | [Novel Room-Temperature Synthesis of Tellurium-Loaded Liquid Scintillators for Neutrinoless Double Beta Decay Search](http://arxiv.org/abs/2509.09316v1) | Yayun Ding, Mengchao Liu et al. | This study establishes an innovative room-temperature synthesis approach for tellurium-diol (Te-diol) compounds, which are crucial components in tellurium-loaded liquid scintillator (Te-LS). The synthesis involves the direct reaction of telluric acid with diols (e.g., 1,2-hexanediol) in methanol under ambient conditions (20$\pm$5{\deg}C) , with the key features of lower energy consumption, enhanced safety, and improved scalability. Mechanistic studies reveal that methanol serves not merely as a solvent but also as a catalyst, playing a critical role in the room-temperature synthesis. The organic amine N,N-dimethyldodecylamine demonstrates dual functionality as both catalyst and stabilizer. The Te-diol compounds enable fabrication of high-performance Te-LS exhibiting exceptional optical transparency ($\Delta Abs$(430nm) $\leq$ 0.0003 per 1% Te loading), achieving long-term spectral stability exceeding or approaching one year for both 1% and 3% Te formulations, and demonstrating a light yield comparable to that achieved by the azeotropic distillation method. The developed protocol offers a green, efficient alternative for large-scale Te-LS production, particularly valuable for next-generation neutrinoless double-beta decay experiments. |
| 2025-09-11 | [Model-Agnostic Open-Set Air-to-Air Visual Object Detection for Reliable UAV Perception](http://arxiv.org/abs/2509.09297v1) | Spyridon Loukovitis, Anastasios Arsenos et al. | Open-set detection is crucial for robust UAV autonomy in air-to-air object detection under real-world conditions. Traditional closed-set detectors degrade significantly under domain shifts and flight data corruption, posing risks to safety-critical applications. We propose a novel, model-agnostic open-set detection framework designed specifically for embedding-based detectors. The method explicitly handles unknown object rejection while maintaining robustness against corrupted flight data. It estimates semantic uncertainty via entropy modeling in the embedding space and incorporates spectral normalization and temperature scaling to enhance open-set discrimination. We validate our approach on the challenging AOT aerial benchmark and through extensive real-world flight tests. Comprehensive ablation studies demonstrate consistent improvements over baseline methods, achieving up to a 10\% relative AUROC gain compared to standard YOLO-based detectors. Additionally, we show that background rejection further strengthens robustness without compromising detection accuracy, making our solution particularly well-suited for reliable UAV perception in dynamic air-to-air environments. |
| 2025-09-11 | [Incentivizing Safer Actions in Policy Optimization for Constrained Reinforcement Learning](http://arxiv.org/abs/2509.09208v1) | Somnath Hazra, Pallab Dasgupta et al. | Constrained Reinforcement Learning (RL) aims to maximize the return while adhering to predefined constraint limits, which represent domain-specific safety requirements. In continuous control settings, where learning agents govern system actions, balancing the trade-off between reward maximization and constraint satisfaction remains a significant challenge. Policy optimization methods often exhibit instability near constraint boundaries, resulting in suboptimal training performance. To address this issue, we introduce a novel approach that integrates an adaptive incentive mechanism in addition to the reward structure to stay within the constraint bound before approaching the constraint boundary. Building on this insight, we propose Incrementally Penalized Proximal Policy Optimization (IP3O), a practical algorithm that enforces a progressively increasing penalty to stabilize training dynamics. Through empirical evaluation on benchmark environments, we demonstrate the efficacy of IP3O compared to the performance of state-of-the-art Safe RL algorithms. Furthermore, we provide theoretical guarantees by deriving a bound on the worst-case error of the optimality achieved by our algorithm. |
| 2025-09-11 | [Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments](http://arxiv.org/abs/2509.09206v1) | Farhad Nawaz, Faizan M. Tariq et al. | Accurately reasoning about future parking spot availability and integrated planning is critical for enabling safe and efficient autonomous valet parking in dynamic, uncertain environments. Unlike existing methods that rely solely on instantaneous observations or static assumptions, we present an approach that predicts future parking spot occupancy by explicitly distinguishing between initially vacant and occupied spots, and by leveraging the predicted motion of dynamic agents. We introduce a probabilistic spot occupancy estimator that incorporates partial and noisy observations within a limited Field-of-View (FoV) model and accounts for the evolving uncertainty of unobserved regions. Coupled with this, we design a strategy planner that adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, and intelligently incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency, safety margins, and trajectory smoothness compared to existing approaches. |
| 2025-09-11 | [FPI-Det: a face--phone Interaction Dataset for phone-use detection and understanding](http://arxiv.org/abs/2509.09111v1) | Jianqin Gao, Tianqi Wang et al. | The widespread use of mobile devices has created new challenges for vision systems in safety monitoring, workplace productivity assessment, and attention management. Detecting whether a person is using a phone requires not only object recognition but also an understanding of behavioral context, which involves reasoning about the relationship between faces, hands, and devices under diverse conditions. Existing generic benchmarks do not fully capture such fine-grained human--device interactions. To address this gap, we introduce the FPI-Det, containing 22{,}879 images with synchronized annotations for faces and phones across workplace, education, transportation, and public scenarios. The dataset features extreme scale variation, frequent occlusions, and varied capture conditions. We evaluate representative YOLO and DETR detectors, providing baseline results and an analysis of performance across object sizes, occlusion levels, and environments. Source code and dataset is available at https://github.com/KvCgRv/FPI-Det. |
| 2025-09-11 | [Content Moderation Futures](http://arxiv.org/abs/2509.09076v1) | Lindsay Blackwell | This study examines the failures and possibilities of contemporary social media governance through the lived experiences of various content moderation professionals. Drawing on participatory design workshops with 33 practitioners in both the technology industry and broader civil society, this research identifies significant structural misalignments between corporate incentives and public interests. While experts agree that successful content moderation is principled, consistent, contextual, proactive, transparent, and accountable, current technology companies fail to achieve these goals, due in part to exploitative labor practices, chronic underinvestment in user safety, and pressures of global scale. I argue that successful governance is undermined by the pursuit of technological novelty and rapid growth, resulting in platforms that necessarily prioritize innovation and expansion over public trust and safety. To counter this dynamic, I revisit the computational history of care work, to motivate present-day solidarity amongst platform governance workers and inspire systemic change. |
| 2025-09-10 | [Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M](http://arxiv.org/abs/2509.09055v1) | Piyush Pant | This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. |
| 2025-09-10 | [An Improved Rapid Performance Analysis Model for Solenoidal Magnetic Radiation Shields](http://arxiv.org/abs/2509.09051v1) | Joseph L. Hesse-Withbroe, Katya S. Arquilla | Astronauts participating in deep-space exploration missions will be exposed to significantly greater amounts of radiation than is typically encountered on Earth or in low Earth orbit (LEO), which poses significant risks to crew health and mission safety. Active magnetic radiation shields based on the Lorentz deflection of charged particles have the potential to reduce astronaut doses with lower mass costs than passive shielding techniques. Typically, active shielding performance is evaluated using high-fidelity Monte Carlo simulations, which are too computationally expensive to evaluate an entire trade space of shield designs. A rapid, semi-analytical model based on the High Charge and Energy Transport code (HZETRN) developed in 2014 provided an alternative method by which to evaluate the performance of solenoidal shields. However, various simplifying assumptions made in the original model have limited its accuracy, and therefore require evaluation and correction. In this work, a number of aspects of the original semi-analytical model are updated and validated by Monte Carlo simulation, then used to recharacterize the design trade space of solenoidal magnetic shields. The updated model predicts improved performance for weaker shields as compared to the original model, but greatly diminished performance for strong shields with bending powers greater than 20 T-m. Overall, the results indicate that magnetic shields enable significant mass savings over passive shields for mission scenarios where the requisite dose reduction is greater than about 60% relative to free space, which includes most exploration missions longer than one year with significant time spent outside LEO. |
| 2025-09-10 | [Design of Reliable and Resilient Electric Power Systems for Wide-Body All-Electric Aircraft](http://arxiv.org/abs/2509.09044v1) | Mona Ghassemi | To achieve net-zero emissions by 2050, all-electric transportation is a promising option. In the U.S., the transportation sector contributes the largest share (29 percent) of greenhouse gas emissions. While electric vehicles are approaching maturity, aviation is only beginning to develop electrified aircraft for commercial flights. More than 75 percent of aviation emissions come from large aircraft, and this impact will worsen with 4-5 percent annual air travel growth. Aircraft electrification has led to two types: more electric aircraft (MEA) and all-electric aircraft (AEA). A MEA replaces subsystems such as hydraulics with electric alternatives, whereas an AEA uses electrically driven subsystems and provides thrust fully from electrochemical energy units (EEUs). For wide-body AEA, thrust demand is about 25 MW plus 1 MW for non-thrust loads, creating major challenges for electric power system (EPS) design. Achieving maximum power density requires minimizing mass and volume. Increasing voltage into the kilovolt range using medium-voltage direct current (MVDC) is a feasible option to enhance power transfer. Consequently, designing an MVDC EPS for wide-body AEA is critical. Because EPS failures could jeopardize passenger safety, reliability and resilience are essential. This chapter presents a load-flow model for DC systems to determine power flows in both normal and single-contingency conditions, followed by analysis of optimal MVDC EPS architectures. A complete EPS for wide-body AEA is introduced, with EEUs and non-propulsion loads located, distances estimated, and flow studies performed. Multiple architectures are evaluated for reliability, power density, power loss, and cost to identify optimal solutions. |
| 2025-09-10 | [YouthSafe: A Youth-Centric Safety Benchmark and Safeguard Model for Large Language Models](http://arxiv.org/abs/2509.08997v1) | Yaman Yu, Yiren Liu et al. | Large Language Models (LLMs) are increasingly used by teenagers and young adults in everyday life, ranging from emotional support and creative expression to educational assistance. However, their unique vulnerabilities and risk profiles remain under-examined in current safety benchmarks and moderation systems, leaving this population disproportionately exposed to harm. In this work, we present Youth AI Risk (YAIR), the first benchmark dataset designed to evaluate and improve the safety of youth LLM interactions. YAIR consists of 12,449 annotated conversation snippets spanning 78 fine grained risk types, grounded in a taxonomy of youth specific harms such as grooming, boundary violation, identity confusion, and emotional overreliance. We systematically evaluate widely adopted moderation models on YAIR and find that existing approaches substantially underperform in detecting youth centered risks, often missing contextually subtle yet developmentally harmful interactions. To address these gaps, we introduce YouthSafe, a real-time risk detection model optimized for youth GenAI contexts. YouthSafe significantly outperforms prior systems across multiple metrics on risk detection and classification, offering a concrete step toward safer and more developmentally appropriate AI interactions for young users. |
| 2025-09-10 | [Monte Carlo Simulation of Spallation and Fission Fragment Distributions for ADS-Related Nuclear Reactions](http://arxiv.org/abs/2509.08996v1) | Sun Wenming | Monte Carlo simulations with the CRISP code were conducted to study spallation and fission fragment distributions induced by intermediate- and high-energy protons and photons on actinide and pre-actinide nuclei. The model accounts for intranuclear cascade, pre-equilibrium, and evaporation-fission competition, enabling consistent treatment of both residues and fission products. Comparisons with experimental data show good agreement in mass and charge distributions, with minor deviations for light fragments. The results highlight the reliability of Monte Carlo approaches for predicting residual nuclei and fragment yields under accelerator-driven system (ADS) conditions. This work provides nuclear data relevant to ADS design, safety, and transmutation analysis |

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



