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
| 2026-08-19 | [Autonomous Cyber Defense in Connected Vehicles: A Multi-Agent Approach to V2X Security](http://arxiv.org/abs/2608.19135v1) | Krishna Teja Medam | A connected vehicle has roughly 100 milliseconds to decide whether an incoming Basic Safety Message is real or fabricated. If a false emergency braking alert reaches the planning pipeline in time, the car brakes - a safety failure triggered by a security failure. Existing intrusion detection systems are not designed to handle that coupling. They operate per vehicle, per message, with static rules - blind to attack patterns that only emerge across a fleet or over time, and blind to the fundamental tension between dropping a suspicious message and dropping a real emergency alert. We propose a three-tier multi-agent architecture that treats this timing constraint as a hard design requirement, not a performance target. At the vehicle level, an onboard agent classifies each incoming V2X message into one of four actions - Accept, Drop, Quarantine, or Escalate - within a 10-millisecond budget, deliberately biased toward Escalate when uncertain, passing ambiguous cases to the roadside edge agent rather than risking a dropped legitimate alert. The edge agent operates across a roadside unit zone with a 50-millisecond budget, fusing threat assessments from multiple vehicles and resolving safety-security conflicts using complementary sensor observations. The cloud tier refines detection models through Byzantine fault-tolerant federated learning and redistributes updated weights to the fleet. Every timing constraint derives directly from the 100-millisecond Basic Safety Message cycles mandated by SAE J2735 and ETSI EN 302 637-2. No existing framework simultaneously assigns standards-grounded latency budgets to all three deployment tiers while treating safety-security conflict resolution as a first-class design constraint. Remaining open problems - adversarial poisoning at the edge and the absence of regulatory frameworks for autonomous security response - are discussed as future work. |
| 2026-08-19 | [Detecting Backdoors in Object Detection via Pre-NMS Prediction Distribution Shift](http://arxiv.org/abs/2608.19088v1) | Longtian Wang, Zhengyu Zhao et al. | Object detection models deployed in safety-critical applications remain vulnerable to backdoor attacks that cause targeted misbehaviors when a hidden trigger is present. Existing detection methods either rely on trigger inversion or exploit architecture-specific assumptions, and critically, representative existing methods fail to generalize reliably to scene-level attacks, where a single trigger induces anomalous behavior across all objects in the scene simultaneously. We present DistScan, a backdoor detection framework based on a simple but previously unexploited observation: backdoor injection systematically shifts a model's pre-NMS prediction class distribution away from its training class frequencies, even on clean inputs without any trigger present. DistScan aggregates intermediate class predictions over a clean validation set and flags a model as backdoored if the resulting distribution deviates significantly from the training class frequencies, requiring no model weight access, no trigger knowledge, and no additional training. Extensive experiments on MS-COCO and PASCAL VOC across two architectures and three scene-level attack scenarios demonstrate that DistScan substantially outperforms existing methods, improving average detection accuracy over the best-performing applicable baseline by 27.32 percentage points. |
| 2026-08-19 | [DA-WAM: Decision-Aligned Future Latents for Driving World Models](http://arxiv.org/abs/2608.19085v1) | Ruiguo Zhong, Benshan Ma et al. | Anticipating how scenes evolve under ego actions is fundamental to safe autonomous driving, yet the full potential of world models for decision-making remains unrealized. The critical challenge lies in ensuring that future modeling is not merely predictive, but decision-informative: the predicted future must directly shape which trajectory is selected. Existing approaches decouple future representation learning from planning optimization, or share predicted states across trajectory candidates, thereby diluting the action-specific consequences that ought to guide selection. To bridge this gap, we propose DA-WAM, a framework that unifies predictive representation learning, action-conditioned future modeling, and trajectory scoring under a single decision-making objective. DA-WAM maintains predictive supervision throughout planner optimization via an online encoder and a stable momentum target, allowing future representations to co-evolve with the driving task. An action-conditioned predictor generates a distinct future latent state per trajectory candidate, which is then evaluated by a future-latent-conditioned factorized scorer. For the expert-matched trajectory, the predicted future latent is supervised by the observed future representation, while safety-critical hard negatives provide additional supervision near planning boundaries. Extensive experiments on NAVSIM-v1 and NAVSIM-v2 demonstrate state-of-the-art performance, while ablations and diagnostic analyses validate the key components. |
| 2026-08-19 | [SPK: Eliciting Structured Prior Knowledge for Interpretable Out-of-Distribution Detection in Real-Time Object Detection](http://arxiv.org/abs/2608.19080v1) | Changshun Wu, Weicheng He et al. | Object detectors often produce over-confident predictions for objects outside their training categories, leading to so-called out-of-distribution (OoD) hallucinations. Existing approaches for detecting or mitigating such hallucinations typically either construct scoring functions directly over learned object detector representations or modify the object detector itself to suppress hallucination emergence. However, the latent priors implicitly encoded in these representations remain largely unexplored and have not been explicitly decoded for OoD detection. To uncover and exploit these latent priors, we propose Structured Prior Knowledge (SPK), a hallucination-oriented framework that explicitly elicits OoD-relevant priors from pretrained object detectors. Specifically, SPK leverages in-distribution data and hallucination-inducing samples as diagnostic supervision to elicit part-level semantic concepts underlying object detector decision-making, rather than using them merely for rejection or object detector adaptation. The elicited semantic priors are further integrated with geometric and contextual priors to form a compact five-dimensional SPK representation for OoD detection. Extensive experiments across diverse object detector architectures and multiple OoD benchmarks demonstrate that SPK achieves state-of-the-art OoD detection. Our findings reveal that pretrained object detectors already encode substantially richer latent knowledge than is typically exploited for OoD detection. More importantly, this knowledge can be explicitly elicited and organized into a compact, structured, and interpretable knowledge space for prediction reliability analysis. This suggests a promising proactive route for improving object detector reliability by explicitly uncovering and leveraging latent priors. Code and data are available at: https://gricad-gitlab.univ-grenoble-alpes.fr/dnn-safety/spk |
| 2026-08-19 | [Robust Risk Under Evolving Uncertainty: A Wasserstein Counterpart of the Entropic Value-at-Risk](http://arxiv.org/abs/2608.19073v1) | Deep Kumar Ganguly, Jan Křetínský | An agent still learning its environment should be cautious while ignorant and bold once confident. The entropic value-at-risk captures this through a robust-optimization identity---a confidence level fixes the radius of a relative-entropy ball of alternative models---but that ball cannot reach catastrophes the nominal deems impossible, precisely what a safe agent must hedge. We instead use an optimal-transport ball and study the coherent risk measure it induces, the Wasserstein entropic value-at-risk. It has a variational dual mirroring the entropic formula (an inverse temperature becomes a transport price), occupies a definite place in the risk hierarchy, and provably accounts for the reachable catastrophes the entropic measure ignores; we verify both dualities numerically. Driving the transport radius by belief entropy then yields a closed-form robust dynamic-programming operator whose caution contracts as the belief sharpens, with a certified safety sandwich and a sharp safety switch. |
| 2026-08-19 | [High-power TCV scenario for conventional and alternative divertor studies](http://arxiv.org/abs/2608.18939v1) | K. Lee, C. Theiler et al. | Alternative divertor configurations (ADCs) must be evaluated under boundary plasma conditions approaching reactor-level values to be considered a reliable, physics-based solution for tokamak power exhaust. Most ADC experiments performed to date were at relatively low exhaust power. This work presents a high-power scenario on the TCV tokamak enabling the study of a wide variety of divertor magnetic shapes under an expanded SOL and power exhaust parameter space. The scenario is characterized by high power levels of electron cyclotron resonance heating ($2.5\,\text{MW}$ fully absorbed in a $\sim1\,\text{m}^{3}$ plasma) at high plasma current (edge safety factor $q_{95}\approx 2.5$), and low upstream separatrix densities ($n_{e,\text{u}}\approx1\times10^{19}\,\text{m}^{-3}$, Greenwald fraction $f_{\text{G}}\approx 0.1$). Stationary parallel heat fluxes up to $100\,\text{MW m}^{-2}$ are measured at the divertor target, an order of magnitude above previous TCV power exhaust studies. The obtained SOL collisionality and Lengyel detachment scaling metric lie within range of values expected in future reactors (SPARC, ITER, ARC). |
| 2026-08-19 | [Breaking the weakest link to evade vision language models](http://arxiv.org/abs/2608.18938v1) | Ilan Zini, Boussad Addad et al. | Vision Language Models (VLMs) have recently emerged as a critical component of multimodal AI systems, enabling joint reasoning over visual and textual inputs in real-world and safety-critical applications. Despite their growing deployment, the robustness of VLMs against adversarial threats remains insufficiently explored, particularly in the context of evasion attacks targeting multimodal alignment. In this work, we investigate the vulnerability of VLMs to adversarial perturbations applied to visual inputs and study two attack settings: untargeted attacks, where the goal is to disrupt the model's interpretation of the original image, and targeted attacks, where the adversary aims to force the model to generate a specific semantic description unrelated to the original image. To efficiently generate adversarial examples, we propose a gradient-based attack method that performs optimization exclusively on the vision encoder of the VLM rather than on the entire multimodal architecture. This design significantly reduces the computational cost and resource requirements of the attack while maintaining strong effectiveness. We evaluate our approach on several open-source VLMs, including Qwen2.5-VL, Granite-Vision, FastVLM, and Phi-3.5-Vision, and show that small, human-imperceptible perturbations can substantially alter the textual interpretation produced by the models. Our findings highlight the vulnerability of modern VLMs to adversarial manipulation and emphasize the need for improved robustness and security mechanisms in multimodal AI systems. |
| 2026-08-19 | [\textsc{TestifAI}: Tomography-Based Testing for Deep Learning Systems](http://arxiv.org/abs/2608.18900v1) | Arooj Arif, Tobias Hartung et al. | As AI systems are increasingly deployed in safety-critical application domains (e.g., autonomous driving), associated risks increase too. Deep learning models underlying modern AI systems, therefore, must undergo thorough testing to ensure their correct behaviour. A single robustness test involves thousands of inferences to empirically verify if a model's outputs remain stable under a bounded perturbation of its inputs. However, existing testing frameworks lack the means to systematically explore and summarise robustness across a combinatorial space of perturbations.   We propose TestifAI, a deep learning testing framework for efficient and accurate estimation of robustness against combinations of perturbations. TestifAI enables users to specify operational conditions as structured spaces of semantic input perturbations (e.g., image blur, brightness and zoom) and discrete severity levels (e.g., low, medium and high). Users can query model robustness for any combination (e.g., "low blur, high brightness, and medium zoom"). To achieve efficiency and accuracy, TestifAI introduces partial model tomography, a novel approach to reconstructing model behaviour in a multi-perturbation space from tests that apply only a small number of perturbations (lower-order projections). To estimate robustness against at least three perturbations, TestifAI trains an auxiliary model on the results of tests involving up to two perturbations only, avoiding execution of an exponential number of tests. Our experiments on five image and language classification tasks show that TestifAI can predict higher-order (3 and 4 perturbations) test outcomes from low-order (1 and 2 perturbations) observations with an aggregate robustness estimation error of less than 7%, while reducing the number of inferences by 60-80%. |
| 2026-08-19 | [EVADE: Evidence-Verified Agentic Diagnosis with Escape](http://arxiv.org/abs/2608.18833v1) | Mohaimenul Azam Khan Raiaan, Nur Mohammad Fahad | Medical vision-language models (VLMs) can achieve high accuracy but remain unreliable: they are systematically overconfident, benefit little from test-time reasoning, and lack the ability to reliably calibrate trust in their own responses. We introduce EVADE (Evidence-Verified Agentic Diagnosis with Escape), an inferential, non-training method that enhances the safety of deploying a single frozen VLM. EVADE responds and, when uncertain, localises the region most diagnostically relevant, re-answers on a zoomed view, and commits only when both the entire image and the zoomed view responses agree; otherwise, it abstains. To directly address verification hallucination in single-model self-checking, our main idea is to verify gate consistency across different image views rather than re-reading the model's own text. Experimental evaluation on VQA-RAD, SLAKE, and PathVQA using Qwen2.5-VL-7B reports that EVADE is the only method that simultaneously improves both calibration and selective risk while maintaining accuracy, reducing expected calibration error (ECE) by up to 45% compared to zero-shot. Chain-of-thought, self-consistency, and self-verification all fail at least one axis. A grounding analysis reports that self-proposed regions perform better at diagnostic structure localisation than centres or random crops. However, a 7B VLM cannot use this localisation to revise answers. Therefore, reliability gains come from the consistency gate and calibrated abstention. |
| 2026-08-19 | [A revised framework for the assessment of psychological safety in autonomous vehicles](http://arxiv.org/abs/2608.18801v1) | Yandika Sirgabsou, Benjamin Hardin et al. | Despite recent technological progress in the development of autonomous vehicles (AVs), their societal acceptability remains a subject of debate as recent research findings point to psychological roadblocks. Concerns arise not only for physical safety but also for potential psychological risks resulting from human interaction with AVs. Psychological concepts such as trust, and perceived safety are well-studied in this context and are found to be determinant factors for the intention to use AVs. Unfortunately, there has been no formalization of the mechanism by which human interaction with AVs may lead to psychological hazards, threatening trust, perceived safety, and acceptability. Furthermore, there has been little prior research that conceptualizes the severity of psychological risk in AVs, and there are no clear guidelines for a systems designer on how to assess and address psychological risk in the AV development context. To address these limitations, this paper extends a theoretical framework for AV psychological safety risk assessment based on an early proposal. The proposed framework consists of an extended risk model for psychological safety including all the key concepts related to psychological safety in AVs, and an assessment method based on the Systems-Theoretic Accident Model and Processes (STAMP). We demonstrate the usefulness of the theoretical framework through a highly automated AV use case scenario, uncovering factors which may lead to psychological risk for an occupant. The use cases provide examples of how to use the framework to extensively evaluate psychological risk and determine vehicle behaviour that could lead to this risk. By developing a theoretical framework for AV psychological safety risk assessment, we provide a foundation and a method that were previously lacking to better enable responsible AV development regarding psychological safety. |
| 2026-08-19 | [Engineering Psychological Safety in Autonomous Vehicles: A Systems-Theoretic Framework for Psychological Safety in Autonomous Vehicles and its Validation in Real-World Scenarios](http://arxiv.org/abs/2608.18778v1) | Yandika Sirgabsou, Benjamin Hardin et al. | Despite rapid technological advances, the societal acceptability of autonomous vehicles (AVs) remains limited by psychological barriers that extend beyond traditional concerns of physical safety. While factors such as trust and perceived safety are known to influence user acceptance, there is a lack of formalized mechanisms and engineering methods to systematically identify, assess, and mitigate psychological risks arising from human-AV interactions. To address this gap, this work proposes and validates a systems-theoretic framework for the assessment of psychological safety in autonomous vehicles. First, a comprehensive psychological safety risk model is defined, extending the Systems-Theoretic Accident Model and Processes (STAMP) to incorporate key psychological constructs such as trust, perceived control, predictability, and perceived support. Based on this model, a hazard analysis method (AV-PsySafe) is developed to systematically identify psychological hazards, unsafe control actions, and loss scenarios, while introducing a Psychological Safety Integrity Level (PsySIL) to support risk prioritization. Second, the applicability and relevance of the framework are evaluated through its deployment in realistic autonomous vehicle scenarios. A structured validation approach is implemented, including a methodological guide, standardized analysis templates, and the collection of analyst feedback. The results demonstrate that the framework can be consistently applied by practitioners, producing meaningful insights into psychological risks. Overall, this work establishes both the theoretical foundations and practical feasibility of a unified approach to co-assessing psychological and physical safety in autonomous systems, contributing to more human-centred and trustworthy AV development. |
| 2026-08-19 | [The Impact of CutMix on Reliability and Robustness in Semantic Segmentation](http://arxiv.org/abs/2608.18715v1) | Steven Landgraf, Markus Ulrich | Ensuring not only high accuracy but also reliable and robust predictions is critical for the deployment of semantic segmentation models in safety-critical applications such as autonomous driving. Despite the widespread use of CutMix - a simple yet powerful data augmentation strategy - its effect on the reliability and robustness in dense predictions tasks remains unexplored. Motivated by recent findings that semi-supervised segmentation methods, where CutMix is a core component, can severely degrade reliability, this study isolates and systematically analyzes the influence of CutMix on segmentation accuracy, calibration, and uncertainty quality. We evaluate two representative architectures, the CNN-based DeepLabV3+ and the transformer-based SegFormer, across both in-domain and out-of-domain scenarios. Our results show that CutMix has only a minor impact on segmentation accuracy but consistently improves the reliability, particularly under distribution shifts. These improvements indicate that CutMix primarily enhances the trustworthiness of the model's calibration and uncertainty rather than the raw segmentation prediction itself. This distinction is crucial for safety-critical deployment, where reliable confidence estimates are as important as raw performance. |
| 2026-08-19 | [A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation](http://arxiv.org/abs/2608.18709v1) | Steven Landgraf, Joceline Hinz et al. | Foundation models are increasingly breaking what seemed to be impossible not long ago by enabling unprecedented accuracy and cross-domain generalization. Yet their lack of interpretability, tendency to be overconfident, and sensitivity to real-world domain shifts pose critical challenges for safety- and mission-critical applications. Uncertainty quantification (UQ) offers a principled way to address these issues, but its integration into segmentation foundation models has yet to be explored. In this paper we present the first systematic evaluation of UQ methods applied to a foundation model for semantic segmentation. We fine-tune a lightweight DPT decoder on top of the pretrained SAM2 encoder to establish a simple yet competitive baseline and benchmark four representative UQ approaches - Monte Carlo Dropout, Deep Sub-Ensemble, Test-Time Augmentation, and Evidential Deep Learning - across Cityscapes, NYUv2, and two challenging out-of-domain settings. Our analysis compares segmentation accuracy, calibration, uncertainty quality, and inference time, revealing clear trade-offs between predictive performance, reliability, and computational cost. These results highlight both the promise and the current limitations of uncertainty-aware foundation models, pointing to the need for future work that jointly optimizes accuracy, robustness, and efficiency for real-world deployment. |
| 2026-08-19 | [Broadband Chiral Primordial Gravitational Waves from Constant-roll Inflation in Parity-violating Symmetric Teleparallel Gravity](http://arxiv.org/abs/2608.18649v1) | Xu Zhang, Chang Liu et al. | We investigate primordial gravitational waves (GWs) generated during constant-roll inflation in a parity-violating extension of symmetric teleparallel gravity. The parity-violating interactions leave the background evolution and linear scalar perturbations unchanged, while inducing velocity birefringence in the tensor sector. Consequently, one of the two circular polarization states undergoes tachyonic amplification, producing a strongly blue and nearly fully chiral tensor spectrum from the cosmic microwave background (CMB) to interferometer scales. We identify viable constant-roll parameter regions consistent with current CMB constraints and determine the largest coupling strength compatible with both CMB B-mode measurements and the LIGO-Virgo-KAGRA (LVK) O1--O4a bound on the stochastic GW background. The predicted CMB B-mode spectra may be detectable by LiteBIRD, while the enhanced high-frequency signal could be accessible to the LISA--Taiji network and LVK O5 run. The model also predicts nonvanishing TB and EB correlations. These results highlight the potential of combining CMB and multi-band GW observations to test parity-violating gravity during inflation. |
| 2026-08-19 | [PILOT Technical Report](http://arxiv.org/abs/2608.18637v1) | Jiuning Lin, Ruiquan Lan et al. | Existing agentic approaches for recommendation system optimization remain fundamentally reactive: they adjust parameters in response to observed metric changes but lack the ability to proactively design controlled experiments, personalize strategies at the user-segment level, or accumulate reusable experimental methodology across tasks. We present PILOT (Proactive Insight Learner for Online Tree-Experiments), an LLM-agent framework that organizes three roles within a constrained control loop where deterministic services enforce all safety, statistical, and permission boundaries: (1) an Experiment Manager that drives the full experiment lifecycle -- task intake, observation governance, anomaly recovery, and postmortem -- by selecting only from a rule-generated legal-command envelope; (2) a Search Planner that proposes candidate decision trees for user-segment-level personalization, invoked only when the Manager requests planning; and (3) a Memory Curator that asynchronously distills experiment outcomes into strategy-level domain knowledge and provenance-tracked methodology, failure-isolated from the main loop. The Manager makes the agent proactive, the Planner enables population-level personalization beyond global tuning, and the Curator turns every completed task into a learning opportunity for the next. Deployed on Taobao's platform with 5 experimental buckets, PILOT is compared against ROAM(Reactive Optimization with Agent-driven Moves), a free-exploration agent without lifecycle governance or structured hypothesis testing. PILOT achieves up to +1.40% IPV, +1.60% Core IPV, +0.96% transaction count, and +1.50% transaction amount, improving over ROAM's best results (+1.00% IPV, +0.90% Core IPV, +0.60% transaction count, +1.13% transaction amount) while raising search efficiency from 53.3% to 93.3% (+40 pp), with no human intervention throughout the experimental cycle. |
| 2026-08-19 | [When Safety Overrides Vision: Exploring Dynamics between Vision Influence and Safety Alignment in Vision-Language Models](http://arxiv.org/abs/2608.18628v1) | Mehak Gupta, Tanmoy Chakraborty | Aligned vision-language models (VLMs) are designed to balance grounded visual reasoning with safe generation behavior. However, we observe a striking phenomenon: under safety-constrained instruction, models frequently abstain from answering questions that remain correctly answerable under default instruction despite receiving identical image-question inputs. This raises a fundamental question: does safety alignment suppress perceptual grounding itself, or does visual evidence remain internally available while generation is redirected toward abstention? In this work, we investigate the internal decoding dynamics underlying safety-induced abstention in aligned VLMs. Across multiple architectures and multimodal benchmarks, we show that abstained generations remain consistently influenced by visual evidence throughout decoding, indicating that perceptual grounding is largely preserved despite refusal behavior. We further demonstrate that, although the representational organization of refusal differs substantially across architectures, safety-constrained instruction consistently alters late-stage hidden-state dynamics toward refusal-oriented decoding. Finally, through targeted activation-level interventions, we show that suppressing refusal-related representations reliably restores grounded answering behavior across models without retraining or modifying visual inputs. Together, these findings reveal a previously underexplored failure mode in aligned VLMs: safety alignment can override grounded visual expression even when perceptual evidence remains internally preserved. |
| 2026-08-19 | [Finality Before Disclosure for Ledger Authenticators in the Quantum Random Oracle Model](http://arxiv.org/abs/2608.18605v1) | Maja Lie, Benjamin Marsh | Public ledgers increasingly authorize state transitions using prior transactions, finalized state, timing, and ordering rather than only a public key, message, and portable signature. We introduce ledger authenticators and $\LAEUF$, an unforgeability experiment for reactive authorization protocols whose public judgment algorithm reads a finalized transcript. The model separates authentication safety from ledger liveness and captures canonical transition freshness, adaptive corruption, exposure before inclusion, censorship, and adversarial ordering. We identify two conditional resource boundaries. An authenticator satisfying our single event conditions yields a contextual one-time signature. Within our rebindable reveal class, safety requires computational post-disclosure non-admissibility. When precursor admission uses only public computation and ledger scheduling, this condition is enforced by closing the evidence eligible to use a disclosed credential. If newly constructed evidence remains admissible after disclosure, censoring the honest reveal gives a forgery. We then define a joint ledger and quantum random oracle execution model in which quantum state persists across classical finalization cuts and oracle evaluations made through the ledger are charged. For a closed finalized target set of size at most $K$, we prove the bound $3β_{\mathsf{cut}}^2+3c_{\mathsf{co}}KQ^2/2^λ+6\ell/2^λ$, where $β_{\mathsf{cut}}$ accounts for fresh openings already present at the cut. A commit, close, reveal authenticator instantiates the framework and obtains a multi-user lifetime QROM bound. |
| 2026-08-19 | [PATE-Forensics: Perception-as-Tool for Explainable Deepfake Forensics with General-Purpose MLLMs](http://arxiv.org/abs/2608.18573v1) | Yaqi Li, Jielun Peng et al. | Existing explainable deepfake forensic methods typically rely on task-adapted MLLM to jointly address detection, localization, and explanation. Inspired by agent-style tool use, we instead introduce a Perception-as-Tool paradigm and instantiate it as PATE-Forensics, which architecturally decouples detection and localization from explanation generation while coupling detection and localization as tightly as possible within a forensic perception tool. The DINOv3-based tool couples a multi-granularity detection module that integrates global, patch-level, and segment-level evidence with a cue-guided localization module by spatializing the patch-level and segment-level evidence into forgery score maps that guide dense mask prediction. The original image and forensic perception outputs produced by the tool form structured forensic context for a general-purpose MLLM, which is guided by prompt constraints to generate explanations without task-specific fine-tuning. On DDL-X Track 3, PATE-Forensics achieves the best official score of 0.89, outperforming the second-ranked team by 0.19 points. Our code is available at https://github.com/yqli00000/PATE-Forensics. |
| 2026-08-19 | [Reducing Technician Search Burden: A Multimodal RAG for Cessna 172 Maintenance Manual](http://arxiv.org/abs/2608.18465v1) | Seongjun Ha, Md Rashedul Islam et al. | Proper use of the aircraft maintenance manual is essential for correct maintenance, providing procedures, diagrams, cautions, and specifications. However, technicians often avoid consulting it because it is difficult to navigate and time-consuming under strict schedules. Retrieval augmented generation (RAG) models have recently been introduced in aircraft maintenance, yet existing models focus solely on textual retrieval. This research therefore targeted the Cessna 172 Maintenance Manual (C172-MM), widely used in general aviation, and developed a multimodal manual retriever (MMR) capable of retrieving multimodal manual pages. Retrieval performance was evaluated using synthetic queries covering procedures, diagrams, caution/safety information, and specifications; the MMR achieved 93.37% recall@5. Beyond retrieval, a multimodal RAG (MRAG) pipeline was examined, in which retrieved pages were input to a vision-language model that generated responses to the synthetic queries, achieving 87.20% semantic similarity to ground-truth answers. Three practical feasibilities were also assessed: inference time, operational cost, and interpretability. Average retrieval time for five pages was 11.93 seconds and response generation took 4.95 seconds, at $0.0091 per query, while interpretability was validated through heatmap visualizations. These results indicate that the MRAG pipeline for the C172-MM can reduce the time technicians spend searching manuals and retrieving multimodal information. |
| 2026-08-19 | [When Clean Signals Are Not Enough: Detecting Structural Ambiguity for Safe Wearable Stress Classification](http://arxiv.org/abs/2608.18397v1) | Saba A. Farahani, Hung Cao et al. | Wearable stress classifiers can achieve strong average performance while failing completely for a particular individual. On WESAD, a Random Forest reaches 93.0% mean accuracy yet yields F1 = 0 for Subject 14, whose cross-signal coupling weakens near stress onset. We call this structural ambiguity: individually plausible physiological channels form an inter-signal pattern that is poorly supported by the person's non-stress reference. We introduce the Individual Conformal Coupling Monitor (ICCM), a lightweight and transparent pre-inference monitor that quantifies subject-specific coupling divergence and routes each window to classify, defer, or abstain without retraining the downstream classifier. Across WESAD (N = 15) and Stress-Predict (N = 35), full-cohort Pearson associations between ambiguity and accuracy are negative (r = -0.607, p = 0.016; r = -0.412, p = 0.014). Robustness analyses temper this finding: rank correlations are not significant, and the WESAD association disappears when Subject 14 is removed. ICCM changes false-positive counts from 29 to 27 and 94 to 92, although neither paired change is significant. It withholds 3 of Subject 14's 21 stress windows but does not repair the missed-stress failure. These results position ICCM as an interpretable signal of unsupported physiology and individual failure, rather than a stand-alone safety guarantee. |

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



