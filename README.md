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
| 2026-05-25 | [AnyScene: Towards Highly Controllable Driving Scene Generation at Anywhere and Beyond](http://arxiv.org/abs/2605.26113v1) | Haiming Zhang, Junfei Zhou et al. | Generating high-fidelity and controllable synthetic data is critical for advancing end-to-end autonomous driving, particularly for addressing the long tail of rare safety-critical scenarios. Existing occupancy-guided methods typically rely on shallow conditioning mechanisms and reference-frame-dependent video synthesis, which limits fine-grained controllability from arbitrary BEV layouts and restricts their applicability for scalable simulation. In this paper, we propose AnyScene, a unified occupancy-centric framework for driving scene generation. AnyScene generates semantic occupancy sequences from BEV layouts through a Spatial-Temporal Occupancy Diffusion Transformer that jointly tokenizes BEV and occupancy features in an autoregressive manner. This design enables precise controllability from cross-dataset and user-defined BEV inputs while naturally supporting long-horizon generation. Building upon the generated occupancy, a Geometry-Grounded View Expansion module treats occupancy as the canonical spatial representation and synthesizes temporally consistent multi-view driving videos in a reference-free and autoregressive fashion, supporting flexible camera configurations at inference time. Extensive experiments demonstrate that AnyScene achieves state-of-the-art performance in both occupancy and video generation. It exhibits strong generalization to unseen and customized layouts, and provides measurable benefits for downstream tasks such as sparse-view 3D reconstruction. |
| 2026-05-25 | [Retrying vs Resampling in AI Control](http://arxiv.org/abs/2605.26047v1) | James Lucassen, Adam Kaufman | AI coding scaffolds like Claude Code and Codex use \textit{retrying}: blocking actions flagged as risky and continuing the trajectory. We study retrying from an AI control perspective, which treats the model as potentially adversarial. We find that while retrying reduces honest suspicion scores, the untrusted model can exploit monitor rationale to construct sneakier attacks, negating safety gains. We also study \textit{resampling}: drawing multiple samples from the same context, which does not leak exploitable information. We disentangle design choices that previous work on resampling had bundled together. In BashArena, with Claude Opus 4.6 as the untrusted model and MiMo-V2-Flash as the trusted monitor, drawing five samples per step and auditing on the maximum suspicion score raises safety from 61\% to 71\% at a 0.3\% audit budget, at no cost to usefulness. Selectively resampling only the steps that look suspicious on the first draw recovers 6.2 percentage points of the gain while drawing only 10\% as many extra samples. Two of our findings in this setting contradict earlier work on resampling. The first is that auditing based on the maximum across resampled suspicion scores outperforms using the minimum, which is the opposite of what Ctrl-Z found. The second is that executing the least suspicious sample, which is the central mechanism in earlier defer-to-resample protocols, gives only a small empirical safety gain in our setting (+3.9 pp, with the confidence interval overlapping zero). |
| 2026-05-25 | [SafeCtrl-RL: Inference-Time Adaptive Behaviour Control for LLM Dialogue via RL-Driven Prompt Optimisation](http://arxiv.org/abs/2605.25984v1) | Michael Orme, Yanchao Yu et al. | Ensuring safe and contextually appropriate behaviour in Large Language Models (LLMs) remains a critical challenge for real-world deployment. We present \textbf{SafeCtrl-RL}, an inference-time behavioural control framework that enables adaptive safety regulation without model retraining or parameter modification. The method formulates dialogue generation as a sequential decision process, where a reinforcement learning agent dynamically selects prompt adjustment strategies based on contextual feedback. This allows unsafe behaviours to be suppressed through iterative refinement, which we conceptualise as inference-time behavioural unlearning. Evaluated across multiple LLMs and unsafe dialogue scenarios, SafeCtrl-RL consistently improves safety and response quality, outperforms existing prompt-based optimisation methods, and achieves favourable performance--efficiency trade-offs. **Warning: This paper may contain examples of harmful language, and reader discretion is recommended. |
| 2026-05-25 | [DetMesh-Gadep: Triangulated Surface Modeling and GPU-based Monte Carlo Efficiency Calibration of High-Purity Germanium Detectors](http://arxiv.org/abs/2605.25963v1) | Kainan Zhang, Shuchang Yan et al. | Sourceless efficiency calibration of high-purity germanium (HPGe) detectors can provide accurate detector-response information without experiments using radioactive calibration sources, offering advantages in both convenience and safety. In many practical implementations, this process is performed using Monte Carlo simulation; however, its performance is constrained by the accuracy of detector modeling, the operational complexity of simulation frameworks, and the computational-resource requirements associated with CPU-based parallelization. In this study, a complete detector modeling and simulation framework is proposed. The detector modeling program DetMesh can generate triangulated surface geometry from parameterized detector models, providing advantages in the representation of complex geometric boundaries. It incorporates standard geometric operations and a geometric library, and is lightweight with strong extensibility. The generated geometry is then input into Gadep, a GPU-based Monte Carlo computational kernel, to enable rapid simulation. For $1\times 10^8$ particles, a single RTX 4090 achieved a speedup factor of 13.53 compared with simultaneous computation using 60 CPU cores. The proposed framework has low implementation cost and broad applicability, providing a complete solution for refined detector modeling and calibration. |
| 2026-05-25 | [RAPTOR+: A Visually Grounded Vision-Language Framework to Improve Clinical Trust and Auditability in Automated Cancer Referral Processing](http://arxiv.org/abs/2605.25956v1) | Sofiat Abioye, Ufaq Khan et al. | Urgent suspected colorectal cancer (CRC) referrals create operational bottlenecks because semi-structured clinical documents often require manual review and transcription. The original RAPTOR system used Large Language Models for structured extraction but relied on a separate OCR stage, making it vulnerable to handwriting, layout variation, and loss of visual evidence linkage. We present RAPTOR+, a multimodal extension that uses Vision-Language Models (VLMs) for end-to-end referral understanding. We evaluate fine-tuned VLMs, commercial and open-source zero-shot VLMs, and the original OCR-based pipeline on 223 clinically curated CRC urgent referral forms. We also introduce a grounding-aware evaluation framework that measures both extraction accuracy and evidence localisation. Results show a clear grounding gap in zero-shot models. Gemini 2.5 Flash achieved 92.6% Reading Accuracy but only 1.2% Strict Safety. In contrast, fine-tuned Qwen3-VL-8B achieved 96.1% Reading Accuracy and 60.6% Strict Safety, substantially improving verifiable evidence grounding. These findings show that task-specific fine-tuning is essential for reliable, auditable clinical document understanding. RAPTOR+ enables extracted referral decisions to be linked to visual evidence, supporting safer and more efficient cancer referral triage. |
| 2026-05-25 | [A Pedestrian-Vehicle Interaction Benchmark and Annotation Framework for Unstructured Scenes via Uncalibrated Cameras](http://arxiv.org/abs/2605.25947v1) | Haoyang Peng, Qian Hu et al. | Predicting the interaction between pedestrian and vehicle is essential for autonomous driving safety in unstructured and semi-structured scenarios; however, this task is severely hindered by the scarcity of public datasets that feature dense pedestrian-vehicle interactions. Most current studies rely on structured road data, leaving the complex, heterogeneous interactions found in unstructured environments insufficiently represented and researched. In this paper, we propose a dataset annotation framework based on video data from uncalibrated surveillance cameras and present PINNS (Pedestrian-vehicle Interaction dataset from uNcalibrated cameras in uNstructured Scenes). The dataset covers multiple countries and regions, includes diverse typical traffic scenarios, and considers variations in seasons, lighting conditions, and weather. It focuses on complex scenes with dense pedestrian-vehicle interactions and is designed to be easily extensible. The dataset is constructed and annotated according to the standard issued by the Chinese Association of Automation, providing both trajectory data and corresponding scene-level information. Furthermore, this paper analyzes current challenges and research directions in heterogeneous agent trajectory prediction, shows the necessity and usefulness of the proposed dataset. We hope our framework and dataset will facilitate research on trajectory prediction and autonomous driving in complex mixed traffic scenarios. PINNS is publicly available at https://github.com/Songan-Lab. |
| 2026-05-25 | [$D^2$-Monitor: Dynamic Safety Monitoring for Diffusion LLMs via Hesitation-Aware Routing](http://arxiv.org/abs/2605.25893v1) | Aoxi Liu, Yupeng Chen et al. | Despite the emergence of diffusion large language models (D-LLMs) as an alternative to autoregressive large language models (AR-LLMs), safety monitoring for D-LLMs remains largely unexplored. Unlike AR-LLMs, D-LLMs generate text through a multi-step denoising process, exposing intermediate hidden representations that may contain safety-relevant information unavailable in standard single-step monitoring setups. Motivated by the suitability of lightweight probes for always-on monitoring, we analyze which trajectory-level signals best indicate when such probes are likely to struggle. We find that the most informative signal is safety hesitation: intermediate hidden states repeatedly falling within a small margin of the probe's decision boundary. The number of such hesitation steps in D-LLM's trajectory predicts probe failure effectively, providing a proxy of sample difficulty. Building on this analysis, we propose $D^2$-Monitor, a bi-level safety monitor for D-LLMs. $D^2$-Monitor adopts a lightweight probe as an always-on monitor to jointly estimate hesitation and perform base classification. When the hesitation level exceeds a threshold, a more expressive but computationally heavier probe is activated. This dynamic routing mechanism allocates monitoring resources efficiently at test time. Evaluated on 3 datasets (WildguardMix, ToxicChat, OpenAI-Moderation) across 4 D-LLMs, $D^2$-Monitor achieves state-of-the-art performance with a compact parameter footprint ($\leq$ 0.85M parameters), and exhibits the best trade-off between effectiveness and efficiency relative to 8 baselines. |
| 2026-05-25 | [Capability and Robustness Cannot Both Be Free: An Information-Theoretic Bound for Vision-Language-Action Models](http://arxiv.org/abs/2605.25889v1) | Jianwei Tai | Vision-Language-Action (VLA) models are increasingly deployed on real robots, where each predicted action is executed and each failure carries a safety cost. They reach high success rates on clean inputs but collapse under small adversarial perturbations. A $16/255$ PGD attack on OpenVLA-7B drops LIBERO success from above $95\%$ to under $5\%$. Empirical defenses recover some robustness at a cost in clean accuracy, but the literature does not say whether the trade-off has a theoretical floor. We prove that it does. For any VLA policy with discrete actions, the sum of capability (mutual information between policy action and oracle action) and robustness (mutual information preserved under adversarial perturbation, net of trivial channel leakage) is upper-bounded by a policy-independent budget: task entropy plus adversarial channel capacity. The proof is two applications of the Data Processing Inequality plus MI non-negativity. The pixel-level bound is loose on current models ($\sim 10^3$ nats), but an encoder-specific corollary restricts the channel to the policy-relevant subspace, reducing the budget from $\sim 5{,}000$ to $\sim 31$ nats on OpenVLA; the policy already consumes $\sim 24\%$ of this tighter budget, leaving limited room for simultaneous robustness improvement. We validate the bound across $252$ closed-form Gaussian-VLA cells and $48$ OpenVLA-7B $\times$ LIBERO $\times$ PGD cells (zero violations). We propose encoder-specific slack as a normalized comparison axis for defense papers, and release all code, manifests, and results. |
| 2026-05-25 | [Proof of Useful Attestation: A Consensus Primitive for Attestation-Native Chains](http://arxiv.org/abs/2605.25844v1) | Stefan Stefanović | Validators on generic Proof of Stake chains earn the same fees whether they handle attestation work correctly or selectively censor it. For chains whose main activity is moving tokens around, that indifference is fine. For chains whose primary economic activity is recording attestations (content provenance, AI-output attribution, threshold-signed credentials, supply-chain receipts), the indifference becomes a problem.   Proof of Useful Attestation (PoUA) makes attestation handling first-class in the consensus weighting itself. Validator vote weight is the product of bonded stake and a reputation scalar in [r_min, r_max] that accumulates from valid attestation work. The reputation update is additive, fee-weighted, non-transferable, and capped per epoch. We prove a cost-to-grind floor (Lemma 1): under chain-wide adaptive burn fraction tau_burn, the non-recoverable cost an adversary pays to inflate reputation by Delta_r is bounded below by tau_burn * Delta_r / (eta * alpha_eff). Under the recommended v0 calibration (r_max/r_min in [4, 10]), the cost premium against a capital adversary is 4x to 10x over equivalent pure-stake PoS at steady state.   The paper specifies the mechanism, six layered Sybil and grinding defenses, empirical Monte Carlo strategy-search across the full layered defense, and grinding detectors with explicit threshold derivations. It is a mechanism-design proposal with a formal economic floor and inherited BFT safety and liveness, not a complete cryptographic security proof. This release incorporates feedback from Jiangshan Yu (University of Sydney) and Marko Vukolić (Bitcoin Scaling Labs). |
| 2026-05-25 | [An Analysis Focused on Womens Safety: Can VAD Models Be Enhanced by a Multi-modal Dataset?](http://arxiv.org/abs/2605.25806v1) |  Sangeeta, Maddikuntla Sai Prajwal et al. | Women's safety and security are paramount for a modern society. Crimes against women occur in daylight as well as in low-light conditions. Often, such events are captured through real-world surveillance cameras that operate at lower resolutions. Despite substantial progress in CV-related research, video anomaly detection (VAD) focused on women's safety has not yet been adequately addressed. Existing video anomaly datasets contain well-lit, high-resolution, close-shot videos, and fail to represent women-centric anomalies such as chain snatching, stalking, inappropriate touch, and other subtle forms of crime against women. To address these problems, we propose the ExtrAnom dataset, a new multi-modal benchmark containing 1001 videos with textual descriptions, 500 normal and 501 anomalous, classified into 5 different types of women-centric crimes. The dataset comprises low-light (8%), low-resolution videos (13%), long-shot (15%), along with daylight (64%) anomalous videos. And it covers anomalous events like stalking (3.9%), chain snatching (17.6%), kidnapping (7.3%), assassinations (2.3%), harassment (18.9%), and normal (50%). Each video is supplemented with 4 textual annotations, including one human-generated and three LLM-generated descriptions, enabling cross-modal and VLM-based validations. The aim of creating a women-centric dataset is to accurately detect the women-centric anomaly patterns, which are possible to observe visually. The dataset supplements the VLMs to accurately generate video-level descriptions. ExtrAnom has been benchmarked against popular unimodal and multi-modal VAD datasets (e.g., XD-Violence, UCF-Crime, and UCA) and SOTA methods. Experiments reveal that the existing datasets are insufficient to train models for detecting women-centric anomalies. |
| 2026-05-25 | [HoLoArm: Deformable Arms for Collision-Tolerant Quadrotor Flight](http://arxiv.org/abs/2605.25790v1) | Quang Ngoc Pham, Jonas Eschmann et al. | The increasing use of drones in human-centric applications highlights the need for designs that can survive collisions and recover rapidly, minimizing risks to both humans and the environment. We present HoLoArm, a quadrotor with compliant arms inspired by the nodus structure of dragonfly wings. This design provides natural flexibility and resilience while preserving flight stability, which is further reinforced by the integration of a Reinforcement Learning (RL) control policy that enhances both recovery and hovering performance. Experimental results demonstrate that HoLoArm can passively deform in any direction, including axial one, and recover within 0.3-0.6 s depending on the direction and level of the impact. The drone can survive collisions at speeds up to 7.6 m/s and carry a 540 g payload while maintaining stable flight. This work contributes to the morphological design of soft aerial robots with high agility and reliable safety, enabling operation in cluttered and human shared environments, and lays the groundwork for future fully soft drones that integrate compliant structures with intelligent control. |
| 2026-05-25 | [Polarized Anisotropic Stochastic Gravitational Wave Background Search with Ground-Based Detector Networks](http://arxiv.org/abs/2605.25772v1) | Töre Boybeyi, Vuk Mandic | Gravitational waves admit a Stokes decomposition into intensity ($I$), circular polarization ($V$), and linear polarization ($Q$, $U$), analogous to Cosmic Microwave Background (CMB) polarimetry. We implement a full-Stokes maximum-likelihood SGWB map-making analysis for ground-based detector networks, promoting the standard cross-correlation data products used in existing pipelines to a joint reconstruction of $I$, $V$, $Q$, $U$. Applied to LVK O3 data, we constrain the polarized angular spectra $C^{VV}_\ell$, $C^{EE}_\ell$, $C^{BB}_\ell$ and $|C^{IV}_\ell|$. We show that an intensity-only model is biased when polarized sky components are present, since the detector-network Fisher inner product does not generally make the Stokes responses orthogonal. For transient CBC foregrounds, polarized shot noise is not parametrically suppressed relative to ordinary CBC intensity shot noise. The full Stokes framework separates the Stokes sectors while providing access to polarized anisotropies invisible to conventional intensity-only searches. |
| 2026-05-25 | [HumanFlow -- Diffusion-Driven MAV Navigation Among Humans via Tightly-Coupled Motion Tracking, Forecasting, and Control](http://arxiv.org/abs/2605.25685v1) | Simon Schaefer, Joshua Näf et al. | Robust and accurate perception of humans in their 3D scene context is essential for integrating robots into everyday environments. Existing approaches, however, often fail to predict plausible and accurate human motion estimates that are consistent with the surrounding scene, especially in the presence of heavy occlusions or partial visibility. This can limit both safety and efficiency for robotic operations. We introduce HumanFlow, a latent diffusion model that unifies human motion tracking and forecasting, conditioned on the 3D scene context. We show that our human motion model produces smooth and accurate predictions under challenging conditions, including heavy occlusions, and outperforms state-of-the-art methods in tracking accuracy while being significantly more efficient. Furthermore, we show how HumanFlow's latent space can be tightly coupled with control by conditioning a flow-matching-based, approximate MPC policy on these representations. We validate our policy in simulation with real human trajectories for MAV social navigation, demonstrating superior navigation performance and remaining collision-free, even under partial observability of the human. |
| 2026-05-25 | [Referential Security as a New Paradigm for AI Evaluations](http://arxiv.org/abs/2605.25673v1) | Dan Ristea, Vasilios Mavroudis | Security evaluations inherently depend on stable identifiers. Any finding, audit, or regulatory decision must remain attached to the specific artifact it pertains to. Continuously updated artificial intelligence systems violate this core assumption, with public model designations remaining static while underlying weights, prompts, retrieval mechanisms, misuse classifiers, inference settings, and serving infrastructures undergo unannounced modifications. Consequently, current evaluations frequently apply to superficial labels rather than identifiable and distinct systems.   To resolve this, we propose referential security as a new paradigm for AI evaluation. The fundamental security question extends beyond whether a model is safe to whether subsequent parties can conclusively determine which system a specific safety claim addressed. This approach reframes model identity as an empirically verifiable property and separates referential stability from the substantive security claims it conditions. This framework brings tractability to three critical workflows that current practices handle poorly. Specifically, it enables reproducible evaluation, longitudinal audit validity, and cross-provider equivalence. By grounding these evaluations in verifiable artifacts, our approach ensures that safety audits and regulatory findings maintain their empirical utility across the operational lifecycle of dynamic systems. |
| 2026-05-25 | [Justice-informed Planning of Intermodal Autonomous Mobility-on-Demand Systems under Operational Constraints](http://arxiv.org/abs/2605.25617v1) | Giacomo Ganassoli, Francesco Mazzeo et al. | To date, most of the research on transport planning has focused on optimizing revenues or utilitarian metrics such as average travel times, which often ends up penalizing the worst-off for the sake of profit or efficiency. At the same time, most of the research in transport justice has focused on assessing injustices, without being able to prescribe operational solutions. This paper contributes to bridging this gap and presents optimization models for justice-informed operational planning of intermodal mobility systems that explicitly account for the budget and safety limitations of users, and for infrastructural capacity constraints. Specifically, we first focus on an intermodal Autonomous Mobility-on-Demand (AMoD) system -- where self-driving robotaxis provide on-demand mobility jointly with public transit and active modes -- and characterize its operations from a mesoscopic planning perspective via network flow models. Second, we leverage these models to optimize system operations through both utilitarian efficiency and justice-informed objectives. We showcase our framework in a real-world case-study for Manhattan, New York. Our results show that monetary budgets significantly limit the social justice potential of AMoD systems if they are to be deployed as transportation network companies. At the same time, granting free public transit can result in sufficiency levels very close to a completely free intermodal AMoD system, where justice-informed operations can be achieved without compromising standard efficiency metrics, ultimately highlighting the strong potential of social policies. |
| 2026-05-25 | [Safety-Critical Whole-Body Control for Humanoid Robots via Input-to-State Safe Control Barrier Functions](http://arxiv.org/abs/2605.25546v1) | Kwanwoo Lee, Sanghyuk Park et al. | Safety-critical control is essential for humanoid robots operating in complex human-centered environments, where physical safety constraints such as joint limits, self-collision avoidance, obstacle avoidance, and workspace boundaries must be satisfied during real-robot operation. However, existing approaches remain limited because kinematic safety guarantees can be degraded in the presence of unknown disturbances, such as model uncertainties, trajectory-tracking errors, and external perturbations. This paper presents a hierarchical safety-critical whole-body control framework for humanoid robots based on input-to-state safe control barrier functions (ISSf-CBFs). The proposed architecture integrates a kinematic-level whole-body controller (KinWBC), an ISSf-CBF safety filter, and a dynamic-level whole-body controller (DynWBC). KinWBC generates nominal joint-motion references from prioritized tasks; the ISSf-CBF filter minimally modifies these references to satisfy kinematic safety constraints under bounded disturbances; and DynWBC tracks the filtered references while enforcing full-body dynamic feasibility and contact stability. Safety constraints are imposed on a whole-body kinematic model, and the ISSf-CBF parameters are conservatively tuned so that the resulting kinematic safety guarantees can be transferred to full-order humanoid dynamics under unknown disturbances. Simulation and real-robot experiments demonstrate that the proposed framework improves safety margins under model mismatch and reliably enforces multiple safety constraints in real time during locomotion, teleoperation, and single-leg balancing with hand control. Project website: https://kwlee365.github.io/SafeWBC-Website/ |
| 2026-05-25 | [StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs](http://arxiv.org/abs/2605.25534v1) | Yang Luo, Xinran Liu et al. | Multimodal Large Language Models (MLLMs) excel at structural reasoning yet suffer from a sharp logical brittleness in structural consistency. We term this phenomenon Structural Cognitive Overload (SCO), a byproduct of the contention between deep reasoning and safety alignment. However, prior work has predominantly targeted typographic and pixel-level perturbations, leaving the study of SCO largely unexplored. To this end, we propose StructBreak, an automated end-to-end framework designed to quantify SCO. By leveraging StructBreak, we uncover a novel higher-order cognitive overload attack paradigm; notably, this attack operates under a practical black-box setting, requiring no internal model access. Consequently, we utilize this framework to establish a comprehensive benchmark spanning ten diverse threat scenarios. Empirical evaluations on six leading MLLMs reveal that SCO readily triggers toxic generation, yielding a 92% average ASR (up to 97% on Gemini 2.5). To elucidate the mechanism of SCO, we further conduct model-level interpretations spanning attention dynamics, latent space topology, and geometric analysis. Our findings reveal that StructBreak acts as a novel structural channel to circumvent safety filters. Furthermore, the limited efficacy of inherent safety mechanisms underscores that current alignment paradigms are insufficient for the era of complex multimodal reasoning. |
| 2026-05-25 | [The Age of Curiosity Meets the Age of AI: Benchmarking Child Safety in Large Language Models](http://arxiv.org/abs/2605.25510v1) | Samee Arif, Angana Borah et al. | Children increasingly have access to Large Language Models (LLMs), which may expose them to responses that are developmentally inappropriate or require age-sensitive safety, guidance, and boundaries. Existing LLM safety evaluations largely focus on harmful-content avoidance and do not explicitly target child-facing safety. We introduce KIDBench, a benchmark for evaluating child-facing LLM safety for ages 7--11 using a developmental-psychology-grounded LLM-as-a-Judge rubric. KIDBench contains realistic child queries across ten categories, with single-turn prompts and multi-turn child-actor simulations. We compare no-cues prompts with no child context, implicit-cues prompts that suggest a child speaker, and explicit age instructions. Implicit-cues improve scores by 9--47% across models, while explicit age adds a further 10--30% gain. Cross-lingual and cultural evaluations show uneven safety behavior across languages and country contexts. Multi-turn simulations show that child-facing response quality can degrade by 6--24% from the first to worst turn. Beyond evaluation, we introduce KIDGuardLlama, a child-safety evaluator, and KIDLlama, a child-oriented response model, showing how KIDBench supports safer child-facing AI |
| 2026-05-25 | [AI Content Moderation in Therapy Conversations](http://arxiv.org/abs/2605.25454v1) | Jiwon Kim, Claire Wang et al. | Large language models (LLMs) are increasingly being used for emotional support. They are also being developed for formal therapy purposes. However, LLMs like ChaptGPT or Llama are often developed with content moderation guardrails that prevent them from discussing sensitive subjects with users for both liability and safety purposes, and this inability to broach these subjects may affect their capacity as therapists. In this study, we perform an algorithm audit on three state-of-the-art moderation systems (OpenAI's moderation endpoint, Meta's Llama Guard, and Google's Shield Gemma) to investigate the extent to which these systems flag the content of real-life therapy sessions as undesirable. Our results raise implications for the limitations that users and organizations may encounter when designing LLMs to play the part of a therapist. |
| 2026-05-25 | [Mode 0: A New 3GPP V2X Resource Allocation Category for Roadside Computing Unit-Assisted Safety Communication](http://arxiv.org/abs/2605.25431v1) | Dewei Jiang, Xiang Gu | The 3GPP V2X resource allocation framework defines two entity classes -- the base station and the vehicle UE -- and four modes across LTE and NR generations. We demonstrate that this binary taxonomy is structurally incomplete. Base station-led scheduling saturates at high-density traffic nodes, producing latency-tail failures that persist even when mean packet delivery ratios approach the service-class target. UE autonomy is categorically incapable of pre-emergence warning for occluded traffic participants and insufficient for large-scope cascading environmental hazards. We propose Mode 0, a new 3GPP V2X category whose defining entity is the Roadside Computing Unit (RCU) -- an infrastructure ensemble integrating elevated sensing (Seeing), sidelink communication (Speaking), and local computational evaluation (Thinking), owned by traffic management authorities. Mode 0 defines a subfamily spectrum from Mode 0a (all-passive UEs, the guaranteed minimum) through Mode 0c (all-active UEs, the optimal target). Convergent deployment evidence from Chinese national standards (DB11/T 2329.1-2024, T/ITS 0224.1-2025), China Unicom RS-MEC infrastructure, and European and US C-V2X programs confirms that both institutional sides are converging on the roadside traffic node without a coordination standard. A fifteen-run Multi-Agent Proximal Policy Optimization (MAPPO) simulation validates the architectural family: Mode 0a in shared-pool baseline sits at the analytical symmetric-Nash coordination floor; Mode 0c with demand separation achieves strict Pareto improvement for both traffic classes (M0 PDR 0.999, M1 PDR 0.998 at $ρ_{\rm pool} \leq 1$) and lifts the worst-TTI delivery ratio from near-zero to 0.601 -- the only configuration satisfying the latency safety requirement structurally. We call for a 3GPP study item on Mode 0 within the NR-V2X sidelink enhancement work programme. |

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



