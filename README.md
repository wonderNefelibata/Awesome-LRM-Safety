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
| 2026-10-27 | [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](http://arxiv.org/abs/2610.37652v1) | Ian Chuang, Hiya Shah et al. | Dexterous manipulation with multi-fingered hands is the next frontier of robotics, yet learned hand skills remain limited to simple grasps or basic repeated motions, such as rotating an object. As robots attempt more demanding in-hand maneuvers, safety, such as not dropping the object, becomes critical, yet it is rarely measured. We study in-hand object pose reaching, a task that demands diverse and difficult grasps: moving an object to almost any position and orientation within reach of the fingers, at any wrist orientation, through long sequences of consecutive goals. We present a recipe for training such policies for a LEAP hand in simulation without human data, built on a range of insights, from diverse resets and goals via generated grasp datasets to an adaptive friction curriculum. With careful system identification, the policies transfer zero-shot to the real hand, using either motion capture or a single wrist-mounted depth camera, and prove remarkably capable and robust. On top of the task policy, we add a learned residual safety filter with adjustable conservativeness, letting us tune the balance between performance and safety and greatly reduce object drops. In 13.3 hours of real-robot evaluation over 4,800 unseen goals across 4 objects and diverse wrist orientations, the state-based policy reaches 95.2% of goals within 2 cm and 0.4 rad, and the filter cuts its drops by 75% while still reaching 91.5% of goals. The policies can also spin a lightbulb held in an arbitrary pose, reach out-of-distribution goals, withstand fast arm motion and perturbations, and, needing only a wrist camera, serve as portable skills such as screwing in a lightbulb. We release the full recipe as a platform for studying performance, reliability, and safety in high-dimensional dexterous systems. Videos and more are available at https://ian-chuang.github.io/dexsafe/. |
| 2026-10-27 | [A Quiet Night in the Millihertz Sky? LISA Forecasts from LIGO/Virgo/KAGRA Populations](http://arxiv.org/abs/2610.37617v1) | Ignacio Magaña Hernandez, Antonella Palmese | The high-mass black hole population inferred from ground-based observations can substantially affect the number of binaries detectable by the Laser Interferometer Space Antenna (LISA). We use population constraints based on data from the latest LIGO/Virgo/KAGRA Gravitational-Wave Transient Catalog (GWTC-5.0), as well as constraints from the O3 search for intermediate-mass black hole binaries, to forecast the number and masses of binary black holes detectable by LISA. Using the released GWTC-5.0 population posterior, we obtain $6.6^{+5.5}_{-2.2}$ expected detections with signal-to-noise ratio $>8$ in 4 years (median and $90\%$ credible interval). The alternative population models we consider use a broken power law with two or three Gaussian peaks for either the primary-mass distribution or a common mass distribution for both black holes. For the three-peak model with the common mass distribution, extrapolating from $300$ to $1000M_\odot$ roughly doubles the median expected number of LISA detections to $\sim14$. The upper end of its $90\%$ credible interval rises from tens to thousands of expected detections, although approximations may limit the reliability of the most extreme counts. For the reference GWTC-5.0 population, $39^{+22}_{-11}\%$ of the binaries detectable by LISA merge during the mission or within $10$ years after its end and are potential multiband sources. Essentially all of these mergers would be detectable by a reference Cosmic Explorer configuration if it is operating at merger. Ground-based measurements of the high-mass population can therefore inform both LISA forecasts and the prospects for observing the same binaries through inspiral and merger. |
| 2026-10-27 | [GNBG-C: A Unified and Configurable Benchmark Generator for Constrained Continuous Optimization](http://arxiv.org/abs/2610.37590v1) | Ali Ghasemzadeh, Erfan Babakhani et al. | Many real-world optimization tasks are governed not only by what should be optimized, but also by what is allowed: a solution must first satisfy physical, operational, safety, budgetary, or structural requirements before its objective value becomes meaningful. Yet benchmark design for constrained continuous optimization often treats these restrictions as fixed properties of predefined test instances. Existing generators and benchmark suites provide essential foundations for reproducible comparison, but they either focus primarily on configurable objective landscapes or expose constraint properties within limited, predefined regimes, making it difficult to ask targeted questions about constraint-induced difficulty: whether an optimizer struggles because feasible regions are small, boundaries are restrictive, variables are tightly coupled, or constraint information is poorly scaled. We introduce GNBG-C, a Generalized Numerical Benchmark Generator for Constrained Continuous Optimization. The central idea is to treat the feasible region itself as a configurable object rather than as an inseparable attachment to a fixed test function. By generating controlled constraint regimes around the known GNBG global minimizer, the generator enables systematic studies in which feasibility pressure can be varied while preserving that point as a global minimizer of the constrained problem. This allows researchers to examine how optimizers discover feasibility, handle constraint pressure, and improve objective quality under diverse and controllable feasibility conditions. GNBG-C therefore provides a flexible and auditable framework for probing the strengths and weaknesses of constrained continuous optimizers under targeted forms of feasibility-side difficulty. |
| 2026-10-27 | [When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance](http://arxiv.org/abs/2610.37561v1) | Walter Endrizzi, Flavio Ragni et al. | Background. Selective prediction systems improve the safety of clinical artificial intelligence by deferring ambiguous cases to human experts while automating high-confidence decisions. However, this safety mechanism relies entirely on the premise that uncertainty scores accurately reflect clinical ambiguity. In prognostic tasks characterized by severe class imbalance, where the clinical outcome of interest is a minority, this assumption breaks down. Standard uncertainty measures cease to capture clinically aligned ambiguity; instead, they become a proxy for class membership, systematically penalizing minority patients regardless of their individual predictive clarity.   Results. Across 56 synthetic conditions and three neurodegenerative cohorts spanning severe class imbalance to near-balanced conditions (multiple sclerosis, Parkinson's Disease, Alzheimer's Disease), our framework consistently maintained bounded, balanced risk across trajectories. In contrast, global metrics either collapsed minority Sensitivity or protected the minority at the expense of majority patients. Applying CCR with standard entropy fails to protect the minority because it breaks within-class monotonicity when tau=0.5. Combining CCR with threshold-aware metrics restores strict monotonicity, correctly identifies ambiguous patients, and safely mitigates minority exclusion.   Conclusions. Under clinical class imbalance, standard uncertainty quantification measures dataset sparsity rather than clinical ambiguity, creating a systemic illusion of safety. This bias is a correctable structural flaw. Clinical pipelines deploying selective prediction under imbalance should adopt threshold-aligned uncertainty measures and class-stratified evaluation. This reduces class-specific risk imbalance, supporting safer selective prediction under the evaluated conditions without silently excluding the most vulnerable patients. |
| 2026-10-27 | [Leveraging Distributed Sensing and Hyperdimensional Computing for Non-Line-of-Sight Passive Object Sensing](http://arxiv.org/abs/2610.37519v1) | Placido Mursia, Marco Rossanese et al. | In smart factory environments, passive entities such as humans and movable obstacles may interfere with autonomous robot operations, posing significant safety risks. Detecting potential conflicts between robots and entities outside their field-of-view (FoV) is therefore of paramount importance. However, conventional passive object detection systems rely on sensors such as LiDAR, radar, or cameras, which operate under direct line-of-sight (LoS) and are limited by obstacles and occlusions. To overcome these limitations, integrated sensing and communication (ISAC) enables environmental perception via the communication infrastructure thanks to cooperation among distributed wireless devices. In this paper, we propose a framework for passive entity sensing beyond the robot's FoV that relies solely on distributed channel measurements from the communication infrastructure to perform risk-aware classification. These measurements are embedded into high-dimensional representations using hyperdimensional computing (HDC), enabling robust and scalable learning without explicit channel reconstruction. Numerical simulations based on a commercial ray tracer and ROS-based validation in realistic smart factory scenarios demonstrate improved safety and enhanced path planning with up to 89% accuracy under severe occlusions, while maintaining low computational complexity. |
| 2026-10-27 | [Follow the Gradient: Tracing Safety Degradation Back to Training Data](http://arxiv.org/abs/2610.37481v1) | Saket Reddy, Amro Abbas et al. | Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned models. We then use SimpleLoRIF to build SafeTulu, a version of Tulu 3 with the 20% of examples most gradient-aligned with unsafe responses removed. Training on SafeTulu reduces GCG attack success rate by up to 22% across Qwen, Llama, and Olmo, outperforming both LLM-as-a-judge and prior gradient-based filtering, while largely preserving or even improving capability. Finally, we analyzed the removed data and identify possibly under-studied sources of safety degradation, such as tool-execution failures and narrative arithmetic. We release SafeTulu along with a Hugging Face Space that lets researchers build their own customized versions of SafeTulu to fine-tune on, such as by changing the removal fraction or filtering only certain subsets. |
| 2026-10-27 | [When Visual Evidence Is Insufficient: How Instructions, Post-Training, and Internal Activations Shape Refusal in Multimodal Language Models](http://arxiv.org/abs/2610.37430v1) | Ananth K Suresh, Aveen Dayal et al. | For reliable deployment, multimodal large language models (MLLMs) should recognize when an image provides insufficient evidence to answer a question and withhold unsupported answers. Yet refusal alone is not necessarily evidence of such recognition: models may refuse because of unrelated safety constraints or without identifying the missing evidence. We study how refusal on visually unanswerable questions is shaped by three sources of control: inference-time instructions, post-training, and internal activation interventions. Across three MLLMs on the MoHoBench benchmark, we systematically manipulate the meaning and placement of refusal instructions, compare supervised and preference-based post-training, and intervene on a refusal-related activation direction derived from naturally answered and refused examples. We find that refusal is highly sensitive to both the meaning and placement of instructions, with model-specific changes at the sample level that are obscured by aggregate refusal rates. Evidence-aware and general refusal instructions have stronger effects than an intentionally mismatched safety instruction. Post-training produces larger increases in refusal, but can also induce responses that combine refusal language with attempted answers. In contrast, activation intervention produces smaller yet transferable changes across prompt conditions, without evidence of indiscriminate refusal on tested visually answerable samples. These findings show that refusal is jointly shaped by external instructions, post-training, and internal representations, highlighting the need to disentangle evidence recognition from a model's learned propensity to refuse. |
| 2026-10-27 | [VLA-Stem: A Lightweight Safety Mechanism for Vision-Language-Action Models via Reinforcement Learning](http://arxiv.org/abs/2610.37378v1) | Bojun Zou, Wenhao Yu et al. | Vision-Language-Action (VLA) models integrate high-level reasoning with low-level action execution for end-to-end robotic control. However, most VLA models target static manipulation in unconstrained environments, overlooking spatial constraints and dynamic obstacles. Fine-tuning VLAs on obstacle-avoidance data is a straightforward solution, but collecting such data is costly, especially for dynamic scenarios, and may degrade their original task capabilities. To address these limitations, we propose VLA-Stem, a lightweight, reinforcement-learning-based safety mechanism for VLA systems. Without fine-tuning the underlying VLA policy, VLA-Stem converts its action chunks into motion references and uses trajectory-level guidance to preserve their spatial intent while allowing flexible motion and timing adjustments. Conditioned on these references, robot proprioception, and local obstacle information, the Stem policy adaptively modifies execution at inference time, preserving the original task intent while enabling responsive adaptation to spatial constraints and dynamic obstacles. For robust and sample-efficient training, we combine teacher-student distillation with curriculum learning. We evaluate VLA-Stem on reference following, static spatial constraints, and dynamic obstacle avoidance. Across six constrained simulation settings, VLA-Stem achieves 52.8% average collision-free task success versus 23.7% for the strongest safety baseline, while preserving base-policy performance in obstacle-free settings; real-world experiments further show consistent collision reductions under static and dynamic obstacles. These results demonstrate that VLA-Stem couples high-level VLA task intent with rapid, reflex-like responses to immediate physical risks, providing a brainstem-inspired mechanism for robust robot execution. |
| 2026-10-27 | [Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories](http://arxiv.org/abs/2610.37369v1) | Zirui Ding, Xinxin Fan et al. | Large language models (LLMs) are usually evaluated by final-output correctness or safety, yet an endpoint label does not reveal how a failure develops throughout the reasoning process. We therefore ask whether ordinary reasoning failures and adversarially induced failures can be characterized using the same internal coordinates. Towards this end, we in this paper study a white-box diagnostic framework CauTrace to disclose the root-cause of why LLM fails from three fundamental inspections: Functional States, Propagation of Representation, Temporal-State Coordination and Reasoning Trajectory. First, seven functional states are creatively defined to represent the support and constraint processing in LLMs. Second, the propagation of representation change measures how the changes among the seven states persist across subsequent Transformer blocks. Third, temporal-state coordination and trajectory topology characterize when the commitment and verification occur relative to the support, and how the dominant state sequence evolves over the full inference process. Finally, we evaluate CauTrace on both clean reasoning failures and paired adversarial perturbations. Across Qwen and LLaMA-family models, the failed traces show weaker Attention-MLP agreement on support-state updates and less verification activity relative to commitment, while the exact representation-level signatures and timing shifts vary by architecture. Held-out repair and steering tests demonstrate consistent operational signals that were in line with our claims for certain models and specific settings, with aligned steering producing the predicted directional changes in process measures. Overall, CauTrace provides a unified diagnostic framework for analyzing failure dynamics through changes in support, commitment, and verification, which would shed light on the in-depth research on LLM security. |
| 2026-10-27 | [Probabilistic Sequential Safety for Stochastic Dynamical Systems](http://arxiv.org/abs/2610.37311v1) | Abhijit Mazumdar, Manuela L. Bujorianu et al. | In this letter, we investigate a notion of probabilistic safety for a stochastic dynamical system. Standard safety notions are based on the idea that an unsafe region of the state-space must be avoided at each instant. In contrast, we study a sequential safety notion which is based on avoiding a tuple of target regions of the state-space. The proposed safety notion is inspired by many practical systems, ranging from modeling jailbreak attacks in large language models (LLMs) to predicting hazardous chemical reactions. In solving the sequential safety problem, we solve a sequential reachability problem for a stochastic dynamical system. |
| 2026-10-27 | [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](http://arxiv.org/abs/2610.37245v1) | Valdini Douglace Lemofouet, Blessing Ngozi Uzor et al. | Safety alignment research for large language models (LLMs) remains concentrated on high-resource languages, leaving open how existing alignment methods perform in low-resource, culturally distinct settings. We study this question for Hausa, a language spoken by more than 200 million people across West Africa. We introduce HausaSafeLM, one of the first Hausa safety preference dataset, built from 10,000 prompts derived from PKU-SafeRLHF: 3,000 human-corrected (gold) and 7,000 machine-translated (silver) examples spanning six harm macro-categories. Using Qwen2.5-7B-Instruct, we compare supervised fine-tuning (SFT) and SFT followed by Direct Preference Optimization (DPO) under both data regimes, evaluating on 500 harmful and 200 benign Hausa prompts.We find that the untuned base model appears highly safe by refusal-rate metrics alone, but this reflects limited Hausa generation ability rather than genuine alignment: its refusals are almost uniformly low quality. SFT improves both refusal fluency and precision and benefits substantially from the larger, noisier silver-tier data, while DPO sharpens refusal precision but comes at a steep, largely data-invariant cost to linguistic quality. No single configuration jointly maximizes safety, helpfulness, and fluency. Our results suggest that current alignment techniques teach models when to refuse more readily than how to do so fluently in underrepresented languages, and that refusal-rate metrics alone are an unreliable signal of safety in low-resource settings. |
| 2026-10-27 | [FastBEVFusion: Simple Multi-Modal Fusion for 3D Object Detection at 50 FPS](http://arxiv.org/abs/2610.37153v1) | Tom Sanitz, Emre Gülsoylu et al. | Reliable real-time 3D object detection based on LiDAR and camera data is critical for safety in autonomous driving, where systems must accurately perceive their surroundings with minimal latency. Many state-of-the-art systems achieve high-quality detection results but fail to meet real-time constraints, limiting their practical use. To address this challenge, we propose FastBEVFusion, a simple, fast, and robust fusion system for 3D object detection. Based on modality-specific features, FastBEVFusion projects these features into semi-aligned modality-specific BEV spaces, without relying on a computationally intensive and potentially error-prone depth estimation. Given these BEV spaces, we propose a light-weight, soft, and dynamic fusion module based on cross-attention, which adapts dynamically to the scene context and enables feature fusion across the entire BEV spaces. Experiments on the complex nuScenes dataset show that FastBEVFusion outperforms state-of-the-art real-time systems, and achieves over 50 FPS without deployment libraries. Additionally, our system shows strong robustness to calibration errors, making FastBEVFusion a powerful 3D object detection system for real-world applications. |
| 2026-10-27 | [Multibranch Ensemble Head in Vision Foundation Models for Efficient Cucumber Phenological Stage Classification](http://arxiv.org/abs/2610.37118v1) | Kostiantyn Khabarlak | Vision Foundation Models (VFMs) pretrained with self-supervised learning, such as DINOv2, provide highly transferable features, yet adapting them to small, domain-shifted datasets is still challenging. We propose a regularizing multibranch ensemble head: a training and inference scheme that pairs low- and high-capacity branches, samples a single branch stochastically at each training step, and averages both branches in probability space at inference. The design adds negligible parameters and essentially no inference overhead over a single heavy head, while improving the accuracy-efficiency trade-off under data-scarce conditions. Accurate classification of plant growth stages is critical for precision agriculture, enabling timely decisions on fertilization, irrigation, and harvesting. Therefore, as a challenging practical case we use cucumber phenological stage classification, in which flowers and fruits are frequently occluded by foliage, and stage transitions manifest primarily as subtle changes in leaf morphology. The NALEF Food Safety dataset contains images of five cucumber growth stages captured under diverse real-world conditions. On it we evaluate a broad range of backbones, including DINOv2 and DINOv3 ViT and ConvNeXt variants, against strongly supervised counterparts (ResNet, EfficientNet, ViT). The proposed DINOv2 ViT-S with multibranch head achieves a Macro F1 of 0.978 and Cohen's Kappa of 0.971, matching the performance of models more than ten times larger in parameters and GFlops, while requiring only 12.2 GFlops and 22.3M parameters, making it deployable on smartphones and edge devices for phenological stage classification. |
| 2026-10-27 | [A hyper-reduction strategy for elastoplastic hydro-mechanical problems in partially saturated soil](http://arxiv.org/abs/2610.37060v1) | Alba Muixí, Lluís Monforte et al. | This work proposes a hyper-reduced order model for coupled hydro-mechanical simulations of elastoplastic deformation and water flow in partially saturated soils. These simulations are used to monitor and assess the safety of geotechnical structures such as earthfill dams, where calibration, uncertainty quantification and scenario analysis require solving the same transient model many times for different parameter values. The displacement and the pore pressure are approximated in a field-split reduced basis obtained by proper orthogonal decomposition, and the Galerkin-projected equations are integrated with decomposed cubature rules, one per balance equation, built with the energy-conserving sampling and weighting (ECSW) method. Each residual is split into an affine part and a nonlinear correction: the affine part is computed exactly, at negligible online cost, so that the cubature rule only needs to approximate the correction, which is localized in space. The construction is independent of the constitutive model choices and is implemented in FEniCSx by assembling the original variational forms on the small set of elements selected by ECSW. The method is tested on two- and three-dimensional dams retaining tailings and water, simulating a raise of the dam and the subsequent consolidation phase, with the friction angle, stiffness and hydraulic conductivity of the dam material and the height of the raise as varying parameters. The hyper-reduced model reproduces the displacement and pressure fields with errors below $0.5\%$ with respect to the finite element solution, achieving speed-ups of up to $43\times$ in two dimensions and $77\times$ in three. These performance gains grow with the extent of the plastic yielding, and the selected elements concentrate in the plastified and unsaturated zones, whose size is set by the physics rather than by the mesh resolution. |
| 2026-10-27 | [State estimation for a hybrid rigid-continuum space manipulator system](http://arxiv.org/abs/2610.37043v1) | Lorenzo Ticozzi, Patricio A. Vela et al. | Spacecraft-mounted robotic manipulators can support long-term operability of orbiting assets by conducting tasks such as on-orbit assembly, repair, upgrade, and refueling. In this context, the integration of continuum/soft robotic mechanisms into space manipulator designs promises to reduce stowage volume and mass, as well as to improve operational safety through the inherent mechanical compliance of soft structures. This work presents a novel state reconstruction pipeline for a hybrid soft-rigid space manipulator, in which the distal link is a tendon-driven continuum robot. The estimation problem is formulated as a maximum a posteriori (MAP) inference problem over the hybrid manipulator's configuration, conditioned on discrete, noisy measurements from joint encoders and a vision-based sensing pipeline. The resulting soft-constrained optimization problem is transcribed as a factor graph, providing a compact representation for fusing heterogeneous measurements with the rigid-soft kinematic model. Furthermore, the adoption of the Geometric Variable Strain (GVS) approach yields a minimum-order parameterization of the configuration, a property expected to facilitate model-based feedback control design in future work. The proposed implementation is validated in simulation on a closed-loop trajectory where, for better realism, position measurements for the continuum link are obtained by means of a simulated visual pipeline leveraging synthetic photorealistic images. The state reconstruction accuracy and computational performance confirm the potential of the proposed framework for hybrid soft-rigid robotic applications in orbit. |
| 2026-10-27 | [RePAIR: Inverting Backdoor Attacks in Semantic Segmentation](http://arxiv.org/abs/2610.37015v1) | Arnold Brosch, Kira Maag | Despite significant advances in computer vision, semantic segmentation models remain vulnerable to backdoor attacks, posing a security risk. These attacks redirect pixels of a chosen victim class to an attacker-specified target class when a trigger is present, while preserving clean predictions. Existing defenses are largely derived from image classification, while dedicated segmentation defenses remain scarce and typically require retraining or model modification. We introduce RePAIR, a post-hoc, label-free, attack-unaware defense based on the observation that backdoor-induced shifts can be stronger in the classifier head than in internal representations, leaving recoverable victim-class semantics. RePAIR infers the victim-target relation from an unlabeled, potentially contaminated image pool, constructs a safety-approved semantic reference, and localizes suspicious pixels by comparing head predictions with internal semantic evidence. It then applies feature-space and output corrections toward the inferred victim semantics, while conservative localization and output constraints limit changes to benign predictions. We model the attack as adversarial probability transport and repair as its approximate reversal, deriving bounds and sufficient conditions for localization, residual attack success, clean-side changes, defense-induced target predictions, and attack-unaware pair identification. Experiments across segmentation architectures and attack settings show that RePAIR substantially reduces attack success while largely preserving clean performance. |
| 2026-10-27 | [Online gas classification at the edge](http://arxiv.org/abs/2610.36999v1) | Ton Juny Pina, Hugh Greatorex et al. | Field applications in chemical sensing, from portable safety monitors to mobile robots, require models that are lightweight, fast to respond, and can be recalibrated from limited labelled data. Current gas classification pipelines assume controlled sampling cycles and offline feature extraction, and rely on computationally expensive models. With a few thousand neurons and sub-milliwatt power consumption, insects reliably classify odours in open environments. Inspired by the insect olfactory system, we present a neural network architecture that classifies raw sensor array readings using a sparse, high-dimensional binary representation. We evaluate it on random sequences of single gases and binary mixtures, introducing new samples dynamically and classifying gas identity from the transient sensor response. In this setting, the architecture out-performs a linear \gls{svm} on gas mixtures and matches an optimised non-linear \gls{svm}, while maintaining online class-level frequency estimates that distinguish familiar from novel stimuli. It relies exclusively on binary weights and Hebbian-based local learning, remaining computationally lightweight and well suited to ultra-low-power neuromorphic processors. We demonstrated the network on-chip with a hazardous gas detection experiment running on the mixed-signal neuromorphic chip TEXEL. The on-chip classifier yielded an accuracy of 74.6\% while running at 482 $\unit{\nano\joule}$ per inference and 27 \unit{\micro\watt} of static-power. |
| 2026-10-27 | [Omni-IGN: Onboard Image-Goal Navigation for Drones with Omnidirectional Vision](http://arxiv.org/abs/2610.36968v1) | Zichen Yan, Rui Huang et al. | Image-goal navigation (ImageNav) requires an agent to reach a location specified by a goal image in an unknown environment. For drones, this task demands exploration, goal- directed navigation, and collision avoidance under limited onboard computing resources. Existing aerial ImageNav methods are built on single-view observations, requiring additional rotation and exploration to search for goal-relevant cues outside the field of view. In this paper, we present Omni-IGN, a fully onboard ImageNav system that integrates quad-fisheye perception, an attention-based reinforcement learning (RL) policy, and a collision-aware safety layer. The policy adaptively fuses goal-relevant features across multiple views to provide directional guidance for continuous navigation. The quad-fisheye RGB images support both ImageNav and obstacle avoidance without a dedicated depth sensor. On Habitat benchmarks, Omni-IGN achieves the best performance among continuous-action baselines. Moreover, flight tests in IsaacSim validate the effectiveness of integrating learning-based depth estimation with the action safety layer for collision avoidance. Finally, real-world experiments demonstrate zero-shot sim-to-real transfer without any fine-tuning or external localization. The code and model checkpoints will be released upon acceptance. |
| 2026-10-27 | [Self-organized hybrid scenario mediated by fishbone modes in the EAST tokamak](http://arxiv.org/abs/2610.36875v1) | J. S. Geng, Y. Q. Chu et al. | Self-organized hybrid scenario mediated by fishbone modes in the Experimental Advanced Superconducting Tokamak (EAST) is reported in this work. In this regime, internal transport barriers(ITBs) and a flat core safety factor profile were maintained simultaneously by fishbone modes. First,the drive of low-frequency flows by fishbone modes is clarified, which subsequently suppresses turbulence transport. Besides, fishbone-induced poloidally symmetric (m = 0) toroidal current sheets were first reported experimentally, which drive outward current redistribution that mitigates core current peaking. Together, these effects enable hybrid operation with stable ITBs and robust magnetohydrodynamic (MHD) stability, revealing a self-organized pathway for sustaining hybrid scenario for fusion reactors. |
| 2026-10-27 | [When Black-Box Robot Policies Fail: Online Sensor Diagnosis and Repair with a Transparent Surrogate](http://arxiv.org/abs/2610.36852v1) | Marc Ziegler, Batu Özen et al. | Robot foundation models may generalize across tasks and embodiments, but their actions still depend on sensors that can fail. When the policy is frozen or inaccessible, can an external monitor diagnose and repair corrupted inputs without access to its internals? We test a structured surrogate that separates periodic gait from sensor-dependent corrections to guide input repair around a frozen locomotion policy. On simulated Go2, offline diagnosis raises walking success from 259/640 to 429/640 episodes; online repair reaches 382/640, with no triggers in 64 held-out fault-free episodes. A matched boosted-tree surrogate predicts clean actions similarly well but yields no detected repair improvement over no repair. Separate fits improve walking on Go1 and ANYmal C, although phase-aware gradients perform better. Evidence remains limited to one policy and a partly screened simulated fault suite; robot foundation model and hardware validation remain open. |

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



