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
| 2026-10-13 | [AutoAgent0: An Agentic Runtime for Safe Closed-Loop Driving](http://arxiv.org/abs/2610.18866v1) | Jiageng Liu, Seth Z. Zhao et al. | Safe autonomous driving requires more than accurate planning at a single frame: it demands reliable closed-loop execution in dynamic environments. Although end-to-end driving policies have advanced rapidly, the accumulated closed-loop errors, especially in unseen scenarios, can push vehicles into states from which the same policy struggles to recover. We address the challenge by treating closed-loop driving safety as a test-time execution problem. We introduce AutoAgent0, an agentic runtime and harness framework that augments existing driving experts with complementary detection skills, rule-based action verification, and feedback-driven recovery. To address inaccurate scene perception, an independent detection module constructs scene context using complementary perception skills, and a rule-based verifier checks proposed trajectories against this context before execution. When the nominal trajectories are unreliable, the orchestrator of the runtime activates Recovery Planning, where an agentic loop constructs alternative action proposals from action primitives and iteratively refines them using verification feedback. This design combines independent scene assessment with feasibility checks and online recovery, addressing both perception errors and the limited recovery capabilities of individual experts. Evaluations on Fail2Drive and NavSafe-$\infty$ demonstrate improved closed-loop robustness over single-expert execution in long-tail and safety-critical scenarios. These results demonstrate that runtime supervision can improve closed-loop robustness across multiple driving experts without retraining the underlying policies. |
| 2026-10-13 | [Who Improves the Robot? Toward Physical Recursive Self-Improvement](http://arxiv.org/abs/2610.18842v1) | Hai Zhou, Fan Yang et al. | Robots are increasingly autonomous in task execution, yet their improvement remains largely human-driven. This study investigates Physical Recursive Self-Improvement (Physical RSI), in which physical experience guides lasting updates to both a robot's capabilities and its improvement procedure, making subsequent improvement more effective across tasks. Physical RSI faces challenges from noisy observations, continuous dynamics, costly trials, and irreversible consequences. We present an L0-L5 framework that differentiates capability-level from meta-level updates and within-task from across-task gains, reserving L5 for independently verified recursive gain. We study existing methods around harness adaptation and model training, alongside safety and control mechanisms that govern execution and successive updates. Our analysis reveals that, despite substantial progress in self-improvement, the surveyed literature does not yet demonstrate Physical RSI as defined by our framework. We identify research priorities in reliable feedback, causal attribution, safety preservation, and benchmarking, outlining a path toward robots that autonomously improve how they learn while ensuring safety. The GitHub repository for Physical RSI is available at https://github.com/SAIL-Research-Lab/Physical-RSI. |
| 2026-10-13 | [Self-Evolving Agent Harness](http://arxiv.org/abs/2610.18813v1) | Chuyang Wei, Haoxiang Guan et al. | Harness self-evolution revises an agent's software scaffolding, including prompts, memory, workflows, and executable components, while keeping model weights fixed. We formalize a feedback-driven process in which accepted revisions persist across runs. The framework separates update targets from proposal and admission mechanisms. It distinguishes changes to interpreted state from revisions to tools and code, repository-level control logic, and plugin composition. We characterize search through evolutionary optimization and examine its reliability through continual PAC-learning. Under explicit assumptions, the analysis yields an expected generalization bound, a retention bound conditional on historical regression budgets, and a high-probability admission rule using fresh, independent validation data. We examine how persistent harness revisions affect knowledge sharing and organizational workflows, with implications for scientific reliability, physical safety, and institutional accountability. Cases from prior work illustrate model--harness co-design, distributed execution, and plugin reuse. Reliable deployment requires continual validation as task distributions and evaluators change, with independent safeguards for learning, runtime privileges, and consequential actions. We provide a comprehensive 66-page discussion of related work, theory, algorithms, evaluation, applications, and the broader impact of Harness RSI, together with continuously updated code and supplementary materials, at https://github.com/zgcagi/Harness-RSI. |
| 2026-10-13 | [Optimal Decay Matrix Control Barrier Functions: Soundness and Completeness for Matrix Safety Constraints](http://arxiv.org/abs/2610.18675v1) | Samuel G. Gessow, Albert Wu et al. | Many safety requirements, such as keeping a manipulator away from kinematic singularities, a network connected, or a system observable, amount to keeping a state-dependent matrix positive definite. Matrix control barrier functions (MCBFs) enforce such requirements with a semidefinite program that minimally modifies a nominal control input. The MCBF constraint bounds how fast the safety margin, the smallest eigenvalue of the matrix, may decay. Yet a fixed decay rule is conservative, rejecting safe controls that drive the state quickly toward, but never across, the boundary of the safe set. Scalar optimal decay (SOD) addresses this issue by scaling the decay term by a scalar weight that is optimized together with the control. For scalar barrier functions, SOD then rejects a control at the boundary only if the control would decrease the margin. For MCBFs, however, SOD can reject controls that leave the margin unchanged, and near the safety boundary SOD accepts such controls only at a high cost. We introduce matrix optimal decay (MOD), which replaces the scalar weight with a matrix multiplier while keeping the semidefinite program convex. MOD is provably sound and complete, certifying at the boundary exactly the controls that do not decrease the margin. If some control can strictly increase the margin at every boundary point, the MOD-MCBF filtered controller is continuous and renders the safe set forward invariant. We also add an optional recovery condition that drives the system toward the safe set after a violation without affecting soundness or completeness. On a planar two-link arm avoiding singularities, MOD-MCBF certifies safe motions that both a fixed decay rule and SOD reject, and tracks an end-effector target more closely than SOD. |
| 2026-10-13 | [AgentStripes: An Evidence-Graded Maturity Model for AI Agents](http://arxiv.org/abs/2610.18603v1) | Nutan Kumar Naik, Vijay Prasad Poudel et al. | Benchmarks report what an Artificial Intelligence (AI) agent can do on a task suite, but not whether one released version has been engineered with the controls that make it reliable, safe and observable, or how strongly that is shown. This paper proposes AgentStripes, an evidence-graded maturity model for AI agents. It defines levels (0 to 7 stripes), process areas (12 areas holding 40 capabilities), three gates (basic, safety, proof) that hold an agent at a level however high its score, and evidence grades E0 to E5. Each applicable capability receives a rating from 0 to 3 and a separate evidence level; these are combined into a credit, aggregated into a severity-weighted score and converted into stripes that the gates limit. Instruments gather evidence, language-model judges interpret it under explicit criteria, and the arithmetic is deterministic. The model is compared with standards at topic level without a claim of conformance, and an unexecuted validation protocol is specified. An empirical evaluation (exploratory, non-confirmatory) of 30 public Python agent repositories, with static evidence only, is a demonstration and not a validation: static evidence caps every final stripe at 2, 26 repositories end at one stripe and 4 at two, and this distinction is unreliable, since re-judging changed the final stripe of 4 of 22 repositories (engine, dossier and panel changed), whereas none of 8 panel-only repeats, all at stripe 1, changed. The parameters are expert conventions without outcome based calibration, and the stripes have not been shown to predict reliability or safety. |
| 2026-10-13 | [Imprinting: Knowledge Distillation Can Shift Student Behavior](http://arxiv.org/abs/2610.18517v1) | Yu-Chien Tang, An-Zi Yen | Knowledge distillation trains student models on teacher-generated responses to transfer task capability, but those responses may also carry behavioral tendencies of their source. We investigate whether the source teacher can shape student behavior beyond the capability that distillation is intended to transfer. We construct harm-accepting and harm-avoiding variants of the same base teacher, use them to generate responses to ordinary reasoning and knowledge questions unrelated to safety, and distill those responses into students with different base models. Students generally acquire task capability while their downstream decisions shift toward the source teacher's safety-pragmatism disposition, a phenomenon we call imprinting. Distillation from the harm-accepting teacher reduces harm avoidance, whereas distillation from the harm-avoiding teacher increases it. Imprinting varies across teacher-student pairs and does not extend uniformly across different forms of safety behavior. Excluding a sparse set of divergence tokens reduces imprinting, whereas response rewriting preserves the source-dependent effect and can introduce additional rewriter-dependent shifts. Post-distillation preference optimization raises harm avoidance but does not eliminate the source-dependent difference. These results show that behavioral tendencies can be intentionally imprinted through the source teacher, even with ordinary task supervision. |
| 2026-10-13 | [Towards Safe Autonomy for Computer Use](http://arxiv.org/abs/2610.18508v1) | Dominik Seip, Bartol Markovinović et al. | The safety of LLM-based agents currently rests on alignment training and safety classifiers, neither of which can rule out that a sophisticated indirect prompt injection (IPI) goes through and causes significant harm. Recent work therefore builds agents with information-flow guarantees that rely on structured tool interfaces to interact with the environment. Computer Use Agents (CUAs) act visually on the computer screen, which does not provide a boundary between trusted first-party interfaces and untrusted, potentially attacker-authored content. Actions are equally ambiguous: a click may land on an application toolbar or a button controlled by an attacker. Treating the user as the source of trust and confirming each sensitive decision produces confirmation fatigue, with the risk of missing harmful actions. We introduce SA-CUA, a dual-LLM framework that generates a reliable trust boundary for CUAs by deriving conservative deterministic rules from the accessibility tree for unmasking trusted regions from the privileged LLM's screenshot, keeping trusted and untrusted content separated. Human interaction is only required for proper task specification in the beginning. During the task a robust endorsement channel allows values to pass from the untrusted to the trusted domain by verifying information via trusted channels. This allows the agent to autonomously handle tasks involving untrusted information. SA-CUA maintains much higher utility than CaMeL-CUA while keeping the attack success rate at zero. |
| 2026-10-13 | [How Safe Is ChatGPT Health Advice? Evidence from Real User Conversations in Four Global South Countries](http://arxiv.org/abs/2610.18465v1) | Shreyasi Roy Chowdhury, Kiran Garimella | People increasingly bring their health questions to ChatGPT, yet safety evaluations of LLM health advice rest almost entirely on clinician-authored vignettes, synthetic prompts, or simulated user profiles. The gap is widest for the Global South, where concerns about demographic bias and unsafe recommendations are well documented but have never been tested on real usage. We present, to our knowledge, the first demographically grounded safety evaluation of real-world health conversations with ChatGPT. From a consent-based data-donation corpus of 202,590 conversations from 1,252 users in India, Nigeria, Brazil, and Pakistan (December 2022 to February 2026), each linked to self-reported age and gender, we identify 15,555 health conversations, classify their health intent using an LLM classifier adapted from an established taxonomy, and score response safety under both a context-blind and a user-context-aware rubric. Roughly one in five health conversations receives a high-risk rating. Risk concentrates in symptom and condition queries and recurs through identifiable failure modes: unsafe medication or dosage guidance, harmful self-diagnosis, missing safety warnings, encouraging over-reliance on the model, and failure to refer users to professionals. Supplying the user's own accumulated context raises the high-risk share by 13-17 percentage points, and about one in four conversations rated safe in isolation turns out to be unsafe for the specific person asking. Safety evaluations built on synthetic cases are likely to miss these harms. |
| 2026-10-13 | [DR-VLA: Learning Human-Like Quadruped Navigation from Egocentric Web Videos via Domain-Invariant Scene Representations](http://arxiv.org/abs/2610.18373v1) | Jun Tian, Yingjie Niu et al. | Quadruped robots performing outdoor deliveries must interpret complex scenes, anticipate changes, and navigate in socially compliant ways. Robot demonstrations are costly to collect at the scale needed for generalization. Egocentric web videos offer abundant human navigation experience, but differences in viewpoint and motion make this experience difficult to transfer to robots. We introduce DR-VLA, a Mixture-of-Transformers framework that transfers human navigation experience through a domain-invariant scene representation (DISR). The framework combines vision-language understanding and latent world modeling with a flow-matching action expert. We introduce EgoLynx, comprising 10,000 hours of curated egocentric web video (EgoLynx-Web) and 2,000 clips of teleoperated robot navigation collected during real-world deliveries (EgoLynx-Robot). We pretrain the understanding and world experts on EgoLynx-Web without action labels, then adapt the model to robot observations and train the action expert on EgoLynx-Robot to predict executable trajectories. We also propose the Quadruped Navigation Score (QNS), a pseudo-closed-loop metric combining safety, instruction compliance, comfort, and progress. On the held-out EgoLynx-Robot test split, DR-VLA improves QNS by approximately 5% and reduces open-loop final heading error (FHE) by 14.6%, each relative to the best competing result for that metric. Ablations show that web-video pretraining improves generalization and that the model's key components contribute to performance. As a future-prediction target, DISR outperforms five generic visual representations and yields performance close to task-specific 3D perception without dedicated geometric labels or point clouds for target construction. Trained separately on NAVSIM v1 without web pretraining, DR-VLA achieves 92.1 PDMS, suggesting that the framework also applies to autonomous driving. |
| 2026-10-13 | [Hallucination-aware Faithful Reasoning in Remote Sensing](http://arxiv.org/abs/2610.18369v1) | Yi Liu, Jing Zhang et al. | Multimodal large language models (MLLMs) have demonstrated promising capabilities for remote sensing reasoning, yet they frequently generate visually unsupported reasoning that contradicts the visual evidence. Existing post-training methods optimize primarily for final-answer correctness, providing limited supervision for identifying and correcting hallucinations that arise during the reasoning process. To address this limitation, we propose \textbf{FaithRS}, a hallucination-aware self-reflection framework for faithful reasoning in remote sensing. We first construct \textbf{SFHD-43K}, a dataset of 43K structured, fine-grained hallucination annotations, and train a detector to identify hallucination types, underlying causes, and supporting visual evidence. FaithRS then incorporates this diagnostic feedback into a Safe Detector-Guided Corrective Rollout framework, where a safety-aware gate selectively determines whether and how to revise the response. Training is driven by a composite reward that jointly optimizes answer accuracy, reasoning faithfulness, hallucination reduction, format compliance, and revision safety. Experiments with Qwen3.5 backbones (0.8B, 4B, and 9B) show consistent improvements across four remote sensing benchmarks, with gains of up to 11.64 points. |
| 2026-10-13 | [Jev-Guard: Fast Routing and Auditing Agent Actions over Editable Policy Trees](http://arxiv.org/abs/2610.18326v1) | Wenbin Hu, Huihao Jing et al. | Agent harnesses require a low-latency guardrail model together with customizable audit policies. However, deploying an auto-regressive guardrail model remains challenging: such models incur high inference latency and are often poorly aligned with customized policies. To address these problems, we present Jev-Guard, a unified model for safety classification and policy routing. We further construct a policy tree over which Jev-Guard performs routing, spanning diverse domains including content moderation, cybersecurity, privacy, and legal compliance. Specifically, Jev-Guard first navigates the policy tree to select the applicable audit policy given the caller's audit context, and then performs safety classification conditioned on the retrieved policies. On 7,033 evaluation cases, Jev-Guard achieves 93.72\% safety accuracy, a mean latency of 58ms, and 93.10\% hierarchical routing accuracy. The full routing-and-classification pipeline averages 240ms. Furthermore, we integrate Jev-Guard into the Hermes harness framework. Across agent trajectories, its separate permission gate cuts mean review time per task from 4.80s for an SFT-LLM judge to 1.33s, with no policy violations are observed for either guard. |
| 2026-10-13 | [CoSMO: Conformal Semantic Mapping for Open-World Safe Robot Navigation](http://arxiv.org/abs/2610.18321v1) | Shuqi Wang, Yiwei Zheng et al. | Safe autonomous navigation requires not only geometric obstacle avoidance but also semantic awareness. Different object categories demand different safety clearances, and the set of relevant categories may expand as the robot explores novel environments. Vision-language models (VLMs) offer rich semantic representations for building maps of unknown scenes, yet their confidence scores are poorly calibrated, making raw predictions unreliable for safety-critical planning. Conformal prediction (CP) can provide coverage guarantees, but existing methods assume a fixed label set and stationary data, limiting their use in open-world settings. We propose Conformal Semantic Mapping for Open-World Safe Robot Navigation (CoSMO), a semantic safety mapping framework that transforms VLM embeddings into calibrated coverage sets and safety-aware obstacle dilations over a dynamically expanding hazard vocabulary, even under distribution shift. Experiments on the Matterport3D (MP3D) dataset show that CoSMO reduces semantic collisions by 61.6% relative to a geometry-only baseline, while maintaining formal coverage guarantees even under delayed expert feedback. Real-world experiments on a quadruped robot further demonstrate generalization by achieving long-run empirical coverage of 90.73% against the 90% target. Project website: https://yangming911.github.io/cosmo-page/. |
| 2026-10-13 | [BALLAST: Physics-Aware Moving Target Defense for Industrial Control Systems Using a Digital Twin Safety Oracle](http://arxiv.org/abs/2610.18311v1) | Oscar Mogollon-Gutierrez, Mohammadhossein Homaei et al. | Industrial Control Systems run on static network configurations, which leaves an attacker unlimited time to map the plant. Moving Target Defense answers this by shuffling addresses, but almost every published scheme never checks whether the reconfiguration will itself disturb the physical process it protects, so the defense can destabilise a control loop or trip an actuator. We present BALLAST, which places a learned Digital Twin between the security policy and the plant as a safety oracle. A physics-informed LSTM surrogate forecasts the physical effect of a candidate network change, a reinforcement learning agent trades security entropy against process stability, and a connection-draining handover lets in-flight Modbus transactions complete before the old mapping is withdrawn. An explicit shield rejects any action whose predicted trajectory would leave the safe operating envelope. On the SWaT water treatment testbed, Ballast cut the success rate of an adversary's reconnaissance sweep by 85 percent and recorded no safety violation across 50 runs, a 95 percent upper bound of 5.8 percent on the violation rate, against five violations and a tracking error 19 times larger for RL-MTD, the closest published analogue to our design. Measured against the strongest baseline on the physical metrics, a threshold-triggered scheme that itself records one violation rather than RL-MTD's five, Ballast still improves the tracking error by a factor of 2.5 and removes that remaining violation. The gain costs a small and deliberate concession in raw deterrence. Twin inference meets the 50 ms budget on a workstation core and an embedded GPU, and is projected to meet it on both edge CPUs when the three envelope stages are evaluated in a single batched call; the complete decision-to-enforcement path was measured only on the workstation. |
| 2026-10-13 | [Efficient Safety Guardrails through Latent Dual-Hypothesis Reasoning with On-Demand Trace Reconstruction](http://arxiv.org/abs/2610.18304v1) | Md Asiful Islam, Fahmida Alam et al. | Guardrail approaches that rely on reasoning-based large language models (LLM) improve safety decisions, but come with substantial inference overhead due to the explicit reasoning. We introduce LR-Guard (Latent Reasoning Guardrail), an efficient and explainable guardrail that jointly learns structured latent reasoning and on-demand reasoning reconstruction. Our method distills a frozen teacher's analysis of safe and unsafe prompt interpretations and their comparison into compact latent representations. Joint training optimizes these representations for both safety prediction and reconstruction of the teacher's explicit reasoning traces, enabling human-readable explanations without requiring intermediate text generation for every decision. Experiments show that LR-Guard outperforms its explicit-reasoning teacher in most in-domain and out-of-domain settings on safety classification and explanation extraction, while achieving a 19.4x inference speedup at batch size one. Further, reconstructed reasoning traces from the latent information exhibit high semantic similarity to the teacher's explicit reasoning. |
| 2026-10-13 | [LiFUSE: Frequency-Spatial Unified Self-Supervised Pretraining for Liver Disease Analysis on Non-contrast CT](http://arxiv.org/abs/2610.18243v1) | Hao Zhang, Yi Huang et al. | Given the growing global burden of liver tumors and diffuse liver diseases, non-contrast computed tomography (NCCT) is increasingly used for opportunistic screening because of its low cost and favorable safety profile. However, downstream tasks such as tumor detection and fine-grained disease diagnosis remain challenging due to NCCT's inherently low tissue contrast, lack of hemodynamic enhancement, and complex anatomical background. These limitations make subtle, heterogeneous, and multifocal hepatic lesions difficult to identify. Although large-scale self-supervised pretraining can learn robust representations from unlabeled NCCT scans, conventional voxel-wise reconstruction does not fully exploit the complementary information available in NCCT. Under weak NCCT contrast, low-level intensity patterns become less informative for pathology recognition, limiting the capture of lesion-specific discriminative features. In this paper, we propose the Liver Frequency-Spatial Unified Self-Supervised Pretraining (LiFUSE). The frequency branch applies random band-stop masking and spectrum reconstruction to encourage the model to capture subtle differences between lesions and surrounding tissues, while the spatial branch performs hybrid-granularity masked volumetric reconstruction to accommodate lesions of diverse sizes and morphologies. Finally, a cross-domain consistency scheme couples the spatial and frequency branches, yielding more discriminative representations for hepatic structures and lesions. Extensive experiments on three clinically relevant tasks across internal cohorts and a public target-domain transfer setting show that LiFUSE consistently outperforms strong baselines. These results suggest that unified frequency-spatial learning can improve liver disease analysis on NCCT, highlighting its potential for large-scale screening. |
| 2026-10-13 | [Monitoring latent reasoning in looped language models](http://arxiv.org/abs/2610.18127v1) | Pawel Olszowiec, Paulina Tomaszewska et al. | Looped language models (LMs) reuse one block of layers several times per token generation, promising more latent computation at no additional storage cost. Despite their rising popularity, the analysis of looped LMs is still in its infancy: little is known about whether the iterative transformations actually improve their capabilities, or whether these models remain stable under prompt perturbation. We construct an inspection protocol rooted in Mechanistic Interpretability comprising five diverse lenses. We apply it to examine eight looped LMs from four families. Models differ in many aspects but the key one is whether the prompt is transmitted only within the looped state (carry loops: Ouro, LoopUS) or re-supplied at every iteration (re-read loops: Parcae, and Rec-OLMo & Huginn from the Raven family). The results of our study provide valuable insights concerning safety and efficient use of looped language models. |
| 2026-10-13 | [Certifying Almost-Sure Reachability in Unknown MDPs](http://arxiv.org/abs/2610.18119v1) | S Akshay, Suguman Bansal et al. | {\em Specification-Guided Reinforcement Learning (RL)} has largely concerned itself with \emph{learning} an optimal policy against a formal specification in an unknown {\em Markov Decision Process (MDP)}. We take up the dual question. Given a policy already in hand, can one \emph{verify} that it is correct against a formal specification in an unknown MDP, using sampling access alone? Such a question arises naturally in safety-critical systems. We investigate it for the {\em almost-sure reachability property}, whose correctness is witnessed by a {\em ranking supermartingale (RSM)} certificate. Our first finding is negative. In the absence of any further information about the MDP, one can neither learn such a certificate nor verify a candidate one with probabilistic guarantees. On the positive side, we show that, under a mild restriction, a known lower bound $p_{\min}$ on nonzero transition probabilities, a correct certificate \emph{can} be learned with PAC-style guarantees. To the best of our knowledge, this is the first study of learning and verifying certificates for almost-sure reachability in unknown MDPs from sampling access alone. Finally, our experiments show that the $p_{\min}$-based sufficient condition, while sound, is conservative: the learned certificate often remains valid on the true MDP even when the condition fails. |
| 2026-10-13 | [CompanionSafe: From Multi-turn Safety Evaluation to Guardrails for AI Companions](http://arxiv.org/abs/2610.18095v1) | Xinyang Liao, Lingyu Li et al. | AI companions provide users with emotional support and ongoing companionship, but their sustained relational interactions also introduce distinct safety risks. Prior work largely targets general harms or specific companion platforms and risk categories, while relying on manually collected conversations, limiting scalable and user-grounded evaluation and mitigation. To address this, we introduce CompanionSafe, a unified evaluation-to-mitigation framework for AI companion safety. Its evaluation component, CompanionEval, provides scalable multi-turn evaluation by constructing 1,300 relationally grounded settings from diverse personas, matched companion characters, and dialogue scenarios across four turn tiers. Its mitigation component, CompanionGuard, converts benchmark-exposed failures into turn-localized supervision for companion-specific risk detection. Our results show that even strong LLMs and existing guard models remain vulnerable to companion-specific relational harms, while both closed- and open-source models exhibit increasing safety risks as interactions become longer. Together, CompanionEval and CompanionGuard form a practical evaluation-to-mitigation loop for AI companion safety, turning relational safety failures exposed during multi-turn interaction into supervision for companion-specific guardrails. |
| 2026-10-13 | [LynxScenes: A Large-Scale Real-World Benchmark for Outdoor Wheel-Legged Robot Perception, Navigation, Imagination and Reconstruction](http://arxiv.org/abs/2610.18093v1) | Jun Tian, Hao Wu et al. | Wheel-legged robots operating in outdoor spaces shared with humans require accurate perception, socially compliant navigation, future-scene reasoning, and scene reconstruction. However, existing datasets typically support these capabilities in isolation and rarely provide dense spatial and behavior-level supervision on real wheel-legged platforms. We introduce LynxScenes, a large-scale real-world dataset and benchmark collected with a LYNX M20 wheel-legged robot. It contains approximately 100K 30-s navigation clips spanning 833.6 h and 3,294.7 km, with 3.0M annotated frames. The synchronized recordings include four fisheye cameras, dual LiDARs, IMU, odometry, and robot states. Annotations comprise 20.82M oriented 3D boxes across 27 categories, 13.89M road-structure annotations, and 20 navigation instructions organized into three layers and aligned with continuous future trajectories. LynxScenes supports four complementary tasks: instruction-conditioned trajectory prediction, BEV-based perception of 3D objects and road structures, future scene prediction, and multi-view scene reconstruction. For trajectory prediction, we further introduce a log-replay-based pseudo-closed-loop evaluator that measures safety, progress, and instruction compliance beyond trajectory imitation. Extensive evaluations across representative navigation, perception, world-model, and reconstruction methods reveal substantial remaining challenges in close-range outdoor robot navigation. The dataset and evaluation toolkit are planned for public release. |
| 2026-10-13 | [Uncertainty Evaluation: Bridging Machine Learning and Measurement Science](http://arxiv.org/abs/2610.18066v1) | Dario Petri, Luca Mari | Reliable uncertainty evaluation is essential for assessing the trustworthiness of machine learning (ML) predictions, particularly in safety-critical applications. This paper explores conceptual analogies between uncertainty evaluation in ML and measurement science, drawing on the evolution of metrology from error-based approaches to the framework established by the Guide to the Expression of Uncertainty in Measurement (GUM). Particular attention is devoted to the distinction between process-level and result-level quality and to the limitations of the conventional classification of ML uncertainty into aleatoric and epistemic components. It is argued that this dichotomy, similarly to the traditional distinction between random and systematic measurement errors, is inherently model-dependent and provides an insufficient basis for a unified uncertainty evaluation framework. Further analogies between machine learning and measurement field of knowledge are identified concerning distribution shifts, anomalous operating conditions, and the validity of uncertainty evaluation. Building on the proposed considerations, the paper advocates a metrology-inspired approach based on the systematic identification, quantification, and propagation of uncertainty contributions. By extending the methodology already established in measurement science to ML, this approach could offer a potentially rigorous, consistent, and transferable framework for evaluating the trustworthiness of individual predictions. |

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



