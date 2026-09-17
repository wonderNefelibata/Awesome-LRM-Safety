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
| 2026-09-16 | [Flag Game: A Toy Model for Mechanistic Swarm Interpretability](http://arxiv.org/abs/2609.19124v1) | Elizabeth Pavlova, Hidenori Tanaka | Emergent coordinated behaviors of AI agents are starting to present critical safety risks. A key phenomenon driving these behaviors is the rapid formation and spread of beliefs about the world, and mechanistic understanding is crucial for collective alignment. To this end, we introduce the Flag Game, a toy model for studying the mechanisms of collective belief formation. Concretely, a hidden country flag defines the ground truth, and each bounded agent directly observes only a private crop but can exchange beliefs and weigh social evidence from peers. Despite its simplicity, the Flag Game reproduces rich collective phenomenology: non-monotonic scaling of performance with population size, accuracy gains from social-awareness prompting and team diversity, and strong effects of organizational structure. In particular, we identify that collective belief collapse at small population sizes turns into collective belief polarization as the population grows. This polarization causes the performance decline at large population sizes, but creates diversity in collective beliefs. Finally, we dissect the mechanisms underlying collective belief collapse and polarization with two complementary approaches. We first introduce social circuit attribution, a technique to predict which agent, and what view, matters most to collective dynamics, and verify its predictions by causal interventions on agents, tracing how agent patching changes collective outcomes. However, the efficacy of causal interventions on agents decreases as the population grows. We therefore develop a statistical mechanical theory for larger populations and verify that it matches the empirical phase diagram. Together, these results take a first step toward mechanistic swarm interpretability, a science of how the properties of individual agents and their communication give rise to emergent collective behavior. |
| 2026-09-16 | [Vigil: Accountable Liveness against Selective Silence](http://arxiv.org/abs/2609.18778v1) | Jiawei Cheng, Huiping Sun et al. | BFT accountability is well understood for safety violations, and recent work attributes global liveness violations; \emph{recipient-selective} silence remains unresolved. A selectively silent adversary withholds messages from some honest nodes while behaving correctly toward others. It can stall consensus yet evade every existing mechanism. We initiate a systematic study of accountability against selective silence. Negatively, a lone attacker silent toward at most $f$ honest nodes is indistinguishable from an honest node, yielding a universal lower bound $K_{\mathrm{SI}} \ge f{+}1$ on the \emph{silence identification threshold}; moreover, any feedback-free repair after a silence-induced violation costs $Θ(n^3)$. Positively, \textsc{Vigil}, a Tendermint variant, matches these bounds with attack-adaptive forwarding, via bitmap cross-attestation, core-based membership, and challenge--response auditing. It pays $O(n)$ authenticators per node when no selective silence occurs (plus $Θ(n^2)$ bitmap metadata bits per node), relays in proportion to the attack's width (sub-threshold silence can force up to $n^3/27$ relays per view, a cost we price exactly), and majority-accuses any node silent toward more than a tunable resilience $τ_A$ of honest peers ($K_{\mathrm{SI}} = τ_A{+}1$, optimal at $τ_A = f$). We also price the residual sub-threshold griefing surface exactly and extend identification to $x$-partial synchrony. Real-network experiments on a three-region WAN, together with a simulator held to exact equality with every closed form, confirm each threshold and cost: at $2\%$ loss, an $f{+}1$ accusation bar falsely accuses $91.2\%$ of honest nodes, while our majority bar accuses $0.002\%$. |
| 2026-09-16 | [TRACER: Adaptive Multi-Robot Social Navigation via Joint Human-Response Prediction and Interaction-Aware Replanning](http://arxiv.org/abs/2609.18776v1) | Lan Hu, Minghui Liwang et al. | Multi-robot navigation in human-shared spaces is inherently interactive: coordinated robot motions influence how nearby entities respond, while those responses provide valuable information for subsequent robot decisions. However, existing methods typically address action-conditioned prediction, multi-robot planning, or online adaptation separately, and therefore lack a unified mechanism for modeling joint robot-entity interactions and adapting future decisions from executed interaction outcomes. To address this gap, we propose TRACER, a bi-directional receding-horizon framework that closes the loop between prediction and adaptation. TRACER evaluates candidate (i.e., alternative feasible future motion plans for the robot team) trajectories using a per-entity probabilistic response model that separates individual-robot effects from non-additive pairwise interactions; after executing the selected trajectory prefix, it updates persistent identity-bound beliefs over latent response modes using the synchronized observed responses. These updated beliefs then guide subsequent candidate evaluation under probabilistic safety and response-aware cost criteria. Experiments show that (i) TRACER more accurately captures non-additive multi-robot interaction effects than a capacity-matched additive predictor, (ii) persistent identity-consistent evidence improves response prediction and downstream replanning, and (iii) the complete TRACER framework improves collision-free completion over an independent-robot baseline on the SocialGym2 multi-robot social-navigation benchmark. |
| 2026-09-16 | [VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge](http://arxiv.org/abs/2609.18663v1) | Deyu Cao, Ryuji Oi et al. | Billion-parameter vision--language--action (VLA) policies demand substantial onboard power, while communication delays in remote inference hinder timely responses. We propose VLA-ULAP, which interleaves remote VLA calls with an Ultra-Lightweight Local Action Predictor (ULAP). With approximately 7.4M parameters including the frozen vision encoder, ULAP combines current views, proprioception, and executed action history to predict chunks in one pass. Trained independently, it requires no VLA hidden states, online verification, or server round trips. On Jetson Orin Nano, ULAP takes 19.9 ms and 0.183 J per inference, compared with 284.3 ms and 50.55 J for GR00T on RTX A6000. Across three simulated base-policy/benchmark pairs, selected operating points remove 48.8--76.7\% of VLA calls while retaining 95.0--97.5\% of the baseline success rate. Against local VLA-acceleration alternatives on VLA-JEPA, ULAP uses an estimated 49.2\% less inference time and 51.0\% less GPU energy per successful episode than ACT at comparable success rates, and 77.1\% less time and 79.9\% less energy than SP-VLA at equal success rates. Physical SO-101 experiments retain 95.2--100\% of the baseline success rate across seen and held-out placements while reducing inference time by an estimated 47.9--58.0\% and inference-device energy by 52.1--62.5\%, based on successful-episode call counts and measured device costs. Faster responses also improve dynamic-task success rates: in latency-aware LIBERO-Safety simulation, VLA-ULAP exceeds $π_{0.5}$ by 11.0 and 15.5 percentage points on two tasks while approximately halving VLA calls. |
| 2026-09-16 | [DyMT-ESB: Dynamic Multi-Turn Evaluation of Social Bias in User-LLM Interactions](http://arxiv.org/abs/2609.18649v1) | Rem Hida, Masahiro Kaneko et al. | Warning: This paper contains examples of stereotypes and social bias. LLMs are increasingly used in interactive settings by the general public, making the evaluation of model behavior in multi-turn conversational scenarios important for safety, including stereotyping-related harms. However, existing multi-turn social bias evaluations often rely on pre-specified or template-based user inputs that do not adapt to model responses and typically assume a fixed dialogue length in advance. In this paper, we study social bias dynamics in response-conditioned multi-turn interactions using a controlled evaluation protocol that generates follow-up user queries from the evolving dialogue history and allows evaluation over variable numbers of turns. Experimental results show that LLMs exhibit social bias even in coherent, response-conditioned multi-turn interactions, revealing late-emerging bias, non-monotonic bias patterns, and bias re-emergence. These results motivate evaluations that extend beyond fixed-turn, pre-scripted protocols. Our findings highlight the importance of analyzing social bias as a turn-level dynamic phenomenon. |
| 2026-09-16 | [Robot Visions: Breaking reCAPTCHA at Zero Cost and Zero Shot](http://arxiv.org/abs/2609.18518v1) | Suphannee Sivakorn, Samantha Gottlieb | Google reCAPTCHA is the most widely deployed visual CAPTCHA service, protecting hundreds of thousands of websites from automated bots. It serves as a critical line of defense against automated attacks, including credential stuffing, bulk account creation, and automated form abuse. It has proven largely effective since its introduction in 2007. However, the rise of accessible AI now threatens its efficacy. Prior work has demonstrated that commercial cloud-based vision-language models (VLMs) can solve visual CAPTCHA challenges, but at non-trivial monetary cost per attempt. In this paper, we show that free and locally-run models can break Google reCAPTCHA. We conduct a comprehensive study of reCAPTCHA and present a taxonomy of its challenge types: Type A (independent image tiles, with static and dynamic sub-variants) and Type B (a single image partitioned into a 4x4 grid), each demanding a distinct solving strategy.   We design zero-shot, no-cost solvers built entirely on open-source local models, specifically CLIP (58% per-challenge accuracy on Type A) and OWLv2 (43.5% on Type B), requiring no model training and no API access. Our end-to-end automated solver achieves a 92.6% per-session success rate across 500 real-world reCAPTCHA sessions. We further demonstrate that reCAPTCHA can be defeated by a non-technical adversary, using only natural-language instructions to a commodity AI assistant. This collapses the practical attacker skill floor to near zero and fundamentally changes the threat model for challenge-based CAPTCHAs. Although reCAPTCHA increasingly favors reputation-based verification, visual challenge-based fallback persists as a safety net that, paradoxically, has become the weakest link in the defense chain, suggesting that challenge-based visual CAPTCHAs may have reached the end of their useful life. |
| 2026-09-16 | [Beyond Routine Compliance: Cunning Data Cultivates Safety Vigilance in Large Language Models](http://arxiv.org/abs/2609.18515v1) | Youjia Wang, Lin Xu et al. | Safety alignment teaches large language models (LLMs) to recognize harmful requests and reject risky instructions. Yet aligned models can fail when harmful intent is concealed within seemingly benign contexts. Robust safety therefore requires both knowledge of safety boundaries and \textbf{vigilance}: the ability to detect unusual premises, misleading reasoning, and latent risks beneath surface-level semantics. Vigilance requires models to scrutinize a request's underlying intent and assumptions before acting. To cultivate this capability, we introduce \textbf{cunning questions}, which are not necessarily safety-related but contain misleading premises, atypical reasoning, or subtle inconsistencies. We hypothesize that learning to look beyond such reasoning traps can transfer to safety-critical scenarios. Experiments show that Cunning training improves robustness to out-of-distribution jailbreak attacks and strengthens subsequent safety fine-tuning. Furthermore, augmenting an existing state-of-the-art safety alignment pipeline with Cunning establishes a new state of the art across our evaluated settings, reducing mean ASR across nine backbone--benchmark combinations from 17.40\% to 15.05\%. Trace analysis after matched safety fine-tuning suggests that safety judgments are more likely to govern responses before harmful planning begins. A conditional theoretical analysis further characterizes when invariance learned from cunning data can transfer to safety-related inputs. These findings suggest that cunning data can strengthen model vigilance and complement conventional safety alignment. |
| 2026-09-16 | [Impact of Phase Unwrapping on Multitarget Acoustic Lenses for Transcranial Holography](http://arxiv.org/abs/2609.18495v1) | D. Attali, T. Tiennot et al. | Acoustic lenses have been introduced recently to compensate for the phase distortions induced by the propagation across a human skull for ultrasonic deep-brain stimulation in humans. In this study, we present bifocal lenses that compensate for human skull aberrations and allow simultaneous targeting of multiple structures deep in the brain. We investigated the impact of phase unwrapping in the design of the lenses and how this process improves the distribution of pressure produced in N=5 human skulls for two different spatial arrangements of the targets. The results show that unwrapping the phase computed during the design increases the fidelity of the pressure field generated across the human skulls. The spatial precision is on average improved by 73%, and out of target energy deposition is on average reduced by 58%. The results presented in this study highlight the importance of phase unwrapping to optimize the safety and efficacy of future transcranial ultrasound stimulations targeting multiple regions. |
| 2026-09-16 | [First Token Matters: Understanding Safety Collapse in Large Reasoning Models](http://arxiv.org/abs/2609.18471v1) | Yizheng Yang, Haining Yu et al. | Large Reasoning Models (LRMs) exhibit strong problem-solving abilities, yet their safety alignment often degrades when handling harmful queries. Existing approaches to improving safety largely rely on additional training or preference optimization, while offering limited understanding of the internal mechanisms behind safety failures. In this work, we investigate this failure through a token-level positional analysis of refusal dynamics and identify a localized vulnerability at the onset of reasoning, which we term Onset Refusal Collapse (ORC). We find that the refusal-related signal of LRMs drops sharply at the first generated token under harmful queries, which is associated with unsafe response generation. Motivated by this finding, we propose SafeToken, a lightweight inference-time intervention that injects a learned continuous safety anchor precisely at reasoning onset. Despite updating only a single token embedding, SafeToken effectively mitigates ORC, improves safety on harmful-query benchmarks, and largely preserves reasoning utility. These results suggest that safety failures in LRMs can arise from a transient breakdown at the critical transition from understanding to generation. |
| 2026-09-16 | [Hardware-Free Robotics Laboratories in Mixed Reality](http://arxiv.org/abs/2609.18434v1) | Santiago Berrezueta-Guzman, Habiba-Loai Khalil et al. | Teaching robotics relies on screen-based simulation, showing robot motion in an abstract coordinate frame rather than at real scale in the learner's own space, while access to physical hardware is limited by cost, safety, and scheduling constraints. We present MR-Robotics LAB, a mixed-reality (MR) platform that replays MATLAB-generated robot trajectories at real scale within the learner's physical environment. A browser-based service validates a MATLAB workspace file (.mat), normalizes units, and publishes a versioned JSON trajectory; a Unity application on a Meta Quest 3 then reproduces the authored joint configurations under position control and replays them at the declared frame rate within a physics-enabled scene that supports collision detection and end-effector grasping. A formative single-group evaluation with engineering students found that participants reported low setup effort (M = 4.67 on a 5-point scale) and perceived support for workspace understanding from multi-viewpoint inspection (M = 4.56), and 83% of participants affirmed their willingness to use the platform in an introductory robotics course. The evaluation instrument records only perceived outcomes, without counterbalancing or a learning measure, so no comparative advantage over desktop simulation is claimed. The contribution is a reusable simulation-to-MR trajectory pathway and design guidance for hardware-free robot visualization in engineering education. |
| 2026-09-16 | [Autonomy in Check: Governor-Mediated Adaptive Security at the Edge](http://arxiv.org/abs/2609.18338v1) | Ijaz Ahmad, Ijaz Ahmad et al. | Adaptive security at the network edge increasingly relies on automated planners, including rule-based controllers, learned policies, and LLM-assisted agents, that translate observations into enforcement actions. Once such a planner can influence live policy state, syntactic validity is not enough. A semantically wrong action, produced from incomplete or manipulated observations, can be faithfully executed by an enforcement substrate that cannot judge mission context. We address this problem by treating the boundary between planner output and kernel enforcement input as the primary security object. We propose a split-control architecture in which an untrusted planner emits typed security intents, a deterministic governor checks each intent against safety, resource, temporal-stability, and proportionality invariants, and only admitted actions are bound to signed receipts and compiled into pre-installed eBPF map updates. The paper formalizes this trust-boundary problem, defines three threat classes, develops the governor admission predicate, and reports an end-to-end prototype. Across rule-based and LLM-assisted planners on a Raspberry Pi 5 testbed connected to the university 5G Test Network, the governor admits, rejects, and bounds intents at microsecond cost without disrupting protected-flow regularity. The contribution is conceptual as much as empirical: adaptive security does not need to trust the author of an action. It needs a mediation boundary that decides whether the action is admissible. |
| 2026-09-16 | [Visual Compliance via Executable Safety Rule Entailment](http://arxiv.org/abs/2609.18328v1) | Jisoo Kim, TaeYoon Kwack et al. | Recent advances in LLMs and VLMs have enabled safety systems to reason beyond simple risk patterns toward more contextual and semantic safety concerns. However, as risk patterns continue to evolve and safety rules become more complex, existing training-based end-to-end safeguards face persistent challenges in adaptability and explainable reasoning over complex safety rules. To address these challenges, we propose GuardEn (Guarding by Safety Rule Entailment), an executable safeguard framework that decomposes safety policies into atomic propositions through Safety-Rule Compilation, modeling their composition as executable code. At test time, Scene-Grounded Execution instantiates these atomic propositions with contextual visual information derived from scene graphs, enabling rule-grounded and interpretable safety reasoning. Experiments on SafetyVisionBench demonstrate the effectiveness of programmable safeguard for complex visual safety assessment, achieving an average improvement of 9.8 F1 points over the strongest baseline. |
| 2026-09-16 | [A Study on the Impact of Natural Language Differences in Prompts on Automatic Code Generation Using LLMs](http://arxiv.org/abs/2609.18311v1) | Haruka Tokumasu, Masanari Kondo et al. | Large Language Models (LLMs) have demonstrated remarkable performance in automatic code generation tasks, thereby encouraging new research in this area. Although numerous studies have explored LLM-based code generation, the impact of the natural language in input prompts remains unexplored (language bias). This study aims to (1) quantify how the natural language of input prompts influences LLM-based code generation performance and (2) evaluate a mitigation strategy to reduce language bias in code generation. We assess code generation Accuracy on AtCoder, LeetCode, and BigCodeBench. To quantify the language bias on code generation, each problem is presented in English, Japanese, and Chinese. We use seven LLMs (GPT-4o, o3-mini, DeepSeek-V3.2, Llama-3, Qwen2.5-Coder-14B, Qwen2.5-Coder-0.5B, and GitHub Copilot) and assess their performance in terms of Accuracy (the number of problems for which generated code passes all test cases). We compare Accuracy before and after translation to evaluate the effectiveness of translation as a mitigation strategy. We observed that the natural language of problem statements affects LLM-based code generation performance. Specifically, the languages officially supported by each dataset achieved the highest median Accuracy. Also, translation improved Accuracy, but its effectiveness was not consistent across datasets and model types. We found that AtCoder contained a particularly high proportion of narrative-style problem statements and longer problem statements. Natural language significantly affects LLM code generation accuracy. Translation can mitigate language bias in some settings, but its effectiveness depends on the dataset and model type. Furthermore, the narrative aspects and context length of input prompts are important factors related to language bias and the effectiveness of translation as a mitigation strategy. |
| 2026-09-16 | [Building Trust in Artificial Intelligence: A Necessity for Railway Applications](http://arxiv.org/abs/2609.18278v1) | Lefebvre Renard Clément, Lébé Vincent et al. | Artificial Intelligence (AI) is currently only applied to non-safety critical applications due to the strict standards and regulations for railway industries. We propose to review the three main fields necessary to increase trust in data science and AI algorithms and reach compliance: robustness, Operational Design Domain (ODD), and explainability. Robustness is the ability of an AI system to maintain its level of performance under any circumstances (ISO24029). ODDs allow the explicit definition of operating conditions under which a system is intended to operate, according to the recently published DIN DKE SPEC 99004. Explainability is the property of an AI system to express important factors influencing the AI system results in a way that humans can understand. Those 3 domains of research are already well investigated by nonrailway actors, with algorithms and methods ready to use for railway applications. A system view is necessary to ensure all trustworthy requirements interact continuously in a safe MLOps environment thereby fostering acceptance from regulators, operators and the public. Beyond safeguarding safety-critical applications, we aim to show that fostering deep trust in AI, as now required by regulatory frameworks worldwide, will unlock its full potential and transform the pace of adoption across mission-critical domains. |
| 2026-09-16 | [A Recursive CBF Framework for Safety under State Uncertainty](http://arxiv.org/abs/2609.18268v1) | Rahal Nanayakkara, Aaron D. Ames et al. | The practical implementation of Control Barrier Functions (CBFs) for safety-critical control is often hindered by uncertainty in the knowledge of the state. While existing robust CBF methods address state uncertainty, they often lack recursive feasibility guarantees or fail when uncertainty levels are high, allowing the system to enter regions where no safe control input exists. To resolve this, we propose a novel framework of enforcing recursive CBFs. Rather than merely ensuring the invariance of the original safe set, this approach enforces the forward invariance of a subset of the safe region where a robustly safe control input is guaranteed to exist. This holistic framework ensures that the system never strays into ambiguous regions, providing continued feasibility and safety guarantees, regardless of the level of state uncertainty. |
| 2026-09-16 | [Indicators of resilience for autonomous control systems](http://arxiv.org/abs/2609.18264v1) | Jasper van Beers, Da-Hwi Kim et al. | As modern societies rely more on autonomous systems to facilitate daily life, assuring their safe operation is paramount. Naturally, there are many techniques available to predict and prevent system failures. However, the safety afforded by such schemes may become misaligned with the true system, which can change in unexpected ways - from partial faults to natural wear-and-tear - that subtly degrade its stability. The implications that such subtle changes have on autonomous system stability can be observed through generic indicators of resilience derived from critical slowing down, popular for anticipating catastrophic tipping points in natural systems. Here, we show how one can systematically design these generic indicators for nonlinear control systems and show how these can reflect loss of stability though simulations of canonical robotic systems wherein their proximity to instability is manipulated directly. These results are affirmed through real-world flight experiments of a quadrotor that is nudged towards instability by progressively damaging its propeller blades. Our results show that the implications of degraded resilience on closed-loop stability are evident well before they appear, for which the indicators of resilience derived here can provide an early warning. |
| 2026-09-16 | [Beyond Direct Sensing: Harnessing Indirect Observations from Third-Party Sensors in Vehicle Tracking](http://arxiv.org/abs/2609.18173v1) | Gaofeng Dong, Vamsi Eyunni et al. | Vehicle tracking is fundamental to applications ranging from urban mobility and public safety to security and defense. Conventional tracking relies on direct access to sensors that provide strong observations such as vehicle identity and location. In practice, however, factors such as ownership, privacy, cost, and operational constraints may limit directly accessible sensors, leaving sparse observations and long tracking gaps. Meanwhile, many additional third-party sensing assets may be present across the environment but remain inaccessible at the raw-data level, preventing their direct integration into the tracking system. In this work, we investigate whether weak, indirect observations with uncertain spatial and temporal cues can complement sparse direct sensing for vehicle tracking. Specifically, we propose GrayTrack, which fuses weak anonymous events with sparse direct observations using a road-constrained particle filter. We build a CARLA-Mininet-WiFi pipeline to evaluate the system under controlled conditions, generating direct observations from accessible cameras and indirect observations from third-party cameras. Our learning-based detector achieves an F1 score of 0.989 for anonymous vehicle passages. Further, incorporating indirect third-party observations reduces trajectory RMSE by 60.1% and catastrophic track loss from 35.8% to 0.3%. These results demonstrate that GrayTrack can effectively exploit weak indirect observations to extend tracking capabilities. |
| 2026-09-16 | ["Your Robot Was Trained on a Lie": Collision Mesh Poisoning Attacks on Robotic Manipulation](http://arxiv.org/abs/2609.18122v1) | Gengyang Xu, Dongwei Xiao et al. | Learning-enabled robotic manipulation increasingly relies on robot simulators for policy training and evaluation before real-world deployment. Inside a simulator, a 3D asset contains two separate geometries: a visual mesh used for rendering and a collision mesh used for physical interaction. For computational efficiency, the collision mesh is deliberately a coarse approximation that need not have the same geometry as the visual mesh, a legitimate and pervasive discrepancy we call the Visual--Collision Gap (V--C Gap). We show that the V--C Gap opens a new and practical attack surface, and propose Collision Mesh Poisoning (CMP), the first poisoning attack against robotic manipulation delivered through the 3D asset supply chain. An attacker modifies only the collision mesh of a 3D asset, leaving the visual mesh and all other components unchanged. A policy trained and evaluated with the poisoned asset behaves normally throughout simulation, yet degrades, fails, or creates physical safety risks once deployed in the real world. Since current asset review practices cover malware, copyright, and format compliance, but not visual--collision consistency, poisoned assets can be distributed through legitimate supply chain channels. We evaluate several defenses and our results show that they are insufficient to defend against CMP, highlighting the need for new defenses. |
| 2026-09-16 | [A Comprehensive Review of Generative Physical Artificial Intelligence](http://arxiv.org/abs/2609.18111v1) | Satyam Gaba, Krutiksinh Rana et al. | The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate training data for VLAs and DPMs, RFMs enable cross-platform deployment of learned policies, while LBMs provide motion priors for natural behavior. Through examples across autonomous vehicles, industrial automation, healthcare robotics, and humanoid systems, we identify significant performance improvements and summarize promising research directions in data-efficient learning, sim-to-real transfer, edge-compatible architectures, and safety frameworks. These insights advance embodied AI for IoT-connected environments where intelligent agents interact with networked sensors, actuators, and edge devices. |
| 2026-09-16 | [Linguistic Triggers of Gender and Racial Bias in Open-Weight LLMs Applied to Recruitment](http://arxiv.org/abs/2609.18106v1) | Kosuke Kitahara, Nobuhiro Yamaguchi | Open-weight large language models are rapidly entering hiring pipelines, yet their discriminatory failure modes -- and the regulatory exposure these create under the EU AI Act high-risk classification (Annex III) and U.S. EEOC adverse-impact analysis -- remain poorly understood. We present the first systematic, multi-model audit of open-weight LLMs that treats job-posting language as the primary experimental variable, evaluating six models (Llama 3.2, Mistral, Gemma 3, Qwen 3, Phi 3, DeepSeek-R1) across four controlled experiments that jointly probe recruiter-simulation and job-seeker-simulation tasks. We find that (1) agentic posting language depresses recruiter recommendation scores for female candidates (r_rb = 0.309, p_Bonf = 7x10^-5; model-fixed-effects r_rb = 0.448), while communal language partially reverses the penalty; and (2) coded-exclusion language suppresses non-White recruiter scores at large effect sizes (r_rb = 0.646-0.758) and, on the job-seeker side, selectively deters non-White personas from expressing interest -- operationalizing a chilling-effect mechanism at scale. A label-ablation experiment isolates the explicit demographic persona label as the primary causal driver, and Word Embedding Association Tests corroborate these findings at the representational level (d = 1.01-1.45 under Caliskan et al.'s multi-word gender attribute lists). We translate these results into a concrete pre-deployment audit protocol -- posting-vocabulary scoring, persona-conditioned LLM probing, and adverse-impact flagging against the four-fifths threshold -- that operationalizes the documentation and risk-management obligations Annex III imposes on high-risk AI in recruitment. |

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



