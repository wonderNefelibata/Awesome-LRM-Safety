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
| 2026-09-28 | [SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents](http://arxiv.org/abs/2609.35596v1) | Saswat Das, Parvati Viswanathan et al. | Self-evolving LLM agents have gained prominence for their ability to improve after deployment by modifying their harness, including their controller instructions, memory management protocols, and reusable tools and skills, in response to user and environment feedback. However, locally useful updates may persist into later tasks where they produce unsafe behavior, even without direct adversarial influence. To study this risk, we introduce SEABench, a benchmark for studying endogenous misalignment arising from agent self-evolution, with 48 longitudinal task sequences that span multiple evolution surfaces, task domains, and harm types in a rich personal-assistant environment. To account for the stochasticity inherent in agentic operations, we provide an adaptive trajectory discovery pipeline that probes for failures while preserving original task intent and supports causal attribution through paired non-evolving agents and attribution scores. Our evaluation across multiple recent LLMs, evolution surfaces, and harm types reveals that self-evolution indeed increases task completion rates but often at the cost of safety failures that are absent for paired non-evolving baseline agents. We also show that qualitatively different safety behaviors emerge across evolution surfaces and harm types. Further, we show that this divergence in safety behavior is reflected in agents' chain-of-thought reasoning, which yields an effective monitoring strategy that can mitigate unsafe behavior with a low false positive rate. |
| 2026-09-28 | [Less Sycophancy, Stronger Refusal? Lessons for AI Safety from Mechanistic Interpretability](http://arxiv.org/abs/2609.35544v1) | Xu Wang, Difan Zou et al. | Reliable refusal of harmful requests is essential to the safe deployment of language models. Because excessive eagerness to please users may undermine existing refusal capabilities, reducing sycophancy offers a potential route to stronger refusal beyond the harmful scenarios covered by safety training. We investigate this possibility using compensatory feature injection (CFI), a training technique designed to limit the acquisition of a target concept by supplying its associated activation during learning. Across three Qwen3.5 base models, we use sparse autoencoders (SAEs) to identify the top-ranked sycophancy feature from paired sycophantic and independent responses, then validate its behavioral influence through inference steering. We subsequently inject the selected feature during supervised fine-tuning on sycophantic targets. Positive injection reduces learned sycophancy after removal (by 62.0% relative to ordinary fine-tuning in 35B-A3B), whereas modest negative injection increases it. Unexpectedly, these reductions in sycophancy do not consistently improve direct refusal of harmful requests, motivating a narrower evaluation of the same harmful intents under user pressure. In this setting, ordinary fine-tuning on sycophantic responses substantially weakens refusal, while selected checkpoints trained with positive injection recover part of the loss, including approximately 95% in 35B-A3B. These findings show that persistent sycophancy reduction does not guarantee stronger direct refusal, while identifying recovery under user pressure as a distinct, conditional benefit of training intervention. |
| 2026-09-28 | [Physics-Guided Conditional Diffusion Model for Rare Event Synthesis and Diagnosis for the Water-Gas Shift Reaction](http://arxiv.org/abs/2609.35499v1) | Md Abrar Rafid Siddique, Bibek Aryal et al. | As the world moves towards sustainable energy sources, hydrogen (H2) can be treated as an eco-friendly alternative to fossil fuels due to its high energy density and zero carbon emissions. The water-gas shift (WGS) reaction is a widely used industrial process for hydrogen production by converting carbon monoxide and steam into hydrogen and carbon dioxide. However, occurrences like severe fouling, catalyst deterioration, and thermal runaway can hamper the reaction kinetics/process safety and decrease the yield of H2. These incidents are rare, and gathering process data under such abnormal conditions is challenging. In this work, we propose a physics-guided conditional diffusion model to generate realistic rare-event trajectories for the WGS reaction. The proposed model integrates a conditional denoising diffusion probabilistic model (CDDPM) with governing laws of the reaction to generate physically consistent process trajectories. The conditioning features allow the model to produce high-quality synthetic profiles for rare-event domains that are typically beyond the training regimes. The generated rare-event trajectories then augment the raw dataset for a balanced distribution between normal and abnormal conditions. We further propose a hazard score to assess the risk severity of the operating condition based on the operating trajectory. Deep learning models are trained with the augmented dataset to diagnose the health status of the reaction. Simulation results show that the proposed physics-guided diffusion model outperforms data-driven models in terms of the quality of synthetic data and diagnosis performance for rare events. |
| 2026-09-28 | [Adaptive Safety Filtering for Frozen ACC Policies via Conformal Residual Calibration](http://arxiv.org/abs/2609.35415v1) | Zhiruo Zhou, Rigaudiere Z. Li et al. | Frozen adaptive cruise control (ACC) policies can violate constraints when deployment dynamics differ from their training conditions. We propose residual-aware conformal action filtering (RACF), which calibrates residuals of a fixed nominal predictor and converts their quantile into an operating margin for finite-model action projection. Completed transitions update margins and candidate selection without retraining the policy. In a registered comparison over 2,400 controller-trial units, Adaptive RACF achieves 94.3% episode safety, improving by 19.9 percentage points over the evaluated nominal CBF-QP baseline while reducing projection frequency from 8.11% to 6.63%. A controlled study isolates a 4.54-point improvement from residual-margin injection. In a separate matched-hardware evaluation, Adaptive reduces mean amortized rollout time by 21.2% relative to Robust CBF-QP, with 161/180 versus 170/180 safe episodes. We characterize conditions linking one-step residual coverage to constraint satisfaction and quantify the observed safety-computation trade-offs. |
| 2026-09-28 | [Structural Alignment for Reliable Industrial AI: Bridging Physical Reality, Data, Models, and Human Intent](http://arxiv.org/abs/2609.35400v1) | Lizhi Xiao, Sihong Wu et al. | Artificial intelligence is increasingly deployed in critical industrial domains, including healthcare, energy grids, subsurface exploration, where failures can have severe consequences for human safety, system stability, and economic outcomes. Yet AI is still evaluated primarily through benchmark accuracy, a model-centric metric that fails to capture the structural complexity and risks of real-world deployment. We propose a framework that views industrial AI reliability as a problem of structural alignment across four interacting worlds: physical, representational, machine, and human cognitive. These worlds are connected through two interfaces: digitalization, linking physical reality to computational representations, and goal encoding, translating human cognition to the machine objectives. Together, they define the space of admissible solutions. We characterize the solution space through four attributes: existence, non-uniqueness, robustness, and interpretability and show how mismatches arise at interfaces and propagate across worlds to produce reliability failures. Applications to healthcare, energy grids, and subsurface exploration illustrate that although dominant failure modes differ across domains, for example, interpretability in healthcare, robustness in energy grids, and non-uniqueness in subsurface exploration, all originate from a shared structural mechanism. By shifting the focus from model-centric evaluation to system-level alignment, this framework offers a principled foundation for assessing and governing reliability in industrial AI systems. |
| 2026-09-28 | [Jailbreaks for Black-Box Uncertainty Quantification in Large Reasoning Models](http://arxiv.org/abs/2609.35350v1) | Lucas Biechy, Cédric Eichler et al. | While Large Reasoning Models (LRMs) excel at complex reasoning, alignment through reinforcement learning often induces systemic overconfidence. In production environments, where logits may be unavailable, robust black-box uncertainty quantification (UQ) is essential for trustworthiness and safety. Focusing on question-answering for LRMs, we show that existing black-box methods, such as paraphrase-based self-consistency and confidence verbalization, offer little to no improvement over simple repeated sampling, suggesting that alignment suppresses useful output variability. We introduce prompt-level relaxation operators that broaden the model's effective output distribution by approximating the effect of an optimal policy obtained with a stronger KL-regularization parameter, hence closer to the reference model. Theoretically, we demonstrate that relaxation improves calibration. We propose Jailbreak for Uncertainty (J4U), a jailbreak-derived technique for UQ that empirically reproduces the behavioral signatures predicted by our relaxation theory. Across 3 datasets and 4 LRMs, including a closed-source production model, J4U's improvement over repeated sampling achieves statistical significance in up to 6 times more LRM-dataset-metric settings than the strongest black-box UQ state-of-the-art baseline we evaluate, with average ECE reductions up to 5 times larger. These results provide a practical tool for UQ in black-box LRM deployment. |
| 2026-09-28 | [Reliability Engineering for AI Systems: Challenges, Methods, and Directions](http://arxiv.org/abs/2609.35316v1) | Rong Pan, Yili Hong et al. | AI reliability concerns whether an AI system performs its intended function dependably over a stated period and under stated operating conditions, with stated evidence. As these systems become more autonomous, that function includes more than a correct output. Retrieval, memory, tool use, permissions, human oversight, and interactions among systems must operate consistently and safely, and, for generative systems, so must the reasoning process that produces the output. Average benchmark accuracy measures capability; it does not quantify this broader reliability claim. This paper adapts established reliability engineering methods, from failure definitions and operational envelopes to FMEA, accelerated testing, field monitoring, and reliability growth, to AI systems. A four-level diagnostic framework classifies failures as component, operational-loop, agentic-conduct, or network and governance failures. Test, evaluation, verification, and validation (TEVV), sequential monitoring, and FRACAS create and refresh evidence. SMART provides statistical guidance for measurement, analysis, assessment, and test planning; the NIST AI Risk Management Framework provides organizational guidance for governance, evaluation, monitoring, and mitigation. Three cases illustrate the program: adversarial testing of a convolutional neural network, perception-error propagation, and autonomous-vehicle disengagements. Established reliability engineering provides a usable foundation; new measurements and safety guardrails are still needed as these systems are self-evolving. |
| 2026-09-28 | [Narrow Multimodal Fine-Tuning Can Induce Emergent Misalignment](http://arxiv.org/abs/2609.35291v1) | Shunchang Liu, Lukas Fluri et al. | Modern AI models are aligned through post-training to adapt them to downstream tasks. Recent work shows that fine-tuning language models on narrow tasks can induce emergent misalignment (EM), causing broadly harmful behaviors beyond the training task. However, EM has been studied almost entirely in text-only tasks, leaving its manifestation in multimodal models unclear. In this paper, we define and analyze EM in the context of vision-language models. We first induce EM via fine-tuning on narrow multimodal tasks targeting vulnerable code, careless household-object use, and conspiratorial interpretations of ordinary scenes. Across fifteen commercial and open-source models with different scales, we find that narrow multimodal fine-tuning can induce coherent and broadly misaligned behavior that transfers to unrelated tasks, including misaligned opinions, visual factual dishonesty, unsafe image generation, vulnerability to visual jailbreaks, and risky agentic actions. We further find that multimodal EM does not depend on the apparent harmfulness of training data but is sensitive to training-evaluation modality alignment. EM can arise under both supervised fine-tuning and preference optimization and can propagate through intermediate reasoning. Finally, we explore several mitigation strategies, including prompt inoculation, benign continued training, and activation-level steering, which can partially reduce EM. Overall, our findings suggest that multimodal EM reflects a behavioral shift rather than a general loss of capability, extending beyond text to the visual modality. |
| 2026-09-28 | [Imprint Reader: From Weight-Update Readout to Behavioral Intervention](http://arxiv.org/abs/2609.35261v1) | Guanxu Chen, Qihao Lin et al. | As language models take a growing role in AI development, a natural aspiration is for them to reflect on their own learning process, as humans do, and use that reflection to improve themselves. At the same time, these models have an advantage that human learners lack, since training leaves parameter-level traces that can, in principle, be inspected directly. However, current models cannot decode these traces into an explicit account of what they have learned. To this end, we introduce the \textit{Imprint Reader}, a model trained with \textit{Semantic Mount-and-Read Tuning} (SaRT) to describe frozen weight updates. SMaRT mounts each update onto the Reader and uses an anchor-free meta-query to elicit a natural-language description, while no-change and random-perturbation controls discourage unsupported claims. On held-out updates, the joint Reader reaches judge-based Pass@100 of $2\%$ for knowledge and $16\%$ for behavior. These results demonstrate the feasibility of natural-language readout while pointing to reliability across updates as the next step. Beyond free-form generation, the Reader provides a differentiable proxy for the gap between a specified target behavior and a candidate weight update. Its coordinate-aligned gradients support intervention through MetaEdit. At a $0.5\%$ pruning rate, Reader-guided selection raises measured harmful-prompt refusal from $57.9\%$ to $64.1\%$ under a safety-maintenance target. Using behavior descriptions without target-task training data, MetaEdit increases the frequency of backtracking and sub-goal expressions in mathematical reasoning traces and raises BFCL Overall from $41.69\%$ to $44.60\%$. |
| 2026-09-28 | [Analyzing Solana's Blocks and Transactions](http://arxiv.org/abs/2609.35171v1) | Yaron Hay, Dvir David Biton et al. | Solana is one of the most popular blockchains, and is arguably the most widely used blockchain for smart contracts, also known as dApps. Understanding the types of smart contracts that are being executed by Solana and their interplay is therefore highly beneficial both for designers of modern blockchains and developers of smart contracts. To that end, in this paper we analyze a million recent Solana blocks. We report statistics about the size of blocks (number of transactions per block), execution time units for individual transactions and fees, and invoked Solana programs. Further, based on the declared readset and writeset of each transaction, as mandated by Solana, we analyze the conflicts and corresponding conflict graphs arising within each block. These latter statistics are important to understand the potential for parallelism in the network, which is one of the main claimed benefits of Solana. The data and code are available in open source. |
| 2026-09-28 | [Tool Mediation Alters Refusal Mechanisms in Large Language Models](http://arxiv.org/abs/2609.35117v1) | Abel Rodríguez, Giuseppe Garofalo et al. | Large language models (LLMs) are increasingly deployed with access to external tools, yet harmful tool-mediated interactions are less likely to be refused when compared to regular conversational ones. As this change in refusal behavior remains underexplored, we investigate its underlying mechanisms across a diverse set of open-weight language models. We find that information about the harmfulness of a request remains strongly encoded in the model's representations and transfers across conversational and tool-mediated inputs. Evidence from representation geometry and neuron-level analysis further indicates that the two interaction modes systematically distribute harm-related computation differently. Crucially, while conversational inputs can be refused at relatively low levels of perceived harmfulness, tool-mediated inputs remain permissive until harmfulness crosses a substantially higher effective refusal threshold. Moreover, tool-mediated refusal is also more brittle: progressively weakening the refusal computation disrupts tool-mediated refusal at lower intervention strengths than conversational refusal, even when benign capabilities remain intact. Together, our findings indicate that tool mediation does not simply reduce the internal perception of harm, but instead impacts its conversion into refusal. Overall, this suggests tool-mediated environments may intrinsically reduce robustness of models to harmful requests, and that conventional safety evaluations may not fully transfer to LLM agents. |
| 2026-09-28 | [RefineDrive: Reliable Failure-Guided Learning for Vision-Language-Action Driving](http://arxiv.org/abs/2609.35078v1) | Zhe Sun, Ziyi Luo et al. | Vision-Language-Action (VLA) models for autonomous driving rely heavily on successful expert demonstrations, leaving model-specific failures underexploited. Learning from these failures is hindered by unreliable diagnoses, poorly matched correction targets, and coarse rewards. We propose RefineDrive, a failure-guided post-training framework that learns from self-generated failures through targeted supervision and safety-aware reinforcement learning. Reliable Diagnosis derives structured, verifiable feedback on collisions and drivable-area violations directly from simulator states. Minimum-Correction Target Retrieval searches a clustered human trajectory bank for nearby corrections that satisfy hard-safety constraints in the current scene, prioritizing preservation of the failed prediction's motion pattern. Conditioned on the driving context and failed trajectory, Correction SFT learns to generate the diagnosis followed by the retrieved correction as a training-only auxiliary task. We then apply GRPO with a Safety-Layered Reward that strictly prioritizes hard-safe trajectories, retains continuous safety feedback for both unsafe and hard-safe trajectories, and rewards driving progress only after hard safety is satisfied. At inference, the policy directly predicts trajectories from the driving context without an explicit diagnosis or repair stage. On NAVSIM v1, RefineDrive improves the 4B base SFT policy from 87.7 to 91.7 PDMS. Using the same checkpoint without additional training, RefineDrive achieves 89.4 EPDMS on the original NAVTEST scenes evaluated with NAVSIM v2 extended metrics. Controlled ablations support the benefits of structured diagnosis supervision, retrieved corrections, and safety-layered optimization for direct planning. |
| 2026-09-28 | [ORPG: Reconciling Multiple Reward Objectives through Objective-wise Policy Gradients](http://arxiv.org/abs/2609.34985v1) | Shicheng Fang, Yiwen Zhao et al. | Multi-reward policy optimization requires a joint update that reflects both the learning signals and the intended relationships among objectives. We introduce Objective-wise Reconciled Policy Gradient (ORPG), which constructs a separate clipped policy objective for each reward and reconciles the resulting gradients into one policy update. For compatible gradients, a cosine-dependent interpolation coordinates their contributions through a partially normalized reference while preserving the norm of their sum. We characterize this update as the unique solution of a spherical directional compromise. For conflicting gradients, projection follows the task's priorities. We evaluate the same compatible rule in helpfulness--safety alignment and correctness--cost optimization for mathematical reasoning. ORPG substantially improves average Useful and Harmless scores over the strongest external baseline on each axis. In mathematics, it achieves the highest average full-budget accuracy and three-budget hypervolume among the compared methods, with more accurate and shorter responses than the initial policy. Component comparisons and training dynamics show the larger contribution of compatible coordination and a complementary benefit from conflict handling. These results support gradient reconciliation for objectives with equal standing and for objectives with an explicit priority. |
| 2026-09-28 | [See it, Say it, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of Emergent Misalignment in LLMs](http://arxiv.org/abs/2609.34970v1) | Weiqiao Que, Ruizhe Li et al. | Safety-aligned LLMs can exhibit emergent misalignment (EM): narrow domain adaptation unexpectedly triggers catastrophic safety failures across unrelated domains. Prior static analyses leave training dynamics unmapped, while existing defenses rely on heuristics that degrade utility. We present a dynamic, second-order geometric study of EM. Tracking training trajectories reveals that directional Hessian curvature concentrates sharply on semantic pivot tokens. Grassmannian projections show that, in most settings, harmful-safe gap widens mainly because safe-gradient overlap declines. Leveraging these insights, we introduce a parameter-level Geometric Mitigation Framework that orthogonally projects empirical harmful gradient subspace out of parameter updates. On Qwen2.5-14B-IT, our defense suppresses free-generation EM by up to 80.0%; across the other three of four open-weight instruction-based model families (3B--20B), where single-layer behavioral EM is already near zero, teacher-forced evaluation shows same harmful subspace controls the conditional support of frozen EM responses. Crucially, these diagnostics unmask the illusion of behavioral safety: the same subspace remains measurable and steerable in models where behavioral EM is near zero. Code: https://github.com/WeiqiaoQUE/mechanistic-emergent-misalignment. |
| 2026-09-28 | [Safe Greenhouse Climate Control Using Lagrangian-Constrained PPO with Kolmogorov-Arnold Networks](http://arxiv.org/abs/2609.34966v1) | Hangzun Liu, Yuling Fan et al. | Greenhouse climate control balances economic return with maintaining temperature, humidity and CO2 within crop-adapted growth ranges. Conventional reinforcement learning (RL) greenhouse controllers use fixed reward penalties to limit climate constraint violations, yet such heuristic penalties cannot explicitly constrain long-term cumulative violations. Poorly tuned weights either lead to overly conservative policies and lower yields, or fail to suppress persistent climate deviations that harm photosynthesis and induce crop diseases. To address this issue, we formulate greenhouse climate regulation as a Constrained Markov Decision Process (CMDP) and use a Lagrangian safe RL framework RCPO-PPO to separate economic optimization and cumulative safety constraints, enabling adaptive penalty adjustment without manual tuning. To handle strong nonlinear, time-varying coupling between greenhouse microclimate and crop growth, Kolmogorov-Arnold Networks (KANs) replace Multi-Layer Perceptrons (MLPs) as policy and value approximators for improved nonlinear representation. Sinusoidal cyclic time features are embedded in observations to capture diurnal environmental periodicity. Simulations use a classic winter lettuce greenhouse model driven by 40-day real weather disturbances. Compared with vanilla penalty-based PPO, our method cuts cumulative climate violations by 18.65% and raises lettuce economic profit by 2.91%, keeping violations stable near the safety threshold. This decoupled CMDP optimization with KAN-based policy representation mitigates long-term climate risks and boosts planting profits, offering a constraint-aware control strategy for precision greenhouse cultivation. |
| 2026-09-28 | [DeShortcut-Align: Decoupling Spurious Shortcuts for Robust Safety Alignment in Large Reasoning Models](http://arxiv.org/abs/2609.34896v1) | Qirui Liu, Yichen Sun et al. | Safety alignment of large reasoning models (LRMs) via supervised fine-tuning (SFT) and reinforcement learning (RL) often yields near-perfect safety scores, yet this apparent success comes at the cost of severe over-refusal and degraded general capabilities. Through systematic empirical analysis, we find that these failures are closely associated with the learning of spurious shortcuts rather than robust intent-sensitive safety evaluation. Specifically, we identify two dominant shortcuts: formatting shortcuts, where refusal behaviors are overly bound to structural prompt templates that frequently appear in safety alignment corpora; and lexical shortcuts, where sensitive keywords reflexively trigger refusals on benign queries. To mitigate reliance on these shortcuts, we propose DeShortcut-Align, a shortcut-decoupling alignment framework that reduces dependence on superficial cues. DeShortcut-Align operates across three coordinated stages: (1) Refusal Sensitivity Attribution, which masks input tokens to quantify their impact on the final refusal response distribution; (2) Attribution-Guided Contrastive Augmentation, which constructs benign contrastive samples using high-sensitivity tokens to mitigate lexical shortcuts; and (3) Counterfactual Consistency Regularization, which constructs template-ablated states via attention blinding to enforce decision consistency across SFT and RL, mitigating formatting shortcut dependence. Experiments on 7B and 14B models demonstrate that DeShortcut-Align significantly improves robustness against template-stripping bypass attacks (reducing performance drops by up to 72%), substantially reduces over-refusal by over 58%, and better preserves general-purpose reasoning capabilities, thereby mitigating the alignment tax commonly observed in safety training. |
| 2026-09-28 | [Synthesizing Update Schedules with Game-Based Extension of Bounded Model Checking](http://arxiv.org/abs/2609.34878v1) | Janis Kröger, Paul Kröger et al. | Ensuring safe software updates in safety-critical systems without interrupting operation and without provisioning and activating cold spare hardware poses a fundamental challenge due to the conflict between system availability and update execution. In this paper, we present a bounded SMT encoding for synthesizing fixed global-time update schedules for timed-games with linear update automata and a fixed number of update transitions. We model the interaction between the system and the update as a two-player timed game. Our key contribution is the synthesis of global time points that define a fixed update schedule which guarantees safe and complete deployment of the update independently of the autonomous system behavior. To this end, we reduce the scheduling problem to a reachability and safety objective and encode it as a quantified SMT problem. We demonstrate it on an example system of a trajectory planner for autonomous driving, showing that the synthesized schedule ensures safe deployment under all admissible executions. |
| 2026-09-28 | [Verifying Graceful Degradation in a Distributed Malware-Detection System with SPIN](http://arxiv.org/abs/2609.34873v1) | Andrei Aldea, Dumitru-Bogdan Prelipcean | Modern endpoint malware detection is distributed: a lightweight agent on each endpoint collects features from a scanned file or process, sends them to a remote server for analysis, and then enforces the returned verdict locally by blocking, quarantining, or disinfecting. Because the endpoint acts on the verdict, the distributed machinery surrounding detection must never turn a transient server failure into a wrong action. We present a formal model, in Promela, of the endpoint decision pipeline of such a system, abstracted from a production architecture at Bitdefender. The model captures the system's graceful-degradation fallback chain: when the primary analysis server times out, the endpoint falls back to an older legacy-protocol server, and failing that to a reduced-signature local scan, before enforcing a verdict. Assuming detection signatures are sound, we specify six safety and liveness properties in linear temporal logic (LTL) and verify them exhaustively with the SPIN model checker. We prove that the fallback machinery never causes a false positive (an enforcement action against a benign file), commits to exactly one verdict per scan even when timed-out responses arrive late, weakens detection strength only in an explicit and ordered way, and always terminates in an enforcement decision, so the pipeline is deadlock-free. Each property is checked to hold non-vacuously, and we report how the state space grows with concurrent scans and endpoints. The work shows how model checking can give strong correctness guarantees for the failure-handling logic of a production security system, a layer that has received little direct formal attention.  |
| 2026-09-28 | [Certified Compilation in the TELEPERM XS Nuclear Safety I&C Platform](http://arxiv.org/abs/2609.34871v1) | Alexandre Berard, Richard B. Kreckel | The large safety instrumentation & control (I&C) systems in civil nuclear power plants (NPPs) are mainly safe-shutdown systems (reactor protection) or limitation and control systems. Framatome's established TELEPERM XS (TXS Core) product family is a digital I&C system platform to cover all these applications. We illustrate the role of verification in the different stages of the software production toolchain, focus on the formal compilation process, and discuss the contribution of the CompCert certified compiler to the safety case of the product. Scrutinizing the object code produced by this compiler has exhibited suboptimal run-time performance in a certain simple but recurring generated code pattern. We explain how formal methods allow us to address this issue in the compiler while simultaneously reducing its trusted computing base (TCB), thereby strengthening the safety case rather than merely preserving it. |
| 2026-09-28 | [Applying Language Models in medical Medicine: Recent Trends and Perspectives](http://arxiv.org/abs/2609.34780v1) | Erik Aerts | The use and applicability of artificial intelligence (AI) in medical research and clinical practice has received increasing attention in the literature over recent years. The emergence of large language models (LLMs) has expanded discussions in regards to applications of AI within healthcare. While traditional deep learning based AI applications in medicine have often focused on specific and defined tasks, LLMs offer broader capabilities and flexibility in working with available data,. At the same time of writing, the integration of LLMs into medical settings raises important questions regarding their reliability, accuracy, transparency, safety, and appropriate role in a medical setting. This text presents and discusses recent talks and articles concerning the application of LLMs in medicine, with particular emphasis on their potential utility in research and clinical practice. It considers both the opportunities offered by these technologies and the challenges associated with their implementation, aiming to provide a perspective on the current and emerging role of LLMs within the medical field. |

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



