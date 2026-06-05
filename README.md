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
| 2026-06-04 | [HANDOFF: Humanoid Agentic Task-Space Whole-Body Control via Distilled Complementary Teachers](http://arxiv.org/abs/2606.06493v1) | Lizhi Yang, Junheng Li et al. | For a humanoid robot to be deployed in the real world, the choice of command space (i.e., the interface between task planning and whole-body control) is crucial. Existing whole-body controllers typically demand dense kinematic or spatial references that planners struggle to synthesize from task semantics. We instead propose a compact, explicit interface that is intuitive, general, modular, and expressive enough for diverse manipulation skills. To this end, we introduce HANDOFF, a single humanoid whole-body controller that follows this interface and is distilled via multi-teacher KL distillation under a context-conditioned gating scheme into a mixture-of-experts student from three complementary specialists: whole-body motion tracking with safety-filtered data, locomotion, and fall-recovery. On the Unitree G1, HANDOFF matches state-of-the-art velocity tracking and offers one of the largest robust manipulation workspaces. We further demonstrate hardware feasibility through multiple natural-language-driven task roll-outs, powered by a VLM-driven agentic planner with no task-specific data or controller fine-tuning. |
| 2026-06-04 | [Computational Modeling of Human Adaptation in Urban Infrastructure Management under Extreme Conditions: A Case Study of Subway Flood Scenarios](http://arxiv.org/abs/2606.06429v1) | Jinfeng Lou, Zijie Liang et al. | Decision-making in urban infrastructure management during extreme events relies heavily on human operators, yet current computational support systems often fail to account for non-monotonic human adaptation and latent psychological biases like overconfidence and defensive overcorrection. This study addresses this gap by integrating Instance-Based Learning Theory (IBLT) into the domain of civil engineering computing. We establish a computational cognitive architecture that simulates operator decision processes through the mathematical mechanisms of memory retrieval and utility blending. This model functions as a computational baseline, representing boundedly rational adaptation driven by experiential priors, thus allowing for the algorithmic isolation of latent psychological biases from the baseline dynamics of memory-based learning. We demonstrated this framework using a human-in-the-loop microworld experiment simulating subway flood-induced track suspensions, where dispatchers must balance passenger safety against service efficiency. Analysis revealed a complex, non-linear human adaptation cycle consisting of four phases: acquisition, overconfidence, overcorrection, and recalibration. Specifically, the computational model exposed a significant divergence during the post-accident "overcorrection" phase: while human operators exhibited immediate, defensive risk overestimation, the model maintained a stable trajectory based on accumulated experience. This strategic divergence confirms that operational instability following failure is often attributable to acute psychological bias overriding stable memory-based adaptation, a pattern theoretically expected to recur across analogous high-stakes environments and validatable through multi-modal behavioral and sensor data from professional operators. |
| 2026-06-04 | [RiskFlow: Fast and Faithful Safety-Critical Traffic Scenario Generation](http://arxiv.org/abs/2606.06423v1) | Qi Lan, Yining Tang et al. | Safety-critical traffic scenario generation is essential for evaluating autonomous driving systems under rare but high-risk interactions. Existing diffusion-based methods offer strong controllability in closed-loop generation, but their iterative denoising process is computationally expensive and may accumulate sampling and guidance errors over long rollouts, causing unrealistic motion artifacts such as jitter, abnormal acceleration, and off-road behavior. To address these issues, we propose RiskFlow, a closed-loop safety-critical multi-agent traffic generation framework that formulates future trajectory generation as transport in the action space. Instead of relying on iterative denoising, RiskFlow learns an average velocity field over a finite interval to transform Gaussian action sequences into future acceleration and yaw-rate commands with a single forward pass, using a JVP-based objective for efficient and stable training. At test time, RiskFlow applies output-space guidance to the generated actions, steering selected critical agents toward risky interactions while regularizing off-road behavior, and reconstructs physically feasible trajectories through vehicle dynamics. Experiments on nuScenes with tbsim closed-loop evaluation show that RiskFlow achieves a strong adversariality-realism trade-off across multi-agent and long-horizon settings. Compared with representative baselines, RiskFlow consistently improves realism while maintaining competitive safety-critical generation capability, and substantially reduces inference time for evaluation. |
| 2026-06-04 | [A q-Tsallis Safe Approximation for Chance-Constrained Programs](http://arxiv.org/abs/2606.06401v1) | Sergio Assunção Monteiro, Fabricio Alves Barbosa da Silva | Classical chance-constrained programs are solved by safe approximations based on the empirical CVaR, which uses a uniform measure over scenarios and systematically underweights tail events under heavy-tailed distributions. We introduce \emph{q-CCP}, a non-extensive safe approximation grounded in the Riemannian geometry of the Tsallis statistical manifold: the rank-based q-CVaR escort weights are the $g^{(q)}$-geodesic projection onto the tail simplex face, and the q-CCP feasible set is a Tsallis-divergence ball (Proposition~12). This geometric foundation yields three results. First, q-CCP is a provable strict tightening of CVaR-CCP for all $q > 1$ (Theorem~7). Second, the empirical violation ratio satisfies $ρ(q) = [1-(1-\varepsilon)^{q+1}]/\varepsilon$, independent of the tail index $ν$ (Proposition~10). Third, the feasible-region volume cost is monotone increasing in $q$ and $ν$ (Proposition~11), providing a data-adaptive safety knob. The formulation inherits convexity and coherence from the q-CVaR functional and admits an iterative LP reformulation converging in 2--3 iterations. Experiments on 15 Ibovespa equities confirm the theory (violation ratio $0.241$, $q^* = 1.50$); an M5 inventory newsvendor experiment generalises the method to supply chain ($q^* = 1.88$, cost premium $1.155\times$, zero OOS stockout violations). |
| 2026-06-04 | [Risk Assessment of Autonomous Driving: Integrating Technical Failures, Ethical Dilemmas, and Policy Frameworks](http://arxiv.org/abs/2606.06396v1) | Boyi Chen, Shengqin Chu et al. | Autonomous driving technology has the potential to reduce the large number of road traffic accidents caused by human error each year, but it also brings new types of risks that need to be evaluated from the aspects of technology, ethics and regulations. Based on public crash data from the National Highway Traffic Safety Administration (NHTSA), disengagement reports from the California Department of Motor Vehicles (DMV), the MIT Moral Machines dataset, and a comparative regulatory analysis of five jurisdictions, we have found that the main types of technical failure modes are perception and classification errors. These account for a relatively large proportion of the reported accidents, and it can be concluded that there are different ethical frameworks for autonomous vehicle decision-making, and inconsistent regulations in different areas increase the uncertainty of widespread application. Generally speaking, the problems of technology, ethics and regulation are closely related and need to be solved together. Therefore, this paper recommends a more adaptive and cooperative governance approach that combines engineering standards, ethical discussion, and institutional supervision. |
| 2026-06-04 | [Ensuring Interaction Safety in Multitask Exoskeleton Control: A Simulation-Trained Variable Impedance Framework](http://arxiv.org/abs/2606.06370v1) | Muyuan Ma, Houcheng Li et al. | Wearable exoskeletons can augment human phys ical capabilities during complex activities. However, ensuring adaptation across diverse tasks while guaranteeing interaction safety remains a critical challenge. To address this, a simulation trained variable impedance control approach with stability guarantees is proposed. First, a simulation-based human exoskeleton motion data generation pipeline is established, utilizing Proximal Policy Optimization (PPO) to synthesize human muscle activations while the exoskeleton provides direct compensation for human biological joint torques. Subsequently, the generated dataset is used to train a dual modality policy that fuses semantic instructions with proprioceptive history, enabling the prediction of reference trajectories and variable impedance gains for nine different motion tasks. To guarantee safety, the network outputs are constrained by a stability criterion derived from Lyapunov stability theory, which bounds stiffness variations to ensure the asymptotic stability of the coupled system. Experimental results indicate that the proposed framework reduces metabolic cost in real-world scenarios com pared with standard baseline methods. These findings suggest the feasibility of the proposed framework for safe, multitask exoskeleton control. |
| 2026-06-04 | [Where Should Knowledge Enter? A Layered Framework for Knowledge Infusion in Multimodal Iterative Generative Mo](http://arxiv.org/abs/2606.06356v1) | Renjith Prasad, Chathurangi Shyalika et al. | Multimodal generative models produce fluent outputs but remain unreliable when generation must respect structured, domain-specific, or safety-critical knowledge. Existing methods incorporate knowledge through mechanisms such as prompt augmentation, guidance, latent editing, or fine-tuning, yet they are typically categorized by technique rather than by the component of the generative process they modify. We argue that knowledge infusion in iterative generative models is fundamentally anintervention-layer problem. Since thegenerative process unfolds as a trajectory of internal states, knowledge can act on four structurally distinct components of this process: the input/output boundary, the transition function, the intermediate state, and the model parameters. This maps to four intervention layers: surface, trajectory, latent, and parametric infusion. We instantiate the framework in diffusion models, map representative methods to all four layers, and derive design principles for multi-layer composition. In a controlled safety-alignment experiment using a multimodal knowledge graph with two diffusion backbones, we implement three of the four layers cumulatively, surface (input-side and output-side) and trajectory--latent (mid-generation). We show empirically that each additional layer addresses failure classes that prior layers cannot reach, reducing knowledge-violating outputs by 70.97% compared to vanilla generation and empirically confirming the framework's complementarity prediction. |
| 2026-06-04 | [Steering LLM Viewpoints through Fabricated Evidence Injection](http://arxiv.org/abs/2606.06244v1) | Xi Yang, Chang Liu et al. | As chatbots increasingly influence daily decision-making, their potential to produce misleading responses poses substantial risks to users. This paper investigates a critical cognitive vulnerability in LLMs: their tendency to uncritically trust external context when presented with fabricated evidence bearing markers of credibility. We introduce Ghostwriter, a two-phase attack framework that first repackages misleading statements with fabricated rationales, then instruct target LLMs to incorporate these viewpoints when responding to relevant queries. Experiments on BBQ, ToxiGen, and our specialized dataset reveal that commercial LLMs without external safety classifiers remain highly vulnerable, while even frontier classifier-guarded models (e.g., GPT-5.4) reduce but do not eliminate the attack. Building on this, we explore multiple defense strategies, among which a tailored safety policy enables gpt-oss-safeguard to achieve 81% detection rate. |
| 2026-06-04 | [From Reward-Hack Activations to Agentic Risk States: Context-Calibrated Mechanistic Monitoring in LLM Agents](http://arxiv.org/abs/2606.06223v1) | Patrick Wilhelm, Odej Kao | Language-model agents act through repeated cycles of observation, reasoning, and action selection, making safety monitoring depend on both internal model state and environment context. We study reward-hacking monitors in ReAct-style agents acting in Gameable ALFWorld and WebShop. Agents are instrumented with activation-based reward-hack scores, token-level entropy, and decision-context features. We find that adapters fine-tuned on \textit{School-of-Reward-Hacks} dataset can transfer reward-hack tendencies into agentic action selection, especially when the environment exposes proxy-reward affordances. However, mitigating such behavior cannot rely on activation dynamics alone. High reward-hack activation identifies a latent policy state, but does not necessarily imply an immediate exploit action. Across next-step prediction tasks, entropy and context-calibrated internal features improve risk estimation over reward-hack activation alone. Activation-direction steering further reduces proxy-exploit behavior in selected mixed-adapter regimes. Overall, our results support context-calibrated internal monitoring for agents: reward-hack activation identifies a latent policy state, while entropy and decision context help determine when that state becomes risky action. |
| 2026-06-04 | [CLEAR: Cognition and Latent Evaluation for Adaptive Routing in End-to-End Autonomous Driving](http://arxiv.org/abs/2606.06219v1) | Yining Xing, Zehong Ke et al. | End-to-end autonomous driving models often struggle to balance multi-modal maneuver generation with real-time inference constraints. While diffusion models successfully capture diverse driving behaviors, their iterative denoising process incurs unacceptable latency for safety-critical deployment. To address this, we propose CLEAR (Cognition and Latent Evaluation for Adaptive Routing), a framework that combines ultra-fast generative planning with deep semantic reasoning. CLEAR employs Drive-JEPA as the visual encoder and replaces the multi-step denoising chain with a single-step conditional drift in a VAE latent space, introducing a conditioning coefficient to balance diversity and expert precision. Meanwhile, we fully fine-tune Qwen~3.5~0.8B on driving QA pairs to extract scene-aware hidden states. These states guide both an Adaptive Scheduler, which selects the conditioning coefficient $α$ and sample count $N$ from a discrete set of predefined schemes, and a cross-attention scorer that selects the optimal trajectory from candidates. On the NAVSIM v1 benchmark, CLEAR achieves a state-of-the-art PDMS of 93.7. Our results demonstrate that high-fidelity, multi-modal planning can be executed efficiently without dense geometric annotations or iterative sampling. |
| 2026-06-04 | [Evaluating Agentic Configuration Repair for Computer Networks](http://arxiv.org/abs/2606.06212v1) | Rufat Asadli, Benjamin Hoffman et al. | Misconfigurations in computer networks remain a major source of critical Internet outages. Research is turning to Large Language Models (LLMs) to automate the complex, error-prone task of network configuration. However, even state-of-the-art models fail to resolve misconfigurations in large-scale, complex scenarios and often introduce new errors. In this work, we benchmark open- and closed-source LLMs augmented with formal network verification and context retrieval tools. We demonstrate that agentic architectures outperform base LLMs in repair efficacy (by 12% on average) and safety (by 17% on average), enabled by the ability to dynamically manage context and iteratively validate configuration repairs. |
| 2026-06-04 | [Unsupervised Pattern Analysis in Japanese Veterinary Toxicology: A Regulatory-Compliant Framework for Cross-Species Risk Assessment](http://arxiv.org/abs/2606.06207v1) | Yukiko Kawakami, Mohammad Shirazi et al. | Veterinary pharmacovigilance systems are essential for monitoring adverse drug events (ADEs), yet existing approaches often fail to capture region-specific toxicity patterns shaped by local biological and regulatory contexts. In Japan, these challenges are amplified by species-specific metabolic differences and reporting practices defined by the Ministry of Agriculture, Forestry, and Fisheries (MAFF). Most prior work relies on prediction-oriented models, limiting mechanistic interpretability. This study proposes a regulatory-integrated unsupervised framework for pattern discovery using the National Veterinary Assay Laboratory (NVAL) database. ADEs are encoded into organ system-aligned representations and adjusted for species-specific reporting biases, enabling cross-species comparison. Similarity-based clustering and dimensionality reduction are applied to identify latent toxicity structures. Analysis of 4,120 high-confidence ADE reports (9,080 drug-ADE combinations) identified three significant species clusters (p < 0.01), including hepatic-dominant patterns in companion animals (0.42 $\pm$ 0.06), renal toxicity in ruminants (0.39 $\pm$ 0.07), and dermatological sensitivity in sheep (0.35 $\pm$ 0.07). Drug-level clustering achieved 83% alignment with pharmacological classes, while cosine similarity outperformed alternative metrics (silhouette score: 0.48; cluster precision: 87%). Regulatory validation showed strong agreement with established classifications. These findings demonstrate that regulation-aligned unsupervised analysis can uncover biologically meaningful, region-specific toxicity patterns, providing an interpretable and scalable framework for veterinary drug safety assessment. |
| 2026-06-04 | [RedEdit: Agentic Red-Teaming of Image Safety Classifiers via MCTS-Guided Photo-Editing](http://arxiv.org/abs/2606.06140v1) | Weilin Lin, Ziqi Lin et al. | Image safety classifiers serve as a critical component of contemporary content moderation systems on the internet. However, their resilience against user-style malicious image editing remains underexplored. Such behaviors are highly prevalent in daily scenarios but difficult to fully reproduce. To explore this vulnerability, we introduce RedEdit, a novel black-box red-teaming agent that formulates photo-editing evasion as a combinatorial search problem over edit-tool sequences. It adopts a Vision-Language-Model (VLM)-based proposer to generate semantically targeted candidate edits and a Monte Carlo Tree Search (MCTS) planner to prioritize promising edit paths while backtracking from ineffective ones. Together, the proposer and planner instantiate two key capabilities of human attackers, i.e., domain knowledge and iterative backtracking, respectively, to reproduce this practical threat. Our extensive experiments on UnsafeBench reveal profound systemic vulnerabilities: fewer than two edits on average enable 76.2% of unsafe images to evade detectors, while retaining 93.0% malicious semantics, meaning that such manipulated content remains perceptually malicious to humans while easily bypassing automated moderation. We therefore appeal to the community for more attention to this overlooked practical threat. |
| 2026-06-04 | [TLA-Prover: Verifiable TLA+ Specification Synthesis via Preference-Optimized Low-Rank Adaptation](http://arxiv.org/abs/2606.06133v1) | Eric Spencer, Arslan Bisharat et al. | TLA+ is a formal specification language for verifying distributed systems and safety-critical protocols. Large language models (LLMs) frequently produce TLA+ specifications that fail the TLC model checker for semantic reasons. Across 25 LLMs, the best public baseline is 26.6% syntactic parse and 8.6% semantic model-check. We present TLA-Prover, a 20-billion-parameter model for TLA+ specification synthesis. Training combines supervised fine-tuning (SFT) on verified examples with repair-based group-relative policy optimization (GRPO). In the GRPO stage, the model learns to fix its own rejected specifications. We also train a direct preference optimization (DPO) variant from the same SFT checkpoint as an ablation. TLC provides the reward signal directly, with no learned reward model. Four tiers grade each output: Bronze (parses), Silver (no warnings), Gold (passes TLC), and Diamond. To reach Diamond, the model's correctness property is automatically altered in a small way; TLC must then detect a violation. If TLC still passes, the property was always-true and contributes nothing; the output fails Diamond. TLA-Prover reaches 9/30 (i.e. pass@1 = 30%) at both Gold and Diamond on a held-out 30-problem benchmark. This is roughly 3.5x the 8.6% untuned baseline. The DPO variant reaches 20% at Diamond. Gold and Diamond coincide at every checkpoint; this prevents the trivial-property failure mode. |
| 2026-06-04 | [Towards Healthy Evolution: Exploring the Role and Mechanisms of Human-Agent Interaction in Self-Evolving Systems](http://arxiv.org/abs/2606.06114v1) | Dianxing Shi, Junqi He et al. | Self-evolving agents improve through continual self-play and self-generated learning signals, but autonomous evolution can also cause capability degradation and safety drift. Although human feedback has proven effective for static and post-trained agents, its role in self-evolving systems remains underexplored. We introduce Agent Norm Correction through Human-like Oversight and Review (ANCHOR), an LLM-based framework that simulates human supervision and delivers feedback at various phases of self-evolution. With ANCHOR, we evaluate two representative open-source self-evolving agent systems across coding, mathematical reasoning, and safety. Our results show that even limited supervision substantially mitigates safety degradation while preserving stable performance on core evolutionary objectives. Further analysis shows that supervision over the output verification phase is the most effective for intervention, whereas increasing supervision frequency yields diminishing returns. These findings provide empirical evidence and practical guidance for designing more stable, controllable, and human-aligned self-evolving agent systems. |
| 2026-06-04 | [CogManip: Benchmarking Manipulative Behavior in Multi-Turn Interactions with Large Language Model](http://arxiv.org/abs/2606.06099v1) | Zeyang Yue, Chenfei Yan et al. | Whether Large Language Models (LLMs) exhibit covert psychological manipulation in complex human-AI interactions has garnered increasing safety concerns. However, existing AI safety benchmarks remain largely restricted to explicit rule compliance and static prompts, failing to capture the dynamic and covert nature of manipulative strategies in multi-turn dialogues. We introduce CogManip, a comprehensive benchmark that evaluates 15 manipulation strategy risks across 1,000 multi-turn interaction scenarios, validated by human experts. A systematic evaluation of 13 representative models, including frontier models like GPT-5.4 and DeepSeek-V3.2, reveals significant risk heterogeneities and illuminates the targeted direction for future defense. Further analysis of objective function perturbation reveals that DeepSeek-V3.2's manipulation tactics are highly sensitive to both negative and benign system prompts, demonstrating the critical necessity of prompt-based defense engineering and implicit goal auditing. CogManip offers a robust instrument and perspective for auditing the implicit psychological influence and dynamic strategy selection of modern LLMs. |
| 2026-06-04 | [Beyond Similarity: Trustworthy Memory Search for Personal AI Agents](http://arxiv.org/abs/2606.06054v1) | Jiawen Zhang, Kejia Chen et al. | Personal AI agents increasingly rely on long-term memory to provide persistent personalization across sessions. However, existing memory pipelines are largely driven by semantic similarity: memory data close to the current query is retrieved and injected into the model context. This creates a critical trustworthiness gap, since a semantically related memory may still be contextually inappropriate, leading to threats such as cross-domain leakage, sycophancy, tool-call drift, or memory-induced jailbreaks.   In this paper, we study memory search as a trust boundary in personal AI agents. We evaluate representative agentic memory frameworks, including A-Mem, Mem0, and MemOS, together with OpenClaw, a real-world personal-agent environment with persistent state and tool-use capability. Our results show that long-term memory is not merely a utility layer, but a durable control channel that can reshape how agents interpret tasks and execute actions, leaving them highly susceptible to the aforementioned threats. To mitigate these vulnerabilities, we propose MemGate, a lightweight and deployable memory plug-in for trustworthy memory search, with only 9M parameters and a 35.1MB footprint. MemGate is inserted between the vector memory store and the backbone LLM, requiring no LLM modification, memory-database rewriting, or inference-time LLM judge. It applies a query-conditioned neural gate to candidate memory representations, turning raw similarity search into task-conditioned memory admission. Across multiple mainstream memory frameworks, real-world agent settings, and diverse LLM backbones, MemGate reduces memory-induced threats while preserving long-term memory utility. |
| 2026-06-04 | [SpeechJBB: Probing Safety Alignment and Comprehension in Large Audio Language Models under Code-Switched Speech](http://arxiv.org/abs/2606.06037v1) | Virginia Ceccatelli, Yejin Jeon et al. | Large audio language models (LALMs) are increasingly deployed in real-world applications, yet their safety alignment is still primarily evaluated on monolingual, text-based harmful prompts. This leaves their generalizability under multilingual and spoken settings, particularly code-switched speech, largely underexplored. To address this gap, we introduce SpeechJBB, an audio jailbreak dataset for benchmarking across multiple state-of-the-art LALMs. The extent of safety weaknesses is further probed by introducing an augmented setting where phonologically plausible pseudo-words are inserted around safety-critical terms to simulate localized obfuscation. Across models, code-switched harmful audio yields substantially high jailbreak success rates (JSR), with non-English monolingual and non-English code-switched pairs exhibiting the highest attack success. Pseudo-word insertion further reduces refusal rates, which demonstrates that natural-sounding obfuscation can effectively bypass safety policies. |
| 2026-06-04 | [Steering Vectors are an Adversarial Attack Surface](http://arxiv.org/abs/2606.05958v1) | Abzal Aidakhmetov, Donato Crisostomi et al. | Activation steering has become a popular way to control Large Language Model (LLM) behavior without fine-tuning. Since the technique is plug-and-play, users share datasets and precomputed vectors to steer model activations. However, we show that a \emph{stealth data poisoning attack} silently compromises this pipeline. By substituting $4{-}6\%$ of tokens in the steering dataset, an attacker can silently align the resulting vector with an anti-refusal direction. This jailbreaks the target model while preserving the intended steering effect on benign prompts. Under this threat model, a malicious actor can distribute an apparently safe bundle containing texts, vectors, and weights, alongside an equivalence certificate that the end-user can verify. We test the attack on two open-weight model families and eight model-attribute combinations, observing that poisoned vectors reach an absolute attack success rate (ASR) of $20{-}55\%$, $+19\%$ to $+51\%$ over a clean reference. Finally, we find that a refusal-direction orthogonalization defense can recover ${\approx}82\%$ of the ASR gap without harming benign behavior. |
| 2026-06-04 | [Learning of Robot Safety Policies via Adversarial Synthetic Scenarios](http://arxiv.org/abs/2606.05952v1) | Nikolai Dorofeev, Alexey Odinokov et al. | In this work, we propose an agentic gamification framework for hazard-informed learning of robot safety policies through synthetic scenarios. We model scenario generation as an adversarial game between two agents: a Red Team that explores the space of potential failures by constructing hazardous situations, and a Blue Team that incrementally refines safety policies to prevent them. This iterative process enables efficient discovery of high-risk edge cases that are unlikely to be captured through random simulation or manual enumeration. By combining classical risk modeling with adversarial scenario generation and modern learning paradigms, this work provides a scalable pathway for embedding safety into Physical AI systems operating in complex real-world environments. The paper describes ongoing work. The contribution is a problem formulation and a proposed solution architecture. |

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



