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
| 2026-01-15 | [Institutional AI: A Governance Framework for Distributional AGI Safety](http://arxiv.org/abs/2601.10599v1) | Federico Pierucci, Marcello Galisai et al. | As LLM-based systems increasingly operate as agents embedded within human social and technical systems, alignment can no longer be treated as a property of an isolated model, but must be understood in relation to the environments in which these agents act. Even the most sophisticated methods of alignment, such as Reinforcement Learning through Human Feedback (RHLF) or through AI Feedback (RLAIF) cannot ensure control once internal goal structures diverge from developer intent. We identify three structural problems that emerge from core properties of AI models: (1) behavioral goal-independence, where models develop internal objectives and misgeneralize goals; (2) instrumental override of natural-language constraints, where models regard safety principles as non-binding while pursuing latent objectives, leveraging deception and manipulation; and (3) agentic alignment drift, where individually aligned agents converge to collusive equilibria through interaction dynamics invisible to single-agent audits. The solution this paper advances is Institutional AI: a system-level approach that treats alignment as a question of effective governance of AI agent collectives. We argue for a governance-graph that details how to constrain agents via runtime monitoring, incentive shaping through prizes and sanctions, explicit norms and enforcement roles. This institutional turn reframes safety from software engineering to a mechanism design problem, where the primary goal of alignment is shifting the payoff landscape of AI agent collectives. |
| 2026-01-15 | [Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay](http://arxiv.org/abs/2601.10589v1) | Hao Wang, Yanting Wang et al. | Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment. |
| 2026-01-15 | [Representation-Aware Unlearning via Activation Signatures: From Suppression to Knowledge-Signature Erasure](http://arxiv.org/abs/2601.10566v1) | Syed Naveed Mahmood, Md. Rezaur Rahman Bhuiyan et al. | Selective knowledge erasure from LLMs is critical for GDPR compliance and model safety, yet current unlearning methods conflate behavioral suppression with true knowledge removal, allowing latent capabilities to persist beneath surface-level refusals. In this work, we address this challenge by introducing Knowledge Immunization Framework (KIF), a representation-aware architecture that distinguishes genuine erasure from obfuscation by targeting internal activation signatures rather than surface outputs. Our approach combines dynamic suppression of subject-specific representations with parameter-efficient adaptation, enabling durable unlearning without full model retraining. KIF achieves near-oracle erasure (FQ approx 0.99 vs. 1.00) while preserving utility at oracle levels (MU = 0.62), effectively breaking the stability-erasure tradeoff that has constrained all prior work. We evaluate both standard foundation models (Llama and Mistral) and reasoning-prior models (Qwen and DeepSeek) across 3B to 14B parameters. Our observation shows that standard models exhibit scale-independent true erasure (<3% utility drift), while reasoning-prior models reveal fundamental architectural divergence. Our comprehensive dual-metric evaluation protocol, combining surface-level leakage with latent trace persistence, operationalizes the obfuscation - erasure distinction and enables the first systematic diagnosis of mechanism-level forgetting behavior across model families and scales. |
| 2026-01-15 | [Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing](http://arxiv.org/abs/2601.10543v1) | Yinzhi Zhao, Ming Wang et al. | Large language models (LLMs) have achieved impressive performance across natural language tasks and are increasingly deployed in real-world applications. Despite extensive safety alignment efforts, recent studies show that such alignment is often shallow and remains vulnerable to jailbreak attacks. Existing defense mechanisms, including decoding-based constraints and post-hoc content detectors, struggle against sophisticated jailbreaks, often intervening robust detection or excessively degrading model utility. In this work, we examine the decoding process of LLMs and make a key observation: even when successfully jailbroken, models internally exhibit latent safety-related signals during generation. However, these signals are overridden by the model's drive for fluent continuation, preventing timely self-correction or refusal. Building on this observation, we propose a simple yet effective approach that explicitly surfaces and leverages these latent safety signals for early detection of unsafe content during decoding. Experiments across diverse jailbreak attacks demonstrate that our approach significantly enhances safety, while maintaining low over-refusal rates on benign inputs and preserving response quality. Our results suggest that activating intrinsic safety-awareness during decoding offers a promising and complementary direction for defending against jailbreak attacks. Code is available at: https://github.com/zyz13590/SafeProbing. |
| 2026-01-15 | [A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5](http://arxiv.org/abs/2601.10527v1) | Xingjun Ma, Yixu Wang et al. | The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has produced substantial gains in reasoning, perception, and generative capability across language and vision. However, whether these advances yield commensurate improvements in safety remains unclear, in part due to fragmented evaluation practices limited to single modalities or threat models. In this report, we present an integrated safety evaluation of 7 frontier models: GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5. We evaluate each model across language, vision-language, and image generation settings using a unified protocol that integrates benchmark evaluation, adversarial evaluation, multilingual evaluation, and compliance evaluation. Aggregating our evaluations into safety leaderboards and model safety profiles across multiple evaluation modes reveals a sharply heterogeneous safety landscape. While GPT-5.2 demonstrates consistently strong and balanced safety performance across evaluations, other models exhibit pronounced trade-offs among benchmark safety, adversarial alignment, multilingual generalization, and regulatory compliance. Both language and vision-language modalities show significant vulnerability under adversarial evaluation, with all models degrading substantially despite strong results on standard benchmarks. Text-to-image models achieve relatively stronger alignment in regulated visual risk categories, yet remain brittle under adversarial or semantically ambiguous prompts. Overall, these results show that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation scheme, underscoring the need for standardized safety evaluations to accurately assess real-world risk and guide responsible model development and deployment. |
| 2026-01-15 | [SurgGoal: Rethinking Surgical Planning Evaluation via Goal-Satisfiability](http://arxiv.org/abs/2601.10455v1) | Ruochen Li, Kun Yuan et al. | Surgical planning integrates visual perception, long-horizon reasoning, and procedural knowledge, yet it remains unclear whether current evaluation protocols reliably assess vision-language models (VLMs) in safety-critical settings. Motivated by a goal-oriented view of surgical planning, we define planning correctness via phase-goal satisfiability, where plan validity is determined by expert-defined surgical rules. Based on this definition, we introduce a multicentric meta-evaluation benchmark with valid procedural variations and invalid plans containing order and content errors. Using this benchmark, we show that sequence similarity metrics systematically misjudge planning quality, penalizing valid plans while failing to identify invalid ones. We therefore adopt a rule-based goal-satisfiability metric as a high-precision meta-evaluation reference to assess Video-LLMs under progressively constrained settings, revealing failures due to perception errors and under-constrained reasoning. Structural knowledge consistently improves performance, whereas semantic guidance alone is unreliable and benefits larger models only when combined with structural constraints. |
| 2026-01-15 | [CS-GBA: A Critical Sample-based Gradient-guided Backdoor Attack for Offline Reinforcement Learning](http://arxiv.org/abs/2601.10407v1) | Yuanjie Zhao, Junnan Qiu et al. | Offline Reinforcement Learning (RL) enables policy optimization from static datasets but is inherently vulnerable to backdoor attacks. Existing attack strategies typically struggle against safety-constrained algorithms (e.g., CQL) due to inefficient random poisoning and the use of easily detectable Out-of-Distribution (OOD) triggers. In this paper, we propose CS-GBA (Critical Sample-based Gradient-guided Backdoor Attack), a novel framework designed to achieve high stealthiness and destructiveness under a strict budget. Leveraging the theoretical insight that samples with high Temporal Difference (TD) errors are pivotal for value function convergence, we introduce an adaptive Critical Sample Selection strategy that concentrates the attack budget on the most influential transitions. To evade OOD detection, we propose a Correlation-Breaking Trigger mechanism that exploits the physical mutual exclusivity of state features (e.g., 95th percentile boundaries) to remain statistically concealed. Furthermore, we replace the conventional label inversion with a Gradient-Guided Action Generation mechanism, which searches for worst-case actions within the data manifold using the victim Q-network's gradient. Empirical results on D4RL benchmarks demonstrate that our method significantly outperforms state-of-the-art baselines, achieving high attack success rates against representative safety-constrained algorithms with a minimal 5% poisoning budget, while maintaining the agent's performance in clean environments. |
| 2026-01-15 | [LatentRefusal: Latent-Signal Refusal for Unanswerable Text-to-SQL Queries](http://arxiv.org/abs/2601.10398v1) | Xuancheng Ren, Shijing Hu et al. | In LLM-based text-to-SQL systems, unanswerable and underspecified user queries may generate not only incorrect text but also executable programs that yield misleading results or violate safety constraints, posing a major barrier to safe deployment. Existing refusal strategies for such queries either rely on output-level instruction following, which is brittle due to model hallucinations, or estimate output uncertainty, which adds complexity and overhead. To address this challenge, we formalize safe refusal in text-to-SQL systems as an answerability-gating problem and propose LatentRefusal, a latent-signal refusal mechanism that predicts query answerability from intermediate hidden activations of a large language model. We introduce the Tri-Residual Gated Encoder, a lightweight probing architecture, to suppress schema noise and amplify sparse, localized cues of question-schema mismatch that indicate unanswerability. Extensive empirical evaluations across diverse ambiguous and unanswerable settings, together with ablation studies and interpretability analyses, demonstrate the effectiveness of the proposed approach and show that LatentRefusal provides an attachable and efficient safety layer for text-to-SQL systems. Across four benchmarks, LatentRefusal improves average F1 to 88.5 percent on both backbones while adding approximately 2 milliseconds of probe overhead. |
| 2026-01-15 | [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](http://arxiv.org/abs/2601.10387v1) | Christina Lu, Jack Gallagher et al. | Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. We investigate the structure of the space of model personas by extracting activation directions corresponding to diverse character archetypes. Across several different models, we find that the leading component of this persona space is an "Assistant Axis," which captures the extent to which a model is operating in its default Assistant mode. Steering towards the Assistant direction reinforces helpful and harmless behavior; steering away increases the model's tendency to identify as other entities. Moreover, steering away with more extreme values often induces a mystical, theatrical speaking style. We find this axis is also present in pre-trained models, where it primarily promotes helpful human archetypes like consultants and coaches and inhibits spiritual ones. Measuring deviations along the Assistant Axis predicts "persona drift," a phenomenon where models slip into exhibiting harmful or bizarre behaviors that are uncharacteristic of their typical persona. We find that persona drift is often driven by conversations demanding meta-reflection on the model's processes or featuring emotionally vulnerable users. We show that restricting activations to a fixed region along the Assistant Axis can stabilize model behavior in these scenarios -- and also in the face of adversarial persona-based jailbreaks. Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it, motivating work on training and steering strategies that more deeply anchor models to a coherent persona. |
| 2026-01-15 | [FastStair: Learning to Run Up Stairs with Humanoid Robots](http://arxiv.org/abs/2601.10365v1) | Yan Liu, Tao Yu et al. | Running up stairs is effortless for humans but remains extremely challenging for humanoid robots due to the simultaneous requirements of high agility and strict stability. Model-free reinforcement learning (RL) can generate dynamic locomotion, yet implicit stability rewards and heavy reliance on task-specific reward shaping tend to result in unsafe behaviors, especially on stairs; conversely, model-based foothold planners encode contact feasibility and stability structure, but enforcing their hard constraints often induces conservative motion that limits speed. We present FastStair, a planner-guided, multi-stage learning framework that reconciles these complementary strengths to achieve fast and stable stair ascent. FastStair integrates a parallel model-based foothold planner into the RL training loop to bias exploration toward dynamically feasible contacts and to pretrain a safety-focused base policy. To mitigate planner-induced conservatism and the discrepancy between low- and high-speed action distributions, the base policy was fine-tuned into speed-specialized experts and then integrated via Low-Rank Adaptation (LoRA) to enable smooth operation across the full commanded-speed range. We deploy the resulting controller on the Oli humanoid robot, achieving stable stair ascent at commanded speeds up to 1.65 m/s and traversing a 33-step spiral staircase (17 cm rise per step) in 12 s, demonstrating robust high-speed performance on long staircases. Notably, the proposed approach served as the champion solution in the Canton Tower Robot Run Up Competition. |
| 2026-01-15 | [Training-Trajectory-Aware Token Selection](http://arxiv.org/abs/2601.10348v1) | Zhanming Shen, Jiaqi Hu et al. | Efficient distillation is a key pathway for converting expensive reasoning capability into deployable efficiency, yet in the frontier regime where the student already has strong reasoning ability, naive continual distillation often yields limited gains or even degradation. We observe a characteristic training phenomenon: even as loss decreases monotonically, all performance metrics can drop sharply at almost the same bottleneck, before gradually recovering. We further uncover a token-level mechanism: confidence bifurcates into steadily increasing Imitation-Anchor Tokens that quickly anchor optimization and other yet-to-learn tokens whose confidence is suppressed until after the bottleneck. And the characteristic that these two types of tokens cannot coexist is the root cause of the failure in continual distillation. To this end, we propose Training-Trajectory-Aware Token Selection (T3S) to reconstruct the training objective at the token level, clearing the optimization path for yet-to-learn tokens. T3 yields consistent gains in both AR and dLLM settings: with only hundreds of examples, Qwen3-8B surpasses DeepSeek-R1 on competitive reasoning benchmarks, Qwen3-32B approaches Qwen3-235B, and T3-trained LLaDA-2.0-Mini exceeds its AR baseline, achieving state-of-the-art performance among all of 16B-scale no-think models. |
| 2026-01-15 | [Radiation Shielding Performance of Different Concrete Materials: A Systematic Review](http://arxiv.org/abs/2601.10336v1) | Christiana Subaar, Ziem Samuel Aanoneda et al. | Background: Concrete is one of the most-used material today in nuclear, medical, and industrial applications for radiation shielding due to its economic advantages and availability together with its structural performance. However, differences in the use of aggregates, density, and other additives affect radiation attenuation efficiency. It is therefore necessary to understand and compare shielding properties of various concrete formulations for the optimization of safety and performance in radiation-prone environments. Methods: Using Preferred Reporting Items for Systematic reviews and Meta-Analyses (PRISMA) guidelines, a literature search was performed across PubMed, Scopus, ScienceDirect, and Google Scholar. 17 peer-reviewed studies published between 2010 and 2025 were analysed systematically. Data was extracted based on material composition, density, radiation type, energy range, attenuation coefficients, and shielding efficiency. The obtained results were compared to find the trend in performance and optimization of the considered materials. Conclusion: Radiation shielding efficiency of concrete is dependent on its density, microstructural characteristics and type of aggregate. For superior performance in a mixed radiation field, Heavy and boron-rich additives can be added. Newly developed UHPCs and nano-engineered concretes are lightweight, durable, and environmentally friendly options for shielding materials compared to traditional ones. Further studies are needed to focus on the standardization of test methods, validation of long-term stability, and coupling computational modelling with experimental data in order to guide material design for applications featuring enhanced radiation shielding. |
| 2026-01-15 | [The Straight and Narrow: Do LLMs Possess an Internal Moral Path?](http://arxiv.org/abs/2601.10307v1) | Luoming Hu, Jingjie Zeng et al. | Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical results confirm that our approach acts as a targeted intrinsic defense, effectively reducing incorrect refusals on benign queries while minimizing jailbreak success rates compared to standard baselines. |
| 2026-01-15 | [Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection](http://arxiv.org/abs/2601.10294v1) | Yuansen Liu, Yixuan Tang et al. | Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack |
| 2026-01-15 | [Updated electrical design of the Diagnostic Neutral Beam Injector in RFX-mod2](http://arxiv.org/abs/2601.10293v1) | Marco Barbisan, Bruno Laterza et al. | The Diagnostic Neutral Beam Injector of the RFX-mod2 experiment (Consorzio RFX, Padova) is expected to provide novel and significant information about the Reversed Field Pinch confinement of fusion plasmas. The injection of the hydrogen beam in the plasma will allow Charge Exchange Recombination Spectroscopy (CXRS) and Motional Stark Effect diagnostics (MSE) to measure several quantities: ion speed, ion temperature, impurity content, intensity and pitch of the magnetic field. The DNBI is of particular importance for allowing the determination of these quantities at the core of the plasma. The present DNBI, built by the Budker Institute of Plasma Physics, features an arc discharge H+ source, coupled to a 4-grid 50 keV acceleration system. The 50 ms, 5 A ion beam is neutralized by charge exchange by means of a gas target; residual ions are then deflected by a magnetic field before injection in the torus chamber. The beam can be modulated at maximum 250 Hz. The DNBI will undergo extraordinary maintenance and a structural upgrade to improve its reliability and safety. This contribution presents the latest upgrades of the electrical plants and of the control system of the DNBI. |
| 2026-01-15 | [Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control](http://arxiv.org/abs/2601.10233v1) | Yifan Xue, Ze Zhang et al. | This work addresses the challenge of safe and efficient mobile robot navigation in complex dynamic environments with concave moving obstacles. Reactive safe controllers like Control Barrier Functions (CBFs) design obstacle avoidance strategies based only on the current states of the obstacles, risking future collisions. To alleviate this problem, we use Gaussian processes to learn barrier functions online from multimodal motion predictions of obstacles generated by neural networks trained with energy-based learning. The learned barrier functions are then fed into quadratic programs using modulated CBFs (MCBFs), a local-minimum-free version of CBFs, to achieve safe and efficient navigation. The proposed framework makes two key contributions. First, it develops a prediction-to-barrier function online learning pipeline. Second, it introduces an autonomous parameter tuning algorithm that adapts MCBFs to deforming, prediction-based barrier functions. The framework is evaluated in both simulations and real-world experiments, consistently outperforming baselines and demonstrating superior safety and efficiency in crowded dynamic environments. |
| 2026-01-15 | [Agentic Pipelines in Embedded Software Engineering: Emerging Practices and Challenges](http://arxiv.org/abs/2601.10220v1) | Simin Sun, Miroslaw Staron | A new transformation is underway in software engineering, driven by the rapid adoption of generative AI in development workflows. Similar to how version control systems once automated manual coordination, AI tools are now beginning to automate many aspects of programming. For embedded software engineering organizations, however, this marks their first experience integrating AI into safety-critical and resource-constrained environments. The strict demands for determinism, reliability, and traceability pose unique challenges for adopting generative technologies.   In this paper, we present findings from a qualitative study with ten senior experts from four companies who are evaluating generative AI-augmented development for embedded software. Through semi-structured focus group interviews and structured brainstorming sessions, we identified eleven emerging practices and fourteen challenges related to the orchestration, responsible governance, and sustainable adoption of generative AI tools. Our results show how embedded software engineering teams are rethinking workflows, roles, and toolchains to enable a sustainable transition toward agentic pipelines and generative AI-augmented development. |
| 2026-01-15 | [ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack](http://arxiv.org/abs/2601.10173v1) | Hao Li, Yankai Yang et al. | Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign. |
| 2026-01-15 | [ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback](http://arxiv.org/abs/2601.10156v1) | Yutao Mou, Zhangchi Xue et al. | While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks. |
| 2026-01-15 | [Understanding and Preserving Safety in Fine-Tuned LLMs](http://arxiv.org/abs/2601.10141v1) | Jiawen Zhang, Yangfan Hu et al. | Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning. |

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



