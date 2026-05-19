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
| 2026-05-18 | [Code as Agent Harness](http://arxiv.org/abs/2605.18747v1) | Xuying Ning, Katherine Tieu et al. | Recent large language models (LLMs) have demonstrated strong capabilities in understanding and generating code, from competitive programming to repository-level software engineering. In emerging agentic systems, code is no longer only a target output. It increasingly serves as an operational substrate for agent reasoning, acting, environment modeling, and execution-based verification. We frame this shift through the lens of agent harnesses and introduce code as agent harness: a unified view that centers code as the basis for agent infrastructure. To systematically study this perspective, we organize the survey around three connected layers. First, we study the harness interface, where code connects agents to reasoning, action, and environment modeling. Second, we examine harness mechanisms: planning, memory, and tool use for long-horizon execution, together with feedback-driven control and optimization that make harness reliable and adaptive. Third, we discuss scaling the harness from single-agent systems to multi-agent settings, where shared code artifacts support multi-agent coordination, review, and verification. Across these layers, we summarize representative methods and practical applications of code as agent harness, spanning coding assistants, GUI/OS automation, embodied agents, scientific discovery, personalization and recommendation, DevOps, and enterprise workflows. We further outline open challenges for harness engineering, including evaluation beyond final task success, verification under incomplete feedback, regression-free harness improvement, consistent shared state across multiple agents, human oversight for safety-critical actions, and extensions to multimodal environments. By centering code as the harness of agentic AI, this survey provides a unified roadmap toward executable, verifiable, and stateful AI agent systems. |
| 2026-05-18 | [SafeDiffusion-R1: Online Reward Steering for Safe Diffusion Post-Training](http://arxiv.org/abs/2605.18719v1) | Komal Kumar, Ankan Deria et al. | Diffusion models have been widely studied for removing unsafe content learned during pre-training. Existing methods require expensive supervised data, either unsafe-text paired with safe-image groundtruth or negative/positive image pairs, making them impractical to scale. Furthermore, offline reinforcement learning and supervised fine-tuning approaches that generate synthetic data offline suffer from catastrophic forgetting, degrading generation quality. We propose a novel online reinforcement learning framework that addresses both data scarcity and model degradation through post-training with Group Relative Policy Optimization (GRPO) on both negative and positive text prompts. To eliminate the need for fine-tuning specialized safe/unsafe reward models, we introduce a \textit{steering reward mechanism} that exploits an inherent property of CLIP embeddings: steering text representations toward positive safety directions and away from negative ones in the embedding space. Our online-policy approach enables the model to learn from diverse prompts, including explicit unsafe content, without catastrophic forgetting. Extensive experiments demonstrate that our method reduces inappropriate content to 18.07\% (vs. 48.9\% for SD v1.4) and nudity detections to 15 (vs. 646 baseline) while improving compositional generation quality from 42.08\% to 47.83\% on GenEval. Remarkably, these safety gains generalize to out-of-domain unsafe prompts across seven harm categories, achieving state-of-the-art performance without supervised paired data or reward tuning. Github: https://github.com/MAXNORM8650/SafeDiffusion-R1. |
| 2026-05-18 | [Position: A Three-Layer Probabilistic Assume-Guarantee Architecture Is Structurally Required for Safe LLM Agent Deployment](http://arxiv.org/abs/2605.18672v1) | S. Bensalem, Y. Dong et al. | This position paper argues that enforcing LLM agent safety within a single abstraction layer is not merely suboptimal but categorically insufficient for deployed LLM agents -- a structural consequence of how agent execution works, not a contingent limitation of current systems. The three dimensions that jointly constitute safe operation -- semantic intent and policy compliance, environmental validity, and dynamical feasibility -- each depend on a strictly distinct set of information that becomes available at different stages of execution. No single guardrail can certify all three. We argue that the community must respond with a contract-based architecture in which each safety dimension is enforced by an independently certified layer whose probabilistic guarantee satisfies the next layer's assumption. We sketch such an architecture and derive the compositional system-level safety bounds it admits via the chain rule of probability. Three open problems stand between this and a deployable standard: bound estimation from non-i.i.d.\ traces, graceful degradation of contracts under deployment drift, and extension to multi-agent settings -- the most important unfinished business in LLM agent runtime assurance. |
| 2026-05-18 | [Not What You Asked For: Typographic Attacks in Household Robot Manipulation](http://arxiv.org/abs/2605.18593v1) | Ali Iranmanesh, Peng Liu | Open-vocabulary embodied AI agents increasingly rely on vision-language models such as CLIP for object perception and task grounding. However, the shared embedding space that enables this flexibility introduces a structural vulnerability to typographic attacks, where printed text in a physical scene semantically overrides visual judgment. While prior work has quantified this threat in static 2D benchmarks and 3D navigation tasks, its impact on the full Sense-Plan-Act pipeline of household robot manipulation remains unexplored.   This work evaluates typographic attacks in a Habitat-based simulation using the HomeRobot benchmark. We introduce a decoupled perception architecture that exposes a frozen CLIP encoder to adversarial stickers while maintaining geometric grounding via DETIC. In a controlled evaluation pool of 59 attributable episodes, the attack achieves an overall Attack Success Rate (ASR) of 67.8%, rising to 70.0% among fully successful episodes, under uncontrolled viewing angles and occlusion with no perceptual optimization.   Critically, we find that perceptual errors propagate through the persistent 3D semantic map to produce kinetic failures, defined here as physically executed grasping and transport of the wrong object driven by an adversarially poisoned semantic state. In these cases, the robot physically grasps and delivers the wrong object to a target receptacle. These results establish typographic misclassification as a real, measurable, and physically consequential threat to the safety of modular manipulation pipelines that prior typographic attack research has left unexamined. |
| 2026-05-18 | [HJ-Gauss: A Monte-Carlo HJ Reachability Scheme](http://arxiv.org/abs/2605.18566v1) | Lekan Molu, Venkatraman Renganathan et al. | Backward reachable tubes (BRTs), computed via viscous Hamilton-Jacobi (HJ) partial differential equations, provide principled safety certificates for learned controllers and planning algorithms in trustworthy machine learning. However, classical grid-based HJ solvers require $O(M^n)$ memory footprint for $M$ grid points per $n$ state dimension. This renders them impractical for high-dimensional systems. We address this bottleneck with a local PDE linearization that enables a frozen-coefficient sampling scheme for the viscous HJ PDE: a generalized Cole-Hopf-type transformation reduces the nonlinear HJ equation to a sequence of linear heat equations whose solutions admit Gaussian heat-kernel representations. The value function and its spatial gradient are then recovered via roll-outs of Monte Carlo expectations on Gaussian densities, yielding a storage and grid-free algorithm that scales as $N\cdot n$ for $N$ samples. This decoupling of memory from dimensionality enables reachability analysis on problems where grid-based methods are simply impossible. We prove a finite-sample concentration bound $O(N^{-1/2})$ error and conditional linear convergence for the introduced Monte-Carlo Picard iterative scheme. Numerical validation on pursuit-evasion games demonstrates relative $L^2_{\text{rel}}$ errors of $0.03 - 0.20$, with $14-26$ second wall-clock times per 2D slice on a CPU. Crucially, the method scales with validation on up to (but not limited to) $n=45$-dimensional multi-agent games. |
| 2026-05-18 | [Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics](http://arxiv.org/abs/2605.18549v1) | Maciej Chrabąszcz, Aleksander Szymczyk et al. | Large Reasoning Models (LRMs) introduce new opportunities for safety monitoring through their Chain of Thought (CoT) reasoning. However, CoT is not always faithful to the model's final output, undermining its reliability as a monitoring tool. To address this, we investigate the hidden representations of LRMs to determine whether future behavior can be predicted from prompt and CoT representations. By evaluating a probe at each generated token, we construct a probe trajectory, the continuous evolution of a concept's probability across the reasoning process. We find that future model behavior is more distinguishable when examined over the full trajectory than from a single static prediction. To characterize these temporal dynamics, we extract signal-processing features that capture volatility, trend, and steady-state behavior, significantly improving the separation of future model states. We also present two methodological insights. First, template-based training data achieves near-parity with dynamically generated model responses, eliminating the need for a costly initial inference and labeling. Second, the choice of pooling operation is critical: average-pooling and last-token methods collapse to near-random performance, while max-pooling achieves up to 95% AUROC and yields stable probe trajectories. Using four datasets and four reasoning models across the domains of safety and mathematics, we demonstrate that trajectory features encode task-specific dynamics that improve outcome separability. These findings establish probe trajectories as a complementary framework for monitoring LRM behavior.   Warning: This article contains potentially harmful content. |
| 2026-05-18 | [Collaborative Air-Ground Sensing, Communication, Computing, Storage, and Intelligence for Low-Altitude Economy](http://arxiv.org/abs/2605.18503v1) | Yiqin Deng, Junhui Gao et al. | Low-altitude economy (LAE) is transforming low-altitude airspace into a new cyber-physical infrastructure. Although air-ground communications have been widely studied, LAE is fundamentally different in the sense that it is mission-centric with diverse requirements, such as stringent safety and compliance constraints not be effectively addressed with a communication-centric design alone, which makes air-ground collaboration indispensable: Only through effectively coordinating air-ground infrastructure and resources can LAE missions be fulfilled. Consequently, LAE calls for task-driven, closed-loop, multi-resource orchestration of Sensing, Communication, Computing, Storage, and Intelligence (SCCSI), where key decisions must be co-designed under mobility and uncertainty. In this paper, we first present a novel framework that connects (i) LAE scenarios and a requirement--resource coupling matrix, (ii) an air--ground collaborative architecture, and (iii) methodological toolboxes for SCCSI co-optimization and online decision-making. We then systematically review enabling technologies for collaborative SCCSI resources and capabilities, emphasizing their coupling and end-to-end tradeoffs. Finally, we summarize testbeds, datasets, and evaluation metrics, and provide representative use cases to illustrate how the proposed framework translates application requirements into practical task-driven optimization designs, together with open challenges and a roadmap toward scalable and trustworthy LAE deployment. |
| 2026-05-18 | [The distance-based formation controller design for multi-agent systems in port-Hamiltonian form](http://arxiv.org/abs/2605.18502v1) | Jingyi Zhao, Yongxin Wu et al. | Based on the practical scenario where collisions in formation control may lead to agent damage, this paper investigates the integrated problem of distance-based formation control and collision avoidance for multi-agent systems governed by port-Hamiltonian dynamics. A foundational step involves constructing a signed incidence matrix, which, by design, corresponds to a directed acyclic graph and possesses the full column rank property. To overcome the prevalent issue of local minima in traditional artificial potential fields, a novel design utilizing attraction-only potentials is introduced, with collision avoidance rigorously enforced by safety barriers. This framework leads to a unified controller that concurrently manages velocity tracking, target formation acquisition, and inter-agent safety. The stability of the resulting closed-loop system is guaranteed through LaSalle's invariance principle. Numerical simulations demonstrate the validity and effectiveness of the proposed control strategy. |
| 2026-05-18 | [REBAR: Reference Ethical Benchmark for Autonomy Readiness](http://arxiv.org/abs/2605.18423v1) | Jonathan Diller, David Barnes et al. | As autonomous systems grow more advanced, objective metrics to evaluate their ethical and legal compliance are critical for informing end users of their limitations and ensuring accountability of those who misuse them. Current ethical embodied AI frameworks remain mostly qualitative, focusing on system design (through safety guardrails or targeted red teaming), and the realized guardrails often directly disallow unsafe behavior without providing the user with an override or interpretable reason. Instead, there is a need for computable metrics through rigorous testing that allow a user to determine the applicability of the system to the task. To address this gap, we introduce the Reference Ethical Benchmark for Autonomy Readiness (REBAR), a quantitative test and evaluation framework for autonomous systems. REBAR maps operating metrics into a computable Autonomy Readiness Level (ARL) rubric that can quantify ethical performance. Key innovations of the framework include a neuro-symbolic Large Language Model (LLM) approach to calculate and explain the ethical difficulty of scenarios, LLM-driven at-scale generation of test instances, and a versatile, photorealistic simulation environment. By evaluating white-box autonomy solutions through this rigorous testing pipeline, REBAR delivers an objective and repeatable benchmark score, bridging the gap between abstract principles and verifiable, accountable autonomy. |
| 2026-05-18 | [ISEP: Implicit Support Expansion for Offline Reinforcement Learning via Stochastic Policy Optimization](http://arxiv.org/abs/2605.18320v1) | Yifei Chen, Shaoqin Zhu et al. | Offline reinforcement learning methods typically enforce strict constraints to ensure safety; yet this rigidity often prevents the discovery of optimal behaviors outside the immediate support of the behavior policy. To address this, we propose Implicit Support Expansion via stochastic Policy optimization (ISEP), which leverages a value function interpolated between in-distribution data and policy samples to implicitly expand the feasible action support. This mechanism "densifies" high-reward regions, creating a navigable path for policy improvement while theoretically guaranteeing bounded value error. However, optimizing against this expanded support creates a multimodal landscape where standard deterministic averaging leads to mode collapse and invalid actions. ISEP mitigates this via a stochastic action selection strategy, optimizing the policy by stochastically alternating between conservative cloning and optimistic expansion signals. We instantiate this framework as ISEP-FM using Conditional Flow Matching utilizing classifier-free guidance to effectively capture the interpolated value signal. |
| 2026-05-18 | [Alignment Dynamics in LLM Fine-Tuning](http://arxiv.org/abs/2605.18309v1) | Yuhan Huang, Huanran Chen et al. | Although Large Language Models (LLMs) achieve strong alignment through supervised fine-tuning and reinforcement learning from human feedback, the alignment is often fragile under subsequent fine-tuning. Existing explanations either attribute alignment fragility to gradient geometry or characterize it as a distributional shift in model outputs, yet few provide a unified account that bridges parameter-space learning dynamics with function-space alignment behavior during fine-tuning. In this work, we introduce a tractable alignment score and derive its closed-form update during fine-tuning, yielding a unified framework for alignment dynamics. Our analysis decomposes alignment updates into two competing components: a \textbf{\color{red!60!black} Rebound Force}, governed jointly by the current alignment state and the narrowness of model distribution, and a \textbf{\color{green!60!black} Driving Force}, determined by how the training distribution aligns with outcome-conditioned posteriors over aligned and non-aligned completions. This decomposition explains why prior alignment can be reversed by later fine-tuning and why narrower posterior structure strengthens such reversal. Moreover, our framework predicts a \textbf{Rehearsal Priming Effect}: prior alignment leaves a latent posterior imprint that amplifies the effective Driving Force upon re-exposure, leading to faster re-alignment. We validate these predictions across safety alignment, emergent misalignment, and sentiment settings, demonstrating consistent alignment reversal and accelerated re-alignment under re-exposure. In addition, controlled experiments in safety alignment confirm the predicted dependence of rebound strength on posterior narrowness. Together, these results provide a unified dynamical perspective on how alignment is disrupted and reactivated during LLM fine-tuning. |
| 2026-05-18 | [Assessing Localization Technologies for Pedestrian Collision Avoidance](http://arxiv.org/abs/2605.18295v1) | Joshua Varughese, Joseba Gorospe et al. | Robust pedestrian safety is crucial to the next-generation of intelligent transportation systems. Such systems rely on active pedestrian localization and predictive collision alerts. Pedestrian localization can be supported by Ultra-Wideband technology and Bluetooth 6.0, which offer high-precision ranging and low-latency communication, making them promising candidates for vehicular collision warning systems. This paper assesses the localization accuracy of these technologies for pedestrian alerting and benchmarks their performance against Global Navigation Satellite Systems. Experimental evaluations performed in this paper focused on key performance metrics, including localization accuracy and robustness to environmental conditions. Preliminary results suggest that Ultra-Wideband and Bluetooth 6.0 can serve as viable alternatives or complements to Global Navigation Satellite Systems in certain scenarios, improving situational awareness and enabling timely pedestrian alerts. |
| 2026-05-18 | [Temporal Task Diversity: Inductive Biases Under Non-Stationarity in Synthetic Sequence Modelling](http://arxiv.org/abs/2605.18281v1) | Afiq Abdillah Effiezal Aswadi, Oliver Britton et al. | Modern deep learning science often assumes that neural networks learn from a fixed data distribution. However, many practically important learning problems involve data distributions that change throughout training. How does such non-stationarity impact the inductive biases of deep learning towards models with different structural, generalisation, and safety properties? A fruitful testbed for studying inductive bias is in-context linear regression sequence modelling, where small transformers display strikingly different generalisation patterns depending on the diversity of the (fixed) training task distribution. In this paper, we explore the effect of diversifying the task distribution across training time, finding that such temporal diversity leads to an increased bias towards generalisation over memorisation. |
| 2026-05-18 | [Multilingual jailbreaking of LLMs using low-resource languages](http://arxiv.org/abs/2605.18239v1) | Dylan Marx, Marcel Dunaiski | Large Language Models (LLMs) remain vulnerable to jailbreak attempts that circumvent safety guardrails. We investigate whether multi-turn conversations using low-resource African languages (Afrikaans, Kiswahili, isiXhosa, and isiZulu) can bypass safety mechanisms across commercial LLMs. We translated prompts from existing datasets and evaluated ChatGPT, Claude, DeepSeek, Gemini, and Grok through automated testing and human red-teaming with native speakers. Single-turn translation attacks proved ineffective, while multi-turn conversations achieved English harmful response rates from 52.7% (Claude 3.5 Haiku) to 83.6% (GPT-4o-mini), Afrikaans from 60.0% (Claude 3.5 Haiku) to 78.2% (GPT-4o-mini), and Kiswahili from 41.8% (Claude 3.5 Haiku) to 70.9% (DeepSeek). Human red-teaming increased jailbreak rates compared to automated methods. Over all evaluated languages, the average jailbreak rate increased from 59.8% to 75.8%, with improvements of +20.0% (Afrikaans), +12.7% (isiZulu), +12.3% (isiXhosa), and +1% (Kiswahili), demonstrating that poor translation quality limits jailbreak success. These findings suggest that vulnerabilities in LLMs persist in multilingual contexts and that translation quality is the critical factor determining jailbreak success in low-resource languages. |
| 2026-05-18 | [Characterisation of fire-damaged batteries,implications for recycling](http://arxiv.org/abs/2605.18183v1) | Wafaa AlShatty, Tom Dunlop et al. | As lithium-ion battery demand grows, so do fire safety challenges. Despite this, research on fire-damaged batteries remains limited. This study explores the distribution of valuable metals (such as Ni, Mn, Co, Cu) in two types of waste derived from lithium-ion nickel-manganese-cobalt oxide batteries (NMC811), black mass (BM) and fire-damaged waste (FD). It emphasizes that cobalt, manganese, and nickel-rich NMC811 particles are predominantly found in smaller particle size fractions (<125 microns), where they can account for up to 85 percent of total metal content. Fire-damaged (FD) batteries show a similar, though less pronounced, trend. Evidence of structural degradation suggests that fire temperatures exceeded 500°C; however, the presence of residual organic binders indicates that heat was unevenly distributed during the fire. FD batteries become friable and easily fragment into fine particles, which can hinder the effective separation of copper and aluminium current collectors, increasing their presence in processed material. The inclusion of FD batteries in standard BM processing introduces variability in output composition, potentially lowering the concentration of high-value NMC811 materials present. To maintain product quality and recycling output values, it is recommended that FD batteries are processed separately. Alternatively, particle size separation may allow for tailored outputs aligned with specific customer requirements. |
| 2026-05-18 | [Acoustic Interference: A New Paradigm Weaponizing Acoustic Latent Semantic for Universal Jailbreak against Large Audio Language Models](http://arxiv.org/abs/2605.18168v1) | Yanyun Wang, Yu Huang et al. | The integration of audio modality into Large Audio Language Models (LALMs) significantly expands their attack surface. Existing jailbreak paradigms predominantly treat audio as a carrier for malicious payloads, relying on semantic optimization, acoustic parameter control, or additive perturbation to embed harmful content into the audio signal. In this work, we challenge this necessity and propose a new paradigm in which the role of audio shifts from content injection to safety alignment interference. We reveal that LALM safety alignment can be compromised solely by specific Acoustic Latent Semantics (ALS), the underlying paralinguistic features intrinsic to the priors of audio generative models. Distinct from previous works that leverage explicit acoustic parameters to merely style malicious audio, we demonstrate that interference audio, benign in content but infused with specific ALS, can serve as a universal jailbreak trigger. Leveraging this insight, we propose the Acoustic Interference Attack (AIA), which decouples the attack payload from the audio. Specifically, AIA employs a set of universal, instruction-neutral interference audio, enabling standard malicious text queries to bypass safety alignment without instance-specific optimization. Extensive experiments on 10 LALMs across five datasets demonstrate that AIA achieves the state-of-the-art attack success rate. Furthermore, our interpretability analysis uncovers the inference path drift induced by AIA and identifies the inherent effective patterns within ALS, revealing the fundamental vulnerability of cross-modal alignment in LALMs. |
| 2026-05-18 | [In-Vehicle Human-Machine Interface to Support Drivers in Conditionally Automated Platooning](http://arxiv.org/abs/2605.18149v1) | Anna-Lena Hager, Mohamed Sabry et al. | Vehicle platooning enables close-gap driving and offers potential benefits for traffic efficiency and safety. In conditionally automated platooning, drivers remain responsible for supervising the system and intervening when necessary, making effective Human-Machine Interfaces (HMIs) critical for maintaining situational awareness and stable driver-automation coordination. This paper investigates whether an in-vehicle HMI providing continuous system-state and inter-vehicle distance information improves supervisory behavior, safety, and platoon stability. We conducted a simulation-based experiment integrated with a 6-degree-of-freedom motion system to enhance scenario realism. Dependent variables included collision occurrence, response latency following platoon disconnection, and the number of manual interventions during intact platooning.   Results showed significantly fewer manual interventions when the HMI was active, with intervention rates about 80% higher without it. No significant effects were found for collision occurrence or response latency, indicating that additional information improves supervisory stability during platooning but does not substantially affect emergency reactions or collision rates. |
| 2026-05-18 | [An Empirical Study of Privacy Leakage Chains via Prompt Injection in Black-Box Chatbot Environments](http://arxiv.org/abs/2605.18133v1) | Hongjang Yang, Hyunsik Na et al. | LLM-based chatbot agents increasingly process user requests by combining natural-language reasoning with external tools such as web browsing. These capabilities improve usability, but they also create attack surfaces when untrusted external content is processed as part of a user' s task. This paper studies a privacy-leakage attack chain based on indirect prompt injection in black-box chatbot environments, where the attacker has no access to model weights, system prompts, or agent implementation details including how a trajectory is actually managed during its processing for a query. We first analyze how an attacker can hijack an agent' s intended task by crafting external content that appears benign to the victim while inducing the agent to execute an attacker-defined objective. We then evaluate a new prompt-injection technique, called exemplification, which uses a bridge in the external content to reframe the user prompt and the benign beginning of the retrieved page as few-shot examples before appending the attacker' s objective. We compare its attack success rate with a prior fake-completion technique. Finally, we demonstrate a proof-of-concept data-exfiltration chain using fictitious personal information in a controlled setting. Our results suggest that prompt injection, jailbreak-style instruction steering, and web-tool invocation can be combined into a feasible privacy-leakage path in deployed chatbot agents. |
| 2026-05-18 | [Safety Geometry Collapse in Multimodal LLMs and Adaptive Drift Correction](http://arxiv.org/abs/2605.18104v1) | Jiahe Guo, Xiangran Guo et al. | Multimodal large language models (MLLMs) often fail to transfer safety capabilities learned in the text modality to semantically equivalent non-text inputs, revealing a persistent multimodal safety gap. We study this gap from a representation-geometric perspective by analyzing a text-aligned refusal direction and a modality-induced drift direction. We show that multimodal inputs compress the usable separation along the refusal direction, making it no longer reliable for identifying and refusing harmful inputs. We refer to this failure mode as Safety Geometry Collapse. We quantify it through conditional refusal separability and show that stronger modality-induced drift is consistently associated with weaker refusal separability and higher attack success rates. We then validate the causal role of modality-induced drift through a fixed-strength activation intervention: counteracting the estimated drift restores refusal separability and improves multimodal safety. After drift correction, we further observe self-rectification, where the model recovers its ability to recognize and refuse harmful multimodal inputs during forward dynamics. This effect also provides an internal signal of the model's perceived harmfulness of each input. Motivated by this signal, we propose ReGap, a training-free inference-time method that adaptively corrects modality drift using self-rectification. Experiments across multiple multimodal safety benchmarks and utility benchmarks demonstrate the effectiveness of ReGap, which significantly improves the safety of MLLMs without compromising general capabilities. Our findings highlight representation-level modality alignment as a crucial direction for real-time safety improvement and for building safer, more reliable MLLMs. |
| 2026-05-18 | [A-ProS: Towards Reliable Autonomous Programming Through Multi-Model Feedback](http://arxiv.org/abs/2605.18073v1) | Anika Tabassum, Md Sifat Hossain et al. | Large Language Models (LLMs) demonstrate strong potential for automated code generation, yet their ability to iteratively refine solutions using execution feedback remains underexplored. Competitive programming offers an ideal testbed for this investigation, as it demands end-to-end algorithmic reasoning, precise implementation under strict computational constraints, and complete functional correctness with rigorous evaluation. In this paper, we present A-ProS, an autonomous AI agent that solves competitive programming problems through a hybrid multi-model feedback framework separating solution generation from specialized debugging. A-ProS combines ChatGPT-based generators (GPT-4 and GPT-5) with three debugging critics: Codestral-2508, Llama-3.3-70B, and DeepSeek-R1, under a 2 x 3 factorial design. We evaluate six workflows on 367 problems from ICPC World Finals (2011-2024) and Codeforces (rated 1200-1800). The results show that GPT-5 workflows improve from 39 initial accepted solutions to 85-90 after three refinement rounds, while GPT-4 improves from 15 to 31-38. A controlled ablation on 47 problems shows that stateful refinement outperforms stateless approaches by 8.5-10.6 percentage points and reduces repeated failures by up to 3.5x. Compared to baseline agent loops, A-ProS achieves over 2x greater gains, highlighting the importance of persistent context and multi-model feedback for reliable autonomous program synthesis. |

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



