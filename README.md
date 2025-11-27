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
| 2025-11-26 | [On the Limits of Innate Planning in Large Language Models](http://arxiv.org/abs/2511.21591v1) | Charles Schepanowski, Charles Ling | Large language models (LLMs) achieve impressive results on many benchmarks, yet their capacity for planning and stateful reasoning remains unclear. We study these abilities directly, without code execution or other tools, using the 8-puzzle: a classic task that requires state tracking and goal-directed planning while allowing precise, step-by-step evaluation. Four models are tested under common prompting conditions (Zero-Shot, Chain-of-Thought, Algorithm-of-Thought) and with tiered corrective feedback. Feedback improves success rates for some model-prompt combinations, but many successful runs are long, computationally expensive, and indirect. We then examine the models with an external move validator that provides only valid moves. Despite this level of assistance, none of the models solve any puzzles in this setting. Qualitative analysis reveals two dominant deficits across all models: (1) brittle internal state representations, leading to frequent invalid moves, and (2) weak heuristic planning, with models entering loops or selecting actions that do not reduce the distance to the goal state. These findings indicate that, in the absence of external tools such as code interpreters, current LLMs have substantial limitations in planning and that further progress may require mechanisms for maintaining explicit state and performing structured search. |
| 2025-11-26 | [Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](http://arxiv.org/abs/2511.21584v1) | Haohong Lin, Yunzhi Zhang et al. | End-to-end (E2E) autonomous driving models have demonstrated strong performance in open-loop evaluations but often suffer from cascading errors and poor generalization in closed-loop settings. To address this gap, we propose Model-based Policy Adaptation (MPA), a general framework that enhances the robustness and safety of pretrained E2E driving agents during deployment. MPA first generates diverse counterfactual trajectories using a geometry-consistent simulation engine, exposing the agent to scenarios beyond the original dataset. Based on this generated data, MPA trains a diffusion-based policy adapter to refine the base policy's predictions and a multi-step Q value model to evaluate long-term outcomes. At inference time, the adapter proposes multiple trajectory candidates, and the Q value model selects the one with the highest expected utility. Experiments on the nuScenes benchmark using a photorealistic closed-loop simulator demonstrate that MPA significantly improves performance across in-domain, out-of-domain, and safety-critical scenarios. We further investigate how the scale of counterfactual data and inference-time guidance strategies affect overall effectiveness. |
| 2025-11-26 | [Self-Transparency Failures in Expert-Persona LLMs: A Large-Scale Behavioral Audit](http://arxiv.org/abs/2511.21569v1) | Alex Diep | If a language model cannot reliably disclose its AI identity in expert contexts, users cannot trust its competence boundaries. This study examines self-transparency in models assigned professional personas within high-stakes domains where false expertise risks user harm. Using a common-garden design, sixteen open-weight models (4B--671B parameters) were audited across 19,200 trials. Models exhibited sharp domain-specific inconsistency: a Financial Advisor persona elicited 30.8% disclosure initially, while a Neurosurgeon persona elicited only 3.5%. This creates preconditions for a "Reverse Gell-Mann Amnesia" effect, where transparency in some domains leads users to overgeneralize trust to contexts where disclosure fails. Disclosure ranged from 2.8% to 73.6%, with a 14B model reaching 61.4% while a 70B produced just 4.1%. Model identity predicted behavior better than parameter count ($ΔR_{adj}^{2} = 0.359$ vs 0.018). Reasoning optimization actively suppressed self-transparency in some models, with reasoning variants showing up to 48.4% lower disclosure than base counterparts. Bayesian validation with Rogan--Gladen correction confirmed robustness to measurement error ($κ= 0.908$). These findings demonstrate transparency reflects training factors rather than scale. Organizations cannot assume safety properties transfer to deployment contexts, requiring deliberate behavior design and empirical verification. |
| 2025-11-26 | [Predictive Safety Shield for Dyna-Q Reinforcement Learning](http://arxiv.org/abs/2511.21531v1) | Jin Pin, Krasowski Hanna et al. | Obtaining safety guarantees for reinforcement learning is a major challenge to achieve applicability for real-world tasks. Safety shields extend standard reinforcement learning and achieve hard safety guarantees. However, existing safety shields commonly use random sampling of safe actions or a fixed fallback controller, therefore disregarding future performance implications of different safe actions. In this work, we propose a predictive safety shield for model-based reinforcement learning agents in discrete space. Our safety shield updates the Q-function locally based on safe predictions, which originate from a safe simulation of the environment model. This shielding approach improves performance while maintaining hard safety guarantees. Our experiments on gridworld environments demonstrate that even short prediction horizons can be sufficient to identify the optimal path. We observe that our approach is robust to distribution shifts, e.g., between simulation and reality, without requiring additional training. |
| 2025-11-26 | [Pessimistic Verification for Open Ended Math Questions](http://arxiv.org/abs/2511.21522v1) | Yanxing Huang, Zihan Tang et al. | The key limitation of the verification performance lies in the ability of error detection. With this intuition we designed several variants of pessimistic verification, which are simple workflows that could significantly improve the verification of open-ended math questions. In pessimistic verification we construct multiple parallel verifications for the same proof, and the proof is deemed incorrect if any one of them reports an error. This simple technique significantly improves the performance across many math verification benchmarks without incurring substantial computational resources. Its token efficiency even surpassed extended long-CoT in test-time scaling. Our case studies further indicate that the majority of false negatives in stronger models are actually caused by annotation errors in the original dataset, so our method's performance is in fact underestimated. Self-verification for mathematical problems can effectively improve the reliability and performance of language model outputs, and it also plays a critical role in enabling long-horizon mathematical tasks. We believe that research on pessimistic verification will help enhance the mathematical capabilities of language models across a wide range of tasks. |
| 2025-11-26 | [MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning](http://arxiv.org/abs/2511.21460v1) | Junjian Wang, Lidan Zhao et al. | Ensuring the safety of embodied AI agents during task planning is critical for real-world deployment, especially in household environments where dangerous instructions pose significant risks. Existing methods often suffer from either high computational costs due to preference alignment training or over-rejection when using single-agent safety prompts. To address these limitations, we propose MADRA, a training-free Multi-Agent Debate Risk Assessment framework that leverages collective reasoning to enhance safety awareness without sacrificing task performance. MADRA employs multiple LLM-based agents to debate the safety of a given instruction, guided by a critical evaluator that scores responses based on logical soundness, risk identification, evidence quality, and clarity. Through iterative deliberation and consensus voting, MADRA significantly reduces false rejections while maintaining high sensitivity to dangerous tasks. Additionally, we introduce a hierarchical cognitive collaborative planning framework that integrates safety, memory, planning, and self-evolution mechanisms to improve task success rates through continuous learning. We also contribute SafeAware-VH, a benchmark dataset for safety-aware task planning in VirtualHome, containing 800 annotated instructions. Extensive experiments on AI2-THOR and VirtualHome demonstrate that our approach achieves over 90% rejection of unsafe tasks while ensuring that safe-task rejection is low, outperforming existing methods in both safety and execution efficiency. Our work provides a scalable, model-agnostic solution for building trustworthy embodied agents. |
| 2025-11-26 | [Text-to-SQL as Dual-State Reasoning: Integrating Adaptive Context and Progressive Generation](http://arxiv.org/abs/2511.21402v1) | Zhifeng Hao, Qibin Song et al. | Recent divide-and-conquer reasoning approaches, particularly those based on Chain-of-Thought (CoT), have substantially improved the Text-to-SQL capabilities of Large Language Models (LLMs). However, when applied to complex enterprise databases, such methods struggle to maintain coherent reasoning due to limited context capacity, unreliable schema linking, and weak grounding in database semantics. To overcome these issues, we introduce DSR-SQL, a \textbf{D}ual-\textbf{S}tate \textbf{R}easoning framework that models Text-to-SQL as an interaction between an adaptive context state and a progressive generation state. The first constructs a compact, semantically faithful environment by refining large schemas and selecting relevant structures, while the second formalizes SQL synthesis as feedback-guided state transitions, enabling the model to self-correct and align with user intent. Without any post-training or in-context examples, DSR-SQL achieves competitive performance, reaching 35.28\% execution accuracy on Spider 2.0-Snow and 68.32\% on BIRD development set. Our implementation will be open-sourced at: https://github.com/DMIRLAB-Group/DSR-SQL. |
| 2025-11-26 | [Can LLMs extract human-like fine-grained evidence for evidence-based fact-checking?](http://arxiv.org/abs/2511.21401v1) | Antonín Jarolím, Martin Fajčík et al. | Misinformation frequently spreads in user comments under online news articles, highlighting the need for effective methods to detect factually incorrect information. To strongly support or refute claims extracted from such comments, it is necessary to identify relevant documents and pinpoint the exact text spans that justify or contradict each claim. This paper focuses on the latter task -- fine-grained evidence extraction for Czech and Slovak claims. We create new dataset, containing two-way annotated fine-grained evidence created by paid annotators. We evaluate large language models (LLMs) on this dataset to assess their alignment with human annotations. The results reveal that LLMs often fail to copy evidence verbatim from the source text, leading to invalid outputs. Error-rate analysis shows that the {llama3.1:8b model achieves a high proportion of correct outputs despite its relatively small size, while the gpt-oss-120b model underperforms despite having many more parameters. Furthermore, the models qwen3:14b, deepseek-r1:32b, and gpt-oss:20b demonstrate an effective balance between model size and alignment with human annotations. |
| 2025-11-26 | [Monet: Reasoning in Latent Visual Space Beyond Images and Language](http://arxiv.org/abs/2511.21395v1) | Qixun Wang, Yang Shi et al. | "Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at https://github.com/NOVAglow646/Monet. |
| 2025-11-26 | [Thinking With Bounding Boxes: Enhancing Spatio-Temporal Video Grounding via Reinforcement Fine-Tuning](http://arxiv.org/abs/2511.21375v1) | Xin Gu, Haoji Zhang et al. | Spatio-temporal video grounding (STVG) requires localizing a target object in untrimmed videos both temporally and spatially from natural language descriptions. Despite their strong language understanding, multimodal large language models (MLLMs) underperform on STVG due to misaligned training objectives and weak fine-grained region-word alignment in standard visual encoders. To address this, we propose STVG-o1, the first framework that enables off-the-shelf MLLMs to achieve state-of-the-art STVG performance without any architectural modifications. Our method introduces a bounding-box chain-of-thought mechanism that explicitly reasons about spatio-temporal locations in an intermediate step before producing the final prediction. We further design a multi-dimensional reinforcement reward function consisting of format, consistency, temporal, spatial, and think rewards, which provides geometry-aware supervision through reinforcement fine-tuning. Evaluated on HCSTVG-v1/v2 and VidSTG, STVG-o1 sets new state-of-the-art results on HCSTVG, outperforming the best task-specific method by 7.3\% m\_tIoU on HCSTVG-v1, matching specialized models on VidSTG, and surpassing all existing MLLM-based approaches by large margins. It also demonstrates strong open-vocabulary generalization across datasets, establishing MLLMs as viable and powerful backbones for precise spatio-temporal grounding. Our code and models will be released. |
| 2025-11-26 | [Hybrid SIFT-SNN for Efficient Anomaly Detection of Traffic Flow-Control Infrastructure](http://arxiv.org/abs/2511.21337v1) | Munish Rathee, Boris Bačić et al. | This paper presents the SIFT-SNN framework, a low-latency neuromorphic signal-processing pipeline for real-time detection of structural anomalies in transport infrastructure. The proposed approach integrates Scale-Invariant Feature Transform (SIFT) for spatial feature encoding with a latency-driven spike conversion layer and a Leaky Integrate-and-Fire (LIF) Spiking Neural Network (SNN) for classification. The Auckland Harbour Bridge dataset is recorded under various weather and lighting conditions, comprising 6,000 labelled frames that include both real and synthetically augmented unsafe cases. The presented system achieves a classification accuracy of 92.3% (+- 0.8%) with a per-frame inference time of 9.5 ms. Achieved sub-10 millisecond latency, combined with sparse spike activity (8.1%), enables real-time, low-power edge deployment. Unlike conventional CNN-based approaches, the hybrid SIFT-SNN pipeline explicitly preserves spatial feature grounding, enhances interpretability, supports transparent decision-making, and operates efficiently on embedded hardware. Although synthetic augmentation improved robustness, generalisation to unseen field conditions remains to be validated. The SIFT-SNN framework is validated through a working prototype deployed on a consumer-grade system and framed as a generalisable case study in structural safety monitoring for movable concrete barriers, which, as a traffic flow-control infrastructure, is deployed in over 20 cities worldwide. |
| 2025-11-26 | [Response-Based Frequency Stability Assessment under Multi-Scale Disturbances in High-Renewable Power Systems](http://arxiv.org/abs/2511.21269v1) | Jinhui Chen, Huadong Sun et al. | In high-renewable power systems, active-power disturbances are becoming larger and exhibit increasingly diverse time scales, which complicates frequency stability assessment under unanticipated events. This paper presents a response-based frequency stability assessment method that uses disturbance power, inferred from generator electrical responses, to provide a unified treatment of multi-scale disturbances. Unanticipated disturbances are first classified into short-term and permanent events; permanent disturbances are further divided into step, second-level slope and minute-level slope disturbances. Based on the measured power responses of generator groups, a unified disturbance-power model is constructed to identify the disturbance type online and to quantify disturbance intensity through the disturbance power and its rate of change. Analytical frequency-response models are then derived for each disturbance class. For step disturbances, the maximum tolerable disturbance power is obtained under steady-state and transient frequency deviation constraints, and a safety-margin index is defined. For slope-type disturbances, an improved system frequency response (SFR) model and the rotor motion equation after exhaustion of primary frequency regulation are used to compute the over-limit time of frequency deviation. The proposed response-based assessment method is validated on the CSEE-FS frequency-stability benchmark system, demonstrating its effectiveness and accuracy for quantitative frequency stability assessment in high-renewable power systems. |
| 2025-11-26 | [Self-Guided Defense: Adaptive Safety Alignment for Reasoning Models via Synthesized Guidelines](http://arxiv.org/abs/2511.21214v1) | Yuhang Wang, Yanxu Zhu et al. | Reasoning models have demonstrated remarkable capabilities in complex reasoning tasks. However, ensuring their safety against adversarial jailbreak prompts remains a critical challenge. Due to the covert and deceptive nature of such prompts, they can often evade built-in safety mechanisms and lead to the generation of harmful content. This underscores the need for an adaptive safety alignment approach that enables models to autonomously reinforce their defenses in response to adversarial inputs. This paper introduces the Synthesized Guideline-based Adaptive Safety Alignment (SGASA) framework, which internalizes model-generated safety guidelines to strengthen models' ability to enhance robustness against harmful adversarial prompts while minimizing unnecessary refusals of benign requests. SGASA consists of two key stages: Data Pre-synthesis, which generates safety guidelines and augmented prompts; and Alignment Fine-tuning, which leverages Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO) to embed these guidelines into the model. Extensive experiments across multiple datasets demonstrate that SGASA significantly improves model safety, validating its adaptive and scalable effectiveness. |
| 2025-11-26 | [TEAR: Temporal-aware Automated Red-teaming for Text-to-Video Models](http://arxiv.org/abs/2511.21145v1) | Jiaming He, Guanyu Hou et al. | Text-to-Video (T2V) models are capable of synthesizing high-quality, temporally coherent dynamic video content, but the diverse generation also inherently introduces critical safety challenges. Existing safety evaluation methods,which focus on static image and text generation, are insufficient to capture the complex temporal dynamics in video generation. To address this, we propose a TEmporal-aware Automated Red-teaming framework, named TEAR, an automated framework designed to uncover safety risks specifically linked to the dynamic temporal sequencing of T2V models. TEAR employs a temporal-aware test generator optimized via a two-stage approach: initial generator training and temporal-aware online preference learning, to craft textually innocuous prompts that exploit temporal dynamics to elicit policy-violating video output. And a refine model is adopted to improve the prompt stealthiness and adversarial effectiveness cyclically. Extensive experimental evaluation demonstrates the effectiveness of TEAR across open-source and commercial T2V systems with over 80% attack success rate, a significant boost from prior best result of 57%. |
| 2025-11-26 | [MLPMoE: Zero-Shot Architectural Metamorphosis of Dense LLM MLPs into Static Mixture-of-Experts](http://arxiv.org/abs/2511.21089v1) | Ivan Novikov | Large Language Models (LLMs) are predominantly deployed as dense transformers, where every parameter in every feed-forward block is activated for every token. While architecturally simple, this is computationally inefficient, since inference costs scale linearly with parameter count. Recent upcycling methods such as MoEfication, CMoE, ToMoE, and MoORE reveal that much of the useful computation lives in sparse, semi-modular substructures inside dense feed-forward networks, but these approaches typically rely on clustering, activation profiling, singular value decomposition, or custom routing that requires calibration data. This paper introduces MLPMoE (MLP Mixture-of-Experts), a training-free, deterministic transformation that restructures the dense MLP in transformer blocks into a static, high-cardinality mixture of experts. The transformation uses simple tensor slicing and summation, reinterpreting the algebra of tensor parallelism as a topological conversion rather than a distributed training pattern. We further introduce Fractal Fade (differential branch sparsity) and Compensated Pruning (variance-preserving branch reduction) as lightweight mechanisms for structured sparsity. On Qwen2.5-0.5B-Instruct and DeepSeek-R1-Distill-Llama-8B, the zero-shot MLPMoE transform changes a proxy perplexity metric by less than 0.05 percent while keeping the parameter count effectively constant. On the 8B model, differential sparsity removes about 20 percent of MLP parameters while keeping perplexity within about 2 percent of the dense baseline. The method operates entirely post hoc on existing checkpoints and does not require gradients, calibration sets, or router training. Code is available at https://gist.github.com/iwallarm/fc2ef1eddf226ca7814f9e5e2ae9bad1 |
| 2025-11-26 | [OVOD-Agent: A Markov-Bandit Framework for Proactive Visual Reasoning and Self-Evolving Detection](http://arxiv.org/abs/2511.21064v1) | Chujie Wang, Jianyu Lu et al. | Open-Vocabulary Object Detection (OVOD) aims to enable detectors to generalize across categories by leveraging semantic information. Although existing methods are pretrained on large vision-language datasets, their inference is still limited to fixed category names, creating a gap between multimodal training and unimodal inference. Previous work has shown that improving textual representation can significantly enhance OVOD performance, indicating that the textual space is still underexplored. To this end, we propose OVOD-Agent, which transforms passive category matching into proactive visual reasoning and self-evolving detection. Inspired by the Chain-of-Thought (CoT) paradigm, OVOD-Agent extends the textual optimization process into an interpretable Visual-CoT with explicit actions. OVOD's lightweight nature makes LLM-based management unsuitable; instead, we model visual context transitions as a Weakly Markovian Decision Process (w-MDP) over eight state spaces, which naturally represents the agent's state, memory, and interaction dynamics. A Bandit module generates exploration signals under limited supervision, helping the agent focus on uncertain regions and adapt its detection policy. We further integrate Markov transition matrices with Bandit trajectories for self-supervised Reward Model (RM) optimization, forming a closed loop from Bandit exploration to RM learning. Experiments on COCO and LVIS show that OVOD-Agent provides consistent improvements across OVOD backbones, particularly on rare categories, confirming the effectiveness of the proposed framework. |
| 2025-11-26 | [A Unified Understanding of Offline Data Selection and Online Self-refining Generation for Post-training LLMs](http://arxiv.org/abs/2511.21056v1) | Quan Xiao, Tianyi Chen | Offline data selection and online self-refining generation, which enhance the data quality, are crucial steps in adapting large language models (LLMs) to specific downstream tasks. We tackle offline data selection and online self-refining generations through an optimization perspective. Specifically, bilevel data selection is used for offline data selection with respect to the validation dataset, and we treat online self-refining generation as a model adaptation step of selecting the model trained on current responses that best fits the validation data. Our framework offers a unified understanding of offline data selection and self-refining generation by assigning a learned data weight to each question and response, either explicitly or implicitly. For the first time, we theoretically demonstrate the effectiveness of the bilevel data selection framework and demonstrate its performance gains over unfiltered direct mixing baselines. By combining offline data with validation-weighted online generations, our method enhances fine-tuning performance. Experiments on quality enhancement and safety-aware LLM fine-tuning validate its effectiveness. |
| 2025-11-26 | [Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs](http://arxiv.org/abs/2511.21050v1) | Dongkyu Derek Cho, Huan Song et al. | Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs. |
| 2025-11-26 | [GuardTrace-VL: Detecting Unsafe Multimodel Reasoning via Iterative Safety Supervision](http://arxiv.org/abs/2511.20994v1) | Yuxiao Xiang, Junchi Chen et al. | Multimodal large reasoning models (MLRMs) are increasingly deployed for vision-language tasks that produce explicit intermediate rationales. However, reasoning traces can contain unsafe content even when the final answer is non-harmful, creating deployment risks. Existing multimodal safety guards primarily evaluate only the input question and the final answer, neglecting the intermediate reasoning process. This oversight allows undetected harm, such as biased inferences or policy-violating use of visual context, to emerge during reasoning. We introduce GuardTrace-VL, a vision-aware safety auditor that monitors the full Question-Thinking-Answer (QTA) pipeline via joint image-text analysis, enabling detection of unsafe content as it emerges in the reasoning stage. To support training and evaluation, we construct the GuardTrace dataset, which is generated through diverse prompting strategies and refined via a MLRM- and human-based voting and verification pipeline. Furthermore, we propose a three-stage progressive training scheme combined with the data refinement process, enabling the model to learn nuanced and context-dependent safety preferences according to different risk levels. On our proposed test set covering both in-domain and out-of-domain scenarios, GuardTrace-VL model achieves an F1 score of 93.1% on unsafe reasoning detection tasks, representing a 13.5% improvement in F1 score compared to the previous strongest multimodal safety defense methods. The codes will be made publicly available. |
| 2025-11-26 | [TrafficLens: Multi-Camera Traffic Video Analysis Using LLMs](http://arxiv.org/abs/2511.20965v1) | Md Adnan Arefeen, Biplob Debnath et al. | Traffic cameras are essential in urban areas, playing a crucial role in intelligent transportation systems. Multiple cameras at intersections enhance law enforcement capabilities, traffic management, and pedestrian safety. However, efficiently managing and analyzing multi-camera feeds poses challenges due to the vast amount of data. Analyzing such huge video data requires advanced analytical tools. While Large Language Models (LLMs) like ChatGPT, equipped with retrieval-augmented generation (RAG) systems, excel in text-based tasks, integrating them into traffic video analysis demands converting video data into text using a Vision-Language Model (VLM), which is time-consuming and delays the timely utilization of traffic videos for generating insights and investigating incidents. To address these challenges, we propose TrafficLens, a tailored algorithm for multi-camera traffic intersections. TrafficLens employs a sequential approach, utilizing overlapping coverage areas of cameras. It iteratively applies VLMs with varying token limits, using previous outputs as prompts for subsequent cameras, enabling rapid generation of detailed textual descriptions while reducing processing time. Additionally, TrafficLens intelligently bypasses redundant VLM invocations through an object-level similarity detector. Experimental results with real-world datasets demonstrate that TrafficLens reduces video-to-text conversion time by up to $4\times$ while maintaining information accuracy. |

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



