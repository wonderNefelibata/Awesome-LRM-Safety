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
| 2025-06-09 | [Solving Inequality Proofs with Large Language Models](http://arxiv.org/abs/2506.07927v1) | Jiayi Sheng, Luna Lyu et al. | Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at https://ineqmath.github.io/. |
| 2025-06-09 | [WeThink: Toward General-purpose Vision-Language Reasoning via Reinforcement Learning](http://arxiv.org/abs/2506.07905v1) | Jie Yang, Feipeng Ma et al. | Building on the success of text-based reasoning models like DeepSeek-R1, extending these capabilities to multimodal reasoning holds great promise. While recent works have attempted to adapt DeepSeek-R1-style reinforcement learning (RL) training paradigms to multimodal large language models (MLLM), focusing on domain-specific tasks like math and visual perception, a critical question remains: How can we achieve the general-purpose visual-language reasoning through RL? To address this challenge, we make three key efforts: (1) A novel Scalable Multimodal QA Synthesis pipeline that autonomously generates context-aware, reasoning-centric question-answer (QA) pairs directly from the given images. (2) The open-source WeThink dataset containing over 120K multimodal QA pairs with annotated reasoning paths, curated from 18 diverse dataset sources and covering various question domains. (3) A comprehensive exploration of RL on our dataset, incorporating a hybrid reward mechanism that combines rule-based verification with model-based assessment to optimize RL training efficiency across various task domains. Across 14 diverse MLLM benchmarks, we demonstrate that our WeThink dataset significantly enhances performance, from mathematical reasoning to diverse general multimodal tasks. Moreover, we show that our automated data pipeline can continuously increase data diversity to further improve model performance. |
| 2025-06-09 | [Secure Distributed Learning for CAVs: Defending Against Gradient Leakage with Leveled Homomorphic Encryption](http://arxiv.org/abs/2506.07894v1) | Muhammad Ali Najjar, Ren-Yi Huang et al. | Federated Learning (FL) enables collaborative model training across distributed clients without sharing raw data, making it a promising approach for privacy-preserving machine learning in domains like Connected and Autonomous Vehicles (CAVs). However, recent studies have shown that exchanged model gradients remain susceptible to inference attacks such as Deep Leakage from Gradients (DLG), which can reconstruct private training data. While existing defenses like Differential Privacy (DP) and Secure Multi-Party Computation (SMPC) offer protection, they often compromise model accuracy. To that end, Homomorphic Encryption (HE) offers a promising alternative by enabling lossless computation directly on encrypted data, thereby preserving both privacy and model utility. However, HE introduces significant computational and communication overhead, which can hinder its practical adoption. To address this, we systematically evaluate various leveled HE schemes to identify the most suitable for FL in resource-constrained environments due to its ability to support fixed-depth computations without requiring costly bootstrapping. Our contributions in this paper include a comprehensive evaluation of HE schemes for real-world FL applications, a selective encryption strategy that targets only the most sensitive gradients to minimize computational overhead, and the development of a full HE-based FL pipeline that effectively mitigates DLG attacks while preserving model accuracy. We open-source our implementation to encourage reproducibility and facilitate adoption in safety-critical domains. |
| 2025-06-09 | [CrosswalkNet: An Optimized Deep Learning Framework for Pedestrian Crosswalk Detection in Aerial Images with High-Performance Computing](http://arxiv.org/abs/2506.07885v1) | Zubin Bhuyan, Yuanchang Xie et al. | With the increasing availability of aerial and satellite imagery, deep learning presents significant potential for transportation asset management, safety analysis, and urban planning. This study introduces CrosswalkNet, a robust and efficient deep learning framework designed to detect various types of pedestrian crosswalks from 15-cm resolution aerial images. CrosswalkNet incorporates a novel detection approach that improves upon traditional object detection strategies by utilizing oriented bounding boxes (OBB), enhancing detection precision by accurately capturing crosswalks regardless of their orientation. Several optimization techniques, including Convolutional Block Attention, a dual-branch Spatial Pyramid Pooling-Fast module, and cosine annealing, are implemented to maximize performance and efficiency. A comprehensive dataset comprising over 23,000 annotated crosswalk instances is utilized to train and validate the proposed framework. The best-performing model achieves an impressive precision of 96.5% and a recall of 93.3% on aerial imagery from Massachusetts, demonstrating its accuracy and effectiveness. CrosswalkNet has also been successfully applied to datasets from New Hampshire, Virginia, and Maine without transfer learning or fine-tuning, showcasing its robustness and strong generalization capability. Additionally, the crosswalk detection results, processed using High-Performance Computing (HPC) platforms and provided in polygon shapefile format, have been shown to accelerate data processing and detection, supporting real-time analysis for safety and mobility applications. This integration offers policymakers, transportation engineers, and urban planners an effective instrument to enhance pedestrian safety and improve urban mobility. |
| 2025-06-09 | [R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation](http://arxiv.org/abs/2506.07826v1) | William Ljungbergh, Bernardo Taveira et al. | Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we will release our dataset and code, see https://research.zenseact.com/publications/R3D2/. |
| 2025-06-09 | [Looking Beyond Visible Cues: Implicit Video Question Answering via Dual-Clue Reasoning](http://arxiv.org/abs/2506.07811v1) | Tieyuan Chen, Huabin Liu et al. | Video Question Answering (VideoQA) aims to answer natural language questions based on the given video, with prior work primarily focusing on identifying the duration of relevant segments, referred to as explicit visual evidence. However, explicit visual evidence is not always directly available, particularly when questions target symbolic meanings or deeper intentions, leading to significant performance degradation. To fill this gap, we introduce a novel task and dataset, $\textbf{I}$mplicit $\textbf{V}$ideo $\textbf{Q}$uestion $\textbf{A}$nswering (I-VQA), which focuses on answering questions in scenarios where explicit visual evidence is inaccessible. Given an implicit question and its corresponding video, I-VQA requires answering based on the contextual visual cues present within the video. To tackle I-VQA, we propose a novel reasoning framework, IRM (Implicit Reasoning Model), incorporating dual-stream modeling of contextual actions and intent clues as implicit reasoning chains. IRM comprises the Action-Intent Module (AIM) and the Visual Enhancement Module (VEM). AIM deduces and preserves question-related dual clues by generating clue candidates and performing relation deduction. VEM enhances contextual visual representation by leveraging key contextual clues. Extensive experiments validate the effectiveness of our IRM in I-VQA tasks, outperforming GPT-4o, OpenAI-o3, and fine-tuned VideoChat2 by $0.76\%$, $1.37\%$, and $4.87\%$, respectively. Additionally, IRM performs SOTA on similar implicit advertisement understanding and future prediction in traffic-VQA. Datasets and codes are available for double-blind review in anonymous repo: https://github.com/tychen-SJTU/Implicit-VideoQA. |
| 2025-06-09 | [Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability](http://arxiv.org/abs/2506.07804v1) | Jie Bao, Chuangyin Dang et al. | As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction. |
| 2025-06-09 | [Deep Equivariant Multi-Agent Control Barrier Functions](http://arxiv.org/abs/2506.07755v1) | Nikolaos Bousias, Lars Lindemann et al. | With multi-agent systems increasingly deployed autonomously at scale in complex environments, ensuring safety of the data-driven policies is critical. Control Barrier Functions have emerged as an effective tool for enforcing safety constraints, yet existing learning-based methods often lack in scalability, generalization and sampling efficiency as they overlook inherent geometric structures of the system. To address this gap, we introduce symmetries-infused distributed Control Barrier Functions, enforcing the satisfaction of intrinsic symmetries on learnable graph-based safety certificates. We theoretically motivate the need for equivariant parametrization of CBFs and policies, and propose a simple, yet efficient and adaptable methodology for constructing such equivariant group-modular networks via the compatible group actions. This approach encodes safety constraints in a distributed data-efficient manner, enabling zero-shot generalization to larger and denser swarms. Through extensive simulations on multi-robot navigation tasks, we demonstrate that our method outperforms state-of-the-art baselines in terms of safety, scalability, and task success rates, highlighting the importance of embedding symmetries in safe distributed neural policies. |
| 2025-06-09 | [RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards](http://arxiv.org/abs/2506.07736v1) | Jingnan Zheng, Xiangtian Ji et al. | Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. |
| 2025-06-09 | [A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs](http://arxiv.org/abs/2506.07696v1) | Yongqi Zhao, Xinrui Zhang et al. | Testing cloud-controlled intelligent connected vehicles (ICVs) requires simulation environments that faithfully emulate both vehicle behavior and realistic communication latencies. This paper proposes a latency-aware co-simulation platform integrating CarMaker and Vissim to evaluate safety and comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two communication latency models, derived from empirical 5G measurements in China and Hungary, are incorporated and statistically modeled using Gamma distributions. A proactive conflict module (PCM) is proposed to dynamically control background vehicles and generate safety-critical scenarios. The platform is validated through experiments involving an exemplary system under test (SUT) across six testing conditions combining two PCM modes (enabled/disabled) and three latency conditions (none, China, Hungary). Safety and comfort are assessed using metrics including collision rate, distance headway, post-encroachment time, and the spectral characteristics of longitudinal acceleration. Results show that the PCM effectively increases driving environment criticality, while V2C latency primarily affects ride comfort. These findings confirm the platform's effectiveness in systematically evaluating cloud-controlled ICVs under diverse testing conditions. |
| 2025-06-09 | [Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models](http://arxiv.org/abs/2506.07645v1) | Maciej Chrabąszcz, Katarzyna Lorenc et al. | Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research. |
| 2025-06-09 | [TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts](http://arxiv.org/abs/2506.07596v1) | Torsten Krauß, Hamid Dashtbani et al. | Machine learning is advancing rapidly, with applications bringing notable benefits, such as improvements in translation and code generation. Models like ChatGPT, powered by Large Language Models (LLMs), are increasingly integrated into daily life. However, alongside these benefits, LLMs also introduce social risks. Malicious users can exploit LLMs by submitting harmful prompts, such as requesting instructions for illegal activities. To mitigate this, models often include a security mechanism that automatically rejects such harmful prompts. However, they can be bypassed through LLM jailbreaks. Current jailbreaks often require significant manual effort, high computational costs, or result in excessive model modifications that may degrade regular utility.   We introduce TwinBreak, an innovative safety alignment removal method. Building on the idea that the safety mechanism operates like an embedded backdoor, TwinBreak identifies and prunes parameters responsible for this functionality. By focusing on the most relevant model layers, TwinBreak performs fine-grained analysis of parameters essential to model utility and safety. TwinBreak is the first method to analyze intermediate outputs from prompts with high structural and content similarity to isolate safety parameters. We present the TwinPrompt dataset containing 100 such twin prompts. Experiments confirm TwinBreak's effectiveness, achieving 89% to 98% success rates with minimal computational requirements across 16 LLMs from five vendors. |
| 2025-06-09 | [Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning](http://arxiv.org/abs/2506.07501v1) | Libo Wang | In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement. |
| 2025-06-09 | [Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models](http://arxiv.org/abs/2506.07468v1) | Mickel Liu, Liwei Jiang et al. | Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL). |
| 2025-06-09 | [When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment](http://arxiv.org/abs/2506.07452v1) | Yuxin Xiao, Sana Tonekaboni et al. | Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety. |
| 2025-06-09 | [Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition](http://arxiv.org/abs/2506.07436v1) | Nishi Chaudhary, S M Jamil Uddin et al. | The recent emergence of multimodal large language models (LLMs) has introduced new opportunities for improving visual hazard recognition on construction sites. Unlike traditional computer vision models that rely on domain-specific training and extensive datasets, modern LLMs can interpret and describe complex visual scenes using simple natural language prompts. However, despite growing interest in their applications, there has been limited investigation into how different LLMs perform in safety-critical visual tasks within the construction domain. To address this gap, this study conducts a comparative evaluation of five state-of-the-art LLMs: Claude-3 Opus, GPT-4.5, GPT-4o, GPT-o3, and Gemini 2.0 Pro, to assess their ability to identify potential hazards from real-world construction images. Each model was tested under three prompting strategies: zero-shot, few-shot, and chain-of-thought (CoT). Zero-shot prompting involved minimal instruction, few-shot incorporated basic safety context and a hazard source mnemonic, and CoT provided step-by-step reasoning examples to scaffold model thinking. Quantitative analysis was performed using precision, recall, and F1-score metrics across all conditions. Results reveal that prompting strategy significantly influenced performance, with CoT prompting consistently producing higher accuracy across models. Additionally, LLM performance varied under different conditions, with GPT-4.5 and GPT-o3 outperforming others in most settings. The findings also demonstrate the critical role of prompt design in enhancing the accuracy and consistency of multimodal LLMs for construction safety applications. This study offers actionable insights into the integration of prompt engineering and LLMs for practical hazard recognition, contributing to the development of more reliable AI-assisted safety systems. |
| 2025-06-09 | [Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures](http://arxiv.org/abs/2506.07402v1) | Yukai Zhou, Sibei Yang et al. | Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about their security. While jailbreak attacks highlight failures under overtly harmful queries, they overlook a critical risk: incorrectly answering harmless-looking inputs can be dangerous and cause real-world harm (Implicit Harm). We systematically reformulate the LLM risk landscape through a structured quadrant perspective based on output factuality and input harmlessness, uncovering an overlooked high-risk region. To investigate this gap, we propose JailFlipBench, a benchmark aims to capture implicit harm, spanning single-modal, multimodal, and factual extension scenarios with diverse evaluation metrics. We further develop initial JailFlip attack methodologies and conduct comprehensive evaluations across multiple open-source and black-box LLMs, show that implicit harm present immediate and urgent real-world risks, calling for broader LLM safety assessments and alignment beyond conventional jailbreak paradigms. |
| 2025-06-09 | [Refusal-Feature-guided Teacher for Safe Finetuning via Data Filtering and Alignment Distillation](http://arxiv.org/abs/2506.07356v1) | Seokil Ham, Yubin Choi et al. | Recently, major AI service providers such as Google and OpenAI have introduced Finetuning-as-a-Service, which enables users to customize Large Language Models (LLMs) for specific downstream tasks using their own data. However, this service is vulnerable to degradation of LLM safety-alignment when user data contains harmful prompts. While some prior works address this issue, fundamentally filtering harmful data from user data remains unexplored. Motivated by our observation that a directional representation reflecting refusal behavior (called the refusal feature) obtained from safety-aligned LLMs can inherently distinguish between harmful and harmless prompts, we propose the Refusal-Feature-guided Teacher (ReFT). Our ReFT model is trained to identify harmful prompts based on the similarity between input prompt features and its refusal feature. During finetuning, the ReFT model serves as a teacher that filters harmful prompts from user data and distills alignment knowledge into the base model. Extensive experiments demonstrate that our ReFT-based finetuning strategy effectively minimizes harmful outputs and enhances finetuning accuracy for user-specific tasks, offering a practical solution for secure and reliable deployment of LLMs in Finetuning-as-a-Service. |
| 2025-06-09 | [Distributed Risk-Sensitive Safety Filters for Uncertain Discrete-Time Systems](http://arxiv.org/abs/2506.07347v1) | Armin Lederer, Erfaun Noorani et al. | Ensuring safety in multi-agent systems is a significant challenge, particularly in settings where centralized coordination is impractical. In this work, we propose a novel risk-sensitive safety filter for discrete-time multi-agent systems with uncertain dynamics that leverages control barrier functions (CBFs) defined through value functions. Our approach relies on centralized risk-sensitive safety conditions based on exponential risk operators to ensure robustness against model uncertainties. We introduce a distributed formulation of the safety filter by deriving two alternative strategies: one based on worst-case anticipation and another on proximity to a known safe policy. By allowing agents to switch between strategies, feasibility can be ensured. Through detailed numerical evaluations, we demonstrate the efficacy of our approach in maintaining safety without being overly conservative. |
| 2025-06-08 | [BR-MPPI: Barrier Rate guided MPPI for Enforcing Multiple Inequality Constraints with Learned Signed Distance Field](http://arxiv.org/abs/2506.07325v1) | Hardik Parwana, Taekyung Kim et al. | Model Predictive Path Integral (MPPI) controller is used to solve unconstrained optimal control problems and Control Barrier Function (CBF) is a tool to impose strict inequality constraints, a.k.a, barrier constraints. In this work, we propose an integration of these two methods that employ CBF-like conditions to guide the control sampling procedure of MPPI. CBFs provide an inequality constraint restricting the rate of change of barrier functions by a classK function of the barrier itself. We instead impose the CBF condition as an equality constraint by choosing a parametric linear classK function and treating this parameter as a state in an augmented system. The time derivative of this parameter acts as an additional control input that is designed by MPPI. A cost function is further designed to reignite Nagumo's theorem at the boundary of the safe set by promoting specific values of classK parameter to enforce safety. Our problem formulation results in an MPPI subject to multiple state and control-dependent equality constraints which are non-trivial to satisfy with randomly sampled control inputs. We therefore also introduce state transformations and control projection operations, inspired by the literature on path planning for manifolds, to resolve the aforementioned issue. We show empirically through simulations and experiments on quadrotor that our proposed algorithm exhibits better sampled efficiency and enhanced capability to operate closer to the safe set boundary over vanilla MPPI. |

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



