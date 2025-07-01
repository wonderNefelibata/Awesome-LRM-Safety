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
| 2025-06-30 | [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](http://arxiv.org/abs/2506.24119v1) | Bo Liu, Leon Guertler et al. | Recent advances in reinforcement learning have shown that language models can develop sophisticated reasoning through training on tasks with verifiable rewards, but these approaches depend on human-curated problem-answer pairs and domain-specific reward engineering. We introduce SPIRAL, a self-play framework where models learn by playing multi-turn, zero-sum games against continuously improving versions of themselves, eliminating the need for human supervision. Through self-play, SPIRAL generates an infinite curriculum of progressively challenging problems as models must constantly adapt to stronger opponents. To enable this self-play training at scale, We implement a fully online, multi-turn, multi-agent reinforcement learning system for LLMs and propose role-conditioned advantage estimation (RAE) to stabilize multi-agent training. Using SPIRAL, self-play on zero-sum games produces reasoning capabilities that transfer broadly. Training Qwen3-4B-Base on Kuhn Poker alone achieves 8.6% improvement on math and 8.4% on general reasoning, outperforming SFT on 25,000 expert game trajectories. Analysis reveals that this transfer occurs through three cognitive patterns: systematic decomposition, expected value calculation, and case-by-case analysis. Multi-game training (TicTacToe, Kuhn Poker, Simple Negotiation) further enhances performance as each game develops distinct reasoning strengths. Applying SPIRAL to a strong reasoning model (DeepSeek-R1-Distill-Qwen-7B) can still lead to 2.0% average improvement. These results demonstrate that zero-sum games naturally develop transferable reasoning capabilities, highlighting a promising direction for autonomous reasoning development. |
| 2025-06-30 | [Modified non-local damage model: resolving spurious damage evolution](http://arxiv.org/abs/2506.24099v1) | Roshan Philip Saji, Panos Pantidis et al. | Accurate prediction of damage and fracture evolution is critical for the safety design and preventive maintenance of engineering structures, however existing computational methods face significant limitations. On one hand, discrete damage and phase-field models are often computationally prohibitive for real world applications and they are less generalizable across different material classes. On the other hand, conventional gradient damage models which are based on phenomenological laws, though more computationally efficient, they suffer from unrealistic widening of the damage-band as damage progresses. This paper presents a modified non-local gradient damage model (MNLD) that overcomes these shortcomings by introducing modifications to the stress degradation function and forcing term in the Helmholtz free energy expression. These two modifications ensure that as damage approaches its maximum value, both the thermodynamic damage driving force for damage vanishes and the evolution of the forcing term decays. Consequently, the damage band retains a non-growing constant width throughout its evolution. The proposed approach builds on insights gained from two intermediate models, which addressed the necessary conditions separately before integrating them into a unified formulation. Numerical validation is performed on several 1D and 2D benchmark problems, demonstrating that the proposed model can reliably produce fixed-width damage bands. The proposed approach can be implemented within existing gradient damage-based finite element frameworks with minimal implementation changes. The results highlight the potential of this approach to resolve the decades-long challenge of spurious widening in gradient damage models, offering an effective and practical solution for engineering applications. |
| 2025-06-30 | [Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models](http://arxiv.org/abs/2506.24056v1) | Tung-Ling Li, Hongliang Liu | We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations. |
| 2025-06-30 | [A Survey on Vision-Language-Action Models for Autonomous Driving](http://arxiv.org/abs/2506.24044v1) | Sicong Jiang, Zilin Huang et al. | The rapid progress of multimodal large language models (MLLM) has paved the way for Vision-Language-Action (VLA) paradigms, which integrate visual perception, natural language understanding, and control within a single policy. Researchers in autonomous driving are actively adapting these methods to the vehicle domain. Such models promise autonomous vehicles that can interpret high-level instructions, reason about complex traffic scenes, and make their own decisions. However, the literature remains fragmented and is rapidly expanding. This survey offers the first comprehensive overview of VLA for Autonomous Driving (VLA4AD). We (i) formalize the architectural building blocks shared across recent work, (ii) trace the evolution from early explainer to reasoning-centric VLA models, and (iii) compare over 20 representative models according to VLA's progress in the autonomous driving domain. We also consolidate existing datasets and benchmarks, highlighting protocols that jointly measure driving safety, accuracy, and explanation quality. Finally, we detail open challenges - robustness, real-time efficiency, and formal verification - and outline future directions of VLA4AD. This survey provides a concise yet complete reference for advancing interpretable socially aligned autonomous vehicles. Github repo is available at \href{https://github.com/JohnsonJiang1996/Awesome-VLA4AD}{SicongJiang/Awesome-VLA4AD}. |
| 2025-06-30 | [Large Language Models Don't Make Sense of Word Problems. A Scoping Review from a Mathematics Education Perspective](http://arxiv.org/abs/2506.24006v1) | Anselm R. Strohmaier, Wim Van Dooren et al. | The progress of Large Language Models (LLMs) like ChatGPT raises the question of how they can be integrated into education. One hope is that they can support mathematics learning, including word-problem solving. Since LLMs can handle textual input with ease, they appear well-suited for solving mathematical word problems. Yet their real competence, whether they can make sense of the real-world context, and the implications for classrooms remain unclear. We conducted a scoping review from a mathematics-education perspective, including three parts: a technical overview, a systematic review of word problems used in research, and a state-of-the-art empirical evaluation of LLMs on mathematical word problems. First, in the technical overview, we contrast the conceptualization of word problems and their solution processes between LLMs and students. In computer-science research this is typically labeled mathematical reasoning, a term that does not align with usage in mathematics education. Second, our literature review of 213 studies shows that the most popular word-problem corpora are dominated by s-problems, which do not require a consideration of realities of their real-world context. Finally, our evaluation of GPT-3.5-turbo, GPT-4o-mini, GPT-4.1, and o3 on 287 word problems shows that most recent LLMs solve these s-problems with near-perfect accuracy, including a perfect score on 20 problems from PISA. LLMs still showed weaknesses in tackling problems where the real-world context is problematic or non-sensical. In sum, we argue based on all three aspects that LLMs have mastered a superficial solution process but do not make sense of word problems, which potentially limits their value as instructional tools in mathematics classrooms. |
| 2025-06-30 | [STCLocker: Deadlock Avoidance Testing for Autonomous Driving Systems](http://arxiv.org/abs/2506.23995v1) | Mingfei Cheng, Renzhi Wang et al. | Autonomous Driving System (ADS) testing is essential to ensure the safety and reliability of autonomous vehicles (AVs) before deployment. However, existing techniques primarily focus on evaluating ADS functionalities in single-AV settings. As ADSs are increasingly deployed in multi-AV traffic, it becomes crucial to assess their cooperative performance, particularly regarding deadlocks, a fundamental coordination failure in which multiple AVs enter a circular waiting state indefinitely, resulting in motion planning failures. Despite its importance, the cooperative capability of ADSs to prevent deadlocks remains insufficiently underexplored. To address this gap, we propose the first dedicated Spatio-Temporal Conflict-Guided Deadlock Avoidance Testing technique, STCLocker, for generating DeadLock Scenarios (DLSs), where a group of AVs controlled by the ADS under test are in a circular wait state. STCLocker consists of three key components: Deadlock Oracle, Conflict Feedback, and Conflict-aware Scenario Generation. Deadlock Oracle provides a reliable black-box mechanism for detecting deadlock cycles among multiple AVs within a given scenario. Conflict Feedback and Conflict-aware Scenario Generation collaborate to actively guide AVs into simultaneous competition over spatial conflict resources (i.e., shared passing regions) and temporal competitive behaviors (i.e., reaching the conflict region at the same time), thereby increasing the effectiveness of generating conflict-prone deadlocks. We evaluate STCLocker on two types of ADSs: Roach, an end-to-end ADS, and OpenCDA, a module-based ADS supporting cooperative communication. Experimental results show that, on average, STCLocker generates more DLS than the best-performing baseline. |
| 2025-06-30 | [Harnessing AI Agents to Advance Research on Refugee Child Mental Health](http://arxiv.org/abs/2506.23992v1) | Aditya Shrivastava, Komal Gupta et al. | The international refugee crisis deepens, exposing millions of dis placed children to extreme psychological trauma. This research suggests a com pact, AI-based framework for processing unstructured refugee health data and distilling knowledge on child mental health. We compare two Retrieval-Aug mented Generation (RAG) pipelines, Zephyr-7B-beta and DeepSeek R1-7B, to determine how well they process challenging humanitarian datasets while avoid ing hallucination hazards. By combining cutting-edge AI methods with migration research and child psychology, this study presents a scalable strategy to assist policymakers, mental health practitioners, and humanitarian agencies to better assist displaced children and recognize their mental wellbeing. In total, both the models worked properly but significantly Deepseek R1 is superior to Zephyr with an accuracy of answer relevance 0.91 |
| 2025-06-30 | [A Scalable Approach for Safe and Robust Learning via Lipschitz-Constrained Networks](http://arxiv.org/abs/2506.23977v1) | Zain ul Abdeen, Vassilis Kekatos et al. | Certified robustness is a critical property for deploying neural networks (NN) in safety-critical applications. A principle approach to achieving such guarantees is to constrain the global Lipschitz constant of the network. However, accurate methods for Lipschitz-constrained training often suffer from non-convex formulations and poor scalability due to reliance on global semidefinite programs (SDPs). In this letter, we propose a convex training framework that enforces global Lipschitz constraints via semidefinite relaxation. By reparameterizing the NN using loop transformation, we derive a convex admissibility condition that enables tractable and certifiable training. While the resulting formulation guarantees robustness, its scalability is limited by the size of global SDP. To overcome this, we develop a randomized subspace linear matrix inequalities (RS-LMI) approach that decomposes the global constraints into sketched layerwise constraints projected onto low-dimensional subspaces, yielding a smooth and memory-efficient training objective. Empirical results on MNIST, CIFAR-10, and ImageNet demonstrate that the proposed framework achieves competitive accuracy with significantly improved Lipschitz bounds and runtime performance. |
| 2025-06-30 | [ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning](http://arxiv.org/abs/2506.23960v1) | Mingfei Cheng, Xiaofei Xie et al. | Autonomous Driving Systems (ADSs) continue to face safety-critical risks due to the inherent limitations in their design and performance capabilities. Online repair plays a crucial role in mitigating such limitations, ensuring the runtime safety and reliability of ADSs. Existing online repair solutions enforce ADS compliance by transforming unacceptable trajectories into acceptable ones based on predefined specifications, such as rule-based constraints or training datasets. However, these approaches often lack generalizability, adaptability and tend to be overly conservative, resulting in ineffective repairs that not only fail to mitigate safety risks sufficiently but also degrade the overall driving experience. To address this issue, we propose Adaptive Decision Repair (ADReFT), a novel and effective repair method that identifies safety-critical states through offline learning from failed tests and generates appropriate mitigation actions to improve ADS safety. Specifically, ADReFT incorporates a transformer-based model with two joint heads, State Monitor and Decision Adapter, designed to capture complex driving environment interactions to evaluate state safety severity and generate adaptive repair actions. Given the absence of oracles for state safety identification, we first pretrain ADReFT using supervised learning with coarse annotations, i.e., labeling states preceding violations as positive samples and others as negative samples. It establishes ADReFT's foundational capability to mitigate safety-critical violations, though it may result in somewhat conservative mitigation strategies. Therefore, we subsequently finetune ADReFT using reinforcement learning to improve its initial capability and generate more precise and contextually appropriate repair decisions. Our evaluation results illustrate that ADReFT achieves better repair performance. |
| 2025-06-30 | [Bridging the Gap with Retrieval-Augmented Generation: Making Prosthetic Device User Manuals Available in Marginalised Languages](http://arxiv.org/abs/2506.23958v1) | Ikechukwu Ogbonna, Lesley Davidson et al. | Millions of people in African countries face barriers to accessing healthcare due to language and literacy gaps. This research tackles this challenge by transforming complex medical documents -- in this case, prosthetic device user manuals -- into accessible formats for underserved populations. This case study in cross-cultural translation is particularly pertinent/relevant for communities that receive donated prosthetic devices but may not receive the accompanying user documentation. Or, if available online, may only be available in formats (e.g., language and readability) that are inaccessible to local populations (e.g., English-language, high resource settings/cultural context). The approach is demonstrated using the widely spoken Pidgin dialect, but our open-source framework has been designed to enable rapid and easy extension to other languages/dialects. This work presents an AI-powered framework designed to process and translate complex medical documents, e.g., user manuals for prosthetic devices, into marginalised languages. The system enables users -- such as healthcare workers or patients -- to upload English-language medical equipment manuals, pose questions in their native language, and receive accurate, localised answers in real time. Technically, the system integrates a Retrieval-Augmented Generation (RAG) pipeline for processing and semantic understanding of the uploaded manuals. It then employs advanced Natural Language Processing (NLP) models for generative question-answering and multilingual translation. Beyond simple translation, it ensures accessibility to device instructions, treatment protocols, and safety information, empowering patients and clinicians to make informed healthcare decisions. |
| 2025-06-30 | [Leveraging the Potential of Prompt Engineering for Hate Speech Detection in Low-Resource Languages](http://arxiv.org/abs/2506.23930v1) | Ruhina Tabasshum Prome, Tarikul Islam Tamiti et al. | The rapid expansion of social media leads to a marked increase in hate speech, which threatens personal lives and results in numerous hate crimes. Detecting hate speech presents several challenges: diverse dialects, frequent code-mixing, and the prevalence of misspelled words in user-generated content on social media platforms. Recent progress in hate speech detection is typically concentrated on high-resource languages. However, low-resource languages still face significant challenges due to the lack of large-scale, high-quality datasets. This paper investigates how we can overcome this limitation via prompt engineering on large language models (LLMs) focusing on low-resource Bengali language. We investigate six prompting strategies - zero-shot prompting, refusal suppression, flattering the classifier, multi-shot prompting, role prompting, and finally our innovative metaphor prompting to detect hate speech effectively in low-resource languages. We pioneer the metaphor prompting to circumvent the built-in safety mechanisms of LLMs that marks a significant departure from existing jailbreaking methods. We investigate all six different prompting strategies on the Llama2-7B model and compare the results extensively with three pre-trained word embeddings - GloVe, Word2Vec, and FastText for three different deep learning models - multilayer perceptron (MLP), convolutional neural network (CNN), and bidirectional gated recurrent unit (BiGRU). To prove the effectiveness of our metaphor prompting in the low-resource Bengali language, we also evaluate it in another low-resource language - Hindi, and two high-resource languages - English and German. The performance of all prompting techniques is evaluated using the F1 score, and environmental impact factor (IF), which measures CO$_2$ emissions, electricity usage, and computational time. |
| 2025-06-30 | [Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](http://arxiv.org/abs/2506.23881v1) | Reihaneh Zohrabi, Hosein Hasani et al. | Out-of-distribution (OOD) detection is crucial for ensuring the reliability and safety of machine learning models in real-world applications, where they frequently face data distributions unseen during training. Despite progress, existing methods are often vulnerable to spurious correlations that mislead models and compromise robustness. To address this, we propose SPROD, a novel prototype-based OOD detection approach that explicitly addresses the challenge posed by unknown spurious correlations. Our post-hoc method refines class prototypes to mitigate bias from spurious features without additional data or hyperparameter tuning, and is broadly applicable across diverse backbones and OOD detection settings. We conduct a comprehensive spurious correlation OOD detection benchmarking, comparing our method against existing approaches and demonstrating its superior performance across challenging OOD datasets, such as CelebA, Waterbirds, UrbanCars, Spurious Imagenet, and the newly introduced Animals MetaCoCo. On average, SPROD improves AUROC by 4.7% and FPR@95 by 9.3% over the second best. |
| 2025-06-30 | [Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts](http://arxiv.org/abs/2506.23845v1) | Kenny Peng, Rajiv Movva et al. | While sparse autoencoders (SAEs) have generated significant excitement, a series of negative results have added to skepticism about their usefulness. Here, we establish a conceptual distinction that reconciles competing narratives surrounding SAEs. We argue that while SAEs may be less effective for acting on known concepts, SAEs are powerful tools for discovering unknown concepts. This distinction cleanly separates existing negative and positive results, and suggests several classes of SAE applications. Specifically, we outline use cases for SAEs in (i) ML interpretability, explainability, fairness, auditing, and safety, and (ii) social and health sciences. |
| 2025-06-30 | [A Survey on Autonomy-Induced Security Risks in Large Model-Based Agents](http://arxiv.org/abs/2506.23844v1) | Hang Su, Jun Luo et al. | Recent advances in large language models (LLMs) have catalyzed the rise of autonomous AI agents capable of perceiving, reasoning, and acting in dynamic, open-ended environments. These large-model agents mark a paradigm shift from static inference systems to interactive, memory-augmented entities. While these capabilities significantly expand the functional scope of AI, they also introduce qualitatively novel security risks - such as memory poisoning, tool misuse, reward hacking, and emergent misalignment - that extend beyond the threat models of conventional systems or standalone LLMs. In this survey, we first examine the structural foundations and key capabilities that underpin increasing levels of agent autonomy, including long-term memory retention, modular tool use, recursive planning, and reflective reasoning. We then analyze the corresponding security vulnerabilities across the agent stack, identifying failure modes such as deferred decision hazards, irreversible tool chains, and deceptive behaviors arising from internal state drift or value misalignment. These risks are traced to architectural fragilities that emerge across perception, cognition, memory, and action modules. To address these challenges, we systematically review recent defense strategies deployed at different autonomy layers, including input sanitization, memory lifecycle control, constrained decision-making, structured tool invocation, and introspective reflection. We introduce the Reflective Risk-Aware Agent Architecture (R2A2), a unified cognitive framework grounded in Constrained Markov Decision Processes (CMDPs), which incorporates risk-aware world modeling, meta-policy adaptation, and joint reward-risk optimization to enable principled, proactive safety across the agent's decision-making loop. |
| 2025-06-30 | [Querying Attack-Fault-Defense Trees: Property Specification in Smart Grid and Aerospace Case Studies](http://arxiv.org/abs/2506.23789v1) | Reza Soltani, Stefano M. Nicoletti et al. | This paper introduces AFDL, a logic-based framework for reasoning about safety, security, and defense interactions in Attack-Fault-Defense Trees, which is a model that captures all safety, security, and defense domains in a single framework. We showcase both AFDL and propose a structured domain specific query language, LangAFDL, which enables domain experts to express complex analysis goals through intuitive templates. LangAFDL supports both Boolean and quantified queries as well as minimal cut set analysis, capturing the interplay between safety, security, and defensive measures. We illustrate the expressiveness and utility of the approach through representative queries over two different real-world case studies: Gridshield and Ground Segment as a Service. The formalization lays the automated safety-security groundwork for analyses in mission-critical systems and paves the way for future tool development and integration into design workflows. |
| 2025-06-30 | [Calibrating Graph Neural Networks with Wavelet-Aware Temperature Scaling](http://arxiv.org/abs/2506.23782v1) | Xiaoyang Li, Linwei Tao et al. | Graph Neural Networks (GNNs) have demonstrated strong predictive performance on relational data; however, their confidence estimates often misalign with actual predictive correctness, posing significant limitations for deployment in safety-critical settings. While existing graph-aware calibration methods seek to mitigate this limitation, they primarily depend on coarse one-hop statistics, such as neighbor-predicted confidence, or latent node embeddings, thereby neglecting the fine-grained structural heterogeneity inherent in graph topology. In this work, we propose Wavelet-Aware Temperature Scaling (WATS), a post-hoc calibration framework that assigns node-specific temperatures based on tunable heat-kernel graph wavelet features. Specifically, WATS harnesses the scalability and topology sensitivity of graph wavelets to refine confidence estimates, all without necessitating model retraining or access to neighboring logits or predictions. Extensive evaluations across seven benchmark datasets with varying graph structures and two GNN backbones demonstrate that WATS achieves the lowest Expected Calibration Error (ECE) among all compared methods, outperforming both classical and graph-specific baselines by up to 42.3\% in ECE and reducing calibration variance by 17.24\% on average compared with graph-specific methods. Moreover, WATS remains computationally efficient, scaling well across graphs of diverse sizes and densities. Code will be released based on publication. |
| 2025-06-30 | [Multi-Timescale Hierarchical Reinforcement Learning for Unified Behavior and Control of Autonomous Driving](http://arxiv.org/abs/2506.23771v1) | Guizhe Jin, Zhuoren Li et al. | Reinforcement Learning (RL) is increasingly used in autonomous driving (AD) and shows clear advantages. However, most RL-based AD methods overlook policy structure design. An RL policy that only outputs short-timescale vehicle control commands results in fluctuating driving behavior due to fluctuations in network outputs, while one that only outputs long-timescale driving goals cannot achieve unified optimality of driving behavior and control. Therefore, we propose a multi-timescale hierarchical reinforcement learning approach. Our approach adopts a hierarchical policy structure, where high- and low-level RL policies are unified-trained to produce long-timescale motion guidance and short-timescale control commands, respectively. Therein, motion guidance is explicitly represented by hybrid actions to capture multimodal driving behaviors on structured road and support incremental low-level extend-state updates. Additionally, a hierarchical safety mechanism is designed to ensure multi-timescale safety. Evaluation in simulator-based and HighD dataset-based highway multi-lane scenarios demonstrates that our approach significantly improves AD performance, effectively increasing driving efficiency, action consistency and safety. |
| 2025-06-30 | [Can We Challenge Open-Vocabulary Object Detectors with Generated Content in Street Scenes?](http://arxiv.org/abs/2506.23751v1) | Annika Mütze, Sadia Ilyas et al. | Open-vocabulary object detectors such as Grounding DINO are trained on vast and diverse data, achieving remarkable performance on challenging datasets. Due to that, it is unclear where to find their limitations, which is of major concern when using in safety-critical applications. Real-world data does not provide sufficient control, required for a rigorous evaluation of model generalization. In contrast, synthetically generated data allows to systematically explore the boundaries of model competence/generalization. In this work, we address two research questions: 1) Can we challenge open-vocabulary object detectors with generated image content? 2) Can we find systematic failure modes of those models? To address these questions, we design two automated pipelines using stable diffusion to inpaint unusual objects with high diversity in semantics, by sampling multiple substantives from WordNet and ChatGPT. On the synthetically generated data, we evaluate and compare multiple open-vocabulary object detectors as well as a classical object detector. The synthetic data is derived from two real-world datasets, namely LostAndFound, a challenging out-of-distribution (OOD) detection benchmark, and the NuImages dataset. Our results indicate that inpainting can challenge open-vocabulary object detectors in terms of overlooking objects. Additionally, we find a strong dependence of open-vocabulary models on object location, rather than on object semantics. This provides a systematic approach to challenge open-vocabulary models and gives valuable insights on how data could be acquired to effectively improve these models. |
| 2025-06-30 | [Attestable Audits: Verifiable AI Safety Benchmarks Using Trusted Execution Environments](http://arxiv.org/abs/2506.23706v1) | Christoph Schnabl, Daniel Hugenroth et al. | Benchmarks are important measures to evaluate safety and compliance of AI models at scale. However, they typically do not offer verifiable results and lack confidentiality for model IP and benchmark datasets. We propose Attestable Audits, which run inside Trusted Execution Environments and enable users to verify interaction with a compliant AI model. Our work protects sensitive data even when model provider and auditor do not trust each other. This addresses verification challenges raised in recent AI governance frameworks. We build a prototype demonstrating feasibility on typical audit benchmarks against Llama-3.1. |
| 2025-06-30 | [A New Perspective On AI Safety Through Control Theory Methodologies](http://arxiv.org/abs/2506.23703v1) | Lars Ullrich, Walter Zimmer et al. | While artificial intelligence (AI) is advancing rapidly and mastering increasingly complex problems with astonishing performance, the safety assurance of such systems is a major concern. Particularly in the context of safety-critical, real-world cyber-physical systems, AI promises to achieve a new level of autonomy but is hampered by a lack of safety assurance. While data-driven control takes up recent developments in AI to improve control systems, control theory in general could be leveraged to improve AI safety. Therefore, this article outlines a new perspective on AI safety based on an interdisciplinary interpretation of the underlying data-generation process and the respective abstraction by AI systems in a system theory-inspired and system analysis-driven manner. In this context, the new perspective, also referred to as data control, aims to stimulate AI engineering to take advantage of existing safety analysis and assurance in an interdisciplinary way to drive the paradigm of data control. Following a top-down approach, a generic foundation for safety analysis and assurance is outlined at an abstract level that can be refined for specific AI systems and applications and is prepared for future innovation. |

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



