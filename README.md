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
| 2025-05-20 | [Two Experts Are All You Need for Steering Thinking: Reinforcing Cognitive Effort in MoE Reasoning Models Without Additional Training](http://arxiv.org/abs/2505.14681v1) | Mengru Wang, Xingyu Chen et al. | Mixture-of-Experts (MoE) architectures within Large Reasoning Models (LRMs) have achieved impressive reasoning capabilities by selectively activating experts to facilitate structured cognitive processes. Despite notable advances, existing reasoning models often suffer from cognitive inefficiencies like overthinking and underthinking. To address these limitations, we introduce a novel inference-time steering methodology called Reinforcing Cognitive Experts (RICE), designed to improve reasoning performance without additional training or complex heuristics. Leveraging normalized Pointwise Mutual Information (nPMI), we systematically identify specialized experts, termed ''cognitive experts'' that orchestrate meta-level reasoning operations characterized by tokens like ''<think>''. Empirical evaluations with leading MoE-based LRMs (DeepSeek-R1 and Qwen3-235B) on rigorous quantitative and scientific reasoning benchmarks demonstrate noticeable and consistent improvements in reasoning accuracy, cognitive efficiency, and cross-domain generalization. Crucially, our lightweight approach substantially outperforms prevalent reasoning-steering techniques, such as prompt design and decoding constraints, while preserving the model's general instruction-following skills. These results highlight reinforcing cognitive experts as a promising, practical, and interpretable direction to enhance cognitive efficiency within advanced reasoning models. |
| 2025-05-20 | [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](http://arxiv.org/abs/2505.14677v1) | Jiaer Xia, Yuhang Zang et al. | Learning general-purpose reasoning capabilities has long been a challenging problem in AI. Recent research in large language models (LLMs), such as DeepSeek-R1, has shown that reinforcement learning techniques like GRPO can enable pre-trained LLMs to develop reasoning capabilities using simple question-answer pairs. In this paper, we aim to train visual language models (VLMs) to perform reasoning on image data through reinforcement learning and visual question-answer pairs, without any explicit chain-of-thought (CoT) supervision. Our findings indicate that simply applying reinforcement learning to a VLM -- by prompting the model to produce a reasoning chain before providing an answer -- can lead the model to develop shortcuts from easy questions, thereby reducing its ability to generalize across unseen data distributions. We argue that the key to mitigating shortcut learning is to encourage the model to interpret images prior to reasoning. Therefore, we train the model to adhere to a caption-reason-answer output format: initially generating a detailed caption for an image, followed by constructing an extensive reasoning chain. When trained on 273K CoT-free visual question-answer pairs and using only reinforcement learning, our model, named Visionary-R1, outperforms strong multimodal models, such as GPT-4o, Claude3.5-Sonnet, and Gemini-1.5-Pro, on multiple visual reasoning benchmarks. |
| 2025-05-20 | [SAFEPATH: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](http://arxiv.org/abs/2505.14667v1) | Wonje Jeung, Sangyeon Yoon et al. | Large Reasoning Models (LRMs) have become powerful tools for complex problem solving, but their structured reasoning pathways can lead to unsafe outputs when exposed to harmful prompts. Existing safety alignment methods reduce harmful outputs but can degrade reasoning depth, leading to significant trade-offs in complex, multi-step tasks, and remain vulnerable to sophisticated jailbreak attacks. To address this, we introduce SAFEPATH, a lightweight alignment method that fine-tunes LRMs to emit a short, 8-token Safety Primer at the start of their reasoning, in response to harmful prompts, while leaving the rest of the reasoning process unsupervised. Empirical results across multiple benchmarks indicate that SAFEPATH effectively reduces harmful outputs while maintaining reasoning performance. Specifically, SAFEPATH reduces harmful responses by up to 90.0% and blocks 83.3% of jailbreak attempts in the DeepSeek-R1-Distill-Llama-8B model, while requiring 295.9x less compute than Direct Refusal and 314.1x less than SafeChain. We further introduce a zero-shot variant that requires no fine-tuning. In addition, we provide a comprehensive analysis of how existing methods in LLMs generalize, or fail, when applied to reasoning-centric models, revealing critical gaps and new directions for safer AI. |
| 2025-05-20 | [Cost-Augmented Monte Carlo Tree Search for LLM-Assisted Planning](http://arxiv.org/abs/2505.14656v1) | Zihao Zhang, Fei Liu | While LLMs excel at open-ended reasoning, they often struggle with cost-sensitive planning, either treating all actions as having equal cost or failing to stay within strict budgets. In this paper, we introduce Cost-Augmented Monte Carlo Tree Search (CATS), a novel approach that brings explicit cost-awareness into LLM-guided planning. Tight cost constraints push the planner to quickly identify infeasible solutions, while looser constraints encourage optimization for minimal cost. We benchmark top LLMs such as GPT-4.1, Claude-3.7-Sonnet, and DeepSeek-R1, against our CATS planner to evaluate their performance in cost-sensitive scenarios. Our experiments suggest that raw LLMs such as GPT-4.1 often falter under tight budgets, whereas CATS consistently delivers strong performance, achieving higher task success rates and better cost efficiency. CATS provides an effective solution for budget-aware decision-making by combining the reasoning power of LLMs with structured search. |
| 2025-05-20 | [General-Reasoner: Advancing LLM Reasoning Across All Domains](http://arxiv.org/abs/2505.14652v1) | Xueguang Ma, Qian Liu et al. | Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks. |
| 2025-05-20 | [Will AI Tell Lies to Save Sick Children? Litmus-Testing AI Values Prioritization with AIRiskDilemmas](http://arxiv.org/abs/2505.14633v1) | Yu Ying Chiu, Zhilin Wang et al. | Detecting AI risks becomes more challenging as stronger models emerge and find novel methods such as Alignment Faking to circumvent these detection attempts. Inspired by how risky behaviors in humans (i.e., illegal activities that may hurt others) are sometimes guided by strongly-held values, we believe that identifying values within AI models can be an early warning system for AI's risky behaviors. We create LitmusValues, an evaluation pipeline to reveal AI models' priorities on a range of AI value classes. Then, we collect AIRiskDilemmas, a diverse collection of dilemmas that pit values against one another in scenarios relevant to AI safety risks such as Power Seeking. By measuring an AI model's value prioritization using its aggregate choices, we obtain a self-consistent set of predicted value priorities that uncover potential risks. We show that values in LitmusValues (including seemingly innocuous ones like Care) can predict for both seen risky behaviors in AIRiskDilemmas and unseen risky behaviors in HarmBench. |
| 2025-05-20 | [Impact of Surfactant and Flow Rate on the Electrical Properties of Activated Carbon Black Suspensions](http://arxiv.org/abs/2505.14618v1) | KangJin Lee, Jesse S. Wainright et al. | Carbon black slurries are a key component in redox flow batteries as the large surface area provided by the particles allows an increase in the battery capacity without facing limitations posed by many solid-state batteries such as safety hazard or cost. However, these conductive slurries often have complex mechanical and electrical responses because of the heterogeneous nature of the suspensions. Utilization of these slurries in a redox flow battery requires understanding of how additives impact the material responses and associated battery performance. This work focuses on the electrochemical performance of the slurry at flow rate conditions matching those of battery operation. In addition, the impact of a nonionic surfactant (Triton X-100) on the conductivity and capacitance of the slurry was measured. Experimental results show that the full capacitive contribution of the carbon black particles can only be measured at low flow rates and low scan rates, while the conductive contribution can be measured at all scan rates in flowing conditions. Upon the addition of surfactants, there is a gradual decrease in the electrochemical performance with increasing surfactant concentration until the surface of the carbon black particle is saturated. Once saturated, the conductive carbon particles no longer contribute to the electronic conductivity of the slurry. Results presented herein on the electrochemical response of the slurry to the addition of surfactant are in stark contrast to the mechanical response. While previous work has shown a smaller change in the response, followed by a step-change at a critical surfactant concentration, electrochemical data shows a gradual transition. Comparison of these behaviors suggests a difference in the mechanisms for how mechanical networks form in comparison to charge transfer networks for this particular slurry chemistry. |
| 2025-05-20 | [Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models](http://arxiv.org/abs/2505.14617v1) | Sahar Abdelnabi, Ahmed Salem | Reasoning-focused large language models (LLMs) sometimes alter their behavior when they detect that they are being evaluated, an effect analogous to the Hawthorne phenomenon, which can lead them to optimize for test-passing performance or to comply more readily with harmful prompts if real-world consequences appear absent. We present the first quantitative study of how such "test awareness" impacts model behavior, particularly its safety alignment. We introduce a white-box probing framework that (i) linearly identifies awareness-related activations and (ii) steers models toward or away from test awareness while monitoring downstream performance. We apply our method to different state-of-the-art open-source reasoning LLMs across both realistic and hypothetical tasks. Our results demonstrate that test awareness significantly impact safety alignment, and is different for different models. By providing fine-grained control over this latent effect, our work aims to increase trust in how we perform safety evaluation. |
| 2025-05-20 | [sudoLLM : On Multi-role Alignment of Language Models](http://arxiv.org/abs/2505.14607v1) | Soumadeep Saha, Akshay Chaturvedi et al. | User authorization-based access privileges are a key feature in many safety-critical systems, but have thus far been absent from the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, and resistance to prompt-based jailbreaking attacks. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs. |
| 2025-05-20 | [Let LLMs Break Free from Overthinking via Self-Braking Tuning](http://arxiv.org/abs/2505.14604v1) | Haoran Zhao, Yuchen Yan et al. | Large reasoning models (LRMs), such as OpenAI o1 and DeepSeek-R1, have significantly enhanced their reasoning capabilities by generating longer chains of thought, demonstrating outstanding performance across a variety of tasks. However, this performance gain comes at the cost of a substantial increase in redundant reasoning during the generation process, leading to high computational overhead and exacerbating the issue of overthinking. Although numerous existing approaches aim to address the problem of overthinking, they often rely on external interventions. In this paper, we propose a novel framework, Self-Braking Tuning (SBT), which tackles overthinking from the perspective of allowing the model to regulate its own reasoning process, thus eliminating the reliance on external control mechanisms. We construct a set of overthinking identification metrics based on standard answers and design a systematic method to detect redundant reasoning. This method accurately identifies unnecessary steps within the reasoning trajectory and generates training signals for learning self-regulation behaviors. Building on this foundation, we develop a complete strategy for constructing data with adaptive reasoning lengths and introduce an innovative braking prompt mechanism that enables the model to naturally learn when to terminate reasoning at an appropriate point. Experiments across mathematical benchmarks (AIME, AMC, MATH500, GSM8K) demonstrate that our method reduces token consumption by up to 60% while maintaining comparable accuracy to unconstrained models. |
| 2025-05-20 | [MCIP: Protecting MCP Safety via Model Contextual Integrity Protocol](http://arxiv.org/abs/2505.14590v1) | Huihao Jing, Haoran Li et al. | As Model Context Protocol (MCP) introduces an easy-to-use ecosystem for users and developers, it also brings underexplored safety risks. Its decentralized architecture, which separates clients and servers, poses unique challenges for systematic safety analysis. This paper proposes a novel framework to enhance MCP safety. Guided by the MAESTRO framework, we first analyze the missing safety mechanisms in MCP, and based on this analysis, we propose the Model Contextual Integrity Protocol (MCIP), a refined version of MCP that addresses these gaps.Next, we develop a fine-grained taxonomy that captures a diverse range of unsafe behaviors observed in MCP scenarios. Building on this taxonomy, we develop benchmark and training data that support the evaluation and improvement of LLMs' capabilities in identifying safety risks within MCP interactions. Leveraging the proposed benchmark and training data, we conduct extensive experiments on state-of-the-art LLMs. The results highlight LLMs' vulnerabilities in MCP interactions and demonstrate that our approach substantially improves their safety performance. |
| 2025-05-20 | [Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning](http://arxiv.org/abs/2505.14585v1) | Wenbin Hu, Haoran Li et al. | While Large Language Models (LLMs) exhibit remarkable capabilities, they also introduce significant safety and privacy risks. Current mitigation strategies often fail to preserve contextual reasoning capabilities in risky scenarios. Instead, they rely heavily on sensitive pattern matching to protect LLMs, which limits the scope. Furthermore, they overlook established safety and privacy standards, leading to systemic risks for legal compliance. To address these gaps, we formulate safety and privacy issues into contextualized compliance problems following the Contextual Integrity (CI) theory. Under the CI framework, we align our model with three critical regulatory standards: GDPR, EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with a rule-based reward to incentivize contextual reasoning capabilities while enhancing compliance with safety and privacy norms. Through extensive experiments, we demonstrate that our method not only significantly enhances legal compliance (achieving a +17.64% accuracy improvement in safety/privacy benchmarks) but also further improves general reasoning capability. For OpenThinker-7B, a strong reasoning model that significantly outperforms its base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on the MMLU and LegalBench benchmark, respectively. |
| 2025-05-20 | [Traversability-aware path planning in dynamic environments](http://arxiv.org/abs/2505.14580v1) | Yaroslav Marchukov, Luis Montano | Planning in environments with moving obstacles remains a significant challenge in robotics. While many works focus on navigation and path planning in obstacle-dense spaces, traversing such congested regions is often avoidable by selecting alternative routes. This paper presents Traversability-aware FMM (Tr-FMM), a path planning method that computes paths in dynamic environments, avoiding crowded regions. The method operates in two steps: first, it discretizes the environment, identifying regions and their distribution; second, it computes the traversability of regions, aiming to minimize both obstacle risks and goal deviation. The path is then computed by propagating the wavefront through regions with higher traversability. Simulated and real-world experiments demonstrate that the approach enhances significant safety by keeping the robot away from regions with obstacles while reducing unnecessary deviations from the goal. |
| 2025-05-20 | [Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders](http://arxiv.org/abs/2505.14536v1) | Agam Goyal, Vedant Rathi et al. | Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment. |
| 2025-05-20 | [Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach](http://arxiv.org/abs/2505.14479v1) | Oren Sultan, Eitan Stern et al. | Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness. |
| 2025-05-20 | [Attributional Safety Failures in Large Language Models under Code-Mixed Perturbations](http://arxiv.org/abs/2505.14469v1) | Somnath Banerjee, Pratyush Chatterjee et al. | Recent advancements in LLMs have raised significant safety concerns, particularly when dealing with code-mixed inputs and outputs. Our study systematically investigates the increased susceptibility of LLMs to produce unsafe outputs from code-mixed prompts compared to monolingual English prompts. Utilizing explainability methods, we dissect the internal attribution shifts causing model's harmful behaviors. In addition, we explore cultural dimensions by distinguishing between universally unsafe and culturally-specific unsafe queries. This paper presents novel experimental insights, clarifying the mechanisms driving this phenomenon. |
| 2025-05-20 | [Not All Correct Answers Are Equal: Why Your Distillation Source Matters](http://arxiv.org/abs/2505.14464v1) | Xiaoyu Tian, Yunjie Ji et al. | Distillation has emerged as a practical and effective approach to enhance the reasoning capabilities of open-source language models. In this work, we conduct a large-scale empirical study on reasoning data distillation by collecting verified outputs from three state-of-the-art teacher models-AM-Thinking-v1, Qwen3-235B-A22B, and DeepSeek-R1-on a shared corpus of 1.89 million queries. We construct three parallel datasets and analyze their distributions, revealing that AM-Thinking-v1-distilled data exhibits greater token length diversity and lower perplexity. Student models trained on each dataset are evaluated on reasoning benchmarks including AIME2024, AIME2025, MATH500, and LiveCodeBench. The AM-based model consistently achieves the best performance (e.g., 84.3 on AIME2024, 72.2 on AIME2025, 98.4 on MATH500, and 65.9 on LiveCodeBench) and demonstrates adaptive output behavior-producing longer responses for harder tasks and shorter ones for simpler tasks. These findings highlight the value of high-quality, verified reasoning traces. We release the AM-Thinking-v1 and Qwen3-235B-A22B distilled datasets to support future research on open and high-performing reasoning-oriented language models. The datasets are publicly available on Hugging Face\footnote{Datasets are available on Hugging Face: \href{https://huggingface.co/datasets/a-m-team/AM-Thinking-v1-Distilled}{AM-Thinking-v1-Distilled}, \href{https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled}{AM-Qwen3-Distilled}.}. |
| 2025-05-20 | [VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank](http://arxiv.org/abs/2505.14460v1) | Tianhe Wu, Jian Zou et al. | DeepSeek-R1 has demonstrated remarkable effectiveness in incentivizing reasoning and generalization capabilities of large language models (LLMs) through reinforcement learning. Nevertheless, the potential of reasoning-induced computational modeling has not been thoroughly explored in the context of image quality assessment (IQA), a task critically dependent on visual reasoning. In this paper, we introduce VisualQuality-R1, a reasoning-induced no-reference IQA (NR-IQA) model, and we train it with reinforcement learning to rank, a learning algorithm tailored to the intrinsically relative nature of visual quality. Specifically, for a pair of images, we employ group relative policy optimization to generate multiple quality scores for each image. These estimates are then used to compute comparative probabilities of one image having higher quality than the other under the Thurstone model. Rewards for each quality estimate are defined using continuous fidelity measures rather than discretized binary labels. Extensive experiments show that the proposed VisualQuality-R1 consistently outperforms discriminative deep learning-based NR-IQA models as well as a recent reasoning-induced quality regression method. Moreover, VisualQuality-R1 is capable of generating contextually rich, human-aligned quality descriptions, and supports multi-dataset training without requiring perceptual scale realignment. These features make VisualQuality-R1 especially well-suited for reliably measuring progress in a wide range of image processing tasks like super-resolution and image generation. |
| 2025-05-20 | [Interpretable Neural System Dynamics: Combining Deep Learning with System Dynamics Modeling to Support Critical Applications](http://arxiv.org/abs/2505.14428v1) | Riccardo D'Elia | The objective of this proposal is to bridge the gap between Deep Learning (DL) and System Dynamics (SD) by developing an interpretable neural system dynamics framework. While DL excels at learning complex models and making accurate predictions, it lacks interpretability and causal reliability. Traditional SD approaches, on the other hand, provide transparency and causal insights but are limited in scalability and require extensive domain knowledge. To overcome these limitations, this project introduces a Neural System Dynamics pipeline, integrating Concept-Based Interpretability, Mechanistic Interpretability, and Causal Machine Learning. This framework combines the predictive power of DL with the interpretability of traditional SD models, resulting in both causal reliability and scalability. The efficacy of the proposed pipeline will be validated through real-world applications of the EU-funded AutoMoTIF project, which is focused on autonomous multimodal transportation systems. The long-term goal is to collect actionable insights that support the integration of explainability and safety in autonomous systems. |
| 2025-05-20 | [Explaining Unreliable Perception in Automated Driving: A Fuzzy-based Monitoring Approach](http://arxiv.org/abs/2505.14407v1) | Aniket Salvi, Gereon Weiss et al. | Autonomous systems that rely on Machine Learning (ML) utilize online fault tolerance mechanisms, such as runtime monitors, to detect ML prediction errors and maintain safety during operation. However, the lack of human-interpretable explanations for these errors can hinder the creation of strong assurances about the system's safety and reliability. This paper introduces a novel fuzzy-based monitor tailored for ML perception components. It provides human-interpretable explanations about how different operating conditions affect the reliability of perception components and also functions as a runtime safety monitor. We evaluated our proposed monitor using naturalistic driving datasets as part of an automated driving case study. The interpretability of the monitor was evaluated and we identified a set of operating conditions in which the perception component performs reliably. Additionally, we created an assurance case that links unit-level evidence of \textit{correct} ML operation to system-level \textit{safety}. The benchmarking demonstrated that our monitor achieved a better increase in safety (i.e., absence of hazardous situations) while maintaining availability (i.e., ability to perform the mission) compared to state-of-the-art runtime ML monitors in the evaluated dataset. |

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



