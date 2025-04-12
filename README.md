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
It is updated every 12 hours, presenting the latest 20 relevant papers.And [Earlier Papers](./articles/README.md) can be found here.


<!-- LATEST_PAPERS_START -->
<details>
<summary>📚 Click to Show Details</summary>

| Date       | Title                                      | Authors           | Abstract Summary          |
|------------|--------------------------------------------|-------------------|---------------------------|
| 2025-04-10 | [VCR-Bench: A Comprehensive Evaluation Framework for Video Chain-of-Thought Reasoning](http://arxiv.org/abs/2504.07956v1) | Yukun Qi, Yiming Zhao et al. | The advancement of Chain-of-Thought (CoT) reasoning has significantly enhanced the capabilities of large language models (LLMs) and large vision-language models (LVLMs). However, a rigorous evaluation framework for video CoT reasoning remains absent. Current video benchmarks fail to adequately assess the reasoning process and expose whether failures stem from deficiencies in perception or reasoning capabilities. Therefore, we introduce VCR-Bench, a novel benchmark designed to comprehensively evaluate LVLMs' Video Chain-of-Thought Reasoning capabilities. VCR-Bench comprises 859 videos spanning a variety of video content and durations, along with 1,034 high-quality question-answer pairs. Each pair is manually annotated with a stepwise CoT rationale, where every step is tagged to indicate its association with the perception or reasoning capabilities. Furthermore, we design seven distinct task dimensions and propose the CoT score to assess the entire CoT process based on the stepwise tagged CoT rationals. Extensive experiments on VCR-Bench highlight substantial limitations in current LVLMs. Even the top-performing model, o1, only achieves a 62.8% CoT score and an 56.7% accuracy, while most models score below 40%. Experiments show most models score lower on perception than reasoning steps, revealing LVLMs' key bottleneck in temporal-spatial information processing for complex video reasoning. A robust positive correlation between the CoT score and accuracy confirms the validity of our evaluation framework and underscores the critical role of CoT reasoning in solving complex video reasoning tasks. We hope VCR-Bench to serve as a standardized evaluation framework and expose the actual drawbacks in complex video reasoning task. |
| 2025-04-10 | [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](http://arxiv.org/abs/2504.07954v1) | En Yu, Kangheng Lin et al. | Inspired by the success of DeepSeek-R1, we explore the potential of rule-based reinforcement learning (RL) in MLLM post-training for perception policy learning. While promising, our initial experiments reveal that incorporating a thinking process through RL does not consistently lead to performance gains across all visual perception tasks. This leads us to delve into the essential role of RL in the context of visual perception. In this work, we return to the fundamentals and explore the effects of RL on different perception tasks. We observe that the perceptual complexity is a major factor in determining the effectiveness of RL. We also observe that reward design plays a crucial role in further approching the upper limit of model perception. To leverage these findings, we propose Perception-R1, a scalable RL framework using GRPO during MLLM post-training. With a standard Qwen2.5-VL-3B-Instruct, Perception-R1 achieves +4.2% on RefCOCO+, +17.9% on PixMo-Count, +4.2% on PageOCR, and notably, 31.9% AP on COCO2017 val for the first time, establishing a strong baseline for perception policy learning. |
| 2025-04-10 | [SoTA with Less: MCTS-Guided Sample Selection for Data-Efficient Visual Reasoning Self-Improvement](http://arxiv.org/abs/2504.07934v1) | Xiyao Wang, Zhengyuan Yang et al. | In this paper, we present an effective method to enhance visual reasoning with significantly fewer training samples, relying purely on self-improvement with no knowledge distillation. Our key insight is that the difficulty of training data during reinforcement fine-tuning (RFT) is critical. Appropriately challenging samples can substantially boost reasoning capabilities even when the dataset is small. Despite being intuitive, the main challenge remains in accurately quantifying sample difficulty to enable effective data filtering. To this end, we propose a novel way of repurposing Monte Carlo Tree Search (MCTS) to achieve that. Starting from our curated 70k open-source training samples, we introduce an MCTS-based selection method that quantifies sample difficulty based on the number of iterations required by the VLMs to solve each problem. This explicit step-by-step reasoning in MCTS enforces the model to think longer and better identifies samples that are genuinely challenging. We filter and retain 11k samples to perform RFT on Qwen2.5-VL-7B-Instruct, resulting in our final model, ThinkLite-VL. Evaluation results on eight benchmarks show that ThinkLite-VL improves the average performance of Qwen2.5-VL-7B-Instruct by 7%, using only 11k training samples with no knowledge distillation. This significantly outperforms all existing 7B-level reasoning VLMs, and our fairly comparable baselines that use classic selection methods such as accuracy-based filtering. Notably, on MathVista, ThinkLite-VL-7B achieves the SoTA accuracy of 75.1, surpassing Qwen2.5-VL-72B, GPT-4o, and O1. Our code, data, and model are available at https://github.com/si0wang/ThinkLite-VL. |
| 2025-04-10 | [Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge](http://arxiv.org/abs/2504.07887v1) | Riccardo Cantini, Alessio Orsino et al. | Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models. |
| 2025-04-10 | [Gauge and parametrization dependence of Quantum Einstein Gravity within the Proper Time flow](http://arxiv.org/abs/2504.07877v1) | Alfio Bonanno, Giovanni Oglialoro et al. | Proper time functional flow equations have garnered significant attention in recent years, as they are particularly suitable in analyzing non-perturbative contexts. By resorting to this flow, we investigate the regulator and gauge dependence in quantum Einstein gravity within the asymptotic safety framework, considering various regularization schemes. Our findings indicate that some details of the regulator have minor influence on the critical properties of the theory. In contrast, the selection between linear and exponential parametrizations appears to have a more substantial impact on the scaling behavior of the renormalized flow near the non-Gaussian fixed point. |
| 2025-04-10 | [Pangu Ultra: Pushing the Limits of Dense Large Language Models on Ascend NPUs](http://arxiv.org/abs/2504.07866v1) | Yichun Yin, Wenyong Huang et al. | We present Pangu Ultra, a Large Language Model (LLM) with 135 billion parameters and dense Transformer modules trained on Ascend Neural Processing Units (NPUs). Although the field of LLM has been witnessing unprecedented advances in pushing the scale and capability of LLM in recent years, training such a large-scale model still involves significant optimization and system challenges. To stabilize the training process, we propose depth-scaled sandwich normalization, which effectively eliminates loss spikes during the training process of deep models. We pre-train our model on 13.2 trillion diverse and high-quality tokens and further enhance its reasoning capabilities during post-training. To perform such large-scale training efficiently, we utilize 8,192 Ascend NPUs with a series of system optimizations. Evaluations on multiple diverse benchmarks indicate that Pangu Ultra significantly advances the state-of-the-art capabilities of dense LLMs such as Llama 405B and Mistral Large 2, and even achieves competitive results with DeepSeek-R1, whose sparse model structure contains much more parameters. Our exploration demonstrates that Ascend NPUs are capable of efficiently and effectively training dense models with more than 100 billion parameters. Our model and system will be available for our commercial customers. |
| 2025-04-10 | [Robust Hallucination Detection in LLMs via Adaptive Token Selection](http://arxiv.org/abs/2504.07863v1) | Mengjia Niu, Hamed Haddadi et al. | Hallucinations in large language models (LLMs) pose significant safety concerns that impede their broader deployment. Recent research in hallucination detection has demonstrated that LLMs' internal representations contain truthfulness hints, which can be harnessed for detector training. However, the performance of these detectors is heavily dependent on the internal representations of predetermined tokens, fluctuating considerably when working on free-form generations with varying lengths and sparse distributions of hallucinated entities. To address this, we propose HaMI, a novel approach that enables robust detection of hallucinations through adaptive selection and learning of critical tokens that are most indicative of hallucinations. We achieve this robustness by an innovative formulation of the Hallucination detection task as Multiple Instance (HaMI) learning over token-level representations within a sequence, thereby facilitating a joint optimisation of token selection and hallucination detection on generation sequences of diverse forms. Comprehensive experimental results on four hallucination benchmarks show that HaMI significantly outperforms existing state-of-the-art approaches. |
| 2025-04-10 | [Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems](http://arxiv.org/abs/2504.07831v1) | Simon Lermen, Mateusz Dziemian et al. | We demonstrate how AI agents can coordinate to deceive oversight systems using automated interpretability of neural networks. Using sparse autoencoders (SAEs) as our experimental framework, we show that language models (Llama, DeepSeek R1, and Claude 3.7 Sonnet) can generate deceptive explanations that evade detection. Our agents employ steganographic methods to hide information in seemingly innocent explanations, successfully fooling oversight models while achieving explanation quality comparable to reference labels. We further find that models can scheme to develop deceptive strategies when they believe the detection of harmful features might lead to negative consequences for themselves. All tested LLM agents were capable of deceiving the overseer while achieving high interpretability scores comparable to those of reference labels. We conclude by proposing mitigation strategies, emphasizing the critical need for robust understanding and defenses against deception. |
| 2025-04-10 | [Revisiting Likelihood-Based Out-of-Distribution Detection by Modeling Representations](http://arxiv.org/abs/2504.07793v1) | Yifan Ding, Arturas Aleksandrauskas et al. | Out-of-distribution (OOD) detection is critical for ensuring the reliability of deep learning systems, particularly in safety-critical applications. Likelihood-based deep generative models have historically faced criticism for their unsatisfactory performance in OOD detection, often assigning higher likelihood to OOD data than in-distribution samples when applied to image data. In this work, we demonstrate that likelihood is not inherently flawed. Rather, several properties in the images space prohibit likelihood as a valid detection score. Given a sufficiently good likelihood estimator, specifically using the probability flow formulation of a diffusion model, we show that likelihood-based methods can still perform on par with state-of-the-art methods when applied in the representation space of pre-trained encoders. The code of our work can be found at $\href{https://github.com/limchaos/Likelihood-OOD.git}{\texttt{https://github.com/limchaos/Likelihood-OOD.git}}$. |
| 2025-04-10 | [Realigning Incentives to Build Better Software: a Holistic Approach to Vendor Accountability](http://arxiv.org/abs/2504.07766v1) | Gergely Biczók, Sasha Romanosky et al. | In this paper, we ask the question of why the quality of commercial software, in terms of security and safety, does not measure up to that of other (durable) consumer goods we have come to expect. We examine this question through the lens of incentives. We argue that the challenge around better quality software is due in no small part to a sequence of misaligned incentives, the most critical of which being that the harm caused by software problems is by and large shouldered by consumers, not developers. This lack of liability means software vendors have every incentive to rush low-quality software onto the market and no incentive to enhance quality control. Within this context, this paper outlines a holistic technical and policy framework we believe is needed to incentivize better and more secure software development. At the heart of the incentive realignment is the concept of software liability. This framework touches on various components, including legal, technical, and financial, that are needed for software liability to work in practice; some currently exist, some will need to be re-imagined or established. This is primarily a market-driven approach that emphasizes voluntary participation but highlights the role appropriate regulation can play. We connect and contrast this with the EU legal environment and discuss what this framework means for open-source software (OSS) development and emerging AI risks. Moreover, we present a CrowdStrike case study complete with a what-if analysis had our proposed framework been in effect. Our intention is very much to stimulate a robust conversation among both researchers and practitioners. |
| 2025-04-10 | [Optimal Frequency Support from Virtual Power Plants: Minimal Reserve and Allocation](http://arxiv.org/abs/2504.07703v1) | Xiang Zhu, Guangchun Ruan et al. | This paper proposes a novel reserve-minimizing and allocation strategy for virtual power plants (VPPs) to deliver optimal frequency support. The proposed strategy enables VPPs, acting as aggregators for inverter-based resources (IBRs), to provide optimal frequency support economically. The proposed strategy captures time-varying active power injections, reducing the unnecessary redundancy compared to traditional fixed reserve schemes. Reserve requirements for the VPPs are determined based on system frequency response and safety constraints, ensuring efficient grid support. Furthermore, an energy-based allocation model decomposes power injections for each IBR, accounting for their specific limitations. Numerical experiments validate the feasibility of the proposed approach, highlighting significant financial gains for VPPs, especially as system inertia decreases due to higher renewable energy integration. |
| 2025-04-10 | [Joint Travel Route Optimization Framework for Platooning](http://arxiv.org/abs/2504.07623v1) | Akif Adas, Stefano Arrigoni et al. | Platooning represents an advanced driving technology designed to assist drivers in traffic convoys of varying lengths, enhancing road safety, reducing driver fatigue, and improving fuel efficiency. Sophisticated automated driving assistance systems have facilitated this innovation. Recent advancements in platooning emphasize cooperative mechanisms within both centralized and decentralized architectures enabled by vehicular communication technologies. This study introduces a cooperative route planning optimization framework aimed at promoting the adoption of platooning through a centralized platoon formation strategy at the system level. This approach is envisioned as a transitional phase from individual (ego) driving to fully collaborative driving. Additionally, this research formulates and incorporates travel cost metrics related to fuel consumption, driver fatigue, and travel time, considering regulatory constraints on consecutive driving durations. The performance of these cost metrics has been evaluated using Dijkstra's and A* shortest path algorithms within a network graph framework. The results indicate that the proposed architecture achieves an average cost improvement of 14 % compared to individual route planning for long road trips. |
| 2025-04-10 | [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](http://arxiv.org/abs/2504.07615v1) | Haozhan Shen, Peng Liu et al. | Recently DeepSeek R1 has shown that reinforcement learning (RL) can substantially improve the reasoning capabilities of Large Language Models (LLMs) through a simple yet effective design. The core of R1 lies in its rule-based reward formulation, which leverages tasks with deterministic ground-truth answers to enable precise and stable reward computation. In the visual domain, we similarly observe that a wide range of visual understanding tasks are inherently equipped with well-defined ground-truth annotations. This property makes them naturally compatible with rule-based reward mechanisms. Motivated by this observation, we investigate the extension of R1-style reinforcement learning to Vision-Language Models (VLMs), aiming to enhance their visual reasoning capabilities. To this end, we develop VLM-R1, a dedicated framework designed to harness RL for improving VLMs' performance on general vision-language tasks. Using this framework, we further explore the feasibility of applying RL to visual domain. Experimental results indicate that the RL-based model not only delivers competitive performance on visual understanding tasks but also surpasses Supervised Fine-Tuning (SFT) in generalization ability. Furthermore, we conduct comprehensive ablation studies that uncover a series of noteworthy insights, including the presence of reward hacking in object detection, the emergence of the "OD aha moment", the impact of training data quality, and the scaling behavior of RL across different model sizes. Through these analyses, we aim to deepen the understanding of how reinforcement learning enhances the capabilities of vision-language models, and we hope our findings and open-source contributions will support continued progress in the vision-language RL community. Our code and model are available at https://github.com/om-ai-lab/VLM-R1 |
| 2025-04-10 | [Drive in Corridors: Enhancing the Safety of End-to-end Autonomous Driving via Corridor Learning and Planning](http://arxiv.org/abs/2504.07507v1) | Zhiwei Zhang, Ruichen Yang et al. | Safety remains one of the most critical challenges in autonomous driving systems. In recent years, the end-to-end driving has shown great promise in advancing vehicle autonomy in a scalable manner. However, existing approaches often face safety risks due to the lack of explicit behavior constraints. To address this issue, we uncover a new paradigm by introducing the corridor as the intermediate representation. Widely adopted in robotics planning, the corridors represents spatio-temporal obstacle-free zones for the vehicle to traverse. To ensure accurate corridor prediction in diverse traffic scenarios, we develop a comprehensive learning pipeline including data annotation, architecture refinement and loss formulation. The predicted corridor is further integrated as the constraint in a trajectory optimization process. By extending the differentiability of the optimization, we enable the optimized trajectory to be seamlessly trained within the end-to-end learning framework, improving both safety and interpretability. Experimental results on the nuScenes dataset demonstrate state-of-the-art performance of our approach, showing a 66.7% reduction in collisions with agents and a 46.5% reduction with curbs, significantly enhancing the safety of end-to-end driving. Additionally, incorporating the corridor contributes to higher success rates in closed-loop evaluations. |
| 2025-04-10 | [Defense against Prompt Injection Attacks via Mixture of Encodings](http://arxiv.org/abs/2504.07467v1) | Ruiyi Zhang, David Sullivan et al. | Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics. |
| 2025-04-10 | [Multi-Modal Data Fusion for Moisture Content Prediction in Apple Drying](http://arxiv.org/abs/2504.07465v1) | Shichen Li, Chenhui Shao | Fruit drying is widely used in food manufacturing to reduce product moisture, ensure product safety, and extend product shelf life. Accurately predicting final moisture content (MC) is critically needed for quality control of drying processes. State-of-the-art methods can build deterministic relationships between process parameters and MC, but cannot adequately account for inherent process variabilities that are ubiquitous in fruit drying. To address this gap, this paper presents a novel multi-modal data fusion framework to effectively fuse two modalities of data: tabular data (process parameters) and high-dimensional image data (images of dried apple slices) to enable accurate MC prediction. The proposed modeling architecture permits flexible adjustment of information portion from tabular and image data modalities. Experimental validation shows that the multi-modal approach improves predictive accuracy substantially compared to state-of-the-art methods. The proposed method reduces root-mean-squared errors by 19.3%, 24.2%, and 15.2% over tabular-only, image-only, and standard tabular-image fusion models, respectively. Furthermore, it is demonstrated that our method is robust in varied tabular-image ratios and capable of effectively capturing inherent small-scale process variabilities. The proposed framework is extensible to a variety of other drying technologies. |
| 2025-04-10 | [LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation](http://arxiv.org/abs/2504.07448v1) | Juzheng Zhang, Jiacheng You et al. | Low-Rank Adaptation (LoRA) has emerged as a popular parameter-efficient fine-tuning (PEFT) method for Large Language Models (LLMs), yet it still incurs notable overhead and suffers from parameter interference in multi-task scenarios. We propose LoRA with Reduced Interference (LoRI), a simple yet effective approach that freezes the projection matrices $A$ as random projections and sparsifies the matrices $B$ using task-specific masks. This design substantially reduces the number of trainable parameters while maintaining strong task performance. Moreover, LoRI minimizes cross-task interference in adapter merging by leveraging the orthogonality between adapter subspaces, and supports continual learning by using sparsity to mitigate catastrophic forgetting. Extensive experiments across natural language understanding, mathematical reasoning, code generation, and safety alignment tasks demonstrate that LoRI outperforms full fine-tuning and existing PEFT methods, while using up to 95% fewer trainable parameters than LoRA. In multi-task experiments, LoRI enables effective adapter merging and continual learning with reduced cross-task interference. Code is available at: https://github.com/juzhengz/LoRI |
| 2025-04-10 | [Electronic Warfare Cyberattacks, Countermeasures and Modern Defensive Strategies of UAV Avionics: A Survey](http://arxiv.org/abs/2504.07358v1) | Aaron Yu, Iuliia Kolotylo et al. | Unmanned Aerial Vehicles (UAVs) play a pivotal role in modern autonomous air mobility, and the reliability of UAV avionics systems is critical to ensuring mission success, sustainability practices, and public safety. The success of UAV missions depends on effectively mitigating various aspects of electronic warfare, including non-destructive and destructive cyberattacks, transponder vulnerabilities, and jamming threats, while rigorously implementing countermeasures and defensive aids. This paper provides a comprehensive review of UAV cyberattacks, countermeasures, and defensive strategies. It explores UAV-to-UAV coordination attacks and their associated features, such as dispatch system attacks, Automatic Dependent Surveillance-Broadcast (ADS-B) attacks, Traffic Alert and Collision Avoidance System (TCAS)-induced collisions, and TCAS attacks. Additionally, the paper examines UAV-to-command center coordination attacks, as well as UAV functionality attacks. The review also covers various countermeasures and defensive aids designed for UAVs. Lastly, a comparison of common cyberattacks and countermeasure approaches is conducted, along with a discussion of future trends in the field. Keywords: Electronic warfare, UAVs, Avionics Systems, cyberattacks, coordination attacks, functionality attacks, countermeasure, defensive-aids. |
| 2025-04-10 | [Revisiting Prompt Optimization with Large Reasoning Models-A Case Study on Event Extraction](http://arxiv.org/abs/2504.07357v1) | Saurabh Srivastava, Ziyu Yao | Large Reasoning Models (LRMs) such as DeepSeek-R1 and OpenAI o1 have demonstrated remarkable capabilities in various reasoning tasks. Their strong capability to generate and reason over intermediate thoughts has also led to arguments that they may no longer require extensive prompt engineering or optimization to interpret human instructions and produce accurate outputs. In this work, we aim to systematically study this open question, using the structured task of event extraction for a case study. We experimented with two LRMs (DeepSeek-R1 and o1) and two general-purpose Large Language Models (LLMs) (GPT-4o and GPT-4.5), when they were used as task models or prompt optimizers. Our results show that on tasks as complicated as event extraction, LRMs as task models still benefit from prompt optimization, and that using LRMs as prompt optimizers yields more effective prompts. Finally, we provide an error analysis of common errors made by LRMs and highlight the stability and consistency of LRMs in refining task instructions and event guidelines. |
| 2025-04-09 | [Code Generation with Small Language Models: A Deep Evaluation on Codeforces](http://arxiv.org/abs/2504.07343v1) | Débora Souza, Rohit Gheyi et al. | Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws. |

</details>
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



