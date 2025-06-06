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
| 2025-06-04 | [Struct2D: A Perception-Guided Framework for Spatial Reasoning in Large Multimodal Models](http://arxiv.org/abs/2506.04220v1) | Fangrui Zhu, Hanhui Wang et al. | Unlocking spatial reasoning in Large Multimodal Models (LMMs) is crucial for enabling intelligent interaction with 3D environments. While prior efforts often rely on explicit 3D inputs or specialized model architectures, we ask: can LMMs reason about 3D space using only structured 2D representations derived from perception? We introduce Struct2D, a perception-guided prompting framework that combines bird's-eye-view (BEV) images with object marks and object-centric metadata, optionally incorporating egocentric keyframes when needed. Using Struct2D, we conduct an in-depth zero-shot analysis of closed-source LMMs (e.g., GPT-o3) and find that they exhibit surprisingly strong spatial reasoning abilities when provided with structured 2D inputs, effectively handling tasks such as relative direction estimation and route planning. Building on these insights, we construct Struct2D-Set, a large-scale instruction tuning dataset with 200K fine-grained QA pairs across eight spatial reasoning categories, generated automatically from 3D indoor scenes. We fine-tune an open-source LMM (Qwen2.5VL) on Struct2D-Set, achieving competitive performance on multiple benchmarks, including 3D question answering, dense captioning, and object grounding. Our approach demonstrates that structured 2D inputs can effectively bridge perception and language reasoning in LMMs-without requiring explicit 3D representations as input. We will release both our code and dataset to support future research. |
| 2025-06-04 | [Pseudo-Simulation for Autonomous Driving](http://arxiv.org/abs/2506.04218v1) | Wei Cao, Marcel Hallgarten et al. | Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations (R^2=0.8) than the best existing open-loop approach (R^2=0.7). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at https://github.com/autonomousvision/navsim. |
| 2025-06-04 | [Does Thinking More always Help? Understanding Test-Time Scaling in Reasoning Models](http://arxiv.org/abs/2506.04210v1) | Soumya Suvra Ghosal, Souradip Chakraborty et al. | Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek R1) have led to a popular belief that extending thinking traces using prompts like "Wait" or "Let me rethink" can improve performance. This raises a natural question: Does thinking more at test-time truly lead to better reasoning? To answer this question, we perform a detailed empirical study across models and benchmarks, which reveals a consistent pattern of initial performance improvements from additional thinking followed by a decline, due to "overthinking". To understand this non-monotonic trend, we consider a simple probabilistic model, which reveals that additional thinking increases output variance-creating an illusion of improved reasoning while ultimately undermining precision. Thus, observed gains from "more thinking" are not true indicators of improved reasoning, but artifacts stemming from the connection between model uncertainty and evaluation metric. This suggests that test-time scaling through extended thinking is not an effective way to utilize the inference thinking budget. Recognizing these limitations, we introduce an alternative test-time scaling approach, parallel thinking, inspired by Best-of-N sampling. Our method generates multiple independent reasoning paths within the same inference budget and selects the most consistent response via majority vote, achieving up to 20% higher accuracy compared to extended thinking. This provides a simple yet effective mechanism for test-time scaling of reasoning models. |
| 2025-06-04 | [Advancing Multimodal Reasoning: From Optimized Cold Start to Staged Reinforcement Learning](http://arxiv.org/abs/2506.04207v1) | Shuang Chen, Yue Guo et al. | Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex textual tasks, many works attempt to incentivize similar capabilities in Multimodal Large Language Models (MLLMs) by directly applying reinforcement learning (RL). However, they still struggle to activate complex reasoning. In this paper, rather than examining multimodal RL in isolation, we delve into current training pipelines and identify three crucial phenomena: 1) Effective cold start initialization is critical for enhancing MLLM reasoning. Intriguingly, we find that initializing with carefully selected text data alone can lead to performance surpassing many recent multimodal reasoning models, even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers from gradient stagnation, which degrades training stability and performance. 3) Subsequent text-only RL training, following the multimodal RL phase, further enhances multimodal reasoning. This staged training approach effectively balances perceptual grounding and cognitive reasoning development. By incorporating the above insights and addressing multimodal RL issues, we introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath, LogicVista, DynaMath, and challenging AIME2024 and AIME2025. |
| 2025-06-04 | [EPiC: Towards Lossless Speedup for Reasoning Training through Edge-Preserving CoT Condensation](http://arxiv.org/abs/2506.04205v1) | Jinghan Jia, Hadi Reisizadeh et al. | Large language models (LLMs) have shown remarkable reasoning capabilities when trained with chain-of-thought (CoT) supervision. However, the long and verbose CoT traces, especially those distilled from large reasoning models (LRMs) such as DeepSeek-R1, significantly increase training costs during the distillation process, where a non-reasoning base model is taught to replicate the reasoning behavior of an LRM. In this work, we study the problem of CoT condensation for resource-efficient reasoning training, aimed at pruning intermediate reasoning steps (i.e., thoughts) in CoT traces, enabling supervised model training on length-reduced CoT data while preserving both answer accuracy and the model's ability to generate coherent reasoning. Our rationale is that CoT traces typically follow a three-stage structure: problem understanding, exploration, and solution convergence. Through empirical analysis, we find that retaining the structure of the reasoning trace, especially the early stage of problem understanding (rich in reflective cues) and the final stage of solution convergence, is sufficient to achieve lossless reasoning supervision. To this end, we propose an Edge-Preserving Condensation method, EPiC, which selectively retains only the initial and final segments of each CoT trace while discarding the middle portion. This design draws an analogy to preserving the "edge" of a reasoning trajectory, capturing both the initial problem framing and the final answer synthesis, to maintain logical continuity. Experiments across multiple model families (Qwen and LLaMA) and benchmarks show that EPiC reduces training time by over 34% while achieving lossless reasoning accuracy on MATH500, comparable to full CoT supervision. To the best of our knowledge, this is the first study to explore thought-level CoT condensation for efficient reasoning model distillation. |
| 2025-06-04 | [OpenThoughts: Data Recipes for Reasoning Models](http://arxiv.org/abs/2506.04178v1) | Etash Guha, Ryan Marten et al. | Reasoning models have made rapid progress on many benchmarks involving math, code, and science. Yet, there are still many open questions about the best training recipes for reasoning since state-of-the-art models often rely on proprietary datasets with little to no public information available. To address this, the goal of the OpenThoughts project is to create open-source datasets for training reasoning models. After initial explorations, our OpenThoughts2-1M dataset led to OpenThinker2-32B, the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks such as AIME and LiveCodeBench. We then improve our dataset further by systematically investigating each step of our data generation pipeline with 1,000+ controlled experiments, which led to OpenThoughts3. Scaling the pipeline to 1.2M examples and using QwQ-32B as teacher yields our OpenThinker3-7B model, which achieves state-of-the-art results: 53% on AIME 2025, 51% on LiveCodeBench 06/24-01/25, and 54% on GPQA Diamond. All of our datasets and models are available on https://openthoughts.ai. |
| 2025-06-04 | [OpenThoughts: Data Recipes for Reasoning Models](http://arxiv.org/abs/2506.04178v2) | Etash Guha, Ryan Marten et al. | Reasoning models have made rapid progress on many benchmarks involving math, code, and science. Yet, there are still many open questions about the best training recipes for reasoning since state-of-the-art models often rely on proprietary datasets with little to no public information available. To address this, the goal of the OpenThoughts project is to create open-source datasets for training reasoning models. After initial explorations, our OpenThoughts2-1M dataset led to OpenThinker2-32B, the first model trained on public reasoning data to match DeepSeek-R1-Distill-32B on standard reasoning benchmarks such as AIME and LiveCodeBench. We then improve our dataset further by systematically investigating each step of our data generation pipeline with 1,000+ controlled experiments, which led to OpenThoughts3. Scaling the pipeline to 1.2M examples and using QwQ-32B as teacher yields our OpenThoughts3-7B model, which achieves state-of-the-art results: 53% on AIME 2025, 51% on LiveCodeBench 06/24-01/25, and 54% on GPQA Diamond - improvements of 15.3, 17.2, and 20.5 percentage points compared to the DeepSeek-R1-Distill-Qwen-7B. All of our datasets and models are available on https://openthoughts.ai. |
| 2025-06-04 | [SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL](http://arxiv.org/abs/2506.04147v1) | Jiaheng Hu, Peter Stone et al. | Building capable household and industrial robots requires mastering the control of versatile, high-degree-of-freedom (DoF) systems such as mobile manipulators. While reinforcement learning (RL) holds promise for autonomously acquiring robot control policies, scaling it to high-DoF embodiments remains challenging. Direct RL in the real world demands both safe exploration and high sample efficiency, which are difficult to achieve in practice. Sim-to-real RL, on the other hand, is often brittle due to the reality gap. This paper introduces SLAC, a method that renders real-world RL feasible for complex embodiments by leveraging a low-fidelity simulator to pretrain a task-agnostic latent action space. SLAC trains this latent action space via a customized unsupervised skill discovery method designed to promote temporal abstraction, disentanglement, and safety, thereby facilitating efficient downstream learning. Once a latent action space is learned, SLAC uses it as the action interface for a novel off-policy RL algorithm to autonomously learn downstream tasks through real-world interactions. We evaluate SLAC against existing methods on a suite of bimanual mobile manipulation tasks, where it achieves state-of-the-art performance. Notably, SLAC learns contact-rich whole-body tasks in under an hour of real-world interactions, without relying on any demonstrations or hand-crafted behavior priors. More information, code, and videos at robo-rl.github.io |
| 2025-06-04 | [macOSWorld: A Multilingual Interactive Benchmark for GUI Agents](http://arxiv.org/abs/2506.04135v1) | Pei Yang, Hai Ci et al. | Graphical User Interface (GUI) agents show promising capabilities for automating computer-use tasks and facilitating accessibility, but existing interactive benchmarks are mostly English-only, covering web-use or Windows, Linux, and Android environments, but not macOS. macOS is a major OS with distinctive GUI patterns and exclusive applications. To bridge the gaps, we present macOSWorld, the first comprehensive benchmark for evaluating GUI agents on macOS. macOSWorld features 202 multilingual interactive tasks across 30 applications (28 macOS-exclusive), with task instructions and OS interfaces offered in 5 languages (English, Chinese, Arabic, Japanese, and Russian). As GUI agents are shown to be vulnerable to deception attacks, macOSWorld also includes a dedicated safety benchmarking subset. Our evaluation on six GUI agents reveals a dramatic gap: proprietary computer-use agents lead at above 30% success rate, while open-source lightweight research models lag at below 2%, highlighting the need for macOS domain adaptation. Multilingual benchmarks also expose common weaknesses, especially in Arabic, with a 27.5% average degradation compared to English. Results from safety benchmarking also highlight that deception attacks are more general and demand immediate attention. macOSWorld is available at https://github.com/showlab/macosworld. |
| 2025-06-04 | [macOSWorld: A Multilingual Interactive Benchmark for GUI Agents](http://arxiv.org/abs/2506.04135v2) | Pei Yang, Hai Ci et al. | Graphical User Interface (GUI) agents show promising capabilities for automating computer-use tasks and facilitating accessibility, but existing interactive benchmarks are mostly English-only, covering web-use or Windows, Linux, and Android environments, but not macOS. macOS is a major OS with distinctive GUI patterns and exclusive applications. To bridge the gaps, we present macOSWorld, the first comprehensive benchmark for evaluating GUI agents on macOS. macOSWorld features 202 multilingual interactive tasks across 30 applications (28 macOS-exclusive), with task instructions and OS interfaces offered in 5 languages (English, Chinese, Arabic, Japanese, and Russian). As GUI agents are shown to be vulnerable to deception attacks, macOSWorld also includes a dedicated safety benchmarking subset. Our evaluation on six GUI agents reveals a dramatic gap: proprietary computer-use agents lead at above 30% success rate, while open-source lightweight research models lag at below 2%, highlighting the need for macOS domain adaptation. Multilingual benchmarks also expose common weaknesses, especially in Arabic, with a 27.5% average degradation compared to English. Results from safety benchmarking also highlight that deception attacks are more general and demand immediate attention. macOSWorld is available at https://github.com/showlab/macosworld. |
| 2025-06-04 | [Contour Errors: An Ego-Centric Metric for Reliable 3D Multi-Object Tracking](http://arxiv.org/abs/2506.04122v1) | Sharang Kaul, Mario Berk et al. | Finding reliable matches is essential in multi-object tracking to ensure the accuracy and reliability of perception systems in safety-critical applications such as autonomous vehicles. Effective matching mitigates perception errors, enhancing object identification and tracking for improved performance and safety. However, traditional metrics such as Intersection over Union (IoU) and Center Point Distances (CPDs), which are effective in 2D image planes, often fail to find critical matches in complex 3D scenes. To address this limitation, we introduce Contour Errors (CEs), an ego or object-centric metric for identifying matches of interest in tracking scenarios from a functional perspective. By comparing bounding boxes in the ego vehicle's frame, contour errors provide a more functionally relevant assessment of object matches. Extensive experiments on the nuScenes dataset demonstrate that contour errors improve the reliability of matches over the state-of-the-art 2D IoU and CPD metrics in tracking-by-detection methods. In 3D car tracking, our results show that Contour Errors reduce functional failures (FPs/FNs) by 80% at close ranges and 60% at far ranges compared to IoU in the evaluation stage. |
| 2025-06-04 | [AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment](http://arxiv.org/abs/2506.04089v1) | Anastasiia Ivanova, Eva Bakaeva et al. | As a part of an embodied agent, Large Language Models (LLMs) are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task ambiguity detection have been proposed. However, it is difficult to compare them because they are tested on different datasets and there is no universal benchmark. For this reason, we propose AmbiK (Ambiguous Tasks in Kitchen Environment), the fully textual dataset of ambiguous instructions addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 1000 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (Human Preferences, Common Sense Knowledge, Safety), with environment descriptions, clarifying questions and answers, user intents, and task plans, for a total of 2000 tasks. We hope that AmbiK will enable researchers to perform a unified comparison of ambiguity detection methods. AmbiK is available at https://github.com/cog-model/AmbiK-dataset. |
| 2025-06-04 | [Multimodal Tabular Reasoning with Privileged Structured Information](http://arxiv.org/abs/2506.04088v1) | Jun-Peng Jiang, Yu Xia et al. | Tabular reasoning involves multi-step information extraction and logical inference over tabular data. While recent advances have leveraged large language models (LLMs) for reasoning over structured tables, such high-quality textual representations are often unavailable in real-world settings, where tables typically appear as images. In this paper, we tackle the task of tabular reasoning from table images, leveraging privileged structured information available during training to enhance multimodal large language models (MLLMs). The key challenges lie in the complexity of accurately aligning structured information with visual representations, and in effectively transferring structured reasoning skills to MLLMs despite the input modality gap. To address these, we introduce TabUlar Reasoning with Bridged infOrmation ({\sc Turbo}), a new framework for multimodal tabular reasoning with privileged structured tables. {\sc Turbo} benefits from a structure-aware reasoning trace generator based on DeepSeek-R1, contributing to high-quality modality-bridged data. On this basis, {\sc Turbo} repeatedly generates and selects the advantageous reasoning paths, further enhancing the model's tabular reasoning ability. Experimental results demonstrate that, with limited ($9$k) data, {\sc Turbo} achieves state-of-the-art performance ($+7.2\%$ vs. previous SOTA) across multiple datasets. |
| 2025-06-04 | [Think Like a Person Before Responding: A Multi-Faceted Evaluation of Persona-Guided LLMs for Countering Hate](http://arxiv.org/abs/2506.04043v1) | Mikel K. Ngueajio, Flor Miriam Plaza-del-Arco et al. | Automated counter-narratives (CN) offer a promising strategy for mitigating online hate speech, yet concerns about their affective tone, accessibility, and ethical risks remain. We propose a framework for evaluating Large Language Model (LLM)-generated CNs across four dimensions: persona framing, verbosity and readability, affective tone, and ethical robustness. Using GPT-4o-Mini, Cohere's CommandR-7B, and Meta's LLaMA 3.1-70B, we assess three prompting strategies on the MT-Conan and HatEval datasets. Our findings reveal that LLM-generated CNs are often verbose and adapted for people with college-level literacy, limiting their accessibility. While emotionally guided prompts yield more empathetic and readable responses, there remain concerns surrounding safety and effectiveness. |
| 2025-06-04 | [Generating Automotive Code: Large Language Models for Software Development and Verification in Safety-Critical Systems](http://arxiv.org/abs/2506.04038v1) | Sven Kirchner, Alois C. Knoll | Developing safety-critical automotive software presents significant challenges due to increasing system complexity and strict regulatory demands. This paper proposes a novel framework integrating Generative Artificial Intelligence (GenAI) into the Software Development Lifecycle (SDLC). The framework uses Large Language Models (LLMs) to automate code generation in languages such as C++, incorporating safety-focused practices such as static verification, test-driven development and iterative refinement. A feedback-driven pipeline ensures the integration of test, simulation and verification for compliance with safety standards. The framework is validated through the development of an Adaptive Cruise Control (ACC) system. Comparative benchmarking of LLMs ensures optimal model selection for accuracy and reliability. Results demonstrate that the framework enables automatic code generation while ensuring compliance with safety-critical requirements, systematically integrating GenAI into automotive software engineering. This work advances the use of AI in safety-critical domains, bridging the gap between state-of-the-art generative models and real-world safety requirements. |
| 2025-06-04 | [Asterinas: A Linux ABI-Compatible, Rust-Based Framekernel OS with a Small and Sound TCB](http://arxiv.org/abs/2506.03876v1) | Yuke Peng, Hongliang Tian et al. | How can one build a feature-rich, general-purpose, Rust-based operating system (OS) with a minimal and sound Trusted Computing Base (TCB) for memory safety? Existing Rust-based OSes fall short due to their improper use of unsafe Rust in kernel development. To address this challenge, we propose a novel OS architecture called framekernel that realizes Rust's full potential to achieve intra-kernel privilege separation, ensuring TCB minimality and soundness. We present OSTD, a streamlined framework for safe Rust OS development, and Asterinas, a Linux ABI-compatible framekernel OS implemented entirely in safe Rust using OSTD. Supporting over 210 Linux system calls, Asterinas delivers performance on par with Linux, while maintaining a minimized, memory-safety TCB of only about 14.0% of the codebase. These results underscore the practicality and benefits of the framekernel architecture in building safe and efficient OSes. |
| 2025-06-04 | [Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](http://arxiv.org/abs/2506.03850v1) | Liang Chen, Xueting Han et al. | Harmful fine-tuning (HFT), performed directly on open-source LLMs or through Fine-tuning-as-a-Service, breaks safety alignment and poses significant threats. Existing methods aim to mitigate HFT risks by learning robust representation on alignment data or making harmful data unlearnable, but they treat each data sample equally, leaving data vulnerability patterns understudied. In this work, we reveal that certain subsets of alignment data are consistently more prone to forgetting during HFT across different fine-tuning tasks. Inspired by these findings, we propose Vulnerability-Aware Alignment (VAA), which estimates data vulnerability, partitions data into "vulnerable" and "invulnerable" groups, and encourages balanced learning using a group distributionally robust optimization (Group DRO) framework. Specifically, VAA learns an adversarial sampler that samples examples from the currently underperforming group and then applies group-dependent adversarial perturbations to the data during training, aiming to encourage a balanced learning process across groups. Experiments across four fine-tuning tasks demonstrate that VAA significantly reduces harmful scores while preserving downstream task performance, outperforming state-of-the-art baselines. |
| 2025-06-04 | [Enhancing Safety of Foundation Models for Visual Navigation through Collision Avoidance via Repulsive Estimation](http://arxiv.org/abs/2506.03834v1) | Joonkyung Kim, Joonyeol Sim et al. | We propose CARE (Collision Avoidance via Repulsive Estimation), a plug-and-play module that enhances the safety of vision-based navigation without requiring additional range sensors or fine-tuning of pretrained models. While recent foundation models using only RGB inputs have shown strong performance, they often fail to generalize in out-of-distribution (OOD) environments with unseen objects or variations in camera parameters (e.g., field of view, pose, or focal length). Without fine-tuning, these models may generate unsafe trajectories that lead to collisions, requiring costly data collection and retraining. CARE addresses this limitation by seamlessly integrating with any RGB-based navigation system that outputs local trajectories, dynamically adjusting them using repulsive force vectors derived from monocular depth maps. We evaluate CARE by combining it with state-of-the-art vision-based navigation models across multiple robot platforms. CARE consistently reduces collision rates (up to 100%) without sacrificing goal-reaching performance and improves collision-free travel distance by up to 10.7x in exploration tasks. |
| 2025-06-04 | [From Theory to Practice: Real-World Use Cases on Trustworthy LLM-Driven Process Modeling, Prediction and Automation](http://arxiv.org/abs/2506.03801v1) | Peter Pfeiffer, Alexander Rombach et al. | Traditional Business Process Management (BPM) struggles with rigidity, opacity, and scalability in dynamic environments while emerging Large Language Models (LLMs) present transformative opportunities alongside risks. This paper explores four real-world use cases that demonstrate how LLMs, augmented with trustworthy process intelligence, redefine process modeling, prediction, and automation. Grounded in early-stage research projects with industrial partners, the work spans manufacturing, modeling, life-science, and design processes, addressing domain-specific challenges through human-AI collaboration. In manufacturing, an LLM-driven framework integrates uncertainty-aware explainable Machine Learning (ML) with interactive dialogues, transforming opaque predictions into auditable workflows. For process modeling, conversational interfaces democratize BPMN design. Pharmacovigilance agents automate drug safety monitoring via knowledge-graph-augmented LLMs. Finally, sustainable textile design employs multi-agent systems to navigate regulatory and environmental trade-offs. We intend to examine tensions between transparency and efficiency, generalization and specialization, and human agency versus automation. By mapping these trade-offs, we advocate for context-sensitive integration prioritizing domain needs, stakeholder values, and iterative human-in-the-loop workflows over universal solutions. This work provides actionable insights for researchers and practitioners aiming to operationalize LLMs in critical BPM environments. |
| 2025-06-04 | [Misalignment or misuse? The AGI alignment tradeoff](http://arxiv.org/abs/2506.03755v1) | Max Hellrigel-Holderbaum, Leonard Dung | Creating systems that are aligned with our goals is seen as a leading approach to create safe and beneficial AI in both leading AI companies and the academic field of AI safety. We defend the view that misaligned AGI - future, generally intelligent (robotic) AI agents - poses catastrophic risks. At the same time, we support the view that aligned AGI creates a substantial risk of catastrophic misuse by humans. While both risks are severe and stand in tension with one another, we show that - in principle - there is room for alignment approaches which do not increase misuse risk. We then investigate how the tradeoff between misalignment and misuse looks empirically for different technical approaches to AI alignment. Here, we argue that many current alignment techniques and foreseeable improvements thereof plausibly increase risks of catastrophic misuse. Since the impacts of AI depend on the social context, we close by discussing important social factors and suggest that to reduce the risk of a misuse catastrophe due to aligned AGI, techniques such as robustness, AI control methods and especially good governance seem essential. |

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



