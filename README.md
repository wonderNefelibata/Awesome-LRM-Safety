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
| 2025-04-21 | [Interpretable Locomotion Prediction in Construction Using a Memory-Driven LLM Agent With Chain-of-Thought Reasoning](http://arxiv.org/abs/2504.15263v1) | Ehsan Ahmadi, Chao Wang | Construction tasks are inherently unpredictable, with dynamic environments and safety-critical demands posing significant risks to workers. Exoskeletons offer potential assistance but falter without accurate intent recognition across diverse locomotion modes. This paper presents a locomotion prediction agent leveraging Large Language Models (LLMs) augmented with memory systems, aimed at improving exoskeleton assistance in such settings. Using multimodal inputs - spoken commands and visual data from smart glasses - the agent integrates a Perception Module, Short-Term Memory (STM), Long-Term Memory (LTM), and Refinement Module to predict locomotion modes effectively. Evaluation reveals a baseline weighted F1-score of 0.73 without memory, rising to 0.81 with STM, and reaching 0.90 with both STM and LTM, excelling with vague and safety-critical commands. Calibration metrics, including a Brier Score drop from 0.244 to 0.090 and ECE from 0.222 to 0.044, affirm improved reliability. This framework supports safer, high-level human-exoskeleton collaboration, with promise for adaptive assistive systems in dynamic industries. |
| 2025-04-21 | [Leveraging Language Models for Automated Patient Record Linkage](http://arxiv.org/abs/2504.15261v1) | Mohammad Beheshti, Lovedeep Gondara et al. | Objective: Healthcare data fragmentation presents a major challenge for linking patient data, necessitating robust record linkage to integrate patient records from diverse sources. This study investigates the feasibility of leveraging language models for automated patient record linkage, focusing on two key tasks: blocking and matching. Materials and Methods: We utilized real-world healthcare data from the Missouri Cancer Registry and Research Center, linking patient records from two independent sources using probabilistic linkage as a baseline. A transformer-based model, RoBERTa, was fine-tuned for blocking using sentence embeddings. For matching, several language models were experimented under fine-tuned and zero-shot settings, assessing their performance against ground truth labels. Results: The fine-tuned blocking model achieved a 92% reduction in the number of candidate pairs while maintaining near-perfect recall. In the matching task, fine-tuned Mistral-7B achieved the best performance with only 6 incorrect predictions. Among zero-shot models, Mistral-Small-24B performed best, with a total of 55 incorrect predictions. Discussion: Fine-tuned language models achieved strong performance in patient record blocking and matching with minimal errors. However, they remain less accurate and efficient than a hybrid rule-based and probabilistic approach for blocking. Additionally, reasoning models like DeepSeek-R1 are impractical for large-scale record linkage due to high computational costs. Conclusion: This study highlights the potential of language models for automating patient record linkage, offering improved efficiency by eliminating the manual efforts required to perform patient record linkage. Overall, language models offer a scalable solution that can enhance data integration, reduce manual effort, and support disease surveillance and research. |
| 2025-04-21 | [FlowReasoner: Reinforcing Query-Level Meta-Agents](http://arxiv.org/abs/2504.15257v1) | Hongcheng Gao, Yue Liu et al. | This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks. The code is available at https://github.com/sail-sg/FlowReasoner. |
| 2025-04-21 | [CRUST-Bench: A Comprehensive Benchmark for C-to-safe-Rust Transpilation](http://arxiv.org/abs/2504.15254v1) | Anirudh Khatry, Robert Zhang et al. | C-to-Rust transpilation is essential for modernizing legacy C code while enhancing safety and interoperability with modern Rust ecosystems. However, no dataset currently exists for evaluating whether a system can transpile C into safe Rust that passes a set of test cases. We introduce CRUST-Bench, a dataset of 100 C repositories, each paired with manually-written interfaces in safe Rust as well as test cases that can be used to validate correctness of the transpilation. By considering entire repositories rather than isolated functions, CRUST-Bench captures the challenges of translating complex projects with dependencies across multiple files. The provided Rust interfaces provide explicit specifications that ensure adherence to idiomatic, memory-safe Rust patterns, while the accompanying test cases enforce functional correctness. We evaluate state-of-the-art large language models (LLMs) on this task and find that safe and idiomatic Rust generation is still a challenging problem for various state-of-the-art methods and techniques. We also provide insights into the errors LLMs usually make in transpiling code from C to safe Rust. The best performing model, OpenAI o1, is able to solve only 15 tasks in a single-shot setting. Improvements on CRUST-Bench would lead to improved transpilation systems that can reason about complex scenarios and help in migrating legacy codebases from C into languages like Rust that ensure memory safety. You can find the dataset and code at https://github.com/anirudhkhatry/CRUST-bench. |
| 2025-04-21 | [MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning](http://arxiv.org/abs/2504.15241v1) | Yahan Yang, Soham Dan et al. | Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation. |
| 2025-04-21 | [A Genetic Fuzzy-Enabled Framework on Robotic Manipulation for In-Space Servicing](http://arxiv.org/abs/2504.15226v1) | Nathan Steffen, Wilhelm Louw et al. | Automation of robotic systems for servicing in cislunar space is becoming extremely important as the number of satellites in orbit increases. Safety is critical in performing satellite maintenance, so the control techniques utilized must be trusted in addition to being highly efficient. In this work, Genetic Fuzzy Trees are combined with the widely used LQR control scheme via Thales' TrUE AI Toolkit to create a trusted and efficient controller for a two-degree-of-freedom planar robotic manipulator that would theoretically be used to perform satellite maintenance. It was found that Genetic Fuzzy-LQR is 18.5% more performant than optimal LQR on average, and that it is incredibly robust to uncertainty. |
| 2025-04-21 | [Existing Industry Practice for the EU AI Act's General-Purpose AI Code of Practice Safety and Security Measures](http://arxiv.org/abs/2504.15181v1) | Lily Stelling, Mick Yang et al. | This report provides a detailed comparison between the measures proposed in the EU AI Act's General-Purpose AI (GPAI) Code of Practice (Third Draft) and current practices adopted by leading AI companies. As the EU moves toward enforcing binding obligations for GPAI model providers, the Code of Practice will be key to bridging legal requirements with concrete technical commitments. Our analysis focuses on the draft's Safety and Security section which is only relevant for the providers of the most advanced models (Commitments II.1-II.16) and excerpts from current public-facing documents quotes that are relevant to each individual measure.   We systematically reviewed different document types - including companies' frontier safety frameworks and model cards - from over a dozen companies, including OpenAI, Anthropic, Google DeepMind, Microsoft, Meta, Amazon, and others. This report is not meant to be an indication of legal compliance nor does it take any prescriptive viewpoint about the Code of Practice or companies' policies. Instead, it aims to inform the ongoing dialogue between regulators and GPAI model providers by surfacing evidence of precedent. |
| 2025-04-21 | [C2RUST-BENCH: A Minimized, Representative Dataset for C-to-Rust Transpilation Evaluation](http://arxiv.org/abs/2504.15144v1) | Melih Sirlanci, Carter Yagemann et al. | Despite the effort in vulnerability detection over the last two decades, memory safety vulnerabilities continue to be a critical problem. Recent reports suggest that the key solution is to migrate to memory-safe languages. To this end, C-to-Rust transpilation becomes popular to resolve memory-safety issues in C programs. Recent works propose C-to-Rust transpilation frameworks; however, a comprehensive evaluation dataset is missing. Although one solution is to put together a large enough dataset, this increases the analysis time in automated frameworks as well as in manual efforts for some cases. In this work, we build a method to select functions from a large set to construct a minimized yet representative dataset to evaluate the C-to-Rust transpilation. We propose C2RUST-BENCH that contains 2,905 functions, which are representative of C-to-Rust transpilation, selected from 15,503 functions of real-world programs. |
| 2025-04-21 | [EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models](http://arxiv.org/abs/2504.15133v1) | Ziwen Xu, Shuxun Wang et al. | In this paper, we introduce EasyEdit2, a framework designed to enable plug-and-play adjustability for controlling Large Language Model (LLM) behaviors. EasyEdit2 supports a wide range of test-time interventions, including safety, sentiment, personality, reasoning patterns, factuality, and language features. Unlike its predecessor, EasyEdit2 features a new architecture specifically designed for seamless model steering. It comprises key modules such as the steering vector generator and the steering vector applier, which enable automatic generation and application of steering vectors to influence the model's behavior without modifying its parameters. One of the main advantages of EasyEdit2 is its ease of use-users do not need extensive technical knowledge. With just a single example, they can effectively guide and adjust the model's responses, making precise control both accessible and efficient. Empirically, we report model steering performance across different LLMs, demonstrating the effectiveness of these techniques. We have released the source code on GitHub at https://github.com/zjunlp/EasyEdit along with a demonstration notebook. In addition, we provide a demo video at https://zjunlp.github.io/project/EasyEdit2/video for a quick introduction. |
| 2025-04-21 | [Safety Co-Option and Compromised National Security: The Self-Fulfilling Prophecy of Weakened AI Risk Thresholds](http://arxiv.org/abs/2504.15088v1) | Heidy Khlaaf, Sarah Myers West | Risk thresholds provide a measure of the level of risk exposure that a society or individual is willing to withstand, ultimately shaping how we determine the safety of technological systems. Against the backdrop of the Cold War, the first risk analyses, such as those devised for nuclear systems, cemented societally accepted risk thresholds against which safety-critical and defense systems are now evaluated. But today, the appropriate risk tolerances for AI systems have yet to be agreed on by global governing efforts, despite the need for democratic deliberation regarding the acceptable levels of harm to human life. Absent such AI risk thresholds, AI technologists-primarily industry labs, as well as "AI safety" focused organizations-have instead advocated for risk tolerances skewed by a purported AI arms race and speculative "existential" risks, taking over the arbitration of risk determinations with life-or-death consequences, subverting democratic processes.   In this paper, we demonstrate how such approaches have allowed AI technologists to engage in "safety revisionism," substituting traditional safety methods and terminology with ill-defined alternatives that vie for the accelerated adoption of military AI uses at the cost of lowered safety and security thresholds. We explore how the current trajectory for AI risk determination and evaluation for foundation model use within national security is poised for a race to the bottom, to the detriment of the US's national security interests. Safety-critical and defense systems must comply with assurance frameworks that are aligned with established risk thresholds, and foundation models are no exception. As such, development of evaluation frameworks for AI-based military systems must preserve the safety and security of US critical and defense infrastructure, and remain in alignment with international humanitarian law. |
| 2025-04-21 | [RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search](http://arxiv.org/abs/2504.15047v1) | Quy-Anh Dang, Chris Ngo et al. | Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming. |
| 2025-04-21 | [aiXamine: LLM Safety and Security Simplified](http://arxiv.org/abs/2504.14985v1) | Fatih Deniz, Dorde Popovic et al. | Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices. |
| 2025-04-21 | [Distributed Time-Varying Gaussian Regression via Kalman Filtering](http://arxiv.org/abs/2504.14900v1) | Nicola Taddei, Riccardo Maggioni et al. | We consider the problem of learning time-varying functions in a distributed fashion, where agents collect local information to collaboratively achieve a shared estimate. This task is particularly relevant in control applications, whenever real-time and robust estimation of dynamic cost/reward functions in safety critical settings has to be performed. In this paper, we,adopt a finite-dimensional approximation of a Gaussian Process, corresponding to a Bayesian linear regression in an appropriate feature space, and propose a new algorithm, DistKP, to track the time-varying coefficients via a distributed Kalman filter. The proposed method works for arbitrary kernels and under weaker assumptions on the time-evolution of the function to learn compared to the literature. We validate our results using a simulation example in which a fleet of Unmanned Aerial Vehicles (UAVs) learns a dynamically changing wind field. |
| 2025-04-21 | [Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey](http://arxiv.org/abs/2504.14891v1) | Aoran Gan, Hao Yu et al. | Recent advancements in Retrieval-Augmented Generation (RAG) have revolutionized natural language processing by integrating Large Language Models (LLMs) with external information retrieval, enabling accurate, up-to-date, and verifiable text generation across diverse applications. However, evaluating RAG systems presents unique challenges due to their hybrid architecture that combines retrieval and generation components, as well as their dependence on dynamic knowledge sources in the LLM era. In response, this paper provides a comprehensive survey of RAG evaluation methods and frameworks, systematically reviewing traditional and emerging evaluation approaches, for system performance, factual accuracy, safety, and computational efficiency in the LLM era. We also compile and categorize the RAG-specific datasets and evaluation frameworks, conducting a meta-analysis of evaluation practices in high-impact RAG research. To the best of our knowledge, this work represents the most comprehensive survey for RAG evaluation, bridging traditional and LLM-driven methods, and serves as a critical resource for advancing RAG development. |
| 2025-04-21 | [ReCraft: Self-Contained Split, Merge, and Membership Change of Raft Protocol](http://arxiv.org/abs/2504.14802v1) | Kezhi Xiong, Soonwon Moon et al. | Designing reconfiguration schemes for consensus protocols is challenging because subtle corner cases during reconfiguration could invalidate the correctness of the protocol. Thus, most systems that embed consensus protocols conservatively implement the reconfiguration and refrain from developing an efficient scheme. Existing implementations often stop the entire system during reconfiguration and rely on a centralized coordinator, which can become a single point of failure. We present ReCraft, a novel reconfiguration protocol for Raft, which supports multi- and single-cluster-level reconfigurations. ReCraft does not rely on external coordinators and blocks minimally. ReCraft enables the sharding of Raft clusters with split and merge reconfigurations and adds a membership change scheme that improves Raft. We prove the safety and liveness of ReCraft and demonstrate its efficiency through implementations in etcd. |
| 2025-04-20 | [Safe Autonomous Environmental Contact for Soft Robots using Control Barrier Functions](http://arxiv.org/abs/2504.14755v1) | Akua K. Dickson, Juan C. Pacheco Garcia et al. | Robots built from soft materials will inherently apply lower environmental forces than their rigid counterparts, and therefore may be more suitable in sensitive settings with unintended contact. However, these robots' applied forces result from both their design and their control system in closed-loop, and therefore, ensuring bounds on these forces requires controller synthesis for safety as well. This article introduces the first feedback controller for a soft manipulator that formally meets a safety specification with respect to environmental contact. In our proof-of-concept setting, the robot's environment has known geometry and is deformable with a known elastic modulus. Our approach maps a bound on applied forces to a safe set of positions of the robot's tip via predicted deformations of the environment. Then, a quadratic program with Control Barrier Functions in its constraints is used to supervise a nominal feedback signal, verifiably maintaining the robot's tip within this safe set. Hardware experiments on a multi-segment soft pneumatic robot demonstrate that the proposed framework successfully constrains its environmental contact forces. This framework represents a fundamental shift in perspective on control and safety for soft robots, defining and implementing a formally verifiable logic specification on their pose and contact forces. |
| 2025-04-20 | [Adaptive Field Effect Planner for Safe Interactive Autonomous Driving on Curved Roads](http://arxiv.org/abs/2504.14747v1) | Qinghao Li, Zhen Tian et al. | Autonomous driving has garnered significant attention for its potential to improve safety, traffic efficiency, and user convenience. However, the dynamic and complex nature of interactive driving poses significant challenges, including the need to navigate non-linear road geometries, handle dynamic obstacles, and meet stringent safety and comfort requirements. Traditional approaches, such as artificial potential fields (APF), often fall short in addressing these complexities independently, necessitating the development of integrated and adaptive frameworks. This paper presents a novel approach to autonomous vehicle navigation that integrates artificial potential fields, Frenet coordinates, and improved particle swarm optimization (IPSO). A dynamic risk field, adapted from traditional APF, is proposed to ensure interactive safety by quantifying risks and dynamically adjusting lane-changing intentions based on surrounding vehicle behavior. Frenet coordinates are utilized to simplify trajectory planning on non-straight roads, while an enhanced quintic polynomial trajectory generator ensures smooth and comfortable path transitions. Additionally, an IPSO algorithm optimizes trajectory selection in real time, balancing safety and user comfort within a feasible input range. The proposed framework is validated through extensive simulations and real-world scenarios, demonstrating its ability to navigate complex traffic environments, maintain safety margins, and generate smooth, dynamically feasible trajectories. |
| 2025-04-20 | [Can We Ignore Labels In Out of Distribution Detection?](http://arxiv.org/abs/2504.14704v1) | Hong Yang, Qi Yu et al. | Out-of-distribution (OOD) detection methods have recently become more prominent, serving as a core element in safety-critical autonomous systems. One major purpose of OOD detection is to reject invalid inputs that could lead to unpredictable errors and compromise safety. Due to the cost of labeled data, recent works have investigated the feasibility of self-supervised learning (SSL) OOD detection, unlabeled OOD detection, and zero shot OOD detection. In this work, we identify a set of conditions for a theoretical guarantee of failure in unlabeled OOD detection algorithms from an information-theoretic perspective. These conditions are present in all OOD tasks dealing with real-world data: I) we provide theoretical proof of unlabeled OOD detection failure when there exists zero mutual information between the learning objective and the in-distribution labels, a.k.a. 'label blindness', II) we define a new OOD task - Adjacent OOD detection - that tests for label blindness and accounts for a previously ignored safety gap in all OOD detection benchmarks, and III) we perform experiments demonstrating that existing unlabeled OOD methods fail under conditions suggested by our label blindness theory and analyze the implications for future research in unlabeled OOD methods. |
| 2025-04-20 | [A Byzantine Fault Tolerance Approach towards AI Safety](http://arxiv.org/abs/2504.14668v1) | John deVadoss, Matthias Artzt | Ensuring that an AI system behaves reliably and as intended, especially in the presence of unexpected faults or adversarial conditions, is a complex challenge. Inspired by the field of Byzantine Fault Tolerance (BFT) from distributed computing, we explore a fault tolerance architecture for AI safety. By drawing an analogy between unreliable, corrupt, misbehaving or malicious AI artifacts and Byzantine nodes in a distributed system, we propose an architecture that leverages consensus mechanisms to enhance AI safety and reliability. |
| 2025-04-20 | [BLACKOUT: Data-Oblivious Computation with Blinded Capabilities](http://arxiv.org/abs/2504.14654v1) | Hossam ElAtali, Merve Gülmez et al. | Lack of memory-safety and exposure to side channels are two prominent, persistent challenges for the secure implementation of software. Memory-safe programming languages promise to significantly reduce the prevalence of memory-safety bugs, but make it more difficult to implement side-channel-resistant code. We aim to address both memory-safety and side-channel resistance by augmenting memory-safe hardware with the ability for data-oblivious programming. We describe an extension to the CHERI capability architecture to provide blinded capabilities that allow data-oblivious computation to be carried out by userspace tasks. We also present BLACKOUT, our realization of blinded capabilities on a FPGA softcore based on the speculative out-of-order CHERI-Toooba processor and extend the CHERI-enabled Clang/LLVM compiler and the CheriBSD operating system with support for blinded capabilities. BLACKOUT makes writing side-channel-resistant code easier by making non-data-oblivious operations via blinded capabilities explicitly fault. Through rigorous evaluation we show that BLACKOUT ensures memory operated on through blinded capabilities is securely allocated, used, and reclaimed and demonstrate that, in benchmarks comparable to those used by previous work, BLACKOUT imposes only a small performance degradation (1.5% geometric mean) compared to the baseline CHERI-Toooba processor. |

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



