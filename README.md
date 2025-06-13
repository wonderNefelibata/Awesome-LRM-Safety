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
| 2025-06-12 | [How Well Can Reasoning Models Identify and Recover from Unhelpful Thoughts?](http://arxiv.org/abs/2506.10979v1) | Sohee Yang, Sang-Woo Lee et al. | Recent reasoning models show the ability to reflect, backtrack, and self-validate their reasoning, which is crucial in spotting mistakes and arriving at accurate solutions. A natural question that arises is how effectively models can perform such self-reevaluation. We tackle this question by investigating how well reasoning models identify and recover from four types of unhelpful thoughts: uninformative rambling thoughts, thoughts irrelevant to the question, thoughts misdirecting the question as a slightly different question, and thoughts that lead to incorrect answers. We show that models are effective at identifying most unhelpful thoughts but struggle to recover from the same thoughts when these are injected into their thinking process, causing significant performance drops. Models tend to naively continue the line of reasoning of the injected irrelevant thoughts, which showcases that their self-reevaluation abilities are far from a general "meta-cognitive" awareness. Moreover, we observe non/inverse-scaling trends, where larger models struggle more than smaller ones to recover from short irrelevant thoughts, even when instructed to reevaluate their reasoning. We demonstrate the implications of these findings with a jailbreak experiment using irrelevant thought injection, showing that the smallest models are the least distracted by harmful-response-triggering thoughts. Overall, our findings call for improvement in self-reevaluation of reasoning models to develop better reasoning and safer systems. |
| 2025-06-12 | [Build the web for agents, not agents for the web](http://arxiv.org/abs/2506.10953v1) | Xing Han Lù, Gaurav Kamath et al. | Recent advancements in Large Language Models (LLMs) and multimodal counterparts have spurred significant interest in developing web agents -- AI systems capable of autonomously navigating and completing tasks within web environments. While holding tremendous promise for automating complex web interactions, current approaches face substantial challenges due to the fundamental mismatch between human-designed interfaces and LLM capabilities. Current methods struggle with the inherent complexity of web inputs, whether processing massive DOM trees, relying on screenshots augmented with additional information, or bypassing the user interface entirely through API interactions. This position paper advocates for a paradigm shift in web agent research: rather than forcing web agents to adapt to interfaces designed for humans, we should develop a new interaction paradigm specifically optimized for agentic capabilities. To this end, we introduce the concept of an Agentic Web Interface (AWI), an interface specifically designed for agents to navigate a website. We establish six guiding principles for AWI design, emphasizing safety, efficiency, and standardization, to account for the interests of all primary stakeholders. This reframing aims to overcome fundamental limitations of existing interfaces, paving the way for more efficient, reliable, and transparent web agent design, which will be a collaborative effort involving the broader ML community. |
| 2025-06-12 | [Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors](http://arxiv.org/abs/2506.10949v1) | Chen Yueh-Han, Nitish Joshi et al. | Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment. |
| 2025-06-12 | [Viability of Future Actions: Robust Safety in Reinforcement Learning via Entropy Regularization](http://arxiv.org/abs/2506.10871v1) | Pierre-François Massiani, Alexander von Rohr et al. | Despite the many recent advances in reinforcement learning (RL), the question of learning policies that robustly satisfy state constraints under unknown disturbances remains open. In this paper, we offer a new perspective on achieving robust safety by analyzing the interplay between two well-established techniques in model-free RL: entropy regularization, and constraints penalization. We reveal empirically that entropy regularization in constrained RL inherently biases learning toward maximizing the number of future viable actions, thereby promoting constraints satisfaction robust to action noise. Furthermore, we show that by relaxing strict safety constraints through penalties, the constrained RL problem can be approximated arbitrarily closely by an unconstrained one and thus solved using standard model-free RL. This reformulation preserves both safety and optimality while empirically improving resilience to disturbances. Our results indicate that the connection between entropy regularization and robustness is a promising avenue for further empirical and theoretical investigation, as it enables robust safety in RL through simple reward shaping. |
| 2025-06-12 | [MultiCoSim: A Python-based Multi-Fidelity Co-Simulation Framework](http://arxiv.org/abs/2506.10869v1) | Quinn Thibeault, Giulia Pedrielli | Simulation is a foundational tool for the analysis and testing of cyber-physical systems (CPS), underpinning activities such as algorithm development, runtime monitoring, and system verification. As CPS grow in complexity and scale, particularly in safety-critical and learning-enabled settings, accurate analysis and synthesis increasingly rely on the rapid use of simulation experiments. Because CPS inherently integrate hardware, software, and physical processes, simulation platforms must support co-simulation of heterogeneous components at varying levels of fidelity. Despite recent advances in high-fidelity modeling of hardware, firmware, and physics, co-simulation in diverse environments remains challenging. These limitations hinder the development of reusable benchmarks and impede the use of simulation for automated and comparative evaluation.   Existing simulation tools often rely on rigid configurations, lack automation support, and present obstacles to portability and modularity. Many are configured through static text files or impose constraints on how simulation components are represented and connected, making it difficult to flexibly compose systems or integrate components across platforms.   To address these challenges, we introduce MultiCoSim, a Python-based simulation framework that enables users to define, compose, and configure simulation components programmatically. MultiCoSim supports distributed, component-based co-simulation and allows seamless substitution and reconfiguration of components. We demonstrate the flexibility of MultiCoSim through case studies that include co-simulations involving custom automaton-based controllers, as well as integration with off-the-shelf platforms like the PX4 autopilot for aerial robotics. These examples highlight MultiCoSim's capability to streamline CPS simulation pipelines for research and development. |
| 2025-06-12 | [Automotive Radar Online Channel Imbalance Estimation via NLMS](http://arxiv.org/abs/2506.10841v1) | Esmaeil Kavousi Ghafi, Oliver Lang et al. | Automotive radars are one of the essential enablers of advanced driver assistance systems (ADASs). Continuous monitoring of the functional safety and reliability of automotive radars is a crucial requirement to prevent accidents and increase road safety. One of the most critical aspects to monitor in this context is radar channel imbalances, as they are a key parameter regarding the reliability of the radar. These imbalances may originate from several parameter variations or hardware fatigues, e.g., a solder ball break (SBB), and may affect some radar processing steps, such as the angle of arrival estimation. In this work, a novel method for online estimation of automotive radar channel imbalances is proposed. The proposed method exploits a normalized least mean squares (NLMS) algorithm as a block in the processing chain of the radar to estimate the channel imbalances. The input of this block is the detected targets in the range-Doppler map of the radar on the road without any prior knowledge on the angular parameters of the targets. This property in combination with low computational complexity of the NLMS, makes the proposed method suitable for online channel imbalance estimation, in parallel to the normal operation of the radar. Furthermore, it features reduced dependency on specific targets of interest and faster update rates of the channel imbalance estimation compared to the majority of state-of-the-art methods. This improvement is achieved by allowing for multiple targets in the angular spectrum, whereas most other methods are restricted to only single targets in the angular spectrum. The performance of the proposed method is validated using various simulation scenarios and is supported by measurement results. |
| 2025-06-12 | [RationalVLA: A Rational Vision-Language-Action Model with Dual System](http://arxiv.org/abs/2506.10826v1) | Wenxuan Song, Jiayi Chen et al. | A fundamental requirement for real-world robotic deployment is the ability to understand and respond to natural language instructions. Existing language-conditioned manipulation tasks typically assume that instructions are perfectly aligned with the environment. This assumption limits robustness and generalization in realistic scenarios where instructions may be ambiguous, irrelevant, or infeasible. To address this problem, we introduce RAtional MAnipulation (RAMA), a new benchmark that challenges models with both unseen executable instructions and defective ones that should be rejected. In RAMA, we construct a dataset with over 14,000 samples, including diverse defective instructions spanning six dimensions: visual, physical, semantic, motion, safety, and out-of-context. We further propose the Rational Vision-Language-Action model (RationalVLA). It is a dual system for robotic arms that integrates the high-level vision-language model with the low-level manipulation policy by introducing learnable latent space embeddings. This design enables RationalVLA to reason over instructions, reject infeasible commands, and execute manipulation effectively. Experiments demonstrate that RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher success rate and 0.94 average task length, while maintaining competitive performance on standard manipulation tasks. Real-world trials further validate its effectiveness and robustness in practical applications. Our project page is https://irpn-eai.github.io/rationalvla. |
| 2025-06-12 | [Barium Calcium Zirconium Titanate Thin Film-Based Capacitive Thermoelectric Converter for Low-Grade Waste Heat](http://arxiv.org/abs/2506.10759v1) | Mohammad K. Al Thehaiban, Vladimir S. Getov et al. | A capacitive thermoelectric device can harvest thermal energy and convert it to electrical energy by employing a temperature-dependent dielectric material whose permittivity sharply changes with temperature. Electricity can be generated by fluctuating the temperature of the capacitor. Currently, capacitive thermoelectric devices are not broadly used, which can be attributed to the low efficiency of the existing solutions, the lack of dielectric materials with suitable temperature non-linearity of the dielectric permittivity, and the complexity of modulating heat flux on the dielectric material. Here, we propose a device based on (Ba0.85Ca0.15)(Ti0.92Zr0.08)O3 and (Ba0.73Ca0.27)(Ti0.98Zr0.02)O3 thin films. The estimated power output under different operation conditions and dynamic workload of an Intel E5-2630 microprocessor show that these thin film materials are promising and can potentially be used for a capacitive thermoelectric converter. |
| 2025-06-12 | [Beyond True or False: Retrieval-Augmented Hierarchical Analysis of Nuanced Claims](http://arxiv.org/abs/2506.10728v1) | Priyanka Kargupta, Runchu Tian et al. | Claims made by individuals or entities are oftentimes nuanced and cannot be clearly labeled as entirely "true" or "false" -- as is frequently the case with scientific and political claims. However, a claim (e.g., "vaccine A is better than vaccine B") can be dissected into its integral aspects and sub-aspects (e.g., efficacy, safety, distribution), which are individually easier to validate. This enables a more comprehensive, structured response that provides a well-rounded perspective on a given problem while also allowing the reader to prioritize specific angles of interest within the claim (e.g., safety towards children). Thus, we propose ClaimSpect, a retrieval-augmented generation-based framework for automatically constructing a hierarchy of aspects typically considered when addressing a claim and enriching them with corpus-specific perspectives. This structure hierarchically partitions an input corpus to retrieve relevant segments, which assist in discovering new sub-aspects. Moreover, these segments enable the discovery of varying perspectives towards an aspect of the claim (e.g., support, neutral, or oppose) and their respective prevalence (e.g., "how many biomedical papers believe vaccine A is more transportable than B?"). We apply ClaimSpect to a wide variety of real-world scientific and political claims featured in our constructed dataset, showcasing its robustness and accuracy in deconstructing a nuanced claim and representing perspectives within a corpus. Through real-world case studies and human evaluation, we validate its effectiveness over multiple baselines. |
| 2025-06-12 | [PREMISE: Scalable and Strategic Prompt Optimization for Efficient Mathematical Reasoning in Large Models](http://arxiv.org/abs/2506.10716v1) | Ye Yu, Yaoning Yu et al. | Large reasoning models (LRMs) such as Claude 3.7 Sonnet and OpenAI o1 achieve strong performance on mathematical benchmarks using lengthy chain-of-thought (CoT) reasoning, but the resulting traces are often unnecessarily verbose. This inflates token usage and cost, limiting deployment in latency-sensitive or API-constrained settings. We introduce PREMISE (PRompt-based Efficient Mathematical Inference with Strategic Evaluation), a prompt-only framework that reduces reasoning overhead without modifying model weights. PREMISE combines trace-level diagnostics with gradient-inspired prompt optimization to minimize redundant computation while preserving answer accuracy. The approach jointly optimizes brevity and correctness through a multi-objective textual search that balances token length and answer validity. Unlike prior work, PREMISE runs in a single-pass black-box interface, so it can be applied directly to commercial LLMs. On GSM8K, SVAMP, and Math500 we match or exceed baseline accuracy ($96\%\rightarrow96\%$ with Claude, $91\%\rightarrow92\%$ with Gemini) while reducing reasoning tokens by up to $87.5\%$ and cutting dollar cost by $69$--$82\%$. These results show that prompt-level optimization is a practical and scalable path to efficient LRM inference without compromising reasoning quality. |
| 2025-06-12 | [GOLIATH: A Decentralized Framework for Data Collection in Intelligent Transportation Systems](http://arxiv.org/abs/2506.10665v1) | Davide Maffiola, Stefano Longari et al. | Intelligent Transportation Systems (ITSs) technology has advanced during the past years, and it is now used for several applications that require vehicles to exchange real-time data, such as in traffic information management. Traditionally, road traffic information has been collected using on-site sensors. However, crowd-sourcing traffic information from onboard sensors or smartphones has become a viable alternative. State-of-the-art solutions currently follow a centralized model where only the service provider has complete access to the collected traffic data and represent a single point of failure and trust. In this paper, we propose GOLIATH, a blockchain-based decentralized framework that runs on the In-Vehicle Infotainment (IVI) system to collect real-time information exchanged between the network's participants. Our approach mitigates the limitations of existing crowd-sourcing centralized solutions by guaranteeing trusted information collection and exchange, fully exploiting the intrinsic distributed nature of vehicles. We demonstrate its feasibility in the context of vehicle positioning and traffic information management. Each vehicle participating in the decentralized network shares its position and neighbors' ones in the form of a transaction recorded on the ledger, which uses a novel consensus mechanism to validate it. We design the consensus mechanism resilient against a realistic set of adversaries that aim to tamper or disable the communication. We evaluate the proposed framework in a simulated (but realistic) environment, which considers different threats and allows showing its robustness and safety properties. |
| 2025-06-12 | [CyFence: Securing Cyber-Physical Controllers via Trusted Execution Environment](http://arxiv.org/abs/2506.10638v1) | Stefano Longari, Alessandro Pozone et al. | In the last decades, Cyber-physical Systems (CPSs) have experienced a significant technological evolution and increased connectivity, at the cost of greater exposure to cyber-attacks. Since many CPS are used in safety-critical systems, such attacks entail high risks and potential safety harms. Although several defense strategies have been proposed, they rarely exploit the cyber-physical nature of the system. In this work, we exploit the nature of CPS by proposing CyFence, a novel architecture that improves the resilience of closed-loop control systems against cyber-attacks by adding a semantic check, used to confirm that the system is behaving as expected. To ensure the security of the semantic check code, we use the Trusted Execution Environment implemented by modern processors. We evaluate CyFence considering a real-world application, consisting of an active braking digital controller, demonstrating that it can mitigate different types of attacks with a negligible computation overhead. |
| 2025-06-12 | [Time Series Forecasting as Reasoning: A Slow-Thinking Approach with Reinforced LLMs](http://arxiv.org/abs/2506.10630v1) | Yucong Luo, Yitong Zhou et al. | To advance time series forecasting (TSF), various methods have been proposed to improve prediction accuracy, evolving from statistical techniques to data-driven deep learning architectures. Despite their effectiveness, most existing methods still adhere to a fast thinking paradigm-relying on extracting historical patterns and mapping them to future values as their core modeling philosophy, lacking an explicit thinking process that incorporates intermediate time series reasoning. Meanwhile, emerging slow-thinking LLMs (e.g., OpenAI-o1) have shown remarkable multi-step reasoning capabilities, offering an alternative way to overcome these issues. However, prompt engineering alone presents several limitations - including high computational cost, privacy risks, and limited capacity for in-depth domain-specific time series reasoning. To address these limitations, a more promising approach is to train LLMs to develop slow thinking capabilities and acquire strong time series reasoning skills. For this purpose, we propose Time-R1, a two-stage reinforcement fine-tuning framework designed to enhance multi-step reasoning ability of LLMs for time series forecasting. Specifically, the first stage conducts supervised fine-tuning for warmup adaptation, while the second stage employs reinforcement learning to improve the model's generalization ability. Particularly, we design a fine-grained multi-objective reward specifically for time series forecasting, and then introduce GRIP (group-based relative importance for policy optimization), which leverages non-uniform sampling to further encourage and optimize the model's exploration of effective reasoning paths. Experiments demonstrate that Time-R1 significantly improves forecast performance across diverse datasets. |
| 2025-06-12 | [Scalable Software Testing in Fast Virtual Platforms: Leveraging SystemC, QEMU and Containerization](http://arxiv.org/abs/2506.10624v1) | Lukas Jünger, Jan Henrik Weinstock et al. | The ever-increasing complexity of HW/SW systems presents a persistent challenge, particularly in safety-critical domains like automotive, where extensive testing is imperative. However, the availability of hardware often lags behind, hindering early-stage software development. To address this, Virtual Platforms (VPs) based on the SystemC TLM-2.0 standard have emerged as a pivotal solution, enabling pre-silicon execution and testing of unmodified target software. In this study, we propose an approach leveraging containerization to encapsulate VPs in order to reduce environment dependencies and enable cloud deployment for fast, parallelized test execution, as well as open-source VP technologies such as QEMU and VCML to obviate the need for seat licenses. To demonstrate the efficacy of our approach, we present an Artificial Intelligence (AI) accelerator VP case study. Through our research, we offer a robust solution to address the challenges posed by the complexity of HW/SW systems, with practical implications for accelerating HW/SW co-development. |
| 2025-06-12 | [SoK: Evaluating Jailbreak Guardrails for Large Language Models](http://arxiv.org/abs/2506.10597v1) | Xunguang Wang, Zhenlan Ji et al. | Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails. |
| 2025-06-12 | [LogiPlan: A Structured Benchmark for Logical Planning and Relational Reasoning in LLMs](http://arxiv.org/abs/2506.10527v1) | Yanan Cai, Ahmed Salem et al. | We introduce LogiPlan, a novel benchmark designed to evaluate the capabilities of large language models (LLMs) in logical planning and reasoning over complex relational structures. Logical relational reasoning is important for applications that may rely on LLMs to generate and query structured graphs of relations such as network infrastructure, knowledge bases, or business process schema. Our framework allows for dynamic variation of task complexity by controlling the number of objects, relations, and the minimum depth of relational chains, providing a fine-grained assessment of model performance across difficulty levels. LogiPlan encompasses three complementary tasks: (1) Plan Generation, where models must construct valid directed relational graphs meeting specified structural constraints; (2) Consistency Detection, testing models' ability to identify inconsistencies in relational structures; and (3) Comparison Question, evaluating models' capacity to determine the validity of queried relationships within a given graph. Additionally, we assess models' self-correction capabilities by prompting them to verify and refine their initial solutions. We evaluate state-of-the-art models including DeepSeek R1, Gemini 2.0 Pro, Gemini 2 Flash Thinking, GPT-4.5, GPT-4o, Llama 3.1 405B, O3-mini, O1, and Claude 3.7 Sonnet across these tasks, revealing significant performance gaps that correlate with model scale and architecture. Our analysis demonstrates that while recent reasoning-enhanced models show promising results on simpler instances, they struggle with more complex configurations requiring deeper logical planning. |
| 2025-06-12 | [Scientists' First Exam: Probing Cognitive Abilities of MLLM via Perception, Understanding, and Reasoning](http://arxiv.org/abs/2506.10521v1) | Yuhao Zhou, Yiheng Wang et al. | Scientific discoveries increasingly rely on complex multimodal reasoning based on information-intensive scientific data and domain-specific expertise. Empowered by expert-level scientific benchmarks, scientific Multimodal Large Language Models (MLLMs) hold the potential to significantly enhance this discovery process in realistic workflows. However, current scientific benchmarks mostly focus on evaluating the knowledge understanding capabilities of MLLMs, leading to an inadequate assessment of their perception and reasoning abilities. To address this gap, we present the Scientists' First Exam (SFE) benchmark, designed to evaluate the scientific cognitive capacities of MLLMs through three interconnected levels: scientific signal perception, scientific attribute understanding, scientific comparative reasoning. Specifically, SFE comprises 830 expert-verified VQA pairs across three question types, spanning 66 multimodal tasks across five high-value disciplines. Extensive experiments reveal that current state-of-the-art GPT-o3 and InternVL-3 achieve only 34.08% and 26.52% on SFE, highlighting significant room for MLLMs to improve in scientific realms. We hope the insights obtained in SFE will facilitate further developments in AI-enhanced scientific discoveries. |
| 2025-06-12 | [J-DDL: Surface Damage Detection and Localization System for Fighter Aircraft](http://arxiv.org/abs/2506.10505v1) | Jin Huang, Mingqiang Wei et al. | Ensuring the safety and extended operational life of fighter aircraft necessitates frequent and exhaustive inspections. While surface defect detection is feasible for human inspectors, manual methods face critical limitations in scalability, efficiency, and consistency due to the vast surface area, structural complexity, and operational demands of aircraft maintenance. We propose a smart surface damage detection and localization system for fighter aircraft, termed J-DDL. J-DDL integrates 2D images and 3D point clouds of the entire aircraft surface, captured using a combined system of laser scanners and cameras, to achieve precise damage detection and localization. Central to our system is a novel damage detection network built on the YOLO architecture, specifically optimized for identifying surface defects in 2D aircraft images. Key innovations include lightweight Fasternet blocks for efficient feature extraction, an optimized neck architecture incorporating Efficient Multiscale Attention (EMA) modules for superior feature aggregation, and the introduction of a novel loss function, Inner-CIOU, to enhance detection accuracy. After detecting damage in 2D images, the system maps the identified anomalies onto corresponding 3D point clouds, enabling accurate 3D localization of defects across the aircraft surface. Our J-DDL not only streamlines the inspection process but also ensures more comprehensive and detailed coverage of large and complex aircraft exteriors. To facilitate further advancements in this domain, we have developed the first publicly available dataset specifically focused on aircraft damage. Experimental evaluations validate the effectiveness of our framework, underscoring its potential to significantly advance automated aircraft inspection technologies. |
| 2025-06-12 | [Towards more efficient quantitative safety validation of residual risk for assisted and automated driving](http://arxiv.org/abs/2506.10363v1) | Daniel Betschinske, Malte Schrimpf et al. | The safety validation of Advanced Driver Assistance Systems (ADAS) and Automated Driving Systems (ADS) increasingly demands efficient and reliable methods to quantify residual risk while adhering to international standards such as ISO 21448. Traditionally, Field Operational Testing (FOT) has been pivotal for macroscopic safety validation of automotive driving functions up to SAE automation level 2. However, state-of-the-art derivations for empirical safety demonstrations using FOT often result in impractical testing efforts, particularly at higher automation levels. Even at lower automation levels, this limitation - coupled with the substantial costs associated with FOT - motivates the exploration of approaches to enhance the efficiency of FOT-based macroscopic safety validation. Therefore, this publication systematically identifies and evaluates state-of-the-art Reduction Approaches (RAs) for FOT, including novel methods reported in the literature. Based on an analysis of ISO 21448, two models are derived: a generic model capturing the argumentation components of the standard, and a base model, exemplarily applied to Automatic Emergency Braking (AEB) systems, establishing a baseline for the real-world driving requirement for a Quantitative Safety Validation of Residual Risk (QSVRR). Subsequently, the RAs are assessed using four criteria: quantifiability, threats to validity, missing links, and black box compatibility, highlighting potential benefits, inherent limitations, and identifying key areas for further research. Our evaluation reveals that, while several approaches offer potential, none are free from missing links or other substantial shortcomings. Moreover, no identified alternative can fully replace FOT, reflecting its crucial role in the safety validation of ADAS and ADS. |
| 2025-06-12 | [The Alignment Trap: Complexity Barriers](http://arxiv.org/abs/2506.10304v1) | Jasper Yao | We establish fundamental computational complexity barriers to verifying AI safety as system capabilities scale. Our main results show that for AI systems with expressiveness EXP$(m)$ above a critical threshold $\tau$, safety verification requires exponential time and is coNP-complete. We formalize the Capability-Risk Scaling (CRS) dynamic, which demonstrates how increasing AI capability drives societal safety requirements toward perfection, creating an inescapable tension with verification complexity. Through four core theorems, we prove that (1) verification complexity grows exponentially with system expressiveness, (2) safe policies comprise at most a $2^{-2^m}$ fraction of the policy space, (3) no finite set of alignment techniques can provide universal coverage, and (4) robust safety properties form measure-zero sets for neural networks. These results characterize an "intractability gap" where practical safety requirements fall within the region of computational intractability. We conclude by presenting a strategic trilemma: AI development must either constrain system complexity to maintain verifiable safety, accept unverifiable risks while scaling capabilities, or develop fundamentally new safety paradigms beyond verification. Our work provides the first systematic complexity-theoretic analysis of AI alignment and establishes rigorous bounds that any safety approach must confront. A formal verification of the core theorems in Lean4 is currently in progress. |

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



