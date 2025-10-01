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
| 2025-09-30 | [AccidentBench: Benchmarking Multimodal Understanding and Reasoning in Vehicle Accidents and Beyond](http://arxiv.org/abs/2509.26636v1) | Shangding Gu, Xiaohan Wang et al. | Rapid advances in multimodal models demand benchmarks that rigorously evaluate understanding and reasoning in safety-critical, dynamic real-world settings. We present AccidentBench, a large-scale benchmark that combines vehicle accident scenarios with Beyond domains, safety-critical settings in air and water that emphasize spatial and temporal reasoning (e.g., navigation, orientation, multi-vehicle motion). The benchmark contains approximately 2000 videos and over 19000 human-annotated question--answer pairs spanning multiple video lengths (short/medium/long) and difficulty levels (easy/medium/hard). Tasks systematically probe core capabilities: temporal, spatial, and intent understanding and reasoning. By unifying accident-centric traffic scenes with broader safety-critical scenarios in air and water, AccidentBench offers a comprehensive, physically grounded testbed for evaluating models under real-world variability. Evaluations of state-of-the-art models (e.g., Gemini-2.5 Pro and GPT-5) show that even the strongest models achieve only about 18% accuracy on the hardest tasks and longest videos, revealing substantial gaps in real-world temporal, spatial, and intent reasoning. AccidentBench is designed to expose these critical gaps and drive the development of multimodal models that are safer, more robust, and better aligned with real-world safety-critical challenges. The code and dataset are available at: https://github.com/SafeRL-Lab/AccidentBench |
| 2025-09-30 | [Robust Safety-Critical Control of Integrator Chains with Mismatched Perturbations via Linear Time-Varying Feedback](http://arxiv.org/abs/2509.26629v1) | Imtiaz Ur Rehman Moussa Labbadi, Amine Abadi et al. | In this paper, we propose a novel safety-critical control framework for a chain of integrators subject to both matched and mismatched perturbations. The core of our approach is a linear, time-varying state-feedback design that simultaneously enforces stability and safety constraints. By integrating backstepping techniques with a quadratic programming (QP) formulation, we develop a systematic procedure to guarantee safety under time-varying gains. We provide rigorous theoretical guarantees for the double integrator case, both in the presence and absence of perturbations, and outline general proofs for extending the methodology to higher-order chains of integrators. This proposed framework thus bridges robustness and safety-critical performance, while overcoming the limitations of existing prescribed-time approaches. |
| 2025-09-30 | [Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models](http://arxiv.org/abs/2509.26626v1) | Siddarth Venkatraman, Vineet Jain et al. | Test-time scaling methods improve the capabilities of large language models (LLMs) by increasing the amount of compute used during inference to make a prediction. Inference-time compute can be scaled in parallel by choosing among multiple independent solutions or sequentially through self-refinement. We propose Recursive Self-Aggregation (RSA), a test-time scaling method inspired by evolutionary methods that combines the benefits of both parallel and sequential scaling. Each step of RSA refines a population of candidate reasoning chains through aggregation of subsets to yield a population of improved solutions, which are then used as the candidate pool for the next iteration. RSA exploits the rich information embedded in the reasoning chains -- not just the final answers -- and enables bootstrapping from partially correct intermediate steps within different chains of thought. Empirically, RSA delivers substantial performance gains with increasing compute budgets across diverse tasks, model families and sizes. Notably, RSA enables Qwen3-4B-Instruct-2507 to achieve competitive performance with larger reasoning models, including DeepSeek-R1 and o3-mini (high), while outperforming purely parallel and sequential scaling strategies across AIME-25, HMMT-25, Reasoning Gym, LiveCodeBench-v6, and SuperGPQA. We further demonstrate that training the model to combine solutions via a novel aggregation-aware reinforcement learning approach yields significant performance gains. Code available at https://github.com/HyperPotatoNeo/RSA. |
| 2025-09-30 | [Uncertainty Quantification for Regression using Proper Scoring Rules](http://arxiv.org/abs/2509.26610v1) | Alexander Fishkov, Kajetan Schweighofer et al. | Quantifying uncertainty of machine learning model predictions is essential for reliable decision-making, especially in safety-critical applications. Recently, uncertainty quantification (UQ) theory has advanced significantly, building on a firm basis of learning with proper scoring rules. However, these advances were focused on classification, while extending these ideas to regression remains challenging. In this work, we introduce a unified UQ framework for regression based on proper scoring rules, such as CRPS, logarithmic, squared error, and quadratic scores. We derive closed-form expressions for the resulting uncertainty measures under practical parametric assumptions and show how to estimate them using ensembles of models. In particular, the derived uncertainty measures naturally decompose into aleatoric and epistemic components. The framework recovers popular regression UQ measures based on predictive variance and differential entropy. Our broad evaluation on synthetic and real-world regression datasets provides guidance for selecting reliable UQ measures. |
| 2025-09-30 | [Neural Network-based Co-design of Output-Feedback Control Barrier Function and Observer](http://arxiv.org/abs/2509.26597v1) | Vaishnavi Jagabathula, Ahan Basu et al. | Control Barrier Functions (CBFs) provide a powerful framework for ensuring safety in dynamical systems. However, their application typically relies on full state information, which is often violated in real-world scenarios due to the availability of partial state information. In this work, we propose a neural network-based framework for the co-design of a safety controller, observer, and CBF for partially observed continuous-time systems. By formulating barrier conditions over an augmented state space, our approach ensures safety without requiring bounded estimation errors or handcrafted barrier functions. All components are jointly trained by formulating appropriate loss functions, and we introduce a validity condition to provide formal safety guarantees beyond the training data. Finally, we demonstrate the effectiveness of the proposed approach through several case studies. |
| 2025-09-30 | [Exploring Large Language Model as an Interactive Sports Coach: Lessons from a Single-Subject Half Marathon Preparation](http://arxiv.org/abs/2509.26593v1) | Kichang Lee | Large language models (LLMs) are emerging as everyday assistants, but their role as longitudinal virtual coaches is underexplored. This two-month single subject case study documents LLM guided half marathon preparation (July-September 2025). Using text based interactions and consumer app logs, the LLM acted as planner, explainer, and occasional motivator. Performance improved from sustaining 2 km at 7min 54sec per km to completing 21.1 km at 6min 30sec per km, with gains in cadence, pace HR coupling, and efficiency index trends. While causal attribution is limited without a control, outcomes demonstrate safe, measurable progress. At the same time, gaps were evident, no realtime sensor integration, text only feedback, motivation support that was user initiated, and limited personalization or safety guardrails. We propose design requirements for next generation systems, persistent athlete models with explicit guardrails, multimodal on device sensing, audio, haptic, visual feedback, proactive motivation scaffolds, and privacy-preserving personalization. This study offers grounded evidence and a design agenda for evolving LLMs from retrospective advisors to closed-loop coaching companions. |
| 2025-09-30 | [OffTopicEval: When Large Language Models Enter the Wrong Chat, Almost Always!](http://arxiv.org/abs/2509.26495v1) | Jingdi Lei, Varun Gumma et al. | Large Language Model (LLM) safety is one of the most pressing challenges for enabling wide-scale deployment. While most studies and global discussions focus on generic harms, such as models assisting users in harming themselves or others, enterprises face a more fundamental concern: whether LLM-based agents are safe for their intended use case. To address this, we introduce operational safety, defined as an LLM's ability to appropriately accept or refuse user queries when tasked with a specific purpose. We further propose OffTopicEval, an evaluation suite and benchmark for measuring operational safety both in general and within specific agentic use cases. Our evaluations on six model families comprising 20 open-weight LLMs reveal that while performance varies across models, all of them remain highly operationally unsafe. Even the strongest models -- Qwen-3 (235B) with 77.77\% and Mistral (24B) with 79.96\% -- fall far short of reliable operational safety, while GPT models plateau in the 62--73\% range, Phi achieves only mid-level scores (48--70\%), and Gemma and Llama-3 collapse to 39.53\% and 23.84\%, respectively. While operational safety is a core model alignment issue, to suppress these failures, we propose prompt-based steering methods: query grounding (Q-ground) and system-prompt grounding (P-ground), which substantially improve OOD refusal. Q-ground provides consistent gains of up to 23\%, while P-ground delivers even larger boosts, raising Llama-3.3 (70B) by 41\% and Qwen-3 (30B) by 27\%. These results highlight both the urgent need for operational safety interventions and the promise of prompt-based steering as a first step toward more reliable LLM-based agents. |
| 2025-09-30 | [STaR-Attack: A Spatio-Temporal and Narrative Reasoning Attack Framework for Unified Multimodal Understanding and Generation Models](http://arxiv.org/abs/2509.26473v1) | Shaoxiong Guo, Tianyi Du et al. | Unified Multimodal understanding and generation Models (UMMs) have demonstrated remarkable capabilities in both understanding and generation tasks. However, we identify a vulnerability arising from the generation-understanding coupling in UMMs. The attackers can use the generative function to craft an information-rich adversarial image and then leverage the understanding function to absorb it in a single pass, which we call Cross-Modal Generative Injection (CMGI). Current attack methods on malicious instructions are often limited to a single modality while also relying on prompt rewriting with semantic drift, leaving the unique vulnerabilities of UMMs unexplored. We propose STaR-Attack, the first multi-turn jailbreak attack framework that exploits unique safety weaknesses of UMMs without semantic drift. Specifically, our method defines a malicious event that is strongly correlated with the target query within a spatio-temporal context. Using the three-act narrative theory, STaR-Attack generates the pre-event and the post-event scenes while concealing the malicious event as the hidden climax. When executing the attack strategy, the opening two rounds exploit the UMM's generative ability to produce images for these scenes. Subsequently, an image-based question guessing and answering game is introduced by exploiting the understanding capability. STaR-Attack embeds the original malicious question among benign candidates, forcing the model to select and answer the most relevant one given the narrative context. Extensive experiments show that STaR-Attack consistently surpasses prior approaches, achieving up to 93.06% ASR on Gemini-2.0-Flash and surpasses the strongest prior baseline, FlipAttack. Our work uncovers a critical yet underdeveloped vulnerability and highlights the need for safety alignments in UMMs. |
| 2025-09-30 | [EQ-Robin: Generating Multiple Minimal Unique-Cause MC/DC Test Suites](http://arxiv.org/abs/2509.26458v1) | Robin Lee, Youngho Nam | Modified Condition/Decision Coverage (MC/DC), particularly its strict Unique-Cause form, is a cornerstone of safety-critical software verification. A recent algorithm, "Robin's Rule," introduced a deterministic method to construct the theoretical minimum of N+1 test cases for Singular Boolean Expressions (SBEs). However, this approach yields only a single test suite, introducing a critical risk: if a test case forming a required 'independence pair' is an illegal input forbidden by system constraints, the suite fails to achieve 100% coverage. This paper proposes EQ-Robin, a lightweight pipeline that systematically generates a family of minimal Unique-Cause MC/DC suites to mitigate this risk. We introduce a method for systematically generating semantically equivalent SBEs by applying algebraic rearrangements to an Abstract Syntax Tree (AST) representation of the expression. By applying Robin's Rule to each structural variant, a diverse set of test suites can be produced. This provides a resilient path to discovering a valid test suite that preserves the N+1 minimality guarantee while navigating real-world constraints. We outline an evaluation plan on TCAS-II-derived SBEs to demonstrate how EQ-Robin offers a practical solution for ensuring robust MC/DC coverage. |
| 2025-09-30 | [Introducing Large Language Models in the Design Flow of Time Sensitive Networking](http://arxiv.org/abs/2509.26368v1) | Rubi Debnath, Luxi Zhao et al. | The growing demand for real-time, safety-critical systems has significantly increased both the adoption and complexity of Time Sensitive Networking (TSN). Configuring an optimized TSN network is highly challenging, requiring careful planning, design, verification, validation, and deployment. Large Language Models (LLMs) have recently demonstrated strong capabilities in solving complex tasks, positioning them as promising candidates for automating end-to-end TSN deployment, referred to as TSN orchestration. This paper outlines the steps involved in TSN orchestration and the associated challenges. To assess the capabilities of existing LLM models, we conduct an initial proof-of-concept case study focused on TSN configuration across multiple models. Building on these insights, we propose an LLM-assisted orchestration framework. Unlike prior research on LLMs in computer networks, which has concentrated on general configuration and management, TSN-specific orchestration has not yet been investigated. We present the building blocks for automating TSN using LLMs, describe the proposed pipeline, and analyze opportunities and limitations for real-world deployment. Finally, we highlight key challenges and research directions, including the development of TSN-focused datasets, standardized benchmark suites, and the integration of external tools such as Network Calculus (NC) engines and simulators. This work provides the first roadmap toward assessing the feasibility of LLM-assisted TSN orchestration. |
| 2025-09-30 | [Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](http://arxiv.org/abs/2509.26354v1) | Shuai Shao, Qihan Ren et al. | Advances in Large Language Models (LLMs) have enabled a new class of self-evolving agents that autonomously improve through interaction with the environment, demonstrating strong capabilities. However, self-evolution also introduces novel risks overlooked by current safety research. In this work, we study the case where an agent's self-evolution deviates in unintended ways, leading to undesirable or even harmful outcomes. We refer to this as Misevolution. To provide a systematic investigation, we evaluate misevolution along four key evolutionary pathways: model, memory, tool, and workflow. Our empirical findings reveal that misevolution is a widespread risk, affecting agents built even on top-tier LLMs (e.g., Gemini-2.5-Pro). Different emergent risks are observed in the self-evolutionary process, such as the degradation of safety alignment after memory accumulation, or the unintended introduction of vulnerabilities in tool creation and reuse. To our knowledge, this is the first study to systematically conceptualize misevolution and provide empirical evidence of its occurrence, highlighting an urgent need for new safety paradigms for self-evolving agents. Finally, we discuss potential mitigation strategies to inspire further research on building safer and more trustworthy self-evolving agents. Our code and data are available at https://github.com/ShaoShuai0605/Misevolution . Warning: this paper includes examples that may be offensive or harmful in nature. |
| 2025-09-30 | [SafeBehavior: Simulating Human-Like Multistage Reasoning to Mitigate Jailbreak Attacks in Large Language Models](http://arxiv.org/abs/2509.26345v1) | Qinjian Zhao, Jiaqi Wang et al. | Large Language Models (LLMs) have achieved impressive performance across diverse natural language processing tasks, but their growing power also amplifies potential risks such as jailbreak attacks that circumvent built-in safety mechanisms. Existing defenses including input paraphrasing, multi step evaluation, and safety expert models often suffer from high computational costs, limited generalization, or rigid workflows that fail to detect subtle malicious intent embedded in complex contexts. Inspired by cognitive science findings on human decision making, we propose SafeBehavior, a novel hierarchical jailbreak defense mechanism that simulates the adaptive multistage reasoning process of humans. SafeBehavior decomposes safety evaluation into three stages: intention inference to detect obvious input risks, self introspection to assess generated responses and assign confidence based judgments, and self revision to adaptively rewrite uncertain outputs while preserving user intent and enforcing safety constraints. We extensively evaluate SafeBehavior against five representative jailbreak attack types including optimization based, contextual manipulation, and prompt based attacks and compare it with seven state of the art defense baselines. Experimental results show that SafeBehavior significantly improves robustness and adaptability across diverse threat scenarios, offering an efficient and human inspired approach to safeguarding LLMs against jailbreak attempts. |
| 2025-09-30 | [PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](http://arxiv.org/abs/2509.26272v1) | Tuan Nguyen, Naseem Khan et al. | The rapid rise of synthetic media has made deepfake detection a critical challenge for online safety and trust. Progress remains constrained by the scarcity of large, high-quality datasets. Although multimodal large language models (LLMs) exhibit strong reasoning capabilities, their performance on deepfake detection is poor, often producing explanations that are misaligned with visual evidence or hallucinatory. To address this limitation, we introduce a reasoning-annotated dataset for deepfake detection and propose Paragraph-level Relative Policy Optimization (PRPO), a reinforcement learning algorithm that aligns LLM reasoning with image content at the paragraph level. Experiments show that PRPO improves detection accuracy by a wide margin and achieves the highest reasoning score of 4.55/5.0. Ablation studies further demonstrate that PRPO significantly outperforms GRPO under test-time conditions. These results underscore the importance of grounding multimodal reasoning in visual evidence to enable more reliable and interpretable deepfake detection. |
| 2025-09-30 | [Sandbagging in a Simple Survival Bandit Problem](http://arxiv.org/abs/2509.26239v1) | Joel Dyer, Daniel Jarne Ornia et al. | Evaluating the safety of frontier AI systems is an increasingly important concern, helping to measure the capabilities of such models and identify risks before deployment. However, it has been recognised that if AI agents are aware that they are being evaluated, such agents may deliberately hide dangerous capabilities or intentionally demonstrate suboptimal performance in safety-related tasks in order to be released and to avoid being deactivated or retrained. Such strategic deception - often known as "sandbagging" - threatens to undermine the integrity of safety evaluations. For this reason, it is of value to identify methods that enable us to distinguish behavioural patterns that demonstrate a true lack of capability from behavioural patterns that are consistent with sandbagging. In this paper, we develop a simple model of strategic deception in sequential decision-making tasks, inspired by the recently developed survival bandit framework. We demonstrate theoretically that this problem induces sandbagging behaviour in optimal rational agents, and construct a statistical test to distinguish between sandbagging and incompetence from sequences of test scores. In simulation experiments, we investigate the reliability of this test in allowing us to distinguish between such behaviours in bandit models. This work aims to establish a potential avenue for developing robust statistical procedures for use in the science of frontier model evaluations. |
| 2025-09-30 | [Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](http://arxiv.org/abs/2509.26238v1) | James Oldfield, Philip Torr et al. | Monitoring large language models' (LLMs) activations is an effective way to detect harmful requests before they lead to unsafe outputs. However, traditional safety monitors often require the same amount of compute for every query. This creates a trade-off: expensive monitors waste resources on easy inputs, while cheap ones risk missing subtle cases. We argue that safety monitors should be flexible--costs should rise only when inputs are difficult to assess, or when more compute is available. To achieve this, we introduce Truncated Polynomial Classifiers (TPCs), a natural extension of linear probes for dynamic activation monitoring. Our key insight is that polynomials can be trained and evaluated progressively, term-by-term. At test-time, one can early-stop for lightweight monitoring, or use more terms for stronger guardrails when needed. TPCs provide two modes of use. First, as a safety dial: by evaluating more terms, developers and regulators can "buy" stronger guardrails from the same model. Second, as an adaptive cascade: clear cases exit early after low-order checks, and higher-order guardrails are evaluated only for ambiguous inputs, reducing overall monitoring costs. On two large-scale safety datasets (WildGuardMix and BeaverTails), for 4 models with up to 30B parameters, we show that TPCs compete with or outperform MLP-based probe baselines of the same size, all the while being more interpretable than their black-box counterparts. Our code is available at http://github.com/james-oldfield/tpc. |
| 2025-09-30 | [Machine Learning Detection of Lithium Plating in Lithium-ion Cells: A Gaussian Process Approach](http://arxiv.org/abs/2509.26234v1) | Ayush Patnaik, Adam B Zufall et al. | Lithium plating during fast charging is a critical degradation mechanism that accelerates capacity fade and can trigger catastrophic safety failures. Recent work has identified a distinctive dQ/dV peak above 4.0 V as a reliable signature of plating onset; however, conventional methods for computing dQ/dV rely on finite differencing with filtering, which amplifies sensor noise and introduces bias in peak location. In this paper, we propose a Gaussian Process (GP) framework for lithium plating detection by directly modeling the charge-voltage relationship Q(V) as a stochastic process with calibrated uncertainty. Leveraging the property that derivatives of GPs remain GPs, we infer dQ/dV analytically and probabilistically from the posterior, enabling robust detection without ad hoc smoothing. The framework provides three key benefits: (i) noise-aware inference with hyperparameters learned from data, (ii) closed-form derivatives with credible intervals for uncertainty quantification, and (iii) scalability to online variants suitable for embedded BMS. Experimental validation on Li-ion coin cells across a range of C-rates (0.2C-1C) and temperatures (0-40\deg C) demonstrates that the GP-based method reliably detects plating peaks under low-temperature, high-rate charging, while correctly reporting no peaks in baseline cases. The concurrence of GP-identified differential peaks, reduced charge throughput, and capacity fade measured via reference performance tests confirms the method's accuracy and robustness, establishing a practical pathway for real-time lithium plating detection. |
| 2025-09-30 | [Beyond Overall Accuracy: Pose- and Occlusion-driven Fairness Analysis in Pedestrian Detection for Autonomous Driving](http://arxiv.org/abs/2509.26166v1) | Mohammad Khoshkdahan, Arman Akbari et al. | Pedestrian detection plays a critical role in autonomous driving (AD), where ensuring safety and reliability is important. While many detection models aim to reduce miss-rates and handle challenges such as occlusion and long-range recognition, fairness remains an underexplored yet equally important concern. In this work, we systematically investigate how variations in the pedestrian pose--including leg status, elbow status, and body orientation--as well as individual joint occlusions, affect detection performance. We evaluate five pedestrian-specific detectors (F2DNet, MGAN, ALFNet, CSP, and Cascade R-CNN) alongside three general-purpose models (YOLOv12 variants) on the EuroCity Persons Dense Pose (ECP-DP) dataset. Fairness is quantified using the Equal Opportunity Difference (EOD) metric across various confidence thresholds. To assess statistical significance and robustness, we apply the Z-test. Our findings highlight biases against pedestrians with parallel legs, straight elbows, and lateral views. Occlusion of lower body joints has a more negative impact on the detection rate compared to the upper body and head. Cascade R-CNN achieves the lowest overall miss-rate and exhibits the smallest bias across all attributes. To the best of our knowledge, this is the first comprehensive pose- and occlusion-aware fairness evaluation in pedestrian detection for AD. |
| 2025-09-30 | [Testing the nature of GW200105 by probing the frequency evolution of eccentricity](http://arxiv.org/abs/2509.26152v1) | Avinash Tiwari, Sajad A. Bhat et al. | GW200105 is a compact binary coalescence (CBC) event, consisting of a neutron star and a black hole, observed in LIGO-Virgo-KAGRA's (LVK's) third observing run (O3). Recent reanalyses of the event using state-of-the-art waveform models have claimed observation of signatures of an eccentric orbit. It has nevertheless been pointed out in the literature that certain physical or modified gravity effects could mimic eccentricity by producing a spurious non-zero eccentricity value, at a given reference frequency, when recovered with an eccentric waveform model. We recently developed a model-independent Eccentricity Evolution Consistency Test (EECT, S. A. Bhat et al. 2025) to identify such mimickers, by comparing the measured frequency $\textit{evolution}$ of eccentricity, $e(f)$, with that expected from General Relativity (GR). In this $\textit{Letter}$, we apply EECT to GW200105 and find that it satisfies EECT within 68% confidence. Our analysis therefore lends complementary support in favour of the eccentricity hypothesis, while also providing a novel test of the consistency of $e(f)$ with GR. |
| 2025-09-30 | [MEDAKA: Construction of Biomedical Knowledge Graphs Using Large Language Models](http://arxiv.org/abs/2509.26128v1) | Asmita Sengupta, David Antony Selby et al. | Knowledge graphs (KGs) are increasingly used to represent biomedical information in structured, interpretable formats. However, existing biomedical KGs often focus narrowly on molecular interactions or adverse events, overlooking the rich data found in drug leaflets. In this work, we present (1) a hackable, end-to-end pipeline to create KGs from unstructured online content using a web scraper and an LLM; and (2) a curated dataset, MEDAKA, generated by applying this method to publicly available drug leaflets. The dataset captures clinically relevant attributes such as side effects, warnings, contraindications, ingredients, dosage guidelines, storage instructions and physical characteristics. We evaluate it through manual inspection and with an LLM-as-a-Judge framework, and compare its coverage with existing biomedical KGs and databases. We expect MEDAKA to support tasks such as patient safety monitoring and drug recommendation. The pipeline can also be used for constructing KGs from unstructured texts in other domains. Code and dataset are available at https://github.com/medakakg/medaka. |
| 2025-09-30 | [Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics](http://arxiv.org/abs/2509.26106v1) | Nakhul Kalaivanan, Senthil Arumugam Muthukumaraswamy et al. | This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety. |

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



