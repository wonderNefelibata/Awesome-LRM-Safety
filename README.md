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
| 2025-11-06 | [GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction](http://arxiv.org/abs/2511.04679v1) | Qingzhou Lu, Yao Feng et al. | Humanoid robots are expected to operate in human-centered environments where safe and natural physical interaction is essential. However, most recent reinforcement learning (RL) policies emphasize rigid tracking and suppress external forces. Existing impedance-augmented approaches are typically restricted to base or end-effector control and focus on resisting extreme forces rather than enabling compliance. We introduce GentleHumanoid, a framework that integrates impedance control into a whole-body motion tracking policy to achieve upper-body compliance. At its core is a unified spring-based formulation that models both resistive contacts (restoring forces when pressing against surfaces) and guiding contacts (pushes or pulls sampled from human motion data). This formulation ensures kinematically consistent forces across the shoulder, elbow, and wrist, while exposing the policy to diverse interaction scenarios. Safety is further supported through task-adjustable force thresholds. We evaluate our approach in both simulation and on the Unitree G1 humanoid across tasks requiring different levels of compliance, including gentle hugging, sit-to-stand assistance, and safe object manipulation. Compared to baselines, our policy consistently reduces peak contact forces while maintaining task success, resulting in smoother and more natural interactions. These results highlight a step toward humanoid robots that can safely and effectively collaborate with humans and handle objects in real-world environments. |
| 2025-11-06 | [Nonparametric Safety Stock Dimensioning: A Data-Driven Approach for Supply Chains of Hardware OEMs](http://arxiv.org/abs/2511.04616v1) | Elvis Agbenyega, Cody Quick | Resilient supply chains are critical, especially for Original Equipment Manufacturers (OEMs) that power today's digital economy. Safety Stock dimensioning-the computation of the appropriate safety stock quantity-is one of several mechanisms to ensure supply chain resiliency, as it protects the supply chain against demand and supply uncertainties. Unfortunately, the major approaches to dimensioning safety stock heavily assume that demand is normally distributed and ignore future demand variability, limiting their applicability in manufacturing contexts where demand is non-normal, intermittent, and highly skewed. In this paper, we propose a data-driven approach that relaxes the assumption of normality, enabling the demand distribution of each inventory item to be analytically determined using Kernel Density Estimation. Also, we extended the analysis from historical demand variability to forecasted demand variability. We evaluated the proposed approach against a normal distribution model in a near-world inventory replenishment simulation. Afterwards, we used a linear optimization model to determine the optimal safety stock configuration. The results from the simulation and linear optimization models showed that the data-driven approach outperformed traditional approaches. In particular, the data-driven approach achieved the desired service levels at lower safety stock levels than the conventional approaches. |
| 2025-11-06 | [From Model to Breach: Towards Actionable LLM-Generated Vulnerabilities Reporting](http://arxiv.org/abs/2511.04538v1) | Cyril Vallez, Alexander Sternfeld et al. | As the role of Large Language Models (LLM)-based coding assistants in software development becomes more critical, so does the role of the bugs they generate in the overall cybersecurity landscape. While a number of LLM code security benchmarks have been proposed alongside approaches to improve the security of generated code, it remains unclear to what extent they have impacted widely used coding LLMs. Here, we show that even the latest open-weight models are vulnerable in the earliest reported vulnerability scenarios in a realistic use setting, suggesting that the safety-functionality trade-off has until now prevented effective patching of vulnerabilities. To help address this issue, we introduce a new severity metric that reflects the risk posed by an LLM-generated vulnerability, accounting for vulnerability severity, generation chance, and the formulation of the prompt that induces vulnerable code generation - Prompt Exposure (PE). To encourage the mitigation of the most serious and prevalent vulnerabilities, we use PE to define the Model Exposure (ME) score, which indicates the severity and prevalence of vulnerabilities a model generates. |
| 2025-11-06 | [Self-mixing-based photoacoustic sensing](http://arxiv.org/abs/2511.04532v1) | Tecla Gabbrielli, Jacopo Pelini et al. | Versatile, ultracompact, easy-to-handle, high-sensitivity sensors are compelling tools for in situ pivotal applications, such as medical diagnostics, security and safety assessments, and environmental control. In this work, we combine photoacoustic spectroscopy and feedback interferometry, proposing a novel trace-gas sensor equipped with a self-mixing readout. This scheme demonstrates a readout sensitivity comparable to that of bulkier state-of-the-art balanced Michelson-interferometric schemes, achieving the same spectroscopic performance in terms of signal-to-noise ratio (SNR) and minimum detection limit (MDL). At the same time, the self-mixing readout benefits from a reduced size and a lower baseline, paving the way for future system downsizing and integration while offering a higher detectability for lower gas concentrations. Moreover, the intrinsic wavelength independence of both self-mixing and photoacoustic techniques allows the applicability and tailorability of the sensor to any desired spectral range. |
| 2025-11-06 | [RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG](http://arxiv.org/abs/2511.04502v1) | Joshua Gao, Quoc Huy Pham et al. | Retrieval-Augmented Generation (RAG) is a critical technique for grounding Large Language Models (LLMs) in factual evidence, yet evaluating RAG systems in specialized, safety-critical domains remains a significant challenge. Existing evaluation frameworks often rely on heuristic-based metrics that fail to capture domain-specific nuances and other works utilize LLM-as-a-Judge approaches that lack validated alignment with human judgment. This paper introduces RAGalyst, an automated, human-aligned agentic framework designed for the rigorous evaluation of domain-specific RAG systems. RAGalyst features an agentic pipeline that generates high-quality, synthetic question-answering (QA) datasets from source documents, incorporating an agentic filtering step to ensure data fidelity. The framework refines two key LLM-as-a-Judge metrics-Answer Correctness and Answerability-using prompt optimization to achieve a strong correlation with human annotations. Applying this framework to evaluate various RAG components across three distinct domains (military operations, cybersecurity, and bridge engineering), we find that performance is highly context-dependent. No single embedding model, LLM, or hyperparameter configuration proves universally optimal. Additionally, we provide an analysis on the most common low Answer Correctness reasons in RAG. These findings highlight the necessity of a systematic evaluation framework like RAGalyst, which empowers practitioners to uncover domain-specific trade-offs and make informed design choices for building reliable and effective RAG systems. RAGalyst is available on our Github. |
| 2025-11-06 | [If I Could Turn Back Time: Temporal Reframing as a Historical Reasoning Task for LLMs](http://arxiv.org/abs/2511.04432v1) | Lars Bungum, Charles Yijia Huang et al. | In this study, we experiment with the ability of LLMs to do temporal reasoning. Using a Norwegian book from 1940 containing trivia questions, we prompt the LLMs to answer the questions as if it were 1940. We also pose the questions in both English and Norwegian. Correct answers are often presented as sentences, and grading is done by means of LLM-as-judge, with sampled checks by a native speaker. Prompting in English consistently gave better results than in Norwegian, an unexpected result. In contrast, using larger LLMs improved results. We tested the DeepSeek-R1, Gemma3, Qwen3, and Llama3.1 model families, and also the largest available LLM especially crafted for Norwegian. |
| 2025-11-06 | [RxSafeBench: Identifying Medication Safety Issues of Large Language Models in Simulated Consultation](http://arxiv.org/abs/2511.04328v1) | Jiahao Zhao, Luxin Xu et al. | Numerous medical systems powered by Large Language Models (LLMs) have achieved remarkable progress in diverse healthcare tasks. However, research on their medication safety remains limited due to the lack of real world datasets, constrained by privacy and accessibility issues. Moreover, evaluation of LLMs in realistic clinical consultation settings, particularly regarding medication safety, is still underexplored. To address these gaps, we propose a framework that simulates and evaluates clinical consultations to systematically assess the medication safety capabilities of LLMs. Within this framework, we generate inquiry diagnosis dialogues with embedded medication risks and construct a dedicated medication safety database, RxRisk DB, containing 6,725 contraindications, 28,781 drug interactions, and 14,906 indication-drug pairs. A two-stage filtering strategy ensures clinical realism and professional quality, resulting in the benchmark RxSafeBench with 2,443 high-quality consultation scenarios. We evaluate leading open-source and proprietary LLMs using structured multiple choice questions that test their ability to recommend safe medications under simulated patient contexts. Results show that current LLMs struggle to integrate contraindication and interaction knowledge, especially when risks are implied rather than explicit. Our findings highlight key challenges in ensuring medication safety in LLM-based systems and provide insights into improving reliability through better prompting and task-specific tuning. RxSafeBench offers the first comprehensive benchmark for evaluating medication safety in LLMs, advancing safer and more trustworthy AI-driven clinical decision support. |
| 2025-11-06 | [AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research](http://arxiv.org/abs/2511.04316v1) | Tim Beyer, Jonas Dornbusch et al. | The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety. |
| 2025-11-06 | [A Tool for Benchmarking Large Language Models' Robustness in Assessing the Realism of Driving Scenarios](http://arxiv.org/abs/2511.04267v1) | Jiahui Wu, Chengjie Lu et al. | In recent years, autonomous driving systems have made significant progress, yet ensuring their safety remains a key challenge. To this end, scenario-based testing offers a practical solution, and simulation-based methods have gained traction due to the high cost and risk of real-world testing. However, evaluating the realism of simulated scenarios remains difficult, creating demand for effective assessment methods. Recent advances show that Large Language Models (LLMs) possess strong reasoning and generalization capabilities, suggesting their potential in assessing scenario realism through scenario-related textual prompts. Motivated by this, we propose DriveRLR, a benchmark tool to assess the robustness of LLMs in evaluating the realism of driving scenarios. DriveRLR generates mutated scenario variants, constructs prompts, which are then used to assess a given LLM's ability and robustness in determining the realism of driving scenarios. We validate DriveRLR on the DeepScenario dataset using three state-of-the-art LLMs: GPT-5, Llama 4 Maverick, and Mistral Small 3.2. Results show that DriveRLR effectively reveals differences in the robustness of various LLMs, demonstrating its effectiveness and practical value in scenario realism assessment. Beyond LLM robustness evaluation, DriveRLR can serve as a practical component in applications such as an objective function to guide scenario generation, supporting simulation-based ADS testing workflows. |
| 2025-11-06 | [Guided by Stars: Interpretable Concept Learning Over Time Series via Temporal Logic Semantics](http://arxiv.org/abs/2511.04244v1) | Irene Ferfoglia, Simone Silvetti et al. | Time series classification is a task of paramount importance, as this kind of data often arises in safety-critical applications. However, it is typically tackled with black-box deep learning methods, making it hard for humans to understand the rationale behind their output. To take on this challenge, we propose a novel approach, STELLE (Signal Temporal logic Embedding for Logically-grounded Learning and Explanation), a neuro-symbolic framework that unifies classification and explanation through direct embedding of trajectories into a space of temporal logic concepts. By introducing a novel STL-inspired kernel that maps raw time series to their alignment with predefined STL formulae, our model jointly optimises accuracy and interpretability, as each prediction is accompanied by the most relevant logical concepts that characterise it. This yields (i) local explanations as human-readable STL conditions justifying individual predictions, and (ii) global explanations as class-characterising formulae. Experiments demonstrate that STELLE achieves competitive accuracy while providing logically faithful explanations, validated on diverse real-world benchmarks. |
| 2025-11-06 | [REMIND: Input Loss Landscapes Reveal Residual Memorization in Post-Unlearning LLMs](http://arxiv.org/abs/2511.04228v1) | Liran Cohen, Yaniv Nemcovesky et al. | Machine unlearning aims to remove the influence of specific training data from a model without requiring full retraining. This capability is crucial for ensuring privacy, safety, and regulatory compliance. Therefore, verifying whether a model has truly forgotten target data is essential for maintaining reliability and trustworthiness. However, existing evaluation methods often assess forgetting at the level of individual inputs. This approach may overlook residual influence present in semantically similar examples. Such influence can compromise privacy and lead to indirect information leakage. We propose REMIND (Residual Memorization In Neighborhood Dynamics), a novel evaluation method aiming to detect the subtle remaining influence of unlearned data and classify whether the data has been effectively forgotten. REMIND analyzes the model's loss over small input variations and reveals patterns unnoticed by single-point evaluations. We show that unlearned data yield flatter, less steep loss landscapes, while retained or unrelated data exhibit sharper, more volatile patterns. REMIND requires only query-based access, outperforms existing methods under similar constraints, and demonstrates robustness across different models, datasets, and paraphrased inputs, making it practical for real-world deployment. By providing a more sensitive and interpretable measure of unlearning effectiveness, REMIND provides a reliable framework to assess unlearning in language models. As a result, REMIND offers a novel perspective on memorization and unlearning. |
| 2025-11-06 | [Black-Box Guardrail Reverse-engineering Attack](http://arxiv.org/abs/2511.04215v1) | Hongwei Yao, Yun Xia et al. | Large language models (LLMs) increasingly employ guardrails to enforce ethical, legal, and application-specific constraints on their outputs. While effective at mitigating harmful responses, these guardrails introduce a new class of vulnerabilities by exposing observable decision patterns. In this work, we present the first study of black-box LLM guardrail reverse-engineering attacks. We propose Guardrail Reverse-engineering Attack (GRA), a reinforcement learning-based framework that leverages genetic algorithm-driven data augmentation to approximate the decision-making policy of victim guardrails. By iteratively collecting input-output pairs, prioritizing divergence cases, and applying targeted mutations and crossovers, our method incrementally converges toward a high-fidelity surrogate of the victim guardrail. We evaluate GRA on three widely deployed commercial systems, namely ChatGPT, DeepSeek, and Qwen3, and demonstrate that it achieves an rule matching rate exceeding 0.92 while requiring less than $85 in API costs. These findings underscore the practical feasibility of guardrail extraction and highlight significant security risks for current LLM safety mechanisms. Our findings expose critical vulnerabilities in current guardrail designs and highlight the urgent need for more robust defense mechanisms in LLM deployment. |
| 2025-11-06 | [Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment](http://arxiv.org/abs/2511.04157v1) | Asma Yamani, Malak Baslyman et al. | Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development. |
| 2025-11-06 | [BAPPA: Benchmarking Agents, Plans, and Pipelines for Automated Text-to-SQL Generation](http://arxiv.org/abs/2511.04153v1) | Fahim Ahmed, Md Mubtasim Ahasan et al. | Text-to-SQL systems provide a natural language interface that can enable even laymen to access information stored in databases. However, existing Large Language Models (LLM) struggle with SQL generation from natural instructions due to large schema sizes and complex reasoning. Prior work often focuses on complex, somewhat impractical pipelines using flagship models, while smaller, efficient models remain overlooked. In this work, we explore three multi-agent LLM pipelines, with systematic performance benchmarking across a range of small to large open-source models: (1) Multi-agent discussion pipeline, where agents iteratively critique and refine SQL queries, and a judge synthesizes the final answer; (2) Planner-Coder pipeline, where a thinking model planner generates stepwise SQL generation plans and a coder synthesizes queries; and (3) Coder-Aggregator pipeline, where multiple coders independently generate SQL queries, and a reasoning agent selects the best query. Experiments on the Bird-Bench Mini-Dev set reveal that Multi-Agent discussion can improve small model performance, with up to 10.6% increase in Execution Accuracy for Qwen2.5-7b-Instruct seen after three rounds of discussion. Among the pipelines, the LLM Reasoner-Coder pipeline yields the best results, with DeepSeek-R1-32B and QwQ-32B planners boosting Gemma 3 27B IT accuracy from 52.4% to the highest score of 56.4%. Codes are available at https://github.com/treeDweller98/bappa-sql. |
| 2025-11-06 | [Experimental Observation of Hidden Multistability in Nonlinear Systems](http://arxiv.org/abs/2511.04150v1) | Kun Zhang, Qicheng Zhang et al. | Multistability, the coexistence of multiple stable states, is a cornerstone of nonlinear dynamical systems, governing their equilibrium, tunability, and emergent complexity. Recently, the concept of hidden multistability, where certain stable states evade detection via conventional continuous parameter sweeping, has garnered increasing attention due to its elusive nature and promising applications. In this Letter, we present the first experimental observation of hidden multistability using a programmable acoustic coupled-cavity platform that integrates competing self-focusing and self-defocusing Kerr nonlinearities. Beyond established bistability, we demonstrate semi- and fully-hidden tristabilities by precisely programming system parameters. Crucially, the hidden stable states, typically inaccessible via the traditional protocol, are unambiguously revealed and dynamically controlled through pulsed excitation, enabling flexible transitions between distinct types of stable states. These experimental findings not only offer new insights into the fundamental physics of emerging hidden multistability, but also unlock new avenues for applications in information storage, information encryption, and safety precaution, where multi-state dynamics could enable advanced control techniques. |
| 2025-11-06 | [Exchange Policy Optimization Algorithm for Semi-Infinite Safe Reinforcement Learning](http://arxiv.org/abs/2511.04147v1) | Jiaming Zhang, Yujie Yang et al. | Safe reinforcement learning (safe RL) aims to respect safety requirements while optimizing long-term performance. In many practical applications, however, the problem involves an infinite number of constraints, known as semi-infinite safe RL (SI-safe RL). Such constraints typically appear when safety conditions must be enforced across an entire continuous parameter space, such as ensuring adequate resource distribution at every spatial location. In this paper, we propose exchange policy optimization (EPO), an algorithmic framework that achieves optimal policy performance and deterministic bounded safety. EPO works by iteratively solving safe RL subproblems with finite constraint sets and adaptively adjusting the active set through constraint expansion and deletion. At each iteration, constraints with violations exceeding the predefined tolerance are added to refine the policy, while those with zero Lagrange multipliers are removed after the policy update. This exchange rule prevents uncontrolled growth of the working set and supports effective policy training. Our theoretical analysis demonstrates that, under mild assumptions, strategies trained via EPO achieve performance comparable to optimal solutions with global constraint violations strictly remaining within a prescribed bound. |
| 2025-11-06 | [Integrating Ergonomics and Manipulability for Upper Limb Postural Optimization in Bimanual Human-Robot Collaboration](http://arxiv.org/abs/2511.04009v1) | Chenzui Li, Yiming Chen et al. | This paper introduces an upper limb postural optimization method for enhancing physical ergonomics and force manipulability during bimanual human-robot co-carrying tasks. Existing research typically emphasizes human safety or manipulative efficiency, whereas our proposed method uniquely integrates both aspects to strengthen collaboration across diverse conditions (e.g., different grasping postures of humans, and different shapes of objects). Specifically, the joint angles of a simplified human skeleton model are optimized by minimizing the cost function to prioritize safety and manipulative capability. To guide humans towards the optimized posture, the reference end-effector poses of the robot are generated through a transformation module. A bimanual model predictive impedance controller (MPIC) is proposed for our human-like robot, CURI, to recalibrate the end effector poses through planned trajectories. The proposed method has been validated through various subjects and objects during human-human collaboration (HHC) and human-robot collaboration (HRC). The experimental results demonstrate significant improvement in muscle conditions by comparing the activation of target muscles before and after optimization. |
| 2025-11-06 | [Design and Detection of Covert Man-in-the-Middle Cyberattacks on Water Treatment Plants](http://arxiv.org/abs/2511.03971v1) | Victor Mattos, João Henrique Schmidt et al. | Cyberattacks targeting critical infrastructures, such as water treatment facilities, represent significant threats to public health, safety, and the environment. This paper introduces a systematic approach for modeling and assessing covert man-in-the-middle (MitM) attacks that leverage system identification techniques to inform the attack design. We focus on the attacker's ability to deploy a covert controller, and we evaluate countermeasures based on the Process-Aware Stealthy Attack Detection (PASAD) anomaly detection method. Using a second-order linear time-invariant with time delay model, representative of water treatment dynamics, we design and simulate stealthy attacks. Our results highlight how factors such as system noise and inaccuracies in the attacker's plant model influence the attack's stealthiness, underscoring the need for more robust detection strategies in industrial control environments. |
| 2025-11-05 | [I Detect What I Don't Know: Incremental Anomaly Learning with Stochastic Weight Averaging-Gaussian for Oracle-Free Medical Imaging](http://arxiv.org/abs/2511.03912v1) | Nand Kumar Yadav, Rodrigue Rizk et al. | Unknown anomaly detection in medical imaging remains a fundamental challenge due to the scarcity of labeled anomalies and the high cost of expert supervision. We introduce an unsupervised, oracle-free framework that incrementally expands a trusted set of normal samples without any anomaly labels. Starting from a small, verified seed of normal images, our method alternates between lightweight adapter updates and uncertainty-gated sample admission. A frozen pretrained vision backbone is augmented with tiny convolutional adapters, ensuring rapid domain adaptation with negligible computational overhead. Extracted embeddings are stored in a compact coreset enabling efficient k-nearest neighbor anomaly (k-NN) scoring. Safety during incremental expansion is enforced by dual probabilistic gates, a sample is admitted into the normal memory only if its distance to the existing coreset lies within a calibrated z-score threshold, and its SWAG-based epistemic uncertainty remains below a seed-calibrated bound. This mechanism prevents drift and false inclusions without relying on generative reconstruction or replay buffers. Empirically, our system steadily refines the notion of normality as unlabeled data arrive, producing substantial gains over baselines. On COVID-CXR, ROC-AUC improves from 0.9489 to 0.9982 (F1: 0.8048 to 0.9746); on Pneumonia CXR, ROC-AUC rises from 0.6834 to 0.8968; and on Brain MRI ND-5, ROC-AUC increases from 0.6041 to 0.7269 and PR-AUC from 0.7539 to 0.8211. These results highlight the effectiveness and efficiency of the proposed framework for real-world, label-scarce medical imaging applications. |
| 2025-11-05 | [QSAFE-V: Quantum-Enhanced Lightweight Authentication Protocol Design for Vehicular Tactile Wireless Networks](http://arxiv.org/abs/2511.03850v1) | Shakil Ahmed, Amika Tabassum et al. | With the rapid advancement of 6G technology, the Tactile Internet is emerging as a novel paradigm of interaction, particularly in intelligent transportation systems, where stringent demands for ultra-low latency and high reliability are prevalent. During the transmission and coordination of autonomous vehicles, malicious adversaries may attempt to compromise control commands or swarm behavior, posing severe threats to road safety and vehicular intelligence. Many existing authentication schemes claim to provide security against conventional attacks. However, recent developments in quantum computing have revealed critical vulnerabilities in these schemes, particularly under quantum-enabled adversarial models. In this context, the design of a quantum-secured, lightweight authentication scheme that is adaptable to vehicular mobility becomes essential. This paper proposes QSAFE-V, a quantum-secured authentication framework for edge-enabled vehicles that surpasses traditional security models. We conduct formal security proofs based on quantum key distribution and quantum adversary models, and also perform context-driven reauthentication analysis based on vehicular behavior. The output of quantum resilience evaluations indicates that QSAFE-V provides robust protection against quantum and contextual attacks. Furthermore, detailed performance analysis reveals that QSAFE-V achieves comparable communication and computation costs to classical schemes, while offering significantly stronger security guarantees under wireless Tactile Internet conditions. |

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



