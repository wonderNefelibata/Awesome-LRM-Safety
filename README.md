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
| 2025-12-12 | [LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems](http://arxiv.org/abs/2512.11750v1) | Ernesto Casablanca, Oliver Schön et al. | Ensuring the safety of AI-enabled systems, particularly in high-stakes domains such as autonomous driving and healthcare, has become increasingly critical. Traditional formal verification tools fall short when faced with systems that embed both opaque, black-box AI components and complex stochastic dynamics. To address these challenges, we introduce LUCID (Learning-enabled Uncertainty-aware Certification of stochastIc Dynamical systems), a verification engine for certifying safety of black-box stochastic dynamical systems from a finite dataset of random state transitions. As such, LUCID is the first known tool capable of establishing quantified safety guarantees for such systems. Thanks to its modular architecture and extensive documentation, LUCID is designed for easy extensibility. LUCID employs a data-driven methodology rooted in control barrier certificates, which are learned directly from system transition data, to ensure formal safety guarantees. We use conditional mean embeddings to embed data into a reproducing kernel Hilbert space (RKHS), where an RKHS ambiguity set is constructed that can be inflated to robustify the result to out-of-distribution behavior. A key innovation within LUCID is its use of a finite Fourier kernel expansion to reformulate a semi-infinite non-convex optimization problem into a tractable linear program. The resulting spectral barrier allows us to leverage the fast Fourier transform to generate the relaxed problem efficiently, offering a scalable yet distributionally robust framework for verifying safety. LUCID thus offers a robust and efficient verification framework, able to handle the complexities of modern black-box systems while providing formal guarantees of safety. These unique capabilities are demonstrated on challenging benchmarks. |
| 2025-12-12 | [MedAI: Evaluating TxAgent's Therapeutic Agentic Reasoning in the NeurIPS CURE-Bench Competition](http://arxiv.org/abs/2512.11682v1) | Tim Cofala, Christian Kalfar et al. | Therapeutic decision-making in clinical medicine constitutes a high-stakes domain in which AI guidance interacts with complex interactions among patient characteristics, disease processes, and pharmacological agents. Tasks such as drug recommendation, treatment planning, and adverse-effect prediction demand robust, multi-step reasoning grounded in reliable biomedical knowledge. Agentic AI methods, exemplified by TxAgent, address these challenges through iterative retrieval-augmented generation (RAG). TxAgent employs a fine-tuned Llama-3.1-8B model that dynamically generates and executes function calls to a unified biomedical tool suite (ToolUniverse), integrating FDA Drug API, OpenTargets, and Monarch resources to ensure access to current therapeutic information. In contrast to general-purpose RAG systems, medical applications impose stringent safety constraints, rendering the accuracy of both the reasoning trace and the sequence of tool invocations critical. These considerations motivate evaluation protocols treating token-level reasoning and tool-usage behaviors as explicit supervision signals. This work presents insights derived from our participation in the CURE-Bench NeurIPS 2025 Challenge, which benchmarks therapeutic-reasoning systems using metrics that assess correctness, tool utilization, and reasoning quality. We analyze how retrieval quality for function (tool) calls influences overall model performance and demonstrate performance gains achieved through improved tool-retrieval strategies. Our work was awarded the Excellence Award in Open Science. Complete information can be found at https://curebench.ai/. |
| 2025-12-12 | [Automating Historical Insight Extraction from Large-Scale Newspaper Archives via Neural Topic Modeling](http://arxiv.org/abs/2512.11635v1) | Keerthana Murugaraj, Salima Lamsiyah et al. | Extracting coherent and human-understandable themes from large collections of unstructured historical newspaper archives presents significant challenges due to topic evolution, Optical Character Recognition (OCR) noise, and the sheer volume of text. Traditional topic-modeling methods, such as Latent Dirichlet Allocation (LDA), often fall short in capturing the complexity and dynamic nature of discourse in historical texts. To address these limitations, we employ BERTopic. This neural topic-modeling approach leverages transformerbased embeddings to extract and classify topics, which, despite its growing popularity, still remains underused in historical research. Our study focuses on articles published between 1955 and 2018, specifically examining discourse on nuclear power and nuclear safety. We analyze various topic distributions across the corpus and trace their temporal evolution to uncover long-term trends and shifts in public discourse. This enables us to more accurately explore patterns in public discourse, including the co-occurrence of themes related to nuclear power and nuclear weapons and their shifts in topic importance over time. Our study demonstrates the scalability and contextual sensitivity of BERTopic as an alternative to traditional approaches, offering richer insights into historical discourses extracted from newspaper archives. These findings contribute to historical, nuclear, and social-science research while reflecting on current limitations and proposing potential directions for future work. |
| 2025-12-12 | [Architecting Large Action Models for Human-in-the-Loop Intelligent Robots](http://arxiv.org/abs/2512.11620v1) | Kanisorn Sangchai, Methasit Boonpun et al. | The realization of intelligent robots, operating autonomously and interacting with other intelligent agents, human or artificial, requires the integration of environment perception, reasoning, and action. Classic Artificial Intelligence techniques for this purpose, focusing on symbolic approaches, have long-ago hit the scalability wall on compute and memory costs. Advances in Large Language Models in the past decade (neural approaches) have resulted in unprecedented displays of capability, at the cost of control, explainability, and interpretability. Large Action Models aim at extending Large Language Models to encompass the full perception, reasoning, and action cycle; however, they typically require substantially more comprehensive training and suffer from the same deficiencies in reliability. Here, we show it is possible to build competent Large Action Models by composing off-the-shelf foundation models, and that their control, interpretability, and explainability can be effected by incorporating symbolic wrappers and associated verification on their outputs, achieving verifiable neuro-symbolic solutions for intelligent robots. Our experiments on a multi-modal robot demonstrate that Large Action Model intelligence does not require massive end-to-end training, but can be achieved by integrating efficient perception models with a logic-driven core. We find that driving action execution through the generation of Planning Domain Definition Language (PDDL) code enables a human-in-the-loop verification stage that effectively mitigates action hallucinations. These results can support practitioners in the design and development of robotic Large Action Models across novel industries, and shed light on the ongoing challenges that must be addressed to ensure safety in the field. |
| 2025-12-12 | [Safe Bayesian optimization across noise models via scenario programming](http://arxiv.org/abs/2512.11580v1) | Abdullah Tokmak, Thomas B. Schön et al. | Safe Bayesian optimization (BO) with Gaussian processes is an effective tool for tuning control policies in safety-critical real-world systems, specifically due to its sample efficiency and safety guarantees. However, most safe BO algorithms assume homoscedastic sub-Gaussian measurement noise, an assumption that does not hold in many relevant applications. In this article, we propose a straightforward yet rigorous approach for safe BO across noise models, including homoscedastic sub-Gaussian and heteroscedastic heavy-tailed distributions. We provide a high-probability bound on the measurement noise via the scenario approach, integrate these bounds into high probability confidence intervals, and prove safety and optimality for our proposed safe BO algorithm. We deploy our algorithm in synthetic examples and in tuning a controller for the Franka Emika manipulator in simulation. |
| 2025-12-12 | [CarlaNCAP: A Framework for Quantifying the Safety of Vulnerable Road Users in Infrastructure-Assisted Collective Perception Using EuroNCAP Scenarios](http://arxiv.org/abs/2512.11551v1) | Jörg Gamerdinger, Sven Teufel et al. | The growing number of road users has significantly increased the risk of accidents in recent years. Vulnerable Road Users (VRUs) are particularly at risk, especially in urban environments where they are often occluded by parked vehicles or buildings. Autonomous Driving (AD) and Collective Perception (CP) are promising solutions to mitigate these risks. In particular, infrastructure-assisted CP, where sensor units are mounted on infrastructure elements such as traffic lights or lamp posts, can help overcome perceptual limitations by providing enhanced points of view, which significantly reduces occlusions. To encourage decision makers to adopt this technology, comprehensive studies and datasets demonstrating safety improvements for VRUs are essential. In this paper, we propose a framework for evaluating the safety improvement by infrastructure-based CP specifically targeted at VRUs including a dataset with safety-critical EuroNCAP scenarios (CarlaNCAP) with 11k frames. Using this dataset, we conduct an in-depth simulation study and demonstrate that infrastructure-assisted CP can significantly reduce accident rates in safety-critical scenarios, achieving up to 100% accident avoidance compared to a vehicle equipped with sensors with only 33%. Code is available at https://github.com/ekut-es/carla_ncap |
| 2025-12-12 | [AI-MASLD Metabolic Dysfunction and Information Steatosis of Large Language Models in Unstructured Clinical Narratives](http://arxiv.org/abs/2512.11544v1) | Yuan Shen, Xiaojun Wu et al. | This study aims to simulate real-world clinical scenarios to systematically evaluate the ability of Large Language Models (LLMs) to extract core medical information from patient chief complaints laden with noise and redundancy, and to verify whether they exhibit a functional decline analogous to Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD). We employed a cross-sectional analysis design based on standardized medical probes, selecting four mainstream LLMs as research subjects: GPT-4o, Gemini 2.5, DeepSeek 3.1, and Qwen3-Max. An evaluation system comprising twenty medical probes across five core dimensions was used to simulate a genuine clinical communication environment. All probes had gold-standard answers defined by clinical experts and were assessed via a double-blind, inverse rating scale by two independent clinicians. The results show that all tested models exhibited functional defects to varying degrees, with Qwen3-Max demonstrating the best overall performance and Gemini 2.5 the worst. Under conditions of extreme noise, most models experienced a functional collapse. Notably, GPT-4o made a severe misjudgment in the risk assessment for pulmonary embolism (PE) secondary to deep vein thrombosis (DVT). This research is the first to empirically confirm that LLMs exhibit features resembling metabolic dysfunction when processing clinical information, proposing the innovative concept of "AI-Metabolic Dysfunction-Associated Steatotic Liver Disease (AI-MASLD)". These findings offer a crucial safety warning for the application of Artificial Intelligence (AI) in healthcare, emphasizing that current LLMs must be used as auxiliary tools under human expert supervision, as there remains a significant gap between their theoretical knowledge and practical clinical application. |
| 2025-12-12 | [Shared Situational Awareness Using Hybrid Zonotopes with Confidence Metric](http://arxiv.org/abs/2512.11493v1) | Vandana Narri, Jonah J. Glunt et al. | Situational awareness for connected and automated vehicles describes the ability to perceive and predict the behavior of other road-users in the near surroundings. However, pedestrians can become occluded by vehicles or infrastructure, creating significant safety risks due to limited visibility. Vehicle-to-everything communication enables the sharing of perception data between connected road-users, allowing for a more comprehensive awareness. The main challenge is how to fuse perception data when measurements are inconsistent with the true locations of pedestrians. Inconsistent measurements can occur due to sensor noise, false positives, or communication issues. This paper employs set-based estimation with constrained zonotopes to compute a confidence metric for the measurement set from each sensor. These sets and their confidences are then fused using hybrid zonotopes. This method can account for inconsistent measurements, enabling reliable and robust fusion of the sensor data. The effectiveness of the proposed method is demonstrated in both simulation and real experiments. |
| 2025-12-12 | [Advances in Ontology--Based Mining of Adverse Drug Reactions](http://arxiv.org/abs/2512.11452v1) | Kenenisa Tadesse Dame, Pietro Belloni et al. | Post--marketing pharmacovigilance is essential for identifying adverse drug reactions (ADRs) that elude detection during pre--marketing clinical trials. This study explores a novel approach that integrates an adverse event (AE) ontology into a zero--inflated negative binomial model to improve ADR detection. By accounting for the biological similarities among correlated AEs and addressing the excess of zero counts, this method more effectively disentangles AE associations. Statistical significance is evaluated using a permutation--based maximum statistic that preserves AE correlations within individual reports. Simulations and an application to real data from the Veneto drug safety database demonstrate that the ontology--based model consistently outperforms classical models such as the Gamma--Poisson Shrinker (GPS). For post--selection inference, we furthermore explore a data thinning technique for convolution--closed families, enabling the creation of independent training and validation datasets while retaining all drug--AE pairs. This approach is compared with conventional random train/test splitting, which may leave some drugs or AEs absent from one subset, and stratified splitting, which requires expanding aggregated counts into individual instances. The data--thinning technique and stratified splitting yield very similar results, with stratified splitting showing a slight benefit, and both clearly outperform random splitting in ensuring reliable and consistent model evaluation. |
| 2025-12-12 | [CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare](http://arxiv.org/abs/2512.11437v1) | Akash Ghosh, Srivarshinee Sridhar et al. | Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages. |
| 2025-12-12 | [Task-Specific Sparse Feature Masks for Molecular Toxicity Prediction with Chemical Language Models](http://arxiv.org/abs/2512.11412v1) | Kwun Sy Lee, Jiawei Chen et al. | Reliable in silico molecular toxicity prediction is a cornerstone of modern drug discovery, offering a scalable alternative to experimental screening. However, the black-box nature of state-of-the-art models remains a significant barrier to adoption, as high-stakes safety decisions demand verifiable structural insights alongside predictive performance. To address this, we propose a novel multi-task learning (MTL) framework designed to jointly enhance accuracy and interpretability. Our architecture integrates a shared chemical language model with task-specific attention modules. By imposing an L1 sparsity penalty on these modules, the framework is constrained to focus on a minimal set of salient molecular fragments for each distinct toxicity endpoint. The resulting framework is trained end-to-end and is readily adaptable to various transformer-based backbones. Evaluated on the ClinTox, SIDER, and Tox21 benchmark datasets, our approach consistently outperforms both single-task and standard MTL baselines. Crucially, the sparse attention weights provide chemically intuitive visualizations that reveal the specific fragments influencing predictions, thereby enhancing insight into the model's decision-making process. |
| 2025-12-12 | [Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](http://arxiv.org/abs/2512.11391v1) | Yifan Niu, Han Xiao et al. | As Large Language Models (LLMs) are increasingly deployed in real-world applications, it is important to ensure their behaviors align with human values, societal norms, and ethical principles. However, safety alignment under Reinforcement Learning (RL) often suffers from forgetting learned general abilities, which is also known as the alignment tax. To address this issue, we introduce Null-Space constrained Policy Optimization (NSPO), a novel RL framework for LLM safety alignment while preserving their core abilities. The safety policy gradients are geometrically projected into the null space of general tasks, thereby mitigating the safety alignment tax. In addition, we theoretically prove that NSPO preserves the model's original core capabilities, while still guaranteeing a descent direction for effective safety alignment. Extensive experiments demonstrate that NSPO outperforms existing methods by a large margin, achieving state-of-the-art safety performance without sacrificing accuracy on general tasks, including math, code, and instruction-following tasks. Notably, NSPO is data-efficient and only requires 40% of public human-annotated safety data from PKU-SafeRLHF to achieve promising safety performance, without a large amount of mixed general tasks data in existing alignment methods. |
| 2025-12-12 | [Out-of-Distribution Segmentation via Wasserstein-Based Evidential Uncertainty](http://arxiv.org/abs/2512.11373v1) | Arnold Brosch, Abdelrahman Eldesokey et al. | Deep neural networks achieve superior performance in semantic segmentation, but are limited to a predefined set of classes, which leads to failures when they encounter unknown objects in open-world scenarios. Recognizing and segmenting these out-of-distribution (OOD) objects is crucial for safety-critical applications such as automated driving. In this work, we present an evidence segmentation framework using a Wasserstein loss, which captures distributional distances while respecting the probability simplex geometry. Combined with Kullback-Leibler regularization and Dice structural consistency terms, our approach leads to improved OOD segmentation performance compared to uncertainty-based approaches. |
| 2025-12-12 | [An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges](http://arxiv.org/abs/2512.11362v1) | Chao Xu, Suyu Zhang et al. | Vision-Language-Action (VLA) models are driving a revolution in robotics, enabling machines to understand instructions and interact with the physical world. This field is exploding with new models and datasets, making it both exciting and challenging to keep pace with. This survey offers a clear and structured guide to the VLA landscape. We design it to follow the natural learning path of a researcher: we start with the basic Modules of any VLA model, trace the history through key Milestones, and then dive deep into the core Challenges that define recent research frontier. Our main contribution is a detailed breakdown of the five biggest challenges in: (1) Representation, (2) Execution, (3) Generalization, (4) Safety, and (5) Dataset and Evaluation. This structure mirrors the developmental roadmap of a generalist agent: establishing the fundamental perception-action loop, scaling capabilities across diverse embodiments and environments, and finally ensuring trustworthy deployment-all supported by the essential data infrastructure. For each of them, we review existing approaches and highlight future opportunities. We position this paper as both a foundational guide for newcomers and a strategic roadmap for experienced researchers, with the dual aim of accelerating learning and inspiring new ideas in embodied intelligence. A live version of this survey, with continuous updates, is maintained on our \href{https://suyuz1.github.io/Survery/}{project page}. |
| 2025-12-12 | [A Multi-Mode Structured Light 3D Imaging System with Multi-Source Information Fusion for Underwater Pipeline Detection](http://arxiv.org/abs/2512.11354v1) | Qinghan Hu, Haijiang Zhu et al. | Underwater pipelines are highly susceptible to corrosion, which not only shorten their service life but also pose significant safety risks. Compared with manual inspection, the intelligent real-time imaging system for underwater pipeline detection has become a more reliable and practical solution. Among various underwater imaging techniques, structured light 3D imaging can restore the sufficient spatial detail for precise defect characterization. Therefore, this paper develops a multi-mode underwater structured light 3D imaging system for pipeline detection (UW-SLD system) based on multi-source information fusion. First, a rapid distortion correction (FDC) method is employed for efficient underwater image rectification. To overcome the challenges of extrinsic calibration among underwater sensors, a factor graph-based parameter optimization method is proposed to estimate the transformation matrix between the structured light and acoustic sensors. Furthermore, a multi-mode 3D imaging strategy is introduced to adapt to the geometric variability of underwater pipelines. Given the presence of numerous disturbances in underwater environments, a multi-source information fusion strategy and an adaptive extended Kalman filter (AEKF) are designed to ensure stable pose estimation and high-accuracy measurements. In particular, an edge detection-based ICP (ED-ICP) algorithm is proposed. This algorithm integrates pipeline edge detection network with enhanced point cloud registration to achieve robust and high-fidelity reconstruction of defect structures even under variable motion conditions. Extensive experiments are conducted under different operation modes, velocities, and depths. The results demonstrate that the developed system achieves superior accuracy, adaptability and robustness, providing a solid foundation for autonomous underwater pipeline detection. |
| 2025-12-12 | [Incremental Validation of Automated Driving Functions using Generic Volumes in Micro- Operational Design Domains](http://arxiv.org/abs/2512.11351v1) | Steffen Schäfer, Martin Cichon | The validation of highly automated, perception-based driving systems must ensure that they function correctly under the full range of real-world conditions. Scenario-based testing is a prominent approach to addressing this challenge, as it involves the systematic simulation of objects and environments. Operational Design Domains (ODDs) are usually described using a taxonomy of qualitative designations for individual objects. However, the process of transitioning from taxonomy to concrete test cases remains unstructured, and completeness is theoretical. This paper introduces a structured method of subdividing the ODD into manageable sections, termed micro-ODDs (mODDs), and deriving test cases with abstract object representations. This concept is demonstrated using a one-dimensional, laterally guided manoeuvre involving a shunting locomotive within a constrained ODD. In this example, mODDs are defined and refined into narrow taxonomies that enable test case generation. Obstacles are represented as generic cubes of varying sizes, providing a simplified yet robust means of evaluating perception performance. A series of tests were conducted in a closed-loop, co-simulated virtual environment featuring photorealistic rendering and simulated LiDAR, GNSS and camera sensors. The results demonstrate how edge cases in obstacle detection can be systematically explored and how perception quality can be evaluated based on observed vehicle behaviour, using crash versus safe stop as the outcome metrics. These findings support the development of a standardised framework for safety argumentation and offer a practical step towards the validation and authorisation of automated driving functions. |
| 2025-12-12 | [Pace: Physics-Aware Attentive Temporal Convolutional Network for Battery Health Estimation](http://arxiv.org/abs/2512.11332v1) | Sara Sameer, Wei Zhang et al. | Batteries are critical components in modern energy systems such as electric vehicles and power grid energy storage. Effective battery health management is essential for battery system safety, cost-efficiency, and sustainability. In this paper, we propose Pace, a physics-aware attentive temporal convolutional network for battery health estimation. Pace integrates raw sensor measurements with battery physics features derived from the equivalent circuit model. We develop three battery-specific modules, including dilated temporal blocks for efficient temporal encoding, chunked attention blocks for context modeling, and a dual-head output block for fusing short- and long-term battery degradation patterns. Together, the modules enable Pace to predict battery health accurately and efficiently in various battery usage conditions. In a large public dataset, Pace performs much better than existing models, achieving an average performance improvement of 6.5 and 2.0x compared to two best-performing baseline models. We further demonstrate its practical viability with a real-time edge deployment on a Raspberry Pi. These results establish Pace as a practical and high-performance solution for battery health analytics. |
| 2025-12-12 | [Few-Shot VLM-Based G-Code and HMI Verification in CNC Machining](http://arxiv.org/abs/2512.11296v1) | Yasaman Hashem Pour, Nazanin Mahjourian et al. | Manual generation of G-code is important for learning the operation of CNC machines. Prior work in G-code verification uses Large-Language Models (LLMs), which primarily examine errors in the written programming. However, CNC machining requires extensive use and knowledge of the Human-Machine Interface (HMI), which displays machine status and errors. LLMs currently lack the capability to leverage knowledge of HMIs due to their inability to access the vision modality. This paper proposes a few-shot VLM-based verification approach that simultaneously evaluates the G-code and the HMI display for errors and safety status. The input dataset includes paired G-code text and associated HMI screenshots from a 15-slant-PRO lathe, including both correct and error-prone cases. To enable few-shot learning, the VLM is provided with a structured JSON schema based on prior heuristic knowledge. After determining the prompts, instances of G-code and HMI that either contain errors or are error free are used as few-shot examples to guide the VLM. The model was then evaluated in comparison to a zero-shot VLM through multiple scenarios of incorrect G-code and HMI errors with respect to per-slot accuracy. The VLM showed that few-shot prompting led to overall enhancement of detecting HMI errors and discrepancies with the G-code for more comprehensive debugging. Therefore, the proposed framework was demonstrated to be suitable for verification of manually generated G-code that is typically developed in CNC training. |
| 2025-12-12 | [When Actions Teach You to Think: Reasoning-Action Synergy via Reinforcement Learning in Conversational Agents](http://arxiv.org/abs/2512.11277v1) | Mrinal Rawat, Arkajyoti Chakraborty et al. | Supervised fine-tuning (SFT) has emerged as one of the most effective ways to improve the performance of large language models (LLMs) in downstream tasks. However, SFT can have difficulty generalizing when the underlying data distribution changes, even when the new data does not fall completely outside the training domain. Recent reasoning-focused models such as o1 and R1 have demonstrated consistent gains over their non-reasoning counterparts, highlighting the importance of reasoning for improved generalization and reliability. However, collecting high-quality reasoning traces for SFT remains challenging -- annotations are costly, subjective, and difficult to scale. To address this limitation, we leverage Reinforcement Learning (RL) to enable models to learn reasoning strategies directly from task outcomes. We propose a pipeline in which LLMs generate reasoning steps that guide both the invocation of tools (e.g., function calls) and the final answer generation for conversational agents. Our method employs Group Relative Policy Optimization (GRPO) with rewards designed around tool accuracy and answer correctness, allowing the model to iteratively refine its reasoning and actions. Experimental results demonstrate that our approach improves both the quality of reasoning and the precision of tool invocations, achieving a 1.5% relative improvement over the SFT model (trained without explicit thinking) and a 40% gain compared to the base of the vanilla Qwen3-1.7B model. These findings demonstrate the promise of unifying reasoning and action learning through RL to build more capable and generalizable conversational agents. |
| 2025-12-12 | [Multi-Objective Reinforcement Learning for Large-Scale Mixed Traffic Control](http://arxiv.org/abs/2512.11247v1) | Iftekharul Islam, Weizi Li | Effective mixed traffic control requires balancing efficiency, fairness, and safety. Existing approaches excel at optimizing efficiency and enforcing safety constraints but lack mechanisms to ensure equitable service, resulting in systematic starvation of vehicles on low-demand approaches. We propose a hierarchical framework combining multi-objective reinforcement learning for local intersection control with strategic routing for network-level coordination. Our approach introduces a Conflict Threat Vector that provides agents with explicit risk signals for proactive conflict avoidance, and a queue parity penalty that ensures equitable service across all traffic streams. Extensive experiments on a real-world network across different robot vehicle (RV) penetration rates demonstrate substantial improvements: up to 53% reductions in average wait time, up to 86% reductions in maximum starvation, and up to 86\% reduction in conflict rate compared to baselines, while maintaining fuel efficiency. Our analysis reveals that strategic routing effectiveness scales with RV penetration, becoming increasingly valuable at higher autonomy levels. The results demonstrate that multi-objective optimization through well-curated reward functions paired with strategic RV routing yields significant benefits in fairness and safety metrics critical for equitable mixed-autonomy deployment. |

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



