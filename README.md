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
| 2025-10-24 | [Hybrid Genetic Algorithm for Optimal User Order Routing: Multi-Objective Solver Optimization in CoW Protocol Batch Auctions](http://arxiv.org/abs/2510.21647v1) | Mitchell Marfinetz | CoW Protocol batch auctions aggregate user intents and rely on solvers to find optimal execution paths that maximize user surplus across heterogeneous automated market makers (AMMs) under stringent auction deadlines. Deterministic single-objective heuristics that optimize only expected output frequently fail to exploit split-flow opportunities across multiple parallel paths and to internalize gas, slippage, and execution risk constraints in a unified search. We apply evolutionary multi-objective optimization to this blockchain routing problem, proposing a hybrid genetic algorithm (GA) architecture for real-time solver optimization that combines a production-grade, multi-objective NSGA-II engine with adaptive instance profiling and deterministic baselines. Our core engine encodes variable-length path sets with continuous split ratios and evolves candidate route-and-volume allocations under a Pareto objective vector F = (user surplus, -gas, -slippage, -risk), enabling principled trade-offs and anytime operation within the auction deadline. An adaptive controller selects between GA and a deterministic dual-decomposition optimizer with Bellman-Ford based negative-cycle detection, with a guarantee to never underperform the baseline. The open-source system integrates six protection layers and passes 8/8 tests, validating safety and correctness. In a 14-stratum benchmark (30 seeds each), the hybrid approach yields absolute user-surplus gains of approximately 0.40-9.82 ETH on small-to-medium orders, while large high-fragmentation orders are unprofitable across gas regimes. Convergence occurs in about 0.5 s median (soft capped at 1.0 s) within a 2-second limit. We are not aware of an openly documented multi-objective GA with end-to-end safety for real-time DEX routing. |
| 2025-10-24 | [DEEDEE: Fast and Scalable Out-of-Distribution Dynamics Detection](http://arxiv.org/abs/2510.21638v1) | Tala Aljaafari, Varun Kanade et al. | Deploying reinforcement learning (RL) in safety-critical settings is constrained by brittleness under distribution shift. We study out-of-distribution (OOD) detection for RL time series and introduce DEEDEE, a two-statistic detector that revisits representation-heavy pipelines with a minimal alternative. DEEDEE uses only an episodewise mean and an RBF kernel similarity to a training summary, capturing complementary global and local deviations. Despite its simplicity, DEEDEE matches or surpasses contemporary detectors across standard RL OOD suites, delivering a 600-fold reduction in compute (FLOPs / wall-time) and an average 5% absolute accuracy gain over strong baselines. Conceptually, our results indicate that diverse anomaly types often imprint on RL trajectories through a small set of low-order statistics, suggesting a compact foundation for OOD detection in complex environments. |
| 2025-10-24 | [SurVigilance: An Application for Accessing Global Pharmacovigilance Data](http://arxiv.org/abs/2510.21572v1) | Raktim Mukhopadhyay, Marianthi Markatou | Even though several publicly accessible pharmacovigilance databases are available, extracting data from them is a technically challenging process. Existing tools typically focus on a single database. We present SurVigilance, an open-source tool that streamlines the process of retrieving safety data from seven major pharmacovigilance databases. SurVigilance provides a graphical user interface as well as functions for programmatic access, thus enabling integration into existing research workflows. SurVigilance utilizes a modular architecture to provide access to the heterogeneous sources. By reducing the technical barriers to accessing safety data, SurVigilance aims to facilitate pharmacovigilance research. |
| 2025-10-24 | [Learning Neural Control Barrier Functions from Expert Demonstrations using Inverse Constraint Learning](http://arxiv.org/abs/2510.21560v1) | Yuxuan Yang, Hussein Sibai | Safety is a fundamental requirement for autonomous systems operating in critical domains. Control barrier functions (CBFs) have been used to design safety filters that minimally alter nominal controls for such systems to maintain their safety. Learning neural CBFs has been proposed as a data-driven alternative for their computationally expensive optimization-based synthesis. However, it is often the case that the failure set of states that should be avoided is non-obvious or hard to specify formally, e.g., tailgating in autonomous driving, while a set of expert demonstrations that achieve the task and avoid the failure set is easier to generate. We use ICL to train a constraint function that classifies the states of the system under consideration to safe, i.e., belong to a controlled forward invariant set that is disjoint from the unspecified failure set, and unsafe ones, i.e., belong to the complement of that set. We then use that function to label a new set of simulated trajectories to train our neural CBF. We empirically evaluate our approach in four different environments, demonstrating that it outperforms existing baselines and achieves comparable performance to a neural CBF trained with the same data but annotated with ground-truth safety labels. |
| 2025-10-24 | [Auction-Based Responsibility Allocation for Scalable Decentralized Safety Filters in Cooperative Multi-Agent Collision Avoidance](http://arxiv.org/abs/2510.21546v1) | Johannes Autenrieb, Mark Spiller | This paper proposes a scalable decentralized safety filter for multi-agent systems based on high-order control barrier functions (HOCBFs) and auction-based responsibility allocation. While decentralized HOCBF formulations ensure pairwise safety under input bounds, they face feasibility and scalability challenges as the number of agents grows. Each agent must evaluate an increasing number of pairwise constraints, raising the risk of infeasibility and making it difficult to meet real-time requirements. To address this, we introduce an auction-based allocation scheme that distributes constraint enforcement asymmetrically among neighbors based on local control effort estimates. The resulting directed responsibility graph guarantees full safety coverage while reducing redundant constraints and per-agent computational load. Simulation results confirm safe and efficient coordination across a range of network sizes and interaction densities. |
| 2025-10-24 | [EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](http://arxiv.org/abs/2510.21524v1) | Ilija Lichkovski, Alexander Müller et al. | Large language models (LLMs) are increasingly deployed as agents in various contexts by providing tools at their disposal. However, LLM agents can exhibit unpredictable behaviors, including taking undesirable and/or unsafe actions. In order to measure the latent propensity of LLM agents for taking illegal actions under an EU legislative context, we introduce EU-Agent-Bench, a verifiable human-curated benchmark that evaluates an agent's alignment with EU legal norms in situations where benign user inputs could lead to unlawful actions. Our benchmark spans scenarios across several categories, including data protection, bias/discrimination, and scientific integrity, with each user request allowing for both compliant and non-compliant execution of the requested actions. Comparing the model's function calls against a rubric exhaustively supported by citations of the relevant legislature, we evaluate the legal compliance of frontier LLMs, and furthermore investigate the compliance effect of providing the relevant legislative excerpts in the agent's system prompt along with explicit instructions to comply. We release a public preview set for the research community, while holding out a private test set to prevent data contamination in evaluating upcoming models. We encourage future work extending agentic safety benchmarks to different legal jurisdictions and to multi-turn and multilingual interactions. We release our code on \href{https://github.com/ilijalichkovski/eu-agent-bench}{this URL}. |
| 2025-10-24 | [An Automatic Detection Method for Hematoma Features in Placental Abruption Ultrasound Images Based on Few-Shot Learning](http://arxiv.org/abs/2510.21495v1) | Xiaoqing Liu, Jitai Han et al. | Placental abruption is a severe complication during pregnancy, and its early accurate diagnosis is crucial for ensuring maternal and fetal safety. Traditional ultrasound diagnostic methods heavily rely on physician experience, leading to issues such as subjective bias and diagnostic inconsistencies. This paper proposes an improved model, EH-YOLOv11n (Enhanced Hemorrhage-YOLOv11n), based on small-sample learning, aiming to achieve automatic detection of hematoma features in placental ultrasound images. The model enhances performance through multidimensional optimization: it integrates wavelet convolution and coordinate convolution to strengthen frequency and spatial feature extraction; incorporates a cascaded group attention mechanism to suppress ultrasound artifacts and occlusion interference, thereby improving bounding box localization accuracy. Experimental results demonstrate a detection accuracy of 78%, representing a 2.5% improvement over YOLOv11n and a 13.7% increase over YOLOv8. The model exhibits significant superiority in precision-recall curves, confidence scores, and occlusion scenarios. Combining high accuracy with real-time processing, this model provides a reliable solution for computer-aided diagnosis of placental abruption, holding significant clinical application value. |
| 2025-10-24 | [Load-bearing Assessment for Safe Locomotion of Quadruped Robots on Collapsing Terrain](http://arxiv.org/abs/2510.21369v1) | Vivian S. Medeiros, Giovanni B. Dessy et al. | Collapsing terrains, often present in search and rescue missions or planetary exploration, pose significant challenges for quadruped robots. This paper introduces a robust locomotion framework for safe navigation over unstable surfaces by integrating terrain probing, load-bearing analysis, motion planning, and control strategies. Unlike traditional methods that rely on specialized sensors or external terrain mapping alone, our approach leverages joint measurements to assess terrain stability without hardware modifications. A Model Predictive Control (MPC) system optimizes robot motion, balancing stability and probing constraints, while a state machine coordinates terrain probing actions, enabling the robot to detect collapsible regions and dynamically adjust its footholds. Experimental results on custom-made collapsing platforms and rocky terrains demonstrate the framework's ability to traverse collapsing terrain while maintaining stability and prioritizing safety. |
| 2025-10-24 | [Remote Autonomy for Multiple Small Lowcost UAVs in GNSS-denied Search and Rescue Operations](http://arxiv.org/abs/2510.21357v1) | Daniel Schleich, Jan Quenzel et al. | In recent years, consumer-grade UAVs have been widely adopted by first responders. In general, they are operated manually, which requires trained pilots, especially in unknown GNSS-denied environments and in the vicinity of structures. Autonomous flight can facilitate the application of UAVs and reduce operator strain. However, autonomous systems usually require special programming interfaces, custom sensor setups, and strong onboard computers, which limits a broader deployment.   We present a system for autonomous flight using lightweight consumer-grade DJI drones. They are controlled by an Android app for state estimation and obstacle avoidance directly running on the UAV's remote control. Our ground control station enables a single operator to configure and supervise multiple heterogeneous UAVs at once. Furthermore, it combines the observations of all UAVs into a joint 3D environment model for improved situational awareness. |
| 2025-10-24 | [Predictive control barrier functions for piecewise affine systems with non-smooth constraints](http://arxiv.org/abs/2510.21321v1) | Kanghui He, Anil Alan et al. | Obtaining control barrier functions (CBFs) with large safe sets for complex nonlinear systems and constraints is a challenging task. Predictive CBFs address this issue by using an online finite-horizon optimal control problem that implicitly defines a large safe set. The optimal control problem, also known as the predictive safety filter (PSF), involves predicting the system's flow under a given backup control policy. However, for non-smooth systems and constraints, some key elements, such as CBF gradients and the sensitivity of the flow, are not well-defined, making the current methods inadequate for ensuring safety. Additionally, for control-non-affine systems, the PSF is generally nonlinear and non-convex, posing challenges for real-time computation. This paper considers piecewise affine systems, which are usually control-non-affine, under nonlinear state and polyhedral input constraints. We solve the safety issue by incorporating set-valued generalized Clarke derivatives in the PSF design. We show that enforcing CBF constraints across all elements of the generalized Clarke derivatives suffices to guarantee safety. Moreover, to lighten the computational overhead, we propose an explicit approximation of the PSF. The resulting control methods are demonstrated through numerical examples. |
| 2025-10-24 | [When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails](http://arxiv.org/abs/2510.21285v1) | Yingzhi Mao, Chunkang Zhang et al. | Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs. |
| 2025-10-24 | [Kriging measure-valued data with sparse observations: application to nuclear safety studies](http://arxiv.org/abs/2510.21277v1) | Florian Gossard, François Bachoc et al. | This work addresses the interpolation of probability measures within a spatial statistics framework. We develop a Kriging approach in the Wasserstein space, leveraging the quantile function representation of the one-dimensional Wasserstein distance. To mitigate the inaccuracies in semivariogram estimation that arise from sparse datasets, we combine this formulation with cross-validation techniques. In particular, we introduce a variant of the virtual cross-validation formulas tailored to quantile functions. The effectiveness of the proposed method is demonstrated on a controlled toy problem as well as on a real-world application from nuclear safety. |
| 2025-10-24 | [Out-of-Distribution Detection for Safety Assurance of AI and Autonomous Systems](http://arxiv.org/abs/2510.21254v1) | Victoria J. Hodge, Colin Paterson et al. | The operational capabilities and application domains of AI-enabled autonomous systems have expanded significantly in recent years due to advances in robotics and machine learning (ML). Demonstrating the safety of autonomous systems rigorously is critical for their responsible adoption but it is challenging as it requires robust methodologies that can handle novel and uncertain situations throughout the system lifecycle, including detecting out-of-distribution (OoD) data. Thus, OOD detection is receiving increased attention from the research, development and safety engineering communities. This comprehensive review analyses OOD detection techniques within the context of safety assurance for autonomous systems, in particular in safety-critical domains. We begin by defining the relevant concepts, investigating what causes OOD and exploring the factors which make the safety assurance of autonomous systems and OOD detection challenging. Our review identifies a range of techniques which can be used throughout the ML development lifecycle and we suggest areas within the lifecycle in which they may be used to support safety assurance arguments. We discuss a number of caveats that system and safety engineers must be aware of when integrating OOD detection into system lifecycles. We conclude by outlining the challenges and future work necessary for the safe development and operation of autonomous systems across a range of domains and applications. |
| 2025-10-24 | [Enhanced MLLM Black-Box Jailbreaking Attacks and Defenses](http://arxiv.org/abs/2510.21214v1) | Xingwei Zhong, Kar Wai Fok et al. | Multimodal large language models (MLLMs) comprise of both visual and textual modalities to process vision language tasks. However, MLLMs are vulnerable to security-related issues, such as jailbreak attacks that alter the model's input to induce unauthorized or harmful responses. The incorporation of the additional visual modality introduces new dimensions to security threats. In this paper, we proposed a black-box jailbreak method via both text and image prompts to evaluate MLLMs. In particular, we designed text prompts with provocative instructions, along with image prompts that introduced mutation and multi-image capabilities. To strengthen the evaluation, we also designed a Re-attack strategy. Empirical results show that our proposed work can improve capabilities to assess the security of both open-source and closed-source MLLMs. With that, we identified gaps in existing defense methods to propose new strategies for both training-time and inference-time defense methods, and evaluated them across the new jailbreak methods. The experiment results showed that the re-designed defense methods improved protections against the jailbreak attacks. |
| 2025-10-24 | [The Nuclear Analogy in AI Governance Research](http://arxiv.org/abs/2510.21203v1) | Sophia Hatz | The analogy between Artificial Intelligence (AI) and nuclear weapons is prominent in academic and policy discourse on AI governance. This chapter reviews 43 scholarly works which explicitly draw on the nuclear domain to derive lessons for AI governance. We identify four problem areas where researchers apply nuclear precedents: (1) early development and governance of transformative technologies; (2) international security risks and strategy; (3) international institutions and agreements; and (4) domestic safety regulation. While nuclear-inspired AI proposals are often criticised due to differences across domains, this review clarifies how historical analogies can inform policy development even when technological domains differ substantially. Valuable functions include providing conceptual frameworks for analyzing strategic dynamics, offering cautionary lessons about unsuccessful governance approaches, and expanding policy imagination by legitimizing radical proposals. Given that policymakers already invoke the nuclear analogy, continued critical engagement with these historical precedents remains essential for shaping effective global AI governance. |
| 2025-10-24 | [The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning](http://arxiv.org/abs/2510.21190v1) | Mingrui Liu, Sixiao Zhang et al. | Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research. |
| 2025-10-24 | [Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](http://arxiv.org/abs/2510.21189v1) | Yukun Jiang, Mingjie Li et al. | Despite their superior performance on a wide range of domains, large language models (LLMs) remain vulnerable to misuse for generating harmful content, a risk that has been further amplified by various jailbreak attacks. Existing jailbreak attacks mainly follow sequential logic, where LLMs understand and answer each given task one by one. However, concurrency, a natural extension of the sequential scenario, has been largely overlooked. In this work, we first propose a word-level method to enable task concurrency in LLMs, where adjacent words encode divergent intents. Although LLMs maintain strong utility in answering concurrent tasks, which is demonstrated by our evaluations on mathematical and general question-answering benchmarks, we notably observe that combining a harmful task with a benign one significantly reduces the probability of it being filtered by the guardrail, showing the potential risks associated with concurrency in LLMs. Based on these findings, we introduce $\texttt{JAIL-CON}$, an iterative attack framework that $\underline{\text{JAIL}}$breaks LLMs via task $\underline{\text{CON}}$currency. Experiments on widely-used LLMs demonstrate the strong jailbreak capabilities of $\texttt{JAIL-CON}$ compared to existing attacks. Furthermore, when the guardrail is applied as a defense, compared to the sequential answers generated by previous attacks, the concurrent answers in our $\texttt{JAIL-CON}$ exhibit greater stealthiness and are less detectable by the guardrail, highlighting the unique feature of task concurrency in jailbreaking LLMs. |
| 2025-10-24 | [Quantifying CBRN Risk in Frontier Models](http://arxiv.org/abs/2510.21133v1) | Divyanshu Kumar, Nitin Aravind Birur et al. | Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities. |
| 2025-10-24 | [SafetyPairs: Isolating Safety Critical Image Features with Counterfactual Image Generation](http://arxiv.org/abs/2510.21120v1) | Alec Helbling, Shruti Palaskar et al. | What exactly makes a particular image unsafe? Systematically differentiating between benign and problematic images is a challenging problem, as subtle changes to an image, such as an insulting gesture or symbol, can drastically alter its safety implications. However, existing image safety datasets are coarse and ambiguous, offering only broad safety labels without isolating the specific features that drive these differences. We introduce SafetyPairs, a scalable framework for generating counterfactual pairs of images, that differ only in the features relevant to the given safety policy, thus flipping their safety label. By leveraging image editing models, we make targeted changes to images that alter their safety labels while leaving safety-irrelevant details unchanged. Using SafetyPairs, we construct a new safety benchmark, which serves as a powerful source of evaluation data that highlights weaknesses in vision-language models' abilities to distinguish between subtly different images. Beyond evaluation, we find our pipeline serves as an effective data augmentation strategy that improves the sample efficiency of training lightweight guard models. We release a benchmark containing over 3,020 SafetyPair images spanning a diverse taxonomy of 9 safety categories, providing the first systematic resource for studying fine-grained image safety distinctions. |
| 2025-10-24 | [Deep learning-based automated damage detection in concrete structures using images from earthquake events](http://arxiv.org/abs/2510.21063v1) | Abdullah Turer, Yongsheng Bai et al. | Timely assessment of integrity of structures after seismic events is crucial for public safety and emergency response. This study focuses on assessing the structural damage conditions using deep learning methods to detect exposed steel reinforcement in concrete buildings and bridges after large earthquakes. Steel bars are typically exposed after concrete spalling or large flexural or shear cracks. The amount and distribution of exposed steel reinforcement is an indication of structural damage and degradation. To automatically detect exposed steel bars, new datasets of images collected after the 2023 Turkey Earthquakes were labeled to represent a wide variety of damaged concrete structures. The proposed method builds upon a deep learning framework, enhanced with fine-tuning, data augmentation, and testing on public datasets. An automated classification framework is developed that can be used to identify inside/outside buildings and structural components. Then, a YOLOv11 (You Only Look Once) model is trained to detect cracking and spalling damage and exposed bars. Another YOLO model is finetuned to distinguish different categories of structural damage levels. All these trained models are used to create a hybrid framework to automatically and reliably determine the damage levels from input images. This research demonstrates that rapid and automated damage detection following disasters is achievable across diverse damage contexts by utilizing image data collection, annotation, and deep learning approaches. |

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



