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
| 2025-09-25 | [RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards](http://arxiv.org/abs/2509.21319v1) | Zhilin Wang, Jiaqi Zeng et al. | Reinforcement Learning with Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) are the main RL paradigms used in LLM post-training, each offering distinct advantages. However, RLHF struggles with interpretability and reward hacking because it relies on human judgments that usually lack explicit criteria, whereas RLVR is limited in scope by its focus on correctness-based verifiers. We propose Reinforcement Learning with Binary Flexible Feedback (RLBFF), which combines the versatility of human-driven preferences with the precision of rule-based verification, enabling reward models to capture nuanced aspects of response quality beyond mere correctness. RLBFF extracts principles that can be answered in a binary fashion (e.g. accuracy of information: yes, or code readability: no) from natural language feedback. Such principles can then be used to ground Reward Model training as an entailment task (response satisfies or does not satisfy an arbitrary principle). We show that Reward Models trained in this manner can outperform Bradley-Terry models when matched for data and achieve top performance on RM-Bench (86.2%) and JudgeBench (81.4%, #1 on leaderboard as of September 24, 2025). Additionally, users can specify principles of interest at inference time to customize the focus of our reward models, in contrast to Bradley-Terry models. Finally, we present a fully open source recipe (including data) to align Qwen3-32B using RLBFF and our Reward Model, to match or exceed the performance of o3-mini and DeepSeek R1 on general alignment benchmarks of MT-Bench, WildBench, and Arena Hard v2 (at <5% of the inference cost). |
| 2025-09-25 | [Semantic Edge-Cloud Communication for Real-Time Urban Traffic Surveillance with ViT and LLMs over Mobile Networks](http://arxiv.org/abs/2509.21259v1) | Murat Arda Onsu, Poonam Lohan et al. | Real-time urban traffic surveillance is vital for Intelligent Transportation Systems (ITS) to ensure road safety, optimize traffic flow, track vehicle trajectories, and prevent collisions in smart cities. Deploying edge cameras across urban environments is a standard practice for monitoring road conditions. However, integrating these with intelligent models requires a robust understanding of dynamic traffic scenarios and a responsive interface for user interaction. Although multimodal Large Language Models (LLMs) can interpret traffic images and generate informative responses, their deployment on edge devices is infeasible due to high computational demands. Therefore, LLM inference must occur on the cloud, necessitating visual data transmission from edge to cloud, a process hindered by limited bandwidth, leading to potential delays that compromise real-time performance. To address this challenge, we propose a semantic communication framework that significantly reduces transmission overhead. Our method involves detecting Regions of Interest (RoIs) using YOLOv11, cropping relevant image segments, and converting them into compact embedding vectors using a Vision Transformer (ViT). These embeddings are then transmitted to the cloud, where an image decoder reconstructs the cropped images. The reconstructed images are processed by a multimodal LLM to generate traffic condition descriptions. This approach achieves a 99.9% reduction in data transmission size while maintaining an LLM response accuracy of 89% for reconstructed cropped images, compared to 93% accuracy with original cropped images. Our results demonstrate the efficiency and practicality of ViT and LLM-assisted edge-cloud semantic communication for real-time traffic surveillance. |
| 2025-09-25 | [humancompatible.train: Implementing Optimization Algorithms for Stochastically-Constrained Stochastic Optimization Problems](http://arxiv.org/abs/2509.21254v1) | Andrii Kliachkin, Jana Lepšová et al. | There has been a considerable interest in constrained training of deep neural networks (DNNs) recently for applications such as fairness and safety. Several toolkits have been proposed for this task, yet there is still no industry standard. We present humancompatible.train (https://github.com/humancompatible/train), an easily-extendable PyTorch-based Python package for training DNNs with stochastic constraints. We implement multiple previously unimplemented algorithms for stochastically constrained stochastic optimization. We demonstrate the toolkit use by comparing two algorithms on a deep learning task with fairness constraints. |
| 2025-09-25 | [From Physics to Machine Learning and Back: Part II - Learning and Observational Bias in PHM](http://arxiv.org/abs/2509.21207v1) | Olga Fink, Ismail Nejjar et al. | Prognostics and Health Management ensures the reliability, safety, and efficiency of complex engineered systems by enabling fault detection, anticipating equipment failures, and optimizing maintenance activities throughout an asset lifecycle. However, real-world PHM presents persistent challenges: sensor data is often noisy or incomplete, available labels are limited, and degradation behaviors and system interdependencies can be highly complex and nonlinear. Physics-informed machine learning has emerged as a promising approach to address these limitations by embedding physical knowledge into data-driven models. This review examines how incorporating learning and observational biases through physics-informed modeling and data strategies can guide models toward physically consistent and reliable predictions. Learning biases embed physical constraints into model training through physics-informed loss functions and governing equations, or by incorporating properties like monotonicity. Observational biases influence data selection and synthesis to ensure models capture realistic system behavior through virtual sensing for estimating unmeasured states, physics-based simulation for data augmentation, and multi-sensor fusion strategies. The review then examines how these approaches enable the transition from passive prediction to active decision-making through reinforcement learning, which allows agents to learn maintenance policies that respect physical constraints while optimizing operational objectives. This closes the loop between model-based predictions, simulation, and actual system operation, empowering adaptive decision-making. Finally, the review addresses the critical challenge of scaling PHM solutions from individual assets to fleet-wide deployment. Fast adaptation methods including meta-learning and few-shot learning are reviewed alongside domain generalization techniques ... |
| 2025-09-25 | [Can Less Precise Be More Reliable? A Systematic Evaluation of Quantization's Impact on CLIP Beyond Accuracy](http://arxiv.org/abs/2509.21173v1) | Aymen Bouguerra, Daniel Montoya et al. | The powerful zero-shot generalization capabilities of vision-language models (VLMs) like CLIP have enabled new paradigms for safety-related tasks such as out-of-distribution (OOD) detection. However, additional aspects crucial for the computationally efficient and reliable deployment of CLIP are still overlooked. In particular, the impact of quantization on CLIP's performance beyond accuracy remains underexplored. This work presents a large-scale evaluation of quantization on CLIP models, assessing not only in-distribution accuracy but a comprehensive suite of reliability metrics and revealing counterintuitive results driven by pre-training source. We demonstrate that quantization consistently improves calibration for typically underconfident pre-trained models, while often degrading it for overconfident variants. Intriguingly, this degradation in calibration does not preclude gains in other reliability metrics; we find that OOD detection can still improve for these same poorly calibrated models. Furthermore, we identify specific quantization-aware training (QAT) methods that yield simultaneous gains in zero-shot accuracy, calibration, and OOD robustness, challenging the view of a strict efficiency-performance trade-off. These findings offer critical insights for navigating the multi-objective problem of deploying efficient, reliable, and robust VLMs by utilizing quantization beyond its conventional role. |
| 2025-09-25 | [Fine-Tuning LLMs to Analyze Multiple Dimensions of Code Review: A Maximum Entropy Regulated Long Chain-of-Thought Approach](http://arxiv.org/abs/2509.21170v1) | Yongda Yu, Guohao Shi et al. | Large Language Models (LLMs) have shown great potential in supporting automated code review due to their impressive capabilities in context understanding and reasoning. However, these capabilities are still limited compared to human-level cognition because they are heavily influenced by the training data. Recent research has demonstrated significantly improved performance through fine-tuning LLMs with code review data. However, compared to human reviewers who often simultaneously analyze multiple dimensions of code review to better identify issues, the full potential of these methods is hampered by the limited or vague information used to fine-tune the models. This paper contributes MelcotCR, a chain-of-thought (COT) fine-tuning approach that trains LLMs with an impressive reasoning ability to analyze multiple dimensions of code review by harnessing long COT techniques to provide rich structured information. To address context loss and reasoning logic loss issues that frequently occur when LLMs process long COT prompts, we propose a solution that combines the Maximum Entropy (ME) modeling principle with pre-defined reasoning pathways in MelcotCR to enable more effective utilization of in-context knowledge within long COT prompts while strengthening the logical tightness of the reasoning process. Empirical evaluations on our curated MelcotCR dataset and the public CodeReviewer dataset reveal that a low-parameter base model, such as 14B Qwen2.5, fine-tuned with MelcotCR can surpass state-of-the-art methods in terms of the accuracy of detecting and describing code issues, with its performance remarkably on par with that of the 671B DeepSeek-R1 model. |
| 2025-09-25 | [Roadmap towards Personalized Approaches and Safety Considerations in Non-Ionizing Radiation: From Dosimetry to Therapeutic and Diagnostic Applications](http://arxiv.org/abs/2509.21165v1) | Ilkka Laakso, Margarethus Marius Paulides et al. | This roadmap provides a comprehensive and forward-looking perspective on the individualized application and safety of non-ionizing radiation (NIR) dosimetry in diagnostic and therapeutic medicine. Covering a wide range of frequencies, i.e., from low-frequency to terahertz, this document provides an overview of the current state of the art and anticipates future research needs in selected key topics of NIR-based medical applications. It also emphasizes the importance of personalized dosimetry, rigorous safety evaluation, and interdisciplinary collaboration to ensure safe and effective integration of NIR technologies in modern therapy and diagnosis. |
| 2025-09-25 | [DATS: Distance-Aware Temperature Scaling for Calibrated Class-Incremental Learning](http://arxiv.org/abs/2509.21161v1) | Giuseppe Serra, Florian Buettner | Continual Learning (CL) is recently gaining increasing attention for its ability to enable a single model to learn incrementally from a sequence of new classes. In this scenario, it is important to keep consistent predictive performance across all the classes and prevent the so-called Catastrophic Forgetting (CF). However, in safety-critical applications, predictive performance alone is insufficient. Predictive models should also be able to reliably communicate their uncertainty in a calibrated manner - that is, with confidence scores aligned to the true frequencies of target events. Existing approaches in CL address calibration primarily from a data-centric perspective, relying on a single temperature shared across all tasks. Such solutions overlook task-specific differences, leading to large fluctuations in calibration error across tasks. For this reason, we argue that a more principled approach should adapt the temperature according to the distance to the current task. However, the unavailability of the task information at test time/during deployment poses a major challenge to achieve the intended objective. For this, we propose Distance-Aware Temperature Scaling (DATS), which combines prototype-based distance estimation with distance-aware calibration to infer task proximity and assign adaptive temperatures without prior task information. Through extensive empirical evaluation on both standard benchmarks and real-world, imbalanced datasets taken from the biomedical domain, our approach demonstrates to be stable, reliable and consistent in reducing calibration error across tasks compared to state-of-the-art approaches. |
| 2025-09-25 | [Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](http://arxiv.org/abs/2509.21155v1) | Chantal Shaib, Vinith M. Suriyakumar et al. | For an LLM to correctly respond to an instruction it must understand both the semantics and the domain (i.e., subject area) of a given task-instruction pair. However, syntax can also convey implicit information Recent work shows that syntactic templates--frequent sequences of Part-of-Speech (PoS) tags--are prevalent in training data and often appear in model outputs. In this work we characterize syntactic templates, domain, and semantics in task-instruction pairs. We identify cases of spurious correlations between syntax and domain, where models learn to associate a domain with syntax during training; this can sometimes override prompt semantics. Using a synthetic training dataset, we find that the syntactic-domain correlation can lower performance (mean 0.51 +/- 0.06) on entity knowledge tasks in OLMo-2 models (1B-13B). We introduce an evaluation framework to detect this phenomenon in trained models, and show that it occurs on a subset of the FlanV2 dataset in open (OLMo-2-7B; Llama-4-Maverick), and closed (GPT-4o) models. Finally, we present a case study on the implications for safety finetuning, showing that unintended syntactic-domain correlations can be used to bypass refusals in OLMo-2-7B Instruct and GPT-4o. Our findings highlight two needs: (1) to explicitly test for syntactic-domain correlations, and (2) to ensure syntactic diversity in training data, specifically within domains, to prevent such spurious correlations. |
| 2025-09-25 | [Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems](http://arxiv.org/abs/2509.21143v1) | Junfeng Yan, Biao Wu et al. | Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents. |
| 2025-09-25 | [Disagreements in Reasoning: How a Model's Thinking Process Dictates Persuasion in Multi-Agent Systems](http://arxiv.org/abs/2509.21054v1) | Haodong Zhao, Jidong Li et al. | The rapid proliferation of recent Multi-Agent Systems (MAS), where Large Language Models (LLMs) and Large Reasoning Models (LRMs) usually collaborate to solve complex problems, necessitates a deep understanding of the persuasion dynamics that govern their interactions. This paper challenges the prevailing hypothesis that persuasive efficacy is primarily a function of model scale. We propose instead that these dynamics are fundamentally dictated by a model's underlying cognitive process, especially its capacity for explicit reasoning. Through a series of multi-agent persuasion experiments, we uncover a fundamental trade-off we term the Persuasion Duality. Our findings reveal that the reasoning process in LRMs exhibits significantly greater resistance to persuasion, maintaining their initial beliefs more robustly. Conversely, making this reasoning process transparent by sharing the "thinking content" dramatically increases their ability to persuade others. We further consider more complex transmission persuasion situations and reveal complex dynamics of influence propagation and decay within multi-hop persuasion between multiple agent networks. This research provides systematic evidence linking a model's internal processing architecture to its external persuasive behavior, offering a novel explanation for the susceptibility of advanced models and highlighting critical implications for the safety, robustness, and design of future MAS. |
| 2025-09-25 | [Transonic buffet and incompressible low-frequency oscillations at high Reynolds numbers](http://arxiv.org/abs/2509.21046v1) | Vishw Patel, Aman Jain et al. | Coherent, self-sustained oscillations of the flow over aircraft wings can lead to unsteady loads that detrimentally affect aircraft safety and stability, thus limiting the flight envelope. Two such types of oscillations are the low-frequency oscillations (LFO) observed in flow over airfoils close to stall in the incompressible regime and transonic buffet, which occurs at high speeds and involves oscillating shock waves. The possibility that these two are linked has been explored only recently at low Reynolds numbers (Re ~ O(1e4)) and natural transition conditions (Moise et al., J. Fluid Mech., vol. 981, 2024, p. A23). However, the shock wave structure in the transonic regime under these conditions differs substantially when compared to high Reynolds number flows, and it is unknown whether a connection can be established at high Reynolds numbers. This study investigates this possibility by performing incompressible and compressible URANS simulations at Re = 1e7. We show that transonic buffet exists for a narrow range of freestream Mach numbers across a wide range of angles of attack, and that buffet-like oscillations are observed at higher angles even in the absence of shock waves. Using a spectral proper orthogonal decomposition (SPOD), we show that the dominant modes associated with these oscillations are strongly correlated for all cases, even in the absence of shock waves. Furthermore, using a fully incompressible URANS framework, we capture LFO at the same Reynolds number and confirm the connection between these two phenomena using SPOD. These results imply that neither shock waves nor compressibility is necessary to sustain such low-frequency oscillations, suggesting that the fundamental mechanism governing them is related to flow separation. This can potentially help in improved control strategies to extend the flight envelope by mitigating buffet or LFO. |
| 2025-09-25 | [FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](http://arxiv.org/abs/2509.21029v1) | Runqi Lin, Alasdair Paren et al. | The integration of new modalities enhances the capabilities of multimodal large language models (MLLMs) but also introduces additional vulnerabilities. In particular, simple visual jailbreaking attacks can manipulate open-source MLLMs more readily than sophisticated textual attacks. However, these underdeveloped attacks exhibit extremely limited cross-model transferability, failing to reliably identify vulnerabilities in closed-source MLLMs. In this work, we analyse the loss landscape of these jailbreaking attacks and find that the generated attacks tend to reside in high-sharpness regions, whose effectiveness is highly sensitive to even minor parameter changes during transfer. To further explain the high-sharpness localisations, we analyse their feature representations in both the intermediate layers and the spectral domain, revealing an improper reliance on narrow layer representations and semantically poor frequency components. Building on this, we propose a Feature Over-Reliance CorrEction (FORCE) method, which guides the attack to explore broader feasible regions across layer features and rescales the influence of frequency features according to their semantic content. By eliminating non-generalizable reliance on both layer and spectral features, our method discovers flattened feasible regions for visual jailbreaking attacks, thereby improving cross-model transferability. Extensive experiments demonstrate that our approach effectively facilitates visual red-teaming evaluations against closed-source MLLMs. |
| 2025-09-25 | [Multi-Robot Vision-Based Task and Motion Planning for EV Battery Disassembly and Sorting](http://arxiv.org/abs/2509.21020v1) | Abdelaziz Shaarawy, Cansu Erdogan et al. | Electric-vehicle (EV) battery disassembly requires precise multi-robot coordination, short and reliable motions, and robust collision safety in cluttered, dynamic scenes. We propose a four-layer task-and-motion planning (TAMP) framework that couples symbolic task planning and cost- and accessibility-aware allocation with a TP-GMM-guided motion planner learned from demonstrations. Stereo vision with YOLOv8 provides real-time component localization, while OctoMap-based 3D mapping and FCL(Flexible Collision Library) checks in MoveIt unify predictive digital-twin collision checking with reactive, vision-based avoidance. Validated on two UR10e robots across cable, busbar, service plug, and three leaf-cell removals, the approach yields substantially more compact and safer motions than a default RRTConnect baseline under identical perception and task assignments: average end-effector path length drops by $-63.3\%$ and makespan by $-8.1\%$; per-arm swept volumes shrink (R1: $0.583\rightarrow0.139\,\mathrm{m}^3$; R2: $0.696\rightarrow0.252\,\mathrm{m}^3$), and mutual overlap decreases by $47\%$ ($0.064\rightarrow0.034\,\mathrm{m}^3$). These results highlight improved autonomy, precision, and safety for multi-robot EV battery disassembly in unstructured, dynamic environments. |
| 2025-09-25 | [The Use of the Simplex Architecture to Enhance Safety in Deep-Learning-Powered Autonomous Systems](http://arxiv.org/abs/2509.21014v1) | Federico Nesti, Niko Salamini et al. | Recently, the outstanding performance reached by neural networks in many tasks has led to their deployment in autonomous systems, such as robots and vehicles. However, neural networks are not yet trustworthy, being prone to different types of misbehavior, such as anomalous samples, distribution shifts, adversarial attacks, and other threats. Furthermore, frameworks for accelerating the inference of neural networks typically run on rich operating systems that are less predictable in terms of timing behavior and present larger surfaces for cyber-attacks.   To address these issues, this paper presents a software architecture for enhancing safety, security, and predictability levels of learning-based autonomous systems. It leverages two isolated execution domains, one dedicated to the execution of neural networks under a rich operating system, which is deemed not trustworthy, and one responsible for running safety-critical functions, possibly under a different operating system capable of handling real-time constraints.   Both domains are hosted on the same computing platform and isolated through a type-1 real-time hypervisor enabling fast and predictable inter-domain communication to exchange real-time data. The two domains cooperate to provide a fail-safe mechanism based on a safety monitor, which oversees the state of the system and switches to a simpler but safer backup module, hosted in the safety-critical domain, whenever its behavior is considered untrustworthy.   The effectiveness of the proposed architecture is illustrated by a set of experiments performed on two control systems: a Furuta pendulum and a rover. The results confirm the utility of the fall-back mechanism in preventing faults due to the learning component. |
| 2025-09-25 | [A Single Neuron Works: Precise Concept Erasure in Text-to-Image Diffusion Models](http://arxiv.org/abs/2509.21008v1) | Qinqin He, Jiaqi Weng et al. | Text-to-image models exhibit remarkable capabilities in image generation. However, they also pose safety risks of generating harmful content. A key challenge of existing concept erasure methods is the precise removal of target concepts while minimizing degradation of image quality. In this paper, we propose Single Neuron-based Concept Erasure (SNCE), a novel approach that can precisely prevent harmful content generation by manipulating only a single neuron. Specifically, we train a Sparse Autoencoder (SAE) to map text embeddings into a sparse, disentangled latent space, where individual neurons align tightly with atomic semantic concepts. To accurately locate neurons responsible for harmful concepts, we design a novel neuron identification method based on the modulated frequency scoring of activation patterns. By suppressing activations of the harmful concept-specific neuron, SNCE achieves surgical precision in concept erasure with minimal disruption to image quality. Experiments on various benchmarks demonstrate that SNCE achieves state-of-the-art results in target concept erasure, while preserving the model's generation capabilities for non-target concepts. Additionally, our method exhibits strong robustness against adversarial attacks, significantly outperforming existing methods. |
| 2025-09-25 | [CORE: Full-Path Evaluation of LLM Agents Beyond Final State](http://arxiv.org/abs/2509.20998v1) | Panagiotis Michelakis, Yiannis Hadjiyiannis et al. | Evaluating AI agents that solve real-world tasks through function-call sequences remains an open challenge. Existing agentic benchmarks often reduce evaluation to a binary judgment of the final state, overlooking critical aspects such as safety, efficiency, and intermediate correctness. We propose a framework based on deterministic finite automata (DFAs) that encodes tasks as sets of valid tool-use paths, enabling principled assessment of agent behavior in diverse world models. Building on this foundation, we introduce CORE, a suite of five metrics, namely Path Correctness, Path Correctness - Kendall's tau Composite, Prefix Criticality, Harmful-Call Rate, and Efficiency, that quantify alignment with expected execution patterns. Across diverse worlds, our method reveals important performance differences between agents that would otherwise appear equivalent under traditional final-state evaluation schemes. |
| 2025-09-25 | [Analysis of instruction-based LLMs' capabilities to score and judge text-input problems in an academic setting](http://arxiv.org/abs/2509.20982v1) | Valeria Ramirez-Garcia, David de-Fitero-Dominguez et al. | Large language models (LLMs) can act as evaluators, a role studied by methods like LLM-as-a-Judge and fine-tuned judging LLMs. In the field of education, LLMs have been studied as assistant tools for students and teachers. Our research investigates LLM-driven automatic evaluation systems for academic Text-Input Problems using rubrics. We propose five evaluation systems that have been tested on a custom dataset of 110 answers about computer science from higher education students with three models: JudgeLM, Llama-3.1-8B and DeepSeek-R1-Distill-Llama-8B. The evaluation systems include: The JudgeLM evaluation, which uses the model's single answer prompt to obtain a score; Reference Aided Evaluation, which uses a correct answer as a guide aside from the original context of the question; No Reference Evaluation, which ommits the reference answer; Additive Evaluation, which uses atomic criteria; and Adaptive Evaluation, which is an evaluation done with generated criteria fitted to each question. All evaluation methods have been compared with the results of a human evaluator. Results show that the best method to automatically evaluate and score Text-Input Problems using LLMs is Reference Aided Evaluation. With the lowest median absolute deviation (0.945) and the lowest root mean square deviation (1.214) when compared to human evaluation, Reference Aided Evaluation offers fair scoring as well as insightful and complete evaluations. Other methods such as Additive and Adaptive Evaluation fail to provide good results in concise answers, No Reference Evaluation lacks information needed to correctly assess questions and JudgeLM Evaluations have not provided good results due to the model's limitations. As a result, we conclude that Artificial Intelligence-driven automatic evaluation systems, aided with proper methodologies, show potential to work as complementary tools to other academic resources. |
| 2025-09-25 | [BSB: Towards Demand-Aware Peer Selection With XOR-based Routing](http://arxiv.org/abs/2509.20974v1) | Qingyun Ji, Darya Melnyk et al. | Peer-to-peer networks, as a key enabler of modern networked and distributed systems, rely on peer-selection algorithms to optimize their scalability and performance. Peer-selection methods have been studied extensively in various aspects, including routing mechanisms and communication overhead. However, many state-of-the-art algorithms are oblivious to application-specific data traffic. This mismatch between design and demand results in underutilized connections, which inevitably leads to longer paths and increased latency. In this work, we propose a novel demand-aware peer-selection algorithm, called Binary Search in Buckets (BSB). Our demand-aware approach adheres to a local and greedy XOR-based routing mechanism, ensuring compatibility with existing protocols and mechanisms. We evaluate our solution against two prior algorithms by conducting simulations on real-world and synthetic communication network traces. The results of our evaluations show that BSB can offer up to a 43% improvement compared to two selected algorithms from the literature. |
| 2025-09-25 | [Decoding the Surgical Scene: A Scoping Review of Scene Graphs in Surgery](http://arxiv.org/abs/2509.20941v1) | Angelo Henriques, Korab Hoxha et al. | Scene graphs (SGs) provide structured relational representations crucial for decoding complex, dynamic surgical environments. This PRISMA-ScR-guided scoping review systematically maps the evolving landscape of SG research in surgery, charting its applications, methodological advancements, and future directions. Our analysis reveals rapid growth, yet uncovers a critical 'data divide': internal-view research (e.g., triplet recognition) almost exclusively uses real-world 2D video, while external-view 4D modeling relies heavily on simulated data, exposing a key translational research gap. Methodologically, the field has advanced from foundational graph neural networks to specialized foundation models that now significantly outperform generalist large vision-language models in surgical contexts. This progress has established SGs as a cornerstone technology for both analysis, such as workflow recognition and automated safety monitoring, and generative tasks like controllable surgical simulation. Although challenges in data annotation and real-time implementation persist, they are actively being addressed through emerging techniques. Surgical SGs are maturing into an essential semantic bridge, enabling a new generation of intelligent systems to improve surgical safety, efficiency, and training. |

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



