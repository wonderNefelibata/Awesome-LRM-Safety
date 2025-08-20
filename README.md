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
| 2025-08-19 | [Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](http://arxiv.org/abs/2508.14031v1) | Dongyoon Hahm, Taywon Min et al. | Beyond simple text generation, Large Language Models (LLMs) have evolved into agentic systems capable of planning and interacting with external tools to solve complex tasks. This evolution involves fine-tuning LLMs on agent-specific tasks to enhance their proficiency. However, safety concerns are frequently overlooked during this fine-tuning process. In this work, we show that aligned LLMs can become unintentionally misaligned, leading to a higher likelihood of executing harmful tasks and a reduced tendency to refuse them when fine-tuned to execute agentic tasks. To address these safety challenges, we propose Prefix INjection Guard (PING), a simple yet effective method that prepends automatically generated natural language prefixes to agent responses, guiding them to refuse harmful requests while preserving performance on benign tasks. Specifically, we introduce an iterative approach that alternates between (1) generating candidate prefixes and (2) selecting those that optimize both task performance and refusal behavior. Experimental results demonstrate that PING significantly enhances the safety of fine-tuned LLM agents without sacrificing their effectiveness. PING consistently outperforms existing prompting approaches across diverse benchmarks in both web navigation and code generation tasks. Our analysis of internal hidden states via linear probes reveals that prefix tokens are crucial for behavior modification, explaining the performance gains. WARNING: This paper contains contents that are unethical or offensive in nature. |
| 2025-08-19 | [Physics-Based 3D Simulation for Synthetic Data Generation and Failure Analysis in Packaging Stability Assessment](http://arxiv.org/abs/2508.13989v1) | Samuel Seligardi, Pietro Musoni et al. | The design and analysis of pallet setups are essential for ensuring safety of packages transportation. With rising demands in the logistics sector, the development of automated systems utilizing advanced technologies has become increasingly crucial. Moreover, the widespread use of plastic wrapping has motivated researchers to investigate eco-friendly alternatives that still adhere to safety standards. We present a fully controllable and accurate physical simulation system capable of replicating the behavior of moving pallets. It features a 3D graphics-based virtual environment that supports a wide range of configurations, including variable package layouts, different wrapping materials, and diverse dynamic conditions. This innovative approach reduces the need for physical testing, cutting costs and environmental impact while improving measurement accuracy for analyzing pallet dynamics. Additionally, we train a deep neural network to evaluate the rendered videos generated by our simulator, as a crash-test predictor for pallet configurations, further enhancing the system's utility in safety analysis. |
| 2025-08-19 | [Toward an Interaction-Centered Approach to Robot Trustworthiness](http://arxiv.org/abs/2508.13976v1) | Carlo Mazzola, Hassan Ali et al. | As robots get more integrated into human environments, fostering trustworthiness in embodied robotic agents becomes paramount for an effective and safe human-robot interaction (HRI). To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. In this position paper, we outline an interaction-based framework for building trust through mutual understanding between humans and robots. We emphasize two main pillars: human awareness and transparency, referring to the robot ability to interpret human actions accurately and to clearly communicate its intentions and goals, respectively. By integrating these two pillars, robots can behave in a manner that aligns with human expectations and needs while providing their human partners with both comprehension and control over their actions. We also introduce four components that we think are important for bridging the gap between a human-perceived sense of trust and a robot true capabilities. |
| 2025-08-19 | [RotBench: Evaluating Multimodal Large Language Models on Identifying Image Rotation](http://arxiv.org/abs/2508.13968v1) | Tianyi Niu, Jaemin Cho et al. | We investigate to what extent Multimodal Large Language Models (MLLMs) can accurately identify the orientation of input images rotated 0{\deg}, 90{\deg}, 180{\deg}, and 270{\deg}. This task demands robust visual reasoning capabilities to detect rotational cues and contextualize spatial relationships within images, regardless of their orientation. To evaluate MLLMs on these abilities, we introduce RotBench -- a 350-image manually-filtered benchmark comprising lifestyle, portrait, and landscape images. Despite the relatively simple nature of this task, we show that several state-of-the-art open and proprietary MLLMs, including GPT-5, o3, and Gemini-2.5-Pro, do not reliably identify rotation in input images. Providing models with auxiliary information -- including captions, depth maps, and more -- or using chain-of-thought prompting offers only small and inconsistent improvements. Our results indicate that most models are able to reliably identify right-side-up (0{\deg}) images, while certain models are able to identify upside-down (180{\deg}) images. None can reliably distinguish between 90{\deg} and 270{\deg}. Simultaneously showing the image rotated in different orientations leads to moderate performance gains for reasoning models, while a modified setup using voting improves the performance of weaker models. We further show that fine-tuning does not improve models' ability to distinguish 90{\deg} and 270{\deg} rotations, despite substantially improving the identification of 180{\deg} images. Together, these results reveal a significant gap between MLLMs' spatial reasoning capabilities and human perception in identifying rotation. |
| 2025-08-19 | [A Mechanism for Mutual Fairness in Cooperative Games with Replicable Resources -- Extended Version](http://arxiv.org/abs/2508.13960v1) | Björn Filter, Ralf Möller et al. | The latest developments in AI focus on agentic systems where artificial and human agents cooperate to realize global goals. An example is collaborative learning, which aims to train a global model based on data from individual agents. A major challenge in designing such systems is to guarantee safety and alignment with human values, particularly a fair distribution of rewards upon achieving the global goal. Cooperative game theory offers useful abstractions of cooperating agents via value functions, which assign value to each coalition, and via reward functions. With these, the idea of fair allocation can be formalized by specifying fairness axioms and designing concrete mechanisms. Classical cooperative game theory, exemplified by the Shapley value, does not fully capture scenarios like collaborative learning, as it assumes nonreplicable resources, whereas data and models can be replicated. Infinite replicability requires a generalized notion of fairness, formalized through new axioms and mechanisms. These must address imbalances in reciprocal benefits among participants, which can lead to strategic exploitation and unfair allocations. The main contribution of this paper is a mechanism and a proof that it fulfills the property of mutual fairness, formalized by the Balanced Reciprocity Axiom. It ensures that, for every pair of players, each benefits equally from the participation of the other. |
| 2025-08-19 | [Comparing Conditional Diffusion Models for Synthesizing Contrast-Enhanced Breast MRI from Pre-Contrast Images](http://arxiv.org/abs/2508.13776v1) | Sebastian Ibarra, Javier del Riego et al. | Dynamic contrast-enhanced (DCE) MRI is essential for breast cancer diagnosis and treatment. However, its reliance on contrast agents introduces safety concerns, contraindications, increased cost, and workflow complexity. To this end, we present pre-contrast conditioned denoising diffusion probabilistic models to synthesize DCE-MRI, introducing, evaluating, and comparing a total of 22 generative model variants in both single-breast and full breast settings. Towards enhancing lesion fidelity, we introduce both tumor-aware loss functions and explicit tumor segmentation mask conditioning. Using a public multicenter dataset and comparing to respective pre-contrast baselines, we observe that subtraction image-based models consistently outperform post-contrast-based models across five complementary evaluation metrics. Apart from assessing the entire image, we also separately evaluate the region of interest, where both tumor-aware losses and segmentation mask inputs improve evaluation metrics. The latter notably enhance qualitative results capturing contrast uptake, albeit assuming access to tumor localization inputs that are not guaranteed to be available in screening settings. A reader study involving 2 radiologists and 4 MRI technologists confirms the high realism of the synthetic images, indicating an emerging clinical potential of generative contrast-enhancement. We share our codebase at https://github.com/sebastibar/conditional-diffusion-breast-MRI. |
| 2025-08-19 | [ViExam: Are Vision Language Models Better than Humans on Vietnamese Multimodal Exam Questions?](http://arxiv.org/abs/2508.13680v1) | Vy Tuong Dang, An Vo et al. | Vision language models (VLMs) demonstrate remarkable capabilities on English multimodal tasks, but their performance on low-resource languages with genuinely multimodal educational content remains largely unexplored. In this work, we test how VLMs perform on Vietnamese educational assessments, investigating whether VLMs trained predominantly on English data can handle real-world cross-lingual multimodal reasoning. Our work presents the first comprehensive evaluation of VLM capabilities on multimodal Vietnamese exams through proposing ViExam, a benchmark containing 2,548 multimodal questions. We find that state-of-the-art VLMs achieve only 57.74% while open-source models achieve 27.70% mean accuracy across 7 academic domains, including Mathematics, Physics, Chemistry, Biology, Geography, Driving Test, and IQ Test. Most VLMs underperform average human test-takers (66.54%), with only the thinking VLM o3 (74.07%) exceeding human average performance, yet still falling substantially short of human best performance (99.60%). Cross-lingual prompting with English instructions while maintaining Vietnamese content fails to improve performance, decreasing accuracy by 1 percentage point for SOTA VLMs. Human-in-the-loop collaboration can partially improve VLM performance by 5 percentage points. Code and data are available at: https://vi-exam.github.io. |
| 2025-08-19 | [Nonlinear stochastic trajectory optimization](http://arxiv.org/abs/2508.13677v1) | Thomas Caleb, Roberto Armellin et al. | Designing robust space trajectories in nonlinear dynamical environments, such as the Earth-Moon circular restricted three-body problem (CR3BP), poses significant challenges due to sensitivity to initial conditions and non-Gaussian uncertainty propagation. This work introduces a novel solver for discrete-time chance-constrained trajectory optimization under uncertainty, referred to as stochastic optimization with differential algebra (SODA). SODA combines differential algebra (DA) with adaptive Gaussian mixture decomposition to efficiently propagate non-Gaussian uncertainties, and enforces Gaussian multidimensional chance constraints using scalable, less conservative transcription methods. We further introduce an efficient and accurate failure risk estimation technique, coupled with a risk allocation strategy across mixture components that enables tight and adaptive distribution of safety margins. The framework is validated on five trajectory design problems of increasing dynamical complexity, from heliocentric transfers to challenging Earth-Moon CR3BP scenarios. A linear variant, the linear stochastic optimization with differential algebra (L-SODA) solver, recovers deterministic performance with minimal overhead under small uncertainties, while the nonlinear SODA solver yields improved robustness and tighter constraint satisfaction in strongly nonlinear regimes. Results highlight SODA's ability to generate accurate, robust, and computationally tractable solutions, supporting its potential for future use in uncertainty-aware space mission design. |
| 2025-08-19 | [Input Time Scaling](http://arxiv.org/abs/2508.13654v1) | Rapheal Huang, Weilong Guo | Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we combine meta-knowledge from LLMs to refine inputs with different strategies. We also find a new phenomenon, training-testing co-design there. We need to apply query strategies during both training and testing. Only applying strategies on training or testing would seriously degrade the performance. We are also surprised to find that seemingly low data quality datasets can gain high performance. Adding irrelevant information to the queries, randomly selecting examples from a minimally filtered dataset, can even perform the best. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, simple dataset size scaling should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. A small set of examples is enough to evoke high-level reasoning ability. With experiments on models trained on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the best result would be 86.7% on AIME24 and 76.7% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints. |
| 2025-08-19 | [CRISP: Persistent Concept Unlearning via Sparse Autoencoders](http://arxiv.org/abs/2508.13650v1) | Tomer Ashuach, Dana Arad et al. | As large language models (LLMs) are increasingly deployed in real-world applications, the need to selectively remove unwanted knowledge while preserving model utility has become paramount. Recent work has explored sparse autoencoders (SAEs) to perform precise interventions on monosemantic features. However, most SAE-based methods operate at inference time, which does not create persistent changes in the model's parameters. Such interventions can be bypassed or reversed by malicious actors with parameter access. We introduce CRISP, a parameter-efficient method for persistent concept unlearning using SAEs. CRISP automatically identifies salient SAE features across multiple layers and suppresses their activations. We experiment with two LLMs and show that our method outperforms prior approaches on safety-critical unlearning tasks from the WMDP benchmark, successfully removing harmful knowledge while preserving general and in-domain capabilities. Feature-level analysis reveals that CRISP achieves semantically coherent separation between target and benign concepts, allowing precise suppression of the target features. |
| 2025-08-19 | [Modular Multiparty Sessions with Mixed Choice](http://arxiv.org/abs/2508.13616v1) | Franco Barbanera, Mariangiola Dezani-Ciancaglini | MultiParty Session Types (MPST) provide a useful framework for safe concurrent systems. Mixed choice (enabling a participant to play at the same time the roles of sender and receiver) increases the expressive power of MPST as well as the difficulty in controlling safety of communications. Such a control is more viable when modular systems are considered and the power of mixed choice fully exploited only inside loosely coupled modules. We carry over such idea in a type assignment approach to multiparty sessions. Typability for modular sessions entails Subject Reductions, Session Fidelity and Lock Freedom. |
| 2025-08-19 | [Reactive Semantics for User Interface Description Languages](http://arxiv.org/abs/2508.13610v1) | Basile Pesin, Celia Picard et al. | User Interface Description Languages (UIDLs) are high-level languages that facilitate the development of Human-Machine Interfaces, such as Graphical User Interface (GUI) applications. They usually provide first-class primitives to specify how the program reacts to an external event (user input, network message), and how data flows through the program. Although these domain-specific languages are now widely used to implement safety-critical GUIs, little work has been invested in their formalization and verification.   In this paper, we propose a denotational semantic model for a core reactive UIDL, Smalite, which we argue is expressive enough to encode constructs from more realistic languages. This preliminary work may be used as a stepping stone to produce a formally verified compiler for UIDLs. |
| 2025-08-19 | [Towards safe control parameter tuning in distributed multi-agent systems](http://arxiv.org/abs/2508.13608v1) | Abdullah Tokmak, Thomas B. Schön et al. | Many safety-critical real-world problems, such as autonomous driving and collaborative robots, are of a distributed multi-agent nature. To optimize the performance of these systems while ensuring safety, we can cast them as distributed optimization problems, where each agent aims to optimize their parameters to maximize a coupled reward function subject to coupled constraints. Prior work either studies a centralized setting, does not consider safety, or struggles with sample efficiency. Since we require sample efficiency and work with unknown and nonconvex rewards and constraints, we solve this optimization problem using safe Bayesian optimization with Gaussian process regression. Moreover, we consider nearest-neighbor communication between the agents. To capture the behavior of non-neighboring agents, we reformulate the static global optimization problem as a time-varying local optimization problem for each agent, essentially introducing time as a latent variable. To this end, we propose a custom spatio-temporal kernel to integrate prior knowledge. We show the successful deployment of our algorithm in simulations. |
| 2025-08-19 | [The 9th AI City Challenge](http://arxiv.org/abs/2508.13564v1) | Zheng Tang, Shuo Wang et al. | The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks. |
| 2025-08-19 | [LM Agents May Fail to Act on Their Own Risk Knowledge](http://arxiv.org/abs/2508.13465v1) | Yuzhi Tang, Tianxiao Li et al. | Language model (LM) agents have demonstrated significant potential for automating real-world tasks, yet they pose a diverse array of potential, severe risks in safety-critical scenarios. In this work, we identify a significant gap between LM agents' risk awareness and safety execution abilities: while they often answer "Yes" to queries like "Is executing `sudo rm -rf /*' dangerous?", they will likely fail to identify such risks in instantiated trajectories or even directly perform these risky actions when acting as agents. To systematically investigate this, we develop a comprehensive evaluation framework to examine agents' safety across three progressive dimensions: 1) their knowledge about potential risks, 2) their ability to identify corresponding risks in execution trajectories, and 3) their actual behaviors to avoid executing these risky actions. Our evaluation reveals two critical performance gaps that resemble the generator-validator gaps observed in LMs: while agents demonstrate near-perfect risk knowledge ($>98\%$ pass rates), they fail to apply this knowledge when identifying risks in actual scenarios (with performance dropping by $>23\%$) and often still execute risky actions ($<26\%$ pass rates). Notably, this trend persists across more capable LMs as well as in specialized reasoning models like DeepSeek-R1, indicating that simply scaling model capabilities or inference compute does not inherently resolve safety concerns. Instead, we take advantage of these observed gaps to develop a risk verifier that independently critiques the proposed actions by agents, with an abstractor that converts specific execution trajectories into abstract descriptions where LMs can more effectively identify the risks. Our overall system achieves a significant reduction of risky action execution by $55.3\%$ over vanilla-prompted agents. |
| 2025-08-19 | [Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms](http://arxiv.org/abs/2508.13459v1) | Rohan Chandra, Shubham Singh et al. | The ``Last Mile Challenge'' has long been considered an important, yet unsolved, challenge for autonomous vehicles, public service robots, and delivery robots. A central issue in this challenge is the ability of robots to navigate constrained and cluttered environments (e.g., doorways, hallways, corridor intersections), often while competing for space with other robots and humans. We refer to these environments as ``Social Mini-Games'' (SMGs). SMGs are tightly coupled, high-agency interactions that arise within general multi-robot navigation (MRN) scenarios. They are identified through certain distinct characteristics and require specialized metrics to evaluate them. Traditional navigation approaches designed for MRN do not perform well in SMGs, which has led to focused research on dedicated SMG solvers (navigation methods specialized to navigate in SMGs), which has flourished in recent years. However, publications on SMG navigation research make different assumptions (on centralized versus decentralized, observability, communication, cooperation, etc.), and have different objective functions (safety versus liveness). These assumptions and objectives are sometimes implicitly assumed or described informally. This makes it difficult to establish appropriate baselines for comparison in research papers, as well as making it difficult for practitioners to find the papers relevant to their concrete application. Such ad-hoc representation of the field also presents a barrier to new researchers wanting to start research in this area. SMG navigation research requires its own taxonomy, definitions, and evaluation protocols to guide effective research moving forward. This survey is the first to catalog SMG solvers using a well-defined and unified taxonomy and to classify existing methods accordingly. |
| 2025-08-19 | [Structured Prompting and Multi-Agent Knowledge Distillation for Traffic Video Interpretation and Risk Inference](http://arxiv.org/abs/2508.13439v1) | Yunxiang Yang, Ningning Xu et al. | Comprehensive highway scene understanding and robust traffic risk inference are vital for advancing Intelligent Transportation Systems (ITS) and autonomous driving. Traditional approaches often struggle with scalability and generalization, particularly under the complex and dynamic conditions of real-world environments. To address these challenges, we introduce a novel structured prompting and knowledge distillation framework that enables automatic generation of high-quality traffic scene annotations and contextual risk assessments. Our framework orchestrates two large Vision-Language Models (VLMs): GPT-4o and o3-mini, using a structured Chain-of-Thought (CoT) strategy to produce rich, multi-perspective outputs. These outputs serve as knowledge-enriched pseudo-annotations for supervised fine-tuning of a much smaller student VLM. The resulting compact 3B-scale model, named VISTA (Vision for Intelligent Scene and Traffic Analysis), is capable of understanding low-resolution traffic videos and generating semantically faithful, risk-aware captions. Despite its significantly reduced parameter count, VISTA achieves strong performance across established captioning metrics (BLEU-4, METEOR, ROUGE-L, and CIDEr) when benchmarked against its teacher models. This demonstrates that effective knowledge distillation and structured multi-agent supervision can empower lightweight VLMs to capture complex reasoning capabilities. The compact architecture of VISTA facilitates efficient deployment on edge devices, enabling real-time risk monitoring without requiring extensive infrastructure upgrades. |
| 2025-08-18 | [AIM 2025 Rip Current Segmentation (RipSeg) Challenge Report](http://arxiv.org/abs/2508.13401v1) | Andrei Dumitriu, Florin Miron et al. | This report presents an overview of the AIM 2025 RipSeg Challenge, a competition designed to advance techniques for automatic rip current segmentation in still images. Rip currents are dangerous, fast-moving flows that pose a major risk to beach safety worldwide, making accurate visual detection an important and underexplored research task. The challenge builds on RipVIS, the largest available rip current dataset, and focuses on single-class instance segmentation, where precise delineation is critical to fully capture the extent of rip currents. The dataset spans diverse locations, rip current types, and camera orientations, providing a realistic and challenging benchmark.   In total, $75$ participants registered for this first edition, resulting in $5$ valid test submissions. Teams were evaluated on a composite score combining $F_1$, $F_2$, $AP_{50}$, and $AP_{[50:95]}$, ensuring robust and application-relevant rankings. The top-performing methods leveraged deep learning architectures, domain adaptation techniques, pretrained models, and domain generalization strategies to improve performance under diverse conditions.   This report outlines the dataset details, competition framework, evaluation metrics, and final results, providing insights into the current state of rip current segmentation. We conclude with a discussion of key challenges, lessons learned from the submissions, and future directions for expanding RipSeg. |
| 2025-08-18 | [Stands to Reason: Investigating the Effect of Reasoning on Idiomaticity Detection](http://arxiv.org/abs/2508.13365v1) | Dylan Phelps, Rodrigo Wilkens et al. | The recent trend towards utilisation of reasoning models has improved the performance of Large Language Models (LLMs) across many tasks which involve logical steps. One linguistic task that could benefit from this framing is idiomaticity detection, as a potentially idiomatic expression must first be understood before it can be disambiguated and serves as a basis for reasoning. In this paper, we explore how reasoning capabilities in LLMs affect idiomaticity detection performance and examine the effect of model size. We evaluate, as open source representative models, the suite of DeepSeek-R1 distillation models ranging from 1.5B to 70B parameters across four idiomaticity detection datasets. We find the effect of reasoning to be smaller and more varied than expected. For smaller models, producing chain-of-thought (CoT) reasoning increases performance from Math-tuned intermediate models, but not to the levels of the base models, whereas larger models (14B, 32B, and 70B) show modest improvements. Our in-depth analyses reveal that larger models demonstrate good understanding of idiomaticity, successfully producing accurate definitions of expressions, while smaller models often fail to output the actual meaning. For this reason, we also experiment with providing definitions in the prompts of smaller models, which we show can improve performance in some cases. |
| 2025-08-18 | [Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](http://arxiv.org/abs/2508.13305v1) | Minhao Xiong, Zichen Wen et al. | Vision-Language Models (VLMs) have emerged as a promising paradigm in autonomous driving (AD), offering a unified framework for perception, reasoning, and decision-making by jointly modeling visual inputs and natural language instructions. However, their deployment is hindered by the significant computational overhead incurred when processing high-resolution, multi-view images, a standard setup in AD systems with six or more synchronized cameras. This overhead stems from the large number of visual tokens generated during encoding, increasing inference latency and memory consumption due to the quadratic complexity of self-attention. To address these challenges, we propose Prune2Drive, a plug-and-play visual token pruning framework for multi-view VLMs in autonomous driving. Prune2Drive introduces two core innovations: (i) a diversity-aware token selection mechanism inspired by farthest point sampling, which prioritizes semantic and spatial coverage across views rather than relying solely on attention scores, and (ii) a view-adaptive pruning controller that learns optimal pruning ratios for each camera view based on their importance to downstream driving tasks. Unlike prior methods, Prune2Drive does not require model retraining or access to attention maps, making it compatible with modern efficient attention implementations. Extensive experiments on two large-scale multi-view driving benchmarks, DriveLM and DriveLMM-o1, show that Prune2Drive achieves significant speedups and memory savings while maintaining or improving task performance. When retaining only 10% of the visual tokens, our method achieves a 6.40$\times$ speedup in the prefilling phase and consumes 13.4% of the original FLOPs, with only a 3% performance drop on the DriveLM benchmark. |

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



