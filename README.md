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
| 2025-04-30 | [A Survey of Interactive Generative Video](http://arxiv.org/abs/2504.21853v1) | Jiwen Yu, Yiran Qin et al. | Interactive Generative Video (IGV) has emerged as a crucial technology in response to the growing demand for high-quality, interactive video content across various domains. In this paper, we define IGV as a technology that combines generative capabilities to produce diverse high-quality video content with interactive features that enable user engagement through control signals and responsive feedback. We survey the current landscape of IGV applications, focusing on three major domains: 1) gaming, where IGV enables infinite exploration in virtual worlds; 2) embodied AI, where IGV serves as a physics-aware environment synthesizer for training agents in multimodal interaction with dynamically evolving scenes; and 3) autonomous driving, where IGV provides closed-loop simulation capabilities for safety-critical testing and validation. To guide future development, we propose a comprehensive framework that decomposes an ideal IGV system into five essential modules: Generation, Control, Memory, Dynamics, and Intelligence. Furthermore, we systematically analyze the technical challenges and future directions in realizing each component for an ideal IGV system, such as achieving real-time generation, enabling open-domain control, maintaining long-term coherence, simulating accurate physics, and integrating causal reasoning. We believe that this systematic analysis will facilitate future research and development in the field of IGV, ultimately advancing the technology toward more sophisticated and practical applications. |
| 2025-04-30 | [Neuro-Symbolic Generation of Explanations for Robot Policies with Weighted Signal Temporal Logic](http://arxiv.org/abs/2504.21841v1) | Mikihisa Yuasa, Ramavarapu S. Sreenivas et al. | Neural network-based policies have demonstrated success in many robotic applications, but often lack human-explanability, which poses challenges in safety-critical deployments. To address this, we propose a neuro-symbolic explanation framework that generates a weighted signal temporal logic (wSTL) specification to describe a robot policy in a interpretable form. Existing methods typically produce explanations that are verbose and inconsistent, which hinders explainability, and loose, which do not give meaningful insights into the underlying policy. We address these issues by introducing a simplification process consisting of predicate filtering, regularization, and iterative pruning. We also introduce three novel explainability evaluation metrics -- conciseness, consistency, and strictness -- to assess explanation quality beyond conventional classification metrics. Our method is validated in three simulated robotic environments, where it outperforms baselines in generating concise, consistent, and strict wSTL explanations without sacrificing classification accuracy. This work bridges policy learning with formal methods, contributing to safer and more transparent decision-making in robotics. |
| 2025-04-30 | [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](http://arxiv.org/abs/2504.21776v1) | Xiaoxi Li, Jiajie Jin et al. | Large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, demonstrate impressive long-horizon reasoning capabilities. However, their reliance on static internal knowledge limits their performance on complex, knowledge-intensive tasks and hinders their ability to produce comprehensive research reports requiring synthesis of diverse web information. To address this, we propose \textbf{WebThinker}, a deep research agent that empowers LRMs to autonomously search the web, navigate web pages, and draft research reports during the reasoning process. WebThinker integrates a \textbf{Deep Web Explorer} module, enabling LRMs to dynamically search, navigate, and extract information from the web when encountering knowledge gaps. It also employs an \textbf{Autonomous Think-Search-and-Draft strategy}, allowing the model to seamlessly interleave reasoning, information gathering, and report writing in real time. To further enhance research tool utilization, we introduce an \textbf{RL-based training strategy} via iterative online Direct Preference Optimization (DPO). Extensive experiments on complex reasoning benchmarks (GPQA, GAIA, WebWalkerQA, HLE) and scientific report generation tasks (Glaive) demonstrate that WebThinker significantly outperforms existing methods and strong proprietary systems. Our approach enhances LRM reliability and applicability in complex scenarios, paving the way for more capable and versatile deep research systems. The code is available at https://github.com/RUC-NLPIR/WebThinker. |
| 2025-04-30 | [CodeFlowBench: A Multi-turn, Iterative Benchmark for Complex Code Generation](http://arxiv.org/abs/2504.21751v1) | Sizhe Wang, Zhengren Wang et al. | Real world development demands code that is readable, extensible, and testable by organizing the implementation into modular components and iteratively reuse pre-implemented code. We term this iterative, multi-turn process codeflow and introduce CodeFlowBench, the first benchmark designed for comprehensively evaluating LLMs' ability to perform codeflow, namely to implement new functionality by reusing existing functions over multiple turns. CodeFlowBench comprises 5258 problems drawn from Codeforces and is continuously updated via an automated pipeline that decomposes each problem into a series of function-level subproblems based on its dependency tree and each subproblem is paired with unit tests. We further propose a novel evaluation framework with tasks and metrics tailored to multi-turn code reuse to assess model performance. In experiments across various LLMs under both multi-turn and single-turn patterns. We observe models' poor performance on CodeFlowBench, with a substantial performance drop in the iterative codeflow scenario. For instance, o1-mini achieves a pass@1 of 20.8% in multi-turn pattern versus 37.8% in single-turn pattern. Further analysis shows that different models excel at different dependency depths, yet all struggle to correctly solve structurally complex problems, highlighting challenges for current LLMs to serve as code generation tools when performing codeflow. Overall, CodeFlowBench offers a comprehensive benchmark and new insights into LLM capabilities for multi-turn, iterative code generation, guiding future advances in code generation tasks. |
| 2025-04-30 | [XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs](http://arxiv.org/abs/2504.21700v1) | Marco Arazzi, Vignesh Kumar Kembu et al. | Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack. |
| 2025-04-30 | [REHEARSE-3D: A Multi-modal Emulated Rain Dataset for 3D Point Cloud De-raining](http://arxiv.org/abs/2504.21699v1) | Abu Mohammed Raisuddin, Jesper Holmblad et al. | Sensor degradation poses a significant challenge in autonomous driving. During heavy rainfall, the interference from raindrops can adversely affect the quality of LiDAR point clouds, resulting in, for instance, inaccurate point measurements. This, in turn, can potentially lead to safety concerns if autonomous driving systems are not weather-aware, i.e., if they are unable to discern such changes. In this study, we release a new, large-scale, multi-modal emulated rain dataset, REHEARSE-3D, to promote research advancements in 3D point cloud de-raining. Distinct from the most relevant competitors, our dataset is unique in several respects. First, it is the largest point-wise annotated dataset, and second, it is the only one with high-resolution LiDAR data (LiDAR-256) enriched with 4D Radar point clouds logged in both daytime and nighttime conditions in a controlled weather environment. Furthermore, REHEARSE-3D involves rain-characteristic information, which is of significant value not only for sensor noise modeling but also for analyzing the impact of weather at a point level. Leveraging REHEARSE-3D, we benchmark raindrop detection and removal in fused LiDAR and 4D Radar point clouds. Our comprehensive study further evaluates the performance of various statistical and deep-learning models. Upon publication, the dataset and benchmark models will be made publicly available at: https://sporsho.github.io/REHEARSE3D. |
| 2025-04-30 | [Using quantum annealing to generate test cases for cyber-physical systems](http://arxiv.org/abs/2504.21684v1) | Hugo Araujo, Xinyi Wang et al. | Quantum computing has emerged as a powerful tool to efficiently solve computational challenges, particularly in simulation and optimisation. However, hardware limitations prevent quantum computers from achieving the full theoretical potential. Among the quantum algorithms, quantum annealing is a prime candidate to solve optimisation problems. This makes it a natural candidate for search-based software testing in the Cyber-Physical Systems (CPS) domain, which demands effective test cases due to their safety-critical nature. This work explores the use of quantum annealing to enhance test case generation for CPS through a mutation-based approach. We encode test case mutation as a binary optimisation problem, and use quantum annealing to identify and target critical regions of the test cases for improvement. Our approach mechanises this process into an algorithm that uses D-Wave's quantum annealer to find the solution. As a main contribution, we offer insights into how quantum annealing can advance software testing methodologies by empirically evaluating the correlation between problem size, hardware limitations, and the effectiveness of the results. Moreover, we compare the proposed method against state-of-the-art classical optimisation algorithms, targeting efficiency (time to generate test cases) and effectiveness (fault detection rates). Results indicate that quantum annealing enables faster test case generation while achieving comparable fault detection performance to state-of-the-art alternatives. |
| 2025-04-30 | [Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs](http://arxiv.org/abs/2504.21680v1) | Pan Suo, Yu-Ming Shang et al. | Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions. |
| 2025-04-30 | [RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations](http://arxiv.org/abs/2504.21605v1) | Jonas Gwozdz, Andreas Both | Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study. |
| 2025-04-30 | [Toward Realization of Low-Altitude Economy Networks: Core Architecture, Integrated Technologies, and Future Directions](http://arxiv.org/abs/2504.21583v1) | Yixian Wang, Geng Sun et al. | The rise of the low-altitude economy (LAE) is propelling urban development and emerging industries by integrating advanced technologies to enhance efficiency, safety, and sustainability in low-altitude operations. The widespread adoption of unmanned aerial vehicles (UAVs) and electric vertical takeoff and landing (eVTOL) aircraft plays a crucial role in enabling key applications within LAE, such as urban logistics, emergency rescue, and aerial mobility. However, unlike traditional UAV networks, LAE networks encounter increased airspace management demands due to dense flying nodes and potential interference with ground communication systems. In addition, there are heightened and extended security risks in real-time operations, particularly the vulnerability of low-altitude aircraft to cyberattacks from ground-based threats. To address these, this paper first explores related standards and core architecture that support the development of LAE networks. Subsequently, we highlight the integration of technologies such as communication, sensing, computing, positioning, navigation, surveillance, flight control, and airspace management. This synergy of multi-technology drives the advancement of real-world LAE applications, particularly in improving operational efficiency, optimizing airspace usage, and ensuring safety. Finally, we outline future research directions for LAE networks, such as intelligent and adaptive optimization, security and privacy protection, sustainable energy and power management, quantum-driven coordination, generative governance, and three-dimensional (3D) airspace coverage, which collectively underscore the potential of collaborative technologies to advance LAE networks. |
| 2025-04-30 | [Provably-Safe, Online System Identification](http://arxiv.org/abs/2504.21486v1) | Bohao Zhang, Zichang Zhou et al. | Precise manipulation tasks require accurate knowledge of payload inertial parameters. Unfortunately, identifying these parameters for unknown payloads while ensuring that the robotic system satisfies its input and state constraints while avoiding collisions with the environment remains a significant challenge. This paper presents an integrated framework that enables robotic manipulators to safely and automatically identify payload parameters while maintaining operational safety guarantees. The framework consists of two synergistic components: an online trajectory planning and control framework that generates provably-safe exciting trajectories for system identification that can be tracked while respecting robot constraints and avoiding obstacles and a robust system identification method that computes rigorous overapproximative bounds on end-effector inertial parameters assuming bounded sensor noise. Experimental validation on a robotic manipulator performing challenging tasks with various unknown payloads demonstrates the framework's effectiveness in establishing accurate parameter bounds while maintaining safety throughout the identification process. The code is available at our project webpage: https://roahmlab.github.io/OnlineSafeSysID/. |
| 2025-04-30 | [An Intermediate Program Representation for Optimizing Stream-Based Languages](http://arxiv.org/abs/2504.21458v1) | Jan Baumeister, Arthur Correnson et al. | Stream-based runtime monitors are safety assurance tools that check at runtime whether the system's behavior satisfies a formal specification. Specifications consist of stream equations, which relate input streams, containing sensor readings and other incoming information, to output streams, representing filtered and aggregated data. This paper presents a framework for the stream-based specification language RTLola. We introduce a new intermediate representation for stream-based languages, the StreamIR, which, like the specification language, operates on streams of unbounded length; while the stream equations are replaced by imperative programs. We developed a set of optimizations based on static analysis of the specification and have implemented an interpreter and a compiler for several target languages. In our evaluation, we measure the performance of several real-world case studies. The results show that using the StreamIR framework reduces the runtime significantly compared to the existing StreamIR interpreter. We evaluate the effect of the optimizations and show that significant performance gains are possible beyond the optimizations of the target language's compiler. While our current implementation is limited to RTLola, the StreamIR is designed to accommodate other stream-based languages, enabling their interpretation and compilation into all available target languages. |
| 2025-04-30 | [Mapping the Human Brain from the Prenatal Period to Infancy Using 3D Magnetic Resonance Imaging](http://arxiv.org/abs/2504.21406v1) | Arnaud Cachia, Jean-François Mangin et al. | Human brain development is a complex and dynamic process that begins during the first weeks of pregnancy and lasts until early adulthood. This chapter focuses on the developmental window from prenatal period to infancy, probably the most dynamic period across the entire lifespan. The availability of non-invasive three-dimensional Magnetic Resonance Imaging (MRI) methodologies has changed the paradigm and allows investigations of the living human brain structure - e.g. micro- and macrostructural features of cortical and subcortical regions and their connections, including cortical sulcation/gyrification, area, and thickness, as well as white matter microstructure and connectivity - beginning in utero. Because of its relative safety, MRI is well-adapted to study individuals at multiple time points and to longitudinally follow the changes in brain structure and function that underlie the early stages of cognitive development. |
| 2025-04-30 | [Retrieval-Enhanced Few-Shot Prompting for Speech Event Extraction](http://arxiv.org/abs/2504.21372v1) | Máté Gedeon | Speech Event Extraction (SpeechEE) is a challenging task that lies at the intersection of Automatic Speech Recognition (ASR) and Natural Language Processing (NLP), requiring the identification of structured event information from spoken language. In this work, we present a modular, pipeline-based SpeechEE framework that integrates high-performance ASR with semantic search-enhanced prompting of Large Language Models (LLMs). Our system first classifies speech segments likely to contain events using a hybrid filtering mechanism including rule-based, BERT-based, and LLM-based models. It then employs few-shot LLM prompting, dynamically enriched via semantic similarity retrieval, to identify event triggers and extract corresponding arguments. We evaluate the pipeline using multiple LLMs (Llama3-8B, GPT-4o-mini, and o1-mini) highlighting significant performance gains with o1-mini, which achieves 63.3% F1 on trigger classification and 27.8% F1 on argument classification, outperforming prior benchmarks. Our results demonstrate that pipeline approaches, when empowered by retrieval-augmented LLMs, can rival or exceed end-to-end systems while maintaining interpretability and modularity. This work provides practical insights into LLM-driven event extraction and opens pathways for future hybrid models combining textual and acoustic features. |
| 2025-04-30 | [ShorterBetter: Guiding Reasoning Models to Find Optimal Inference Length for Efficient Reasoning](http://arxiv.org/abs/2504.21370v1) | Jingyang Yi, Jiazheng Wang | Reasoning models such as OpenAI o3 and DeepSeek-R1 have demonstrated strong performance on reasoning-intensive tasks through extended Chain-of-Thought (CoT) prompting. While longer reasoning traces can facilitate a more thorough exploration of solution paths for complex problems, researchers have observed that these models often "overthink", leading to inefficient inference. In this paper, we introduce ShorterBetter, a simple yet effective reinforcement learning methed that enables reasoning language models to discover their own optimal CoT lengths without human intervention. By sampling multiple outputs per problem and defining the Sample Optimal Length (SOL) as the shortest correct response among all the outputs, our method dynamically guides the model toward optimal inference lengths. Applied to the DeepSeek-Distill-Qwen-1.5B model, ShorterBetter achieves up to an 80% reduction in output length on both in-domain and out-of-domain reasoning tasks while maintaining accuracy. Our analysis shows that overly long reasoning traces often reflect loss of reasoning direction, and thus suggests that the extended CoT produced by reasoning models is highly compressible. |
| 2025-04-30 | [Simple Visual Artifact Detection in Sora-Generated Videos](http://arxiv.org/abs/2504.21334v1) | Misora Sugiyama, Hirokatsu Kataoka | The December 2024 release of OpenAI's Sora, a powerful video generation model driven by natural language prompts, highlights a growing convergence between large language models (LLMs) and video synthesis. As these multimodal systems evolve into video-enabled LLMs (VidLLMs), capable of interpreting, generating, and interacting with visual content, understanding their limitations and ensuring their safe deployment becomes essential. This study investigates visual artifacts frequently found and reported in Sora-generated videos, which can compromise quality, mislead viewers, or propagate disinformation. We propose a multi-label classification framework targeting four common artifact label types: label 1: boundary / edge defects, label 2: texture / noise issues, label 3: movement / joint anomalies, and label 4: object mismatches / disappearances. Using a dataset of 300 manually annotated frames extracted from 15 Sora-generated videos, we trained multiple 2D CNN architectures (ResNet-50, EfficientNet-B3 / B4, ViT-Base). The best-performing model trained by ResNet-50 achieved an average multi-label classification accuracy of 94.14%. This work supports the broader development of VidLLMs by contributing to (1) the creation of datasets for video quality evaluation, (2) interpretable artifact-based analysis beyond language metrics, and (3) the identification of visual risks relevant to factuality and safety. |
| 2025-04-30 | [Phi-4-reasoning Technical Report](http://arxiv.org/abs/2504.21318v1) | Marah Abdin, Sahaj Agarwal et al. | We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of "teachable" prompts-selected for the right level of complexity and diversity-and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models. |
| 2025-04-30 | [Annotating and Auditing the Safety Properties of Unsafe Rust](http://arxiv.org/abs/2504.21312v1) | Zihao Rao, Hongliang Tian et al. | Unsafe code is a critical topic in ensuring the security of system software development in Rust. It is the sole source of potential undefined behaviors, assuming the compiler is sound. To avoid the misuse of unsafe code, Rust developers should provide clear safety property annotations for unsafe APIs. However, there is limited official guidance and few best practices for annotating unsafe code. Even the current best practices for safety property annotations in the Rust standard library are ad hoc and informal. In this paper, we design a domain-specific language to describe the safety properties of unsafe APIs, which may serve as a precursor for automated verification in the future. Furthermore, to ensure that the caller of an unsafe API properly delegates the safety property required by the callee, we propose a novel unsafety propagation graph to model the usage and propagation of unsafe code. Based on this graph, we further introduce a method to partition the graph into smaller graphs, such that each graph serves as a self-contained audit unit for examining the soundness of unsafe code encapsulation and safety property annotation. We applied our approach to the Rust standard library, and the experimental results demonstrate that our method is both practical and effective. Additionally, we have fixed safety property description issues in 23 APIs. |
| 2025-04-30 | [The Dual Power of Interpretable Token Embeddings: Jailbreaking Attacks and Defenses for Diffusion Model Unlearning](http://arxiv.org/abs/2504.21307v1) | Siyi Chen, Yimeng Zhang et al. | Despite the remarkable generalization capabilities of diffusion models, recent studies have shown that these models can memorize and generate harmful content when prompted with specific text instructions. Although fine-tuning approaches have been developed to mitigate this issue by unlearning harmful concepts, these methods can be easily circumvented through jailbreaking attacks. This indicates that the harmful concept has not been fully erased from the model. However, existing attack methods, while effective, lack interpretability regarding why unlearned models still retain the concept, thereby hindering the development of defense strategies. In this work, we address these limitations by proposing an attack method that learns an orthogonal set of interpretable attack token embeddings. The attack token embeddings can be decomposed into human-interpretable textual elements, revealing that unlearned models still retain the target concept through implicit textual components. Furthermore, these attack token embeddings are robust and transferable across text prompts, initial noises, and unlearned models. Finally, leveraging this diverse set of embeddings, we design a defense method applicable to both our proposed attack and existing attack methods. Experimental results demonstrate the effectiveness of both our attack and defense strategies. |
| 2025-04-30 | [Assessing LLM code generation quality through path planning tasks](http://arxiv.org/abs/2504.21276v1) | Wanyi Chen, Meng-Wen Su et al. | As LLM-generated code grows in popularity, more evaluation is needed to assess the risks of using such tools, especially for safety-critical applications such as path planning. Existing coding benchmarks are insufficient as they do not reflect the context and complexity of safety-critical applications. To this end, we assessed six LLMs' abilities to generate the code for three different path-planning algorithms and tested them on three maps of various difficulties. Our results suggest that LLM-generated code presents serious hazards for path planning applications and should not be applied in safety-critical contexts without rigorous testing. |

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



