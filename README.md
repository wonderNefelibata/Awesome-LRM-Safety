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
| 2025-05-13 | [PCS-UQ: Uncertainty Quantification via the Predictability-Computability-Stability Framework](http://arxiv.org/abs/2505.08784v1) | Abhineet Agarwal, Michael Xiao et al. | As machine learning (ML) models are increasingly deployed in high-stakes domains, trustworthy uncertainty quantification (UQ) is critical for ensuring the safety and reliability of these models. Traditional UQ methods rely on specifying a true generative model and are not robust to misspecification. On the other hand, conformal inference allows for arbitrary ML models but does not consider model selection, which leads to large interval sizes. We tackle these drawbacks by proposing a UQ method based on the predictability, computability, and stability (PCS) framework for veridical data science proposed by Yu and Kumbier. Specifically, PCS-UQ addresses model selection by using a prediction check to screen out unsuitable models. PCS-UQ then fits these screened algorithms across multiple bootstraps to assess inter-sample variability and algorithmic instability, enabling more reliable uncertainty estimates. Further, we propose a novel calibration scheme that improves local adaptivity of our prediction sets. Experiments across $17$ regression and $6$ classification datasets show that PCS-UQ achieves the desired coverage and reduces width over conformal approaches by $\approx 20\%$. Further, our local analysis shows PCS-UQ often achieves target coverage across subgroups while conformal methods fail to do so. For large deep-learning models, we propose computationally efficient approximation schemes that avoid the expensive multiple bootstrap trainings of PCS-UQ. Across three computer vision benchmarks, PCS-UQ reduces prediction set size over conformal methods by $20\%$. Theoretically, we show a modified PCS-UQ algorithm is a form of split conformal inference and achieves the desired coverage with exchangeable data. |
| 2025-05-13 | [HealthBench: Evaluating Large Language Models Towards Improved Human Health](http://arxiv.org/abs/2505.08775v1) | Rahul K. Arora, Jason Wei et al. | We present HealthBench, an open-source benchmark measuring the performance and safety of large language models in healthcare. HealthBench consists of 5,000 multi-turn conversations between a model and an individual user or healthcare professional. Responses are evaluated using conversation-specific rubrics created by 262 physicians. Unlike previous multiple-choice or short-answer benchmarks, HealthBench enables realistic, open-ended evaluation through 48,562 unique rubric criteria spanning several health contexts (e.g., emergencies, transforming clinical data, global health) and behavioral dimensions (e.g., accuracy, instruction following, communication). HealthBench performance over the last two years reflects steady initial progress (compare GPT-3.5 Turbo's 16% to GPT-4o's 32%) and more rapid recent improvements (o3 scores 60%). Smaller models have especially improved: GPT-4.1 nano outperforms GPT-4o and is 25 times cheaper. We additionally release two HealthBench variations: HealthBench Consensus, which includes 34 particularly important dimensions of model behavior validated via physician consensus, and HealthBench Hard, where the current top score is 32%. We hope that HealthBench grounds progress towards model development and applications that benefit human health. |
| 2025-05-13 | [DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models](http://arxiv.org/abs/2505.08744v1) | Xiaoyang Chen, Xinan Dai et al. | To advance the mathematical proficiency of large language models (LLMs), the DeepMath team has launched an open-source initiative aimed at developing an open mathematical LLM and systematically evaluating its mathematical creativity. This paper represents the initial contribution of this initiative. While recent developments in mathematical LLMs have predominantly emphasized reasoning skills, as evidenced by benchmarks on elementary to undergraduate-level mathematical tasks, the creative capabilities of these models have received comparatively little attention, and evaluation datasets remain scarce. To address this gap, we propose an evaluation criteria for mathematical creativity and introduce DeepMath-Creative, a novel, high-quality benchmark comprising constructive problems across algebra, geometry, analysis, and other domains. We conduct a systematic evaluation of mainstream LLMs' creative problem-solving abilities using this dataset. Experimental results show that even under lenient scoring criteria -- emphasizing core solution components and disregarding minor inaccuracies, such as small logical gaps, incomplete justifications, or redundant explanations -- the best-performing model, O3 Mini, achieves merely 70% accuracy, primarily on basic undergraduate-level constructive tasks. Performance declines sharply on more complex problems, with models failing to provide substantive strategies for open problems. These findings suggest that, although current LLMs display a degree of constructive proficiency on familiar and lower-difficulty problems, such performance is likely attributable to the recombination of memorized patterns rather than authentic creative insight or novel synthesis. |
| 2025-05-13 | [LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs](http://arxiv.org/abs/2505.08704v1) | K M Sajjadul Islam, Ayesha Siddika Nipu et al. | Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting. |
| 2025-05-13 | [Granite-speech: open-source speech-aware LLMs with strong English ASR capabilities](http://arxiv.org/abs/2505.08699v1) | George Saon, Avihu Dekel et al. | Granite-speech LLMs are compact and efficient speech language models specifically designed for English ASR and automatic speech translation (AST). The models were trained by modality aligning the 2B and 8B parameter variants of granite-3.3-instruct to speech on publicly available open-source corpora containing audio inputs and text targets consisting of either human transcripts for ASR or automatically generated translations for AST. Comprehensive benchmarking shows that on English ASR, which was our primary focus, they outperform several competitors' models that were trained on orders of magnitude more proprietary data, and they keep pace on English-to-X AST for major European languages, Japanese, and Chinese. The speech-specific components are: a conformer acoustic encoder using block attention and self-conditioning trained with connectionist temporal classification, a windowed query-transformer speech modality adapter used to do temporal downsampling of the acoustic embeddings and map them to the LLM text embedding space, and LoRA adapters to further fine-tune the text LLM. Granite-speech-3.3 operates in two modes: in speech mode, it performs ASR and AST by activating the encoder, projector, and LoRA adapters; in text mode, it calls the underlying granite-3.3-instruct model directly (without LoRA), essentially preserving all the text LLM capabilities and safety. Both models are freely available on HuggingFace (https://huggingface.co/ibm-granite/granite-speech-3.3-2b and https://huggingface.co/ibm-granite/granite-speech-3.3-8b) and can be used for both research and commercial purposes under a permissive Apache 2.0 license. |
| 2025-05-13 | [MC-Swarm: Minimal-Communication Multi-Agent Trajectory Planning and Deadlock Resolution for Quadrotor Swarm](http://arxiv.org/abs/2505.08593v1) | Yunwoo Lee, Jungwon Park | For effective multi-agent trajectory planning, it is important to consider lightweight communication and its potential asynchrony. This paper presents a distributed trajectory planning algorithm for a quadrotor swarm that operates asynchronously and requires no communication except during the initial planning phase. Moreover, our algorithm guarantees no deadlock under asynchronous updates and absence of communication during flight. To effectively ensure these points, we build two main modules: coordination state updater and trajectory optimizer. The coordination state updater computes waypoints for each agent toward its goal and performs subgoal optimization while considering deadlocks, as well as safety constraints with respect to neighbor agents and obstacles. Then, the trajectory optimizer generates a trajectory that ensures collision avoidance even with the asynchronous planning updates of neighboring agents. We provide a theoretical guarantee of collision avoidance with deadlock resolution and evaluate the effectiveness of our method in complex simulation environments, including random forests and narrow-gap mazes. Additionally, to reduce the total mission time, we design a faster coordination state update using lightweight communication. Lastly, our approach is validated through extensive simulations and real-world experiments with cluttered environment scenarios. |
| 2025-05-13 | [Thermal Detection of People with Mobility Restrictions for Barrier Reduction at Traffic Lights Controlled Intersections](http://arxiv.org/abs/2505.08568v1) | Xiao Ni, Carsten Kuehnel et al. | Rapid advances in deep learning for computer vision have driven the adoption of RGB camera-based adaptive traffic light systems to improve traffic safety and pedestrian comfort. However, these systems often overlook the needs of people with mobility restrictions. Moreover, the use of RGB cameras presents significant challenges, including limited detection performance under adverse weather or low-visibility conditions, as well as heightened privacy concerns. To address these issues, we propose a fully automated, thermal detector-based traffic light system that dynamically adjusts signal durations for individuals with walking impairments or mobility burden and triggers the auditory signal for visually impaired individuals, thereby advancing towards barrier-free intersection for all users. To this end, we build the thermal dataset for people with mobility restrictions (TD4PWMR), designed to capture diverse pedestrian scenarios, particularly focusing on individuals with mobility aids or mobility burden under varying environmental conditions, such as different lighting, weather, and crowded urban settings. While thermal imaging offers advantages in terms of privacy and robustness to adverse conditions, it also introduces inherent hurdles for object detection due to its lack of color and fine texture details and generally lower resolution of thermal images. To overcome these limitations, we develop YOLO-Thermal, a novel variant of the YOLO architecture that integrates advanced feature extraction and attention mechanisms for enhanced detection accuracy and robustness in thermal imaging. Experiments demonstrate that the proposed thermal detector outperforms existing detectors, while the proposed traffic light system effectively enhances barrier-free intersection. The source codes and dataset are available at https://github.com/leon2014dresden/YOLO-THERMAL. |
| 2025-05-13 | [Synthesis of safety certificates for discrete-time uncertain systems via convex optimization](http://arxiv.org/abs/2505.08559v1) | Marta Fochesato, Han Wang et al. | We study the problem of co-designing control barrier functions and linear state feedback controllers for discrete-time linear systems affected by additive disturbances. For disturbances of bounded magnitude, we provide a semi-definite program whose feasibility implies the existence of a control law and a certificate ensuring safety in the infinite horizon with respect to the worst-case disturbance realization in the uncertainty set. For disturbances with unbounded support, we rely on martingale theory to derive a second semi-definite program whose feasibility provides probabilistic safety guarantees holding joint-in-time over a finite time horizon. We examine several extensions, including (i) encoding of different types of input constraints, (ii) robustification against distributional ambiguity around the true distribution, (iii) design of safety filters, and (iv) extension to general safety specifications such as obstacle avoidance. |
| 2025-05-13 | [Area Comparison of CHERIoT and PMP in Ibex](http://arxiv.org/abs/2505.08541v1) | Samuel Riedel, Marno van der Maas et al. | Memory safety is a critical concern for modern embedded systems, particularly in security-sensitive applications. This paper explores the area impact of adding memory safety extensions to the Ibex RISC-V core, focusing on physical memory protection (PMP) and Capability Hardware Extension to RISC-V for Internet of Things (CHERIoT). We synthesise the extended Ibex cores using a commercial tool targeting the open FreePDK45 process and provide a detailed area breakdown and discussion of the results.   The PMP configuration we consider is one with 16 PMP regions. We find that the extensions increase the core size by 24 thousand gate-equivalent (kGE) for PMP and 33 kGE for CHERIoT. The increase is mainly due to the additional state required to store information about protected memory. While this increase amounts to 42% for PMP and 57% for CHERIoT in Ibex's area, its effect on the overall system is minimal. In a complete system-on-chip (SoC), like the secure microcontroller OpenTitan Earl Grey, where the core represents only a fraction of the total area, the estimated system-wide overhead is 0.6% for PMP and 1% for CHERIoT. Given the security benefits these extensions provide, the area trade-off is justified, making Ibex a compelling choice for secure embedded applications. |
| 2025-05-13 | [Weighted Rewriting: Semiring Semantics for Abstract Reduction Systems](http://arxiv.org/abs/2505.08496v1) | Emma Ahrens, Jan-Christoph Kassing et al. | We present novel semiring semantics for abstract reduction systems (ARSs). More precisely, we provide a weighted version of ARSs, where the reduction steps induce weights from a semiring. Inspired by provenance analysis in database theory and logic, we obtain a formalism that can be used for provenance analysis of arbitrary ARSs. Our semantics handle (possibly unbounded) non-determinism and possibly infinite reductions. Moreover, we develop several techniques to prove upper and lower bounds on the weights resulting from our semantics, and show that in this way one obtains a uniform approach to analyze several different properties like termination, derivational complexity, space complexity, safety, as well as combinations of these properties. |
| 2025-05-13 | [Explaining Autonomous Vehicles with Intention-aware Policy Graphs](http://arxiv.org/abs/2505.08404v1) | Sara Montese, Victor Gimenez-Abalos et al. | The potential to improve road safety, reduce human driving error, and promote environmental sustainability have enabled the field of autonomous driving to progress rapidly over recent decades. The performance of autonomous vehicles has significantly improved thanks to advancements in Artificial Intelligence, particularly Deep Learning. Nevertheless, the opacity of their decision-making, rooted in the use of accurate yet complex AI models, has created barriers to their societal trust and regulatory acceptance, raising the need for explainability. We propose a post-hoc, model-agnostic solution to provide teleological explanations for the behaviour of an autonomous vehicle in urban environments. Building on Intention-aware Policy Graphs, our approach enables the extraction of interpretable and reliable explanations of vehicle behaviour in the nuScenes dataset from global and local perspectives. We demonstrate the potential of these explanations to assess whether the vehicle operates within acceptable legal boundaries and to identify possible vulnerabilities in autonomous driving datasets and models. |
| 2025-05-13 | [Towards Contamination Resistant Benchmarks](http://arxiv.org/abs/2505.08389v1) | Rahmatullah Musawi, Sheng Lu | The rapid development of large language models (LLMs) has transformed the landscape of natural language processing. Evaluating LLMs properly is crucial for understanding their potential and addressing concerns such as safety. However, LLM evaluation is confronted by various factors, among which contamination stands out as a key issue that undermines the reliability of evaluations. In this work, we introduce the concept of contamination resistance to address this challenge. We propose a benchmark based on Caesar ciphers (e.g., "ab" to "bc" when the shift is 1), which, despite its simplicity, is an excellent example of a contamination resistant benchmark. We test this benchmark on widely used LLMs under various settings, and we find that these models struggle with this benchmark when contamination is controlled. Our findings reveal issues in current LLMs and raise important questions regarding their true capabilities. Our work contributes to the development of contamination resistant benchmarks, enabling more rigorous LLM evaluation and offering insights into the true capabilities and limitations of LLMs. |
| 2025-05-13 | [Investigating Resolution Strategies for Workspace-Occlusion in Augmented Virtuality](http://arxiv.org/abs/2505.08312v1) | Nico Feld, Pauline Bimberg et al. | Augmented Virtuality integrates physical content into virtual environments, but the occlusion of physical by virtual content is a challenge. This unwanted occlusion may disrupt user interactions with physical devices and compromise safety and usability. This paper investigates two resolution strategies to address this issue: Redirected Walking, which subtly adjusts the user's movement to maintain physical-virtual alignment, and Automatic Teleport Rotation, which realigns the virtual environment during travel. A user study set in a virtual forest demonstrates that both methods effectively reduce occlusion. While in our testbed, Automatic Teleport Rotation achieves higher occlusion resolution, it is suspected to increase cybersickness compared to the less intrusive Redirected Walking approach. |
| 2025-05-13 | [AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale](http://arxiv.org/abs/2505.08311v1) | Yunjie Ji, Xiaoyu Tian et al. | We present AM-Thinking-v1, a 32B dense language model that advances the frontier of reasoning, embodying the collaborative spirit of open-source innovation. Outperforming DeepSeek-R1 and rivaling leading Mixture-of-Experts (MoE) models like Qwen3-235B-A22B and Seed1.5-Thinking, AM-Thinking-v1 achieves impressive scores of 85.3 on AIME 2024, 74.4 on AIME 2025, and 70.3 on LiveCodeBench, showcasing state-of-the-art mathematical and coding capabilities among open-source models of similar scale.   Built entirely from the open-source Qwen2.5-32B base model and publicly available queries, AM-Thinking-v1 leverages a meticulously crafted post-training pipeline - combining supervised fine-tuning and reinforcement learning - to deliver exceptional reasoning capabilities. This work demonstrates that the open-source community can achieve high performance at the 32B scale, a practical sweet spot for deployment and fine-tuning. By striking a balance between top-tier performance and real-world usability, we hope AM-Thinking-v1 inspires further collaborative efforts to harness mid-scale models, pushing reasoning boundaries while keeping accessibility at the core of innovation. We have open-sourced our model on \href{https://huggingface.co/a-m-team/AM-Thinking-v1}{Hugging Face}. |
| 2025-05-13 | [Feasibility-Aware Pessimistic Estimation: Toward Long-Horizon Safety in Offline RL](http://arxiv.org/abs/2505.08179v1) | Zhikun Tao, Gang Xiong et al. | Offline safe reinforcement learning(OSRL) derives constraint-satisfying policies from pre-collected datasets, offers a promising avenue for deploying RL in safety-critical real-world domains such as robotics. However, the majority of existing approaches emphasize only short-term safety, neglecting long-horizon considerations. Consequently, they may violate safety constraints and fail to ensure sustained protection during online deployment. Moreover, the learned policies often struggle to handle states and actions that are not present or out-of-distribution(OOD) from the offline dataset, and exhibit limited sample efficiency. To address these challenges, we propose a novel framework Feasibility-Aware offline Safe Reinforcement Learning with CVAE-based Pessimism (FASP). First, we employ Hamilton-Jacobi (H-J) reachability analysis to generate reliable safety labels, which serve as supervisory signals for training both a conditional variational autoencoder (CVAE) and a safety classifier. This approach not only ensures high sampling efficiency but also provides rigorous long-horizon safety guarantees. Furthermore, we utilize pessimistic estimation methods to estimate the Q-value of reward and cost, which mitigates the extrapolation errors induces by OOD actions, and penalize unsafe actions to enabled the agent to proactively avoid high-risk behaviors. Moreover, we theoretically prove the validity of this pessimistic estimation. Extensive experiments on DSRL benchmarks demonstrate that FASP algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety. |
| 2025-05-13 | [Foundation Models Knowledge Distillation For Battery Capacity Degradation Forecast](http://arxiv.org/abs/2505.08151v1) | Joey Chan, Zhen Chen et al. | Accurate estimation of lithium-ion battery capacity degradation is critical for enhancing the reliability and safety of battery operations. Traditional expert models, tailored to specific scenarios, provide isolated estimations. With the rapid advancement of data-driven techniques, a series of general-purpose time-series foundation models have been developed. However, foundation models specifically designed for battery capacity degradation remain largely unexplored. To enable zero-shot generalization in battery degradation prediction using large model technology, this study proposes a degradation-aware fine-tuning strategy for time-series foundation models. We apply this strategy to fine-tune the Timer model on approximately 10 GB of open-source battery charge discharge data. Validation on our released CycleLife-SJTUIE dataset demonstrates that the fine-tuned Battery-Timer possesses strong zero-shot generalization capability in capacity degradation forecasting. To address the computational challenges of deploying large models, we further propose a knowledge distillation framework that transfers the knowledge of pre-trained foundation models into compact expert models. Distillation results across several state-of-the-art time-series expert models confirm that foundation model knowledge significantly improves the multi-condition generalization of expert models. |
| 2025-05-12 | [Asymptotic grand unification in SO(10) with one extra dimension](http://arxiv.org/abs/2505.08068v1) | Gao-Xiang Fang, Zhi-Wei Wang et al. | Asymptotic grand unification provides an alternative approach to gradually unify gauge couplings in the UV limit, where they reach a non-trivial UV fixed point. Using an economical and realistic particle content setup, we demonstrate that asymptotic grand unification can be achieved in a 5D SO(10) model with one extra dimension. The top, bottom and tau masses are split, and the smallness of the neutrino mass is explained via inverse seesaw. One intermediate scale, the Pati-Salam symmetry breaking scale, is included below the compactification scale. Due to the absence of large-dimensional Higgs representations, gauge couplings exhibit asymptotic safety and are thus asymptotically unified, regardless of their initial values. In contrast, Yukawa couplings can achieve asymptotic freedom if the negative gauge contributions dominate over the positive Yukawa terms, requiring exact unification at the compactification scale. The widely-used 126-dimensional Higgs is not recommended in this 5D asymptotic SO(10) GUT, as it tends to drive the gauge beta function positive, compromising asymptotic safety. |
| 2025-05-12 | [Justified Evidence Collection for Argument-based AI Fairness Assurance](http://arxiv.org/abs/2505.08064v1) | Alpay Sabuncuoglu, Christopher Burr et al. | It is well recognised that ensuring fair AI systems is a complex sociotechnical challenge, which requires careful deliberation and continuous oversight across all stages of a system's lifecycle, from defining requirements to model deployment and deprovisioning. Dynamic argument-based assurance cases, which present structured arguments supported by evidence, have emerged as a systematic approach to evaluating and mitigating safety risks and hazards in AI-enabled system development and have also been extended to deal with broader normative goals such as fairness and explainability. This paper introduces a systems-engineering-driven framework, supported by software tooling, to operationalise a dynamic approach to argument-based assurance in two stages. In the first stage, during the requirements planning phase, a multi-disciplinary and multi-stakeholder team define goals and claims to be established (and evidenced) by conducting a comprehensive fairness governance process. In the second stage, a continuous monitoring interface gathers evidence from existing artefacts (e.g. metrics from automated tests), such as model, data, and use case documentation, to support these arguments dynamically. The framework's effectiveness is demonstrated through an illustrative case study in finance, with a focus on supporting fairness-related arguments. |
| 2025-05-12 | [FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning](http://arxiv.org/abs/2505.08054v1) | Zhehao Zhang, Weijie Xu et al. | Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities. |
| 2025-05-12 | [Safety and optimality in learning-based control at low computational cost](http://arxiv.org/abs/2505.08026v1) | Dominik Baumann, Krzysztof Kowalczyk et al. | Applying machine learning methods to physical systems that are supposed to act in the real world requires providing safety guarantees. However, methods that include such guarantees often come at a high computational cost, making them inapplicable to large datasets and embedded devices with low computational power. In this paper, we propose CoLSafe, a computationally lightweight safe learning algorithm whose computational complexity grows sublinearly with the number of data points. We derive both safety and optimality guarantees and showcase the effectiveness of our algorithm on a seven-degrees-of-freedom robot arm. |

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



