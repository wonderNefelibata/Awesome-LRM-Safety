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
| 2025-05-08 | [Reasoning Models Don't Always Say What They Think](http://arxiv.org/abs/2505.05410v1) | Yanda Chen, Joe Benton et al. | Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. |
| 2025-05-08 | [PillarMamba: Learning Local-Global Context for Roadside Point Cloud via Hybrid State Space Model](http://arxiv.org/abs/2505.05397v1) | Zhang Zhang, Chao Sun et al. | Serving the Intelligent Transport System (ITS) and Vehicle-to-Everything (V2X) tasks, roadside perception has received increasing attention in recent years, as it can extend the perception range of connected vehicles and improve traffic safety. However, roadside point cloud oriented 3D object detection has not been effectively explored. To some extent, the key to the performance of a point cloud detector lies in the receptive field of the network and the ability to effectively utilize the scene context. The recent emergence of Mamba, based on State Space Model (SSM), has shaken up the traditional convolution and transformers that have long been the foundational building blocks, due to its efficient global receptive field. In this work, we introduce Mamba to pillar-based roadside point cloud perception and propose a framework based on Cross-stage State-space Group (CSG), called PillarMamba. It enhances the expressiveness of the network and achieves efficient computation through cross-stage feature fusion. However, due to the limitations of scan directions, state space model faces local connection disrupted and historical relationship forgotten. To address this, we propose the Hybrid State-space Block (HSB) to obtain the local-global context of roadside point cloud. Specifically, it enhances neighborhood connections through local convolution and preserves historical memory through residual attention. The proposed method outperforms the state-of-the-art methods on the popular large scale roadside benchmark: DAIR-V2X-I. The code will be released soon. |
| 2025-05-08 | [Robust Online Learning with Private Information](http://arxiv.org/abs/2505.05341v1) | Kyohei Okumura | This paper investigates the robustness of online learning algorithms when learners possess private information. No-external-regret algorithms, prevalent in machine learning, are vulnerable to strategic manipulation, allowing an adaptive opponent to extract full surplus. Even standard no-weak-external-regret algorithms, designed for optimal learning in stationary environments, exhibit similar vulnerabilities. This raises a fundamental question: can a learner simultaneously prevent full surplus extraction by adaptive opponents while maintaining optimal performance in well-behaved environments? To address this, we model the problem as a two-player repeated game, where the learner with private information plays against the environment, facing ambiguity about the environment's types: stationary or adaptive. We introduce \emph{partial safety} as a key design criterion for online learning algorithms to prevent full surplus extraction. We then propose the \emph{Explore-Exploit-Punish} (\textsf{EEP}) algorithm and prove that it satisfies partial safety while achieving optimal learning in stationary environments, and has a variant that delivers improved welfare performance. Our findings highlight the risks of applying standard online learning algorithms in strategic settings with adverse selection. We advocate for a shift toward online learning algorithms that explicitly incorporate safeguards against strategic manipulation while ensuring strong learning performance. |
| 2025-05-08 | [Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation](http://arxiv.org/abs/2505.05235v1) | Luca Marzari, Isabella Mastroeni et al. | Traditional methods for formal verification (FV) of deep neural networks (DNNs) are constrained by a binary encoding of safety properties, where a model is classified as either safe or unsafe (robust or not robust). This binary encoding fails to capture the nuanced safety levels within a model, often resulting in either overly restrictive or too permissive requirements. In this paper, we introduce a novel problem formulation called Abstract DNN-Verification, which verifies a hierarchical structure of unsafe outputs, providing a more granular analysis of the safety aspect for a given DNN. Crucially, by leveraging abstract interpretation and reasoning about output reachable sets, our approach enables assessing multiple safety levels during the FV process, requiring the same (in the worst case) or even potentially less computational effort than the traditional binary verification approach. Specifically, we demonstrate how this formulation allows rank adversarial inputs according to their abstract safety level violation, offering a more detailed evaluation of the model's safety and robustness. Our contributions include a theoretical exploration of the relationship between our novel abstract safety formulation and existing approaches that employ abstract interpretation for robustness verification, complexity analysis of the novel problem introduced, and an empirical evaluation considering both a complex deep reinforcement learning task (based on Habitat 3.0) and standard DNN-Verification benchmarks. |
| 2025-05-08 | [PaniCar: Securing the Perception of Advanced Driving Assistance Systems Against Emergency Vehicle Lighting](http://arxiv.org/abs/2505.05183v1) | Elad Feldman, Jacob Shams et al. | The safety of autonomous cars has come under scrutiny in recent years, especially after 16 documented incidents involving Teslas (with autopilot engaged) crashing into parked emergency vehicles (police cars, ambulances, and firetrucks). While previous studies have revealed that strong light sources often introduce flare artifacts in the captured image, which degrade the image quality, the impact of flare on object detection performance remains unclear. In this research, we unveil PaniCar, a digital phenomenon that causes an object detector's confidence score to fluctuate below detection thresholds when exposed to activated emergency vehicle lighting. This vulnerability poses a significant safety risk, and can cause autonomous vehicles to fail to detect objects near emergency vehicles. In addition, this vulnerability could be exploited by adversaries to compromise the security of advanced driving assistance systems (ADASs). We assess seven commercial ADASs (Tesla Model 3, "manufacturer C", HP, Pelsee, AZDOME, Imagebon, Rexing), four object detectors (YOLO, SSD, RetinaNet, Faster R-CNN), and 14 patterns of emergency vehicle lighting to understand the influence of various technical and environmental factors. We also evaluate four SOTA flare removal methods and show that their performance and latency are insufficient for real-time driving constraints. To mitigate this risk, we propose Caracetamol, a robust framework designed to enhance the resilience of object detectors against the effects of activated emergency vehicle lighting. Our evaluation shows that on YOLOv3 and Faster RCNN, Caracetamol improves the models' average confidence of car detection by 0.20, the lower confidence bound by 0.33, and reduces the fluctuation range by 0.33. In addition, Caracetamol is capable of processing frames at a rate of between 30-50 FPS, enabling real-time ADAS car detection. |
| 2025-05-08 | [Duplex Self-Aligning Resonant Beam Communications and Power Transfer with Coupled Spatially Distributed Laser Resonator](http://arxiv.org/abs/2505.05107v1) | Mingliang Xiong, Qingwen Liu et al. | Sustainable energy supply and high-speed communications are two significant needs for mobile electronic devices. This paper introduces a self-aligning resonant beam system for simultaneous light information and power transfer (SLIPT), employing a novel coupled spatially distributed resonator (CSDR). The system utilizes a resonant beam for efficient power delivery and a second-harmonic beam for concurrent data transmission, inherently minimizing echo interference and enabling bidirectional communication. Through comprehensive analyses, we investigate the CSDR's stable region, beam evolution, and power characteristics in relation to working distance and device parameters. Numerical simulations validate the CSDR-SLIPT system's feasibility by identifying a stable beam waist location for achieving accurate mode-match coupling between two spatially distributed resonant cavities and demonstrating its operational range and efficient power delivery across varying distances. The research reveals the system's benefits in terms of both safety and energy transmission efficiency. We also demonstrate the trade-off among the reflectivities of the cavity mirrors in the CSDR. These findings offer valuable design insights for resonant beam systems, advancing SLIPT with significant potential for remote device connectivity. |
| 2025-05-08 | [DispBench: Benchmarking Disparity Estimation to Synthetic Corruptions](http://arxiv.org/abs/2505.05091v1) | Shashank Agnihotri, Amaan Ansari et al. | Deep learning (DL) has surpassed human performance on standard benchmarks, driving its widespread adoption in computer vision tasks. One such task is disparity estimation, estimating the disparity between matching pixels in stereo image pairs, which is crucial for safety-critical applications like medical surgeries and autonomous navigation. However, DL-based disparity estimation methods are highly susceptible to distribution shifts and adversarial attacks, raising concerns about their reliability and generalization. Despite these concerns, a standardized benchmark for evaluating the robustness of disparity estimation methods remains absent, hindering progress in the field.   To address this gap, we introduce DispBench, a comprehensive benchmarking tool for systematically assessing the reliability of disparity estimation methods. DispBench evaluates robustness against synthetic image corruptions such as adversarial attacks and out-of-distribution shifts caused by 2D Common Corruptions across multiple datasets and diverse corruption scenarios. We conduct the most extensive performance and robustness analysis of disparity estimation methods to date, uncovering key correlations between accuracy, reliability, and generalization. Open-source code for DispBench: https://github.com/shashankskagnihotri/benchmarking_robustness/tree/disparity_estimation/final/disparity_estimation |
| 2025-05-08 | [Adaptive Contextual Embedding for Robust Far-View Borehole Detection](http://arxiv.org/abs/2505.05008v1) | Xuesong Liu, Tianyu Hao et al. | In controlled blasting operations, accurately detecting densely distributed tiny boreholes from far-view imagery is critical for operational safety and efficiency. However, existing detection methods often struggle due to small object scales, highly dense arrangements, and limited distinctive visual features of boreholes. To address these challenges, we propose an adaptive detection approach that builds upon existing architectures (e.g., YOLO) by explicitly leveraging consistent embedding representations derived through exponential moving average (EMA)-based statistical updates.   Our method introduces three synergistic components: (1) adaptive augmentation utilizing dynamically updated image statistics to robustly handle illumination and texture variations; (2) embedding stabilization to ensure consistent and reliable feature extraction; and (3) contextual refinement leveraging spatial context for improved detection accuracy. The pervasive use of EMA in our method is particularly advantageous given the limited visual complexity and small scale of boreholes, allowing stable and robust representation learning even under challenging visual conditions. Experiments on a challenging proprietary quarry-site dataset demonstrate substantial improvements over baseline YOLO-based architectures, highlighting our method's effectiveness in realistic and complex industrial scenarios. |
| 2025-05-08 | [A Vehicle System for Navigating Among Vulnerable Road Users Including Remote Operation](http://arxiv.org/abs/2505.04982v1) | Oscar de Groot, Alberto Bertipaglia et al. | We present a vehicle system capable of navigating safely and efficiently around Vulnerable Road Users (VRUs), such as pedestrians and cyclists. The system comprises key modules for environment perception, localization and mapping, motion planning, and control, integrated into a prototype vehicle. A key innovation is a motion planner based on Topology-driven Model Predictive Control (T-MPC). The guidance layer generates multiple trajectories in parallel, each representing a distinct strategy for obstacle avoidance or non-passing. The underlying trajectory optimization constrains the joint probability of collision with VRUs under generic uncertainties. To address extraordinary situations ("edge cases") that go beyond the autonomous capabilities - such as construction zones or encounters with emergency responders - the system includes an option for remote human operation, supported by visual and haptic guidance. In simulation, our motion planner outperforms three baseline approaches in terms of safety and efficiency. We also demonstrate the full system in prototype vehicle tests on a closed track, both in autonomous and remotely operated modes. |
| 2025-05-08 | [LVLM-MPC Collaboration for Autonomous Driving: A Safety-Aware and Task-Scalable Control Architecture](http://arxiv.org/abs/2505.04980v1) | Kazuki Atsuta, Kohei Honda et al. | This paper proposes a novel Large Vision-Language Model (LVLM) and Model Predictive Control (MPC) integration framework that delivers both task scalability and safety for Autonomous Driving (AD). LVLMs excel at high-level task planning across diverse driving scenarios. However, since these foundation models are not specifically designed for driving and their reasoning is not consistent with the feasibility of low-level motion planning, concerns remain regarding safety and smooth task switching. This paper integrates LVLMs with MPC Builder, which automatically generates MPCs on demand, based on symbolic task commands generated by the LVLM, while ensuring optimality and safety. The generated MPCs can strongly assist the execution or rejection of LVLM-driven task switching by providing feedback on the feasibility of the given tasks and generating task-switching-aware MPCs. Our approach provides a safe, flexible, and adaptable control framework, bridging the gap between cutting-edge foundation models and reliable vehicle operation. We demonstrate the effectiveness of our approach through a simulation experiment, showing that our system can safely and effectively handle highway driving while maintaining the flexibility and adaptability of LVLMs. |
| 2025-05-08 | [Belief Filtering for Epistemic Control in Linguistic State Space](http://arxiv.org/abs/2505.04927v1) | Sebastian Dumbrava | We examine belief filtering as a mechanism for the epistemic control of artificial agents, focusing on the regulation of internal cognitive states represented as linguistic expressions. This mechanism is developed within the Semantic Manifold framework, where belief states are dynamic, structured ensembles of natural language fragments. Belief filters act as content-aware operations on these fragments across various cognitive transitions. This paper illustrates how the inherent interpretability and modularity of such a linguistically-grounded cognitive architecture directly enable belief filtering, offering a principled approach to agent regulation. The study highlights the potential for enhancing AI safety and alignment through structured interventions in an agent's internal semantic space and points to new directions for architecturally embedded cognitive governance. |
| 2025-05-08 | [Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models](http://arxiv.org/abs/2505.04921v1) | Yunxin Li, Zhenyu Liu et al. | Reasoning lies at the heart of intelligence, shaping the ability to make decisions, draw conclusions, and generalize across domains. In artificial intelligence, as systems increasingly operate in open, uncertain, and multimodal environments, reasoning becomes essential for enabling robust and adaptive behavior. Large Multimodal Reasoning Models (LMRMs) have emerged as a promising paradigm, integrating modalities such as text, images, audio, and video to support complex reasoning capabilities and aiming to achieve comprehensive perception, precise understanding, and deep reasoning. As research advances, multimodal reasoning has rapidly evolved from modular, perception-driven pipelines to unified, language-centric frameworks that offer more coherent cross-modal understanding. While instruction tuning and reinforcement learning have improved model reasoning, significant challenges remain in omni-modal generalization, reasoning depth, and agentic behavior. To address these issues, we present a comprehensive and structured survey of multimodal reasoning research, organized around a four-stage developmental roadmap that reflects the field's shifting design philosophies and emerging capabilities. First, we review early efforts based on task-specific modules, where reasoning was implicitly embedded across stages of representation, alignment, and fusion. Next, we examine recent approaches that unify reasoning into multimodal LLMs, with advances such as Multimodal Chain-of-Thought (MCoT) and multimodal reinforcement learning enabling richer and more structured reasoning chains. Finally, drawing on empirical insights from challenging benchmarks and experimental cases of OpenAI O3 and O4-mini, we discuss the conceptual direction of native large multimodal reasoning models (N-LMRMs), which aim to support scalable, agentic, and adaptive reasoning and planning in complex, real-world environments. |
| 2025-05-08 | [GCN-Based Throughput-Oriented Handover Management in Dense 5G Vehicular Networks](http://arxiv.org/abs/2505.04894v1) | Nazanin Mehregan, Robson E. De Grande | The rapid advancement of 5G has transformed vehicular networks, offering high bandwidth, low latency, and fast data rates essential for real-time applications in smart cities and vehicles. These improvements enhance traffic safety and entertainment services. However, the limited coverage and frequent handovers in 5G networks cause network instability, especially in high-mobility environments due to the ping-pong effect. This paper presents TH-GCN (Throughput-oriented Graph Convolutional Network), a novel approach for optimizing handover management in dense 5G networks. Using graph neural networks (GNNs), TH-GCN models vehicles and base stations as nodes in a dynamic graph enriched with features such as signal quality, throughput, vehicle speed, and base station load. By integrating both user equipment and base station perspectives, this dual-centric approach enables adaptive, real-time handover decisions that improve network stability. Simulation results show that TH-GCN reduces handovers by up to 78 percent and improves signal quality by 10 percent, outperforming existing methods. |
| 2025-05-08 | [Federated Learning for Cyber Physical Systems: A Comprehensive Survey](http://arxiv.org/abs/2505.04873v1) | Minh K. Quan, Pubudu N. Pathirana et al. | The integration of machine learning (ML) in cyber physical systems (CPS) is a complex task due to the challenges that arise in terms of real-time decision making, safety, reliability, device heterogeneity, and data privacy. There are also open research questions that must be addressed in order to fully realize the potential of ML in CPS. Federated learning (FL), a distributed approach to ML, has become increasingly popular in recent years. It allows models to be trained using data from decentralized sources. This approach has been gaining popularity in the CPS field, as it integrates computer, communication, and physical processes. Therefore, the purpose of this work is to provide a comprehensive analysis of the most recent developments of FL-CPS, including the numerous application areas, system topologies, and algorithms developed in recent years. The paper starts by discussing recent advances in both FL and CPS, followed by their integration. Then, the paper compares the application of FL in CPS with its applications in the internet of things (IoT) in further depth to show their connections and distinctions. Furthermore, the article scrutinizes how FL is utilized in critical CPS applications, e.g., intelligent transportation systems, cybersecurity services, smart cities, and smart healthcare solutions. The study also includes critical insights and lessons learned from various FL-CPS implementations. The paper's concluding section delves into significant concerns and suggests avenues for further research in this fast-paced and dynamic era. |
| 2025-05-07 | [PR2: Peephole Raw Pointer Rewriting with LLMs for Translating C to Safer Rust](http://arxiv.org/abs/2505.04852v1) | Yifei Gao, Chengpeng Wang et al. | There has been a growing interest in translating C code to Rust due to Rust's robust memory and thread safety guarantees. Tools such as C2RUST enable syntax-guided transpilation from C to semantically equivalent Rust code. However, the resulting Rust programs often rely heavily on unsafe constructs--particularly raw pointers--which undermines Rust's safety guarantees. This paper aims to improve the memory safety of Rust programs generated by C2RUST by eliminating raw pointers. Specifically, we propose a peephole raw pointer rewriting technique that lifts raw pointers in individual functions to appropriate Rust data structures. Technically, PR2 employs decision-tree-based prompting to guide the pointer lifting process. Additionally, it leverages code change analysis to guide the repair of errors introduced during rewriting, effectively addressing errors encountered during compilation and test case execution. We implement PR2 as a prototype and evaluate it using gpt-4o-mini on 28 real-world C projects. The results show that PR2 successfully eliminates 13.22% of local raw pointers across these projects, significantly enhancing the safety of the translated Rust code. On average, PR2 completes the transformation of a project in 5.44 hours, at an average cost of $1.46. |
| 2025-05-07 | [Comparative Study of Generative Models for Early Detection of Failures in Medical Devices](http://arxiv.org/abs/2505.04845v1) | Binesh Sadanandan, Bahareh Arghavani Nobar et al. | The medical device industry has significantly advanced by integrating sophisticated electronics like microchips and field-programmable gate arrays (FPGAs) to enhance the safety and usability of life-saving devices. These complex electro-mechanical systems, however, introduce challenging failure modes that are not easily detectable with conventional methods. Effective fault detection and mitigation become vital as reliance on such electronics grows. This paper explores three generative machine learning-based approaches for fault detection in medical devices, leveraging sensor data from surgical staplers,a class 2 medical device. Historically considered low-risk, these devices have recently been linked to an increasing number of injuries and fatalities. The study evaluates the performance and data requirements of these machine-learning approaches, highlighting their potential to enhance device safety. |
| 2025-05-07 | [Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs](http://arxiv.org/abs/2505.04806v1) | Chetan Pathade | Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security. |
| 2025-05-07 | [Replay to Remember (R2R): An Efficient Uncertainty-driven Unsupervised Continual Learning Framework Using Generative Replay](http://arxiv.org/abs/2505.04787v1) | Sriram Mandalika, Harsha Vardhan et al. | Continual Learning entails progressively acquiring knowledge from new data while retaining previously acquired knowledge, thereby mitigating ``Catastrophic Forgetting'' in neural networks. Our work presents a novel uncertainty-driven Unsupervised Continual Learning framework using Generative Replay, namely ``Replay to Remember (R2R)''. The proposed R2R architecture efficiently uses unlabelled and synthetic labelled data in a balanced proportion using a cluster-level uncertainty-driven feedback mechanism and a VLM-powered generative replay module. Unlike traditional memory-buffer methods that depend on pretrained models and pseudo-labels, our R2R framework operates without any prior training. It leverages visual features from unlabeled data and adapts continuously using clustering-based uncertainty estimation coupled with dynamic thresholding. Concurrently, a generative replay mechanism along with DeepSeek-R1 powered CLIP VLM produces labelled synthetic data representative of past experiences, resembling biological visual thinking that replays memory to remember and act in new, unseen tasks. Extensive experimental analyses are carried out in CIFAR-10, CIFAR-100, CINIC-10, SVHN and TinyImageNet datasets. Our proposed R2R approach improves knowledge retention, achieving a state-of-the-art performance of 98.13%, 73.06%, 93.41%, 95.18%, 59.74%, respectively, surpassing state-of-the-art performance by over 4.36%. |
| 2025-05-07 | [Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization](http://arxiv.org/abs/2505.04578v1) | Wenjun Cao | Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models. |
| 2025-05-07 | [Stow: Robotic Packing of Items into Fabric Pods](http://arxiv.org/abs/2505.04572v1) | Nicolas Hudson, Josh Hooks et al. | This paper presents a compliant manipulation system capable of placing items onto densely packed shelves. The wide diversity of items and strict business requirements for high producing rates and low defect generation have prohibited warehouse robotics from performing this task. Our innovations in hardware, perception, decision-making, motion planning, and control have enabled this system to perform over 500,000 stows in a large e-commerce fulfillment center. The system achieves human levels of packing density and speed while prioritizing work on overhead shelves to enhance the safety of humans working alongside the robots. |

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



