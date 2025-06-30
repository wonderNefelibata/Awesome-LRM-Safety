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
| 2025-06-27 | [ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks](http://arxiv.org/abs/2506.22423v1) | Pritam Dash, Ethan Chan et al. | Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training. |
| 2025-06-27 | [Sequential Diagnosis with Language Models](http://arxiv.org/abs/2506.22405v1) | Harsha Nori, Mayank Daswani et al. | Artificial intelligence holds great promise for expanding access to expert medical knowledge and reasoning. However, most evaluations of language models rely on static vignettes and multiple-choice questions that fail to reflect the complexity and nuance of evidence-based medicine in real-world settings. In clinical practice, physicians iteratively formulate and revise diagnostic hypotheses, adapting each subsequent question and test to what they've just learned, and weigh the evolving evidence before committing to a final diagnosis. To emulate this iterative process, we introduce the Sequential Diagnosis Benchmark, which transforms 304 diagnostically challenging New England Journal of Medicine clinicopathological conference (NEJM-CPC) cases into stepwise diagnostic encounters. A physician or AI begins with a short case abstract and must iteratively request additional details from a gatekeeper model that reveals findings only when explicitly queried. Performance is assessed not just by diagnostic accuracy but also by the cost of physician visits and tests performed. We also present the MAI Diagnostic Orchestrator (MAI-DxO), a model-agnostic orchestrator that simulates a panel of physicians, proposes likely differential diagnoses and strategically selects high-value, cost-effective tests. When paired with OpenAI's o3 model, MAI-DxO achieves 80% diagnostic accuracy--four times higher than the 20% average of generalist physicians. MAI-DxO also reduces diagnostic costs by 20% compared to physicians, and 70% compared to off-the-shelf o3. When configured for maximum accuracy, MAI-DxO achieves 85.5% accuracy. These performance gains with MAI-DxO generalize across models from the OpenAI, Gemini, Claude, Grok, DeepSeek, and Llama families. We highlight how AI systems, when guided to think iteratively and act judiciously, can advance diagnostic precision and cost-effectiveness in clinical care. |
| 2025-06-27 | [V2X Intention Sharing for Cooperative Electrically Power-Assisted Cycles](http://arxiv.org/abs/2506.22223v1) | Felipe Valle Quiroz, Johan Elfing et al. | This paper introduces a novel intention-sharing mechanism for Electrically Power-Assisted Cycles (EPACs) within V2X communication frameworks, enhancing the ETSI VRU Awareness Message (VAM) protocol. The method replaces discrete predicted trajectory points with a compact elliptical geographical area representation derived via quadratic polynomial fitting and Least Squares Method (LSM). This approach encodes trajectory predictions with fixed-size data payloads, independent of the number of forecasted points, enabling higher-frequency transmissions and improved network reliability. Simulation results demonstrate superior inter-packet gap (IPG) performance compared to standard ETSI VAMs, particularly under constrained communication conditions. A physical experiment validates the feasibility of real-time deployment on embedded systems. The method supports scalable, low-latency intention sharing, contributing to cooperative perception and enhanced safety for vulnerable road users in connected and automated mobility ecosystems. Finally, we discuss the viability of LSM and open the door to other methods for prediction. |
| 2025-06-27 | [A Different Approach to AI Safety: Proceedings from the Columbia Convening on Openness in Artificial Intelligence and AI Safety](http://arxiv.org/abs/2506.22183v1) | Camille François, Ludovic Péran et al. | The rapid rise of open-weight and open-source foundation models is intensifying the obligation and reshaping the opportunity to make AI systems safe. This paper reports outcomes from the Columbia Convening on AI Openness and Safety (San Francisco, 19 Nov 2024) and its six-week preparatory programme involving more than forty-five researchers, engineers, and policy leaders from academia, industry, civil society, and government. Using a participatory, solutions-oriented process, the working groups produced (i) a research agenda at the intersection of safety and open source AI; (ii) a mapping of existing and needed technical interventions and open source tools to safely and responsibly deploy open foundation models across the AI development workflow; and (iii) a mapping of the content safety filter ecosystem with a proposed roadmap for future research and development. We find that openness -- understood as transparent weights, interoperable tooling, and public governance -- can enhance safety by enabling independent scrutiny, decentralized mitigation, and culturally plural oversight. However, significant gaps persist: scarce multimodal and multilingual benchmarks, limited defenses against prompt-injection and compositional attacks in agentic systems, and insufficient participatory mechanisms for communities most affected by AI harms. The paper concludes with a roadmap of five priority research directions, emphasizing participatory inputs, future-proof content filters, ecosystem-wide safety infrastructure, rigorous agentic safeguards, and expanded harm taxonomies. These recommendations informed the February 2025 French AI Action Summit and lay groundwork for an open, plural, and accountable AI safety discipline. |
| 2025-06-27 | [ASVSim (AirSim for Surface Vehicles): A High-Fidelity Simulation Framework for Autonomous Surface Vehicle Research](http://arxiv.org/abs/2506.22174v1) | Bavo Lesy, Siemen Herremans et al. | The transport industry has recently shown significant interest in unmanned surface vehicles (USVs), specifically for port and inland waterway transport. These systems can improve operational efficiency and safety, which is especially relevant in the European Union, where initiatives such as the Green Deal are driving a shift towards increased use of inland waterways. At the same time, a shortage of qualified personnel is accelerating the adoption of autonomous solutions. However, there is a notable lack of open-source, high-fidelity simulation frameworks and datasets for developing and evaluating such solutions. To address these challenges, we introduce AirSim For Surface Vehicles (ASVSim), an open-source simulation framework specifically designed for autonomous shipping research in inland and port environments. The framework combines simulated vessel dynamics with marine sensor simulation capabilities, including radar and camera systems and supports the generation of synthetic datasets for training computer vision models and reinforcement learning agents. Built upon Cosys-AirSim, ASVSim provides a comprehensive platform for developing autonomous navigation algorithms and generating synthetic datasets. The simulator supports research of both traditional control methods and deep learning-based approaches. Through limited experiments, we demonstrate the potential of the simulator in these research areas. ASVSim is provided as an open-source project under the MIT license, making autonomous navigation research accessible to a larger part of the ocean engineering community. |
| 2025-06-27 | [Learning Distributed Safe Multi-Agent Navigation via Infinite-Horizon Optimal Graph Control](http://arxiv.org/abs/2506.22117v1) | Fenglan Wang, Xinguo Shu et al. | Distributed multi-agent navigation faces inherent challenges due to the competing requirements of maintaining safety and achieving goal-directed behavior, particularly for agents with limited sensing range operating in unknown environments with dense obstacles. Existing approaches typically project predefined goal-reaching controllers onto control barrier function (CBF) constraints, often resulting in conservative and suboptimal trade-offs between safety and goal-reaching performance. We propose an infinite-horizon CBF-constrained optimal graph control formulation for distributed safe multi-agent navigation. By deriving the analytical solution structure, we develop a novel Hamilton-Jacobi-Bellman (HJB)-based learning framework to approximate the solution. In particular, our algorithm jointly learns a CBF and a distributed control policy, both parameterized by graph neural networks (GNNs), along with a value function that robustly guides agents toward their goals. Moreover, we introduce a state-dependent parameterization of Lagrange multipliers, enabling dynamic trade-offs between safety and performance. Unlike traditional short-horizon, quadratic programming-based CBF methods, our approach leverages long-horizon optimization to proactively avoid deadlocks and navigate complex environments more effectively. Extensive simulation results demonstrate substantial improvements in safety and task success rates across various agent dynamics, with strong scalability and generalization to large-scale teams in previously unseen environments. Real-world experiments using Crazyflie drone swarms on challenging antipodal position-swapping tasks further validate the practicality, generalizability, and robustness of the proposed HJB-GNN learning framework. |
| 2025-06-27 | [Pedestrian Intention and Trajectory Prediction in Unstructured Traffic Using IDD-PeD](http://arxiv.org/abs/2506.22111v1) | Ruthvik Bokkasam, Shankar Gangisetty et al. | With the rapid advancements in autonomous driving, accurately predicting pedestrian behavior has become essential for ensuring safety in complex and unpredictable traffic conditions. The growing interest in this challenge highlights the need for comprehensive datasets that capture unstructured environments, enabling the development of more robust prediction models to enhance pedestrian safety and vehicle navigation. In this paper, we introduce an Indian driving pedestrian dataset designed to address the complexities of modeling pedestrian behavior in unstructured environments, such as illumination changes, occlusion of pedestrians, unsignalized scene types and vehicle-pedestrian interactions. The dataset provides high-level and detailed low-level comprehensive annotations focused on pedestrians requiring the ego-vehicle's attention. Evaluation of the state-of-the-art intention prediction methods on our dataset shows a significant performance drop of up to $\mathbf{15\%}$, while trajectory prediction methods underperform with an increase of up to $\mathbf{1208}$ MSE, defeating standard pedestrian datasets. Additionally, we present exhaustive quantitative and qualitative analysis of intention and trajectory baselines. We believe that our dataset will open new challenges for the pedestrian behavior research community to build robust models. Project Page: https://cvit.iiit.ac.in/research/projects/cvit-projects/iddped |
| 2025-06-27 | [Query as Test: An Intelligent Driving Test and Data Storage Method for Integrated Cockpit-Vehicle-Road Scenarios](http://arxiv.org/abs/2506.22068v1) | Shengyue Yao, Runqing Guo et al. | With the deep penetration of Artificial Intelligence (AI) in the transportation sector, intelligent cockpits, autonomous driving, and intelligent road networks are developing at an unprecedented pace. However, the data ecosystems of these three key areas are increasingly fragmented and incompatible. Especially, existing testing methods rely on data stacking, fail to cover all edge cases, and lack flexibility. To address this issue, this paper introduces the concept of "Query as Test" (QaT). This concept shifts the focus from rigid, prescripted test cases to flexible, on-demand logical queries against a unified data representation. Specifically, we identify the need for a fundamental improvement in data storage and representation, leading to our proposal of "Extensible Scenarios Notations" (ESN). ESN is a novel declarative data framework based on Answer Set Programming (ASP), which uniformly represents heterogeneous multimodal data from the cockpit, vehicle, and road as a collection of logical facts and rules. This approach not only achieves deep semantic fusion of data, but also brings three core advantages: (1) supports complex and flexible semantic querying through logical reasoning; (2) provides natural interpretability for decision-making processes; (3) allows for on-demand data abstraction through logical rules, enabling fine-grained privacy protection. We further elaborate on the QaT paradigm, transforming the functional validation and safety compliance checks of autonomous driving systems into logical queries against the ESN database, significantly enhancing the expressiveness and formal rigor of the testing. Finally, we introduce the concept of "Validation-Driven Development" (VDD), which suggests to guide developments by logical validation rather than quantitative testing in the era of Large Language Models, in order to accelerating the iteration and development process. |
| 2025-06-27 | [Building Trustworthy Cognitive Monitoring for Safety-Critical Human Tasks: A Phased Methodological Approach](http://arxiv.org/abs/2506.22066v1) | Maciej Grzeszczuk, Grzegorz Pochwatko et al. | Operators performing high-stakes, safety-critical tasks - such as air traffic controllers, surgeons, or mission control personnel - must maintain exceptional cognitive performance under variable and often stressful conditions. This paper presents a phased methodological approach to building cognitive monitoring systems for such environments. By integrating insights from human factors research, simulation-based training, sensor technologies, and fundamental psychological principles, the proposed framework supports real-time performance assessment with minimum intrusion. The approach begins with simplified simulations and evolves towards operational contexts. Key challenges addressed include variability in workload, the effects of fatigue and stress, thus the need for adaptive monitoring for early warning support mechanisms. The methodology aims to improve situational awareness, reduce human error, and support decision-making without undermining operator autonomy. Ultimately, the work contributes to the development of resilient and transparent systems in domains where human performance is critical to safety. |
| 2025-06-27 | [Lost at the Beginning of Reasoning](http://arxiv.org/abs/2506.22058v1) | Baohao Liao, Xinyi Chen et al. | Recent advancements in large language models (LLMs) have significantly advanced complex reasoning capabilities, particularly through extended chain-of-thought (CoT) reasoning that incorporates mechanisms such as backtracking, self-reflection and self-correction. Despite these developments, the self-correction abilities of LLMs during long CoT reasoning remain underexplored. And recent findings on overthinking suggest that such models often engage in unnecessarily redundant reasoning. In this work, we empirically show that the first reasoning step exerts a disproportionately large influence on the final prediction - errors introduced at this stage can substantially degrade subsequent reasoning quality. This phenomenon is consistently observed across two state-of-the-art open-source reasoning model families: DeepSeek-R1 and Qwen3. To address this, we propose an efficient sampling strategy that leverages a reward model to identify and retain high-quality first reasoning steps while discarding suboptimal ones, achieving up to a 70% reduction in inference cost without sacrificing accuracy. Finally, we introduce a new benchmark specifically constructed with deliberately flawed first reasoning steps to systematically evaluate model self-correction capabilities, offering a foundation for future research on robust reasoning in LLMs. |
| 2025-06-27 | [Evaluating Redundancy Mitigation in Vulnerable Road User Awareness Messages for Bicycles](http://arxiv.org/abs/2506.22052v1) | Nico Ostendorf, Keno Garlichs et al. | V2X communication has become crucial for enhancing road safety, especially for Vulnerable Road Users (VRU) such as pedestrians and cyclists. However, the increasing number of devices communicating on the same channels will lead to significant channel load. To address this issue this study evaluates the effectiveness of Redundancy Mitigation (RM) for VRU Awareness Messages (VAM), focusing specifically on cyclists. The objective of RM is to minimize the transmission of redundant information. We conducted a simulation study using a urban scenario with a high bicycle density based on traffic data from Hannover, Germany. This study assessed the impact of RM on channel load, measured by Channel Busy Ratio (CBR), and safety, measured by VRU Perception Rate (VPR) in simulation. To evaluate the accuracy and reliability of the RM mechanisms, we analyzed the actual differences in position, speed, and heading between the ego VRU and the VRU, which was assumed to be redundant. Our findings indicate that while RM can reduce channel congestion, it also leads to a decrease in VPR. The analysis of actual differences revealed that the RM mechanism standardized by ETSI often uses outdated information, leading to significant discrepancies in position, speed, and heading, which could result in dangerous situations. To address these limitations, we propose an adapted RM mechanism that improves the balance between reducing channel load and maintaining VRU awareness. The adapted approach shows a significant reduction in maximum CBR and a less significant decrease in VPR compared to the standardized RM. Moreover, it demonstrates better performance in the actual differences in position, speed, and heading, thereby enhancing overall safety. Our results highlight the need for further research to optimize RM techniques and ensure they effectively enhance V2X communication without compromising the safety of VRUs. |
| 2025-06-27 | [Hyper-modal Imputation Diffusion Embedding with Dual-Distillation for Federated Multimodal Knowledge Graph Completion](http://arxiv.org/abs/2506.22036v1) | Ying Zhang, Yu Zhao et al. | With the increasing multimodal knowledge privatization requirements, multimodal knowledge graphs in different institutes are usually decentralized, lacking of effective collaboration system with both stronger reasoning ability and transmission safety guarantees. In this paper, we propose the Federated Multimodal Knowledge Graph Completion (FedMKGC) task, aiming at training over federated MKGs for better predicting the missing links in clients without sharing sensitive knowledge. We propose a framework named MMFeD3-HidE for addressing multimodal uncertain unavailability and multimodal client heterogeneity challenges of FedMKGC. (1) Inside the clients, our proposed Hyper-modal Imputation Diffusion Embedding model (HidE) recovers the complete multimodal distributions from incomplete entity embeddings constrained by available modalities. (2) Among clients, our proposed Multimodal FeDerated Dual Distillation (MMFeD3) transfers knowledge mutually between clients and the server with logit and feature distillation to improve both global convergence and semantic consistency. We propose a FedMKGC benchmark for a comprehensive evaluation, consisting of a general FedMKGC backbone named MMFedE, datasets with heterogeneous multimodal information, and three groups of constructed baselines. Experiments conducted on our benchmark validate the effectiveness, semantic consistency, and convergence robustness of MMFeD3-HidE. |
| 2025-06-27 | [R1-Track: Direct Application of MLLMs to Visual Object Tracking via Reinforcement Learning](http://arxiv.org/abs/2506.21980v1) | Biao Wang, Wenwen Li | Visual single object tracking aims to continuously localize and estimate the scale of a target in subsequent video frames, given only its initial state in the first frame. This task has traditionally been framed as a template matching problem, evolving through major phases including correlation filters, two-stream networks, and one-stream networks with significant progress achieved. However, these methods typically require explicit classification and regression modeling, depend on supervised training with large-scale datasets, and are limited to the single task of tracking, lacking flexibility. In recent years, multi-modal large language models (MLLMs) have advanced rapidly. Open-source models like Qwen2.5-VL, a flagship MLLMs with strong foundational capabilities, demonstrate excellent performance in grounding tasks. This has spurred interest in applying such models directly to visual tracking. However, experiments reveal that Qwen2.5-VL struggles with template matching between image pairs (i.e., tracking tasks). Inspired by deepseek-R1, we fine-tuned Qwen2.5-VL using the group relative policy optimization (GRPO) reinforcement learning method on a small-scale dataset with a rule-based reward function. The resulting model, R1-Track, achieved notable performance on the GOT-10k benchmark. R1-Track supports flexible initialization via bounding boxes or text descriptions while retaining most of the original model's general capabilities. And we further discuss potential improvements for R1-Track. This rough technical report summarizes our findings as of May 2025. |
| 2025-06-27 | [Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses](http://arxiv.org/abs/2506.21972v1) | Mohamed Ahmed, Mohamed Abdelmouty et al. | The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries. |
| 2025-06-27 | [Identifying High-Risk Areas for Traffic Collisions in Montgomery, Maryland Using KDE and Spatial Autocorrelation Analysis](http://arxiv.org/abs/2506.21930v1) | Stanislav Liashkov | Despite a global decline in motor vehicle crash fatalities due to improved research and road safety policies, road traffic injuries remain a significant public health concern. The World Health Organization 2023 report highlights that road traffic injuries are the leading cause of death among individuals aged 5-29, with over half of fatalities involving pedestrians, cyclists, and motorcyclists. This study addresses this critical issue by identifying high-risk areas in Montgomery County, Maryland, contributing to the global goal of halving road traffic deaths and injuries by 2030. Using Kernel Density Estimation (KDE) and spatial autocorrelation analysis, we estimate collision densities and identify hotspots for targeted interventions. Our findings reveal significant spatial clustering of traffic collisions, with distinct patterns in densely populated urban areas and rural regions, offering valuable insights for policymakers to enhance road safety. |
| 2025-06-27 | [SODA: Out-of-Distribution Detection in Domain-Shifted Point Clouds via Neighborhood Propagation](http://arxiv.org/abs/2506.21892v1) | Adam Goodge, Xun Xu et al. | As point cloud data increases in prevalence in a variety of applications, the ability to detect out-of-distribution (OOD) point cloud objects becomes critical for ensuring model safety and reliability. However, this problem remains under-explored in existing research. Inspired by success in the image domain, we propose to exploit advances in 3D vision-language models (3D VLMs) for OOD detection in point cloud objects. However, a major challenge is that point cloud datasets used to pre-train 3D VLMs are drastically smaller in size and object diversity than their image-based counterparts. Critically, they often contain exclusively computer-designed synthetic objects. This leads to a substantial domain shift when the model is transferred to practical tasks involving real objects scanned from the physical environment. In this paper, our empirical experiments show that synthetic-to-real domain shift significantly degrades the alignment of point cloud with their associated text embeddings in the 3D VLM latent space, hindering downstream performance. To address this, we propose a novel methodology called SODA which improves the detection of OOD point clouds through a neighborhood-based score propagation scheme. SODA is inference-based, requires no additional model training, and achieves state-of-the-art performance over existing approaches across datasets and problem settings. |
| 2025-06-27 | [Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](http://arxiv.org/abs/2506.21876v1) | Qiyue Gao, Xinyu Pi et al. | Internal world models (WMs) enable agents to understand the world's state and predict transitions, serving as the basis for advanced deliberative reasoning. Recent large Vision-Language Models (VLMs), such as OpenAI o3, GPT-4o and Gemini, exhibit potential as general-purpose WMs. While the latest studies have evaluated and shown limitations in specific capabilities such as visual understanding, a systematic evaluation of VLMs' fundamental WM abilities remains absent. Drawing on comparative psychology and cognitive science, we propose a two-stage framework that assesses Perception (visual, spatial, temporal, quantitative, and motion) and Prediction (mechanistic simulation, transitive inference, compositional inference) to provide an atomic evaluation of VLMs as WMs. Guided by this framework, we introduce WM-ABench, a large-scale benchmark comprising 23 fine-grained evaluation dimensions across 6 diverse simulated environments with controlled counterfactual simulations. Through 660 experiments on 15 latest commercial and open-source VLMs, we find that these models exhibit striking limitations in basic world modeling abilities. For instance, almost all models perform at near-random accuracy when distinguishing motion trajectories. Additionally, they lack disentangled understanding -- e.g., some models tend to believe blue objects move faster than green ones. More rich results and analyses reveal significant gaps between VLMs and human-level world modeling. |
| 2025-06-26 | [Adaptive Multipath-Based SLAM for Distributed MIMO Systems](http://arxiv.org/abs/2506.21798v1) | Xuhong Li, Benjamin J. B. Deutschmann et al. | Localizing users and mapping the environment using radio signals is a key task in emerging applications such as reliable communications, location-aware security, and safety critical navigation. Recently introduced multipath-based simultaneous localization and mapping (MP-SLAM) can jointly localize a mobile agent and the reflective surfaces in radio frequency (RF) environments. Most existing MP-SLAM methods assume that map features and their corresponding RF propagation paths are statistically independent, which neglects inherent dependencies arising when a single reflective surface contributes to different propagation paths or when an agent communicates with more than one base station. Previous approaches that aim to fuse information across propagation paths are limited by their inability to perform ray tracing in environments with nonconvex geometries. In this paper, we propose a Bayesian MP-SLAM method for distributed MIMO systems that addresses this limitation. In particular, we use amplitude statistics to establish adaptive time-varying detection probabilities. Based on the resulting "soft" ray-tracing strategy, our method can fuse information across propagation paths in RF environments with nonconvex geometries. A Bayesian estimation method for the joint estimation of map features and agent position is established by applying the message passing rules of the sum-product algorithm (SPA) to the factor graph that represents the proposed statistical model. We also introduce an improved proposal PDF for particle-based computation of SPA messages. This proposal PDF enables the early detection of new surfaces that are solely supported by double-bounce paths. Our method is validated using synthetic RF measurements in a challenging scenario with nonconvex geometries. The results demonstrate that it can provide accurate localization and mapping estimates as well as attain the posterior CRLB. |
| 2025-06-26 | [Detecting Land with Reflected Light Spectroscopy to Rule Out Waterworld O2 Biosignature False Positives](http://arxiv.org/abs/2506.21790v1) | Anna Grace Ulses, Joshua Krissansen-Totton et al. | The search for life outside our solar system is at the forefront of modern astronomy, and telescopes such as the Habitable Worlds Observatory (HWO) are being designed to identify biosignatures. Molecular oxygen, O2, is considered a promising indication of life, yet substantial abiotic O2 may accumulate from H2O photolysis and hydrogen escape on a lifeless, fully (100%) ocean-covered terrestrial planet when surface O2 sinks are suppressed. This so-called waterworld false positive scenario could be ruled out with land detection because exposed land precludes extremely deep oceans (~50 Earth oceans) given topographic limits set by the crushing strength of rocks. Land detection is possible because plausible geologic surfaces exhibit increasing reflectance with wavelength in the visible, whereas liquid water and ice/snow have flat or decreasing reflectance, respectively. Here, we present reflected light retrievals to demonstrate that HWO could detect land on an exo-Earth in the disk-averaged spectrum. Given a signal-to-noise ratio of 20 spectrum, Earth-like land fractions can be confidently detected with 0.3-1.1 um spectral coverage (resolution R~140 in the visible, R~7 in the UV, with Earth-like atmosphere and clouds). We emphasize the need for UV spectroscopy down to at least 0.3 um to break an O3-land degeneracy. We find that the SNR and resolution requirements in the visible/UV imply that a larger aperture (~8 m) will be necessary to ensure the observing times required for land detection are feasible for most HWO terrestrial habitable zone targets. These results strongly inform the HWO minimum requirements to corroborate possible oxygen biosignatures. |
| 2025-06-26 | [Stochastic Neural Control Barrier Functions](http://arxiv.org/abs/2506.21697v1) | Hongchao Zhang, Manan Tayal et al. | Control Barrier Functions (CBFs) are utilized to ensure the safety of control systems. CBFs act as safety filters in order to provide safety guarantees without compromising system performance. These safety guarantees rely on the construction of valid CBFs. Due to their complexity, CBFs can be represented by neural networks, known as neural CBFs (NCBFs). Existing works on the verification of the NCBF focus on the synthesis and verification of NCBFs in deterministic settings, leaving the stochastic NCBFs (SNCBFs) less studied. In this work, we propose a verifiably safe synthesis for SNCBFs. We consider the cases of smooth SNCBFs with twice-differentiable activation functions and SNCBFs that utilize the Rectified Linear Unit or ReLU activation function. We propose a verification-free synthesis framework for smooth SNCBFs and a verification-in-the-loop synthesis framework for both smooth and ReLU SNCBFs. and we validate our frameworks in three cases, namely, the inverted pendulum, Darboux, and the unicycle model. |

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



