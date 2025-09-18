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
| 2025-09-17 | [GLIDE: A Coordinated Aerial-Ground Framework for Search and Rescue in Unknown Environments](http://arxiv.org/abs/2509.14210v1) | Seth Farrell, Chenghao Li et al. | We present a cooperative aerial-ground search-and-rescue (SAR) framework that pairs two unmanned aerial vehicles (UAVs) with an unmanned ground vehicle (UGV) to achieve rapid victim localization and obstacle-aware navigation in unknown environments. We dub this framework Guided Long-horizon Integrated Drone Escort (GLIDE), highlighting the UGV's reliance on UAV guidance for long-horizon planning. In our framework, a goal-searching UAV executes real-time onboard victim detection and georeferencing to nominate goals for the ground platform, while a terrain-scouting UAV flies ahead of the UGV's planned route to provide mid-level traversability updates. The UGV fuses aerial cues with local sensing to perform time-efficient A* planning and continuous replanning as information arrives. Additionally, we present a hardware demonstration (using a GEM e6 golf cart as the UGV and two X500 UAVs) to evaluate end-to-end SAR mission performance and include simulation ablations to assess the planning stack in isolation from detection. Empirical results demonstrate that explicit role separation across UAVs, coupled with terrain scouting and guided planning, improves reach time and navigation safety in time-critical SAR missions. |
| 2025-09-17 | [Breaking the Cycle of Incarceration With Targeted Mental Health Outreach: A Case Study in Machine Learning for Public Policy](http://arxiv.org/abs/2509.14129v1) | Kit T. Rodolfa, Erika Salomon et al. | Many incarcerated individuals face significant and complex challenges, including mental illness, substance dependence, and homelessness, yet jails and prisons are often poorly equipped to address these needs. With little support from the existing criminal justice system, these needs can remain untreated and worsen, often leading to further offenses and a cycle of incarceration with adverse outcomes both for the individual and for public safety, with particularly large impacts on communities of color that continue to widen the already extensive racial disparities in criminal justice outcomes. Responding to these failures, a growing number of criminal justice stakeholders are seeking to break this cycle through innovative approaches such as community-driven and alternative approaches to policing, mentoring, community building, restorative justice, pretrial diversion, holistic defense, and social service connections. Here we report on a collaboration between Johnson County, Kansas, and Carnegie Mellon University to perform targeted, proactive mental health outreach in an effort to reduce reincarceration rates.   This paper describes the data used, our predictive modeling approach and results, as well as the design and analysis of a field trial conducted to confirm our model's predictive power, evaluate the impact of this targeted outreach, and understand at what level of reincarceration risk outreach might be most effective. Through this trial, we find that our model is highly predictive of new jail bookings, with more than half of individuals in the trial's highest-risk group returning to jail in the following year. Outreach was most effective among these highest-risk individuals, with impacts on mental health utilization, EMS dispatches, and criminal justice involvement. |
| 2025-09-17 | [Safe Sliding Mode Control in Position for Double Integrator Systems](http://arxiv.org/abs/2509.14121v1) | Marco A. Gomez, Christopher D. Cruz-Ancona | We address the problem of robust safety control design for double integrator systems. We show that, when the constraints are defined only on position states, it is possible to construct a safe sliding domain from the dynamic of a simple integrator that is already safe. On this domain, the closed-loop trajectories remain robust and safe against uncertainties and disturbances. Furthermore, we design a controller gain that guarantees convergence to the safe sliding domain while avoiding the given unsafe set. The concept is initially developed for first-order sliding mode and is subsequently generalized to an adaptive framework, ensuring that trajectories remain confined to a predefined vicinity of the sliding domain, outside the unsafe region. |
| 2025-09-17 | [An Automaton-based Characterisation of First-Order Logic over Infinite Trees](http://arxiv.org/abs/2509.14090v1) | Massimo Benerecetti, Dario Della Monica et al. | In this paper, we study First Order Logic (FO) over (unordered) infinite trees and its connection with branching-time temporal logics. More specifically, we provide an automata-theoretic characterisation of FO interpreted over infinite trees. To this end, two different classes of hesitant tree automata are introduced and proved to capture precisely the expressive power of two branching time temporal logics, denoted polcCTLp and cCTL*[f], which are, respectively, a restricted version of counting CTL with past and counting CTL* over finite paths, both of which have been previously shown equivalent to FO over infinite trees. The two automata characterisations naturally lead to normal forms for the two temporal logics, and highlight the fact that FO can only express properties of the tree branches which are either safety or co-safety in nature. |
| 2025-09-17 | [Constraint-Consistent Control of Task-Based and Kinematic RCM Constraints for Surgical Robots](http://arxiv.org/abs/2509.14075v1) | Yu Li, Hamid Sadeghian et al. | Robotic-assisted minimally invasive surgery (RAMIS) requires precise enforcement of the remote center of motion (RCM) constraint to ensure safe tool manipulation through a trocar. Achieving this constraint under dynamic and interactive conditions remains challenging, as existing control methods either lack robustness at the torque level or do not guarantee consistent RCM constraint satisfaction. This paper proposes a constraint-consistent torque controller that treats the RCM as a rheonomic holonomic constraint and embeds it into a projection-based inverse-dynamics framework. The method unifies task-level and kinematic formulations, enabling accurate tool-tip tracking while maintaining smooth and efficient torque behavior. The controller is validated both in simulation and on a RAMIS training platform, and is benchmarked against state-of-the-art approaches. Results show improved RCM constraint satisfaction, reduced required torque, and robust performance by improving joint torque smoothness through the consistency formulation under clinically relevant scenarios, including spiral trajectories, variable insertion depths, moving trocars, and human interaction. These findings demonstrate the potential of constraint-consistent torque control to enhance safety and reliability in surgical robotics. The project page is available at: https://rcmpc-cube.github.io |
| 2025-09-17 | [Flow-driven hysteresis in the transition boiling regime](http://arxiv.org/abs/2509.14042v1) | Alessandro Gabbana, Xander M. de Wit et al. | Transition boiling is an intermediate regime occurring between nucleate boiling, where bubbles at the surface efficiently carry heat away, and film boiling, where a layer of vapor formed over the surface insulates the system reducing heat transfer. This regime is inherently unstable and typically occurs near the boiling crisis, where the system approaches the maximum heat flux. Transition boiling hysteresis remains a central open problem in phase-change heat transfer, with critical implications for industrial cooling systems and nuclear reactor safety, since entering this regime sharply reduces heat removal potentially leading to overheating or component damage. We investigate the mechanisms driving hysteresis in the transition boiling regime through large-scale three-dimensional numerical simulations, providing clearcut evidence that hysteresis occurs even under idealized conditions of pool boiling on flat surfaces at constant temperature. This demonstrates that hysteresis arises purely from the flow dynamics of the liquid-vapor system, rather than from surface properties or defects. Moreover, we disclose strong asymmetries in the transition dynamics between nucleate and film boiling. During heating, the transition is abrupt and memory-less, whereas, upon decreasing the surface temperature, it is more complex, with the emergence of metastable coexisting states that can delay the transition. |
| 2025-09-17 | [SHaRe-RL: Structured, Interactive Reinforcement Learning for Contact-Rich Industrial Assembly Tasks](http://arxiv.org/abs/2509.13949v1) | Jannick Stranghöner, Philipp Hartmann et al. | High-mix low-volume (HMLV) industrial assembly, common in small and medium-sized enterprises (SMEs), requires the same precision, safety, and reliability as high-volume automation while remaining flexible to product variation and environmental uncertainty. Current robotic systems struggle to meet these demands. Manual programming is brittle and costly to adapt, while learning-based methods suffer from poor sample efficiency and unsafe exploration in contact-rich tasks. To address this, we present SHaRe-RL, a reinforcement learning framework that leverages multiple sources of prior knowledge. By (i) structuring skills into manipulation primitives, (ii) incorporating human demonstrations and online corrections, and (iii) bounding interaction forces with per-axis compliance, SHaRe-RL enables efficient and safe online learning for long-horizon, contact-rich industrial assembly tasks. Experiments on the insertion of industrial Harting connector modules with 0.2-0.4 mm clearance demonstrate that SHaRe-RL achieves reliable performance within practical time budgets. Our results show that process expertise, without requiring robotics or RL knowledge, can meaningfully contribute to learning, enabling safer, more robust, and more economically viable deployment of RL for industrial assembly. |
| 2025-09-17 | [Iterative Prompt Refinement for Safer Text-to-Image Generation](http://arxiv.org/abs/2509.13760v1) | Jinwoo Jeon, JunHyeok Oh et al. | Text-to-Image (T2I) models have made remarkable progress in generating images from text prompts, but their output quality and safety still depend heavily on how prompts are phrased. Existing safety methods typically refine prompts using large language models (LLMs), but they overlook the images produced, which can result in unsafe outputs or unnecessary changes to already safe prompts. To address this, we propose an iterative prompt refinement algorithm that uses Vision Language Models (VLMs) to analyze both the input prompts and the generated images. By leveraging visual feedback, our method refines prompts more effectively, improving safety while maintaining user intent and reliability comparable to existing LLM-based approaches. Additionally, we introduce a new dataset labeled with both textual and visual safety signals using off-the-shelf multi-modal LLM, enabling supervised fine-tuning. Experimental results demonstrate that our approach produces safer outputs without compromising alignment with user intent, offering a practical solution for generating safer T2I content. Our code is available at https://github.com/ku-dmlab/IPR. \textbf{\textcolor{red}WARNING: This paper contains examples of harmful or inappropriate images generated by models. |
| 2025-09-17 | [A Study on Thinking Patterns of Large Reasoning Models in Code Generation](http://arxiv.org/abs/2509.13758v1) | Kevin Halim, Sin G. Teo et al. | Currently, many large language models (LLMs) are utilized for software engineering tasks such as code generation. The emergence of more advanced models known as large reasoning models (LRMs), such as OpenAI's o3, DeepSeek R1, and Qwen3. They have demonstrated the capability of performing multi-step reasoning. Despite the advancement in LRMs, little attention has been paid to systematically analyzing the reasoning patterns these models exhibit and how such patterns influence the generated code. This paper presents a comprehensive study aimed at investigating and uncovering the reasoning behavior of LRMs during code generation. We prompted several state-of-the-art LRMs of varying sizes with code generation tasks and applied open coding to manually annotate the reasoning traces. From this analysis, we derive a taxonomy of LRM reasoning behaviors, encompassing 15 reasoning actions across four phases.   Our empirical study based on the taxonomy reveals a series of findings. First, we identify common reasoning patterns, showing that LRMs generally follow a human-like coding workflow, with more complex tasks eliciting additional actions such as scaffolding, flaw detection, and style checks. Second, we compare reasoning across models, finding that Qwen3 exhibits iterative reasoning while DeepSeek-R1-7B follows a more linear, waterfall-like approach. Third, we analyze the relationship between reasoning and code correctness, showing that actions such as unit test creation and scaffold generation strongly support functional outcomes, with LRMs adapting strategies based on task context. Finally, we evaluate lightweight prompting strategies informed by these findings, demonstrating the potential of context- and reasoning-oriented prompts to improve LRM-generated code. Our results offer insights and practical implications for advancing automatic code generation. |
| 2025-09-17 | [Reinforcement Learning for Robotic Insertion of Flexible Cables in Industrial Settings](http://arxiv.org/abs/2509.13731v1) | Jeongwoo Park, Seabin Lee et al. | The industrial insertion of flexible flat cables (FFCs) into receptacles presents a significant challenge owing to the need for submillimeter precision when handling the deformable cables. In manufacturing processes, FFC insertion with robotic manipulators often requires laborious human-guided trajectory generation. While Reinforcement Learning (RL) offers a solution to automate this task without modeling complex properties of FFCs, the nondeterminism caused by the deformability of FFCs requires significant efforts and time on training. Moreover, training directly in a real environment is dangerous as industrial robots move fast and possess no safety measure. We propose an RL algorithm for FFC insertion that leverages a foundation model-based real-to-sim approach to reduce the training time and eliminate the risk of physical damages to robots and surroundings. Training is done entirely in simulation, allowing for random exploration without the risk of physical damages. Sim-to-real transfer is achieved through semantic segmentation masks which leave only those visual features relevant to the insertion tasks such as the geometric and spatial information of the cables and receptacles. To enhance generality, we use a foundation model, Segment Anything Model 2 (SAM2). To eleminate human intervention, we employ a Vision-Language Model (VLM) to automate the initial prompting of SAM2 to find segmentation masks. In the experiments, our method exhibits zero-shot capabilities, which enable direct deployments to real environments without fine-tuning. |
| 2025-09-17 | [Conducting Mission-Critical Voice Experiments with Automated Speech Recognition and Crowdsourcing](http://arxiv.org/abs/2509.13724v1) | Jan Janak, Kahlil Dozier et al. | Mission-critical voice (MCV) communications systems have been a critical tool for the public safety community for over eight decades. Public safety users expect MCV systems to operate reliably and consistently, particularly in challenging conditions. Because of these expectations, the Public Safety Communications Research (PSCR) Division of the National Institute of Standards and Technology (NIST) has been interested in correlating impairments in MCV communication systems and public safety user quality of experience (QoE). Previous research has studied MCV voice quality and intelligibility in a controlled environment. However, such research has been limited by the challenges inherent in emulating real-world environmental conditions. Additionally, there is the question of the best metric to use to reflect QoE accurately.   This paper describes our efforts to develop the methodology and tools for human-subject experiments with MCV. We illustrate their use in human-subject experiments in emulated real-world environments. The tools include a testbed for emulating real-world MCV systems and an automated speech recognition (ASR) robot approximating human subjects in transcription tasks. We evaluate QoE through a Levenshtein Distance-based metric, arguing it is a suitable proxy for measuring comprehension and the QoE. We conducted human-subject studies with Amazon MTurk volunteers to understand the influence of selected system parameters and impairments on human subject performance and end-user QoE. We also compare the performance of several ASR system configurations with human-subject performance. We find that humans generally perform better than ASR in accuracy-related MCV tasks and that the codec significantly influences the end-user QoE and ASR performance. |
| 2025-09-17 | [Automated Triaging and Transfer Learning of Incident Learning Safety Reports Using Large Language Representational Models](http://arxiv.org/abs/2509.13706v1) | Peter Beidler, Mark Nguyen et al. | PURPOSE: Incident reports are an important tool for safety and quality improvement in healthcare, but manual review is time-consuming and requires subject matter expertise. Here we present a natural language processing (NLP) screening tool to detect high-severity incident reports in radiation oncology across two institutions.   METHODS AND MATERIALS: We used two text datasets to train and evaluate our NLP models: 7,094 reports from our institution (Inst.), and 571 from IAEA SAFRON (SF), all of which had severity scores labeled by clinical content experts. We trained and evaluated two types of models: baseline support vector machines (SVM) and BlueBERT which is a large language model pretrained on PubMed abstracts and hospitalized patient data. We assessed for generalizability of our model in two ways. First, we evaluated models trained using Inst.-train on SF-test. Second, we trained a BlueBERT_TRANSFER model that was first fine-tuned on Inst.-train then on SF-train before testing on SF-test set. To further analyze model performance, we also examined a subset of 59 reports from our Inst. dataset, which were manually edited for clarity.   RESULTS Classification performance on the Inst. test achieved AUROC 0.82 using SVM and 0.81 using BlueBERT. Without cross-institution transfer learning, performance on the SF test was limited to an AUROC of 0.42 using SVM and 0.56 using BlueBERT. BlueBERT_TRANSFER, which was fine-tuned on both datasets, improved the performance on SF test to AUROC 0.78. Performance of SVM, and BlueBERT_TRANSFER models on the manually curated Inst. reports (AUROC 0.85 and 0.74) was similar to human performance (AUROC 0.81).   CONCLUSION: In summary, we successfully developed cross-institution NLP models on incident report text from radiation oncology centers. These models were able to detect high-severity reports similarly to humans on a curated dataset. |
| 2025-09-17 | [InfraMind: A Novel Exploration-based GUI Agentic Framework for Mission-critical Industrial Management](http://arxiv.org/abs/2509.13704v1) | Liangtao Lin, Zhaomeng Zhu et al. | Mission-critical industrial infrastructure, such as data centers, increasingly depends on complex management software. Its operations, however, pose significant challenges due to the escalating system complexity, multi-vendor integration, and a shortage of expert operators. While Robotic Process Automation (RPA) offers partial automation through handcrafted scripts, it suffers from limited flexibility and high maintenance costs. Recent advances in Large Language Model (LLM)-based graphical user interface (GUI) agents have enabled more flexible automation, yet these general-purpose agents face five critical challenges when applied to industrial management, including unfamiliar element understanding, precision and efficiency, state localization, deployment constraints, and safety requirements. To address these issues, we propose InfraMind, a novel exploration-based GUI agentic framework specifically tailored for industrial management systems. InfraMind integrates five innovative modules to systematically resolve different challenges in industrial management: (1) systematic search-based exploration with virtual machine snapshots for autonomous understanding of complex GUIs; (2) memory-driven planning to ensure high-precision and efficient task execution; (3) advanced state identification for robust localization in hierarchical interfaces; (4) structured knowledge distillation for efficient deployment with lightweight models; and (5) comprehensive, multi-layered safety mechanisms to safeguard sensitive operations. Extensive experiments on both open-source and commercial DCIM platforms demonstrate that our approach consistently outperforms existing frameworks in terms of task success rate and operational efficiency, providing a rigorous and scalable solution for industrial management automation. |
| 2025-09-17 | [Multi-Threaded Software Model Checking via Parallel Trace Abstraction Refinement](http://arxiv.org/abs/2509.13699v1) | Max Barth, Marie-Christine Jakobs | Automatic software verification is a valuable means for software quality assurance. However, automatic verification and in particular software model checking can be time-consuming, which hinders their practical applicability e.g., the use in continuous integration. One solution to address the issue is to reduce the response time of the verification procedure by leveraging today's multi-core CPUs.   In this paper, we propose a solution to parallelize trace abstraction, an abstraction-based approach to software model checking. The underlying idea of our approach is to parallelize the abstraction refinement. More concretely, our approach analyzes different traces (syntactic program paths) that could violate the safety property in parallel. We realize our parallelized version of trace abstraction in the verification tool Ulti mate Automizer and perform a thorough evaluation. Our evaluation shows that our parallelization is more effective than sequential trace abstraction and can provide results significantly faster on many time-consuming tasks. Also, our approach is more effective than DSS, a recent parallel approach to abstraction-based software model checking. |
| 2025-09-17 | [Is GPT-4o mini Blinded by its Own Safety Filters? Exposing the Multimodal-to-Unimodal Bottleneck in Hate Speech Detection](http://arxiv.org/abs/2509.13608v1) | Niruthiha Selvanayagam, Ted Kurti | As Large Multimodal Models (LMMs) become integral to daily digital life, understanding their safety architectures is a critical problem for AI Alignment. This paper presents a systematic analysis of OpenAI's GPT-4o mini, a globally deployed model, on the difficult task of multimodal hate speech detection. Using the Hateful Memes Challenge dataset, we conduct a multi-phase investigation on 500 samples to probe the model's reasoning and failure modes. Our central finding is the experimental identification of a "Unimodal Bottleneck," an architectural flaw where the model's advanced multimodal reasoning is systematically preempted by context-blind safety filters. A quantitative validation of 144 content policy refusals reveals that these overrides are triggered in equal measure by unimodal visual 50% and textual 50% content. We further demonstrate that this safety system is brittle, blocking not only high-risk imagery but also benign, common meme formats, leading to predictable false positives. These findings expose a fundamental tension between capability and safety in state-of-the-art LMMs, highlighting the need for more integrated, context-aware alignment strategies to ensure AI systems can be deployed both safely and effectively. |
| 2025-09-16 | [Leg-Arm Coordinated Operation for Curtain Wall Installation](http://arxiv.org/abs/2509.13595v1) | Xiao Liu, Weijun Wang et al. | With the acceleration of urbanization, the number of high-rise buildings and large public facilities is increasing, making curtain walls an essential component of modern architecture with widespread applications. Traditional curtain wall installation methods face challenges such as variable on-site terrain, high labor intensity, low construction efficiency, and significant safety risks. Large panels often require multiple workers to complete installation. To address these issues, based on a hexapod curtain wall installation robot, we design a hierarchical optimization-based whole-body control framework for coordinated arm-leg planning tailored to three key tasks: wall installation, ceiling installation, and floor laying. This framework integrates the motion of the hexapod legs with the operation of the folding arm and the serial-parallel manipulator. We conduct experiments on the hexapod curtain wall installation robot to validate the proposed control method, demonstrating its capability in performing curtain wall installation tasks. Our results confirm the effectiveness of the hierarchical optimization-based arm-leg coordination framework for the hexapod robot, laying the foundation for its further application in complex construction site environments. |
| 2025-09-16 | [TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning](http://arxiv.org/abs/2509.13579v1) | Momchil S. Tomov, Sang Uk Lee et al. | We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving. |
| 2025-09-16 | [Overview of Dialog System Evaluation Track: Dimensionality, Language, Culture and Safety at DSTC 12](http://arxiv.org/abs/2509.13569v1) | John Mendonça, Lining Zhang et al. | The rapid advancement of Large Language Models (LLMs) has intensified the need for robust dialogue system evaluation, yet comprehensive assessment remains challenging. Traditional metrics often prove insufficient, and safety considerations are frequently narrowly defined or culturally biased. The DSTC12 Track 1, "Dialog System Evaluation: Dimensionality, Language, Culture and Safety," is part of the ongoing effort to address these critical gaps. The track comprised two subtasks: (1) Dialogue-level, Multi-dimensional Automatic Evaluation Metrics, and (2) Multilingual and Multicultural Safety Detection. For Task 1, focused on 10 dialogue dimensions, a Llama-3-8B baseline achieved the highest average Spearman's correlation (0.1681), indicating substantial room for improvement. In Task 2, while participating teams significantly outperformed a Llama-Guard-3-1B baseline on the multilingual safety subset (top ROC-AUC 0.9648), the baseline proved superior on the cultural subset (0.5126 ROC-AUC), highlighting critical needs in culturally-aware safety. This paper describes the datasets and baselines provided to participants, as well as submission evaluation results for each of the two proposed subtasks. |
| 2025-09-16 | [The impact of modeling approaches on controlling safety-critical, highly perturbed systems: the case for data-driven models](http://arxiv.org/abs/2509.13531v1) | Piotr Łaszkiewicz, Maria Carvalho et al. | This paper evaluates the impact of three system models on the reference trajectory tracking error of the LQR optimal controller, in the challenging problem of guidance and control of the state of a system under strong perturbations and reconfiguration. We compared a smooth Linear Time Variant system learned from data (DD-LTV) with state of the art Linear Time Variant (LTV) system identification methods, showing its superiority in the task of state propagation. Moreover, we have found that DD-LTV allows for better performance in terms of trajectory tracking error than the standard solutions of a Linear Time Invariant (LTI) system model, and comparable performance to a linearized Linear Time Variant (L-LTV) system model. We tested the three approaches on the perturbed and time varying spring-mass-damper systems. |
| 2025-09-16 | [Multimodal Hate Detection Using Dual-Stream Graph Neural Networks](http://arxiv.org/abs/2509.13515v1) | Jiangbei Yue, Shuonan Yang et al. | Hateful videos present serious risks to online safety and real-world well-being, necessitating effective detection methods. Although multimodal classification approaches integrating information from several modalities outperform unimodal ones, they typically neglect that even minimal hateful content defines a video's category. Specifically, they generally treat all content uniformly, instead of emphasizing the hateful components. Additionally, existing multimodal methods cannot systematically capture structured information in videos, limiting the effectiveness of multimodal fusion. To address these limitations, we propose a novel multimodal dual-stream graph neural network model. It constructs an instance graph by separating the given video into several instances to extract instance-level features. Then, a complementary weight graph assigns importance weights to these features, highlighting hateful instances. Importance weights and instance features are combined to generate video labels. Our model employs a graph-based framework to systematically model structured relationships within and across modalities. Extensive experiments on public datasets show that our model is state-of-the-art in hateful video classification and has strong explainability. Code is available: https://github.com/Multimodal-Intelligence-Lab-MIL/MultiHateGNN. |

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



