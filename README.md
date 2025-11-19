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
| 2025-11-18 | [Robust Verification of Controllers under State Uncertainty via Hamilton-Jacobi Reachability Analysis](http://arxiv.org/abs/2511.14755v1) | Albert Lin, Alessandro Pinto et al. | As perception-based controllers for autonomous systems become increasingly popular in the real world, it is important that we can formally verify their safety and performance despite perceptual uncertainty. Unfortunately, the verification of such systems remains challenging, largely due to the complexity of the controllers, which are often nonlinear, nonconvex, learning-based, and/or black-box. Prior works propose verification algorithms that are based on approximate reachability methods, but they often restrict the class of controllers and systems that can be handled or result in overly conservative analyses. Hamilton-Jacobi (HJ) reachability analysis is a popular formal verification tool for general nonlinear systems that can compute optimal reachable sets under worst-case system uncertainties; however, its application to perception-based systems is currently underexplored. In this work, we propose RoVer-CoRe, a framework for the Robust Verification of Controllers via HJ Reachability. To the best of our knowledge, RoVer-CoRe is the first HJ reachability-based framework for the verification of perception-based systems under perceptual uncertainty. Our key insight is to concatenate the system controller, observation function, and the state estimation modules to obtain an equivalent closed-loop system that is readily compatible with existing reachability frameworks. Within RoVer-CoRe, we propose novel methods for formal safety verification and robust controller design. We demonstrate the efficacy of the framework in case studies involving aircraft taxiing and NN-based rover navigation. Code is available at the link in the footnote. |
| 2025-11-18 | [Talk, Snap, Complain: Validation-Aware Multimodal Expert Framework for Fine-Grained Customer Grievances](http://arxiv.org/abs/2511.14693v1) | Rishu Kumar Singh, Navneet Shreya et al. | Existing approaches to complaint analysis largely rely on unimodal, short-form content such as tweets or product reviews. This work advances the field by leveraging multimodal, multi-turn customer support dialogues, where users often share both textual complaints and visual evidence (e.g., screenshots, product photos) to enable fine-grained classification of complaint aspects and severity. We introduce VALOR, a Validation-Aware Learner with Expert Routing, tailored for this multimodal setting. It employs a multi-expert reasoning setup using large-scale generative models with Chain-of-Thought (CoT) prompting for nuanced decision-making. To ensure coherence between modalities, a semantic alignment score is computed and integrated into the final classification through a meta-fusion strategy. In alignment with the United Nations Sustainable Development Goals (UN SDGs), the proposed framework supports SDG 9 (Industry, Innovation and Infrastructure) by advancing AI-driven tools for robust, scalable, and context-aware service infrastructure. Further, by enabling structured analysis of complaint narratives and visual context, it contributes to SDG 12 (Responsible Consumption and Production) by promoting more responsive product design and improved accountability in consumer services. We evaluate VALOR on a curated multimodal complaint dataset annotated with fine-grained aspect and severity labels, showing that it consistently outperforms baseline models, especially in complex complaint scenarios where information is distributed across text and images. This study underscores the value of multimodal interaction and expert validation in practical complaint understanding systems. Resources related to data and codes are available here: https://github.com/sarmistha-D/VALOR |
| 2025-11-18 | [Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks](http://arxiv.org/abs/2511.14592v1) | Xianhui Meng, Yuchen Zhang et al. | Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible. |
| 2025-11-18 | [Direct Neutron Detectors based on Carborane Containing Conjugated Polymers](http://arxiv.org/abs/2511.14561v1) | Aled Horner, Fani E. Taifakou et al. | Thermal neutron detectors are crucial to a wide range of applications, including nuclear safety and security, cancer treatment, space research, non-destructive testing, and more. However, neutrons are notoriously difficult to capture due to their absence of charge, and only a handful of isotopes have a sufficient neutron cross-section. Meanwhile, commercially available $^3$He gas filled proportional counters suffer from depleting $^3$He feedstocks and complex device structures. In this work, we explore the potential of a carborane containing conjugated polymer ($o$CbT$_2$-NDI) as a thermal neutron detector. The natural abundance of $^{10}$B in such a polymer enables intrinsic thermal neutron capture of the material, making it the first demonstration of an organic semiconductor with such capabilities. In addition, we show that thermal neutron detection can be achieved also by adding a $^{10}$B$_4$C sensitiser additive to the analogous carborane-free polymer PNDI(2OD)2T, whereas unsensitised PNDI(2OD)2T control devices only respond to the fast neutron component of the radiation field. This approach allows us to disentangle the fast and thermal neutron responses of the devices tested and compare the relative performance of the two approaches to thermal neutron detection. Both the carborane containing and the $^{10}$B$_4$C sensitised devices displayed enhancement due to thermal neutrons, above that of the unsensitised polymer. The detector response is found to be linear with flux up to $1.796\,\times\,10^7\,$cm$^{-2}$s$^{-1}$ n$_{th}\bar{v}$ and saturates at high drive biases. This study demonstrates the viability of carboranyl polymers as neutron detectors, highlights the inherent chemical tuneability of organic semiconductors, and opens the possibility of their application to a number of different low-cost, scalable, and easily processable detector technologies. |
| 2025-11-18 | [Aerial Assistance System for Automated Firefighting during Turntable Ladder Operations](http://arxiv.org/abs/2511.14504v1) | Jan Quenzel, Valerij Sekin et al. | Fires in industrial facilities pose special challenges to firefighters, e.g., due to the sheer size and scale of the buildings. The resulting visual obstructions impair firefighting accuracy, further compounded by inaccurate assessments of the fire's location. Such imprecision simultaneously increases the overall damage and prolongs the fire-brigades operation unnecessarily.   We propose an automated assistance system for firefighting using a motorized fire monitor on a turntable ladder with aerial support from an unmanned aerial vehicle (UAV). The UAV flies autonomously within an obstacle-free flight funnel derived from geodata, detecting and localizing heat sources. An operator supervises the operation on a handheld controller and selects a fire target in reach. After the selection, the UAV automatically plans and traverses between two triangulation poses for continued fire localization. Simultaneously, our system steers the fire monitor to ensure the water jet reaches the detected heat source. In preliminary tests, our assistance system successfully localized multiple heat sources and directed a water jet towards the fires. |
| 2025-11-18 | [Operationalizing Pluralistic Values in Large Language Model Alignment Reveals Trade-offs in Safety, Inclusivity, and Model Behavior](http://arxiv.org/abs/2511.14476v1) | Dalia Ali, Dora Zhao et al. | Although large language models (LLMs) are increasingly trained using human feedback for safety and alignment with human values, alignment decisions often overlook human social diversity. This study examines how incorporating pluralistic values affects LLM behavior by systematically evaluating demographic variation and design parameters in the alignment pipeline. We collected alignment data from US and German participants (N = 1,095, 27,375 ratings) who rated LLM responses across five dimensions: Toxicity, Emotional Awareness (EA), Sensitivity, Stereotypical Bias, and Helpfulness. We fine-tuned multiple Large Language Models and Large Reasoning Models using preferences from different social groups while varying rating scales, disagreement handling methods, and optimization techniques. The results revealed systematic demographic effects: male participants rated responses 18% less toxic than female participants; conservative and Black participants rated responses 27.9% and 44% more emotionally aware than liberal and White participants, respectively. Models fine-tuned on group-specific preferences exhibited distinct behaviors. Technical design choices showed strong effects: the preservation of rater disagreement achieved roughly 53% greater toxicity reduction than majority voting, and 5-point scales yielded about 22% more reduction than binary formats; and Direct Preference Optimization (DPO) consistently outperformed Group Relative Policy Optimization (GRPO) in multi-value optimization. These findings represent a preliminary step in answering a critical question: How should alignment balance expert-driven and user-driven signals to ensure both safety and fair representation? |
| 2025-11-18 | [Learning Subglacial Bed Topography from Sparse Radar with Physics-Guided Residuals](http://arxiv.org/abs/2511.14473v1) | Bayu Adhi Tama, Jianwu Wang et al. | Accurate subglacial bed topography is essential for ice sheet modeling, yet radar observations are sparse and uneven. We propose a physics-guided residual learning framework that predicts bed thickness residuals over a BedMachine prior and reconstructs bed from the observed surface. A DeepLabV3+ decoder over a standard encoder (e.g.,ResNet-50) is trained with lightweight physics and data terms: multi-scale mass conservation, flow-aligned total variation, Laplacian damping, non-negativity of thickness, a ramped prior-consistency term, and a masked Huber fit to radar picks modulated by a confidence map. To measure real-world generalization, we adopt leakage-safe blockwise hold-outs (vertical/horizontal) with safety buffers and report metrics only on held-out cores. Across two Greenland sub-regions, our approach achieves strong test-core accuracy and high structural fidelity, outperforming U-Net, Attention U-Net, FPN, and a plain CNN. The residual-over-prior design, combined with physics, yields spatially coherent, physically plausible beds suitable for operational mapping under domain shift. |
| 2025-11-18 | [From Flash to Crater: Morphological and Spectral Analysis of the Brightest Lunar Impact on 11 September 2013 using LRO Data](http://arxiv.org/abs/2511.14442v1) | J. L. Rizos, L. M. Lara et al. | We present a comprehensive morphological and spectrophotometric analysis of the lunar impact that occurred on September 11, 2013, based on pre- and post-event observations by the Lunar Reconnaissance Orbiter (LRO). The crater formed exhibits a rim-to-rim diameter of $35 \pm 0.7$ m, a depth of $4.9 \pm 0.4$ m, and an ejecta blanket extending over 2 km with an area of approximately $7 \times 10^{5}\,$m$^{2}$. The ejecta shows a pronounced asymmetry and, assuming uniform distribution, an average thickness limit of $\sim 2$ mm. Spectral analysis using WAC images reveals a consistent reddening of the central ejecta region, with an average 16.54 % increase in spectral slope between 321 nm and 643 nm, marking the first reported detection of color changes resulting from a lunar impact. We evaluated several scaling laws and found that the Gault et al. (1974) formulation most accurately reproduces the observed crater size. Furthermore, luminous efficiency values below $η= 2 \times 10^{-3}$ and higher projectile densities are most consistent with the morphology of the ejecta. The impact direction inferred from this pattern is not compatible with the radiant of the September $\varepsilon$-Perseids stream. Moreover, an independent probability analysis yields a greater than 96 % likelihood that the event was caused by a sporadic meteoroid. Our results also demonstrate the potential of WAC imagery for the automated detection of new lunar craters, which can improve statistical estimates of the current impact flux. This methodology offers a powerful complement to high-resolution imaging, with important implications for both lunar safety and planetary defense. |
| 2025-11-18 | [MedBench v4: A Robust and Scalable Benchmark for Evaluating Chinese Medical Language Models, Multimodal Models, and Intelligent Agents](http://arxiv.org/abs/2511.14439v1) | Jinru Ding, Lu Lu et al. | Recent advances in medical large language models (LLMs), multimodal models, and agents demand evaluation frameworks that reflect real clinical workflows and safety constraints. We present MedBench v4, a nationwide, cloud-based benchmarking infrastructure comprising over 700,000 expert-curated tasks spanning 24 primary and 91 secondary specialties, with dedicated tracks for LLMs, multimodal models, and agents. Items undergo multi-stage refinement and multi-round review by clinicians from more than 500 institutions, and open-ended responses are scored by an LLM-as-a-judge calibrated to human ratings. We evaluate 15 frontier models. Base LLMs reach a mean overall score of 54.1/100 (best: Claude Sonnet 4.5, 62.5/100), but safety and ethics remain low (18.4/100). Multimodal models perform worse overall (mean 47.5/100; best: GPT-5, 54.9/100), with solid perception yet weaker cross-modal reasoning. Agents built on the same backbones substantially improve end-to-end performance (mean 79.8/100), with Claude Sonnet 4.5-based agents achieving up to 85.3/100 overall and 88.9/100 on safety tasks. MedBench v4 thus reveals persisting gaps in multimodal reasoning and safety for base models, while showing that governance-aware agentic orchestration can markedly enhance benchmarked clinical readiness without sacrificing capability. By aligning tasks with Chinese clinical guidelines and regulatory priorities, the platform offers a practical reference for hospitals, developers, and policymakers auditing medical AI. |
| 2025-11-18 | [Towards A Catalogue of Requirement Patterns for Space Robotic Missions](http://arxiv.org/abs/2511.14438v1) | Mahdi Etumi, Hazel M. Taylor et al. | In the development of safety and mission-critical systems, including autonomous space robotic missions, complex behaviour is captured during the requirements elicitation phase. Requirements are typically expressed using natural language which is ambiguous and not amenable to formal verification methods that can provide robust guarantees of system behaviour. To support the definition of formal requirements, specification patterns provide reusable, logic-based templates. A suite of robotic specification patterns, along with their formalisation in NASA's Formal Requirements Elicitation Tool (FRET) already exists. These pre-existing requirement patterns are domain agnostic and, in this paper we explore their applicability for space missions. To achieve this we carried out a literature review of existing space missions and formalised their requirements using FRET, contributing a corpus of space mission requirements. We categorised these requirements using pre-existing specification patterns which demonstrated their applicability in space missions. However, not all of the requirements that we formalised corresponded to an existing pattern so we have contributed 5 new requirement specification patterns as well as several variants of the existing and new patterns. We also conducted an expert evaluation of the new patterns, highlighting their benefits and limitations. |
| 2025-11-18 | [Model Learning for Adjusting the Level of Automation in HCPS](http://arxiv.org/abs/2511.14437v1) | Mehrnoush Hajnorouzi, Astrid Rakow et al. | The steadily increasing level of automation in human-centred systems demands rigorous design methods for analysing and controlling interactions between humans and automated components, especially in safety-critical applications. The variability of human behaviour poses particular challenges for formal verification and synthesis. We present a model-based framework that enables design-time exploration of safe shared-control strategies in human-automation systems. The approach combines active automata learning -- to derive coarse, finite-state abstractions of human behaviour from simulations -- with game-theoretic reactive synthesis to determine whether a controller can guarantee safety when interacting with these models. If no such strategy exists, the framework supports iterative refinement of the human model or adjustment of the automation's controllable actions. A driving case study, integrating automata learning with reactive synthesis in UPPAAL, illustrates the applicability of the framework on a simplified driving scenario and its potential for analysing shared-control strategies in human-centred cyber-physical systems. |
| 2025-11-18 | [Watchdogs and Oracles: Runtime Verification Meets Large Language Models for Autonomous Systems](http://arxiv.org/abs/2511.14435v1) | Angelo Ferrando | Assuring the safety and trustworthiness of autonomous systems is particularly difficult when learning-enabled components and open environments are involved. Formal methods provide strong guarantees but depend on complete models and static assumptions. Runtime verification (RV) complements them by monitoring executions at run time and, in its predictive variants, by anticipating potential violations. Large language models (LLMs), meanwhile, excel at translating natural language into formal artefacts and recognising patterns in data, yet they remain error-prone and lack formal guarantees. This vision paper argues for a symbiotic integration of RV and LLMs. RV can serve as a guardrail for LLM-driven autonomy, while LLMs can extend RV by assisting specification capture, supporting anticipatory reasoning, and helping to handle uncertainty. We outline how this mutual reinforcement differs from existing surveys and roadmaps, discuss challenges and certification implications, and identify future research directions towards dependable autonomy. |
| 2025-11-18 | [Achieving Safe Control Online through Integration of Harmonic Control Lyapunov-Barrier Functions with Unsafe Object-Centric Action Policies](http://arxiv.org/abs/2511.14434v1) | Marlow Fawn, Matthias Scheutz | We propose a method for combining Harmonic Control Lyapunov-Barrier Functions (HCLBFs) derived from Signal Temporal Logic (STL) specifications with any given robot policy to turn an unsafe policy into a safe one with formal guarantees.  The two components are combined via HCLBF-derived safety certificates, thus producing commands that preserve both safety and task-driven behavior.  We demonstrate with a simple proof-of-concept implementation for an object-centric force-based policy trained through reinforcement learning for a movement task of a stationary robot arm that is able to avoid colliding with obstacles on a table top after combining the policy with the safety constraints.  The proposed method can be generalized to more complex specifications and dynamic task settings. |
| 2025-11-18 | [Safe-ROS: An Architecture for Autonomous Robots in Safety-Critical Domains](http://arxiv.org/abs/2511.14433v1) | Diana C. Benjumea, Marie Farrell et al. | Deploying autonomous robots in safety-critical domains requires architectures that ensure operational effectiveness and safety compliance. In this paper, we contribute the Safe-ROS architecture for developing reliable and verifiable autonomous robots in such domains. It features two distinct subsystems: (1) an intelligent control system that is responsible for normal/routine operations, and (2) a Safety System consisting of Safety Instrumented Functions (SIFs) that provide formally verifiable independent oversight. We demonstrate Safe-ROS on an AgileX Scout Mini robot performing autonomous inspection in a nuclear environment. One safety requirement is selected and instantiated as a SIF. To support verification, we implement the SIF as a cognitive agent, programmed to stop the robot whenever it detects that it is too close to an obstacle. We verify that the agent meets the safety requirement and integrate it into the autonomous inspection. This integration is also verified, and the full deployment is validated in a Gazebo simulation, and lab testing. We evaluate this architecture in the context of the UK nuclear sector, where safety and regulation are crucial aspects of deployment. Success criteria include the development of a formal property from the safety requirement, implementation, and verification of the SIF, and the integration of the SIF into the operational robotic autonomous system. Our results demonstrate that the  Safe-ROS architecture can provide safety verifiable oversight while deploying autonomous robots in safety-critical domains, offering a robust framework that can be extended to additional requirements and various applications. |
| 2025-11-18 | [Abstract Scene Graphs: Formalizing and Monitoring Spatial Properties of Automated Driving Functions](http://arxiv.org/abs/2511.14430v1) | Ishan Saxena, Bernd Westphal et al. | Automated Driving Functions (ADFs) need to comply with spatial properties of varied complexity while driving on public roads. Since such situations are safety-critical in nature, it is necessary to continuously check ADFs for compliance with their spatial properties. Due to their complexity, such spatial properties need to be formalized to enable their automated checking. Scene Graphs (SGs) allow for an explicit structured representation of objects present in a traffic scene and their spatial relationships to each other. In this paper, we build upon the SG construct and propose the Abstract Scene Graph (ASG) formalism to formalize spatial properties of ADFs. We show using real-world examples how spatial properties can be formalized using ASGs. Finally, we present a framework that uses ASGs to perform Runtime Monitoring of ADFs. To this end, we also show algorithmically how a spatial property formalized as an ASG can be satisfied by ADF system behaviour. |
| 2025-11-18 | [Context-aware, Ante-hoc Explanations of Driving Behaviour](http://arxiv.org/abs/2511.14428v1) | Dominik Grundt, Ishan Saxena et al. | Autonomous vehicles (AVs) must be both safe and trustworthy to gain social acceptance and become a viable option for everyday public transportation. Explanations about the system behaviour can increase safety and trust in AVs. Unfortunately, explaining the system behaviour of AI-based driving functions is particularly challenging, as decision-making processes are often opaque. The field of Explainability Engineering tackles this challenge by developing explanation models at design time. These models are designed from system design artefacts and stakeholder needs to develop correct and good explanations. To support this field, we propose an approach that enables context-aware, ante-hoc explanations of (un)expectable driving manoeuvres at runtime. The visual yet formal language Traffic Sequence Charts is used to formalise explanation contexts, as well as corresponding (un)expectable driving manoeuvres. A dedicated runtime monitoring enables context-recognition and ante-hoc presentation of explanations at runtime. In combination, we aim to support the bridging of correct and good explanations. Our method is demonstrated in a simulated overtaking. |
| 2025-11-18 | [Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education](http://arxiv.org/abs/2511.14423v1) | Xin Yi, Yue Li et al. | Large Language Models (LLMs) are increasingly integrated into educational applications. However, they remain vulnerable to jailbreak and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. Existing studies mainly focus on general safety evaluations, with limited attention to the unique safety requirements of educational scenarios. To address this gap, we construct EduHarm, a benchmark containing safe-unsafe instruction pairs across five representative educational scenarios, enabling systematic safety evaluation of educational LLMs. Furthermore, we propose a three-stage shield framework (TSSF) for educational LLMs that simultaneously mitigates both jailbreak and fine-tuning attacks. First, safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Second, layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Finally, defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TSSF effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning. |
| 2025-11-18 | [Enhancing LLM-based Autonomous Driving with Modular Traffic Light and Sign Recognition](http://arxiv.org/abs/2511.14391v1) | Fabian Schmidt, Noushiq Mohammed Kayilan Abdul Nazar et al. | Large Language Models (LLMs) are increasingly used for decision-making and planning in autonomous driving, showing promising reasoning capabilities and potential to generalize across diverse traffic situations. However, current LLM-based driving agents lack explicit mechanisms to enforce traffic rules and often struggle to reliably detect small, safety-critical objects such as traffic lights and signs. To address this limitation, we introduce TLS-Assist, a modular redundancy layer that augments LLM-based autonomous driving agents with explicit traffic light and sign recognition. TLS-Assist converts detections into structured natural language messages that are injected into the LLM input, enforcing explicit attention to safety-critical cues. The framework is plug-and-play, model-agnostic, and supports both single-view and multi-view camera setups. We evaluate TLS-Assist in a closed-loop setup on the LangAuto benchmark in CARLA. The results demonstrate relative driving performance improvements of up to 14% over LMDrive and 7% over BEVDriver, while consistently reducing traffic light and sign infractions. We publicly release the code and models on https://github.com/iis-esslingen/TLS-Assist. |
| 2025-11-18 | [Emergent Cooperative Driving Strategies for Stop-and-Go Wave Mitigation via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2511.14378v1) | Raphael Korbmacher, Daniel Straub et al. | Stop-and-go waves in traffic flow pose a persistent challenge, compromising safety, efficiency, and environmental sustainability. This paper introduces a novel mitigation strategy discovered through training multi-agent deep reinforcement learning (DRL) agents in a simulated ring-road environment. The agents autonomously develop a cooperative driving policy, where most vehicles maintain minimal headways to maximize throughput, while a single "buffer" vehicle adopts a larger headway to absorb perturbations and prevent wave propagation. This strategy enhances stability without sacrificing overall flow. We further demonstrate that adapting this cooperative strategy to classical car-following models, such as the Intelligent Driver Model (IDM), yields improved stability and traffic efficiency. Furthermore, we show within a parametrised linear framework, that the cooperative strategy can optimise system performance under stability constraints. Our findings offer promising insights for future autonomous vehicle systems and highway management. |
| 2025-11-18 | [ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](http://arxiv.org/abs/2511.14342v1) | Xingwei He, Qianru Zhang et al. | Instruction-following is a critical capability of Large Language Models (LLMs). While existing works primarily focus on assessing how well LLMs adhere to user instructions, they often overlook scenarios where instructions contain conflicting constraints-a common occurrence in complex prompts. The behavior of LLMs under such conditions remains under-explored. To bridge this gap, we introduce ConInstruct, a benchmark specifically designed to assess LLMs' ability to detect and resolve conflicts within user instructions. Using this dataset, we evaluate LLMs' conflict detection performance and analyze their conflict resolution behavior. Our experiments reveal two key findings: (1) Most proprietary LLMs exhibit strong conflict detection capabilities, whereas among open-source models, only DeepSeek-R1 demonstrates similarly strong performance. DeepSeek-R1 and Claude-4.5-Sonnet achieve the highest average F1-scores at 91.5% and 87.3%, respectively, ranking first and second overall. (2) Despite their strong conflict detection abilities, LLMs rarely explicitly notify users about the conflicts or request clarification when faced with conflicting constraints. These results underscore a critical shortcoming in current LLMs and highlight an important area for future improvement when designing instruction-following LLMs. |

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



