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
| 2025-11-14 | [Who Moved My Distribution? Conformal Prediction for Interactive Multi-Agent Systems](http://arxiv.org/abs/2511.11567v1) | Allen Emmanuel Binny, Anushri Dixit | Uncertainty-aware prediction is essential for safe motion planning, especially when using learned models to forecast the behavior of surrounding agents. Conformal prediction is a statistical tool often used to produce uncertainty-aware prediction regions for machine learning models. Most existing frameworks utilizing conformal prediction-based uncertainty predictions assume that the surrounding agents are non-interactive. This is because in closed-loop, as uncertainty-aware agents change their behavior to account for prediction uncertainty, the surrounding agents respond to this change, leading to a distribution shift which we call endogenous distribution shift. To address this challenge, we introduce an iterative conformal prediction framework that systematically adapts the uncertainty-aware ego-agent controller to the endogenous distribution shift. The proposed method provides probabilistic safety guarantees while adapting to the evolving behavior of reactive, non-ego agents. We establish a model for the endogenous distribution shift and provide the conditions for the iterative conformal prediction pipeline to converge under such a distribution shift. We validate our framework in simulation for 2- and 3- agent interaction scenarios, demonstrating collision avoidance without resulting in overly conservative behavior and an overall improvement in success rates of up to 9.6% compared to other conformal prediction-based baselines. |
| 2025-11-14 | [CertiA360: Enhance Compliance Agility in Aerospace Software Development](http://arxiv.org/abs/2511.11550v1) | J. Antonio Dantas Macedo, Hugo Fernandes et al. | Agile methods are characterised by iterative and incremental processes with a strong focus on flexibility and accommodating changing requirements based on either technical, regulatory, or stakeholder feedback. However, integrating Agile methods into safety-critical system development in the aerospace industry presents substantial challenges due to its strict compliance requirements, such as those outlined in the DO-178C standard. To achieve this vision, the flexibility of Agile must align with the rigorous certification guidelines, which emphasize documentation, traceability of requirements across different levels and disciplines, and comprehensive verification and validation (V&V) activities. The research work described in this paper proposes a way of using the strengths of the flexible nature of Agile methods to automate and manage change requests throughout the whole software development lifecycle, ensuring robust traceability, regulatory compliance and ultimately facilitating successful certification. This study proposes CertiA360, a tool designed to help teams improve requirement maturity, automate the changes in traceability, and align with the regulatory objectives. The tool was designed and validated in close collaboration with aerospace industry experts, using their feedback to ensure practical application and real-life effectiveness. The feedback collected demonstrated that the automation given by CertiA360 may reduce manual effort and allow response to changing requirements while ensuring compliance with DO-178C. While the tool is not yet qualified under DO-330 (Tool Qualification), findings suggest that when tailored appropriately, Agile methods can not only coexist with the requirements of safety-system development and certification in highly regulated domains like aerospace, but also add efficiency. |
| 2025-11-14 | [Scalable Policy Evaluation with Video World Models](http://arxiv.org/abs/2511.11520v1) | Wei-Cheng Tseng, Jinwei Gu et al. | Training generalist policies for robotic manipulation has shown great promise, as they enable language-conditioned, multi-task behaviors across diverse scenarios. However, evaluating these policies remains difficult because real-world testing is expensive, time-consuming, and labor-intensive. It also requires frequent environment resets and carries safety risks when deploying unproven policies on physical robots. Manually creating and populating simulation environments with assets for robotic manipulation has not addressed these issues, primarily due to the significant engineering effort required and the often substantial sim-to-real gap, both in terms of physics and rendering. In this paper, we explore the use of action-conditional video generation models as a scalable way to learn world models for policy evaluation. We demonstrate how to incorporate action conditioning into existing pre-trained video generation models. This allows leveraging internet-scale in-the-wild online videos during the pre-training stage, and alleviates the need for a large dataset of paired video-action data, which is expensive to collect for robotic manipulation. Our paper examines the effect of dataset diversity, pre-trained weight and common failure cases for the proposed evaluation pipeline.Our experiments demonstrate that, across various metrics, including policy ranking and the correlation between actual policy values and predicted policy values, these models offer a promising approach for evaluating policies without requiring real-world interactions. |
| 2025-11-14 | [A Comparative Evaluation of Prominent Methods in Autonomous Vehicle Certification](http://arxiv.org/abs/2511.11484v1) | Mustafa Erdem Kırmızıgül, Hasan Feyzi Doğruyol et al. | The "Vision Zero" policy, introduced by the Swedish Parliament in 1997, aims to eliminate fatalities and serious injuries resulting from traffic accidents. To achieve this goal, the use of self-driving vehicles in traffic is envisioned and a roadmap for the certification of self-driving vehicles is aimed to be determined. However, it is still unclear how the basic safety requirements that autonomous vehicles must meet will be verified and certified, and which methods will be used. This paper focuses on the comparative evaluation of the prominent methods planned to be used in the certification process of autonomous vehicles. It examines the prominent methods used in the certification process, develops a pipeline for the certification process of autonomous vehicles, and determines the stages, actors, and areas where the addressed methods can be applied. |
| 2025-11-14 | [SimTac: A Physics-Based Simulator for Vision-Based Tactile Sensing with Biomorphic Structures](http://arxiv.org/abs/2511.11456v1) | Xuyang Zhang, Jiaqi Jiang et al. | Tactile sensing in biological organisms is deeply intertwined with morphological form, such as human fingers, cat paws, and elephant trunks, which enables rich and adaptive interactions through a variety of geometrically complex structures. In contrast, vision-based tactile sensors in robotics have been limited to simple planar geometries, with biomorphic designs remaining underexplored. To address this gap, we present SimTac, a physics-based simulation framework for the design and validation of biomorphic tactile sensors. SimTac consists of particle-based deformation modeling, light-field rendering for photorealistic tactile image generation, and a neural network for predicting mechanical responses, enabling accurate and efficient simulation across a wide range of geometries and materials. We demonstrate the versatility of SimTac by designing and validating physical sensor prototypes inspired by biological tactile structures and further demonstrate its effectiveness across multiple Sim2Real tactile tasks, including object classification, slip detection, and contact safety assessment. Our framework bridges the gap between bio-inspired design and practical realisation, expanding the design space of tactile sensors and paving the way for tactile sensing systems that integrate morphology and sensing to enable robust interaction in unstructured environments. |
| 2025-11-14 | [Heterogeneous CACC Coexistence: Simulation, Analysis, and Modeling](http://arxiv.org/abs/2511.11429v1) | Lorenzo Ghiro, Marco Franceschini et al. | The design of Cooperative Adaptive Cruise Control (CACC) algorithms for vehicle platooning has been extensively investigated, leading to a wide range of approaches with different requirements and performance. Most existing studies evaluate these algorithms under the assumption of homogeneous platoons, i.e., when all platoon members adopt the same CACC. However, market competition is likely to result in vehicles from different manufacturers implementing distinct CACCs. This raises fundamental questions about whether heterogeneous vehicles can safely cooperate within a platoon and what performance can be achieved. To date, these questions have received little attention, as heterogeneous platoons are difficult to model and analyze. In this work, we introduce the concept of mixed platoons, i.e., platoons made of vehicles running heterogeneous CACCs, and we study their performance through simulation-based experiments. We consider mixtures of three well-established CACCs from the literature. In the first part of the paper, we study a single mixed platoon in isolation to understand the microscopic effects on safety: we evaluate the performance of various CACC-mixtures across speed change and emergency braking scenarios. In the second part, we examine a high-density ring-road scenario to assess macroscopic impacts on safety, comfort, and traffic throughput, especially comparing throughput results with those obtained from vehicles controlled by a standard Adaptive Cruise Control (ACC) or by human drivers. Our findings highlight that some combinations of CACCs can operate robustly and safely, while others exhibit critical limitations in safety, comfort, or efficiency. These results emphasize the need for careful system design and the development of theoretical frameworks for modeling heterogeneous platoons. |
| 2025-11-14 | [Multi-Phase Spacecraft Trajectory Optimization via Transformer-Based Reinforcement Learning](http://arxiv.org/abs/2511.11402v1) | Amit Jain, Victor Rodriguez-Fernandez et al. | Autonomous spacecraft control for mission phases such as launch, ascent, stage separation, and orbit insertion remains a critical challenge due to the need for adaptive policies that generalize across dynamically distinct regimes. While reinforcement learning (RL) has shown promise in individual astrodynamics tasks, existing approaches often require separate policies for distinct mission phases, limiting adaptability and increasing operational complexity. This work introduces a transformer-based RL framework that unifies multi-phase trajectory optimization through a single policy architecture, leveraging the transformer's inherent capacity to model extended temporal contexts. Building on proximal policy optimization (PPO), our framework replaces conventional recurrent networks with a transformer encoder-decoder structure, enabling the agent to maintain coherent memory across mission phases spanning seconds to minutes during critical operations. By integrating a Gated Transformer-XL (GTrXL) architecture, the framework eliminates manual phase transitions while maintaining stability in control decisions. We validate our approach progressively: first demonstrating near-optimal performance on single-phase benchmarks (double integrator and Van der Pol oscillator), then extending to multiphase waypoint navigation variants, and finally tackling a complex multiphase rocket ascent problem that includes atmospheric flight, stage separation, and vacuum operations. Results demonstrate that the transformer-based framework not only matches analytical solutions in simple cases but also effectively learns coherent control policies across dynamically distinct regimes, establishing a foundation for scalable autonomous mission planning that reduces reliance on phase-specific controllers while maintaining compatibility with safety-critical verification protocols. |
| 2025-11-14 | [Universal Safety Controllers with Learned Prophecies](http://arxiv.org/abs/2511.11390v1) | Bernd Finkbeiner, Niklas Metzger et al. | \emph{Universal Safety Controllers (USCs)} are a promising logical control framework that guarantees the satisfaction of a given temporal safety specification when applied to any realizable plant model. Unlike traditional methods, which synthesize one logical controller over a given detailed plant model, USC synthesis constructs a \emph{generic controller} whose outputs are conditioned by plant behavior, called \emph{prophecies}. Thereby, USCs offer strong generalization and scalability benefits over classical logical controllers. However, the exact computation and verification of prophecies remain computationally challenging. In this paper, we introduce an approximation algorithm for USC synthesis that addresses these limitations via learning. Instead of computing exact prophecies, which reason about sets of trees via automata, we only compute under- and over-approximations from (small) example plants and infer computation tree logic (CTL) formulas as representations of prophecies. The resulting USC generalizes to unseen plants via a verification step and offers improved efficiency and explainability through small and concise CTL prophecies, which remain human-readable and interpretable. Experimental results demonstrate that our learned prophecies remain generalizable, yet are significantly more compact and interpretable than their exact tree automata representations. |
| 2025-11-14 | [EcoAlign: An Economically Rational Framework for Efficient LVLM Alignment](http://arxiv.org/abs/2511.11301v1) | Ruoxi Cheng, Haoxuan Ma et al. | Large Vision-Language Models (LVLMs) exhibit powerful reasoning capabilities but suffer sophisticated jailbreak vulnerabilities. Fundamentally, aligning LVLMs is not just a safety challenge but a problem of economic efficiency. Current alignment methods struggle with the trade-off between safety, utility, and operational costs. Critically, a focus solely on final outputs (process-blindness) wastes significant computational budget on unsafe deliberation. This flaw allows harmful reasoning to be disguised with benign justifications, thereby circumventing simple additive safety scores. To address this, we propose EcoAlign, an inference-time framework that reframes alignment as an economically rational search by treating the LVLM as a boundedly rational agent. EcoAlign incrementally expands a thought graph and scores actions using a forward-looking function (analogous to net present value) that dynamically weighs expected safety, utility, and cost against the remaining budget. To prevent deception, path safety is enforced via the weakest-link principle. Extensive experiments across 3 closed-source and 2 open-source models on 6 datasets show that EcoAlign matches or surpasses state-of-the-art safety and utility at a lower computational cost, thereby offering a principled, economical pathway to robust LVLM alignment. |
| 2025-11-14 | [UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios](http://arxiv.org/abs/2511.11252v1) | Mohamed Amine Ferrag, Abderrahmane Lakas et al. | Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at https://github.com/maferrag/UAVBench |
| 2025-11-14 | [One-to-N Backdoor Attack in 3D Point Cloud via Spherical Trigger](http://arxiv.org/abs/2511.11210v1) | Dongmei Shan, Wei Lian et al. | Backdoor attacks represent a critical threat to deep learning systems, particularly in safety-sensitive 3D domains such as autonomous driving and robotics. However, existing backdoor attacks for 3D point clouds have been limited to a rigid one-to-one paradigm. To address this, we present the first one-to-N backdoor framework for 3D vision, based on a novel, configurable spherical trigger. Our key insight is to leverage the spatial properties of spheres as a parameter space, allowing a single trigger design to encode multiple target classes. We establish a theoretical foundation for one-to-N backdoor attacks in 3D, demonstrating that poisoned models can map distinct trigger configurations to different target labels. Experimental results systematically validate this conclusion across multiple datasets and model architectures, achieving high attack success rates (up to 100\%) while maintaining accuracy on clean data. This work establishes a crucial benchmark for multi-target threats in 3D vision and provides the foundational understanding needed to secure future 3D-driven intelligent systems. |
| 2025-11-14 | [Advancing IoT System Dependability: A Deep Dive into Management and Operation Plane Separation](http://arxiv.org/abs/2511.11204v1) | Luoyao Hao, Shuo Zhang et al. | We propose to enhance the dependability of large-scale IoT systems by separating the management and operation plane. We innovate the management plane to enforce overarching policies, such as safety norms, operation standards, and energy restrictions, and integrate multi-faceted management entities, including regulatory agencies and manufacturers, while the current IoT operational workflow remains unchanged. Central to the management plane is a meticulously designed, identity-independent policy framework that employs flexible descriptors rather than fixed identifiers, allowing for proactive deployment of overarching policies with adaptability to system changes. Our evaluation across three datasets indicates that the proposed framework can achieve near-optimal expressiveness and dependable policy enforcement. |
| 2025-11-14 | [Geospatial Chain of Thought Reasoning for Enhanced Visual Question Answering on Satellite Imagery](http://arxiv.org/abs/2511.11198v1) | Shambhavi Shanker, Manikandan Padmanaban et al. | Geospatial chain of thought (CoT) reasoning is essential for advancing Visual Question Answering (VQA) on satellite imagery, particularly in climate related applications such as disaster monitoring, infrastructure risk assessment, urban resilience planning, and policy support. Existing VQA models enable scalable interpretation of remote sensing data but often lack the structured reasoning required for complex geospatial queries. We propose a VQA framework that integrates CoT reasoning with Direct Preference Optimization (DPO) to improve interpretability, robustness, and accuracy. By generating intermediate rationales, the model better handles tasks involving detection, classification, spatial relations, and comparative analysis, which are critical for reliable decision support in high stakes climate domains. Experiments show that CoT supervision improves accuracy by 34.9\% over direct baselines, while DPO yields additional gains in accuracy and reasoning quality. The resulting system advances VQA for multispectral Earth observation by enabling richer geospatial reasoning and more effective climate use cases. |
| 2025-11-14 | [Hindsight Distillation Reasoning with Knowledge Encouragement Preference for Knowledge-based Visual Question Answering](http://arxiv.org/abs/2511.11132v1) | Yu Zhao, Ying Zhang et al. | Knowledge-based Visual Question Answering (KBVQA) necessitates external knowledge incorporation beyond cross-modal understanding. Existing KBVQA methods either utilize implicit knowledge in multimodal large language models (MLLMs) via in-context learning or explicit knowledge via retrieval augmented generation. However, their reasoning processes remain implicit, without explicit multi-step trajectories from MLLMs. To address this gap, we provide a Hindsight Distilled Reasoning (HinD) framework with Knowledge Encouragement Preference Optimization (KEPO), designed to elicit and harness internal knowledge reasoning ability in MLLMs. First, to tackle the reasoning supervision problem, we propose to emphasize the hindsight wisdom of MLLM by prompting a frozen 7B-size MLLM to complete the reasoning process between the question and its ground truth answer, constructing Hindsight-Zero training data. Then we self-distill Hindsight-Zero into Chain-of-Thought (CoT) Generator and Knowledge Generator, enabling the generation of sequential steps and discrete facts. Secondly, to tackle the misalignment between knowledge correctness and confidence, we optimize the Knowledge Generator with KEPO, preferring under-confident but helpful knowledge over the over-confident but unhelpful one. The generated CoT and sampled knowledge are then exploited for answer prediction. Experiments on OK-VQA and A-OKVQA validate the effectiveness of HinD, showing that HinD with elicited reasoning from 7B-size MLLM achieves superior performance without commercial model APIs or outside knowledge. |
| 2025-11-14 | [Automata-less Monitoring via Trace-Checking (Extended Version)](http://arxiv.org/abs/2511.11072v1) | Andrea Brunello, Luca Geatti et al. | In runtime verification, monitoring consists of analyzing the current execution of a system and determining, on the basis of the observed finite trace, whether all its possible continuations satisfy or violate a given specification. This is typically done by synthesizing a monitor--often a Deterministic Finite State Automaton (DFA)--from logical specifications expressed in Linear Temporal Logic (LTL) or in its finite-word variant (LTLf). Unfortunately, the size of the resulting DFA may incur a doubly exponential blow-up in the size of the formula. In this paper, we identify some conditions under which monitoring can be done without constructing such a DFA. We build on the notion of intentionally safe and cosafe formulas, introduced in [Kupferman & Vardi, FMSD, 2001], to show that monitoring of these formulas can be carried out through trace-checking, that is, by directly evaluating them on the current system trace, with a polynomial complexity in the size of both the trace and the formula. In addition, we investigate the complexity of recognizing intentionally safe and cosafe formulas for the safety and cosafety fragments of LTL and LTLf. As for LTLf, we show that all formulas in these fragments are intentionally safe and cosafe, thus removing the need for the check. As for LTL, we prove that the problem is in PSPACE, significantly improving over the EXPSPACE complexity of full LTL. |
| 2025-11-14 | [Key Decision-Makers in Multi-Agent Debates: Who Holds the Power?](http://arxiv.org/abs/2511.11040v1) | Qian Zhang, Yan Zheng et al. | Recent studies on LLM agent scaling have highlighted the potential of Multi-Agent Debate (MAD) to enhance reasoning abilities. However, the critical aspect of role allocation strategies remains underexplored. In this study, we demonstrate that allocating roles with differing viewpoints to specific positions significantly impacts MAD's performance in reasoning tasks. Specifically, we find a novel role allocation strategy, "Truth Last", which can improve MAD performance by up to 22% in reasoning tasks. To address the issue of unknown truth in practical applications, we propose the Multi-Agent Debate Consistency (MADC) strategy, which systematically simulates and optimizes its core mechanisms. MADC incorporates path consistency to assess agreement among independent roles, simulating the role with the highest consistency score as the truth. We validated MADC across a range of LLMs (9 models), including the DeepSeek-R1 Distilled Models, on challenging reasoning tasks. MADC consistently demonstrated advanced performance, effectively overcoming MAD's performance bottlenecks and providing a crucial pathway for further improvements in LLM agent scaling. |
| 2025-11-14 | [SALT-V: Lightweight Authentication for 5G V2X Broadcasting](http://arxiv.org/abs/2511.11028v1) | Liu Cao, Weizheng Wang et al. | Vehicle-to-Everything (V2X) communication faces a critical authentication dilemma: traditional public-key schemes like ECDSA provide strong security but impose 2 ms verification delays unsuitable for collision avoidance, while symmetric approaches like TESLA achieve microsecond-level efficiency at the cost of 20-100 ms key disclosure latency. Neither meets 5G New Radio (NR)-V2X's stringent requirements for both immediate authentication and computational efficiency. This paper presents SALT-V, a novel hybrid authentication framework that reconciles this fundamental trade-off through intelligent protocol stratification. SALT-V employs ECDSA signatures for 10% of traffic (BOOT frames) to establish sender trust, then leverages this trust anchor to authenticate 90% of messages (DATA frames) using lightweight GMAC operations. The core innovation - an Ephemeral Session Tag (EST) whitelist mechanism - enables 95% of messages to achieve immediate verification without waiting for key disclosure, while Bloom filter integration provides O(1) revocation checking in 1 us. Comprehensive evaluation demonstrates that SALT-V achieves 0.035 ms average computation time (57x faster than pure ECDSA), 1 ms end-to-end latency, 41-byte overhead, and linear scalability to 2000 vehicles, making it the first practical solution to satisfy all safety-critical requirements for real-time V2X deployment. |
| 2025-11-14 | [Region of Attraction Estimate Learning and Verification for Nonlinear Systems using Neural-Network-based Lyapunov Functions](http://arxiv.org/abs/2511.11026v1) | Adel Bechihi, Aristotelis Kapnopoulos | Estimating the Region of Attraction (RoA) for nonlinear dynamical systems is a fundamental problem in control theory, with direct implications for stability analysis and safe controller design. Traditional approaches rely on analytically derived Lyapunov functions, which are often conservative and challenging to construct for high-dimensional or highly nonlinear systems. In this work, we propose a data-driven framework for learning and verifying RoA estimates for nonlinear systems using neural-network-based Lyapunov functions. Our method employs a composite Lyapunov function that combines a quadratic term with a neural-network-based component, providing both structure and flexibility. We introduce a novel homogeneous loss function for training, which removes the imbalance typically caused by the two non-homogeneous Lyapunov conditions. Together, these two aspects enable efficient training of the Lyapunov candidate. To guarantee the correctness of the learned Lyapunov function, we employ a Satisfiability Modulo Theories (SMT) solver to formally verify the stability results. Lastly, we perform a deeper analysis near the origin to overcome numerical artifacts, ensuring strict asymptotic stability. We demonstrate the effectiveness of our approach on benchmark nonlinear systems, showing that it significantly reduces conservatism compared to traditional Lyapunov methods while maintaining verifiability. This framework bridges the gap between function approximation and stability certification, paving the way for scalable safety analysis in learning-based control and safety-critical applications. |
| 2025-11-14 | [AI and Worker Well-Being: Differential Impacts Across Generational Cohorts and Genders](http://arxiv.org/abs/2511.11021v1) | Voraprapa Nakavachara | This paper investigates the relationship between AI use and worker well-being outcomes such as mental health, job enjoyment, and physical health and safety, using microdata from the OECD AI Surveys across seven countries. The results reveal that AI users are significantly more likely to report improvements across all three outcomes, with effects ranging from 8.9% to 21.3%. However, these benefits vary by generation and gender. Generation Y (1981-1996) shows the strongest gains across all dimensions, while Generation X (1965-1980) reports moderate improvements in mental health and job enjoyment. In contrast, Generation Z (1997-2012) benefits only in job enjoyment. As digital natives already familiar with technology, Gen Z workers may not receive additional gains in mental or physical health from AI, though they still experience increased enjoyment from using it. Baby Boomers (born before 1965) experience limited benefits, as they may not find these tools as engaging or useful. Women report stronger mental health gains, whereas men report greater improvements in physical health. These findings suggest that AI's workplace impact is uneven and shaped by demographic factors, career stage, and the nature of workers' roles. |
| 2025-11-14 | [Data Poisoning Vulnerabilities Across Healthcare AI Architectures: A Security Threat Analysis](http://arxiv.org/abs/2511.11020v1) | Farhad Abtahi, Fernando Seoane et al. | Healthcare AI systems face major vulnerabilities to data poisoning that current defenses and regulations cannot adequately address. We analyzed eight attack scenarios in four categories: architectural attacks on convolutional neural networks, large language models, and reinforcement learning agents; infrastructure attacks exploiting federated learning and medical documentation systems; critical resource allocation attacks affecting organ transplantation and crisis triage; and supply chain attacks targeting commercial foundation models. Our findings indicate that attackers with access to only 100-500 samples can compromise healthcare AI regardless of dataset size, often achieving over 60 percent success, with detection taking an estimated 6 to 12 months or sometimes not occurring at all. The distributed nature of healthcare infrastructure creates many entry points where insiders with routine access can launch attacks with limited technical skill. Privacy laws such as HIPAA and GDPR can unintentionally shield attackers by restricting the analyses needed for detection. Supply chain weaknesses allow a single compromised vendor to poison models across 50 to 200 institutions. The Medical Scribe Sybil scenario shows how coordinated fake patient visits can poison data through legitimate clinical workflows without requiring a system breach. Current regulations lack mandatory adversarial robustness testing, and federated learning can worsen risks by obscuring attribution. We recommend multilayer defenses including required adversarial testing, ensemble-based detection, privacy-preserving security mechanisms, and international coordination on AI security standards. We also question whether opaque black-box models are suitable for high-stakes clinical decisions, suggesting a shift toward interpretable systems with verifiable safety guarantees. |

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



