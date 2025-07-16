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
| 2025-07-15 | [LLM-based ambiguity detection in natural language instructions for collaborative surgical robots](http://arxiv.org/abs/2507.11525v1) | Ana Davila, Jacinto Colan et al. | Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action. |
| 2025-07-15 | [Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety](http://arxiv.org/abs/2507.11473v1) | Tomek Korbak, Mikita Balesni et al. | AI systems that "think" in human language offer a unique opportunity for AI safety: we can monitor their chains of thought (CoT) for the intent to misbehave. Like all other known AI oversight methods, CoT monitoring is imperfect and allows some misbehavior to go unnoticed. Nevertheless, it shows promise and we recommend further research into CoT monitorability and investment in CoT monitoring alongside existing safety methods. Because CoT monitorability may be fragile, we recommend that frontier model developers consider the impact of development decisions on CoT monitorability. |
| 2025-07-15 | [Human-Robot collaboration in surgery: Advances and challenges towards autonomous surgical assistants](http://arxiv.org/abs/2507.11460v1) | Jacinto Colan, Ana Davila et al. | Human-robot collaboration in surgery represents a significant area of research, driven by the increasing capability of autonomous robotic systems to assist surgeons in complex procedures. This systematic review examines the advancements and persistent challenges in the development of autonomous surgical robotic assistants (ASARs), focusing specifically on scenarios where robots provide meaningful and active support to human surgeons. Adhering to the PRISMA guidelines, a comprehensive literature search was conducted across the IEEE Xplore, Scopus, and Web of Science databases, resulting in the selection of 32 studies for detailed analysis. Two primary collaborative setups were identified: teleoperation-based assistance and direct hands-on interaction. The findings reveal a growing research emphasis on ASARs, with predominant applications currently in endoscope guidance, alongside emerging progress in autonomous tool manipulation. Several key challenges hinder wider adoption, including the alignment of robotic actions with human surgeon preferences, the necessity for procedural awareness within autonomous systems, the establishment of seamless human-robot information exchange, and the complexities of skill acquisition in shared workspaces. This review synthesizes current trends, identifies critical limitations, and outlines future research directions essential to improve the reliability, safety, and effectiveness of human-robot collaboration in surgical environments. |
| 2025-07-15 | [A Risk-Aware Adaptive Robust MPC with Learned Uncertainty Quantification](http://arxiv.org/abs/2507.11420v1) | Mingcong Li | Solving chance-constrained optimal control problems for systems subject to non-stationary uncertainties is a significant challenge.Conventional robust model predictive control (MPC) often yields excessive conservatism by relying on static worst-case assumptions, while standard stochastic MPC methods struggle when underlying uncertainty distributions are unknown a priori.This article presents a Risk-Aware Adaptive Robust MPC (RAAR-MPC) framework,a hierarchical architecture that systematically orchestrates a novel synthesis of proactive, learning-based risk assessment and reactive risk regulation. The framework employs a medium-frequency risk assessment engine, which leverages Gaussian process regression and active learning, to construct a tight, data-driven characterization of the prediction error set from operational data.Concurrently, a low-timescale outer loop implements a self-correcting update law for an adaptive safety margin to precisely regulate the empirical risk and compensate for unmodeled dynamics.This dual-timescale adaptation enables the system to rigorously satisfy chance constraints with a user-defined probability, while minimizing the conservatism inherent in traditional approaches.We formally establish that the interplay between these adaptive components guarantees recursive feasibility and ensures the closed-loop system satisfies the chance constraints up to a user-defined risk level with high probability.Numerical experiments on a benchmark DC-DC converter under non-stationary parametric uncertainties demonstrate that our framework precisely achieves the target risk level, resulting in a significantly lower average cost compared to state-of-the-art robust and stochastic MPC strategies. |
| 2025-07-15 | [Inverse Optimal Control with Constraint Relaxation](http://arxiv.org/abs/2507.11392v1) | Rahel Rickenbach, Amon Lahr et al. | Inverse optimal control (IOC) is a promising paradigm for learning and mimicking optimal control strategies from capable demonstrators, or gaining a deeper understanding of their intentions, by estimating an unknown objective function from one or more corresponding optimal control sequences. When computing estimates from demonstrations in environments with safety-preserving inequality constraints, acknowledging their presence in the chosen IOC method is crucial given their strong influence on the final control strategy. However, solution strategies capable of considering inequality constraints, such as the inverse Karush-Kuhn-Tucker approach, rely on their correct activation and fulfillment; a restrictive assumption when dealing with noisy demonstrations. To overcome this problem, we leverage the concept of exact penalty functions for IOC and show preservation of estimation accuracy. Considering noisy demonstrations, we then illustrate how the usage of penalty functions reduces the number of unknown variables and how their approximations enhance the estimation method's capacity to account for wrong constraint activations within a polytopic-constrained environment. The proposed method is evaluated for three systems in simulation, outperforming traditional relaxation approaches for noisy demonstrations. |
| 2025-07-15 | [From Observational Data to Clinical Recommendations: A Causal Framework for Estimating Patient-level Treatment Effects and Learning Policies](http://arxiv.org/abs/2507.11381v1) | Rom Gutman, Shimon Sheiba et al. | We propose a framework for building patient-specific treatment recommendation models, building on the large recent literature on learning patient-level causal models and inspired by the target trial paradigm of Hernan and Robins. We focus on safety and validity, including the crucial issue of causal identification when using observational data. We do not provide a specific model, but rather a way to integrate existing methods and know-how into a practical pipeline. We further provide a real world use-case of treatment optimization for patients with heart failure who develop acute kidney injury during hospitalization. The results suggest our pipeline can improve patient outcomes over the current treatment regime. |
| 2025-07-15 | [Foundation Models for Logistics: Toward Certifiable, Conversational Planning Interfaces](http://arxiv.org/abs/2507.11352v1) | Yunhao Yang, Neel P. Bhatt et al. | Logistics operators, from battlefield coordinators rerouting airlifts ahead of a storm to warehouse managers juggling late trucks, often face life-critical decisions that demand both domain expertise and rapid and continuous replanning. While popular methods like integer programming yield logistics plans that satisfy user-defined logical constraints, they are slow and assume an idealized mathematical model of the environment that does not account for uncertainty. On the other hand, large language models (LLMs) can handle uncertainty and promise to accelerate replanning while lowering the barrier to entry by translating free-form utterances into executable plans, yet they remain prone to misinterpretations and hallucinations that jeopardize safety and cost. We introduce a neurosymbolic framework that pairs the accessibility of natural-language dialogue with verifiable guarantees on goal interpretation. It converts user requests into structured planning specifications, quantifies its own uncertainty at the field and token level, and invokes an interactive clarification loop whenever confidence falls below an adaptive threshold. A lightweight model, fine-tuned on just 100 uncertainty-filtered examples, surpasses the zero-shot performance of GPT-4.1 while cutting inference latency by nearly 50%. These preliminary results highlight a practical path toward certifiable, real-time, and user-aligned decision-making for complex logistics. |
| 2025-07-15 | [The downgrading semantics of memory safety](http://arxiv.org/abs/2507.11282v1) | René Rydhof Hansen, Andreas Stenbæk Larsen et al. | Memory safety is traditionally characterized in terms of bad things that cannot happen, an approach that is often criticized as unprincipled. Prior work suggest a connection between memory safety and noninterference, but no satisfactory semantic notion of memory safety is currently known.   This work proposes a notion of gradual allocator independence that accurately captures many allocator-specific aspects of memory safety. We consider a low-level language with access to an allocator that provides malloc and free primitives in a flat memory model. Pointers are just integers, and as such it is trivial to write memory-unsafe programs. The basic intuition of gradual allocator independence is that of noninterference, namely that allocators must not influence program execution. This intuition is refined in two important ways to account for the allocators running out-of-memory and for programs to have pointer-to-integer casts. The key insight of the definition is to treat these extensions as forms of downgrading and give them satisfactory technical treatment using the state-of-the-art information flow machinery. |
| 2025-07-15 | [Development of an Autonomous Mobile Robotic System for Efficient and Precise Disinfection](http://arxiv.org/abs/2507.11270v1) | Ting-Wei Ou, Jia-Hao Jiang et al. | The COVID-19 pandemic has severely affected public health, healthcare systems, and daily life, especially amid resource shortages and limited workers. This crisis has underscored the urgent need for automation in hospital environments, particularly disinfection, which is crucial to controlling virus transmission and improving the safety of healthcare personnel and patients. Ultraviolet (UV) light disinfection, known for its high efficiency, has been widely adopted in hospital settings. However, most existing research focuses on maximizing UV coverage while paying little attention to the impact of human activity on virus distribution. To address this issue, we propose a mobile robotic system for UV disinfection focusing on the virus hotspot. The system prioritizes disinfection in high-risk areas and employs an approach for optimized UV dosage to ensure that all surfaces receive an adequate level of UV exposure while significantly reducing disinfection time. It not only improves disinfection efficiency but also minimizes unnecessary exposure in low-risk areas. In two representative hospital scenarios, our method achieves the same disinfection effectiveness while reducing disinfection time by 30.7% and 31.9%, respectively. The video of the experiment is available at: https://youtu.be/wHcWzOcoMPM. |
| 2025-07-15 | [AI Agent Architecture for Decentralized Trading of Alternative Assets](http://arxiv.org/abs/2507.11117v1) | Ailiya Borjigin, Cong He et al. | Decentralized trading of real-world alternative assets (e.g., gold) requires bridging physical asset custody with blockchain systems while meeting strict requirements for compliance, liquidity, and risk management. We present GoldMine OS, a research oriented architecture that employs multiple specialized AI agents to automate and secure the tokenization and exchange of physical gold into a blockchain based stablecoin ("OZ"). Our approach combines on chain smart contracts for critical risk controls with off chain AI agents for decision making, blending the transparency and reliability of blockchains with the flexibility of AI driven automation. We describe four cooperative agents (Compliance, Token Issuance, Market Making, and Risk Control) and a coordinating core, and evaluate the system through simulation and a controlled pilot deployment. In experiments the prototype delivers on demand token issuance in under 1.2 s, more than 100 times faster than manual workflows. The Market Making agent maintains tight liquidity with spreads often below 0.5 percent even under volatile conditions. Fault injection tests show resilience: an oracle price spoofing attack is detected and mitigated within 10 s, and a simulated vault mis reporting halts issuance immediately with minimal user impact. The architecture scales to 5000 transactions per second with 10000 concurrent users in benchmarks. These results indicate that an AI agent based decentralized exchange for alternative assets can satisfy rigorous performance and safety requirements. We discuss broader implications for democratizing access to traditionally illiquid assets and explain how our governance model -- multi signature agent updates and on chain community voting on risk parameters -- provides ongoing transparency, adaptability, and formal assurance of system integrity. |
| 2025-07-15 | [The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs](http://arxiv.org/abs/2507.11097v1) | Zichen Wen, Jiashu Qu et al. | Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA. |
| 2025-07-15 | [Optimizing Fluid Antenna Configurations for Constructive Interference Precoding](http://arxiv.org/abs/2507.11093v1) | Wenxuan Sun, Mingjie Shao et al. | The fluid antenna system (FAS) has emerged as a new physical-layer concept to provide enhanced propagation conditions for multiuser multiple-input multiple-output (MIMO) communications over conventional fixed arrays. This work focuses on minimizing the maximum symbol error probability (SEP) under $M$-ary phase shift keying (MPSK) signaling in a multiuser downlink equipped with FAS, where each antenna moves within nonoverlapping intervals. This specific problem of joint SEP minimization with FAS and constructive interference (CI) precoding has not been previously addressed. The resulting problem turns out to be a nonconvex and nonsmooth optimization challenge. We transform the SEP minimization problem into a safety margin maximization problem in constructive interference precoding. Then, we customize a smoothing technique and a block coordinate descent (BCD) algorithm, with emphasis on low computational complexity. Simulation results show that our approach can reduce bit error rate (BER) compared to both the fixed arrays and FAS designed by existing particle swarm optimization (PSO). Also, our approach shows attractively low computational complexity compared to PSO benchmarks. |
| 2025-07-15 | [Real-Time Foreign Object Recognition Based on Improved Wavelet Scattering Deep Network and Edge Computing](http://arxiv.org/abs/2507.11043v1) | He Zhichao, Shen Xiangyu et al. | The increasing penetration rate of new energy in the power system has put forward higher requirements for the operation and maintenance of substations and transmission lines. Using the Unmanned Aerial Vehicles (UAV) to identify foreign object in real time can quickly and effectively eliminate potential safety hazards. However, due to the limited computation power, the captured image cannot be real-time processed on edge devices in UAV locally. To overcome this problem, a lightweight model based on an improved wavelet scatter deep network is proposed. This model contains improved wavelet scattering network for extracting the scatter coefficients and modulus coefficients of image single channel, replacing the role of convolutional layer and pooling layer in convolutional neural network. The following 3 fully connected layers, also constituted a simplified Multilayer Perceptron (MLP), are used to classify the extracted features. Experiments prove that the model constructed with biorthogonal wavelets basis is able to recognize and classify the foreign object in edge devices such as Raspberry Pi and Jetson Nano, with accuracy higher than 90% and inference time less than 7ms for 720P (1280*720) images. Further experiments demonstrate that the recognition accuracy of our model is 1.1% higher than YOLOv5s and 0.3% higher than YOLOv8s. |
| 2025-07-15 | [Approximate solutions to games of ordered preference](http://arxiv.org/abs/2507.11021v1) | Pau de las Heras Molins, Eric Roy-Almonacid et al. | Autonomous vehicles must balance ranked objectives, such as minimizing travel time, ensuring safety, and coordinating with traffic. Games of ordered preference effectively model these interactions but become computationally intractable as the time horizon, number of players, or number of preference levels increase. While receding horizon frameworks mitigate long-horizon intractability by solving sequential shorter games, often warm-started, they do not resolve the complexity growth inherent in existing methods for solving games of ordered preference. This paper introduces a solution strategy that avoids excessive complexity growth by approximating solutions using lexicographic iterated best response (IBR) in receding horizon, termed "lexicographic IBR over time." Lexicographic IBR over time uses past information to accelerate convergence. We demonstrate through simulated traffic scenarios that lexicographic IBR over time efficiently computes approximate-optimal solutions for receding horizon games of ordered preference, converging towards generalized Nash equilibria. |
| 2025-07-15 | [Indium Hydroxide Ceramic Targets: A Breakthrough in High-Mobility Thin-Film Transistor Technology](http://arxiv.org/abs/2507.11011v1) | Hikaru Sadahira, Prashant R. Ghediya et al. | Thin-film transistors composed of a hydrogen-containing indium oxide active layer are promising candidates for backplane devices in next-generation flat panel displays, offering higher definition and faster operation. However, the hydrogen incorporation process during film deposition poses challenges for scalable and industrial development due to both safety and controllability issues. Here, we demonstrate that using indium hydroxide ceramic as the target material for film deposition overcomes the difficulties associated with hydrogen gas usage. We sintered commercially available indium hydroxide powder using a conventional ceramic process at 150-250{\deg}C in air and utilized it for the deposition of hydrogen-incorporated indium oxide films via pulsed laser deposition. The resulting indium oxide films, after thermal annealing, contained a sufficient concentration of hydrogen and exhibited very high electron mobility due to significantly grown grains. Furthermore, we confirmed that the fabricated thin-film transistors exhibited comparably high performance to those produced using the gas-phase hydrogen incorporation method. This approach offers a practical pathway for hydrogen-containing indium oxide-based thin-film transistors in next-generation flat panel displays. |
| 2025-07-15 | [Learning to Tune Like an Expert: Interpretable and Scene-Aware Navigation via MLLM Reasoning and CVAE-Based Adaptation](http://arxiv.org/abs/2507.11001v1) | Yanbo Wang, Zipeng Fang et al. | Service robots are increasingly deployed in diverse and dynamic environments, where both physical layouts and social contexts change over time and across locations. In these unstructured settings, conventional navigation systems that rely on fixed parameters often fail to generalize across scenarios, resulting in degraded performance and reduced social acceptance. Although recent approaches have leveraged reinforcement learning to enhance traditional planners, these methods often fail in real-world deployments due to poor generalization and limited simulation diversity, which hampers effective sim-to-real transfer. To tackle these issues, we present LE-Nav, an interpretable and scene-aware navigation framework that leverages multi-modal large language model reasoning and conditional variational autoencoders to adaptively tune planner hyperparameters. To achieve zero-shot scene understanding, we utilize one-shot exemplars and chain-of-thought prompting strategies. Additionally, a conditional variational autoencoder captures the mapping between natural language instructions and navigation hyperparameters, enabling expert-level tuning. Experiments show that LE-Nav can generate hyperparameters achieving human-level tuning across diverse planners and scenarios. Real-world navigation trials and a user study on a smart wheelchair platform demonstrate that it outperforms state-of-the-art methods on quantitative metrics such as success rate, efficiency, safety, and comfort, while receiving higher subjective scores for perceived safety and social acceptance. Code is available at https://github.com/Cavendish518/LE-Nav. |
| 2025-07-15 | [Data-Driven Safety Certificates of Infinite Networks with Unknown Models and Interconnection Topologies](http://arxiv.org/abs/2507.10979v1) | Mahdieh Zaker, Amy Nejati et al. | Infinite networks are complex interconnected systems comprising a countably infinite number of subsystems, where counting them precisely poses a significant challenge due to the seemingly endless interconnected nature of the network (e.g., counting vehicles on the road). In such scenarios, the presence of infinitely many subsystems within the network renders the existing analysis frameworks tailored for finite networks inapplicable to infinite ones. This paper is concerned with offering a data-driven approach, within a compositional framework, for the safety certification of infinite networks with both unknown mathematical models and interconnection topologies. Given the immense computational complexity stemming from the extensive dimension of infinite networks, our approach capitalizes on the joint dissipativity-type properties of subsystems, characterized by storage certificates. We introduce innovative compositional data-driven conditions to construct a barrier certificate for the infinite network leveraging storage certificates of its unknown subsystems derived from data, while offering correctness guarantees across the network safety. We demonstrate that our compositional data-driven reasoning eliminates the requirement for checking the traditional dissipativity condition, which typically mandates precise knowledge of the interconnection topology. In addition, while existing data-driven literature demonstrates an exponential trend in sample complexity with respect to network size, we showcase that our compositional strategy notably reduces it to a linear scale in terms of the number of subsystems. We illustrate our data-driven results on two physical infinite networks with unknown models and interconnection topologies. |
| 2025-07-15 | [SMART-Merge Planner: A Safe Merging and Real-Time Motion Planner for Autonomous Highway On-Ramp Merging](http://arxiv.org/abs/2507.10968v1) | Toktam Mohammadnejad, Jovin D'sa et al. | Merging onto a highway is a complex driving task that requires identifying a safe gap, adjusting speed, often interactions to create a merging gap, and completing the merge maneuver within a limited time window while maintaining safety and driving comfort. In this paper, we introduce a Safe Merging and Real-Time Merge (SMART-Merge) planner, a lattice-based motion planner designed to facilitate safe and comfortable forced merging. By deliberately adapting cost terms to the unique challenges of forced merging and introducing a desired speed heuristic, SMART-Merge planner enables the ego vehicle to merge successfully while minimizing the merge time. We verify the efficiency and effectiveness of the proposed merge planner through high-fidelity CarMaker simulations on hundreds of highway merge scenarios. Our proposed planner achieves the success rate of 100% as well as completes the merge maneuver in the shortest amount of time compared with the baselines, demonstrating our planner's capability to handle complex forced merge tasks and provide a reliable and robust solution for autonomous highway merge. The simulation result videos are available at https://sites.google.com/view/smart-merge-planner/home. |
| 2025-07-15 | [Artificial Finance: How AI Thinks About Money](http://arxiv.org/abs/2507.10933v1) | Orhan Erdem, Ragavi Pobbathi Ashok | In this paper, we explore how large language models (LLMs) approach financial decision-making by systematically comparing their responses to those of human participants across the globe. We posed a set of commonly used financial decision-making questions to seven leading LLMs, including five models from the GPT series(GPT-4o, GPT-4.5, o1, o3-mini), Gemini 2.0 Flash, and DeepSeek R1. We then compared their outputs to human responses drawn from a dataset covering 53 nations. Our analysis reveals three main results. First, LLMs generally exhibit a risk-neutral decision-making pattern, favoring choices aligned with expected value calculations when faced with lottery-type questions. Second, when evaluating trade-offs between present and future, LLMs occasionally produce responses that appear inconsistent with normative reasoning. Third, when we examine cross-national similarities, we find that the LLMs' aggregate responses most closely resemble those of participants from Tanzania. These findings contribute to the understanding of how LLMs emulate human-like decision behaviors and highlight potential cultural and training influences embedded within their outputs. |
| 2025-07-15 | [Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](http://arxiv.org/abs/2507.10923v1) | Yuhao Wang, Keyan Ding et al. | Protein language models have emerged as powerful tools for sequence generation, offering substantial advantages in functional optimization and denovo design. However, these models also present significant risks of generating harmful protein sequences, such as those that enhance viral transmissibility or evade immune responses. These concerns underscore critical biosafety and ethical challenges. To address these issues, we propose a Knowledge-guided Preference Optimization (KPO) framework that integrates prior knowledge via a Protein Safety Knowledge Graph. This framework utilizes an efficient graph pruning strategy to identify preferred sequences and employs reinforcement learning to minimize the risk of generating harmful proteins. Experimental results demonstrate that KPO effectively reduces the likelihood of producing hazardous sequences while maintaining high functionality, offering a robust safety assurance framework for applying generative models in biotechnology. |

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



