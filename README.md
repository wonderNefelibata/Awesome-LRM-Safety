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
| 2025-07-30 | [DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](http://arxiv.org/abs/2507.22813v1) | Hossein Mirzaei, Zeinab Taghavi et al. | Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL. |
| 2025-07-30 | [Bayesian Optimization applied for accelerated Virtual Validation of the Autonomous Driving Function](http://arxiv.org/abs/2507.22769v1) | Satyesh Shanker Awasthi, Mohammed Irshadh Ismaaeel Sathyamangalam Imran et al. | Rigorous Verification and Validation (V&V) of Autonomous Driving Functions (ADFs) is paramount for ensuring the safety and public acceptance of Autonomous Vehicles (AVs). Current validation relies heavily on simulation to achieve sufficient test coverage within the Operational Design Domain (ODD) of a vehicle, but exhaustively exploring the vast parameter space of possible scenarios is computationally expensive and time-consuming. This work introduces a framework based on Bayesian Optimization (BO) to accelerate the discovery of critical scenarios. We demonstrate the effectiveness of the framework on an Model Predictive Controller (MPC)-based motion planner, showing that it identifies hazardous situations, such as off-road events, using orders of magnitude fewer simulations than brute-force Design of Experiments (DoE) methods. Furthermore, this study investigates the scalability of the framework in higher-dimensional parameter spaces and its ability to identify multiple, distinct critical regions within the ODD of the motion planner used as the case study . |
| 2025-07-30 | [Of Good Demons and Bad Angels: Guaranteeing Safe Control under Finite Precision](http://arxiv.org/abs/2507.22760v1) | Samuel Teuber, Debasmita Lohar et al. | As neural networks (NNs) become increasingly prevalent in safety-critical neural network-controlled cyber-physical systems (NNCSs), formally guaranteeing their safety becomes crucial. For these systems, safety must be ensured throughout their entire operation, necessitating infinite-time horizon verification. To verify the infinite-time horizon safety of NNCSs, recent approaches leverage Differential Dynamic Logic (dL). However, these dL-based guarantees rely on idealized, real-valued NN semantics and fail to account for roundoff errors introduced by finite-precision implementations. This paper bridges the gap between theoretical guarantees and real-world implementations by incorporating robustness under finite-precision perturbations -- in sensing, actuation, and computation -- into the safety verification. We model the problem as a hybrid game between a good Demon, responsible for control actions, and a bad Angel, introducing perturbations. This formulation enables formal proofs of robustness w.r.t. a given (bounded) perturbation. Leveraging this bound, we employ state-of-the-art mixed-precision fixed-point tuners to synthesize sound and efficient implementations, thus providing a complete end-to-end solution. We evaluate our approach on case studies from the automotive and aeronautics domains, producing efficient NN implementations with rigorous infinite-time horizon safety guarantees. |
| 2025-07-30 | [Social-Pose: Enhancing Trajectory Prediction with Human Body Pose](http://arxiv.org/abs/2507.22742v1) | Yang Gao, Saeed Saadatnejad et al. | Accurate human trajectory prediction is one of the most crucial tasks for autonomous driving, ensuring its safety. Yet, existing models often fail to fully leverage the visual cues that humans subconsciously communicate when navigating the space. In this work, we study the benefits of predicting human trajectories using human body poses instead of solely their Cartesian space locations in time. We propose `Social-pose', an attention-based pose encoder that effectively captures the poses of all humans in a scene and their social relations. Our method can be integrated into various trajectory prediction architectures. We have conducted extensive experiments on state-of-the-art models (based on LSTM, GAN, MLP, and Transformer), and showed improvements over all of them on synthetic (Joint Track Auto) and real (Human3.6M, Pedestrians and Cyclists in Road Traffic, and JRDB) datasets. We also explored the advantages of using 2D versus 3D poses, as well as the effect of noisy poses and the application of our pose-based predictor in robot navigation scenarios. |
| 2025-07-30 | [Investigating Hallucination in Conversations for Low Resource Languages](http://arxiv.org/abs/2507.22720v1) | Amit Das, Md. Najib Hasan et al. | Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our study extends this investigation to conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a comprehensive analysis of a dataset to examine both factual and linguistic errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0, DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated responses in Mandarin but generate a significantly higher number of hallucinations in Hindi and Farsi. |
| 2025-07-30 | [Safe Deployment of Offline Reinforcement Learning via Input Convex Action Correction](http://arxiv.org/abs/2507.22640v1) | Alex Durkin, Jasper Stolte et al. | Offline reinforcement learning (offline RL) offers a promising framework for developing control strategies in chemical process systems using historical data, without the risks or costs of online experimentation. This work investigates the application of offline RL to the safe and efficient control of an exothermic polymerisation continuous stirred-tank reactor. We introduce a Gymnasium-compatible simulation environment that captures the reactor's nonlinear dynamics, including reaction kinetics, energy balances, and operational constraints. The environment supports three industrially relevant scenarios: startup, grade change down, and grade change up. It also includes reproducible offline datasets generated from proportional-integral controllers with randomised tunings, providing a benchmark for evaluating offline RL algorithms in realistic process control tasks.   We assess behaviour cloning and implicit Q-learning as baseline algorithms, highlighting the challenges offline agents face, including steady-state offsets and degraded performance near setpoints. To address these issues, we propose a novel deployment-time safety layer that performs gradient-based action correction using input convex neural networks (PICNNs) as learned cost models. The PICNN enables real-time, differentiable correction of policy actions by descending a convex, state-conditioned cost surface, without requiring retraining or environment interaction.   Experimental results show that offline RL, particularly when combined with convex action correction, can outperform traditional control approaches and maintain stability across all scenarios. These findings demonstrate the feasibility of integrating offline RL with interpretable and safety-aware corrections for high-stakes chemical process control, and lay the groundwork for more reliable data-driven automation in industrial systems. |
| 2025-07-30 | [Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs](http://arxiv.org/abs/2507.22564v1) | Xikang Yang, Biyu Zhou et al. | Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems. |
| 2025-07-30 | [Accident-Driven Congestion Prediction and Simulation: An Explainable Framework Using Advanced Clustering and Bayesian Networks](http://arxiv.org/abs/2507.22529v1) | Kranthi Kumar Talluri, Galia Weidl et al. | Traffic congestion due to uncertainties, such as accidents, is a significant issue in urban areas, as the ripple effect of accidents causes longer delays, increased emissions, and safety concerns. To address this issue, we propose a robust framework for predicting the impact of accidents on congestion. We implement Automated Machine Learning (AutoML)-enhanced Deep Embedding Clustering (DEC) to assign congestion labels to accident data and predict congestion probability using a Bayesian Network (BN). The Simulation of Urban Mobility (SUMO) simulation is utilized to evaluate the correctness of BN predictions using evidence-based scenarios. Results demonstrate that the AutoML-enhanced DEC has outperformed traditional clustering approaches. The performance of the proposed BN model achieved an overall accuracy of 95.6%, indicating its ability to understand the complex relationship of accidents causing congestion. Validation in SUMO with evidence-based scenarios demonstrated that the BN model's prediction of congestion states closely matches those of SUMO, indicating the high reliability of the proposed BN model in ensuring smooth urban mobility. |
| 2025-07-30 | [Green Wave as an Integral Part for the Optimization of Traffic Efficiency and Safety: A Survey](http://arxiv.org/abs/2507.22511v1) | Kranthi Kumar Talluri, Christopher Stang et al. | Green Wave provides practical and advanced solutions to improve traffic efficiency and safety through network coordination. Nevertheless, the complete potential of Green Wave systems has yet to be explored. Utilizing emerging technologies and advanced algorithms, such as AI or V2X, would aid in achieving more robust traffic management strategies, especially when integrated with Green Wave. This work comprehensively surveys existing traffic control strategies that enable Green Waves and analyzes their impact on future traffic management systems and urban infrastructure. Understanding previous research on traffic management and its effect on traffic efficiency and safety helps explore the integration of Green Wave solutions with smart city initiatives for effective traffic signal coordination. This paper also discusses the advantages of using Green Wave strategies for emission reduction and considers road safety issues for vulnerable road users, such as pedestrians and cyclists. Finally, the existing challenges and research gaps in building robust and successful Green Wave systems are discussed to articulate explicitly the future requirement of sustainable urban transport. |
| 2025-07-30 | [Characterizing Material Effects On Direct ToF Signal Response In Optical Tactile Systems](http://arxiv.org/abs/2507.22456v1) | I. Aulika, A. Ogurcovs et al. | Optical tactile sensing holds transformative potential for robotics, particularly in collaborative environments where touch perception enhances safety, adaptability, and cognitive interaction. However, traditional tactile technologies based on total internal reflection (TIR) and frustrated total internal reflection (FTIR) - such as those used in touchscreen systems - face significant limitations. These include reliance on multiple infrared light sources and cameras, as well as poor adaptability to the complex, curved geometries often found in robotic systems. To address these challenges, we recently introduced OptoSkin, an advanced optical tactile sensor based on direct Time-of-Flight (ToF) technology, enabling touch and pressure detection. In this study, we investigate how specific material properties, particularly light scattering, influence the sensitivity of contact point detection under direct ToF sensing. Four materials with distinct scattering coefficients were selected to assess their impact on signal quality across different contact scenarios involving various target surfaces. |
| 2025-07-30 | [RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function](http://arxiv.org/abs/2507.22446v1) | Yunrui Yu, Kafeng Wang et al. | Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms. |
| 2025-07-30 | [Operationalization of Scenario-Based Safety Assessment of Automated Driving Systems](http://arxiv.org/abs/2507.22433v1) | Olaf Op den Camp, Erwin de Gelder | Before introducing an Automated Driving System (ADS) on the road at scale, the manufacturer must conduct some sort of safety assurance. To structure and harmonize the safety assurance process, the UNECE WP.29 Working Party on Automated/Autonomous and Connected Vehicles (GRVA) is developing the New Assessment/Test Method (NATM) that indicates what steps need to be taken for safety assessment of an ADS. In this paper, we will show how to practically conduct safety assessment making use of a scenario database, and what additional steps must be taken to fully operationalize the NATM. In addition, we will elaborate on how the use of scenario databases fits with methods developed in the Horizon Europe projects that focus on safety assessment following the NATM approach. |
| 2025-07-30 | [Comparing Normalizing Flows with Kernel Density Estimation in Estimating Risk of Automated Driving Systems](http://arxiv.org/abs/2507.22429v1) | Erwin de Gelder, Maren Buermann et al. | The development of safety validation methods is essential for the safe deployment and operation of Automated Driving Systems (ADSs). One of the goals of safety validation is to prospectively evaluate the risk of an ADS dealing with real-world traffic. Scenario-based assessment is a widely-used approach, where test cases are derived from real-world driving data. To allow for a quantitative analysis of the system performance, the exposure of the scenarios must be accurately estimated. The exposure of scenarios at parameter level is expressed using a Probability Density Function (PDF). However, assumptions about the PDF, such as parameter independence, can introduce errors, while avoiding assumptions often leads to oversimplified models with limited parameters to mitigate the curse of dimensionality.   This paper considers the use of Normalizing Flows (NF) for estimating the PDF of the parameters. NF are a class of generative models that transform a simple base distribution into a complex one using a sequence of invertible and differentiable mappings, enabling flexible, high-dimensional density estimation without restrictive assumptions on the PDF's shape. We demonstrate the effectiveness of NF in quantifying risk and risk uncertainty of an ADS, comparing its performance with Kernel Density Estimation (KDE), a traditional method for non-parametric PDF estimation. While NF require more computational resources compared to KDE, NF is less sensitive to the curse of dimensionality. As a result, NF can improve risk uncertainty estimation, offering a more precise assessment of an ADS's safety.   This work illustrates the potential of NF in scenario-based safety. Future work involves experimenting more with using NF for scenario generation and optimizing the NF architecture, transformation types, and training hyperparameters to further enhance their applicability. |
| 2025-07-30 | [On the Definition of Intelligence](http://arxiv.org/abs/2507.22423v1) | Kei-Sing Ng | To engineer AGI, we should first capture the essence of intelligence in a species-agnostic form that can be evaluated, while being sufficiently general to encompass diverse paradigms of intelligent behavior, including reinforcement learning, generative models, classification, analogical reasoning, and goal-directed decision-making. We propose a general criterion based on sample fidelity: intelligence is the ability, given sample(s) from a category, to generate sample(s) from the same category. We formalise this intuition as {\epsilon}-category intelligence: it is {\epsilon}-intelligent with respect to a category if no chosen admissible distinguisher can separate generated from original samples beyond tolerance {\epsilon}. We present the formal framework, outline empirical protocols, and discuss implications for evaluation, safety, and generalization. |
| 2025-07-30 | [Safety Evaluation of Motion Plans Using Trajectory Predictors as Forward Reachable Set Estimators](http://arxiv.org/abs/2507.22389v1) | Kaustav Chakraborty, Zeyuan Feng et al. | The advent of end-to-end autonomy stacks - often lacking interpretable intermediate modules - has placed an increased burden on ensuring that the final output, i.e., the motion plan, is safe in order to validate the safety of the entire stack. This requires a safety monitor that is both complete (able to detect all unsafe plans) and sound (does not flag safe plans). In this work, we propose a principled safety monitor that leverages modern multi-modal trajectory predictors to approximate forward reachable sets (FRS) of surrounding agents. By formulating a convex program, we efficiently extract these data-driven FRSs directly from the predicted state distributions, conditioned on scene context such as lane topology and agent history. To ensure completeness, we leverage conformal prediction to calibrate the FRS and guarantee coverage of ground-truth trajectories with high probability. To preserve soundness in out-of-distribution (OOD) scenarios or under predictor failure, we introduce a Bayesian filter that dynamically adjusts the FRS conservativeness based on the predictor's observed performance. We then assess the safety of the ego vehicle's motion plan by checking for intersections with these calibrated FRSs, ensuring the plan remains collision-free under plausible future behaviors of others. Extensive experiments on the nuScenes dataset show our approach significantly improves soundness while maintaining completeness, offering a practical and reliable safety monitor for learned autonomy stacks. |
| 2025-07-30 | [Magentic-UI: Towards Human-in-the-loop Agentic Systems](http://arxiv.org/abs/2507.22358v1) | Hussein Mozannar, Gagan Bansal et al. | AI agents powered by large language models are increasingly capable of autonomously completing complex, multi-step tasks using external tools. Yet, they still fall short of human-level performance in most domains including computer use, software development, and research. Their growing autonomy and ability to interact with the outside world, also introduces safety and security risks including potentially misaligned actions and adversarial manipulation. We argue that human-in-the-loop agentic systems offer a promising path forward, combining human oversight and control with AI efficiency to unlock productivity from imperfect systems. We introduce Magentic-UI, an open-source web interface for developing and studying human-agent interaction. Built on a flexible multi-agent architecture, Magentic-UI supports web browsing, code execution, and file manipulation, and can be extended with diverse tools via Model Context Protocol (MCP). Moreover, Magentic-UI presents six interaction mechanisms for enabling effective, low-cost human involvement: co-planning, co-tasking, multi-tasking, action guards, and long-term memory. We evaluate Magentic-UI across four dimensions: autonomous task completion on agentic benchmarks, simulated user testing of its interaction capabilities, qualitative studies with real users, and targeted safety assessments. Our findings highlight Magentic-UI's potential to advance safe and efficient human-agent collaboration. |
| 2025-07-30 | [Risk-inclusive Contextual Bandits for Early Phase Clinical Trials](http://arxiv.org/abs/2507.22344v1) | Rohit Kanrar, Chunlin Li et al. | Early-phase clinical trials face the challenge of selecting optimal drug doses that balance safety and efficacy due to uncertain dose-response relationships and varied participant characteristics. Traditional randomized dose allocation often exposes participants to sub-optimal doses by not considering individual covariates, necessitating larger sample sizes and prolonging drug development. This paper introduces a risk-inclusive contextual bandit algorithm that utilizes multi-arm bandit (MAB) strategies to optimize dosing through participant-specific data integration. By combining two separate Thompson samplers, one for efficacy and one for safety, the algorithm enhances the balance between efficacy and safety in dose allocation. The effect sizes are estimated with a generalized version of asymptotic confidence sequences (AsympCS, Waudby-Smith et al., 2024), offering a uniform coverage guarantee for sequential causal inference over time. The validity of AsympCS is also established in the MAB setup with a possibly mis-specified model. The empirical results demonstrate the strengths of this method in optimizing dose allocation compared to randomized allocations and traditional contextual bandits focused solely on efficacy. Moreover, an application on real data generated from a recent Phase IIb study aligns with actual findings. |
| 2025-07-30 | [An Explainable Emotion Alignment Framework for LLM-Empowered Agent in Metaverse Service Ecosystem](http://arxiv.org/abs/2507.22326v1) | Qun Ma, Xiao Xue et al. | Metaverse service is a product of the convergence between Metaverse and service systems, designed to address service-related challenges concerning digital avatars, digital twins, and digital natives within Metaverse. With the rise of large language models (LLMs), agents now play a pivotal role in Metaverse service ecosystem, serving dual functions: as digital avatars representing users in the virtual realm and as service assistants (or NPCs) providing personalized support. However, during the modeling of Metaverse service ecosystems, existing LLM-based agents face significant challenges in bridging virtual-world services with real-world services, particularly regarding issues such as character data fusion, character knowledge association, and ethical safety concerns. This paper proposes an explainable emotion alignment framework for LLM-based agents in Metaverse Service Ecosystem. It aims to integrate factual factors into the decision-making loop of LLM-based agents, systematically demonstrating how to achieve more relational fact alignment for these agents. Finally, a simulation experiment in the Offline-to-Offline food delivery scenario is conducted to evaluate the effectiveness of this framework, obtaining more realistic social emergence. |
| 2025-07-29 | [Multi-Agent Path Finding Among Dynamic Uncontrollable Agents with Statistical Safety Guarantees](http://arxiv.org/abs/2507.22282v1) | Kegan J. Strawn, Thomy Phan et al. | Existing multi-agent path finding (MAPF) solvers do not account for uncertain behavior of uncontrollable agents. We present a novel variant of Enhanced Conflict-Based Search (ECBS), for both one-shot and lifelong MAPF in dynamic environments with uncontrollable agents. Our method consists of (1) training a learned predictor for the movement of uncontrollable agents, (2) quantifying the prediction error using conformal prediction (CP), a tool for statistical uncertainty quantification, and (3) integrating these uncertainty intervals into our modified ECBS solver. Our method can account for uncertain agent behavior, comes with statistical guarantees on collision-free paths for one-shot missions, and scales to lifelong missions with a receding horizon sequence of one-shot instances. We run our algorithm, CP-Solver, across warehouse and game maps, with competitive throughput and reduced collisions. |
| 2025-07-29 | [Promoting Online Safety by Simulating Unsafe Conversations with LLMs](http://arxiv.org/abs/2507.22267v1) | Owen Hoffman, Kangze Peng et al. | Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM. |

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



