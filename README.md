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
| 2025-05-06 | [WebGen-Bench: Evaluating LLMs on Generating Interactive and Functional Websites from Scratch](http://arxiv.org/abs/2505.03733v1) | Zimu Lu, Yunqiao Yang et al. | LLM-based agents have demonstrated great potential in generating and managing code within complex codebases. In this paper, we introduce WebGen-Bench, a novel benchmark designed to measure an LLM-based agent's ability to create multi-file website codebases from scratch. It contains diverse instructions for website generation, created through the combined efforts of human annotators and GPT-4o. These instructions span three major categories and thirteen minor categories, encompassing nearly all important types of web applications. To assess the quality of the generated websites, we use GPT-4o to generate test cases targeting each functionality described in the instructions, and then manually filter, adjust, and organize them to ensure accuracy, resulting in 647 test cases. Each test case specifies an operation to be performed on the website and the expected result after the operation. To automate testing and improve reproducibility, we employ a powerful web-navigation agent to execute tests on the generated websites and determine whether the observed responses align with the expected results. We evaluate three high-performance code-agent frameworks, Bolt.diy, OpenHands, and Aider, using multiple proprietary and open-source LLMs as engines. The best-performing combination, Bolt.diy powered by DeepSeek-R1, achieves only 27.8\% accuracy on the test cases, highlighting the challenging nature of our benchmark. Additionally, we construct WebGen-Instruct, a training set consisting of 6,667 website-generation instructions. Training Qwen2.5-Coder-32B-Instruct on Bolt.diy trajectories generated from a subset of this training set achieves an accuracy of 38.2\%, surpassing the performance of the best proprietary model. |
| 2025-05-06 | [Toward a Harmonized Approach -- Requirement-based Structuring of a Safety Assurance Argumentation for Automated Vehicles](http://arxiv.org/abs/2505.03709v1) | M. Loba, N. F. Salem et al. | Despite increasing testing operation on public roads, media reports on incidents show that safety issues remain to this day. One major cause factoring into this circumstance is high development uncertainty that manufacturers face when deploying these systems in an open context. In particular, one challenge is establishing a valid argument at design time that the vehicle will exhibit reasonable residual risk when operating in its intended operational design domain. Regulations, such as the European Implementing Regulation 2022/1426, require manufacturers to provide a safety assurance argumentation for SAE-Level-4 automated vehicles. While there is extensive literature on assurance cases for safety-critical systems, the domain of automated driving lacks explicit requirements regarding the creation of safety assurance argumentations. In this paper, we aim to narrow this gap by elaborating a requirement-based approach. We derive structural requirements for an argumentation from literature and supplement these with requirements derived from stakeholder concerns. We implement the requirements, yielding a proposal for an overall argumentation structure. The resulting "safety arguments" argue over four topic complexes: The developed product, the underlying process including its conformance/compliance to standards/laws, as well as the argumentations' context and soundness. Finally, we instantiate this structure with respect to domain-specific needs and principles. |
| 2025-05-06 | [Frenet Corridor Planner: An Optimal Local Path Planning Framework for Autonomous Driving](http://arxiv.org/abs/2505.03695v1) | Faizan M. Tariq, Zheng-Hang Yeh et al. | Motivated by the requirements for effectiveness and efficiency, path-speed decomposition-based trajectory planning methods have widely been adopted for autonomous driving applications. While a global route can be pre-computed offline, real-time generation of adaptive local paths remains crucial. Therefore, we present the Frenet Corridor Planner (FCP), an optimization-based local path planning strategy for autonomous driving that ensures smooth and safe navigation around obstacles. Modeling the vehicles as safety-augmented bounding boxes and pedestrians as convex hulls in the Frenet space, our approach defines a drivable corridor by determining the appropriate deviation side for static obstacles. Thereafter, a modified space-domain bicycle kinematics model enables path optimization for smoothness, boundary clearance, and dynamic obstacle risk minimization. The optimized path is then passed to a speed planner to generate the final trajectory. We validate FCP through extensive simulations and real-world hardware experiments, demonstrating its efficiency and effectiveness. |
| 2025-05-06 | [Demonstrating ViSafe: Vision-enabled Safety for High-speed Detect and Avoid](http://arxiv.org/abs/2505.03694v1) | Parv Kapoor, Ian Higgins et al. | Assured safe-separation is essential for achieving seamless high-density operation of airborne vehicles in a shared airspace. To equip resource-constrained aerial systems with this safety-critical capability, we present ViSafe, a high-speed vision-only airborne collision avoidance system. ViSafe offers a full-stack solution to the Detect and Avoid (DAA) problem by tightly integrating a learning-based edge-AI framework with a custom multi-camera hardware prototype designed under SWaP-C constraints. By leveraging perceptual input-focused control barrier functions (CBF) to design, encode, and enforce safety thresholds, ViSafe can provide provably safe runtime guarantees for self-separation in high-speed aerial operations. We evaluate ViSafe's performance through an extensive test campaign involving both simulated digital twins and real-world flight scenarios. By independently varying agent types, closure rates, interaction geometries, and environmental conditions (e.g., weather and lighting), we demonstrate that ViSafe consistently ensures self-separation across diverse scenarios. In first-of-its-kind real-world high-speed collision avoidance tests with closure rates reaching 144 km/h, ViSafe sets a new benchmark for vision-only autonomous collision avoidance, establishing a new standard for safety in high-speed aerial navigation. |
| 2025-05-06 | [Moral Testing of Autonomous Driving Systems](http://arxiv.org/abs/2505.03683v1) | Wenbing Tang, Mingfei Cheng et al. | Autonomous Driving System (ADS) testing plays a crucial role in their development, with the current focus primarily on functional and safety testing. However, evaluating the non-functional morality of ADSs, particularly their decision-making capabilities in unavoidable collision scenarios, is equally important to ensure the systems' trustworthiness and public acceptance. Unfortunately, testing ADS morality is nearly impossible due to the absence of universal moral principles. To address this challenge, this paper first extracts a set of moral meta-principles derived from existing moral experiments and well-established social science theories, aiming to capture widely recognized and common-sense moral values for ADSs. These meta-principles are then formalized as quantitative moral metamorphic relations, which act as the test oracle. Furthermore, we propose a metamorphic testing framework to systematically identify potential moral issues. Finally, we illustrate the implementation of the framework and present typical violation cases using the VIRES VTD simulator and its built-in ADS. |
| 2025-05-06 | [BURNS: Backward Underapproximate Reachability for Neural-Feedback-Loop Systems](http://arxiv.org/abs/2505.03643v1) | Chelsea Sidrane, Jana Tumova | Learning-enabled planning and control algorithms are increasingly popular, but they often lack rigorous guarantees of performance or safety. We introduce an algorithm for computing underapproximate backward reachable sets of nonlinear discrete time neural feedback loops. We then use the backward reachable sets to check goal-reaching properties. Our algorithm is based on overapproximating the system dynamics function to enable computation of underapproximate backward reachable sets through solutions of mixed-integer linear programs. We rigorously analyze the soundness of our algorithm and demonstrate it on a numerical example. Our work expands the class of properties that can be verified for learning-enabled systems. |
| 2025-05-06 | [Backstepping Reach-avoid Controller Synthesis for Multi-input Multi-output Systems with Mixed Relative Degrees](http://arxiv.org/abs/2505.03612v1) | Jianqiang Ding, Dingran Yuan et al. | Designing controllers with provable formal guarantees has become an urgent requirement for cyber-physical systems in safety-critical scenarios. Beyond addressing scalability in high-dimensional implementations, controller synthesis methodologies separating safety and reachability objectives may risk optimization infeasibility due to conflicting constraints, thereby significantly undermining their applicability in practical applications. In this paper, by leveraging feedback linearization and backstepping techniques, we present a novel framework for constructing provable reach-avoid formal certificates tailored to multi-input multi-output systems. Based on this, we developed a systematic synthesis approach for controllers with reach-avoid guarantees, which ensures that the outputs of the system eventually enter the predefined target set while staying within the required safe set. Finally, we demonstrate the effectiveness of our method through simulations. |
| 2025-05-06 | [LlamaFirewall: An open source guardrail system for building secure AI agents](http://arxiv.org/abs/2505.03574v1) | Sahana Chennabasappa, Cyrus Nikolaidis et al. | Large language models (LLMs) have evolved from simple chatbots into autonomous agents capable of performing complex tasks such as editing production code, orchestrating workflows, and taking higher-stakes actions based on untrusted inputs like webpages and emails. These capabilities introduce new security risks that existing security measures, such as model fine-tuning or chatbot-focused guardrails, do not fully address. Given the higher stakes and the absence of deterministic solutions to mitigate these risks, there is a critical need for a real-time guardrail monitor to serve as a final layer of defense, and support system level, use case specific safety policy definition and enforcement. We introduce LlamaFirewall, an open-source security focused guardrail framework designed to serve as a final layer of defense against security risks associated with AI Agents. Our framework mitigates risks such as prompt injection, agent misalignment, and insecure code risks through three powerful guardrails: PromptGuard 2, a universal jailbreak detector that demonstrates clear state of the art performance; Agent Alignment Checks, a chain-of-thought auditor that inspects agent reasoning for prompt injection and goal misalignment, which, while still experimental, shows stronger efficacy at preventing indirect injections in general scenarios than previously proposed approaches; and CodeShield, an online static analysis engine that is both fast and extensible, aimed at preventing the generation of insecure or dangerous code by coding agents. Additionally, we include easy-to-use customizable scanners that make it possible for any developer who can write a regular expression or an LLM prompt to quickly update an agent's security guardrails. |
| 2025-05-06 | [Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models](http://arxiv.org/abs/2505.03469v1) | Bin Yu, Hang Yuan et al. | Recent advances in large language models have demonstrated that Supervised Fine-Tuning (SFT) with Chain-of-Thought (CoT) reasoning data distilled from large reasoning models (e.g., DeepSeek R1) can effectively transfer reasoning capabilities to non-reasoning models. However, models fine-tuned with this approach inherit the "overthinking" problem from teacher models, producing verbose and redundant reasoning chains during inference. To address this challenge, we propose \textbf{L}ong-\textbf{S}hort Chain-of-Thought \textbf{Mixture} \textbf{S}upervised \textbf{F}ine-\textbf{T}uning (\textbf{LS-Mixture SFT}), which combines long CoT reasoning dataset with their short counterparts obtained through structure-preserved rewriting. Our experiments demonstrate that models trained using the LS-Mixture SFT method, compared to those trained with direct SFT, achieved an average accuracy improvement of 2.3\% across various benchmarks while substantially reducing model response length by approximately 47.61\%. This work offers an approach to endow non-reasoning models with reasoning capabilities through supervised fine-tuning while avoiding the inherent overthinking problems inherited from teacher models, thereby enabling efficient reasoning in the fine-tuned models. |
| 2025-05-06 | [Close-Fitting Dressing Assistance Based on State Estimation of Feet and Garments with Semantic-based Visual Attention](http://arxiv.org/abs/2505.03400v1) | Takuma Tsukakoshi, Tamon Miyake et al. | As the population continues to age, a shortage of caregivers is expected in the future. Dressing assistance, in particular, is crucial for opportunities for social participation. Especially dressing close-fitting garments, such as socks, remains challenging due to the need for fine force adjustments to handle the friction or snagging against the skin, while considering the shape and position of the garment. This study introduces a method uses multi-modal information including not only robot's camera images, joint angles, joint torques, but also tactile forces for proper force interaction that can adapt to individual differences in humans. Furthermore, by introducing semantic information based on object concepts, rather than relying solely on RGB data, it can be generalized to unseen feet and background. In addition, incorporating depth data helps infer relative spatial relationship between the sock and the foot. To validate its capability for semantic object conceptualization and to ensure safety, training data were collected using a mannequin, and subsequent experiments were conducted with human subjects. In experiments, the robot successfully adapted to previously unseen human feet and was able to put socks on 10 participants, achieving a higher success rate than Action Chunking with Transformer and Diffusion Policy. These results demonstrate that the proposed model can estimate the state of both the garment and the foot, enabling precise dressing assistance for close-fitting garments. |
| 2025-05-06 | [Miniature multihole airflow sensor for lightweight aircraft over wide speed and angular range](http://arxiv.org/abs/2505.03331v1) | Lukas Stuber, Simon Jeger et al. | An aircraft's airspeed, angle of attack, and angle of side slip are crucial to its safety, especially when flying close to the stall regime. Various solutions exist, including pitot tubes, angular vanes, and multihole pressure probes. However, current sensors are either too heavy (>30 g) or require large airspeeds (>20 m/s), making them unsuitable for small uncrewed aerial vehicles. We propose a novel multihole pressure probe, integrating sensing electronics in a single-component structure, resulting in a mechanically robust and lightweight sensor (9 g), which we released to the public domain. Since there is no consensus on two critical design parameters, tip shape (conical vs spherical) and hole spacing (distance between holes), we provide a study on measurement accuracy and noise generation using wind tunnel experiments. The sensor is calibrated using a multivariate polynomial regression model over an airspeed range of 3-27 m/s and an angle of attack/sideslip range of +-35{\deg}, achieving a mean absolute error of 0.44 m/s and 0.16{\deg}. Finally, we validated the sensor in outdoor flights near the stall regime. Our probe enabled accurate estimations of airspeed, angle of attack and sideslip during different acrobatic manoeuvres. Due to its size and weight, this sensor will enable safe flight for lightweight, uncrewed aerial vehicles flying at low speeds close to the stall regime. |
| 2025-05-06 | [Ψ-Arena: Interactive Assessment and Optimization of LLM-based Psychological Counselors with Tripartite Feedback](http://arxiv.org/abs/2505.03293v1) | Shijing Zhu, Zhuang Chen et al. | Large language models (LLMs) have shown promise in providing scalable mental health support, while evaluating their counseling capability remains crucial to ensure both efficacy and safety. Existing evaluations are limited by the static assessment that focuses on knowledge tests, the single perspective that centers on user experience, and the open-loop framework that lacks actionable feedback. To address these issues, we propose {\Psi}-Arena, an interactive framework for comprehensive assessment and optimization of LLM-based counselors, featuring three key characteristics: (1) Realistic arena interactions that simulate real-world counseling through multi-stage dialogues with psychologically profiled NPC clients, (2) Tripartite evaluation that integrates assessments from the client, counselor, and supervisor perspectives, and (3) Closed-loop optimization that iteratively improves LLM counselors using diagnostic feedback. Experiments across eight state-of-the-art LLMs show significant performance variations in different real-world scenarios and evaluation perspectives. Moreover, reflection-based optimization results in up to a 141% improvement in counseling performance. We hope PsychoArena provides a foundational resource for advancing reliable and human-aligned LLM applications in mental healthcare. |
| 2025-05-06 | [Enabling Robots to Autonomously Search Dynamic Cluttered Post-Disaster Environments](http://arxiv.org/abs/2505.03283v1) | Karlo Rado, Mirko Baglioni et al. | Robots will bring search and rescue (SaR) in disaster response to another level, in case they can autonomously take over dangerous SaR tasks from humans. A main challenge for autonomous SaR robots is to safely navigate in cluttered environments with uncertainties, while avoiding static and moving obstacles. We propose an integrated control framework for SaR robots in dynamic, uncertain environments, including a computationally efficient heuristic motion planning system that provides a nominal (assuming there are no uncertainties) collision-free trajectory for SaR robots and a robust motion tracking system that steers the robot to track this reference trajectory, taking into account the impact of uncertainties. The control architecture guarantees a balanced trade-off among various SaR objectives, while handling the hard constraints, including safety. The results of various computer-based simulations, presented in this paper, showed significant out-performance (of up to 42.3%) of the proposed integrated control architecture compared to two commonly used state-of-the-art methods (Rapidly-exploring Random Tree and Artificial Potential Function) in reaching targets (e.g., trapped victims in SaR) safely, collision-free, and in the shortest possible time. |
| 2025-05-06 | [RADE: Learning Risk-Adjustable Driving Environment via Multi-Agent Conditional Diffusion](http://arxiv.org/abs/2505.03178v1) | Jiawei Wang, Xintao Yan et al. | Generating safety-critical scenarios in high-fidelity simulations offers a promising and cost-effective approach for efficient testing of autonomous vehicles. Existing methods typically rely on manipulating a single vehicle's trajectory through sophisticated designed objectives to induce adversarial interactions, often at the cost of realism and scalability. In this work, we propose the Risk-Adjustable Driving Environment (RADE), a simulation framework that generates statistically realistic and risk-adjustable traffic scenes. Built upon a multi-agent diffusion architecture, RADE jointly models the behavior of all agents in the environment and conditions their trajectories on a surrogate risk measure. Unlike traditional adversarial methods, RADE learns risk-conditioned behaviors directly from data, preserving naturalistic multi-agent interactions with controllable risk levels. To ensure physical plausibility, we incorporate a tokenized dynamics check module that efficiently filters generated trajectories using a motion vocabulary. We validate RADE on the real-world rounD dataset, demonstrating that it preserves statistical realism across varying risk levels and naturally increases the likelihood of safety-critical events as the desired risk level grows up. Our results highlight RADE's potential as a scalable and realistic tool for AV safety evaluation. |
| 2025-05-06 | [VISLIX: An XAI Framework for Validating Vision Models with Slice Discovery and Analysis](http://arxiv.org/abs/2505.03132v1) | Xinyuan Yan, Xiwei Xuan et al. | Real-world machine learning models require rigorous evaluation before deployment, especially in safety-critical domains like autonomous driving and surveillance. The evaluation of machine learning models often focuses on data slices, which are subsets of the data that share a set of characteristics. Data slice finding automatically identifies conditions or data subgroups where models underperform, aiding developers in mitigating performance issues. Despite its popularity and effectiveness, data slicing for vision model validation faces several challenges. First, data slicing often needs additional image metadata or visual concepts, and falls short in certain computer vision tasks, such as object detection. Second, understanding data slices is a labor-intensive and mentally demanding process that heavily relies on the expert's domain knowledge. Third, data slicing lacks a human-in-the-loop solution that allows experts to form hypothesis and test them interactively. To overcome these limitations and better support the machine learning operations lifecycle, we introduce VISLIX, a novel visual analytics framework that employs state-of-the-art foundation models to help domain experts analyze slices in computer vision models. Our approach does not require image metadata or visual concepts, automatically generates natural language insights, and allows users to test data slice hypothesis interactively. We evaluate VISLIX with an expert study and three use cases, that demonstrate the effectiveness of our tool in providing comprehensive insights for validating object detection models. |
| 2025-05-05 | [A Modal-Space Formulation for Momentum Observer Contact Estimation and Effects of Uncertainty for Continuum Robots](http://arxiv.org/abs/2505.03044v1) | Garrison L. H. Johnston, Neel Shihora et al. | Contact detection for continuum and soft robots has been limited in past works to statics or kinematics-based methods with assumed circular bending curvature or known bending profiles. In this paper, we adapt the generalized momentum observer contact estimation method to continuum robots. This is made possible by leveraging recent results for real-time shape sensing of continuum robots along with a modal-space representation of the robot dynamics. In addition to presenting an approach for estimating the generalized forces due to contact via a momentum observer, we present a constrained optimization method to identify the wrench imparted on the robot during contact. We also present an approach for investigating the effects of unmodeled deviations in the robot's dynamic state on the contact detection method and we validate our algorithm by simulations and experiments. We also compare the performance of the momentum observer to the joint force deviation method, a direct estimation approach using the robot's full dynamic model. We also demonstrate a basic extension of the method to multisegment continuum robots. Results presented in this work extend dynamic contact detection to the domain of continuum and soft robots and can be used to improve the safety of large-scale continuum robots for human-robot collaboration. |
| 2025-05-05 | [Evidence of a fraction of LIGO/Virgo/KAGRA events coming from active galactic nuclei](http://arxiv.org/abs/2505.02924v1) | Liang-Gui Zhu, Xian Chen | The formation channels of the gravitational-wave (GW) sources detected by LIGO/Virgo/KAGRA (LVK) remain poorly constrained. Active galactic nucleus (AGN) has been proposed as one of the potential hosts but the fraction of GW events originating from AGNs has not been quantified. Here, we constrain the AGN-origin fraction $f_{\rm agn}$ by analyzing the spatial correlation between GW source localizations ($O1\!-\!O4$a) and AGN distributions (SDSS DR16). We find evidence of an excess of low-luminosity ($L_{\rm bol} \le 10^{45}~\!\mathrm{erg~s}^{-1}$) as well as low-Eddington ratio ($\lambda_{\rm Edd} \le 0.05$) AGNs around the LVK events, the explanation of which requires $f_{\rm agn} = 0.39^{+0.41}_{-0.32}$ and $0.29^{+0.40}_{-0.25}$ (90\% confidence level) of the LVK events originating from these respective AGN populations. Monte Carlo simulations confirm that this correlation is unlikely to arise from random coincidence, further supported by anomalous variation of the error of $f_{\rm agn}$ with GW event counts. These results provide the first observational evidence for GW sources coming from either low-luminosity or low-accretion-rate AGNs, offering critical insights into the environmental dependencies of the formation of GW sources. |
| 2025-05-05 | [Stabilizing dark matter with quantum scale symmetry](http://arxiv.org/abs/2505.02803v1) | Abhishek Chikkaballi, Kamila Kowalska et al. | In the context of gauge-Yukawa theories with trans-Planckian asymptotic safety, quantum scale symmetry can prevent the appearance in the Lagrangian of couplings that would otherwise be allowed by the gauge symmetry. Such couplings correspond to irrelevant Gaussian fixed points of the renormalization group flow. Their absence in the theory implies that different sectors of the gauge-Yukawa theory are secluded from one another, in similar fashion to the effects of a global or a discrete symmetry. As an example, we impose the trans-Planckian scale symmetry on a model of Grand Unification based on the gauge group SU(6), showing that it leads to the emergence of several fermionic WIMP dark matter candidates whose coupling strengths are entirely predicted by the UV completion. |
| 2025-05-05 | [When Your Own Output Becomes Your Training Data: Noise-to-Meaning Loops and a Formal RSI Trigger](http://arxiv.org/abs/2505.02888v1) | Rintaro Ando | We present Noise-to-Meaning Recursive Self-Improvement (N2M-RSI), a minimal formal model showing that once an AI agent feeds its own outputs back as inputs and crosses an explicit information-integration threshold, its internal complexity will grow without bound under our assumptions. The framework unifies earlier ideas on self-prompting large language models, G\"odelian self-reference, and AutoML, yet remains implementation-agnostic. The model furthermore scales naturally to interacting swarms of agents, hinting at super-linear effects once communication among instances is permitted. For safety reasons, we omit system-specific implementation details and release only a brief, model-agnostic toy prototype in Appendix C. |
| 2025-05-05 | [A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning and Inference-time Scaling Law](http://arxiv.org/abs/2505.02665v1) | Qianjun Pan, Wenkai Ji et al. | This survey explores recent advancements in reasoning large language models (LLMs) designed to mimic "slow thinking" - a reasoning process inspired by human cognition, as described in Kahneman's Thinking, Fast and Slow. These models, like OpenAI's o1, focus on scaling computational resources dynamically during complex tasks, such as math reasoning, visual reasoning, medical diagnosis, and multi-agent debates. We present the development of reasoning LLMs and list their key technologies. By synthesizing over 100 studies, it charts a path toward LLMs that combine human-like deep thinking with scalable efficiency for reasoning. The review breaks down methods into three categories: (1) test-time scaling dynamically adjusts computation based on task complexity via search and sampling, dynamic verification; (2) reinforced learning refines decision-making through iterative improvement leveraging policy networks, reward models, and self-evolution strategies; and (3) slow-thinking frameworks (e.g., long CoT, hierarchical processes) that structure problem-solving with manageable steps. The survey highlights the challenges and further directions of this domain. Understanding and advancing the reasoning abilities of LLMs is crucial for unlocking their full potential in real-world applications, from scientific discovery to decision support systems. |

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



