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
| 2026-01-27 | [Estimating Trust in Human-Robot Collaboration through Behavioral Indicators and Explainability](http://arxiv.org/abs/2601.19856v1) | Giulio Campagna, Marta Lagomarsino et al. | Industry 5.0 focuses on human-centric collaboration between humans and robots, prioritizing safety, comfort, and trust. This study introduces a data-driven framework to assess trust using behavioral indicators. The framework employs a Preference-Based Optimization algorithm to generate trust-enhancing trajectories based on operator feedback. This feedback serves as ground truth for training machine learning models to predict trust levels from behavioral indicators. The framework was tested in a chemical industry scenario where a robot assisted a human operator in mixing chemicals. Machine learning models classified trust with over 80\% accuracy, with the Voting Classifier achieving 84.07\% accuracy and an AUC-ROC score of 0.90. These findings underscore the effectiveness of data-driven methods in assessing trust within human-robot collaboration, emphasizing the valuable role behavioral indicators play in predicting the dynamics of human trust. |
| 2026-01-27 | [Zero-Shot Stance Detection in the Wild: Dynamic Target Generation and Multi-Target Adaptation](http://arxiv.org/abs/2601.19802v1) | Aohua Li, Yuanshuo Zhang et al. | Current stance detection research typically relies on predicting stance based on given targets and text. However, in real-world social media scenarios, targets are neither predefined nor static but rather complex and dynamic. To address this challenge, we propose a novel task: zero-shot stance detection in the wild with Dynamic Target Generation and Multi-Target Adaptation (DGTA), which aims to automatically identify multiple target-stance pairs from text without prior target knowledge. We construct a Chinese social media stance detection dataset and design multi-dimensional evaluation metrics. We explore both integrated and two-stage fine-tuning strategies for large language models (LLMs) and evaluate various baseline models. Experimental results demonstrate that fine-tuned LLMs achieve superior performance on this task: the two-stage fine-tuned Qwen2.5-7B attains the highest comprehensive target recognition score of 66.99%, while the integrated fine-tuned DeepSeek-R1-Distill-Qwen-7B achieves a stance detection F1 score of 79.26%. |
| 2026-01-27 | [GAVEL: Towards rule-based safety through activation monitoring](http://arxiv.org/abs/2601.19768v1) | Shir Rozenfeld, Rahul Pankajakshan et al. | Large language models (LLMs) are increasingly paired with activation-based monitoring to detect and prevent harmful behaviors that may not be apparent at the surface-text level. However, existing activation safety approaches, trained on broad misuse datasets, struggle with poor precision, limited flexibility, and lack of interpretability. This paper introduces a new paradigm: rule-based activation safety, inspired by rule-sharing practices in cybersecurity. We propose modeling activations as cognitive elements (CEs), fine-grained, interpretable factors such as ''making a threat'' and ''payment processing'', that can be composed to capture nuanced, domain-specific behaviors with higher precision. Building on this representation, we present a practical framework that defines predicate rules over CEs and detects violations in real time. This enables practitioners to configure and update safeguards without retraining models or detectors, while supporting transparency and auditability. Our results show that compositional rule-based activation safety improves precision, supports domain customization, and lays the groundwork for scalable, interpretable, and auditable AI governance. We will release GAVEL as an open-source framework and provide an accompanying automated rule creation tool. |
| 2026-01-27 | [RvB: Automating AI System Hardening via Iterative Red-Blue Games](http://arxiv.org/abs/2601.19726v1) | Lige Huang, Zicheng Liu et al. | The dual offensive and defensive utility of Large Language Models (LLMs) highlights a critical gap in AI security: the lack of unified frameworks for dynamic, iterative adversarial adaptation hardening. To bridge this gap, we propose the Red Team vs. Blue Team (RvB) framework, formulated as a training-free, sequential, imperfect-information game. In this process, the Red Team exposes vulnerabilities, driving the Blue Team to learning effective solutions without parameter updates. We validate our framework across two challenging domains: dynamic code hardening against CVEs and guardrail optimization against jailbreaks. Our empirical results show that this interaction compels the Blue Team to learn fundamental defensive principles, leading to robust remediations that are not merely overfitted to specific exploits. RvB achieves Defense Success Rates of 90\% and 45\% across the respective tasks while maintaining near 0\% False Positive Rates, significantly surpassing baselines. This work establishes the iterative adversarial interaction framework as a practical paradigm that automates the continuous hardening of AI systems. |
| 2026-01-27 | [Enhancing Worker Safety in Harbors Using Quadruped Robots](http://arxiv.org/abs/2601.19643v1) | Zoe Betta, Davide Corongiu et al. | Infrastructure inspection is becoming increasingly relevant in the field of robotics due to its significant impact on ensuring workers' safety. The harbor environment presents various challenges in designing a robotic solution for inspection, given the complexity of daily operations. This work introduces an initial phase to identify critical areas within the port environment. Following this, a preliminary solution using a quadruped robot for inspecting these critical areas is analyzed. |
| 2026-01-27 | [R^3: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning](http://arxiv.org/abs/2601.19620v1) | Zhizheng Jiang, Kang Zhao et al. | Large reasoning models (LRMs) aim to solve diverse and complex problems through structured reasoning. Recent advances in group-based policy optimization methods have shown promise in enabling stable advantage estimation without reliance on process-level annotations. However, these methods rely on advantage gaps induced by high-quality samples within the same batch, which makes the training process fragile and inefficient when intra-group advantages collapse under challenging tasks. To address these problems, we propose a reinforcement learning mechanism named \emph{\textbf{R^3}} that along three directions: (1) a \emph{cross-context \underline{\textbf{R}}eplay} strategy that maintains the intra-group advantage by recalling valuable examples from historical trajectories of the same query, (2) an \emph{in-context self-\underline{\textbf{R}}eflection} mechanism enabling models to refine outputs by leveraging past failures, and (3) a \emph{structural entropy \underline{\textbf{R}}anking reward}, which assigns relative rewards to truncated or failed samples by ranking responses based on token-level entropy patterns, capturing both local exploration and global stability. We implement our method on Deepseek-R1-Distill-Qwen-1.5B and train it on the DeepscaleR-40k in the math domain. Experiments demonstrate our method achieves SoTA performance on several math benchmarks, representing significant improvements and fewer reasoning tokens over the base models. Code and model will be released. |
| 2026-01-27 | [Safe Exploration via Policy Priors](http://arxiv.org/abs/2601.19612v1) | Manuel Wendl, Yarden As et al. | Safe exploration is a key requirement for reinforcement learning (RL) agents to learn and adapt online, beyond controlled (e.g. simulated) environments. In this work, we tackle this challenge by utilizing suboptimal yet conservative policies (e.g., obtained from offline data or simulators) as priors. Our approach, SOOPER, uses probabilistic dynamics models to optimistically explore, yet pessimistically fall back to the conservative policy prior if needed. We prove that SOOPER guarantees safety throughout learning, and establish convergence to an optimal policy by bounding its cumulative regret. Extensive experiments on key safe RL benchmarks and real-world hardware demonstrate that SOOPER is scalable, outperforms the state-of-the-art and validate our theoretical guarantees in practice. |
| 2026-01-27 | [Exposure-Aware Beamforming for mmWave Systems: From EM Theory to Thermal Compliance](http://arxiv.org/abs/2601.19587v1) | Zihan Zhou, Ang Chen et al. | Electromagnetic (EM) exposure compliance has long been recognized as a crucial aspect of communications terminal designs. However, accurately assessing the impact of EM exposure for proper design strategies remains challenging. In this paper, we develop a long-term thermal EM exposure constraint model and propose a novel adaptive exposure-aware beamforming design for an mmWave uplink system. Specifically, we first establish an equivalent channel model based on Maxwell's radiation equations, which accurately captures the EM physical effects. Then, we derive a closed-form thermal impulse response model from the Pennes bioheat transfer equation (BHTE), characterizing the thermal inertia of tissue. Inspired by this model, we formulate a beamforming optimization problem that translates rigid instantaneous exposure limits into a flexible long-term thermal budget constraint. Furthermore, we develop a low-complexity online beamforming algorithm based on Lyapunov optimization theory, obtaining a closed-form near-optimal solution. Simulation results demonstrate that the proposed algorithm effectively stabilizes tissue temperature near a predefined safety threshold and significantly outperforms the conventional scheme with instantaneous exposure constraints. |
| 2026-01-27 | [ScenePilot-Bench: A Large-Scale Dataset and Benchmark for Evaluation of Vision-Language Models in Autonomous Driving](http://arxiv.org/abs/2601.19582v1) | Yujin Wang, Yutong Zheng et al. | In this paper, we introduce ScenePilot-Bench, a large-scale first-person driving benchmark designed to evaluate vision-language models (VLMs) in autonomous driving scenarios. ScenePilot-Bench is built upon ScenePilot-4K, a diverse dataset comprising 3,847 hours of driving videos, annotated with multi-granularity information including scene descriptions, risk assessments, key participant identification, ego trajectories, and camera parameters. The benchmark features a four-axis evaluation suite that assesses VLM capabilities in scene understanding, spatial perception, motion planning, and GPT-Score, with safety-aware metrics and cross-region generalization settings. We benchmark representative VLMs on ScenePilot-Bench, providing empirical analyses that clarify current performance boundaries and identify gaps for driving-oriented reasoning. ScenePilot-Bench offers a comprehensive framework for evaluating and advancing VLMs in safety-critical autonomous driving contexts. |
| 2026-01-27 | [Automated Safety Benchmarking: A Multi-agent Pipeline for LVLMs](http://arxiv.org/abs/2601.19507v1) | Xiangyang Zhu, Yuan Tian et al. | Large vision-language models (LVLMs) exhibit remarkable capabilities in cross-modal tasks but face significant safety challenges, which undermine their reliability in real-world applications. Efforts have been made to build LVLM safety evaluation benchmarks to uncover their vulnerability. However, existing benchmarks are hindered by their labor-intensive construction process, static complexity, and limited discriminative power. Thus, they may fail to keep pace with rapidly evolving models and emerging risks. To address these limitations, we propose VLSafetyBencher, the first automated system for LVLM safety benchmarking. VLSafetyBencher introduces four collaborative agents: Data Preprocessing, Generation, Augmentation, and Selection agents to construct and select high-quality samples. Experiments validates that VLSafetyBencher can construct high-quality safety benchmarks within one week at a minimal cost. The generated benchmark effectively distinguish safety, with a safety rate disparity of 70% between the most and least safe models. |
| 2026-01-27 | [Reinforcement Learning Goal-Reaching Control with Guaranteed Lyapunov-Like Stabilizer for Mobile Robots](http://arxiv.org/abs/2601.19499v1) | Mehdi Heydari Shahna, Seyed Adel Alizadeh Kolagar et al. | Reinforcement learning (RL) can be highly effective at learning goal-reaching policies, but it typically does not provide formal guarantees that the goal will always be reached. A common approach to provide formal goal-reaching guarantees is to introduce a shielding mechanism that restricts the agent to actions that satisfy predefined safety constraints. The main challenge here is integrating this mechanism with RL so that learning and exploration remain effective without becoming overly conservative. Hence, this paper proposes an RL-based control framework that provides formal goal-reaching guarantees for wheeled mobile robots operating in unstructured environments. We first design a real-time RL policy with a set of 15 carefully defined reward terms. These rewards encourage the robot to reach both static and dynamic goals while generating sufficiently smooth command signals that comply with predefined safety specifications, which is critical in practice. Second, a Lyapunov-like stabilizer layer is integrated into the benchmark RL framework as a policy supervisor to formally strengthen the goal-reaching control while preserving meaningful exploration of the state action space. The proposed framework is suitable for real-time deployment in challenging environments, as it provides a formal guarantee of convergence to the intended goal states and compensates for uncertainties by generating real-time control signals based on the current state, while respecting real-world motion constraints. The experimental results show that the proposed Lyapunov-like stabilizer consistently improves the benchmark RL policies, boosting the goal-reaching rate from 84.6% to 99.0%, sharply reducing failures, and improving efficiency. |
| 2026-01-27 | [LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment](http://arxiv.org/abs/2601.19487v1) | Haonan Zhang, Dongxia Wang et al. | Safety-aligned LLMs suffer from two failure modes: jailbreak (answering harmful inputs) and over-refusal (declining benign queries). Existing vector steering methods adjust the magnitude of answer vectors, but this creates a fundamental trade-off -- reducing jailbreak increases over-refusal and vice versa. We identify the root cause: LLMs encode the decision to answer (answer vector $v_a$) and the judgment of input safety (benign vector $v_b$) as nearly orthogonal directions, treating them as independent processes. We propose LLM-VA, which aligns $v_a$ with $v_b$ through closed-form weight updates, making the model's willingness to answer causally dependent on its safety assessment -- without fine-tuning or architectural changes. Our method identifies vectors at each layer using SVMs, selects safety-relevant layers, and iteratively aligns vectors via minimum-norm weight modifications. Experiments on 12 LLMs demonstrate that LLM-VA achieves 11.45% higher F1 than the best baseline while preserving 95.92% utility, and automatically adapts to each model's safety bias without manual tuning. Code and models are available at https://hotbento.github.io/LLM-VA-Web/. |
| 2026-01-27 | [Physical Human-Robot Interaction: A Critical Review of Safety Constraints](http://arxiv.org/abs/2601.19462v1) | Riccardo Zanella, Federico Califano et al. | This paper aims to provide a clear and rigorous understanding of commonly recognized safety constraints in physical human-robot interaction, i.e. ISO/TS 15066, by examining how they are obtained and which assumptions support them. We clarify the interpretation and practical impact of key simplifying assumptions, show how these modeling choices affect both safety and performance across the system, and indicate specific design parameters that can be adjusted in safety-critical control implementations. Numerical examples are provided to quantify performance degradation induced by common approximations and simplifying design choices. Furthermore, the fundamental role of energy in safety assessment is emphasized, and focused insights are offered on the existing body of work concerning energy-based safety methodologies. |
| 2026-01-27 | [From Internal Diagnosis to External Auditing: A VLM-Driven Paradigm for Online Test-Time Backdoor Defense](http://arxiv.org/abs/2601.19448v1) | Binyan Xu, Fan Yang et al. | Deep Neural Networks remain inherently vulnerable to backdoor attacks. Traditional test-time defenses largely operate under the paradigm of internal diagnosis methods like model repairing or input robustness, yet these approaches are often fragile under advanced attacks as they remain entangled with the victim model's corrupted parameters. We propose a paradigm shift from Internal Diagnosis to External Semantic Auditing, arguing that effective defense requires decoupling safety from the victim model via an independent, semantically grounded auditor. To this end, we present a framework harnessing Universal Vision-Language Models (VLMs) as evolving semantic gatekeepers. We introduce PRISM (Prototype Refinement & Inspection via Statistical Monitoring), which overcomes the domain gap of general VLMs through two key mechanisms: a Hybrid VLM Teacher that dynamically refines visual prototypes online, and an Adaptive Router powered by statistical margin monitoring to calibrate gating thresholds in real-time. Extensive evaluation across 17 datasets and 11 attack types demonstrates that PRISM achieves state-of-the-art performance, suppressing Attack Success Rate to <1% on CIFAR-10 while improving clean accuracy, establishing a new standard for model-agnostic, externalized security. |
| 2026-01-27 | [eHMI for All -- Investigating the Effect of External Communication of Automated Vehicles on Pedestrians, Manual Drivers, and Cyclists in Virtual Reality](http://arxiv.org/abs/2601.19440v1) | Mark Colley, Simon Kopp et al. | With automated vehicles (AVs), the absence of a human operator could necessitate external Human-Machine Interfaces (eHMIs) to communicate with other road users. Existing research primarily focuses on pedestrian-AV interactions, with limited attention given to other road users, such as cyclists and drivers of manually driven vehicles. So far, no studies have compared the effects of eHMIs across these three road user roles. Therefore, we conducted a within-subjects virtual reality experiment (N=40), evaluating the subjective and objective impact of an eHMI communicating the AV's intention to pedestrians, cyclists, and drivers under various levels of distraction (no distraction, visual noise, interference). eHMIs positively influenced safety perceptions, trust, perceived usefulness, and mental demand across all roles. While distraction and road user roles showed significant main effects, interaction effects were only observed in perceived usability. Thus, a unified eHMI design is effective, facilitating the standardization and broader adoption of eHMIs in diverse traffic. |
| 2026-01-27 | [Ad Insertion in LLM-Generated Responses](http://arxiv.org/abs/2601.19435v1) | Shengwei Xu, Zhaohua Chen et al. | Sustainable monetization of Large Language Models (LLMs) remains a critical open challenge. Traditional search advertising, which relies on static keywords, fails to capture the fleeting, context-dependent user intents--the specific information, goods, or services a user seeks--embedded in conversational flows. Beyond the standard goal of social welfare maximization, effective LLM advertising imposes additional requirements on contextual coherence (ensuring ads align semantically with transient user intents) and computational efficiency (avoiding user interaction latency), as well as adherence to ethical and regulatory standards, including preserving privacy and ensuring explicit ad disclosure. Although various recent solutions have explored bidding on token-level and query-level, both categories of approaches generally fail to holistically satisfy this multifaceted set of constraints.   We propose a practical framework that resolves these tensions through two decoupling strategies. First, we decouple ad insertion from response generation to ensure safety and explicit disclosure. Second, we decouple bidding from specific user queries by using ``genres'' (high-level semantic clusters) as a proxy. This allows advertisers to bid on stable categories rather than sensitive real-time response, reducing computational burden and privacy risks. We demonstrate that applying the VCG auction mechanism to this genre-based framework yields approximately dominant strategy incentive compatibility (DSIC) and individual rationality (IR), as well as approximately optimal social welfare, while maintaining high computational efficiency. Finally, we introduce an "LLM-as-a-Judge" metric to estimate contextual coherence. Our experiments show that this metric correlates strongly with human ratings (Spearman's $ρ\approx 0.66$), outperforming 80% of individual human evaluators. |
| 2026-01-27 | [MIRAGE: Enabling Real-Time Automotive Mediated Reality](http://arxiv.org/abs/2601.19385v1) | Pascal Jansen, Julian Britten et al. | Traffic is inherently dangerous, with around 1.19 million fatalities annually. Automotive Mediated Reality (AMR) can enhance driving safety by overlaying critical information (e.g., outlines, icons, text) on key objects to improve awareness, altering objects' appearance to simplify traffic situations, and diminishing their appearance to minimize distractions. However, real-world AMR evaluation remains limited due to technical challenges. To fill this sim-to-real gap, we present MIRAGE, an open-source tool that enables real-time AMR in real vehicles. MIRAGE implements 15 effects across the AMR spectrum of augmented, diminished, and modified reality using state-of-the-art computational models for object detection and segmentation, depth estimation, and inpainting. In an on-road expert user study (N=9) of MIRAGE, participants enjoyed the AMR experience while pointing out technical limitations and identifying use cases for AMR. We discuss these results in relation to prior work and outline implications for AMR ethics and interaction design. |
| 2026-01-27 | [Self-Supervised Path Planning in Unstructured Environments via Global-Guided Differentiable Hard Constraint Projection](http://arxiv.org/abs/2601.19354v1) | Ziqian Wang, Chenxi Fang et al. | Deploying deep learning agents for autonomous navigation in unstructured environments faces critical challenges regarding safety, data scarcity, and limited computational resources. Traditional solvers often suffer from high latency, while emerging learning-based approaches struggle to ensure deterministic feasibility. To bridge the gap from embodied to embedded intelligence, we propose a self-supervised framework incorporating a differentiable hard constraint projection layer for runtime assurance. To mitigate data scarcity, we construct a Global-Guided Artificial Potential Field (G-APF), which provides dense supervision signals without manual labeling. To enforce actuator limitations and geometric constraints efficiently, we employ an adaptive neural projection layer, which iteratively rectifies the coarse network output onto the feasible manifold. Extensive benchmarks on a test set of 20,000 scenarios demonstrate an 88.75\% success rate, substantiating the enhanced operational safety. Closed-loop experiments in CARLA further validate the physical realizability of the planned paths under dynamic constraints. Furthermore, deployment verification on an NVIDIA Jetson Orin NX confirms an inference latency of 94 ms, showing real-time feasibility on resource-constrained embedded hardware. This framework offers a generalized paradigm for embedding physical laws into neural architectures, providing a viable direction for solving constrained optimization in mechatronics. Source code is available at: https://github.com/wzq-13/SSHC.git. |
| 2026-01-27 | [Eco-Driving Control for Electric Vehicles with Multi-Speed Transmission: Optimizing Vehicle Speed and Powertrain Operation in Dynamic Environments](http://arxiv.org/abs/2601.19340v1) | Suiyi He, Zongxuan Sun | This article presents an eco-driving algorithm for electric vehicles featuring multi-speed transmissions. The proposed controller is formulated as a co-optimization problem, simultaneously optimizing both vehicle longitudinal speed and powertrain operation to maximize energy efficiency. Constraints derived from a connected vehicle based traffic prediction algorithm are used to ensure traffic safety and smooth traffic flow in dynamic environments with multiple signalized intersections and mixed traffic. By simplifying the complex, nonlinear mixed integer problem, the proposed controller achieves computational efficiency, enabling real-time implementation. To evaluate its performance, traffic scenarios from both Simulation of Urban MObility (SUMO) and real-world road tests are employed. The results demonstrate a notable reduction in energy consumption by up to 11.36\% over an \SI{18}{\km} drive. |
| 2026-01-27 | [SETA: Statistical Fault Attribution for Compound AI Systems](http://arxiv.org/abs/2601.19337v1) | Sayak Chowdhury, Meenakshi D'Souza | Modern AI systems increasingly comprise multiple interconnected neural networks to tackle complex inference tasks. Testing such systems for robustness and safety entails significant challenges. Current state-of-the-art robustness testing techniques, whether black-box or white-box, have been proposed and implemented for single-network models and do not scale well to multi-network pipelines. We propose a modular robustness testing framework that applies a given set of perturbations to test data. Our testing framework supports (1) a component-wise system analysis to isolate errors and (2) reasoning about error propagation across the neural network modules. The testing framework is architecture and modality agnostic and can be applied across domains. We apply the framework to a real-world autonomous rail inspection system composed of multiple deep networks and successfully demonstrate how our approach enables fine-grained robustness analysis beyond conventional end-to-end metrics. |

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



