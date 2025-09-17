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
| 2025-09-16 | [HARMONIC: A Content-Centric Cognitive Robotic Architecture](http://arxiv.org/abs/2509.13279v1) | Sanjay Oruganti, Sergei Nirenburg et al. | This paper introduces HARMONIC, a cognitive-robotic architecture designed for robots in human-robotic teams. HARMONIC supports semantic perception interpretation, human-like decision-making, and intentional language communication. It addresses the issues of safety and quality of results; aims to solve problems of data scarcity, explainability, and safety; and promotes transparency and trust. Two proof-of-concept HARMONIC-based robotic systems are demonstrated, each implemented in both a high-fidelity simulation environment and on physical robotic platforms. |
| 2025-09-16 | [Safety Critical Model Predictive Control Using Discrete-Time Control Density Functions](http://arxiv.org/abs/2509.13257v1) | Sriram S. K. S. Narayanan, Sajad Ahmadi et al. | This paper presents MPC-CDF, a new approach integrating control density functions (CDFs) within a model predictive control (MPC) framework to ensure safety-critical control in nonlinear dynamical systems. By using the dual formulation of the navigation problem, we incorporate CDFs into the MPC framework, ensuring both convergence and safety in a discrete-time setting. These density functions are endowed with a physical interpretation, where the associated measure signifies the occupancy of system trajectories. Leveraging this occupancy-based perspective, we synthesize safety-critical controllers using the proposed MPC-CDF framework. We illustrate the safety properties of this framework using a unicycle model and compare it with a control barrier function-based method. The efficacy of this approach is demonstrated in the autonomous safe navigation of an underwater vehicle, which avoids complex and arbitrary obstacles while achieving the desired level of safety. |
| 2025-09-16 | [Investigating the Performance of EKF, UKF, and PF for Quadrotor Position Estimation in Hurricane Wind Disturbances](http://arxiv.org/abs/2509.13243v1) | Ahmed A. Elgohary, Benjamin Gwinnell et al. | Natural disasters, such as hurricanes and typhoons, pose significant challenges to public safety and infrastructure. While government agencies rely on multi million dollar UAV systems for storm data collection and disaster response, smaller drones lack the ability to autonomously adapt to rapidly changing environmental conditions, such as turbulent winds. This project investigates the implementation of advanced state estimation filters, Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filter (PF), to enhance the control and adaptability of quadrotor drones in uncertain wind profiles modeled by Von Karman turbulence. While the EKF relies on linearization techniques using the Taylor series expansion, which can struggle under high nonlinearity, the UKF leverages sigma points for better performance in nonlinear systems without requiring Jacobian computations. The PF, on the other hand, addresses non-Gaussian noise and severe nonlinearities by employing a large number of particles, albeit at a high computational cost. To enhance accuracy and minimize estimation errors, a genetic algorithm (GA) was employed to optimally tune the process and measurement noise covariances (Q and R matrices) as well as UKF specific parameters alpha, beta, and kappa. This study highlights the tradeoff between accuracy, computational efficiency, and smoothing capabilities across these filters. Despite its robustness, the PF suffered from computational inefficiencies due to the high state dimensionality, while the EKF demonstrated faster computation but lower adaptability in nonlinear conditions. The UKF emerged as a balanced approach, achieving superior performance in capturing dynamic wind disturbances. |
| 2025-09-16 | [Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance](http://arxiv.org/abs/2509.13210v1) | Ligang Chang, Shengkai Xu et al. | Violence detection in public surveillance is critical for public safety. This study addresses challenges such as small-scale targets, complex environments, and real-time temporal analysis. We propose Vi-SAFE, a spatial-temporal framework that integrates an enhanced YOLOv8 with a Temporal Segment Network (TSN) for video surveillance. The YOLOv8 model is optimized with GhostNetV3 as a lightweight backbone, an exponential moving average (EMA) attention mechanism, and pruning to reduce computational cost while maintaining accuracy. YOLOv8 and TSN are trained separately on pedestrian and violence datasets, where YOLOv8 extracts human regions and TSN performs binary classification of violent behavior. Experiments on the RWF-2000 dataset show that Vi-SAFE achieves an accuracy of 0.88, surpassing TSN alone (0.77) and outperforming existing methods in both accuracy and efficiency, demonstrating its effectiveness for public safety surveillance. Code is available at https://anonymous.4open.science/r/Vi-SAFE-3B42/README.md. |
| 2025-09-16 | [ROOM: A Physics-Based Continuum Robot Simulator for Photorealistic Medical Datasets Generation](http://arxiv.org/abs/2509.13177v1) | Salvatore Esposito, Matías Mattamala et al. | Continuum robots are advancing bronchoscopy procedures by accessing complex lung airways and enabling targeted interventions. However, their development is limited by the lack of realistic training and test environments: Real data is difficult to collect due to ethical constraints and patient safety concerns, and developing autonomy algorithms requires realistic imaging and physical feedback. We present ROOM (Realistic Optical Observation in Medicine), a comprehensive simulation framework designed for generating photorealistic bronchoscopy training data. By leveraging patient CT scans, our pipeline renders multi-modal sensor data including RGB images with realistic noise and light specularities, metric depth maps, surface normals, optical flow and point clouds at medically relevant scales. We validate the data generated by ROOM in two canonical tasks for medical robotics -- multi-view pose estimation and monocular depth estimation, demonstrating diverse challenges that state-of-the-art methods must overcome to transfer to these medical settings. Furthermore, we show that the data produced by ROOM can be used to fine-tune existing depth estimation models to overcome these challenges, also enabling other downstream applications such as navigation. We expect that ROOM will enable large-scale data generation across diverse patient anatomies and procedural scenarios that are challenging to capture in clinical settings. Code and data: https://github.com/iamsalvatore/room. |
| 2025-09-16 | [TeraSim-World: Worldwide Safety-Critical Data Synthesis for End-to-End Autonomous Driving](http://arxiv.org/abs/2509.13164v1) | Jiawei Wang, Haowei Sun et al. | Safe and scalable deployment of end-to-end (E2E) autonomous driving requires extensive and diverse data, particularly safety-critical events. Existing data are mostly generated from simulators with a significant sim-to-real gap or collected from on-road testing that is costly and unsafe. This paper presents TeraSim-World, an automated pipeline that synthesizes realistic and geographically diverse safety-critical data for E2E autonomous driving at anywhere in the world. Starting from an arbitrary location, TeraSim-World retrieves real-world maps and traffic demand from geospatial data sources. Then, it simulates agent behaviors from naturalistic driving datasets, and orchestrates diverse adversities to create corner cases. Informed by street views of the same location, it achieves photorealistic, geographically grounded sensor rendering via the frontier video generation model Cosmos-Drive. By bridging agent and sensor simulations, TeraSim-World provides a scalable and critical~data synthesis framework for training and evaluation of E2E autonomous driving systems. |
| 2025-09-16 | [Space-Time Trade-off in Bounded Iterated Memory](http://arxiv.org/abs/2509.13157v1) | Guillermo Toyos-Marfurt, Petr Kuznetsov | The celebrated asynchronous computability theorem (ACT) characterizes tasks solvable in the read-write shared-memory model using the unbounded full-information protocol, where in every round of computation, each process shares its complete knowledge of the system with the other processes. Therefore, ACT assumes shared-memory variables of unbounded capacity. It has been recently shown that boundedvariables can achieve the same computational power at the expense of extra rounds. However, the exact relationship between the bit capacity of the shared memory and the number of rounds required in order to implement one round of the full-information protocol remained unknown.   In this paper, we focus on the asymptotic round complexity of bounded iterated shared-memory algorithms that simulate, up to isomorphism, the unbounded full-information protocol. We relate the round complexity to the number of processes $n$, the number of iterations of the full information protocol $r$, and the bit size per shared-memory entry $b$. By analyzing the corresponding protocol complex, a combinatorial structure representing reachable states, we derive necessary conditions and present a bounded full-information algorithm tailored to the bits available $b$ per shared memory entry. We show that for $n>2$, the round complexity required to implement the full-information protocol satisfies $\Omega((n!)^{r-1} \cdot 2^{n-b})$. Our results apply to a range of iterated shared-memory models, from regular read-write registers to atomic and immediate snapshots. Moreover, our bounded full-information algorithm is asymptotically optimal for the iterated collect model and within a linear factor $n$ of optimal for the snapshot-based models. |
| 2025-09-16 | [An Uncertainty-Weighted Decision Transformer for Navigation in Dense, Complex Driving Scenarios](http://arxiv.org/abs/2509.13132v1) | Zhihao Zhang, Chengyang Peng et al. | Autonomous driving in dense, dynamic environments requires decision-making systems that can exploit both spatial structure and long-horizon temporal dependencies while remaining robust to uncertainty. This work presents a novel framework that integrates multi-channel bird's-eye-view occupancy grids with transformer-based sequence modeling for tactical driving in complex roundabout scenarios. To address the imbalance between frequent low-risk states and rare safety-critical decisions, we propose the Uncertainty-Weighted Decision Transformer (UWDT). UWDT employs a frozen teacher transformer to estimate per-token predictive entropy, which is then used as a weight in the student model's loss function. This mechanism amplifies learning from uncertain, high-impact states while maintaining stability across common low-risk transitions. Experiments in a roundabout simulator, across varying traffic densities, show that UWDT consistently outperforms other baselines in terms of reward, collision rate, and behavioral stability. The results demonstrate that uncertainty-aware, spatial-temporal transformers can deliver safer and more efficient decision-making for autonomous driving in complex traffic environments. |
| 2025-09-16 | [Model Predictive Control with Reference Learning for Soft Robotic Intracranial Pressure Waveform Modulation](http://arxiv.org/abs/2509.13109v1) | Fabian Flürenbrock, Yanick Büchel et al. | This paper introduces a learning-based control framework for a soft robotic actuator system designed to modulate intracranial pressure (ICP) waveforms, which is essential for studying cerebrospinal fluid dynamics and pathological processes underlying neurological disorders. A two-layer framework is proposed to safely achieve a desired ICP waveform modulation. First, a model predictive controller (MPC) with a disturbance observer is used for offset-free tracking of the system's motor position reference trajectory under safety constraints. Second, to address the unknown nonlinear dependence of ICP on the motor position, we employ a Bayesian optimization (BO) algorithm used for online learning of a motor position reference trajectory that yields the desired ICP modulation. The framework is experimentally validated using a test bench with a brain phantom that replicates realistic ICP dynamics in vitro. Compared to a previously employed proportional-integral-derivative controller, the MPC reduces mean and maximum motor position reference tracking errors by 83 % and 73 %, respectively. In less than 20 iterations, the BO algorithm learns a motor position reference trajectory that yields an ICP waveform with the desired mean and amplitude. |
| 2025-09-16 | [When Inverse Data Outperforms: Exploring the Pitfalls of Mixed Data in Multi-Stage Fine-Tuning](http://arxiv.org/abs/2509.13079v1) | Mengyi Deng, Xin Li et al. | Existing work has shown that o1-level performance can be achieved with limited data distillation, but most existing methods focus on unidirectional supervised fine-tuning (SFT), overlooking the intricate interplay between diverse reasoning patterns. In this paper, we construct r1k, a high-quality reverse reasoning dataset derived by inverting 1,000 forward examples from s1k, and examine how SFT and Direct Preference Optimization (DPO) affect alignment under bidirectional reasoning objectives. SFT on r1k yields a 1.6%--6.8% accuracy improvement over s1k across evaluated benchmarks. However, naively mixing forward and reverse data during SFT weakens the directional distinction. Although DPO can partially recover this distinction, it also suppresses less preferred reasoning paths by shifting the probability mass toward irrelevant outputs. These findings suggest that mixed reasoning data introduce conflicting supervision signals, underscoring the need for robust and direction-aware alignment strategies. |
| 2025-09-16 | [Digital Sovereignty Control Framework for Military AI-based Cyber Security](http://arxiv.org/abs/2509.13072v1) | Clara Maathuis, Kasper Cools | In today's evolving threat landscape, ensuring digital sovereignty has become mandatory for military organizations, especially given their increased development and investment in AI-driven cyber security solutions. To this end, a multi-angled framework is proposed in this article in order to define and assess digital sovereign control of data and AI-based models for military cyber security. This framework focuses on aspects such as context, autonomy, stakeholder involvement, and mitigation of risks in this domain. Grounded on the concepts of digital sovereignty and data sovereignty, the framework aims to protect sensitive defence assets against threats such as unauthorized access, ransomware, and supply-chain attacks. This approach reflects the multifaceted nature of digital sovereignty by preserving operational autonomy, assuring security and safety, securing privacy, and fostering ethical compliance of both military systems and decision-makers. At the same time, the framework addresses interoperability challenges among allied forces, strategic and legal considerations, and the integration of emerging technologies by considering a multidisciplinary approach that enhances the resilience and preservation of control over (critical) digital assets. This is done by adopting a design oriented research where systematic literature review is merged with critical thinking and analysis of field incidents in order to assure the effectivity and realism of the framework proposed. |
| 2025-09-16 | [A Dantzig-Wolfe Reformulation for Automated Aircraft Arrival Routing and Scheduling](http://arxiv.org/abs/2509.13065v1) | Roghayeh Hajizadeh, Tatiana Polishchuk et al. | We consider the problem of computing aircraft arrival routes in a terminal maneuvering area (TMA) together with an automated scheduling of all the arrivals within a given time interval. The arrival routes are modeled as energy-efficient continuous-descent operations, such that separation based on wake-turbulence categories is guaranteed within the TMA. We propose a new model based on a Dantzig-Wolfe reformulation of a previous model for this problem. As in the previous model, we include tree consistency across consecutive planning intervals. However, the reformulation enables us to further improve the model and also consider aircraft that remain in the TMA from the previous period, a feature critical for operational safety. In computational experiments for Stockholm Arlanda airport, the new model consistently outperforms the previous one: we obtain solutions within 5 seconds to 12.65 minutes compared to 40.9 hours with the old model for instances of half hours with high traffic. In addition, we are able to solve instances of a full hour of arriving aircraft with high traffic (33 aircraft) within 22.22 to 58.57 minutes, whereas the old model could not solve these instances at all. While we schedule all aircraft as continuous-descent arrivals, our model can be applied to any type of speed profiles for the arriving aircraft. |
| 2025-09-16 | [Multi-Model Synthetic Training for Mission-Critical Small Language Models](http://arxiv.org/abs/2509.13047v1) | Nolan Platt, Pragyansmita Nayak | Large Language Models (LLMs) have demonstrated remarkable capabilities across many domains, yet their application to specialized fields remains constrained by the scarcity and complexity of domain-specific training data. We present a novel approach that achieves a 261x cost reduction for maritime intelligence by using LLMs as one-time teachers rather than using them directly for inference. Our method transforms 3.2 billion Automatic Identification System (AIS) vessel tracking records into 21,543 synthetic question and answer pairs through multi-model generation (GPT-4o and o3-mini), preventing overfitting and ensuring accurate reasoning. The resulting fine-tuned Qwen2.5-7B model achieves 75% accuracy on maritime tasks, while being substantially cheaper than using a larger model for inference. We show that smaller, cheaper models -- when fine tuned properly -- can provide similar accuracy compared to larger models that are prohibitively expensive. Our work contributes to the growing field of synthetic dataset generation for specialized AI applications and presents a highly reproducible framework for domains where manual annotation is infeasible. Beyond expanding research in the growing field of specialized small language models, our approach has immediate applications in maritime safety, security operations, and vessel traffic management systems in various industries. |
| 2025-09-16 | [ReTrack: Data Unlearning in Diffusion Models through Redirecting the Denoising Trajectory](http://arxiv.org/abs/2509.13007v1) | Qitan Shi, Cheng Jin et al. | Diffusion models excel at generating high-quality, diverse images but suffer from training data memorization, raising critical privacy and safety concerns. Data unlearning has emerged to mitigate this issue by removing the influence of specific data without retraining from scratch. We propose ReTrack, a fast and effective data unlearning method for diffusion models. ReTrack employs importance sampling to construct a more efficient fine-tuning loss, which we approximate by retaining only dominant terms. This yields an interpretable objective that redirects denoising trajectories toward the $k$-nearest neighbors, enabling efficient unlearning while preserving generative quality. Experiments on MNIST T-Shirt, CelebA-HQ, CIFAR-10, and Stable Diffusion show that ReTrack achieves state-of-the-art performance, striking the best trade-off between unlearning strength and generation quality preservation. |
| 2025-09-16 | [Momentum-Based Access and Speed Control for Improved Safety in Heterogeneous Road Networks](http://arxiv.org/abs/2509.12944v1) | Felix Wieberneit, Emanuele Crisostomi et al. | The increasing variety of means of transportation, including light vehicles like e-scooters and e-bikes, together with the increasing weight of conventional vehicles due to electrification and consumer preferences for SUVs, are raising serious concerns regarding the safety of road networks. In this paper we design a two-level control algorithm to improve the safety of heterogeneous networks: first, an access control strategy decreases the heterogeneity of the network depending on actual traffic conditions; then, a speed control strategy mitigates the probability of serious injuries in potential collisions. Both control strategies are designed based on momentum considerations, as this is regarded as the most influential variable to assess injury risk. The road network mobility simulator SUMO is adopted to implement and validate our proposed control strategies. |
| 2025-09-16 | [Jailbreaking Large Language Models Through Content Concretization](http://arxiv.org/abs/2509.12937v1) | Johan Wahréus, Ahmed Hussain et al. | Large Language Models (LLMs) are increasingly deployed for task automation and content generation, yet their safety mechanisms remain vulnerable to circumvention through different jailbreaking techniques. In this paper, we introduce \textit{Content Concretization} (CC), a novel jailbreaking technique that iteratively transforms abstract malicious requests into concrete, executable implementations. CC is a two-stage process: first, generating initial LLM responses using lower-tier, less constrained safety filters models, then refining them through higher-tier models that process both the preliminary output and original prompt. We evaluate our technique using 350 cybersecurity-specific prompts, demonstrating substantial improvements in jailbreak Success Rates (SRs), increasing from 7\% (no refinements) to 62\% after three refinement iterations, while maintaining a cost of 7.5\textcent~per prompt. Comparative A/B testing across nine different LLM evaluators confirms that outputs from additional refinement steps are consistently rated as more malicious and technically superior. Moreover, manual code analysis reveals that generated outputs execute with minimal modification, although optimal deployment typically requires target-specific fine-tuning. With eventual improved harmful code generation, these results highlight critical vulnerabilities in current LLM safety frameworks. |
| 2025-09-16 | [Rethinking the Evaluation of Alignment Methods: Insights into Diversity, Generalisation, and Safety](http://arxiv.org/abs/2509.12936v1) | Denis Janiak, Julia Moska et al. | Large language models (LLMs) require careful alignment to balance competing objectives - factuality, safety, conciseness, proactivity, and diversity. Existing studies focus on individual techniques or specific dimensions, lacking a holistic assessment of the inherent trade-offs. We propose a unified evaluation framework that compares LLM alignment methods (PPO, DPO, ORPO, KTO) across these five axes, using both in-distribution and out-of-distribution datasets. Leveraging a specialized LLM-as-Judge prompt, validated through human studies, we reveal that DPO and KTO excel in factual accuracy, PPO and DPO lead in safety, and PPO best balances conciseness with proactivity. Our findings provide insights into trade-offs of common alignment methods, guiding the development of more balanced and reliable LLMs. |
| 2025-09-16 | [The Anatomy of Alignment: Decomposing Preference Optimization by Steering Sparse Features](http://arxiv.org/abs/2509.12934v1) | Jeremias Ferrao, Matthijs van der Lende et al. | Aligning large language models is critical for their usability and safety. However, the prevailing approach of Reinforcement Learning from Human Feedback (RLHF) induces diffuse, opaque parameter changes, making it difficult to discern what the model has internalized. Hence, we introduce Feature Steering with Reinforcement Learning (FSRL), a transparent alignment framework that trains a lightweight adapter to steer behavior by modulating interpretable features from a Sparse Autoencoder (SAE). First, we demonstrate that FSRL is an effective method for preference optimization and is comparable with current RLHF methods. We then perform mechanistic analysis on the trained adapter, and find that its policy systematically promotes style features over explicit alignment concepts, suggesting that the preference optimization process rewards stylistic presentation as a proxy for quality. Ultimately, we hope that FSRL provides a tool for both interpretable model control and diagnosing the internal mechanisms of alignment. |
| 2025-09-16 | [Spotting the Unfriendly Robot -- Towards better Metrics for Interactions](http://arxiv.org/abs/2509.12912v1) | Raphael Wenzel, Malte Probst | Establishing standardized metrics for Social Robot Navigation (SRN) algorithms for assessing the quality and social compliance of robot behavior around humans is essential for SRN research. Currently, commonly used evaluation metrics lack the ability to quantify how cooperative an agent behaves in interaction with humans. Concretely, in a simple frontal approach scenario, no metric specifically captures if both agents cooperate or if one agent stays on collision course and the other agent is forced to evade. To address this limitation, we propose two new metrics, a conflict intensity metric and the responsibility metric. Together, these metrics are capable of evaluating the quality of human-robot interactions by showing how much a given algorithm has contributed to reducing a conflict and which agent actually took responsibility of the resolution. This work aims to contribute to the development of a comprehensive and standardized evaluation methodology for SRN, ultimately enhancing the safety, efficiency, and social acceptance of robots in human-centric environments. |
| 2025-09-16 | [Safe Reinforcement Learning using Action Projection: Safeguard the Policy or the Environment?](http://arxiv.org/abs/2509.12833v1) | Hannah Markgraf, Shamburaj Sawant et al. | Projection-based safety filters, which modify unsafe actions by mapping them to the closest safe alternative, are widely used to enforce safety constraints in reinforcement learning (RL). Two integration strategies are commonly considered: Safe environment RL (SE-RL), where the safeguard is treated as part of the environment, and safe policy RL (SP-RL), where it is embedded within the policy through differentiable optimization layers. Despite their practical relevance in safety-critical settings, a formal understanding of their differences is lacking. In this work, we present a theoretical comparison of SE-RL and SP-RL. We identify a key distinction in how each approach is affected by action aliasing, a phenomenon in which multiple unsafe actions are projected to the same safe action, causing information loss in the policy gradients. In SE-RL, this effect is implicitly approximated by the critic, while in SP-RL, it manifests directly as rank-deficient Jacobians during backpropagation through the safeguard. Our contributions are threefold: (i) a unified formalization of SE-RL and SP-RL in the context of actor-critic algorithms, (ii) a theoretical analysis of their respective policy gradient estimates, highlighting the role of action aliasing, and (iii) a comparative study of mitigation strategies, including a novel penalty-based improvement for SP-RL that aligns with established SE-RL practices. Empirical results support our theoretical predictions, showing that action aliasing is more detrimental for SP-RL than for SE-RL. However, with appropriate improvement strategies, SP-RL can match or outperform improved SE-RL across a range of environments. These findings provide actionable insights for choosing and refining projection-based safe RL methods based on task characteristics. |

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



