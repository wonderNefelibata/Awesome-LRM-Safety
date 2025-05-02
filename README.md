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
| 2025-05-01 | [Towards Autonomous Micromobility through Scalable Urban Simulation](http://arxiv.org/abs/2505.00690v1) | Wayne Wu, Honglin He et al. | Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations. |
| 2025-05-01 | [Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions](http://arxiv.org/abs/2505.00671v1) | Chenggang Wang, Xinyi Wang et al. | The safety of training task policies and their subsequent application using reinforcement learning (RL) methods has become a focal point in the field of safe RL. A central challenge in this area remains the establishment of theoretical guarantees for safety during both the learning and deployment processes. Given the successful implementation of Control Barrier Function (CBF)-based safety strategies in a range of control-affine robotic systems, CBF-based safe RL demonstrates significant promise for practical applications in real-world scenarios. However, integrating these two approaches presents several challenges. First, embedding safety optimization within the RL training pipeline requires that the optimization outputs be differentiable with respect to the input parameters, a condition commonly referred to as differentiable optimization, which is non-trivial to solve. Second, the differentiable optimization framework confronts significant efficiency issues, especially when dealing with multi-constraint problems. To address these challenges, this paper presents a CBF-based safe RL architecture that effectively mitigates the issues outlined above. The proposed approach constructs a continuous AND logic approximation for the multiple constraints using a single composite CBF. By leveraging this approximation, a close-form solution of the quadratic programming is derived for the policy network in RL, thereby circumventing the need for differentiable optimization within the end-to-end safe RL pipeline. This strategy significantly reduces computational complexity because of the closed-form solution while maintaining safety guarantees. Simulation results demonstrate that, in comparison to existing approaches relying on differentiable optimization, the proposed method significantly reduces training computational costs while ensuring provable safety throughout the training process. |
| 2025-05-01 | [DeepCritic: Deliberate Critique with Large Language Models](http://arxiv.org/abs/2505.00662v1) | Wenkai Yang, Jingwen Chen et al. | As Large Language Models (LLMs) are rapidly evolving, providing accurate feedback and scalable oversight on their outputs becomes an urgent and critical problem. Leveraging LLMs as critique models to achieve automated supervision is a promising solution. In this work, we focus on studying and enhancing the math critique ability of LLMs. Current LLM critics provide critiques that are too shallow and superficial on each step, leading to low judgment accuracy and struggling to offer sufficient feedback for the LLM generator to correct mistakes. To tackle this issue, we propose a novel and effective two-stage framework to develop LLM critics that are capable of deliberately critiquing on each reasoning step of math solutions. In the first stage, we utilize Qwen2.5-72B-Instruct to generate 4.5K long-form critiques as seed data for supervised fine-tuning. Each seed critique consists of deliberate step-wise critiques that includes multi-perspective verifications as well as in-depth critiques of initial critiques for each reasoning step. Then, we perform reinforcement learning on the fine-tuned model with either existing human-labeled data from PRM800K or our automatically annotated data obtained via Monte Carlo sampling-based correctness estimation, to further incentivize its critique ability. Our developed critique model built on Qwen2.5-7B-Instruct not only significantly outperforms existing LLM critics (including the same-sized DeepSeek-R1-distill models and GPT-4o) on various error identification benchmarks, but also more effectively helps the LLM generator refine erroneous steps through more detailed feedback. |
| 2025-05-01 | [Catastrophic Liability: Managing Systemic Risks in Frontier AI Development](http://arxiv.org/abs/2505.00616v1) | Aidan Kierans, Kaley Rittichier et al. | As artificial intelligence systems grow more capable and autonomous, frontier AI development poses potential systemic risks that could affect society at a massive scale. Current practices at many AI labs developing these systems lack sufficient transparency around safety measures, testing procedures, and governance structures. This opacity makes it challenging to verify safety claims or establish appropriate liability when harm occurs. Drawing on liability frameworks from nuclear energy, aviation software, and healthcare, we propose a comprehensive approach to safety documentation and accountability in frontier AI development. |
| 2025-05-01 | [100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models](http://arxiv.org/abs/2505.00551v1) | Chong Zhang, Yue Deng et al. | The recent development of reasoning language models (RLMs) represents a novel evolution in large language models. In particular, the recent release of DeepSeek-R1 has generated widespread social impact and sparked enthusiasm in the research community for exploring the explicit reasoning paradigm of language models. However, the implementation details of the released models have not been fully open-sourced by DeepSeek, including DeepSeek-R1-Zero, DeepSeek-R1, and the distilled small models. As a result, many replication studies have emerged aiming to reproduce the strong performance achieved by DeepSeek-R1, reaching comparable performance through similar training procedures and fully open-source data resources. These works have investigated feasible strategies for supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR), focusing on data preparation and method design, yielding various valuable insights. In this report, we provide a summary of recent replication studies to inspire future research. We primarily focus on SFT and RLVR as two main directions, introducing the details for data construction, method design and training procedure of current replication studies. Moreover, we conclude key findings from the implementation details and experimental results reported by these studies, anticipating to inspire future research. We also discuss additional techniques of enhancing RLMs, highlighting the potential of expanding the application scope of these models, and discussing the challenges in development. By this survey, we aim to help researchers and developers of RLMs stay updated with the latest advancements, and seek to inspire new ideas to further enhance RLMs. |
| 2025-05-01 | [Linear Phase Balancing Scheme using Voltage Unbalance Sensitivities in Multi-phase Power Distribution Grids](http://arxiv.org/abs/2505.00519v1) | Rahul K. Gupta | Power distribution networks, especially in North America, are often unbalanced due to the mix of single-, two- and three-phase networks as well as due to the high penetration of single-phase devices at the distribution level such as electric vehicle (EV) chargers and single-phase solar plants. However, the network operator must adhere to the voltage unbalance levels within the limits specified by IEEE, IEC, and NEMA standards for the safety of the equipment as well as the efficiency of the network operation. Existing works have proposed active and reactive power control in the network to minimize imbalances. However, these optimization problems are highly nonlinear and nonconvex due to the inherent non-linearity of unbalanced metrics and power-flow equations. In this work, we propose a linearization approach of unbalance metrics such as voltage unbalance factors (VUF), phase voltage unbalance rate (PVUR), and line voltage unbalance rate (LVUR) using the first order Taylor's approximation. This linearization is then applied to the phase balancing control scheme; it is formulated as a feedback approach where the linearization is updated successively after the active/reactive control setpoint has been actuated and shows improvement in voltage imbalances. We demonstrate the application of the proposed scheme on a standard IEEE benchmark test case, demonstrating its effectiveness. |
| 2025-05-01 | [Safety-Critical Traffic Simulation with Guided Latent Diffusion Model](http://arxiv.org/abs/2505.00515v1) | Mingxing Peng, Ruoyu Yao et al. | Safety-critical traffic simulation plays a crucial role in evaluating autonomous driving systems under rare and challenging scenarios. However, existing approaches often generate unrealistic scenarios due to insufficient consideration of physical plausibility and suffer from low generation efficiency. To address these limitations, we propose a guided latent diffusion model (LDM) capable of generating physically realistic and adversarial safety-critical traffic scenarios. Specifically, our model employs a graph-based variational autoencoder (VAE) to learn a compact latent space that captures complex multi-agent interactions while improving computational efficiency. Within this latent space, the diffusion model performs the denoising process to produce realistic trajectories. To enable controllable and adversarial scenario generation, we introduce novel guidance objectives that drive the diffusion process toward producing adversarial and behaviorally realistic driving behaviors. Furthermore, we develop a sample selection module based on physical feasibility checks to further enhance the physical plausibility of the generated scenarios. Extensive experiments on the nuScenes dataset demonstrate that our method achieves superior adversarial effectiveness and generation efficiency compared to existing baselines while maintaining a high level of realism. Our work provides an effective tool for realistic safety-critical scenario simulation, paving the way for more robust evaluation of autonomous driving systems. |
| 2025-05-01 | [Variational OOD State Correction for Offline Reinforcement Learning](http://arxiv.org/abs/2505.00503v1) | Ke Jiang, Wen Jiang et al. | The performance of Offline reinforcement learning is significantly impacted by the issue of state distributional shift, and out-of-distribution (OOD) state correction is a popular approach to address this problem. In this paper, we propose a novel method named Density-Aware Safety Perception (DASP) for OOD state correction. Specifically, our method encourages the agent to prioritize actions that lead to outcomes with higher data density, thereby promoting its operation within or the return to in-distribution (safe) regions. To achieve this, we optimize the objective within a variational framework that concurrently considers both the potential outcomes of decision-making and their density, thus providing crucial contextual information for safe decision-making. Finally, we validate the effectiveness and feasibility of our proposed method through extensive experimental evaluations on the offline MuJoCo and AntMaze suites. |
| 2025-05-01 | [A Generalised Framework for Property-Driven Machine Learning](http://arxiv.org/abs/2505.00466v1) | Thomas Flinkow, Marco Casadio et al. | Neural networks have been shown to frequently fail to satisfy critical safety and correctness properties after training, highlighting the pressing need for training methods that incorporate such properties directly. While adversarial training can be used to improve robustness to small perturbations within $\epsilon$-cubes, domains other than computer vision -- such as control systems and natural language processing -- may require more flexible input region specifications via generalised hyper-rectangles. Meanwhile, differentiable logics offer a way to encode arbitrary logical constraints as additional loss terms that guide the learning process towards satisfying these constraints. In this paper, we investigate how these two complementary approaches can be unified within a single framework for property-driven machine learning. We show that well-known properties from the literature are subcases of this general approach, and we demonstrate its practical effectiveness on a case study involving a neural network controller for a drone system. Our framework is publicly available at https://github.com/tflinkow/property-driven-ml. |
| 2025-05-01 | [Toward Automated Regulatory Decision-Making: Trustworthy Medical Device Risk Classification with Multimodal Transformers and Self-Training](http://arxiv.org/abs/2505.00422v1) | Yu Han, Aaron Ceross et al. | Accurate classification of medical device risk levels is essential for regulatory oversight and clinical safety. We present a Transformer-based multimodal framework that integrates textual descriptions and visual information to predict device regulatory classification. The model incorporates a cross-attention mechanism to capture intermodal dependencies and employs a self-training strategy for improved generalization under limited supervision. Experiments on a real-world regulatory dataset demonstrate that our approach achieves up to 90.4% accuracy and 97.9% AUROC, significantly outperforming text-only (77.2%) and image-only (54.8%) baselines. Compared to standard multimodal fusion, the self-training mechanism improved SVM performance by 3.3 percentage points in accuracy (from 87.1% to 90.4%) and 1.4 points in macro-F1, suggesting that pseudo-labeling can effectively enhance generalization under limited supervision. Ablation studies further confirm the complementary benefits of both cross-modal attention and self-training. |
| 2025-05-01 | [Safety in the Face of Adversity: Achieving Zero Constraint Violation in Online Learning with Slowly Changing Constraints](http://arxiv.org/abs/2505.00398v1) | Bassel Hamoud, Ilnura Usmanova et al. | We present the first theoretical guarantees for zero constraint violation in Online Convex Optimization (OCO) across all rounds, addressing dynamic constraint changes. Unlike existing approaches in constrained OCO, which allow for occasional safety breaches, we provide the first approach for maintaining strict safety under the assumption of gradually evolving constraints, namely the constraints change at most by a small amount between consecutive rounds. This is achieved through a primal-dual approach and Online Gradient Ascent in the dual space. We show that employing a dichotomous learning rate enables ensuring both safety, via zero constraint violation, and sublinear regret. Our framework marks a departure from previous work by providing the first provable guarantees for maintaining absolute safety in the face of changing constraints in OCO. |
| 2025-05-01 | [Optimizing Deep Neural Networks using Safety-Guided Self Compression](http://arxiv.org/abs/2505.00350v1) | Mohammad Zbeeb, Mariam Salman et al. | The deployment of deep neural networks on resource-constrained devices necessitates effective model com- pression strategies that judiciously balance the reduction of model size with the preservation of performance. This study introduces a novel safety-driven quantization framework that leverages preservation sets to systematically prune and quantize neural network weights, thereby optimizing model complexity without compromising accuracy. The proposed methodology is rigorously evaluated on both a convolutional neural network (CNN) and an attention-based language model, demonstrating its applicability across diverse architectural paradigms. Experimental results reveal that our framework achieves up to a 2.5% enhancement in test accuracy relative to the original unquantized models while maintaining 60% of the initial model size. In comparison to conventional quantization techniques, our approach not only augments generalization by eliminating parameter noise and retaining essential weights but also reduces variance, thereby ensuring the retention of critical model features. These findings underscore the efficacy of safety-driven quantization as a robust and reliable strategy for the efficient optimization of deep learn- ing models. The implementation and comprehensive experimental evaluations of our framework are publicly accessible at GitHub. |
| 2025-05-01 | [Vehicular Communication Security: Multi-Channel and Multi-Factor Authentication](http://arxiv.org/abs/2505.00340v1) | Marco De Vincenzi, Shuyang Sun et al. | Secure and reliable communications are crucial for Intelligent Transportation Systems (ITSs), where Vehicle-to-Infrastructure (V2I) communication plays a key role in enabling mobility-enhancing and safety-critical services. Current V2I authentication relies on credential-based methods over wireless Non-Line-of-Sight (NLOS) channels, leaving them exposed to remote impersonation and proximity attacks. To mitigate these risks, we propose a unified Multi-Channel, Multi-Factor Authentication (MFA) scheme that combines NLOS cryptographic credentials with a Line-of-Sight (LOS) visual channel. Our approach leverages a challenge-response security paradigm: the infrastructure issues challenges and the vehicle's headlights respond by flashing a structured sequence containing encoded security data. Deep learning models on the infrastructure side then decode the embedded information to authenticate the vehicle. Real-world experimental evaluations demonstrate high test accuracy, reaching an average of 95% and 96.6%, respectively, under various lighting, weather, speed, and distance conditions. Additionally, we conducted extensive experiments on three state-of-the-art deep learning models, including detailed ablation studies for decoding the flashing sequence. Our results indicate that the optimal architecture employs a dual-channel design, enabling simultaneous decoding of the flashing sequence and extraction of vehicle spatial and locational features for robust authentication. |
| 2025-05-01 | [AI2-Active Safety: AI-enabled Interaction-aware Active Safety Analysis with Vehicle Dynamics](http://arxiv.org/abs/2505.00322v1) | Keshu Wu, Zihao Li et al. | This paper introduces an AI-enabled, interaction-aware active safety analysis framework that accounts for groupwise vehicle interactions. Specifically, the framework employs a bicycle model-augmented with road gradient considerations-to accurately capture vehicle dynamics. In parallel, a hypergraph-based AI model is developed to predict probabilistic trajectories of ambient traffic. By integrating these two components, the framework derives vehicle intra-spacing over a 3D road surface as the solution of a stochastic ordinary differential equation, yielding high-fidelity surrogate safety measures such as time-to-collision (TTC). To demonstrate its effectiveness, the framework is analyzed using stochastic numerical methods comprising 4th-order Runge-Kutta integration and AI inference, generating probability-weighted high-fidelity TTC (HF-TTC) distributions that reflect complex multi-agent maneuvers and behavioral uncertainties. Evaluated with HF-TTC against traditional constant-velocity TTC and non-interaction-aware approaches on highway datasets, the proposed framework offers a systematic methodology for active safety analysis with enhanced potential for improving safety perception in complex traffic environments. |
| 2025-05-01 | [Beyond Quadratic Costs: A Bregman Divergence Approach to H$_\infty$ Control](http://arxiv.org/abs/2505.00319v1) | Joudi Hajar, Reza Ghane et al. | This paper presents a novel extension of the H$_\infty$ control framework that generalizes the traditional quadratic cost formulation to accommodate strictly convex, nonquadratic functions for the state, control input, and disturbance. This new formulation not only captures additional noise characteristics but also supports a range of performance objectives-including sparse control, safety constraints, and other tailored behaviors-beyond what is possible with quadratic costs. We derive a closed-form solution of a central controller that minimizes the worst-case performance ratio under the proposed cost structure. Furthermore, we develop Riccati-like equations that impose necessary and sufficient conditions on the nonquadratic cost functions, thereby ensuring the existence of a robust solution. Finally, we rigorously establish Lyapunov stability for the closed-loop system. The proposed framework bridges robust control theory with modern approaches in machine learning and signal processing, offering enhanced flexibility and improved performance in complex control scenarios. |
| 2025-05-01 | [Beyond Quadratic Costs in LQR: Bregman Divergence Control](http://arxiv.org/abs/2505.00317v1) | Babak Hassibi, Joudi Hajar et al. | In the past couple of decades, the use of ``non-quadratic" convex cost functions has revolutionized signal processing, machine learning, and statistics, allowing one to customize solutions to have desired structures and properties. However, the situation is not the same in control where the use of quadratic costs still dominates, ostensibly because determining the ``value function", i.e., the optimal expected cost-to-go, which is critical to the construction of the optimal controller, becomes computationally intractable as soon as one considers general convex costs. As a result, practitioners often resort to heuristics and approximations, such as model predictive control that only looks a few steps into the future. In the quadratic case, the value function is easily determined by solving Riccati equations. In this work, we consider a special class of convex cost functions constructed from Bregman divergence and show how, with appropriate choices, they can be used to fully extend the framework developed for the quadratic case. The resulting optimal controllers are infinite horizon, come with stability guarantees, and have state-feedback, or estimated state-feedback, laws. They exhibit a much wider range of behavior than their quadratic counterparts since the feedback laws are nonlinear. The approach can be applied to several cases of interest, including safety control, sparse control, and bang-bang control. |
| 2025-05-01 | [J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators](http://arxiv.org/abs/2505.00306v1) | Shivani Guptasarma, Matthew Strong et al. | J-PARSE is a method for smooth first-order inverse kinematic control of a serial manipulator near kinematic singularities. The commanded end-effector velocity is interpreted component-wise, according to the available mobility in each dimension of the task space. First, a substitute "Safety" Jacobian matrix is created, keeping the aspect ratio of the manipulability ellipsoid above a threshold value. The desired motion is then projected onto non-singular and singular directions, and the latter projection scaled down by a factor informed by the threshold value. A right-inverse of the non-singular Safety Jacobian is applied to the modified command. In the absence of joint limits and collisions, this ensures smooth transition into and out of low-rank poses, guaranteeing asymptotic stability for target poses within the workspace, and stability for those outside. Velocity control with J-PARSE is benchmarked against the Least-Squares and Damped Least-Squares inversions of the Jacobian, and shows high accuracy in reaching and leaving singular target poses. By expanding the available workspace of manipulators, the method finds applications in servoing, teleoperation, and learning. |
| 2025-04-30 | [PSN Game: Game-theoretic Planning via a Player Selection Network](http://arxiv.org/abs/2505.00213v1) | Tianyu Qiu, Eric Ouano et al. | While game-theoretic planning frameworks are effective at modeling multi-agent interactions, they require solving optimization problems with hundreds or thousands of variables, resulting in long computation times that limit their use in large-scale, real-time systems. To address this issue, we propose PSN Game: a novel game-theoretic planning framework that reduces runtime by learning a Player Selection Network (PSN). A PSN outputs a player selection mask that distinguishes influential players from less relevant ones, enabling the ego player to solve a smaller, masked game involving only selected players. By reducing the number of variables in the optimization problem, PSN directly lowers computation time. The PSN Game framework is more flexible than existing player selection methods as it i) relies solely on observations of players' past trajectories, without requiring full state, control, or other game-specific information; and ii) requires no online parameter tuning. We train PSNs in an unsupervised manner using a differentiable dynamic game solver, with reference trajectories from full-player games guiding the learning. Experiments in both simulated scenarios and human trajectory datasets demonstrate that i) PSNs outperform baseline selection methods in trajectory smoothness and length, while maintaining comparable safety and achieving a 10x speedup in runtime; and ii) PSNs generalize effectively to real-world scenarios without fine-tuning. By selecting only the most relevant players for decision-making, PSNs offer a general mechanism for reducing planning complexity that can be seamlessly integrated into existing multi-agent planning frameworks. |
| 2025-04-30 | [Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems](http://arxiv.org/abs/2505.00212v1) | Shaokun Zhang, Ming Yin et al. | Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at https://github.com/mingyin1/Agents_Failure_Attribution |
| 2025-04-30 | [Real-World Gaps in AI Governance Research](http://arxiv.org/abs/2505.00174v1) | Ilan Strauss, Isobel Moure et al. | Drawing on 1,178 safety and reliability papers from 9,439 generative AI papers (January 2020 - March 2025), we compare research outputs of leading AI companies (Anthropic, Google DeepMind, Meta, Microsoft, and OpenAI) and AI universities (CMU, MIT, NYU, Stanford, UC Berkeley, and University of Washington). We find that corporate AI research increasingly concentrates on pre-deployment areas -- model alignment and testing & evaluation -- while attention to deployment-stage issues such as model bias has waned. Significant research gaps exist in high-risk deployment domains, including healthcare, finance, misinformation, persuasive and addictive features, hallucinations, and copyright. Without improved observability into deployed AI, growing corporate concentration could deepen knowledge deficits. We recommend expanding external researcher access to deployment data and systematic observability of in-market AI behaviors. |

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



