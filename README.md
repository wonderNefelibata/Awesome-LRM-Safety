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
| 2025-04-24 | [Integrating Learning-Based Manipulation and Physics-Based Locomotion for Whole-Body Badminton Robot Control](http://arxiv.org/abs/2504.17771v1) | Haochen Wang, Zhiwei Shi et al. | Learning-based methods, such as imitation learning (IL) and reinforcement learning (RL), can produce excel control policies over challenging agile robot tasks, such as sports robot. However, no existing work has harmonized learning-based policy with model-based methods to reduce training complexity and ensure the safety and stability for agile badminton robot control. In this paper, we introduce \ourmethod, a novel hybrid control system for agile badminton robots. Specifically, we propose a model-based strategy for chassis locomotion which provides a base for arm policy. We introduce a physics-informed ``IL+RL'' training framework for learning-based arm policy. In this train framework, a model-based strategy with privileged information is used to guide arm policy training during both IL and RL phases. In addition, we train the critic model during IL phase to alleviate the performance drop issue when transitioning from IL to RL. We present results on our self-engineered badminton robot, achieving 94.5% success rate against the serving machine and 90.7% success rate against human players. Our system can be easily generalized to other agile mobile manipulation tasks such as agile catching and table tennis. Our project website: https://dreamstarring.github.io/HAMLET/. |
| 2025-04-24 | [Conformal Segmentation in Industrial Surface Defect Detection with Statistical Guarantees](http://arxiv.org/abs/2504.17721v1) | Cheng Shen, Yuewei Liu | In industrial settings, surface defects on steel can significantly compromise its service life and elevate potential safety risks. Traditional defect detection methods predominantly rely on manual inspection, which suffers from low efficiency and high costs. Although automated defect detection approaches based on Convolutional Neural Networks(e.g., Mask R-CNN) have advanced rapidly, their reliability remains challenged due to data annotation uncertainties during deep model training and overfitting issues. These limitations may lead to detection deviations when processing the given new test samples, rendering automated detection processes unreliable. To address this challenge, we first evaluate the detection model's practical performance through calibration data that satisfies the independent and identically distributed (i.i.d) condition with test data. Specifically, we define a loss function for each calibration sample to quantify detection error rates, such as the complement of recall rate and false discovery rate. Subsequently, we derive a statistically rigorous threshold based on a user-defined risk level to identify high-probability defective pixels in test images, thereby constructing prediction sets (e.g., defect regions). This methodology ensures that the expected error rate (mean error rate) on the test set remains strictly bounced by the predefined risk level. Additionally, we observe a negative correlation between the average prediction set size and the risk level on the test set, establishing a statistically rigorous metric for assessing detection model uncertainty. Furthermore, our study demonstrates robust and efficient control over the expected test set error rate across varying calibration-to-test partitioning ratios, validating the method's adaptability and operational effectiveness. |
| 2025-04-24 | [Safety in Large Reasoning Models: A Survey](http://arxiv.org/abs/2504.17704v1) | Cheng Wang, Yue Liu et al. | Large Reasoning Models (LRMs) have exhibited extraordinary prowess in tasks like mathematics and coding, leveraging their advanced reasoning capabilities. Nevertheless, as these capabilities progress, significant concerns regarding their vulnerabilities and safety have arisen, which can pose challenges to their deployment and application in real-world settings. This paper presents a comprehensive survey of LRMs, meticulously exploring and summarizing the newly emerged safety risks, attacks, and defense strategies. By organizing these elements into a detailed taxonomy, this work aims to offer a clear and structured understanding of the current safety landscape of LRMs, facilitating future research and development to enhance the security and reliability of these powerful models. |
| 2025-04-24 | [Using mathematical models of heart cells to assess the safety of new pharmaceutical drugs](http://arxiv.org/abs/2504.17694v1) | Gary R. Mirams | Many drugs have been withdrawn from the market worldwide, at a cost of billions of dollars, because of patient fatalities due to them unexpectedly disturbing heart rhythm. Even drugs for ailments as mild as hay fever have been withdrawn due to an unacceptable increase in risk of these heart rhythm disturbances. Consequently, the whole pharmaceutical industry expends a huge effort in checking all new drugs for any unwanted side effects on the heart. The predominant root cause has been identified as drug molecules blocking ionic current flows in the heart. Block of individual types of ionic currents can now be measured experimentally at an early stage of drug development, and this is the standard screening approach for a number of ion currents in many large pharmaceutical companies. However, clinical risk is a complex function of the degree of block of many different types of cardiac ion currents, and this is difficult to understand by looking at results of these screens independently. By using ordinary differential equation models for the electrical activity of heart cells (electrophysiology models) we can integrate information from different types of currents, to predict the effect on whole heart cells and subsequent risk of side effects. The resulting simulations can provide a more accurate summary of the risk of a drug earlier in development and hence more cheaply than the pre-existing approaches. |
| 2025-04-24 | [Data-Driven Calibration of Prediction Sets in Large Vision-Language Models Based on Inductive Conformal Prediction](http://arxiv.org/abs/2504.17671v1) | Yuanchang Ye, Weiyan Wen | This study addresses the critical challenge of hallucination mitigation in Large Vision-Language Models (LVLMs) for Visual Question Answering (VQA) tasks through a Split Conformal Prediction (SCP) framework. While LVLMs excel in multi-modal reasoning, their outputs often exhibit hallucinated content with high confidence, posing risks in safety-critical applications. We propose a model-agnostic uncertainty quantification method that integrates dynamic threshold calibration and cross-modal consistency verification. By partitioning data into calibration and test sets, the framework computes nonconformity scores to construct prediction sets with statistical guarantees under user-defined risk levels ($\alpha$). Key innovations include: (1) rigorous control of \textbf{marginal coverage} to ensure empirical error rates remain strictly below $\alpha$; (2) dynamic adjustment of prediction set sizes inversely with $\alpha$, filtering low-confidence outputs; (3) elimination of prior distribution assumptions and retraining requirements. Evaluations on benchmarks (ScienceQA, MMMU) with eight LVLMs demonstrate that SCP enforces theoretical guarantees across all $\alpha$ values. The framework achieves stable performance across varying calibration-to-test split ratios, underscoring its robustness for real-world deployment in healthcare, autonomous systems, and other safety-sensitive domains. This work bridges the gap between theoretical reliability and practical applicability in multi-modal AI systems, offering a scalable solution for hallucination detection and uncertainty-aware decision-making. |
| 2025-04-24 | [Data-Driven Calibration of Prediction Sets in Large Vision-Language Models Based on Inductive Conformal Prediction](http://arxiv.org/abs/2504.17671v2) | Yuanchang Ye, Weiyan Wen | This study addresses the critical challenge of hallucination mitigation in Large Vision-Language Models (LVLMs) for Visual Question Answering (VQA) tasks through a Split Conformal Prediction (SCP) framework. While LVLMs excel in multi-modal reasoning, their outputs often exhibit hallucinated content with high confidence, posing risks in safety-critical applications. We propose a model-agnostic uncertainty quantification method that integrates dynamic threshold calibration and cross-modal consistency verification. By partitioning data into calibration and test sets, the framework computes nonconformity scores to construct prediction sets with statistical guarantees under user-defined risk levels ($\alpha$). Key innovations include: (1) rigorous control of \textbf{marginal coverage} to ensure empirical error rates remain strictly below $\alpha$; (2) dynamic adjustment of prediction set sizes inversely with $\alpha$, filtering low-confidence outputs; (3) elimination of prior distribution assumptions and retraining requirements. Evaluations on benchmarks (ScienceQA, MMMU) with eight LVLMs demonstrate that SCP enforces theoretical guarantees across all $\alpha$ values. The framework achieves stable performance across varying calibration-to-test split ratios, underscoring its robustness for real-world deployment in healthcare, autonomous systems, and other safety-sensitive domains. This work bridges the gap between theoretical reliability and practical applicability in multi-modal AI systems, offering a scalable solution for hallucination detection and uncertainty-aware decision-making. |
| 2025-04-24 | [Unifying Complementarity Constraints and Control Barrier Functions for Safe Whole-Body Robot Control](http://arxiv.org/abs/2504.17647v1) | Rafael I. Cabral Muchacho, Riddhiman Laha et al. | Safety-critical whole-body robot control demands reactive methods that ensure collision avoidance in real-time. Complementarity constraints and control barrier functions (CBF) have emerged as core tools for ensuring such safety constraints, and each represents a well-developed field. Despite addressing similar problems, their connection remains largely unexplored. This paper bridges this gap by formally proving the equivalence between these two methodologies for sampled-data, first-order systems, considering both single and multiple constraint scenarios. By demonstrating this equivalence, we provide a unified perspective on these techniques. This unification has theoretical and practical implications, facilitating the cross-application of robustness guarantees and algorithmic improvements between complementarity and CBF frameworks. We discuss these synergistic benefits and motivate future work in the comparison of the methods in more general cases. |
| 2025-04-24 | [Portability of Optimizations from SC to TSO](http://arxiv.org/abs/2504.17646v1) | Akshay Gopalakrishnan, Clark Verbrugge | It is well recognized that the safety of compiler optimizations is at risk in a concurrent context. Existing approaches primarily rely on context-free thread-local guarantees, and prohibit optimizations that introduce a data-race. However, compilers utilize global context-specific information, exposing safe optimizations that may violate such guarantees as well as introduce a race. Such optimizations need to individually be proven safe for each language model. An alternate approach to this would be proving them safe for an intuitive model (like interleaving semantics), and then determine their portability across other concurrent models. In this paper, we address this problem of porting across models of concurrency. We first identify a global guarantee on optimizations portable from Sequential Consistency (SC) to Total Store Order (TSO). Our guarantee is in the form of constraints specifying the syntactic changes an optimization must not incur. We then show these constraints correlate to prohibiting the introduction of triangular races, a subset of data-race relevant to TSO. We conclude by showing how such race inducing optimizations relate to porting across Strong Release Acquire (SRA), a known causally consistent memory model. |
| 2025-04-24 | [Safe to Stay: Psychological Safety Sustains Participation in Pull-based Open Source Projects](http://arxiv.org/abs/2504.17510v1) | Emeralda Sesari, Federica Sarro et al. | Psychological safety is the belief that team members can speak up or make mistakes without fear of negative consequences. While it is recognized as important in traditional software teams, its role in open-source development remains understudied. Yet, open-source contributors often collaborate without formal roles or structures, where interpersonal relationship can make or break participation. In this study, we examine whether team-level psychological safety, inferred from code review activities, is associated with contributors' continued participation in open-source projects. Code review is a central and collaborative activity in modern software development, which offers a rich context for observing team interactions. Based on 60,684 pull requests, we construct a psychological safety index using cues such as merge decisions, comment activity, interaction diversity, and mentions. We analyze its relationship with contributors' short-term (after 1 year) and long-term (after 4-5 years) sustained participation using three logistic regression models. Our findings show that contributors are more likely to remain active in repositories with higher levels of psychological safety. Psychological safety is positively associated with both short-term and future sustained participation. However, when prior participation is included, it becomes the stronger predictor of future sustained participation, while the effect of psychological safety becomes smaller. This study introduces a scalable approach to study psychological safety through pull request data and provides new evidence that it matters in open-source development. |
| 2025-04-24 | [Assessing the Capability of Large Language Models for Domain-Specific Ontology Generation](http://arxiv.org/abs/2504.17402v1) | Anna Sofia Lippolis, Mohammad Javad Saeedizade et al. | Large Language Models (LLMs) have shown significant potential for ontology engineering. However, it is still unclear to what extent they are applicable to the task of domain-specific ontology generation. In this study, we explore the application of LLMs for automated ontology generation and evaluate their performance across different domains. Specifically, we investigate the generalizability of two state-of-the-art LLMs, DeepSeek and o1-preview, both equipped with reasoning capabilities, by generating ontologies from a set of competency questions (CQs) and related user stories. Our experimental setup comprises six distinct domains carried out in existing ontology engineering projects and a total of 95 curated CQs designed to test the models' reasoning for ontology engineering. Our findings show that with both LLMs, the performance of the experiments is remarkably consistent across all domains, indicating that these methods are capable of generalizing ontology generation tasks irrespective of the domain. These results highlight the potential of LLM-based approaches in achieving scalable and domain-agnostic ontology construction and lay the groundwork for further research into enhancing automated reasoning and knowledge representation techniques. |
| 2025-04-24 | [Highly Accurate and Diverse Traffic Data: The DeepScenario Open 3D Dataset](http://arxiv.org/abs/2504.17371v1) | Oussema Dhaouadi, Johannes Meier et al. | Accurate 3D trajectory data is crucial for advancing autonomous driving. Yet, traditional datasets are usually captured by fixed sensors mounted on a car and are susceptible to occlusion. Additionally, such an approach can precisely reconstruct the dynamic environment in the close vicinity of the measurement vehicle only, while neglecting objects that are further away. In this paper, we introduce the DeepScenario Open 3D Dataset (DSC3D), a high-quality, occlusion-free dataset of 6 degrees of freedom bounding box trajectories acquired through a novel monocular camera drone tracking pipeline. Our dataset includes more than 175,000 trajectories of 14 types of traffic participants and significantly exceeds existing datasets in terms of diversity and scale, containing many unprecedented scenarios such as complex vehicle-pedestrian interaction on highly populated urban streets and comprehensive parking maneuvers from entry to exit. DSC3D dataset was captured in five various locations in Europe and the United States and include: a parking lot, a crowded inner-city, a steep urban intersection, a federal highway, and a suburban intersection. Our 3D trajectory dataset aims to enhance autonomous driving systems by providing detailed environmental 3D representations, which could lead to improved obstacle interactions and safety. We demonstrate its utility across multiple applications including motion prediction, motion planning, scenario mining, and generative reactive traffic agents. Our interactive online visualization platform and the complete dataset are publicly available at app.deepscenario.com, facilitating research in motion prediction, behavior modeling, and safety validation. |
| 2025-04-24 | [Highly Accurate and Diverse Traffic Data: The DeepScenario Open 3D Dataset](http://arxiv.org/abs/2504.17371v2) | Oussema Dhaouadi, Johannes Meier et al. | Accurate 3D trajectory data is crucial for advancing autonomous driving. Yet, traditional datasets are usually captured by fixed sensors mounted on a car and are susceptible to occlusion. Additionally, such an approach can precisely reconstruct the dynamic environment in the close vicinity of the measurement vehicle only, while neglecting objects that are further away. In this paper, we introduce the DeepScenario Open 3D Dataset (DSC3D), a high-quality, occlusion-free dataset of 6 degrees of freedom bounding box trajectories acquired through a novel monocular camera drone tracking pipeline. Our dataset includes more than 175,000 trajectories of 14 types of traffic participants and significantly exceeds existing datasets in terms of diversity and scale, containing many unprecedented scenarios such as complex vehicle-pedestrian interaction on highly populated urban streets and comprehensive parking maneuvers from entry to exit. DSC3D dataset was captured in five various locations in Europe and the United States and include: a parking lot, a crowded inner-city, a steep urban intersection, a federal highway, and a suburban intersection. Our 3D trajectory dataset aims to enhance autonomous driving systems by providing detailed environmental 3D representations, which could lead to improved obstacle interactions and safety. We demonstrate its utility across multiple applications including motion prediction, motion planning, scenario mining, and generative reactive traffic agents. Our interactive online visualization platform and the complete dataset are publicly available at https://app.deepscenario.com, facilitating research in motion prediction, behavior modeling, and safety validation. |
| 2025-04-24 | [AUTHENTICATION: Identifying Rare Failure Modes in Autonomous Vehicle Perception Systems using Adversarially Guided Diffusion Models](http://arxiv.org/abs/2504.17179v1) | Mohammad Zarei, Melanie A Jutras et al. | Autonomous Vehicles (AVs) rely on artificial intelligence (AI) to accurately detect objects and interpret their surroundings. However, even when trained using millions of miles of real-world data, AVs are often unable to detect rare failure modes (RFMs). The problem of RFMs is commonly referred to as the "long-tail challenge", due to the distribution of data including many instances that are very rarely seen. In this paper, we present a novel approach that utilizes advanced generative and explainable AI techniques to aid in understanding RFMs. Our methods can be used to enhance the robustness and reliability of AVs when combined with both downstream model training and testing. We extract segmentation masks for objects of interest (e.g., cars) and invert them to create environmental masks. These masks, combined with carefully crafted text prompts, are fed into a custom diffusion model. We leverage the Stable Diffusion inpainting model guided by adversarial noise optimization to generate images containing diverse environments designed to evade object detection models and expose vulnerabilities in AI systems. Finally, we produce natural language descriptions of the generated RFMs that can guide developers and policymakers to improve the safety and reliability of AV systems. |
| 2025-04-23 | [Opt-ODENet: A Neural ODE Framework with Differentiable QP Layers for Safe and Stable Control Design (longer version)](http://arxiv.org/abs/2504.17139v1) | Keyan Miao, Liqun Zhao et al. | Designing controllers that achieve task objectives while ensuring safety is a key challenge in control systems. This work introduces Opt-ODENet, a Neural ODE framework with a differentiable Quadratic Programming (QP) optimization layer to enforce constraints as hard requirements. Eliminating the reliance on nominal controllers or large datasets, our framework solves the optimal control problem directly using Neural ODEs. Stability and convergence are ensured through Control Lyapunov Functions (CLFs) in the loss function, while Control Barrier Functions (CBFs) embedded in the QP layer enforce real-time safety. By integrating the differentiable QP layer with Neural ODEs, we demonstrate compatibility with the adjoint method for gradient computation, enabling the learning of the CBF class-$\mathcal{K}$ function and control network parameters. Experiments validate its effectiveness in balancing safety and performance. |
| 2025-04-23 | [Steering the CensorShip: Uncovering Representation Vectors for LLM "Thought" Control](http://arxiv.org/abs/2504.17130v1) | Hannah Cyberey, David Evans | Large language models (LLMs) have transformed the way we access information. These models are often tuned to refuse to comply with requests that are considered harmful and to produce responses that better align with the preferences of those who control the models. To understand how this "censorship" works. We use representation engineering techniques to study open-weights safety-tuned models. We present a method for finding a refusal--compliance vector that detects and controls the level of censorship in model outputs. We also analyze recent reasoning LLMs, distilled from DeepSeek-R1, and uncover an additional dimension of censorship through "thought suppression". We show a similar approach can be used to find a vector that suppresses the model's reasoning process, allowing us to remove censorship by applying the negative multiples of this vector |
| 2025-04-23 | [Peer-Aware Cost Estimation in Nonlinear General-Sum Dynamic Games for Mutual Learning and Intent Inference](http://arxiv.org/abs/2504.17129v1) | Seyed Yousef Soltanian, Wenlong Zhang | Human-robot interactions can be modeled as incomplete-information general-sum dynamic games since the objective functions of both agents are not explicitly known to each other. However, solving for equilibrium policies for such games presents a major challenge, especially if the games involve nonlinear underlying dynamics. To simplify the problem, existing work often assumes that one agent is an expert with complete information about its peer, which can lead to biased estimates and failures in coordination. To address this challenge, we propose a nonlinear peer-aware cost estimation (N-PACE) algorithm for general-sum dynamic games. In N-PACE, using iterative linear quadratic (LQ) approximation of the nonlinear general-sum game, each agent explicitly models the learning dynamics of its peer agent while inferring their objective functions, leading to unbiased fast learning in inferring the unknown objective function of the peer agent, which is critical for task completion and safety assurance. Additionally, we demonstrate how N-PACE enables \textbf{intent communication} in such multi-agent systems by explicitly modeling the peer's learning dynamics. |
| 2025-04-23 | [Discovering the Precursors of Traffic Breakdowns Using Spatiotemporal Graph Attribution Networks](http://arxiv.org/abs/2504.17109v1) | Zhaobin Mo, Xiangyi Liao et al. | Understanding and predicting the precursors of traffic breakdowns is critical for improving road safety and traffic flow management. This paper presents a novel approach combining spatiotemporal graph neural networks (ST-GNNs) with Shapley values to identify and interpret traffic breakdown precursors. By extending Shapley explanation methods to a spatiotemporal setting, our proposed method bridges the gap between black-box neural network predictions and interpretable causes. We demonstrate the method on the Interstate-24 data, and identify that road topology and abrupt braking are major factors that lead to traffic breakdowns. |
| 2025-04-23 | [Safety Pretraining: Toward the Next Generation of Safe AI](http://arxiv.org/abs/2504.16980v1) | Pratyush Maini, Sachin Goyal et al. | As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. We present a data-centric pretraining framework that builds safety into the model from the start. Our contributions include: (i) a safety classifier trained on 10,000 GPT-4 labeled examples, used to filter 600B tokens; (ii) the largest synthetic safety dataset to date (100B tokens) generated via recontextualization of harmful web data; (iii) RefuseWeb and Moral Education datasets that convert harmful prompts into refusal dialogues and web-style educational material; (iv) Harmfulness-Tag annotations injected during pretraining to flag unsafe content and steer away inference from harmful generations; and (v) safety evaluations measuring base model behavior before instruction tuning. Our safety-pretrained models reduce attack success rates from 38.8% to 8.4% with no performance degradation on standard LLM safety benchmarks. |
| 2025-04-23 | [Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving](http://arxiv.org/abs/2504.16923v1) | Jacob Levy, Jason Gibson et al. | High-speed off-road autonomous driving presents unique challenges due to complex, evolving terrain characteristics and the difficulty of accurately modeling terrain-vehicle interactions. While dynamics models used in model-based control can be learned from real-world data, they often struggle to generalize to unseen terrain, making real-time adaptation essential. We propose a novel framework that combines a Kalman filter-based online adaptation scheme with meta-learned parameters to address these challenges. Offline meta-learning optimizes the basis functions along which adaptation occurs, as well as the adaptation parameters, while online adaptation dynamically adjusts the onboard dynamics model in real time for model-based control. We validate our approach through extensive experiments, including real-world testing on a full-scale autonomous off-road vehicle, demonstrating that our method outperforms baseline approaches in prediction accuracy, performance, and safety metrics, particularly in safety-critical scenarios. Our results underscore the effectiveness of meta-learned dynamics model adaptation, advancing the development of reliable autonomous systems capable of navigating diverse and unseen environments. Video is available at: https://youtu.be/cCKHHrDRQEA |
| 2025-04-23 | [Learning Verifiable Control Policies Using Relaxed Verification](http://arxiv.org/abs/2504.16879v1) | Puja Chaudhury, Alexander Estornell et al. | To provide safety guarantees for learning-based control systems, recent work has developed formal verification methods to apply after training ends. However, if the trained policy does not meet the specifications, or there is conservatism in the verification algorithm, establishing these guarantees may not be possible. Instead, this work proposes to perform verification throughout training to ultimately aim for policies whose properties can be evaluated throughout runtime with lightweight, relaxed verification algorithms. The approach is to use differentiable reachability analysis and incorporate new components into the loss function. Numerical experiments on a quadrotor model and unicycle model highlight the ability of this approach to lead to learned control policies that satisfy desired reach-avoid and invariance specifications. |

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



