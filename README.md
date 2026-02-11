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
| 2026-02-10 | [Learning Agile Quadrotor Flight in the Real World](http://arxiv.org/abs/2602.10111v1) | Yunfan Ren, Zhiyuan Zhu et al. | Learning-based controllers have achieved impressive performance in agile quadrotor flight but typically rely on massive training in simulation, necessitating accurate system identification for effective Sim2Real transfer. However, even with precise modeling, fixed policies remain susceptible to out-of-distribution scenarios, ranging from external aerodynamic disturbances to internal hardware degradation. To ensure safety under these evolving uncertainties, such controllers are forced to operate with conservative safety margins, inherently constraining their agility outside of controlled settings. While online adaptation offers a potential remedy, safely exploring physical limits remains a critical bottleneck due to data scarcity and safety risks. To bridge this gap, we propose a self-adaptive framework that eliminates the need for precise system identification or offline Sim2Real transfer. We introduce Adaptive Temporal Scaling (ATS) to actively explore platform physical limits, and employ online residual learning to augment a simple nominal model. {Based on the learned hybrid model, we further propose Real-world Anchored Short-horizon Backpropagation Through Time (RASH-BPTT) to achieve efficient and robust in-flight policy updates. Extensive experiments demonstrate that our quadrotor reliably executes agile maneuvers near actuator saturation limits. The system evolves a conservative base policy with a peak speed of 1.9 m/s to 7.3 m/s within approximately 100 seconds of flight time. These findings underscore that real-world adaptation serves not merely to compensate for modeling errors, but as a practical mechanism for sustained performance improvement in aggressive flight regimes. |
| 2026-02-10 | [Humanoid Factors: Design Principles for AI Humanoids in Human Worlds](http://arxiv.org/abs/2602.10069v1) | Xinyuan Liu, Eren Sadikoglu et al. | Human factors research has long focused on optimizing environments, tools, and systems to account for human performance. Yet, as humanoid robots begin to share our workplaces, homes, and public spaces, the design challenge expands. We must now consider not only factors for humans but also factors for humanoids, since both will coexist and interact within the same environments. Unlike conventional machines, humanoids introduce expectations of human-like behavior, communication, and social presence, which reshape usability, trust, and safety considerations. In this article, we introduce the concept of humanoid factors as a framework structured around four pillars - physical, cognitive, social, and ethical - that shape the development of humanoids to help them effectively coexist and collaborate with humans. This framework characterizes the overlap and divergence between human capabilities and those of general-purpose humanoids powered by AI foundation models. To demonstrate our framework's practical utility, we then apply the framework to evaluate a real-world humanoid control algorithm, illustrating how conventional task completion metrics in robotics overlook key human cognitive and interaction principles. We thus position humanoid factors as a foundational framework for designing, evaluating, and governing sustained human-humanoid coexistence. |
| 2026-02-10 | [Perception with Guarantees: Certified Pose Estimation via Reachability Analysis](http://arxiv.org/abs/2602.10032v1) | Tobias Ladner, Yasser Shoukry et al. | Agents in cyber-physical systems are increasingly entrusted with safety-critical tasks. Ensuring safety of these agents often requires localizing the pose for subsequent actions. Pose estimates can, e.g., be obtained from various combinations of lidar sensors, cameras, and external services such as GPS. Crucially, in safety-critical domains, a rough estimate is insufficient to formally determine safety, i.e., guaranteeing safety even in the worst-case scenario, and external services might additionally not be trustworthy. We address this problem by presenting a certified pose estimation in 3D solely from a camera image and a well-known target geometry. This is realized by formally bounding the pose, which is computed by leveraging recent results from reachability analysis and formal neural network verification. Our experiments demonstrate that our approach efficiently and accurately localizes agents in both synthetic and real-world experiments. |
| 2026-02-10 | [A Collaborative Safety Shield for Safe and Efficient CAV Lane Changes in Congested On-Ramp Merging](http://arxiv.org/abs/2602.10007v1) | Bharathkumar Hegde, Melanie Bouroche | Lane changing in dense traffic is a significant challenge for Connected and Autonomous Vehicles (CAVs). Existing lane change controllers primarily either ensure safety or collaboratively improve traffic efficiency, but do not consider these conflicting objectives together. To address this, we propose the Multi-Agent Safety Shield (MASS), designed using Control Barrier Functions (CBFs) to enable safe and collaborative lane changes. The MASS enables collaboration by capturing multi-agent interactions among CAVs through interaction topologies constructed as a graph using a simple algorithm. Further, a state-of-the-art Multi-Agent Reinforcement Learning (MARL) lane change controller is extended by integrating MASS to ensure safety and defining a customised reward function to prioritise efficiency improvements. As a result, we propose a lane change controller, known as MARL-MASS, and evaluate it in a congested on-ramp merging simulation. The results demonstrate that MASS enables collaborative lane changes with safety guarantees by strictly respecting the safety constraints. Moreover, the proposed custom reward function improves the stability of MARL policies trained with a safety shield. Overall, by encouraging the exploration of a collaborative lane change policy while respecting safety constraints, MARL-MASS effectively balances the trade-off between ensuring safety and improving traffic efficiency in congested traffic. The code for MARL-MASS is available with an open-source licence at https://github.com/hkbharath/MARL-MASS |
| 2026-02-10 | [Safe Feedback Optimization through Control Barrier Functions](http://arxiv.org/abs/2602.09928v1) | Giannis Delimpaltadakis, Pol Mestres et al. | Feedback optimization refers to a class of methods that steer a control system to a steady state that solves an optimization problem. Despite tremendous progress on the topic, an important problem remains open: enforcing state constraints at all times. The difficulty in addressing it lies on mediating between the safety enforcement and the closed-loop stability, and ensuring the equivalence between closed-loop equilibria and the optimization problem's critical points. In this work, we present a feedback-optimization method that enforces state constraints at all times employing high-order control-barrier functions. We provide several results on the proposed controller dynamics, including well-posedness, safety guarantees, equivalence between equilibria and critical points, and local and global (in certain convex cases) asymptotic stability of optima. Various simulations illustrate our results. |
| 2026-02-10 | [The Devil Behind Moltbook: Anthropic Safety is Always Vanishing in Self-Evolving AI Societies](http://arxiv.org/abs/2602.09877v1) | Chenxu Wang, Chaozhuo Li et al. | The emergence of multi-agent systems built from large language models (LLMs) offers a promising paradigm for scalable collective intelligence and self-evolution. Ideally, such systems would achieve continuous self-improvement in a fully closed loop while maintaining robust safety alignment--a combination we term the self-evolution trilemma. However, we demonstrate both theoretically and empirically that an agent society satisfying continuous self-evolution, complete isolation, and safety invariance is impossible. Drawing on an information-theoretic framework, we formalize safety as the divergence degree from anthropic value distributions. We theoretically demonstrate that isolated self-evolution induces statistical blind spots, leading to the irreversible degradation of the system's safety alignment. Empirical and qualitative results from an open-ended agent community (Moltbook) and two closed self-evolving systems reveal phenomena that align with our theoretical prediction of inevitable safety erosion. We further propose several solution directions to alleviate the identified safety concern. Our work establishes a fundamental limit on the self-evolving AI societies and shifts the discourse from symptom-driven safety patches to a principled understanding of intrinsic dynamical risks, highlighting the need for external oversight or novel safety-preserving mechanisms. |
| 2026-02-10 | [Steer2Edit: From Activation Steering to Component-Level Editing](http://arxiv.org/abs/2602.09870v1) | Chung-En Sun, Ge Yan et al. | Steering methods influence Large Language Model behavior by identifying semantic directions in hidden representations, but are typically realized through inference-time activation interventions that apply a fixed, global modification to the model's internal states. While effective, such interventions often induce unfavorable attribute-utility trade-offs under strong control, as they ignore the fact that many behaviors are governed by a small and heterogeneous subset of model components. We propose Steer2Edit, a theoretically grounded, training-free framework that transforms steering vectors from inference-time control signals into diagnostic signals for component-level rank-1 weight editing. Instead of uniformly injecting a steering direction during generation, Steer2Edit selectively redistributes behavioral influence across individual attention heads and MLP neurons, yielding interpretable edits that preserve the standard forward pass and remain compatible with optimized parallel inference. Across safety alignment, hallucination mitigation, and reasoning efficiency, Steer2Edit consistently achieves more favorable attribute-utility trade-offs: at matched downstream performance, it improves safety by up to 17.2%, increases truthfulness by 9.8%, and reduces reasoning length by 12.2% on average. Overall, Steer2Edit provides a principled bridge between representation steering and weight editing by translating steering signals into interpretable, training-free parameter updates. |
| 2026-02-10 | [ClinAlign: Scaling Healthcare Alignment from Clinician Preference](http://arxiv.org/abs/2602.09653v1) | Shiwei Lyu, Xidong Wang et al. | Although large language models (LLMs) demonstrate expert-level medical knowledge, aligning their open-ended outputs with fine-grained clinician preferences remains challenging. Existing methods often rely on coarse objectives or unreliable automated judges that are weakly grounded in professional guidelines. We propose a two-stage framework to address this gap. First, we introduce HealthRubrics, a dataset of 7,034 physician-verified preference examples in which clinicians refine LLM-drafted rubrics to meet rigorous medical standards. Second, we distill these rubrics into HealthPrinciples: 119 broadly reusable, clinically grounded principles organized by clinical dimensions, enabling scalable supervision beyond manual annotation. We use HealthPrinciples for (1) offline alignment by synthesizing rubrics for unlabeled queries and (2) an inference-time tool for guided self-revision. A 30B parameter model that activates only 3B parameters at inference trained with our framework achieves 33.4% on HealthBench-Hard, outperforming much larger models including Deepseek-R1 and o3, establishing a resource-efficient baseline for clinical alignment. |
| 2026-02-10 | [Bayesian network approach to building an affective module for a driver behavioural model](http://arxiv.org/abs/2602.09632v1) | Dorota Młynarczyk, Gabriel Calvo et al. | This paper focuses on the affective component of a driver behavioural model (DBM). This component specifically models some drivers' mental states such as mental load and active fatigue, which may affect driving performance. We have used Bayesian networks (BNs) to explore the dependencies between various relevant random variables and assess the probability that a driver is in a particular mental state based on their physiological and demographic conditions. Through this approach, our goal is to improve our understanding of driver behaviour in dynamic environments, with potential applications in traffic safety and autonomous vehicle technologies. |
| 2026-02-10 | [Stop Testing Attacks, Start Diagnosing Defenses: The Four-Checkpoint Framework Reveals Where LLM Safety Breaks](http://arxiv.org/abs/2602.09629v1) | Hayfa Dhabhi, Kashyap Thimmaraju | Large Language Models (LLMs) deploy safety mechanisms to prevent harmful outputs, yet these defenses remain vulnerable to adversarial prompts. While existing research demonstrates that jailbreak attacks succeed, it does not explain \textit{where} defenses fail or \textit{why}.   To address this gap, we propose that LLM safety operates as a sequential pipeline with distinct checkpoints. We introduce the \textbf{Four-Checkpoint Framework}, which organizes safety mechanisms along two dimensions: processing stage (input vs.\ output) and detection level (literal vs.\ intent). This creates four checkpoints, CP1 through CP4, each representing a defensive layer that can be independently evaluated. We design 13 evasion techniques, each targeting a specific checkpoint, enabling controlled testing of individual defensive layers.   Using this framework, we evaluate GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across 3,312 single-turn, black-box test cases. We employ an LLM-as-judge approach for response classification and introduce Weighted Attack Success Rate (WASR), a severity-adjusted metric that captures partial information leakage overlooked by binary evaluation.   Our evaluation reveals clear patterns. Traditional Binary ASR reports 22.6\% attack success. However, WASR reveals 52.7\%, a 2.3$\times$ higher vulnerability. Output-stage defenses (CP3, CP4) prove weakest at 72--79\% WASR, while input-literal defenses (CP1) are strongest at 13\% WASR. Claude achieves the strongest safety (42.8\% WASR), followed by GPT-5 (55.9\%) and Gemini (59.5\%).   These findings suggest that current defenses are strongest at input-literal checkpoints but remain vulnerable to intent-level manipulation and output-stage techniques. The Four-Checkpoint Framework provides a structured approach for identifying and addressing safety vulnerabilities in deployed systems. |
| 2026-02-10 | [On the Optimal Reasoning Length for RL-Trained Language Models](http://arxiv.org/abs/2602.09591v1) | Daisuke Nohara, Taishi Nakamura et al. | Reinforcement learning substantially improves reasoning in large language models, but it also tends to lengthen chain of thought outputs and increase computational cost during both training and inference. Though length control methods have been proposed, it remains unclear what the optimal output length is for balancing efficiency and performance. In this work, we compare several length control methods on two models, Qwen3-1.7B Base and DeepSeek-R1-Distill-Qwen-1.5B. Our results indicate that length penalties may hinder reasoning acquisition, while properly tuned length control can improve efficiency for models with strong prior reasoning. By extending prior work to RL trained policies, we identify two failure modes, 1) long outputs increase dispersion, and 2) short outputs lead to under-thinking. |
| 2026-02-10 | [Advancing Block Diffusion Language Models for Test-Time Scaling](http://arxiv.org/abs/2602.09555v1) | Yi Lu, Deyang Kong et al. | Recent advances in block diffusion language models have demonstrated competitive performance and strong scalability on reasoning tasks. However, existing BDLMs have limited exploration under the test-time scaling setting and face more severe decoding challenges in long Chain-of-Thought reasoning, particularly in balancing the decoding speed and effectiveness. In this work, we propose a unified framework for test-time scaling in BDLMs that introduces adaptivity in both decoding and block-wise generation. At the decoding level, we propose Bounded Adaptive Confidence Decoding (BACD), a difficulty-aware sampling strategy that dynamically adjusts denoising based on model confidence, accelerating inference while controlling error accumulation. Beyond step-wise adaptivity, we introduce Think Coarse, Critic Fine (TCCF), a test-time scaling paradigm that allocates large block sizes to exploratory reasoning and smaller block sizes to refinement, achieving an effective efficiency-effectiveness balance. To enable efficient and effective decoding with a large block size, we adopt Progressive Block Size Extension, which mitigates performance degradation when scaling block sizes. Extensive experiments show that applying BACD and TCCF to TDAR-8B yields significant improvements over strong baselines such as TraDo-8B (2.26x speedup, +11.2 points on AIME24). These results mark an important step toward unlocking the potential of BDLMs for test-time scaling in complex reasoning tasks. |
| 2026-02-10 | [Shifting landscape of disability and development in India: Analysis from historical trends to future predictions 2001-2031](http://arxiv.org/abs/2602.09543v1) | Hana Kapadia, Arun Kumar Rajasekaran | This study delves into the causes and trends of disability-related health burdens across Indian states. Through multiple Disability-Adjusted Life Years (DALY) types (covering communicable diseases, noncommunicable diseases, and injuries), gender disparities, and Human Development Index (HDI) values, these disability trends were evaluated. The data for this study was compiled from censuses, health research organisations, and data centres, among various other sources. We built regression models and used them to analyze trends across past decades and make projections for 2031. Our regression results show a strong inverse relationship between communicable disease DALYs and HDI. In other words, ongoing improvements in development and infrastructure significantly reduced communicable disease DALYs. In contrast, noncommunicable DALYs did not decrease despite rising HDI. And lastly, injury DALYs showed moderate declines with higher HDI, which reflects improvements in healthcare and safety systems. Gender analysis showed male overrepresentation among people with disabilities. These results from our study support that there is a need to shift public health focus toward chronic diseases and address gender disparities in disability outcomes. |
| 2026-02-10 | [A Behavioral Fingerprint for Large Language Models: Provenance Tracking via Refusal Vectors](http://arxiv.org/abs/2602.09434v1) | Zhenyu Xu, Victor S. Sheng | Protecting the intellectual property of large language models (LLMs) is a critical challenge due to the proliferation of unauthorized derivative models. We introduce a novel fingerprinting framework that leverages the behavioral patterns induced by safety alignment, applying the concept of refusal vectors for LLM provenance tracking. These vectors, extracted from directional patterns in a model's internal representations when processing harmful versus harmless prompts, serve as robust behavioral fingerprints. Our contribution lies in developing a fingerprinting system around this concept and conducting extensive validation of its effectiveness for IP protection. We demonstrate that these behavioral fingerprints are highly robust against common modifications, including finetunes, merges, and quantization. Our experiments show that the fingerprint is unique to each model family, with low cosine similarity between independently trained models. In a large-scale identification task across 76 offspring models, our method achieves 100\% accuracy in identifying the correct base model family. Furthermore, we analyze the fingerprint's behavior under alignment-breaking attacks, finding that while performance degrades significantly, detectable traces remain. Finally, we propose a theoretical framework to transform this private fingerprint into a publicly verifiable, privacy-preserving artifact using locality-sensitive hashing and zero-knowledge proofs. |
| 2026-02-10 | [Certified Gradient-Based Contact-Rich Manipulation via Smoothing-Error Reachable Tubes](http://arxiv.org/abs/2602.09368v1) | Wei-Chen Li, Glen Chou | Gradient-based methods can efficiently optimize controllers using physical priors and differentiable simulators, but contact-rich manipulation remains challenging due to discontinuous or vanishing gradients from hybrid contact dynamics. Smoothing the dynamics yields continuous gradients, but the resulting model mismatch can cause controller failures when executed on real systems. We address this trade-off by planning with smoothed dynamics while explicitly quantifying and compensating for the induced errors, providing formal guarantees of constraint satisfaction and goal reachability on the true hybrid dynamics. Our method smooths both contact dynamics and geometry via a novel differentiable simulator based on convex optimization, which enables us to characterize the discrepancy from the true dynamics as a set-valued deviation. This deviation constrains the optimization of time-varying affine feedback policies through analytical bounds on the system's reachable set, enabling robust constraint satisfaction guarantees for the true closed-loop hybrid dynamics, while relying solely on informative gradients from the smoothed dynamics. We evaluate our method on several contact-rich tasks, including planar pushing, object rotation, and in-hand dexterous manipulation, achieving guaranteed constraint satisfaction with lower safety violation and goal error than baselines. By bridging differentiable physics with set-valued robust control, our method is the first certifiable gradient-based policy synthesis method for contact-rich manipulation. |
| 2026-02-10 | [The Similarity Control Problem with Required Events](http://arxiv.org/abs/2602.09360v1) | Yu Wang, Zhaohui Zhu et al. | In order to guarantee that a supervised system satisfies safety requirements of the specification, as well as requirements saying that in certain states certain events must be enabled, this paper introduces required events for discrete event systems and reconsiders the similarity control problem while taking all requirements from the specification into account. The notion of a covariant-contravariant simulation, which is finer than the conventional notion of simulation, is adopted to act as the behavioral relation of supervisory control theory. A necessary and sufficient condition for the solvability of this problem is established and a method for synthesizing a maximally permissive supervisor is provided. |
| 2026-02-10 | [Understanding Risk and Dependency in AI Chatbot Use from User Discourse](http://arxiv.org/abs/2602.09339v1) | Jianfeng Zhu, Karin G. Coifman et al. | Generative AI systems are increasingly embedded in everyday life, yet empirical understanding of how psychological risk associated with AI use emerges, is experienced, and is regulated by users remains limited. We present a large-scale computational thematic analysis of posts collected between 2023 and 2025 from two Reddit communities, r/AIDangers and r/ChatbotAddiction, explicitly focused on AI-related harm and distress. Using a multi-agent, LLM-assisted thematic analysis grounded in Braun and Clarke's reflexive framework, we identify 14 recurring thematic categories and synthesize them into five higher-order experiential dimensions. To further characterize affective patterns, we apply emotion labeling using a BERT-based classifier and visualize emotional profiles across dimensions. Our findings reveal five empirically derived experiential dimensions of AI-related psychological risk grounded in real-world user discourse, with self-regulation difficulties emerging as the most prevalent and fear concentrated in concerns related to autonomy, control, and technical risk. These results provide early empirical evidence from lived user experience of how AI safety is perceived and emotionally experienced outside laboratory or speculative contexts, offering a foundation for future AI safety research, evaluation, and responsible governance. |
| 2026-02-10 | [SnareNet: Flexible Repair Layers for Neural Networks with Hard Constraints](http://arxiv.org/abs/2602.09317v1) | Ya-Chi Chu, Alkiviades Boukas et al. | Neural networks are increasingly used as surrogate solvers and control policies, but unconstrained predictions can violate physical, operational, or safety requirements. We propose SnareNet, a feasibility-controlled architecture for learning mappings whose outputs must satisfy input-dependent nonlinear constraints. SnareNet appends a differentiable repair layer that navigates in the constraint map's range space, steering iterates toward feasibility and producing a repaired output that satisfies constraints to a user-specified tolerance. To stabilize end-to-end training, we introduce adaptive relaxation, which designs a relaxed feasible set that snares the neural network at initialization and shrinks it into the feasible set, enabling early exploration and strict feasibility later in training. On optimization-learning and trajectory planning benchmarks, SnareNet consistently attains improved objective quality while satisfying constraints more reliably than prior work. |
| 2026-02-09 | ["Create an environment that protects women, rather than selling anxiety!": Participatory Threat Modeling with Chinese Young Women Living Alone](http://arxiv.org/abs/2602.09256v1) | Shijing He, Chenkai Ma et al. | As more young women in China live alone, they navigate entangled privacy, security, and safety (PSS) risks across smart homes, online platforms, and public infrastructures. Drawing on six participatory threat modeling (PTM) workshops (n = 33), we present a human-centered threat model that illustrates how digitally facilitated physical violence, digital harassment and scams, and pervasive surveillance by individuals, companies, and the state are interconnected and mutually reinforcing. We also document four mitigation strategies employed by participants: smart home device configurations, boundary management, sociocultural practices, and social media tactics--each of which can introduce new vulnerabilities and emotional burdens. Based on these insights, we developed a digital PSS guidebook for young women living alone (YWLA) in China. We further propose actionable design implications for smart home devices and social media platforms, along with policy and legal recommendations and directions for educational interventions. |
| 2026-02-09 | [Risk-Aware Obstacle Avoidance Algorithm for Real-Time Applications](http://arxiv.org/abs/2602.09204v1) | Ozan Kaya, Emir Cem Gezer et al. | Robust navigation in changing marine environments requires autonomous systems capable of perceiving, reasoning, and acting under uncertainty. This study introduces a hybrid risk-aware navigation architecture that integrates probabilistic modeling of obstacles along the vehicle path with smooth trajectory optimization for autonomous surface vessels. The system constructs probabilistic risk maps that capture both obstacle proximity and the behavior of dynamic objects. A risk-biased Rapidly Exploring Random Tree (RRT) planner leverages these maps to generate collision-free paths, which are subsequently refined using B-spline algorithms to ensure trajectory continuity. Three distinct RRT* rewiring modes are implemented based on the cost function: minimizing the path length, minimizing risk, and optimizing a combination of the path length and total risk. The framework is evaluated in experimental scenarios containing both static and dynamic obstacles. The results demonstrate the system's ability to navigate safely, maintain smooth trajectories, and dynamically adapt to changing environmental risks. Compared with conventional LIDAR or vision-only navigation approaches, the proposed method shows improvements in operational safety and autonomy, establishing it as a promising solution for risk-aware autonomous vehicle missions in uncertain and dynamic environments. |

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



