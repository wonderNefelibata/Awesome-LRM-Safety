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
| 2025-05-16 | [SHIELD: Safety on Humanoids via CBFs In Expectation on Learned Dynamics](http://arxiv.org/abs/2505.11494v1) | Lizhi Yang, Blake Werner et al. | Robot learning has produced remarkably effective ``black-box'' controllers for complex tasks such as dynamic locomotion on humanoids. Yet ensuring dynamic safety, i.e., constraint satisfaction, remains challenging for such policies. Reinforcement learning (RL) embeds constraints heuristically through reward engineering, and adding or modifying constraints requires retraining. Model-based approaches, like control barrier functions (CBFs), enable runtime constraint specification with formal guarantees but require accurate dynamics models. This paper presents SHIELD, a layered safety framework that bridges this gap by: (1) training a generative, stochastic dynamics residual model using real-world data from hardware rollouts of the nominal controller, capturing system behavior and uncertainties; and (2) adding a safety layer on top of the nominal (learned locomotion) controller that leverages this model via a stochastic discrete-time CBF formulation enforcing safety constraints in probability. The result is a minimally-invasive safety layer that can be added to the existing autonomy stack to give probabilistic guarantees of safety that balance risk and performance. In hardware experiments on an Unitree G1 humanoid, SHIELD enables safe navigation (obstacle avoidance) through varied indoor and outdoor environments using a nominal (unknown) RL controller and onboard perception. |
| 2025-05-16 | [UMArm: Untethered, Modular, Wearable, Soft Pneumatic Arm](http://arxiv.org/abs/2505.11476v1) | Runze Zuo, Dong Heon Han et al. | Robotic arms are essential to modern industries, however, their adaptability to unstructured environments remains limited. Soft robotic arms, particularly those actuated pneumatically, offer greater adaptability in unstructured environments and enhanced safety for human-robot interaction. However, current pneumatic soft arms are constrained by limited degrees of freedom, precision, payload capacity, and reliance on bulky external pressure regulators. In this work, a novel pneumatically driven rigid-soft hybrid arm, ``UMArm'', is presented. The shortcomings of pneumatically actuated soft arms are addressed by densely integrating high-force-to-weight-ratio, self-regulated McKibben actuators onto a lightweight rigid spine structure. The modified McKibben actuators incorporate valves and controllers directly inside, eliminating the need for individual pressure lines and external regulators, significantly reducing system weight and complexity. Full untethered operation, high payload capacity, precision, and directionally tunable compliance are achieved by the UMArm. Portability is demonstrated through a wearable assistive arm experiment, and versatility is showcased by reconfiguring the system into an inchworm robot. The results of this work show that the high-degree-of-freedom, external-regulator-free pneumatically driven arm systems like the UMArm possess great potential for real-world unstructured environments. |
| 2025-05-16 | [REACT: Runtime-Enabled Active Collision-avoidance Technique for Autonomous Driving](http://arxiv.org/abs/2505.11474v1) | Heye Huang, Hao Cheng et al. | Achieving rapid and effective active collision avoidance in dynamic interactive traffic remains a core challenge for autonomous driving. This paper proposes REACT (Runtime-Enabled Active Collision-avoidance Technique), a closed-loop framework that integrates risk assessment with active avoidance control. By leveraging energy transfer principles and human-vehicle-road interaction modeling, REACT dynamically quantifies runtime risk and constructs a continuous spatial risk field. The system incorporates physically grounded safety constraints such as directional risk and traffic rules to identify high-risk zones and generate feasible, interpretable avoidance behaviors. A hierarchical warning trigger strategy and lightweight system design enhance runtime efficiency while ensuring real-time responsiveness. Evaluations across four representative high-risk scenarios including car-following braking, cut-in, rear-approaching, and intersection conflict demonstrate REACT's capability to accurately identify critical risks and execute proactive avoidance. Its risk estimation aligns closely with human driver cognition (i.e., warning lead time < 0.4 s), achieving 100% safe avoidance with zero false alarms or missed detections. Furthermore, it exhibits superior real-time performance (< 50 ms latency), strong foresight, and generalization. The lightweight architecture achieves state-of-the-art accuracy, highlighting its potential for real-time deployment in safety-critical autonomous systems. |
| 2025-05-16 | [Disentangling Reasoning and Knowledge in Medical Large Language Models](http://arxiv.org/abs/2505.11462v1) | Rahul Thapa, Qingyang Wu et al. | Medical reasoning in large language models (LLMs) aims to emulate clinicians' diagnostic thinking, but current benchmarks such as MedQA-USMLE, MedMCQA, and PubMedQA often mix reasoning with factual recall. We address this by separating 11 biomedical QA benchmarks into reasoning- and knowledge-focused subsets using a PubMedBERT classifier that reaches 81 percent accuracy, comparable to human performance. Our analysis shows that only 32.8 percent of questions require complex reasoning. We evaluate biomedical models (HuatuoGPT-o1, MedReason, m1) and general-domain models (DeepSeek-R1, o4-mini, Qwen3), finding consistent gaps between knowledge and reasoning performance. For example, m1 scores 60.5 on knowledge but only 47.1 on reasoning. In adversarial tests where models are misled with incorrect initial reasoning, biomedical models degrade sharply, while larger or RL-trained general models show more robustness. To address this, we train BioMed-R1 using fine-tuning and reinforcement learning on reasoning-heavy examples. It achieves the strongest performance among similarly sized models. Further gains may come from incorporating clinical case reports and training with adversarial and backtracking scenarios. |
| 2025-05-16 | [CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs](http://arxiv.org/abs/2505.11413v1) | Sijia Chen, Xiaomin Li et al. | Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions. |
| 2025-05-16 | [Phare: A Safety Probe for Large Language Models](http://arxiv.org/abs/2505.11365v1) | Pierre Le Jeune, Benoît Malésieux et al. | Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems. |
| 2025-05-16 | [Sobolev Training of End-to-End Optimization Proxies](http://arxiv.org/abs/2505.11342v1) | Andrew W. Rosemberg, Joaquim Dias Garcia et al. | Optimization proxies - machine learning models trained to approximate the solution mapping of parametric optimization problems in a single forward pass - offer dramatic reductions in inference time compared to traditional iterative solvers. This work investigates the integration of solver sensitivities into such end to end proxies via a Sobolev training paradigm and does so in two distinct settings: (i) fully supervised proxies, where exact solver outputs and sensitivities are available, and (ii) self supervised proxies that rely only on the objective and constraint structure of the underlying optimization problem. By augmenting the standard training loss with directional derivative information extracted from the solver, the proxy aligns both its predicted solutions and local derivatives with those of the optimizer. Under Lipschitz continuity assumptions on the true solution mapping, matching first order sensitivities is shown to yield uniform approximation error proportional to the training set covering radius. Empirically, different impacts are observed in each studied setting. On three large Alternating Current Optimal Power Flow benchmarks, supervised Sobolev training cuts mean squared error by up to 56 percent and the median worst case constraint violation by up to 400 percent while keeping the optimality gap below 0.22 percent. For a mean variance portfolio task trained without labeled solutions, self supervised Sobolev training halves the average optimality gap in the medium risk region (standard deviation above 10 percent of budget) and matches the baseline elsewhere. Together, these results highlight Sobolev training whether supervised or self supervised as a path to fast reliable surrogates for safety critical large scale optimization workloads. |
| 2025-05-16 | [Explaining Strategic Decisions in Multi-Agent Reinforcement Learning for Aerial Combat Tactics](http://arxiv.org/abs/2505.11311v1) | Ardian Selmonaj, Alessandro Antonucci et al. | Artificial intelligence (AI) is reshaping strategic planning, with Multi-Agent Reinforcement Learning (MARL) enabling coordination among autonomous agents in complex scenarios. However, its practical deployment in sensitive military contexts is constrained by the lack of explainability, which is an essential factor for trust, safety, and alignment with human strategies. This work reviews and assesses current advances in explainability methods for MARL with a focus on simulated air combat scenarios. We proceed by adapting various explainability techniques to different aerial combat scenarios to gain explanatory insights about the model behavior. By linking AI-generated tactics with human-understandable reasoning, we emphasize the need for transparency to ensure reliable deployment and meaningful human-machine interaction. By illuminating the crucial importance of explainability in advancing MARL for operational defense, our work supports not only strategic planning but also the training of military personnel with insightful and comprehensible analyses. |
| 2025-05-16 | [LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios](http://arxiv.org/abs/2505.11247v1) | Mingxing Peng, Yuting Xie et al. | Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios. |
| 2025-05-16 | [Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs](http://arxiv.org/abs/2505.11227v1) | Zhangying Feng, Qianglong Chen et al. | The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models. |
| 2025-05-16 | [HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization](http://arxiv.org/abs/2505.11225v1) | Chengyu Huang, Zhengxin Zhang et al. | While scaling the length of responses at test-time has been shown to markedly improve the reasoning abilities and performance of large language models (LLMs), it often results in verbose outputs and increases inference cost. Prior approaches for efficient test-time scaling, typically using universal budget constraints or query-level length optimization, do not leverage historical information from previous encounters with the same problem during training. We hypothesize that this limits their ability to progressively make solutions more concise over time. To address this, we present History-Aware Policy Optimization (HAPO), which keeps track of a history state (e.g., the minimum length over previously generated correct responses) for each problem. HAPO employs a novel length reward function based on this history state to incentivize the discovery of correct solutions that are more concise than those previously found. Crucially, this reward structure avoids overly penalizing shorter incorrect responses with the goal of facilitating exploration towards more efficient solutions. By combining this length reward with a correctness reward, HAPO jointly optimizes for correctness and efficiency. We use HAPO to train DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, and Qwen-2.5-1.5B-Instruct, and evaluate HAPO on several math benchmarks that span various difficulty levels. Experiment results demonstrate that HAPO effectively induces LLMs' concise reasoning abilities, producing length reductions of 33-59% with accuracy drops of only 2-5%. |
| 2025-05-16 | [Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration](http://arxiv.org/abs/2505.11191v1) | Kasra Borazjani, Payam Abdisarabshali et al. | As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: Foundation Models (FMs) provide a pathway toward generalization across tasks and modalities, whereas Federated Learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied environments. In this vision paper, we introduce Federated Foundation Models (FFMs) for embodied AI, a new paradigm that unifies the strengths of multi-modal multi-task (M3T) FMs with the privacy-preserving distributed nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying FFMs in embodied AI systems, along with the associated trade-offs. |
| 2025-05-16 | [Human-Aligned Bench: Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans](http://arxiv.org/abs/2505.11141v1) | Yansheng Qiu, Li Xiao et al. | The goal of achieving Artificial General Intelligence (AGI) is to imitate humans and surpass them. Models such as OpenAI's o1, o3, and DeepSeek's R1 have demonstrated that large language models (LLMs) with human-like reasoning capabilities exhibit exceptional performance and are being gradually integrated into multimodal large language models (MLLMs). However, whether these models possess capabilities comparable to humans in handling reasoning tasks remains unclear at present. In this paper, we propose Human-Aligned Bench, a benchmark for fine-grained alignment of multimodal reasoning with human performance. Specifically, we collected 9,794 multimodal questions that solely rely on contextual reasoning, including bilingual (Chinese and English) multimodal questions and pure text-based questions, encompassing four question types: visual reasoning, definition judgment, analogical reasoning, and logical judgment. More importantly, each question is accompanied by human success rates and options that humans are prone to choosing incorrectly. Extensive experiments on the Human-Aligned Bench reveal notable differences between the performance of current MLLMs in multimodal reasoning and human performance. The findings on our benchmark provide insights into the development of the next-generation models. |
| 2025-05-16 | [Scaling Reasoning can Improve Factuality in Large Language Models](http://arxiv.org/abs/2505.11140v1) | Mike Zhang, Johannes Bjerva et al. | Recent studies on large language model (LLM) reasoning capabilities have demonstrated promising improvements in model performance by leveraging a lengthy thinking process and additional computational resources during inference, primarily in tasks involving mathematical reasoning (Muennighoff et al., 2025). However, it remains uncertain if longer reasoning chains inherently enhance factual accuracy, particularly beyond mathematical contexts. In this work, we thoroughly examine LLM reasoning within complex open-domain question-answering (QA) scenarios. We initially distill reasoning traces from advanced, large-scale reasoning models (QwQ-32B and DeepSeek-R1-671B), then fine-tune a variety of models ranging from smaller, instruction-tuned variants to larger architectures based on Qwen2.5. To enrich reasoning traces, we introduce factual information from knowledge graphs in the form of paths into our reasoning traces. Our experimental setup includes four baseline approaches and six different instruction-tuned models evaluated across a benchmark of six datasets, encompassing over 22.6K questions. Overall, we carry out 168 experimental runs and analyze approximately 1.7 million reasoning traces. Our findings indicate that, within a single run, smaller reasoning models achieve noticeable improvements in factual accuracy compared to their original instruction-tuned counterparts. Moreover, our analysis demonstrates that adding test-time compute and token budgets factual accuracy consistently improves by 2-8%, further confirming the effectiveness of test-time scaling for enhancing performance and consequently improving reasoning accuracy in open-domain QA tasks. We release all the experimental artifacts for further research. |
| 2025-05-16 | [LLM-Enhanced Symbolic Control for Safety-Critical Applications](http://arxiv.org/abs/2505.11077v1) | Amir Bayat, Alessandro Abate et al. | Motivated by Smart Manufacturing and Industry 4.0, we introduce a framework for synthesizing Abstraction-Based Controller Design (ABCD) for reach-avoid problems from Natural Language (NL) specifications using Large Language Models (LLMs). A Code Agent interprets an NL description of the control problem and translates it into a formal language interpretable by state-of-the-art symbolic control software, while a Checker Agent verifies the correctness of the generated code and enhances safety by identifying specification mismatches. Evaluations show that the system handles linguistic variability and improves robustness over direct planning with LLMs. The proposed approach lowers the barrier to formal control synthesis by enabling intuitive, NL-based task definition while maintaining safety guarantees through automated validation. |
| 2025-05-16 | [A Multi-modal Fusion Network for Terrain Perception Based on Illumination Aware](http://arxiv.org/abs/2505.11066v1) | Rui Wang, Shichun Yang et al. | Road terrains play a crucial role in ensuring the driving safety of autonomous vehicles (AVs). However, existing sensors of AVs, including cameras and Lidars, are susceptible to variations in lighting and weather conditions, making it challenging to achieve real-time perception of road conditions. In this paper, we propose an illumination-aware multi-modal fusion network (IMF), which leverages both exteroceptive and proprioceptive perception and optimizes the fusion process based on illumination features. We introduce an illumination-perception sub-network to accurately estimate illumination features. Moreover, we design a multi-modal fusion network which is able to dynamically adjust weights of different modalities according to illumination features. We enhance the optimization process by pre-training of the illumination-perception sub-network and incorporating illumination loss as one of the training constraints. Extensive experiments demonstrate that the IMF shows a superior performance compared to state-of-the-art methods. The comparison results with single modality perception methods highlight the comprehensive advantages of multi-modal fusion in accurately perceiving road terrains under varying lighting conditions. Our dataset is available at: https://github.com/lindawang2016/IMF. |
| 2025-05-16 | [Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](http://arxiv.org/abs/2505.11063v1) | Changyue Jiang, Xudong Pan et al. | LLM-based autonomous agents possess capabilities such as reasoning, tool invocation, and environment interaction, enabling the execution of complex multi-step tasks. The internal reasoning process, i.e., thought, of behavioral trajectory significantly influences tool usage and subsequent actions but can introduce potential risks. Even minor deviations in the agent's thought may trigger cascading effects leading to irreversible safety incidents. To address the safety alignment challenges in long-horizon behavioral trajectories, we propose Thought-Aligner, a plug-in dynamic thought correction module. Utilizing a lightweight and resource-efficient model, Thought-Aligner corrects each high-risk thought on the fly before each action execution. The corrected thought is then reintroduced to the agent, ensuring safer subsequent decisions and tool interactions. Importantly, Thought-Aligner modifies only the reasoning phase without altering the underlying agent framework, making it easy to deploy and widely applicable to various agent frameworks. To train the Thought-Aligner model, we construct an instruction dataset across ten representative scenarios and simulate ReAct execution trajectories, generating 5,000 diverse instructions and more than 11,400 safe and unsafe thought pairs. The model is fine-tuned using contrastive learning techniques. Experiments across three agent safety benchmarks involving 12 different LLMs demonstrate that Thought-Aligner raises agent behavioral safety from approximately 50% in the unprotected setting to 90% on average. Additionally, Thought-Aligner maintains response latency below 100ms with minimal resource usage, demonstrating its capability for efficient deployment, broad applicability, and timely responsiveness. This method thus provides a practical dynamic safety solution for the LLM-based agents. |
| 2025-05-16 | [GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning](http://arxiv.org/abs/2505.11049v1) | Yue Liu, Shengfang Zhai et al. | To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model's reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average. We release data, code, and models (3B/7B) of GuardReasoner-VL at https://github.com/yueliu1999/GuardReasoner-VL/ |
| 2025-05-16 | [GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models](http://arxiv.org/abs/2505.10983v1) | Haozheng Luo, Chenghao Qiu et al. | We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. |
| 2025-05-16 | [Exploration of amorphous V$_2$O$_5$ as cathode for magnesium batteries](http://arxiv.org/abs/2505.10967v1) | Vijay Choyal, Debsundar Dey et al. | Development of energy storage technologies that can exhibit higher energy densities, better safety, and lower supply-chain constraints than the current state-of-the-art Li-ion batteries (LIBs) is crucial for our transition into sustainable energy use. In this context, Mg batteries (MBs) offer a promising pathway to design energy storage systems with superior volumetric energy densities than LIBs but require the development of positive electrodes (cathodes) exhibiting high energy and power densities. Notably, amorphous materials that lack long range order can exhibit `flatter' potential energy surfaces than crystalline frameworks, possibly resulting in faster Mg$^{2+}$ motion. Here, we use a combination of ab initio molecular dynamics (AIMD), and machine learned interatomic potential (MLIP) based calculations to explore amorphous V$_2$O$_5$ as a potential cathode for MBs. Using an AIMD-generated dataset, we train and validate moment tensor potentials that can accurately model amorphous (Mg)V$_2$O$_5$ Due to the amorphization of V$_2$O$_5$, we observe a 10-14% drop in the average Mg intercalation voltage $-$ but the voltage remains higher than sulfide Mg cathodes. Importantly, we find a $\sim$seven (five) orders of magnitude higher Mg$^{2+}$ diffusivity in amorphous MgV$_2$O$_5$ than its crystalline version (thiospinel-Mg$_x$Ti$_2$S$_4$), which is directly attributable to the amorphization of the structure. Also, we note the Mg$^{2+}$ motion in the amorphous structure is significantly cross-correlated at low temperatures, with the correlation decreasing with increase in temperature. Thus, our work highlights the potential of amorphous V$_2$O$_5$ as a cathode that can exhibit both high energy and power densities, resulting in the practical deployment of MBs. |

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



