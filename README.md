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
| 2025-05-15 | [Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning](http://arxiv.org/abs/2505.10547v1) | Milan Ganai, Rohan Sinha et al. | Foundation models can provide robust high-level reasoning on appropriate safety interventions in hazardous scenarios beyond a robot's training data, i.e. out-of-distribution (OOD) failures. However, due to the high inference latency of Large Vision and Language Models, current methods rely on manually defined intervention policies to enact fallbacks, thereby lacking the ability to plan generalizable, semantically safe motions. To overcome these challenges we present FORTRESS, a framework that generates and reasons about semantically safe fallback strategies in real time to prevent OOD failures. At a low frequency in nominal operations, FORTRESS uses multi-modal reasoners to identify goals and anticipate failure modes. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. |
| 2025-05-15 | [Large Language Models for Cancer Communication: Evaluating Linguistic Quality, Safety, and Accessibility in Generative AI](http://arxiv.org/abs/2505.10472v1) | Agnik Saha, Victoria Churchill et al. | Effective communication about breast and cervical cancers remains a persistent health challenge, with significant gaps in public understanding of cancer prevention, screening, and treatment, potentially leading to delayed diagnoses and inadequate treatments. This study evaluates the capabilities and limitations of Large Language Models (LLMs) in generating accurate, safe, and accessible cancer-related information to support patient understanding. We evaluated five general-purpose and three medical LLMs using a mixed-methods evaluation framework across linguistic quality, safety and trustworthiness, and communication accessibility and affectiveness. Our approach utilized quantitative metrics, qualitative expert ratings, and statistical analysis using Welch's ANOVA, Games-Howell, and Hedges' g. Our results show that general-purpose LLMs produced outputs of higher linguistic quality and affectiveness, while medical LLMs demonstrate greater communication accessibility. However, medical LLMs tend to exhibit higher levels of potential harm, toxicity, and bias, reducing their performance in safety and trustworthiness. Our findings indicate a duality between domain-specific knowledge and safety in health communications. The results highlight the need for intentional model design with targeted improvements, particularly in mitigating harm and bias, and improving safety and affectiveness. This study provides a comprehensive evaluation of LLMs for cancer communication, offering critical insights for improving AI-generated health content and informing future development of accurate, safe, and accessible digital health tools. |
| 2025-05-15 | [Formalising Human-in-the-Loop: Computational Reductions, Failure Modes, and Legal-Moral Responsibility](http://arxiv.org/abs/2505.10426v1) | Maurice Chiodo, Dennis Müller et al. | The legal compliance and safety of different Human-in-the-loop (HITL) setups for AI can vary greatly. This manuscript aims to identify new ways of choosing between such setups, and shows that there is an unavoidable trade-off between the attribution of legal responsibility and the technical explainability of AI. We begin by using the notion of oracle machines from computability theory to formalise different HITL setups, distinguishing between trivial human monitoring, single endpoint human action, and highly involved interaction between the human(s) and the AI. These correspond to total functions, many-one reductions, and Turing reductions respectively. A taxonomy categorising HITL failure modes is then presented, highlighting the limitations on what any HITL setup can actually achieve. Our approach then identifies oversights from UK and EU legal frameworks, which focus on certain HITL setups which may not always achieve the desired ethical, legal, and sociotechnical outcomes. We suggest areas where the law should recognise the effectiveness of different HITL setups and assign responsibility in these contexts, avoiding unnecessary and unproductive human "scapegoating". Overall, we show how HITL setups involve many technical design decisions, and can be prone to failures which are often out of the humans' control. This opens up a new analytic perspective on the challenges arising in the creation of HITL setups, helping inform AI developers and lawmakers on designing HITL to better achieve their desired outcomes. |
| 2025-05-15 | [FactsR: A Safer Method for Producing High Quality Healthcare Documentation](http://arxiv.org/abs/2505.10360v1) | Victor Petrén Bach Hansen, Lasse Krogsbøll et al. | There are now a multitude of AI-scribing solutions for healthcare promising the utilization of large language models for ambient documentation. However, these AI scribes still rely on one-shot, or few-shot prompts for generating notes after the consultation has ended, employing little to no reasoning. This risks long notes with an increase in hallucinations, misrepresentation of the intent of the clinician, and reliance on the proofreading of the clinician to catch errors. A dangerous combination for patient safety if vigilance is compromised by workload and fatigue. In this paper, we introduce a method for extracting salient clinical information in real-time alongside the healthcare consultation, denoted Facts, and use that information recursively to generate the final note. The FactsR method results in more accurate and concise notes by placing the clinician-in-the-loop of note generation, while opening up new use cases within real-time decision support. |
| 2025-05-15 | [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](http://arxiv.org/abs/2505.10320v1) | Chenxi Whitehouse, Tianlu Wang et al. | The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses. |
| 2025-05-15 | [AttentionGuard: Transformer-based Misbehavior Detection for Secure Vehicular Platoons](http://arxiv.org/abs/2505.10273v1) | Hexu Li, Konstantinos Kalogiannis et al. | Vehicle platooning, with vehicles traveling in close formation coordinated through Vehicle-to-Everything (V2X) communications, offers significant benefits in fuel efficiency and road utilization. However, it is vulnerable to sophisticated falsification attacks by authenticated insiders that can destabilize the formation and potentially cause catastrophic collisions. This paper addresses this challenge: misbehavior detection in vehicle platooning systems. We present AttentionGuard, a transformer-based framework for misbehavior detection that leverages the self-attention mechanism to identify anomalous patterns in mobility data. Our proposal employs a multi-head transformer-encoder to process sequential kinematic information, enabling effective differentiation between normal mobility patterns and falsification attacks across diverse platooning scenarios, including steady-state (no-maneuver) operation, join, and exit maneuvers. Our evaluation uses an extensive simulation dataset featuring various attack vectors (constant, gradual, and combined falsifications) and operational parameters (controller types, vehicle speeds, and attacker positions). Experimental results demonstrate that AttentionGuard achieves up to 0.95 F1-score in attack detection, with robust performance maintained during complex maneuvers. Notably, our system performs effectively with minimal latency (100ms decision intervals), making it suitable for real-time transportation safety applications. Comparative analysis reveals superior detection capabilities and establishes the transformer-encoder as a promising approach for securing Cooperative Intelligent Transport Systems (C-ITS) against sophisticated insider threats. |
| 2025-05-15 | [Sage Deer: A Super-Aligned Driving Generalist Is Your Copilot](http://arxiv.org/abs/2505.10257v1) | Hao Lu, Jiaqi Tang et al. | The intelligent driving cockpit, an important part of intelligent driving, needs to match different users' comfort, interaction, and safety needs. This paper aims to build a Super-Aligned and GEneralist DRiving agent, SAGE DeeR. Sage Deer achieves three highlights: (1) Super alignment: It achieves different reactions according to different people's preferences and biases. (2) Generalist: It can understand the multi-view and multi-mode inputs to reason the user's physiological indicators, facial emotions, hand movements, body movements, driving scenarios, and behavioral decisions. (3) Self-Eliciting: It can elicit implicit thought chains in the language space to further increase generalist and super-aligned abilities. Besides, we collected multiple data sets and built a large-scale benchmark. This benchmark measures the deer's perceptual decision-making ability and the super alignment's accuracy. |
| 2025-05-15 | [Towards Safe Robot Foundation Models Using Inductive Biases](http://arxiv.org/abs/2505.10219v1) | Maximilian Tölle, Theo Gruner et al. | Safety is a critical requirement for the real-world deployment of robotic systems. Unfortunately, while current robot foundation models show promising generalization capabilities across a wide variety of tasks, they fail to address safety, an important aspect for ensuring long-term operation. Current robot foundation models assume that safe behavior should emerge by learning from a sufficiently large dataset of demonstrations. However, this approach has two clear major drawbacks. Firstly, there are no formal safety guarantees for a behavior cloning policy trained using supervised learning. Secondly, without explicit knowledge of any safety constraints, the policy may require an unreasonable number of additional demonstrations to even approximate the desired constrained behavior. To solve these key issues, we show how we can instead combine robot foundation models with geometric inductive biases using ATACOM, a safety layer placed after the foundation policy that ensures safe state transitions by enforcing action constraints. With this approach, we can ensure formal safety guarantees for generalist policies without providing extensive demonstrations of safe behavior, and without requiring any specific fine-tuning for safety. Our experiments show that our approach can be beneficial both for classical manipulation tasks, where we avoid unwanted collisions with irrelevant objects, and for dynamic tasks, such as the robot air hockey environment, where we can generate fast trajectories respecting complex tasks and joint space constraints. |
| 2025-05-15 | [LanTu: Dynamics-Enhanced Deep Learning for Eddy-Resolving Ocean Forecasting](http://arxiv.org/abs/2505.10191v1) | Qingyu Zheng, Qi Shao et al. | Mesoscale eddies dominate the spatiotemporal multiscale variability of the ocean, and their impact on the energy cascade of the global ocean cannot be ignored. Eddy-resolving ocean forecasting is providing more reliable protection for fisheries and navigational safety, but also presents significant scientific challenges and high computational costs for traditional numerical models. Artificial intelligence (AI)-based weather and ocean forecasting systems are becoming powerful tools that balance forecast performance with computational efficiency. However, the complex multiscale features in the ocean dynamical system make AI models still face many challenges in mesoscale eddy forecasting (especially regional modelling). Here, we develop LanTu, a regional eddy-resolving ocean forecasting system based on dynamics-enhanced deep learning. We incorporate cross-scale interactions into LanTu and construct multiscale physical constraint for optimising LanTu guided by knowledge of eddy dynamics in order to improve the forecasting skill of LanTu for mesoscale evolution. The results show that LanTu outperforms the existing advanced operational numerical ocean forecasting system (NOFS) and AI-based ocean forecasting system (AI-OFS) in temperature, salinity, sea level anomaly and current prediction, with a lead time of more than 10 days. Our study highlights that dynamics-enhanced deep learning (LanTu) can be a powerful paradigm for eddy-resolving ocean forecasting. |
| 2025-05-15 | [Closure and Complexity of Temporal Causality](http://arxiv.org/abs/2505.10186v1) | Mishel Carelli, Bernd Finkbeiner et al. | Temporal causality defines what property causes some observed temporal behavior (the effect) in a given computation, based on a counterfactual analysis of similar computations. In this paper, we study its closure properties and the complexity of computing causes. For the former, we establish that safety, reachability, and recurrence properties are all closed under causal inference: If the effect is from one of these property classes, then the cause for this effect is from the same class. We also show that persistence and obligation properties are not closed in this way. These results rest on a topological characterization of causes which makes them applicable to a wide range of similarity relations between computations. Finally, our complexity analysis establishes improved upper bounds for computing causes for safety, reachability, and recurrence properties. We also present the first lower bounds for all of the classes. |
| 2025-05-15 | [Knowledge-Based Aerospace Engineering -- A Systematic Literature Review](http://arxiv.org/abs/2505.10142v1) | Tim Wittenborg, Ildar Baimuratov et al. | The aerospace industry operates at the frontier of technological innovation while maintaining high standards regarding safety and reliability. In this environment, with an enormous potential for re-use and adaptation of existing solutions and methods, Knowledge-Based Engineering (KBE) has been applied for decades. The objective of this study is to identify and examine state-of-the-art knowledge management practices in the field of aerospace engineering. Our contributions include: 1) A SWARM-SLR of over 1,000 articles with qualitative analysis of 164 selected articles, supported by two aerospace engineering domain expert surveys. 2) A knowledge graph of over 700 knowledge-based aerospace engineering processes, software, and data, formalized in the interoperable Web Ontology Language (OWL) and mapped to Wikidata entries where possible. The knowledge graph is represented on the Open Research Knowledge Graph (ORKG), and an aerospace Wikibase, for reuse and continuation of structuring aerospace engineering knowledge exchange. 3) Our resulting intermediate and final artifacts of the knowledge synthesis, available as a Zenodo dataset. This review sets a precedent for structured, semantic-based approaches to managing aerospace engineering knowledge. By advancing these principles, research, and industry can achieve more efficient design processes, enhanced collaboration, and a stronger commitment to sustainable aviation. |
| 2025-05-15 | [Dark LLMs: The Growing Threat of Unaligned AI Models](http://arxiv.org/abs/2505.10066v1) | Michael Fire, Yitzhak Elbazis et al. | Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated. |
| 2025-05-15 | [Enhancing performance in bolt torque tightening using a connected torque wrench and augmented reality](http://arxiv.org/abs/2505.10047v1) | Adeline Fau, Mina Ghobrial et al. | Modern production rates and the increasing complexity of mechanical systems require efficient and effective manufacturing and assembly processes. The transition to Industry 4.0, supported by the deployment of innovative tools such as Augmented Reality (AR), equips the industry to tackle future challenges. Among critical processes, the assembly and tightening of bolted joints stand out due to their significant safety and economic implications across various industrial sectors. This study proposes an innovative tightening method designed to enhance the reliability of bolted assembly tightening through the use of Augmented Reality and connected tools. A 6-Degrees-of-Freedom (6-DoF) tracked connected torque wrench assists the operator during tightening, ensuring each screw is tightened to the correct torque. The effectiveness of this method is compared with the conventional tightening method using paper instructions. Participants in the study carried out tightening sequences on two simple parts with multiple screws. The study evaluates the impact of the proposed method on task performance and its acceptability to operators. The tracked connected torque wrench provides considerable assistance to the operators, including wrench control and automatic generation of tightening reports. The results suggest that the AR-based method has the potential to ensure reliable torque tightening of bolted joints. |
| 2025-05-15 | [Application of YOLOv8 in monocular downward multiple Car Target detection](http://arxiv.org/abs/2505.10016v1) | Shijie Lyu | Autonomous driving technology is progressively transforming traditional car driving methods, marking a significant milestone in modern transportation. Object detection serves as a cornerstone of autonomous systems, playing a vital role in enhancing driving safety, enabling autonomous functionality, improving traffic efficiency, and facilitating effective emergency responses. However, current technologies such as radar for environmental perception, cameras for road perception, and vehicle sensor networks face notable challenges, including high costs, vulnerability to weather and lighting conditions, and limited resolution.To address these limitations, this paper presents an improved autonomous target detection network based on YOLOv8. By integrating structural reparameterization technology, a bidirectional pyramid structure network model, and a novel detection pipeline into the YOLOv8 framework, the proposed approach achieves highly efficient and precise detection of multi-scale, small, and remote objects. Experimental results demonstrate that the enhanced model can effectively detect both large and small objects with a detection accuracy of 65%, showcasing significant advancements over traditional methods.This improved model holds substantial potential for real-world applications and is well-suited for autonomous driving competitions, such as the Formula Student Autonomous China (FSAC), particularly excelling in scenarios involving single-target and small-object detection. |
| 2025-05-15 | [A Survey on Open-Source Edge Computing Simulators and Emulators: The Computing and Networking Convergence Perspective](http://arxiv.org/abs/2505.09995v1) | Jianpeng Qi, Chao Liu et al. | Edge computing, with its low latency, dynamic scalability, and location awareness, along with the convergence of computing and communication paradigms, has been successfully applied in critical domains such as industrial IoT, smart healthcare, smart homes, and public safety. This paper provides a comprehensive survey of open-source edge computing simulators and emulators, presented in our GitHub repository (https://github.com/qijianpeng/awesome-edge-computing), emphasizing the convergence of computing and networking paradigms. By examining more than 40 tools, including CloudSim, NS-3, and others, we identify the strengths and limitations in simulating and emulating edge environments. This survey classifies these tools into three categories: packet-level, application-level, and emulators. Furthermore, we evaluate them across five dimensions, ranging from resource representation to resource utilization. The survey highlights the integration of different computing paradigms, packet processing capabilities, support for edge environments, user-defined metric interfaces, and scenario visualization. The findings aim to guide researchers in selecting appropriate tools for developing and validating advanced computing and networking technologies. |
| 2025-05-15 | [Provably safe and human-like car-following behaviors: Part 2. A parsimonious multi-phase model with projected braking](http://arxiv.org/abs/2505.09988v1) | Wen-Long Jin | Ensuring safe and human-like trajectory planning for automated vehicles amidst real-world uncertainties remains a critical challenge. While existing car-following models often struggle to consistently provide rigorous safety proofs alongside human-like acceleration and deceleration patterns, we introduce a novel multi-phase projection-based car-following model. This model is designed to balance safety and performance by incorporating bounded acceleration and deceleration rates while emulating key human driving principles. Building upon a foundation of fundamental driving principles and a multi-phase dynamical systems analysis (detailed in Part 1 of this study \citep{jin2025WA20-02_Part1}), we first highlight the limitations of extending standard models like Newell's with simple bounded deceleration. Inspired by human drivers' anticipatory behavior, we mathematically define and analyze projected braking profiles for both leader and follower vehicles, establishing safety criteria and new phase definitions based on the projected braking lead-vehicle problem. The proposed parsimonious model combines an extended Newell's model for nominal driving with a new control law for scenarios requiring projected braking. Using speed-spacing phase plane analysis, we provide rigorous mathematical proofs of the model's adherence to defined safe and human-like driving principles, including collision-free operation, bounded deceleration, and acceptable safe stopping distance, under reasonable initial conditions. Numerical simulations validate the model's superior performance in achieving both safety and human-like braking profiles for the stationary lead-vehicle problem. Finally, we discuss the model's implications and future research directions. |
| 2025-05-15 | [Provably safe and human-like car-following behaviors: Part 1. Analysis of phases and dynamics in standard models](http://arxiv.org/abs/2505.09987v1) | Wen-Long Jin | Trajectory planning is essential for ensuring safe driving in the face of uncertainties related to communication, sensing, and dynamic factors such as weather, road conditions, policies, and other road users. Existing car-following models often lack rigorous safety proofs and the ability to replicate human-like driving behaviors consistently. This article applies multi-phase dynamical systems analysis to well-known car-following models to highlight the characteristics and limitations of existing approaches. We begin by formulating fundamental principles for safe and human-like car-following behaviors, which include zeroth-order principles for comfort and minimum jam spacings, first-order principles for speeds and time gaps, and second-order principles for comfort acceleration/deceleration bounds as well as braking profiles. From a set of these zeroth- and first-order principles, we derive Newell's simplified car-following model. Subsequently, we analyze phases within the speed-spacing plane for the stationary lead-vehicle problem in Newell's model and its extensions, which incorporate both bounded acceleration and deceleration. We then analyze the performance of the Intelligent Driver Model and the Gipps model. Through this analysis, we highlight the limitations of these models with respect to some of the aforementioned principles. Numerical simulations and empirical observations validate the theoretical insights. Finally, we discuss future research directions to further integrate safety, human-like behaviors, and vehicular automation in car-following models, which are addressed in Part 2 of this study \citep{jin2025WA20-02_Part2}, where we develop a novel multi-phase projection-based car-following model that addresses the limitations identified here. |
| 2025-05-15 | [Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data](http://arxiv.org/abs/2505.09974v1) | Adel ElZemity, Budi Arief et al. | The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. We present a systematic evaluation of safety risks in fine-tuned LLMs for cyber security applications. Using the OWASP Top 10 for LLM Applications framework, we assessed seven open-source LLMs: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. Our evaluation shows that fine-tuning reduces safety resilience across all tested LLMs (e.g., the safety score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). We propose and evaluate a safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path forward for developing safer fine-tuning methodologies. This work offers a systematic evaluation for safety risks in LLMs, enabling safer adoption of generative AI in sensitive domains, and contributing towards the development of secure, trustworthy, and ethically aligned LLMs. |
| 2025-05-15 | [Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents](http://arxiv.org/abs/2505.09970v1) | Mrinal Rawat, Ambuje Gupta et al. | The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset. |
| 2025-05-15 | [Advanced Crash Causation Analysis for Freeway Safety: A Large Language Model Approach to Identifying Key Contributing Factors](http://arxiv.org/abs/2505.09949v1) | Ahmed S. Abdelrahman, Mohamed Abdel-Aty et al. | Understanding the factors contributing to traffic crashes and developing strategies to mitigate their severity is essential. Traditional statistical methods and machine learning models often struggle to capture the complex interactions between various factors and the unique characteristics of each crash. This research leverages large language model (LLM) to analyze freeway crash data and provide crash causation analysis accordingly. By compiling 226 traffic safety studies related to freeway crashes, a training dataset encompassing environmental, driver, traffic, and geometric design factors was created. The Llama3 8B model was fine-tuned using QLoRA to enhance its understanding of freeway crashes and their contributing factors, as covered in these studies. The fine-tuned Llama3 8B model was then used to identify crash causation without pre-labeled data through zero-shot classification, providing comprehensive explanations to ensure that the identified causes were reasonable and aligned with existing research. Results demonstrate that LLMs effectively identify primary crash causes such as alcohol-impaired driving, speeding, aggressive driving, and driver inattention. Incorporating event data, such as road maintenance, offers more profound insights. The model's practical applicability and potential to improve traffic safety measures were validated by a high level of agreement among researchers in the field of traffic safety, as reflected in questionnaire results with 88.89%. This research highlights the complex nature of traffic crashes and how LLMs can be used for comprehensive analysis of crash causation and other contributing factors. Moreover, it provides valuable insights and potential countermeasures to aid planners and policymakers in developing more effective and efficient traffic safety practices. |

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



