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
| 2025-05-02 | [Neutrino mass generation in asymptotically safe gravity](http://arxiv.org/abs/2505.01422v1) | Gustavo P. de Brito, Astrid Eichhorn et al. | There exist several distinct phenomenological models to generate neutrino masses. We explore, which of these models can consistently be embedded in a quantum theory of gravity and matter. We proceed by invoking a minimal number of degrees of freedom beyond the Standard Model. Thus, we first investigate whether the Weinberg operator, a dimension-five-operator that generates neutrino masses without requiring degrees of freedom beyond the Standard Model, can arise in asymptotically safe quantum gravity. We find a negative answer with far-reaching consequences: new degrees of freedom beyond gravity and the Standard Model are necessary to give neutrinos a mass in the asymptotic-safety paradigm. Second, we explore whether the type-I Seesaw mechanism is viable and discover an upper bound on the Seesaw scale. The bound depends on the mass of the visible neutrino. We find a numerical value of $10^{14}\, \rm GeV$ for this bound when neglecting neutrino mixing for a visible mass of $10^{-10}\, \rm GeV$. Conversely, for the most ``natural" value of the Seesaw scale in a quantum-gravity setting, which is the Planck scale, we predict an upper bound for the neutrino mass of the visible neutrino of approximately $10^{-15}\, \rm GeV$. Third, we explore whether neutrinos could also be Pseudo-Dirac-neutrinos in asymptotic safety and find that this possibility can be accommodated. |
| 2025-05-02 | [Evaluating Frontier Models for Stealth and Situational Awareness](http://arxiv.org/abs/2505.01420v1) | Mary Phuong, Roland S. Zimmermann et al. | Recent work has demonstrated the plausibility of frontier AI models scheming -- knowingly and covertly pursuing an objective misaligned with its developer's intentions. Such behavior could be very hard to detect, and if present in future advanced systems, could pose severe loss of control risk. It is therefore important for AI developers to rule out harm from scheming prior to model deployment. In this paper, we present a suite of scheming reasoning evaluations measuring two types of reasoning capabilities that we believe are prerequisites for successful scheming: First, we propose five evaluations of ability to reason about and circumvent oversight (stealth). Second, we present eleven evaluations for measuring a model's ability to instrumentally reason about itself, its environment and its deployment (situational awareness). We demonstrate how these evaluations can be used as part of a scheming inability safety case: a model that does not succeed on these evaluations is almost certainly incapable of causing severe harm via scheming in real deployment. We run our evaluations on current frontier models and find that none of them show concerning levels of either situational awareness or stealth. |
| 2025-05-02 | [Helping Big Language Models Protect Themselves: An Enhanced Filtering and Summarization System](http://arxiv.org/abs/2505.01315v1) | Sheikh Samit Muhaimin, Spyridon Mastorakis | The recent growth in the use of Large Language Models has made them vulnerable to sophisticated adversarial assaults, manipulative prompts, and encoded malicious inputs. Existing countermeasures frequently necessitate retraining models, which is computationally costly and impracticable for deployment. Without the need for retraining or fine-tuning, this study presents a unique defense paradigm that allows LLMs to recognize, filter, and defend against adversarial or malicious inputs on their own. There are two main parts to the suggested framework: (1) A prompt filtering module that uses sophisticated Natural Language Processing (NLP) techniques, including zero-shot classification, keyword analysis, and encoded content detection (e.g. base64, hexadecimal, URL encoding), to detect, decode, and classify harmful inputs; and (2) A summarization module that processes and summarizes adversarial research literature to give the LLM context-aware defense knowledge. This approach strengthens LLMs' resistance to adversarial exploitation by fusing text extraction, summarization, and harmful prompt analysis. According to experimental results, this integrated technique has a 98.71% success rate in identifying harmful patterns, manipulative language structures, and encoded prompts. By employing a modest amount of adversarial research literature as context, the methodology also allows the model to react correctly to harmful inputs with a larger percentage of jailbreak resistance and refusal rate. While maintaining the quality of LLM responses, the framework dramatically increases LLM's resistance to hostile misuse, demonstrating its efficacy as a quick and easy substitute for time-consuming, retraining-based defenses. |
| 2025-05-02 | [Document Retrieval Augmented Fine-Tuning (DRAFT) for safety-critical software assessments](http://arxiv.org/abs/2505.01307v1) | Regan Bolton, Mohammadreza Sheikhfathollahi et al. | Safety critical software assessment requires robust assessment against complex regulatory frameworks, a process traditionally limited by manual evaluation. This paper presents Document Retrieval-Augmented Fine-Tuning (DRAFT), a novel approach that enhances the capabilities of a large language model (LLM) for safety-critical compliance assessment. DRAFT builds upon existing Retrieval-Augmented Generation (RAG) techniques by introducing a novel fine-tuning framework that accommodates our dual-retrieval architecture, which simultaneously accesses both software documentation and applicable reference standards. To fine-tune DRAFT, we develop a semi-automated dataset generation methodology that incorporates variable numbers of relevant documents with meaningful distractors, closely mirroring real-world assessment scenarios. Experiments with GPT-4o-mini demonstrate a 7% improvement in correctness over the baseline model, with qualitative improvements in evidence handling, response structure, and domain-specific reasoning. DRAFT represents a practical approach to improving compliance assessment systems while maintaining the transparency and evidence-based reasoning essential in regulatory domains. |
| 2025-05-02 | [Contactless pulse rate assessment: Results and insights for application in driving simulator](http://arxiv.org/abs/2505.01299v1) | Đorđe D. Nešković, Kristina Stojmenova Pečečnik et al. | Camera-based monitoring of Pulse Rate (PR) enables continuous and unobtrusive assessment of driver's state, allowing estimation of fatigue or stress that could impact traffic safety. Commonly used wearable Photoplethysmography (PPG) sensors, while effective, suffer from motion artifacts and user discomfort. This study explores the feasibility of non-contact PR assessment using facial video recordings captured by a Red, Green, and Blue (RGB) camera in a driving simulation environment. The proposed approach detects subtle skin color variations due to blood flow and compares extracted PR values against reference measurements from a wearable wristband Empatica E4. We evaluate the impact of Eulerian Video Magnification (EVM) on signal quality and assess statistical differences in PR between age groups. Data obtained from 80 recordings from 64 healthy subjects covering a PR range of 45-160 bpm are analyzed, and signal extraction accuracy is quantified using metrics, such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE). Results show that EVM slightly improves PR estimation accuracy, reducing MAE from 6.48 bpm to 5.04 bpm and RMSE from 7.84 bpm to 6.38 bpm. A statistically significant difference is found between older and younger groups with both video-based and ground truth evaluation procedures. Additionally, we discuss Empatica E4 bias and its potential impact on the overall assessment of contact measurements. Altogether the findings demonstrate the feasibility of camera-based PR monitoring in dynamic environments and its potential integration into driving simulators for real-time physiological assessment. |
| 2025-05-02 | [2DXformer: Dual Transformers for Wind Power Forecasting with Dual Exogenous Variables](http://arxiv.org/abs/2505.01286v1) | Yajuan Zhang, Jiahai Jiang et al. | Accurate wind power forecasting can help formulate scientific dispatch plans, which is of great significance for maintaining the safety, stability, and efficient operation of the power system. In recent years, wind power forecasting methods based on deep learning have focused on extracting the spatiotemporal correlations among data, achieving significant improvements in forecasting accuracy. However, they exhibit two limitations. First, there is a lack of modeling for the inter-variable relationships, which limits the accuracy of the forecasts. Second, by treating endogenous and exogenous variables equally, it leads to unnecessary interactions between the endogenous and exogenous variables, increasing the complexity of the model. In this paper, we propose the 2DXformer, which, building upon the previous work's focus on spatiotemporal correlations, addresses the aforementioned two limitations. Specifically, we classify the inputs of the model into three types: exogenous static variables, exogenous dynamic variables, and endogenous variables. First, we embed these variables as variable tokens in a channel-independent manner. Then, we use the attention mechanism to capture the correlations among exogenous variables. Finally, we employ a multi-layer perceptron with residual connections to model the impact of exogenous variables on endogenous variables. Experimental results on two real-world large-scale datasets indicate that our proposed 2DXformer can further improve the performance of wind power forecasting. The code is available in this repository: \href{https://github.com/jseaj/2DXformer}{https://github.com/jseaj/2DXformer}. |
| 2025-05-02 | [Self-moderation in the decentralized era: decoding blocking behavior on Bluesky](http://arxiv.org/abs/2505.01174v1) | Carlo Bono, Nick Liu et al. | Moderation and blocking behavior, both closely related to the mitigation of abuse and misinformation on social platforms, are fundamental mechanisms for maintaining healthy online communities. However, while centralized platforms typically employ top-down moderation, decentralized networks rely on users to self-regulate through mechanisms like blocking actions to safeguard their online experience. Given the novelty of the decentralized paradigm, addressing self-moderation is critical for understanding how community safety and user autonomy can be effectively balanced. This study examines user blocking on Bluesky, a decentralized social networking platform, providing a comprehensive analysis of over three months of user activity through the lens of blocking behaviour. We define profiles based on 86 features that describe user activity, content characteristics, and network interactions, addressing two primary questions: (1) Is the likelihood of a user being blocked inferable from their online behavior? and (2) What behavioral features are associated with an increased likelihood of being blocked? Our findings offer valuable insights and contribute with a robust analytical framework to advance research in moderation on decentralized social networks. |
| 2025-05-02 | [VTS-LLM: Domain-Adaptive LLM Agent for Enhancing Awareness in Vessel Traffic Services through Natural Language](http://arxiv.org/abs/2505.00989v1) | Sijin Sun, Liangbin Zhao et al. | Vessel Traffic Services (VTS) are essential for maritime safety and regulatory compliance through real-time traffic management. However, with increasing traffic complexity and the prevalence of heterogeneous, multimodal data, existing VTS systems face limitations in spatiotemporal reasoning and intuitive human interaction. In this work, we propose VTS-LLM Agent, the first domain-adaptive large LLM agent tailored for interactive decision support in VTS operations. We formalize risk-prone vessel identification as a knowledge-augmented Text-to-SQL task, combining structured vessel databases with external maritime knowledge. To support this, we construct a curated benchmark dataset consisting of a custom schema, domain-specific corpus, and a query-SQL test set in multiple linguistic styles. Our framework incorporates NER-based relational reasoning, agent-based domain knowledge injection, semantic algebra intermediate representation, and query rethink mechanisms to enhance domain grounding and context-aware understanding. Experimental results show that VTS-LLM outperforms both general-purpose and SQL-focused baselines under command-style, operational-style, and formal natural language queries, respectively. Moreover, our analysis provides the first empirical evidence that linguistic style variation introduces systematic performance challenges in Text-to-SQL modeling. This work lays the foundation for natural language interfaces in vessel traffic services and opens new opportunities for proactive, LLM-driven maritime real-time traffic management. |
| 2025-05-02 | [Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models](http://arxiv.org/abs/2505.00972v1) | Yuewen Mei, Tong Nie et al. | Simulation-based testing is crucial for validating autonomous vehicles (AVs), yet existing scenario generation methods either overfit to common driving patterns or operate in an offline, non-interactive manner that fails to expose rare, safety-critical corner cases. In this paper, we introduce an online, retrieval-augmented large language model (LLM) framework for generating safety-critical driving scenarios. Our method first employs an LLM-based behavior analyzer to infer the most dangerous intent of the background vehicle from the observed state, then queries additional LLM agents to synthesize feasible adversarial trajectories. To mitigate catastrophic forgetting and accelerate adaptation, we augment the framework with a dynamic memorization and retrieval bank of intent-planner pairs, automatically expanding its behavioral library when novel intents arise. Evaluations using the Waymo Open Motion Dataset demonstrate that our model reduces the mean minimum time-to-collision from 1.62 to 1.08 s and incurs a 75% collision rate, substantially outperforming baselines. |
| 2025-05-02 | [A SCADE Model Verification Method Based on B-Model Transformation](http://arxiv.org/abs/2505.00967v1) | Xili Hou, Keming Wang et al. | Due to the limitations of SCADE models in expressing and verifying abstract specifications in safety-critical systems, this study proposes a formal verification framework based on the B-Method. By establishing a semantic equivalence transformation mechanism from SCADE models to B models, a hierarchical mapping rule set is constructed, covering type systems, control flow structures, and state machines. This effectively addresses key technical challenges such as loop-equivalent transformation proof for high-order operators and modeling of temporal logic storage structures. The proposed method innovatively leverages the abstraction capabilities of B-Method in set theory and first-order logic, overcoming the constraints of native verification tools of SCADE in complex specification descriptions. It successfully verifies abstract specifications that are difficult to model directly in SCADE. Experimental results show that the transformed B models achieve a higher defect detection rate and improved verification efficiency in the ProB verification environment compared to the native verifier of SCADE, significantly enhancing the formal verification capability of safety-critical systems. This study provides a cross-model verification paradigm for embedded control systems in avionics, rail transportation, and other domains, demonstrating substantial engineering application value. |
| 2025-05-02 | [Llama-Nemotron: Efficient Reasoning Models](http://arxiv.org/abs/2505.00949v1) | Akhiad Bercovich, Itay Levy et al. | We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM. |
| 2025-05-02 | [SSRLBot: Designing and Developing an LLM-based Agent using Socially Shared Regulated Learning](http://arxiv.org/abs/2505.00945v1) | Xiaoshan Huang, Jie Gao et al. | Large language model (LLM)-based agents are increasingly used to support human experts by streamlining complex tasks and offering actionable insights. However, their application in multi-professional decision-making, particularly in teamwork contexts, remains underexplored. This design-based study addresses that gap by developing LLM functions to enhance collaboration, grounded in the Socially Shared Regulation of Learning (SSRL) framework and applied to medical diagnostic teamwork. SSRL emphasizes metacognitive, cognitive, motivational, and emotional processes in shared learning, focusing on how teams manage these processes to improve decision-making. This paper introduces SSRLBot, a prototype chatbot designed to help team members reflect on both their diagnostic performance and key SSRL skills. Its core functions include summarizing dialogues, analyzing SSRL behaviors, evaluating diagnostic outcomes, annotating SSRL markers in conversation, assessing their impact on performance, and identifying interpersonal regulatory dynamics. We compare SSRLBot's capabilities with those of Gemini-1.5, GPT-3.5, and Deepseek-R1 in a case study. SSRLBot demonstrates stronger alignment with SSRL theory, offering detailed evaluations that link behaviors to regulatory dimensions and suggesting improvements for collaboration. By integrating SSRL theory with LLM capabilities, SSRLBot contributes a novel tool for enhancing team-based decision-making and collaborative learning in high-stakes environments, such as medical education. |
| 2025-05-01 | [Learning Neural Control Barrier Functions from Offline Data with Conservatism](http://arxiv.org/abs/2505.00908v1) | Ihab Tabbara, Hussein Sibai | Safety filters, particularly those based on control barrier functions, have gained increased interest as effective tools for safe control of dynamical systems. Existing correct-by-construction synthesis algorithms, however, suffer from the curse of dimensionality. Deep learning approaches have been proposed in recent years to address this challenge. In this paper, we contribute to this line of work by proposing an algorithm for training control barrier functions from offline datasets. Our algorithm trains the filter to not only prevent the system from reaching unsafe states but also out-of-distribution ones, at which the filter would be unreliable. It is inspired by Conservative Q-learning, an offline reinforcement learning algorithm. We call its outputs Conservative Control Barrier Functions (CCBFs). Our empirical results demonstrate that CCBFs outperform existing methods in maintaining safety and out-of-distribution avoidance while minimally affecting task performance. |
| 2025-05-01 | [Inattentional Blindness with Augmented Reality HUDS: An On-road Study](http://arxiv.org/abs/2505.00879v1) | Nayara de Oliveira Faria, Joseph L. Gabbard | As the integration of augmented reality (AR) technology in head-up displays (HUDs) becomes more prevalent in vehicles, it is crucial to understand how to design and evaluate AR interfaces to ensure safety. With new AR displays capable of rendering images with larger field of views and at varying depths, the visual and cognitive separation between graphical and real-world visual stimuli will be increasingly more difficult to quantify as will drivers' ability to efficiently allocate visual attention between the two sets of stimuli. In this study, we present a user study that serves as a crucial first step in gaining insight into inattentional blindness while using AR in surface transportation, where understanding is currently limited. Our primary goal is to investigate how the visual demand of AR tasks influences drivers' ability to detect stimuli, and whether the nature of the stimuli itself plays a role in this effect. To address these questions, we designed an on-road user study aimed at producing a more realistic and ecologically valid understanding of the phenomenon.   Our results show that drivers' ability to timely detect stimuli in the environment decreased as the AR task visual demand increased demonstrated by both detection performance and inattentional blindness metrics. Further, inattentional blindness caused by AR displays appears to be more prevalent within drivers' central field of view. We conclude by discussing implications towards a safety-centric evaluation framework for AR HUDs. |
| 2025-05-01 | [From Texts to Shields: Convergence of Large Language Models and Cybersecurity](http://arxiv.org/abs/2505.00841v1) | Tao Li, Ya-Ting Yang et al. | This report explores the convergence of large language models (LLMs) and cybersecurity, synthesizing interdisciplinary insights from network security, artificial intelligence, formal methods, and human-centered design. It examines emerging applications of LLMs in software and network security, 5G vulnerability analysis, and generative security engineering. The report highlights the role of agentic LLMs in automating complex tasks, improving operational efficiency, and enabling reasoning-driven security analytics. Socio-technical challenges associated with the deployment of LLMs -- including trust, transparency, and ethical considerations -- can be addressed through strategies such as human-in-the-loop systems, role-specific training, and proactive robustness testing. The report further outlines critical research challenges in ensuring interpretability, safety, and fairness in LLM-based systems, particularly in high-stakes domains. By integrating technical advances with organizational and societal considerations, this report presents a forward-looking research agenda for the secure and effective adoption of LLMs in cybersecurity. |
| 2025-05-01 | [IberFire -- a detailed creation of a spatio-temporal dataset for wildfire risk assessment in Spain](http://arxiv.org/abs/2505.00837v1) | Julen Ercibengoa, Meritxell Gómez-Omella et al. | Wildfires pose a critical environmental issue to ecosystems, economies, and public safety, particularly in Mediterranean regions such as Spain. Accurate predictive models rely on high-resolution spatio-temporal data to capture the complex interplay of environmental and anthropogenic factors. To address the lack of localised and fine-grained datasets in Spain, this work introduces IberFire, a spatio-temporal datacube at 1 km x 1 km x 1-day resolution covering mainland Spain and the Balearic Islands from December 2007 to December 2024. IberFire integrates 260 features across eight main categories: auxiliary features, fire history, geography, topography, meteorology, vegetation indices, human activity, and land cover. All features are derived from open-access sources, ensuring transparency and real-time applicability. The data processing pipeline was implemented entirely using open-source tools, and the codebase has been made publicly available. This work not only enhances spatio-temporal granularity and feature diversity compared to existing European datacubes but also provides a reproducible methodology for constructing similar datasets. IberFire supports advanced wildfire risk modelling through Machine Learning (ML) and Deep Learning (DL) techniques, enables climate pattern analysis and informs strategic planning in fire prevention and land management. The dataset is publicly available on Zenodo to promote open research and collaboration. |
| 2025-05-01 | [HMCF: A Human-in-the-loop Multi-Robot Collaboration Framework Based on Large Language Models](http://arxiv.org/abs/2505.00820v1) | Zhaoxing Li, Wenbo Wu et al. | Rapid advancements in artificial intelligence (AI) have enabled robots to performcomplex tasks autonomously with increasing precision. However, multi-robot systems (MRSs) face challenges in generalization, heterogeneity, and safety, especially when scaling to large-scale deployments like disaster response. Traditional approaches often lack generalization, requiring extensive engineering for new tasks and scenarios, and struggle with managing diverse robots. To overcome these limitations, we propose a Human-in-the-loop Multi-Robot Collaboration Framework (HMCF) powered by large language models (LLMs). LLMs enhance adaptability by reasoning over diverse tasks and robot capabilities, while human oversight ensures safety and reliability, intervening only when necessary. Our framework seamlessly integrates human oversight, LLM agents, and heterogeneous robots to optimize task allocation and execution. Each robot is equipped with an LLM agent capable of understanding its capabilities, converting tasks into executable instructions, and reducing hallucinations through task verification and human supervision. Simulation results show that our framework outperforms state-of-the-art task planning methods, achieving higher task success rates with an improvement of 4.76%. Real-world tests demonstrate its robust zero-shot generalization feature and ability to handle diverse tasks and environments with minimal human intervention. |
| 2025-05-01 | [Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures](http://arxiv.org/abs/2505.00779v1) | Junwon Seo, Kensuke Nakamura et al. | Recent advances in generative world models have enabled classical safe control methods, such as Hamilton-Jacobi (HJ) reachability, to generalize to complex robotic systems operating directly from high-dimensional sensor observations. However, obtaining comprehensive coverage of all safety-critical scenarios during world model training is extremely challenging. As a result, latent safety filters built on top of these models may miss novel hazards and even fail to prevent known ones, overconfidently misclassifying risky out-of-distribution (OOD) situations as safe. To address this, we introduce an uncertainty-aware latent safety filter that proactively steers robots away from both known and unseen failures. Our key idea is to use the world model's epistemic uncertainty as a proxy for identifying unseen potential hazards. We propose a principled method to detect OOD world model predictions by calibrating an uncertainty threshold via conformal prediction. By performing reachability analysis in an augmented state space-spanning both the latent representation and the epistemic uncertainty-we synthesize a latent safety filter that can reliably safeguard arbitrary policies from both known and unseen safety hazards. In simulation and hardware experiments on vision-based control tasks with a Franka manipulator, we show that our uncertainty-aware safety filter preemptively detects potential unsafe scenarios and reliably proposes safe, in-distribution actions. Video results can be found on the project website at https://cmu-intentlab.github.io/UNISafe |
| 2025-05-01 | [Towards Autonomous Micromobility through Scalable Urban Simulation](http://arxiv.org/abs/2505.00690v1) | Wayne Wu, Honglin He et al. | Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations. |
| 2025-05-01 | [Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions](http://arxiv.org/abs/2505.00671v1) | Chenggang Wang, Xinyi Wang et al. | The safety of training task policies and their subsequent application using reinforcement learning (RL) methods has become a focal point in the field of safe RL. A central challenge in this area remains the establishment of theoretical guarantees for safety during both the learning and deployment processes. Given the successful implementation of Control Barrier Function (CBF)-based safety strategies in a range of control-affine robotic systems, CBF-based safe RL demonstrates significant promise for practical applications in real-world scenarios. However, integrating these two approaches presents several challenges. First, embedding safety optimization within the RL training pipeline requires that the optimization outputs be differentiable with respect to the input parameters, a condition commonly referred to as differentiable optimization, which is non-trivial to solve. Second, the differentiable optimization framework confronts significant efficiency issues, especially when dealing with multi-constraint problems. To address these challenges, this paper presents a CBF-based safe RL architecture that effectively mitigates the issues outlined above. The proposed approach constructs a continuous AND logic approximation for the multiple constraints using a single composite CBF. By leveraging this approximation, a close-form solution of the quadratic programming is derived for the policy network in RL, thereby circumventing the need for differentiable optimization within the end-to-end safe RL pipeline. This strategy significantly reduces computational complexity because of the closed-form solution while maintaining safety guarantees. Simulation results demonstrate that, in comparison to existing approaches relying on differentiable optimization, the proposed method significantly reduces training computational costs while ensuring provable safety throughout the training process. |

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



