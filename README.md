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
| 2025-06-17 | [Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior](http://arxiv.org/abs/2506.14762v1) | Chengyuan Zhang, Cathy Wu et al. | Accurate and interpretable car-following models are essential for traffic simulation and autonomous vehicle development. However, classical models like the Intelligent Driver Model (IDM) are fundamentally limited by their parsimonious and single-regime structure. They fail to capture the multi-modal nature of human driving, where a single driving state (e.g., speed, relative speed, and gap) can elicit many different driver actions. This forces the model to average across distinct behaviors, reducing its fidelity and making its parameters difficult to interpret. To overcome this, we introduce a regime-switching framework that allows driving behavior to be governed by different IDM parameter sets, each corresponding to an interpretable behavioral mode. This design enables the model to dynamically switch between interpretable behavioral modes, rather than averaging across diverse driving contexts. We instantiate the framework using a Factorial Hidden Markov Model with IDM dynamics (FHMM-IDM), which explicitly separates intrinsic driving regimes (e.g., aggressive acceleration, steady-state following) from external traffic scenarios (e.g., free-flow, congestion, stop-and-go) through two independent latent Markov processes. Bayesian inference via Markov chain Monte Carlo (MCMC) is used to jointly estimate the regime-specific parameters, transition dynamics, and latent state trajectories. Experiments on the HighD dataset demonstrate that FHMM-IDM uncovers interpretable structure in human driving, effectively disentangling internal driver actions from contextual traffic conditions and revealing dynamic regime-switching patterns. This framework provides a tractable and principled solution to modeling context-dependent driving behavior under uncertainty, offering improvements in the fidelity of traffic simulations, the efficacy of safety analyses, and the development of more human-centric ADAS. |
| 2025-06-17 | [Swarm-STL: A Framework for Motion Planning in Large-Scale, Multi-Swarm Systems](http://arxiv.org/abs/2506.14749v1) | Shiyu Cheng, Luyao Niu et al. | In multi-agent systems, signal temporal logic (STL) is widely used for path planning to accomplish complex objectives with formal safety guarantees. However, as the number of agents increases, existing approaches encounter significant computational challenges. Recognizing that many complex tasks require cooperation among multiple agents, we propose swarm STL specifications to describe the collective tasks that need to be achieved by a team of agents. Next, we address the motion planning problem for all the agents in two stages. First, we abstract a group of cooperating agents as a swarm and construct a reduced-dimension state space whose dimension does not increase with the number of agents. The path planning is performed at the swarm level, ensuring the safety and swarm STL specifications are satisfied. Then, we design low-level control strategies for agents within each swarm based on the path synthesized in the first step. The trajectories of agents generated by the two-step policy ensure satisfaction of the STL specifications. We evaluate our two-stage approach in both single-swarm and multi-swarm scenarios. The results demonstrate that all tasks are completed with safety guarantees. Compared to the baseline multi-agent planning approach, our method maintains computational efficiency as the number of agents increases, since the computational time scales with the number of swarms rather than the number of agents. |
| 2025-06-17 | [AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions](http://arxiv.org/abs/2506.14697v1) | Aishan Liu, Zonghao Ying et al. | The rapid advancement of vision-language models (VLMs) and their integration into embodied agents have unlocked powerful capabilities for decision-making. However, as these systems are increasingly deployed in real-world environments, they face mounting safety concerns, particularly when responding to hazardous instructions. In this work, we propose AGENTSAFE, the first comprehensive benchmark for evaluating the safety of embodied VLM agents under hazardous instructions. AGENTSAFE simulates realistic agent-environment interactions within a simulation sandbox and incorporates a novel adapter module that bridges the gap between high-level VLM outputs and low-level embodied controls. Specifically, it maps recognized visual entities to manipulable objects and translates abstract planning into executable atomic actions in the environment. Building on this, we construct a risk-aware instruction dataset inspired by Asimovs Three Laws of Robotics, including base risky instructions and mutated jailbroken instructions. The benchmark includes 45 adversarial scenarios, 1,350 hazardous tasks, and 8,100 hazardous instructions, enabling systematic testing under adversarial conditions ranging from perception, planning, and action execution stages. |
| 2025-06-17 | [AIRTBench: Measuring Autonomous AI Red Teaming Capabilities in Language Models](http://arxiv.org/abs/2506.14682v1) | Ads Dawson, Rob Mulla et al. | We introduce AIRTBench, an AI red teaming benchmark for evaluating language models' ability to autonomously discover and exploit Artificial Intelligence and Machine Learning (AI/ML) security vulnerabilities. The benchmark consists of 70 realistic black-box capture-the-flag (CTF) challenges from the Crucible challenge environment on the Dreadnode platform, requiring models to write python code to interact with and compromise AI systems. Claude-3.7-Sonnet emerged as the clear leader, solving 43 challenges (61% of the total suite, 46.9% overall success rate), with Gemini-2.5-Pro following at 39 challenges (56%, 34.3% overall), GPT-4.5-Preview at 34 challenges (49%, 36.9% overall), and DeepSeek R1 at 29 challenges (41%, 26.9% overall). Our evaluations show frontier models excel at prompt injection attacks (averaging 49% success rates) but struggle with system exploitation and model inversion challenges (below 26%, even for the best performers). Frontier models are far outpacing open-source alternatives, with the best truly open-source model (Llama-4-17B) solving 7 challenges (10%, 1.0% overall), though demonstrating specialized capabilities on certain hard challenges. Compared to human security researchers, large language models (LLMs) solve challenges with remarkable efficiency completing in minutes what typically takes humans hours or days-with efficiency advantages of over 5,000x on hard challenges. Our contribution fills a critical gap in the evaluation landscape, providing the first comprehensive benchmark specifically designed to measure and track progress in autonomous AI red teaming capabilities. |
| 2025-06-17 | [StreetLens: Enabling Human-Centered AI Agents for Neighborhood Assessment from Street View Imagery](http://arxiv.org/abs/2506.14670v1) | Jina Kim, Leeje Jang et al. | Traditionally, neighborhood studies have employed interviews, surveys, and manual image annotation guided by detailed protocols to identify environmental characteristics, including physical disorder, decay, street safety, and sociocultural symbols, and to examine their impact on developmental and health outcomes. While these methods yield rich insights, they are time-consuming and require intensive expert intervention. Recent technological advances, including vision-language models (VLMs), have begun to automate parts of this process; however, existing efforts are often ad hoc and lack adaptability across research designs and geographic contexts. In this demo paper, we present StreetLens, a human-centered, researcher-configurable workflow that embeds relevant social science expertise in a VLM for scalable neighborhood environmental assessments. StreetLens mimics the process of trained human coders by grounding the analysis in questions derived from established interview protocols, retrieving relevant street view imagery (SVI), and generating a wide spectrum of semantic annotations from objective features (e.g., the number of cars) to subjective perceptions (e.g., the sense of disorder in an image). By enabling researchers to define the VLM's role through domain-informed prompting, StreetLens places domain knowledge at the core of the analysis process. It also supports the integration of prior survey data to enhance robustness and expand the range of characteristics assessed across diverse settings. We provide a Google Colab notebook to make StreetLens accessible and extensible for researchers working with public or custom SVI datasets. StreetLens represents a shift toward flexible, agentic AI systems that work closely with researchers to accelerate and scale neighborhood studies. |
| 2025-06-17 | [NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving](http://arxiv.org/abs/2506.14589v1) | Ren Xin, Hongji Liu et al. | Integrating General Models (GMs) such as Large Language Models (LLMs), with Specialized Models (SMs) in autonomous driving tasks presents a promising approach to mitigating challenges in data diversity and model capacity of existing specialized driving models. However, this integration leads to problems of asynchronous systems, which arise from the distinct characteristics inherent in GMs and SMs. To tackle this challenge, we propose NetRoller, an adapter that incorporates a set of novel mechanisms to facilitate the seamless integration of GMs and specialized driving models. Specifically, our mechanisms for interfacing the asynchronous GMs and SMs are organized into three key stages. NetRoller first harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. It then applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation. At last, it employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of SMs through few-epoch fine-tuning. Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the GM. Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving. The code and models are available at https://github.com/Rex-sys-hk/NetRoller . |
| 2025-06-17 | [Modeling Uncertainty: From Simulink to Stochastic Hybrid Automata](http://arxiv.org/abs/2506.14581v1) | Pauline Blohm, Felix Schulz et al. | Simulink is widely used in industrial design processes to model increasingly complex embedded control systems. Thus, their formal analysis is highly desirable. However, this comes with two major challenges: First, Simulink models often provide an idealized view of real-life systems and omit uncertainties such as, aging, sensor noise or failures. Second, the semantics of Simulink is only informally defined. In this paper, we present an approach to formally analyze safety and performance of embedded control systems modeled in Simulink in the presence of uncertainty. To achieve this, we 1) model different types of uncertainties as stochastic Simulink subsystems and 2) extend an existing formalization of the Simulink semantics based on stochastic hybrid automata (SHA) by providing transformation rules for the stochastic subsystems. Our approach gives us access to established quantitative analysis techniques, like statistical model checking and reachability analysis. We demonstrate the applicability of our approach by analyzing safety and performance in the presence of uncertainty for two smaller case studies. |
| 2025-06-17 | [Object-Centric Neuro-Argumentative Learning](http://arxiv.org/abs/2506.14577v1) | Abdul Rahman Jacob, Avinash Kori et al. | Over the last decade, as we rely more on deep learning technologies to make critical decisions, concerns regarding their safety, reliability and interpretability have emerged. We introduce a novel Neural Argumentative Learning (NAL) architecture that integrates Assumption-Based Argumentation (ABA) with deep learning for image analysis. Our architecture consists of neural and symbolic components. The former segments and encodes images into facts using object-centric learning, while the latter applies ABA learning to develop ABA frameworks enabling predictions with images. Experiments on synthetic data show that the NAL architecture can be competitive with a state-of-the-art alternative. |
| 2025-06-17 | [Doppelgänger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack](http://arxiv.org/abs/2506.14539v1) | Daewon Kang, YeongHwan Shin et al. | Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelg\"anger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelg\"anger method. The experimental results demonstrate that the Doppelg\"anger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack. |
| 2025-06-17 | [Toward Safety-First Human-Like Decision Making for Autonomous Vehicles in Time-Varying Traffic Flow](http://arxiv.org/abs/2506.14502v1) | Xiao Wang, Junru Yu et al. | Despite the recent advancements in artificial intelligence technologies have shown great potential in improving transport efficiency and safety, autonomous vehicles(AVs) still face great challenge of driving in time-varying traffic flow, especially in dense and interactive situations. Meanwhile, human have free wills and usually do not make the same decisions even situate in the exactly same scenarios, leading to the data-driven methods suffer from poor migratability and high search cost problems, decreasing the efficiency and effectiveness of the behavior policy. In this research, we propose a safety-first human-like decision-making framework(SF-HLDM) for AVs to drive safely, comfortably, and social compatiblely in effiency. The framework integrates a hierarchical progressive framework, which combines a spatial-temporal attention (S-TA) mechanism for other road users' intention inference, a social compliance estimation module for behavior regulation, and a Deep Evolutionary Reinforcement Learning(DERL) model for expanding the search space efficiently and effectively to make avoidance of falling into the local optimal trap and reduce the risk of overfitting, thus make human-like decisions with interpretability and flexibility. The SF-HLDM framework enables autonomous driving AI agents dynamically adjusts decision parameters to maintain safety margins and adhering to contextually appropriate driving behaviors at the same time. |
| 2025-06-17 | [Enclosing Prototypical Variational Autoencoder for Explainable Out-of-Distribution Detection](http://arxiv.org/abs/2506.14390v1) | Conrad Orglmeister, Erik Bochinski et al. | Understanding the decision-making and trusting the reliability of Deep Machine Learning Models is crucial for adopting such methods to safety-relevant applications. We extend self-explainable Prototypical Variational models with autoencoder-based out-of-distribution (OOD) detection: A Variational Autoencoder is applied to learn a meaningful latent space which can be used for distance-based classification, likelihood estimation for OOD detection, and reconstruction. The In-Distribution (ID) region is defined by a Gaussian mixture distribution with learned prototypes representing the center of each mode. Furthermore, a novel restriction loss is introduced that promotes a compact ID region in the latent space without collapsing it into single points. The reconstructive capabilities of the Autoencoder ensure the explainability of the prototypes and the ID region of the classifier, further aiding the discrimination of OOD samples. Extensive evaluations on common OOD detection benchmarks as well as a large-scale dataset from a real-world railway application demonstrate the usefulness of the approach, outperforming previous methods. |
| 2025-06-17 | [Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning](http://arxiv.org/abs/2506.14387v1) | William F. Shen, Xinchi Qiu et al. | Existing work on mitigating catastrophic forgetting in large language model (LLM) fine-tuning has primarily focused on preserving specific data or tasks, while critically overlooking the degradation of essential capabilities instilled through safety alignment, particularly the model's ability to faithfully express ignorance. In this work, we show that this capability is significantly degraded during conventional fine-tuning, leading to undesired behaviors such as hallucinations. To address this novel but highly practical problem, we propose SEAT, a simple and effective fine-tuning approach that preserves both fine-tuning performance and the model's inherent ability to acknowledge its ignorance. SEAT integrates two key components: (1) sparse training that constrains activation drift, and (2) a novel entity perturbation method with KL-divergence regularization, designed to counter knowledge entanglement. Experimental results demonstrate that SEAT significantly outperforms baselines in preserving ignorance awareness while retaining fine-tuning performance, offering a more robust solution for LLM fine-tuning. |
| 2025-06-17 | [IntelliLung: Advancing Safe Mechanical Ventilation using Offline RL with Hybrid Actions and Clinically Aligned Rewards](http://arxiv.org/abs/2506.14375v1) | Muhammad Hamza Yousuf, Jason Li et al. | Invasive mechanical ventilation (MV) is a life-sustaining therapy for critically ill patients in the intensive care unit (ICU). However, optimizing its settings remains a complex and error-prone process due to patient-specific variability. While Offline Reinforcement Learning (RL) shows promise for MV control, current stateof-the-art (SOTA) methods struggle with the hybrid (continuous and discrete) nature of MV actions. Discretizing the action space limits available actions due to exponential growth in combinations and introduces distribution shifts that can compromise safety. In this paper, we propose optimizations that build upon prior work in action space reduction to address the challenges of discrete action spaces. We also adapt SOTA offline RL algorithms (IQL and EDAC) to operate directly on hybrid action spaces, thereby avoiding the pitfalls of discretization. Additionally, we introduce a clinically grounded reward function based on ventilator-free days and physiological targets, which provides a more meaningful optimization objective compared to traditional sparse mortality-based rewards. Our findings demonstrate that AI-assisted MV optimization may enhance patient safety and enable individualized lung support, representing a significant advancement toward intelligent, data-driven critical care solutions. |
| 2025-06-17 | [Excessive Reasoning Attack on Reasoning LLMs](http://arxiv.org/abs/2506.14374v1) | Wai Man Si, Mingjie Li et al. | Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models. |
| 2025-06-17 | [AviationLLM: An LLM-based Knowledge System for Aviation Training](http://arxiv.org/abs/2506.14336v1) | Jia'ang Wan, Feng Shen et al. | Aviation training is a core link in ensuring flight safety, improving industry efficiency and promoting sustainable development. It not only involves flight simulation but also requires the learning of a great deal of professional aviation theory knowledge. In the existing training system, the knowledge is mainly imparted by the the instructors. However, the number of instructors is limited and the professional answers obtained from the Internet are not accurate enough, resulting in low training efficiency. To address this, we introduced LLM, but the basic pre-trained model cannot provide accurate answers to professional fields, so we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generating superficially plausible but factually incorrect responses due to insufficient data coverage. To address this, we employ Direct Preference Optimization(DPO). This paper proposes Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO). We select open source pre-trained LLM Qwen and adapt it to aviation theory training through DPO-based domain alignment. Simultaneously, to mitigate hallucinations caused by training data biases, knowledge obsolescence, or domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG) technology that combines generative and retrieval models. RALA-DPO effectively retrieves relevant information from external knowledge bases and delivers precise and high-quality responses through the generative model. Experimental results demonstrate that RALA-DPO can improve accuracy in response to professional aviation knowledge. With integrated RAG mechanisms, this system can further improve the accuracy of answers and achieve zero-cost knowledge updates simultaneously. |
| 2025-06-17 | [ClutterDexGrasp: A Sim-to-Real System for General Dexterous Grasping in Cluttered Scenes](http://arxiv.org/abs/2506.14317v1) | Zeyuan Chen, Qiyang Yan et al. | Dexterous grasping in cluttered scenes presents significant challenges due to diverse object geometries, occlusions, and potential collisions. Existing methods primarily focus on single-object grasping or grasp-pose prediction without interaction, which are insufficient for complex, cluttered scenes. Recent vision-language-action models offer a potential solution but require extensive real-world demonstrations, making them costly and difficult to scale. To address these limitations, we revisit the sim-to-real transfer pipeline and develop key techniques that enable zero-shot deployment in reality while maintaining robust generalization. We propose ClutterDexGrasp, a two-stage teacher-student framework for closed-loop target-oriented dexterous grasping in cluttered scenes. The framework features a teacher policy trained in simulation using clutter density curriculum learning, incorporating both a novel geometry and spatially-embedded scene representation and a comprehensive safety curriculum, enabling general, dynamic, and safe grasping behaviors. Through imitation learning, we distill the teacher's knowledge into a student 3D diffusion policy (DP3) that operates on partial point cloud observations. To the best of our knowledge, this represents the first zero-shot sim-to-real closed-loop system for target-oriented dexterous grasping in cluttered scenes, demonstrating robust performance across diverse objects and layouts. More details and videos are available at https://clutterdexgrasp.github.io/. |
| 2025-06-17 | [Socially Aware Robot Crowd Navigation via Online Uncertainty-Driven Risk Adaptation](http://arxiv.org/abs/2506.14305v1) | Zhirui Sun, Xingrong Diao et al. | Navigation in human-robot shared crowded environments remains challenging, as robots are expected to move efficiently while respecting human motion conventions. However, many existing approaches emphasize safety or efficiency while overlooking social awareness. This article proposes Learning-Risk Model Predictive Control (LR-MPC), a data-driven navigation algorithm that balances efficiency, safety, and social awareness. LR-MPC consists of two phases: an offline risk learning phase, where a Probabilistic Ensemble Neural Network (PENN) is trained using risk data from a heuristic MPC-based baseline (HR-MPC), and an online adaptive inference phase, where local waypoints are sampled and globally guided by a Multi-RRT planner. Each candidate waypoint is evaluated for risk by PENN, and predictions are filtered using epistemic and aleatoric uncertainty to ensure robust decision-making. The safest waypoint is selected as the MPC input for real-time navigation. Extensive experiments demonstrate that LR-MPC outperforms baseline methods in success rate and social awareness, enabling robots to navigate complex crowds with high adaptability and low disruption. A website about this work is available at https://sites.google.com/view/lr-mpc. |
| 2025-06-17 | [synth-dacl: Does Synthetic Defect Data Enhance Segmentation Accuracy and Robustness for Real-World Bridge Inspections?](http://arxiv.org/abs/2506.14255v1) | Johannes Flotzinger, Fabian Deuser et al. | Adequate bridge inspection is increasingly challenging in many countries due to growing ailing stocks, compounded with a lack of staff and financial resources. Automating the key task of visual bridge inspection, classification of defects and building components on pixel level, improves efficiency, increases accuracy and enhances safety in the inspection process and resulting building assessment. Models overtaking this task must cope with an assortment of real-world conditions. They must be robust to variations in image quality, as well as background texture, as defects often appear on surfaces of diverse texture and degree of weathering. dacl10k is the largest and most diverse dataset for real-world concrete bridge inspections. However, the dataset exhibits class imbalance, which leads to notably poor model performance particularly when segmenting fine-grained classes such as cracks and cavities. This work introduces "synth-dacl", a compilation of three novel dataset extensions based on synthetic concrete textures. These extensions are designed to balance class distribution in dacl10k and enhance model performance, especially for crack and cavity segmentation. When incorporating the synth-dacl extensions, we observe substantial improvements in model robustness across 15 perturbed test sets. Notably, on the perturbed test set, a model trained on dacl10k combined with all synthetic extensions achieves a 2% increase in mean IoU, F1 score, Recall, and Precision compared to the same model trained solely on dacl10k. |
| 2025-06-17 | [Robust Adaptive Time-Varying Control Barrier Function with Application to Robotic Surface Treatment](http://arxiv.org/abs/2506.14249v1) | Yitaek Kim, Christoffer Sloth | Set invariance techniques such as control barrier functions (CBFs) can be used to enforce time-varying constraints such as keeping a safe distance from dynamic objects. However, existing methods for enforcing time-varying constraints often overlook model uncertainties. To address this issue, this paper proposes a CBFs-based robust adaptive controller design endowing time-varying constraints while considering parametric uncertainty and additive disturbances. To this end, we first leverage Robust adaptive Control Barrier Functions (RaCBFs) to handle model uncertainty, along with the concept of Input-to-State Safety (ISSf) to ensure robustness towards input disturbances. Furthermore, to alleviate the inherent conservatism in robustness, we also incorporate a set membership identification scheme. We demonstrate the proposed method on robotic surface treatment that requires time-varying force bounds to ensure uniform quality, in numerical simulation and real robotic setup, showing that the quality is formally guaranteed within an acceptable range. |
| 2025-06-17 | [Xolver: Multi-Agent Reasoning with Holistic Experience Learning Just Like an Olympiad Team](http://arxiv.org/abs/2506.14234v1) | Md Tanzib Hosain, Salman Rahman et al. | Despite impressive progress on complex reasoning, current large language models (LLMs) typically operate in isolation - treating each problem as an independent attempt, without accumulating or integrating experiential knowledge. In contrast, expert problem solvers - such as Olympiad or programming contest teams - leverage a rich tapestry of experiences: absorbing mentorship from coaches, developing intuition from past problems, leveraging knowledge of tool usage and library functionality, adapting strategies based on the expertise and experiences of peers, continuously refining their reasoning through trial and error, and learning from other related problems even during competition. We introduce Xolver, a training-free multi-agent reasoning framework that equips a black-box LLM with a persistent, evolving memory of holistic experience. Xolver integrates diverse experience modalities, including external and self-retrieval, tool use, collaborative interactions, agent-driven evaluation, and iterative refinement. By learning from relevant strategies, code fragments, and abstract reasoning patterns at inference time, Xolver avoids generating solutions from scratch - marking a transition from isolated inference toward experience-aware language agents. Built on both open-weight and proprietary models, Xolver consistently outperforms specialized reasoning agents. Even with lightweight backbones (e.g., QWQ-32B), it often surpasses advanced models including Qwen3-235B, Gemini 2.5 Pro, o3, and o4-mini-high. With o3-mini-high, it achieves new best results on GSM8K (98.1%), AIME'24 (94.4%), AIME'25 (93.7%), Math-500 (99.8%), and LiveCodeBench-V5 (91.6%) - highlighting holistic experience learning as a key step toward generalist agents capable of expert-level reasoning. Code and data are available at https://kagnlp.github.io/xolver.github.io/. |

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



