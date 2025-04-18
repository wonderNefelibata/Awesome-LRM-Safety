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
| 2025-04-16 | [Accountable Liveness](http://arxiv.org/abs/2504.12218v1) | Andrew Lewis-Pye, Joachim Neu et al. | Safety and liveness are the two classical security properties of consensus protocols. Recent works have strengthened safety with accountability: should any safety violation occur, a sizable fraction of adversary nodes can be proven to be protocol violators. This paper studies to what extent analogous accountability guarantees are achievable for liveness. To reveal the full complexity of this question, we introduce an interpolation between the classical synchronous and partially-synchronous models that we call the $x$-partially-synchronous network model in which, intuitively, at most an $x$ fraction of the time steps in any sufficiently long interval are asynchronous (and, as with a partially-synchronous network, all time steps are synchronous following the passage of an unknown "global stablization time"). We prove a precise characterization of the parameter regime in which accountable liveness is achievable: if and only if $x < 1/2$ and $f < n/2$, where $n$ denotes the number of nodes and $f$ the number of nodes controlled by an adversary. We further refine the problem statement and our analysis by parameterizing by the number of violating nodes identified following a liveness violation, and provide evidence that the guarantees achieved by our protocol are near-optimal (as a function of $x$ and $f$). Our results provide rigorous foundations for liveness-accountability heuristics such as the "inactivity leaks" employed in Ethereum. |
| 2025-04-16 | [GripMap: An Efficient, Spatially Resolved Constraint Framework for Offline and Online Trajectory Planning in Autonomous Racing](http://arxiv.org/abs/2504.12115v1) | Frederik Werner, Ann-Kathrin Schwehn et al. | Conventional trajectory planning approaches for autonomous vehicles often assume a fixed vehicle model that remains constant regardless of the vehicle's location. This overlooks the critical fact that the tires and the surface are the two force-transmitting partners in vehicle dynamics; while the tires stay with the vehicle, surface conditions vary with location. Recognizing these challenges, this paper presents a novel framework for spatially resolving dynamic constraints in both offline and online planning algorithms applied to autonomous racing. We introduce the GripMap concept, which provides a spatial resolution of vehicle dynamic constraints in the Frenet frame, allowing adaptation to locally varying grip conditions. This enables compensation for location-specific effects, more efficient vehicle behavior, and increased safety, unattainable with spatially invariant vehicle models. The focus is on low storage demand and quick access through perfect hashing. This framework proved advantageous in real-world applications in the presented form. Experiments inspired by autonomous racing demonstrate its effectiveness. In future work, this framework can serve as a foundational layer for developing future interpretable learning algorithms that adjust to varying grip conditions in real-time. |
| 2025-04-16 | [Towards LLM Agents for Earth Observation](http://arxiv.org/abs/2504.12110v1) | Chia Hsiang Kao, Wenting Zhao et al. | Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at https://iandrover.github.io/UnivEarth. |
| 2025-04-16 | [Reasoning-Based AI for Startup Evaluation (R.A.I.S.E.): A Memory-Augmented, Multi-Step Decision Framework](http://arxiv.org/abs/2504.12090v1) | Jack Preuveneers, Joseph Ternasky et al. | We present a novel framework that bridges the gap between the interpretability of decision trees and the advanced reasoning capabilities of large language models (LLMs) to predict startup success. Our approach leverages chain-of-thought prompting to generate detailed reasoning logs, which are subsequently distilled into structured, human-understandable logical rules. The pipeline integrates multiple enhancements - efficient data ingestion, a two-step refinement process, ensemble candidate sampling, simulated reinforcement learning scoring, and persistent memory - to ensure both stable decision-making and transparent output. Experimental evaluations on curated startup datasets demonstrate that our combined pipeline improves precision by 54% from 0.225 to 0.346 and accuracy by 50% from 0.46 to 0.70 compared to a standalone OpenAI o3 model. Notably, our model achieves over 2x the precision of a random classifier (16%). By combining state-of-the-art AI reasoning with explicit rule-based explanations, our method not only augments traditional decision-making processes but also facilitates expert intervention and continuous policy refinement. This work lays the foundation for the implementation of interpretable LLM-powered decision frameworks in high-stakes investment environments and other domains that require transparent and data-driven insights. |
| 2025-04-16 | [Selective Demonstration Retrieval for Improved Implicit Hate Speech Detection](http://arxiv.org/abs/2504.12082v1) | Yumin Kim, Hwanhee Lee | Hate speech detection is a crucial area of research in natural language processing, essential for ensuring online community safety. However, detecting implicit hate speech, where harmful intent is conveyed in subtle or indirect ways, remains a major challenge. Unlike explicit hate speech, implicit expressions often depend on context, cultural subtleties, and hidden biases, making them more challenging to identify consistently. Additionally, the interpretation of such speech is influenced by external knowledge and demographic biases, resulting in varied detection results across different language models. Furthermore, Large Language Models often show heightened sensitivity to toxic language and references to vulnerable groups, which can lead to misclassifications. This over-sensitivity results in false positives (incorrectly identifying harmless statements as hateful) and false negatives (failing to detect genuinely harmful content). Addressing these issues requires methods that not only improve detection precision but also reduce model biases and enhance robustness. To address these challenges, we propose a novel method, which utilizes in-context learning without requiring model fine-tuning. By adaptively retrieving demonstrations that focus on similar groups or those with the highest similarity scores, our approach enhances contextual comprehension. Experimental results show that our method outperforms current state-of-the-art techniques. Implementation details and code are available at TBD. |
| 2025-04-16 | [Observational properties of regular black holes in Asymptotic Safety](http://arxiv.org/abs/2504.12072v1) | Abdybek Urmanov, Hrishikesh Chakrabarty et al. | We consider the observational properties of a spherically symmetric, static regular black hole within the framework of asymptotic safety (AS) as proposed by Bonanno et al. The metric resembles the Schwarzschild solution in the classical limit. The departure from Schwarzschild at small scales is controlled by a single free parameter related to the ultraviolet (UV) cutoff of the theory. We investigated null and time-like geodesics around the AS metric, including circular orbits, photon rings and lensing effects. In particular we focused on the optical properties of thin accretion disks in the equatorial plane of the object and compared them with those of accretion disks in the Schwarzschild metric. We found that the radiation flux, luminosity, and efficiency of the accretion disk increase with the value of the free parameter. Using a spacetime generic open-source relativistic ray-tracing code, we simulate the K$\alpha$ iron line profiles emitted by the disk and analyze their deviation from that of the Schwarzschild geometry. |
| 2025-04-16 | [Contract-based hierarchical control using predictive feasibility value functions](http://arxiv.org/abs/2504.12036v1) | Felix Berkel, Kim Peter Wabersich et al. | Today's control systems are often characterized by modularity and safety requirements to handle complexity, resulting in hierarchical control structures. Although hierarchical model predictive control offers favorable properties, achieving a provably safe, yet modular design remains a challenge. This paper introduces a contract-based hierarchical control strategy to improve the performance of control systems facing challenges related to model inconsistency and independent controller design across hierarchies. We consider a setup where a higher-level controller generates references that affect the constraints of a lower-level controller, which is based on a soft-constrained MPC formulation. The optimal slack variables serve as the basis for a contract that allows the higher-level controller to assess the feasibility of the reference trajectory without exact knowledge of the model, constraints, and cost of the lower-level controller. To ensure computational efficiency while maintaining model confidentiality, we propose using an explicit function approximation, such as a neural network, to represent the cost of optimal slack values. The approach is tested for a hierarchical control setup consisting of a planner and a motion controller as commonly found in autonomous driving. |
| 2025-04-16 | [Global Patterns of Extreme Temperature Teleconnections Using Climate Network Analysis](http://arxiv.org/abs/2504.12008v1) | Yuhao Feng, Jun Meng et al. | Extreme weather events, rare yet profoundly impactful, are often accompanied by severe conditions. Increasing global temperatures are poised to exacerbate these events, resulting in greater human casualties, economic losses, and ecological destruction. Complex global climate interactions, known as teleconnections, can lead to widespread repercussions triggered by localized extreme weather. Understanding these teleconnection patterns is crucial for weather forecasting, enhancing safety, and advancing climate science. Here, we employ climate network analysis to uncover teleconnection patterns associated with extreme temperature fluctuations, including both extreme warming and cooling events occurring on a daily basis. Our study results demonstrate that the distances of significant teleconnections initially conform to a power-law decay, signifying a decline in connectivity with distance. However, this power-law decay tendency breaks beyond a certain threshold distance, suggesting the existence of long-distance connections. Additionally, we uncover a greater prevalence of long-distance connectivity among extreme cooling events compared to extreme warming events. The global pattern of teleconnections is, in part, driven by the mechanism of Rossby waves, which serve as a rapid conduit for inducing correlated fluctuations in both pressure and temperature. These results enhance our understanding of the multiscale nature of climate teleconnections and hold significant implications for improving weather forecasting and assessing climate risks in a warming world. |
| 2025-04-16 | [Comment on Path integral measure and RG equations for gravity](http://arxiv.org/abs/2504.12006v1) | Aaron Held, Benjamin Knorr et al. | Asymptotic safety is a candidate for a predictive quantum theory of gravity and matter. Recent works (arXiv:2412.10194 and arXiv:2412.14108) challenged this scenario. We show that their arguments fail on a basic level. |
| 2025-04-16 | [Leveraging Machine Learning Models to Predict the Outcome of Digital Medical Triage Interviews](http://arxiv.org/abs/2504.11977v1) | Sofia Krylova, Fabian Schmidt et al. | Many existing digital triage systems are questionnaire-based, guiding patients to appropriate care levels based on information (e.g., symptoms, medical history, and urgency) provided by the patients answering questionnaires. Such a system often uses a deterministic model with predefined rules to determine care levels. It faces challenges with incomplete triage interviews since it can only assist patients who finish the process. In this study, we explore the use of machine learning (ML) to predict outcomes of unfinished interviews, aiming to enhance patient care and service quality. Predicting triage outcomes from incomplete data is crucial for patient safety and healthcare efficiency. Our findings show that decision-tree models, particularly LGBMClassifier and CatBoostClassifier, achieve over 80\% accuracy in predicting outcomes from complete interviews while having a linear correlation between the prediction accuracy and interview completeness degree. For example, LGBMClassifier achieves 88,2\% prediction accuracy for interviews with 100\% completeness, 79,6\% accuracy for interviews with 80\% completeness, 58,9\% accuracy for 60\% completeness, and 45,7\% accuracy for 40\% completeness. The TabTransformer model demonstrated exceptional accuracy of over 80\% for all degrees of completeness but required extensive training time, indicating a need for more powerful computational resources. The study highlights the linear correlation between interview completeness and predictive power of the decision-tree models. |
| 2025-04-16 | [SemDiff: Generating Natural Unrestricted Adversarial Examples via Semantic Attributes Optimization in Diffusion Models](http://arxiv.org/abs/2504.11923v1) | Zeyu Dai, Shengcai Liu et al. | Unrestricted adversarial examples (UAEs), allow the attacker to create non-constrained adversarial examples without given clean samples, posing a severe threat to the safety of deep learning models. Recent works utilize diffusion models to generate UAEs. However, these UAEs often lack naturalness and imperceptibility due to simply optimizing in intermediate latent noises. In light of this, we propose SemDiff, a novel unrestricted adversarial attack that explores the semantic latent space of diffusion models for meaningful attributes, and devises a multi-attributes optimization approach to ensure attack success while maintaining the naturalness and imperceptibility of generated UAEs. We perform extensive experiments on four tasks on three high-resolution datasets, including CelebA-HQ, AFHQ and ImageNet. The results demonstrate that SemDiff outperforms state-of-the-art methods in terms of attack success rate and imperceptibility. The generated UAEs are natural and exhibit semantically meaningful changes, in accord with the attributes' weights. In addition, SemDiff is found capable of evading different defenses, which further validates its effectiveness and threatening. |
| 2025-04-16 | [Rethinking the Generation of High-Quality CoT Data from the Perspective of LLM-Adaptive Question Difficulty Grading](http://arxiv.org/abs/2504.11919v1) | Qianjin Yu, Keyu Wu et al. | Recently, DeepSeek-R1 (671B) (DeepSeek-AIet al., 2025) has demonstrated its excellent reasoning ability in complex tasks and has publiclyshared its methodology. This provides potentially high-quality chain-of-thought (CoT) data for stimulating the reasoning abilities of small-sized large language models (LLMs). To generate high-quality CoT data for different LLMs, we seek an efficient method for generating high-quality CoT data with LLM-Adaptive questiondifficulty levels. First, we grade the difficulty of the questions according to the reasoning ability of the LLMs themselves and construct a LLM-Adaptive question database. Second, we sample the problem database based on a distribution of difficulty levels of the questions and then use DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) to generate the corresponding high-quality CoT data with correct answers. Thanks to the construction of CoT data with LLM-Adaptive difficulty levels, we have significantly reduced the cost of data generation and enhanced the efficiency of model supervised fine-tuning (SFT). Finally, we have validated the effectiveness and generalizability of the proposed method in the fields of complex mathematical competitions and code generation tasks. Notably, with only 2k high-quality mathematical CoT data, our ZMath-32B surpasses DeepSeek-Distill-32B in math reasoning task. Similarly, with only 2k high-quality code CoT data, our ZCode-32B surpasses DeepSeek-Distill-32B in code reasoning tasks. |
| 2025-04-16 | [A Graph-Based Reinforcement Learning Approach with Frontier Potential Based Reward for Safe Cluttered Environment Exploration](http://arxiv.org/abs/2504.11907v1) | Gabriele Calzolari, Vidya Sumathy et al. | Autonomous exploration of cluttered environments requires efficient exploration strategies that guarantee safety against potential collisions with unknown random obstacles. This paper presents a novel approach combining a graph neural network-based exploration greedy policy with a safety shield to ensure safe navigation goal selection. The network is trained using reinforcement learning and the proximal policy optimization algorithm to maximize exploration efficiency while reducing the safety shield interventions. However, if the policy selects an infeasible action, the safety shield intervenes to choose the best feasible alternative, ensuring system consistency. Moreover, this paper proposes a reward function that includes a potential field based on the agent's proximity to unexplored regions and the expected information gain from reaching them. Overall, the approach investigated in this paper merges the benefits of the adaptability of reinforcement learning-driven exploration policies and the guarantee ensured by explicit safety mechanisms. Extensive evaluations in simulated environments demonstrate that the approach enables efficient and safe exploration in cluttered environments. |
| 2025-04-16 | [MDHP-Net: Detecting an Emerging Time-exciting Threat in IVN](http://arxiv.org/abs/2504.11867v1) | Qi Liu, Yanchen Liu et al. | The integration of intelligent and connected technologies in modern vehicles, while offering enhanced functionalities through Electronic Control Unit (ECU) and interfaces like OBD-II and telematics, also exposes the vehicle's in-vehicle network (IVN) to potential cyberattacks. Unlike prior work, we identify a new time-exciting threat model against IVN. These attacks inject malicious messages that exhibit a time-exciting effect, gradually manipulating network traffic to disrupt vehicle operations and compromise safety-critical functions. We systematically analyze the characteristics of the threat: dynamism, time-exciting impact, and low prior knowledge dependency. To validate its practicality, we replicate the attack on a real Advanced Driver Assistance System via Controller Area Network (CAN), exploiting Unified Diagnostic Service vulnerabilities and proposing four attack strategies. While CAN's integrity checks mitigate attacks, Ethernet migration (e.g., DoIP/SOME/IP) introduces new surfaces. We further investigate the feasibility of time-exciting threat under SOME/IP. To detect time-exciting threat, we introduce MDHP-Net, leveraging Multi-Dimentional Hawkes Process (MDHP) and temporal and message-wise feature extracting structures. Meanwhile, to estimate MDHP parameters, we developed the first GPU-optimized gradient descent solver for MDHP (MDHP-GDS). These modules significantly improves the detection rate under time-exciting attacks in multi-ECU IVN system. To address data scarcity, we release STEIA9, the first open-source dataset for time-exciting attacks, covering 9 Ethernet-based attack scenarios. Extensive experiments on STEIA9 (9 attack scenarios) show MDHP-Net outperforms 3 baselines, confirming attack feasibility and detection efficacy. |
| 2025-04-16 | [Visualization Analysis and Impedance Analysis for the Aging Behavior Assessment of 18650 Cells](http://arxiv.org/abs/2504.11861v1) | Yihan Shi, Qingrui Pan et al. | This work presents a comprehensive study on the aging behavior of 18650-type lithium-ion batteries, focusing on the uneven intercalation of lithium ions during fast charging processes. It introduces a novel approach using color visual recognition technology to analyze color changes in the graphite anode, indicative of lithiation levels. The study employs X-ray diffraction (XRD) and Distribution of Relaxation Time (DRT) techniques to validate and analyze the observations. The study emphasizes the significance of electrode impedance, the positioning of battery tabs, and electrolyte distribution in influencing the aging dynamics of lithium-ion batteries. Furthermore, the paper presents an innovative impedance Transport-Line Model, specifically developed to capture the evolution of polarization impedance over time. This model offers a deeper understanding of the internal mechanisms driving battery aging, providing valuable insights for the design and optimization of lithium-ion batteries. The research represents a significant contribution to the field, shedding light on the complex aging processes in lithium-ion batteries, particularly under the conditions of fast charging. This could lead to improved battery performance, longevity, and safety, which are critical for the wide range of applications that depend on these energy storage systems. |
| 2025-04-16 | [Ultra-Efficient Kidney Stone Fragment Removal via Spinner-Induced Synergistic Circulation and Spiral Flow](http://arxiv.org/abs/2504.11847v1) | Yilong Chang, Jasmine Vallejo et al. | Kidney stones can cause severe pain and complications such as chronic kidney disease or kidney failure. Retrograde intrarenal surgery (RIRS), which uses laser lithotripsy to fragment stones for removal via a ureteroscope, is widely adopted due to its safety and effectiveness. However, conventional fragment removal methods using basketing and vacuum-assisted aspiration are inefficient, as they can capture only 1 to 3 fragments (1--3\,mm in size) per pass, often requiring dozens to hundreds of ureteroscope passes during a single procedure to completely remove the fragments. These limitations lead to prolonged procedures and residual fragments that contribute to high recurrence rates. To address these limitations, we present a novel spinner device that enables ultra-efficient fragment removal through spinning-induced localized suction. The spinner generates a three-dimensional spiral and circulating flow field that dislodges and draws fragments into its cavity even from distances over 20\,mm, eliminating the need to chase fragments. It can capture over 60 fragments (0.5--2\,mm) or over 15 larger fragments (2--3\,mm) in a single pass, significantly improving removal efficiency. In this work, the spinner design is optimized via computational fluid dynamics to maximize suction performance. \textit{In vitro} testing demonstrates near 100\% capture rates for up to 60 fragments in a single operation and superior large-distance capture efficacy compared to vacuum-assisted methods. \textit{Ex vivo} testing of the integrated spinner-ureteroscope system in a porcine kidney confirmed its high performance by capturing 45 fragments in just 4 seconds during a single pass and achieving complete fragment clearance within a few passes. |
| 2025-04-16 | [Support is All You Need for Certified VAE Training](http://arxiv.org/abs/2504.11831v1) | Changming Xu, Debangshu Banerjee et al. | Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees. |
| 2025-04-16 | [Multi-goal Rapidly Exploring Random Tree with Safety and Dynamic Constraints for UAV Cooperative Path Planning](http://arxiv.org/abs/2504.11823v1) | Thu Hang Khuat, Duy-Nam Bui et al. | Cooperative path planning is gaining its importance due to the increasing demand on using multiple unmanned aerial vehicles (UAVs) for complex missions. This work addresses the problem by introducing a new algorithm named MultiRRT that extends the rapidly exploring random tree (RRT) to generate paths for a group of UAVs to reach multiple goal locations at the same time. We first derive the dynamics constraint of the UAV and include it in the problem formulation. MultiRRT is then developed, taking into account the cooperative requirements and safe constraints during its path-searching process. The algorithm features two new mechanisms, node reduction and Bezier interpolation, to ensure the feasibility and optimality of the paths generated. Importantly, the interpolated paths are proven to meet the safety and dynamics constraints imposed by obstacles and the UAVs. A number of simulations, comparisons, and experiments have been conducted to evaluate the performance of the proposed approach. The results show that MultiRRT can generate collision-free paths for multiple UAVs to reach their goals with better scores in path length and smoothness metrics than state-of-the-art RRT variants including Theta-RRT, FN-RRT, RRT*, and RRT*-Smart. The generated paths are also tested in practical flights with real UAVs to evaluate their validity for cooperative tasks. The source code of the algorithm is available at https://github.com/duynamrcv/multi-target_RRT |
| 2025-04-16 | [Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports](http://arxiv.org/abs/2504.11717v1) | Donggeon David Oh, Justin Lidard et al. | We propose a human-centered safety filter (HCSF) for shared autonomy that significantly enhances system safety without compromising human agency. Our HCSF is built on a neural safety value function, which we first learn scalably through black-box interactions and then use at deployment to enforce a novel quality control barrier function (Q-CBF) safety constraint. Since this Q-CBF safety filter does not require any knowledge of the system dynamics for both synthesis and runtime safety monitoring and intervention, our method applies readily to complex, black-box shared autonomy systems. Notably, our HCSF's CBF-based interventions modify the human's actions minimally and smoothly, avoiding the abrupt, last-moment corrections delivered by many conventional safety filters. We validate our approach in a comprehensive in-person user study using Assetto Corsa-a high-fidelity car racing simulator with black-box dynamics-to assess robustness in "driving on the edge" scenarios. We compare both trajectory data and drivers' perceptions of our HCSF assistance against unassisted driving and a conventional safety filter. Experimental results show that 1) compared to having no assistance, our HCSF improves both safety and user satisfaction without compromising human agency or comfort, and 2) relative to a conventional safety filter, our proposed HCSF boosts human agency, comfort, and satisfaction while maintaining robustness. |
| 2025-04-16 | [Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports](http://arxiv.org/abs/2504.11717v2) | Donggeon David Oh, Justin Lidard et al. | We propose a human-centered safety filter (HCSF) for shared autonomy that significantly enhances system safety without compromising human agency. Our HCSF is built on a neural safety value function, which we first learn scalably through black-box interactions and then use at deployment to enforce a novel state-action control barrier function (Q-CBF) safety constraint. Since this Q-CBF safety filter does not require any knowledge of the system dynamics for both synthesis and runtime safety monitoring and intervention, our method applies readily to complex, black-box shared autonomy systems. Notably, our HCSF's CBF-based interventions modify the human's actions minimally and smoothly, avoiding the abrupt, last-moment corrections delivered by many conventional safety filters. We validate our approach in a comprehensive in-person user study using Assetto Corsa-a high-fidelity car racing simulator with black-box dynamics-to assess robustness in "driving on the edge" scenarios. We compare both trajectory data and drivers' perceptions of our HCSF assistance against unassisted driving and a conventional safety filter. Experimental results show that 1) compared to having no assistance, our HCSF improves both safety and user satisfaction without compromising human agency or comfort, and 2) relative to a conventional safety filter, our proposed HCSF boosts human agency, comfort, and satisfaction while maintaining robustness. |

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



