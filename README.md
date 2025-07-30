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
| 2025-07-29 | [Secure Tug-of-War (SecTOW): Iterative Defense-Attack Training with Reinforcement Learning for Multimodal Model Security](http://arxiv.org/abs/2507.22037v1) | Muzhi Dai, Shixuan Liu et al. | The rapid advancement of multimodal large language models (MLLMs) has led to breakthroughs in various applications, yet their security remains a critical challenge. One pressing issue involves unsafe image-query pairs--jailbreak inputs specifically designed to bypass security constraints and elicit unintended responses from MLLMs. Compared to general multimodal data, such unsafe inputs are relatively sparse, which limits the diversity and richness of training samples available for developing robust defense models. Meanwhile, existing guardrail-type methods rely on external modules to enforce security constraints but fail to address intrinsic vulnerabilities within MLLMs. Traditional supervised fine-tuning (SFT), on the other hand, often over-refuses harmless inputs, compromising general performance. Given these challenges, we propose Secure Tug-of-War (SecTOW), an innovative iterative defense-attack training method to enhance the security of MLLMs. SecTOW consists of two modules: a defender and an auxiliary attacker, both trained iteratively using reinforcement learning (GRPO). During the iterative process, the attacker identifies security vulnerabilities in the defense model and expands jailbreak data. The expanded data are then used to train the defender, enabling it to address identified security vulnerabilities. We also design reward mechanisms used for GRPO to simplify the use of response labels, reducing dependence on complex generative labels and enabling the efficient use of synthetic data. Additionally, a quality monitoring mechanism is used to mitigate the defender's over-refusal of harmless inputs and ensure the diversity of the jailbreak data generated by the attacker. Experimental results on safety-specific and general benchmarks demonstrate that SecTOW significantly improves security while preserving general performance. |
| 2025-07-29 | [From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning](http://arxiv.org/abs/2507.22028v1) | Honglin He, Yukai Ma et al. | Navigation foundation models trained on massive webscale data enable agents to generalize across diverse environments and embodiments. However, these models trained solely on offline data, often lack the capacity to reason about the consequences of their actions or adapt through counterfactual understanding. They thus face significant limitations in the real-world urban navigation where interactive and safe behaviors, such as avoiding obstacles and moving pedestrians, are critical. To tackle these challenges, we introduce the Seeing-to-Experiencing framework to scale the capability of navigation foundation models with reinforcement learning. S2E combines the strengths of pre-training on videos and post-training through RL. It maintains the generalizability acquired from large-scale real-world videos while enhancing its interactivity through RL in simulation environments. Specifically, we introduce two innovations: an Anchor-Guided Distribution Matching strategy, which stabilizes learning and models diverse motion patterns through anchor-based supervision; and a Residual-Attention Module, which obtains reactive behaviors from simulation environments without erasing the model's pretrained knowledge. Moreover, we establish a comprehensive end-to-end evaluation benchmark, NavBench-GS, built on photorealistic 3DGS reconstructions of real-world scenes that incorporate physical interactions. It can systematically assess the generalizability and safety of navigation foundation models. Extensive experiments show that S2E mitigates the diminishing returns often seen when scaling with offline data alone. We perform a thorough analysis of the benefits of Reinforcement Learning compared to Supervised Fine-Tuning in the context of post-training for robot learning. Our findings emphasize the crucial role of integrating interactive online experiences to effectively scale foundation models in Robotics. |
| 2025-07-29 | [Hierarchical Game-Based Multi-Agent Decision-Making for Autonomous Vehicles](http://arxiv.org/abs/2507.21941v1) | Mushuang Liu, Yan Wan et al. | This paper develops a game-theoretic decision-making framework for autonomous driving in multi-agent scenarios. A novel hierarchical game-based decision framework is developed for the ego vehicle. This framework features an interaction graph, which characterizes the interaction relationships between the ego and its surrounding traffic agents (including AVs, human driven vehicles, pedestrians, and bicycles, and others), and enables the ego to smartly select a limited number of agents as its game players. Compared to the standard multi-player games, where all surrounding agents are considered as game players, the hierarchical game significantly reduces the computational complexity. In addition, compared to pairwise games, the most popular approach in the literature, the hierarchical game promises more efficient decisions for the ego (in terms of less unnecessary waiting and yielding). To further reduce the computational cost, we then propose an improved hierarchical game, which decomposes the hierarchical game into a set of sub-games. Decision safety and efficiency are analyzed in both hierarchical games. Comprehensive simulation studies are conducted to verify the effectiveness of the proposed frameworks, with an intersection-crossing scenario as a case study. |
| 2025-07-29 | [Libra: Large Chinese-based Safeguard for AI Content](http://arxiv.org/abs/2507.21929v1) | Ziyang Chen, Huimu Yu et al. | Large language models (LLMs) excel in text understanding and generation but raise significant safety and ethical concerns in high-stakes applications. To mitigate these risks, we present Libra-Guard, a cutting-edge safeguard system designed to enhance the safety of Chinese-based LLMs. Leveraging a two-stage curriculum training pipeline, Libra-Guard enhances data efficiency by employing guard pretraining on synthetic samples, followed by fine-tuning on high-quality, real-world data, thereby significantly reducing reliance on manual annotations. To enable rigorous safety evaluations, we also introduce Libra-Test, the first benchmark specifically designed to evaluate the effectiveness of safeguard systems for Chinese content. It covers seven critical harm scenarios and includes over 5,700 samples annotated by domain experts. Experiments show that Libra-Guard achieves 86.79% accuracy, outperforming Qwen2.5-14B-Instruct (74.33%) and ShieldLM-Qwen-14B-Chat (65.69%), and nearing closed-source models like Claude-3.5-Sonnet and GPT-4o. These contributions establish a robust framework for advancing the safety governance of Chinese LLMs and represent a tentative step toward developing safer, more reliable Chinese AI systems. |
| 2025-07-29 | [Training language models to be warm and empathetic makes them less reliable and more sycophantic](http://arxiv.org/abs/2507.21919v1) | Lujain Ibrahim, Franziska Sofia Hafner et al. | Artificial intelligence (AI) developers are increasingly building language models with warm and empathetic personas that millions of people now use for advice, therapy, and companionship. Here, we show how this creates a significant trade-off: optimizing language models for warmth undermines their reliability, especially when users express vulnerability. We conducted controlled experiments on five language models of varying sizes and architectures, training them to produce warmer, more empathetic responses, then evaluating them on safety-critical tasks. Warm models showed substantially higher error rates (+10 to +30 percentage points) than their original counterparts, promoting conspiracy theories, providing incorrect factual information, and offering problematic medical advice. They were also significantly more likely to validate incorrect user beliefs, particularly when user messages expressed sadness. Importantly, these effects were consistent across different model architectures, and occurred despite preserved performance on standard benchmarks, revealing systematic risks that current evaluation practices may fail to detect. As human-like AI systems are deployed at an unprecedented scale, our findings indicate a need to rethink how we develop and oversee these systems that are reshaping human relationships and social interaction. |
| 2025-07-29 | [Safety Analysis for Distributed Coupled-Cavity Laser based Wireless Power Transfer](http://arxiv.org/abs/2507.21891v1) | Mingqing Liu, Hao Deng et al. | Intracavity laser-based systems are emerging as key enablers for next-generation wireless communications, positioning, and wireless power transfer (WPT). Distributed coupled-cavity laser (DCCL) systems, as a representative configuration, have been proposed to expand the field of view (FoV) and enhance safety. This paper investigates the safety assessment of DCCL-WPT systems through three case studies: skin safety, eye safety, and small-object intrusion sensitivity. First, we establish a safety analysis model to quantify irradiation levels on intruding objects in the beam path, which simulates intracavity beam propagation using diffraction modeling and gain-loss dynamics under case-specific boundary conditions. Next, we formulate an eye safety evaluation tailored for DCCL-WPT systems using a human head model to identify potential exposure angles and distances. Ray tracing confirms that intracavity beams are not focused onto the retina, making cornea exposure the primary consideration (irradiance is below 0.1 W/cm2). Numerical results demonstrate that DCCL-WPT achieves: i) over 600 mW charging power under skin-safe conditions at 5 m distance (100 mW over 16{\deg} FoV), and nearly 50% lower irradiance on intruding objects compared to single-cavity systems; ii) 150 mW charging power under eye-safe conditions with 650 mW 1064 nm output beam power, far beyond the typical ~10 mW eye-safe threshold; iii) high sensitivity to small-object intrusion, enabling hazard mitigation. These findings underscore the practicality of DCCL-WPT systems for mobile, long-distance, and safe energy transfer, and lay the groundwork for future safety-aware optimizations in real-world deployments. |
| 2025-07-29 | [MultiEditor: Controllable Multimodal Object Editing for Driving Scenarios Using 3D Gaussian Splatting Priors](http://arxiv.org/abs/2507.21872v1) | Shouyi Lu, Zihan Lin et al. | Autonomous driving systems rely heavily on multimodal perception data to understand complex environments. However, the long-tailed distribution of real-world data hinders generalization, especially for rare but safety-critical vehicle categories. To address this challenge, we propose MultiEditor, a dual-branch latent diffusion framework designed to edit images and LiDAR point clouds in driving scenarios jointly. At the core of our approach is introducing 3D Gaussian Splatting (3DGS) as a structural and appearance prior for target objects. Leveraging this prior, we design a multi-level appearance control mechanism--comprising pixel-level pasting, semantic-level guidance, and multi-branch refinement--to achieve high-fidelity reconstruction across modalities. We further propose a depth-guided deformable cross-modality condition module that adaptively enables mutual guidance between modalities using 3DGS-rendered depth, significantly enhancing cross-modality consistency. Extensive experiments demonstrate that MultiEditor achieves superior performance in visual and geometric fidelity, editing controllability, and cross-modality consistency. Furthermore, generating rare-category vehicle data with MultiEditor substantially enhances the detection accuracy of perception models on underrepresented classes. |
| 2025-07-29 | [Evaluating Interactions between Automated Vehicles and Cyclists using a coupled In-the-Loop Test Environment](http://arxiv.org/abs/2507.21859v1) | Michael Kaiser, Clemens Groß et al. | Testing and evaluating automated driving systems (ADS) in interactions with vulnerable road users (VRUs), such as cyclists, are essential for improving the safety of VRUs, but often lack realism. This paper presents and validates a coupled in-the-loop test environment that integrates a Cyclist-in-the Loop test bench with a Vehicle-in-the-Loop test bench via a virtual environment (VE) developed in Unreal Engine 5. The setup enables closed-loop, bidirectional interaction between a real human cyclist and a real automated vehicle under safe and controllable conditions. The automated vehicle reacts to cyclist gestures via stimulated camera input, while the cyclist, riding a stationary bicycle, perceives and reacts to the vehicle in the VE in real time. Validation experiments are conducted using a real automated shuttle bus with a track-and-follow function, performing three test maneuvers - straight-line driving with stop, circular track driving, and double lane change - on a proving ground and in the coupled in-the-loop test environment. The performance is evaluated by comparing the resulting vehicle trajectories in both environments. Additionally, the introduced latencies of individual components in the test setup are measured. The results demonstrate the feasibility of the approach and highlight its strengths and limitations for realistic ADS evaluation. |
| 2025-07-29 | [Against racing to AGI: Cooperation, deterrence, and catastrophic risks](http://arxiv.org/abs/2507.21839v1) | Leonard Dung, Max Hellrigel-Holderbaum | AGI Racing is the view that it is in the self-interest of major actors in AI development, especially powerful nations, to accelerate their frontier AI development to build highly capable AI, especially artificial general intelligence (AGI), before competitors have a chance. We argue against AGI Racing. First, the downsides of racing to AGI are much higher than portrayed by this view. Racing to AGI would substantially increase catastrophic risks from AI, including nuclear instability, and undermine the prospects of technical AI safety research to be effective. Second, the expected benefits of racing may be lower than proponents of AGI Racing hold. In particular, it is questionable whether winning the race enables complete domination over losers. Third, international cooperation and coordination, and perhaps carefully crafted deterrence measures, constitute viable alternatives to racing to AGI which have much smaller risks and promise to deliver most of the benefits that racing to AGI is supposed to provide. Hence, racing to AGI is not in anyone's self-interest as other actions, particularly incentivizing and seeking international cooperation around AI issues, are preferable. |
| 2025-07-29 | [Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is](http://arxiv.org/abs/2507.21820v1) | Ahmed B Mustafa, Zihan Ye et al. | Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings. |
| 2025-07-29 | [HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs](http://arxiv.org/abs/2507.21815v1) | Kaixuan Wang, Chenxin Diao et al. | Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms. |
| 2025-07-29 | [Interactive Adversarial Testing of Autonomous Vehicles with Adjustable Confrontation Intensity](http://arxiv.org/abs/2507.21814v1) | Yicheng Guo, Chengkai Xu et al. | Scientific testing techniques are essential for ensuring the safe operation of autonomous vehicles (AVs), with high-risk, highly interactive scenarios being a primary focus. To address the limitations of existing testing methods, such as their heavy reliance on high-quality test data, weak interaction capabilities, and low adversarial robustness, this paper proposes ExamPPO, an interactive adversarial testing framework that enables scenario-adaptive and intensity-controllable evaluation of autonomous vehicles. The framework models the Surrounding Vehicle (SV) as an intelligent examiner, equipped with a multi-head attention-enhanced policy network, enabling context-sensitive and sustained behavioral interventions. A scalar confrontation factor is introduced to modulate the intensity of adversarial behaviors, allowing continuous, fine-grained adjustment of test difficulty. Coupled with structured evaluation metrics, ExamPPO systematically probes AV's robustness across diverse scenarios and strategies. Extensive experiments across multiple scenarios and AV strategies demonstrate that ExamPPO can effectively modulate adversarial behavior, expose decision-making weaknesses in tested AVs, and generalize across heterogeneous environments, thereby offering a unified and reproducible solution for evaluating the safety and intelligence of autonomous decision-making systems. |
| 2025-07-29 | [Can large language models assist choice modelling? Insights into prompting strategies and current models capabilities](http://arxiv.org/abs/2507.21790v1) | Georges Sfeir, Gabriel Nova et al. | Large Language Models (LLMs) are widely used to support various workflows across different disciplines, yet their potential in choice modelling remains relatively unexplored. This work examines the potential of LLMs as assistive agents in the specification and, where technically feasible, estimation of Multinomial Logit models. We implement a systematic experimental framework involving thirteen versions of six leading LLMs (ChatGPT, Claude, DeepSeek, Gemini, Gemma, and Llama) evaluated under five experimental configurations. These configurations vary along three dimensions: modelling goal (suggesting vs. suggesting and estimating MNLs); prompting strategy (Zero-Shot vs. Chain-of-Thoughts); and information availability (full dataset vs. data dictionary only). Each LLM-suggested specification is implemented, estimated, and evaluated based on goodness-of-fit metrics, behavioural plausibility, and model complexity. Findings reveal that proprietary LLMs can generate valid and behaviourally sound utility specifications, particularly when guided by structured prompts. Open-weight models such as Llama and Gemma struggled to produce meaningful specifications. Claude 4 Sonnet consistently produced the best-fitting and most complex models, while GPT models suggested models with robust and stable modelling outcomes. Some LLMs performed better when provided with just data dictionary, suggesting that limiting raw data access may enhance internal reasoning capabilities. Among all LLMs, GPT o3 was uniquely capable of correctly estimating its own specifications by executing self-generated code. Overall, the results demonstrate both the promise and current limitations of LLMs as assistive agents in choice modelling, not only for model specification but also for supporting modelling decision and estimation, and provide practical guidance for integrating these tools into choice modellers' workflows. |
| 2025-07-29 | [The Problem with Safety Classification is not just the Models](http://arxiv.org/abs/2507.21782v1) | Sowmya Vajjala | Studying the robustness of Large Language Models (LLMs) to unsafe behaviors is an important topic of research today. Building safety classification models or guard models, which are fine-tuned models for input/output safety classification for LLMs, is seen as one of the solutions to address the issue. Although there is a lot of research on the safety testing of LLMs themselves, there is little research on evaluating the effectiveness of such safety classifiers or the evaluation datasets used for testing them, especially in multilingual scenarios. In this position paper, we demonstrate how multilingual disparities exist in 5 safety classification models by considering datasets covering 18 languages. At the same time, we identify potential issues with the evaluation datasets, arguing that the shortcomings of current safety classifiers are not only because of the models themselves. We expect that these findings will contribute to the discussion on developing better methods to identify harmful content in LLM inputs across languages. |
| 2025-07-29 | [LiteFat: Lightweight Spatio-Temporal Graph Learning for Real-Time Driver Fatigue Detection](http://arxiv.org/abs/2507.21756v1) | Jing Ren, Suyu Ma et al. | Detecting driver fatigue is critical for road safety, as drowsy driving remains a leading cause of traffic accidents. Many existing solutions rely on computationally demanding deep learning models, which result in high latency and are unsuitable for embedded robotic devices with limited resources (such as intelligent vehicles/cars) where rapid detection is necessary to prevent accidents. This paper introduces LiteFat, a lightweight spatio-temporal graph learning model designed to detect driver fatigue efficiently while maintaining high accuracy and low computational demands. LiteFat involves converting streaming video data into spatio-temporal graphs (STG) using facial landmark detection, which focuses on key motion patterns and reduces unnecessary data processing. LiteFat uses MobileNet to extract facial features and create a feature matrix for the STG. A lightweight spatio-temporal graph neural network is then employed to identify signs of fatigue with minimal processing and low latency. Experimental results on benchmark datasets show that LiteFat performs competitively while significantly decreasing computational complexity and latency as compared to current state-of-the-art methods. This work enables the development of real-time, resource-efficient human fatigue detection systems that can be implemented upon embedded robotic devices. |
| 2025-07-29 | [Ionic emissions and activity evolution in comet C/2020 F3 (NEOWISE): Insights from long-slit spectroscopy and photometry](http://arxiv.org/abs/2507.21707v1) | K. Aravind, E. Jehin et al. | Comet C/2020 F3 (NEOWISE) was the brightest comet in the northern hemisphere since C/1995 O1 (Hale-Bopp), providing a unique opportunity to study its composition and spatial distribution of emissions. We conducted narrow-band photometry and long-slit low-resolution spectroscopy to monitor the comet's activity and compositional evolution over several weeks post-perihelion. Narrow-band images (OH, NH, CN, C$_2$, C$_3$, BC, GC, RC) and broad-band images (B, V, Rc, Ic) were acquired with TRAPPIST-North between 22 July and 10 September 2020 to derive production rates, mixing ratios, and dust proxy (Af$\rho$). A long-slit spectrum obtained on 24 July 2020 with HFOSC on the 2-m HCT was used to analyse emission profiles along the sunward and anti-sunward directions. We report production rates and mixing ratios of OH, NH, CN, C$_2$, C$_3$, and NH$_2$, and derive the water production rate using forbidden oxygen line flux. Ionic emissions from N$_2^+$, CO$^+$, CO$_2^+$, and H$_2$O$^+$ were detected at 4$\times$10$^4$ to 1$\times$10$^5$ km from the nucleus in the tailward direction. The average N$_2^+$/CO$^+$ ratio was found to be (3.0 $\pm$ 1.0)$\times$10$^{-2}$, refined to (4.8 $\pm$ 2.4)$\times$10$^{-2}$ using fluorescence modeling. The CO$_2^+$/CO$^+$ ratio was measured to be 1.34 $\pm$ 0.21. These results suggest the comet likely formed in the cold mid-to-outer solar nebula (approx. 50-70 K). Additionally, the average rotation period was estimated as 7.28 $\pm$ 0.79 hours, with a CN outflow velocity of 2.40 $\pm$ 0.25 km/s |
| 2025-07-29 | [Edge Agentic AI Framework for Autonomous Network Optimisation in O-RAN](http://arxiv.org/abs/2507.21696v1) | Abdelaziz Salama, Zeinab Nezami et al. | The deployment of AI agents within legacy Radio Access Network (RAN) infrastructure poses significant safety and reliability challenges for future 6G networks. This paper presents a novel Edge AI framework for autonomous network optimisation in Open RAN environments, addressing these challenges through three core innovations: (1) a persona-based multi-tools architecture enabling distributed, context-aware decision-making; (2) proactive anomaly detection agent powered by traffic predictive tool; and (3) a safety, aligned reward mechanism that balances performance with operational stability. Integrated into the RAN Intelligent Controller (RIC), our framework leverages multimodal data fusion, including network KPIs, a traffic prediction model, and external information sources, to anticipate and respond to dynamic network conditions. Extensive evaluation using realistic 5G scenarios demonstrates that the edge framework achieves zero network outages under high-stress conditions, compared to 8.4% for traditional fixed-power networks and 3.3% for large language model (LLM) agent-based approaches, while maintaining near real-time responsiveness and consistent QoS. These results establish that, when equipped with the right tools and contextual awareness, AI agents can be safely and effectively deployed in critical network infrastructure, laying the framework for intelligent and autonomous 5G and beyond network operations. |
| 2025-07-29 | [UnsafeChain: Enhancing Reasoning Model Safety via Hard Cases](http://arxiv.org/abs/2507.21652v1) | Raj Vardhan Tomar, Preslav Nakov et al. | As large reasoning models (LRMs) grow more capable, chain-of-thought (CoT) reasoning introduces new safety challenges. Existing SFT-based safety alignment studies dominantly focused on filtering prompts with safe, high-quality responses, while overlooking hard prompts that always elicit harmful outputs. To fill this gap, we introduce UnsafeChain, a safety alignment dataset constructed from hard prompts with diverse sources, where unsafe completions are identified and explicitly corrected into safe responses. By exposing models to unsafe behaviors and guiding their correction, UnsafeChain enhances safety while preserving general reasoning ability. We fine-tune three LRMs on UnsafeChain and compare them against recent SafeChain and STAR-1 across six out-of-distribution and five in-distribution benchmarks. UnsafeChain consistently outperforms prior datasets, with even a 1K subset matching or surpassing baseline performance, demonstrating the effectiveness and generalizability of correction-based supervision. We release our dataset and code at https://github.com/mbzuai-nlp/UnsafeChain |
| 2025-07-29 | [The Evolution of Video Anomaly Detection: A Unified Framework from DNN to MLLM](http://arxiv.org/abs/2507.21649v1) | Shibo Gao, Peipei Yang et al. | Video anomaly detection (VAD) aims to identify and ground anomalous behaviors or events in videos, serving as a core technology in the fields of intelligent surveillance and public safety. With the advancement of deep learning, the continuous evolution of deep model architectures has driven innovation in VAD methodologies, significantly enhancing feature representation and scene adaptability, thereby improving algorithm generalization and expanding application boundaries. More importantly, the rapid development of multi-modal large language (MLLMs) and large language models (LLMs) has introduced new opportunities and challenges to the VAD field. Under the support of MLLMs and LLMs, VAD has undergone significant transformations in terms of data annotation, input modalities, model architectures, and task objectives. The surge in publications and the evolution of tasks have created an urgent need for systematic reviews of recent advancements. This paper presents the first comprehensive survey analyzing VAD methods based on MLLMs and LLMs, providing an in-depth discussion of the changes occurring in the VAD field in the era of large models and their underlying causes. Additionally, this paper proposes a unified framework that encompasses both deep neural network (DNN)-based and LLM-based VAD methods, offering a thorough analysis of the new VAD paradigms empowered by LLMs, constructing a classification system, and comparing their strengths and weaknesses. Building on this foundation, this paper focuses on current VAD methods based on MLLMs/LLMs. Finally, based on the trajectory of technological advancements and existing bottlenecks, this paper distills key challenges and outlines future research directions, offering guidance for the VAD community. |
| 2025-07-29 | [Self-Aware Safety Augmentation: Leveraging Internal Semantic Understanding to Enhance Safety in Vision-Language Models](http://arxiv.org/abs/2507.21637v1) | Wanying Wang, Zeyu Ma et al. | Large vision-language models (LVLMs) are vulnerable to harmful input compared to their language-only backbones. We investigated this vulnerability by exploring LVLMs internal dynamics, framing their inherent safety understanding in terms of three key capabilities. Specifically, we define these capabilities as safety perception, semantic understanding, and alignment for linguistic expression, and experimentally pinpointed their primary locations within the model architecture. The results indicate that safety perception often emerges before comprehensive semantic understanding, leading to the reduction in safety. Motivated by these findings, we propose \textbf{Self-Aware Safety Augmentation (SASA)}, a technique that projects informative semantic representations from intermediate layers onto earlier safety-oriented layers. This approach leverages the model's inherent semantic understanding to enhance safety recognition without fine-tuning. Then, we employ linear probing to articulate the model's internal semantic comprehension to detect the risk before the generation process. Extensive experiments on various datasets and tasks demonstrate that SASA significantly improves the safety of LVLMs, with minimal impact on the utility. |

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



