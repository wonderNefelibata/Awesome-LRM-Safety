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
| 2025-11-05 | [LLM-enhanced Air Quality Monitoring Interface via Model Context Protocol](http://arxiv.org/abs/2511.03706v1) | Yu-Erh Pan, Ayesha Siddika Nipu | Air quality monitoring is central to environmental sustainability and public health, yet traditional systems remain difficult for non-expert users to interpret due to complex visualizations, limited interactivity, and high deployment costs. Recent advances in Large Language Models (LLMs) offer new opportunities to make sensor data more accessible, but their tendency to produce hallucinations limits reliability in safety-critical domains. To address these challenges, we present an LLM-enhanced Air Monitoring Interface (AMI) that integrates real-time sensor data with a conversational interface via the Model Context Protocol (MCP). Our system grounds LLM outputs in live environmental data, enabling accurate, context-aware responses while reducing hallucination risk. The architecture combines a Django-based backend, a responsive user dashboard, and a secure MCP server that exposes system functions as discoverable tools, allowing the LLM to act as an active operator rather than a passive responder. Expert evaluation demonstrated high factual accuracy (4.78), completeness (4.82), and minimal hallucinations (4.84), on a scale of 5, supported by inter-rater reliability analysis. These results highlight the potential of combining LLMs with standardized tool protocols to create reliable, secure, and user-friendly interfaces for real-time environmental monitoring. |
| 2025-11-05 | [A Constant-Gain Equation-Error Framework for Airliner Aerodynamic Monitoring Using QAR Data](http://arxiv.org/abs/2511.03678v1) | Ruiying Wen, Yuntao Dai et al. | Monitoring the in-service aerodynamic performance of airliners is critical for operational efficiency and safety, but using operational Quick Access Recorder (QAR) data for this purpose presents significant challenges. This paper first establishes that the absence of key parameters, particularly aircraft moments of inertia, makes conventional state-propagation filters fundamentally unsuitable for this application. This limitation necessitates a decoupled, Equation-Error Method (EEM). However, we then demonstrate through a comparative analysis that standard recursive estimators with time-varying gains, such as Recursive Least Squares (RLS), also fail within an EEM framework, exhibiting premature convergence or instability when applied to low-excitation cruise data. To overcome these dual challenges, we propose and validate the Constant-Gain Equation-Error Method (CG-EEM). This framework employs a custom estimator with a constant, Kalman-like gain, which is perfectly suited to the stationary, low-signal-to-noise characteristics of cruise flight. The CG-EEM is extensively validated on a large, multi-fleet dataset of over 200 flights, where it produces highly consistent, physically plausible aerodynamic parameters and correctly identifies known performance differences between aircraft types. The result is a robust, scalable, and computationally efficient tool for fleet-wide performance monitoring and the early detection of performance degradation. |
| 2025-11-05 | [Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning](http://arxiv.org/abs/2511.03591v1) | Qingyi Chen, Ruiqi Ni et al. | Safe multi-agent motion planning (MAMP) under task-induced constraints is a critical challenge in robotics. Many real-world scenarios require robots to navigate dynamic environments while adhering to manifold constraints imposed by tasks. For example, service robots must carry cups upright while avoiding collisions with humans or other robots. Despite recent advances in decentralized MAMP for high-dimensional systems, incorporating manifold constraints remains difficult. To address this, we propose a manifold-constrained Hamilton-Jacobi reachability (HJR) learning framework for decentralized MAMP. Our method solves HJR problems under manifold constraints to capture task-aware safety conditions, which are then integrated into a decentralized trajectory optimization planner. This enables robots to generate motion plans that are both safe and task-feasible without requiring assumptions about other agents' policies. Our approach generalizes across diverse manifold-constrained tasks and scales effectively to high-dimensional multi-agent manipulation problems. Experiments show that our method outperforms existing constrained motion planners and operates at speeds suitable for real-world applications. Video demonstrations are available at https://youtu.be/RYcEHMnPTH8 . |
| 2025-11-05 | [MultiZebraLogic: A Multilingual Logical Reasoning Benchmark](http://arxiv.org/abs/2511.03553v1) | Sofie Helene Bruun, Dan Saattrup Smart | Measuring the full abilities of large language models (LLMs) requires benchmarks representing multiple tasks. We aim to create large, high-quality datasets for comparison of logical reasoning skills across several languages and of suitable difficulty for LLMs of various reasoning ability. We explore multiple ways of increasing difficulty. We generate zebra puzzles in multiple languages, themes, sizes and including 14 different clue types and 8 red herring types (uninformative clues). We find puzzle sizes 2x3 and 4x5 are sufficiently challenging for GPT-4o mini (a non-reasoning model) and o3-mini (a reasoning model), respectively. Including 5 red herrings decreases o3-mini puzzle-level accuracy on 4x5 puzzles by 15$\pm$7 %. Scores of o3-mini on 4x5 puzzles are not significantly affected by use of English vs. Danish or the common houses theme vs. the country-specific smoerrebroed theme. We find no correlation between difficulty and the selected clue types. Datasets of 128+1024 puzzles are published as MultiZebraLogic in each of nine Germanic languages for sizes 2x3 and 4x5. We publish code for puzzle generation, designed for adaptablity into more languages and themes. |
| 2025-11-05 | [Silenced Biases: The Dark Side LLMs Learned to Refuse](http://arxiv.org/abs/2511.03369v1) | Rom Himelstein, Amit LeVi et al. | Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues. |
| 2025-11-05 | [Core-Shell Confinement Blocks Hydride Formation: The Impact of Surface Oxides on Hydrogen Sorption in Nanoporous FeTi](http://arxiv.org/abs/2511.03349v1) | Lukas Schweiger, Florian Spieckermann et al. | Metal hydrides remain an intriguing alternative to conventional gaseous and liquid hydrogen storage methods, offering high volumetric storage density and enhanced hydrogen storage safety at ambient conditions. In this regard, the intermetallic compound FeTi is one of the most promising storage materials. However, its widespread industrial application remains challenging due to the need for activation, slow initial kinetics, large hysteresis, and high material costs. In this study, we aim to overcome these limitations by devising an alternative synthesis pathway to prepare nanoporous and ultra-fine porous FeTi with controlled grain and ligament sizes, allowing us to study the obtained well-defined microstructures in detail. In particular, we observe the confinement of the FeTi phase by surface oxides, which can be correlated with the hydrogen sorption properties of the respective material. These experimental results are further supported by an analytical model allowing the calculation of the absorption pressure as a function of microstructure-dependent elastic stresses. Additionally, we show that such stresses also influence the absorption-desorption hysteresis. This study lays the groundwork for the controlled and systematic study of the processing-structure-properties relations in metal hydrides and FeTi in particular, thereby paving the way to cost-effective and efficient hydrogen storage solutions based on metal hydrides. |
| 2025-11-05 | [First global gyrokinetic profile predictions of ITER burning plasma](http://arxiv.org/abs/2511.03336v1) | A. Di Siena, C. Bourdelle et al. | In this work, we present the first global gyrokinetic simulations of the ITER baseline scenario operating at 15 MA using GENE-Tango electrostatic and electromagnetic simulations. The modeled radial region spans close to the magnetic axis up to rho_tor = 0.6. Our results show a pronounced density peaking, moderated by electromagnetic fluctuations. The predicted fusion gain for this scenario is Q = 12.2, aligning well with ITER's mission objectives. We further characterize the turbulence spectra and find that electromagnetic modes, such as microtearing modes, kinetic ballooning modes, and Alfvenic ion temperature gradient modes at low binormal wave numbers, play a critical role in the core transport of this ITER scenario, necessitating high numerical resolution for accurate modeling. Local flux-tube simulations qualitatively reproduce the key features observed in the global gyrokinetic simulations but exhibit a much higher sensitivity to profile gradients, reflecting increased stiffness, likely due to the linearization of the equilibrium profiles and safety factor. Our study also reveals that the imposed external toroidal rotation profiles have a negligible impact on turbulent transport, as their magnitudes are substantially lower than the dominant linear growth rates. Furthermore, we demonstrate that the safety factor profile is of paramount importance: scenarios featuring flat q profiles with near-zero magnetic shear lead to the destabilization of kinetic ballooning modes in the plasma core, significantly enhancing turbulent transport and potentially degrading confinement. Finally, although electron temperature gradient turbulence initially appears large, sometimes exceeding ion-scale transport levels, it is ultimately quenched over long timescales by secular evolution of zonal flows, which are weakly damped under the very low collisionality conditions expected in ITER. |
| 2025-11-05 | [Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs](http://arxiv.org/abs/2511.03271v1) | Yize Liu, Yunyun Hou et al. | Large Language Models (LLMs) have been widely deployed across various applications, yet their potential security and ethical risks have raised increasing concerns. Existing research employs red teaming evaluations, utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs. However, these approaches often lack exploration of successful dialogue trajectories within the attack space, and they tend to overlook the considerable overhead associated with the attack process. To address these limitations, this paper first introduces a theoretical model based on dynamically weighted graph topology, abstracting the multi-turn attack process as a path planning problem. Based on this framework, we propose ABC, an enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a collaborative search mechanism with employed, onlooker, and scout bees. This algorithm significantly improves the efficiency of optimal attack path search while substantially reducing the average number of queries required. Empirical evaluations on three open-source and two proprietary language models demonstrate the effectiveness of our approach, achieving attack success rates above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and outperforming existing baselines. Furthermore, it achieves comparable success with only 26 queries on average, significantly reducing red teaming overhead and highlighting its superior efficiency. |
| 2025-11-05 | [IEC3D-AD: A 3D Dataset of Industrial Equipment Components for Unsupervised Point Cloud Anomaly Detection](http://arxiv.org/abs/2511.03267v1) | Bingyang Guo, Hongjie Li et al. | 3D anomaly detection (3D-AD) plays a critical role in industrial manufacturing, particularly in ensuring the reliability and safety of core equipment components. Although existing 3D datasets like Real3D-AD and MVTec 3D-AD offer broad application support, they fall short in capturing the complexities and subtle defects found in real industrial environments. This limitation hampers precise anomaly detection research, especially for industrial equipment components (IEC) such as bearings, rings, and bolts. To address this challenge, we have developed a point cloud anomaly detection dataset (IEC3D-AD) specific to real industrial scenarios. This dataset is directly collected from actual production lines, ensuring high fidelity and relevance. Compared to existing datasets, IEC3D-AD features significantly improved point cloud resolution and defect annotation granularity, facilitating more demanding anomaly detection tasks. Furthermore, inspired by generative 2D-AD methods, we introduce a novel 3D-AD paradigm (GMANet) on IEC3D-AD. This paradigm generates synthetic point cloud samples based on geometric morphological analysis, then reduces the margin and increases the overlap between normal and abnormal point-level features through spatial discrepancy optimization. Extensive experiments demonstrate the effectiveness of our method on both IEC3D-AD and other datasets. |
| 2025-11-05 | [Death by a Thousand Prompts: Open Model Vulnerability Analysis](http://arxiv.org/abs/2511.03247v1) | Amy Chang, Nicholas Conley et al. | Open-weight models provide researchers and developers with accessible foundations for diverse downstream applications. We tested the safety and security postures of eight open-weight large language models (LLMs) to identify vulnerabilities that may impact subsequent fine-tuning and deployment. Using automated adversarial testing, we measured each model's resilience against single-turn and multi-turn prompt injection and jailbreak attacks. Our findings reveal pervasive vulnerabilities across all tested models, with multi-turn attacks achieving success rates between 25.86\% and 92.78\% -- representing a $2\times$ to $10\times$ increase over single-turn baselines. These results underscore a systemic inability of current open-weight models to maintain safety guardrails across extended interactions. We assess that alignment strategies and lab priorities significantly influence resilience: capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma 3 exhibit more balanced performance.   The analysis concludes that open-weight models, while crucial for innovation, pose tangible operational and ethical risks when deployed without layered security controls. These findings are intended to inform practitioners and developers of the potential risks and the value of professional AI security solutions to mitigate exposure. Addressing multi-turn vulnerabilities is essential to ensure the safe, reliable, and responsible deployment of open-weight LLMs in enterprise and public domains. We recommend adopting a security-first design philosophy and layered protections to ensure resilient deployments of open-weight models. |
| 2025-11-05 | [Provable Separations between Memorization and Generalization in Diffusion Models](http://arxiv.org/abs/2511.03202v1) | Zeqi Ye, Qijie Zhu et al. | Diffusion models have achieved remarkable success across diverse domains, but they remain vulnerable to memorization -- reproducing training data rather than generating novel outputs. This not only limits their creative potential but also raises concerns about privacy and safety. While empirical studies have explored mitigation strategies, theoretical understanding of memorization remains limited. We address this gap through developing a dual-separation result via two complementary perspectives: statistical estimation and network approximation. From the estimation side, we show that the ground-truth score function does not minimize the empirical denoising loss, creating a separation that drives memorization. From the approximation side, we prove that implementing the empirical score function requires network size to scale with sample size, spelling a separation compared to the more compact network representation of the ground-truth score function. Guided by these insights, we develop a pruning-based method that reduces memorization while maintaining generation quality in diffusion transformers. |
| 2025-11-05 | [Collaborative Assembly Policy Learning of a Sightless Robot](http://arxiv.org/abs/2511.03189v1) | Zeqing Zhang, Weifeng Lu et al. | This paper explores a physical human-robot collaboration (pHRC) task involving the joint insertion of a board into a frame by a sightless robot and a human operator. While admittance control is commonly used in pHRC tasks, it can be challenging to measure the force/torque applied by the human for accurate human intent estimation, limiting the robot's ability to assist in the collaborative task. Other methods that attempt to solve pHRC tasks using reinforcement learning (RL) are also unsuitable for the board-insertion task due to its safety constraints and sparse rewards. Therefore, we propose a novel RL approach that utilizes a human-designed admittance controller to facilitate more active robot behavior and reduce human effort. Through simulation and real-world experiments, we demonstrate that our approach outperforms admittance control in terms of success rate and task completion time. Additionally, we observed a significant reduction in measured force/torque when using our proposed approach compared to admittance control. The video of the experiments is available at https://youtu.be/va07Gw6YIog. |
| 2025-11-05 | [A Proprietary Model-Based Safety Response Framework for AI Agents](http://arxiv.org/abs/2511.03138v1) | Qi Li, Jianjun Xu et al. | With the widespread application of Large Language Models (LLMs), their associated security issues have become increasingly prominent, severely constraining their trustworthy deployment in critical domains. This paper proposes a novel safety response framework designed to systematically safeguard LLMs at both the input and output levels. At the input level, the framework employs a supervised fine-tuning-based safety classification model. Through a fine-grained four-tier taxonomy (Safe, Unsafe, Conditionally Safe, Focused Attention), it performs precise risk identification and differentiated handling of user queries, significantly enhancing risk coverage and business scenario adaptability, and achieving a risk recall rate of 99.3%. At the output level, the framework integrates Retrieval-Augmented Generation (RAG) with a specifically fine-tuned interpretation model, ensuring all responses are grounded in a real-time, trustworthy knowledge base. This approach eliminates information fabrication and enables result traceability. Experimental results demonstrate that our proposed safety control model achieves a significantly higher safety score on public safety evaluation benchmarks compared to the baseline model, TinyR1-Safety-8B. Furthermore, on our proprietary high-risk test set, the framework's components attained a perfect 100% safety score, validating their exceptional protective capabilities in complex risk scenarios. This research provides an effective engineering pathway for building high-security, high-trust LLM applications. |
| 2025-11-05 | [Control Barrier Function for Aligning Large Language Models](http://arxiv.org/abs/2511.03121v1) | Yuya Miyaoka, Masaki Inoue | This paper proposes a control-based framework for aligning large language models (LLMs) by leveraging a control barrier function (CBF) to ensure user-desirable text generation. The presented framework applies the CBF safety filter to the predicted token generated from the baseline LLM, to intervene in the generated text. The safety filter includes two significant advantages: this safety filter is an add-on type, allowing it to be used for alignment purposes without fine-tuning the baseline LLM, and if there is an evaluation model regarding the desired alignment, it can be directly applied to the filter design. The overall text-generation system is implemented with open-source language models, aiming to generate positive text. |
| 2025-11-05 | [Control Barrier Function for Aligning Large Language Models](http://arxiv.org/abs/2511.03121v2) | Yuya Miyaoka, Masaki Inoue | This paper proposes a control-based framework for aligning large language models (LLMs) by leveraging a control barrier function (CBF) to ensure user-desirable text generation. The presented framework applies the CBF safety filter to the predicted token generated from the baseline LLM, to intervene in the generated text. The safety filter includes two significant advantages: this safety filter is an add-on type, allowing it to be used for alignment purposes without fine-tuning the baseline LLM, and if there is an evaluation model regarding the desired alignment, it can be directly applied to the filter design. The overall text-generation system is implemented with open-source language models, aiming to generate positive text. |
| 2025-11-05 | [Large language models require a new form of oversight: capability-based monitoring](http://arxiv.org/abs/2511.03106v1) | Katherine C. Kellogg, Bingyang Ye et al. | The rapid adoption of large language models (LLMs) in healthcare has been accompanied by scrutiny of their oversight. Existing monitoring approaches, inherited from traditional machine learning (ML), are task-based and founded on assumed performance degradation arising from dataset drift. In contrast, with LLMs, inevitable model degradation due to changes in populations compared to the training dataset cannot be assumed, because LLMs were not trained for any specific task in any given population. We therefore propose a new organizing principle guiding generalist LLM monitoring that is scalable and grounded in how these models are developed and used in practice: capability-based monitoring. Capability-based monitoring is motivated by the fact that LLMs are generalist systems whose overlapping internal capabilities are reused across numerous downstream tasks. Instead of evaluating each downstream task independently, this approach organizes monitoring around shared model capabilities, such as summarization, reasoning, translation, or safety guardrails, in order to enable cross-task detection of systemic weaknesses, long-tail errors, and emergent behaviors that task-based monitoring may miss. We describe considerations for developers, organizational leaders, and professional societies for implementing a capability-based monitoring approach. Ultimately, capability-based monitoring will provide a scalable foundation for safe, adaptive, and collaborative monitoring of LLMs and future generalist artificial intelligence models in healthcare. |
| 2025-11-05 | [ISC-Perception: A Hybrid Computer Vision Dataset for Object Detection in Novel Steel Assembly](http://arxiv.org/abs/2511.03098v1) | Miftahur Rahman, Samuel Adebayo et al. | The Intermeshed Steel Connection (ISC) system, when paired with robotic manipulators, can accelerate steel-frame assembly and improve worker safety by eliminating manual assembly. Dependable perception is one of the initial stages for ISC-aware robots. However, this is hampered by the absence of a dedicated image corpus, as collecting photographs on active construction sites is logistically difficult and raises safety and privacy concerns. In response, we introduce ISC-Perception, the first hybrid dataset expressly designed for ISC component detection. It blends procedurally rendered CAD images, game-engine photorealistic scenes, and a limited, curated set of real photographs, enabling fully automatic labelling of the synthetic portion. We explicitly account for all human effort to produce the dataset, including simulation engine and scene setup, asset preparation, post-processing scripts and quality checks; our total human time to generate a 10,000-image dataset was 30.5,h versus 166.7,h for manual labelling at 60,s per image (-81.7%). A manual pilot on a representative image with five instances of ISC members took 60,s (maximum 80,s), anchoring the manual baseline. Detectors trained on ISC-Perception achieved a mean Average Precision at IoU 0.50 of 0.756, substantially surpassing models trained on synthetic-only or photorealistic-only data. On a 1,200-frame bench test, we report mAP@0.50/mAP@[0.50:0.95] of 0.943/0.823. By bridging the data gap for construction-robotics perception, ISC-Perception facilitates rapid development of custom object detectors and is freely available for research and industrial use upon request. |
| 2025-11-05 | [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/abs/2511.03092v1) | Jonathan Li, Nasim Farahini et al. | The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+ context length support have resulted in increasing demands for on-chip memory to support large KV caches. Techniques such as StreamingLLM and SnapKV demonstrate how to control KV cache size while maintaining model accuracy. Yet, these techniques are not commonly used within industrial deployments using frameworks like vLLM or SGLang. The reason is twofold: on one hand, the static graphs and continuous batching methodology employed by these frameworks make it difficult to admit modifications to the standard multi-head attention algorithm, while on the other hand, the accuracy implications of such techniques on modern instruction-following and reasoning models are not well understood, obfuscating the need for implementing these techniques. In this paper, we explore these accuracy implications on Llama-3.1-8B-Instruct and DeepSeek-R1, and develop SnapStream, a KV cache compression method that can be deployed at scale. We demonstrate the efficacy of SnapStream in a 16-way tensor-parallel deployment of DeepSeek-671B on SambaNova SN40L accelerators running at 128k context length and up to 1832 tokens per second in a real production setting. SnapStream enables $4\times$ improved on-chip memory usage and introduces minimal accuracy degradation on LongBench-v2, AIME24 and LiveCodeBench. To the best of our knowledge, this is the first implementation of sparse KV attention techniques deployed in a production inference system with static graphs and continuous batching. |
| 2025-11-05 | [SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators](http://arxiv.org/abs/2511.03092v2) | Jonathan Li, Nasim Farahini et al. | The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+ context length support have resulted in increasing demands for on-chip memory to support large KV caches. Techniques such as StreamingLLM and SnapKV demonstrate how to control KV cache size while maintaining model accuracy. Yet, these techniques are not commonly used within industrial deployments using frameworks like vLLM or SGLang. The reason is twofold: on one hand, the static graphs and continuous batching methodology employed by these frameworks make it difficult to admit modifications to the standard multi-head attention algorithm, while on the other hand, the accuracy implications of such techniques on modern instruction-following and reasoning models are not well understood, obfuscating the need for implementing these techniques. In this paper, we explore these accuracy implications on Llama-3.1-8B-Instruct and DeepSeek-R1, and develop SnapStream, a KV cache compression method that can be deployed at scale. We demonstrate the efficacy of SnapStream in a 16-way tensor-parallel deployment of DeepSeek-671B on SambaNova SN40L accelerators running at 128k context length and up to 1832 tokens per second in a real production setting. SnapStream enables $4\times$ improved on-chip memory usage and introduces minimal accuracy degradation on LongBench-v2, AIME24 and LiveCodeBench. To the best of our knowledge, this is the first implementation of sparse KV attention techniques deployed in a production inference system with static graphs and continuous batching. |
| 2025-11-04 | [A Collaborative Reasoning Framework for Anomaly Diagnostics in Underwater Robotics](http://arxiv.org/abs/2511.03075v1) | Markus Buchholz, Ignacio Carlucho et al. | The safe deployment of autonomous systems in safety-critical settings requires a paradigm that combines human expertise with AI-driven analysis, especially when anomalies are unforeseen. We introduce AURA (Autonomous Resilience Agent), a collaborative framework for anomaly and fault diagnostics in robotics. AURA integrates large language models (LLMs), a high-fidelity digital twin (DT), and human-in-the-loop interaction to detect and respond to anomalous behavior in real time. The architecture uses two agents with clear roles: (i) a low-level State Anomaly Characterization Agent that monitors telemetry and converts signals into a structured natural-language problem description, and (ii) a high-level Diagnostic Reasoning Agent that conducts a knowledge-grounded dialogue with an operator to identify root causes, drawing on external sources. Human-validated diagnoses are then converted into new training examples that refine the low-level perceptual model. This feedback loop progressively distills expert knowledge into the AI, transforming it from a static tool into an adaptive partner. We describe the framework's operating principles and provide a concrete implementation, establishing a pattern for trustworthy, continually improving human-robot teams. |

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



