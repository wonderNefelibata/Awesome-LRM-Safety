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
| 2025-04-28 | [Modular Machine Learning: An Indispensable Path towards New-Generation Large Language Models](http://arxiv.org/abs/2504.20020v1) | Xin Wang, Haoyang Li et al. | Large language models (LLMs) have dramatically advanced machine learning research including natural language processing, computer vision, data mining, etc., yet they still exhibit critical limitations in reasoning, factual consistency, and interpretability. In this paper, we introduce a novel learning paradigm -- Modular Machine Learning (MML) -- as an essential approach toward new-generation LLMs. MML decomposes the complex structure of LLMs into three interdependent components: modular representation, modular model, and modular reasoning, aiming to enhance LLMs' capability of counterfactual reasoning, mitigating hallucinations, as well as promoting fairness, safety, and transparency. Specifically, the proposed MML paradigm can: i) clarify the internal working mechanism of LLMs through the disentanglement of semantic components; ii) allow for flexible and task-adaptive model design; iii) enable interpretable and logic-driven decision-making process. We present a feasible implementation of MML-based LLMs via leveraging advanced techniques such as disentangled representation learning, neural architecture search and neuro-symbolic learning. We critically identify key challenges, such as the integration of continuous neural and discrete symbolic processes, joint optimization, and computational scalability, present promising future research directions that deserve further exploration. Ultimately, the integration of the MML paradigm with LLMs has the potential to bridge the gap between statistical (deep) learning and formal (logical) reasoning, thereby paving the way for robust, adaptable, and trustworthy AI systems across a wide range of real-world applications. |
| 2025-04-28 | [Socially-Aware Autonomous Driving: Inferring Yielding Intentions for Safer Interactions](http://arxiv.org/abs/2504.20004v1) | Jing Wang, Yan Jin et al. | Since the emergence of autonomous driving technology, it has advanced rapidly over the past decade. It is becoming increasingly likely that autonomous vehicles (AVs) would soon coexist with human-driven vehicles (HVs) on the roads. Currently, safety and reliable decision-making remain significant challenges, particularly when AVs are navigating lane changes and interacting with surrounding HVs. Therefore, precise estimation of the intentions of surrounding HVs can assist AVs in making more reliable and safe lane change decision-making. This involves not only understanding their current behaviors but also predicting their future motions without any direct communication. However, distinguishing between the passing and yielding intentions of surrounding HVs still remains ambiguous. To address the challenge, we propose a social intention estimation algorithm rooted in Directed Acyclic Graph (DAG), coupled with a decision-making framework employing Deep Reinforcement Learning (DRL) algorithms. To evaluate the method's performance, the proposed framework can be tested and applied in a lane-changing scenario within a simulated environment. Furthermore, the experiment results demonstrate how our approach enhances the ability of AVs to navigate lane changes safely and efficiently on roads. |
| 2025-04-28 | [Mitigating Societal Cognitive Overload in the Age of AI: Challenges and Directions](http://arxiv.org/abs/2504.19990v1) | Salem Lahlou | Societal cognitive overload, driven by the deluge of information and complexity in the AI age, poses a critical challenge to human well-being and societal resilience. This paper argues that mitigating cognitive overload is not only essential for improving present-day life but also a crucial prerequisite for navigating the potential risks of advanced AI, including existential threats. We examine how AI exacerbates cognitive overload through various mechanisms, including information proliferation, algorithmic manipulation, automation anxieties, deregulation, and the erosion of meaning. The paper reframes the AI safety debate to center on cognitive overload, highlighting its role as a bridge between near-term harms and long-term risks. It concludes by discussing potential institutional adaptations, research directions, and policy considerations that arise from adopting an overload-resilient perspective on human-AI alignment, suggesting pathways for future exploration rather than prescribing definitive solutions. |
| 2025-04-28 | [HJRNO: Hamilton-Jacobi Reachability with Neural Operators](http://arxiv.org/abs/2504.19989v1) | Yankai Li, Mo Chen | Ensuring the safety of autonomous systems under uncertainty is a critical challenge. Hamilton-Jacobi reachability (HJR) analysis is a widely used method for guaranteeing safety under worst-case disturbances. Traditional HJR methods provide safety guarantees but suffer from the curse of dimensionality, limiting their scalability to high-dimensional systems or varying environmental conditions. In this work, we propose HJRNO, a neural operator-based framework for solving backward reachable tubes (BRTs) efficiently and accurately. By leveraging the Fourier Neural Operator (FNO), HJRNO learns a mapping between value functions, enabling fast inference with strong generalization across different obstacle shapes, system configurations, and hyperparameters. We demonstrate that HJRNO achieves low error on random obstacle scenarios and generalizes effectively across varying system dynamics. These results suggest that HJRNO offers a promising foundation model approach for scalable, real-time safety analysis in autonomous systems. |
| 2025-04-28 | [Can AI Agents Design and Implement Drug Discovery Pipelines?](http://arxiv.org/abs/2504.19912v1) | Khachik Smbatyan, Tsolak Ghukasyan et al. | The rapid advancement of artificial intelligence, particularly autonomous agentic systems based on Large Language Models (LLMs), presents new opportunities to accelerate drug discovery by improving in-silico modeling and reducing dependence on costly experimental trials. Current AI agent-based systems demonstrate proficiency in solving programming challenges and conducting research, indicating an emerging potential to develop software capable of addressing complex problems such as pharmaceutical design and drug discovery. This paper introduces DO Challenge, a benchmark designed to evaluate the decision-making abilities of AI agents in a single, complex problem resembling virtual screening scenarios. The benchmark challenges systems to independently develop, implement, and execute efficient strategies for identifying promising molecular structures from extensive datasets, while navigating chemical space, selecting models, and managing limited resources in a multi-objective context. We also discuss insights from the DO Challenge 2025, a competition based on the proposed benchmark, which showcased diverse strategies explored by human participants. Furthermore, we present the Deep Thought multi-agent system, which demonstrated strong performance on the benchmark, outperforming most human teams. Among the language models tested, Claude 3.7 Sonnet, Gemini 2.5 Pro and o3 performed best in primary agent roles, and GPT-4o, Gemini 2.0 Flash were effective in auxiliary roles. While promising, the system's performance still fell short of expert-designed solutions and showed high instability, highlighting both the potential and current limitations of AI-driven methodologies in transforming drug discovery and broader scientific research. |
| 2025-04-28 | [Human-Centered AI and Autonomy in Robotics: Insights from a Bibliometric Study](http://arxiv.org/abs/2504.19848v1) | Simona Casini, Pietro Ducange et al. | The development of autonomous robotic systems offers significant potential for performing complex tasks with precision and consistency. Recent advances in Artificial Intelligence (AI) have enabled more capable intelligent automation systems, addressing increasingly complex challenges. However, this progress raises questions about human roles in such systems. Human-Centered AI (HCAI) aims to balance human control and automation, ensuring performance enhancement while maintaining creativity, mastery, and responsibility. For real-world applications, autonomous robots must balance task performance with reliability, safety, and trustworthiness. Integrating HCAI principles enhances human-robot collaboration and ensures responsible operation.   This paper presents a bibliometric analysis of intelligent autonomous robotic systems, utilizing SciMAT and VOSViewer to examine data from the Scopus database. The findings highlight academic trends, emerging topics, and AI's role in self-adaptive robotic behaviour, with an emphasis on HCAI architecture. These insights are then projected onto the IBM MAPE-K architecture, with the goal of identifying how these research results map into actual robotic autonomous systems development efforts for real-world scenarios. |
| 2025-04-28 | [Measuring Train Driver Performance as Key to Approval of Driverless Trains](http://arxiv.org/abs/2504.19735v1) | Rustam Tagiew, Prasannavenkatesh Balaji | Points 2.1.4(b), 2.4.2(b) and 2.4.3(b) in Annex I of Implementing Regulation (EU) No. 402/2013 allow a simplified approach for the safety approval of computer vision systems for driverless trains, if they have 'similar' functions and interfaces as the replaced human driver. The human driver is not replaced one-to-one by a technical system - only a limited set of cognitive functions are replaced. However, performance in the most challenging function, obstacle detection, is difficult to quantify due to the deficiency of published measurement results. This article summarizes the data published so far. This article also goes a long way to remedy this situation by providing a new public and anonymized dataset of 711 train driver performance measurements from controlled experiments. The measurements are made for different speeds, obstacle sizes, train protection systems and obstacle color contrasts respectively. The measured values are reaction time and distance to the obstacle. The goal of this paper is an unbiased and exhaustive description of the presented dataset for research, standardization and regulation. Further project related information including the dataset and source code is available at https://atosense-02371c.usercontent.opencode.de/ |
| 2025-04-28 | [Hector UI: A Flexible Human-Robot User Interface for (Semi-)Autonomous Rescue and Inspection Robots](http://arxiv.org/abs/2504.19728v1) | Stefan Fabian, Oskar von Stryk | The remote human operator's user interface (UI) is an important link to make the robot an efficient extension of the operator's perception and action. In rescue applications, several studies have investigated the design of operator interfaces based on observations during major robotics competitions or field deployments. Based on this research, guidelines for good interface design were empirically identified. The investigations on the UIs of teams participating in competitions are often based on external observations during UI application, which may miss some relevant requirements for UI flexibility. In this work, we present an open-source and flexibly configurable user interface based on established guidelines and its exemplary use for wheeled, tracked, and walking robots. We explain the design decisions and cover the insights we have gained during its highly successful applications in multiple robotics competitions and evaluations. The presented UI can also be adapted for other robots with little effort and is available as open source. |
| 2025-04-28 | [Open-set Anomaly Segmentation in Complex Scenarios](http://arxiv.org/abs/2504.19706v1) | Song Xia, Yi Yu et al. | Precise segmentation of out-of-distribution (OoD) objects, herein referred to as anomalies, is crucial for the reliable deployment of semantic segmentation models in open-set, safety-critical applications, such as autonomous driving. Current anomalous segmentation benchmarks predominantly focus on favorable weather conditions, resulting in untrustworthy evaluations that overlook the risks posed by diverse meteorological conditions in open-set environments, such as low illumination, dense fog, and heavy rain. To bridge this gap, this paper introduces the ComsAmy, a challenging benchmark specifically designed for open-set anomaly segmentation in complex scenarios. ComsAmy encompasses a wide spectrum of adverse weather conditions, dynamic driving environments, and diverse anomaly types to comprehensively evaluate the model performance in realistic open-world scenarios. Our extensive evaluation of several state-of-the-art anomalous segmentation models reveals that existing methods demonstrate significant deficiencies in such challenging scenarios, highlighting their serious safety risks for real-world deployment. To solve that, we propose a novel energy-entropy learning (EEL) strategy that integrates the complementary information from energy and entropy to bolster the robustness of anomaly segmentation under complex open-world environments. Additionally, a diffusion-based anomalous training data synthesizer is proposed to generate diverse and high-quality anomalous images to enhance the existing copy-paste training data synthesizer. Extensive experimental results on both public and ComsAmy benchmarks demonstrate that our proposed diffusion-based synthesizer with energy and entropy learning (DiffEEL) serves as an effective and generalizable plug-and-play method to enhance existing models, yielding an average improvement of around 4.96% in $\rm{AUPRC}$ and 9.87% in $\rm{FPR}_{95}$. |
| 2025-04-28 | [$\texttt{SAGE}$: A Generic Framework for LLM Safety Evaluation](http://arxiv.org/abs/2504.19674v1) | Madhur Jindal, Hari Shrawgi et al. | Safety evaluation of Large Language Models (LLMs) has made progress and attracted academic interest, but it remains challenging to keep pace with the rapid integration of LLMs across diverse applications. Different applications expose users to various harms, necessitating application-specific safety evaluations with tailored harms and policies. Another major gap is the lack of focus on the dynamic and conversational nature of LLM systems. Such potential oversights can lead to harms that go unnoticed in standard safety benchmarks. This paper identifies the above as key requirements for robust LLM safety evaluation and recognizing that current evaluation methodologies do not satisfy these, we introduce the $\texttt{SAGE}$ (Safety AI Generic Evaluation) framework. $\texttt{SAGE}$ is an automated modular framework designed for customized and dynamic harm evaluations. It utilizes adversarial user models that are system-aware and have unique personalities, enabling a holistic red-teaming evaluation. We demonstrate $\texttt{SAGE}$'s effectiveness by evaluating seven state-of-the-art LLMs across three applications and harm policies. Our experiments with multi-turn conversational evaluations revealed a concerning finding that harm steadily increases with conversation length. Furthermore, we observe significant disparities in model behavior when exposed to different user personalities and scenarios. Our findings also reveal that some models minimize harmful outputs by employing severe refusal tactics that can hinder their usefulness. These insights highlight the necessity of adaptive and context-specific testing to ensure better safety alignment and safer deployment of LLMs in real-world scenarios. |
| 2025-04-28 | [Fooling the Decoder: An Adversarial Attack on Quantum Error Correction](http://arxiv.org/abs/2504.19651v1) | Jerome Lenssen, Alexandru Paler | Neural network decoders are becoming essential for achieving fault-tolerant quantum computations. However, their internal mechanisms are poorly understood, hindering our ability to ensure their reliability and security against adversarial attacks. Leading machine learning decoders utilize recurrent and transformer models (e.g., AlphaQubit), with reinforcement learning (RL) playing a key role in training advanced transformer models (e.g., DeepSeek R1). In this work, we target a basic RL surface code decoder (DeepQ) to create the first adversarial attack on quantum error correction. By applying state-of-the-art white-box methods, we uncover vulnerabilities in this decoder, demonstrating an attack that reduces the logical qubit lifetime in memory experiments by up to five orders of magnitude. We validate that this attack exploits a genuine weakness, as the decoder exhibits robustness against noise fluctuations, is largely unaffected by substituting the referee decoder, responsible for episode termination, with an MWPM decoder, and demonstrates fault tolerance at checkable code distances. This attack highlights the susceptibility of machine learning-based QEC and underscores the importance of further research into robust QEC methods. |
| 2025-04-28 | [ARMOR: Adaptive Meshing with Reinforcement Optimization for Real-time 3D Monitoring in Unexposed Scenes](http://arxiv.org/abs/2504.19624v1) | Yizhe Zhang, Jianping Li et al. | Unexposed environments, such as lava tubes, mines, and tunnels, are among the most complex yet strategically significant domains for scientific exploration and infrastructure development. Accurate and real-time 3D meshing of these environments is essential for applications including automated structural assessment, robotic-assisted inspection, and safety monitoring. Implicit neural Signed Distance Fields (SDFs) have shown promising capabilities in online meshing; however, existing methods often suffer from large projection errors and rely on fixed reconstruction parameters, limiting their adaptability to complex and unstructured underground environments such as tunnels, caves, and lava tubes. To address these challenges, this paper proposes ARMOR, a scene-adaptive and reinforcement learning-based framework for real-time 3D meshing in unexposed environments. The proposed method was validated across more than 3,000 meters of underground environments, including engineered tunnels, natural caves, and lava tubes. Experimental results demonstrate that ARMOR achieves superior performance in real-time mesh reconstruction, reducing geometric error by 3.96\% compared to state-of-the-art baselines, while maintaining real-time efficiency. The method exhibits improved robustness, accuracy, and adaptability, indicating its potential for advanced 3D monitoring and mapping in challenging unexposed scenarios. The project page can be found at: https://yizhezhang0418.github.io/armor.github.io/ |
| 2025-04-28 | [AI Alignment in Medical Imaging: Unveiling Hidden Biases Through Counterfactual Analysis](http://arxiv.org/abs/2504.19621v1) | Haroui Ma, Francesco Quinzan et al. | Machine learning (ML) systems for medical imaging have demonstrated remarkable diagnostic capabilities, but their susceptibility to biases poses significant risks, since biases may negatively impact generalization performance. In this paper, we introduce a novel statistical framework to evaluate the dependency of medical imaging ML models on sensitive attributes, such as demographics. Our method leverages the concept of counterfactual invariance, measuring the extent to which a model's predictions remain unchanged under hypothetical changes to sensitive attributes. We present a practical algorithm that combines conditional latent diffusion models with statistical hypothesis testing to identify and quantify such biases without requiring direct access to counterfactual data. Through experiments on synthetic datasets and large-scale real-world medical imaging datasets, including \textsc{cheXpert} and MIMIC-CXR, we demonstrate that our approach aligns closely with counterfactual fairness principles and outperforms standard baselines. This work provides a robust tool to ensure that ML diagnostic systems generalize well, e.g., across demographic groups, offering a critical step towards AI safety in healthcare. Code: https://github.com/Neferpitou3871/AI-Alignment-Medical-Imaging. |
| 2025-04-28 | [Crowd Detection Using Very-Fine-Resolution Satellite Imagery](http://arxiv.org/abs/2504.19546v1) | Tong Xiao, Qunming Wang et al. | Accurate crowd detection (CD) is critical for public safety and historical pattern analysis, yet existing methods relying on ground and aerial imagery suffer from limited spatio-temporal coverage. The development of very-fine-resolution (VFR) satellite sensor imagery (e.g., ~0.3 m spatial resolution) provides unprecedented opportunities for large-scale crowd activity analysis, but it has never been considered for this task. To address this gap, we proposed CrowdSat-Net, a novel point-based convolutional neural network, which features two innovative components: Dual-Context Progressive Attention Network (DCPAN) to improve feature representation of individuals by aggregating scene context and local individual characteristics, and High-Frequency Guided Deformable Upsampler (HFGDU) that recovers high-frequency information during upsampling through frequency-domain guided deformable convolutions. To validate the effectiveness of CrowdSat-Net, we developed CrowdSat, the first VFR satellite imagery dataset designed specifically for CD tasks, comprising over 120k manually labeled individuals from multi-source satellite platforms (Beijing-3N, Jilin-1 Gaofen-04A and Google Earth) across China. In the experiments, CrowdSat-Net was compared with five state-of-the-art point-based CD methods (originally designed for ground or aerial imagery) using CrowdSat and achieved the largest F1-score of 66.12% and Precision of 73.23%, surpassing the second-best method by 1.71% and 2.42%, respectively. Moreover, extensive ablation experiments validated the importance of the DCPAN and HFGDU modules. Furthermore, cross-regional evaluation further demonstrated the spatial generalizability of CrowdSat-Net. This research advances CD capability by providing both a newly developed network architecture for CD and a pioneering benchmark dataset to facilitate future CD development. |
| 2025-04-28 | [Security Steerability is All You Need](http://arxiv.org/abs/2504.19521v1) | Itay Hazan, Idan Habler et al. | The adoption of Generative AI (GenAI) in various applications inevitably comes with expanding the attack surface, combining new security threats along with the traditional ones. Consequently, numerous research and industrial initiatives aim to mitigate these security threats in GenAI by developing metrics and designing defenses. However, while most of the GenAI security work focuses on universal threats (e.g. manipulating the LLM to generate forbidden content), there is significantly less discussion on application-level security and how to mitigate it.   Thus, in this work we adopt an application-centric approach to GenAI security, and show that while LLMs cannot protect against ad-hoc application specific threats, they can provide the framework for applications to protect themselves against such threats. Our first contribution is defining Security Steerability - a novel security measure for LLMs, assessing the model's capability to adhere to strict guardrails that are defined in the system prompt ('Refrain from discussing about politics'). These guardrails, in case effective, can stop threats in the presence of malicious users who attempt to circumvent the application and cause harm to its providers.   Our second contribution is a methodology to measure the security steerability of LLMs, utilizing two newly-developed datasets: VeganRibs assesses the LLM behavior in forcing specific guardrails that are not security per se in the presence of malicious user that uses attack boosters (jailbreaks and perturbations), and ReverseText takes this approach further and measures the LLM ability to force specific treatment of the user input as plain text while do user try to give it additional meanings... |
| 2025-04-28 | [BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text](http://arxiv.org/abs/2504.19467v1) | Jiageng Wu, Bowen Gu et al. | Large language models (LLMs) hold great promise for medical applications and are evolving rapidly, with new models being released at an accelerated pace. However, current evaluations of LLMs in clinical contexts remain limited. Most existing benchmarks rely on medical exam-style questions or PubMed-derived text, failing to capture the complexity of real-world electronic health record (EHR) data. Others focus narrowly on specific application scenarios, limiting their generalizability across broader clinical use. To address this gap, we present BRIDGE, a comprehensive multilingual benchmark comprising 87 tasks sourced from real-world clinical data sources across nine languages. We systematically evaluated 52 state-of-the-art LLMs (including DeepSeek-R1, GPT-4o, Gemini, and Llama 4) under various inference strategies. With a total of 13,572 experiments, our results reveal substantial performance variation across model sizes, languages, natural language processing tasks, and clinical specialties. Notably, we demonstrate that open-source LLMs can achieve performance comparable to proprietary models, while medically fine-tuned LLMs based on older architectures often underperform versus updated general-purpose models. The BRIDGE and its corresponding leaderboard serve as a foundational resource and a unique reference for the development and evaluation of new LLMs in real-world clinical text understanding. |
| 2025-04-28 | [JailbreaksOverTime: Detecting Jailbreak Attacks Under Distribution Shift](http://arxiv.org/abs/2504.19440v1) | Julien Piet, Xiao Huang et al. | Safety and security remain critical concerns in AI deployment. Despite safety training through reinforcement learning with human feedback (RLHF) [ 32], language models remain vulnerable to jailbreak attacks that bypass safety guardrails. Universal jailbreaks - prefixes that can circumvent alignment for any payload - are particularly concerning. We show empirically that jailbreak detection systems face distribution shift, with detectors trained at one point in time performing poorly against newer exploits. To study this problem, we release JailbreaksOverTime, a comprehensive dataset of timestamped real user interactions containing both benign requests and jailbreak attempts collected over 10 months. We propose a two-pronged method for defenders to detect new jailbreaks and continuously update their detectors. First, we show how to use continuous learning to detect jailbreaks and adapt rapidly to new emerging jailbreaks. While detectors trained at a single point in time eventually fail due to drift, we find that universal jailbreaks evolve slowly enough for self-training to be effective. Retraining our detection model weekly using its own labels - with no new human labels - reduces the false negative rate from 4% to 0.3% at a false positive rate of 0.1%. Second, we introduce an unsupervised active monitoring approach to identify novel jailbreaks. Rather than classifying inputs directly, we recognize jailbreaks by their behavior, specifically, their ability to trigger models to respond to known-harmful prompts. This approach has a higher false negative rate (4.1%) than supervised methods, but it successfully identified some out-of-distribution attacks that were missed by the continuous learning approach. |
| 2025-04-27 | [Doxing via the Lens: Revealing Privacy Leakage in Image Geolocation for Agentic Multi-Modal Large Reasoning Model](http://arxiv.org/abs/2504.19373v1) | Weidi Luo, Qiming Zhang et al. | The increasing capabilities of agentic multi-modal large reasoning models, such as ChatGPT o3, have raised critical concerns regarding privacy leakage through inadvertent image geolocation. In this paper, we conduct the first systematic and controlled study on the potential privacy risks associated with visual reasoning abilities of ChatGPT o3. We manually collect and construct a dataset comprising 50 real-world images that feature individuals alongside privacy-relevant environmental elements, capturing realistic and sensitive scenarios for analysis. Our experimental evaluation reveals that ChatGPT o3 can predict user locations with high precision, achieving street-level accuracy (within one mile) in 60% of cases. Through analysis, we identify key visual cues, including street layout and front yard design, that significantly contribute to the model inference success. Additionally, targeted occlusion experiments demonstrate that masking critical features effectively mitigates geolocation accuracy, providing insights into potential defense mechanisms. Our findings highlight an urgent need for privacy-aware development for agentic multi-modal large reasoning models, particularly in applications involving private imagery. |
| 2025-04-27 | [Synthesis of Discrete-time Control Barrier Functions for Polynomial Systems Based on Sum-of-Squares Programming](http://arxiv.org/abs/2504.19330v1) | Erfan Shakhesi, W. P. M. H. et al. | Discrete-time Control Barrier Functions (DTCBFs) are commonly utilized in the literature as a powerful tool for synthesizing control policies that guarantee safety of discrete-time dynamical systems. However, the systematic synthesis of DTCBFs in a computationally efficient way is at present an important open problem. This article first proposes a novel alternating-descent approach based on Sum-of-Squares programming to synthesize quadratic DTCBFs and corresponding polynomial control policies for discrete-time control-affine polynomial systems with input constraints and semi-algebraic safe sets. Subsequently, two distinct approaches are introduced to extend the proposed method to the synthesis of higher-degree polynomial DTCBFs. To demonstrate its efficacy, we apply the proposed method to numerical case studies. |
| 2025-04-27 | [Learned Perceptive Forward Dynamics Model for Safe and Platform-aware Robotic Navigation](http://arxiv.org/abs/2504.19322v1) | Pascal Roth, Jonas Frey et al. | Ensuring safe navigation in complex environments requires accurate real-time traversability assessment and understanding of environmental interactions relative to the robot`s capabilities. Traditional methods, which assume simplified dynamics, often require designing and tuning cost functions to safely guide paths or actions toward the goal. This process is tedious, environment-dependent, and not generalizable.To overcome these issues, we propose a novel learned perceptive Forward Dynamics Model (FDM) that predicts the robot`s future state conditioned on the surrounding geometry and history of proprioceptive measurements, proposing a more scalable, safer, and heuristic-free solution. The FDM is trained on multiple years of simulated navigation experience, including high-risk maneuvers, and real-world interactions to incorporate the full system dynamics beyond rigid body simulation. We integrate our perceptive FDM into a zero-shot Model Predictive Path Integral (MPPI) planning framework, leveraging the learned mapping between actions, future states, and failure probability. This allows for optimizing a simplified cost function, eliminating the need for extensive cost-tuning to ensure safety. On the legged robot ANYmal, the proposed perceptive FDM improves the position estimation by on average 41% over competitive baselines, which translates into a 27% higher navigation success rate in rough simulation environments. Moreover, we demonstrate effective sim-to-real transfer and showcase the benefit of training on synthetic and real data. Code and models are made publicly available under https://github.com/leggedrobotics/fdm. |

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



