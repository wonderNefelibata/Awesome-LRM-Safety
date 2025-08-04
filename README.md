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
| 2025-08-01 | [Cross-Dataset Semantic Segmentation Performance Analysis: Unifying NIST Point Cloud City Datasets for 3D Deep Learning](http://arxiv.org/abs/2508.00822v1) | Alexander Nikitas Dimopoulos, Joseph Grasso | This study analyzes semantic segmentation performance across heterogeneously labeled point-cloud datasets relevant to public safety applications, including pre-incident planning systems derived from lidar scans. Using NIST's Point Cloud City dataset (Enfield and Memphis collections), we investigate challenges in unifying differently labeled 3D data. Our methodology employs a graded schema with the KPConv architecture, evaluating performance through IoU metrics on safety-relevant features. Results indicate performance variability: geometrically large objects (e.g. stairs, windows) achieve higher segmentation performance, suggesting potential for navigational context, while smaller safety-critical features exhibit lower recognition rates. Performance is impacted by class imbalance and the limited geometric distinction of smaller objects in typical lidar scans, indicating limitations in detecting certain safety-relevant features using current point-cloud methods. Key identified challenges include insufficient labeled data, difficulties in unifying class labels across datasets, and the need for standardization. Potential directions include automated labeling and multi-dataset learning strategies. We conclude that reliable point-cloud semantic segmentation for public safety necessitates standardized annotation protocols and improved labeling techniques to address data heterogeneity and the detection of small, safety-critical elements. |
| 2025-08-01 | [Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data](http://arxiv.org/abs/2508.00741v1) | Sohaib Imran, Rob Lamb et al. | Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety. |
| 2025-08-01 | [Evac-Cast: An Interpretable Machine-Learning Framework for Evacuation Forecasts Across Hurricanes and Wildfires](http://arxiv.org/abs/2508.00650v1) | Bo Li, Chenyue Liu et al. | Evacuation is critical for disaster safety, yet agencies lack timely, accurate, and transparent tools for evacuation prediction. This study introduces Evac-Cast, an interpretable machine learning framework that predicts tract-level evacuation rates using over 20 features derived from four dimensions: hazard intensity, community vulnerability, evacuation readiness, and built environment. Using an XGBoost model trained on multi-source, large-scale datasets for two hurricanes (Ian 2022, Milton 2024) and two wildfires (Kincade 2019, Palisades--Eaton 2025), Evac-Cast achieves mean absolute errors of 4.5% and 3.5% for hurricane and wildfire events, respectively. SHAP analysis reveals a consistent feature importance hierarchy across hazards, led by hazard intensity. Notably, the models perform well without explicit psychosocial variables, suggesting that macro-level proxies effectively encode behavioral signals traditionally captured through time-consuming surveys. This work offers a survey-free, high-resolution approach for predicting and understanding evacuation in hazard events, which could serve as a data-driven tool to support decision-making in emergency management. |
| 2025-08-01 | [LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks](http://arxiv.org/abs/2508.00602v1) | Francesco Panebianco, Stefano Bonfanti et al. | The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard. |
| 2025-08-01 | [A Context-Aware Dual-Metric Framework for Confidence Estimation in Large Language Models](http://arxiv.org/abs/2508.00600v1) | Mingruo Yuan, Shuyi Zhang et al. | Accurate confidence estimation is essential for trustworthy large language models (LLMs) systems, as it empowers the user to determine when to trust outputs and enables reliable deployment in safety-critical applications. Current confidence estimation methods for LLMs neglect the relevance between responses and contextual information, a crucial factor in output quality evaluation, particularly in scenarios where background knowledge is provided. To bridge this gap, we propose CRUX (Context-aware entropy Reduction and Unified consistency eXamination), the first framework that integrates context faithfulness and consistency for confidence estimation via two novel metrics. First, contextual entropy reduction represents data uncertainty with the information gain through contrastive sampling with and without context. Second, unified consistency examination captures potential model uncertainty through the global consistency of the generated answers with and without context. Experiments across three benchmark datasets (CoQA, SQuAD, QuAC) and two domain-specific datasets (BioASQ, EduQG) demonstrate CRUX's effectiveness, achieving the highest AUROC than existing baselines. |
| 2025-08-01 | [Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving](http://arxiv.org/abs/2508.00589v1) | Stefan Englmeier, Max A. Büttner et al. | Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset. |
| 2025-08-01 | [OmniUnet: A Multimodal Network for Unstructured Terrain Segmentation on Planetary Rovers Using RGB, Depth, and Thermal Imagery](http://arxiv.org/abs/2508.00580v1) | Raul Castilla-Arquillo, Carlos Perez-del-Pulgar et al. | Robot navigation in unstructured environments requires multimodal perception systems that can support safe navigation. Multimodality enables the integration of complementary information collected by different sensors. However, this information must be processed by machine learning algorithms specifically designed to leverage heterogeneous data. Furthermore, it is necessary to identify which sensor modalities are most informative for navigation in the target environment. In Martian exploration, thermal imagery has proven valuable for assessing terrain safety due to differences in thermal behaviour between soil types. This work presents OmniUnet, a transformer-based neural network architecture for semantic segmentation using RGB, depth, and thermal (RGB-D-T) imagery. A custom multimodal sensor housing was developed using 3D printing and mounted on the Martian Rover Testbed for Autonomy (MaRTA) to collect a multimodal dataset in the Bardenas semi-desert in northern Spain. This location serves as a representative environment of the Martian surface, featuring terrain types such as sand, bedrock, and compact soil. A subset of this dataset was manually labeled to support supervised training of the network. The model was evaluated both quantitatively and qualitatively, achieving a pixel accuracy of 80.37% and demonstrating strong performance in segmenting complex unstructured terrain. Inference tests yielded an average prediction time of 673 ms on a resource-constrained computer (Jetson Orin Nano), confirming its suitability for on-robot deployment. The software implementation of the network and the labeled dataset have been made publicly available to support future research in multimodal terrain perception for planetary robotics. |
| 2025-08-01 | [Activation-Guided Local Editing for Jailbreaking Attacks](http://arxiv.org/abs/2508.00555v1) | Jiecong Wang, Haoran Li et al. | Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at https://github.com/yunsaijc/AGILE. |
| 2025-08-01 | [Utilizing Deep Learning for Enhanced Tritium Detection in CCDs](http://arxiv.org/abs/2508.00532v1) | E. Rofors, R. Heller et al. | This study explores the use of charge-coupled devices (CCDs) for detecting low-energy beta particles from tritium decay - a critical signal for nuclear safety, nuclear nonproliferation, and environmental monitoring. We employ a dual approach utilizing both measured CCD data and detailed Geant4 simulations. Our analysis compares classical techniques with advanced deep learning methods, including convolutional neural networks (CNNs), autoencoders trained exclusively on tritium data, and preliminary studies on boosted decision trees (BDTs). The CNN, trained on mixed signal/background datasets, demonstrates superior classification performance, while the autoencoder shows the potential of unsupervised, background-agnostic strategies. These results highlight the excellent sensitivity achievable thanks to the background rejection made possible by information-rich CCD data, paving the way for improved portable tritium monitoring. |
| 2025-08-01 | [Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking](http://arxiv.org/abs/2508.00500v1) | Haoyu Wang, Chris M. Poskitt et al. | Large Language Model (LLM) agents exhibit powerful autonomous capabilities across domains such as robotics, virtual assistants, and web automation. However, their stochastic behavior introduces significant safety risks that are difficult to anticipate. Existing rule-based enforcement systems, such as AgentSpec, focus on developing reactive safety rules, which typically respond only when unsafe behavior is imminent or has already occurred. These systems lack foresight and struggle with long-horizon dependencies and distribution shifts. To address these limitations, we propose Pro2Guard, a proactive runtime enforcement framework grounded in probabilistic reachability analysis. Pro2Guard abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it anticipates future risks by estimating the probability of reaching unsafe states, triggering interventions before violations occur when the predicted risk exceeds a user-defined threshold. By incorporating semantic validity checks and leveraging PAC bounds, Pro2Guard ensures statistical reliability while approximating the underlying ground-truth model. We evaluate Pro2Guard extensively across two safety-critical domains: embodied household agents and autonomous vehicles. In embodied agent tasks, Pro2Guard enforces safety early on up to 93.6% of unsafe tasks using low thresholds, while configurable modes (e.g., reflect) allow balancing safety with task success, maintaining up to 80.4% task completion. In autonomous driving scenarios, Pro2Guard achieves 100% prediction of traffic law violations and collisions, anticipating risks up to 38.66 seconds ahead. |
| 2025-08-01 | [HyPCV-Former: Hyperbolic Spatio-Temporal Transformer for 3D Point Cloud Video Anomaly Detection](http://arxiv.org/abs/2508.00473v1) | Jiaping Cao, Kangkang Zhou et al. | Video anomaly detection is a fundamental task in video surveillance, with broad applications in public safety and intelligent monitoring systems. Although previous methods leverage Euclidean representations in RGB or depth domains, such embeddings are inherently limited in capturing hierarchical event structures and spatio-temporal continuity. To address these limitations, we propose HyPCV-Former, a novel hyperbolic spatio-temporal transformer for anomaly detection in 3D point cloud videos. Our approach first extracts per-frame spatial features from point cloud sequences via point cloud extractor, and then embeds them into Lorentzian hyperbolic space, which better captures the latent hierarchical structure of events. To model temporal dynamics, we introduce a hyperbolic multi-head self-attention (HMHA) mechanism that leverages Lorentzian inner products and curvature-aware softmax to learn temporal dependencies under non-Euclidean geometry. Our method performs all feature transformations and anomaly scoring directly within full Lorentzian space rather than via tangent space approximation. Extensive experiments demonstrate that HyPCV-Former achieves state-of-the-art performance across multiple anomaly categories, with a 7\% improvement on the TIMo dataset and a 5.6\% gain on the DAD dataset compared to benchmarks. The code will be released upon paper acceptance. |
| 2025-08-01 | [Loop Invariant Generation: A Hybrid Framework of Reasoning optimised LLMs and SMT Solvers](http://arxiv.org/abs/2508.00419v1) | Varun Bharti, Shashwat Jha et al. | Loop invariants are essential for proving the correctness of programs with loops. Developing loop invariants is challenging, and fully automatic synthesis cannot be guaranteed for arbitrary programs. Some approaches have been proposed to synthesize loop invariants using symbolic techniques and more recently using neural approaches. These approaches are able to correctly synthesize loop invariants only for subsets of standard benchmarks. In this work, we investigate whether modern, reasoning-optimized large language models can do better. We integrate OpenAI's O1, O1-mini, and O3-mini into a tightly coupled generate-and-check pipeline with the Z3 SMT solver, using solver counterexamples to iteratively guide invariant refinement. We use Code2Inv benchmark, which provides C programs along with their formal preconditions and postconditions. On this benchmark of 133 tasks, our framework achieves 100% coverage (133 out of 133), outperforming the previous best of 107 out of 133, while requiring only 1-2 model proposals per instance and 14-55 seconds of wall-clock time. These results demonstrate that LLMs possess latent logical reasoning capabilities which can help automate loop invariant synthesis. While our experiments target C-specific programs, this approach should be generalizable to other imperative languages. |
| 2025-08-01 | [iSafetyBench: A video-language benchmark for safety in industrial environment](http://arxiv.org/abs/2508.00399v1) | Raiyaan Abdullah, Yogesh Singh Rawat et al. | Recent advances in vision-language models (VLMs) have enabled impressive generalization across diverse video understanding tasks under zero-shot settings. However, their capabilities in high-stakes industrial domains-where recognizing both routine operations and safety-critical anomalies is essential-remain largely underexplored. To address this gap, we introduce iSafetyBench, a new video-language benchmark specifically designed to evaluate model performance in industrial environments across both normal and hazardous scenarios. iSafetyBench comprises 1,100 video clips sourced from real-world industrial settings, annotated with open-vocabulary, multi-label action tags spanning 98 routine and 67 hazardous action categories. Each clip is paired with multiple-choice questions for both single-label and multi-label evaluation, enabling fine-grained assessment of VLMs in both standard and safety-critical contexts. We evaluate eight state-of-the-art video-language models under zero-shot conditions. Despite their strong performance on existing video benchmarks, these models struggle with iSafetyBench-particularly in recognizing hazardous activities and in multi-label scenarios. Our results reveal significant performance gaps, underscoring the need for more robust, safety-aware multimodal models for industrial applications. iSafetyBench provides a first-of-its-kind testbed to drive progress in this direction. The dataset is available at: https://github.com/raiyaan-abdullah/iSafety-Bench. |
| 2025-08-01 | [Advancing Welding Defect Detection in Maritime Operations via Adapt-WeldNet and Defect Detection Interpretability Analysis](http://arxiv.org/abs/2508.00381v1) | Kamal Basha S, Athira Nambiar | Weld defect detection is crucial for ensuring the safety and reliability of piping systems in the oil and gas industry, especially in challenging marine and offshore environments. Traditional non-destructive testing (NDT) methods often fail to detect subtle or internal defects, leading to potential failures and costly downtime. Furthermore, existing neural network-based approaches for defect classification frequently rely on arbitrarily selected pretrained architectures and lack interpretability, raising safety concerns for deployment. To address these challenges, this paper introduces ``Adapt-WeldNet", an adaptive framework for welding defect detection that systematically evaluates various pre-trained architectures, transfer learning strategies, and adaptive optimizers to identify the best-performing model and hyperparameters, optimizing defect detection and providing actionable insights. Additionally, a novel Defect Detection Interpretability Analysis (DDIA) framework is proposed to enhance system transparency. DDIA employs Explainable AI (XAI) techniques, such as Grad-CAM and LIME, alongside domain-specific evaluations validated by certified ASNT NDE Level II professionals. Incorporating a Human-in-the-Loop (HITL) approach and aligning with the principles of Trustworthy AI, DDIA ensures the reliability, fairness, and accountability of the defect detection system, fostering confidence in automated decisions through expert validation. By improving both performance and interpretability, this work enhances trust, safety, and reliability in welding defect detection systems, supporting critical operations in offshore and marine environments. |
| 2025-08-01 | [R1-ACT: Efficient Reasoning Model Safety Alignment by Activating Safety Knowledge](http://arxiv.org/abs/2508.00324v1) | Yeonjun In, Wonjoong Kim et al. | Although large reasoning models (LRMs) have demonstrated impressive capabilities on complex tasks, recent studies reveal that these models frequently fulfill harmful user instructions, raising significant safety concerns. In this paper, we investigate the underlying cause of LRM safety risks and find that models already possess sufficient safety knowledge but fail to activate it during reasoning. Based on this insight, we propose R1-Act, a simple and efficient post-training method that explicitly triggers safety knowledge through a structured reasoning process. R1-Act achieves strong safety improvements while preserving reasoning performance, outperforming prior alignment methods. Notably, it requires only 1,000 training examples and 90 minutes of training on a single RTX A6000 GPU. Extensive experiments across multiple LRM backbones and sizes demonstrate the robustness, scalability, and practical efficiency of our approach. |
| 2025-08-01 | [GV-VAD : Exploring Video Generation for Weakly-Supervised Video Anomaly Detection](http://arxiv.org/abs/2508.00312v1) | Suhang Cai, Xiaohao Peng et al. | Video anomaly detection (VAD) plays a critical role in public safety applications such as intelligent surveillance. However, the rarity, unpredictability, and high annotation cost of real-world anomalies make it difficult to scale VAD datasets, which limits the performance and generalization ability of existing models. To address this challenge, we propose a generative video-enhanced weakly-supervised video anomaly detection (GV-VAD) framework that leverages text-conditioned video generation models to produce semantically controllable and physically plausible synthetic videos. These virtual videos are used to augment training data at low cost. In addition, a synthetic sample loss scaling strategy is utilized to control the influence of generated synthetic samples for efficient training. The experiments show that the proposed framework outperforms state-of-the-art methods on UCF-Crime datasets. The code is available at https://github.com/Sumutan/GV-VAD.git. |
| 2025-08-01 | [Privacy-Preserving Driver Drowsiness Detection with Spatial Self-Attention and Federated Learning](http://arxiv.org/abs/2508.00287v1) | Tran Viet Khoa, Do Hai Son et al. | Driver drowsiness is one of the main causes of road accidents and is recognized as a leading contributor to traffic-related fatalities. However, detecting drowsiness accurately remains a challenging task, especially in real-world settings where facial data from different individuals is decentralized and highly diverse. In this paper, we propose a novel framework for drowsiness detection that is designed to work effectively with heterogeneous and decentralized data. Our approach develops a new Spatial Self-Attention (SSA) mechanism integrated with a Long Short-Term Memory (LSTM) network to better extract key facial features and improve detection performance. To support federated learning, we employ a Gradient Similarity Comparison (GSC) that selects the most relevant trained models from different operators before aggregation. This improves the accuracy and robustness of the global model while preserving user privacy. We also develop a customized tool that automatically processes video data by extracting frames, detecting and cropping faces, and applying data augmentation techniques such as rotation, flipping, brightness adjustment, and zooming. Experimental results show that our framework achieves a detection accuracy of 89.9% in the federated learning settings, outperforming existing methods under various deployment scenarios. The results demonstrate the effectiveness of our approach in handling real-world data variability and highlight its potential for deployment in intelligent transportation systems to enhance road safety through early and reliable drowsiness detection. |
| 2025-07-31 | [Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs](http://arxiv.org/abs/2508.00161v1) | Ziqian Zhong, Aditi Raghunathan | The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.   In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.   For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation.   Our implementation can be found at https://github.com/fjzzq2002/WeightWatch. |
| 2025-07-31 | [Model-Based Soft Maximization of Suitable Metrics of Long-Term Human Power](http://arxiv.org/abs/2508.00159v1) | Jobst Heitzig, Ram Potham | Power is a key concept in AI safety: power-seeking as an instrumental goal, sudden or gradual disempowerment of humans, power balance in human-AI interaction and international AI governance. At the same time, power as the ability to pursue diverse goals is essential for wellbeing.   This paper explores the idea of promoting both safety and wellbeing by forcing AI agents explicitly to empower humans and to manage the power balance between humans and AI agents in a desirable way. Using a principled, partially axiomatic approach, we design a parametrizable and decomposable objective function that represents an inequality- and risk-averse long-term aggregate of human power. It takes into account humans' bounded rationality and social norms, and, crucially, considers a wide variety of possible human goals.   We derive algorithms for computing that metric by backward induction or approximating it via a form of multi-agent reinforcement learning from a given world model. We exemplify the consequences of (softly) maximizing this metric in a variety of paradigmatic situations and describe what instrumental sub-goals it will likely imply. Our cautious assessment is that softly maximizing suitable aggregate metrics of human power might constitute a beneficial objective for agentic AI systems that is safer than direct utility-based objectives. |
| 2025-07-31 | [Integrating Opinion Dynamics into Safety Control for Decentralized Airplane Encounter Resolution](http://arxiv.org/abs/2508.00156v1) | Shuhao Qi, Zhiqi Tang et al. | As the airspace becomes increasingly congested, decentralized conflict resolution methods for airplane encounters have become essential. While decentralized safety controllers can prevent dangerous midair collisions, they do not always ensure prompt conflict resolution. As a result, airplane progress may be blocked for extended periods in certain situations. To address this blocking phenomenon, this paper proposes integrating bio-inspired nonlinear opinion dynamics into the airplane safety control framework, thereby guaranteeing both safety and blocking-free resolution. In particular, opinion dynamics enable the safety controller to achieve collaborative decision-making for blocking resolution and facilitate rapid, safe coordination without relying on communication or preset rules. Extensive simulation results validate the improved flight efficiency and safety guarantees. This study provides practical insights into the design of autonomous controllers for airplanes. |

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



