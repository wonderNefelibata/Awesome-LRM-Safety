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
| 2025-10-15 | [GAPS: A Clinically Grounded, Automated Benchmark for Evaluating AI Clinicians](http://arxiv.org/abs/2510.13734v1) | Xiuyuan Chen, Tao Sun et al. | Current benchmarks for AI clinician systems, often based on multiple-choice exams or manual rubrics, fail to capture the depth, robustness, and safety required for real-world clinical practice. To address this, we introduce the GAPS framework, a multidimensional paradigm for evaluating \textbf{G}rounding (cognitive depth), \textbf{A}dequacy (answer completeness), \textbf{P}erturbation (robustness), and \textbf{S}afety. Critically, we developed a fully automated, guideline-anchored pipeline to construct a GAPS-aligned benchmark end-to-end, overcoming the scalability and subjectivity limitations of prior work. Our pipeline assembles an evidence neighborhood, creates dual graph and tree representations, and automatically generates questions across G-levels. Rubrics are synthesized by a DeepResearch agent that mimics GRADE-consistent, PICO-driven evidence review in a ReAct loop. Scoring is performed by an ensemble of large language model (LLM) judges. Validation confirmed our automated questions are high-quality and align with clinician judgment. Evaluating state-of-the-art models on the benchmark revealed key failure modes: performance degrades sharply with increased reasoning depth (G-axis), models struggle with answer completeness (A-axis), and they are highly vulnerable to adversarial perturbations (P-axis) as well as certain safety issues (S-axis). This automated, clinically-grounded approach provides a reproducible and scalable method for rigorously evaluating AI clinician systems and guiding their development toward safer, more reliable clinical practice. |
| 2025-10-15 | [From Refusal to Recovery: A Control-Theoretic Approach to Generative AI Guardrails](http://arxiv.org/abs/2510.13727v1) | Ravi Pandya, Madison Bland et al. | Generative AI systems are increasingly assisting and acting on behalf of end users in practical settings, from digital shopping assistants to next-generation autonomous cars. In this context, safety is no longer about blocking harmful content, but about preempting downstream hazards like financial or physical harm. Yet, most AI guardrails continue to rely on output classification based on labeled datasets and human-specified criteria,making them brittle to new hazardous situations. Even when unsafe conditions are flagged, this detection offers no path to recovery: typically, the AI system simply refuses to act--which is not always a safe choice. In this work, we argue that agentic AI safety is fundamentally a sequential decision problem: harmful outcomes arise from the AI system's continually evolving interactions and their downstream consequences on the world. We formalize this through the lens of safety-critical control theory, but within the AI model's latent representation of the world. This enables us to build predictive guardrails that (i) monitor an AI system's outputs (actions) in real time and (ii) proactively correct risky outputs to safe ones, all in a model-agnostic manner so the same guardrail can be wrapped around any AI model. We also offer a practical training recipe for computing such guardrails at scale via safety-critical reinforcement learning. Our experiments in simulated driving and e-commerce settings demonstrate that control-theoretic guardrails can reliably steer LLM agents clear of catastrophic outcomes (from collisions to bankruptcy) while preserving task performance, offering a principled dynamic alternative to today's flag-and-block guardrails. |
| 2025-10-15 | [Risk-adaptive Activation Steering for Safe Multimodal Large Language Models](http://arxiv.org/abs/2510.13698v1) | Jonghyun Park, Minhyuk Seo et al. | One of the key challenges of modern AI models is ensuring that they provide helpful responses to benign queries while refusing malicious ones. But often, the models are vulnerable to multimodal queries with harmful intent embedded in images. One approach for safety alignment is training with extensive safety datasets at the significant costs in both dataset curation and training. Inference-time alignment mitigates these costs, but introduces two drawbacks: excessive refusals from misclassified benign queries and slower inference speed due to iterative output adjustments. To overcome these limitations, we propose to reformulate queries to strengthen cross-modal attention to safety-critical image regions, enabling accurate risk assessment at the query level. Using the assessed risk, it adaptively steers activations to generate responses that are safe and helpful without overhead from iterative output adjustments. We call this Risk-adaptive Activation Steering (RAS). Extensive experiments across multiple benchmarks on multimodal safety and utility demonstrate that the RAS significantly reduces attack success rates, preserves general task performance, and improves inference speed over prior inference-time defenses. |
| 2025-10-15 | [International AI Safety Report 2025: First Key Update: Capabilities and Risk Implications](http://arxiv.org/abs/2510.13653v1) | Yoshua Bengio, Stephen Clare et al. | Since the publication of the first International AI Safety Report, AI capabilities have continued to improve across key domains. New training techniques that teach AI systems to reason step-by-step and inference-time enhancements have primarily driven these advances, rather than simply training larger models. As a result, general-purpose AI systems can solve more complex problems in a range of domains, from scientific research to software development. Their performance on benchmarks that measure performance in coding, mathematics, and answering expert-level science questions has continued to improve, though reliability challenges persist, with systems excelling on some tasks while failing completely on others. These capability improvements also have implications for multiple risks, including risks from biological weapons and cyber attacks. Finally, they pose new challenges for monitoring and controllability. This update examines how AI capabilities have improved since the first Report, then focuses on key risk areas where substantial new evidence warrants updated assessments. |
| 2025-10-15 | [Towards Adversarial Robustness and Uncertainty Quantification in DINOv2-based Few-Shot Anomaly Detection](http://arxiv.org/abs/2510.13643v1) | Akib Mohammed Khan, Bartosz Krawczyk | Foundation models such as DINOv2 have shown strong performance in few-shot anomaly detection, yet two key questions remain unexamined: (i) how susceptible are these detectors to adversarial perturbations; and (ii) how well do their anomaly scores reflect calibrated uncertainty? Building on AnomalyDINO, a training-free deep nearest-neighbor detector over DINOv2 features, we present one of the first systematic studies of adversarial attacks and uncertainty estimation in this setting. To enable white-box gradient attacks while preserving test-time behavior, we attach a lightweight linear head to frozen DINOv2 features only for crafting perturbations. Using this heuristic, we evaluate the impact of FGSM across the MVTec-AD and VisA datasets and observe consistent drops in F1, AUROC, AP, and G-mean, indicating that imperceptible perturbations can flip nearest-neighbor relations in feature space to induce confident misclassification. Complementing robustness, we probe reliability and find that raw anomaly scores are poorly calibrated, revealing a gap between confidence and correctness that limits safety-critical use. As a simple, strong baseline toward trustworthiness, we apply post-hoc Platt scaling to the anomaly scores for uncertainty estimation. The resulting calibrated posteriors yield significantly higher predictive entropy on adversarially perturbed inputs than on clean ones, enabling a practical flagging mechanism for attack detection while reducing calibration error (ECE). Our findings surface concrete vulnerabilities in DINOv2-based few-shot anomaly detectors and establish an evaluation protocol and baseline for robust, uncertainty-aware anomaly detection. We argue that adversarial robustness and principled uncertainty quantification are not optional add-ons but essential capabilities if anomaly detection systems are to be trustworthy and ready for real-world deployment. |
| 2025-10-15 | [AVAR-Net: A Lightweight Audio-Visual Anomaly Recognition Framework with a Benchmark Dataset](http://arxiv.org/abs/2510.13630v1) | Amjid Ali, Zulfiqar Ahmad Khan et al. | Anomaly recognition plays a vital role in surveillance, transportation, healthcare, and public safety. However, most existing approaches rely solely on visual data, making them unreliable under challenging conditions such as occlusion, low illumination, and adverse weather. Moreover, the absence of large-scale synchronized audio-visual datasets has hindered progress in multimodal anomaly recognition. To address these limitations, this study presents AVAR-Net, a lightweight and efficient audio-visual anomaly recognition framework designed for real-world environments. AVAR-Net consists of four main modules: an audio feature extractor, a video feature extractor, fusion strategy, and a sequential pattern learning network that models cross-modal relationships for anomaly recognition. Specifically, the Wav2Vec2 model extracts robust temporal features from raw audio, while MobileViT captures both local and global visual representations from video frames. An early fusion mechanism combines these modalities, and a Multi-Stage Temporal Convolutional Network (MTCN) model that learns long-range temporal dependencies within the fused representation, enabling robust spatiotemporal reasoning. A novel Visual-Audio Anomaly Recognition (VAAR) dataset, is also introduced, serving as a medium-scale benchmark containing 3,000 real-world videos with synchronized audio across ten diverse anomaly classes. Experimental evaluations demonstrate that AVAR-Net achieves 89.29% accuracy on VAAR and 88.56% Average Precision on the XD-Violence dataset, improving Average Precision by 2.8% over existing state-of-the-art methods. These results highlight the effectiveness, efficiency, and generalization capability of the proposed framework, as well as the utility of VAAR as a benchmark for advancing multimodal anomaly recognition research. |
| 2025-10-15 | [Towards Robust Knowledge Removal in Federated Learning with High Data Heterogeneity](http://arxiv.org/abs/2510.13606v1) | Riccardo Santi, Riccardo Salami et al. | Nowdays, there are an abundance of portable devices capable of collecting large amounts of data and with decent computational power. This opened the possibility to train AI models in a distributed manner, preserving the participating clients' privacy. However, because of privacy regulations and safety requirements, elimination upon necessity of a client contribution to the model has become mandatory. The cleansing process must satisfy specific efficacy and time requirements. In recent years, research efforts have produced several knowledge removal methods, but these require multiple communication rounds between the data holders and the process coordinator. This can cause the unavailability of an effective model up to the end of the removal process, which can result in a disservice to the system users. In this paper, we introduce an innovative solution based on Task Arithmetic and the Neural Tangent Kernel, to rapidly remove a client's influence from a model. |
| 2025-10-15 | [Who Speaks for the Trigger? Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers](http://arxiv.org/abs/2510.13462v1) | Xin Zhao, Xiaojun Chen et al. | Large language models (LLMs) with Mixture-of-Experts (MoE) architectures achieve impressive performance and efficiency by dynamically routing inputs to specialized subnetworks, known as experts. However, this sparse routing mechanism inherently exhibits task preferences due to expert specialization, introducing a new and underexplored vulnerability to backdoor attacks. In this work, we investigate the feasibility and effectiveness of injecting backdoors into MoE-based LLMs by exploiting their inherent expert routing preferences. We thus propose BadSwitch, a novel backdoor framework that integrates task-coupled dynamic trigger optimization with a sensitivity-guided Top-S expert tracing mechanism. Our approach jointly optimizes trigger embeddings during pretraining while identifying S most sensitive experts, subsequently constraining the Top-K gating mechanism to these targeted experts. Unlike traditional backdoor attacks that rely on superficial data poisoning or model editing, BadSwitch primarily embeds malicious triggers into expert routing paths with strong task affinity, enabling precise and stealthy model manipulation. Through comprehensive evaluations across three prominent MoE architectures (Switch Transformer, QwenMoE, and DeepSeekMoE), we demonstrate that BadSwitch can efficiently hijack pre-trained models with up to 100% success rate (ASR) while maintaining the highest clean accuracy (ACC) among all baselines. Furthermore, BadSwitch exhibits strong resilience against both text-level and model-level defense mechanisms, achieving 94.07% ASR and 87.18% ACC on the AGNews dataset. Our analysis of expert activation patterns reveals fundamental insights into MoE vulnerabilities. We anticipate this work will expose security risks in MoE systems and contribute to advancing AI safety. |
| 2025-10-15 | [Physics-Informed Neural Network Modeling of Vehicle Collision Dynamics in Precision Immobilization Technique Maneuvers](http://arxiv.org/abs/2510.13461v1) | Yangye Jiang, Jiachen Wang et al. | Accurate prediction of vehicle collision dynamics is crucial for advanced safety systems and post-impact control applications, yet existing methods face inherent trade-offs among computational efficiency, prediction accuracy, and data requirements. This paper proposes a dual Physics-Informed Neural Network framework addressing these challenges through two complementary networks. The first network integrates Gaussian Mixture Models with PINN architecture to learn impact force distributions from finite element analysis data while enforcing momentum conservation and energy consistency constraints. The second network employs an adaptive PINN with dynamic constraint weighting to predict post-collision vehicle dynamics, featuring an adaptive physics guard layer that prevents unrealistic predictions whil e preserving data-driven learning capabilities. The framework incorporates uncertainty quantification through time-varying parameters and enables rapid adaptation via fine-tuning strategies. Validation demonstrates significant improvements: the impact force model achieves relative errors below 15.0% for force prediction on finite element analysis (FEA) datasets, while the vehicle dynamics model reduces average trajectory prediction error by 63.6% compared to traditional four-degree-of-freedom models in scaled vehicle experiments. The integrated system maintains millisecond-level computational efficiency suitable for real-time applications while providing probabilistic confidence bounds essential for safety-critical control. Comprehensive validation through FEA simulation, dynamic modeling, and scaled vehicle experiments confirms the framework's effectiveness for Precision Immobilization Technique scenarios and general collision dynamics prediction. |
| 2025-10-15 | [Protect: Towards Robust Guardrailing Stack for Trustworthy Enterprise LLM Systems](http://arxiv.org/abs/2510.13351v1) | Karthik Avinash, Nikhil Pareek et al. | The increasing deployment of Large Language Models (LLMs) across enterprise and mission-critical domains has underscored the urgent need for robust guardrailing systems that ensure safety, reliability, and compliance. Existing solutions often struggle with real-time oversight, multi-modal data handling, and explainability -- limitations that hinder their adoption in regulated environments. Existing guardrails largely operate in isolation, focused on text alone making them inadequate for multi-modal, production-scale environments. We introduce Protect, natively multi-modal guardrailing model designed to operate seamlessly across text, image, and audio inputs, designed for enterprise-grade deployment. Protect integrates fine-tuned, category-specific adapters trained via Low-Rank Adaptation (LoRA) on an extensive, multi-modal dataset covering four safety dimensions: toxicity, sexism, data privacy, and prompt injection. Our teacher-assisted annotation pipeline leverages reasoning and explanation traces to generate high-fidelity, context-aware labels across modalities. Experimental results demonstrate state-of-the-art performance across all safety dimensions, surpassing existing open and proprietary models such as WildGuard, LlamaGuard-4, and GPT-4.1. Protect establishes a strong foundation for trustworthy, auditable, and production-ready safety systems capable of operating across text, image, and audio modalities. |
| 2025-10-15 | [Five Years of Clinical Application of independent Monte Carlo-Based Patient-Specific Quality Assurance at the Maastro Proton Therapy Center](http://arxiv.org/abs/2510.13300v1) | Ilaria Rinaldi, Giorgio Cartechini et al. | At the Maastro Proton Therapy Center in Maastricht, patient-specific quality assurance (PSQA) using an independent GPU-accelerated Monte Carlo (MC) calculation has fully replaced conventional measurements, which are time-consuming and have limited sensitivity to clinically relevant errors.   A fully automated and robust pipeline was developed, integrating two clinical workflows based on the fast MC code Fred. The system is fully operational, and automatic verification reports are part of daily clinical practice.   The first workflow performs a pre-treatment dose recalculation in Fred using the planning CT and clinical plan. The second uses Fred with machine log files to verify the actually delivered dose. Both generate automatic reports for clinical review.   Over five years, this workflow has become part of routine clinical operations, providing robust 3D dosimetric verification in heterogeneous anatomies. So far, Fred has recalculated more than 6000 pre-treatment plans and 3513 log file-based PSQA cases, saving an estimated 4090 hours of QA work. The pipeline identified true negatives and detected two planning-related failures that would have been missed by conventional measurements. No false positives or negatives were observed, confirming high accuracy and reliability.   The MC-based PSQA pipeline offers an efficient, sensitive, and clinically meaningful alternative to measurement-based QA in pencil beam scanning proton therapy. By eliminating routine measurements, it saves resources while improving patient safety and treatment quality. Five years of experience confirm that measurement-less MC-based PSQA is a viable and superior approach, providing full 3D verification and early error detection - a practical blueprint for other proton therapy centres. |
| 2025-10-15 | [Partitioned Scheduling for DAG Tasks Considering Probabilistic Execution Time](http://arxiv.org/abs/2510.13279v1) | Fuma Omori, Atsushi Yano et al. | Autonomous driving systems, critical for safety, require real-time guarantees and can be modeled as DAGs. Their acceleration features, such as caches and pipelining, often result in execution times below the worst-case. Thus, a probabilistic approach ensuring constraint satisfaction within a probability threshold is more suitable than worst-case guarantees for these systems. This paper considers probabilistic guarantees for DAG tasks by utilizing the results of probabilistic guarantees for single processors, which have been relatively more advanced than those for multi-core processors. This paper proposes a task set partitioning method that guarantees schedulability under the partitioned scheduling. The evaluation on randomly generated DAG task sets demonstrates that the proposed method schedules more task sets with a smaller mean analysis time compared to existing probabilistic schedulability analysis for DAGs. The evaluation also compares four bin-packing heuristics, revealing Item-Centric Worst-Fit-Decreasing schedules the most task sets. |
| 2025-10-15 | [Real-Time Crowd Counting for Embedded Systems with Lightweight Architecture](http://arxiv.org/abs/2510.13250v1) | Zhiyuan Zhao, Yubin Wen et al. | Crowd counting is a task of estimating the number of the crowd through images, which is extremely valuable in the fields of intelligent security, urban planning, public safety management, and so on. However, the existing counting methods have some problems in practical application on embedded systems for these fields, such as excessive model parameters, abundant complex calculations, etc. The practical application of embedded systems requires the model to be real-time, which means that the model is fast enough. Considering the aforementioned problems, we design a super real-time model with a stem-encoder-decoder structure for crowd counting tasks, which achieves the fastest inference compared with state-of-the-arts. Firstly, large convolution kernels in the stem network are used to enlarge the receptive field, which effectively extracts detailed head information. Then, in the encoder part, we use conditional channel weighting and multi-branch local fusion block to merge multi-scale features with low computational consumption. This part is crucial to the super real-time performance of the model. Finally, the feature pyramid networks are added to the top of the encoder to alleviate its incomplete fusion problems. Experiments on three benchmarks show that our network is suitable for super real-time crowd counting on embedded systems, ensuring competitive accuracy. At the same time, the proposed network reasoning speed is the fastest. Specifically, the proposed network achieves 381.7 FPS on NVIDIA GTX 1080Ti and 71.9 FPS on NVIDIA Jetson TX1. |
| 2025-10-15 | [SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs](http://arxiv.org/abs/2510.13190v1) | Juan Ren, Mark Dras et al. | Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs. |
| 2025-10-15 | [DSCD: Large Language Model Detoxification with Self-Constrained Decoding](http://arxiv.org/abs/2510.13183v1) | Ming Dong, Jinkui Zhang et al. | Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLM detoxification without parameter fine-tuning. DSCD strengthens the inner next-token distribution of the safety layer while weakening that of hallucination and toxic layers during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-and-play capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD's effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD's potential as a practical and scalable solution for safer LLM deployments. |
| 2025-10-15 | [Safe Driving in Occluded Environments](http://arxiv.org/abs/2510.13114v1) | Zhuoyuan Wang, Tongyao Jia et al. | Ensuring safe autonomous driving in the presence of occlusions poses a significant challenge in its policy design. While existing model-driven control techniques based on set invariance can handle visible risks, occlusions create latent risks in which safety-critical states are not observable. Data-driven techniques also struggle to handle latent risks because direct mappings from risk-critical objects in sensor inputs to safe actions cannot be learned without visible risk-critical objects. Motivated by these challenges, in this paper, we propose a probabilistic safety certificate for latent risk. Our key technical enabler is the application of probabilistic invariance: It relaxes the strict observability requirements imposed by set-invariance methods that demand the knowledge of risk-critical states. The proposed techniques provide linear action constraints that confine the latent risk probability within tolerance. Such constraints can be integrated into model predictive controllers or embedded in data-driven policies to mitigate latent risks. The proposed method is tested using the CARLA simulator and compared with a few existing techniques. The theoretical and empirical analysis jointly demonstrate that the proposed methods assure long-term safety in real-time control in occluded environments without being overly conservative and with transparency to exposed risks. |
| 2025-10-15 | [TRUSTVIS: A Multi-Dimensional Trustworthiness Evaluation Framework for Large Language Models](http://arxiv.org/abs/2510.13106v1) | Ruoyu Sun, Da Song et al. | As Large Language Models (LLMs) continue to revolutionize Natural Language Processing (NLP) applications, critical concerns about their trustworthiness persist, particularly in safety and robustness. To address these challenges, we introduce TRUSTVIS, an automated evaluation framework that provides a comprehensive assessment of LLM trustworthiness. A key feature of our framework is its interactive user interface, designed to offer intuitive visualizations of trustworthiness metrics. By integrating well-known perturbation methods like AutoDAN and employing majority voting across various evaluation methods, TRUSTVIS not only provides reliable results but also makes complex evaluation processes accessible to users. Preliminary case studies on models like Vicuna-7b, Llama2-7b, and GPT-3.5 demonstrate the effectiveness of our framework in identifying safety and robustness vulnerabilities, while the interactive interface allows users to explore results in detail, empowering targeted model improvements. Video Link: https://youtu.be/k1TrBqNVg8g |
| 2025-10-15 | [A Multi-dimensional Semantic Surprise Framework Based on Low-Entropy Semantic Manifolds for Fine-Grained Out-of-Distribution Detection](http://arxiv.org/abs/2510.13093v1) | Ningkang Peng, Yuzhe Mao et al. | Out-of-Distribution (OOD) detection is a cornerstone for the safe deployment of AI systems in the open world. However, existing methods treat OOD detection as a binary classification problem, a cognitive flattening that fails to distinguish between semantically close (Near-OOD) and distant (Far-OOD) unknown risks. This limitation poses a significant safety bottleneck in applications requiring fine-grained risk stratification. To address this, we propose a paradigm shift from a conventional probabilistic view to a principled information-theoretic framework. We formalize the core task as quantifying the Semantic Surprise of a new sample and introduce a novel ternary classification challenge: In-Distribution (ID) vs. Near-OOD vs. Far-OOD. The theoretical foundation of our work is the concept of Low-Entropy Semantic Manifolds, which are explicitly structured to reflect the data's intrinsic semantic hierarchy. To construct these manifolds, we design a Hierarchical Prototypical Network. We then introduce the Semantic Surprise Vector (SSV), a universal probe that decomposes a sample's total surprise into three complementary and interpretable dimensions: conformity, novelty, and ambiguity. To evaluate performance on this new task, we propose the Normalized Semantic Risk (nSR), a cost-sensitive metric. Experiments demonstrate that our framework not only establishes a new state-of-the-art (sota) on the challenging ternary task, but its robust representations also achieve top results on conventional binary benchmarks, reducing the False Positive Rate by over 60% on datasets like LSUN. |
| 2025-10-15 | [Imperative Quantum Programming with Ownership and Borrowing in Guppy](http://arxiv.org/abs/2510.13082v1) | Mark Koch, Agustín Borgna et al. | Linear types enforce no-cloning and no-deleting theorems in functional quantum programming. However, in imperative quantum programming, they have not gained widespread adoption. This work aims to develop a quantum type system that combines ergonomic linear typing with imperative semantics and maintains safety guarantees. All ideas presented here have been implemented in Quantinuum's Guppy programming language. |
| 2025-10-15 | [ADPerf: Investigating and Testing Performance in Autonomous Driving Systems](http://arxiv.org/abs/2510.13078v1) | Tri Minh-Triet Pham, Diego Elias Costa et al. | Obstacle detection is crucial to the operation of autonomous driving systems, which rely on multiple sensors, such as cameras and LiDARs, combined with code logic and deep learning models to detect obstacles for time-sensitive decisions. Consequently, obstacle detection latency is critical to the safety and effectiveness of autonomous driving systems. However, the latency of the obstacle detection module and its resilience to various changes in the LiDAR point cloud data are not yet fully understood. In this work, we present the first comprehensive investigation on measuring and modeling the performance of the obstacle detection modules in two industry-grade autonomous driving systems, i.e., Apollo and Autoware. Learning from this investigation, we introduce ADPerf, a tool that aims to generate realistic point cloud data test cases that can expose increased detection latency. Increasing latency decreases the availability of the detected obstacles and stresses the capabilities of subsequent modules in autonomous driving systems, i.e., the modules may be negatively impacted by the increased latency in obstacle detection.   We applied ADPerf to stress-test the performance of widely used 3D obstacle detection modules in autonomous driving systems, as well as the propagation of such tests on trajectory prediction modules. Our evaluation highlights the need to conduct performance testing of obstacle detection components, especially 3D obstacle detection, as they can be a major bottleneck to increased latency of the autonomous driving system. Such an adverse outcome will also further propagate to other modules, reducing the overall reliability of autonomous driving systems. |

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



