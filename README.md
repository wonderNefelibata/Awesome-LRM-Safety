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
| 2025-08-21 | [End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning](http://arxiv.org/abs/2508.15746v1) | Qiaoyu Zheng, Yuze Sun et al. | Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL.   Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See https://github.com/MAGIC-AI4Med/Deep-DxSearch. |
| 2025-08-21 | [Probability Density from Latent Diffusion Models for Out-of-Distribution Detection](http://arxiv.org/abs/2508.15737v1) | Joonas Järve, Karl Kaspar Haavel et al. | Despite rapid advances in AI, safety remains the main bottleneck to deploying machine-learning systems. A critical safety component is out-of-distribution detection: given an input, decide whether it comes from the same distribution as the training data. In generative models, the most natural OOD score is the data likelihood. Actually, under the assumption of uniformly distributed OOD data, the likelihood is even the optimal OOD detector, as we show in this work. However, earlier work reported that likelihood often fails in practice, raising doubts about its usefulness. We explore whether, in practice, the representation space also suffers from the inability to learn good density estimation for OOD detection, or if it is merely a problem of the pixel space typically used in generative models. To test this, we trained a Variational Diffusion Model not on images, but on the representation space of a pre-trained ResNet-18 to assess the performance of our likelihood-based detector in comparison to state-of-the-art methods from the OpenOOD suite. |
| 2025-08-21 | [SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models](http://arxiv.org/abs/2508.15648v1) | Peng Ding, Wen Sun et al. | Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO. |
| 2025-08-21 | [A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification](http://arxiv.org/abs/2508.15588v1) | Ahmed Nasir, Abdelhafid Zenati | The application of reinforcement learning to safety-critical systems is limited by the lack of formal methods for verifying the robustness and safety of learned policies. This paper introduces a novel framework that addresses this gap by analyzing the combination of an RL agent and its environment as a discrete-time autonomous dynamical system. By leveraging tools from dynamical systems theory, specifically the Finite-Time Lyapunov Exponent (FTLE), we identify and visualize Lagrangian Coherent Structures (LCS) that act as the hidden "skeleton" governing the system's behavior. We demonstrate that repelling LCS function as safety barriers around unsafe regions, while attracting LCS reveal the system's convergence properties and potential failure modes, such as unintended "trap" states. To move beyond qualitative visualization, we introduce a suite of quantitative metrics, Mean Boundary Repulsion (MBR), Aggregated Spurious Attractor Strength (ASAS), and Temporally-Aware Spurious Attractor Strength (TASAS), to formally measure a policy's safety margin and robustness. We further provide a method for deriving local stability guarantees and extend the analysis to handle model uncertainty. Through experiments in both discrete and continuous control environments, we show that this framework provides a comprehensive and interpretable assessment of policy behavior, successfully identifying critical flaws in policies that appear successful based on reward alone. |
| 2025-08-21 | [SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking](http://arxiv.org/abs/2508.15526v1) | Xiangyang Zhu, Yuan Tian et al. | The rapid proliferation of large language models (LLMs) has intensified the requirement for reliable safety evaluation to uncover model vulnerabilities. To this end, numerous LLM safety evaluation benchmarks are proposed. However, existing benchmarks generally rely on labor-intensive manual curation, which causes excessive time and resource consumption. They also exhibit significant redundancy and limited difficulty. To alleviate these problems, we introduce SafetyFlow, the first agent-flow system designed to automate the construction of LLM safety benchmarks. SafetyFlow can automatically build a comprehensive safety benchmark in only four days without any human intervention by orchestrating seven specialized agents, significantly reducing time and resource cost. Equipped with versatile tools, the agents of SafetyFlow ensure process and cost controllability while integrating human expertise into the automatic pipeline. The final constructed dataset, SafetyFlowBench, contains 23,446 queries with low redundancy and strong discriminative power. Our contribution includes the first fully automated benchmarking pipeline and a comprehensive safety benchmark. We evaluate the safety of 49 advanced LLMs on our dataset and conduct extensive experiments to validate our efficacy and efficiency. |
| 2025-08-21 | [Mini-Batch Robustness Verification of Deep Neural Networks](http://arxiv.org/abs/2508.15454v1) | Saar Tzour-Shaday, Dana Drachsler Cohen | Neural network image classifiers are ubiquitous in many safety-critical applications. However, they are susceptible to adversarial attacks. To understand their robustness to attacks, many local robustness verifiers have been proposed to analyze $\epsilon$-balls of inputs. Yet, existing verifiers introduce a long analysis time or lose too much precision, making them less effective for a large set of inputs. In this work, we propose a new approach to local robustness: group local robustness verification. The key idea is to leverage the similarity of the network computations of certain $\epsilon$-balls to reduce the overall analysis time. We propose BaVerLy, a sound and complete verifier that boosts the local robustness verification of a set of $\epsilon$-balls by dynamically constructing and verifying mini-batches. BaVerLy adaptively identifies successful mini-batch sizes, accordingly constructs mini-batches of $\epsilon$-balls that have similar network computations, and verifies them jointly. If a mini-batch is verified, all $\epsilon$-balls are proven robust. Otherwise, one $\epsilon$-ball is suspected as not being robust, guiding the refinement. In the latter case, BaVerLy leverages the analysis results to expedite the analysis of that $\epsilon$-ball as well as the other $\epsilon$-balls in the batch. We evaluate BaVerLy on fully connected and convolutional networks for MNIST and CIFAR-10. Results show that BaVerLy scales the common one by one verification by 2.3x on average and up to 4.1x, in which case it reduces the total analysis time from 24 hours to 6 hours. |
| 2025-08-21 | [Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection](http://arxiv.org/abs/2508.15449v1) | Chengcan Wu, Zeming Wei et al. | While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in https://github.com/ChengcanWu/MRP. |
| 2025-08-21 | [Lang2Lift: A Framework for Language-Guided Pallet Detection and Pose Estimation Integrated in Autonomous Outdoor Forklift Operation](http://arxiv.org/abs/2508.15427v1) | Huy Hoang Nguyen, Johannes Huemer et al. | The logistics and construction industries face persistent challenges in automating pallet handling, especially in outdoor environments with variable payloads, inconsistencies in pallet quality and dimensions, and unstructured surroundings. In this paper, we tackle automation of a critical step in pallet transport: the pallet pick-up operation. Our work is motivated by labor shortages, safety concerns, and inefficiencies in manually locating and retrieving pallets under such conditions. We present Lang2Lift, a framework that leverages foundation models for natural language-guided pallet detection and 6D pose estimation, enabling operators to specify targets through intuitive commands such as "pick up the steel beam pallet near the crane." The perception pipeline integrates Florence-2 and SAM-2 for language-grounded segmentation with FoundationPose for robust pose estimation in cluttered, multi-pallet outdoor scenes under variable lighting. The resulting poses feed into a motion planning module for fully autonomous forklift operation. We validate Lang2Lift on the ADAPT autonomous forklift platform, achieving 0.76 mIoU pallet segmentation accuracy on a real-world test dataset. Timing and error analysis demonstrate the system's robustness and confirm its feasibility for deployment in operational logistics and construction environments. Video demonstrations are available at https://eric-nguyen1402.github.io/lang2lift.github.io/ |
| 2025-08-21 | [Integrated Take-off Management and Trajectory Optimization for Merging Control in Urban Air Mobility Corridors](http://arxiv.org/abs/2508.15395v1) | Yingqi Liu, Tianlu Pan et al. | Urban Air Mobility (UAM) has the potential to revolutionize daily transportation, offering rapid and efficient aerial mobility services. Take-off and merging phases are critical for air corridor operations, requiring the coordination of take-off aircraft and corridor traffic while ensuring safety and seamless transition. This paper proposes an integrated take-off management and trajectory optimization for merging control in UAM corridors. We first introduce a novel take-off airspace design. To our knowledge, this paper is one of the first to propose a structured design for take-off airspace. Based on the take-off airspace design, we devise a hierarchical coordinated take-off and merging management (HCTMM) strategy. To be specific, the take-off airspace design can simplify aircraft dynamics and thus reduce the dimensionality of the trajectory optimization problem whilst mitigating obstacle avoidance complexities. The HCTMM strategy strictly ensures safety and improves the efficiency of take-off and merging operations. At the tactical level, a scheduling algorithm coordinates aircraft take-off times and selects dynamic merging points to reduce conflicts and ensure smooth take-off and merging processes. At the operational level, a trajectory optimization strategy ensures that each aircraft reaches the dynamic merging point efficiently while satisfying safety constraints. Simulation results show that, compared to representative strategies with fixed or dynamic merging points, the HCTMM strategy significantly improves operational efficiency and reduces computational burden, while ensuring safety under various corridor traffic conditions. Further results confirm the scalability of the HCTMM strategy and the computational efficiency enabled by the proposed take-off airspace design. |
| 2025-08-21 | [Unveiling Trust in Multimodal Large Language Models: Evaluation, Analysis, and Mitigation](http://arxiv.org/abs/2508.15370v1) | Yichi Zhang, Yao Huang et al. | The trustworthiness of Multimodal Large Language Models (MLLMs) remains an intense concern despite the significant progress in their capabilities. Existing evaluation and mitigation approaches often focus on narrow aspects and overlook risks introduced by the multimodality. To tackle these challenges, we propose MultiTrust-X, a comprehensive benchmark for evaluating, analyzing, and mitigating the trustworthiness issues of MLLMs. We define a three-dimensional framework, encompassing five trustworthiness aspects which include truthfulness, robustness, safety, fairness, and privacy; two novel risk types covering multimodal risks and cross-modal impacts; and various mitigation strategies from the perspectives of data, model architecture, training, and inference algorithms. Based on the taxonomy, MultiTrust-X includes 32 tasks and 28 curated datasets, enabling holistic evaluations over 30 open-source and proprietary MLLMs and in-depth analysis with 8 representative mitigation methods. Our extensive experiments reveal significant vulnerabilities in current models, including a gap between trustworthiness and general capabilities, as well as the amplification of potential risks in base LLMs by both multimodal training and inference. Moreover, our controlled analysis uncovers key limitations in existing mitigation strategies that, while some methods yield improvements in specific aspects, few effectively address overall trustworthiness, and many introduce unexpected trade-offs that compromise model utility. These findings also provide practical insights for future improvements, such as the benefits of reasoning to better balance safety and performance. Based on these insights, we introduce a Reasoning-Enhanced Safety Alignment (RESA) approach that equips the model with chain-of-thought reasoning ability to discover the underlying risks, achieving state-of-the-art results. |
| 2025-08-21 | [Binary black hole population inference combining confident and marginal events from the $\tt{IAS\text{-}HM}$ search pipeline](http://arxiv.org/abs/2508.15350v1) | Ajit Kumar Mehta, Digvijay Wadekar et al. | We present the population properties of binary black hole mergers identified by the $\tt{IAS\text{-}HM}$ pipeline (which incorporates higher-order modes in the search templates) during the third observing run (O3) of the LIGO, Virgo, and KAGRA (LVK) detectors. In our population inference analysis, instead of only using events above a sharp cut based on a particular detection threshold (e.g., false alarm rate), we use a Bayesian framework to consistently include both marginal and confident events. We find that our inference based solely on highly significant events ($p_{\mathrm{astro}} \sim 1$) is broadly consistent with the GWTC-3 population analysis performed by the LVK collaboration. However, incorporating marginal events into the analysis leads to a preference for stronger redshift evolution in the merger rate and an increased density of asymmetric mass-ratio mergers relative to the GWTC-3 analysis, while remaining within its allowed parameter ranges. Using simple parametric models to describe the binary black hole population, we estimate a merger rate density of $32.4^{+18.5}_{-12.2}\ \mathrm{Gpc}^{-3}\,\mathrm{yr}^{-1}$ at redshift $z = 0.2$, and a redshift evolution parameter of $\kappa = 4.4^{+1.9}_{-2.0}$. Assuming a power-law form for the mass ratio distribution ($\propto q^{\beta}$), we infer $\beta = 0.1^{+1.9}_{-1.4}$, indicating a relatively flat distribution. These results highlight the potential impact of marginal events on population inferences and motivate future analyses with data from upcoming observing runs. |
| 2025-08-21 | [VideoEraser: Concept Erasure in Text-to-Video Diffusion Models](http://arxiv.org/abs/2508.15314v1) | Naen Xu, Jinghuai Zhang et al. | The rapid growth of text-to-video (T2V) diffusion models has raised concerns about privacy, copyright, and safety due to their potential misuse in generating harmful or misleading content. These models are often trained on numerous datasets, including unauthorized personal identities, artistic creations, and harmful materials, which can lead to uncontrolled production and distribution of such content. To address this, we propose VideoEraser, a training-free framework that prevents T2V diffusion models from generating videos with undesirable concepts, even when explicitly prompted with those concepts. Designed as a plug-and-play module, VideoEraser can seamlessly integrate with representative T2V diffusion models via a two-stage process: Selective Prompt Embedding Adjustment (SPEA) and Adversarial-Resilient Noise Guidance (ARNG). We conduct extensive evaluations across four tasks, including object erasure, artistic style erasure, celebrity erasure, and explicit content erasure. Experimental results show that VideoEraser consistently outperforms prior methods regarding efficacy, integrity, fidelity, robustness, and generalizability. Notably, VideoEraser achieves state-of-the-art performance in suppressing undesirable content during T2V generation, reducing it by 46% on average across four tasks compared to baselines. |
| 2025-08-21 | [Retrieval-Augmented Review Generation for Poisoning Recommender Systems](http://arxiv.org/abs/2508.15252v1) | Shiyi Yang, Xinshu Li et al. | Recent studies have shown that recommender systems (RSs) are highly vulnerable to data poisoning attacks, where malicious actors inject fake user profiles, including a group of well-designed fake ratings, to manipulate recommendations. Due to security and privacy constraints in practice, attackers typically possess limited knowledge of the victim system and thus need to craft profiles that have transferability across black-box RSs. To maximize the attack impact, the profiles often remains imperceptible. However, generating such high-quality profiles with the restricted resources is challenging. Some works suggest incorporating fake textual reviews to strengthen the profiles; yet, the poor quality of the reviews largely undermines the attack effectiveness and imperceptibility under the practical setting.   To tackle the above challenges, in this paper, we propose to enhance the quality of the review text by harnessing in-context learning (ICL) capabilities of multimodal foundation models. To this end, we introduce a demonstration retrieval algorithm and a text style transfer strategy to augment the navie ICL. Specifically, we propose a novel practical attack framework named RAGAN to generate high-quality fake user profiles, which can gain insights into the robustness of RSs. The profiles are generated by a jailbreaker and collaboratively optimized on an instructional agent and a guardian to improve the attack transferability and imperceptibility. Comprehensive experiments on various real-world datasets demonstrate that RAGAN achieves the state-of-the-art poisoning attack performance. |
| 2025-08-21 | [EMNLP: Educator-role Moral and Normative Large Language Models Profiling](http://arxiv.org/abs/2508.15250v1) | Yilin Jiang, Mingzi Zhang et al. | Simulating Professions (SP) enables Large Language Models (LLMs) to emulate professional roles. However, comprehensive psychological and ethical evaluation in these contexts remains lacking. This paper introduces EMNLP, an Educator-role Moral and Normative LLMs Profiling framework for personality profiling, moral development stage measurement, and ethical risk under soft prompt injection. EMNLP extends existing scales and constructs 88 teacher-specific moral dilemmas, enabling profession-oriented comparison with human teachers. A targeted soft prompt injection set evaluates compliance and vulnerability in teacher SP. Experiments on 12 LLMs show teacher-role LLMs exhibit more idealized and polarized personalities than human teachers, excel in abstract moral reasoning, but struggle with emotionally complex situations. Models with stronger reasoning are more vulnerable to harmful prompt injection, revealing a paradox between capability and safety. The model temperature and other hyperparameters have limited influence except in some risk behaviors. This paper presents the first benchmark to assess ethical and psychological alignment of teacher-role LLMs for educational AI. Resources are available at https://e-m-n-l-p.github.io/. |
| 2025-08-21 | [STAGNet: A Spatio-Temporal Graph and LSTM Framework for Accident Anticipation](http://arxiv.org/abs/2508.15216v1) | Vipooshan Vipulananthan, Kumudu Mohottala et al. | Accident prediction and timely warnings play a key role in improving road safety by reducing the risk of injury to road users and minimizing property damage. Advanced Driver Assistance Systems (ADAS) are designed to support human drivers and are especially useful when they can anticipate potential accidents before they happen. While many existing systems depend on a range of sensors such as LiDAR, radar, and GPS, relying solely on dash-cam video input presents a more challenging but a more cost-effective and easily deployable solution. In this work, we incorporate better spatio-temporal features and aggregate them through a recurrent network to improve upon state-of-the-art graph neural networks for predicting accidents from dash-cam videos. Experiments using three publicly available datasets show that our proposed STAGNet model achieves higher average precision and mean time-to-collision values than previous methods, both when cross-validated on a given dataset and when trained and tested on different datasets. |
| 2025-08-21 | [Adversarial Agent Behavior Learning in Autonomous Driving Using Deep Reinforcement Learning](http://arxiv.org/abs/2508.15207v1) | Arjun Srinivasan, Anubhav Paras et al. | Existing approaches in reinforcement learning train an agent to learn desired optimal behavior in an environment with rule based surrounding agents. In safety critical applications such as autonomous driving it is crucial that the rule based agents are modelled properly. Several behavior modelling strategies and IDM models are used currently to model the surrounding agents. We present a learning based method to derive the adversarial behavior for the rule based agents to cause failure scenarios. We evaluate our adversarial agent against all the rule based agents and show the decrease in cumulative reward. |
| 2025-08-21 | [SafeLLM: Unlearning Harmful Outputs from Large Language Models against Jailbreak Attacks](http://arxiv.org/abs/2508.15182v1) | Xiangman Li, Xiaodong Wu et al. | Jailbreak attacks pose a serious threat to the safety of Large Language Models (LLMs) by crafting adversarial prompts that bypass alignment mechanisms, causing the models to produce harmful, restricted, or biased content. In this paper, we propose SafeLLM, a novel unlearning-based defense framework that unlearn the harmful knowledge from LLMs while preserving linguistic fluency and general capabilities. SafeLLM employs a three-stage pipeline: (1) dynamic unsafe output detection using a hybrid approach that integrates external classifiers with model-internal evaluations; (2) token-level harmful content tracing through feedforward network (FFN) activations to localize harmful knowledge; and (3) constrained optimization to suppress unsafe behavior without degrading overall model quality. SafeLLM achieves targeted and irreversible forgetting by identifying and neutralizing FFN substructures responsible for harmful generation pathways. Extensive experiments on prominent LLMs (Vicuna, LLaMA, and GPT-J) across multiple jailbreak benchmarks show that SafeLLM substantially reduces attack success rates while maintaining high general-purpose performance. Compared to standard defense methods such as supervised fine-tuning and direct preference optimization, SafeLLM offers stronger safety guarantees, more precise control over harmful behavior, and greater robustness to unseen attacks. Moreover, SafeLLM maintains the general performance after the harmful knowledge unlearned. These results highlight unlearning as a promising direction for scalable and effective LLM safety. |
| 2025-08-21 | [Reliable Multi-view 3D Reconstruction for `Just-in-time' Edge Environments](http://arxiv.org/abs/2508.15158v1) | Md. Nurul Absur, Abhinav Kumar et al. | Multi-view 3D reconstruction applications are revolutionizing critical use cases that require rapid situational-awareness, such as emergency response, tactical scenarios, and public safety. In many cases, their near-real-time latency requirements and ad-hoc needs for compute resources necessitate adoption of `Just-in-time' edge environments where the system is set up on the fly to support the applications during the mission lifetime. However, reliability issues can arise from the inherent dynamism and operational adversities of such edge environments, resulting in spatiotemporally correlated disruptions that impact the camera operations, which can lead to sustained degradation of reconstruction quality. In this paper, we propose a novel portfolio theory inspired edge resource management strategy for reliable multi-view 3D reconstruction against possible system disruptions. Our proposed methodology can guarantee reconstruction quality satisfaction even when the cameras are prone to spatiotemporally correlated disruptions. The portfolio theoretic optimization problem is solved using a genetic algorithm that converges quickly for realistic system settings. Using publicly available and customized 3D datasets, we demonstrate the proposed camera selection strategy's benefits in guaranteeing reliable 3D reconstruction against traditional baseline strategies, under spatiotemporal disruptions. |
| 2025-08-21 | [Software Model Checking via Summary-Guided Search (Extended Version)](http://arxiv.org/abs/2508.15137v1) | Ruijie Fang, Zachary Kincaid et al. | In this work, we describe a new software model-checking algorithm called GPS. GPS treats the task of model checking a program as a directed search of the program states, guided by a compositional, summary-based static analysis. The summaries produced by static analysis are used both to prune away infeasible paths and to drive test generation to reach new, unexplored program states. GPS can find both proofs of safety and counter-examples to safety (i.e., inputs that trigger bugs), and features a novel two-layered search strategy that renders it particularly efficient at finding bugs in programs featuring long, input-dependent error paths. To make GPS refutationally complete (in the sense that it will find an error if one exists, if it is allotted enough time), we introduce an instrumentation technique and show that it helps GPS achieve refutation-completeness without sacrificing overall performance. We benchmarked GPS on a suite of benchmarks including both programs from the Software Verification Competition (SV-COMP) and from prior literature, and found that our implementation of GPS outperforms state-of-the-art software model checkers (including the top performers in SV-COMP ReachSafety-Loops category), both in terms of the number of benchmarks solved and in terms of running time. |
| 2025-08-21 | [Wide-spectrum security of quantum key distribution](http://arxiv.org/abs/2508.15136v1) | Hao Tan, Mikhail Petrov et al. | Implementations of quantum key distribution (QKD) need vulnerability assessment against loopholes in their optical scheme. Most of the optical attacks involve injecting or receiving extraneous light via the communication channel. An eavesdropper can choose her attack wavelengths arbitrarily within the quantum channel passband to maximise the attack performance, exploiting spectral transparency windows of system components. Here we propose a wide-spectrum security evaluation methodology to achieve full optical spectrum safety for QKD systems. This technique requires transmittance characterisation in a wide spectral band with a high sensitivity. We report a testbench that characterises insertion loss of fiber-optic components in a wide spectral range of 400 to 2300 nm and up to 70 dB dynamic range. To illustrate practical application of the proposed methodology, we give a full Trojan-horse attack analysis for some typical QKD system configurations and discuss briefly induced-photorefraction and detector-backflash attacks. Our methodology can be used for certification of QKD systems. |

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



