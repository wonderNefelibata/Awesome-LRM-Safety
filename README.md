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
| 2025-07-02 | [How Well Does GPT-4o Understand Vision? Evaluating Multimodal Foundation Models on Standard Computer Vision Tasks](http://arxiv.org/abs/2507.01955v1) | Rahul Ramachandran, Ali Garjani et al. | Multimodal foundation models, such as GPT-4o, have recently made remarkable progress, but it is not clear where exactly these models stand in terms of understanding vision. In this paper, we benchmark the performance of popular multimodal foundation models (GPT-4o, o4-mini, Gemini 1.5 Pro and Gemini 2.0 Flash, Claude 3.5 Sonnet, Qwen2-VL, Llama 3.2) on standard computer vision tasks (semantic segmentation, object detection, image classification, depth and surface normal prediction) using established datasets (e.g., COCO, ImageNet and its variants, etc).   The main challenges to performing this are: 1) most models are trained to output text and cannot natively express versatile domains, such as segments or 3D geometry, and 2) many leading models are proprietary and accessible only at an API level, i.e., there is no weight access to adapt them. We address these challenges by translating standard vision tasks into equivalent text-promptable and API-compatible tasks via prompt chaining to create a standardized benchmarking framework.   We observe that 1) the models are not close to the state-of-the-art specialist models at any task. However, 2) they are respectable generalists; this is remarkable as they are presumably trained on primarily image-text-based tasks. 3) They perform semantic tasks notably better than geometric ones. 4) While the prompt-chaining techniques affect performance, better models exhibit less sensitivity to prompt variations. 5) GPT-4o performs the best among non-reasoning models, securing the top position in 4 out of 6 tasks, 6) reasoning models, e.g. o3, show improvements in geometric tasks, and 7) a preliminary analysis of models with native image generation, like the latest GPT-4o, shows they exhibit quirks like hallucinations and spatial misalignments. |
| 2025-07-02 | [Test-Time Scaling with Reflective Generative Model](http://arxiv.org/abs/2507.01951v1) | Zixiao Wang, Yuxin Wang et al. | We introduce our first reflective generative model MetaStone-S1, which obtains OpenAI o3's performance via the self-supervised process reward model (SPRM). Through sharing the backbone network and using task-specific heads for next token prediction and process scoring respectively, SPRM successfully integrates the policy model and process reward model(PRM) into a unified interface without extra process annotation, reducing over 99% PRM parameters for efficient reasoning. Equipped with SPRM, MetaStone-S1 is naturally suitable for test time scaling (TTS), and we provide three reasoning effort modes (low, medium, and high), based on the controllable thinking length. Moreover, we empirically establish a scaling law that reveals the relationship between total thinking computation and TTS performance. Experiments demonstrate that our MetaStone-S1 achieves comparable performance to OpenAI-o3-mini's series with only 32B parameter size. To support the research community, we have open-sourced MetaStone-S1 at https://github.com/MetaStone-AI/MetaStone-S1. |
| 2025-07-02 | [AI4Research: A Survey of Artificial Intelligence for Scientific Research](http://arxiv.org/abs/2507.01903v1) | Qiguang Chen, Mingda Yang et al. | Recent advancements in artificial intelligence (AI), particularly in large language models (LLMs) such as OpenAI-o1 and DeepSeek-R1, have demonstrated remarkable capabilities in complex domains such as logical reasoning and experimental coding. Motivated by these advancements, numerous studies have explored the application of AI in the innovation process, particularly in the context of scientific research. These AI technologies primarily aim to develop systems that can autonomously conduct research processes across a wide range of scientific disciplines. Despite these significant strides, a comprehensive survey on AI for Research (AI4Research) remains absent, which hampers our understanding and impedes further development in this field. To address this gap, we present a comprehensive survey and offer a unified perspective on AI4Research. Specifically, the main contributions of our work are as follows: (1) Systematic taxonomy: We first introduce a systematic taxonomy to classify five mainstream tasks in AI4Research. (2) New frontiers: Then, we identify key research gaps and highlight promising future directions, focusing on the rigor and scalability of automated experiments, as well as the societal impact. (3) Abundant applications and resources: Finally, we compile a wealth of resources, including relevant multidisciplinary applications, data corpora, and tools. We hope our work will provide the research community with quick access to these resources and stimulate innovative breakthroughs in AI4Research. |
| 2025-07-02 | [Out-of-Distribution Detection Methods Answer the Wrong Questions](http://arxiv.org/abs/2507.01831v1) | Yucen Lily Li, Daohan Lu et al. | To detect distribution shifts and improve model safety, many out-of-distribution (OOD) detection methods rely on the predictive uncertainty or features of supervised models trained on in-distribution data. In this paper, we critically re-examine this popular family of OOD detection procedures, and we argue that these methods are fundamentally answering the wrong questions for OOD detection. There is no simple fix to this misalignment, since a classifier trained only on in-distribution classes cannot be expected to identify OOD points; for instance, a cat-dog classifier may confidently misclassify an airplane if it contains features that distinguish cats from dogs, despite generally appearing nothing alike. We find that uncertainty-based methods incorrectly conflate high uncertainty with being OOD, while feature-based methods incorrectly conflate far feature-space distance with being OOD. We show how these pathologies manifest as irreducible errors in OOD detection and identify common settings where these methods are ineffective. Additionally, interventions to improve OOD detection such as feature-logit hybrid methods, scaling of model and data size, epistemic uncertainty representation, and outlier exposure also fail to address this fundamental misalignment in objectives. We additionally consider unsupervised density estimation and generative models for OOD detection, which we show have their own fundamental limitations. |
| 2025-07-02 | [Towards Design and Development of a Concentric Tube Steerable Drilling Robot for Creating S-shape Tunnels for Pelvic Fixation Procedures](http://arxiv.org/abs/2507.01811v1) | Yash Kulkarni, Susheela Sharma et al. | Current pelvic fixation techniques rely on rigid drilling tools, which inherently constrain the placement of rigid medical screws in the complex anatomy of pelvis. These constraints prevent medical screws from following anatomically optimal pathways and force clinicians to fixate screws in linear trajectories. This suboptimal approach, combined with the unnatural placement of the excessively long screws, lead to complications such as screw misplacement, extended surgery times, and increased radiation exposure due to repeated X-ray images taken ensure to safety of procedure. To address these challenges, in this paper, we present the design and development of a unique 4 degree-of-freedom (DoF) pelvic concentric tube steerable drilling robot (pelvic CT-SDR). The pelvic CT-SDR is capable of creating long S-shaped drilling trajectories that follow the natural curvatures of the pelvic anatomy. The performance of the pelvic CT-SDR was thoroughly evaluated through several S-shape drilling experiments in simulated bone phantoms. |
| 2025-07-02 | [Are Vision Transformer Representations Semantically Meaningful? A Case Study in Medical Imaging](http://arxiv.org/abs/2507.01788v1) | Montasir Shams, Chashi Mahiul Islam et al. | Vision transformers (ViTs) have rapidly gained prominence in medical imaging tasks such as disease classification, segmentation, and detection due to their superior accuracy compared to conventional deep learning models. However, due to their size and complex interactions via the self-attention mechanism, they are not well understood. In particular, it is unclear whether the representations produced by such models are semantically meaningful. In this paper, using a projected gradient-based algorithm, we show that their representations are not semantically meaningful and they are inherently vulnerable to small changes. Images with imperceptible differences can have very different representations; on the other hand, images that should belong to different semantic classes can have nearly identical representations. Such vulnerability can lead to unreliable classification results; for example, unnoticeable changes cause the classification accuracy to be reduced by over 60\%. %. To the best of our knowledge, this is the first work to systematically demonstrate this fundamental lack of semantic meaningfulness in ViT representations for medical image classification, revealing a critical challenge for their deployment in safety-critical systems. |
| 2025-07-02 | [Probing Evaluation Awareness of Language Models](http://arxiv.org/abs/2507.01786v1) | Jord Nguyen, Khiem Hoang et al. | Language models can distinguish between testing and deployment phases -- a capability known as evaluation awareness. This has significant safety and policy implications, potentially undermining the reliability of evaluations that are central to AI governance frameworks and voluntary industry commitments. In this paper, we study evaluation awareness in Llama-3.3-70B-Instruct. We show that linear probes can separate real-world evaluation and deployment prompts, suggesting that current models internally represent this distinction. We also find that current safety evaluations are correctly classified by the probes, suggesting that they already appear artificial or inauthentic to models. Our findings underscore the importance of ensuring trustworthy evaluations and understanding deceptive capabilities. More broadly, our work showcases how model internals may be leveraged to support blackbox methods in safety audits, especially for future models more competent at evaluation awareness and deception. |
| 2025-07-02 | [Signals and Symptoms: ICS Attack Dataset From Railway Cyber Range](http://arxiv.org/abs/2507.01768v1) | Anis Yusof, Yuancheng Liu et al. | The prevalence of cyberattacks on Industrial Control Systems (ICS) has highlighted the necessity for robust security measures and incident response to protect critical infrastructure. This is prominent when Operational Technology (OT) systems undergo digital transformation by integrating with Information Technology (IT) systems to enhance operational efficiency, adaptability, and safety. To support analysts in staying abreast of emerging attack patterns, there is a need for ICS datasets that reflect indicators representative of contemporary cyber threats. To address this, we conduct two ICS cyberattack simulations to showcase the impact of trending ICS cyberattacks on a railway cyber range that resembles the railway infrastructure. The attack scenario is designed to blend trending attack trends with attack patterns observed from historical ICS incidents. The resulting evidence is collected as datasets, serving as an essential resource for cyberattack analysis. This captures key indicators that are relevant to the current threat landscape, augmenting the effectiveness of security systems and analysts to protect against ICS cyber threats. |
| 2025-07-02 | [Globality and Regions](http://arxiv.org/abs/2507.01664v1) | Hector Gramaglia | We obtain a characterization of global variables by unifying abstraction with region abstraction in a region-based language. More precisely, in a previous work a language called global was presented, whose virtue is to provide a conceptually clear way of introducing imperative operations in a functional language. Memory safety is provided by the concept of linear protection, which connects the global system to a linear one. In this paper we show that the concept of global variable provided by the global language arises from the Tofte and Talping's region language through the unification of abstraction and region abstraction. |
| 2025-07-02 | [Time-Varying Coverage Control: A Distributed Tracker-Planner MPC Framework](http://arxiv.org/abs/2507.01567v1) | Patrick Benito Eberhard, Johannes Köhler et al. | Time-varying coverage control addresses the challenge of coordinating multiple agents covering an environment where regions of interest change over time. This problem has broad applications, including the deployment of autonomous taxis and coordination in search and rescue operations. The achievement of effective coverage is complicated by the presence of time-varying density functions, nonlinear agent dynamics, and stringent system and safety constraints. In this paper, we present a distributed multi-agent control framework for time-varying coverage under nonlinear constrained dynamics. Our approach integrates a reference trajectory planner and a tracking model predictive control (MPC) scheme, which operate at different frequencies within a multi-rate framework. For periodic density functions, we demonstrate closed-loop convergence to an optimal configuration of trajectories and provide formal guarantees regarding constraint satisfaction, collision avoidance, and recursive feasibility. Additionally, we propose an efficient algorithm capable of handling nonperiodic density functions, making the approach suitable for practical applications. Finally, we validate our method through hardware experiments using a fleet of four miniature race cars. |
| 2025-07-02 | [SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](http://arxiv.org/abs/2507.01513v1) | Beitao Chen, Xinyu Lyu et al. | By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility. |
| 2025-07-02 | [A Large Language Model for Chemistry and Retrosynthesis Predictions](http://arxiv.org/abs/2507.01444v1) | Yueqing Zhang, Wentao Liu et al. | Large language models (LLM) have achieved impressive progress across a broad range of general-purpose tasks, but their effectiveness in chemistry remains limited due to scarce domain-specific datasets and the demand for precise symbolic and structural reasoning. Here we introduce ECNU-ChemGPT(name after East China Normal University), a chemistry-specialized LLM engineered for deep chemical knowledge understanding and accurate retrosynthetic route planning. Our approach is distinguished by four key strategies: structured prompt-based knowledge distillation from authoritative chemistry textbooks to construct a high-quality question-answering dataset; domain-specific prompt engineering using curated chemical keywords, combined with LLMs APIs for data derivation and knowledge distillation; large-scale fine-tuning on a meticulously cleaned and enriched Pistachio reaction dataset to enhance retrosynthesis prediction accuracy; and integration of BrainGPT, a dynamic multi-model scheduling framework that enables task-specific invocation of multiple specialized models trained for diverse chemistry-related tasks. ECNU-ChemGPT exhibits superior performance on chemistry question-answering and retrosynthetic planning benchmarks, outperforming leading general-purpose models-including Deepseek-R1, Qwen-2.5, and GPT-4o. In retrosynthesis, it achieves a Top-1 accuracy of 68.3% on the USPTO_50K dataset and successfully reconstructed 13 complete experimental pathways for real-world drug molecules from medicinal chemistry journals. These results underscore the effectiveness of domain-adapted fine-tuning combined with dynamic multi-model task scheduling, providing a scalable and robust solution for chemical knowledge question answering and retrosynthetic planning. |
| 2025-07-02 | [Approximation-free Control of Unknown Euler-Lagrangian Systems under Input Constraints](http://arxiv.org/abs/2507.01426v1) | Ratnangshu Das, Pushpak Jagtap | In this paper, we present a novel funnel-based tracking control algorithm for robotic systems with unknown dynamics and prescribed input constraints. The Euler-Lagrange formulation, a common modeling approach for robotic systems, has been adopted in this study to address the trade-off between performance and actuator safety. We establish feasibility conditions that ensure tracking errors evolve within predefined funnel bounds while maintaining bounded control efforts, a crucial consideration for robots with limited actuation capabilities. We propose two approximation-free control strategies for scenarios where these conditions are violated: one actively corrects the error, and the other stops further deviation. Finally, we demonstrate the robust performance and safety of the approach through simulations and experimental validations. This work represents a significant advancement in funnel-based control, enhancing its applicability to real-world robotics systems with input constraints. |
| 2025-07-02 | [Dose-Escalation Trial Protocols that Extend Naturally to Admit Titration](http://arxiv.org/abs/2507.01370v1) | David C. Norris | Dose-escalation trials in oncology drug development still today typically aim to identify 1-size-fits-all dose recommendations, as arbitrary quantiles of the toxicity thresholds evident in patient samples. In the late 1990s efforts to individualize dosing emerged fleetingly in the oncology trial methods literature, but these have gained little traction due to a nexus of conceptual, technical, commercial, and regulatory barriers. To reduce the activation energy needed for transforming current 1-size-fits-all dose-escalation trial designs to the dose-titration designs required for patient-centered dose individualization, we demonstrate a categorical formulation of dose-escalation protocols that extends readily to allow gradual introduction of dose titration.   Central to this formulation is a symmetric monoidal preorder on the accessible states of dose-escalation trials, embodying pharmacologic intuitions regarding dose-monotonicity of drug toxicity and ethical intuitions relating to the therapeutic intent of such trials. A trial protocol that assigns doses to enrolling participants consistently with these intuitions is then a monotone map from this preorder to the sequence of doses being trialed. We illustrate this formulation by reference to the ubiquitous 3+3 dose-escalation design, which despite its many widely discussed flaws remains familiar to oncology trialists and moreover has available an executable specification in Prolog. Remarkably, examined in light of our preorder the 3+3 protocol discloses a new flaw not previously described: a non-monotone dose recommendation. The right Kan extension approximates this protocol from the side of safety, dissolving its triplet cohorts to allow incremental enrollment, and rectifying said non-monotonicity. It also facilitates accelerated enrollment while toxicity assessments remain pending, as well as discretionary dose titration. |
| 2025-07-02 | [Activation Reward Models for Few-Shot Model Alignment](http://arxiv.org/abs/2507.01368v1) | Tianning Chai, Chancharik Mitra et al. | Aligning Large Language Models (LLMs) and Large Multimodal Models (LMMs) to human preferences is a central challenge in improving the quality of the models' generative outputs for real-world applications. A common approach is to use reward modeling to encode preferences, enabling alignment via post-training using reinforcement learning. However, traditional reward modeling is not easily adaptable to new preferences because it requires a separate reward model, commonly trained on large preference datasets. To address this, we introduce Activation Reward Models (Activation RMs) -- a novel few-shot reward modeling method that leverages activation steering to construct well-aligned reward signals using minimal supervision and no additional model finetuning. Activation RMs outperform existing few-shot reward modeling approaches such as LLM-as-a-judge with in-context learning, voting-based scoring, and token probability scoring on standard reward modeling benchmarks. Furthermore, we demonstrate the effectiveness of Activation RMs in mitigating reward hacking behaviors, highlighting their utility for safety-critical applications. Toward this end, we propose PreferenceHack, a novel few-shot setting benchmark, the first to test reward models on reward hacking in a paired preference format. Finally, we show that Activation RM achieves state-of-the-art performance on this benchmark, surpassing even GPT-4o. |
| 2025-07-02 | [3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](http://arxiv.org/abs/2507.01367v1) | Tianrui Lou, Xiaojun Jia et al. | Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA. |
| 2025-07-02 | [Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](http://arxiv.org/abs/2507.01352v1) | Chris Yuhao Liu, Liang Zeng et al. | Despite the critical role of reward models (RMs) in reinforcement learning from human feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches that incorporate advanced training techniques have not yielded meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while large language models perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling, achieving state-of-the-art performance across seven major reward model benchmarks. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality. |
| 2025-07-02 | [Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs](http://arxiv.org/abs/2507.01334v1) | Nifu Dan, Yujun Cai et al. | Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains. |
| 2025-07-02 | [AI Meets Maritime Training: Precision Analytics for Enhanced Safety and Performance](http://arxiv.org/abs/2507.01274v1) | Vishakha Lall, Yisi Liu | Traditional simulator-based training for maritime professionals is critical for ensuring safety at sea but often depends on subjective trainer assessments of technical skills, behavioral focus, communication, and body language, posing challenges such as subjectivity, difficulty in measuring key features, and cognitive limitations. Addressing these issues, this study develops an AI-driven framework to enhance maritime training by objectively assessing trainee performance through visual focus tracking, speech recognition, and stress detection, improving readiness for high-risk scenarios. The system integrates AI techniques, including visual focus determination using eye tracking, pupil dilation analysis, and computer vision; communication analysis through a maritime-specific speech-to-text model and natural language processing; communication correctness using large language models; and mental stress detection via vocal pitch. Models were evaluated on data from simulated maritime scenarios with seafarers exposed to controlled high-stress events. The AI algorithms achieved high accuracy, with ~92% for visual detection, ~91% for maritime speech recognition, and ~90% for stress detection, surpassing existing benchmarks. The system provides insights into visual attention, adherence to communication checklists, and stress levels under demanding conditions. This study demonstrates how AI can transform maritime training by delivering objective performance analytics, enabling personalized feedback, and improving preparedness for real-world operational challenges. |
| 2025-07-02 | [LLM-based Realistic Safety-Critical Driving Video Generation](http://arxiv.org/abs/2507.01264v1) | Yongjie Fu, Ruijian Zha et al. | Designing diverse and safety-critical driving scenarios is essential for evaluating autonomous driving systems. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) for few-shot code generation to automatically synthesize driving scenarios within the CARLA simulator, which has flexibility in scenario scripting, efficient code-based control of traffic participants, and enforcement of realistic physical dynamics. Given a few example prompts and code samples, the LLM generates safety-critical scenario scripts that specify the behavior and placement of traffic participants, with a particular focus on collision events. To bridge the gap between simulation and real-world appearance, we integrate a video generation pipeline using Cosmos-Transfer1 with ControlNet, which converts rendered scenes into realistic driving videos. Our approach enables controllable scenario generation and facilitates the creation of rare but critical edge cases, such as pedestrian crossings under occlusion or sudden vehicle cut-ins. Experimental results demonstrate the effectiveness of our method in generating a wide range of realistic, diverse, and safety-critical scenarios, offering a promising tool for simulation-based testing of autonomous vehicles. |

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



