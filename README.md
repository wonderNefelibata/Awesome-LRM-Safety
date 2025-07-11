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
| 2025-07-10 | [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](http://arxiv.org/abs/2507.07999v1) | Haochen Wang, Xiangtai Li et al. | Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at https://github.com/Haochen-Wang409/TreeVGR. |
| 2025-07-10 | [Automating Expert-Level Medical Reasoning Evaluation of Large Language Models](http://arxiv.org/abs/2507.07988v1) | Shuang Zhou, Wenya Xie et al. | As large language models (LLMs) become increasingly integrated into clinical decision-making, ensuring transparent and trustworthy reasoning is essential. However, existing evaluation strategies of LLMs' medical reasoning capability either suffer from unsatisfactory assessment or poor scalability, and a rigorous benchmark remains lacking. To address this, we introduce MedThink-Bench, a benchmark designed for rigorous, explainable, and scalable assessment of LLMs' medical reasoning. MedThink-Bench comprises 500 challenging questions across ten medical domains, each annotated with expert-crafted step-by-step rationales. Building on this, we propose LLM-w-Ref, a novel evaluation framework that leverages fine-grained rationales and LLM-as-a-Judge mechanisms to assess intermediate reasoning with expert-level fidelity while maintaining scalability. Experiments show that LLM-w-Ref exhibits a strong positive correlation with expert judgments. Benchmarking twelve state-of-the-art LLMs, we find that smaller models (e.g., MedGemma-27B) can surpass larger proprietary counterparts (e.g., OpenAI-o3). Overall, MedThink-Bench offers a foundational tool for evaluating LLMs' medical reasoning, advancing their safe and responsible deployment in clinical practice. |
| 2025-07-10 | [Multimodal Framework for Explainable Autonomous Driving: Integrating Video, Sensor, and Textual Data for Enhanced Decision-Making and Transparency](http://arxiv.org/abs/2507.07938v1) | Abolfazl Zarghani, Amirhossein Ebrahimi et al. | Autonomous vehicles (AVs) are poised to redefine transportation by enhancing road safety, minimizing human error, and optimizing traffic efficiency. The success of AVs depends on their ability to interpret complex, dynamic environments through diverse data sources, including video streams, sensor measurements, and contextual textual information. However, seamlessly integrating these multimodal inputs and ensuring transparency in AI-driven decisions remain formidable challenges. This study introduces a novel multimodal framework that synergistically combines video, sensor, and textual data to predict driving actions while generating human-readable explanations, fostering trust and regulatory compliance. By leveraging VideoMAE for spatiotemporal video analysis, a custom sensor fusion module for real-time data processing, and BERT for textual comprehension, our approach achieves robust decision-making and interpretable outputs. Evaluated on the BDD-X (21113 samples) and nuScenes (1000 scenes) datasets, our model reduces training loss from 5.7231 to 0.0187 over five epochs, attaining an action prediction accuracy of 92.5% and a BLEU-4 score of 0.75 for explanation quality, outperforming state-of-the-art methods. Ablation studies confirm the critical role of each modality, while qualitative analyses and human evaluations highlight the model's ability to produce contextually rich, user-friendly explanations. These advancements underscore the transformative potential of multimodal integration and explainability in building safe, transparent, and trustworthy AV systems, paving the way for broader societal adoption of autonomous driving technologies. |
| 2025-07-10 | [KeyDroid: A Large-Scale Analysis of Secure Key Storage in Android Apps](http://arxiv.org/abs/2507.07927v1) | Jenny Blessing, Ross J. Anderson et al. | Most contemporary mobile devices offer hardware-backed storage for cryptographic keys, user data, and other sensitive credentials. Such hardware protects credentials from extraction by an adversary who has compromised the main operating system, such as a malicious third-party app. Since 2011, Android app developers can access trusted hardware via the Android Keystore API. In this work, we conduct the first comprehensive survey of hardware-backed key storage in Android devices. We analyze 490 119 Android apps, collecting data on how trusted hardware is used by app developers (if used at all) and cross-referencing our findings with sensitive user data collected by each app, as self-reported by developers via the Play Store's data safety labels.   We find that despite industry-wide initiatives to encourage adoption, 56.3% of apps self-reporting as processing sensitive user data do not use Android's trusted hardware capabilities at all, while just 5.03% of apps collecting some form of sensitive data use the strongest form of trusted hardware, a secure element distinct from the main processor. To better understand the potential downsides of using secure hardware, we conduct the first empirical analysis of trusted hardware performance in mobile devices, measuring the runtime of common cryptographic operations across both software- and hardware-backed keystores. We find that while hardware-backed key storage using a coprocessor is viable for most common cryptographic operations, secure elements capable of preventing more advanced attacks make performance infeasible for symmetric encryption with non-negligible payloads and any kind of asymmetric encryption. |
| 2025-07-10 | [Improving AEBS Validation Through Objective Intervention Classification Leveraging the Prediction Divergence Principle](http://arxiv.org/abs/2507.07872v1) | Daniel Betschinske, Steven Peters | The safety validation of automatic emergency braking system (AEBS) requires accurately distinguishing between false positive (FP) and true positive (TP) system activations. While simulations allow straightforward differentiation by comparing scenarios with and without interventions, analyzing activations from open-loop resimulations - such as those from field operational testing (FOT) - is more complex. This complexity arises from scenario parameter uncertainty and the influence of driver interventions in the recorded data. Human labeling is frequently used to address these challenges, relying on subjective assessments of intervention necessity or situational criticality, potentially introducing biases and limitations. This work proposes a rule-based classification approach leveraging the Prediction Divergence Principle (PDP) to address those issues. Applied to a simplified AEBS, the proposed method reveals key strengths, limitations, and system requirements for effective implementation. The findings suggest that combining this approach with human labeling may enhance the transparency and consistency of classification, thereby improving the overall validation process. While the rule set for classification derived in this work adopts a conservative approach, the paper outlines future directions for refinement and broader applicability. Finally, this work highlights the potential of such methods to complement existing practices, paving the way for more reliable and reproducible AEBS validation frameworks. |
| 2025-07-10 | [Set-Based Control Barrier Functions and Safety Filters](http://arxiv.org/abs/2507.07805v1) | Kim P. Wabersich, Felix Berkel et al. | High performance and formal safety guarantees are common requirements for industrial control applications. Control barrier function (CBF) methods provide a systematic approach to the modularization of safety and performance. However, the design of such CBFs can be challenging, which limits their applicability to large-scale or data-driven systems. This paper introduces the concept of a set-based CBF for linear systems with convex constraints. By leveraging control invariant sets from reachability analysis and predictive control, the set-based CBF is defined implicitly through the minimal scaling of such a set to contain the current system state. This approach enables the development of implicit, data-driven, and high-dimensional CBF representations. The paper demonstrates the design of a safety filter using set-based CBFs, which is suitable for real-time implementations and learning-based approximations to reduce online computational demands. The effectiveness of the method is illustrated through comprehensive simulations on a high-dimensional mass-spring-damper system and a motion control task, and it is validated experimentally using an electric drive application with short sampling times, highlighting its practical benefits for safety-critical control. |
| 2025-07-10 | [Collaborative Human-Robot Surgery for Mandibular Angle Split Osteotomy: Optical Tracking based Approach](http://arxiv.org/abs/2507.07794v1) | Zhe Han, Huanyu Tian et al. | Mandibular Angle Split Osteotomy (MASO) is a significant procedure in oral and maxillofacial surgery. Despite advances in technique and instrumentation, its success still relies heavily on the surgeon's experience. In this work, a human-robot collaborative system is proposed to perform MASO according to a preoperative plan and under guidance of a surgeon. A task decomposition methodology is used to divide the collaborative surgical procedure into three subtasks: (1) positional control and (2) orientation control, both led by the robot for precise alignment; and (3) force-control, managed by surgeon to ensure safety. Additionally, to achieve patient tracking without the need for a skull clamp, an optical tracking system (OTS) is utilized. Movement of the patient mandibular is measured with an optical-based tracker mounted on a dental occlusal splint. A registration method and Robot-OTS calibration method are introduced to achieve reliable navigation within our framework. The experiments of drilling were conducted on the realistic phantom model, which demonstrated that the average error between the planned and actual drilling points is 1.85mm. |
| 2025-07-10 | [GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing](http://arxiv.org/abs/2507.07735v1) | Peiyan Zhang, Haibo Jin et al. | Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models. |
| 2025-07-10 | [EEvAct: Early Event-Based Action Recognition with High-Rate Two-Stream Spiking Neural Networks](http://arxiv.org/abs/2507.07734v1) | Michael Neumeier, Jules Lecomte et al. | Recognizing human activities early is crucial for the safety and responsiveness of human-robot and human-machine interfaces. Due to their high temporal resolution and low latency, event-based vision sensors are a perfect match for this early recognition demand. However, most existing processing approaches accumulate events to low-rate frames or space-time voxels which limits the early prediction capabilities. In contrast, spiking neural networks (SNNs) can process the events at a high-rate for early predictions, but most works still fall short on final accuracy. In this work, we introduce a high-rate two-stream SNN which closes this gap by outperforming previous work by 2% in final accuracy on the large-scale THU EACT-50 dataset. We benchmark the SNNs within a novel early event-based recognition framework by reporting Top-1 and Top-5 recognition scores for growing observation time. Finally, we exemplify the impact of these methods on a real-world task of early action triggering for human motion capture in sports. |
| 2025-07-10 | [From Domain Documents to Requirements: Retrieval-Augmented Generation in the Space Industry](http://arxiv.org/abs/2507.07689v1) | Chetan Arora, Fanyu Wang et al. | Requirements engineering (RE) in the space industry is inherently complex, demanding high precision, alignment with rigorous standards, and adaptability to mission-specific constraints. Smaller space organisations and new entrants often struggle to derive actionable requirements from extensive, unstructured documents such as mission briefs, interface specifications, and regulatory standards. In this innovation opportunity paper, we explore the potential of Retrieval-Augmented Generation (RAG) models to support and (semi-)automate requirements generation in the space domain. We present a modular, AI-driven approach that preprocesses raw space mission documents, classifies them into semantically meaningful categories, retrieves contextually relevant content from domain standards, and synthesises draft requirements using large language models (LLMs). We apply the approach to a real-world mission document from the space domain to demonstrate feasibility and assess early outcomes in collaboration with our industry partner, Starbound Space Solutions. Our preliminary results indicate that the approach can reduce manual effort, improve coverage of relevant requirements, and support lightweight compliance alignment. We outline a roadmap toward broader integration of AI in RE workflows, intending to lower barriers for smaller organisations to participate in large-scale, safety-critical missions. |
| 2025-07-10 | [Synthetic MC via Biological Transmitters: Therapeutic Modulation of the Gut-Brain Axis](http://arxiv.org/abs/2507.07604v1) | Sebastian Lotter, Elisabeth Mohr et al. | Synthetic molecular communication (SMC) is a key enabler for future healthcare systems in which Internet of Bio-Nano-Things (IoBNT) devices facilitate the continuous monitoring of a patient's biochemical signals. To close the loop between sensing and actuation, both the detection and the generation of in-body molecular communication (MC) signals is key. However, generating signals inside the human body, e.g., via synthetic nanodevices, poses a challenge in SMC, due to technological obstacles as well as legal, safety, and ethical issues. Hence, this paper considers an SMC system in which signals are generated indirectly via the modulation of a natural in-body MC system, namely the gut-brain axis (GBA). Therapeutic GBA modulation is already established as treatment for neurological diseases, e.g., drug refractory epilepsy (DRE), and performed via the administration of nutritional supplements or specific diets. However, the molecular signaling pathways that mediate the effect of such treatments are mostly unknown. Consequently, existing treatments are standardized or designed heuristically and able to help only some patients while failing to help others. In this paper, we propose to leverage personal health data, e.g., gathered by in-body IoBNT devices, to design more versatile and robust GBA modulation-based treatments as compared to the existing ones. To show the feasibility of our approach, we define a catalog of theoretical requirements for therapeutic GBA modulation. Then, we propose a machine learning model to verify these requirements for practical scenarios when only limited data on the GBA modulation exists. By evaluating the proposed model on several datasets, we confirm its excellent accuracy in identifying different modulators of the GBA. Finally, we utilize the proposed model to identify specific modulatory pathways that play an important role for therapeutic GBA modulation. |
| 2025-07-10 | [Enhancing Vaccine Safety Surveillance: Extracting Vaccine Mentions from Emergency Department Triage Notes Using Fine-Tuned Large Language Models](http://arxiv.org/abs/2507.07599v1) | Sedigh Khademi, Jim Black et al. | This study evaluates fine-tuned Llama 3.2 models for extracting vaccine-related information from emergency department triage notes to support near real-time vaccine safety surveillance. Prompt engineering was used to initially create a labeled dataset, which was then confirmed by human annotators. The performance of prompt-engineered models, fine-tuned models, and a rule-based approach was compared. The fine-tuned Llama 3 billion parameter model outperformed other models in its accuracy of extracting vaccine names. Model quantization enabled efficient deployment in resource-constrained environments. Findings demonstrate the potential of large language models in automating data extraction from emergency department notes, supporting efficient vaccine safety surveillance and early detection of emerging adverse events following immunization issues. |
| 2025-07-10 | [Vaccine Hesitancy on YouTube: a Competition between Health and Politics](http://arxiv.org/abs/2507.07517v1) | Yelena Mejova, Michele Tizzani | YouTube has rapidly emerged as a predominant platform for content consumption, effectively displacing conventional media such as television and news outlets. A part of the enormous video stream uploaded to this platform includes health-related content, both from official public health organizations, and from any individual or group that can make an account. The quality of information available on YouTube is a critical point of public health safety, especially when concerning major interventions, such as vaccination. This study differentiates itself from previous efforts of auditing YouTube videos on this topic by conducting a systematic daily collection of posted videos mentioning vaccination for the duration of 3 months. We show that the competition for the public's attention is between public health messaging by institutions and individual educators on one side, and commentators on society and politics on the other, the latest contributing the most to the videos expressing stances against vaccination. Videos opposing vaccination are more likely to mention politicians and publication media such as podcasts, reports, and news analysis, on the other hand, videos in favor are more likely to mention specific diseases or health-related topics. Finally, we find that, at the time of analysis, only 2.7% of the videos have been taken down (by the platform or the channel), despite 20.8% of the collected videos having a vaccination hesitant stance, pointing to a lack of moderation activity for hesitant content. The availability of high-quality information is essential to improve awareness and compliance with public health interventions. Our findings help characterize the public discourse around vaccination on one of the largest media platforms, disentangling the role of the different creators and their stances, and as such, they provide important insights for public health communication policy. |
| 2025-07-10 | [Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision](http://arxiv.org/abs/2507.07460v1) | Jeonghoon Song, Sunghun Kim et al. | Out-of-Distribution (OoD) segmentation is critical for safety-sensitive applications like autonomous driving. However, existing mask-based methods often suffer from boundary imprecision, inconsistent anomaly scores within objects, and false positives from background noise. We propose \textbf{\textit{Objectomaly}}, an objectness-aware refinement framework that incorporates object-level priors. Objectomaly consists of three stages: (1) Coarse Anomaly Scoring (CAS) using an existing OoD backbone, (2) Objectness-Aware Score Calibration (OASC) leveraging SAM-generated instance masks for object-level score normalization, and (3) Meticulous Boundary Precision (MBP) applying Laplacian filtering and Gaussian smoothing for contour refinement. Objectomaly achieves state-of-the-art performance on key OoD segmentation benchmarks, including SMIYC AnomalyTrack/ObstacleTrack and RoadAnomaly, improving both pixel-level (AuPRC up to 96.99, FPR$_{95}$ down to 0.07) and component-level (F1$-$score up to 83.44) metrics. Ablation studies and qualitative results on real-world driving videos further validate the robustness and generalizability of our method. Code will be released upon publication. |
| 2025-07-10 | [Towards Safe Autonomous Driving: A Real-Time Safeguarding Concept for Motion Planning Algorithms](http://arxiv.org/abs/2507.07444v1) | Korbinian Moller, Rafael Neher et al. | Ensuring the functional safety of motion planning modules in autonomous vehicles remains a critical challenge, especially when dealing with complex or learning-based software. Online verification has emerged as a promising approach to monitor such systems at runtime, yet its integration into embedded real-time environments remains limited. This work presents a safeguarding concept for motion planning that extends prior approaches by introducing a time safeguard. While existing methods focus on geometric and dynamic feasibility, our approach additionally monitors the temporal consistency of planning outputs to ensure timely system response. A prototypical implementation on a real-time operating system evaluates trajectory candidates using constraint-based feasibility checks and cost-based plausibility metrics. Preliminary results show that the safeguarding module operates within real-time bounds and effectively detects unsafe trajectories. However, the full integration of the time safeguard logic and fallback strategies is ongoing. This study contributes a modular and extensible framework for runtime trajectory verification and highlights key aspects for deployment on automotive-grade hardware. Future work includes completing the safeguarding logic and validating its effectiveness through hardware-in-the-loop simulations and vehicle-based testing. The code is available at: https://github.com/TUM-AVS/motion-planning-supervisor |
| 2025-07-10 | [DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search](http://arxiv.org/abs/2507.07426v1) | Zerui Yang, Yuwei Wan et al. | Recent advances in large language models have demonstrated considerable potential in scientific domains such as drug discovery. However, their effectiveness remains constrained when reasoning extends beyond the knowledge acquired during pretraining. Conventional approaches, such as fine-tuning or retrieval-augmented generation, face limitations in either imposing high computational overhead or failing to fully exploit structured scientific data. To overcome these challenges, we propose DrugMCTS, a novel framework that synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree Search for drug repurposing. The framework employs five specialized agents tasked with retrieving and analyzing molecular and protein information, thereby enabling structured and iterative reasoning. Without requiring domain-specific fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate that DrugMCTS achieves substantially higher recall and robustness compared to both general-purpose LLMs and deep learning baselines. Our results highlight the importance of structured reasoning, agent-based collaboration, and feedback-driven search mechanisms in advancing LLM applications for drug discovery. |
| 2025-07-10 | [Corvid: Improving Multimodal Large Language Models Towards Chain-of-Thought Reasoning](http://arxiv.org/abs/2507.07424v1) | Jingjing Jiang, Chao Ma et al. | Recent advancements in multimodal large language models (MLLMs) have demonstrated exceptional performance in multimodal perception and understanding. However, leading open-source MLLMs exhibit significant limitations in complex and structured reasoning, particularly in tasks requiring deep reasoning for decision-making and problem-solving. In this work, we present Corvid, an MLLM with enhanced chain-of-thought (CoT) reasoning capabilities. Architecturally, Corvid incorporates a hybrid vision encoder for informative visual representation and a meticulously designed connector (GateMixer) to facilitate cross-modal alignment. To enhance Corvid's CoT reasoning capabilities, we introduce MCoT-Instruct-287K, a high-quality multimodal CoT instruction-following dataset, refined and standardized from diverse public reasoning sources. Leveraging this dataset, we fine-tune Corvid with a two-stage CoT-formatted training approach to progressively enhance its step-by-step reasoning abilities. Furthermore, we propose an effective inference-time scaling strategy that enables Corvid to mitigate over-reasoning and under-reasoning through self-verification. Extensive experiments demonstrate that Corvid outperforms existing o1-like MLLMs and state-of-the-art MLLMs with similar parameter scales, with notable strengths in mathematical reasoning and science problem-solving. Project page: https://mm-vl.github.io/corvid. |
| 2025-07-10 | [Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models](http://arxiv.org/abs/2507.07406v1) | Jikesh Thapa, Gurrehmat Chahal et al. | Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks. |
| 2025-07-09 | [On the Impossibility of Separating Intelligence from Judgment: The Computational Intractability of Filtering for AI Alignment](http://arxiv.org/abs/2507.07341v1) | Sarah Ball, Greg Gluch et al. | With the increased deployment of large language models (LLMs), one concern is their potential misuse for generating harmful content. Our work studies the alignment challenge, with a focus on filters to prevent the generation of unsafe information. Two natural points of intervention are the filtering of the input prompt before it reaches the model, and filtering the output after generation. Our main results demonstrate computational challenges in filtering both prompts and outputs. First, we show that there exist LLMs for which there are no efficient prompt filters: adversarial prompts that elicit harmful behavior can be easily constructed, which are computationally indistinguishable from benign prompts for any efficient filter. Our second main result identifies a natural setting in which output filtering is computationally intractable. All of our separation results are under cryptographic hardness assumptions. In addition to these core findings, we also formalize and study relaxed mitigation approaches, demonstrating further computational barriers. We conclude that safety cannot be achieved by designing filters external to the LLM internals (architecture and weights); in particular, black-box access to the LLM will not suffice. Based on our technical results, we argue that an aligned AI system's intelligence cannot be separated from its judgment. |
| 2025-07-09 | [Medical Red Teaming Protocol of Language Models: On the Importance of User Perspectives in Healthcare Settings](http://arxiv.org/abs/2507.07248v1) | Minseon Kim, Jean-Philippe Corbeil et al. | As the performance of large language models (LLMs) continues to advance, their adoption is expanding across a wide range of domains, including the medical field. The integration of LLMs into medical applications raises critical safety concerns, particularly due to their use by users with diverse roles, e.g. patients and clinicians, and the potential for model's outputs to directly affect human health. Despite the domain-specific capabilities of medical LLMs, prior safety evaluations have largely focused only on general safety benchmarks. In this paper, we introduce a safety evaluation protocol tailored to the medical domain in both patient user and clinician user perspectives, alongside general safety assessments and quantitatively analyze the safety of medical LLMs. We bridge a gap in the literature by building the PatientSafetyBench containing 466 samples over 5 critical categories to measure safety from the perspective of the patient. We apply our red-teaming protocols on the MediPhi model collection as a case study. To our knowledge, this is the first work to define safety evaluation criteria for medical LLMs through targeted red-teaming taking three different points of view - patient, clinician, and general user - establishing a foundation for safer deployment in medical domains. |

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



