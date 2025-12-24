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
| 2025-12-23 | [AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition](http://arxiv.org/abs/2512.20407v1) | Rajdeep Chatterjee, Sudip Chakrabarty et al. | Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used across diverse domains, including logistics, agriculture, surveillance, and defense. While these systems provide numerous benefits, their misuse raises safety and security concerns, making effective detection mechanisms essential. Acoustic sensing offers a low-cost and non-intrusive alternative to vision or radar-based detection, as drone propellers generate distinctive sound patterns. This study introduces AUDRON (AUdio-based Drone Recognition Network), a hybrid deep learning framework for drone sound detection, employing a combination of Mel-Frequency Cepstral Coefficients (MFCC), Short-Time Fourier Transform (STFT) spectrograms processed with convolutional neural networks (CNNs), recurrent layers for temporal modeling, and autoencoder-based representations. Feature-level fusion integrates complementary information before classification. Experimental evaluation demonstrates that AUDRON effectively differentiates drone acoustic signatures from background noise, achieving high accuracy while maintaining generalizability across varying conditions. AUDRON achieves 98.51 percent and 97.11 percent accuracy in binary and multiclass classification. The results highlight the advantage of combining multiple feature representations with deep learning for reliable acoustic drone detection, suggesting the framework's potential for deployment in security and surveillance applications where visual or radar sensing may be limited. |
| 2025-12-23 | [ChatGPT: Excellent Paper! Accept It. Editor: Imposter Found! Review Rejected](http://arxiv.org/abs/2512.20405v1) | Kanchon Gharami, Sanjiv Kumar Sarkar et al. | Large Language Models (LLMs) like ChatGPT are now widely used in writing and reviewing scientific papers. While this trend accelerates publication growth and reduces human workload, it also introduces serious risks. Papers written or reviewed by LLMs may lack real novelty, contain fabricated or biased results, or mislead downstream research that others depend on. Such issues can damage reputations, waste resources, and even endanger lives when flawed studies influence medical or safety-critical systems. This research explores both the offensive and defensive sides of this growing threat. On the attack side, we demonstrate how an author can inject hidden prompts inside a PDF that secretly guide or "jailbreak" LLM reviewers into giving overly positive feedback and biased acceptance. On the defense side, we propose an "inject-and-detect" strategy for editors, where invisible trigger prompts are embedded into papers; if a review repeats or reacts to these triggers, it reveals that the review was generated by an LLM, not a human. This method turns prompt injections from vulnerability into a verification tool. We outline our design, expected model behaviors, and ethical safeguards for deployment. The goal is to expose how fragile today's peer-review process becomes under LLM influence and how editorial awareness can help restore trust in scientific evaluation. |
| 2025-12-23 | [Toward Explaining Large Language Models in Software Engineering Tasks](http://arxiv.org/abs/2512.20328v1) | Antonio Vitale, Khai-Nguyen Nguyen et al. | Recent progress in Large Language Models (LLMs) has substantially advanced the automation of software engineering (SE) tasks, enabling complex activities such as code generation and code summarization. However, the black-box nature of LLMs remains a major barrier to their adoption in high-stakes and safety-critical domains, where explainability and transparency are vital for trust, accountability, and effective human supervision. Despite increasing interest in explainable AI for software engineering, existing methods lack domain-specific explanations aligned with how practitioners reason about SE artifacts. To address this gap, we introduce FeatureSHAP, the first fully automated, model-agnostic explainability framework tailored to software engineering tasks. Based on Shapley values, FeatureSHAP attributes model outputs to high-level input features through systematic input perturbation and task-specific similarity comparisons, while remaining compatible with both open-source and proprietary LLMs. We evaluate FeatureSHAP on two bi-modal SE tasks: code generation and code summarization. The results show that FeatureSHAP assigns less importance to irrelevant input features and produces explanations with higher fidelity than baseline methods. A practitioner survey involving 37 participants shows that FeatureSHAP helps practitioners better interpret model outputs and make more informed decisions. Collectively, FeatureSHAP represents a meaningful step toward practical explainable AI in software engineering. FeatureSHAP is available at https://github.com/deviserlab/FeatureSHAP. |
| 2025-12-23 | [AprielGuard](http://arxiv.org/abs/2512.20293v1) | Jaykumar Kasundra, Anjaneya Praharaj et al. | Safeguarding large language models (LLMs) against unsafe or adversarial behavior is critical as they are increasingly deployed in conversational and agentic settings. Existing moderation tools often treat safety risks (e.g. toxicity, bias) and adversarial threats (e.g. prompt injections, jailbreaks) as separate problems, limiting their robustness and generalizability. We introduce AprielGuard, an 8B parameter safeguard model that unify these dimensions within a single taxonomy and learning framework. AprielGuard is trained on a diverse mix of open and synthetic data covering standalone prompts, multi-turn conversations, and agentic workflows, augmented with structured reasoning traces to improve interpretability. Across multiple public and proprietary benchmarks, AprielGuard achieves strong performance in detecting harmful content and adversarial manipulations, outperforming existing opensource guardrails such as Llama-Guard and Granite Guardian, particularly in multi-step and reasoning intensive scenarios. By releasing the model, we aim to advance transparent and reproducible research on reliable safeguards for LLMs. |
| 2025-12-23 | [Auditing Reproducibility in Non-Targeted Analysis: 103 LC/GC--HRMS Tools Reveal Temporal Divergence Between Openness and Operability](http://arxiv.org/abs/2512.20279v1) | Sarah Alsubaie, Sakhaa Alsaedi et al. | In 2008, melamine in infant formula forced laboratories across three continents to verify a compound they had never monitored. Non-targeted analysis using LC/GC-HRMS handles these cases. But when findings trigger regulatory action, reproducibility becomes operational: can an independent laboratory repeat the analysis and reach the same conclusion?   We assessed 103 tools (2004-2025) against six pillars drawn from FAIR and BP4NTA principles: laboratory validation (C1), data availability (C2), code availability (C3), standardised formats (C4), knowledge integration (C5), and portable implementation (C6). Health contributed 51 tools, Pharma 31, and Chemistry 21.   Nine in ten tools shared data (C2, 90/103, 87%). Fewer than four in ten supported portable implementations (C6, 40/103, 39%). Validation and portability rarely appeared together (C1+C6, 18/103, 17%). Over twenty-one years, openness climbed from 56% to 86% while operability dropped from 55% to 43%. No tool addressed food safety.   Journal data-sharing policies increased what authors share but not what reviewers can run. Tools became easier to find but harder to execute. Strengthening C1, C4, and C6 would turn documented artifacts into workflows that external laboratories can replay. |
| 2025-12-23 | [Graph-Symbolic Policy Enforcement and Control (G-SPEC): A Neuro-Symbolic Framework for Safe Agentic AI in 5G Autonomous Networks](http://arxiv.org/abs/2512.20275v1) | Divya Vijay, Vignesh Ethiraj | As networks evolve toward 5G Standalone and 6G, operators face orchestration challenges that exceed the limits of static automation and Deep Reinforcement Learning. Although Large Language Model (LLM) agents offer a path toward intent-based networking, they introduce stochastic risks, including topology hallucinations and policy non-compliance. To mitigate this, we propose Graph-Symbolic Policy Enforcement and Control (G-SPEC), a neuro-symbolic framework that constrains probabilistic planning with deterministic verification. The architecture relies on a Governance Triad - a telecom-adapted agent (TSLAM-4B), a Network Knowledge Graph (NKG), and SHACL constraints. We evaluated G-SPEC on a simulated 450-node 5G Core, achieving zero safety violations and a 94.1% remediation success rate, significantly outperforming the 82.4% baseline. Ablation analysis indicates that NKG validation drives the majority of safety gains (68%), followed by SHACL policies (24%). Scalability tests on topologies ranging from 10K to 100K nodes demonstrate that validation latency scales as $O(k^{1.2})$ where $k$ is subgraph size. With a processing overhead of 142ms, G-SPEC is viable for SMO-layer operations. |
| 2025-12-23 | [Robust safety design for strict-feedback nonlinear systems via observer-based linear time varying feedback](http://arxiv.org/abs/2512.20226v1) | Imtiaz Ur Rehman, Moussa Labbadi et al. | This paper develops a robust safety-critical control method for nonlinear strictfeedback systems with mismatched disturbances. Using a state transformation and a linear time-varying disturbance observer, the system is converted into a form that enables safe control design. The approach ensures forward invariance of the safety set and also applies to disturbancefree systems. Safety is proven for all cases, and a numerical example illustrates the results. |
| 2025-12-23 | [LiteFusion: Taming 3D Object Detectors from Vision-Based to Multi-Modal with Minimal Adaptation](http://arxiv.org/abs/2512.20217v1) | Xiangxuan Ren, Zhongdao Wang et al. | 3D object detection is fundamental for safe and robust intelligent transportation systems. Current multi-modal 3D object detectors often rely on complex architectures and training strategies to achieve higher detection accuracy. However, these methods heavily rely on the LiDAR sensor so that they suffer from large performance drops when LiDAR is absent, which compromises the robustness and safety of autonomous systems in practical scenarios. Moreover, existing multi-modal detectors face difficulties in deployment on diverse hardware platforms, such as NPUs and FPGAs, due to their reliance on 3D sparse convolution operators, which are primarily optimized for NVIDIA GPUs. To address these challenges, we reconsider the role of LiDAR in the camera-LiDAR fusion paradigm and introduce a novel multi-modal 3D detector, LiteFusion. Instead of treating LiDAR point clouds as an independent modality with a separate feature extraction backbone, LiteFusion utilizes LiDAR data as a complementary source of geometric information to enhance camera-based detection. This straightforward approach completely eliminates the reliance on a 3D backbone, making the method highly deployment-friendly. Specifically, LiteFusion integrates complementary features from LiDAR points into image features within a quaternion space, where the orthogonal constraints are well-preserved during network training. This helps model domain-specific relations across modalities, yielding a compact cross-modal embedding. Experiments on the nuScenes dataset show that LiteFusion improves the baseline vision-based detector by +20.4% mAP and +19.7% NDS with a minimal increase in parameters (1.1%) without using dedicated LiDAR encoders. Notably, even in the absence of LiDAR input, LiteFusion maintains strong results , highlighting its favorable robustness and effectiveness across diverse fusion paradigms and deployment scenarios. |
| 2025-12-23 | [Reaching Agreement Among Reasoning LLM Agents](http://arxiv.org/abs/2512.20184v1) | Chaoyi Ruan, Yiliang Wang et al. | Multi-agent systems have extended the capability of agentic AI. Instead of single inference passes, multiple agents perform collective reasoning to derive high quality answers. However, existing multi-agent orchestration relies on static heuristic workflows such as fixed loop limits and barrier synchronization. These ad-hoc approaches waste computational resources, incur high latency due to stragglers, and risk finalizing transient agreements. We argue that reliable multi-agent reasoning requires a formal foundation analogous to classical distributed consensus problem.   To that end, we propose a formal model of the multi-agent refinement problem. The model includes definitions of the correctness guarantees and formal semantics of agent reasoning. We then introduce Aegean, a consensus protocol designed for stochastic reasoning agents that solves multi-agent refinement. We implement the protocol in Aegean-Serve, a consensus-aware serving engine that performs incremental quorum detection across concurrent agent executions, enabling early termination when sufficient agents converge. Evaluation using four mathematical reasoning benchmarks shows that Aegean provides provable safety and liveness guarantees while reducing latency by 1.2--20$\times$ compared to state-of-the-art baselines, maintaining answer quality within 2.5%. Consistent gains across both local GPU deployments and commercial API providers validate that consensus-based orchestration eliminates straggler delays without sacrificing correctness. |
| 2025-12-23 | [FaithLens: Detecting and Explaining Faithfulness Hallucination](http://arxiv.org/abs/2512.20182v1) | Shuzheng Si, Qingyi Wang et al. | Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness. |
| 2025-12-23 | [Offline Safe Policy Optimization From Heterogeneous Feedback](http://arxiv.org/abs/2512.20173v1) | Ze Gong, Pradeep Varakantham et al. | Offline Preference-based Reinforcement Learning (PbRL) learns rewards and policies aligned with human preferences without the need for extensive reward engineering and direct interaction with human annotators. However, ensuring safety remains a critical challenge across many domains and tasks. Previous works on safe RL from human feedback (RLHF) first learn reward and cost models from offline data, then use constrained RL to optimize a safe policy. While such an approach works in the contextual bandits settings (LLMs), in long horizon continuous control tasks, errors in rewards and costs accumulate, leading to impairment in performance when used with constrained RL methods. To address these challenges, (a) instead of indirectly learning policies (from rewards and costs), we introduce a framework that learns a policy directly based on pairwise preferences regarding the agent's behavior in terms of rewards, as well as binary labels indicating the safety of trajectory segments; (b) we propose \textsc{PreSa} (Preference and Safety Alignment), a method that combines preference learning module with safety alignment in a constrained optimization problem. This optimization problem is solved within a Lagrangian paradigm that directly learns reward-maximizing safe policy \textit{without explicitly learning reward and cost models}, avoiding the need for constrained RL; (c) we evaluate our approach on continuous control tasks with both synthetic and real human feedback. Empirically, our method successfully learns safe policies with high rewards, outperforming state-of-the-art baselines, and offline safe RL approaches with ground-truth reward and cost. |
| 2025-12-23 | [Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography](http://arxiv.org/abs/2512.20168v1) | Songze Li, Jiameng Cheng et al. | By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems. |
| 2025-12-23 | [MolAct: An Agentic RL Framework for Molecular Editing and Property Optimization](http://arxiv.org/abs/2512.20135v1) | Zhuo Yang, Yeyun chen et al. | Molecular editing and optimization are multi-step problems that require iteratively improving properties while keeping molecules chemically valid and structurally similar. We frame both tasks as sequential, tool-guided decisions and introduce MolAct, an agentic reinforcement learning framework that employs a two-stage training paradigm: first building editing capability, then optimizing properties while reusing the learned editing behaviors. To the best of our knowledge, this is the first work to formalize molecular design as an Agentic Reinforcement Learning problem, where an LLM agent learns to interleave reasoning, tool-use, and molecular optimization. The framework enables agents to interact in multiple turns, invoking chemical tools for validity checking, property assessment, and similarity control, and leverages their feedback to refine subsequent edits. We instantiate the MolAct framework to train two model families: MolEditAgent for molecular editing tasks and MolOptAgent for molecular optimization tasks. In molecular editing, MolEditAgent-7B delivers 100, 95, and 98 valid add, delete, and substitute edits, outperforming strong closed "thinking" baselines such as DeepSeek-R1; MolEditAgent-3B approaches the performance of much larger open "thinking" models like Qwen3-32B-think. In molecular optimization, MolOptAgent-7B (trained on MolEditAgent-7B) surpasses the best closed "thinking" baseline (e.g., Claude 3.7) on LogP and remains competitive on solubility, while maintaining balanced performance across other objectives. These results highlight that treating molecular design as a multi-step, tool-augmented process is key to reliable and interpretable improvements. |
| 2025-12-23 | [Multi Modal Attention Networks with Uncertainty Quantification for Automated Concrete Bridge Deck Delamination Detection](http://arxiv.org/abs/2512.20113v1) | Alireza Moayedikia, Sattar Dorafshan | Deteriorating civil infrastructure requires automated inspection techniques overcoming limitations of visual assessment. While Ground Penetrating Radar and Infrared Thermography enable subsurface defect detection, single modal approaches face complementary constraints radar struggles with moisture and shallow defects, while thermography exhibits weather dependency and limited depth. This paper presents a multi modal attention network fusing radar temporal patterns with thermal spatial signatures for bridge deck delamination detection. Our architecture introduces temporal attention for radar processing, spatial attention for thermal features, and cross modal fusion with learnable embeddings discovering complementary defect patterns invisible to individual sensors. We incorporate uncertainty quantification through Monte Carlo dropout and learned variance estimation, decomposing uncertainty into epistemic and aleatoric components for safety critical decisions. Experiments on five bridge datasets reveal that on balanced to moderately imbalanced data, our approach substantially outperforms baselines in accuracy and AUC representing meaningful improvements over single modal and concatenation based fusion. Ablation studies demonstrate cross modal attention provides critical gains beyond within modality attention, while multi head mechanisms achieve improved calibration. Uncertainty quantification reduces calibration error, enabling selective prediction by rejecting uncertain cases. However, under extreme class imbalance, attention mechanisms show vulnerability to majority class collapse. These findings provide actionable guidance: attention based architecture performs well across typical scenarios, while extreme imbalance requires specialized techniques. Our system maintains deployment efficiency, enabling real time inspection with characterized capabilities and limitations. |
| 2025-12-22 | [Scalably Enhancing the Clinical Validity of a Task Benchmark with Physician Oversight](http://arxiv.org/abs/2512.19691v1) | Junze Ye, Daniel Tawfik et al. | Automating the calculation of clinical risk scores offers a significant opportunity to reduce physician administrative burden and enhance patient care. The current standard for evaluating this capability is MedCalc-Bench, a large-scale dataset constructed using LLM-based feature extraction and rule-based aggregation. However, treating such model-generated benchmarks as static oracles risks enshrining historical model errors as evaluation gold standards, a problem dangerously amplified when these datasets serve as reward signals for Reinforcement Learning (RL). In this work, we propose viewing benchmarks for complex tasks such as clinical score computation as ''in-progress living documents'' that should be periodically re-evaluated as the processes for creating them improve. We introduce a systematic, physician-in-the-loop pipeline that leverages advanced agentic verifiers to audit and relabel MedCalc-Bench, utilizing automated triage to reserve scarce clinician attention for the most contentious instances. Our audit reveals that a notable fraction of original labels diverge from medical ground truth due to extraction errors, calculator logic mismatches, and clinical ambiguity. To study whether this label noise meaningfully impacts downstream RL training, we fine-tune a Qwen3-8B model via Group Relative Policy Optimization (GRPO) and demonstrate that training on corrected labels yields an 8.7% absolute improvement in accuracy over the original baseline -- validating that label noise materially affects model evaluation. These findings underscore that in safety-critical domains, rigorous benchmark maintenance is a prerequisite for genuine model alignment. |
| 2025-12-22 | [Optimal-coupling-observer AV motion control securing comfort in the presence of cyber attacks](http://arxiv.org/abs/2512.19679v1) | Farzam Tajdari, Georgios Papaioannou et al. | The security of Automated Vehicles (AVs) is an important emerging area of research in traffic safety. Methods have been published and evaluated in experimental vehicles to secure safe AV control in the presence of attacks, but human motion comfort is rarely investigated in such studies.   In this paper, we present an innovative optimal-coupling-observer-based framework that rejects the impact of bounded sensor attacks in a network of connected and automated vehicles from safety and comfort point of view.   We demonstrate its performance in car following with cooperative adaptive cruise control for platoons with redundant distance and velocity sensors.   The error dynamics are formulated as a Linear Time Variant (LTV) system, resulting in complex stability conditions that are investigated using a Linear Matrix Inequality (LMI) approach guaranteeing global asymptotic stability.   We prove the capability of the framework to secure occupants' safety and comfort in the presence of bounded attacks. In the onset of attack, the framework rapidly detects attacked sensors and switches to the most reliable observer eliminating attacked sensors, even with modest attack magnitudes. Without our proposed method, severe (but bounded) attacks result in collisions and major discomfort. With our method, attacks had negligible effects on motion comfort evaluated using ISO-2631 Ride Comfort and Motion Sickness indexes. The results pave the path to bring comfort to the forefront of AVs security. |
| 2025-12-22 | [Optimal Uncertainty Quantification under General Moment Constraints on Input Subdomains](http://arxiv.org/abs/2512.19572v1) | Rong Jin, Xingsheng Sun | We present an optimal uncertainty quantification (OUQ) framework for systems whose uncertain inputs are characterized by truncated moment constraints defined over subdomains. Based on this partial information, rigorous optimal upper and lower bounds on the probability of failure (PoF) are derived over the admissible set of probability measures, providing a principled basis for system safety certification. We formulate the OUQ problem under general subdomain moment constraints and develop a high-performance computational framework to compute the optimal bounds. This approach transforms the original infinite-dimensional optimization problems into finite-dimensional unconstrained ones parameterized solely by free canonical moments. To address the prohibitive cost of PoF evaluation in high-dimensional settings, we incorporate inverse transform sampling (ITS), enabling efficient and accurate PoF estimation within the OUQ optimization. We also demonstrate that constraining inputs only by zeroth-order moments over subdomains yields a formulation equivalent to evidence theory. Three groups of numerical examples demonstrate the framework's effectiveness and scalability. Results show that increasing the number of subdomains or the moment order systematically tightens the bound interval. For high-dimensional problems, the ITS strategy reduces computational costs by up to two orders of magnitude while maintaining relative error below 1%. Furthermore, we identify regimes where optimal bounds are sensitive to subdomain partitioning or higher-order moments, guiding uncertainty reduction efforts for safety certification. |
| 2025-12-22 | [Results of the 2024 CommonRoad Motion Planning Competition for Autonomous Vehicles](http://arxiv.org/abs/2512.19564v1) | Yanliang Huang, Xia Yan et al. | Over the past decade, a wide range of motion planning approaches for autonomous vehicles has been developed to handle increasingly complex traffic scenarios. However, these approaches are rarely compared on standardized benchmarks, limiting the assessment of relative strengths and weaknesses. To address this gap, we present the setup and results of the 4th CommonRoad Motion Planning Competition held in 2024, conducted using the CommonRoad benchmark suite. This annual competition provides an open-source and reproducible framework for benchmarking motion planning algorithms. The benchmark scenarios span highway and urban environments with diverse traffic participants, including passenger cars, buses, and bicycles. Planner performance is evaluated along four dimensions: efficiency, safety, comfort, and compliance with selected traffic rules. This report introduces the competition format and provides a comparison of representative high-performing planners from the 2023 and 2024 editions. |
| 2025-12-22 | [Augmenting Intelligence: A Hybrid Framework for Scalable and Stable Explanations](http://arxiv.org/abs/2512.19557v1) | Lawrence Krukrubo, Julius Odede et al. | Current approaches to Explainable AI (XAI) face a "Scalability-Stability Dilemma." Post-hoc methods (e.g., LIME, SHAP) may scale easily but suffer from instability, while supervised explanation frameworks (e.g., TED) offer stability but require prohibitive human effort to label every training instance. This paper proposes a Hybrid LRR-TED framework that addresses this dilemma through a novel "Asymmetry of Discovery." When applied to customer churn prediction, we demonstrate that automated rule learners (GLRM) excel at identifying broad "Safety Nets" (retention patterns) but struggle to capture specific "Risk Traps" (churn triggers)-a phenomenon we term the Anna Karenina Principle of Churn. By initialising the explanation matrix with automated safety rules and augmenting it with a Pareto-optimal set of just four human-defined risk rules, our approach achieves 94.00% predictive accuracy. This configuration outperforms the full 8-rule manual expert baseline while reducing human annotation effort by 50%, proposing a shift in the paradigm for Human-in-the-Loop AI: moving experts from the role of "Rule Writers" to "Exception Handlers." |
| 2025-12-22 | [Ferro-hydrodynamics of droplet necking filaments](http://arxiv.org/abs/2512.19459v1) | Neeladri Sekhar Bera, Apurba Roy et al. | We explore the necking, filament thinning, and pinchoff dynamics of ferrofluid droplets within a magnetic field, via a simple and low-cost experimental method. In our studies, both the Ohnesorge number Oh and the Deborah number De are O1, a typically inaccessible regime with conventional extensional rheometers. Under magnetic forcing, the nanoparticles assemble into field aligned, chainlike structures, that generate a tunable magnetoelastic response, and markedly alter the extensional flow. Although behaving as Newtonian liquids in the absence of a magnetic field, the field induces extensional thickening, and the emergence of beads on a string BOAS structures in the ferrofluid filaments, a non-Newtonian signature. By combining controlled elongation with high speed imaging, we directly quantify the magnetic field-dependent extensional viscosity and relaxation time. Our findings underscore how magnetically induced microstructures govern filament stability and extensional dynamics in ferrofluids. |

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



