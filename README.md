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
| 2025-12-29 | [Le Cam Distortion: A Decision-Theoretic Framework for Robust Transfer Learning](http://arxiv.org/abs/2512.23617v1) | Deniz Akdemir | Distribution shift is the defining challenge of real-world machine learning. The dominant paradigm--Unsupervised Domain Adaptation (UDA)--enforces feature invariance, aligning source and target representations via symmetric divergence minimization [Ganin et al., 2016]. We demonstrate that this approach is fundamentally flawed: when domains are unequally informative (e.g., high-quality vs degraded sensors), strict invariance necessitates information destruction, causing "negative transfer" that can be catastrophic in safety-critical applications [Wang et al., 2019].   We propose a decision-theoretic framework grounded in Le Cam's theory of statistical experiments [Le Cam, 1986], using constructive approximations to replace symmetric invariance with directional simulability. We introduce Le Cam Distortion, quantified by the Deficiency Distance $δ(E_1, E_2)$, as a rigorous upper bound for transfer risk conditional on simulability. Our framework enables transfer without source degradation by learning a kernel that simulates the target from the source. Across five experiments (genomics, vision, reinforcement learning), Le Cam Distortion achieves: (1) near-perfect frequency estimation in HLA genomics (correlation $r=0.999$, matching classical methods), (2) zero source utility loss in CIFAR-10 image classification (81.2% accuracy preserved vs 34.7% drop for CycleGAN), and (3) safe policy transfer in RL control where invariance-based methods suffer catastrophic collapse. Le Cam Distortion provides the first principled framework for risk-controlled transfer learning in domains where negative transfer is unacceptable: medical imaging, autonomous systems, and precision medicine. |
| 2025-12-29 | [Unsupervised Learning for Detection of Rare Driving Scenarios](http://arxiv.org/abs/2512.23585v1) | Dat Le, Thomas Manhardt et al. | The detection of rare and hazardous driving scenarios is a critical challenge for ensuring the safety and reliability of autonomous systems. This research explores an unsupervised learning framework for detecting rare and extreme driving scenarios using naturalistic driving data (NDD). We leverage the recently proposed Deep Isolation Forest (DIF), an anomaly detection algorithm that combines neural network-based feature representations with Isolation Forests (IFs), to identify non-linear and complex anomalies. Data from perception modules, capturing vehicle dynamics and environmental conditions, is preprocessed into structured statistical features extracted from sliding windows. The framework incorporates t-distributed stochastic neighbor embedding (t-SNE) for dimensionality reduction and visualization, enabling better interpretability of detected anomalies. Evaluation is conducted using a proxy ground truth, combining quantitative metrics with qualitative video frame inspection. Our results demonstrate that the proposed approach effectively identifies rare and hazardous driving scenarios, providing a scalable solution for anomaly detection in autonomous driving systems. Given the study's methodology, it was unavoidable to depend on proxy ground truth and manually defined feature combinations, which do not encompass the full range of real-world driving anomalies or their nuanced contextual dependencies. |
| 2025-12-29 | [ProGuard: Towards Proactive Multimodal Safeguard](http://arxiv.org/abs/2512.23573v1) | Shaohan Yu, Lijun Li et al. | The rapid evolution of generative models has led to a continuous emergence of multimodal safety risks, exposing the limitations of existing defense methods. To address these challenges, we propose ProGuard, a vision-language proactive guard that identifies and describes out-of-distribution (OOD) safety risks without the need for model adjustments required by traditional reactive approaches. We first construct a modality-balanced dataset of 87K samples, each annotated with both binary safety labels and risk categories under a hierarchical multimodal safety taxonomy, effectively mitigating modality bias and ensuring consistent moderation across text, image, and text-image inputs. Based on this dataset, we train our vision-language base model purely through reinforcement learning (RL) to achieve efficient and concise reasoning. To approximate proactive safety scenarios in a controlled setting, we further introduce an OOD safety category inference task and augment the RL objective with a synonym-bank-based similarity reward that encourages the model to generate concise descriptions for unseen unsafe categories. Experimental results show that ProGuard achieves performance comparable to closed-source large models on binary safety classification, substantially outperforms existing open-source guard models on unsafe content categorization. Most notably, ProGuard delivers a strong proactive moderation ability, improving OOD risk detection by 52.6% and OOD risk description by 64.8%. |
| 2025-12-29 | [PurifyGen: A Risk-Discrimination and Semantic-Purification Model for Safe Text-to-Image Generation](http://arxiv.org/abs/2512.23546v1) | Zongsheng Cao, Yangfan He et al. | Recent advances in diffusion models have notably enhanced text-to-image (T2I) generation quality, but they also raise the risk of generating unsafe content. Traditional safety methods like text blacklisting or harmful content classification have significant drawbacks: they can be easily circumvented or require extensive datasets and extra training. To overcome these challenges, we introduce PurifyGen, a novel, training-free approach for safe T2I generation that retains the model's original weights. PurifyGen introduces a dual-stage strategy for prompt purification. First, we evaluate the safety of each token in a prompt by computing its complementary semantic distance, which measures the semantic proximity between the prompt tokens and concept embeddings from predefined toxic and clean lists. This enables fine-grained prompt classification without explicit keyword matching or retraining. Tokens closer to toxic concepts are flagged as risky. Second, for risky prompts, we apply a dual-space transformation: we project toxic-aligned embeddings into the null space of the toxic concept matrix, effectively removing harmful semantic components, and simultaneously align them into the range space of clean concepts. This dual alignment purifies risky prompts by both subtracting unsafe semantics and reinforcing safe ones, while retaining the original intent and coherence. We further define a token-wise strategy to selectively replace only risky token embeddings, ensuring minimal disruption to safe content. PurifyGen offers a plug-and-play solution with theoretical grounding and strong generalization to unseen prompts and models. Extensive testing shows that PurifyGen surpasses current methods in reducing unsafe content across five datasets and competes well with training-dependent approaches. The code can refer to https://github.com/AI-Researcher-Team/PurifyGen. |
| 2025-12-29 | [Trustworthy Machine Learning under Distribution Shifts](http://arxiv.org/abs/2512.23524v1) | Zhuo Huang | Machine Learning (ML) has been a foundational topic in artificial intelligence (AI), providing both theoretical groundwork and practical tools for its exciting advancements. From ResNet for visual recognition to Transformer for vision-language alignment, the AI models have achieved superior capability to humans. Furthermore, the scaling law has enabled AI to initially develop general intelligence, as demonstrated by Large Language Models (LLMs). To this stage, AI has had an enormous influence on society and yet still keeps shaping the future for humanity. However, distribution shift remains a persistent ``Achilles' heel'', fundamentally limiting the reliability and general usefulness of ML systems. Moreover, generalization under distribution shift would also cause trust issues for AIs. Motivated by these challenges, my research focuses on \textit{Trustworthy Machine Learning under Distribution Shifts}, with the goal of expanding AI's robustness, versatility, as well as its responsibility and reliability. We carefully study the three common distribution shifts into: (1) Perturbation Shift, (2) Domain Shift, and (3) Modality Shift. For all scenarios, we also rigorously investigate trustworthiness via three aspects: (1) Robustness, (2) Explainability, and (3) Adaptability. Based on these dimensions, we propose effective solutions and fundamental insights, meanwhile aiming to enhance the critical ML problems, such as efficiency, adaptability, and safety. |
| 2025-12-29 | [Why AI Safety Requires Uncertainty, Incomplete Preferences, and Non-Archimedean Utilities](http://arxiv.org/abs/2512.23508v1) | Alessio Benavoli, Alessandro Facchini et al. | How can we ensure that AI systems are aligned with human values and remain safe? We can study this problem through the frameworks of the AI assistance and the AI shutdown games. The AI assistance problem concerns designing an AI agent that helps a human to maximise their utility function(s). However, only the human knows these function(s); the AI assistant must learn them. The shutdown problem instead concerns designing AI agents that: shut down when a shutdown button is pressed; neither try to prevent nor cause the pressing of the shutdown button; and otherwise accomplish their task competently. In this paper, we show that addressing these challenges requires AI agents that can reason under uncertainty and handle both incomplete and non-Archimedean preferences. |
| 2025-12-29 | [Robust Deep Learning Control with Guaranteed Performance for Safe and Reliable Robotization in Heavy-Duty Machinery](http://arxiv.org/abs/2512.23505v1) | Mehdi Heydari Shahna | Today's heavy-duty mobile machines (HDMMs) face two transitions: from diesel-hydraulic actuation to clean electric systems driven by climate goals, and from human supervision toward greater autonomy. Diesel-hydraulic systems have long dominated, so full electrification, via direct replacement or redesign, raises major technical and economic challenges. Although advanced artificial intelligence (AI) could enable higher autonomy, adoption in HDMMs is limited by strict safety requirements, and these machines still rely heavily on human supervision.   This dissertation develops a control framework that (1) simplifies control design for electrified HDMMs through a generic modular approach that is energy-source independent and supports future modifications, and (2) defines hierarchical control policies that partially integrate AI while guaranteeing safety-defined performance and stability.   Five research questions align with three lines of investigation: a generic robust control strategy for multi-body HDMMs with strong stability across actuation types and energy sources; control solutions that keep strict performance under uncertainty and faults while balancing robustness and responsiveness; and methods to interpret and trust black-box learning strategies so they can be integrated stably and verified against international safety standards.   The framework is validated in three case studies spanning different actuators and conditions, covering heavy-duty mobile robots and robotic manipulators. Results appear in five peer-reviewed publications and one unpublished manuscript, advancing nonlinear control and robotics and supporting both transitions. |
| 2025-12-29 | [ML Compass: Navigating Capability, Cost, and Compliance Trade-offs in AI Model Deployment](http://arxiv.org/abs/2512.23487v1) | Vassilis Digalakis, Ramayya Krishnan et al. | We study how organizations should select among competing AI models when user utility, deployment costs, and compliance requirements jointly matter. Widely used capability leaderboards do not translate directly into deployment decisions, creating a capability--deployment gap; to bridge it, we take a systems-level view in which model choice is tied to application outcomes, operating constraints, and a capability--cost frontier. We develop ML Compass, a framework that treats model selection as constrained optimization over this frontier. On the theory side, we characterize optimal model configurations under a parametric frontier and show a three-regime structure in optimal internal measures: some dimensions are pinned at compliance minima, some saturate at maximum levels, and the remainder take interior values governed by frontier curvature. We derive comparative statics that quantify how budget changes, regulatory tightening, and technological progress propagate across capability dimensions and costs. On the implementation side, we propose a pipeline that (i) extracts low-dimensional internal measures from heterogeneous model descriptors, (ii) estimates an empirical frontier from capability and cost data, (iii) learns a user- or task-specific utility function from interaction outcome data, and (iv) uses these components to target capability--cost profiles and recommend models. We validate ML Compass with two case studies: a general-purpose conversational setting using the PRISM Alignment dataset and a healthcare setting using a custom dataset we build using HealthBench. In both environments, our framework produces recommendations -- and deployment-aware leaderboards based on predicted deployment value under constraints -- that can differ materially from capability-only rankings, and clarifies how trade-offs between capability, cost, and safety shape optimal model choice. |
| 2025-12-29 | [Assessing behaviour coverage in a multi-agent system simulation for autonomous vehicle testing](http://arxiv.org/abs/2512.23445v1) | Manuel Franco-Vivo | As autonomous vehicle technology advances, ensuring the safety and reliability of these systems becomes paramount. Consequently, comprehensive testing methodologies are essential to evaluate the performance of autonomous vehicles in diverse and complex real-world scenarios. This study focuses on the behaviour coverage analysis of a multi-agent system simulation designed for autonomous vehicle testing, and provides a systematic approach to measure and assess behaviour coverage within the simulation environment. By defining a set of driving scenarios, and agent interactions, we evaluate the extent to which the simulation encompasses a broad range of behaviours relevant to autonomous driving.   Our findings highlight the importance of behaviour coverage in validating the effectiveness and robustness of autonomous vehicle systems. Through the analysis of behaviour coverage metrics and coverage-based testing, we identify key areas for improvement and optimization in the simulation framework. Thus, a Model Predictive Control (MPC) pedestrian agent is proposed, where its objective function is formulated to encourage \textit{interesting} tests while promoting a more realistic behaviour than other previously studied pedestrian agents. This research contributes to advancing the field of autonomous vehicle testing by providing insights into the comprehensive evaluation of system behaviour in simulated environments. The results offer valuable implications for enhancing the safety, reliability, and performance of autonomous vehicles through rigorous testing methodologies. |
| 2025-12-29 | [Adaptive Fusion Graph Network for 3D Strain Field Prediction in Solid Rocket Motor Grains](http://arxiv.org/abs/2512.23443v1) | Jiada Huang, Hao Ma et al. | Local high strain in solid rocket motor grains is a primary cause of structural failure. However, traditional numerical simulations are computationally expensive, and existing surrogate models cannot explicitly establish geometric models and accurately capture high-strain regions. Therefore, this paper proposes an adaptive graph network, GrainGNet, which employs an adaptive pooling dynamic node selection mechanism to effectively preserve the key mechanical features of structurally critical regions, while concurrently utilising feature fusion to transmit deep features and enhance the model's representational capacity. In the joint prediction task involving four sequential conditions--curing and cooling, storage, overloading, and ignition--GrainGNet reduces the mean squared error by 62.8% compared to the baseline graph U-Net model, with only a 5.2% increase in parameter count and an approximately sevenfold improvement in training efficiency. Furthermore, in the high-strain regions of debonding seams, the prediction error is further reduced by 33% compared to the second-best method, offering a computationally efficient and high-fidelity approach to evaluate motor structural safety. |
| 2025-12-29 | [An SLO Driven and Cost-Aware Autoscaling Framework for Kubernetes](http://arxiv.org/abs/2512.23415v1) | Vinoth Punniyamoorthy, Bikesh Kumar et al. | Kubernetes provides native autoscaling mechanisms, including the Horizontal Pod Autoscaler, Vertical Pod Autoscaler, and node-level autoscalers, to enable elastic resource management for cloud-native applications. However, production environments frequently experience Service Level Objective violations and cost inefficiencies due to reactive scaling behavior, limited use of application-level signals, and opaque control logic. This paper investigates how Kubernetes autoscaling can be enhanced using AIOps principles to jointly satisfy SLO and cost constraints under diverse workload patterns without compromising safety or operational transparency. We present a gap-driven analysis of existing autoscaling approaches and propose a safe and explainable multi-signal autoscaling framework that integrates SLO-aware and cost-conscious control with lightweight demand forecasting. Experimental evaluation using representative microservice and event-driven workloads shows that the proposed approach reduces SLO violation duration by up to 31 percent, improves scaling response time by 24 percent, and lowers infrastructure cost by 18 percent compared to default and tuned Kubernetes autoscaling baselines, while maintaining stable and auditable control behavior. These results demonstrate that AIOps-driven, SLO-first autoscaling can significantly improve the reliability, efficiency, and operational trustworthiness of Kubernetes-based cloud platforms. |
| 2025-12-29 | [Explainable Neural Inverse Kinematics for Obstacle-Aware Robotic Manipulation: A Comparative Analysis of IKNet Variants](http://arxiv.org/abs/2512.23312v1) | Sheng-Kai Chen, Yi-Ling Tsai et al. | Deep neural networks have accelerated inverse-kinematics (IK) inference to the point where low cost manipulators can execute complex trajectories in real time, yet the opaque nature of these models contradicts the transparency and safety requirements emerging in responsible AI regulation. This study proposes an explainability centered workflow that integrates Shapley-value attribution with physics-based obstacle avoidance evaluation for the ROBOTIS OpenManipulator-X. Building upon the original IKNet, two lightweight variants-Improved IKNet with residual connections and Focused IKNet with position-orientation decoupling are trained on a large, synthetically generated pose-joint dataset. SHAP is employed to derive both global and local importance rankings, while the InterpretML toolkit visualizes partial-dependence patterns that expose non-linear couplings between Cartesian poses and joint angles. To bridge algorithmic insight and robotic safety, each network is embedded in a simulator that subjects the arm to randomized single and multi-obstacle scenes; forward kinematics, capsule-based collision checks, and trajectory metrics quantify the relationship between attribution balance and physical clearance. Qualitative heat maps reveal that architectures distributing importance more evenly across pose dimensions tend to maintain wider safety margins without compromising positional accuracy. The combined analysis demonstrates that explainable AI(XAI) techniques can illuminate hidden failure modes, guide architectural refinements, and inform obstacle aware deployment strategies for learning based IK. The proposed methodology thus contributes a concrete path toward trustworthy, data-driven manipulation that aligns with emerging responsible-AI standards. |
| 2025-12-29 | [Agentic Physical AI toward a Domain-Specific Foundation Model for Nuclear Reactor Control](http://arxiv.org/abs/2512.23292v1) | Yoonpyo Lee, Kazuma Kobayashi et al. | The prevailing paradigm in AI for physical systems, scaling general-purpose foundation models toward universal multimodal reasoning, confronts a fundamental barrier at the control interface. Recent benchmarks show that even frontier vision-language models achieve only 50-53% accuracy on basic quantitative physics tasks, behaving as approximate guessers that preserve semantic plausibility while violating physical constraints. This input unfaithfulness is not a scaling deficiency but a structural limitation. Perception-centric architectures optimize parameter-space imitation, whereas safety-critical control demands outcome-space guarantees over executed actions. Here, we present a fundamentally different pathway toward domain-specific foundation models by introducing compact language models operating as Agentic Physical AI, in which policy optimization is driven by physics-based validation rather than perceptual inference. We train a 360-million-parameter model on synthetic reactor control scenarios, scaling the dataset from 10^3 to 10^5 examples. This induces a sharp phase transition absent in general-purpose models. Small-scale systems exhibit high-variance imitation with catastrophic tail risk, while large-scale models undergo variance collapse exceeding 500x reduction, stabilizing execution-level behavior. Despite balanced exposure to four actuation families, the model autonomously rejects approximately 70% of the training distribution and concentrates 95% of runtime execution on a single-bank strategy. Learned representations transfer across distinct physics and continuous input modalities without architectural modification. |
| 2025-12-29 | [Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation](http://arxiv.org/abs/2512.23260v1) | Dianyun Wang, Qingsen Ma et al. | Parameter-efficient fine-tuning has become the dominant paradigm for adapting large language models to downstream tasks. Low-rank adaptation methods such as LoRA operate under the assumption that task-relevant weight updates reside in a low-rank subspace, yet this subspace is learned implicitly from data in a black-box manner, offering no interpretability or direct control. We hypothesize that this difficulty stems from polysemanticity--individual dimensions encoding multiple entangled concepts. To address this, we leverage pre-trained Sparse Autoencoders (SAEs) to identify task-relevant features in a disentangled feature space, then construct an explicit, interpretable low-rank subspace to guide adapter initialization. We provide theoretical analysis proving that under monosemanticity assumptions, SAE-based subspace identification achieves arbitrarily small recovery error, while direct identification in polysemantic space suffers an irreducible error floor. On safety alignment, our method achieves up to 99.6% safety rate--exceeding full fine-tuning by 7.4 percentage points and approaching RLHF-based methods--while updating only 0.19-0.24% of parameters. Crucially, our method provides interpretable insights into the learned alignment subspace through the semantic grounding of SAE features. Our work demonstrates that incorporating mechanistic interpretability into the fine-tuning process can simultaneously improve both performance and transparency. |
| 2025-12-29 | [Physics-Inspired Modeling and Content Adaptive Routing in an Infrared Gas Leak Detection Network](http://arxiv.org/abs/2512.23234v1) | Dongsheng Li, Chaobo Chen et al. | Detecting infrared gas leaks is critical for environmental monitoring and industrial safety, yet remains difficult because plumes are faint, small, semitransparent, and have weak, diffuse boundaries. We present physics-edge hybrid gas dynamic routing network (PEG-DRNet). First, we introduce the Gas Block, a diffusion-convection unit modeling gas transport: a local branch captures short-range variations, while a large-kernel branch captures long-range propagation. An edge-gated learnable fusion module balances local detail and global context, strengthening weak-contrast plume and contour cues. Second, we propose the adaptive gradient and phase edge operator (AGPEO), computing reliable edge priors from multi-directional gradients and phase-consistent responses. These are transformed by a multi-scale edge perception module (MSEPM) into hierarchical edge features that reinforce boundaries. Finally, the content-adaptive sparse routing path aggregation network (CASR-PAN), with adaptive information modulation modules for fusion and self, selectively propagates informative features across scales based on edge and content cues, improving cross-scale discriminability while reducing redundancy. Experiments on the IIG dataset show that PEG-DRNet achieves an overall AP of 29.8\%, an AP$_{50}$ of 84.3\%, and a small-object AP of 25.3\%, surpassing the RT-DETR-R18 baseline by 3.0\%, 6.5\%, and 5.3\%, respectively, while requiring only 43.7 Gflops and 14.9 M parameters. The proposed PEG-DRNet achieves superior overall performance with the best balance of accuracy and computational efficiency, outperforming existing CNN and Transformer detectors in AP and AP$_{50}$ on the IIG and LangGas dataset. |
| 2025-12-29 | [LogosQ: A High-Performance and Type-Safe Quantum Computing Library in Rust](http://arxiv.org/abs/2512.23183v1) | Shiwen An, Jiayi Wang et al. | Developing robust and high performance quantum software is challenging due to the dynamic nature of existing Python-based frameworks, which often suffer from runtime errors and scalability bottlenecks. In this work, we present LogosQ, a high performance backend agnostic quantum computing library implemented in Rust that enforces correctness through compile time type safety. Unlike existing tools, LogosQ leverages Rust static analysis to eliminate entire classes of runtime errors, particularly in parameter-shift rule gradient computations for variational algorithms. We introduce novel optimization techniques, including direct state-vector manipulation, adaptive parallel processing, and an FFT optimized Quantum Fourier Transform, which collectively deliver speedups of up to 900 times for state preparation (QFT) and 2 to 5 times for variational workloads over Python frameworks (PennyLane, Qiskit), 6 to 22 times over Julia implementations (Yao), and competitive performance with Q sharp. Beyond performance, we validate numerical stability through variational quantum eigensolver (VQE) experiments on molecular hydrogen and XYZ Heisenberg models, achieving chemical accuracy even in edge cases where other libraries fail. By combining the safety of systems programming with advanced circuit optimization, LogosQ establishes a new standard for reliable and efficient quantum simulation. |
| 2025-12-29 | [EquaCode: A Multi-Strategy Jailbreak Approach for Large Language Models via Equation Solving and Code Completion](http://arxiv.org/abs/2512.23173v1) | Zhen Liang, Hai Huang et al. | Large language models (LLMs), such as ChatGPT, have achieved remarkable success across a wide range of fields. However, their trustworthiness remains a significant concern, as they are still susceptible to jailbreak attacks aimed at eliciting inappropriate or harmful responses. However, existing jailbreak attacks mainly operate at the natural language level and rely on a single attack strategy, limiting their effectiveness in comprehensively assessing LLM robustness. In this paper, we propose Equacode, a novel multi-strategy jailbreak approach for large language models via equation-solving and code completion. This approach transforms malicious intent into a mathematical problem and then requires the LLM to solve it using code, leveraging the complexity of cross-domain tasks to divert the model's focus toward task completion rather than safety constraints. Experimental results show that Equacode achieves an average success rate of 91.19% on the GPT series and 98.65% across 3 state-of-the-art LLMs, all with only a single query. Further, ablation experiments demonstrate that EquaCode outperforms either the mathematical equation module or the code module alone. This suggests a strong synergistic effect, thereby demonstrating that multi-strategy approach yields results greater than the sum of its parts. |
| 2025-12-29 | [Evaluating Parameter Efficient Methods for RLVR](http://arxiv.org/abs/2512.23165v1) | Qingyu Yin, Yulun Wu et al. | We systematically evaluate Parameter-Efficient Fine-Tuning (PEFT) methods under the paradigm of Reinforcement Learning with Verifiable Rewards (RLVR). RLVR incentivizes language models to enhance their reasoning capabilities through verifiable feedback; however, while methods like LoRA are commonly used, the optimal PEFT architecture for RLVR remains unidentified. In this work, we conduct the first comprehensive evaluation of over 12 PEFT methodologies across the DeepSeek-R1-Distill families on mathematical reasoning benchmarks. Our empirical results challenge the default adoption of standard LoRA with three main findings. First, we demonstrate that structural variants, such as DoRA, AdaLoRA, and MiSS, consistently outperform LoRA. Second, we uncover a spectral collapse phenomenon in SVD-informed initialization strategies (\textit{e.g.,} PiSSA, MiLoRA), attributing their failure to a fundamental misalignment between principal-component updates and RL optimization. Furthermore, our ablations reveal that extreme parameter reduction (\textit{e.g.,} VeRA, Rank-1) severely bottlenecks reasoning capacity. We further conduct ablation studies and scaling experiments to validate our findings. This work provides a definitive guide for advocating for more exploration for parameter-efficient RL methods. |
| 2025-12-29 | [Multi-Agent Framework for Threat Mitigation and Resilience in AI-Based Systems](http://arxiv.org/abs/2512.23132v1) | Armstrong Foundjem, Lionel Nganyewou Tidjon et al. | Machine learning (ML) underpins foundation models in finance, healthcare, and critical infrastructure, making them targets for data poisoning, model extraction, prompt injection, automated jailbreaking, and preference-guided black-box attacks that exploit model comparisons. Larger models can be more vulnerable to introspection-driven jailbreaks and cross-modal manipulation. Traditional cybersecurity lacks ML-specific threat modeling for foundation, multimodal, and RAG systems. Objective: Characterize ML security risks by identifying dominant TTPs, vulnerabilities, and targeted lifecycle stages. Methods: We extract 93 threats from MITRE ATLAS (26), AI Incident Database (12), and literature (55), and analyze 854 GitHub/Python repositories. A multi-agent RAG system (ChatGPT-4o, temp 0.4) mines 300+ articles to build an ontology-driven threat graph linking TTPs, vulnerabilities, and stages. Results: We identify unreported threats including commercial LLM API model stealing, parameter memorization leakage, and preference-guided text-only jailbreaks. Dominant TTPs include MASTERKEY-style jailbreaking, federated poisoning, diffusion backdoors, and preference optimization leakage, mainly impacting pre-training and inference. Graph analysis reveals dense vulnerability clusters in libraries with poor patch propagation. Conclusion: Adaptive, ML-specific security frameworks, combining dependency hygiene, threat intelligence, and monitoring, are essential to mitigate supply-chain and inference risks across the ML lifecycle. |
| 2025-12-29 | [It's a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents](http://arxiv.org/abs/2512.23128v1) | Karolina Korgul, Yushi Yang et al. | Web-based agents powered by large language models are increasingly used for tasks such as email management or professional networking. Their reliance on dynamic web content, however, makes them vulnerable to prompt injection attacks: adversarial instructions hidden in interface elements that persuade the agent to divert from its original task. We introduce the Task-Redirecting Agent Persuasion Benchmark (TRAP), an evaluation for studying how persuasion techniques misguide autonomous web agents on realistic tasks. Across six frontier models, agents are susceptible to prompt injection in 25\% of tasks on average (13\% for GPT-5 to 43\% for DeepSeek-R1), with small interface or contextual changes often doubling success rates and revealing systemic, psychologically driven vulnerabilities in web-based agents. We also provide a modular social-engineering injection framework with controlled experiments on high-fidelity website clones, allowing for further benchmark expansion. |

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



