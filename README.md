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
| 2025-12-02 | [The Moral Consistency Pipeline: Continuous Ethical Evaluation for Large Language Models](http://arxiv.org/abs/2512.03026v1) | Saeid Jamshidi, Kawser Wazed Nafi et al. | The rapid advancement and adaptability of Large Language Models (LLMs) highlight the need for moral consistency, the capacity to maintain ethically coherent reasoning across varied contexts. Existing alignment frameworks, structured approaches designed to align model behavior with human ethical and social norms, often rely on static datasets and post-hoc evaluations, offering limited insight into how ethical reasoning may evolve across different contexts or temporal scales. This study presents the Moral Consistency Pipeline (MoCoP), a dataset-free, closed-loop framework for continuously evaluating and interpreting the moral stability of LLMs. MoCoP combines three supporting layers: (i) lexical integrity analysis, (ii) semantic risk estimation, and (iii) reasoning-based judgment modeling within a self-sustaining architecture that autonomously generates, evaluates, and refines ethical scenarios without external supervision. Our empirical results on GPT-4-Turbo and DeepSeek suggest that MoCoP effectively captures longitudinal ethical behavior, revealing a strong inverse relationship between ethical and toxicity dimensions (correlation rET = -0.81, p value less than 0.001) and a near-zero association with response latency (correlation rEL approximately equal to 0). These findings demonstrate that moral coherence and linguistic safety tend to emerge as stable and interpretable characteristics of model behavior rather than short-term fluctuations. Furthermore, by reframing ethical evaluation as a dynamic, model-agnostic form of moral introspection, MoCoP offers a reproducible foundation for scalable, continuous auditing and advances the study of computational morality in autonomous AI systems. |
| 2025-12-02 | [Invasive Context Engineering to Control Large Language Models](http://arxiv.org/abs/2512.03001v1) | Thomas Rivasseau | Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations. |
| 2025-12-02 | [Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities](http://arxiv.org/abs/2512.02973v1) | Yuan Xiong, Ziqi Miao et al. | While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs. |
| 2025-12-02 | [Lumos: Let there be Language Model System Certification](http://arxiv.org/abs/2512.02966v1) | Isha Chaudhary, Vedaant Jain et al. | We introduce the first principled framework, Lumos, for specifying and formally certifying Language Model System (LMS) behaviors. Lumos is an imperative probabilistic programming DSL over graphs, with constructs to generate independent and identically distributed prompts for LMS. It offers a structured view of prompt distributions via graphs, forming random prompts from sampled subgraphs. Lumos supports certifying LMS for arbitrary prompt distributions via integration with statistical certifiers. We provide hybrid (operational and denotational) semantics for Lumos, providing a rigorous way to interpret the specifications. Using only a small set of composable constructs, Lumos can encode existing LMS specifications, including complex relational and temporal specifications. It also facilitates specifying new properties - we present the first safety specifications for vision-language models (VLMs) in autonomous driving scenarios developed with Lumos. Using these, we show that the state-of-the-art VLM Qwen-VL exhibits critical safety failures, producing incorrect and unsafe responses with at least 90% probability in right-turn scenarios under rainy driving conditions, revealing substantial safety risks. Lumos's modular structure allows easy modification of the specifications, enabling LMS certification to stay abreast with the rapidly evolving threat landscape. We further demonstrate that specification programs written in Lumos enable finding specific failure cases exhibited by state-of-the-art LMS. Lumos is the first systematic and extensible language-based framework for specifying and certifying LMS behaviors, paving the way for a wider adoption of LMS certification. |
| 2025-12-02 | [A Lightweight Real-Time Low-Light Enhancement Network for Embedded Automotive Vision Systems](http://arxiv.org/abs/2512.02965v1) | Yuhan Chen, Yicui Shi et al. | In low-light environments like nighttime driving, image degradation severely challenges in-vehicle camera safety. Since existing enhancement algorithms are often too computationally intensive for vehicular applications, we propose UltraFast-LieNET, a lightweight multi-scale shifted convolutional network for real-time low-light image enhancement. We introduce a Dynamic Shifted Convolution (DSConv) kernel with only 12 learnable parameters for efficient feature extraction. By integrating DSConv with varying shift distances, a Multi-scale Shifted Residual Block (MSRB) is constructed to significantly expand the receptive field. To mitigate lightweight network instability, a residual structure and a novel multi-level gradient-aware loss function are incorporated. UltraFast-LieNET allows flexible parameter configuration, with a minimum size of only 36 parameters. Results on the LOLI-Street dataset show a PSNR of 26.51 dB, outperforming state-of-the-art methods by 4.6 dB while utilizing only 180 parameters. Experiments across four benchmark datasets validate its superior balance of real-time performance and enhancement quality under limited resources. Code is available at https://githubhttps://github.com/YuhanChen2024/UltraFast-LiNET |
| 2025-12-02 | [Belobog: Move Language Fuzzing Framework For Real-World Smart Contracts](http://arxiv.org/abs/2512.02918v1) | Wanxu Xia, Ziqiao Kong et al. | Move is a research-oriented programming language design for secure and verifiable smart contract development and has been widely used in managing billions of digital assets in blockchains, such as Sui and Aptos. Move features a strong static type system and explicit resource semantics to enforce safety properties such as the prevention of data races, invalid asset transfers, and entry vulnerabilities. However, smart contracts written in Move may still contain certain vulnerabilities that are beyond the reach of its type system. It is thus essential to validate Move smart contracts. Unfortunately, due to its strong type system, existing smart contract fuzzers are ineffective in producing syntactically or semantically valid transactions to test Move smart contracts. This paper introduces the first fuzzing framework, Belobog, for Move smart contracts. Belobog is type-aware and ensures that all generated and mutated transactions are well-typed. More specifically, for a target Move smart contract, Belobog first constructs a type graph based on Move's type system, and then generates or mutates a transaction based on the graph trace derived from the type graph. In order to overcome the complex checks in Move smart contracts, we further design and implement a concolic executor in Belobog. We evaluated Belobog on 109 real-world Move smart contract projects. The experimental results show that Belobog is able to detect 100\% critical and 79\% major vulnerabilities manually audited by human experts. We further selected two recent notorious incidents in Move smart contracts, i.e., Cetus and Nemo. Belobog successfully reproduced full exploits for both of them, without any prior knowledge. |
| 2025-12-02 | [Statistical-Symbolic Verification of Perception-Based Autonomous Systems using State-Dependent Conformal Prediction](http://arxiv.org/abs/2512.02893v1) | Yuang Geng, Thomas Waite et al. | Reachability analysis has been a prominent way to provide safety guarantees for neurally controlled autonomous systems, but its direct application to neural perception components is infeasible due to imperfect or intractable perception models. Typically, this issue has been bypassed by complementing reachability with statistical analysis of perception error, say with conformal prediction (CP). However, existing CP methods for time-series data often provide conservative bounds. The corresponding error accumulation over time has made it challenging to combine statistical bounds with symbolic reachability in a way that is provable, scalable, and minimally conservative. To reduce conservatism and improve scalability, our key insight is that perception error varies significantly with the system's dynamical state. This article proposes state-dependent conformal prediction, which exploits that dependency in constructing tight high-confidence bounds on perception error. Based on this idea, we provide an approach to partition the state space, using a genetic algorithm, so as to optimize the tightness of conformal bounds. Finally, since using these bounds in reachability analysis leads to additional uncertainty and branching in the resulting hybrid system, we propose a branch-merging reachability algorithm that trades off uncertainty for scalability so as to enable scalable and tight verification. The evaluation of our verification methodology on two complementary case studies demonstrates reduced conservatism compared to the state of the art. |
| 2025-12-02 | [SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots](http://arxiv.org/abs/2512.02851v1) | Iana Zhura, Sausar Karaf et al. | Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100\% navigation success and 0.09 s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation. |
| 2025-12-02 | [VLM as Strategist: Adaptive Generation of Safety-critical Testing Scenarios via Guided Diffusion](http://arxiv.org/abs/2512.02844v1) | Xinzheng Wu, Junyi Chen et al. | The safe deployment of autonomous driving systems (ADSs) relies on comprehensive testing and evaluation. However, safety-critical scenarios that can effectively expose system vulnerabilities are extremely sparse in the real world. Existing scenario generation methods face challenges in efficiently constructing long-tail scenarios that ensure fidelity, criticality, and interactivity, while particularly lacking real-time dynamic response capabilities to the vehicle under test (VUT). To address these challenges, this paper proposes a safety-critical testing scenario generation framework that integrates the high-level semantic understanding capabilities of Vision Language Models (VLMs) with the fine-grained generation capabilities of adaptive guided diffusion models. The framework establishes a three-layer hierarchical architecture comprising a strategic layer for VLM-directed scenario generation objective determination, a tactical layer for guidance function formulation, and an operational layer for guided diffusion execution. We first establish a high-quality fundamental diffusion model that learns the data distribution of real driving scenarios. Next, we design an adaptive guided diffusion method that enables real-time, precise control of background vehicles (BVs) in closed-loop simulation. The VLM is then incorporated to autonomously generate scenario generation objectives and guidance functions through deep scenario understanding and risk reasoning, ultimately guiding the diffusion model to achieve VLM-directed scenario generation. Experimental results demonstrate that the proposed method can efficiently generate realistic, diverse, and highly interactive safety-critical testing scenarios. Furthermore, case studies validate the adaptability and VLM-directed generation performance of the proposed method. |
| 2025-12-02 | [A benchmark dataset for evaluating Syndrome Differentiation and Treatment in large language models](http://arxiv.org/abs/2512.02816v1) | Kunning Li, Jianbin Guo et al. | The emergence of Large Language Models (LLMs) within the Traditional Chinese Medicine (TCM) domain presents an urgent need to assess their clinical application capabilities. However, such evaluations are challenged by the individualized, holistic, and diverse nature of TCM's "Syndrome Differentiation and Treatment" (SDT). Existing benchmarks are confined to knowledge-based question-answering or the accuracy of syndrome differentiation, often neglecting assessment of treatment decision-making. Here, we propose a comprehensive, clinical case-based benchmark spearheaded by TCM experts, and a specialized reward model employed to quantify prescription-syndrome congruence. Data annotation follows a rigorous pipeline. This benchmark, designated TCM-BEST4SDT, encompasses four tasks, including TCM Basic Knowledge, Medical Ethics, LLM Content Safety, and SDT. The evaluation framework integrates three mechanisms, namely selected-response evaluation, judge model evaluation, and reward model evaluation. The effectiveness of TCM-BEST4SDT was corroborated through experiments on 15 mainstream LLMs, spanning both general and TCM domains. To foster the development of intelligent TCM research, TCM-BEST4SDT is now publicly available. |
| 2025-12-02 | [CogDrive: Cognition-Driven Multimodal Prediction-Planning Fusion for Safe Autonomy](http://arxiv.org/abs/2512.02777v1) | Heye Huang, Yibin Yang et al. | Safe autonomous driving in mixed traffic requires a unified understanding of multimodal interactions and dynamic planning under uncertainty. Existing learning based approaches struggle to capture rare but safety critical behaviors, while rule based systems often lack adaptability in complex interactions. To address these limitations, CogDrive introduces a cognition driven multimodal prediction and planning framework that integrates explicit modal reasoning with safety aware trajectory optimization. The prediction module adopts cognitive representations of interaction modes based on topological motion semantics and nearest neighbor relational encoding. With a differentiable modal loss and multimodal Gaussian decoding, CogDrive learns sparse and unbalanced interaction behaviors and improves long horizon trajectory prediction. The planning module incorporates an emergency response concept and optimizes safety stabilized trajectories, where short term consistent branches ensure safety during replanning cycles and long term branches support smooth and collision free motion under low probability switching modes. Experiments on Argoverse2 and INTERACTION datasets show that CogDrive achieves strong performance in trajectory accuracy and miss rate, while closed loop simulations confirm adaptive behavior in merge and intersection scenarios. By combining cognitive multimodal prediction with safety oriented planning, CogDrive offers an interpretable and reliable paradigm for safe autonomy in complex traffic. |
| 2025-12-02 | [CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer](http://arxiv.org/abs/2512.02711v1) | Lavish Bansal, Naman Mishra | Ensuring content safety in large language models (LLMs) is essential for their deployment in real-world applications. However, existing safety guardrails are predominantly tailored for high-resource languages, leaving a significant portion of the world's population underrepresented who communicate in low-resource languages. To address this, we introduce CREST (CRoss-lingual Efficient Safety Transfer), a parameter-efficient multilingual safety classification model that supports 100 languages with only 0.5B parameters. By training on a strategically chosen subset of only 13 high-resource languages, our model utilizes cluster-based cross-lingual transfer from a few to 100 languages, enabling effective generalization to both unseen high-resource and low-resource languages. This approach addresses the challenge of limited training data in low-resource settings. We conduct comprehensive evaluations across six safety benchmarks to demonstrate that CREST outperforms existing state-of-the-art guardrails of comparable scale and achieves competitive results against models with significantly larger parameter counts (2.5B parameters and above). Our findings highlight the limitations of language-specific guardrails and underscore the importance of developing universal, language-agnostic safety systems that can scale effectively to serve global populations. |
| 2025-12-02 | [Beyond Single-Agent Safety: A Taxonomy of Risks in LLM-to-LLM Interactions](http://arxiv.org/abs/2512.02682v1) | Piercosma Bisconti, Marcello Galisai et al. | This paper examines why safety mechanisms designed for human-model interaction do not scale to environments where large language models (LLMs) interact with each other. Most current governance practices still rely on single-agent safety containment, prompts, fine-tuning, and moderation layers that constrain individual model behavior but leave the dynamics of multi-model interaction ungoverned. These mechanisms assume a dyadic setting: one model responding to one user under stable oversight. Yet research and industrial development are rapidly shifting toward LLM-to-LLM ecosystems, where outputs are recursively reused as inputs across chains of agents. In such systems, local compliance can aggregate into collective failure even when every model is individually aligned. We propose a conceptual transition from model-level safety to system-level safety, introducing the framework of the Emergent Systemic Risk Horizon (ESRH) to formalize how instability arises from interaction structure rather than from isolated misbehavior. The paper contributes (i) a theoretical account of collective risk in interacting LLMs, (ii) a taxonomy connecting micro, meso, and macro-level failure modes, and (iii) a design proposal for InstitutionalAI, an architecture for embedding adaptive oversight within multi-agent systems. |
| 2025-12-02 | [Rural Connectivity Inequalities in Finland and Sweden: Evidence, Measures, and Policy Reflections](http://arxiv.org/abs/2512.02649v1) | Sameera Bandaranayake, Amirreza Moradi et al. | Persistent rural-urban disparities in broadband connectivity remain a major policy challenge, even in digitally advanced countries. This paper examines how these inequalities manifest in northern Finland and Sweden, where sparse populations, long distances, and seasonal variations in demand create persistent gaps in service quality and reliability. Drawing on survey data (n = 148), in-depth interviews, and spatial analysis, the study explores the lived experience of connectivity in Arctic rural communities and introduces a novel Cellular Coverage Inequality (CCI) Index. The index combines measures of rurality and network performance to quantify spatial disparities that are masked by national coverage statistics. Results reveal that headline indicators overstate inclusiveness, while local users report chronic connectivity gaps affecting work, safety, and access to services. Building on these findings, the paper outlines policy reflections in six areas: shared infrastructure and roaming frameworks, spectrum flexibility for rural operators, performance-based Quality-of-Service monitoring, standardized and transparent reporting, temporal and seasonal capacity management, and digital-skills initiatives. Together, these recommendations highlight the need for multidimensional metrics and governance mechanisms that link technical performance, spatial equity, and user experience. The analysis contributes to ongoing debates on how broadband policy in sparsely populated regions can move beyond nominal coverage targets toward genuine inclusion and reliability. |
| 2025-12-02 | [RULER-Bench: Probing Rule-based Reasoning Abilities of Next-level Video Generation Models for Vision Foundation Intelligence](http://arxiv.org/abs/2512.02622v1) | Xuming He, Zehao Fan et al. | Recent advances in video generation have enabled the synthesis of videos with strong temporal consistency and impressive visual quality, marking a crucial step toward vision foundation models. To evaluate these video generation models, existing benchmarks primarily focus on factors related to visual perception and understanding, like visual aesthetics, instruction adherence, and temporal coherence. However, the rule-based reasoning capabilities of video generation models remain largely unexplored. Although recent studies have carried out preliminary explorations into whether video models can serve as zero-shot learners, they still lack a fine-grained decomposition of reasoning capabilities and a comprehensive evaluation protocol. To address this gap, we introduce RULER-Bench, a benchmark designed to evaluate the reasoning ability of video generation models from the perspective of cognitive rules. Built upon two fundamental paradigms: text-to-video and image-to-video, RULER-Bench covers 40 representative tasks spanning six rule categories with 622 high-quality annotated instances. For the evaluation of each generated video, we construct a checklist covering four metrics and leverage GPT-o3 to assign scores to each question, achieving 85% alignment with human judgements. Extensive experiments show that the state-of-the-art model achieves only 48.87% on the rule coherence metric, highlighting significant room for improvement in the reasoning capability of next-level video models. We expect that the insight obtained from RULER-Bench will facilitate further development of reasoning-aware video generation, advancing video generation models toward vision foundation intelligence. |
| 2025-12-02 | [Spoken Conversational Agents with Large Language Models](http://arxiv.org/abs/2512.02593v1) | Chao-Han Huck Yang, Andreas Stolcke et al. | Spoken conversational agents are converging toward voice-native LLMs. This tutorial distills the path from cascaded ASR/NLU to end-to-end, retrieval-and vision-grounded systems. We frame adaptation of text LLMs to audio, cross-modal alignment, and joint speech-text training; review datasets, metrics, and robustness across accents and compare design choices (cascaded vs. E2E, post-ASR correction, streaming). We link industrial assistants to current open-domain and task-oriented agents, highlight reproducible baselines, and outline open problems in privacy, safety, and evaluation. Attendees leave with practical recipes and a clear systems-level roadmap. |
| 2025-12-02 | [Reframing Human-Robot Interaction Through Extended Reality: Unlocking Safer, Smarter, and More Empathic Interactions with Virtual Robots and Foundation Models](http://arxiv.org/abs/2512.02569v1) | Yuchong Zhang, Yong Ma et al. | This perspective reframes human-robot interaction (HRI) through extended reality (XR), arguing that virtual robots powered by large foundation models (FMs) can serve as cognitively grounded, empathic agents. Unlike physical robots, XR-native agents are unbound by hardware constraints and can be instantiated, adapted, and scaled on demand, while still affording embodiment and co-presence. We synthesize work across XR, HRI, and cognitive AI to show how such agents can support safety-critical scenarios, socially and cognitively empathic interaction across domains, and outreaching physical capabilities with XR and AI integration. We then discuss how multimodal large FMs (e.g., large language model, large vision model, and vision-language model) enable context-aware reasoning, affect-sensitive situations, and long-term adaptation, positioning virtual robots as cognitive and empathic mediators rather than mere simulation assets. At the same time, we highlight challenges and potential risks, including overtrust, cultural and representational bias, privacy concerns around biometric sensing, and data governance and transparency. The paper concludes by outlining a research agenda for human-centered, ethically grounded XR agents - emphasizing multi-layered evaluation frameworks, multi-user ecosystems, mixed virtual-physical embodiment, and societal and ethical design practices to envision XR-based virtual agents powered by FMs as reshaping future HRI into a more efficient and adaptive paradigm. |
| 2025-12-02 | [Feedback Loops and Code Perturbations in LLM-based Software Engineering: A Case Study on a C-to-Rust Translation System](http://arxiv.org/abs/2512.02567v1) | Martin Weiss, Jesko Hecking-Harbusch et al. | The advent of strong generative AI has a considerable impact on various software engineering tasks such as code repair, test generation, or language translation. While tools like GitHub Copilot are already in widespread use in interactive settings, automated approaches require a higher level of reliability before being usable in industrial practice. In this paper, we focus on three aspects that directly influence the quality of the results: a) the effect of automated feedback loops, b) the choice of Large Language Model (LLM), and c) the influence of behavior-preserving code changes.   We study the effect of these three variables on an automated C-to-Rust translation system. Code translation from C to Rust is an attractive use case in industry due to Rust's safety guarantees. The translation system is based on a generate-and-check pattern, in which Rust code generated by the LLM is automatically checked for compilability and behavioral equivalence with the original C code. For negative checking results, the LLM is re-prompted in a feedback loop to repair its output. These checks also allow us to evaluate and compare the respective success rates of the translation system when varying the three variables.   Our results show that without feedback loops LLM selection has a large effect on translation success. However, when the translation system uses feedback loops the differences across models diminish. We observe this for the average performance of the system as well as its robustness under code perturbations. Finally, we also identify that diversity provided by code perturbations can even result in improved system performance. |
| 2025-12-02 | [Intervention Strategies for Fairness and Efficiency at Autonomous Single-Intersection Traffic Flows](http://arxiv.org/abs/2512.02562v1) | Salman Ghori, Ania Adil et al. | Intersections present significant challenges in traffic management, where ensuring safety and efficiency is essential for effective flow. However, these goals are often achieved at the expense of fairness, which is critical for trustworthiness and long-term sustainability. This paper investigates how the timing of centralized intervention affects the management of autonomous agents at a signal-less, orthogonal intersection, while satisfying safety constraints, evaluating efficiency, and ensuring fairness. A mixed-integer linear programming (MILP) approach is used to optimize agent coordination within a circular control zone centered at the intersection. We introduce the concept of fairness, measured via pairwise reversal counts, and incorporate fairness constraints into the MILP framework. We then study the relationship between fairness and system efficiency and its impact on platoon formation. Finally, simulation studies analyze the effectiveness of early versus late intervention strategies and fairness-aware control, focusing on safe, efficient, and robust management of agents within the control zone. |
| 2025-12-02 | [Aetheria: A multimodal interpretable content safety framework based on multi-agent debate and collaboration](http://arxiv.org/abs/2512.02530v1) | Yuxiang He, Jian Zhao et al. | The exponential growth of digital content presents significant challenges for content safety. Current moderation systems, often based on single models or fixed pipelines, exhibit limitations in identifying implicit risks and providing interpretable judgment processes. To address these issues, we propose Aetheria, a multimodal interpretable content safety framework based on multi-agent debate and collaboration.Employing a collaborative architecture of five core agents, Aetheria conducts in-depth analysis and adjudication of multimodal content through a dynamic, mutually persuasive debate mechanism, which is grounded by RAG-based knowledge retrieval.Comprehensive experiments on our proposed benchmark (AIR-Bench) validate that Aetheria not only generates detailed and traceable audit reports but also demonstrates significant advantages over baselines in overall content safety accuracy, especially in the identification of implicit risks. This framework establishes a transparent and interpretable paradigm, significantly advancing the field of trustworthy AI content moderation. |

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



