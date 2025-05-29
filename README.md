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
| 2025-05-28 | [Sherlock: Self-Correcting Reasoning in Vision-Language Models](http://arxiv.org/abs/2505.22651v1) | Yi Ding, Ruqi Zhang | Reasoning Vision-Language Models (VLMs) have shown promising performance on complex multimodal tasks. However, they still face significant challenges: they are highly sensitive to reasoning errors, require large volumes of annotated data or accurate verifiers, and struggle to generalize beyond specific domains. To address these limitations, we explore self-correction as a strategy to enhance reasoning VLMs. We first conduct an in-depth analysis of reasoning VLMs' self-correction abilities and identify key gaps. Based on our findings, we introduce Sherlock, a self-correction and self-improvement training framework. Sherlock introduces a trajectory-level self-correction objective, a preference data construction method based on visual perturbation, and a dynamic $\beta$ for preference tuning. Once the model acquires self-correction capabilities using only 20k randomly sampled annotated data, it continues to self-improve without external supervision. Built on the Llama3.2-Vision-11B model, Sherlock achieves remarkable results across eight benchmarks, reaching an average accuracy of 64.1 with direct generation and 65.4 after self-correction. It outperforms LLaVA-CoT (63.2), Mulberry (63.9), and LlamaV-o1 (63.4) while using less than 20% of the annotated data. |
| 2025-05-28 | [SimProcess: High Fidelity Simulation of Noisy ICS Physical Processes](http://arxiv.org/abs/2505.22638v1) | Denis Donadel, Gabriele Crestanello et al. | Industrial Control Systems (ICS) manage critical infrastructures like power grids and water treatment plants. Cyberattacks on ICSs can disrupt operations, causing severe economic, environmental, and safety issues. For example, undetected pollution in a water plant can put the lives of thousands at stake. ICS researchers have increasingly turned to honeypots -- decoy systems designed to attract attackers, study their behaviors, and eventually improve defensive mechanisms. However, existing ICS honeypots struggle to replicate the ICS physical process, making them susceptible to detection. Accurately simulating the noise in ICS physical processes is challenging because different factors produce it, including sensor imperfections and external interferences.   In this paper, we propose SimProcess, a novel framework to rank the fidelity of ICS simulations by evaluating how closely they resemble real-world and noisy physical processes. It measures the simulation distance from a target system by estimating the noise distribution with machine learning models like Random Forest. Unlike existing solutions that require detailed mathematical models or are limited to simple systems, SimProcess operates with only a timeseries of measurements from the real system, making it applicable to a broader range of complex dynamic systems. We demonstrate the framework's effectiveness through a case study using real-world power grid data from the EPIC testbed. We compare the performance of various simulation methods, including static and generative noise techniques. Our model correctly classifies real samples with a recall of up to 1.0. It also identifies Gaussian and Gaussian Mixture as the best distribution to simulate our power systems, together with a generative solution provided by an autoencoder, thereby helping developers to improve honeypot fidelity. Additionally, we make our code publicly available. |
| 2025-05-28 | [Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs](http://arxiv.org/abs/2505.22548v1) | Changhao Song, Yazhou Zhang et al. | Emotion understanding includes basic tasks (e.g., sentiment/emotion classification) and advanced tasks (e.g., sarcasm/humor detection). Current methods rely on fixed-length CoT reasoning, failing to adapt to the varying complexity of emotions. We propose a task-adaptive reasoning framework that employs DeepSeek-R1 to generate variable-length reasoning chains for different emotion tasks. By combining fine-tuning with reinforcement learning, we design a composite reward function that balances four objectives: prediction accuracy, adaptive reasoning depth control, structural diversity in reasoning paths, and suppression of repetitive logic. This approach achieves dynamic context-sensitive inference while enabling LLMs to autonomously develop deep reasoning capabilities. Experimental results demonstrate consistent improvements in both Acc and F1 scores across four tasks: emotion, sentiment, humor, and sarcasm. Notably, peak enhancements reached 3.56% F1 (2.76% Acc) for basic tasks and 37.95% F1 (23.14% Acc) for advanced tasks. Our work bridges rigid CoT reasoning and emotional complexity through adaptive-depth analysis. |
| 2025-05-28 | [GeoDrive: 3D Geometry-Informed Driving World Model with Precise Action Control](http://arxiv.org/abs/2505.22421v1) | Anthony Chen, Wenzhao Zheng et al. | Recent advancements in world models have revolutionized dynamic environment simulation, allowing systems to foresee future states and assess potential actions. In autonomous driving, these capabilities help vehicles anticipate the behavior of other road users, perform risk-aware planning, accelerate training in simulation, and adapt to novel scenarios, thereby enhancing safety and reliability. Current approaches exhibit deficiencies in maintaining robust 3D geometric consistency or accumulating artifacts during occlusion handling, both critical for reliable safety assessment in autonomous navigation tasks. To address this, we introduce GeoDrive, which explicitly integrates robust 3D geometry conditions into driving world models to enhance spatial understanding and action controllability. Specifically, we first extract a 3D representation from the input frame and then obtain its 2D rendering based on the user-specified ego-car trajectory. To enable dynamic modeling, we propose a dynamic editing module during training to enhance the renderings by editing the positions of the vehicles. Extensive experiments demonstrate that our method significantly outperforms existing models in both action accuracy and 3D spatial awareness, leading to more realistic, adaptable, and reliable scene modeling for safer autonomous driving. Additionally, our model can generalize to novel trajectories and offers interactive scene editing capabilities, such as object editing and object trajectory control. |
| 2025-05-28 | [Mitigating Overthinking in Large Reasoning Models via Manifold Steering](http://arxiv.org/abs/2505.22411v1) | Yao Huang, Huanran Chen et al. | Recent advances in Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in solving complex tasks such as mathematics and coding. However, these models frequently exhibit a phenomenon known as overthinking during inference, characterized by excessive validation loops and redundant deliberation, leading to substantial computational overheads. In this paper, we aim to mitigate overthinking by investigating the underlying mechanisms from the perspective of mechanistic interpretability. We first showcase that the tendency of overthinking can be effectively captured by a single direction in the model's activation space and the issue can be eased by intervening the activations along this direction. However, this efficacy soon reaches a plateau and even deteriorates as the intervention strength increases. We therefore systematically explore the activation space and find that the overthinking phenomenon is actually tied to a low-dimensional manifold, which indicates that the limited effect stems from the noises introduced by the high-dimensional steering direction. Based on this insight, we propose Manifold Steering, a novel approach that elegantly projects the steering direction onto the low-dimensional activation manifold given the theoretical approximation of the interference noise. Extensive experiments on DeepSeek-R1 distilled models validate that our method reduces output tokens by up to 71% while maintaining and even improving the accuracy on several mathematical benchmarks. Our method also exhibits robust cross-domain transferability, delivering consistent token reduction performance in code generation and knowledge-based QA tasks. Code is available at: https://github.com/Aries-iai/Manifold_Steering. |
| 2025-05-28 | [Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Deployment Settings](http://arxiv.org/abs/2505.22356v1) | Angéline Pouget, Mohammad Yaghini et al. | Deploying machine learning models in safety-critical domains poses a key challenge: ensuring reliable model performance on downstream user data without access to ground truth labels for direct validation. We propose the suitability filter, a novel framework designed to detect performance deterioration by utilizing suitability signals -- model output features that are sensitive to covariate shifts and indicative of potential prediction errors. The suitability filter evaluates whether classifier accuracy on unlabeled user data shows significant degradation compared to the accuracy measured on the labeled test dataset. Specifically, it ensures that this degradation does not exceed a pre-specified margin, which represents the maximum acceptable drop in accuracy. To achieve reliable performance evaluation, we aggregate suitability signals for both test and user data and compare these empirical distributions using statistical hypothesis testing, thus providing insights into decision uncertainty. Our modular method adapts to various models and domains. Empirical evaluations across different classification tasks demonstrate that the suitability filter reliably detects performance deviations due to covariate shift. This enables proactive mitigation of potential failures in high-stakes applications. |
| 2025-05-28 | [Skywork Open Reasoner 1 Technical Report](http://arxiv.org/abs/2505.22312v1) | Jujie He, Jiacai Liu et al. | The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets. |
| 2025-05-28 | [Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](http://arxiv.org/abs/2505.22298v1) | Yifan Lu, Jing Li et al. | Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs. |
| 2025-05-28 | [Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs](http://arxiv.org/abs/2505.22293v1) | Samuel Frontull, Thomas Ströhle | Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation, sometimes even outperforming traditional neural systems. However, previous research has highlighted the challenges of using LLMs, particularly with prompt engineering, for low-resource languages. In this work, we introduce Fragment-Shot Prompting, a novel in-context learning method that segments input and retrieves translation examples based on syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables translation without direct parallel data. We evaluate these methods using GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot Prompting is effective for translating into and between the studied low-resource languages, with syntactic coverage positively correlating with translation quality; (2) Models with stronger reasoning abilities make more effective use of retrieved knowledge, generally produce better translations, and enable Pivoted Fragment-Shot to significantly improve translation quality between the Ladin variants; and (3) prompt engineering offers limited, if any, improvements when translating from a low-resource to a high-resource language, where zero-shot prompting already yields satisfactory results. We publicly release our code and the retrieval corpora. |
| 2025-05-28 | [Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models](http://arxiv.org/abs/2505.22271v1) | Yongcan Yu, Yanbo Wang et al. | While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM. |
| 2025-05-28 | [LiDAR Based Semantic Perception for Forklifts in Outdoor Environments](http://arxiv.org/abs/2505.22258v1) | Benjamin Serfling, Hannes Reichert et al. | In this study, we present a novel LiDAR-based semantic segmentation framework tailored for autonomous forklifts operating in complex outdoor environments. Central to our approach is the integration of a dual LiDAR system, which combines forward-facing and downward-angled LiDAR sensors to enable comprehensive scene understanding, specifically tailored for industrial material handling tasks. The dual configuration improves the detection and segmentation of dynamic and static obstacles with high spatial precision. Using high-resolution 3D point clouds captured from two sensors, our method employs a lightweight yet robust approach that segments the point clouds into safety-critical instance classes such as pedestrians, vehicles, and forklifts, as well as environmental classes such as driveable ground, lanes, and buildings. Experimental validation demonstrates that our approach achieves high segmentation accuracy while satisfying strict runtime requirements, establishing its viability for safety-aware, fully autonomous forklift navigation in dynamic warehouse and yard environments. |
| 2025-05-28 | [BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain](http://arxiv.org/abs/2505.22240v1) | Yunsoo Kim, Yusuf Abdulle et al. | Biomedical reasoning often requires traversing interconnected relationships across entities such as drugs, diseases, and proteins. Despite the increasing prominence of large language models (LLMs), existing benchmarks lack the ability to evaluate multi-hop reasoning in the biomedical domain, particularly for queries involving one-to-many and many-to-many relationships. This gap leaves the critical challenges of biomedical multi-hop reasoning underexplored. To address this, we introduce BioHopR, a novel benchmark designed to evaluate multi-hop, multi-answer reasoning in structured biomedical knowledge graphs. Built from the comprehensive PrimeKG, BioHopR includes 1-hop and 2-hop reasoning tasks that reflect real-world biomedical complexities.   Evaluations of state-of-the-art models reveal that O3-mini, a proprietary reasoning-focused model, achieves 37.93% precision on 1-hop tasks and 14.57% on 2-hop tasks, outperforming proprietary models such as GPT4O and open-source biomedical models including HuatuoGPT-o1-70B and Llama-3.3-70B. However, all models exhibit significant declines in multi-hop performance, underscoring the challenges of resolving implicit reasoning steps in the biomedical domain. By addressing the lack of benchmarks for multi-hop reasoning in biomedical domain, BioHopR sets a new standard for evaluating reasoning capabilities and highlights critical gaps between proprietary and open-source models while paving the way for future advancements in biomedical LLMs. |
| 2025-05-28 | [Optimal kernel regression bounds under energy-bounded noise](http://arxiv.org/abs/2505.22235v1) | Amon Lahr, Johannes Köhler et al. | Non-conservative uncertainty bounds are key for both assessing an estimation algorithm's accuracy and in view of downstream tasks, such as its deployment in safety-critical contexts. In this paper, we derive a tight, non-asymptotic uncertainty bound for kernel-based estimation, which can also handle correlated noise sequences. Its computation relies on a mild norm-boundedness assumption on the unknown function and the noise, returning the worst-case function realization within the hypothesis class at an arbitrary query input location. The value of this function is shown to be given in terms of the posterior mean and covariance of a Gaussian process for an optimal choice of the measurement noise covariance. By rigorously analyzing the proposed approach and comparing it with other results in the literature, we show its effectiveness in returning tight and easy-to-compute bounds for kernel-based estimates. |
| 2025-05-28 | [Thermal Modeling and Optimal Allocation of Avionics Safety-critical Tasks on Heterogeneous MPSoCs](http://arxiv.org/abs/2505.22214v1) | Ondřej Benedikt, Michal Sojka et al. | Multi-Processor Systems-on-Chip (MPSoC) can deliver high performance needed in many industrial domains, including aerospace. However, their high power consumption, combined with avionics safety standards, brings new thermal management challenges. This paper investigates techniques for offline thermal-aware allocation of periodic tasks on heterogeneous MPSoCs running at a fixed clock frequency, as required in avionics. The goal is to find the assignment of tasks to (i) cores and (ii) temporal isolation windows while minimizing the MPSoC temperature. To achieve that, we propose and analyze three power models, and integrate them within several novel optimization approaches based on heuristics, a black-box optimizer, and Integer Linear Programming (ILP). We perform the experimental evaluation on three popular MPSoC platforms (NXP i.MX8QM MEK, NXP i.MX8QM Ixora, NVIDIA TX2) and observe a difference of up to 5.5{\deg}C among the tested methods (corresponding to a 22% reduction w.r.t. the ambient temperature). We also show that our method, integrating the empirical power model with the ILP, outperforms the other methods on all tested platforms. |
| 2025-05-28 | [Pitfalls of Rule- and Model-based Verifiers -- A Case Study on Mathematical Reasoning](http://arxiv.org/abs/2505.22203v1) | Yuzhen Huang, Weihao Zeng et al. | Trustworthy verifiers are essential for the success of reinforcement learning with verifiable reward (RLVR), which is the core methodology behind various large reasoning models such as DeepSeek-R1. In complex domains like mathematical reasoning, rule-based verifiers have been widely adopted in previous works to train strong reasoning models. However, the reliability of these verifiers and their impact on the RL training process remain poorly understood. In this work, we take mathematical reasoning as a case study and conduct a comprehensive analysis of various verifiers in both static evaluation and RL training scenarios. First, we find that current open-source rule-based verifiers often fail to recognize equivalent answers presented in different formats across multiple commonly used mathematical datasets, resulting in non-negligible false negative rates. This limitation adversely affects RL training performance and becomes more pronounced as the policy model gets stronger. Subsequently, we investigate model-based verifiers as a potential solution to address these limitations. While the static evaluation shows that model-based verifiers achieve significantly higher verification accuracy, further analysis and RL training results imply that they are highly susceptible to hacking, where they misclassify certain patterns in responses as correct (i.e., false positives). This vulnerability is exploited during policy model optimization, leading to artificially inflated rewards. Our findings underscore the unique risks inherent to both rule-based and model-based verifiers, aiming to offer valuable insights to develop more robust reward systems in reinforcement learning. |
| 2025-05-28 | [Accountable, Scalable and DoS-resilient Secure Vehicular Communication](http://arxiv.org/abs/2505.22162v1) | Hongyu Jin, Panos Papadimitratos | Paramount to vehicle safety, broadcasted Cooperative Awareness Messages (CAMs) and Decentralized Environmental Notification Messages (DENMs) are pseudonymously authenticated for security and privacy protection, with each node needing to have all incoming messages validated within an expiration deadline. This creates an asymmetry that can be easily exploited by external adversaries to launch a clogging Denial of Service (DoS) attack: each forged VC message forces all neighboring nodes to cryptographically validate it; at increasing rates, easy to generate forged messages gradually exhaust processing resources and severely degrade or deny timely validation of benign CAMs/DENMs. The result can be catastrophic when awareness of neighbor vehicle positions or critical reports are missed. We address this problem making the standardized VC pseudonymous authentication DoS-resilient. We propose efficient cryptographic constructs, which we term message verification facilitators, to prioritize processing resources for verification of potentially valid messages among bogus messages and verify multiple messages based on one signature verification. Any message acceptance is strictly based on public-key based message authentication/verification for accountability, i.e., non-repudiation is not sacrificed, unlike symmetric key based approaches. This further enables drastic misbehavior detection, also exploiting the newly introduced facilitators, based on probabilistic signature verification and cross-checking over multiple facilitators verifying the same message; while maintaining verification latency low even when under attack, trading off modest communication overhead. Our facilitators can also be used for efficient discovery and verification of DENM or any event-driven message, including misbehavior evidence used for our scheme. |
| 2025-05-28 | [Adapting Segment Anything Model for Power Transmission Corridor Hazard Segmentation](http://arxiv.org/abs/2505.22105v1) | Hang Chen, Maoyuan Ye et al. | Power transmission corridor hazard segmentation (PTCHS) aims to separate transmission equipment and surrounding hazards from complex background, conveying great significance to maintaining electric power transmission safety. Recently, the Segment Anything Model (SAM) has emerged as a foundational vision model and pushed the boundaries of segmentation tasks. However, SAM struggles to deal with the target objects in complex transmission corridor scenario, especially those with fine structure. In this paper, we propose ELE-SAM, adapting SAM for the PTCHS task. Technically, we develop a Context-Aware Prompt Adapter to achieve better prompt tokens via incorporating global-local features and focusing more on key regions. Subsequently, to tackle the hazard objects with fine structure in complex background, we design a High-Fidelity Mask Decoder by leveraging multi-granularity mask features and then scaling them to a higher resolution. Moreover, to train ELE-SAM and advance this field, we construct the ELE-40K benchmark, the first large-scale and real-world dataset for PTCHS including 44,094 image-mask pairs. Experimental results for ELE-40K demonstrate the superior performance that ELE-SAM outperforms the baseline model with the average 16.8% mIoU and 20.6% mBIoU performance improvement. Moreover, compared with the state-of-the-art method on HQSeg-44K, the average 2.9% mIoU and 3.8% mBIoU absolute improvements further validate the effectiveness of our method on high-quality generic object segmentation. The source code and dataset are available at https://github.com/Hhaizee/ELE-SAM. |
| 2025-05-28 | [Efficient Dynamic Shielding for Parametric Safety Specifications](http://arxiv.org/abs/2505.22104v1) | Davide Corsi, Kaushik Mallik et al. | Shielding has emerged as a promising approach for ensuring safety of AI-controlled autonomous systems. The algorithmic goal is to compute a shield, which is a runtime safety enforcement tool that needs to monitor and intervene the AI controller's actions if safety could be compromised otherwise. Traditional shields are designed statically for a specific safety requirement. Therefore, if the safety requirement changes at runtime due to changing operating conditions, the shield needs to be recomputed from scratch, causing delays that could be fatal. We introduce dynamic shields for parametric safety specifications, which are succinctly represented sets of all possible safety specifications that may be encountered at runtime. Our dynamic shields are statically designed for a given safety parameter set, and are able to dynamically adapt as the true safety specification (permissible by the parameters) is revealed at runtime. The main algorithmic novelty lies in the dynamic adaptation procedure, which is a simple and fast algorithm that utilizes known features of standard safety shields, like maximal permissiveness. We report experimental results for a robot navigation problem in unknown territories, where the safety specification evolves as new obstacles are discovered at runtime. In our experiments, the dynamic shields took a few minutes for their offline design, and took between a fraction of a second and a few seconds for online adaptation at each step, whereas the brute-force online recomputation approach was up to 5 times slower. |
| 2025-05-28 | [From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving](http://arxiv.org/abs/2505.22067v1) | Xinyu Xia, Xingjun Ma et al. | Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions. |
| 2025-05-28 | [A Comparative Study of Fuzzers and Static Analysis Tools for Finding Memory Unsafety in C and C++](http://arxiv.org/abs/2505.22052v1) | Keno Hassler, Philipp Görz et al. | Even today, over 70% of security vulnerabilities in critical software systems result from memory safety violations. To address this challenge, fuzzing and static analysis are widely used automated methods to discover such vulnerabilities. Fuzzing generates random program inputs to identify faults, while static analysis examines source code to detect potential vulnerabilities. Although these techniques share a common goal, they take fundamentally different approaches and have evolved largely independently.   In this paper, we present an empirical analysis of five static analyzers and 13 fuzzers, applied to over 100 known security vulnerabilities in C/C++ programs. We measure the number of bug reports generated for each vulnerability to evaluate how the approaches differ and complement each other. Moreover, we randomly sample eight bug-containing functions, manually analyze all bug reports therein, and quantify false-positive rates. We also assess limits to bug discovery, ease of use, resource requirements, and integration into the development process. We find that both techniques discover different types of bugs, but there are clear winners for each. Developers should consider these tools depending on their specific workflow and usability requirements. Based on our findings, we propose future directions to foster collaboration between these research domains. |

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



