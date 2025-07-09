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
| 2025-07-08 | [UQLM: A Python Package for Uncertainty Quantification in Large Language Models](http://arxiv.org/abs/2507.06196v1) | Dylan Bouchard, Mohit Singh Chauhan et al. | Hallucinations, defined as instances where Large Language Models (LLMs) generate false or misleading content, pose a significant challenge that impacts the safety and trust of downstream applications. We introduce UQLM, a Python package for LLM hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers that compute response-level confidence scores ranging from 0 to 1. This library provides an off-the-shelf solution for UQ-based hallucination detection that can be easily integrated to enhance the reliability of LLM outputs. |
| 2025-07-08 | [Evaluation of Habitat Robotics using Large Language Models](http://arxiv.org/abs/2507.06157v1) | William Li, Lei Hamilton et al. | This paper focuses on evaluating the effectiveness of Large Language Models at solving embodied robotic tasks using the Meta PARTNER benchmark. Meta PARTNR provides simplified environments and robotic interactions within randomized indoor kitchen scenes. Each randomized kitchen scene is given a task where two robotic agents cooperatively work together to solve the task. We evaluated multiple frontier models on Meta PARTNER environments. Our results indicate that reasoning models like OpenAI o3-mini outperform non-reasoning models like OpenAI GPT-4o and Llama 3 when operating in PARTNR's robotic embodied environments. o3-mini displayed outperform across centralized, decentralized, full observability, and partial observability configurations. This provides a promising avenue of research for embodied robotic development. |
| 2025-07-08 | [OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](http://arxiv.org/abs/2507.06134v1) | Sanidhya Vijayvargiya, Aditya Bharat Soni et al. | Recent advances in AI agents capable of solving complex, everyday tasks, from scheduling to customer service, have enabled deployment in real-world settings, but their possibilities for unsafe behavior demands rigorous evaluation. While prior benchmarks have attempted to assess agent safety, most fall short by relying on simulated environments, narrow task domains, or unrealistic tool abstractions. We introduce OpenAgentSafety, a comprehensive and modular framework for evaluating agent behavior across eight critical risk categories. Unlike prior work, our framework evaluates agents that interact with real tools, including web browsers, code execution environments, file systems, bash shells, and messaging platforms; and supports over 350 multi-turn, multi-user tasks spanning both benign and adversarial user intents. OpenAgentSafety is designed for extensibility, allowing researchers to add tools, tasks, websites, and adversarial strategies with minimal effort. It combines rule-based analysis with LLM-as-judge assessments to detect both overt and subtle unsafe behaviors. Empirical analysis of five prominent LLMs in agentic scenarios reveals unsafe behavior in 51.2% of safety-vulnerable tasks with Claude-Sonnet-3.7, to 72.7% with o3-mini, highlighting critical safety vulnerabilities and the need for stronger safeguards before real-world deployment. |
| 2025-07-08 | [Safe Domain Randomization via Uncertainty-Aware Out-of-Distribution Detection and Policy Adaptation](http://arxiv.org/abs/2507.06111v1) | Mohamad H. Danesh, Maxime Wabartha et al. | Deploying reinforcement learning (RL) policies in real-world involves significant challenges, including distribution shifts, safety concerns, and the impracticality of direct interactions during policy refinement. Existing methods, such as domain randomization (DR) and off-dynamics RL, enhance policy robustness by direct interaction with the target domain, an inherently unsafe practice. We propose Uncertainty-Aware RL (UARL), a novel framework that prioritizes safety during training by addressing Out-Of-Distribution (OOD) detection and policy adaptation without requiring direct interactions in target domain. UARL employs an ensemble of critics to quantify policy uncertainty and incorporates progressive environmental randomization to prepare the policy for diverse real-world conditions. By iteratively refining over high-uncertainty regions of the state space in simulated environments, UARL enhances robust generalization to the target domain without explicitly training on it. We evaluate UARL on MuJoCo benchmarks and a quadrupedal robot, demonstrating its effectiveness in reliable OOD detection, improved performance, and enhanced sample efficiency compared to baselines. |
| 2025-07-08 | [SCCRUB: Surface Cleaning Compliant Robot Utilizing Bristles](http://arxiv.org/abs/2507.06053v1) | Jakub F. Kowalewski, Keeyon Hajjafar et al. | Scrubbing surfaces is a physically demanding and time-intensive task. Removing adhered contamination requires substantial friction generated through pressure and torque or high lateral forces. Rigid robotic manipulators, while capable of exerting these forces, are usually confined to structured environments isolated from humans due to safety risks. In contrast, soft robot arms can safely work around humans and adapt to environmental uncertainty, but typically struggle to transmit the continuous torques or lateral forces necessary for scrubbing. Here, we demonstrate a soft robotic arm scrubbing adhered residues using torque and pressure, a task traditionally challenging for soft robots. We train a neural network to learn the arm's inverse kinematics and elasticity, which enables open-loop force and position control. Using this learned model, the robot successfully scrubbed burnt food residue from a plate and sticky fruit preserve from a toilet seat, removing an average of 99.7% of contamination. This work demonstrates how soft robots, capable of exerting continuous torque, can effectively and safely scrub challenging contamination from surfaces. |
| 2025-07-08 | [CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations](http://arxiv.org/abs/2507.06043v1) | Xiaohu Li, Yunfeng Ning et al. | Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN. |
| 2025-07-08 | [RabakBench: Scaling Human Annotations to Construct Localized Multilingual Safety Benchmarks for Low-Resource Languages](http://arxiv.org/abs/2507.05980v1) | Gabriel Chua, Leanne Tan et al. | Large language models (LLMs) and their safety classifiers often perform poorly on low-resource languages due to limited training data and evaluation benchmarks. This paper introduces RabakBench, a new multilingual safety benchmark localized to Singapore's unique linguistic context, covering Singlish, Chinese, Malay, and Tamil. RabakBench is constructed through a scalable three-stage pipeline: (i) Generate - adversarial example generation by augmenting real Singlish web content with LLM-driven red teaming; (ii) Label - semi-automated multi-label safety annotation using majority-voted LLM labelers aligned with human judgments; and (iii) Translate - high-fidelity translation preserving linguistic nuance and toxicity across languages. The final dataset comprises over 5,000 safety-labeled examples across four languages and six fine-grained safety categories with severity levels. Evaluations of 11 popular open-source and closed-source guardrail classifiers reveal significant performance degradation. RabakBench not only enables robust safety evaluation in Southeast Asian multilingual settings but also offers a reproducible framework for building localized safety datasets in low-resource environments. The benchmark dataset, including the human-verified translations, and evaluation code are publicly available. |
| 2025-07-08 | [High-Resolution Visual Reasoning via Multi-Turn Grounding-Based Reinforcement Learning](http://arxiv.org/abs/2507.05920v1) | Xinyu Huang, Yuhao Dong et al. | State-of-the-art large multi-modal models (LMMs) face challenges when processing high-resolution images, as these inputs are converted into enormous visual tokens, many of which are irrelevant to the downstream task. In this paper, we propose Multi-turn Grounding-based Policy Optimization (MGPO), an end-to-end reinforcement learning (RL) framework that enables LMMs to iteratively focus on key visual regions by automatically cropping sub-images, based on model-predicted grounding coordinates within a multi-turn conversation framework. Compared to supervised fine-tuning (SFT), which requires costly additional grounding annotations, our approach highlights that LMMs can emerge robust grounding abilities during the RL training process, leveraging only a binary reward function derived from the correctness of the final answer. Additionally, we observe that LMMs struggle to autonomously trigger visual grounding during the rollout process. To address this cold start problem, we design a multi-turn conversational template and restrict policy loss computation to model outputs generated across multiple dialogue rounds, thereby promoting stable optimization. Extensive experiments demonstrate that, when trained on standard visual-question-short answering data without grounding annotations, MGPO effectively elicits stronger grounding capabilities compared to GRPO, leading to 5.4\% improvement on in-distribution MME-Realworld and 5.2\% improvement on the challenging out-of-distribution (OOD) V* Bench. Notably, MGPO post-training on Qwen2.5-VL-7B with 21K samples surpasses OpenAI's o1 and GPT-4o models on the OOD V* Bench. Codes are available at https://github.com/EvolvingLMMs-Lab/MGPO. |
| 2025-07-08 | [Towards Solar Altitude Guided Scene Illumination](http://arxiv.org/abs/2507.05812v1) | Samed Doğan, Maximilian Hoh et al. | The development of safe and robust autonomous driving functions is heavily dependent on large-scale, high-quality sensor data. However, real-word data acquisition demands intensive human labor and is strongly limited by factors such as labeling cost, driver safety protocols and diverse scenario coverage. Thus, multiple lines of work focus on the conditional generation of synthetic camera sensor data. We identify a significant gap in research regarding daytime variation, presumably caused by the scarcity of available labels. Consequently, we present the solar altitude as global conditioning variable. It is readily computable from latitude-longitude coordinates and local time, eliminating the need for extensive manual labeling. Our work is complemented by a tailored normalization approach, targeting the sensitivity of daylight towards small numeric changes in altitude. We demonstrate its ability to accurately capture lighting characteristics and illumination-dependent image noise in the context of diffusion models. |
| 2025-07-08 | [Unraveling $K(1690)$ as a pseudoscalar $ud\bar{d}\bar{s}$ tetraquark state](http://arxiv.org/abs/2507.05726v1) | Jin-Peng Zhang, Xu-Liang Chen et al. | The recent observed $K (1690)$ has been identified as a supernumerary pseudoscalar resonance signal in the strange-meson spectrum predicted by quark model calculations. It is the best candidate of a strange crypto-exotic state. In this work, we systematically study the hadron masses of $ud\bar{d}\bar{s}$ tetraquark states with $J^P = 0^-$ in the method of QCD sum rules (QCDSR). For ten interpolating currents, we calculate the correlation functions up to dimension-8 nonperturbative condensates. To calculate the tri-gluon condensate, we comprehensively consider the contributions from different operators with and without covariant derivatives. The infrared (IR) safety can be guaranteed for the completely calculated tri-gluon condensate by properly addressing the IR divergences in Feynman diagrams. It is demonstrated that the tri-gluon condensate provides significant contributions to the sum-rule analyses in these light tetraquark systems. Our results support the interpretation of $K (1690)$ resonance to be a pseudoscalar $ud\bar{d}\bar{s}$ tetraquark state. |
| 2025-07-08 | [Divergent Realities: A Comparative Analysis of Human Expert vs. Artificial Intelligence Based Generation and Evaluation of Treatment Plans in Dermatology](http://arxiv.org/abs/2507.05716v1) | Dipayan Sengupta, Saumya Panda | Background: Evaluating AI-generated treatment plans is a key challenge as AI expands beyond diagnostics, especially with new reasoning models. This study compares plans from human experts and two AI models (a generalist and a reasoner), assessed by both human peers and a superior AI judge.   Methods: Ten dermatologists, a generalist AI (GPT-4o), and a reasoning AI (o3) generated treatment plans for five complex dermatology cases. The anonymized, normalized plans were scored in two phases: 1) by the ten human experts, and 2) by a superior AI judge (Gemini 2.5 Pro) using an identical rubric.   Results: A profound 'evaluator effect' was observed. Human experts scored peer-generated plans significantly higher than AI plans (mean 7.62 vs. 7.16; p=0.0313), ranking GPT-4o 6th (mean 7.38) and the reasoning model, o3, 11th (mean 6.97). Conversely, the AI judge produced a complete inversion, scoring AI plans significantly higher than human plans (mean 7.75 vs. 6.79; p=0.0313). It ranked o3 1st (mean 8.20) and GPT-4o 2nd, placing all human experts lower.   Conclusions: The perceived quality of a clinical plan is fundamentally dependent on the evaluator's nature. An advanced reasoning AI, ranked poorly by human experts, was judged as superior by a sophisticated AI, revealing a deep gap between experience-based clinical heuristics and data-driven algorithmic logic. This paradox presents a critical challenge for AI integration, suggesting the future requires synergistic, explainable human-AI systems that bridge this reasoning gap to augment clinical care. |
| 2025-07-08 | [DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving](http://arxiv.org/abs/2507.05710v1) | Hyeongchan Ham, Heejin Ahn | Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence. |
| 2025-07-08 | [Noninvasive Focused Ultrasound Spinal Cord Stimulation in Humans: A Computational Feasibility Study](http://arxiv.org/abs/2507.05702v1) | Koharu Isomura, Wenwei Yu et al. | Background: Trans-spinal FUS (tsFUS) has recently been shown promise in modulating spinal reflexes in rodents, opening new avenues for spinal cord interventions in motor control and pain management. However, anatomical differences between rodents and human spinal cords require careful targeting strategies and transducer design adaptations for human applications. Aim: This study aims to computationally explore the feasibility of tsFUS in the human spinal cord by leveraging the intervertebral acoustic window and vertebral lamina. Method: Acoustic simulations were performed using an anatomically detailed human spinal cord model with an adapted single-element focusing transducer (SEFT) to investigate the focality and intensity of the acoustic quantities generated within the spinal cord. Results: Sonication through the intervertebral acoustic window using an adapted transducer achieved approximately 2-fold higher intensity and up to 20% greater beam overlap compared to commercial SEFT. Precise transducer positioning was critical; a 10 mm vertical shift resulted in a reduction of target intensity by approximately 7-fold. The vertebral level also substantially influenced the sonication outcomes, with the thoracic spine achieving 6-fold higher intensity than the cervical level. Sonication through the vertebral lamina resulted in approximately 2.5-fold higher intraspinal intensity in the cervical spine. Conclusion and Significance: This study presents the first systematic, anatomically realistic computational model for the feasibility of tsFUS in humans. Quantifying the trade-offs between acoustic path, vertebral level, and transducer geometry offers foundational design and procedural guidelines for the design and safety of spinal FUS protocols, supporting future studies of trans-spinal FUS in humans. |
| 2025-07-08 | [AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs](http://arxiv.org/abs/2507.05687v1) | Shangzhan Li, Zefan Wang et al. | Kernel development in deep learning requires optimizing computational units across hardware while balancing memory management, parallelism, and hardware-specific optimizations through extensive empirical tuning. Although domain-specific languages like Triton simplify GPU programming by abstracting low-level details, developers must still manually tune critical parameters such as tile sizes and memory access patterns through iterative experimentation, creating substantial barriers to optimal performance and wider adoption. In this work, we introduce AutoTriton, the first model dedicated to Triton programming powered by reinforcement learning (RL). AutoTriton performs supervised fine-tuning (SFT) to be equipped with essential Triton programming expertise using a high-quality data gathering pipeline, and conducts RL with Group Relative Policy Optimization (GRPO) algorithm, combining a rule-based reward and an execution-based reward to further improve Triton programming ability, sequentially. Experiments across five evaluation channels of TritonBench and KernelBench illustrate that our 8B model AutoTriton achieves performance comparable to mainstream large models, including Claude-4-Sonnet and DeepSeek-R1-0528. Further experimental analysis demonstrates the crucial role of each module within AutoTriton, including the SFT stage, the RL stage, and the reward design strategy. These findings underscore the promise of RL for automatically generating high-performance kernels, and since high-performance kernels are core components of AI systems, this breakthrough establishes an important foundation for building more efficient AI systems. The model and code will be available at https://github.com/AI9Stars/AutoTriton. |
| 2025-07-08 | [TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data](http://arxiv.org/abs/2507.05660v1) | Aravind Cheruvu, Shravya Kanchi et al. | Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL). |
| 2025-07-08 | [Detecting and Mitigating Reward Hacking in Reinforcement Learning Systems: A Comprehensive Empirical Study](http://arxiv.org/abs/2507.05619v1) | Ibne Farabi Shihab, Sanjeda Akter et al. | Reward hacking in Reinforcement Learning (RL) systems poses a critical threat to the deployment of autonomous agents, where agents exploit flaws in reward functions to achieve high scores without fulfilling intended objectives. Despite growing awareness of this problem, systematic detection and mitigation approaches remain limited. This paper presents a large-scale empirical study of reward hacking across diverse RL environments and algorithms. We analyze 15,247 training episodes across 15 RL environments (Atari, MuJoCo, custom domains) and 5 algorithms (PPO, SAC, DQN, A3C, Rainbow), implementing automated detection algorithms for six categories of reward hacking: specification gaming, reward tampering, proxy optimization, objective misalignment, exploitation patterns, and wireheading. Our detection framework achieves 78.4% precision and 81.7% recall across environments, with computational overhead under 5%. Through controlled experiments varying reward function properties, we demonstrate that reward density and alignment with true objectives significantly impact hacking frequency ($p < 0.001$, Cohen's $d = 1.24$). We validate our approach through three simulated application studies representing recommendation systems, competitive gaming, and robotic control scenarios. Our mitigation techniques reduce hacking frequency by up to 54.6% in controlled scenarios, though we find these trade-offs are more challenging in practice due to concept drift, false positive costs, and adversarial adaptation. All detection algorithms, datasets, and experimental protocols are publicly available to support reproducible research in RL safety. |
| 2025-07-08 | [Domain adaptation of large language models for geotechnical applications](http://arxiv.org/abs/2507.05613v1) | Lei Fan, Fangxue Liu et al. | Recent developments in large language models (LLMs) are opening up new opportunities in geotechnical engineering and engineering geology. While general-purpose LLMs possess broad capabilities, effective application in geotechnics often requires domain-specific adaptation. Such tailored LLMs are increasingly employed to streamline geotechnical workflows. This paper presents the first survey of the adaptation and application of LLMs in geotechnical engineering. It outlines key methodologies for adaptation to geotechnical domain, including prompt engineering, retrieval-augmented generation, domain-adaptive pretraining, and fine-tuning. The survey examines the state-of-the-art applications of geotechnical-adapted LLMs, including geological interpretation, subsurface characterization, site planning, design calculations, numerical modeling, safety and risk assessment, and educational tutoring. It also analyzes benefits and limitations of geotechnical-adapted LLMs, and identifies promising directions for future research in this interdisciplinary discipline. The findings serve as a valuable resource for practitioners seeking to integrate LLMs into geotechnical practice, while also providing a foundation to stimulate further investigation within the academic community. |
| 2025-07-08 | [Structured Task Solving via Modular Embodied Intelligence: A Case Study on Rubik's Cube](http://arxiv.org/abs/2507.05607v1) | Chongshan Fan, Shenghai Yuan | This paper presents Auto-RubikAI, a modular autonomous planning framework that integrates a symbolic Knowledge Base (KB), a vision-language model (VLM), and a large language model (LLM) to solve structured manipulation tasks exemplified by Rubik's Cube restoration. Unlike traditional robot systems based on predefined scripts, or modern approaches relying on pretrained networks and large-scale demonstration data, Auto-RubikAI enables interpretable, multi-step task execution with minimal data requirements and no prior demonstrations. The proposed system employs a KB module to solve group-theoretic restoration steps, overcoming LLMs' limitations in symbolic reasoning. A VLM parses RGB-D input to construct a semantic 3D scene representation, while the LLM generates structured robotic control code via prompt chaining. This tri-module architecture enables robust performance under spatial uncertainty. We deploy Auto-RubikAI in both simulation and real-world settings using a 7-DOF robotic arm, demonstrating effective Sim-to-Real adaptation without retraining. Experiments show a 79% end-to-end task success rate across randomized configurations. Compared to CFOP, DeepCubeA, and Two-Phase baselines, our KB-enhanced method reduces average solution steps while maintaining interpretability and safety. Auto-RubikAI provides a cost-efficient, modular foundation for embodied task planning in smart manufacturing, robotics education, and autonomous execution scenarios. Code, prompts, and hardware modules will be released upon publication. |
| 2025-07-08 | [Semi-Supervised Defect Detection via Conditional Diffusion and CLIP-Guided Noise Filtering](http://arxiv.org/abs/2507.05588v1) | Shuai Li, Shihan Chen et al. | In the realm of industrial quality inspection, defect detection stands as a critical component, particularly in high-precision, safety-critical sectors such as automotive components aerospace, and medical devices. Traditional methods, reliant on manual inspection or early image processing algorithms, suffer from inefficiencies, high costs, and limited robustness. This paper introduces a semi-supervised defect detection framework based on conditional diffusion (DSYM), leveraging a two-stage collaborative training mechanism and a staged joint optimization strategy. The framework utilizes labeled data for initial training and subsequently incorporates unlabeled data through the generation of pseudo-labels. A conditional diffusion model synthesizes multi-scale pseudo-defect samples, while a CLIP cross-modal feature-based noise filtering mechanism mitigates label contamination. Experimental results on the NEU-DET dataset demonstrate a 78.4% mAP@0.5 with the same amount of labeled data as traditional supervised methods, and 75.1% mAP@0.5 with only 40% of the labeled data required by the original supervised model, showcasing significant advantages in data efficiency. This research provides a high-precision, low-labeling-dependent solution for defect detection in industrial quality inspection scenarios. The work of this article has been open-sourced at https://github.com/cLin-c/Semisupervised-DSYM. |
| 2025-07-08 | [Towards Measurement Theory for Artificial Intelligence](http://arxiv.org/abs/2507.05587v1) | Elija Perrier | We motivate and outline a programme for a formal theory of measurement of artificial intelligence. We argue that formalising measurement for AI will allow researchers, practitioners, and regulators to: (i) make comparisons between systems and the evaluation methods applied to them; (ii) connect frontier AI evaluations with established quantitative risk analysis techniques drawn from engineering and safety science; and (iii) foreground how what counts as AI capability is contingent upon the measurement operations and scales we elect to use. We sketch a layered measurement stack, distinguish direct from indirect observables, and signpost how these ingredients provide a pathway toward a unified, calibratable taxonomy of AI phenomena. |

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



