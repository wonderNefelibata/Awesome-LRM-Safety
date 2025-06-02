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
| 2025-05-30 | [Open CaptchaWorld: A Comprehensive Web-based Platform for Testing and Benchmarking Multimodal LLM Agents](http://arxiv.org/abs/2505.24878v1) | Yaxin Luo, Zhaoyi Li et al. | CAPTCHAs have been a critical bottleneck for deploying web agents in real-world applications, often blocking them from completing end-to-end automation tasks. While modern multimodal LLM agents have demonstrated impressive performance in static perception tasks, their ability to handle interactive, multi-step reasoning challenges like CAPTCHAs is largely untested. To address this gap, we introduce Open CaptchaWorld, the first web-based benchmark and platform specifically designed to evaluate the visual reasoning and interaction capabilities of MLLM-powered agents through diverse and dynamic CAPTCHA puzzles. Our benchmark spans 20 modern CAPTCHA types, totaling 225 CAPTCHAs, annotated with a new metric we propose: CAPTCHA Reasoning Depth, which quantifies the number of cognitive and motor steps required to solve each puzzle. Experimental results show that humans consistently achieve near-perfect scores, state-of-the-art MLLM agents struggle significantly, with success rates at most 40.0% by Browser-Use Openai-o3, far below human-level performance, 93.3%. This highlights Open CaptchaWorld as a vital benchmark for diagnosing the limits of current multimodal agents and guiding the development of more robust multimodal reasoning systems. Code and Data are available at this https URL. |
| 2025-05-30 | [Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning](http://arxiv.org/abs/2505.24850v1) | Shuyao Xu, Cheng Peng et al. | Recent advances in model distillation demonstrate that data from advanced reasoning models (e.g., DeepSeek-R1, OpenAI's o1) can effectively transfer complex reasoning abilities to smaller, efficient student models. However, standard practices employ rejection sampling, discarding incorrect reasoning examples -- valuable, yet often underutilized data. This paper addresses the critical question: How can both positive and negative distilled reasoning traces be effectively leveraged to maximize LLM reasoning performance in an offline setting? To this end, We propose Reinforcement Distillation (REDI), a two-stage framework. Stage 1 learns from positive traces via Supervised Fine-Tuning (SFT). Stage 2 further refines the model using both positive and negative traces through our proposed REDI objective. This novel objective is a simple, reference-free loss function that outperforms established methods like DPO and SimPO in this distillation context. Our empirical evaluations demonstrate REDI's superiority over baseline Rejection Sampling SFT or SFT combined with DPO/SimPO on mathematical reasoning tasks. Notably, the Qwen-REDI-1.5B model, post-trained on just 131k positive and negative examples from the open Open-R1 dataset, achieves an 83.1% score on MATH-500 (pass@1). Its performance matches or surpasses that of DeepSeek-R1-Distill-Qwen-1.5B (a model post-trained on 800k proprietary data) across various mathematical reasoning benchmarks, establishing a new state-of-the-art for 1.5B models post-trained offline with openly available data. |
| 2025-05-30 | [RealDrive: Retrieval-Augmented Driving with Diffusion Models](http://arxiv.org/abs/2505.24808v1) | Wenhao Ding, Sushant Veer et al. | Learning-based planners generate natural human-like driving behaviors by learning to reason about nuanced interactions from data, overcoming the rigid behaviors that arise from rule-based planners. Nonetheless, data-driven approaches often struggle with rare, safety-critical scenarios and offer limited controllability over the generated trajectories. To address these challenges, we propose RealDrive, a Retrieval-Augmented Generation (RAG) framework that initializes a diffusion-based planning policy by retrieving the most relevant expert demonstrations from the training dataset. By interpolating between current observations and retrieved examples through a denoising process, our approach enables fine-grained control and safe behavior across diverse scenarios, leveraging the strong prior provided by the retrieved scenario. Another key insight we produce is that a task-relevant retrieval model trained with planning-based objectives results in superior planning performance in our framework compared to a task-agnostic retriever. Experimental results demonstrate improved generalization to long-tail events and enhanced trajectory diversity compared to standard learning-based planners -- we observe a 40% reduction in collision rate on the Waymo Open Motion dataset with RAG. |
| 2025-05-30 | [DiG-Net: Enhancing Quality of Life through Hyper-Range Dynamic Gesture Recognition in Assistive Robotics](http://arxiv.org/abs/2505.24786v1) | Eran Bamani Beeri, Eden Nissinman et al. | Dynamic hand gestures play a pivotal role in assistive human-robot interaction (HRI), facilitating intuitive, non-verbal communication, particularly for individuals with mobility constraints or those operating robots remotely. Current gesture recognition methods are mostly limited to short-range interactions, reducing their utility in scenarios demanding robust assistive communication from afar. In this paper, we introduce a novel approach designed specifically for assistive robotics, enabling dynamic gesture recognition at extended distances of up to 30 meters, thereby significantly improving accessibility and quality of life. Our proposed Distance-aware Gesture Network (DiG-Net) effectively combines Depth-Conditioned Deformable Alignment (DADA) blocks with Spatio-Temporal Graph modules, enabling robust processing and classification of gesture sequences captured under challenging conditions, including significant physical attenuation, reduced resolution, and dynamic gesture variations commonly experienced in real-world assistive environments. We further introduce the Radiometric Spatio-Temporal Depth Attenuation Loss (RSTDAL), shown to enhance learning and strengthen model robustness across varying distances. Our model demonstrates significant performance improvement over state-of-the-art gesture recognition frameworks, achieving a recognition accuracy of 97.3% on a diverse dataset with challenging hyper-range gestures. By effectively interpreting gestures from considerable distances, DiG-Net significantly enhances the usability of assistive robots in home healthcare, industrial safety, and remote assistance scenarios, enabling seamless and intuitive interactions for users regardless of physical limitations |
| 2025-05-30 | [Neural Network-based Universal Formulas for Control](http://arxiv.org/abs/2505.24744v1) | Pol Mestres, Jorge Cortés et al. | We study the problem of designing a controller that satisfies an arbitrary number of affine inequalities at every point in the state space. This is motivated by the use of guardrails in autonomous systems. Indeed, a variety of key control objectives, such as stability, safety, and input saturation, are guaranteed by closed-loop systems whose controllers satisfy such inequalities. Many works in the literature design such controllers as the solution to a state-dependent quadratic program (QP) whose constraints are precisely the inequalities. When the input dimension and number of constraints are high, computing a solution of this QP in real time can become computationally burdensome. Additionally, the solution of such optimization problems is not smooth in general, which can degrade the performance of the system. This paper provides a novel method to design a smooth controller that satisfies an arbitrary number of affine constraints. This why we refer to it as a universal formula for control. The controller is given at every state as the minimizer of a strictly convex function. To avoid computing the minimizer of such function in real time, we introduce a method based on neural networks (NN) to approximate the controller. Remarkably, this NN can be used to solve the controller design problem for any task with less than a fixed input dimension and number of affine constraints, and is completely independent of the state dimension. Additionally, we show that the NN-based controller only needs to be trained with datapoints from a compact set in the state space, which significantly simplifies the training process. Various simulations showcase the performance of the proposed solution. |
| 2025-05-30 | [Multi-Domain ABSA Conversation Dataset Generation via LLMs for Real-World Evaluation and Model Comparison](http://arxiv.org/abs/2505.24701v1) | Tejul Pandit, Meet Raval et al. | Aspect-Based Sentiment Analysis (ABSA) offers granular insights into opinions but often suffers from the scarcity of diverse, labeled datasets that reflect real-world conversational nuances. This paper presents an approach for generating synthetic ABSA data using Large Language Models (LLMs) to address this gap. We detail the generation process aimed at producing data with consistent topic and sentiment distributions across multiple domains using GPT-4o. The quality and utility of the generated data were evaluated by assessing the performance of three state-of-the-art LLMs (Gemini 1.5 Pro, Claude 3.5 Sonnet, and DeepSeek-R1) on topic and sentiment classification tasks. Our results demonstrate the effectiveness of the synthetic data, revealing distinct performance trade-offs among the models: DeepSeekR1 showed higher precision, Gemini 1.5 Pro and Claude 3.5 Sonnet exhibited strong recall, and Gemini 1.5 Pro offered significantly faster inference. We conclude that LLM-based synthetic data generation is a viable and flexible method for creating valuable ABSA resources, facilitating research and model evaluation without reliance on limited or inaccessible real-world labeled data. |
| 2025-05-30 | [TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis](http://arxiv.org/abs/2505.24672v1) | Xiaorui Wu, Xiaofeng Mao et al. | Large Language Models (LLMs) excel in various natural language processing tasks but remain vulnerable to generating harmful content or being exploited for malicious purposes. Although safety alignment datasets have been introduced to mitigate such risks through supervised fine-tuning (SFT), these datasets often lack comprehensive risk coverage. Most existing datasets focus primarily on lexical diversity while neglecting other critical dimensions. To address this limitation, we propose a novel analysis framework to systematically measure the risk coverage of alignment datasets across three essential dimensions: Lexical Diversity, Malicious Intent, and Jailbreak Tactics. We further introduce TRIDENT, an automated pipeline that leverages persona-based, zero-shot LLM generation to produce diverse and comprehensive instructions spanning these dimensions. Each harmful instruction is paired with an ethically aligned response, resulting in two datasets: TRIDENT-Core, comprising 26,311 examples, and TRIDENT-Edge, with 18,773 examples. Fine-tuning Llama 3.1-8B on TRIDENT-Edge demonstrates substantial improvements, achieving an average 14.29% reduction in Harm Score, and a 20% decrease in Attack Success Rate compared to the best-performing baseline model fine-tuned on the WildBreak dataset. |
| 2025-05-30 | [Benchmarking Large Language Models for Cryptanalysis and Mismatched-Generalization](http://arxiv.org/abs/2505.24621v1) | Utsav Maskey, Chencheng Zhu et al. | Recent advancements in Large Language Models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis a critical area for data security and encryption has not yet been thoroughly explored in LLM evaluations. To address this gap, we evaluate cryptanalytic potential of state of the art LLMs on encrypted texts generated using a range of cryptographic algorithms. We introduce a novel benchmark dataset comprising diverse plain texts spanning various domains, lengths, writing styles, and topics paired with their encrypted versions. Using zero-shot and few shot settings, we assess multiple LLMs for decryption accuracy and semantic comprehension across different encryption schemes. Our findings reveal key insights into the strengths and limitations of LLMs in side-channel communication while raising concerns about their susceptibility to jailbreaking attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security. |
| 2025-05-30 | [AMIA: Automatic Masking and Joint Intention Analysis Makes LVLMs Robust Jailbreak Defenders](http://arxiv.org/abs/2505.24519v1) | Yuqi Zhang, Yuchun Miao et al. | We introduce AMIA, a lightweight, inference-only defense for Large Vision-Language Models (LVLMs) that (1) Automatically Masks a small set of text-irrelevant image patches to disrupt adversarial perturbations, and (2) conducts joint Intention Analysis to uncover and mitigate hidden harmful intents before response generation. Without any retraining, AMIA improves defense success rates across diverse LVLMs and jailbreak benchmarks from an average of 52.4% to 81.7%, preserves general utility with only a 2% average accuracy drop, and incurs only modest inference overhead. Ablation confirms both masking and intention analysis are essential for a robust safety-utility trade-off. |
| 2025-05-30 | [Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting](http://arxiv.org/abs/2505.24511v1) | Jiahao Wang, Mingyue Cheng et al. | Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks. |
| 2025-05-30 | [How can AI reduce wrist injuries in the workplace?](http://arxiv.org/abs/2505.24510v1) | Roberto F. Pitzalis, Nicholas Cartocci et al. | This paper explores the development of a control and sensor strategy for an industrial wearable wrist exoskeleton by classifying and predicting workers' actions. The study evaluates the correlation between exerted force and effort intensity, along with sensor strategy optimization, for designing purposes. Using data from six healthy subjects in a manufacturing plant, this paper presents EMG-based models for wrist motion classification and force prediction. Wrist motion recognition is achieved through a pattern recognition algorithm developed with surface EMG data from an 8-channel EMG sensor (Myo Armband); while a force regression model uses wrist and hand force measurements from a commercial handheld dynamometer (Vernier GoDirect Hand Dynamometer). This control strategy forms the foundation for a streamlined exoskeleton architecture designed for industrial applications, focusing on simplicity, reduced costs, and minimal sensor use while ensuring reliable and effective assistance. |
| 2025-05-30 | [How can AI reduce fall injuries in the workplace?](http://arxiv.org/abs/2505.24507v1) | Nicholas Cartocci, Antonios E. Gkikakis et al. | Fall-caused injuries are common in all types of work environments, including offices. They are the main cause of absences longer than three days, especially for small and medium-sized businesses (SMEs). However, data, data amount, data heterogeneity, and stringent processing time constraints continue to pose challenges to real-time fall detection. This work proposes a new approach based on a recurrent neural network (RNN) for Fall Detection and a Kolmogorov-Arnold Network (KAN) to estimate the time of impact of the fall. The approach is tested on SisFall, a dataset consisting of 2706 Activities of Daily Living (ADLs) and 1798 falls recorded by three sensors. The results show that the proposed approach achieves an average TPR of 82.6% and TNR of 98.4% for fall sequences and 94.4% in ADL. Besides, the Root Mean Squared Error of the estimated time of impact is approximately 160ms. |
| 2025-05-30 | [TimeHC-RL: Temporal-aware Hierarchical Cognitive Reinforcement Learning for Enhancing LLMs' Social Intelligence](http://arxiv.org/abs/2505.24500v1) | Guiyang Hou, Xing Gao et al. | Recently, Large Language Models (LLMs) have made significant progress in IQ-related domains that require careful thinking, such as mathematics and coding. However, enhancing LLMs' cognitive development in social domains, particularly from a post-training perspective, remains underexplored. Recognizing that the social world follows a distinct timeline and requires a richer blend of cognitive modes (from intuitive reactions (System 1) and surface-level thinking to deliberate thinking (System 2)) than mathematics, which primarily relies on System 2 cognition (careful, step-by-step reasoning), we introduce Temporal-aware Hierarchical Cognitive Reinforcement Learning (TimeHC-RL) for enhancing LLMs' social intelligence. In our experiments, we systematically explore improving LLMs' social intelligence and validate the effectiveness of the TimeHC-RL method, through five other post-training paradigms and two test-time intervention paradigms on eight datasets with diverse data patterns. Experimental results reveal the superiority of our proposed TimeHC-RL method compared to the widely adopted System 2 RL method. It gives the 7B backbone model wings, enabling it to rival the performance of advanced models like DeepSeek-R1 and OpenAI-O3. Additionally, the systematic exploration from post-training and test-time interventions perspectives to improve LLMs' social intelligence has uncovered several valuable insights. |
| 2025-05-30 | [Real-time Fall Prevention system for the Next-generation of Workers](http://arxiv.org/abs/2505.24487v1) | Nicholas Cartocci, Antonios E. Gkikakis et al. | Developing a general-purpose wearable real-time fall-detection system is still a challenging task, especially for healthy and strong subjects, such as industrial workers that work in harsh environments. In this work, we present a hybrid approach for fall detection and prevention, which uses the dynamic model of an inverted pendulum to generate simulations of falling that are then fed to a deep learning framework. The output is a signal to activate a fall mitigation mechanism when the subject is at risk of harm. The advantage of this approach is that abstracted models can be used to efficiently generate training data for thousands of different subjects with different falling initial conditions, something that is practically impossible with real experiments. This approach is suitable for a specific type of fall, where the subjects fall without changing their initial configuration significantly, and it is the first step toward a general-purpose wearable device, with the aim of reducing fall-associated injuries in industrial environments, which can improve the safety of workers. |
| 2025-05-30 | [Evaluating Gemini in an arena for learning](http://arxiv.org/abs/2505.24477v1) | LearnLM Team, Abhinit Modi et al. | Artificial intelligence (AI) is poised to transform education, but the research community lacks a robust, general benchmark to evaluate AI models for learning. To assess state-of-the-art support for educational use cases, we ran an "arena for learning" where educators and pedagogy experts conduct blind, head-to-head, multi-turn comparisons of leading AI models. In particular, $N = 189$ educators drew from their experience to role-play realistic learning use cases, interacting with two models sequentially, after which $N = 206$ experts judged which model better supported the user's learning goals. The arena evaluated a slate of state-of-the-art models: Gemini 2.5 Pro, Claude 3.7 Sonnet, GPT-4o, and OpenAI o3. Excluding ties, experts preferred Gemini 2.5 Pro in 73.2% of these match-ups -- ranking it first overall in the arena. Gemini 2.5 Pro also demonstrated markedly higher performance across key principles of good pedagogy. Altogether, these results position Gemini 2.5 Pro as a leading model for learning. |
| 2025-05-30 | [Learning Safety Constraints for Large Language Models](http://arxiv.org/abs/2505.24445v1) | Xin Chen, Yarden As et al. | Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space. |
| 2025-05-30 | [Model Unlearning via Sparse Autoencoder Subspace Guided Projections](http://arxiv.org/abs/2505.24428v1) | Xu Wang, Zihao Li et al. | Large language models (LLMs) store vast amounts of information, making them powerful yet raising privacy and safety concerns when selective knowledge removal is required. Existing unlearning strategies, ranging from gradient-based fine-tuning and model editing to sparse autoencoder (SAE) steering, either lack interpretability or fail to provide a robust defense against adversarial prompts. We propose SAE-Guided Subspace Projection Unlearning (SSPU), a novel framework that leverages SAE features to drive targeted updates in the model's parameter space, enabling precise, interpretable, and robust unlearning. SSPU's three-stage pipeline performs data-driven layer and feature selection, subspace construction via QR decomposition, and constrained optimization that controls activations into an "irrelevant" subspace while preserving retained knowledge. Overall, we use SAE features to construct a subspace that supervises unlearning, refining the loss and adding a regularization term to guide interpretable parameter updates. In experiments on the WMDP-Cyber forget set and three utility benchmarks (MMLU, TruthfulQA, GSM8K), SSPU reduces harmful knowledge accuracy by 3.22% compared to the strongest baseline. It also improves adversarial robustness, lowering malicious accuracy under jailbreak prompts compared to baselines. Our findings expose the limitations of prior unlearning methods and demonstrate how interpretable subspace-guided optimization can achieve robust, controllable model behavior. |
| 2025-05-30 | [Pangu DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](http://arxiv.org/abs/2505.24332v1) | Wenxuan Shi, Haochen Tan et al. | Information seeking demands iterative evidence gathering and reflective reasoning, yet large language models (LLMs) still struggle with it in open-web question answering. Existing methods rely on static prompting rules or training with Wikipedia-based corpora and retrieval environments, limiting adaptability to the real-world web environment where ambiguity, conflicting evidence, and noise are prevalent. These constrained training settings hinder LLMs from learning to dynamically decide when and where to search, and how to adjust search depth and frequency based on informational demands. We define this missing capacity as Search Intensity Scaling (SIS)--the emergent skill to intensify search efforts under ambiguous or conflicting conditions, rather than settling on overconfident, under-verification answers.   To study SIS, we introduce WebPuzzle, the first dataset designed to foster information-seeking behavior in open-world internet environments. WebPuzzle consists of 24K training instances and 275 test questions spanning both wiki-based and open-web queries. Building on this dataset, we propose DeepDiver, a Reinforcement Learning (RL) framework that promotes SIS by encouraging adaptive search policies through exploration under a real-world open-web environment. Experimental results show that Pangu-7B-Reasoner empowered by DeepDiver achieve performance on real-web tasks comparable to the 671B-parameter DeepSeek-R1. We detail DeepDiver's training curriculum from cold-start supervised fine-tuning to a carefully designed RL phase, and present that its capability of SIS generalizes from closed-form QA to open-ended tasks such as long-form writing. Our contributions advance adaptive information seeking in LLMs and provide a valuable benchmark and dataset for future research. |
| 2025-05-30 | [EgoExOR: An Ego-Exo-Centric Operating Room Dataset for Surgical Activity Understanding](http://arxiv.org/abs/2505.24287v1) | Ege Özsoy, Arda Mamur et al. | Operating rooms (ORs) demand precise coordination among surgeons, nurses, and equipment in a fast-paced, occlusion-heavy environment, necessitating advanced perception models to enhance safety and efficiency. Existing datasets either provide partial egocentric views or sparse exocentric multi-view context, but do not explore the comprehensive combination of both. We introduce EgoExOR, the first OR dataset and accompanying benchmark to fuse first-person and third-person perspectives. Spanning 94 minutes (84,553 frames at 15 FPS) of two emulated spine procedures, Ultrasound-Guided Needle Insertion and Minimally Invasive Spine Surgery, EgoExOR integrates egocentric data (RGB, gaze, hand tracking, audio) from wearable glasses, exocentric RGB and depth from RGB-D cameras, and ultrasound imagery. Its detailed scene graph annotations, covering 36 entities and 22 relations (568,235 triplets), enable robust modeling of clinical interactions, supporting tasks like action recognition and human-centric perception. We evaluate the surgical scene graph generation performance of two adapted state-of-the-art models and offer a new baseline that explicitly leverages EgoExOR's multimodal and multi-perspective signals. This new dataset and benchmark set a new foundation for OR perception, offering a rich, multimodal resource for next-generation clinical perception. |
| 2025-05-30 | [FABLE: A Novel Data-Flow Analysis Benchmark on Procedural Text for Large Language Model Evaluation](http://arxiv.org/abs/2505.24258v1) | Vishal Pallagani, Nitin Gupta et al. | Understanding how data moves, transforms, and persists, known as data flow, is fundamental to reasoning in procedural tasks. Despite their fluency in natural and programming languages, large language models (LLMs), although increasingly being applied to decisions with procedural tasks, have not been systematically evaluated for their ability to perform data-flow reasoning. We introduce FABLE, an extensible benchmark designed to assess LLMs' understanding of data flow using structured, procedural text. FABLE adapts eight classical data-flow analyses from software engineering: reaching definitions, very busy expressions, available expressions, live variable analysis, interval analysis, type-state analysis, taint analysis, and concurrency analysis. These analyses are instantiated across three real-world domains: cooking recipes, travel routes, and automated plans. The benchmark includes 2,400 question-answer pairs, with 100 examples for each domain-analysis combination. We evaluate three types of LLMs: a reasoning-focused model (DeepSeek-R1 8B), a general-purpose model (LLaMA 3.1 8B), and a code-specific model (Granite Code 8B). Each model is tested using majority voting over five sampled completions per prompt. Results show that the reasoning model achieves higher accuracy, but at the cost of over 20 times slower inference compared to the other models. In contrast, the general-purpose and code-specific models perform close to random chance. FABLE provides the first diagnostic benchmark to systematically evaluate data-flow reasoning and offers insights for developing models with stronger procedural understanding. |

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



