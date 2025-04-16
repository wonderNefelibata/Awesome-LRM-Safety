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
| 2025-04-15 | [A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce](http://arxiv.org/abs/2504.11343v1) | Wei Xiong, Jiarui Yao et al. | Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training. |
| 2025-04-15 | [Robust MPC for Uncertain Linear Systems -- Combining Model Adaptation and Iterative Learning](http://arxiv.org/abs/2504.11261v1) | Hannes Petrenz, Johannes Köhler et al. | This paper presents a robust adaptive learning Model Predictive Control (MPC) framework for linear systems with parametric uncertainties and additive disturbances performing iterative tasks. The approach iteratively refines the parameter estimates using set membership estimation. Performance enhancement over iterations is achieved by learning the terminal cost from data. Safety is enforced using a terminal set, which is also learned iteratively. The proposed method guarantees recursive feasibility, constraint satisfaction, and a robust bound on the closed-loop cost. Numerical simulations on a mass-spring-damper system demonstrate improved computational efficiency and control performance compared to an existing robust adaptive MPC approach. |
| 2025-04-15 | [Towards Automated Safety Requirements Derivation Using Agent-based RAG](http://arxiv.org/abs/2504.11243v1) | Balahari Vignesh Balu, Florian Geissler et al. | We study the automated derivation of safety requirements in a self-driving vehicle use case, leveraging LLMs in combination with agent-based retrieval-augmented generation. Conventional approaches that utilise pre-trained LLMs to assist in safety analyses typically lack domain-specific knowledge. Existing RAG approaches address this issue, yet their performance deteriorates when handling complex queries and it becomes increasingly harder to retrieve the most relevant information. This is particularly relevant for safety-relevant applications. In this paper, we propose the use of agent-based RAG to derive safety requirements and show that the retrieved information is more relevant to the queries. We implement an agent-based approach on a document pool of automotive standards and the Apollo case study, as a representative example of an automated driving perception system. Our solution is tested on a data set of safety requirement questions and answers, extracted from the Apollo data. Evaluating a set of selected RAG metrics, we present and discuss advantages of a agent-based approach compared to default RAG methods. |
| 2025-04-15 | [Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs](http://arxiv.org/abs/2504.11239v1) | Chang Yang, Ruiyu Wang et al. | Reasoning is the fundamental capability of large language models (LLMs). Due to the rapid progress of LLMs, there are two main issues of current benchmarks: i) these benchmarks can be crushed in a short time (less than 1 year), and ii) these benchmarks may be easily hacked. To handle these issues, we propose the ever-scalingness for building the benchmarks which are uncrushable, unhackable, auto-verifiable and general. This paper presents Nondeterministic Polynomial-time Problem Challenge (NPPC), an ever-scaling reasoning benchmark for LLMs. Specifically, the NPPC has three main modules: i) npgym, which provides a unified interface of 25 well-known NP-complete problems and can generate any number of instances with any levels of complexities, ii) npsolver: which provides a unified interface to evaluate the problem instances with both online and offline models via APIs and local deployments, respectively, and iii) npeval: which provides the comprehensive and ready-to-use tools to analyze the performances of LLMs over different problems, the number of tokens, the aha moments, the reasoning errors and the solution errors. Extensive experiments over widely-used LLMs demonstrate: i) NPPC can successfully decrease the performances of advanced LLMs' performances to below 10%, demonstrating that NPPC is uncrushable, ii) DeepSeek-R1, Claude-3.7-Sonnet, and o1/o3-mini are the most powerful LLMs, where DeepSeek-R1 outperforms Claude-3.7-Sonnet and o1/o3-mini in most NP-complete problems considered, and iii) the numbers of tokens, aha moments in the advanced LLMs, e.g., Claude-3.7-Sonnet and DeepSeek-R1, are observed first to increase and then decrease when the problem instances become more and more difficult. We believe that NPPC is the first ever-scaling reasoning benchmark, serving as the uncrushable and unhackable testbed for LLMs toward artificial general intelligence (AGI). |
| 2025-04-15 | [Benchmarking Next-Generation Reasoning-Focused Large Language Models in Ophthalmology: A Head-to-Head Evaluation on 5,888 Items](http://arxiv.org/abs/2504.11186v1) | Minjie Zou, Sahana Srinivasan et al. | Recent advances in reasoning-focused large language models (LLMs) mark a shift from general LLMs toward models designed for complex decision-making, a crucial aspect in medicine. However, their performance in specialized domains like ophthalmology remains underexplored. This study comprehensively evaluated and compared the accuracy and reasoning capabilities of four newly developed reasoning-focused LLMs, namely DeepSeek-R1, OpenAI o1, o3-mini, and Gemini 2.0 Flash-Thinking. Each model was assessed using 5,888 multiple-choice ophthalmology exam questions from the MedMCQA dataset in zero-shot setting. Quantitative evaluation included accuracy, Macro-F1, and five text-generation metrics (ROUGE-L, METEOR, BERTScore, BARTScore, and AlignScore), computed against ground-truth reasonings. Average inference time was recorded for a subset of 100 randomly selected questions. Additionally, two board-certified ophthalmologists qualitatively assessed clarity, completeness, and reasoning structure of responses to differential diagnosis questions.O1 (0.902) and DeepSeek-R1 (0.888) achieved the highest accuracy, with o1 also leading in Macro-F1 (0.900). The performance of models across the text-generation metrics varied: O3-mini excelled in ROUGE-L (0.151), o1 in METEOR (0.232), DeepSeek-R1 and o3-mini tied for BERTScore (0.673), DeepSeek-R1 (-4.105) and Gemini 2.0 Flash-Thinking (-4.127) performed best in BARTScore, while o3-mini (0.181) and o1 (0.176) led AlignScore. Inference time across the models varied, with DeepSeek-R1 being slowest (40.4 seconds) and Gemini 2.0 Flash-Thinking fastest (6.7 seconds). Qualitative evaluation revealed that DeepSeek-R1 and Gemini 2.0 Flash-Thinking tended to provide detailed and comprehensive intermediate reasoning, whereas o1 and o3-mini displayed concise and summarized justifications. |
| 2025-04-15 | [Exploring Backdoor Attack and Defense for LLM-empowered Recommendations](http://arxiv.org/abs/2504.11182v1) | Liangbo Ning, Wenqi Fan et al. | The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner. |
| 2025-04-15 | [A Real-time Anomaly Detection Method for Robots based on a Flexible and Sparse Latent Space](http://arxiv.org/abs/2504.11170v1) | Taewook Kang, Bum-Jae You et al. | The growing demand for robots to operate effectively in diverse environments necessitates the need for robust real-time anomaly detection techniques during robotic operations. However, deep learning-based models in robotics face significant challenges due to limited training data and highly noisy signal features. In this paper, we present Sparse Masked Autoregressive Flow-based Adversarial AutoEncoders model to address these problems. This approach integrates Masked Autoregressive Flow model into Adversarial AutoEncoders to construct a flexible latent space and utilize Sparse autoencoder to efficiently focus on important features, even in scenarios with limited feature space. Our experiments demonstrate that the proposed model achieves a 4.96% to 9.75% higher area under the receiver operating characteristic curve for pick-and-place robotic operations with randomly placed cans, compared to existing state-of-the-art methods. Notably, it showed up to 19.67% better performance in scenarios involving collisions with lightweight objects. Additionally, unlike the existing state-of-the-art model, our model performs inferences within 1 millisecond, ensuring real-time anomaly detection. These capabilities make our model highly applicable to machine learning-based robotic safety systems in dynamic environments. The code will be made publicly available after acceptance. |
| 2025-04-15 | [Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails](http://arxiv.org/abs/2504.11168v1) | William Hackett, Lewis Birch et al. | Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems. |
| 2025-04-15 | [Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models](http://arxiv.org/abs/2504.11106v1) | Jiangtao Liu, Zhaoxin Wang et al. | Recent advancements in Text-to-Image (T2I) generation have significantly enhanced the realism and creativity of generated images. However, such powerful generative capabilities pose risks related to the production of inappropriate or harmful content. Existing defense mechanisms, including prompt checkers and post-hoc image checkers, are vulnerable to sophisticated adversarial attacks. In this work, we propose TCBS-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. By iteratively optimizing tokens near these boundaries, TCBS-Attack generates semantically coherent adversarial prompts capable of bypassing multiple defensive layers in T2I models. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 45\% and an ASR-1 of 21\% on jailbreaking full-chain T2I models, significantly surpassing baseline methods. |
| 2025-04-15 | [Neural Control Barrier Functions from Physics Informed Neural Networks](http://arxiv.org/abs/2504.11045v1) | Shreenabh Agrawal, Manan Tayal et al. | As autonomous systems become increasingly prevalent in daily life, ensuring their safety is paramount. Control Barrier Functions (CBFs) have emerged as an effective tool for guaranteeing safety; however, manually designing them for specific applications remains a significant challenge. With the advent of deep learning techniques, recent research has explored synthesizing CBFs using neural networks-commonly referred to as neural CBFs. This paper introduces a novel class of neural CBFs that leverages a physics-inspired neural network framework by incorporating Zubov's Partial Differential Equation (PDE) within the context of safety. This approach provides a scalable methodology for synthesizing neural CBFs applicable to high-dimensional systems. Furthermore, by utilizing reciprocal CBFs instead of zeroing CBFs, the proposed framework allows for the specification of flexible, user-defined safe regions. To validate the effectiveness of the approach, we present case studies on three different systems: an inverted pendulum, autonomous ground navigation, and aerial navigation in obstacle-laden environments. |
| 2025-04-15 | [DRIFT open dataset: A drone-derived intelligence for traffic analysis in urban environmen](http://arxiv.org/abs/2504.11019v1) | Hyejin Lee, Seokjun Hong et al. | Reliable traffic data are essential for understanding urban mobility and developing effective traffic management strategies. This study introduces the DRone-derived Intelligence For Traffic analysis (DRIFT) dataset, a large-scale urban traffic dataset collected systematically from synchronized drone videos at approximately 250 meters altitude, covering nine interconnected intersections in Daejeon, South Korea. DRIFT provides high-resolution vehicle trajectories that include directional information, processed through video synchronization and orthomap alignment, resulting in a comprehensive dataset of 81,699 vehicle trajectories. Through our DRIFT dataset, researchers can simultaneously analyze traffic at multiple scales - from individual vehicle maneuvers like lane-changes and safety metrics such as time-to-collision to aggregate network flow dynamics across interconnected urban intersections. The DRIFT dataset is structured to enable immediate use without additional preprocessing, complemented by open-source models for object detection and trajectory extraction, as well as associated analytical tools. DRIFT is expected to significantly contribute to academic research and practical applications, such as traffic flow analysis and simulation studies. The dataset and related resources are publicly accessible at https://github.com/AIxMobility/The-DRIFT. |
| 2025-04-15 | [Drivers and barriers of adopting shared micromobility: a latent class clustering model on the attitudes towards shared micromobility as part of public transport trips in the Netherlands](http://arxiv.org/abs/2504.10943v1) | Nejc Geržinič, Mark van Hagen et al. | Shared micromobility (SMM) is often cited as a solution to the first/last mile problem of public transport (train) travel, yet when implemented, they often do not get adopted by a broader travelling public. A large part of behavioural adoption is related to peoples' attitudes and perceptions. In this paper, we develop an adjusted behavioural framework, based on the UTAUT2 technology acceptance framework. We carry out an exploratory factor analysis (EFA) to obtain attitudinal factors which we then use to perform a latent class cluster analysis (LCCA), with the goal of studying the potential adoption of SMM and to assess the various drivers and barriers as perceived by different user groups. Our findings suggest there are six distinct user groups with varying intention to use shared micromobility: Progressives, Conservatives, Hesitant participants, Bold innovators, Anxious observers and Skilled sceptics. Bold innovators and Progressives tend to be the most open to adopting SMM and are also able to do so. Hesitant participants would like to, but find it difficult or dangerous to use, while Skilled sceptics are capable and confident, but have limited intention of using it. Conservatives and Anxious observers are most negative about SMM, finding it difficult to use and dangerous. In general, factors relating to technological savviness, ease-of-use, physical safety and societal perception seem to be the biggest barriers to wider adoption. Younger, highly educated males are the group most likely and open to using shared micromobility, while older individuals with lower incomes and a lower level of education tend to be the least likely. |
| 2025-04-15 | [Safe-Construct: Redefining Construction Safety Violation Recognition as 3D Multi-View Engagement Task](http://arxiv.org/abs/2504.10880v1) | Aviral Chharia, Tianyu Ren et al. | Recognizing safety violations in construction environments is critical yet remains underexplored in computer vision. Existing models predominantly rely on 2D object detection, which fails to capture the complexities of real-world violations due to: (i) an oversimplified task formulation treating violation recognition merely as object detection, (ii) inadequate validation under realistic conditions, (iii) absence of standardized baselines, and (iv) limited scalability from the unavailability of synthetic dataset generators for diverse construction scenarios. To address these challenges, we introduce Safe-Construct, the first framework that reformulates violation recognition as a 3D multi-view engagement task, leveraging scene-level worker-object context and 3D spatial understanding. We also propose the Synthetic Indoor Construction Site Generator (SICSG) to create diverse, scalable training data, overcoming data limitations. Safe-Construct achieves a 7.6% improvement over state-of-the-art methods across four violation types. We rigorously evaluate our approach in near-realistic settings, incorporating four violations, four workers, 14 objects, and challenging conditions like occlusions (worker-object, worker-worker) and variable illumination (back-lighting, overexposure, sunlight). By integrating 3D multi-view spatial understanding and synthetic data generation, Safe-Construct sets a new benchmark for scalable and robust safety monitoring in high-risk industries. Project Website: https://Safe-Construct.github.io/Safe-Construct |
| 2025-04-15 | [Hallucination-Aware Generative Pretrained Transformer for Cooperative Aerial Mobility Control](http://arxiv.org/abs/2504.10831v1) | Hyojun Ahn, Seungcheol Oh et al. | This paper proposes SafeGPT, a two-tiered framework that integrates generative pretrained transformers (GPTs) with reinforcement learning (RL) for efficient and reliable unmanned aerial vehicle (UAV) last-mile deliveries. In the proposed design, a Global GPT module assigns high-level tasks such as sector allocation, while an On-Device GPT manages real-time local route planning. An RL-based safety filter monitors each GPT decision and overrides unsafe actions that could lead to battery depletion or duplicate visits, effectively mitigating hallucinations. Furthermore, a dual replay buffer mechanism helps both the GPT modules and the RL agent refine their strategies over time. Simulation results demonstrate that SafeGPT achieves higher delivery success rates compared to a GPT-only baseline, while substantially reducing battery consumption and travel distance. These findings validate the efficacy of combining GPT-based semantic reasoning with formal safety guarantees, contributing a viable solution for robust and energy-efficient UAV logistics. |
| 2025-04-15 | [LayoutCoT: Unleashing the Deep Reasoning Potential of Large Language Models for Layout Generation](http://arxiv.org/abs/2504.10829v1) | Hengyu Shi, Junhao Su et al. | Conditional layout generation aims to automatically generate visually appealing and semantically coherent layouts from user-defined constraints. While recent methods based on generative models have shown promising results, they typically require substantial amounts of training data or extensive fine-tuning, limiting their versatility and practical applicability. Alternatively, some training-free approaches leveraging in-context learning with Large Language Models (LLMs) have emerged, but they often suffer from limited reasoning capabilities and overly simplistic ranking mechanisms, which restrict their ability to generate consistently high-quality layouts. To this end, we propose LayoutCoT, a novel approach that leverages the reasoning capabilities of LLMs through a combination of Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) techniques. Specifically, LayoutCoT transforms layout representations into a standardized serialized format suitable for processing by LLMs. A Layout-aware RAG is used to facilitate effective retrieval and generate a coarse layout by LLMs. This preliminary layout, together with the selected exemplars, is then fed into a specially designed CoT reasoning module for iterative refinement, significantly enhancing both semantic coherence and visual quality. We conduct extensive experiments on five public datasets spanning three conditional layout generation tasks. Experimental results demonstrate that LayoutCoT achieves state-of-the-art performance without requiring training or fine-tuning. Notably, our CoT reasoning module enables standard LLMs, even those without explicit deep reasoning abilities, to outperform specialized deep-reasoning models such as deepseek-R1, highlighting the potential of our approach in unleashing the deep reasoning capabilities of LLMs for layout generation tasks. |
| 2025-04-15 | [The Sword of Damocles in ViTs: Computational Redundancy Amplifies Adversarial Transferability](http://arxiv.org/abs/2504.10804v1) | Jiani Liu, Zhiyuan Wang et al. | Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures. |
| 2025-04-15 | [Products of Recursive Programs for Hypersafety Verification](http://arxiv.org/abs/2504.10800v1) | Ruotong Cheng, Azadeh Farzan | We study the problem of automated hypersafety verification of infinite-state recursive programs. We propose an infinite class of product programs, specifically designed with recursion in mind, that reduce the hypersafety verification of a recursive program to standard safety verification. For this, we combine insights from language theory and concurrency theory to propose an algorithmic solution for constructing an infinite class of recursive product programs. One key insight is that, using the simple theory of visibly pushdown languages, one can maintain the recursive structure of syntactic program alignments which is vital to constructing a new product program that can be viewed as a classic recursive program -- that is, one that can be executed on a single stack. Another key insight is that techniques from concurrency theory can be generalized to help define product programs based on the view that the parallel composition of individual recursive programs includes all possible alignments from which a sound set of alignments that faithfully preserve the satisfaction of the hypersafety property can be selected. On the practical side, we formulate a family of parametric canonical product constructions that are intuitive to programmers and can be used as building blocks to specify recursive product programs for the purpose of relational and hypersafety verification, with the idea that the right product program can be verified automatically using existing techniques. We demonstrate the effectiveness of these techniques through an implementation and highly promising experimental results. |
| 2025-04-14 | [ReasonDrive: Efficient Visual Question Answering for Autonomous Vehicles with Reasoning-Enhanced Small Vision-Language Models](http://arxiv.org/abs/2504.10757v1) | Amirhosein Chahe, Lifeng Zhou | Vision-language models (VLMs) show promise for autonomous driving but often lack transparent reasoning capabilities that are critical for safety. We investigate whether explicitly modeling reasoning during fine-tuning enhances VLM performance on driving decision tasks. Using GPT-4o, we generate structured reasoning chains for driving scenarios from the DriveLM benchmark with category-specific prompting strategies. We compare reasoning-based fine-tuning, answer-only fine-tuning, and baseline instruction-tuned models across multiple small VLM families (Llama 3.2, Llava 1.5, and Qwen 2.5VL). Our results demonstrate that reasoning-based fine-tuning consistently outperforms alternatives, with Llama3.2-11B-reason achieving the highest performance. Models fine-tuned with reasoning show substantial improvements in accuracy and text generation quality, suggesting explicit reasoning enhances internal representations for driving decisions. These findings highlight the importance of transparent decision processes in safety-critical domains and offer a promising direction for developing more interpretable autonomous driving systems. |
| 2025-04-14 | [The Jailbreak Tax: How Useful are Your Jailbreak Outputs?](http://arxiv.org/abs/2504.10694v1) | Kristina Nikolić, Luze Sun et al. | Jailbreak attacks bypass the guardrails of large language models to produce harmful outputs. In this paper, we ask whether the model outputs produced by existing jailbreaks are actually useful. For example, when jailbreaking a model to give instructions for building a bomb, does the jailbreak yield good instructions? Since the utility of most unsafe answers (e.g., bomb instructions) is hard to evaluate rigorously, we build new jailbreak evaluation sets with known ground truth answers, by aligning models to refuse questions related to benign and easy-to-evaluate topics (e.g., biology or math). Our evaluation of eight representative jailbreaks across five utility benchmarks reveals a consistent drop in model utility in jailbroken responses, which we term the jailbreak tax. For example, while all jailbreaks we tested bypass guardrails in models aligned to refuse to answer math, this comes at the expense of a drop of up to 92% in accuracy. Overall, our work proposes the jailbreak tax as a new important metric in AI safety, and introduces benchmarks to evaluate existing and future jailbreaks. We make the benchmark available at https://github.com/ethz-spylab/jailbreak-tax |
| 2025-04-14 | [Beyond Chains of Thought: Benchmarking Latent-Space Reasoning Abilities in Large Language Models](http://arxiv.org/abs/2504.10615v1) | Thilo Hagendorff, Sarah Fabi | Large language models (LLMs) can perform reasoning computations both internally within their latent space and externally by generating explicit token sequences like chains of thought. Significant progress in enhancing reasoning abilities has been made by scaling test-time compute. However, understanding and quantifying model-internal reasoning abilities - the inferential "leaps" models make between individual token predictions - remains crucial. This study introduces a benchmark (n = 4,000 items) designed to quantify model-internal reasoning in different domains. We achieve this by having LLMs indicate the correct solution to reasoning problems not through descriptive text, but by selecting a specific language of their initial response token that is different from English, the benchmark language. This not only requires models to reason beyond their context window, but also to overrise their default tendency to respond in the same language as the prompt, thereby posing an additional cognitive strain. We evaluate a set of 18 LLMs, showing significant performance variations, with GPT-4.5 achieving the highest accuracy (74.7%), outperforming models like Grok-2 (67.2%), and Llama 3.1 405B (65.6%). Control experiments and difficulty scaling analyses suggest that while LLMs engage in internal reasoning, we cannot rule out heuristic exploitations under certain conditions, marking an area for future investigation. Our experiments demonstrate that LLMs can "think" via latent-space computations, revealing model-internal inference strategies that need further understanding, especially regarding safety-related concerns such as covert planning, goal-seeking, or deception emerging without explicit token traces. |

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



