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
| 2025-05-21 | [On the creation of narrow AI: hierarchy and nonlocality of neural network skills](http://arxiv.org/abs/2505.15811v1) | Eric J. Michaud, Asher Parker-Sartori et al. | We study the problem of creating strong, yet narrow, AI systems. While recent AI progress has been driven by the training of large general-purpose foundation models, the creation of smaller models specialized for narrow domains could be valuable for both efficiency and safety. In this work, we explore two challenges involved in creating such systems, having to do with basic properties of how neural networks learn and structure their representations. The first challenge regards when it is possible to train narrow models from scratch. Through experiments on a synthetic task, we find that it is sometimes necessary to train networks on a wide distribution of data to learn certain narrow skills within that distribution. This effect arises when skills depend on each other hierarchically, and training on a broad distribution introduces a curriculum which substantially accelerates learning. The second challenge regards how to transfer particular skills from large general models into small specialized models. We find that model skills are often not perfectly localized to a particular set of prunable components. However, we find that methods based on pruning can still outperform distillation. We investigate the use of a regularization objective to align desired skills with prunable components while unlearning unnecessary skills. |
| 2025-05-21 | [Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering](http://arxiv.org/abs/2505.15805v1) | Hwan Chang, Yumin Kim et al. | As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security. |
| 2025-05-21 | [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](http://arxiv.org/abs/2505.15801v1) | Yuchen Yan, Jin Jiang et al. | Large reasoning models such as OpenAI o1 and DeepSeek-R1 have achieved remarkable performance in the domain of reasoning. A key component of their training is the incorporation of verifiable rewards within reinforcement learning (RL). However, existing reward benchmarks do not evaluate reference-based reward systems, leaving researchers with limited understanding of the accuracy of verifiers used in RL. In this paper, we introduce two benchmarks, VerifyBench and VerifyBench-Hard, designed to assess the performance of reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Current models still show considerable room for improvement on both VerifyBench and VerifyBench-Hard, especially smaller-scale models. Furthermore, we conduct a thorough and comprehensive analysis of evaluation results, offering insights for understanding and developing reference-based reward systems. Our proposed benchmarks serve as effective tools for guiding the development of verifier accuracy and the reasoning capabilities of models trained via RL in reasoning tasks. |
| 2025-05-21 | [HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving](http://arxiv.org/abs/2505.15793v1) | Zhiwen Chen, Bo Leng et al. | Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to hallucinations.Evaluations show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios. |
| 2025-05-21 | [Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval](http://arxiv.org/abs/2505.15753v1) | Taiye Chen, Zeming Wei et al. | Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication. |
| 2025-05-21 | [Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses](http://arxiv.org/abs/2505.15738v1) | Xiaoxue Yang, Bozhidar Stevanoski et al. | Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs. |
| 2025-05-21 | [Safe Control for Pursuit-Evasion with Density Functions](http://arxiv.org/abs/2505.15718v1) | Mustafa Bozdag, Arya Honarpisheh et al. | This letter presents a density function based safe control synthesis framework for the pursuit-evasion problem. We extend safety analysis to dynamic unsafe sets by formulating a reach-avoid type pursuit-evasion differential game as a robust safe control problem. Using density functions and semi-algebraic set definitions, we derive sufficient conditions for weak eventuality and evasion, reformulating the problem into a convex sum-of-squares program solvable via standard semidefinite programming solvers. This approach avoids the computational complexity of solving the Hamilton-Jacobi-Isaacs partial differential equation, offering a scalable and efficient framework. Numerical simulations demonstrate the efficacy of the proposed method. |
| 2025-05-21 | [Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling](http://arxiv.org/abs/2505.15715v1) | He Hu, Yucheng Zhou et al. | Large language models (LLMs) hold significant potential for mental health support, capable of generating empathetic responses and simulating therapeutic conversations. However, existing LLM-based approaches often lack the clinical grounding necessary for real-world psychological counseling, particularly in explicit diagnostic reasoning aligned with standards like the DSM/ICD and incorporating diverse therapeutic modalities beyond basic empathy or single strategies. To address these critical limitations, we propose PsyLLM, the first large language model designed to systematically integrate both diagnostic and therapeutic reasoning for mental health counseling. To develop the PsyLLM, we propose a novel automated data synthesis pipeline. This pipeline processes real-world mental health posts, generates multi-turn dialogue structures, and leverages LLMs guided by international diagnostic standards (e.g., DSM/ICD) and multiple therapeutic frameworks (e.g., CBT, ACT, psychodynamic) to simulate detailed clinical reasoning processes. Rigorous multi-dimensional filtering ensures the generation of high-quality, clinically aligned dialogue data. In addition, we introduce a new benchmark and evaluation protocol, assessing counseling quality across four key dimensions: comprehensiveness, professionalism, authenticity, and safety. Our experiments demonstrate that PsyLLM significantly outperforms state-of-the-art baseline models on this benchmark. |
| 2025-05-21 | [Advancing LLM Safe Alignment with Safety Representation Ranking](http://arxiv.org/abs/2505.15710v1) | Tianqi Du, Zeming Wei et al. | The rapid advancement of large language models (LLMs) has demonstrated milestone success in a variety of tasks, yet their potential for generating harmful content has raised significant safety concerns. Existing safety evaluation approaches typically operate directly on textual responses, overlooking the rich information embedded in the model's internal representations. In this paper, we propose Safety Representation Ranking (SRR), a listwise ranking framework that selects safe responses using hidden states from the LLM itself. SRR encodes both instructions and candidate completions using intermediate transformer representations and ranks candidates via a lightweight similarity-based scorer. Our approach directly leverages internal model states and supervision at the list level to capture subtle safety signals. Experiments across multiple benchmarks show that SRR significantly improves robustness to adversarial prompts. Our code will be available upon publication. |
| 2025-05-21 | [SwarmDiff: Swarm Robotic Trajectory Planning in Cluttered Environments via Diffusion Transformer](http://arxiv.org/abs/2505.15679v1) | Kang Ding, Chunxuan Jiao et al. | Swarm robotic trajectory planning faces challenges in computational efficiency, scalability, and safety, particularly in complex, obstacle-dense environments. To address these issues, we propose SwarmDiff, a hierarchical and scalable generative framework for swarm robots. We model the swarm's macroscopic state using Probability Density Functions (PDFs) and leverage conditional diffusion models to generate risk-aware macroscopic trajectory distributions, which then guide the generation of individual robot trajectories at the microscopic level. To ensure a balance between the swarm's optimal transportation and risk awareness, we integrate Wasserstein metrics and Conditional Value at Risk (CVaR). Additionally, we introduce a Diffusion Transformer (DiT) to improve sampling efficiency and generation quality by capturing long-range dependencies. Extensive simulations and real-world experiments demonstrate that SwarmDiff outperforms existing methods in computational efficiency, trajectory validity, and scalability, making it a reliable solution for swarm robotic trajectory planning. |
| 2025-05-21 | [Enhancing Monte Carlo Dropout Performance for Uncertainty Quantification](http://arxiv.org/abs/2505.15671v1) | Hamzeh Asgharnezhad, Afshar Shamsi et al. | Knowing the uncertainty associated with the output of a deep neural network is of paramount importance in making trustworthy decisions, particularly in high-stakes fields like medical diagnosis and autonomous systems. Monte Carlo Dropout (MCD) is a widely used method for uncertainty quantification, as it can be easily integrated into various deep architectures. However, conventional MCD often struggles with providing well-calibrated uncertainty estimates. To address this, we introduce innovative frameworks that enhances MCD by integrating different search solutions namely Grey Wolf Optimizer (GWO), Bayesian Optimization (BO), and Particle Swarm Optimization (PSO) as well as an uncertainty-aware loss function, thereby improving the reliability of uncertainty quantification. We conduct comprehensive experiments using different backbones, namely DenseNet121, ResNet50, and VGG16, on various datasets, including Cats vs. Dogs, Myocarditis, Wisconsin, and a synthetic dataset (Circles). Our proposed algorithm outperforms the MCD baseline by 2-3% on average in terms of both conventional accuracy and uncertainty accuracy while achieving significantly better calibration. These results highlight the potential of our approach to enhance the trustworthiness of deep learning models in safety-critical applications. |
| 2025-05-21 | [Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models](http://arxiv.org/abs/2505.15634v1) | Zihao Li, Xu Wang et al. | Large Language Models (LLMs) demonstrate the ability to solve reasoning and mathematical problems using the Chain-of-Thought (CoT) technique. Expanding CoT length, as seen in models such as DeepSeek-R1, significantly enhances this reasoning for complex problems, but requires costly and high-quality long CoT data and fine-tuning. This work, inspired by the deep thinking paradigm of DeepSeek-R1, utilizes a steering technique to enhance the reasoning ability of an LLM without external datasets. Our method first employs Sparse Autoencoders (SAEs) to extract interpretable features from vanilla CoT. These features are then used to steer the LLM's internal states during generation. Recognizing that many LLMs do not have corresponding pre-trained SAEs, we further introduce a novel SAE-free steering algorithm, which directly computes steering directions from the residual activations of an LLM, obviating the need for an explicit SAE. Experimental results demonstrate that both our SAE-based and subsequent SAE-free steering algorithms significantly enhance the reasoning capabilities of LLMs. |
| 2025-05-21 | [Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](http://arxiv.org/abs/2505.15612v1) | Wei Liu, Ruochen Zhou et al. | Large Reasoning Models (LRMs) have shown remarkable capabilities in solving complex problems through reinforcement learning (RL), particularly by generating long reasoning traces. However, these extended outputs often exhibit substantial redundancy, which limits the efficiency of LRMs. In this paper, we investigate RL-based approaches to promote reasoning efficiency. Specifically, we first present a unified framework that formulates various efficient reasoning methods through the lens of length-based reward shaping. Building on this perspective, we propose a novel Length-bAsed StEp Reward shaping method (LASER), which employs a step function as the reward, controlled by a target length. LASER surpasses previous methods, achieving a superior Pareto-optimal balance between performance and efficiency. Next, we further extend LASER based on two key intuitions: (1) The reasoning behavior of the model evolves during training, necessitating reward specifications that are also adaptive and dynamic; (2) Rather than uniformly encouraging shorter or longer chains of thought (CoT), we posit that length-based reward shaping should be difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries. This approach is expected to facilitate a combination of fast and slow thinking, leading to a better overall tradeoff. The resulting method is termed LASER-D (Dynamic and Difficulty-aware). Experiments on DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and DeepSeek-R1-Distill-Qwen-32B show that our approach significantly enhances both reasoning performance and response length efficiency. For instance, LASER-D and its variant achieve a +6.1 improvement on AIME2024 while reducing token usage by 63%. Further analysis reveals our RL-based compression produces more concise reasoning patterns with less redundant "self-reflections". Resources are at https://github.com/hkust-nlp/Laser. |
| 2025-05-21 | [Self-powered smart contact lenses: a multidisciplinary approach to micro-scale energy and 900 MHz - 1.1 GHz bandwidth microfabricated loop antennas communication systems](http://arxiv.org/abs/2505.15593v1) | Patrice Salzenstein, Blandine Guichardaz et al. | Smart contact lenses are at the forefront of integrating microelectronics, biomedical engineering, and optics into wearable technologies. This work addresses a key obstacle in their development: achieving autonomous power without compromising safety or miniaturization. We examine energy harvesting strategies using intrinsic ocular sources-particularly tear salinity and eyelid motion-to enable sustainable operation without external batteries. The study emphasizes compact loop antennas operating between 900 MHz and 1.1 GHz as critical for wireless data transmission and power management. Material choices, signal integrity, and biocompatibility are also discussed. By presenting recent advances in 3D-printed optics, antenna integration, and energy systems, we propose a conceptual framework for the next generation of smart lenses, enabling real-time health monitoring and vision enhancement through self-powered, compact devices. |
| 2025-05-21 | [From learning to safety: A Direct Data-Driven Framework for Constrained Control](http://arxiv.org/abs/2505.15515v1) | Kanghui He, Shengling Shi et al. | Ensuring safety in the sense of constraint satisfaction for learning-based control is a critical challenge, especially in the model-free case. While safety filters address this challenge in the model-based setting by modifying unsafe control inputs, they typically rely on predictive models derived from physics or data. This reliance limits their applicability for advanced model-free learning control methods. To address this gap, we propose a new optimization-based control framework that determines safe control inputs directly from data. The benefit of the framework is that it can be updated through arbitrary model-free learning algorithms to pursue optimal performance. As a key component, the concept of direct data-driven safety filters (3DSF) is first proposed. The framework employs a novel safety certificate, called the state-action control barrier function (SACBF). We present three different schemes to learn the SACBF. Furthermore, based on input-to-state safety analysis, we present the error-to-state safety analysis framework, which provides formal guarantees on safety and recursive feasibility even in the presence of learning inaccuracies. The proposed control framework bridges the gap between model-free learning-based control and constrained control, by decoupling performance optimization from safety enforcement. Simulations on vehicle control illustrate the superior performance regarding constraint satisfaction and task achievement compared to model-based methods and reward shaping. |
| 2025-05-21 | [Multilingual Test-Time Scaling via Initial Thought Transfer](http://arxiv.org/abs/2505.15508v1) | Prasoon Bajpai, Tanmoy Chakraborty | Test-time scaling has emerged as a widely adopted inference-time strategy for boosting reasoning performance. However, its effectiveness has been studied almost exclusively in English, leaving its behavior in other languages largely unexplored. We present the first systematic study of test-time scaling in multilingual settings, evaluating DeepSeek-R1-Distill-LLama-8B and DeepSeek-R1-Distill-Qwen-7B across both high- and low-resource Latin-script languages. Our findings reveal that the relative gains from test-time scaling vary significantly across languages. Additionally, models frequently switch to English mid-reasoning, even when operating under strictly monolingual prompts. We further show that low-resource languages not only produce initial reasoning thoughts that differ significantly from English but also have lower internal consistency across generations in their early reasoning. Building on our findings, we introduce MITT (Multilingual Initial Thought Transfer), an unsupervised and lightweight reasoning prefix-tuning approach that transfers high-resource reasoning prefixes to enhance test-time scaling across all languages, addressing inconsistencies in multilingual reasoning performance. MITT significantly boosts DeepSeek-R1-Distill-Qwen-7B's reasoning performance, especially for underrepresented languages. |
| 2025-05-21 | [Coloring Between the Lines: Personalization in the Null Space of Planning Constraints](http://arxiv.org/abs/2505.15503v1) | Tom Silver, Rajat Kumar Jenamani et al. | Generalist robots must personalize in-the-wild to meet the diverse needs and preferences of long-term users. How can we enable flexible personalization without sacrificing safety or competency? This paper proposes Coloring Between the Lines (CBTL), a method for personalization that exploits the null space of constraint satisfaction problems (CSPs) used in robot planning. CBTL begins with a CSP generator that ensures safe and competent behavior, then incrementally personalizes behavior by learning parameterized constraints from online interaction. By quantifying uncertainty and leveraging the compositionality of planning constraints, CBTL achieves sample-efficient adaptation without environment resets. We evaluate CBTL in (1) three diverse simulation environments; (2) a web-based user study; and (3) a real-robot assisted feeding system, finding that CBTL consistently achieves more effective personalization with fewer interactions than baselines. Our results demonstrate that CBTL provides a unified and practical approach for continual, flexible, active, and safe robot personalization. Website: https://emprise.cs.cornell.edu/cbtl/ |
| 2025-05-21 | [Certified Neural Approximations of Nonlinear Dynamics](http://arxiv.org/abs/2505.15497v1) | Frederik Baymler Mathiesen, Nikolaus Vertovec et al. | Neural networks hold great potential to act as approximate models of nonlinear dynamical systems, with the resulting neural approximations enabling verification and control of such systems. However, in safety-critical contexts, the use of neural approximations requires formal bounds on their closeness to the underlying system. To address this fundamental challenge, we propose a novel, adaptive, and parallelizable verification method based on certified first-order models. Our approach provides formal error bounds on the neural approximations of dynamical systems, allowing them to be safely employed as surrogates by interpreting the error bound as bounded disturbances acting on the approximated dynamics. We demonstrate the effectiveness and scalability of our method on a range of established benchmarks from the literature, showing that it outperforms the state-of-the-art. Furthermore, we highlight the flexibility of our framework by applying it to two novel scenarios not previously explored in this context: neural network compression and an autoencoder-based deep learning architecture for learning Koopman operators, both yielding compelling results. |
| 2025-05-21 | [Comprehensive Evaluation and Analysis for NSFW Concept Erasure in Text-to-Image Diffusion Models](http://arxiv.org/abs/2505.15450v1) | Die Chen, Zhiwen Li et al. | Text-to-image diffusion models have gained widespread application across various domains, demonstrating remarkable creative potential. However, the strong generalization capabilities of diffusion models can inadvertently lead to the generation of not-safe-for-work (NSFW) content, posing significant risks to their safe deployment. While several concept erasure methods have been proposed to mitigate the issue associated with NSFW content, a comprehensive evaluation of their effectiveness across various scenarios remains absent. To bridge this gap, we introduce a full-pipeline toolkit specifically designed for concept erasure and conduct the first systematic study of NSFW concept erasure methods. By examining the interplay between the underlying mechanisms and empirical observations, we provide in-depth insights and practical guidance for the effective application of concept erasure methods in various real-world scenarios, with the aim of advancing the understanding of content safety in diffusion models and establishing a solid foundation for future research and development in this critical area. |
| 2025-05-21 | [Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries](http://arxiv.org/abs/2505.15420v1) | Yuhao Wang, Wenjie Qu et al. | Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, but they are vulnerable to privacy risks from data extraction attacks. Existing extraction methods typically rely on malicious inputs such as prompt injection or jailbreaking, making them easily detectable via input- or output-level detection. In this paper, we introduce Implicit Knowledge Extraction Attack (IKEA), which conducts knowledge extraction on RAG systems through benign queries. IKEA first leverages anchor concepts to generate queries with the natural appearance, and then designs two mechanisms to lead to anchor concept thoroughly 'explore' the RAG's privacy knowledge: (1) Experience Reflection Sampling, which samples anchor concepts based on past query-response patterns to ensure the queries' relevance to RAG documents; (2) Trust Region Directed Mutation, which iteratively mutates anchor concepts under similarity constraints to further exploit the embedding space. Extensive experiments demonstrate IKEA's effectiveness under various defenses, surpassing baselines by over 80% in extraction efficiency and 90% in attack success rate. Moreover, the substitute RAG system built from IKEA's extractions consistently outperforms those based on baseline methods across multiple evaluation tasks, underscoring the significant privacy risk in RAG systems. |

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



