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
| 2025-12-15 | [Comparative Analysis of LLM Abliteration Methods: A Cross-Architecture Evaluation](http://arxiv.org/abs/2512.13655v1) | Richard J. Young | Safety alignment mechanisms in large language models prevent responses to harmful queries through learned refusal behavior, yet these same mechanisms impede legitimate research applications including cognitive modeling, adversarial testing, and security analysis. While abliteration techniques enable surgical removal of refusal representations through directional orthogonalization, the relative effectiveness of available implementations remains uncharacterized. This study evaluates four abliteration tools (Heretic, DECCP, ErisForge, FailSpy) across sixteen instruction-tuned models (7B-14B parameters), reporting tool compatibility on all 16 models and quantitative metrics on subsets dictated by tool support. Single-pass methods demonstrated superior capability preservation on the benchmarked subset (avg GSM8K change across three models: ErisForge -0.28 pp; DECCP -0.13 pp), while Bayesian-optimized abliteration produced variable distribution shift (KL divergence: 0.043-1.646) with model-dependent capability impact. These findings provide researchers with evidence-based selection criteria for abliteration tool deployment across diverse model architectures. The principal finding indicates that mathematical reasoning capabilities exhibit the highest sensitivity to abliteration interventions, with GSM8K change ranging from +1.51 pp to -18.81 pp (-26.5% relative) depending on tool selection and model architecture. |
| 2025-12-15 | [Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Models](http://arxiv.org/abs/2512.13607v1) | Boxin Wang, Chankyu Lee et al. | Building general-purpose reasoning models with reinforcement learning (RL) entails substantial cross-domain heterogeneity, including large variation in inference-time response lengths and verification latency. Such variability complicates the RL infrastructure, slows training, and makes training curriculum (e.g., response length extension) and hyperparameter selection challenging. In this work, we propose cascaded domain-wise reinforcement learning (Cascade RL) to develop general-purpose reasoning models, Nemotron-Cascade, capable of operating in both instruct and deep thinking modes. Departing from conventional approaches that blend heterogeneous prompts from different domains, Cascade RL orchestrates sequential, domain-wise RL, reducing engineering complexity and delivering state-of-the-art performance across a wide range of benchmarks. Notably, RLHF for alignment, when used as a pre-step, boosts the model's reasoning ability far beyond mere preference optimization, and subsequent domain-wise RLVR stages rarely degrade the benchmark performance attained in earlier domains and may even improve it (see an illustration in Figure 1). Our 14B model, after RL, outperforms its SFT teacher, DeepSeek-R1-0528, on LiveCodeBench v5/v6/Pro and achieves silver-medal performance in the 2025 International Olympiad in Informatics (IOI). We transparently share our training and data recipes. |
| 2025-12-15 | [Near-Field Perception for Safety Enhancement of Autonomous Mobile Robots in Manufacturing Environments](http://arxiv.org/abs/2512.13561v1) | Li-Wei Shih, Ruo-Syuan Mei et al. | Near-field perception is essential for the safe operation of autonomous mobile robots (AMRs) in manufacturing environments. Conventional ranging sensors such as light detection and ranging (LiDAR) and ultrasonic devices provide broad situational awareness but often fail to detect small objects near the robot base. To address this limitation, this paper presents a three-tier near-field perception framework. The first approach employs light-discontinuity detection, which projects a laser stripe across the near-field zone and identifies interruptions in the stripe to perform fast, binary cutoff sensing for obstacle presence. The second approach utilizes light-displacement measurement to estimate object height by analyzing the geometric displacement of a projected stripe in the camera image, which provides quantitative obstacle height information with minimal computational overhead. The third approach employs a computer vision-based object detection model on embedded AI hardware to classify objects, enabling semantic perception and context-aware safety decisions. All methods are implemented on a Raspberry Pi 5 system, achieving real-time performance at 25 or 50 frames per second. Experimental evaluation and comparative analysis demonstrate that the proposed hierarchy balances precision, computation, and cost, thereby providing a scalable perception solution for enabling safe operations of AMRs in manufacturing environments. |
| 2025-12-15 | [neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings](http://arxiv.org/abs/2512.13481v1) | Ojas Pungalia, Rashi Upadhyay et al. | Envy is a common human behavior that shapes competitiveness and can alter outcomes in team settings. As large language models (LLMs) increasingly act on behalf of humans in collaborative and competitive workflows, there is a pressing need to evaluate whether and under what conditions they exhibit envy-like preferences. In this paper, we test whether LLMs show envy-like behavior toward each other. We considered two scenarios: (1) A point allocation game that tests whether a model tries to win over its peer. (2) A workplace setting observing behaviour when recognition is unfair. Our findings reveal consistent evidence of envy-like patterns in certain LLMs, with large variation across models and contexts. For instance, GPT-5-mini and Claude-3.7-Sonnet show a clear tendency to pull down the peer model to equalize outcomes, whereas Mistral-Small-3.2-24B instead focuses on maximizing its own individual gains. These results highlight the need to consider competitive dispositions as a safety and design factor in LLM-based multi-agent systems. |
| 2025-12-15 | [A Data Annotation Requirements Representation and Specification (DARS)](http://arxiv.org/abs/2512.13444v1) | Yi Peng, Hina Saeeda et al. | With the rise of AI-enabled cyber-physical systems, data annotation has become a critical yet often overlooked process in the development of these intelligent information systems. Existing work in requirements engineering (RE) has explored how requirements for AI systems and their data can be represented. However, related interviews with industry professionals show that data annotations and their related requirements introduce distinct challenges, indicating a need for annotation-specific requirement representations. We propose the Data Annotation Requirements Representation and Specification (DARS), including an Annotation Negotiation Card to align stakeholders on objectives and constraints, and a Scenario-Based Annotation Specification to express atomic and verifiable data annotation requirements. We evaluate DARS with an automotive perception case related to an ongoing project, and a mapping against 18 real-world data annotation error types. The results suggest that DARS mitigates root causes of completeness, accuracy, and consistency annotation errors. By integrating DARS into RE, this work improves the reliability of safety-critical systems using data annotations and demonstrates how engineering frameworks must evolve for data-dependent components of today's intelligent information systems. |
| 2025-12-15 | [Taxonomy of type II orientifold flux vacua in 3D](http://arxiv.org/abs/2512.13433v1) | Álvaro Arboleya, Gabriele Casagrande et al. | We complete the study initiated in \cite{Arboleya:2024vnp} and investigate three-dimensional (3D) flux vacua of type II orientifold reductions on twisted tori that include a single type of spacetime-filling O$p$-plane with $\,p=2,\ldots,9$. Restricting to $\textrm{SO}(3)$-invariant setups -- also known as RSTU-models -- and setting axions to zero, we exhaustively chart the landscape of 3D orientifold flux vacua. It consists of $\,56\,$ inequivalent multi-parametric families of AdS$_3$ and Mkw$_3$ flux vacua, all of them without negative masses in the spectrum of scalar fluctuations. Performing T-dualities, all the inequivalent vacua can be found in type IIB with either O9-, O5- or O3-planes. We show that: $i)\,$ type IIB with O$9$ fails to stabilise the volume of the internal space. $ii)\,$ type IIB with O5 realises the only cases of scale-separated AdS$_{3}$ vacua. $iii)\,$ type IIB with O$3$ provides only Mkw$_{3}$ vacua exhibiting a rich variety of supersymmetry-breaking patterns. Assuming that some mechanism (possibly non-perturbative) fixes the unstabilised modulus in type IIB with O$9$, we re-examine the possibility of achieving a weakly-coupled and scale-separated regime. |
| 2025-12-15 | [Neural Control Barrier Functions for Signal Temporal Logic Specifications with Input Constraints](http://arxiv.org/abs/2512.13344v1) | Vaishnavi Jagabathula, Pushpak Jagtap | Signal Temporal Logic (STL) provides a powerful framework to describe complex tasks involving temporal and logical behavior in dynamical systems. In this work, we address the problem of synthesizing controllers for continuous-time systems under STL specifications with input constraints. We propose a neural network-based framework for synthesizing time-varying control barrier functions (TVCBF) and their corresponding controllers for systems to fulfill STL specifications while respecting input constraints. We formulate barrier conditions incorporating the spatial and temporal logic of the given STL specification. Additionally, we introduce a validity condition to provide formal safety guarantees across the entire state space. Finally, we demonstrate the effectiveness of the proposed approach through several simulation studies considering different STL tasks. |
| 2025-12-15 | [FROC: A Unified Framework with Risk-Optimized Control for Machine Unlearning in LLMs](http://arxiv.org/abs/2512.13337v1) | Si Qi Goh, Yongsen Zheng et al. | Machine unlearning (MU) seeks to eliminate the influence of specific training examples from deployed models. As large language models (LLMs) become widely used, managing risks arising from insufficient forgetting or utility loss is increasingly crucial. Current MU techniques lack effective mechanisms for evaluating and controlling these risks, hindering the selection of strategies that appropriately balance safety and utility, and raising trust concerns surrounding the "right to be forgotten." To address these issues, we propose FROC, a unified framework with Risk-Optimized Control for machine unlearning in LLMs. FROC is built around a conformal-style risk-control formulation that expresses a user-specified risk budget on unlearning behavior. This probability-based constraint enables FROC to compare MU strategies, identify feasible operating regions, and guide hyperparameter selection according to desired trade-offs between forgetting sufficiency and utility preservation. To operationalize this constraint, FROC introduces a smoothly varying continuous risk model that aggregates forgetting deficiency and utility degradation into a single configuration-level score. Building on conformal risk analysis, FROC computes (1) the Conformal Unlearning Risk (CUR), a data-driven estimated value on the probability that forgotten samples continue to influence model predictions, and (2) risk-controlled configuration sets, which identify unlearning hyperparameters that are valid under the specified risk budget. Experiments across multiple LLM MU methods demonstrate that FROC produces stable, interpretable risk landscapes and reveals consistent relationships between unlearning configurations, semantic shift, and utility impact. FROC reframes MU as a controllable, risk-aware process and offers a practical foundation for managing unlearning behavior in large-scale LLM deployments. |
| 2025-12-15 | [On the impact of geometric variance on the performance of formed parts: A probabilistic approach on the example of airbag pressure bins](http://arxiv.org/abs/2512.13302v1) | Lukas Schnelle, Niklas Fehlemann et al. | Scatter in properties resulting from manufacturing is a great challenge in lightweight design, requiring consideration of not only the average mechanical performance but also the variance which is done e.g., by conservative safety factors. One contributor to this variance is the inherent geometric variability in the formed part. To isolate and quantify this effect, we present a probabilistic numerical study, aiming to assess the impact of geometric variance on the resulting part performance. By modelling geometric deviations stochastically, we aim to establish a correlation between the variance in geometry with the resulting variance in performance. The study is done on the example of an airbag pressure bin, where a better understanding of this correlation is crucial, as it allows for the design of a lighter part without changing the manufacturing process. Instead, we aim to implement more targeted and effective quality assurance, informed by the performance impact of geometric deviations. |
| 2025-12-15 | [Post-Training and Test-Time Scaling of Generative Agent Behavior Models for Interactive Autonomous Driving](http://arxiv.org/abs/2512.13262v1) | Hyunki Seong, Jeong-Kyun Lee et al. | Learning interactive motion behaviors among multiple agents is a core challenge in autonomous driving. While imitation learning models generate realistic trajectories, they often inherit biases from datasets dominated by safe demonstrations, limiting robustness in safety-critical cases. Moreover, most studies rely on open-loop evaluation, overlooking compounding errors in closed-loop execution. We address these limitations with two complementary strategies. First, we propose Group Relative Behavior Optimization (GRBO), a reinforcement learning post-training method that fine-tunes pretrained behavior models via group relative advantage maximization with human regularization. Using only 10% of the training dataset, GRBO improves safety performance by over 40% while preserving behavioral realism. Second, we introduce Warm-K, a warm-started Top-K sampling strategy that balances consistency and diversity in motion selection. Our Warm-K method-based test-time scaling enhances behavioral consistency and reactivity at test time without retraining, mitigating covariate shift and reducing performance discrepancies. Demo videos are available in the supplementary material. |
| 2025-12-15 | [Multi-directional Safe Rectangle Corridor-Based MPC for Nonholonomic Robots Navigation in Cluttered Environment](http://arxiv.org/abs/2512.13215v1) | Yinsong Qu, Yunxiang Li et al. | Autonomous Mobile Robots (AMRs) have become indispensable in industrial applications due to their operational flexibility and efficiency. Navigation serves as a crucial technical foundation for accomplishing complex tasks. However, navigating AMRs in dense, cluttered, and semi-structured environments remains challenging, primarily due to nonholonomic vehicle dynamics, interactions with mixed static/dynamic obstacles, and the non-convex constrained nature of such operational spaces. To solve these problems, this paper proposes an Improved Sequential Model Predictive Control (ISMPC) navigation framework that systematically reformulates navigation tasks as sequential switched optimal control problems. The framework addresses the aforementioned challenges through two key innovations: 1) Implementation of a Multi-Directional Safety Rectangular Corridor (MDSRC) algorithm, which encodes the free space through rectangular convex regions to avoid collision with static obstacles, eliminating redundant computational burdens and accelerating solver convergence; 2) A sequential MPC navigation framework that integrates corridor constraints with barrier function constraints is proposed to achieve static and dynamic obstacle avoidance. The ISMPC navigation framework enables direct velocity generation for AMRs, simplifying traditional navigation algorithm architectures. Comparative experiments demonstrate the framework's superiority in free-space utilization ( an increase of 41.05$\%$ in the average corridor area) while maintaining real-time computational performance (average corridors generation latency of 3 ms). |
| 2025-12-15 | [Clinical transfusion-outcomes research: A practical guide](http://arxiv.org/abs/2512.13155v1) | Sarah J Valk, Camila Caram-Deelder et al. | Clinical transfusion-outcomes research faces unique methodological challenges compared with other areas of clinical research. These challenges arise because patients frequently receive multiple transfusions, each unit originates from a different donor, and the probability of receiving specific blood product characteristics is influenced by external, often uncontrollable, factors. These complexities complicate causal inference in observational studies of transfusion effectiveness and safety. This guide addresses key challenges in observational transfusion research, with a focus on time-varying exposure, time-varying confounding, and treatment-confounder feedback. Using the example of donor sex and pregnancy history in relation to recipient mortality, we illustrate the strengths and limitations of commonly used analytical approaches. We compare restriction-based analyses, time-varying Cox regression, and inverse probability weighted marginal structural models using a large observational dataset of male transfusion recipients. In the applied example, restriction and conventional time-varying approaches suggested an increased mortality risk associated with transfusion of red blood cells from ever-pregnant female donors compared with male-only donors (hazard ratio [HR] 1.22; 95% CI 1.05-1.42 and HR 1.21; 95% CI 1.04-1.41, respectively). In contrast, inverse probability of treatment and censoring weighted analyses, which account for treatment-confounder feedback, showed no evidence of an association (HR 1.01; 95% CI 0.85-1.20). These findings demonstrate how conventional methods can yield biased estimates when complex longitudinal structures are not adequately handled. We provide practical guidance on study design, target trial emulation, and the use of g-methods, including a reproducible tutorial and example dataset, to support valid causal inference in clinical transfusion research. |
| 2025-12-15 | [Can AI Understand What We Cannot Say? Measuring Multilevel Alignment Through Abortion Stigma Across Cognitive, Interpersonal, and Structural Levels](http://arxiv.org/abs/2512.13142v1) | Anika Sharma, Malavika Mampally et al. | As large language models increasingly mediate stigmatized health decisions, their capacity to genuinely understand complex psychological and physiological phenomena remains poorly evaluated. Can AI understand what we cannot say? We investigate whether LLMs coherently represent abortion stigma across the cognitive, interpersonal, and structural levels where it operates. We systematically tested 627 demographically diverse personas across five leading LLMs using the validated Individual Level Abortion Stigma Scale (ILAS). Our multilevel analysis examined whether models coherently represent stigma at the cognitive level (self-judgment), interpersonal level (anticipated judgment and isolation), and structural level (community condemnation and disclosure patterns), as well as overall stigma. Models fail tests of genuine understanding across all levels. They overestimate interpersonal stigma while underestimating cognitive stigma, assume uniform community condemnation, introduce demographic biases absent from human validation data, miss the empirically validated stigma-secrecy relationship, and contradict themselves within theoretical constructs. These patterns reveal that current alignment approaches ensure appropriate language but not coherent multilevel understanding. This work provides empirical evidence that current LLMs lack coherent multilevel understanding of psychological and physiological constructs. AI safety in high-stakes contexts demands new approaches to design (multilevel coherence), evaluation (continuous auditing), governance and regulation (mandatory audits, accountability, deployment restrictions), and AI literacy in domains where understanding what people cannot say determines whether support helps or harms. |
| 2025-12-15 | [From Overfitting to Reliability: Introducing the Hierarchical Approximate Bayesian Neural Network](http://arxiv.org/abs/2512.13111v1) | Hayk Amirkhanian, Marco F. Huber | In recent years, neural networks have revolutionized various domains, yet challenges such as hyperparameter tuning and overfitting remain significant hurdles. Bayesian neural networks offer a framework to address these challenges by incorporating uncertainty directly into the model, yielding more reliable predictions, particularly for out-of-distribution data. This paper presents Hierarchical Approximate Bayesian Neural Network, a novel approach that uses a Gaussian-inverse-Wishart distribution as a hyperprior of the network's weights to increase both the robustness and performance of the model. We provide analytical representations for the predictive distribution and weight posterior, which amount to the calculation of the parameters of Student's t-distributions in closed form with linear complexity with respect to the number of weights. Our method demonstrates robust performance, effectively addressing issues of overfitting and providing reliable uncertainty estimates, particularly for out-of-distribution tasks. Experimental results indicate that HABNN not only matches but often outperforms state-of-the-art models, suggesting a promising direction for future applications in safety-critical environments. |
| 2025-12-15 | [LikeBench: Evaluating Subjective Likability in LLMs for Personalization](http://arxiv.org/abs/2512.13077v1) | Md Awsafur Rahman, Adam Gabrys et al. | A personalized LLM should remember user facts, apply them correctly, and adapt over time to provide responses that the user prefers. Existing LLM personalization benchmarks are largely centered on two axes: accurately recalling user information and accurately applying remembered information in downstream tasks. We argue that a third axis, likability, is both subjective and central to user experience, yet under-measured by current benchmarks. To measure likability holistically, we introduce LikeBench, a multi-session, dynamic evaluation framework that measures likability across multiple dimensions by how much an LLM can adapt over time to a user's preferences to provide more likable responses. In LikeBench, the LLMs engage in conversation with a simulated user and learn preferences only from the ongoing dialogue. As the interaction unfolds, models try to adapt to responses, and after each turn, they are evaluated for likability across seven dimensions by the same simulated user. To the best of our knowledge, we are the first to decompose likability into multiple diagnostic metrics: emotional adaptation, formality matching, knowledge adaptation, reference understanding, conversation length fit, humor fit, and callback, which makes it easier to pinpoint where a model falls short. To make the simulated user more realistic and discriminative, LikeBench uses fine-grained, psychologically grounded descriptive personas rather than the coarse high/low trait rating based personas used in prior work. Our benchmark shows that strong memory performance does not guarantee high likability: DeepSeek R1, with lower memory accuracy (86%, 17 facts/profile), outperformed Qwen3 by 28% on likability score despite Qwen3's higher memory accuracy (93%, 43 facts/profile). Even SOTA models like GPT-5 adapt well in short exchanges but show only limited robustness in longer, noisier interactions. |
| 2025-12-15 | [Bi-Erasing: A Bidirectional Framework for Concept Removal in Diffusion Models](http://arxiv.org/abs/2512.13039v1) | Hao Chen, Yiwei Wang et al. | Concept erasure, which fine-tunes diffusion models to remove undesired or harmful visual concepts, has become a mainstream approach to mitigating unsafe or illegal image generation in text-to-image models.However, existing removal methods typically adopt a unidirectional erasure strategy by either suppressing the target concept or reinforcing safe alternatives, making it difficult to achieve a balanced trade-off between concept removal and generation quality. To address this limitation, we propose a novel Bidirectional Image-Guided Concept Erasure (Bi-Erasing) framework that performs concept suppression and safety enhancement simultaneously. Specifically, based on the joint representation of text prompts and corresponding images, Bi-Erasing introduces two decoupled image branches: a negative branch responsible for suppressing harmful semantics and a positive branch providing visual guidance for safe alternatives. By jointly optimizing these complementary directions, our approach achieves a balance between erasure efficacy and generation usability. In addition, we apply mask-based filtering to the image branches to prevent interference from irrelevant content during the erasure process. Across extensive experiment evaluations, the proposed Bi-Erasing outperforms baseline methods in balancing concept removal effectiveness and visual fidelity. |
| 2025-12-15 | [Safe Control of Multi-Agent Systems with Minimal Communication](http://arxiv.org/abs/2512.13021v1) | Mo Yang, Jing Yu et al. | In many multi-agent systems, communication is limited by bandwidth, latency, and energy constraints. Designing controllers that achieve coordination and safety with minimal communication is critical for scalable and reliable deployment. This paper presents a method for designing controllers that minimize inter-agent communication in multi-agent systems while satisfying safety and coordination requirements, while conforming to communication delay constraints. The control synthesis problem is cast as a rank minimization problem, where a convex relaxation is obtained via system level synthesis. Simulation results on various tasks, including trajectory tracking with relative and heterogeneous sensing, demonstrate that the proposed method significantly reduces inter-agent transmission compared to baseline approaches. |
| 2025-12-15 | [Large Language Models for Power System Applications: A Comprehensive Literature Survey](http://arxiv.org/abs/2512.13004v1) | Muhammad Sarwar, Muhammad Rizwan et al. | This comprehensive literature review examines the emerging applications of Large Language Models (LLMs) in power system engineering. Through a systematic analysis of recent research published between 2020 and 2025, we explore how LLMs are being integrated into various aspects of power system operations, planning, and management. The review covers key application areas including fault diagnosis, load forecasting, cybersecurity, control and optimization, system planning, simulation, and knowledge management. Our findings indicate that while LLMs show promising potential in enhancing power system operations through their advanced natural language processing and reasoning capabilities, significant challenges remain in their practical implementation. These challenges include limited domain-specific training data, concerns about reliability and safety in critical infrastructure, and the need for enhanced explainability. The review also highlights emerging trends such as the development of power system-specific LLMs and hybrid approaches combining LLMs with traditional power engineering methods. We identify crucial research directions for advancing the field, including the development of specialized architectures, improved security frameworks, and enhanced integration with existing power system tools. This survey provides power system researchers and practitioners with a comprehensive overview of the current state of LLM applications in the field and outlines future pathways for research and development. |
| 2025-12-15 | [Cisco Integrated AI Security and Safety Framework Report](http://arxiv.org/abs/2512.12921v1) | Amy Chang, Tiffany Saade et al. | Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements. |
| 2025-12-15 | [CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs](http://arxiv.org/abs/2512.12914v1) | Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao et al. | Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains. |

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



