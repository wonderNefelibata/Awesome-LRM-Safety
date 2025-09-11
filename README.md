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
| 2025-09-10 | [A Survey of Reinforcement Learning for Large Reasoning Models](http://arxiv.org/abs/2509.08827v1) | Kaiyan Zhang, Yuxin Zuo et al. | In this paper, we survey recent advances in Reinforcement Learning (RL) for reasoning with Large Language Models (LLMs). RL has achieved remarkable success in advancing the frontier of LLM capabilities, particularly in addressing complex logical tasks such as mathematics and coding. As a result, RL has emerged as a foundational methodology for transforming LLMs into LRMs. With the rapid progress of the field, further scaling of RL for LRMs now faces foundational challenges not only in computational resources but also in algorithm design, training data, and infrastructure. To this end, it is timely to revisit the development of this domain, reassess its trajectory, and explore strategies to enhance the scalability of RL toward Artificial SuperIntelligence (ASI). In particular, we examine research applying RL to LLMs and LRMs for reasoning abilities, especially since the release of DeepSeek-R1, including foundational components, core problems, training resources, and downstream applications, to identify future opportunities and directions for this rapidly evolving area. We hope this review will promote future research on RL for broader reasoning models. Github: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs |
| 2025-09-10 | [RoboChemist: Long-Horizon and Safety-Compliant Robotic Chemical Experimentation](http://arxiv.org/abs/2509.08820v1) | Zongzheng Zhang, Chenghao Yue et al. | Robotic chemists promise to both liberate human experts from repetitive tasks and accelerate scientific discovery, yet remain in their infancy. Chemical experiments involve long-horizon procedures over hazardous and deformable substances, where success requires not only task completion but also strict compliance with experimental norms. To address these challenges, we propose \textit{RoboChemist}, a dual-loop framework that integrates Vision-Language Models (VLMs) with Vision-Language-Action (VLA) models. Unlike prior VLM-based systems (e.g., VoxPoser, ReKep) that rely on depth perception and struggle with transparent labware, and existing VLA systems (e.g., RDT, pi0) that lack semantic-level feedback for complex tasks, our method leverages a VLM to serve as (1) a planner to decompose tasks into primitive actions, (2) a visual prompt generator to guide VLA models, and (3) a monitor to assess task success and regulatory compliance. Notably, we introduce a VLA interface that accepts image-based visual targets from the VLM, enabling precise, goal-conditioned control. Our system successfully executes both primitive actions and complete multi-step chemistry protocols. Results show 23.57% higher average success rate and a 0.298 average increase in compliance rate over state-of-the-art VLA baselines, while also demonstrating strong generalization to objects and tasks. |
| 2025-09-10 | [Merge-of-Thought Distillation](http://arxiv.org/abs/2509.08814v1) | Zhanming Shen, Zeyu Qin et al. | Efficient reasoning distillation for long chain-of-thought (CoT) models is increasingly constrained by the assumption of a single oracle teacher, despite practical availability of multiple candidate teachers and growing CoT corpora. We revisit teacher selection and observe that different students have different "best teachers," and even for the same student the best teacher can vary across datasets. Therefore, to unify multiple teachers' reasoning abilities into student with overcoming conflicts among various teachers' supervision, we propose Merge-of-Thought Distillation (MoT), a lightweight framework that alternates between teacher-specific supervised fine-tuning branches and weight-space merging of the resulting student variants. On competition math benchmarks, using only about 200 high-quality CoT samples, applying MoT to a Qwen3-14B student surpasses strong models including DEEPSEEK-R1, QWEN3-30B-A3B, QWEN3-32B, and OPENAI-O1, demonstrating substantial gains. Besides, MoT consistently outperforms the best single-teacher distillation and the naive multi-teacher union, raises the performance ceiling while mitigating overfitting, and shows robustness to distribution-shifted and peer-level teachers. Moreover, MoT reduces catastrophic forgetting, improves general reasoning beyond mathematics and even cultivates a better teacher, indicating that consensus-filtered reasoning features transfer broadly. These results position MoT as a simple, scalable route to efficiently distilling long CoT capabilities from diverse teachers into compact students. |
| 2025-09-10 | [Extended Version: Security and Privacy Perceptions of Pakistani Facebook Matrimony Group Users](http://arxiv.org/abs/2509.08782v1) | Mah Jan Dorazahi, Deepthi Mungara et al. | In Pakistan, where dating apps are subject to censorship, Facebook matrimony groups -- also referred to as marriage groups -- serve as alternative virtual spaces for members to search for potential life partners. To participate in these groups, members often share sensitive personal information such as photos, addresses, and phone numbers, which exposes them to risks such as fraud, blackmail, and identity theft. To better protect users of Facebook matrimony groups, we need to understand aspects related to user safety, such as how users perceive risks, what influences their trust in sharing personal information, and how they navigate security and privacy concerns when seeking potential partners online. In this study, through 23 semi-structured interviews, we explore how Pakistani users of Facebook matrimony groups perceive and navigate risks of sharing personal information, and how cultural norms and expectations influence their behavior in these groups.   We find elevated privacy concerns among participants, leading them to share limited personal information and creating mistrust among potential partners. Many also expressed concerns about the authenticity of profiles and major security risks, such as identity theft, harassment, and social judgment. Our work highlights the challenges of safely navigating Facebook matrimony groups in Pakistan and offers recommendations for such as implementing stronger identity verification by group admins, enforcing stricter cybersecurity laws, clear platform guidelines to ensure accountability, and technical feature enhancements -- including restricting screenshots, picture downloads, and implementing anonymous chats -- to protect user data and build trust. |
| 2025-09-10 | [Joint Model-based Model-free Diffusion for Planning with Constraints](http://arxiv.org/abs/2509.08775v1) | Wonsuhk Jung, Utkarsh A. Mishra et al. | Model-free diffusion planners have shown great promise for robot motion planning, but practical robotic systems often require combining them with model-based optimization modules to enforce constraints, such as safety. Naively integrating these modules presents compatibility challenges when diffusion's multi-modal outputs behave adversarially to optimization-based modules. To address this, we introduce Joint Model-based Model-free Diffusion (JM2D), a novel generative modeling framework. JM2D formulates module integration as a joint sampling problem to maximize compatibility via an interaction potential, without additional training. Using importance sampling, JM2D guides modules outputs based only on evaluations of the interaction potential, thus handling non-differentiable objectives commonly arising from non-convex optimization modules. We evaluate JM2D via application to aligning diffusion planners with safety modules on offline RL and robot manipulation. JM2D significantly improves task performance compared to conventional safety filters without sacrificing safety. Further, we show that conditional generation is a special case of JM2D and elucidate key design choices by comparing with SOTA gradient-based and projection-based diffusion planners. More details at: https://jm2d-corl25.github.io/. |
| 2025-09-10 | [Reconfigurable Holographic Surfaces and Near Field Communication for Non-Terrestrial Networks: Potential and Challenges](http://arxiv.org/abs/2509.08770v1) | Muhammad Ali Jamshed, Muhammad Ahmed Mohsin et al. | To overcome the challenges of ultra-low latency, ubiquitous coverage, and soaring data rates, this article presents a combined use of Near Field Communication (NFC) and Reconfigurable Holographic Surfaces (RHS) for Non-Terrestrial Networks (NTN). A system architecture has been presented, which shows that the integration of RHS with NTN platforms such as satellites, High Altitute Platform Stations (HAPS), and Uncrewed Aerial Vehicles (UAV) can achieve precise beamforming and intelligent wavefront control in near-field regions, enhancing Energy Efficiency (EE), spectral utilization, and spatial resolution. Moreover, key applications, challenges, and future directions have been identified to fully adopt this integration. In addition, a use case analysis has been presented to improve the EE of the system in a public safety use case scenario, further strengthening the UAV-RHS fusion. |
| 2025-09-10 | [X-Teaming Evolutionary M2S: Automated Discovery of Multi-turn to Single-turn Jailbreak Templates](http://arxiv.org/abs/2509.08729v1) | Hyunjun Kim, Junwoo Ha et al. | Multi-turn-to-single-turn (M2S) compresses iterative red-teaming into one structured prompt, but prior work relied on a handful of manually written templates. We present X-Teaming Evolutionary M2S, an automated framework that discovers and optimizes M2S templates through language-model-guided evolution. The system pairs smart sampling from 12 sources with an LLM-as-judge inspired by StrongREJECT and records fully auditable logs.   Maintaining selection pressure by setting the success threshold to $\theta = 0.70$, we obtain five evolutionary generations, two new template families, and 44.8% overall success (103/230) on GPT-4.1. A balanced cross-model panel of 2,500 trials (judge fixed) shows that structural gains transfer but vary by target; two models score zero at the same threshold. We also find a positive coupling between prompt length and score, motivating length-aware judging.   Our results demonstrate that structure-level search is a reproducible route to stronger single-turn probes and underscore the importance of threshold calibration and cross-model evaluation. Code, configurations, and artifacts are available at https://github.com/hyunjun1121/M2S-x-teaming. |
| 2025-09-10 | [Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing](http://arxiv.org/abs/2509.08721v1) | Jeffrey Amico, Gabriel Passamani Andrade et al. | Post-training language models (LMs) with reinforcement learning (RL) can enhance their complex reasoning capabilities without supervised fine-tuning, as demonstrated by DeepSeek-R1-Zero. However, effectively utilizing RL for LMs requires significant parallelization to scale-up inference, which introduces non-trivial technical challenges (e.g. latency, memory, and reliability) alongside ever-growing financial costs. We present Swarm sAmpling Policy Optimization (SAPO), a fully decentralized and asynchronous RL post-training algorithm. SAPO is designed for decentralized networks of heterogenous compute nodes, where each node manages its own policy model(s) while "sharing" rollouts with others in the network; no explicit assumptions about latency, model homogeneity, or hardware are required and nodes can operate in silo if desired. As a result, the algorithm avoids common bottlenecks in scaling RL post-training while also allowing (and even encouraging) new possibilities. By sampling rollouts "shared" across the network, it enables "Aha moments" to propagate, thereby bootstrapping the learning process. In this paper we show SAPO achieved cumulative reward gains of up to 94% in controlled experiments. We also share insights from tests on a network with thousands of nodes contributed by Gensyn community members running the algorithm on diverse hardware and models during an open-source demo. |
| 2025-09-10 | [Generative AI as a Safety Net for Survey Question Refinement](http://arxiv.org/abs/2509.08702v1) | Erica Ann Metheney, Lauren Yehle | Writing survey questions that easily and accurately convey their intent to a variety of respondents is a demanding and high-stakes task. Despite the extensive literature on best practices, the number of considerations to keep in mind is vast and even small errors can render collected data unusable for its intended purpose. The process of drafting initial questions, checking for known sources of error, and developing solutions to those problems requires considerable time, expertise, and financial resources. Given the rising costs of survey implementation and the critical role that polls play in media, policymaking, and research, it is vital that we utilize all available tools to protect the integrity of survey data and the financial investments made to obtain it. Since its launch in 2022, ChatGPT and other generative AI model platforms have been integrated into everyday life processes and workflows, particularly pertaining to text revision. While many researchers have begun exploring how generative AI may assist with questionnaire design, we have implemented a prompt experiment to systematically test what kind of feedback on survey questions an average ChatGPT user can expect. Results from our zero--shot prompt experiment, which randomized the version of ChatGPT and the persona given to the model, shows that generative AI is a valuable tool today, even for an average AI user, and suggests that AI will play an increasingly prominent role in the evolution of survey development best practices as precise tools are developed. |
| 2025-09-10 | [A Local-Phase Framework for the BaTi_{1-x}Zr_xO_3$ Phase Diagram: From Ferroelectricity to Dipolar Glass](http://arxiv.org/abs/2509.08662v1) | M. Sepliarsky, F. Aquistapace et al. | We apply a first-principles-based atomistic model to investigate the BaTi(1-x)Zr(x)O3 phase diagram, focusing on both macroscopic and local structural changes. Our approach, which combines molecular dynamics with machine learning techniques, accurately captures the influence of Ti and Zr cations on their local environment and its evolution with composition and temperature. The computed phase diagram shows excellent agreement with existing experimental and theoretical data. Beyond reproducing known results, our analysis reveals that the behavior of the solid solution across different compositions and temperatures can be understood in terms of coexisting Ti cells with different symmetries, whose stability depends on the local B-site configuration. This local-phase-based approach provides a unified description of the distinct regions of the solid solution, including ferroelectric, relaxor, and dipolar glass phases, and captures the continuous evolution from one regime to another. Our findings demonstrate how atomic-level distortions drive the complex macroscopic behavior of the material. |
| 2025-09-10 | [AutoODD: Agentic Audits via Bayesian Red Teaming in Black-Box Models](http://arxiv.org/abs/2509.08638v1) | Rebecca Martin, Jay Patrikar et al. | Specialized machine learning models, regardless of architecture and training, are susceptible to failures in deployment. With their increasing use in high risk situations, the ability to audit these models by determining their operational design domain (ODD) is crucial in ensuring safety and compliance. However, given the high-dimensional input spaces, this process often requires significant human resources and domain expertise. To alleviate this, we introduce \coolname, an LLM-Agent centric framework for automated generation of semantically relevant test cases to search for failure modes in specialized black-box models. By leveraging LLM-Agents as tool orchestrators, we aim to fit a uncertainty-aware failure distribution model on a learned text-embedding manifold by projecting the high-dimension input space to low-dimension text-embedding latent space. The LLM-Agent is tasked with iteratively building the failure landscape by leveraging tools for generating test-cases to probe the model-under-test (MUT) and recording the response. The agent also guides the search using tools to probe uncertainty estimate on the low dimensional manifold. We demonstrate this process in a simple case using models trained with missing digits on the MNIST dataset and in the real world setting of vision-based intruder detection for aerial vehicles. |
| 2025-09-10 | [Memshare: Memory Sharing for Multicore Computation in R with an Application to Feature Selection by Mutual Information using PDE](http://arxiv.org/abs/2509.08632v1) | Michael C. Thrun, Julian Märte | We present memshare\footnote{The Software package is published as a CRAN package under https://CRAN.R-project.org/package=memshare, a package that enables shared memory multicore computation in R by allocating buffers in C++ shared memory and exposing them to R through ALTREP views. We compare memshare to SharedObject (Bioconductor) discuss semantics and safety, and report a 2x speedup over SharedObject with no additional resident memory in a column wise apply benchmark. Finally, we illustrate a downstream analytics use case: feature selection by mutual information in which densities are estimated per feature via Pareto Density Estimation (PDE). The analytical use-case is an RNA seq dataset consisting of N=10,446 cases and d=19,637 gene expressions requiring roughly n_threads * 10GB of memory in the case of using parallel R sessions. Such and larger use-cases are common in big data analytics and make R feel limiting sometimes which is mitigated by the addition of the library presented in this work. |
| 2025-09-10 | [AdsQA: Towards Advertisement Video Understanding](http://arxiv.org/abs/2509.08621v1) | Xinwei Long, Kai Tian et al. | Large language models (LLMs) have taken a great step towards AGI. Meanwhile, an increasing number of domain-specific problems such as math and programming boost these general-purpose models to continuously evolve via learning deeper expertise. Now is thus the time further to extend the diversity of specialized applications for knowledgeable LLMs, though collecting high quality data with unexpected and informative tasks is challenging. In this paper, we propose to use advertisement (ad) videos as a challenging test-bed to probe the ability of LLMs in perceiving beyond the objective physical content of common visual domain. Our motivation is to take full advantage of the clue-rich and information-dense ad videos' traits, e.g., marketing logic, persuasive strategies, and audience engagement. Our contribution is three-fold: (1) To our knowledge, this is the first attempt to use ad videos with well-designed tasks to evaluate LLMs. We contribute AdsQA, a challenging ad Video QA benchmark derived from 1,544 ad videos with 10,962 clips, totaling 22.7 hours, providing 5 challenging tasks. (2) We propose ReAd-R, a Deepseek-R1 styled RL model that reflects on questions, and generates answers via reward-driven optimization. (3) We benchmark 14 top-tier LLMs on AdsQA, and our \texttt{ReAd-R}~achieves the state-of-the-art outperforming strong competitors equipped with long-chain reasoning capabilities by a clear margin. |
| 2025-09-10 | [Real-time CBCT reconstructions using Krylov solvers in repeated scanning procedures](http://arxiv.org/abs/2509.08574v1) | Fred Vickers Hastings, S M Ragib Shahriar Islam et al. | This work introduces a new efficient iterative solver for the reconstruction of real-time cone-beam computed tomography (CBCT), which is based on the Prior Image Constrained Compressed Sensing (PICCS) regularization and leverages the efficiency of Krylov subspace methods. In particular, we focus on the setting where a sequence of under-sampled CT scans are taken on the same object with only local changes (e.g. changes in a tumour size or the introduction of a surgical tool). This is very common, for example, in image-guided surgery, where the amount of measurements is limited to ensure the safety of the patient. In this case, we can also typically assume that a (good) initial reconstruction for the solution exists, coming from a previously over-sampled scan, so we can use this information to aid the subsequent reconstructions. The effectiveness of this method is demonstrated in both a synthetic scan and using real CT data, where it can be observed that the PICCS framework is very effective for the reduction of artifacts, and that the new method is faster than other common alternatives used in the same setting. |
| 2025-09-10 | [HumanAgencyBench: Scalable Evaluation of Human Agency Support in AI Assistants](http://arxiv.org/abs/2509.08494v1) | Benjamin Sturgeon, Daniel Samuelson et al. | As humans delegate more tasks and decisions to artificial intelligence (AI), we risk losing control of our individual and collective futures. Relatively simple algorithmic systems already steer human decision-making, such as social media feed algorithms that lead people to unintentionally and absent-mindedly scroll through engagement-optimized content. In this paper, we develop the idea of human agency by integrating philosophical and scientific theories of agency with AI-assisted evaluation methods: using large language models (LLMs) to simulate and validate user queries and to evaluate AI responses. We develop HumanAgencyBench (HAB), a scalable and adaptive benchmark with six dimensions of human agency based on typical AI use cases. HAB measures the tendency of an AI assistant or agent to Ask Clarifying Questions, Avoid Value Manipulation, Correct Misinformation, Defer Important Decisions, Encourage Learning, and Maintain Social Boundaries. We find low-to-moderate agency support in contemporary LLM-based assistants and substantial variation across system developers and dimensions. For example, while Anthropic LLMs most support human agency overall, they are the least supportive LLMs in terms of Avoid Value Manipulation. Agency support does not appear to consistently result from increasing LLM capabilities or instruction-following behavior (e.g., RLHF), and we encourage a shift towards more robust safety and alignment targets. |
| 2025-09-10 | [Too Helpful, Too Harmless, Too Honest or Just Right?](http://arxiv.org/abs/2509.08486v1) | Gautam Siddharth Kashyap, Mark Dras et al. | Large Language Models (LLMs) exhibit strong performance across a wide range of NLP tasks, yet aligning their outputs with the principles of Helpfulness, Harmlessness, and Honesty (HHH) remains a persistent challenge. Existing methods often optimize for individual alignment dimensions in isolation, leading to trade-offs and inconsistent behavior. While Mixture-of-Experts (MoE) architectures offer modularity, they suffer from poorly calibrated routing, limiting their effectiveness in alignment tasks. We propose TrinityX, a modular alignment framework that incorporates a Mixture of Calibrated Experts (MoCaE) within the Transformer architecture. TrinityX leverages separately trained experts for each HHH dimension, integrating their outputs through a calibrated, task-adaptive routing mechanism that combines expert signals into a unified, alignment-aware representation. Extensive experiments on three standard alignment benchmarks-Alpaca (Helpfulness), BeaverTails (Harmlessness), and TruthfulQA (Honesty)-demonstrate that TrinityX outperforms strong baselines, achieving relative improvements of 32.5% in win rate, 33.9% in safety score, and 28.4% in truthfulness. In addition, TrinityX reduces memory usage and inference latency by over 40% compared to prior MoE-based approaches. Ablation studies highlight the importance of calibrated routing, and cross-model evaluations confirm TrinityX's generalization across diverse LLM backbones. |
| 2025-09-10 | [LD-ViCE: Latent Diffusion Model for Video Counterfactual Explanations](http://arxiv.org/abs/2509.08422v1) | Payal Varshney, Adriano Lucieri et al. | Video-based AI systems are increasingly adopted in safety-critical domains such as autonomous driving and healthcare. However, interpreting their decisions remains challenging due to the inherent spatiotemporal complexity of video data and the opacity of deep learning models. Existing explanation techniques often suffer from limited temporal coherence, insufficient robustness, and a lack of actionable causal insights. Current counterfactual explanation methods typically do not incorporate guidance from the target model, reducing semantic fidelity and practical utility. We introduce Latent Diffusion for Video Counterfactual Explanations (LD-ViCE), a novel framework designed to explain the behavior of video-based AI models. Compared to previous approaches, LD-ViCE reduces the computational costs of generating explanations by operating in latent space using a state-of-the-art diffusion model, while producing realistic and interpretable counterfactuals through an additional refinement step. Our experiments demonstrate the effectiveness of LD-ViCE across three diverse video datasets, including EchoNet-Dynamic (cardiac ultrasound), FERV39k (facial expression), and Something-Something V2 (action recognition). LD-ViCE outperforms a recent state-of-the-art method, achieving an increase in R2 score of up to 68% while reducing inference time by half. Qualitative analysis confirms that LD-ViCE generates semantically meaningful and temporally coherent explanations, offering valuable insights into the target model behavior. LD-ViCE represents a valuable step toward the trustworthy deployment of AI in safety-critical domains. |
| 2025-09-10 | [An Iterative LLM Framework for SIBT utilizing RAG-based Adaptive Weight Optimization](http://arxiv.org/abs/2509.08407v1) | Zhuo Xiao, Qinglong Yao et al. | Seed implant brachytherapy (SIBT) is an effective cancer treatment modality; however, clinical planning often relies on manual adjustment of objective function weights, leading to inefficiencies and suboptimal results. This study proposes an adaptive weight optimization framework for SIBT planning, driven by large language models (LLMs). A locally deployed DeepSeek-R1 LLM is integrated with an automatic planning algorithm in an iterative loop. Starting with fixed weights, the LLM evaluates plan quality and recommends new weights in the next iteration. This process continues until convergence criteria are met, after which the LLM conducts a comprehensive evaluation to identify the optimal plan. A clinical knowledge base, constructed and queried via retrieval-augmented generation (RAG), enhances the model's domain-specific reasoning. The proposed method was validated on 23 patient cases, showing that the LLM-assisted approach produces plans that are comparable to or exceeding clinically approved and fixed-weight plans, in terms of dose homogeneity for the clinical target volume (CTV) and sparing of organs at risk (OARs). The study demonstrates the potential use of LLMs in SIBT planning automation. |
| 2025-09-10 | [Safety Factories -- a Manifesto](http://arxiv.org/abs/2509.08285v1) | Carmen Cârlan, Daniel Ratiu et al. | Modern cyber-physical systems are operated by complex software that increasingly takes over safety-critical functions. Software enables rapid iterations and continuous delivery of new functionality that meets the ever-changing expectations of users. As high-speed development requires discipline, rigor, and automation, software factories are used. These entail methods and tools used for software development, such as build systems and pipelines. To keep up with the rapid evolution of software, we need to bridge the disconnect in methods and tools between software development and safety engineering today. We need to invest more in formality upfront - capturing safety work products in semantically rich models that are machine-processable, defining automatic consistency checks, and automating the generation of documentation - to benefit later. Transferring best practices from software to safety engineering is worth exploring. We advocate for safety factories, which integrate safety tooling and methods into software development pipelines. |
| 2025-09-10 | [Enhancing 6G Network Security and Incident Response through Integrated VNF and SDN Technologies](http://arxiv.org/abs/2509.08274v1) | Abdul Razaque, Abitkhanova Zhadyra Abitkhanovna | Low-speed internet can negatively affect incident response in a number of ways, including decreased teamwork, delayed detection, inefficient action, and elevated risk. Delayed data acquisition and processing may result from inadequate internet connectivity, hindering security teams' ability to obtain the necessary information for timely and effective responses. Each of these factors may augment the organization's susceptibility to security incidents and their subsequent ramifications. This article establishes a virtual network function service delivery network (VNFSDN) through the integration of virtual network function (VNF) and software-defined networking (SDN) technologies. The VNFSDN approach enhances network security effectiveness and efficiency while reducing the danger of breaches. This method assists security services in rapidly assessing vast quantities of data generated by 6G networks. VNFSDN adapts dynamically to changing safety requirements and connection conditions through the use of SDN and VNF. This flexibility enables enterprises to mitigate or halt the impact of cyberattacks by swiftly identifying and addressing security threats. The VNFSDN enhances network resilience, allowing operators to proactively mitigate possible security attacks and minimize downtime. The incorporation of machine learning and artificial intelligence into VNFSDN can significantly improve network security and threat detection capabilities. The VNFSDN integrates VNF and SDN technologies to deliver security services that analyze vast quantities of 6G data in real time. As security requirements and network conditions evolve, it adapts dynamically to enhance network resilience and facilitate proactive threat detection. |

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



