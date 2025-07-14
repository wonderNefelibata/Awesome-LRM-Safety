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
| 2025-07-11 | [Optimistic Exploration for Risk-Averse Constrained Reinforcement Learning](http://arxiv.org/abs/2507.08793v1) | James McCarthy, Radu Marinescu et al. | Risk-averse Constrained Reinforcement Learning (RaCRL) aims to learn policies that minimise the likelihood of rare and catastrophic constraint violations caused by an environment's inherent randomness. In general, risk-aversion leads to conservative exploration of the environment which typically results in converging to sub-optimal policies that fail to adequately maximise reward or, in some cases, fail to achieve the goal. In this paper, we propose an exploration-based approach for RaCRL called Optimistic Risk-averse Actor Critic (ORAC), which constructs an exploratory policy by maximising a local upper confidence bound of the state-action reward value function whilst minimising a local lower confidence bound of the risk-averse state-action cost value function. Specifically, at each step, the weighting assigned to the cost value is increased or decreased if it exceeds or falls below the safety constraint value. This way the policy is encouraged to explore uncertain regions of the environment to discover high reward states whilst still satisfying the safety constraints. Our experimental results demonstrate that the ORAC approach prevents convergence to sub-optimal policies and improves significantly the reward-cost trade-off in various continuous control tasks such as Safety-Gymnasium and a complex building energy management environment CityLearn. |
| 2025-07-11 | [A Personalised Formal Verification Framework for Monitoring Activities of Daily Living of Older Adults Living Independently in Their Homes](http://arxiv.org/abs/2507.08701v1) | Ricardo Contreras, Filip Smola et al. | There is an imperative need to provide quality of life to a growing population of older adults living independently. Personalised solutions that focus on the person and take into consideration their preferences and context are key. In this work, we introduce a framework for representing and reasoning about the Activities of Daily Living of older adults living independently at home. The framework integrates data from sensors and contextual information that aggregates semi-structured interviews, home layouts and sociological observations from the participants. We use these data to create formal models, personalised for each participant according to their preferences and context. We formulate requirements that are specific to each individual as properties encoded in Linear Temporal Logic and use a model checker to verify whether each property is satisfied by the model. When a property is violated, a counterexample is generated giving the cause of the violation. We demonstrate the framework's generalisability by applying it to different participants, highlighting its potential to enhance the safety and well-being of older adults ageing in place. |
| 2025-07-11 | [A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1](http://arxiv.org/abs/2507.08621v1) | Marcin Pietroń, Rafał Olszowski et al. | Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings. |
| 2025-07-11 | [Agentic Large Language Models for Conceptual Systems Engineering and Design](http://arxiv.org/abs/2507.08619v1) | Soheyl Massoudi, Mark Fuge | Early-stage engineering design involves complex, iterative reasoning, yet existing large language model (LLM) workflows struggle to maintain task continuity and generate executable models. We evaluate whether a structured multi-agent system (MAS) can more effectively manage requirements extraction, functional decomposition, and simulator code generation than a simpler two-agent system (2AS). The target application is a solar-powered water filtration system as described in a cahier des charges. We introduce the Design-State Graph (DSG), a JSON-serializable representation that bundles requirements, physical embodiments, and Python-based physics models into graph nodes. A nine-role MAS iteratively builds and refines the DSG, while the 2AS collapses the process to a Generator-Reflector loop. Both systems run a total of 60 experiments (2 LLMs - Llama 3.3 70B vs reasoning-distilled DeepSeek R1 70B x 2 agent configurations x 3 temperatures x 5 seeds). We report a JSON validity, requirement coverage, embodiment presence, code compatibility, workflow completion, runtime, and graph size. Across all runs, both MAS and 2AS maintained perfect JSON integrity and embodiment tagging. Requirement coverage remained minimal (less than 20\%). Code compatibility peaked at 100\% under specific 2AS settings but averaged below 50\% for MAS. Only the reasoning-distilled model reliably flagged workflow completion. Powered by DeepSeek R1 70B, the MAS generated more granular DSGs (average 5-6 nodes) whereas 2AS mode-collapsed. Structured multi-agent orchestration enhanced design detail. Reasoning-distilled LLM improved completion rates, yet low requirements and fidelity gaps in coding persisted. |
| 2025-07-11 | [STRAP: Spatial-Temporal Risk-Attentive Vehicle Trajectory Prediction for Autonomous Driving](http://arxiv.org/abs/2507.08563v1) | Xinyi Ning, Zilin Bian et al. | Accurate vehicle trajectory prediction is essential for ensuring safety and efficiency in fully autonomous driving systems. While existing methods primarily focus on modeling observed motion patterns and interactions with other vehicles, they often neglect the potential risks posed by the uncertain or aggressive behaviors of surrounding vehicles. In this paper, we propose a novel spatial-temporal risk-attentive trajectory prediction framework that incorporates a risk potential field to assess perceived risks arising from behaviors of nearby vehicles. The framework leverages a spatial-temporal encoder and a risk-attentive feature fusion decoder to embed the risk potential field into the extracted spatial-temporal feature representations for trajectory prediction. A risk-scaled loss function is further designed to improve the prediction accuracy of high-risk scenarios, such as short relative spacing. Experiments on the widely used NGSIM and HighD datasets demonstrate that our method reduces average prediction errors by 4.8% and 31.2% respectively compared to state-of-the-art approaches, especially in high-risk scenarios. The proposed framework provides interpretable, risk-aware predictions, contributing to more robust decision-making for autonomous driving systems. |
| 2025-07-11 | [Computing Floating-Point Errors by Injecting Perturbations](http://arxiv.org/abs/2507.08467v1) | Youshuai Tan, Zhanwei Zhang et al. | Floating-point programs form the foundation of modern science and engineering, providing the essential computational framework for a wide range of applications, such as safety-critical systems, aerospace engineering, and financial analysis. Floating-point errors can lead to severe consequences. Although floating-point errors widely exist, only a subset of inputs may trigger significant errors in floating-point programs. Therefore, it is crucial to determine whether a given input could produce such errors. Researchers tend to take the results of high-precision floating-point programs as oracles for detecting floating-point errors, which introduces two main limitations: (1) difficulty of implementation and (2) prolonged execution time. The two recent tools, ATOMU and FPCC, can partially address these issues. However, ATOMU suffers from false positives; while FPCC, though eliminating false positives, operates at a considerably slower speed.   To address these two challenges, we propose a novel approach named PI-detector to computing floating-point errors effectively and efficiently. Our approach is based on the observation that floating-point errors stem from large condition numbers in atomic operations (such as addition and subtraction), which then propagate and accumulate. PI-detector injects small perturbations into the operands of individual atomic operations within the program and compares the outcomes of the original program with the perturbed version to compute floating-point errors. We evaluate PI-detector with datasets from ATOMU and HSED, as well as a complex linear system-solving program. Experimental results demonstrate that PI-detector can perform efficient and accurate floating-point error computation. |
| 2025-07-11 | [Modeling Wallet-Level Behavioral Shifts Post-FTX Collapse: An XAI-Driven GLM Study on Ethereum Transactions](http://arxiv.org/abs/2507.08455v1) | Benjamin Gillen, Rashmi Ranjan Bhuyan et al. | The Ethereum blockchain plays a central role in the broader cryptocurrency ecosystem, enabling a wide range of financial activity through the use of smart contracts. This paper investigates how individual Ethereum wallets responded to the collapse of FTX, one of the largest centralized cryptocurrency exchanges. Moving beyond price-based event studies, we adopt a bottom-up approach using granular wallet-level data. We construct a representative sample of Ethereum addresses and analyze their transaction behavior before and after the collapse using an explainable artificial intelligence (XAI) framework. Our proposed framework addresses data scarcity in high-resolution wallet-level daily transactions by employing a calibrated zero-inflated generalized linear fixed effects model. Our analysis quantifies distinct shifts in transaction intensity and stablecoin usage, highlighting a flight to safety within the ecosystem. These findings underscore the value of a bottom-up methodology for quantifying the user-level impact of blockchain-based shocks, offering insights beyond traditional price-level analysis through wallet-level data. |
| 2025-07-11 | [Prediction of Lane Change Intentions of Human Drivers using an LSTM, a CNN and a Transformer](http://arxiv.org/abs/2507.08365v1) | Francesco De Cristofaro, Felix Hofbaur et al. | Lane changes of preceding vehicles have a great impact on the motion planning of automated vehicles especially in complex traffic situations. Predicting them would benefit the public in terms of safety and efficiency. While many research efforts have been made in this direction, few concentrated on predicting maneuvers within a set time interval compared to predicting at a set prediction time. In addition, there exist a lack of comparisons between different architectures to try to determine the best performing one and to assess how to correctly choose the input for such models. In this paper the structure of an LSTM, a CNN and a Transformer network are described and implemented to predict the intention of human drivers to perform a lane change. We show how the data was prepared starting from a publicly available dataset (highD), which features were used, how the networks were designed and finally we compare the results of the three networks with different configurations of input data. We found that transformer networks performed better than the other networks and was less affected by overfitting. The accuracy of the method spanned from $82.79\%$ to $96.73\%$ for different input configurations and showed overall good performances considering also precision and recall. |
| 2025-07-11 | [Improving gravitational wave search sensitivity with TIER: Trigger Inference using Extended strain Representation](http://arxiv.org/abs/2507.08318v1) | Digvijay Wadekar, Arush Pimpalkar et al. | We introduce a machine learning (ML) framework called $\texttt{TIER}$ for improving the sensitivity of gravitational wave search pipelines. Typically, search pipelines only use a small region of strain data in the vicinity of a candidate signal to construct the detection statistic. However, extended strain data ($\sim 10$ s) in the candidate's vicinity can also carry valuable complementary information. We show that this information can be efficiently captured by ML classifier models trained on sparse summary representation/features of the extended data. Our framework is easy to train and can be used with already existing candidates from any search pipeline, and without requiring expensive injection campaigns. Furthermore, the output of our model can be easily integrated into the detection statistic of a search pipeline. Using $\texttt{TIER}$ on triggers from the $\texttt{IAS-HM}$ pipeline, we find up to $\sim 20\%$ improvement in sensitive volume time in LIGO-Virgo-Kagra O3 data, with improvements concentrated in regions of high masses and unequal mass ratios. Applying our framework increases the significance of several near-threshold gravitational-wave candidates, especially in the pair-instability mass gap and intermediate-mass black hole (IMBH) ranges. |
| 2025-07-11 | [KAT-V1: Kwai-AutoThink Technical Report](http://arxiv.org/abs/2507.08297v1) | Zizheng Zhan, Ken Deng et al. | We present Kwaipilot-AutoThink (KAT), an open-source 40B large language model developed to address the overthinking problem in reasoning-intensive tasks, where an automatic thinking training paradigm is proposed to dynamically switch between reasoning and non-reasoning modes based on task complexity. Specifically, first, we construct the dual-regime dataset based on a novel tagging pipeline and a multi-agent synthesis strategy, and then we apply Multi-Token Prediction (MTP)-enhanced knowledge distillation, enabling efficient and fine-grained reasoning transfer with minimal pretraining cost. Besides, we implement a cold-start initialization strategy that introduces mode-selection priors using majority-vote signals and intent-aware prompting. Finally, we propose Step-SRPO, a reinforcement learning algorithm that incorporates intermediate supervision into the GRPO framework, offering structured guidance over both reasoning-mode selection and response accuracy. Extensive experiments across multiple benchmarks demonstrate that KAT consistently matches or even outperforms current state-of-the-art models, including DeepSeek-R1-0528 and Qwen3-235B-A22B, across a wide range of reasoning-intensive tasks while reducing token usage by up to approximately 30\%. Beyond academic evaluation, KAT has been successfully deployed in Kwaipilot (i.e., Kuaishou's internal coding assistant), and improves real-world development workflows with high accuracy, efficiency, and controllable reasoning behaviors. Moreover, we are actively training a 200B Mixture-of-Experts (MoE) with 40B activation parameters, where the early-stage results already demonstrate promising improvements in performance and efficiency, further showing the scalability of the AutoThink paradigm. |
| 2025-07-11 | [Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training](http://arxiv.org/abs/2507.08284v1) | Aleksei Ilin, Gor Matevosyan et al. | We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems. |
| 2025-07-11 | [Agent Safety Alignment via Reinforcement Learning](http://arxiv.org/abs/2507.08270v1) | Zeyang Sha, Hanling Tian et al. | The emergence of autonomous Large Language Model (LLM) agents capable of tool usage has introduced new safety risks that go beyond traditional conversational misuse. These agents, empowered to execute external functions, are vulnerable to both user-initiated threats (e.g., adversarial prompts) and tool-initiated threats (e.g., malicious outputs from compromised tools). In this paper, we propose the first unified safety-alignment framework for tool-using agents, enabling models to handle both channels of threat via structured reasoning and sandboxed reinforcement learning. We introduce a tri-modal taxonomy, including benign, malicious, and sensitive for both user prompts and tool responses, and define a policy-driven decision model. Our framework employs a custom-designed sandbox environment that simulates real-world tool execution and allows fine-grained reward shaping. Through extensive evaluations on public and self-built benchmarks, including Agent SafetyBench, InjecAgent, and BFCL, we demonstrate that our safety-aligned agents significantly improve resistance to security threats while preserving strong utility on benign tasks. Our results show that safety and effectiveness can be jointly optimized, laying the groundwork for trustworthy deployment of autonomous LLM agents. |
| 2025-07-11 | [Maneuver Detection via a Confidence Dominance Maneuver Indicator](http://arxiv.org/abs/2507.08234v1) | Xingyu Zhou, Roberto Armellin et al. | Accurate and efficient maneuver detection is critical for ensuring the safety and predictability of spacecraft trajectories. This paper presents a novel maneuver detection approach based on comparing the confidence levels associated with the orbital state estimation and the observation likelihood. First, a confidence-dominance maneuver indicator (CDMI) is proposed by setting a confidence level for the state estimation and computing the maximum likelihood of the observation and its confidence level. The CDMI then flag a maneuver when the observation's confidence level exceeds that of the state estimation, indicating that the observation is unlikely under the no-maneuver hypothesis while maintaining consistency with the prior state estimation confidence. To efficiently compute the maximum likelihood of the observation and obtain the CDMI, a recursive polynomial optimization method is developed, taking advantage of convex optimization and polynomial approximation. In addition, an integrated CDMI approach is developed to eliminate the need to manually select the state confidence level. The integrated CDMI approach maintains high detection accuracy while simultaneously providing an indication of maneuver likelihood, thereby enhancing robustness and practical applicability. The performance of the proposed CDMI-based maneuver detection approaches is evaluated against an optimal control distance metric and two mixture-based approaches. The simulation results demonstrate that the proposed integrated CDMI approach can achieve up to 99.33\% detection accuracy, at least 10% higher than the competing methods, while substantially reducing computational costs. |
| 2025-07-10 | [A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking](http://arxiv.org/abs/2507.08207v1) | Zhengye Han, Quanyan Zhu | As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking. |
| 2025-07-10 | [AmpLyze: A Deep Learning Model for Predicting the Hemolytic Concentration](http://arxiv.org/abs/2507.08162v1) | Peng Qiu, Hanqi Feng et al. | Red-blood-cell lysis (HC50) is the principal safety barrier for antimicrobial-peptide (AMP) therapeutics, yet existing models only say "toxic" or "non-toxic." AmpLyze closes this gap by predicting the actual HC50 value from sequence alone and explaining the residues that drive toxicity. The model couples residue-level ProtT5/ESM2 embeddings with sequence-level descriptors in dual local and global branches, aligned by a cross-attention module and trained with log-cosh loss for robustness to assay noise. The optimal AmpLyze model reaches a PCC of 0.756 and an MSE of 0.987, outperforming classical regressors and the state-of-the-art. Ablations confirm that both branches are essential, and cross-attention adds a further 1% PCC and 3% MSE improvement. Expected-Gradients attributions reveal known toxicity hotspots and suggest safer substitutions. By turning hemolysis assessment into a quantitative, sequence-based, and interpretable prediction, AmpLyze facilitates AMP design and offers a practical tool for early-stage toxicity screening. |
| 2025-07-10 | [AI for NONMEM Coding in Pharmacometrics Research and Education: Shortcut or Pitfall?](http://arxiv.org/abs/2507.08144v1) | Wenhao Zheng, Wanbing Wang et al. | Artificial intelligence (AI) is increasingly being explored as a tool to support pharmacometric modeling, particularly in addressing the coding challenges associated with NONMEM. In this study, we evaluated the ability of seven AI agents to generate NONMEM codes across 13 pharmacometrics tasks, including a range of population pharmacokinetic (PK) and pharmacodynamic (PD) models. We further developed a standardized scoring rubric to assess code accuracy and created an optimized prompt to improve AI agent performance. Our results showed that the OpenAI o1 and gpt-4.1 models achieved the best performance, both generating codes with great accuracy for all tasks when using our optimized prompt. Overall, AI agents performed well in writing basic NONMEM model structures, providing a useful foundation for pharmacometrics model coding. However, user review and refinement remain essential, especially for complex models with special dataset alignment or advanced coding techniques. We also discussed the applications of AI in pharmacometrics education, particularly strategies to prevent over-reliance on AI for coding. This work provides a benchmark for current AI agents performance in NONMEM coding and introduces a practical prompt that can facilitate more accurate and efficient use of AI in pharmacometrics research and education. |
| 2025-07-10 | [Audit, Alignment, and Optimization of LM-Powered Subroutines with Application to Public Comment Processing](http://arxiv.org/abs/2507.08109v1) | Reilly Raab, Mike Parker et al. | The advent of language models (LMs) has the potential to dramatically accelerate tasks that may be cast to text-processing; however, real-world adoption is hindered by concerns regarding safety, explainability, and bias. How can we responsibly leverage LMs in a transparent, auditable manner -- minimizing risk and allowing human experts to focus on informed decision-making rather than data-processing or prompt engineering? In this work, we propose a framework for declaring statically typed, LM-powered subroutines (i.e., callable, function-like procedures) for use within conventional asynchronous code -- such that sparse feedback from human experts is used to improve the performance of each subroutine online (i.e., during use). In our implementation, all LM-produced artifacts (i.e., prompts, inputs, outputs, and data-dependencies) are recorded and exposed to audit on demand. We package this framework as a library to support its adoption and continued development. While this framework may be applicable across several real-world decision workflows (e.g., in healthcare and legal fields), we evaluate it in the context of public comment processing as mandated by the 1969 National Environmental Protection Act (NEPA): Specifically, we use this framework to develop "CommentNEPA," an application that compiles, organizes, and summarizes a corpus of public commentary submitted in response to a project requiring environmental review. We quantitatively evaluate the application by comparing its outputs (when operating without human feedback) to historical ``ground-truth'' data as labelled by human annotators during the preparation of official environmental impact statements. |
| 2025-07-10 | [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology](http://arxiv.org/abs/2507.07999v1) | Haochen Wang, Xiangtai Li et al. | Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at https://github.com/Haochen-Wang409/TreeVGR. |
| 2025-07-10 | [Automating Expert-Level Medical Reasoning Evaluation of Large Language Models](http://arxiv.org/abs/2507.07988v1) | Shuang Zhou, Wenya Xie et al. | As large language models (LLMs) become increasingly integrated into clinical decision-making, ensuring transparent and trustworthy reasoning is essential. However, existing evaluation strategies of LLMs' medical reasoning capability either suffer from unsatisfactory assessment or poor scalability, and a rigorous benchmark remains lacking. To address this, we introduce MedThink-Bench, a benchmark designed for rigorous, explainable, and scalable assessment of LLMs' medical reasoning. MedThink-Bench comprises 500 challenging questions across ten medical domains, each annotated with expert-crafted step-by-step rationales. Building on this, we propose LLM-w-Ref, a novel evaluation framework that leverages fine-grained rationales and LLM-as-a-Judge mechanisms to assess intermediate reasoning with expert-level fidelity while maintaining scalability. Experiments show that LLM-w-Ref exhibits a strong positive correlation with expert judgments. Benchmarking twelve state-of-the-art LLMs, we find that smaller models (e.g., MedGemma-27B) can surpass larger proprietary counterparts (e.g., OpenAI-o3). Overall, MedThink-Bench offers a foundational tool for evaluating LLMs' medical reasoning, advancing their safe and responsible deployment in clinical practice. |
| 2025-07-10 | [Multimodal Framework for Explainable Autonomous Driving: Integrating Video, Sensor, and Textual Data for Enhanced Decision-Making and Transparency](http://arxiv.org/abs/2507.07938v1) | Abolfazl Zarghani, Amirhossein Ebrahimi et al. | Autonomous vehicles (AVs) are poised to redefine transportation by enhancing road safety, minimizing human error, and optimizing traffic efficiency. The success of AVs depends on their ability to interpret complex, dynamic environments through diverse data sources, including video streams, sensor measurements, and contextual textual information. However, seamlessly integrating these multimodal inputs and ensuring transparency in AI-driven decisions remain formidable challenges. This study introduces a novel multimodal framework that synergistically combines video, sensor, and textual data to predict driving actions while generating human-readable explanations, fostering trust and regulatory compliance. By leveraging VideoMAE for spatiotemporal video analysis, a custom sensor fusion module for real-time data processing, and BERT for textual comprehension, our approach achieves robust decision-making and interpretable outputs. Evaluated on the BDD-X (21113 samples) and nuScenes (1000 scenes) datasets, our model reduces training loss from 5.7231 to 0.0187 over five epochs, attaining an action prediction accuracy of 92.5% and a BLEU-4 score of 0.75 for explanation quality, outperforming state-of-the-art methods. Ablation studies confirm the critical role of each modality, while qualitative analyses and human evaluations highlight the model's ability to produce contextually rich, user-friendly explanations. These advancements underscore the transformative potential of multimodal integration and explainability in building safe, transparent, and trustworthy AV systems, paving the way for broader societal adoption of autonomous driving technologies. |

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



