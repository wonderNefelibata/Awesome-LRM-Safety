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
| 2025-05-12 | [A Theoretical Framework for Explaining Reinforcement Learning with Shapley Values](http://arxiv.org/abs/2505.07797v1) | Daniel Beechey, Thomas M. S. Smith et al. | Reinforcement learning agents can achieve superhuman performance, but their decisions are often difficult to interpret. This lack of transparency limits deployment, especially in safety-critical settings where human trust and accountability are essential. In this work, we develop a theoretical framework for explaining reinforcement learning through the influence of state features, which represent what the agent observes in its environment. We identify three core elements of the agent-environment interaction that benefit from explanation: behaviour (what the agent does), performance (what the agent achieves), and value estimation (what the agent expects to achieve). We treat state features as players cooperating to produce each element and apply Shapley values, a principled method from cooperative game theory, to identify the influence of each feature. This approach yields a family of mathematically grounded explanations with clear semantics and theoretical guarantees. We use illustrative examples to show how these explanations align with human intuition and reveal novel insights. Our framework unifies and extends prior work, making explicit the assumptions behind existing approaches, and offers a principled foundation for more interpretable and trustworthy reinforcement learning. |
| 2025-05-12 | [Learning from Peers in Reasoning Models](http://arxiv.org/abs/2505.07787v1) | Tongxu Luo, Wenyu Du et al. | Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the "Prefix Dominance Trap". Inspired by psychological findings that peer interaction can promote self-correction without negatively impacting already accurate individuals, we propose **Learning from Peers** (LeaP) to address this phenomenon. Specifically, every tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference. However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively. To address this, we fine-tune them into our **LeaP-T** model series. Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements. For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points. Notably, our fine-tuned LeaP-T-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024. In-depth analysis reveals LeaP's robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty. LeaP marks a milestone by enabling LRMs to collaborate during reasoning. Our code, datasets, and models are available at https://learning-from-peers.github.io/ . |
| 2025-05-12 | [Must Read: A Systematic Survey of Computational Persuasion](http://arxiv.org/abs/2505.07775v1) | Nimet Beyza Bozdag, Shuhaib Mehri et al. | Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models. |
| 2025-05-12 | [Clouds can enhance direct imaging detection of O2 and O3 on terrestrial exoplanets](http://arxiv.org/abs/2505.07760v1) | Huanzhou Yang, Michelle Hu et al. | Clouds are often considered a highly uncertain barrier for detecting biosignatures on exoplanets, especially given intuition gained from transit surveys. However, for direct imaging reflected light observations, clouds could increase the observational signal by increasing reflected light. Here we constrain the impact of clouds on the detection of O2 and O3 by a direct imaging telescope such as the Habitable Worlds Observatory (HWO) using observations simulated with the Planetary Spectrum Generator (PSG). We first perform sensitivity tests to show that low clouds enhance O2 and O3 detectability while high clouds diminish it, and the effect is greater when cloud particles are smaller. We next apply clouds produced by the cloud microphysics model CARMA with varied planetary parameters and clouds drawn from observations of different types of clouds on Earth to PSG. We find that clouds are likely to increase the SNR of O2 and O3 for terrestrial exoplanets under a wide range of scenarios. This work provides important constraints on the impact of clouds on observations by telescopes including HWO. |
| 2025-05-12 | [Emotion-Gradient Metacognitive RSI (Part I): Theoretical Foundations and Single-Agent Architecture](http://arxiv.org/abs/2505.07757v1) | Rintaro Ando | We present the Emotion-Gradient Metacognitive Recursive Self-Improvement (EG-MRSI) framework, a novel architecture that integrates introspective metacognition, emotion-based intrinsic motivation, and recursive self-modification into a unified theoretical system. The framework is explicitly capable of overwriting its own learning algorithm under formally bounded risk. Building upon the Noise-to-Meaning RSI (N2M-RSI) foundation, EG-MRSI introduces a differentiable intrinsic reward function driven by confidence, error, novelty, and cumulative success. This signal regulates both a metacognitive mapping and a self-modification operator constrained by provable safety mechanisms. We formally define the initial agent configuration, emotion-gradient dynamics, and RSI trigger conditions, and derive a reinforcement-compatible optimization objective that guides the agent's development trajectory. Meaning Density and Meaning Conversion Efficiency are introduced as quantifiable metrics of semantic learning, closing the gap between internal structure and predictive informativeness. This Part I paper establishes the single-agent theoretical foundations of EG-MRSI. Future parts will extend this framework to include safety certificates and rollback protocols (Part II), collective intelligence mechanisms (Part III), and feasibility constraints including thermodynamic and computational limits (Part IV). Together, the EG-MRSI series provides a rigorous, extensible foundation for open-ended and safe AGI. |
| 2025-05-12 | [Assessing the Chemical Intelligence of Large Language Models](http://arxiv.org/abs/2505.07735v1) | Nicholas T. Runcie, Charlotte M. Deane et al. | Large Language Models are versatile, general-purpose tools with a wide range of applications. Recently, the advent of "reasoning models" has led to substantial improvements in their abilities in advanced problem-solving domains such as mathematics and software engineering. In this work, we assessed the ability of reasoning models to directly perform chemistry tasks, without any assistance from external tools. We created a novel benchmark, called ChemIQ, which consists of 796 questions assessing core concepts in organic chemistry, focused on molecular comprehension and chemical reasoning. Unlike previous benchmarks, which primarily use multiple choice formats, our approach requires models to construct short-answer responses, more closely reflecting real-world applications. The reasoning models, exemplified by OpenAI's o3-mini, correctly answered 28%-59% of questions depending on the reasoning level used, with higher reasoning levels significantly increasing performance on all tasks. These models substantially outperformed the non-reasoning model, GPT-4o, which achieved only 7% accuracy. We found that Large Language Models can now convert SMILES strings to IUPAC names, a task earlier models were unable to perform. Additionally, we show that the latest reasoning models can elucidate structures from 1H and 13C NMR data, correctly generating SMILES strings for 74% of molecules containing up to 10 heavy atoms, and in one case solving a structure comprising 21 heavy atoms. For each task, we found evidence that the reasoning process mirrors that of a human chemist. Our results demonstrate that the latest reasoning models have the ability to perform advanced chemical reasoning. |
| 2025-05-12 | [Non-Conservative Data-driven Safe Control Design for Nonlinear Systems with Polyhedral Safe Sets](http://arxiv.org/abs/2505.07733v1) | Amir Modares, Bosen Lian et al. | This paper presents a data-driven nonlinear safe control design approach for discrete-time systems under parametric uncertainties and additive disturbances. We first characterize a new control structure from which a data-based representation of closed-loop systems is obtained. This data-based closed-loop system is composed of two parts: 1) a parametrized linear closed-loop part and a parametrized nonlinear remainder closed-loop part. We show that using the standard practice or learning a robust controller to ensure safety while treating the remaining nonlinearities as disturbances brings about significant challenges in terms of computational complexity and conservatism. To overcome these challenges, we develop a novel nonlinear safe control design approach in which the closed-loop nonlinear remainders are learned, rather than canceled, in a control-oriented fashion while preserving the computational efficiency. To this end, a primal-dual optimization framework is leveraged in which the control gains are learned to enforce the second-order optimality on the closed-loop nonlinear remainders. This allows us to account for nonlinearities in the design for the sake of safety rather than treating them as disturbances. This new controller parameterization and design approach reduces the computational complexity and the conservatism of designing a safe nonlinear controller. A simulation example is then provided to show the effectiveness of the proposed data-driven controller. |
| 2025-05-12 | [Hybrid Control Strategies for Safe and Adaptive Robot-Assisted Dressing](http://arxiv.org/abs/2505.07710v1) | Yasmin Rafiq, Baslin A. James et al. | Safety, reliability, and user trust are crucial in human-robot interaction (HRI) where the robots must address hazards in real-time. This study presents hazard driven low-level control strategies implemented in robot-assisted dressing (RAD) scenarios where hazards like garment snags and user discomfort in real-time can affect task performance and user safety. The proposed control mechanisms include: (1) Garment Snagging Control Strategy, which detects excessive forces and either seeks user intervention via a chatbot or autonomously adjusts its trajectory, and (2) User Discomfort/Pain Mitigation Strategy, which dynamically reduces velocity based on user feedback and aborts the task if necessary. We used physical dressing trials in order to evaluate these control strategies. Results confirm that integrating force monitoring with user feedback improves safety and task continuity. The findings emphasise the need for hybrid approaches that balance autonomous intervention, user involvement, and controlled task termination, supported by bi-directional interaction and real-time user-driven adaptability, paving the way for more responsive and personalised HRI systems. |
| 2025-05-12 | [S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models](http://arxiv.org/abs/2505.07686v1) | Muzhi Dai, Chenxu Yang et al. | As Test-Time Scaling emerges as an active research focus in the large language model community, advanced post-training methods increasingly emphasize extending chain-of-thought (CoT) generation length, thereby enhancing reasoning capabilities to approach Deepseek R1-like reasoning models. However, recent studies reveal that reasoning models (even Qwen3) consistently exhibit excessive thought redundancy in CoT generation. This overthinking problem stems from conventional outcome-reward reinforcement learning's systematic neglect in regulating intermediate reasoning steps. This paper proposes Serial-Group Decaying-Reward Policy Optimization (namely S-GRPO), a novel reinforcement learning method that empowers models with the capability to determine the sufficiency of reasoning steps, subsequently triggering early exit of CoT generation. Specifically, unlike GRPO, which samples multiple possible completions (parallel group) in parallel, we select multiple temporal positions in the generation of one CoT to allow the model to exit thinking and instead generate answers (serial group), respectively. For the correct answers in a serial group, we assign rewards that decay according to positions, with lower rewards towards the later ones, thereby reinforcing the model's behavior to generate higher-quality answers at earlier phases with earlier exits of thinking. Empirical evaluations demonstrate compatibility with state-of-the-art reasoning models, including Qwen3 and Deepseek-distill models, achieving 35.4% ~ 61.1\% sequence length reduction with 0.72% ~ 6.08% accuracy improvements across GSM8K, AIME 2024, AMC 2023, MATH-500, and GPQA Diamond benchmarks. |
| 2025-05-12 | [Deep Learning Advances in Vision-Based Traffic Accident Anticipation: A Comprehensive Review of Methods,Datasets,and Future Directions](http://arxiv.org/abs/2505.07611v1) | Yi Zhang, Wenye Zhou et al. | Traffic accident prediction and detection are critical for enhancing road safety,and vision-based traffic accident anticipation (Vision-TAA) has emerged as a promising approach in the era of deep learning.This paper reviews 147 recent studies,focusing on the application of supervised,unsupervised,and hybrid deep learning models for accident prediction,alongside the use of real-world and synthetic datasets.Current methodologies are categorized into four key approaches: image and video feature-based prediction, spatiotemporal feature-based prediction, scene understanding,and multimodal data fusion.While these methods demonstrate significant potential,challenges such as data scarcity,limited generalization to complex scenarios,and real-time performance constraints remain prevalent. This review highlights opportunities for future research,including the integration of multimodal data fusion, self-supervised learning,and Transformer-based architectures to enhance prediction accuracy and scalability.By synthesizing existing advancements and identifying critical gaps, this paper provides a foundational reference for developing robust and adaptive Vision-TAA systems,contributing to road safety and traffic management. |
| 2025-05-12 | [Concept-Level Explainability for Auditing & Steering LLM Responses](http://arxiv.org/abs/2505.07610v1) | Kenza Amara, Rita Sevastjanova et al. | As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior. |
| 2025-05-12 | [MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining](http://arxiv.org/abs/2505.07608v1) | Xiaomi LLM-Core Team, : et al. | We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at https://github.com/xiaomimimo/MiMo. |
| 2025-05-12 | [Finite-Sample-Based Reachability for Safe Control with Gaussian Process Dynamics](http://arxiv.org/abs/2505.07594v1) | Manish Prajapat, Johannes Köhler et al. | Gaussian Process (GP) regression is shown to be effective for learning unknown dynamics, enabling efficient and safety-aware control strategies across diverse applications. However, existing GP-based model predictive control (GP-MPC) methods either rely on approximations, thus lacking guarantees, or are overly conservative, which limits their practical utility. To close this gap, we present a sampling-based framework that efficiently propagates the model's epistemic uncertainty while avoiding conservatism. We establish a novel sample complexity result that enables the construction of a reachable set using a finite number of dynamics functions sampled from the GP posterior. Building on this, we design a sampling-based GP-MPC scheme that is recursively feasible and guarantees closed-loop safety and stability with high probability. Finally, we showcase the effectiveness of our method on two numerical examples, highlighting accurate reachable set over-approximation and safe closed-loop performance. |
| 2025-05-12 | [SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models](http://arxiv.org/abs/2505.07584v1) | Huining Cui, Wei Liu | The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security. |
| 2025-05-12 | [Byam: Fixing Breaking Dependency Updates with Large Language Models](http://arxiv.org/abs/2505.07522v1) | Frank Reyes, May Mahmoud et al. | Application Programming Interfaces (APIs) facilitate the integration of third-party dependencies within the code of client applications. However, changes to an API, such as deprecation, modification of parameter names or types, or complete replacement with a new API, can break existing client code. These changes are called breaking dependency updates; It is often tedious for API users to identify the cause of these breaks and update their code accordingly. In this paper, we explore the use of Large Language Models (LLMs) to automate client code updates in response to breaking dependency updates. We evaluate our approach on the BUMP dataset, a benchmark for breaking dependency updates in Java projects. Our approach leverages LLMs with advanced prompts, including information from the build process and from the breaking dependency analysis. We assess effectiveness at three granularity levels: at the build level, the file level, and the individual compilation error level. We experiment with five LLMs: Google Gemini-2.0 Flash, OpenAI GPT4o-mini, OpenAI o3-mini, Alibaba Qwen2.5-32b-instruct, and DeepSeek V3. Our results show that LLMs can automatically repair breaking updates. Among the considered models, OpenAI's o3-mini is the best, able to completely fix 27% of the builds when using prompts that include contextual information such as the buggy line, API differences, error messages, and step-by-step reasoning instructions. Also, it fixes 78% of the individual compilation errors. Overall, our findings demonstrate the potential for LLMs to fix compilation errors due to breaking dependency updates, supporting developers in their efforts to stay up-to-date with changes in their dependencies. |
| 2025-05-12 | [Promising Topics for U.S.-China Dialogues on AI Risks and Governance](http://arxiv.org/abs/2505.07468v1) | Saad Siddiqui, Lujain Ibrahim et al. | Cooperation between the United States and China, the world's leading artificial intelligence (AI) powers, is crucial for effective global AI governance and responsible AI development. Although geopolitical tensions have emphasized areas of conflict, in this work, we identify potential common ground for productive dialogue by conducting a systematic analysis of more than 40 primary AI policy and corporate governance documents from both nations. Specifically, using an adapted version of the AI Governance and Regulatory Archive (AGORA) - a comprehensive repository of global AI governance documents - we analyze these materials in their original languages to identify areas of convergence in (1) sociotechnical risk perception and (2) governance approaches. We find strong and moderate overlap in several areas such as on concerns about algorithmic transparency, system reliability, agreement on the importance of inclusive multi-stakeholder engagement, and AI's role in enhancing safety. These findings suggest that despite strategic competition, there exist concrete opportunities for bilateral U.S.-China cooperation in the development of responsible AI. Thus, we present recommendations for furthering diplomatic dialogues that can facilitate such cooperation. Our analysis contributes to understanding how different international governance frameworks might be harmonized to promote global responsible AI development. |
| 2025-05-12 | [AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review](http://arxiv.org/abs/2505.07374v1) | Zhiye Xie, Enmei Tu et al. | With the increasing demands for safety, efficiency, and sustainability in global shipping, Automatic Identification System (AIS) data plays an increasingly important role in maritime monitoring. AIS data contains spatial-temporal variation patterns of vessels that hold significant research value in the marine domain. However, due to its massive scale, the full potential of AIS data has long remained untapped. With its powerful sequence modeling capabilities, particularly its ability to capture long-range dependencies and complex temporal dynamics, the Transformer model has emerged as an effective tool for processing AIS data. Therefore, this paper reviews the research on Transformer-based AIS data-driven maritime monitoring, providing a comprehensive overview of the current applications of Transformer models in the marine field. The focus is on Transformer-based trajectory prediction methods, behavior detection, and prediction techniques. Additionally, this paper collects and organizes publicly available AIS datasets from the reviewed papers, performing data filtering, cleaning, and statistical analysis. The statistical results reveal the operational characteristics of different vessel types, providing data support for further research on maritime monitoring tasks. Finally, we offer valuable suggestions for future research, identifying two promising research directions. Datasets are available at https://github.com/eyesofworld/Maritime-Monitoring. |
| 2025-05-12 | [Enabling Privacy-Aware AI-Based Ergonomic Analysis](http://arxiv.org/abs/2505.07306v1) | Sander De Coninck, Emilio Gamba et al. | Musculoskeletal disorders (MSDs) are a leading cause of injury and productivity loss in the manufacturing industry, incurring substantial economic costs. Ergonomic assessments can mitigate these risks by identifying workplace adjustments that improve posture and reduce strain. Camera-based systems offer a non-intrusive, cost-effective method for continuous ergonomic tracking, but they also raise significant privacy concerns. To address this, we propose a privacy-aware ergonomic assessment framework utilizing machine learning techniques. Our approach employs adversarial training to develop a lightweight neural network that obfuscates video data, preserving only the essential information needed for human pose estimation. This obfuscation ensures compatibility with standard pose estimation algorithms, maintaining high accuracy while protecting privacy. The obfuscated video data is transmitted to a central server, where state-of-the-art keypoint detection algorithms extract body landmarks. Using multi-view integration, 3D keypoints are reconstructed and evaluated with the Rapid Entire Body Assessment (REBA) method. Our system provides a secure, effective solution for ergonomic monitoring in industrial environments, addressing both privacy and workplace safety concerns. |
| 2025-05-12 | [Synthetic Similarity Search in Automotive Production](http://arxiv.org/abs/2505.07256v1) | Christoph Huber, Ludwig Schleeh et al. | Visual quality inspection in automotive production is essential for ensuring the safety and reliability of vehicles. Computer vision (CV) has become a popular solution for these inspections due to its cost-effectiveness and reliability. However, CV models require large, annotated datasets, which are costly and time-consuming to collect. To reduce the need for extensive training data, we propose a novel image classification pipeline that combines similarity search using a vision-based foundation model with synthetic data. Our approach leverages a DINOv2 model to transform input images into feature vectors, which are then compared to pre-classified reference images using cosine distance measurements. By utilizing synthetic data instead of real images as references, our pipeline achieves high classification accuracy without relying on real data. We evaluate this approach in eight real-world inspection scenarios and demonstrate that it meets the high performance requirements of production environments. |
| 2025-05-12 | [Continuous-Time Control Synthesis for Multiple Quadrotors under Signal Temporal Logic Specifications](http://arxiv.org/abs/2505.07240v1) | Yating Yuan | Ensuring continuous-time control of multiple quadrotors in constrained environments under signal temporal logic (STL) specifications is challenging due to nonlinear dynamics, safety constraints, and disturbances. This letter proposes a two-stage framework to address this challenge. First, exponentially decaying tracking error bounds are derived with multidimensional geometric control gains obtained via differential evolution. These bounds are less conservative, while the resulting tracking errors exhibit smaller oscillations and improved transient performance. Second, leveraging the time-varying bounds, a mixed-integer convex programming (MICP) formulation generates piecewise B\'ezier reference trajectories that satisfy STL and velocity limits, while ensuring inter-agent safety through convex-hull properties. Simulation results demonstrate that the proposed approach enables formally verifiable multi-agent coordination in constrained environments, with provable tracking guarantees under bounded disturbances. |

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



