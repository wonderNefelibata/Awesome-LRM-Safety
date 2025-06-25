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
| 2025-06-24 | [Persona Features Control Emergent Misalignment](http://arxiv.org/abs/2506.19823v1) | Miles Wang, Tom Dupré la Tour et al. | Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. discovered that fine-tuning GPT-4o on intentionally insecure code causes "emergent misalignment," where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a "model diffing" approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several "misaligned persona" features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment. |
| 2025-06-24 | [Optimization Studies of Radiation Shielding for the PIP-II Project at Fermilab](http://arxiv.org/abs/2506.19763v1) | Alajos Makovec, Dali Georgobiani et al. | The PIP-II project at Fermilab, which includes an 800-MeV superconducting LINAC, demands rigorous radiation shielding optimization to meet safety requirements. We updated the MARS geometry model to reflect new magnet and collimator designs and introduced high-resolution detector planes to better capture radiation field distributions. To overcome the significant computational demands, we implemented a well-known branching technique that drastically reduced simulation runtimes while maintaining statistical integrity. This was achieved through particle splitting and the application of Russian Roulette techniques. Additionally, new graphical tools were created to streamline data visualization and MARS code usability. |
| 2025-06-24 | [Formalization and security analysis of the Bridgeless protocol](http://arxiv.org/abs/2506.19730v1) | Orestis Alpos, Oleg Fomenko et al. | This paper formalizes the proves the security of the Bridgeless protocol, a protocol able to bridge tokens between various chains. The Bridgeless protocol is run by a set of validators, responsible for verifying deposit transactions on the source chain and generating the corresponding withdrawals on the target chain. The protocol is designed to be chain-agnostic and the validators interact with each supported chain via a chain client. It currently supports EVM-compatible chains, the Zano, and the Bitcoin chains. The paper formalizes all involved subprotocols and describes the conditions under which the protocol maintains safety and liveness. |
| 2025-06-24 | [A Verification Methodology for Safety Assurance of Robotic Autonomous Systems](http://arxiv.org/abs/2506.19622v1) | Mustafa Adam, David A. Anisi et al. | Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems. |
| 2025-06-24 | [Probabilistic modelling and safety assurance of an agriculture robot providing light-treatment](http://arxiv.org/abs/2506.19620v1) | Mustafa Adam, Kangfeng Ye et al. | Continued adoption of agricultural robots postulates the farmer's trust in the reliability, robustness and safety of the new technology. This motivates our work on safety assurance of agricultural robots, particularly their ability to detect, track and avoid obstacles and humans. This paper considers a probabilistic modelling and risk analysis framework for use in the early development phases. Starting off with hazard identification and a risk assessment matrix, the behaviour of the mobile robot platform, sensor and perception system, and any humans present are captured using three state machines. An auto-generated probabilistic model is then solved and analysed using the probabilistic model checker PRISM. The result provides unique insight into fundamental development and engineering aspects by quantifying the effect of the risk mitigation actions and risk reduction associated with distinct design concepts. These include implications of adopting a higher performance and more expensive Object Detection System or opting for a more elaborate warning system to increase human awareness. Although this paper mainly focuses on the initial concept-development phase, the proposed safety assurance framework can also be used during implementation, and subsequent deployment and operation phases. |
| 2025-06-24 | [PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty](http://arxiv.org/abs/2506.19563v1) | Jinwen He, Yiyang Lu et al. | Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications. |
| 2025-06-24 | [A Spline-Based Stress Function Approach for the Principle of Minimum Complementary Energy](http://arxiv.org/abs/2506.19534v1) | Fabian Key, Lukas Freinberger | In computational engineering, ensuring the integrity and safety of structures in fields such as aerospace and civil engineering relies on accurate stress prediction. However, analytical methods are limited to simple test cases, and displacement-based finite element methods (FEMs), while commonly used, require a large number of unknowns to achieve high accuracy; stress-based numerical methods have so far failed to provide a simple and effective alternative. This work aims to develop a novel numerical approach that overcomes these limitations by enabling accurate stress prediction with improved flexibility for complex geometries and boundary conditions and fewer degrees of freedom (DOFs). The proposed method is based on a spline-based stress function formulation for the principle of minimum complementary energy, which we apply to plane, linear elastostatics. The method is first validated against an analytical power series solution and then tested on two test cases challenging for current state-of-the-art numerical schemes, a bi-layer cantilever with anisotropic material behavior, and a cantilever with a non-prismatic, parabolic-shaped beam geometry. Results demonstrate that our approach, unlike analytical methods, can be easily applied to general geometries and boundary conditions, and achieves stress accuracy comparable to that reported in the literature for displacement-based FEMs, while requiring significantly fewer DOFs. This novel spline-based stress function approach thus provides an efficient and flexible tool for accurate stress prediction, with promising applications in structural analysis and numerical design. |
| 2025-06-24 | [Automatic Posology Structuration : What role for LLMs?](http://arxiv.org/abs/2506.19525v1) | Natalia Bobkova, Laura Zanella-Calzada et al. | Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use. |
| 2025-06-24 | [Visual hallucination detection in large vision-language models via evidential conflict](http://arxiv.org/abs/2506.19513v1) | Tao Huang, Zhekun Liu et al. | Despite the remarkable multimodal capabilities of Large Vision-Language Models (LVLMs), discrepancies often occur between visual inputs and textual outputs--a phenomenon we term visual hallucination. This critical reliability gap poses substantial risks in safety-critical Artificial Intelligence (AI) applications, necessitating a comprehensive evaluation benchmark and effective detection methods. Firstly, we observe that existing visual-centric hallucination benchmarks mainly assess LVLMs from a perception perspective, overlooking hallucinations arising from advanced reasoning capabilities. We develop the Perception-Reasoning Evaluation Hallucination (PRE-HAL) dataset, which enables the systematic evaluation of both perception and reasoning capabilities of LVLMs across multiple visual semantics, such as instances, scenes, and relations. Comprehensive evaluation with this new benchmark exposed more visual vulnerabilities, particularly in the more challenging task of relation reasoning. To address this issue, we propose, to the best of our knowledge, the first Dempster-Shafer theory (DST)-based visual hallucination detection method for LVLMs through uncertainty estimation. This method aims to efficiently capture the degree of conflict in high-level features at the model inference phase. Specifically, our approach employs simple mass functions to mitigate the computational complexity of evidence combination on power sets. We conduct an extensive evaluation of state-of-the-art LVLMs, LLaVA-v1.5, mPLUG-Owl2 and mPLUG-Owl3, with the new PRE-HAL benchmark. Experimental results indicate that our method outperforms five baseline uncertainty metrics, achieving average AUROC improvements of 4%, 10%, and 7% across three LVLMs. Our code is available at https://github.com/HT86159/Evidential-Conflict. |
| 2025-06-24 | [5 Days, 5 Stories: Using Technology to Promote Empathy in the Workplace](http://arxiv.org/abs/2506.19495v1) | Russell Beale, Eugenia Sergueeva | Empathy is widely recognized as a vital attribute for effective collaboration and communication in the workplace, yet developing empathic skills and fostering it among colleagues remains a challenge. This study explores the potential of a collaborative digital storytelling platform - In Your Shoes - designed to promote empathic listening and interpersonal understanding through the structured exchange of personal narratives. A one-week intervention was conducted with employees from multiple organizations using the platform. Employing a mixed methods approach, we assessed quantitative changes in empathy using the Empathy Quotient (EQ) and qualitatively analyzed participant experiences through grounded theory. While quantitative analysis revealed no statistically significant shift in dispositional empathy, qualitative findings suggested the tool facilitated situational empathy, prompted self-reflection, improved emotional resonance, and enhanced workplace relationships. Participants reported feelings of psychological safety, connection, and, in some cases, therapeutic benefits from sharing and responding to stories. These results highlight the promise of asynchronous, structured narrative-based digital tools for supporting empathic engagement in professional settings, offering insights for the design of emotionally intelligent workplace technologies. |
| 2025-06-24 | [A Survey on Soft Robot Adaptability: Implementations, Applications, and Prospects](http://arxiv.org/abs/2506.19397v1) | Zixi Chen, Di Wu et al. | Soft robots, compared to rigid robots, possess inherent advantages, including higher degrees of freedom, compliance, and enhanced safety, which have contributed to their increasing application across various fields. Among these benefits, adaptability is particularly noteworthy. In this paper, adaptability in soft robots is categorized into external and internal adaptability. External adaptability refers to the robot's ability to adjust, either passively or actively, to variations in environments, object properties, geometries, and task dynamics. Internal adaptability refers to the robot's ability to cope with internal variations, such as manufacturing tolerances or material aging, and to generalize control strategies across different robots. As the field of soft robotics continues to evolve, the significance of adaptability has become increasingly pronounced. In this review, we summarize various approaches to enhancing the adaptability of soft robots, including design, sensing, and control strategies. Additionally, we assess the impact of adaptability on applications such as surgery, wearable devices, locomotion, and manipulation. We also discuss the limitations of soft robotics adaptability and prospective directions for future research. By analyzing adaptability through the lenses of implementation, application, and challenges, this paper aims to provide a comprehensive understanding of this essential characteristic in soft robotics and its implications for diverse applications. |
| 2025-06-24 | [Trajectory Prediction in Dynamic Object Tracking: A Critical Study](http://arxiv.org/abs/2506.19341v1) | Zhongping Dong, Liming Chen et al. | This study provides a detailed analysis of current advancements in dynamic object tracking (DOT) and trajectory prediction (TP) methodologies, including their applications and challenges. It covers various approaches, such as feature-based, segmentation-based, estimation-based, and learning-based methods, evaluating their effectiveness, deployment, and limitations in real-world scenarios. The study highlights the significant impact of these technologies in automotive and autonomous vehicles, surveillance and security, healthcare, and industrial automation, contributing to safety and efficiency. Despite the progress, challenges such as improved generalization, computational efficiency, reduced data dependency, and ethical considerations still exist. The study suggests future research directions to address these challenges, emphasizing the importance of multimodal data integration, semantic information fusion, and developing context-aware systems, along with ethical and privacy-preserving frameworks. |
| 2025-06-24 | [MSR-Align: Policy-Grounded Multimodal Alignment for Safety-Aware Reasoning in Vision-Language Models](http://arxiv.org/abs/2506.19257v1) | Yinan Xia, Yilei Jiang et al. | Vision-Language Models (VLMs) have achieved remarkable progress in multimodal reasoning tasks through enhanced chain-of-thought capabilities. However, this advancement also introduces novel safety risks, as these models become increasingly vulnerable to harmful multimodal prompts that can trigger unethical or unsafe behaviors. Existing safety alignment approaches, primarily designed for unimodal language models, fall short in addressing the complex and nuanced threats posed by multimodal inputs. Moreover, current safety datasets lack the fine-grained, policy-grounded reasoning required to robustly align reasoning-capable VLMs. In this work, we introduce {MSR-Align}, a high-quality Multimodal Safety Reasoning dataset tailored to bridge this gap. MSR-Align supports fine-grained, deliberative reasoning over standardized safety policies across both vision and text modalities. Our data generation pipeline emphasizes multimodal diversity, policy-grounded reasoning, and rigorous quality filtering using strong multimodal judges. Extensive experiments demonstrate that fine-tuning VLMs on MSR-Align substantially improves robustness against both textual and vision-language jailbreak attacks, while preserving or enhancing general reasoning performance. MSR-Align provides a scalable and effective foundation for advancing the safety alignment of reasoning-capable VLMs. Our dataset is made publicly available at https://huggingface.co/datasets/Leigest/MSR-Align. |
| 2025-06-24 | [Robust Behavior Cloning Via Global Lipschitz Regularization](http://arxiv.org/abs/2506.19250v1) | Shili Wu, Yizhao Jin et al. | Behavior Cloning (BC) is an effective imitation learning technique and has even been adopted in some safety-critical domains such as autonomous vehicles. BC trains a policy to mimic the behavior of an expert by using a dataset composed of only state-action pairs demonstrated by the expert, without any additional interaction with the environment. However, During deployment, the policy observations may contain measurement errors or adversarial disturbances. Since the observations may deviate from the true states, they can mislead the agent into making sub-optimal actions. In this work, we use a global Lipschitz regularization approach to enhance the robustness of the learned policy network. We then show that the resulting global Lipschitz property provides a robustness certificate to the policy with respect to different bounded norm perturbations. Then, we propose a way to construct a Lipschitz neural network that ensures the policy robustness. We empirically validate our theory across various environments in Gymnasium. Keywords: Robust Reinforcement Learning; Behavior Cloning; Lipschitz Neural Network |
| 2025-06-24 | [Inference-Time Reward Hacking in Large Language Models](http://arxiv.org/abs/2506.19248v1) | Hadi Khalaf, Claudio Mayrink Verdun et al. | A common paradigm to improve the performance of large language models is optimizing for a reward model. Reward models assign a numerical score to LLM outputs indicating, for example, which response would likely be preferred by a user or is most aligned with safety goals. However, reward models are never perfect. They inevitably function as proxies for complex desiderata such as correctness, helpfulness, and safety. By overoptimizing for a misspecified reward, we can subvert intended alignment goals and reduce overall performance -- a phenomenon commonly referred to as reward hacking. In this work, we characterize reward hacking in inference-time alignment and demonstrate when and how we can mitigate it by hedging on the proxy reward. We study this phenomenon under Best-of-$n$ (BoN) and Soft-Best-of-$n$ (SBoN), and we introduce Best-of-Poisson (BoP) that provides an efficient, near-exact approximation of the optimal reward-KL divergence policy at inference time. We show that the characteristic pattern of hacking as observed in practice (where the true reward first increases before declining) is an inevitable property of a broad class of inference-time mechanisms, including BoN and BoP. To counter this effect, hedging offers a tactical choice to avoid placing undue confidence in high but potentially misleading proxy reward signals. We introduce HedgeTune, an efficient algorithm to find the optimal inference-time parameter and avoid reward hacking. We demonstrate through experiments that hedging mitigates reward hacking and achieves superior distortion-reward tradeoffs with minimal computational overhead. |
| 2025-06-24 | [RecLLM-R1: A Two-Stage Training Paradigm with Reinforcement Learning and Chain-of-Thought v1](http://arxiv.org/abs/2506.19235v1) | Yu Xie, Xingkai Ren et al. | Traditional recommendation systems often grapple with "filter bubbles", underutilization of external knowledge, and a disconnect between model optimization and business policy iteration. To address these limitations, this paper introduces RecLLM-R1, a novel recommendation framework leveraging Large Language Models (LLMs) and drawing inspiration from the DeepSeek R1 methodology. The framework initiates by transforming user profiles, historical interactions, and multi-faceted item attributes into LLM-interpretable natural language prompts through a carefully engineered data construction process. Subsequently, a two-stage training paradigm is employed: the initial stage involves Supervised Fine-Tuning (SFT) to imbue the LLM with fundamental recommendation capabilities. The subsequent stage utilizes Group Relative Policy Optimization (GRPO), a reinforcement learning technique, augmented with a Chain-of-Thought (CoT) mechanism. This stage guides the model through multi-step reasoning and holistic decision-making via a flexibly defined reward function, aiming to concurrently optimize recommendation accuracy, diversity, and other bespoke business objectives. Empirical evaluations on a real-world user behavior dataset from a large-scale social media platform demonstrate that RecLLM-R1 significantly surpasses existing baseline methods across a spectrum of evaluation metrics, including accuracy, diversity, and novelty. It effectively mitigates the filter bubble effect and presents a promising avenue for the integrated optimization of recommendation models and policies under intricate business goals. |
| 2025-06-23 | [Command-V: Pasting LLM Behaviors via Activation Profiles](http://arxiv.org/abs/2506.19140v1) | Barry Wang, Avi Schwarzschild et al. | Retrofitting large language models (LLMs) with new behaviors typically requires full finetuning or distillation-costly steps that must be repeated for every architecture. In this work, we introduce Command-V, a backpropagation-free behavior transfer method that copies an existing residual activation adapter from a donor model and pastes its effect into a recipient model. Command-V profiles layer activations on a small prompt set, derives linear converters between corresponding layers, and applies the donor intervention in the recipient's activation space. This process does not require access to the original training data and needs minimal compute. In three case studies-safety-refusal enhancement, jailbreak facilitation, and automatic chain-of-thought reasoning--Command-V matches or exceeds the performance of direct finetuning while using orders of magnitude less compute. Our code and data are accessible at https://github.com/GithuBarry/Command-V/. |
| 2025-06-23 | [Enhancing Evacuation Safety: Detecting Post-Nuclear Event Radiation Levels in an Urban Area](http://arxiv.org/abs/2506.19044v1) | Ellis Duncalfe, Milena Radenkovic | The detonation of an improvised nuclear device (IND) in an urban area would cause catastrophic damage, followed by hazardous radioactive fallout. Timely dissemination of radiation data is crucial for evacuation and casualty reduction. However, conventional communication infrastructure is likely to be severely disrupted. This study designs and builds a pseudorealistic, geospatially and temporally dynamic post-nuclear event (PNE) scenario using the Opportunistic Network Environment (ONE) simulator. It integrates radiation sensing by emergency responders, unmanned aerial vehicles (UAVs), and civilian devices as dynamic nodes within Delay-Tolerant Networks (DTNs). The performance of two DTN routing protocols, Epidemic and PRoPHET, was evaluated across multiple PNE phases. Both protocols achieve high message delivery rates, with PRoPHET exhibiting lower network overhead but higher latency. Findings demonstrate the potential of DTN-based solutions to support emergency response and evacuation safety by ensuring critical radiation data propagation despite severe infrastructure damage. |
| 2025-06-23 | [ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](http://arxiv.org/abs/2506.18896v1) | Jiaru Zou, Ling Yang et al. | Process Reward Models (PRMs) have recently emerged as a powerful framework for supervising intermediate reasoning steps in large language models (LLMs). Previous PRMs are primarily trained on model final output responses and struggle to evaluate intermediate thinking trajectories robustly, especially in the emerging setting of trajectory-response outputs generated by frontier reasoning models like Deepseek-R1. In this work, we introduce ReasonFlux-PRM, a novel trajectory-aware PRM explicitly designed to evaluate the trajectory-response type of reasoning traces. ReasonFlux-PRM incorporates both step-level and trajectory-level supervision, enabling fine-grained reward assignment aligned with structured chain-of-thought data. We adapt ReasonFlux-PRM to support reward supervision under both offline and online settings, including (i) selecting high-quality model distillation data for downstream supervised fine-tuning of smaller models, (ii) providing dense process-level rewards for policy optimization during reinforcement learning, and (iii) enabling reward-guided Best-of-N test-time scaling. Empirical results on challenging downstream benchmarks such as AIME, MATH500, and GPQA-Diamond demonstrate that ReasonFlux-PRM-7B selects higher quality data than strong PRMs (e.g., Qwen2.5-Math-PRM-72B) and human-curated baselines. Furthermore, our derived ReasonFlux-PRM-7B yields consistent performance improvements, achieving average gains of 12.1% in supervised fine-tuning, 4.5% in reinforcement learning, and 6.3% in test-time scaling. We also release our efficient ReasonFlux-PRM-1.5B for resource-constrained applications and edge deployment. Projects: https://github.com/Gen-Verse/ReasonFlux |
| 2025-06-23 | [OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization](http://arxiv.org/abs/2506.18880v1) | Yiyou Sun, Shawn Hu et al. | Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency. |

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



