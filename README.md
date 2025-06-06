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
| 2025-06-05 | [Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets](http://arxiv.org/abs/2506.05346v1) | Lei Hsiung, Tianyu Pang et al. | Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers. |
| 2025-06-05 | [Control Tax: The Price of Keeping AI in Check](http://arxiv.org/abs/2506.05296v1) | Mikhail Terekhov, Zhen Ning David Liu et al. | The rapid integration of agentic AI into high-stakes real-world applications requires robust oversight mechanisms. The emerging field of AI Control (AIC) aims to provide such an oversight mechanism, but practical adoption depends heavily on implementation overhead. To study this problem better, we introduce the notion of Control tax -- the operational and financial cost of integrating control measures into AI pipelines. Our work makes three key contributions to the field of AIC: (1) we introduce a theoretical framework that quantifies the Control Tax and maps classifier performance to safety assurances; (2) we conduct comprehensive evaluations of state-of-the-art language models in adversarial settings, where attacker models insert subtle backdoors into code while monitoring models attempt to detect these vulnerabilities; and (3) we provide empirical financial cost estimates for control protocols and develop optimized monitoring strategies that balance safety and cost-effectiveness while accounting for practical constraints like auditing budgets. Our framework enables practitioners to make informed decisions by systematically connecting safety guarantees with their costs, advancing AIC through principled economic feasibility assessment across different deployment contexts. |
| 2025-06-05 | [Backward Responsibility in Transition Systems Beyond Safety](http://arxiv.org/abs/2506.05192v1) | Christel Baier, Rio Klatt et al. | As the complexity of software systems rises, methods for explaining their behaviour are becoming ever-more important. When a system fails, it is critical to determine which of its components are responsible for this failure. Within the verification community, one approach uses graph games and the Shapley value to ascribe a responsibility value to every state of a transition system. As this is done with respect to a specific failure, it is called backward responsibility.   This paper provides tight complexity bounds for backward responsibility for reachability, B\"uchi and parity objectives. For B\"uchi objectives, a polynomial algorithm is given to determine the set of responsible states. To analyse systems that are too large for standard methods, the paper presents a novel refinement algorithm that iteratively computes responsibility and demonstrates its utility with a prototypical implementation. |
| 2025-06-05 | [Towards provable probabilistic safety for scalable embodied AI systems](http://arxiv.org/abs/2506.05171v1) | Linxuan He, Qing-Shan Jia et al. | Embodied AI systems, comprising AI models and physical plants, are increasingly prevalent across various applications. Due to the rarity of system failures, ensuring their safety in complex operating environments remains a major challenge, which severely hinders their large-scale deployment in safety-critical domains, such as autonomous vehicles, medical devices, and robotics. While achieving provable deterministic safety--verifying system safety across all possible scenarios--remains theoretically ideal, the rarity and complexity of corner cases make this approach impractical for scalable embodied AI systems. To address this challenge, we introduce provable probabilistic safety, which aims to ensure that the residual risk of large-scale deployment remains below a predefined threshold. Instead of attempting exhaustive safety proof across all corner cases, this paradigm establishes a probabilistic safety boundary on overall system performance, leveraging statistical methods to enhance feasibility and scalability. A well-defined probabilistic safety boundary enables embodied AI systems to be deployed at scale while allowing for continuous refinement of safety guarantees. Our work focuses on three core questions: what is provable probabilistic safety, how to prove the probabilistic safety, and how to achieve the provable probabilistic safety. By bridging the gap between theoretical safety assurance and practical deployment, our work offers a pathway toward safer, large-scale adoption of embodied AI systems in safety-critical applications. |
| 2025-06-05 | [Whole-Body Constrained Learning for Legged Locomotion via Hierarchical Optimization](http://arxiv.org/abs/2506.05115v1) | Haoyu Wang, Ruyi Zhou et al. | Reinforcement learning (RL) has demonstrated impressive performance in legged locomotion over various challenging environments. However, due to the sim-to-real gap and lack of explainability, unconstrained RL policies deployed in the real world still suffer from inevitable safety issues, such as joint collisions, excessive torque, or foot slippage in low-friction environments. These problems limit its usage in missions with strict safety requirements, such as planetary exploration, nuclear facility inspection, and deep-sea operations. In this paper, we design a hierarchical optimization-based whole-body follower, which integrates both hard and soft constraints into RL framework to make the robot move with better safety guarantees. Leveraging the advantages of model-based control, our approach allows for the definition of various types of hard and soft constraints during training or deployment, which allows for policy fine-tuning and mitigates the challenges of sim-to-real transfer. Meanwhile, it preserves the robustness of RL when dealing with locomotion in complex unstructured environments. The trained policy with introduced constraints was deployed in a hexapod robot and tested in various outdoor environments, including snow-covered slopes and stairs, demonstrating the great traversability and safety of our approach. |
| 2025-06-05 | [FG 2025 TrustFAA: the First Workshop on Towards Trustworthy Facial Affect Analysis: Advancing Insights of Fairness, Explainability, and Safety (TrustFAA)](http://arxiv.org/abs/2506.05095v1) | Jiaee Cheong, Yang Liu et al. | With the increasing prevalence and deployment of Emotion AI-powered facial affect analysis (FAA) tools, concerns about the trustworthiness of these systems have become more prominent. This first workshop on "Towards Trustworthy Facial Affect Analysis: Advancing Insights of Fairness, Explainability, and Safety (TrustFAA)" aims to bring together researchers who are investigating different challenges in relation to trustworthiness-such as interpretability, uncertainty, biases, and privacy-across various facial affect analysis tasks, including macro/ micro-expression recognition, facial action unit detection, other corresponding applications such as pain and depression detection, as well as human-robot interaction and collaboration. In alignment with FG2025's emphasis on ethics, as demonstrated by the inclusion of an Ethical Impact Statement requirement for this year's submissions, this workshop supports FG2025's efforts by encouraging research, discussion and dialogue on trustworthy FAA. |
| 2025-06-05 | [Tech-ASan: Two-stage check for Address Sanitizer](http://arxiv.org/abs/2506.05022v1) | Yixuan Cao, Yuhong Feng et al. | Address Sanitizer (ASan) is a sharp weapon for detecting memory safety violations, including temporal and spatial errors hidden in C/C++ programs during execution. However, ASan incurs significant runtime overhead, which limits its efficiency in testing large software. The overhead mainly comes from sanitizer checks due to the frequent and expensive shadow memory access. Over the past decade, many methods have been developed to speed up ASan by eliminating and accelerating sanitizer checks, however, they either fail to adequately eliminate redundant checks or compromise detection capabilities. To address this issue, this paper presents Tech-ASan, a two-stage check based technique to accelerate ASan with safety assurance. First, we propose a novel two-stage check algorithm for ASan, which leverages magic value comparison to reduce most of the costly shadow memory accesses. Second, we design an efficient optimizer to eliminate redundant checks, which integrates a novel algorithm for removing checks in loops. Third, we implement Tech-ASan as a memory safety tool based on the LLVM compiler infrastructure. Our evaluation using the SPEC CPU2006 benchmark shows that Tech-ASan outperforms the state-of-the-art methods with 33.70% and 17.89% less runtime overhead than ASan and ASan--, respectively. Moreover, Tech-ASan detects 56 fewer false negative cases than ASan and ASan-- when testing on the Juliet Test Suite under the same redzone setting. |
| 2025-06-05 | [Characterization of NbTi wires for the electron-ion collider project](http://arxiv.org/abs/2506.05004v1) | Jun Lu, Jeremy Levitan Noah Gavin et al. | The Electron-Ion Collider (EIC) is a proposed machine to explore the behaviour of the fundamental particles and forces that bind atomic nuclei together. The design and construction of the EIC are underway at Brookhaven National Laboratory in collaboration with Thomas Jefferson National Accelerator Facility. EIC will use several different types of superconducting strands for magnets near the interaction region (IR). At beam injection, the magnetic field is usually very low compared with its maximum operating field. This usually creates considerable field errors mainly generated from magnetization current in superconducting strands even using very fine filament. The accurate magnetization measurement results from those superconducting strands will be critical for the calculation and future correction of magnetic field for EIC. In this work, we characterized three billets of superconductor NbTi strands. The magnetization was measured at 4.2 K and 1.9 K in magnetic fields below 1.5 T. The critical current at 4.2 K and in magnetic field down to 5 T were also measured. Other properties that are important for the safety margin of superconducting magnet fabrication, operation, and quench protection such as residual-resistance-ratio (RRR), filament diameter, Cu to non-Cu ratio, twist pitch, and mechanical properties at 77 K will also be presented. |
| 2025-06-05 | [Mathematical Reasoning for Unmanned Aerial Vehicles: A RAG-Based Approach for Complex Arithmetic Reasoning](http://arxiv.org/abs/2506.04998v1) | Mehdi Azarafza, Mojtaba Nayyeri et al. | Autonomous UAV operation necessitates reliable mathematical reasoning for tasks such as trajectory planning and power management. While traditional flight control relies on hardcoded equations, recent Large Language Models (LLMs) offer potential for more flexible problem-solving but struggle with reliably selecting and applying correct mathematical formulations and executing precise multi-step arithmetic. We propose RAG-UAV, a retrieval-augmented generation framework designed to improve the mathematical reasoning of several LLMs (including GPT o1/Turbo, Llama-3.2/3.3, Mistral, and DeepSeek R1) in UAV-specific contexts by providing access to relevant domain literature. To conduct an initial assessment, we introduce the UAV-Math-Bench, a small problem set comprising 20 UAV-centric mathematical problems across four difficulty levels. Our experiments demonstrate that incorporating retrieval substantially increases exact answer accuracy (achieving up to 75% with o1), reduces instances of incorrect formulation selection (from 25% without RAG to 5% with RAG), decreases numerical errors, reducing Mean Squared Error (MSE) by orders of magnitude for the best-performing models. This pilot study indicates that RAG can enable general-purpose LLMs to function as more reliable tools for engineering analysis, although direct real-time flight control requires further investigation and validation on a larger scale. All benchmark data, question and answer are publicly available. |
| 2025-06-05 | [Evaluating Prompt-Driven Chinese Large Language Models: The Influence of Persona Assignment on Stereotypes and Safeguards](http://arxiv.org/abs/2506.04975v1) | Geng Liu, Li Feng et al. | Recent research has highlighted that assigning specific personas to large language models (LLMs) can significantly increase harmful content generation. Yet, limited attention has been given to persona-driven toxicity in non-Western contexts, particularly in Chinese-based LLMs. In this paper, we perform a large-scale, systematic analysis of how persona assignment influences refusal behavior and response toxicity in Qwen, a widely-used Chinese language model. Utilizing fine-tuned BERT classifiers and regression analysis, our study reveals significant gender biases in refusal rates and demonstrates that certain negative personas can amplify toxicity toward Chinese social groups by up to 60-fold compared to the default model. To mitigate this toxicity, we propose an innovative multi-model feedback strategy, employing iterative interactions between Qwen and an external evaluator, which effectively reduces toxic outputs without costly model retraining. Our findings emphasize the necessity of culturally specific analyses for LLMs safety and offer a practical framework for evaluating and enhancing ethical alignment in LLM-generated content. |
| 2025-06-05 | [ICPC-Eval: Probing the Frontiers of LLM Reasoning with Competitive Programming Contests](http://arxiv.org/abs/2506.04894v1) | Shiyi Xu, Yiwen Hu et al. | With the significant progress of large reasoning models in complex coding and reasoning tasks, existing benchmarks, like LiveCodeBench and CodeElo, are insufficient to evaluate the coding capabilities of large language models (LLMs) in real competition environments. Moreover, current evaluation metrics such as Pass@K fail to capture the reflective abilities of reasoning models. To address these challenges, we propose \textbf{ICPC-Eval}, a top-level competitive coding benchmark designed to probing the frontiers of LLM reasoning. ICPC-Eval includes 118 carefully curated problems from 11 recent ICPC contests held in various regions of the world, offering three key contributions: 1) A challenging realistic ICPC competition scenario, featuring a problem type and difficulty distribution consistent with actual contests. 2) A robust test case generation method and a corresponding local evaluation toolkit, enabling efficient and accurate local evaluation. 3) An effective test-time scaling evaluation metric, Refine@K, which allows iterative repair of solutions based on execution feedback. The results underscore the significant challenge in evaluating complex reasoning abilities: top-tier reasoning models like DeepSeek-R1 often rely on multi-turn code feedback to fully unlock their in-context reasoning potential when compared to non-reasoning counterparts. Furthermore, despite recent advancements in code generation, these models still lag behind top-performing human teams. We release the benchmark at: https://github.com/RUCAIBox/Slow_Thinking_with_LLMs |
| 2025-06-05 | [Safe Planning and Policy Optimization via World Model Learning](http://arxiv.org/abs/2506.04828v1) | Artem Latyshev, Gregory Gorbov et al. | Reinforcement Learning (RL) applications in real-world scenarios must prioritize safety and reliability, which impose strict constraints on agent behavior. Model-based RL leverages predictive world models for action planning and policy optimization, but inherent model inaccuracies can lead to catastrophic failures in safety-critical settings. We propose a novel model-based RL framework that jointly optimizes task performance and safety. To address world model errors, our method incorporates an adaptive mechanism that dynamically switches between model-based planning and direct policy execution. We resolve the objective mismatch problem of traditional model-based approaches using an implicit world model. Furthermore, our framework employs dynamic safety thresholds that adapt to the agent's evolving capabilities, consistently selecting actions that surpass safe policy suggestions in both performance and safety. Experiments demonstrate significant improvements over non-adaptive methods, showing that our approach optimizes safety and performance simultaneously rather than merely meeting minimum safety requirements. The proposed framework achieves robust performance on diverse safety-critical continuous control tasks, outperforming existing methods. |
| 2025-06-05 | [Multi-Layer GRPO: Enhancing Reasoning and Self-Correction in Large Language Models](http://arxiv.org/abs/2506.04746v1) | Fei Ding, Baiqiao Wang et al. | The Group Relative Policy Optimization (GRPO) algorithm has demonstrated considerable success in enhancing the reasoning capabilities of large language models (LLMs), as evidenced by DeepSeek-R1. However, the absence of intermediate supervision in GRPO frequently leads to inefficient exploration dynamics. A single error in a complex reasoning chain can invalidate the entire solution, resulting in abrupt reward vanishing and compromising training stability.To address these challenges, we propose MGRPO (Multi-layer GRPO). MGRPO operates in two layers: the first layer employs standard GRPO to generate an initial response. This response, along with the original query, is then fed into a second-layer GRPO process. This second layer is specifically trained to identify and correct errors in the initial response, effectively creating a self-correction loop. This mechanism provides implicit process-level supervision by rewarding successful error correction, without requiring an explicit, densely-annotated reward model. Experimental results on several mathematical reasoning benchmarks demonstrate that MGRPO significantly outperforms standard GRPO, achieving superior performance by fostering both reasoning and self-correction abilities. |
| 2025-06-05 | [Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design](http://arxiv.org/abs/2506.04734v1) | Lin Sun, Weihong Lin et al. | Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models. |
| 2025-06-05 | [HoliSafe: Holistic Safety Benchmarking and Modeling with Safety Meta Token for Vision-Language Model](http://arxiv.org/abs/2506.04704v1) | Youngwan Lee, Kangsan Kim et al. | Despite emerging efforts to enhance the safety of Vision-Language Models (VLMs), current approaches face two main shortcomings. 1) Existing safety-tuning datasets and benchmarks only partially consider how image-text interactions can yield harmful content, often overlooking contextually unsafe outcomes from seemingly benign pairs. This narrow coverage leaves VLMs vulnerable to jailbreak attacks in unseen configurations. 2) Prior methods rely primarily on data-centric tuning, with limited architectural innovations to intrinsically strengthen safety. We address these gaps by introducing a holistic safety dataset and benchmark, HoliSafe, that spans all five safe/unsafe image-text combinations, providing a more robust basis for both training and evaluation. We further propose SafeLLaVA, a novel VLM augmented with a learnable safety meta token and a dedicated safety head. The meta token encodes harmful visual cues during training, intrinsically guiding the language model toward safer responses, while the safety head offers interpretable harmfulness classification aligned with refusal rationales. Experiments show that SafeLLaVA, trained on HoliSafe, achieves state-of-the-art safety performance across multiple VLM benchmarks. Additionally, the HoliSafe benchmark itself reveals critical vulnerabilities in existing models. We hope that HoliSafe and SafeLLaVA will spur further research into robust and interpretable VLM safety, expanding future avenues for multimodal alignment. |
| 2025-06-05 | [Energy Consumption Optimization for Autonomous Vehicles via Positive Control Input Minimization](http://arxiv.org/abs/2506.04685v1) | Andreas Hadjigeorgiou, Stelios Timotheou | Autonomous vehicles (AVs) present a unique opportunity to improve the sustainability of transportation systems by adopting eco-driving strategies that reduce energy consumption and emissions. This paper introduces a novel surrogate model for energy and fuel consumption that minimizes Positive Control Input (PCI). Unlike conventional objectives such as squared acceleration, which often misrepresent actual energy usage, PCI provides a more accurate and optimization-friendly alternative. Building on PCI, we propose ECO+, a convex, time-based trajectory optimization framework that ensures safety and passenger comfort while optimizing energy use for AVs approaching an intersection. To improve computational efficiency, quadratic resistive forces are approximated using piecewise affine segments, resulting in a linear programming formulation. ECO+ is validated using empirical fuel and electric energy models and benchmarked against established optimization strategies, including a state-of-the-art nonlinear solver. Simulation results demonstrate that ECO+ consistently outperforms baseline methods in reducing energy consumption, even under strict comfort constraints and in scenarios involving a leading vehicle. Moreover, initializing a nonlinear solver with ECO+ yields only marginal gains, indicating that ECO+ is effective as a standalone eco-driving strategy. These findings highlight ECO+ as a practical, scalable, and computationally efficient solution for enhancing the sustainability of autonomous urban mobility systems. |
| 2025-06-05 | [Real-Time LPV-Based Non-Linear Model Predictive Control for Robust Trajectory Tracking in Autonomous Vehicles](http://arxiv.org/abs/2506.04684v1) | Nitish Kumar, Rajalakshmi Pachamuthu | This paper presents the development and implementation of a Model Predictive Control (MPC) framework for trajectory tracking in autonomous vehicles under diverse driving conditions. The proposed approach incorporates a modular architecture that integrates state estimation, vehicle dynamics modeling, and optimization to ensure real-time performance. The state-space equations are formulated in a Linear Parameter Varying (LPV) form, and a curvature-based tuning method is introduced to optimize weight matrices for varying trajectories. The MPC framework is implemented using the Robot Operating System (ROS) for parallel execution of state estimation and control optimization, ensuring scalability and minimal latency. Extensive simulations and real-time experiments were conducted on multiple predefined trajectories, demonstrating high accuracy with minimal cross-track and orientation errors, even under aggressive maneuvers and high-speed conditions. The results highlight the robustness and adaptability of the proposed system, achieving seamless alignment between simulated and real-world performance. This work lays the foundation for dynamic weight tuning and integration into cooperative autonomous navigation systems, paving the way for enhanced safety and efficiency in autonomous driving applications. |
| 2025-06-05 | [Normative Conflicts and Shallow AI Alignment](http://arxiv.org/abs/2506.04679v1) | Raphaël Millière | The progress of AI systems such as large language models (LLMs) raises increasingly pressing concerns about their safe deployment. This paper examines the value alignment problem for LLMs, arguing that current alignment strategies are fundamentally inadequate to prevent misuse. Despite ongoing efforts to instill norms such as helpfulness, honesty, and harmlessness in LLMs through fine-tuning based on human preferences, they remain vulnerable to adversarial attacks that exploit conflicts between these norms. I argue that this vulnerability reflects a fundamental limitation of existing alignment methods: they reinforce shallow behavioral dispositions rather than endowing LLMs with a genuine capacity for normative deliberation. Drawing from on research in moral psychology, I show how humans' ability to engage in deliberative reasoning enhances their resilience against similar adversarial tactics. LLMs, by contrast, lack a robust capacity to detect and rationally resolve normative conflicts, leaving them susceptible to manipulation; even recent advances in reasoning-focused LLMs have not addressed this vulnerability. This ``shallow alignment'' problem carries significant implications for AI safety and regulation, suggesting that current approaches are insufficient for mitigating potential harms posed by increasingly capable AI systems. |
| 2025-06-05 | [E-bike agents: Large Language Model-Driven E-Bike Accident Analysis and Severity Prediction](http://arxiv.org/abs/2506.04654v1) | Zhichao Yang, Jiashu He et al. | Electric bicycles (e-bikes) are rapidly increasing in use, raising safety concerns due to a rise in accident reports. However, e-bike incident reports often use unstructured narrative formats, which hinders quantitative safety analysis. This study introduces E-bike agents, a framework that uses large language models (LLM) powered agents to classify and extract safety variables from unstructured incident reports. Our framework consists of four LLM agents, handling data classification, information extraction, injury cause determination, and component linkage, to extract the key factors that could lead to E-bike accidents and cause varying severity levels. Furthermore, we used an ordered logit model to examine the relationship between the severity of the incident and the factors retrieved, such as gender, the type of cause, and environmental conditions. Our research shows that equipment issues are slightly more common than human-related ones, but human-related incidents are more often fatal. Specifically, pedals, tires, and brakes are frequent contributors to accidents. The model achieves a high weighted F1 score of 0.87 in classification accuracy, highlighting the potential of using LLMs to extract unstructured data in niche domains, such as transportation. Our method offers a scalable solution to improve e-bike safety analytics and provides actionable information for policy makers, designers, and regulators. |
| 2025-06-05 | [Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations](http://arxiv.org/abs/2506.04633v1) | Linjie Li, Mahtab Bigverdi et al. | Spatial cognition is essential for human intelligence, enabling problem-solving through visual simulations rather than solely relying on verbal reasoning. However, existing AI benchmarks primarily assess verbal reasoning, neglecting the complexities of non-verbal, multi-step visual simulation. We introduce STARE(Spatial Transformations and Reasoning Evaluation), a benchmark designed to rigorously evaluate multimodal large language models on tasks better solved through multi-step visual simulation. STARE features 4K tasks spanning foundational geometric transformations (2D and 3D), integrated spatial reasoning (cube net folding and tangram puzzles), and real-world spatial reasoning (perspective and temporal reasoning), reflecting practical cognitive challenges like object assembly, mechanical diagram interpretation, and everyday spatial navigation. Our evaluations show that models excel at reasoning over simpler 2D transformations, but perform close to random chance on more complex tasks like 3D cube net folding and tangram puzzles that require multi-step visual simulations. Humans achieve near-perfect accuracy but take considerable time (up to 28.9s) on complex tasks, significantly speeding up (down by 7.5 seconds on average) with intermediate visual simulations. In contrast, models exhibit inconsistent performance gains from visual simulations, improving on most tasks but declining in specific cases like tangram puzzles (GPT-4o, o1) and cube net folding (Claude-3.5, Gemini-2.0 Flash), indicating that models may not know how to effectively leverage intermediate visual information. |

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



