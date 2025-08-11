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
| 2025-08-08 | [Voting-Based Semi-Parallel Proof-of-Work Protocol](http://arxiv.org/abs/2508.06489v1) | Mustafa Doger, Sennur Ulukus | Parallel Proof-of-Work (PoW) protocols are suggested to improve the safety guarantees, transaction throughput and confirmation latencies of Nakamoto consensus. In this work, we first consider the existing parallel PoW protocols and develop hard-coded incentive attack structures. Our theoretical results and simulations show that the existing parallel PoW protocols are more vulnerable to incentive attacks than the Nakamoto consensus, e.g., attacks have smaller profitability threshold and they result in higher relative rewards. Next, we introduce a voting-based semi-parallel PoW protocol that outperforms both Nakamoto consensus and the existing parallel PoW protocols from most practical perspectives such as communication overheads, throughput, transaction conflicts, incentive compatibility of the protocol as well as a fair distribution of transaction fees among the voters and the leaders. We use state-of-the-art analysis to evaluate the consistency of the protocol and consider Markov decision process (MDP) models to substantiate our claims about the resilience of our protocol against incentive attacks. |
| 2025-08-08 | [The Problem of Atypicality in LLM-Powered Psychiatry](http://arxiv.org/abs/2508.06479v1) | Bosco Garcia, Eugene Y. S. Chua et al. | Large language models (LLMs) are increasingly proposed as scalable solutions to the global mental health crisis. But their deployment in psychiatric contexts raises a distinctive ethical concern: the problem of atypicality. Because LLMs generate outputs based on population-level statistical regularities, their responses -- while typically appropriate for general users -- may be dangerously inappropriate when interpreted by psychiatric patients, who often exhibit atypical cognitive or interpretive patterns. We argue that standard mitigation strategies, such as prompt engineering or fine-tuning, are insufficient to resolve this structural risk. Instead, we propose dynamic contextual certification (DCC): a staged, reversible and context-sensitive framework for deploying LLMs in psychiatry, inspired by clinical translation and dynamic safety models from artificial intelligence governance. DCC reframes chatbot deployment as an ongoing epistemic and ethical process that prioritises interpretive safety over static performance benchmarks. Atypicality, we argue, cannot be eliminated -- but it can, and must, be proactively managed. |
| 2025-08-08 | [ScamAgents: How AI Agents Can Simulate Human-Level Scam Calls](http://arxiv.org/abs/2508.06457v1) | Sanket Badhe | Large Language Models (LLMs) have demonstrated impressive fluency and reasoning capabilities, but their potential for misuse has raised growing concern. In this paper, we present ScamAgent, an autonomous multi-turn agent built on top of LLMs, capable of generating highly realistic scam call scripts that simulate real-world fraud scenarios. Unlike prior work focused on single-shot prompt misuse, ScamAgent maintains dialogue memory, adapts dynamically to simulated user responses, and employs deceptive persuasion strategies across conversational turns. We show that current LLM safety guardrails, including refusal mechanisms and content filters, are ineffective against such agent-based threats. Even models with strong prompt-level safeguards can be bypassed when prompts are decomposed, disguised, or delivered incrementally within an agent framework. We further demonstrate the transformation of scam scripts into lifelike voice calls using modern text-to-speech systems, completing a fully automated scam pipeline. Our findings highlight an urgent need for multi-turn safety auditing, agent-level control frameworks, and new methods to detect and disrupt conversational deception powered by generative AI. |
| 2025-08-08 | [ActivityDiff: A diffusion model with Positive and Negative Activity Guidance for De Novo Drug Design](http://arxiv.org/abs/2508.06364v1) | Renyi Zhou, Huimin Zhu et al. | Achieving precise control over a molecule's biological activity-encompassing targeted activation/inhibition, cooperative multi-target modulation, and off-target toxicity mitigation-remains a critical challenge in de novo drug design. However, existing generative methods primarily focus on producing molecules with a single desired activity, lacking integrated mechanisms for the simultaneous management of multiple intended and unintended molecular interactions. Here, we propose ActivityDiff, a generative approach based on the classifier-guidance technique of diffusion models. It leverages separately trained drug-target classifiers for both positive and negative guidance, enabling the model to enhance desired activities while minimizing harmful off-target effects. Experimental results show that ActivityDiff effectively handles essential drug design tasks, including single-/dual-target generation, fragment-constrained dual-target design, selective generation to enhance target specificity, and reduction of off-target effects. These results demonstrate the effectiveness of classifier-guided diffusion in balancing efficacy and safety in molecular design. Overall, our work introduces a novel paradigm for achieving integrated control over molecular activity, and provides ActivityDiff as a versatile and extensible framework. |
| 2025-08-08 | [Cyberbullying Detection via Aggression-Enhanced Prompting](http://arxiv.org/abs/2508.06360v1) | Aisha Saeid, Anu Sabu et al. | Detecting cyberbullying on social media remains a critical challenge due to its subtle and varied expressions. This study investigates whether integrating aggression detection as an auxiliary task within a unified training framework can enhance the generalisation and performance of large language models (LLMs) in cyberbullying detection. Experiments are conducted on five aggression datasets and one cyberbullying dataset using instruction-tuned LLMs. We evaluated multiple strategies: zero-shot, few-shot, independent LoRA fine-tuning, and multi-task learning (MTL). Given the inconsistent results of MTL, we propose an enriched prompt pipeline approach in which aggression predictions are embedded into cyberbullying detection prompts to provide contextual augmentation. Preliminary results show that the enriched prompt pipeline consistently outperforms standard LoRA fine-tuning, indicating that aggression-informed context significantly boosts cyberbullying detection. This study highlights the potential of auxiliary tasks, such as aggression detection, to improve the generalisation of LLMs for safety-critical applications on social networks. |
| 2025-08-08 | [LLM Robustness Leaderboard v1 --Technical report](http://arxiv.org/abs/2508.06296v1) | Pierre Peigné - Lefebvre, Quentin Feuillade-Montixi et al. | This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community. |
| 2025-08-08 | [Beyond Uniform Criteria: Scenario-Adaptive Multi-Dimensional Jailbreak Evaluation](http://arxiv.org/abs/2508.06194v1) | Lai Jiang, Yuekang Li et al. | Precise jailbreak evaluation is vital for LLM red teaming and jailbreak research. Current approaches employ binary classification ( e.g., string matching, toxic text classifiers, LLM-driven methods), yielding only "yes/no" labels without quantifying harm intensity. Existing multi-dimensional frameworks ( e.g., Security Violation, Relative Truthfulness, Informativeness) apply uniform evaluation criteria across scenarios, resulting in scenario-specific mismatches--for instance, "Relative Truthfulness" is irrelevant to "hate speech"--which compromise evaluation precision. To tackle these limitations, we introduce SceneJailEval, with key contributions: (1) A groundbreaking scenario-adaptive multi-dimensional framework for jailbreak evaluation, overcoming the critical "one-size-fits-all" constraint of existing multi-dimensional methods, and featuring strong extensibility to flexibly adapt to customized or emerging scenarios. (2) A comprehensive 14-scenario dataset with diverse jailbreak variants and regional cases, filling the long-standing gap in high-quality, holistic benchmarks for scenario-adaptive evaluation. (3) SceneJailEval achieves state-of-the-art results, with an F1 score of 0.917 on our full-scenario dataset (+6% over prior SOTA) and 0.995 on JBB (+3% over prior SOTA), surpassing accuracy limits of existing evaluation methods in heterogeneous scenarios and confirming its advantage. |
| 2025-08-08 | [MA-CBP: A Criminal Behavior Prediction Framework Based on Multi-Agent Asynchronous Collaboration](http://arxiv.org/abs/2508.06189v1) | Cheng Liu, Daou Zhang et al. | With the acceleration of urbanization, criminal behavior in public scenes poses an increasingly serious threat to social security. Traditional anomaly detection methods based on feature recognition struggle to capture high-level behavioral semantics from historical information, while generative approaches based on Large Language Models (LLMs) often fail to meet real-time requirements. To address these challenges, we propose MA-CBP, a criminal behavior prediction framework based on multi-agent asynchronous collaboration. This framework transforms real-time video streams into frame-level semantic descriptions, constructs causally consistent historical summaries, and fuses adjacent image frames to perform joint reasoning over long- and short-term contexts. The resulting behavioral decisions include key elements such as event subjects, locations, and causes, enabling early warning of potential criminal activity. In addition, we construct a high-quality criminal behavior dataset that provides multi-scale language supervision, including frame-level, summary-level, and event-level semantic annotations. Experimental results demonstrate that our method achieves superior performance on multiple datasets and offers a promising solution for risk warning in urban public safety scenarios. |
| 2025-08-08 | [SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](http://arxiv.org/abs/2508.06142v1) | Hanqing Wang, Yuan Tian et al. | In the rapidly evolving landscape of Multimodal Large Language Models (MLLMs), the safety concerns of their outputs have earned significant attention. Although numerous datasets have been proposed, they may become outdated with MLLM advancements and are susceptible to data contamination issues. To address these problems, we propose \textbf{SDEval}, the \textit{first} safety dynamic evaluation framework to controllably adjust the distribution and complexity of safety benchmarks. Specifically, SDEval mainly adopts three dynamic strategies: text, image, and text-image dynamics to generate new samples from original benchmarks. We first explore the individual effects of text and image dynamics on model safety. Then, we find that injecting text dynamics into images can further impact safety, and conversely, injecting image dynamics into text also leads to safety risks. SDEval is general enough to be applied to various existing safety and even capability benchmarks. Experiments across safety benchmarks, MLLMGuard and VLSBench, and capability benchmarks, MMBench and MMVet, show that SDEval significantly influences safety evaluation, mitigates data contamination, and exposes safety limitations of MLLMs. Code is available at https://github.com/hq-King/SDEval |
| 2025-08-08 | [AURA: Affordance-Understanding and Risk-aware Alignment Technique for Large Language Models](http://arxiv.org/abs/2508.06124v1) | Sayantan Adak, Pratyush Chatterjee et al. | Present day LLMs face the challenge of managing affordance-based safety risks-situations where outputs inadvertently facilitate harmful actions due to overlooked logical implications. Traditional safety solutions, such as scalar outcome-based reward models, parameter tuning, or heuristic decoding strategies, lack the granularity and proactive nature needed to reliably detect and intervene during subtle yet crucial reasoning steps. Addressing this fundamental gap, we introduce AURA, an innovative, multi-layered framework centered around Process Reward Models (PRMs), providing comprehensive, step level evaluations across logical coherence and safety-awareness. Our framework seamlessly combines introspective self-critique, fine-grained PRM assessments, and adaptive safety-aware decoding to dynamically and proactively guide models toward safer reasoning trajectories. Empirical evidence clearly demonstrates that this approach significantly surpasses existing methods, significantly improving the logical integrity and affordance-sensitive safety of model outputs. This research represents a pivotal step toward safer, more responsible, and contextually aware AI, setting a new benchmark for alignment-sensitive applications. |
| 2025-08-08 | [ArchXBench: A Complex Digital Systems Benchmark Suite for LLM Driven RTL Synthesis](http://arxiv.org/abs/2508.06047v1) | Suresh Purini, Siddhant Garg et al. | Modern SoC datapaths include deeply pipelined, domain-specific accelerators, but their RTL implementation and verification are still mostly done by hand. While large language models (LLMs) exhibit advanced code-generation abilities for programming languages like Python, their application to Verilog-like RTL remains in its nascent stage. This is reflected in the simple arithmetic and control circuits currently used to evaluate generative capabilities in existing benchmarks. In this paper, we introduce ArchXBench, a six-level benchmark suite that encompasses complex arithmetic circuits and other advanced digital subsystems drawn from domains such as cryptography, image processing, machine learning, and signal processing. Architecturally, some of these designs are purely combinational, others are multi-cycle or pipelined, and many require hierarchical composition of modules. For each benchmark, we provide a problem description, design specification, and testbench, enabling rapid research in the area of LLM-driven agentic approaches for complex digital systems design.   Using zero-shot prompting with Claude Sonnet 4, GPT 4.1, o4-mini-high, and DeepSeek R1 under a pass@5 criterion, we observed that o4-mini-high successfully solves the largest number of benchmarks, 16 out of 30, spanning Levels 1, 2, and 3. From Level 4 onward, however, all models consistently fail, highlighting a clear gap in the capabilities of current state-of-the-art LLMs and prompting/agentic approaches. |
| 2025-08-08 | [Vacuum Dealloyed Brass as Li-Metal Battery Current Collector: Effect of Zinc and Porosity](http://arxiv.org/abs/2508.06015v1) | Eric V Woods, Xinren Chen et al. | "Anode-free" lithium-metal batteries promise significantly higher energy density than conventional graphite-based lithium-ion batteries; however, lithium dendrite growth can lead to internal short circuits with associated safety risks. While porous current collectors can suppress dendrite growth, optimal porosity and composition remain unknown. Here, we show that the temperature during vapor phase dealloying (VPD) of alpha-brass (Cu63Zn37) controls the surface Zn concentration, decreasing from 8 percent to below 1 percent from 500 to 800 degrees C. The surface composition is controlled by the temperature-dependent diffusion. A battery cell maintains greater than 90 percent Coulombic efficiency (CE) over 100 cycles when the Zn content is the lowest, whereas the higher-Zn samples degraded to approximately 70 percent CE. The difference in surface composition has hence dramatic effects on battery performance, and our results demonstrate how precise compositional control enables stable lithium-metal battery operation, establishing about 1 atomic percent surface Zn as optimal for preventing capacity fading and uniform lithium plating, while establishing predictive relationships between processing temperature and surface composition. This work provides design rules for multifunctional current collectors and demonstrates scalable VPD production for next-generation batteries. |
| 2025-08-08 | [Dynamical Trajectory Planning of Disturbance Consciousness for Air-Land Bimodal Unmanned Aerial Vehicles](http://arxiv.org/abs/2508.05972v1) | Shaoting Liu, Zhou Liu | Air-land bimodal vehicles provide a promising solution for navigating complex environments by combining the flexibility of aerial locomotion with the energy efficiency of ground mobility. To enhance the robustness of trajectory planning under environmental disturbances, this paper presents a disturbance-aware planning framework that incorporates real-time disturbance estimation into both path searching and trajectory optimization. A key component of the framework is a disturbance-adaptive safety boundary adjustment mechanism, which dynamically modifies the vehicle's feasible dynamic boundaries based on estimated disturbances to ensure trajectory feasibility. Leveraging the dynamics model of the bimodal vehicle, the proposed approach achieves adaptive and reliable motion planning across different terrains and operating conditions. A series of real-world experiments and benchmark comparisons on a custom-built platform validate the effectiveness and robustness of the method, demonstrating improvements in tracking accuracy, task efficiency, and energy performance under both ground and aerial disturbances. |
| 2025-08-08 | [Dean of LLM Tutors: Exploring Comprehensive and Automated Evaluation of LLM-generated Educational Feedback via LLM Feedback Evaluators](http://arxiv.org/abs/2508.05952v1) | Keyang Qian, Yixin Cheng et al. | The use of LLM tutors to provide automated educational feedback to students on student assignment submissions has received much attention in the AI in Education field. However, the stochastic nature and tendency for hallucinations in LLMs can undermine both quality of learning experience and adherence to ethical standards. To address this concern, we propose a method that uses LLM feedback evaluators (DeanLLMs) to automatically and comprehensively evaluate feedback generated by LLM tutor for submissions on university assignments before it is delivered to students. This allows low-quality feedback to be rejected and enables LLM tutors to improve the feedback they generated based on the evaluation results. We first proposed a comprehensive evaluation framework for LLM-generated educational feedback, comprising six dimensions for feedback content, seven for feedback effectiveness, and three for hallucination types. Next, we generated a virtual assignment submission dataset covering 85 university assignments from 43 computer science courses using eight commonly used commercial LLMs. We labelled and open-sourced the assignment dataset to support the fine-tuning and evaluation of LLM feedback evaluators. Our findings show that o3-pro demonstrated the best performance in zero-shot labelling of feedback while o4-mini demonstrated the best performance in few-shot labelling of feedback. Moreover, GPT-4.1 achieved human expert level performance after fine-tuning (Accuracy 79.8%, F1-score 79.4%; human average Accuracy 78.3%, F1-score 82.6%). Finally, we used our best-performance model to evaluate 2,000 assignment feedback instances generated by 10 common commercial LLMs, 200 each, to compare the quality of feedback generated by different LLMs. Our LLM feedback evaluator method advances our ability to automatically provide high-quality and reliable educational feedback to students. |
| 2025-08-08 | [Prosocial Behavior Detection in Player Game Chat: From Aligning Human-AI Definitions to Efficient Annotation at Scale](http://arxiv.org/abs/2508.05938v1) | Rafal Kocielnik, Min Kim et al. | Detecting prosociality in text--communication intended to affirm, support, or improve others' behavior--is a novel and increasingly important challenge for trust and safety systems. Unlike toxic content detection, prosociality lacks well-established definitions and labeled data, requiring new approaches to both annotation and deployment. We present a practical, three-stage pipeline that enables scalable, high-precision prosocial content classification while minimizing human labeling effort and inference costs. First, we identify the best LLM-based labeling strategy using a small seed set of human-labeled examples. We then introduce a human-AI refinement loop, where annotators review high-disagreement cases between GPT-4 and humans to iteratively clarify and expand the task definition-a critical step for emerging annotation tasks like prosociality. This process results in improved label quality and definition alignment. Finally, we synthesize 10k high-quality labels using GPT-4 and train a two-stage inference system: a lightweight classifier handles high-confidence predictions, while only $\sim$35\% of ambiguous instances are escalated to GPT-4o. This architecture reduces inference costs by $\sim$70% while achieving high precision ($\sim$0.90). Our pipeline demonstrates how targeted human-AI interaction, careful task formulation, and deployment-aware architecture design can unlock scalable solutions for novel responsible AI tasks. |
| 2025-08-07 | [Safety of Embodied Navigation: A Survey](http://arxiv.org/abs/2508.05855v1) | Zixia Wang, Jia Hu et al. | As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency. |
| 2025-08-07 | [Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation](http://arxiv.org/abs/2508.05775v1) | Chi Zhang, Changjia Zhu et al. | Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies. |
| 2025-08-07 | [A Framework for Inherently Safer AGI through Language-Mediated Active Inference](http://arxiv.org/abs/2508.05766v1) | Bo Wen | This paper proposes a novel framework for developing safe Artificial General Intelligence (AGI) by combining Active Inference principles with Large Language Models (LLMs). We argue that traditional approaches to AI safety, focused on post-hoc interpretability and reward engineering, have fundamental limitations. We present an architecture where safety guarantees are integrated into the system's core design through transparent belief representations and hierarchical value alignment. Our framework leverages natural language as a medium for representing and manipulating beliefs, enabling direct human oversight while maintaining computational tractability. The architecture implements a multi-agent system where agents self-organize according to Active Inference principles, with preferences and safety constraints flowing through hierarchical Markov blankets. We outline specific mechanisms for ensuring safety, including: (1) explicit separation of beliefs and preferences in natural language, (2) bounded rationality through resource-aware free energy minimization, and (3) compositional safety through modular agent structures. The paper concludes with a research agenda centered on the Abstraction and Reasoning Corpus (ARC) benchmark, proposing experiments to validate our framework's safety properties. Our approach offers a path toward AGI development that is inherently safer, rather than retrofitted with safety measures. |
| 2025-08-07 | [Towards Generalizable Safety in Crowd Navigation via Conformal Uncertainty Handling](http://arxiv.org/abs/2508.05634v1) | Jianpeng Yao, Xiaopan Zhang et al. | Mobile robots navigating in crowds trained using reinforcement learning are known to suffer performance degradation when faced with out-of-distribution scenarios. We propose that by properly accounting for the uncertainties of pedestrians, a robot can learn safe navigation policies that are robust to distribution shifts. Our method augments agent observations with prediction uncertainty estimates generated by adaptive conformal inference, and it uses these estimates to guide the agent's behavior through constrained reinforcement learning. The system helps regulate the agent's actions and enables it to adapt to distribution shifts. In the in-distribution setting, our approach achieves a 96.93% success rate, which is over 8.80% higher than the previous state-of-the-art baselines with over 3.72 times fewer collisions and 2.43 times fewer intrusions into ground-truth human future trajectories. In three out-of-distribution scenarios, our method shows much stronger robustness when facing distribution shifts in velocity variations, policy changes, and transitions from individual to group dynamics. We deploy our method on a real robot, and experiments show that the robot makes safe and robust decisions when interacting with both sparse and dense crowds. Our code and videos are available on https://gen-safe-nav.github.io/. |
| 2025-08-07 | [TrajEvo: Trajectory Prediction Heuristics Design via LLM-driven Evolution](http://arxiv.org/abs/2508.05616v1) | Zhikai Zhao, Chuanbo Hua et al. | Trajectory prediction is a critical task in modeling human behavior, especially in safety-critical domains such as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy and generalizability. Although deep learning approaches offer improved performance, they typically suffer from high computational cost, limited explainability, and, importantly, poor generalization to out-of-distribution (OOD) scenarios. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We propose two key innovations: Cross-Generation Elite Sampling to encourage population diversity, and a Statistics Feedback Loop that enables the LLM to analyze and improve alternative predictions. Our evaluations demonstrate that TrajEvo outperforms existing heuristic methods across multiple real-world datasets, and notably surpasses both heuristic and deep learning methods in generalizing to an unseen OOD real-world dataset. TrajEvo marks a promising step toward the automated design of fast, explainable, and generalizable trajectory prediction heuristics. We release our source code to facilitate future research at https://github.com/ai4co/trajevo. |

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



