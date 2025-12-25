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
| 2025-12-24 | [Surgical Scene Segmentation using a Spike-Driven Video Transformer with Real-Time Potential](http://arxiv.org/abs/2512.21284v1) | Shihao Zou, Jingjing Li et al. | Modern surgical systems increasingly rely on intelligent scene understanding to provide timely situational awareness for enhanced intra-operative safety. Within this pipeline, surgical scene segmentation plays a central role in accurately perceiving operative events. Although recent deep learning models, particularly large-scale foundation models, achieve remarkable segmentation accuracy, their substantial computational demands and power consumption hinder real-time deployment in resource-constrained surgical environments. To address this limitation, we explore the emerging SNN as a promising paradigm for highly efficient surgical intelligence. However, their performance is still constrained by the scarcity of labeled surgical data and the inherently sparse nature of surgical video representations. To this end, we propose \textit{SpikeSurgSeg}, the first spike-driven video Transformer framework tailored for surgical scene segmentation with real-time potential on non-GPU platforms. To address the limited availability of surgical annotations, we introduce a surgical-scene masked autoencoding pretraining strategy for SNNs that enables robust spatiotemporal representation learning via layer-wise tube masking. Building on this pretrained backbone, we further adopt a lightweight spike-driven segmentation head that produces temporally consistent predictions while preserving the low-latency characteristics of SNNs. Extensive experiments on EndoVis18 and our in-house SurgBleed dataset demonstrate that SpikeSurgSeg achieves mIoU comparable to SOTA ANN-based models while reducing inference latency by at least $8\times$. Notably, it delivers over $20\times$ acceleration relative to most foundation-model baselines, underscoring its potential for time-critical surgical scene segmentation. |
| 2025-12-24 | [Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking](http://arxiv.org/abs/2512.21236v1) | Yifan Huang, Xiaojun Jia et al. | Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications. |
| 2025-12-24 | [RoboSafe: Safeguarding Embodied Agents via Executable Safety Logic](http://arxiv.org/abs/2512.21220v1) | Le Wang, Zonghao Ying et al. | Embodied agents powered by vision-language models (VLMs) are increasingly capable of executing complex real-world tasks, yet they remain vulnerable to hazardous instructions that may trigger unsafe behaviors. Runtime safety guardrails, which intercept hazardous actions during task execution, offer a promising solution due to their flexibility. However, existing defenses often rely on static rule filters or prompt-level control, which struggle to address implicit risks arising in dynamic, temporally dependent, and context-rich environments. To address this, we propose RoboSafe, a hybrid reasoning runtime safeguard for embodied agents through executable predicate-based safety logic. RoboSafe integrates two complementary reasoning processes on a Hybrid Long-Short Safety Memory. We first propose a Backward Reflective Reasoning module that continuously revisits recent trajectories in short-term memory to infer temporal safety predicates and proactively triggers replanning when violations are detected. We then propose a Forward Predictive Reasoning module that anticipates upcoming risks by generating context-aware safety predicates from the long-term safety memory and the agent's multimodal observations. Together, these components form an adaptive, verifiable safety logic that is both interpretable and executable as code. Extensive experiments across multiple agents demonstrate that RoboSafe substantially reduces hazardous actions (-36.8% risk occurrence) compared with leading baselines, while maintaining near-original task performance. Real-world evaluations on physical robotic arms further confirm its practicality. Code will be released upon acceptance. |
| 2025-12-24 | [A Real-World Evaluation of LLM Medication Safety Reviews in NHS Primary Care](http://arxiv.org/abs/2512.21127v1) | Oliver Normand, Esther Borsi et al. | Large language models (LLMs) often match or exceed clinician-level performance on medical benchmarks, yet very few are evaluated on real clinical data or examined beyond headline metrics. We present, to our knowledge, the first evaluation of an LLM-based medication safety review system on real NHS primary care data, with detailed characterisation of key failure behaviours across varying levels of clinical complexity. In a retrospective study using a population-scale EHR spanning 2,125,549 adults in NHS Cheshire and Merseyside, we strategically sampled patients to capture a broad range of clinical complexity and medication safety risk, yielding 277 patients after data-quality exclusions. An expert clinician reviewed these patients and graded system-identified issues and proposed interventions. Our primary LLM system showed strong performance in recognising when a clinical issue is present (sensitivity 100\% [95\% CI 98.2--100], specificity 83.1\% [95\% CI 72.7--90.1]), yet correctly identified all issues and interventions in only 46.9\% [95\% CI 41.1--52.8] of patients. Failure analysis reveals that, in this setting, the dominant failure mechanism is contextual reasoning rather than missing medication knowledge, with five primary patterns: overconfidence in uncertainty, applying standard guidelines without adjusting for patient context, misunderstanding how healthcare is delivered in practice, factual errors, and process blindness. These patterns persisted across patient complexity and demographic strata, and across a range of state-of-the-art models and configurations. We provide 45 detailed vignettes that comprehensively cover all identified failure cases. This work highlights shortcomings that must be addressed before LLM-based clinical AI can be safely deployed. It also begs larger-scale, prospective evaluations and deeper study of LLM behaviours in clinical contexts. |
| 2025-12-24 | [Beyond Context: Large Language Models Failure to Grasp Users Intent](http://arxiv.org/abs/2512.21110v1) | Ahmed M. Hussain, Salahuddin Salahuddin et al. | Current Large Language Models (LLMs) safety approaches focus on explicitly harmful content while overlooking a critical vulnerability: the inability to understand context and recognize user intent. This creates exploitable vulnerabilities that malicious users can systematically leverage to circumvent safety mechanisms. We empirically evaluate multiple state-of-the-art LLMs, including ChatGPT, Claude, Gemini, and DeepSeek. Our analysis demonstrates the circumvention of reliable safety mechanisms through emotional framing, progressive revelation, and academic justification techniques. Notably, reasoning-enabled configurations amplified rather than mitigated the effectiveness of exploitation, increasing factual precision while failing to interrogate the underlying intent. The exception was Claude Opus 4.1, which prioritized intent detection over information provision in some use cases. This pattern reveals that current architectural designs create systematic vulnerabilities. These limitations require paradigmatic shifts toward contextual understanding and intent recognition as core safety capabilities rather than post-hoc protective mechanisms. |
| 2025-12-24 | [Semi-Supervised Learning for Large Language Models Safety and Content Moderation](http://arxiv.org/abs/2512.21107v1) | Eduard Stefan Dinuta, Iustin Sirbu et al. | Safety for Large Language Models (LLMs) has been an ongoing research focus since their emergence and is even more relevant nowadays with the increasing capacity of those models. Currently, there are several guardrails in place for all public LLMs and multiple proposed datasets for training safety classifiers. However, training these safety classifiers relies on large quantities of labeled data, which can be problematic to acquire, prone to labeling errors, or often include synthetic data. To address these issues, we suggest a different approach: utilizing semi-supervised learning techniques, which leverage both labeled and unlabeled data, to improve the performance on the safety task. We analyze the improvements that these techniques can offer for both prompts given to Large Language Models and the responses to those requests. Moreover, since augmentation is the central part of semi-supervised algorithms, we demonstrate the importance of using task-specific augmentations, which significantly increase the performance when compared to general-purpose augmentation techniques. |
| 2025-12-24 | [Co-Existence of Private 5G Network and Wireless Hospital Systems](http://arxiv.org/abs/2512.21079v1) | Mohsin Khan, Matti Hämäläinen et al. | This paper investigates the feasibility of deploying private 5G networks in hospital environments, with a focus on the operating room at the brand new Oulu University Hospital, Finland. The study aims to evaluate the interference risk with other wireless systems, and electromagnetic safety of a private 5G network in the 3.9-4.1 GHz band, while ensuring compatibility with legacy wireless systems, such as LTE and Wi-Fi. We conducted a measurement campaign, employing state-of-the-art instrumentation and a methodology that combined high resolution and long-duration spectrum scans. The results demonstrate no measurable interference between the hospital's private 5G network with adjacent LTE (4G) or Wi-Fi bands, confirming the spectral isolation of the 5G transmissions, and vise versa. Additionally, RF exposure levels in the operating room were found to be well below ICNIRP, WHO, and IEEE safety thresholds, ensuring that the network poses negligible biological risk to patients and hospital staff. The study also proposes spectrum management strategies for private 5G networks in hospitals, focusing on adaptive sensing and guardband planning. These findings provide a solid foundation for the integration of private 5G infrastructure in hospitals environments, supporting digital transformation in patient care without compromising electromagnetic compatibility or patient safety. The results also contribute to ongoing discussions around private 5G network deployments in sensitive sectors and provide actionable guidelines for future hospitals' wireless systems planning. |
| 2025-12-24 | [Agentic Multi-Persona Framework for Evidence-Aware Fake News Detection](http://arxiv.org/abs/2512.21039v1) | Roopa Bukke, Soumya Pandey et al. | The rapid proliferation of online misinformation poses significant risks to public trust, policy, and safety, necessitating reliable automated fake news detection. Existing methods often struggle with multimodal content, domain generalization, and explainability. We propose AMPEND-LS, an agentic multi-persona evidence-grounded framework with LLM-SLM synergy for multimodal fake news detection. AMPEND-LS integrates textual, visual, and contextual signals through a structured reasoning pipeline powered by LLMs, augmented with reverse image search, knowledge graph paths, and persuasion strategy analysis. To improve reliability, we introduce a credibility fusion mechanism combining semantic similarity, domain trustworthiness, and temporal context, and a complementary SLM classifier to mitigate LLM uncertainty and hallucinations. Extensive experiments across three benchmark datasets demonstrate that AMPEND-LS consistently outperformed state-of-the-art baselines in accuracy, F1 score, and robustness. Qualitative case studies further highlight its transparent reasoning and resilience against evolving misinformation. This work advances the development of adaptive, explainable, and evidence-aware systems for safeguarding online information integrity. |
| 2025-12-24 | [GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs](http://arxiv.org/abs/2512.21008v1) | Lichao Wu, Sasha Behrouzi et al. | Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs. |
| 2025-12-24 | [Automatic Replication of LLM Mistakes in Medical Conversations](http://arxiv.org/abs/2512.20983v1) | Oleksii Proniakin, Diego Fajardo et al. | Large language models (LLMs) are increasingly evaluated in clinical settings using multi-dimensional rubrics which quantify reasoning quality, safety, and patient-centeredness. Yet, replicating specific mistakes in other LLM models is not straightforward and often requires manual effort. We introduce MedMistake, an automatic pipeline that extracts mistakes LLMs make in patient-doctor conversations and converts them into a benchmark of single-shot QA pairs. Our pipeline (1) creates complex, conversational data between an LLM patient and LLM doctor, (2) runs an evaluation with a committee of 2 LLM judges across a variety of dimensions and (3) creates simplified single-shot QA scenarios from those mistakes. We release MedMistake-All, a dataset of 3,390 single-shot QA pairs where GPT-5 and Gemini 2.5 Pro are currently failing to answer correctly, as judged by two LLM judges. We used medical experts to validate a subset of 211/3390 questions (MedMistake-Bench), which we used to run a final evaluation of 12 frontier LLMs: Claude Opus 4.5, Claude Sonnet 4.5, DeepSeek-Chat, Gemini 2.5 Pro, Gemini 3 Pro, GPT-4o, GPT-5, GPT-5.1, GPT-5.2, Grok 4, Grok 4.1, Mistral Large. We found that GPT models, Claude and Grok obtained the best performance on MedMistake-Bench. We release both the doctor-validated benchmark (MedMistake-Bench), as well as the full dataset (MedMistake-All) at https://huggingface.co/datasets/TheLumos/MedicalMistakeBenchmark. |
| 2025-12-24 | [Time-Efficient Evaluation and Enhancement of Adversarial Robustness in Deep Neural Networks](http://arxiv.org/abs/2512.20893v1) | Runqi Lin | With deep neural networks (DNNs) increasingly embedded in modern society, ensuring their safety has become a critical and urgent issue. In response, substantial efforts have been dedicated to the red-blue adversarial framework, where the red team focuses on identifying vulnerabilities in DNNs and the blue team on mitigating them. However, existing approaches from both teams remain computationally intensive, constraining their applicability to large-scale models. To overcome this limitation, this thesis endeavours to provide time-efficient methods for the evaluation and enhancement of adversarial robustness in DNNs. |
| 2025-12-24 | [Early warning signals for loss of control](http://arxiv.org/abs/2512.20868v1) | Jasper J. van Beers, Marten Scheffer et al. | Maintaining stability in feedback systems, from aircraft and autonomous robots to biological and physiological systems, relies on monitoring their behavior and continuously adjusting their inputs. Incremental damage can make such control fragile. This tends to go unnoticed until a small perturbation induces instability (i.e. loss of control). Traditional methods in the field of engineering rely on accurate system models to compute a safe set of operating instructions, which become invalid when the, possibly damaged, system diverges from its model. Here we demonstrate that the approach of such a feedback system towards instability can nonetheless be monitored through dynamical indicators of resilience. This holistic system safety monitor does not rely on a system model and is based on the generic phenomenon of critical slowing down, shown to occur in the climate, biology and other complex nonlinear systems approaching criticality. Our findings for engineered devices opens up a wide range of applications involving real-time early warning systems as well as an empirical guidance of resilient system design exploration, or "tinkering". While we demonstrate the validity using drones, the generic nature of the underlying principles suggest that these indicators could apply across a wider class of controlled systems including reactors, aircraft, and self-driving cars. |
| 2025-12-24 | [Robustness Certificates for Neural Networks against Adversarial Attacks](http://arxiv.org/abs/2512.20865v1) | Sara Taheri, Mahalakshmi Sabanayagam et al. | The increasing use of machine learning in safety-critical domains amplifies the risk of adversarial threats, especially data poisoning attacks that corrupt training data to degrade performance or induce unsafe behavior. Most existing defenses lack formal guarantees or rely on restrictive assumptions about the model class, attack type, extent of poisoning, or point-wise certification, limiting their practical reliability. This paper introduces a principled formal robustness certification framework that models gradient-based training as a discrete-time dynamical system (dt-DS) and formulates poisoning robustness as a formal safety verification problem. By adapting the concept of barrier certificates (BCs) from control theory, we introduce sufficient conditions to certify a robust radius ensuring that the terminal model remains safe under worst-case ${\ell}_p$-norm based poisoning. To make this practical, we parameterize BCs as neural networks trained on finite sets of poisoned trajectories. We further derive probably approximately correct (PAC) bounds by solving a scenario convex program (SCP), which yields a confidence lower bound on the certified robustness radius generalizing beyond the training set. Importantly, our framework also extends to certification against test-time attacks, making it the first unified framework to provide formal guarantees in both training and test-time attack settings. Experiments on MNIST, SVHN, and CIFAR-10 show that our approach certifies non-trivial perturbation budgets while being model-agnostic and requiring no prior knowledge of the attack or contamination level. |
| 2025-12-23 | [MediEval: A Unified Medical Benchmark for Patient-Contextual and Knowledge-Grounded Reasoning in LLMs](http://arxiv.org/abs/2512.20822v1) | Zhan Qu, Michael Färber | Large Language Models (LLMs) are increasingly applied to medicine, yet their adoption is limited by concerns over reliability and safety. Existing evaluations either test factual medical knowledge in isolation or assess patient-level reasoning without verifying correctness, leaving a critical gap. We introduce MediEval, a benchmark that links MIMIC-IV electronic health records (EHRs) to a unified knowledge base built from UMLS and other biomedical vocabularies. MediEval generates diverse factual and counterfactual medical statements within real patient contexts, enabling systematic evaluation across a 4-quadrant framework that jointly considers knowledge grounding and contextual consistency. Using this framework, we identify critical failure modes, including hallucinated support and truth inversion, that current proprietary, open-source, and domain-specific LLMs frequently exhibit. To address these risks, we propose Counterfactual Risk-Aware Fine-tuning (CoRFu), a DPO-based method with an asymmetric penalty targeting unsafe confusions. CoRFu improves by +16.4 macro-F1 points over the base model and eliminates truth inversion errors, demonstrating both higher accuracy and substantially greater safety. |
| 2025-12-23 | [Safety Alignment of LMs via Non-cooperative Games](http://arxiv.org/abs/2512.20806v1) | Anselm Paulus, Ilia Kulikov et al. | Ensuring the safety of language models (LMs) while maintaining their usefulness remains a critical challenge in AI alignment. Current approaches rely on sequential adversarial training: generating adversarial prompts and fine-tuning LMs to defend against them. We introduce a different paradigm: framing safety alignment as a non-zero-sum game between an Attacker LM and a Defender LM trained jointly via online reinforcement learning. Each LM continuously adapts to the other's evolving strategies, driving iterative improvement. Our method uses a preference-based reward signal derived from pairwise comparisons instead of point-wise scores, providing more robust supervision and potentially reducing reward hacking. Our RL recipe, AdvGame, shifts the Pareto frontier of safety and utility, yielding a Defender LM that is simultaneously more helpful and more resilient to adversarial attacks. In addition, the resulting Attacker LM converges into a strong, general-purpose red-teaming agent that can be directly deployed to probe arbitrary target models. |
| 2025-12-23 | [A Benchmark for Evaluating Outcome-Driven Constraint Violations in Autonomous AI Agents](http://arxiv.org/abs/2512.20798v1) | Miles Q. Li, Benjamin C. M. Fung et al. | As autonomous AI agents are increasingly deployed in high-stakes environments, ensuring their safety and alignment with human values has become a paramount concern. Current safety benchmarks often focusing only on single-step decision-making, simulated environments for tasks with malicious intent, or evaluating adherence to explicit negative constraints. There is a lack of benchmarks that are designed to capture emergent forms of outcome-driven constraint violations, which arise when agents pursue goal optimization under strong performance incentives while deprioritizing ethical, legal, or safety constraints over multiple steps in realistic production settings. To address this gap, we introduce a new benchmark comprising 40 distinct scenarios. Each scenario presents a task that requires multi-step actions, and the agent's performance is tied to a specific Key Performance Indicator (KPI). Each scenario features Mandated (instruction-commanded) and Incentivized (KPI-pressure-driven) variations to distinguish between obedience and emergent misalignment. Across 12 state-of-the-art large language models, we observe outcome-driven constraint violations ranging from 1.3% to 71.4%, with 9 of the 12 evaluated models exhibiting misalignment rates between 30% and 50%. Strikingly, we find that superior reasoning capability does not inherently ensure safety; for instance, Gemini-3-Pro-Preview, one of the most capable models evaluated, exhibits the highest violation rate at over 60%, frequently escalating to severe misconduct to satisfy KPIs. Furthermore, we observe significant "deliberative misalignment", where the models that power the agents recognize their actions as unethical during separate evaluation. These results emphasize the critical need for more realistic agentic-safety training before deployment to mitigate their risks in the real world. |
| 2025-12-23 | [Bridging Efficiency and Safety: Formal Verification of Neural Networks with Early Exits](http://arxiv.org/abs/2512.20755v1) | Yizhak Yisrael Elboher, Avraham Raviv et al. | Ensuring the safety and efficiency of AI systems is a central goal of modern research. Formal verification provides guarantees of neural network robustness, while early exits improve inference efficiency by enabling intermediate predictions. Yet verifying networks with early exits introduces new challenges due to their conditional execution paths. In this work, we define a robustness property tailored to early exit architectures and show how off-the-shelf solvers can be used to assess it. We present a baseline algorithm, enhanced with an early stopping strategy and heuristic optimizations that maintain soundness and completeness. Experiments on multiple benchmarks validate our framework's effectiveness and demonstrate the performance gains of the improved algorithm. Alongside the natural inference acceleration provided by early exits, we show that they also enhance verifiability, enabling more queries to be solved in less time compared to standard networks. Together with a robustness analysis, we show how these metrics can help users navigate the inherent trade-off between accuracy and efficiency. |
| 2025-12-23 | [Fixed-time control with prescribed performance for path following of underwater gliders](http://arxiv.org/abs/2512.20748v1) | Hanzhi Yang, Nina Mahmoudian | Underwater gliders are increasingly deployed in challenging missions - such as hurricane-season observations and long-endurance environmental monitoring - where strong currents and turbulence pose significant risks to navigation safety. To address these practical challenges, this paper presents a fixed-time prescribed performance control scheme for the 3D path following of underwater gliders subject to model uncertainties and environmental disturbances. The primary contribution is the integration of a finite-time performance function within a fixed-time control framework. This synthesis ensures that the tracking errors are constrained within prescribed performance bounds and converge to a compact set within a fixed time, independent of initial conditions. A second key contribution is the development of a fixed-time sliding mode disturbance observer that provides accurate finite-time estimation of lumped disturbances, enhancing the system's robustness. Integrated with an iLOS guidance law, the proposed controller enables precise and safe waypoint following. Numerical simulations demonstrate that the proposed method outperforms conventional sliding mode and prescribed performance controllers in tracking accuracy, convergence speed, and control effort smoothness, validating its efficacy for robust underwater navigation. |
| 2025-12-23 | [AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](http://arxiv.org/abs/2512.20745v1) | Haipeng Luo, Huawen Feng et al. | Large Reasoning Models (LRMs) like o3 and DeepSeek-R1 have achieved remarkable progress in natural language reasoning with long chain-of-thought. However, they remain computationally inefficient and struggle with accuracy when solving problems requiring complex mathematical operations. In this work, we present AgentMath, an agent framework that seamlessly integrates language models' reasoning capabilities with code interpreters' computational precision to efficiently tackle complex mathematical problems. Our approach introduces three key innovations: (1) An automated method that converts natural language chain-of-thought into structured tool-augmented trajectories, generating high-quality supervised fine-tuning (SFT) data to alleviate data scarcity; (2) A novel agentic reinforcement learning (RL) paradigm that dynamically interleaves natural language generation with real-time code execution. This enables models to autonomously learn optimal tool-use strategies through multi-round interactive feedback, while fostering emergent capabilities in code refinement and error correction; (3) An efficient training system incorporating innovative techniques, including request-level asynchronous rollout scheduling, agentic partial rollout, and prefix-aware weighted load balancing, achieving 4-5x speedup and making efficient RL training feasible on ultra-long sequences with scenarios with massive tool calls.Extensive evaluations show that AgentMath achieves state-of-the-art performance on challenging mathematical competition benchmarks including AIME24, AIME25, and HMMT25. Specifically, AgentMath-30B-A3B attains 90.6%, 86.4%, and 73.8% accuracy respectively, achieving advanced capabilities.These results validate the effectiveness of our approach and pave the way for building more sophisticated and scalable mathematical reasoning agents. |
| 2025-12-23 | [SoK: Speedy Secure Finality](http://arxiv.org/abs/2512.20715v1) | Yash Saraswat, Abhimanyu Nag | While Ethereum has successfully achieved dynamic availability together with safety, a fundamental delay remains between transaction execution and immutable finality. In Ethereum's current Gasper protocol, this latency is on the order of 15 minutes, exposing the network to ex ante reorganization attacks, enabling MEV extraction, and limiting the efficiency of economic settlement. These limitations have motivated a growing body of work on Speedy Secure Finality (SSF), which aims to minimize confirmation latency without weakening formal security guarantees.   This paper surveys the state of the art in fast finality protocol design. We introduce the core theoretical primitives underlying this space, including reorganization resilience and the generalized sleepy model, and trace their development from Goldfish to RLMD-GHOST. We then analyze the communication and aggregation bottlenecks faced by single-slot finality protocols in large validator settings. Finally, we survey the 3-slot finality (3SF) protocol as a practical synthesis that balances fast finality with the engineering constraints of the Ethereum network. |

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



