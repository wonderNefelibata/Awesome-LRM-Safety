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
| 2025-06-11 | [From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring](http://arxiv.org/abs/2506.09996v1) | Yang Li, Qiang Sheng et al. | Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO. |
| 2025-06-11 | [Assessing a Safety Case: Bottom-up Guidance for Claims and Evidence Evaluation](http://arxiv.org/abs/2506.09929v1) | Scott Schnelle, Francesca Favaro et al. | As Automated Driving Systems (ADS) technology advances, ensuring safety and public trust requires robust assurance frameworks, with safety cases emerging as a critical tool toward such a goal. This paper explores an approach to assess how a safety case is supported by its claims and evidence, toward establishing credibility for the overall case. Starting from a description of the building blocks of a safety case (claims, evidence, and optional format-dependent entries), this paper delves into the assessment of support of each claim through the provided evidence. Two domains of assessment are outlined for each claim: procedural support (formalizing process specification) and implementation support (demonstrating process application). Additionally, an assessment of evidence status is also undertaken, independently from the claims support. Scoring strategies and evaluation guidelines are provided, including detailed scoring tables for claim support and evidence status assessment. The paper further discusses governance, continual improvement, and timing considerations for safety case assessments. Reporting of results and findings is contextualized within its primary use for internal decision-making on continual improvement efforts. The presented approach builds on state of the art auditing practices, but specifically tackles the question of judging the credibility of a safety case. While not conclusive on its own, it provides a starting point toward a comprehensive "Case Credibility Assessment" (CCA), starting from the evaluation of the support for each claim (individually and in aggregate), as well as every piece of evidence provided. By delving into the technical intricacies of ADS safety cases, this work contributes to the ongoing discourse on safety assurance and aims to facilitate the responsible integration of ADS technology into society. |
| 2025-06-11 | [OctoNav: Towards Generalist Embodied Navigation](http://arxiv.org/abs/2506.09839v1) | Chen Gao, Liankai Jin et al. | Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. |
| 2025-06-11 | [Superstudent intelligence in thermodynamics](http://arxiv.org/abs/2506.09822v1) | Rebecca Loubet, Pascal Zittlau et al. | In this short note, we report and analyze a striking event: OpenAI's large language model o3 has outwitted all students in a university exam on thermodynamics. The thermodynamics exam is a difficult hurdle for most students, where they must show that they have mastered the fundamentals of this important topic. Consequently, the failure rates are very high, A-grades are rare - and they are considered proof of the students' exceptional intellectual abilities. This is because pattern learning does not help in the exam. The problems can only be solved by knowledgeably and creatively combining principles of thermodynamics. We have given our latest thermodynamics exam not only to the students but also to OpenAI's most powerful reasoning model, o3, and have assessed the answers of o3 exactly the same way as those of the students. In zero-shot mode, the model o3 solved all problems correctly, better than all students who took the exam; its overall score was in the range of the best scores we have seen in more than 10,000 similar exams since 1985. This is a turning point: machines now excel in complex tasks, usually taken as proof of human intellectual capabilities. We discuss the consequences this has for the work of engineers and the education of future engineers. |
| 2025-06-11 | [CoRT: Code-integrated Reasoning within Thinking](http://arxiv.org/abs/2506.09820v1) | Chengpeng Li, Zhengyang Tang et al. | Large Reasoning Models (LRMs) like o1 and DeepSeek-R1 have shown remarkable progress in natural language reasoning with long chain-of-thought (CoT), yet they remain inefficient or inaccurate when handling complex mathematical operations. Addressing these limitations through computational tools (e.g., computation libraries and symbolic solvers) is promising, but it introduces a technical challenge: Code Interpreter (CI) brings external knowledge beyond the model's internal text representations, thus the direct combination is not efficient. This paper introduces CoRT, a post-training framework for teaching LRMs to leverage CI effectively and efficiently. As a first step, we address the data scarcity issue by synthesizing code-integrated reasoning data through Hint-Engineering, which strategically inserts different hints at appropriate positions to optimize LRM-CI interaction. We manually create 30 high-quality samples, upon which we post-train models ranging from 1.5B to 32B parameters, with supervised fine-tuning, rejection fine-tuning and reinforcement learning. Our experimental results demonstrate that Hint-Engineering models achieve 4\% and 8\% absolute improvements on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B respectively, across five challenging mathematical reasoning datasets. Furthermore, Hint-Engineering models use about 30\% fewer tokens for the 32B model and 50\% fewer tokens for the 1.5B model compared with the natural language models. The models and code are available at https://github.com/ChengpengLi1003/CoRT. |
| 2025-06-11 | [Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving](http://arxiv.org/abs/2506.09800v1) | Haochen Liu, Tianyu Li et al. | End-to-end autonomous driving has emerged as a promising paradigm for directly mapping sensor inputs to planning maneuvers using learning-based modular integrations. However, existing imitation learning (IL)-based models suffer from generalization to hard cases, and a lack of corrective feedback loop under post-deployment. While reinforcement learning (RL) offers a potential solution to tackle hard cases with optimality, it is often hindered by overfitting to specific driving cases, resulting in catastrophic forgetting of generalizable knowledge and sample inefficiency. To overcome these challenges, we propose Reinforced Refinement with Self-aware Expansion (R2SE), a novel learning pipeline that constantly refines hard domain while keeping generalizable driving policy for model-agnostic end-to-end driving systems. Through reinforcement fine-tuning and policy expansion that facilitates continuous improvement, R2SE features three key components: 1) Generalist Pretraining with hard-case allocation trains a generalist imitation learning (IL) driving system while dynamically identifying failure-prone cases for targeted refinement; 2) Residual Reinforced Specialist Fine-tuning optimizes residual corrections using reinforcement learning (RL) to improve performance in hard case domain while preserving global driving knowledge; 3) Self-aware Adapter Expansion dynamically integrates specialist policies back into the generalist model, enhancing continuous performance improvement. Experimental results in closed-loop simulation and real-world datasets demonstrate improvements in generalization, safety, and long-horizon policy robustness over state-of-the-art E2E systems, highlighting the effectiveness of reinforce refinement for scalable autonomous driving. |
| 2025-06-11 | [Translating a VDM Model of a Medical Device into Kapture](http://arxiv.org/abs/2506.09636v1) | Joe Hare, Leo Freitas et al. | As the complexity of safety-critical medical devices increases, so does the need for clear, verifiable, software requirements. This paper explores the use of Kapture, a formal modelling tool developed by D-RisQ, to translate an existing formal VDM model of a medical implant for treating focal epilepsy called CANDO. The work was undertaken without prior experience in formal methods. The paper assess Kapture's usability, the challenges of formal modelling, and the effectiveness of the translated model. The result is a model in Kapture which covers over 90% of the original VDM model, and produces matching traces of results. While several issues were encountered during design and implementation, mainly due to the initial learning curve, this paper demonstrates that complex systems can be effectively modelled in Kapture by inexperienced users and highlights some difficulties in translating VDM specifications to Kapture. |
| 2025-06-11 | [Effective Red-Teaming of Policy-Adherent Agents](http://arxiv.org/abs/2506.09600v1) | Itay Nakash, George Kour et al. | Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks |
| 2025-06-11 | [Understanding the Performance and Power of LLM Inferencing on Edge Accelerators](http://arxiv.org/abs/2506.09554v1) | Mayank Arya, Yogesh Simmhan | Large Language Models (LLMs) have demonstrated exceptional benefits to a wide range of domains, for tasks as diverse as code generation and robot navigation. While LLMs are usually served from cloud data centers, mission-critical and privacy-sensitive applications may require local hosting of open LLM models. Given the large GPU memory footprint needed for LLMs, edge accelerators such as Nvidia Jetson Orin AGX with 64GB of shared GPU-CPU RAM are a compelling choice. However, the feasibility and performance of LLM inference on edge accelerators is under-explored. This study presents a detailed evaluation of LLM inference on the NVIDIA Jetson Orin AGX, on four SOTA models ranging from 2.7B to 32.8B parameters, such as Meta Llama3.1, Microsoft-Phi2, Deepseek-R1-Qwen.We investigate the impact of varying batch sizes, sequence lengths, and quantization levels on latency, throughput, and perplexity, and also explore various custom power modes on the Orin AGX to perform power and energy consumption analysis. Our findings offer interesting insights on the trade-offs between efficiency, inference speed and resource use, e.g., increasing the sequence length causes a decrease in token throughput and quantization causes smaller LLMs to be slower. These results can help optimize LLM serving on edge accelerators for practical applications. |
| 2025-06-11 | [Enhancing Human-Robot Collaboration: A Sim2Real Domain Adaptation Algorithm for Point Cloud Segmentation in Industrial Environments](http://arxiv.org/abs/2506.09552v1) | Fatemeh Mohammadi Amin, Darwin G. Caldwell et al. | The robust interpretation of 3D environments is crucial for human-robot collaboration (HRC) applications, where safety and operational efficiency are paramount. Semantic segmentation plays a key role in this context by enabling a precise and detailed understanding of the environment. Considering the intense data hunger for real-world industrial annotated data essential for effective semantic segmentation, this paper introduces a pioneering approach in the Sim2Real domain adaptation for semantic segmentation of 3D point cloud data, specifically tailored for HRC. Our focus is on developing a network that robustly transitions from simulated environments to real-world applications, thereby enhancing its practical utility and impact on a safe HRC.   In this work, we propose a dual-stream network architecture (FUSION) combining Dynamic Graph Convolutional Neural Networks (DGCNN) and Convolutional Neural Networks (CNN) augmented with residual layers as a Sim2Real domain adaptation algorithm for an industrial environment. The proposed model was evaluated on real-world HRC setups and simulation industrial point clouds, it showed increased state-of-the-art performance, achieving a segmentation accuracy of 97.76%, and superior robustness compared to existing methods. |
| 2025-06-11 | [Automated Synthesis of Formally Verified Multi-Abstraction Function Summaries](http://arxiv.org/abs/2506.09550v1) | Fanpeng Yang, Xu Ma et al. | Function summaries, which characterize the behavior of code segments (typically functions) through preconditions and postconditions, are essential for understanding, reusing, and verifying software, particularly in safety-critical domains like aerospace embedded systems. However, these mission-critical legacy code serving as a valuable reused asset often lacks formal specifications. It is challenging to automatically generate function summaries for C programs, due to the existence of complex features such as loops, nested function calls, pointer aliasing, and so on. Moreover, function summaries should support multiple abstraction levels to meet diverse requirements, e.g. precise summaries capturing full functionality for formal verification and intuitive summaries for human understanding.   To address these challenges, we first propose a novel framework that combines symbolic execution, large language models (LLMs), and formal verification to generate Relatively Strongest Postconditions (RSPs) and build function summaries that fully capture program behavior. Our approach leverages VST-A's symbolic execution to precisely track program execution paths and state transitions, employs LLMs to infer loop invariants based on predefined templates, and uses Frama-C to guarantee soundness of generated summaries in an iterative refinement loop. Furthermore, from generated RSPs, we automatically synthesize strongest non-redundant postconditions expressed within given domain specific language. We compare our approach with existing work through extensive experiments. |
| 2025-06-11 | [Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning](http://arxiv.org/abs/2506.09501v1) | Jiayi Yuan, Hao Li et al. | Large Language Models (LLMs) are now integral across various domains and have demonstrated impressive performance. Progress, however, rests on the premise that benchmark scores are both accurate and reproducible. We demonstrate that the reproducibility of LLM performance is fragile: changing system configuration such as evaluation batch size, GPU count, and GPU version can introduce significant difference in the generated responses. This issue is especially pronounced in reasoning models, where minor rounding differences in early tokens can cascade into divergent chains of thought, ultimately affecting accuracy. For instance, under bfloat16 precision with greedy decoding, a reasoning model like DeepSeek-R1-Distill-Qwen-7B can exhibit up to 9% variation in accuracy and 9,000 tokens difference in response length due to differences in GPU count, type, and evaluation batch size. We trace the root cause of this variability to the non-associative nature of floating-point arithmetic under limited numerical precision. This work presents the first systematic investigation into how numerical precision affects reproducibility in LLM inference. Through carefully controlled experiments across various hardware, software, and precision settings, we quantify when and how model outputs diverge. Our analysis reveals that floating-point precision -- while critical for reproducibility -- is often neglected in evaluation practices. Inspired by this, we develop a lightweight inference pipeline, dubbed LayerCast, that stores weights in 16-bit precision but performs all computations in FP32, balancing memory efficiency with numerical stability. Code is available at https://github.com/nanomaoli/llm_reproducibility. |
| 2025-06-11 | [Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation](http://arxiv.org/abs/2506.09485v1) | Yuxin Liu, Zhenghao Peng et al. | Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work. |
| 2025-06-11 | [Optimization and Control Technologies for Renewable-Dominated Hydrogen-Blended Integrated Gas-Electricity System: A Review](http://arxiv.org/abs/2506.09447v1) | Wenxin Liu, Jiakun Fang et al. | The growing coupling among electricity, gas, and hydrogen systems is driven by green hydrogen blending into existing natural gas pipelines, paving the way toward a renewable-dominated energy future. However, the integration poses significant challenges, particularly ensuring efficient and safe operation under varying hydrogen penetration and infrastructure adaptability. This paper reviews progress in optimization and control technologies for hydrogen-blended integrated gas-electricity system. First, key technologies and international demonstration projects are introduced to provide an overview of current developments. Besides, advances in gas-electricity system integration, including modeling, scheduling, planning and market design, are reviewed respectively. Then, the potential for cross-system fault propagation is highlighted, and practical methods for safety analysis and control are proposed. Finally, several possible research directions are introduced, aiming to ensure efficient renewable integration and reliable operation. |
| 2025-06-11 | [Securing Open RAN: A Survey of Cryptographic Challenges and Emerging Solutions for 5G](http://arxiv.org/abs/2506.09418v1) | Ryan Barker, Fatemeh Afghah | The advent of Open Radio Access Networks (O-RAN) introduces modularity and flexibility into 5G deployments but also surfaces novel security challenges across disaggregated interfaces. This literature review synthesizes recent research across thirteen academic and industry sources, examining vulnerabilities such as cipher bidding-down attacks, partial encryption exposure on control/user planes, and performance trade-offs in securing O-RAN interfaces like E2 and O1. The paper surveys key cryptographic tools -- SNOW-V, AES-256, and ZUC-256 -- evaluating their throughput, side-channel resilience, and adaptability to heterogeneous slices (eMBB, URLLC, mMTC). Emphasis is placed on emerging testbeds and AI-driven controllers that facilitate dynamic orchestration, anomaly detection, and secure configuration. We conclude by outlining future research directions, including hardware offloading, cross-layer cipher adaptation, and alignment with 3GPP TS 33.501 and O-RAN Alliance security mandates, all of which point toward the need for integrated, zero-trust architectures in 6G. |
| 2025-06-11 | [Token Constraint Decoding Improves Robustness on Question Answering for Large Language Models](http://arxiv.org/abs/2506.09408v1) | Jui-Ming Yao, Hao-Yuan Chen et al. | Large Language Models (LLMs) have demonstrated impressive performance on multiple-choice question answering (MCQA) benchmarks, yet they remain highly vulnerable to minor input perturbations. In this paper, we introduce and evaluate Token Constraint Decoding (TCD). This simple yet effective inference-time algorithm enforces alignment between token-level predictions to enhance robustness in noisy settings. Through extensive experiments on CommonsenseQA, MMLU, and MMLU-Pro, we show that TCD, especially when paired with prompt engineering (PE) fixes, significantly restores performance degraded by input noise, yielding up to +39\% absolute gains for weaker models like Gemma3 1B. Penalty sweep analyses further reveal that TCD implicitly regularizes overconfident outputs, with different models requiring distinct penalty schedules to maximize resilience. Our findings establish TCD as a practical, model-agnostic approach for improving reasoning stability under real-world imperfections and pave the way for more reliable deployment of LLMs in safety-critical or user-facing applications. |
| 2025-06-11 | [SAGE: Exploring the Boundaries of Unsafe Concept Domain with Semantic-Augment Erasing](http://arxiv.org/abs/2506.09363v1) | Hongguang Zhu, Yunchao Wei et al. | Diffusion models (DMs) have achieved significant progress in text-to-image generation. However, the inevitable inclusion of sensitive information during pre-training poses safety risks, such as unsafe content generation and copyright infringement. Concept erasing finetunes weights to unlearn undesirable concepts, and has emerged as a promising solution. However, existing methods treat unsafe concept as a fixed word and repeatedly erase it, trapping DMs in ``word concept abyss'', which prevents generalized concept-related erasing. To escape this abyss, we introduce semantic-augment erasing which transforms concept word erasure into concept domain erasure by the cyclic self-check and self-erasure. It efficiently explores and unlearns the boundary representation of concept domain through semantic spatial relationships between original and training DMs, without requiring additional preprocessed data. Meanwhile, to mitigate the retention degradation of irrelevant concepts while erasing unsafe concepts, we further propose the global-local collaborative retention mechanism that combines global semantic relationship alignment with local predicted noise preservation, effectively expanding the retentive receptive field for irrelevant concepts. We name our method SAGE, and extensive experiments demonstrate the comprehensive superiority of SAGE compared with other methods in the safe generation of DMs. The code and weights will be open-sourced at https://github.com/KevinLight831/SAGE. |
| 2025-06-11 | [DAVSP: Safety Alignment for Large Vision-Language Models via Deep Aligned Visual Safety Prompt](http://arxiv.org/abs/2506.09353v1) | Yitong Zhang, Jia Li et al. | Large Vision-Language Models (LVLMs) have achieved impressive progress across various applications but remain vulnerable to malicious queries that exploit the visual modality. Existing alignment approaches typically fail to resist malicious queries while preserving utility on benign ones effectively. To address these challenges, we propose Deep Aligned Visual Safety Prompt (DAVSP), which is built upon two key innovations. First, we introduce the Visual Safety Prompt, which appends a trainable padding region around the input image. It preserves visual features and expands the optimization space. Second, we propose Deep Alignment, a novel approach to train the visual safety prompt through supervision in the model's activation space. It enhances the inherent ability of LVLMs to perceive malicious queries, achieving deeper alignment than prior works. Extensive experiments across five benchmarks on two representative LVLMs demonstrate that DAVSP effectively resists malicious queries while preserving benign input utility. Furthermore, DAVSP exhibits great cross-model generation ability. Ablation studies further reveal that both the Visual Safety Prompt and Deep Alignment are essential components, jointly contributing to its overall effectiveness. The code is publicly available at https://github.com/zhangyitonggg/DAVSP. |
| 2025-06-11 | ["Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions](http://arxiv.org/abs/2506.09354v1) | Kellie Yu Hui Sim, Roy Ka-Wei Lee et al. | Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care. |
| 2025-06-10 | [Efficient Edge Deployment of Quantized YOLOv4-Tiny for Aerial Emergency Object Detection on Raspberry Pi 5](http://arxiv.org/abs/2506.09300v1) | Sindhu Boddu, Arindam Mukherjee | This paper presents the deployment and performance evaluation of a quantized YOLOv4-Tiny model for real-time object detection in aerial emergency imagery on a resource-constrained edge device the Raspberry Pi 5. The YOLOv4-Tiny model was quantized to INT8 precision using TensorFlow Lite post-training quantization techniques and evaluated for detection speed, power consumption, and thermal feasibility under embedded deployment conditions. The quantized model achieved an inference time of 28.2 ms per image with an average power consumption of 13.85 W, demonstrating a significant reduction in power usage compared to its FP32 counterpart. Detection accuracy remained robust across key emergency classes such as Ambulance, Police, Fire Engine, and Car Crash. These results highlight the potential of low-power embedded AI systems for real-time deployment in safety-critical emergency response applications. |

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



