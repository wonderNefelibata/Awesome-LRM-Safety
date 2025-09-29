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
| 2025-09-26 | [VoiceAssistant-Eval: Benchmarking AI Assistants across Listening, Speaking, and Viewing](http://arxiv.org/abs/2509.22651v1) | Ke Wang, Houxing Ren et al. | The growing capabilities of large language models and multimodal systems have spurred interest in voice-first AI assistants, yet existing benchmarks are inadequate for evaluating the full range of these systems' capabilities. We introduce VoiceAssistant-Eval, a comprehensive benchmark designed to assess AI assistants across listening, speaking, and viewing. VoiceAssistant-Eval comprises 10,497 curated examples spanning 13 task categories. These tasks include natural sounds, music, and spoken dialogue for listening; multi-turn dialogue, role-play imitation, and various scenarios for speaking; and highly heterogeneous images for viewing. To demonstrate its utility, we evaluate 21 open-source models and GPT-4o-Audio, measuring the quality of the response content and speech, as well as their consistency. The results reveal three key findings: (1) proprietary models do not universally outperform open-source models; (2) most models excel at speaking tasks but lag in audio understanding; and (3) well-designed smaller models can rival much larger ones. Notably, the mid-sized Step-Audio-2-mini (7B) achieves more than double the listening accuracy of LLaMA-Omni2-32B-Bilingual. However, challenges remain: multimodal (audio plus visual) input and role-play voice imitation tasks are difficult for current models, and significant gaps persist in robustness and safety alignment. VoiceAssistant-Eval identifies these gaps and establishes a rigorous framework for evaluating and guiding the development of next-generation AI assistants. Code and data will be released at https://mathllm.github.io/VoiceAssistantEval/ . |
| 2025-09-26 | [Likelihood-free inference for gravitational-wave data analysis and public alerts](http://arxiv.org/abs/2509.22561v1) | Ethan Marx, Deep Chatterjee et al. | Rapid and reliable detection and dissemination of source parameter estimation data products from gravitational-wave events, especially sky localization, is critical for maximizing the potential of multi-messenger astronomy. Machine learning based detection and parameter estimation algorithms are emerging as production ready alternatives to traditional approaches. Here, we report validation studies of AMPLFI, a likelihood-free inference solution to low-latency parameter estimation of binary black holes. We use simulated signals added into data from the LIGO-Virgo-KAGRA's (LVK's) third observing run (O3) to compare sky localization performance with BAYESTAR, the algorithm currently in production for rapid sky localization of candidates from matched-filter pipelines. We demonstrate sky localization performance, measured by searched area and volume, to be equivalent with BAYESTAR. We show accurate reconstruction of source parameters with uncertainties for use distributing low-latency coarse-grained chirp mass information. In addition, we analyze several candidate events reported by the LVK in the third gravitational-wave transient catalog (GWTC-3) and show consistency with the LVK's analysis. Altogether, we demonstrate AMPLFI's ability to produce data products for low-latency public alerts. |
| 2025-09-26 | [OFMU: Optimization-Driven Framework for Machine Unlearning](http://arxiv.org/abs/2509.22483v1) | Sadia Asif, Mohammad Mohammadi Amiri | Large language models deployed in sensitive applications increasingly require the ability to unlearn specific knowledge, such as user requests, copyrighted materials, or outdated information, without retraining from scratch to ensure regulatory compliance, user privacy, and safety. This task, known as machine unlearning, aims to remove the influence of targeted data (forgetting) while maintaining performance on the remaining data (retention). A common approach is to formulate this as a multi-objective problem and reduce it to a single-objective problem via scalarization, where forgetting and retention losses are combined using a weighted sum. However, this often results in unstable training dynamics and degraded model utility due to conflicting gradient directions. To address these challenges, we propose OFMU, a penalty-based bi-level optimization framework that explicitly prioritizes forgetting while preserving retention through a hierarchical structure. Our method enforces forgetting via an inner maximization step that incorporates a similarity-aware penalty to decorrelate the gradients of the forget and retention objectives, and restores utility through an outer minimization step. To ensure scalability, we develop a two-loop algorithm with provable convergence guarantees under both convex and non-convex regimes. We further provide a rigorous theoretical analysis of convergence rates and show that our approach achieves better trade-offs between forgetting efficacy and model utility compared to prior methods. Extensive experiments across vision and language benchmarks demonstrate that OFMU consistently outperforms existing unlearning methods in both forgetting efficacy and retained utility. |
| 2025-09-26 | [Safe-by-Design: Approximate Nonlinear Model Predictive Control with Real Time Feasibility](http://arxiv.org/abs/2509.22422v1) | Jan Olucak, Arthur Castello B. de Oliveira et al. | This paper establishes relationships between continuous-time, receding horizon, nonlinear model predictive control (MPC) and control Lyapunov and control barrier functions (CLF/CBF). We show that, if the cost function "behaves well" for points in the terminal set, then the optimal value function and the feasible set, respectively, define a compatible CLF/CBF pair on the MPC's region of attraction. We then proceed to prove that any approximation of the value function and the feasible set also define a CLF/CBF pair, as long as those approximations satisfy the same "well behavedness" condition; and that a feasible state feedback can be computed by solving an infinitesimal version of the MPC problem. This methodology permits the formulation of continuous-time small-sized quadratic programs for feedback and enables approximate solutions of the nonlinear model predictive controller with theoretical safety and convergence guarantee. Finally, we demonstrate the effectiveness of the proposed approach when compared to other constrained control techniques through numerical experiments for nonlinear constrained spacecraft control. |
| 2025-09-26 | [X-ray and neural network based in-situ identification of the melt pool during the additive manufacturing of a stainless steel part](http://arxiv.org/abs/2509.22409v1) | Loic Jegou, Valerie Kaftandjian et al. | Laser Metal Deposition with Powder (LMDp) is an additive manufacturing technique used for repairing metal components or producing parts with intricate geometries. However, a comprehensive understanding of the melt pool dynamics, which significantly influences the final properties of LMDp-fabricated parts, remains limited. Non-destructive testing is highly valuable for conducting in-situ controls during manufacturing. X-ray imaging offers the ability to penetrate metallic parts and detect defects such as porosity. In the context of additive manufacturing, X-rays can be employed to visualize the shape of the melt pool during the fabrication process. The contrast between the liquid and solid phases, due to their density differences, should be observable in the radioscopy images. The experimental setup required to perform such a test on an industrial additive manufacturing installation consists of a movable X-ray source that produces polychromatic beams, a detector, and extensive lead shielding to ensure X-ray safety. In-situ observations of the melt pool were conducted during the deposition of ten successive layers of stainless steel 316L (SS316L). The polychromatic nature of the X-ray beam, however, rendered traditional image analysis methods ineffective for detecting contrast variations. To address this challenge, neural networks trained on simulated data (thermal and X-ray) were employed, providing a solution to identify the melt pool in low-contrast radioscopic images. The architecture inspired by VGG16 demonstrated promising results, confirming the potential for in-situ non-destructive testing using X-ray imaging in industrial additive manufacturing processes. |
| 2025-09-26 | [Closing the Safety Gap: Surgical Concept Erasure in Visual Autoregressive Models](http://arxiv.org/abs/2509.22400v1) | Xinhao Zhong, Yimin Zhou et al. | The rapid progress of visual autoregressive (VAR) models has brought new opportunities for text-to-image generation, but also heightened safety concerns. Existing concept erasure techniques, primarily designed for diffusion models, fail to generalize to VARs due to their next-scale token prediction paradigm. In this paper, we first propose a novel VAR Erasure framework VARE that enables stable concept erasure in VAR models by leveraging auxiliary visual tokens to reduce fine-tuning intensity. Building upon this, we introduce S-VARE, a novel and effective concept erasure method designed for VAR, which incorporates a filtered cross entropy loss to precisely identify and minimally adjust unsafe visual tokens, along with a preservation loss to maintain semantic fidelity, addressing the issues such as language drift and reduced diversity introduce by na\"ive fine-tuning. Extensive experiments demonstrate that our approach achieves surgical concept erasure while preserving generation quality, thereby closing the safety gap in autoregressive text-to-image generation by earlier methods. |
| 2025-09-26 | [A Multi-Modality Evaluation of the Reality Gap in Autonomous Driving Systems](http://arxiv.org/abs/2509.22379v1) | Stefano Carlo Lambertenghi, Mirena Flores Valdez et al. | Simulation-based testing is a cornerstone of Autonomous Driving System (ADS) development, offering safe and scalable evaluation across diverse driving scenarios. However, discrepancies between simulated and real-world behavior, known as the reality gap, challenge the transferability of test results to deployed systems. In this paper, we present a comprehensive empirical study comparing four representative testing modalities: Software-in-the-Loop (SiL), Vehicle-in-the-Loop (ViL), Mixed-Reality (MR), and full real-world testing. Using a small-scale physical vehicle equipped with real sensors (camera and LiDAR) and its digital twin, we implement each setup and evaluate two ADS architectures (modular and end-to-end) across diverse indoor driving scenarios involving real obstacles, road topologies, and indoor environments. We systematically assess the impact of each testing modality along three dimensions of the reality gap: actuation, perception, and behavioral fidelity. Our results show that while SiL and ViL setups simplify critical aspects of real-world dynamics and sensing, MR testing improves perceptual realism without compromising safety or control. Importantly, we identify the conditions under which failures do not transfer across testing modalities and isolate the underlying dimensions of the gap responsible for these discrepancies. Our findings offer actionable insights into the respective strengths and limitations of each modality and outline a path toward more robust and transferable validation of autonomous driving systems. |
| 2025-09-26 | [Investigating Faithfulness in Large Audio Language Models](http://arxiv.org/abs/2509.22363v1) | Lovenya Jain, Pooneh Mousavi et al. | Faithfulness measures whether chain-of-thought (CoT) representations accurately reflect a model's decision process and can be used as reliable explanations. Prior work has shown that CoTs from text-based LLMs are often unfaithful. This question has not been explored for large audio-language models (LALMs), where faithfulness is critical for safety-sensitive applications. Reasoning in LALMs is also more challenging, as models must first extract relevant clues from audio before reasoning over them. In this paper, we investigate the faithfulness of CoTs produced by several LALMs by applying targeted interventions, including paraphrasing, filler token injection, early answering, and introducing mistakes, on two challenging reasoning datasets: SAKURA and MMAR. After going through the aforementioned interventions across several datasets and tasks, our experiments suggest that, LALMs generally produce CoTs that appear to be faithful to their underlying decision processes. |
| 2025-09-26 | [RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation](http://arxiv.org/abs/2509.22356v1) | Enguang Liu, Siyuan Liang et al. | The safety and reliability of embodied agents rely on accurate and unbiased visual perception. However, existing benchmarks mainly emphasize generalization and robustness under perturbations, while systematic quantification of visual bias remains scarce. This gap limits a deeper understanding of how perception influences decision-making stability. To address this issue, we propose RoboView-Bias, the first benchmark specifically designed to systematically quantify visual bias in robotic manipulation, following a principle of factor isolation. Leveraging a structured variant-generation framework and a perceptual-fairness validation protocol, we create 2,127 task instances that enable robust measurement of biases induced by individual visual factors and their interactions. Using this benchmark, we systematically evaluate three representative embodied agents across two prevailing paradigms and report three key findings: (i) all agents exhibit significant visual biases, with camera viewpoint being the most critical factor; (ii) agents achieve their highest success rates on highly saturated colors, indicating inherited visual preferences from underlying VLMs; and (iii) visual biases show strong, asymmetric coupling, with viewpoint strongly amplifying color-related bias. Finally, we demonstrate that a mitigation strategy based on a semantic grounding layer substantially reduces visual bias by approximately 54.5\% on MOKA. Our results highlight that systematic analysis of visual bias is a prerequisite for developing safe and reliable general-purpose embodied agents. |
| 2025-09-26 | [Advancing Natural Language Formalization to First Order Logic with Fine-tuned LLMs](http://arxiv.org/abs/2509.22338v1) | Felix Vossel, Till Mossakowski et al. | Automating the translation of natural language to first-order logic (FOL) is crucial for knowledge representation and formal methods, yet remains challenging. We present a systematic evaluation of fine-tuned LLMs for this task, comparing architectures (encoder-decoder vs. decoder-only) and training strategies. Using the MALLS and Willow datasets, we explore techniques like vocabulary extension, predicate conditioning, and multilingual training, introducing metrics for exact match, logical equivalence, and predicate alignment. Our fine-tuned Flan-T5-XXL achieves 70% accuracy with predicate lists, outperforming GPT-4o and even the DeepSeek-R1-0528 model with CoT reasoning ability as well as symbolic systems like ccg2lambda. Key findings show: (1) predicate availability boosts performance by 15-20%, (2) T5 models surpass larger decoder-only LLMs, and (3) models generalize to unseen logical arguments (FOLIO dataset) without specific training. While structural logic translation proves robust, predicate extraction emerges as the main bottleneck. |
| 2025-09-26 | [Trust and Human Autonomy after Cobot Failures: Communication is Key for Industry 5.0](http://arxiv.org/abs/2509.22298v1) | Felix Glawe, Laura Kremer et al. | Collaborative robots (cobots) are a core technology of Industry 4.0. Industry 4.0 uses cyber-physical systems, IoT and smart automation to improve efficiency and data-driven decision-making. Cobots, as cyber-physical systems, enable the introduction of lightweight automation to smaller companies through their flexibility, low cost and ability to work alongside humans, while keeping humans and their skills in the loop. Industry 5.0, the evolution of Industry 4.0, places the worker at the centre of its principles: The physical and mental well-being of the worker is the main goal of new technology design, not just productivity, efficiency and safety standards. Within this concept, human trust in cobots and human autonomy are important. While trust is essential for effective and smooth interaction, the workers' perception of autonomy is key to intrinsic motivation and overall well-being. As failures are an inevitable part of technological systems, this study aims to answer the question of how system failures affect trust in cobots as well as human autonomy, and how they can be recovered afterwards. Therefore, a VR experiment (n = 39) was set up to investigate the influence of a cobot failure and its severity on human autonomy and trust in the cobot. Furthermore, the influence of transparent communication about the failure and next steps was investigated. The results show that both trust and autonomy suffer after cobot failures, with the severity of the failure having a stronger negative impact on trust, but not on autonomy. Both trust and autonomy can be partially restored by transparent communication. |
| 2025-09-26 | [Jailbreaking on Text-to-Video Models via Scene Splitting Strategy](http://arxiv.org/abs/2509.22292v1) | Wonjun Lee, Haon Park et al. | Along with the rapid advancement of numerous Text-to-Video (T2V) models, growing concerns have emerged regarding their safety risks. While recent studies have explored vulnerabilities in models like LLMs, VLMs, and Text-to-Image (T2I) models through jailbreak attacks, T2V models remain largely unexplored, leaving a significant safety gap. To address this gap, we introduce SceneSplit, a novel black-box jailbreak method that works by fragmenting a harmful narrative into multiple scenes, each individually benign. This approach manipulates the generative output space, the abstract set of all potential video outputs for a given prompt, using the combination of scenes as a powerful constraint to guide the final outcome. While each scene individually corresponds to a wide and safe space where most outcomes are benign, their sequential combination collectively restricts this space, narrowing it to an unsafe region and significantly increasing the likelihood of generating a harmful video. This core mechanism is further enhanced through iterative scene manipulation, which bypasses the safety filter within this constrained unsafe region. Additionally, a strategy library that reuses successful attack patterns further improves the attack's overall effectiveness and robustness. To validate our method, we evaluate SceneSplit across 11 safety categories on T2V models. Our results show that it achieves a high average Attack Success Rate (ASR) of 77.2% on Luma Ray2, 84.1% on Hailuo, and 78.2% on Veo2, significantly outperforming the existing baseline. Through this work, we demonstrate that current T2V safety mechanisms are vulnerable to attacks that exploit narrative structure, providing new insights for understanding and improving the safety of T2V models. |
| 2025-09-26 | [Rule-Based Reinforcement Learning for Document Image Classification with Vision Language Models](http://arxiv.org/abs/2509.22283v1) | Michael Jungo, Andreas Fischer | Rule-based reinforcement learning has been gaining popularity ever since DeepSeek-R1 has demonstrated its success through simple verifiable rewards. In the domain of document analysis, reinforcement learning is not as prevalent, even though many downstream tasks may benefit from the emerging properties of reinforcement learning, particularly the enhanced reason capabilities. We study the effects of rule-based reinforcement learning with the task of Document Image Classification which is one of the most commonly studied downstream tasks in document analysis. We find that reinforcement learning tends to have better generalisation capabilities to out-of-distritbution data, which we examine in three different scenarios, namely out-of-distribution images, unseen classes and different modalities. Our code is available at https://github.com/jungomi/vision-finetune. |
| 2025-09-26 | [Towards a more realistic evaluation of machine learning models for bearing fault diagnosis](http://arxiv.org/abs/2509.22267v1) | João Paulo Vieira, Victor Afonso Bauler et al. | Reliable detection of bearing faults is essential for maintaining the safety and operational efficiency of rotating machinery. While recent advances in machine learning (ML), particularly deep learning, have shown strong performance in controlled settings, many studies fail to generalize to real-world applications due to methodological flaws, most notably data leakage. This paper investigates the issue of data leakage in vibration-based bearing fault diagnosis and its impact on model evaluation. We demonstrate that common dataset partitioning strategies, such as segment-wise and condition-wise splits, introduce spurious correlations that inflate performance metrics. To address this, we propose a rigorous, leakage-free evaluation methodology centered on bearing-wise data partitioning, ensuring no overlap between the physical components used for training and testing. Additionally, we reformulate the classification task as a multi-label problem, enabling the detection of co-occurring fault types and the use of prevalence-independent metrics such as Macro AUROC. Beyond preventing leakage, we also examine the effect of dataset diversity on generalization, showing that the number of unique training bearings is a decisive factor for achieving robust performance. We evaluate our methodology on three widely adopted datasets: CWRU, Paderborn University (PU), and University of Ottawa (UORED-VAFCLS). This study highlights the importance of leakage-aware evaluation protocols and provides practical guidelines for dataset partitioning, model selection, and validation, fostering the development of more trustworthy ML systems for industrial fault diagnosis applications. |
| 2025-09-26 | [Safety Compliance: Rethinking LLM Safety Reasoning through the Lens of Compliance](http://arxiv.org/abs/2509.22250v1) | Wenbin Hu, Huihao Jing et al. | The proliferation of Large Language Models (LLMs) has demonstrated remarkable capabilities, elevating the critical importance of LLM safety. However, existing safety methods rely on ad-hoc taxonomy and lack a rigorous, systematic protection, failing to ensure safety for the nuanced and complex behaviors of modern LLM systems. To address this problem, we solve LLM safety from legal compliance perspectives, named safety compliance. In this work, we posit relevant established legal frameworks as safety standards for defining and measuring safety compliance, including the EU AI Act and GDPR, which serve as core legal frameworks for AI safety and data security in Europe. To bridge the gap between LLM safety and legal compliance, we first develop a new benchmark for safety compliance by generating realistic LLM safety scenarios seeded with legal statutes. Subsequently, we align Qwen3-8B using Group Policy Optimization (GRPO) to construct a safety reasoner, Compliance Reasoner, which effectively aligns LLMs with legal standards to mitigate safety risks. Our comprehensive experiments demonstrate that the Compliance Reasoner achieves superior performance on the new benchmark, with average improvements of +10.45% for the EU AI Act and +11.85% for GDPR. |
| 2025-09-26 | [A Correct by Construction Fault Tolerant Voter for Input Selection of a Control System](http://arxiv.org/abs/2509.22236v1) | Arif Ali AP, Jasine Babu et al. | Safety-critical systems use redundant input units to improve their reliability and fault tolerance. A voting logic is then used to select a reliable input from the redundant sources. A fault detection and isolation rules help in selecting input units that can participate in voting. This work deals with the formal requirement formulation, design, verification and synthesis of a generic voting unit for an $N$-modular redundant measurement system used for control applications in avionics systems. The work follows a correct-by-construction approach, using the Rocq theorem prover. |
| 2025-09-26 | [UrbanFeel: A Comprehensive Benchmark for Temporal and Perceptual Understanding of City Scenes through Human Perspective](http://arxiv.org/abs/2509.22228v1) | Jun He, Yi Lin et al. | Urban development impacts over half of the global population, making human-centered understanding of its structural and perceptual changes essential for sustainable development. While Multimodal Large Language Models (MLLMs) have shown remarkable capabilities across various domains, existing benchmarks that explore their performance in urban environments remain limited, lacking systematic exploration of temporal evolution and subjective perception of urban environment that aligns with human perception. To address these limitations, we propose UrbanFeel, a comprehensive benchmark designed to evaluate the performance of MLLMs in urban development understanding and subjective environmental perception. UrbanFeel comprises 14.3K carefully constructed visual questions spanning three cognitively progressive dimensions: Static Scene Perception, Temporal Change Understanding, and Subjective Environmental Perception. We collect multi-temporal single-view and panoramic street-view images from 11 representative cities worldwide, and generate high-quality question-answer pairs through a hybrid pipeline of spatial clustering, rule-based generation, model-assisted prompting, and manual annotation. Through extensive evaluation of 20 state-of-the-art MLLMs, we observe that Gemini-2.5 Pro achieves the best overall performance, with its accuracy approaching human expert levels and narrowing the average gap to just 1.5\%. Most models perform well on tasks grounded in scene understanding. In particular, some models even surpass human annotators in pixel-level change detection. However, performance drops notably in tasks requiring temporal reasoning over urban development. Additionally, in the subjective perception dimension, several models reach human-level or even higher consistency in evaluating dimension such as beautiful and safety. |
| 2025-09-26 | [Thinking in Many Modes: How Composite Reasoning Elevates Large Language Model Performance with Limited Data](http://arxiv.org/abs/2509.22224v1) | Zishan Ahmad, Saisubramaniam Gopalakrishnan | Large Language Models (LLMs), despite their remarkable capabilities, rely on singular, pre-dominant reasoning paradigms, hindering their performance on intricate problems that demand diverse cognitive strategies. To address this, we introduce Composite Reasoning (CR), a novel reasoning approach empowering LLMs to dynamically explore and combine multiple reasoning styles like deductive, inductive, and abductive for more nuanced problem-solving. Evaluated on scientific and medical question-answering benchmarks, our approach outperforms existing baselines like Chain-of-Thought (CoT) and also surpasses the accuracy of DeepSeek-R1 style reasoning (SR) capabilities, while demonstrating superior sample efficiency and adequate token usage. Notably, CR adaptively emphasizes domain-appropriate reasoning styles. It prioritizes abductive and deductive reasoning for medical question answering, but shifts to causal, deductive, and inductive methods for scientific reasoning. Our findings highlight that by cultivating internal reasoning style diversity, LLMs acquire more robust, adaptive, and efficient problem-solving abilities. |
| 2025-09-26 | [Context Parametrization with Compositional Adapters](http://arxiv.org/abs/2509.22158v1) | Josip Jukić, Martin Tutek et al. | Large language models (LLMs) often seamlessly adapt to new tasks through in-context learning (ICL) or supervised fine-tuning (SFT). However, both of these approaches face key limitations: ICL is inefficient when handling many demonstrations, and SFT incurs training overhead while sacrificing flexibility. Mapping instructions or demonstrations from context directly into adapter parameters offers an appealing alternative. While prior work explored generating adapters based on a single input context, it has overlooked the need to integrate multiple chunks of information. To address this gap, we introduce CompAs, a meta-learning framework that translates context into adapter parameters with a compositional structure. Adapters generated this way can be merged algebraically, enabling instructions, demonstrations, or retrieved passages to be seamlessly combined without reprocessing long prompts. Critically, this approach yields three benefits: lower inference cost, robustness to long-context instability, and establishes a principled solution when input exceeds the model's context window. Furthermore, CompAs encodes information into adapter parameters in a reversible manner, enabling recovery of input context through a decoder, facilitating safety and security. Empirical results on diverse multiple-choice and extractive question answering tasks show that CompAs outperforms ICL and prior generator-based methods, especially when scaling to more inputs. Our work establishes composable adapter generation as a practical and efficient alternative for scaling LLM deployment. |
| 2025-09-26 | [Which Values Matter to Socially Assistive Robots in Elder Care Settings? Empirically Investigating Values That Should Be Embedded in SARs from a Multi-Stakeholder Perspective](http://arxiv.org/abs/2509.22146v1) | Vivienne Jia Zhong, Theresa Schmiedel | The integration of socially assistive robots (SARs) in elder care settings has the potential to address critical labor shortages while enhancing the quality of care. However, the design of SARs must align with the values of various stakeholders to ensure their acceptance and efficacy. This study empirically investigates the values that should be embedded in SARs from a multi-stakeholder perspective, including care receivers, caregivers, therapists, relatives, and other involved parties. Utilizing a combination of semi-structured interviews and focus groups, we identify a wide range of values related to safety, trust, care, privacy, and autonomy, and illustrate how stakeholders interpret these values in real-world care environments. Our findings reveal several value tensions and propose potential resolutions to these tensions. Additionally, the study highlights under-researched values such as calmness and collaboration, which are critical in fostering a supportive and efficient care environment. Our work contributes to the understanding of value-sensitive design of SARs and aids practitioners in developing SARs that align with human values, ultimately promoting socially responsible applications in elder care settings. |

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



