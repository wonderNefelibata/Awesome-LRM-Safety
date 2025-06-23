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
| 2025-06-20 | [VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning](http://arxiv.org/abs/2506.17221v1) | Zhangyang Qi, Zhixiong Zhang et al. | Vision-Language Navigation (VLN) is a core challenge in embodied AI, requiring agents to navigate real-world environments using natural language instructions. Current language model-based navigation systems operate on discrete topological graphs, limiting path planning to predefined node connections. We propose VLN-R1, an end-to-end framework that leverages Large Vision-Language Models (LVLM) to directly translate egocentric video streams into continuous navigation actions, adopting GRPO-based training inspired by DeepSeek-R1. To enable effective training, we first construct the VLN-Ego dataset using a 3D simulator, Habitat, and propose Long-Short Memory Sampling to balance historical and current observations. While large language models can supervise complete textual instructions, they lack fine-grained action-level control. Our framework employs a two-stage training approach: a) Supervised fine-tuning (SFT) to align the model's action sequence text predictions with expert demonstrations, followed by b) Reinforcement fine-tuning (RFT) enhanced with a Time-Decayed Reward (TDR) mechanism that strategically weights multi-step future actions. Experimental results show VLN-R1 achieves strong performance on VLN-CE benchmark. VLN-R1 proves LVLMs can drive embodied navigation and enhance task-specific reasoning through data-efficient, reward-driven post-training. |
| 2025-06-20 | [Fine-Tuning Lowers Safety and Disrupts Evaluation Consistency](http://arxiv.org/abs/2506.17209v1) | Kathleen C. Fraser, Hillary Dawkins et al. | Fine-tuning a general-purpose large language model (LLM) for a specific domain or task has become a routine procedure for ordinary users. However, fine-tuning is known to remove the safety alignment features of the model, even when the fine-tuning data does not contain any harmful content. We consider this to be a critical failure mode of LLMs due to the widespread uptake of fine-tuning, combined with the benign nature of the "attack". Most well-intentioned developers are likely unaware that they are deploying an LLM with reduced safety. On the other hand, this known vulnerability can be easily exploited by malicious actors intending to bypass safety guardrails. To make any meaningful progress in mitigating this issue, we first need reliable and reproducible safety evaluations. In this work, we investigate how robust a safety benchmark is to trivial variations in the experimental procedure, and the stochastic nature of LLMs. Our initial experiments expose surprising variance in the results of the safety evaluation, even when seemingly inconsequential changes are made to the fine-tuning setup. Our observations have serious implications for how researchers in this field should report results to enable meaningful comparisons in the future. |
| 2025-06-20 | [$^{50}$Cr and $^{53}$Cr neutron capture cross sections measurement at the n_TOF facility at CERN](http://arxiv.org/abs/2506.17161v1) | P. Pérez-Maroto, C. Guerrero et al. | $^{50,53}$Cr are very relevant in criticality safety benchmarks related to nuclear reactors. The discrepancies between the neutron capture cross section evaluations have an important effect on the $k_{eff}$ and $k_{\infty}$ in criticality benchmarks particularly sensitive to chromium. The $^{50,53}$Cr(n,$\gamma$) cross sections is to be determined between 1 and 100 keV with an 8-10% accuracy following the requirements of the NEA High Priority Request List (HPRL) to solve the current discrepancies. We have measured the neutron capture cross sections by the time-of-flight technique at the EAR1 experimental area of the n_TOF facility, using an array of four C$_6$D$_6$ detectors with very low neutron sensitivity. The highly-enriched samples used are significantly thinner than in previous measurements, thus minimizing the multiple-scattering effects. We have produced, and analyzed with the R-matrix analysis code SAMMY, capture yields featuring 33 resonances of $^{50}$Cr and 51 of $^{53}$Cr with an accuracy between 5% and 9%, hence fulfilling the requirements made by the NEA. The differential and integral cross sections have been compared to previous data and evaluations. The new $^{50,53}$Cr(n,$\gamma$) cross sections measured at the CERN n\TOF facility provide a valuable input for upcoming evaluations, which are deemed necessary given that the results presented herein do not support the increase in both cross sections proposed in the recent INDEN evaluation. |
| 2025-06-20 | [Mathematical Proof as a Litmus Test: Revealing Failure Modes of Advanced Large Reasoning Models](http://arxiv.org/abs/2506.17114v1) | Dadi Guo, Jiayu Liu et al. | Large reasoning models (e.g., R1, o3) have demonstrated remarkable mathematical problem-solving abilities. However, the high reported accuracy of these advanced models on popular datasets, reliance on purely numerical evaluation and potential benchmark leakage, often masks their true reasoning shortcomings. To address this, we propose leveraging the inherent rigor and methodological complexity of mathematical proofs as a diagnostic tool to expose these hidden failures. Specifically, we introduce the RFMDataset (Reveal Failure Modes), a collection of 200 diverse mathematical proof problems, and thoroughly evaluate advanced models' performance on it. Our in-depth analysis of their failures uncovers 10 fine-grained error types, which shows fundamental limitations in current large reasoning models: 1) large reasoning models grapple profoundly with mathematical proofs, with some generating entirely correct proofs for less than 20% of problems and failing even on basic ones; 2) models exhibit a diverse spectrum of reasoning failures, prominently demonstrating the lack of guarantees for the correctness and rigor of single-step reasoning; and 3) models show hallucination and incompleteness during the reasoning process. Our findings reveal that models' self-reflection is insufficient to resolve the current logical dilemmas, necessitating formalized and fine-grained logical training. |
| 2025-06-20 | [Are Bias Evaluation Methods Biased ?](http://arxiv.org/abs/2506.17111v1) | Lina Berrayana, Sean Rooney et al. | The creation of benchmarks to evaluate the safety of Large Language Models is one of the key activities within the trusted AI community. These benchmarks allow models to be compared for different aspects of safety such as toxicity, bias, harmful behavior etc. Independent benchmarks adopt different approaches with distinct data sets and evaluation methods. We investigate how robust such benchmarks are by using different approaches to rank a set of representative models for bias and compare how similar are the overall rankings. We show that different but widely used bias evaluations methods result in disparate model rankings. We conclude with recommendations for the community in the usage of such benchmarks. |
| 2025-06-20 | [A low-cost plasma source aimed for medical applications using Ar as the working gas](http://arxiv.org/abs/2506.17072v1) | Fellype do Nascimento, Bruno Henrique da Silva Leal et al. | Because of the advances in equipment and several studies on medical applications of atmospheric pressure plasmas over the last few years, plasma sources are now being produced for clinical use. It has been demonstrated that plasma sources can be constructed in such a way that their operation and application are safe, and that they present relevant clinical results. However, the manufacturing and operating costs of plasma sources can be decisive for their adoption in medical procedures. In this work, a simple modification of a plasma source that has a low production cost was investigated in order to evaluate the viability of its use for medical applications using argon as the working gas. Without this modification, the plasma source produces high levels of electrical current, which makes it unfeasible for use in medical applications. With the proposed modification, the treatment is no longer carried out using the plasma jet directly, but rather with the post-discharge effluent enriched with reactive oxygen and nitrogen species. As a result, the electrical current reaching the target remains below the safety threshold established for plasma applications in humans. Application tests on water activation indicate that both operating conditions can yield comparable outcomes. |
| 2025-06-20 | [From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers](http://arxiv.org/abs/2506.17052v1) | Jingtong Su, Julia Kempe et al. | Transformers have achieved state-of-the-art performance across language and vision tasks. This success drives the imperative to interpret their internal mechanisms with the dual goals of enhancing performance and improving behavioral control. Attribution methods help advance interpretability by assigning model outputs associated with a target concept to specific model components. Current attribution research primarily studies multi-layer perceptron neurons and addresses relatively simple concepts such as factual associations (e.g., Paris is located in France). This focus tends to overlook the impact of the attention mechanism and lacks a unified approach for analyzing more complex concepts. To fill these gaps, we introduce Scalable Attention Module Discovery (SAMD), a concept-agnostic method for mapping arbitrary, complex concepts to specific attention heads of general transformer models. We accomplish this by representing each concept as a vector, calculating its cosine similarity with each attention head, and selecting the TopK-scoring heads to construct the concept-associated attention module. We then propose Scalar Attention Module Intervention (SAMI), a simple strategy to diminish or amplify the effects of a concept by adjusting the attention module using only a single scalar parameter. Empirically, we demonstrate SAMD on concepts of varying complexity, and visualize the locations of their corresponding modules. Our results demonstrate that module locations remain stable before and after LLM post-training, and confirm prior work on the mechanics of LLM multilingualism. Through SAMI, we facilitate jailbreaking on HarmBench (+72.7%) by diminishing "safety" and improve performance on the GSM8K benchmark (+1.6%) by amplifying "reasoning". Lastly, we highlight the domain-agnostic nature of our approach by suppressing the image classification accuracy of vision transformers on ImageNet. |
| 2025-06-20 | [Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs](http://arxiv.org/abs/2506.16962v1) | Haoran Sun, Yankai Jiang et al. | Multimodal large language models (MLLMs) have begun to demonstrate robust reasoning capabilities on general tasks, yet their application in the medical domain remains in its early stages. Constructing chain-of-thought (CoT) training data is essential for bolstering the reasoning abilities of medical MLLMs. However, existing approaches exhibit a deficiency in offering a comprehensive framework for searching and evaluating effective reasoning paths towards critical diagnosis. To address this challenge, we propose Mentor-Intern Collaborative Search (MICS), a novel reasoning-path searching scheme to generate rigorous and effective medical CoT data. MICS first leverages mentor models to initialize the reasoning, one step at a time, then prompts each intern model to continue the thinking along those initiated paths, and finally selects the optimal reasoning path according to the overall reasoning performance of multiple intern models. The reasoning performance is determined by an MICS-Score, which assesses the quality of generated reasoning paths. Eventually, we construct MMRP, a multi-task medical reasoning dataset with ranked difficulty, and Chiron-o1, a new medical MLLM devised via a curriculum learning strategy, with robust visual question-answering and generalizable reasoning capabilities. Extensive experiments demonstrate that Chiron-o1, trained on our CoT dataset constructed using MICS, achieves state-of-the-art performance across a list of medical visual question answering and reasoning benchmarks. Codes are available at GitHub - manglu097/Chiron-o1: Enhancing Step-by-Step and Verifiable Medical Reasoning in MLLMs |
| 2025-06-20 | [Orbital Collision: An Indigenously Developed Web-based Space Situational Awareness Platform](http://arxiv.org/abs/2506.16892v1) | Partha Chowdhury, Harsha M et al. | This work presents an indigenous web based platform Orbital Collision (OrCo), created by the Space Systems Laboratory at IIIT Delhi, to enhance Space Situational Awareness (SSA) by predicting collision probabilities of space objects using Two Line Elements (TLE) data. The work highlights the growing challenges of congestion in the Earth's orbital environment, mainly due to space debris and defunct satellites, which increase collision risks. It employs several methods for propagating orbital uncertainty and calculating the collision probability. The performance of the platform is evaluated through accuracy assessments and efficiency metrics, in order to improve the tracking of space objects and ensure the safety of the satellite in congested space. |
| 2025-06-20 | [Revolutionizing Validation and Verification: Explainable Testing Methodologies for Intelligent Automotive Decision-Making Systems](http://arxiv.org/abs/2506.16876v1) | Halit Eris, Stefan Wagner | Autonomous Driving Systems (ADS) use complex decision-making (DM) models with multimodal sensory inputs, making rigorous validation and verification (V&V) essential for safety and reliability. These models pose challenges in diagnosing failures, tracing anomalies, and maintaining transparency, with current manual testing methods being inefficient and labor-intensive. This vision paper presents a methodology that integrates explainability, transparency, and interpretability into V&V processes. We propose refining V&V requirements through literature reviews and stakeholder input, generating explainable test scenarios via large language models (LLMs), and enabling real-time validation in simulation environments. Our framework includes test oracle, explanation generation, and a test chatbot, with empirical studies planned to evaluate improvements in diagnostic efficiency and transparency. Our goal is to streamline V&V, reduce resources, and build user trust in autonomous technologies. |
| 2025-06-20 | [ParkFormer: A Transformer-Based Parking Policy with Goal Embedding and Pedestrian-Aware Control](http://arxiv.org/abs/2506.16856v1) | Jun Fu, Bin Tian et al. | Autonomous parking plays a vital role in intelligent vehicle systems, particularly in constrained urban environments where high-precision control is required. While traditional rule-based parking systems struggle with environmental uncertainties and lack adaptability in crowded or dynamic scenes, human drivers demonstrate the ability to park intuitively without explicit modeling. Inspired by this observation, we propose a Transformer-based end-to-end framework for autonomous parking that learns from expert demonstrations. The network takes as input surround-view camera images, goal-point representations, ego vehicle motion, and pedestrian trajectories. It outputs discrete control sequences including throttle, braking, steering, and gear selection. A novel cross-attention module integrates BEV features with target points, and a GRU-based pedestrian predictor enhances safety by modeling dynamic obstacles. We validate our method on the CARLA 0.9.14 simulator in both vertical and parallel parking scenarios. Experiments show our model achieves a high success rate of 96.57\%, with average positional and orientation errors of 0.21 meters and 0.41 degrees, respectively. The ablation studies further demonstrate the effectiveness of key modules such as pedestrian prediction and goal-point attention fusion. The code and dataset will be released at: https://github.com/little-snail-f/ParkFormer. |
| 2025-06-20 | [The status quo of fire evacuation drills in nursery schools](http://arxiv.org/abs/2506.16848v1) | Hana Najmanová, Petr Novák et al. | Fire drills are a commonly-used training method for improving how people act in emergency situations. This article deals with fire drills in early childhood education facilities and analyses data gathered from an online survey of 1151 Czech nursery schools (23.5% of nursery schools invited to participate, 21.5% of all officially registered nursery schools in Czechia). It provides recommendations for improving the effectiveness of fire drills in nursery schools. Results suggest that regular (typically annual) fire drills are common training practices at these schools, but more frequent fire drills led to significantly fewer issues during evacuation. The most frequent evacuation issue encountered during fire drills, according to our data, was slow movement. Many children needed assistance, notably on stairs; nursery schools located only on the ground floor reported fewer issues during evacuation than other schools. The purpose, design, and implementation of fire drills with our study confirming results from prior studies must be properly integrated into complex and systematic fire safety education programs that specifically reflect the characteristics, needs, and possible limitations of preschool children. |
| 2025-06-20 | [Accountability of Robust and Reliable AI-Enabled Systems: A Preliminary Study and Roadmap](http://arxiv.org/abs/2506.16831v1) | Filippo Scaramuzza, Damian A. Tamburri et al. | This vision paper presents initial research on assessing the robustness and reliability of AI-enabled systems, and key factors in ensuring their safety and effectiveness in practical applications, including a focus on accountability. By exploring evolving definitions of these concepts and reviewing current literature, the study highlights major challenges and approaches in the field. A case study is used to illustrate real-world applications, emphasizing the need for innovative testing solutions. The incorporation of accountability is crucial for building trust and ensuring responsible AI development. The paper outlines potential future research directions and identifies existing gaps, positioning robustness, reliability, and accountability as vital areas for the development of trustworthy AI systems of the future. |
| 2025-06-20 | [MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning](http://arxiv.org/abs/2506.16792v1) | Muyang Zheng, Yuanzhi Yao et al. | Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks--methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version--order-determining optimization. Extensive experiments across two open-source models and four closed-source models demonstrate that MIST achieves competitive attack success rates and attack transferability compared with other state-of-the-art white-box and black-box jailbreak methods. Additionally, we conduct experiments on computational efficiency to validate the practical viability of MIST. |
| 2025-06-20 | [Cross-Modal Obfuscation for Jailbreak Attacks on Large Vision-Language Models](http://arxiv.org/abs/2506.16760v1) | Lei Jiang, Zixun Zhang et al. | Large Vision-Language Models (LVLMs) demonstrate exceptional performance across multimodal tasks, yet remain vulnerable to jailbreak attacks that bypass built-in safety mechanisms to elicit restricted content generation. Existing black-box jailbreak methods primarily rely on adversarial textual prompts or image perturbations, yet these approaches are highly detectable by standard content filtering systems and exhibit low query and computational efficiency. In this work, we present Cross-modal Adversarial Multimodal Obfuscation (CAMO), a novel black-box jailbreak attack framework that decomposes malicious prompts into semantically benign visual and textual fragments. By leveraging LVLMs' cross-modal reasoning abilities, CAMO covertly reconstructs harmful instructions through multi-step reasoning, evading conventional detection mechanisms. Our approach supports adjustable reasoning complexity and requires significantly fewer queries than prior attacks, enabling both stealth and efficiency. Comprehensive evaluations conducted on leading LVLMs validate CAMO's effectiveness, showcasing robust performance and strong cross-model transferability. These results underscore significant vulnerabilities in current built-in safety mechanisms, emphasizing an urgent need for advanced, alignment-aware security and safety solutions in vision-language systems. |
| 2025-06-20 | [A Scalable Post-Processing Pipeline for Large-Scale Free-Space Multi-Agent Path Planning with PiBT](http://arxiv.org/abs/2506.16748v1) | Arjo Chakravarty, Michael X. Grey et al. | Free-space multi-agent path planning remains challenging at large scales. Most existing methods either offer optimality guarantees but do not scale beyond a few dozen agents, or rely on grid-world assumptions that do not generalize well to continuous space. In this work, we propose a hybrid, rule-based planning framework that combines Priority Inheritance with Backtracking (PiBT) with a novel safety-aware path smoothing method. Our approach extends PiBT to 8-connected grids and selectively applies string-pulling based smoothing while preserving collision safety through local interaction awareness and a fallback collision resolution step based on Safe Interval Path Planning (SIPP). This design allows us to reduce overall path lengths while maintaining real-time performance. We demonstrate that our method can scale to over 500 agents in large free-space environments, outperforming existing any-angle and optimal methods in terms of runtime, while producing near-optimal trajectories in sparse domains. Our results suggest this framework is a promising building block for scalable, real-time multi-agent navigation in robotics systems operating beyond grid constraints. |
| 2025-06-20 | [VLM-Empowered Multi-Mode System for Efficient and Safe Planetary Navigation](http://arxiv.org/abs/2506.16703v1) | Sinuo Cheng, Ruyi Zhou et al. | The increasingly complex and diverse planetary exploration environment requires more adaptable and flexible rover navigation strategy. In this study, we propose a VLM-empowered multi-mode system to achieve efficient while safe autonomous navigation for planetary rovers. Vision-Language Model (VLM) is used to parse scene information by image inputs to achieve a human-level understanding of terrain complexity. Based on the complexity classification, the system switches to the most suitable navigation mode, composing of perception, mapping and planning modules designed for different terrain types, to traverse the terrain ahead before reaching the next waypoint. By integrating the local navigation system with a map server and a global waypoint generation module, the rover is equipped to handle long-distance navigation tasks in complex scenarios. The navigation system is evaluated in various simulation environments. Compared to the single-mode conservative navigation method, our multi-mode system is able to bootstrap the time and energy efficiency in a long-distance traversal with varied type of obstacles, enhancing efficiency by 79.5%, while maintaining its avoidance capabilities against terrain hazards to guarantee rover safety. More system information is shown at https://chengsn1234.github.io/multi-mode-planetary-navigation/. |
| 2025-06-20 | [Exploring Traffic Simulation and Cybersecurity Strategies Using Large Language Models](http://arxiv.org/abs/2506.16699v1) | Lu Gao, Yongxin Liu et al. | Intelligent Transportation Systems (ITS) are increasingly vulnerable to sophisticated cyberattacks due to their complex, interconnected nature. Ensuring the cybersecurity of these systems is paramount to maintaining road safety and minimizing traffic disruptions. This study presents a novel multi-agent framework leveraging Large Language Models (LLMs) to enhance traffic simulation and cybersecurity testing. The framework automates the creation of traffic scenarios, the design of cyberattack strategies, and the development of defense mechanisms. A case study demonstrates the framework's ability to simulate a cyberattack targeting connected vehicle broadcasts, evaluate its impact, and implement a defense mechanism that significantly mitigates traffic delays. Results show a 10.2 percent increase in travel time during an attack, which is reduced by 3.3 percent with the defense strategy. This research highlights the potential of LLM-driven multi-agent systems in advancing transportation cybersecurity and offers a scalable approach for future research in traffic simulation and cyber defense. |
| 2025-06-20 | [PPTP: Performance-Guided Physiological Signal-Based Trust Prediction in Human-Robot Collaboration](http://arxiv.org/abs/2506.16677v1) | Hao Guo, Wei Fan et al. | Trust prediction is a key issue in human-robot collaboration, especially in construction scenarios where maintaining appropriate trust calibration is critical for safety and efficiency. This paper introduces the Performance-guided Physiological signal-based Trust Prediction (PPTP), a novel framework designed to improve trust assessment. We designed a human-robot construction scenario with three difficulty levels to induce different trust states. Our approach integrates synchronized multimodal physiological signals (ECG, GSR, and EMG) with collaboration performance evaluation to predict human trust levels. Individual physiological signals are processed using collaboration performance information as guiding cues, leveraging the standardized nature of collaboration performance to compensate for individual variations in physiological responses. Extensive experiments demonstrate the efficacy of our cross-modality fusion method in significantly improving trust classification performance. Our model achieves over 81% accuracy in three-level trust classification, outperforming the best baseline method by 6.7%, and notably reaches 74.3% accuracy in high-resolution seven-level classification, which is a first in trust prediction research. Ablation experiments further validate the superiority of physiological signal processing guided by collaboration performance assessment. |
| 2025-06-19 | [LLMs in Coding and their Impact on the Commercial Software Engineering Landscape](http://arxiv.org/abs/2506.16653v1) | Vladislav Belozerov, Peter J Barclay et al. | Large-language-model coding tools are now mainstream in software engineering. But as these same tools move human effort up the development stack, they present fresh dangers: 10% of real prompts leak private data, 42% of generated snippets hide security flaws, and the models can even ``agree'' with wrong ideas, a trait called sycophancy. We argue that firms must tag and review every AI-generated line of code, keep prompts and outputs inside private or on-premises deployments, obey emerging safety regulations, and add tests that catch sycophantic answers -- so they can gain speed without losing security and accuracy. |

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



