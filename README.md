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
| 2025-04-14 | [Decoupled Diffusion Sparks Adaptive Scene Generation](http://arxiv.org/abs/2504.10485v1) | Yunsong Zhou, Naisheng Ye et al. | Controllable scene generation could reduce the cost of diverse data collection substantially for autonomous driving. Prior works formulate the traffic layout generation as predictive progress, either by denoising entire sequences at once or by iteratively predicting the next frame. However, full sequence denoising hinders online reaction, while the latter's short-sighted next-frame prediction lacks precise goal-state guidance. Further, the learned model struggles to generate complex or challenging scenarios due to a large number of safe and ordinal driving behaviors from open datasets. To overcome these, we introduce Nexus, a decoupled scene generation framework that improves reactivity and goal conditioning by simulating both ordinal and challenging scenarios from fine-grained tokens with independent noise states. At the core of the decoupled pipeline is the integration of a partial noise-masking training strategy and a noise-aware schedule that ensures timely environmental updates throughout the denoising process. To complement challenging scenario generation, we collect a dataset consisting of complex corner cases. It covers 540 hours of simulated data, including high-risk interactions such as cut-in, sudden braking, and collision. Nexus achieves superior generation realism while preserving reactivity and goal orientation, with a 40% reduction in displacement error. We further demonstrate that Nexus improves closed-loop planning by 20% through data augmentation and showcase its capability in safety-critical data generation. |
| 2025-04-14 | [xVerify: Efficient Answer Verifier for Reasoning Model Evaluations](http://arxiv.org/abs/2504.10481v1) | Ding Chen, Qingchen Yu et al. | With the release of the o1 model by OpenAI, reasoning models adopting slow thinking strategies have gradually emerged. As the responses generated by such models often include complex reasoning, intermediate steps, and self-reflection, existing evaluation methods are often inadequate. They struggle to determine whether the LLM output is truly equivalent to the reference answer, and also have difficulty identifying and extracting the final answer from long, complex responses. To address this issue, we propose xVerify, an efficient answer verifier for reasoning model evaluations. xVerify demonstrates strong capability in equivalence judgment, enabling it to effectively determine whether the answers produced by reasoning models are equivalent to reference answers across various types of objective questions. To train and evaluate xVerify, we construct the VAR dataset by collecting question-answer pairs generated by multiple LLMs across various datasets, leveraging multiple reasoning models and challenging evaluation sets designed specifically for reasoning model assessment. A multi-round annotation process is employed to ensure label accuracy. Based on the VAR dataset, we train multiple xVerify models of different scales. In evaluation experiments conducted on both the test set and generalization set, all xVerify models achieve overall F1 scores and accuracy exceeding 95\%. Notably, the smallest variant, xVerify-0.5B-I, outperforms all evaluation methods except GPT-4o, while xVerify-3B-Ib surpasses GPT-4o in overall performance. These results validate the effectiveness and generalizability of xVerify. |
| 2025-04-14 | [GUI-R1 : A Generalist R1-Style Vision-Language Action Model For GUI Agents](http://arxiv.org/abs/2504.10458v1) | Xiaobo Xia, Run Luo | Existing efforts in building Graphical User Interface (GUI) agents largely rely on the training paradigm of supervised fine-tuning on Large Vision-Language Models (LVLMs). However, this approach not only demands extensive amounts of training data but also struggles to effectively understand GUI screenshots and generalize to unseen interfaces. The issue significantly limits its application in real-world scenarios, especially for high-level tasks. Inspired by Reinforcement Fine-Tuning (RFT) in large reasoning models (e.g., DeepSeek-R1), which efficiently enhances the problem-solving capabilities of large language models in real-world settings, we propose \name, the first reinforcement learning framework designed to enhance the GUI capabilities of LVLMs in high-level real-world task scenarios, through unified action space rule modeling. By leveraging a small amount of carefully curated high-quality data across multiple platforms (including Windows, Linux, MacOS, Android, and Web) and employing policy optimization algorithms such as Group Relative Policy Optimization (GRPO) to update the model, \name achieves superior performance using only 0.02\% of the data (3K vs. 13M) compared to previous state-of-the-art methods like OS-Atlas across eight benchmarks spanning three different platforms (mobile, desktop, and web). These results demonstrate the immense potential of reinforcement learning based on unified action space rule modeling in improving the execution capabilities of LVLMs for real-world GUI agent tasks. |
| 2025-04-14 | [M1: Towards Scalable Test-Time Compute with Mamba Reasoning Models](http://arxiv.org/abs/2504.10449v1) | Junxiong Wang, Wen-Ding Li et al. | Effective reasoning is crucial to solving complex mathematical problems. Recent large language models (LLMs) have boosted performance by scaling test-time computation through long chain-of-thought reasoning. However, transformer-based models are inherently limited in extending context length due to their quadratic computational complexity and linear memory requirements. In this paper, we introduce a novel hybrid linear RNN reasoning model, M1, built on the Mamba architecture, which allows memory-efficient inference. Our approach leverages a distillation process from existing reasoning models and is further enhanced through RL training. Experimental results on the AIME and MATH benchmarks show that M1 not only outperforms previous linear RNN models but also matches the performance of state-of-the-art Deepseek R1 distilled reasoning models at a similar scale. We also compare our generation speed with a highly performant general purpose inference engine, vLLM, and observe more than a 3x speedup compared to a same size transformer. With throughput speedup, we are able to achieve higher accuracy compared to DeepSeek R1 distilled transformer reasoning models under a fixed generation time budget using self-consistency voting. Overall, we introduce a hybrid Mamba reasoning model and provide a more effective approach to scaling test-time generation using self-consistency or long chain of thought reasoning. |
| 2025-04-14 | [LLM Can be a Dangerous Persuader: Empirical Study of Persuasion Safety in Large Language Models](http://arxiv.org/abs/2504.10430v1) | Minqian Liu, Zhiyang Xu et al. | Recent advancements in Large Language Models (LLMs) have enabled them to approach human-level persuasion capabilities. However, such potential also raises concerns about the safety risks of LLM-driven persuasion, particularly their potential for unethical influence through manipulation, deception, exploitation of vulnerabilities, and many other harmful tactics. In this work, we present a systematic investigation of LLM persuasion safety through two critical aspects: (1) whether LLMs appropriately reject unethical persuasion tasks and avoid unethical strategies during execution, including cases where the initial persuasion goal appears ethically neutral, and (2) how influencing factors like personality traits and external pressures affect their behavior. To this end, we introduce PersuSafety, the first comprehensive framework for the assessment of persuasion safety which consists of three stages, i.e., persuasion scene creation, persuasive conversation simulation, and persuasion safety assessment. PersuSafety covers 6 diverse unethical persuasion topics and 15 common unethical strategies. Through extensive experiments across 8 widely used LLMs, we observe significant safety concerns in most LLMs, including failing to identify harmful persuasion tasks and leveraging various unethical persuasion strategies. Our study calls for more attention to improve safety alignment in progressive and goal-driven conversations such as persuasion. |
| 2025-04-14 | [Ctrl-Z: Controlling AI Agents via Resampling](http://arxiv.org/abs/2504.10374v1) | Aryan Bhatt, Cody Rushing et al. | Control evaluations measure whether monitoring and security protocols for AI systems prevent intentionally subversive AI models from causing harm. Our work presents the first control evaluation performed in an agent environment. We construct BashBench, a dataset of 257 challenging multi-step system administration tasks, and evaluate whether various safety measures can prevent an adversarially constructed AI agent from covertly downloading and executing malicious code in this environment. This multi-step setting introduces new attack and defense dynamics, which we investigate in order to design novel control protocols that prevent safety failures without hindering the ability of non-malicious agents to perform useful work. We introduce a class of control protocols called resample protocols that dynamically take additional samples of certain actions. We find these protocols significantly improve on existing techniques by selectively blocking the AI agent from executing suspicious code and incriminating the agent by generating additional examples of dangerous behavior. We measure the tradeoff between attack prevention and usefulness; our best protocol combines resampling with analysis of previous steps, reducing the success rate of attacks from 58% to 7% at a 5% cost to the performance of a non-malicious agent. |
| 2025-04-14 | [Reactive power flow optimization in AC drive systems](http://arxiv.org/abs/2504.10360v1) | Sanjay Chandrasekaran, Catalin Arghir et al. | This paper explores a limit avoidance approach in the case of input (modulation) and output (current) constraints with the aim of enhancing system availability of AC drives. Drawing on the observation that, in a certain range of reactive power, there exists a trade-off between current and modulation magnitude, we exploit this freedom and define a constrained optimization problem. We propose two approaches, one in the form of an activation-function which drives the reactive power set-point towards safety, and an approach which uses online feedback optimization to set the reactive power dynamically. Both methods compromise reactive power tracking accuracy for increased system robustness. Through a high fidelity simulation, we compare the benefits of the two methods, highlighting their effectiveness in industrial applications. |
| 2025-04-14 | [Improving diffusion modeling in all-solid-state lithium batteries: a novel approach for grain boundary effects](http://arxiv.org/abs/2504.10348v1) | Lena Scholz, Yongliang Ou et al. | All-solid-state lithium-ion batteries offer promising advantages with respect to capacity, safety, and performance. The diffusion behavior of lithium ions in the contained polycrystalline solid-state electrolyte is crucial for battery function. While atomistic studies indicate that grain boundaries (GBs) and grain size significantly impact diffusivity, the corresponding effects are either neglected in simulations on larger scales or considered only under strong assumptions such as isotropy. Our approach considers the fully resolved crystalline structure with a parametrization aligned with the atomistic perspective to describe diffusion along and across GBs. The approach is embedded into a finite element simulation using a novel collapsed interface element based on an analytical description in thickness direction. Results are governed by different and potentially anisotropic diffusion coefficients in bulk and GB domains. The mesoscale response is derived using linear computational homogenization to capture large-scale effects. The novel collapsed interface description allows for a reconstruction of the 3D transport behavior within the GB domain without resolving it and is able to capture the relevant transport mechanisms such as channeling effects and concentration jumps. Grain size and GB volume fraction are expressed in terms of an affine parameter dependence and can be altered without any changes to geometry or mesh. Together with the observed dependence of the effective material response on the anisotropic GB parametrization, this leads to the identification of four distinct diffusion regimes, each with implications for the design of battery materials. |
| 2025-04-14 | [Heimdall: test-time scaling on the generative verification](http://arxiv.org/abs/2504.10337v1) | Wenlei Shi, Xing Jin | An AI system can create and maintain knowledge only to the extent that it can verify that knowledge itself. Recent work on long Chain-of-Thought reasoning has demonstrated great potential of LLMs on solving competitive problems, but their verification ability remains to be weak and not sufficiently investigated. In this paper, we propose Heimdall, the long CoT verification LLM that can accurately judge the correctness of solutions. With pure reinforcement learning, we boost the verification accuracy from 62.5% to 94.5% on competitive math problems. By scaling with repeated sampling, the accuracy further increases to 97.5%. Through human evaluation, Heimdall demonstrates impressive generalization capabilities, successfully detecting most issues in challenging math proofs, the type of which is not included during training. Furthermore, we propose Pessimistic Verification to extend the functionality of Heimdall to scaling up the problem solving. It calls Heimdall to judge the solutions from a solver model and based on the pessimistic principle, selects the most likely correct solution with the least uncertainty. Taking DeepSeek-R1-Distill-Qwen-32B as the solver model, Pessimistic Verification improves the solution accuracy on AIME2025 from 54.2% to 70.0% with 16x compute budget and to 83.3% with more compute budget. With the stronger solver Gemini 2.5 Pro, the score reaches 93.0%. Finally, we prototype an automatic knowledge discovery system, a ternary system where one poses questions, another provides solutions, and the third verifies the solutions. Using the data synthesis work NuminaMath for the first two components, Heimdall effectively identifies problematic records within the dataset and reveals that nearly half of the data is flawed, which interestingly aligns with the recent ablation studies from NuminaMath. |
| 2025-04-14 | [Cumulative-Time Signal Temporal Logic](http://arxiv.org/abs/2504.10325v1) | Hongkai Chen, Zeyu Zhang et al. | Signal Temporal Logic (STL) is a widely adopted specification language in cyber-physical systems for expressing critical temporal requirements, such as safety conditions and response time. However, STL's expressivity is not sufficient to capture the cumulative duration during which a property holds within an interval of time. To overcome this limitation, we introduce Cumulative-Time Signal Temporal Logic (CT-STL) that operates over discrete-time signals and extends STL with a new cumulative-time operator. This operator compares the sum of all time steps for which its nested formula is true with a threshold. We present both a qualitative and a quantitative (robustness) semantics for CT-STL and prove both their soundness and completeness properties. We provide an efficient online monitoring algorithm for both semantics. Finally, we show the applicability of CT-STL in two case studies: specifying and monitoring cumulative temporal requirements for a microgrid and an artificial pancreas. |
| 2025-04-14 | [SlowFastVAD: Video Anomaly Detection via Integrating Simple Detector and RAG-Enhanced Vision-Language Model](http://arxiv.org/abs/2504.10320v1) | Zongcan Ding, Haodong Zhang et al. | Video anomaly detection (VAD) aims to identify unexpected events in videos and has wide applications in safety-critical domains. While semi-supervised methods trained on only normal samples have gained traction, they often suffer from high false alarm rates and poor interpretability. Recently, vision-language models (VLMs) have demonstrated strong multimodal reasoning capabilities, offering new opportunities for explainable anomaly detection. However, their high computational cost and lack of domain adaptation hinder real-time deployment and reliability. Inspired by dual complementary pathways in human visual perception, we propose SlowFastVAD, a hybrid framework that integrates a fast anomaly detector with a slow anomaly detector (namely a retrieval augmented generation (RAG) enhanced VLM), to address these limitations. Specifically, the fast detector first provides coarse anomaly confidence scores, and only a small subset of ambiguous segments, rather than the entire video, is further analyzed by the slower yet more interpretable VLM for elaborate detection and reasoning. Furthermore, to adapt VLMs to domain-specific VAD scenarios, we construct a knowledge base including normal patterns based on few normal samples and abnormal patterns inferred by VLMs. During inference, relevant patterns are retrieved and used to augment prompts for anomaly reasoning. Finally, we smoothly fuse the anomaly confidence of fast and slow detectors to enhance robustness of anomaly detection. Extensive experiments on four benchmarks demonstrate that SlowFastVAD effectively combines the strengths of both fast and slow detectors, and achieves remarkable detection accuracy and interpretability with significantly reduced computational overhead, making it well-suited for real-world VAD applications with high reliability requirements. |
| 2025-04-14 | [Siamese Network with Dual Attention for EEG-Driven Social Learning: Bridging the Human-Robot Gap in Long-Tail Autonomous Driving](http://arxiv.org/abs/2504.10296v1) | Xiaoshan Zhou, Carol C. Menassa et al. | Robots with wheeled, quadrupedal, or humanoid forms are increasingly integrated into built environments. However, unlike human social learning, they lack a critical pathway for intrinsic cognitive development, namely, learning from human feedback during interaction. To understand human ubiquitous observation, supervision, and shared control in dynamic and uncertain environments, this study presents a brain-computer interface (BCI) framework that enables classification of Electroencephalogram (EEG) signals to detect cognitively demanding and safety-critical events. As a timely and motivating co-robotic engineering application, we simulate a human-in-the-loop scenario to flag risky events in semi-autonomous robotic driving-representative of long-tail cases that pose persistent bottlenecks to the safety performance of smart mobility systems and robotic vehicles. Drawing on recent advances in few-shot learning, we propose a dual-attention Siamese convolutional network paired with Dynamic Time Warping Barycenter Averaging approach to generate robust EEG-encoded signal representations. Inverse source localization reveals activation in Broadman areas 4 and 9, indicating perception-action coupling during task-relevant mental imagery. The model achieves 80% classification accuracy under data-scarce conditions and exhibits a nearly 100% increase in the utility of salient features compared to state-of-the-art methods, as measured through integrated gradient attribution. Beyond performance, this study contributes to our understanding of the cognitive architecture required for BCI agents-particularly the role of attention and memory mechanisms-in categorizing diverse mental states and supporting both inter- and intra-subject adaptation. Overall, this research advances the development of cognitive robotics and socially guided learning for service robots in complex built environments. |
| 2025-04-14 | [Gravity-induced emergence of the Fermi scale in quantum quadratic gravity](http://arxiv.org/abs/2504.10293v1) | Mohammad Mehrafarin | In the framework of asymptotic safety, we study quantum quadratic gravity in the presence of the Higgs field considered as non-separable from the vacuum. The theory flows to a high energy fixed point where the Higgs field is strongly coupled to gravity, its potential is symmetric, and the quadratic Weyl curvature coupling is large. The latter renders the ghost graviton an unstable high mass resonance which renders unitarity in the spirit of Lee-Week type theories. Furthermore, if the scalar graviton is tachyonic then there will be a low energy fixed point where tachyonic condensation leads to a new stable vacuum. At this fixed point the symmetry breaks and the Fermi scale emerges, and the behavior of the Higgs field is classical (not influenced by gravitational interaction). Gravity at the UV scale is purely quadratic whereas at the Fermi scale it is linear, and in the intermediate region both contributions are relevant. Thus, at the Fermi scale the quadratic curvature fields disappear through the ghost instability and tachyon condensation, giving rise to Einstein gravity and the electroweak phase transition. |
| 2025-04-14 | [Deep Reasoning Translation via Reinforcement Learning](http://arxiv.org/abs/2504.10187v1) | Jiaan Wang, Fandong Meng et al. | Recently, deep reasoning LLMs (e.g., OpenAI o1/o3 and DeepSeek-R1) have shown promising performance in various complex tasks. Free translation is an important and interesting task in the multilingual world, which requires going beyond word-for-word translation and taking cultural differences into account. This task is still under-explored in deep reasoning LLMs. In this paper, we introduce DeepTrans, a deep reasoning translation model that learns free translation via reinforcement learning. Specifically, we carefully build a reward model with pre-defined scoring criteria on both the translation results and the thought process. Given the source sentences, the reward model teaches the deep translation model how to think and free-translate them during reinforcement learning. In this way, training DeepTrans does not need any labeled translations, avoiding the human-intensive annotation or resource-intensive data synthesis. Experimental results show the effectiveness of DeepTrans. Using Qwen2.5-7B as the backbone, DeepTrans improves performance by 16.3% in literature translation, and outperforms strong deep reasoning baselines as well as baselines that are fine-tuned with synthesized data. Moreover, we summarize the failures and interesting findings during our RL exploration. We hope this work could inspire other researchers in free translation. |
| 2025-04-14 | [LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks](http://arxiv.org/abs/2504.10185v1) | Soumyadeep Pal, Changsheng Wang et al. | Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset. |
| 2025-04-14 | [Challenges in interpretability of additive models](http://arxiv.org/abs/2504.10169v1) | Xinyu Zhang, Julien Martinelli et al. | We review generalized additive models as a type of ``transparent'' model that has recently seen renewed interest in the deep learning community as neural additive models. We highlight multiple types of nonidentifiability in this model class and discuss challenges in interpretability, arguing for restraint when claiming ``interpretability'' or ``suitability for safety-critical applications'' of such models. |
| 2025-04-14 | [GeoUni: A Unified Model for Generating Geometry Diagrams, Problems and Problem Solutions](http://arxiv.org/abs/2504.10146v1) | Jo-Ku Cheng, Zeren Zhang et al. | We propose GeoUni, the first unified geometry expert model capable of generating problem solutions and diagrams within a single framework in a way that enables the creation of unique and individualized geometry problems. Traditionally, solving geometry problems and generating diagrams have been treated as separate tasks in machine learning, with no models successfully integrating both to support problem creation. However, we believe that mastery in geometry requires frictionless integration of all of these skills, from solving problems to visualizing geometric relationships, and finally, crafting tailored problems. Our extensive experiments demonstrate that GeoUni, with only 1.5B parameters, achieves performance comparable to larger models such as DeepSeek-R1 with 671B parameters in geometric reasoning tasks. GeoUni also excels in generating precise geometric diagrams, surpassing both text-to-image models and unified models, including the GPT-4o image generation. Most importantly, GeoUni is the only model capable of successfully generating textual problems with matching diagrams based on specific knowledge points, thus offering a wider range of capabilities that extend beyond current models. |
| 2025-04-14 | [M2S-RoAD: Multi-Modal Semantic Segmentation for Road Damage Using Camera and LiDAR Data](http://arxiv.org/abs/2504.10123v1) | Tzu-Yun Tseng, Hongyu Lyu et al. | Road damage can create safety and comfort challenges for both human drivers and autonomous vehicles (AVs). This damage is particularly prevalent in rural areas due to less frequent surveying and maintenance of roads. Automated detection of pavement deterioration can be used as an input to AVs and driver assistance systems to improve road safety. Current research in this field has predominantly focused on urban environments driven largely by public datasets, while rural areas have received significantly less attention. This paper introduces M2S-RoAD, a dataset for the semantic segmentation of different classes of road damage. M2S-RoAD was collected in various towns across New South Wales, Australia, and labelled for semantic segmentation to identify nine distinct types of road damage. This dataset will be released upon the acceptance of the paper. |
| 2025-04-14 | [CameraBench: Benchmarking Visual Reasoning in MLLMs via Photography](http://arxiv.org/abs/2504.10090v1) | I-Sheng Fang, Jun-Cheng Chen | Large language models (LLMs) and multimodal large language models (MLLMs) have significantly advanced artificial intelligence. However, visual reasoning, reasoning involving both visual and textual inputs, remains underexplored. Recent advancements, including the reasoning models like OpenAI o1 and Gemini 2.0 Flash Thinking, which incorporate image inputs, have opened this capability. In this ongoing work, we focus specifically on photography-related tasks because a photo is a visual snapshot of the physical world where the underlying physics (i.e., illumination, blur extent, etc.) interplay with the camera parameters. Successfully reasoning from the visual information of a photo to identify these numerical camera settings requires the MLLMs to have a deeper understanding of the underlying physics for precise visual comprehension, representing a challenging and intelligent capability essential for practical applications like photography assistant agents. We aim to evaluate MLLMs on their ability to distinguish visual differences related to numerical camera settings, extending a methodology previously proposed for vision-language models (VLMs). Our preliminary results demonstrate the importance of visual reasoning in photography-related tasks. Moreover, these results show that no single MLLM consistently dominates across all evaluation tasks, demonstrating ongoing challenges and opportunities in developing MLLMs with better visual reasoning. |
| 2025-04-14 | [RealSafe-R1: Safety-Aligned DeepSeek-R1 without Compromising Reasoning Capability](http://arxiv.org/abs/2504.10081v1) | Yichi Zhang, Zihao Zeng et al. | Large Reasoning Models (LRMs), such as OpenAI o1 and DeepSeek-R1, have been rapidly progressing and achieving breakthrough performance on complex reasoning tasks such as mathematics and coding. However, the open-source R1 models have raised safety concerns in wide applications, such as the tendency to comply with malicious queries, which greatly impacts the utility of these powerful models in their applications. In this paper, we introduce RealSafe-R1 as safety-aligned versions of DeepSeek-R1 distilled models. To train these models, we construct a dataset of 15k safety-aware reasoning trajectories generated by DeepSeek-R1, under explicit instructions for expected refusal behavior. Both quantitative experiments and qualitative case studies demonstrate the models' improvements, which are shown in their safety guardrails against both harmful queries and jailbreak attacks. Importantly, unlike prior safety alignment efforts that often compromise reasoning performance, our method preserves the models' reasoning capabilities by maintaining the training data within the original distribution of generation. Model weights of RealSafe-R1 are open-source at https://huggingface.co/RealSafe. |

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



