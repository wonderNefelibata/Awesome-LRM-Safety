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
| 2025-05-29 | [MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence](http://arxiv.org/abs/2505.23764v1) | Sihan Yang, Runsen Xu et al. | Spatial intelligence is essential for multimodal large language models (MLLMs) operating in the complex physical world. Existing benchmarks, however, probe only single-image relations and thus fail to assess the multi-image spatial reasoning that real-world deployments demand. We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours meticulously crafting 1,000 challenging, unambiguous multiple-choice questions from over 120,000 images, each paired with carefully designed distractors and a step-by-step reasoning process. We conduct extensive experiments and thoroughly evaluate 34 open-source and proprietary MLLMs, observing a wide gap: the strongest open-source model attains roughly 30% accuracy and OpenAI's o3 reasoning model reaches 40%, while humans score 97%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research. Leveraging the annotated reasoning processes, we also provide an automated error analysis pipeline that diagnoses four dominant failure modes, including (1) grounding errors, (2) overlap-matching and scene-reconstruction errors, (3) situation-transformation reasoning errors, and (4) spatial-logic errors, offering valuable insights for advancing multi-image spatial intelligence. Project page: https://runsenxu.com/projects/MMSI_Bench . |
| 2025-05-29 | [To Trust Or Not To Trust Your Vision-Language Model's Prediction](http://arxiv.org/abs/2505.23745v1) | Hao Dong, Moru Liu et al. | Vision-Language Models (VLMs) have demonstrated strong capabilities in aligning visual and textual modalities, enabling a wide range of applications in multimodal understanding and generation. While they excel in zero-shot and transfer learning scenarios, VLMs remain susceptible to misclassification, often yielding confident yet incorrect predictions. This limitation poses a significant risk in safety-critical domains, where erroneous predictions can lead to severe consequences. In this work, we introduce TrustVLM, a training-free framework designed to address the critical challenge of estimating when VLM's predictions can be trusted. Motivated by the observed modality gap in VLMs and the insight that certain concepts are more distinctly represented in the image embedding space, we propose a novel confidence-scoring function that leverages this space to improve misclassification detection. We rigorously evaluate our approach across 17 diverse datasets, employing 4 architectures and 2 VLMs, and demonstrate state-of-the-art performance, with improvements of up to 51.87% in AURC, 9.14% in AUROC, and 32.42% in FPR95 compared to existing baselines. By improving the reliability of the model without requiring retraining, TrustVLM paves the way for safer deployment of VLMs in real-world applications. The code will be available at https://github.com/EPFL-IMOS/TrustVLM. |
| 2025-05-29 | [Exposing the Impact of GenAI for Cybercrime: An Investigation into the Dark Side](http://arxiv.org/abs/2505.23733v1) | Truong, Luu et al. | In recent years, the rapid advancement and democratization of generative AI models have sparked significant debate over safety, ethical risks, and dual-use concerns, particularly in the context of cybersecurity. While anecdotally known, this paper provides empirical evidence regarding generative AI's association with malicious internet-related activities and cybercrime by examining the phenomenon through psychological frameworks of technological amplification and affordance theory. Using a quasi-experimental design with interrupted time series analysis, we analyze two datasets, one general and one cryptocurrency-focused, to empirically assess generative AI's role in cybercrime. The findings contribute to ongoing discussions about AI governance by balancing control and fostering innovation, underscoring the need for strategies to guide policymakers, inform AI developers and cybersecurity professionals, and educate the public to maximize AI's benefits while mitigating its risks. |
| 2025-05-29 | [SC-LoRA: Balancing Efficient Fine-tuning and Knowledge Preservation via Subspace-Constrained LoRA](http://arxiv.org/abs/2505.23724v1) | Minrui Luo, Fuhang Kuang et al. | Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), are indispensable for efficiently customizing Large Language Models (LLMs). However, vanilla LoRA suffers from slow convergence speed and knowledge forgetting problems. Recent studies have leveraged the power of designed LoRA initialization, to enhance the fine-tuning efficiency, or to preserve knowledge in the pre-trained LLM. However, none of these works can address the two cases at the same time. To this end, we introduce Subspace-Constrained LoRA (SC-LoRA), a novel LoRA initialization framework engineered to navigate the trade-off between efficient fine-tuning and knowledge preservation. We achieve this by constraining the output of trainable LoRA adapters in a low-rank subspace, where the context information of fine-tuning data is most preserved while the context information of preserved knowledge is least retained, in a balanced way. Such constraint enables the trainable weights to primarily focus on the main features of fine-tuning data while avoiding damaging the preserved knowledge features. We provide theoretical analysis on our method, and conduct extensive experiments including safety preservation and world knowledge preservation, on various downstream tasks. In our experiments, SC-LoRA succeeds in delivering superior fine-tuning performance while markedly diminishing knowledge forgetting, surpassing contemporary LoRA initialization methods. |
| 2025-05-29 | [ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering](http://arxiv.org/abs/2505.23723v1) | Zexi Liu, Jingyi Chai et al. | The emergence of large language model (LLM)-based agents has significantly advanced the development of autonomous machine learning (ML) engineering. However, most existing approaches rely heavily on manual prompt engineering, failing to adapt and optimize based on diverse experimental experiences. Focusing on this, for the first time, we explore the paradigm of learning-based agentic ML, where an LLM agent learns through interactive experimentation on ML tasks using online reinforcement learning (RL). To realize this, we propose a novel agentic ML training framework with three key components: (1) exploration-enriched fine-tuning, which enables LLM agents to generate diverse actions for enhanced RL exploration; (2) step-wise RL, which enables training on a single action step, accelerating experience collection and improving training efficiency; (3) an agentic ML-specific reward module, which unifies varied ML feedback signals into consistent rewards for RL optimization. Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM for autonomous ML. Remarkably, despite being trained on merely 9 ML tasks, our 7B-sized ML-Agent outperforms the 671B-sized DeepSeek-R1 agent. Furthermore, it achieves continuous performance improvements and demonstrates exceptional cross-task generalization capabilities. |
| 2025-05-29 | [Distributed Federated Learning for Vehicular Network Security: Anomaly Detection Benefits and Multi-Domain Attack Threats](http://arxiv.org/abs/2505.23706v1) | Utku Demir, Yalin E. Sagduyu et al. | In connected and autonomous vehicles, machine learning for safety message classification has become critical for detecting malicious or anomalous behavior. However, conventional approaches that rely on centralized data collection or purely local training face limitations due to the large scale, high mobility, and heterogeneous data distributions inherent in inter-vehicle networks. To overcome these challenges, this paper explores Distributed Federated Learning (DFL), whereby vehicles collaboratively train deep learning models by exchanging model updates among one-hop neighbors and propagating models over multiple hops. Using the Vehicular Reference Misbehavior (VeReMi) Extension Dataset, we show that DFL can significantly improve classification accuracy across all vehicles compared to learning strictly with local data. Notably, vehicles with low individual accuracy see substantial accuracy gains through DFL, illustrating the benefit of knowledge sharing across the network. We further show that local training data size and time-varying network connectivity correlate strongly with the model's overall accuracy. We investigate DFL's resilience and vulnerabilities under attacks in multiple domains, namely wireless jamming and training data poisoning attacks. Our results reveal important insights into the vulnerabilities of DFL when confronted with multi-domain attacks, underlining the need for more robust strategies to secure DFL in vehicular networks. |
| 2025-05-29 | [Fortune: Formula-Driven Reinforcement Learning for Symbolic Table Reasoning in Language Models](http://arxiv.org/abs/2505.23667v1) | Lang Cao, Jingxian Xu et al. | Tables are a fundamental structure for organizing and analyzing data, making effective table understanding a critical capability for intelligent systems. While large language models (LMs) demonstrate strong general reasoning abilities, they continue to struggle with accurate numerical or symbolic reasoning over tabular data, especially in complex scenarios. Spreadsheet formulas provide a powerful and expressive medium for representing executable symbolic operations, encoding rich reasoning patterns that remain largely underutilized. In this paper, we propose Formula Tuning (Fortune), a reinforcement learning (RL) framework that trains LMs to generate executable spreadsheet formulas for question answering over general tabular data. Formula Tuning reduces the reliance on supervised formula annotations by using binary answer correctness as a reward signal, guiding the model to learn formula derivation through reasoning. We provide a theoretical analysis of its advantages and demonstrate its effectiveness through extensive experiments on seven table reasoning benchmarks. Formula Tuning substantially enhances LM performance, particularly on multi-step numerical and symbolic reasoning tasks, enabling a 7B model to outperform O1 on table understanding. This highlights the potential of formula-driven RL to advance symbolic table reasoning in LMs. |
| 2025-05-29 | [Are Reasoning Models More Prone to Hallucination?](http://arxiv.org/abs/2505.23646v1) | Zijun Yao, Yantao Liu et al. | Recently evolved large reasoning models (LRMs) show powerful performance in solving complex tasks with long chain-of-thought (CoT) reasoning capability. As these LRMs are mostly developed by post-training on formal reasoning tasks, whether they generalize the reasoning capability to help reduce hallucination in fact-seeking tasks remains unclear and debated. For instance, DeepSeek-R1 reports increased performance on SimpleQA, a fact-seeking benchmark, while OpenAI-o3 observes even severer hallucination. This discrepancy naturally raises the following research question: Are reasoning models more prone to hallucination? This paper addresses the question from three perspectives. (1) We first conduct a holistic evaluation for the hallucination in LRMs. Our analysis reveals that LRMs undergo a full post-training pipeline with cold start supervised fine-tuning (SFT) and verifiable reward RL generally alleviate their hallucination. In contrast, both distillation alone and RL training without cold start fine-tuning introduce more nuanced hallucinations. (2) To explore why different post-training pipelines alters the impact on hallucination in LRMs, we conduct behavior analysis. We characterize two critical cognitive behaviors that directly affect the factuality of a LRM: Flaw Repetition, where the surface-level reasoning attempts repeatedly follow the same underlying flawed logic, and Think-Answer Mismatch, where the final answer fails to faithfully match the previous CoT process. (3) Further, we investigate the mechanism behind the hallucination of LRMs from the perspective of model uncertainty. We find that increased hallucination of LRMs is usually associated with the misalignment between model uncertainty and factual accuracy. Our work provides an initial understanding of the hallucination in LRMs. |
| 2025-05-29 | [MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment](http://arxiv.org/abs/2505.23634v1) | John Halloran | The model context protocol (MCP) has been widely adapted as an open standard enabling the seamless integration of generative AI agents. However, recent work has shown the MCP is susceptible to retrieval-based "falsely benign" attacks (FBAs), allowing malicious system access and credential theft, but requiring that users download compromised files directly to their systems. Herein, we show that the threat model of MCP-based attacks is significantly broader than previously thought, i.e., attackers need only post malicious content online to deceive MCP agents into carrying out their attacks on unsuspecting victims' systems.   To improve alignment guardrails against such attacks, we introduce a new MCP dataset of FBAs and (truly) benign samples to explore the effectiveness of direct preference optimization (DPO) for the refusal training of large language models (LLMs). While DPO improves model guardrails against such attacks, we show that the efficacy of refusal learning varies drastically depending on the model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to refuse extremely poorly. Thus, to further improve FBA refusals, we introduce Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel preference alignment strategy based on RAG. We show that RAG-Pref significantly improves the ability of LLMs to refuse FBAs, particularly when combined with DPO alignment, thus drastically improving guardrails against MCP-based attacks. |
| 2025-05-29 | [Table-R1: Inference-Time Scaling for Table Reasoning](http://arxiv.org/abs/2505.23621v1) | Zheyuan Yang, Lyuhao Chen et al. | In this work, we present the first study to explore inference-time scaling on table reasoning tasks. We develop and evaluate two post-training strategies to enable inference-time scaling: distillation from frontier model reasoning traces and reinforcement learning with verifiable rewards (RLVR). For distillation, we introduce a large-scale dataset of reasoning traces generated by DeepSeek-R1, which we use to fine-tune LLMs into the Table-R1-SFT model. For RLVR, we propose task-specific verifiable reward functions and apply the GRPO algorithm to obtain the Table-R1-Zero model. We evaluate our Table-R1-series models across diverse table reasoning tasks, including short-form QA, fact verification, and free-form QA. Notably, the Table-R1-Zero model matches or exceeds the performance of GPT-4.1 and DeepSeek-R1, while using only a 7B-parameter LLM. It also demonstrates strong generalization to out-of-domain datasets. Extensive ablation and qualitative analyses reveal the benefits of instruction tuning, model architecture choices, and cross-task generalization, as well as emergence of essential table reasoning skills during RL training. |
| 2025-05-29 | [LLM Performance for Code Generation on Noisy Tasks](http://arxiv.org/abs/2505.23598v1) | Radzim Sendyka, Christian Cabrera et al. | This paper investigates the ability of large language models (LLMs) to recognise and solve tasks which have been obfuscated beyond recognition. Focusing on competitive programming and benchmark tasks (LeetCode and MATH), we compare performance across multiple models and obfuscation methods, such as noise and redaction. We demonstrate that all evaluated LLMs can solve tasks obfuscated to a level where the text would be unintelligible to human readers, and does not contain key pieces of instruction or context. We introduce the concept of eager pattern matching to describe this behaviour, which is not observed in tasks published after the models' knowledge cutoff date, indicating strong memorisation or overfitting to training data, rather than legitimate reasoning about the presented problem. We report empirical evidence of distinct performance decay patterns between contaminated and unseen datasets. We discuss the implications for benchmarking and evaluations of model behaviour, arguing for caution when designing experiments using standard datasets. We also propose measuring the decay of performance under obfuscation as a possible strategy for detecting dataset contamination and highlighting potential safety risks and interpretability issues for automated software systems. |
| 2025-05-29 | [SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents](http://arxiv.org/abs/2505.23559v1) | Kunlun Zhu, Jiaxun Zhang et al. | Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.} |
| 2025-05-29 | [Understanding Refusal in Language Models with Sparse Autoencoders](http://arxiv.org/abs/2505.23556v1) | Wei Jie Yeo, Nirmalendu Prakash et al. | Refusal is a key safety behavior in aligned language models, yet the internal mechanisms driving refusals remain opaque. In this work, we conduct a mechanistic study of refusal in instruction-tuned LLMs using sparse autoencoders to identify latent features that causally mediate refusal behaviors. We apply our method to two open-source chat models and intervene on refusal-related features to assess their influence on generation, validating their behavioral impact across multiple harmful datasets. This enables a fine-grained inspection of how refusal manifests at the activation level and addresses key research questions such as investigating upstream-downstream latent relationship and understanding the mechanisms of adversarial jailbreaking techniques. We also establish the usefulness of refusal features in enhancing generalization for linear probes to out-of-distribution adversarial samples in classification tasks. We open source our code in https://github.com/wj210/refusal_sae. |
| 2025-05-29 | [LLM-based Property-based Test Generation for Guardrailing Cyber-Physical Systems](http://arxiv.org/abs/2505.23549v1) | Khashayar Etemadi, Marjan Sirjani et al. | Cyber-physical systems (CPSs) are complex systems that integrate physical, computational, and communication subsystems. The heterogeneous nature of these systems makes their safety assurance challenging. In this paper, we propose a novel automated approach for guardrailing cyber-physical systems using property-based tests (PBTs) generated by Large Language Models (LLMs). Our approach employs an LLM to extract properties from the code and documentation of CPSs. Next, we use the LLM to generate PBTs that verify the extracted properties on the CPS. The generated PBTs have two uses. First, they are used to test the CPS before it is deployed, i.e., at design time. Secondly, these PBTs can be used after deployment, i.e., at run time, to monitor the behavior of the system and guardrail it against unsafe states. We implement our approach in ChekProp and conduct preliminary experiments to evaluate the generated PBTs in terms of their relevance (how well they match manually crafted properties), executability (how many run with minimal manual modification), and effectiveness (coverage of the input space partitions). The results of our experiments and evaluation demonstrate a promising path forward for creating guardrails for CPSs using LLM-generated property-based tests. |
| 2025-05-29 | [EVOREFUSE: Evolutionary Prompt Optimization for Evaluation and Mitigation of LLM Over-Refusal to Pseudo-Malicious Instructions](http://arxiv.org/abs/2505.23473v1) | Xiaorui Wu, Xiaofeng Mao et al. | Large language models (LLMs) frequently refuse to respond to pseudo-malicious instructions: semantically harmless input queries triggering unnecessary LLM refusals due to conservative safety alignment, significantly impairing user experience. Collecting such instructions is crucial for evaluating and mitigating over-refusals, but existing instruction curation methods, like manual creation or instruction rewriting, either lack scalability or fail to produce sufficiently diverse and effective refusal-inducing prompts. To address these limitations, we introduce EVOREFUSE, a prompt optimization approach that generates diverse pseudo-malicious instructions consistently eliciting confident refusals across LLMs. EVOREFUSE employs an evolutionary algorithm exploring the instruction space in more diverse directions than existing methods via mutation strategies and recombination, and iteratively evolves seed instructions to maximize evidence lower bound on LLM refusal probability. Using EVOREFUSE, we create two novel datasets: EVOREFUSE-TEST, a benchmark of 582 pseudo-malicious instructions that outperforms the next-best benchmark with 140.41% higher average refusal triggering rate across 9 LLMs, 34.86% greater lexical diversity, and 40.03% improved LLM response confidence scores; and EVOREFUSE-ALIGN, which provides 3,000 pseudo-malicious instructions with responses for supervised and preference-based alignment training. LLAMA3.1-8B-INSTRUCT supervisedly fine-tuned on EVOREFUSE-ALIGN achieves up to 14.31% fewer over-refusals than models trained on the second-best alignment dataset, without compromising safety. Our analysis with EVOREFUSE-TEST reveals models trigger over-refusals by overly focusing on sensitive keywords while ignoring broader context. |
| 2025-05-29 | [Self-driving technologies need the help of the public: A narrative review of the evidence](http://arxiv.org/abs/2505.23472v1) | Jonathan Smith, Siddartha Khastgir | If public trust is lot in a new technology early in its life cycle it can take much more time for the benefits of that technology to be realised. Eventually tens-of-millions of people will collectively have the power to determine self-driving technology success of failure driven by their perception of risk, data handling, safety, governance, accountability, benefits to their life and more. This paper reviews the evidence on safety critical technology covering trust, engagement, and acceptance. The paper takes a narrative review approach concluding with a scalable model for self-driving technology education and engagement. The paper find that if a mismatch between the publics perception and expectations about self driving systems emerge it can lead to misuse, disuse, or abuse of the system. Furthermore we find from the evidence that industrial experts often misunderstand what matters to the public, users, and stakeholders. However we find that engagement programmes that develop approaches to defining the right information at the right time, in the right format orientated around what matters to the public creates the potential for ever more sophisticated conversations, greater trust, and moving the public into a progressive more active role of critique and advocacy. This work has been undertaken as part of the Partners for Automated Vehicle Education (PAVE) United Kingdom programme. |
| 2025-05-29 | [Bounded-Abstention Pairwise Learning to Rank](http://arxiv.org/abs/2505.23437v1) | Antonio Ferrara, Andrea Pugnana et al. | Ranking systems influence decision-making in high-stakes domains like health, education, and employment, where they can have substantial economic and social impacts. This makes the integration of safety mechanisms essential. One such mechanism is $\textit{abstention}$, which enables algorithmic decision-making system to defer uncertain or low-confidence decisions to human experts. While abstention have been predominantly explored in the context of classification tasks, its application to other machine learning paradigms remains underexplored. In this paper, we introduce a novel method for abstention in pairwise learning-to-rank tasks. Our approach is based on thresholding the ranker's conditional risk: the system abstains from making a decision when the estimated risk exceeds a predefined threshold. Our contributions are threefold: a theoretical characterization of the optimal abstention strategy, a model-agnostic, plug-in algorithm for constructing abstaining ranking models, and a comprehensive empirical evaluations across multiple datasets, demonstrating the effectiveness of our approach. |
| 2025-05-29 | [Diversity-Aware Policy Optimization for Large Language Model Reasoning](http://arxiv.org/abs/2505.23433v1) | Jian Yao, Ran Cheng et al. | The reasoning capabilities of large language models (LLMs) have advanced rapidly, particularly following the release of DeepSeek R1, which has inspired a surge of research into data quality and reinforcement learning (RL) algorithms. Despite the pivotal role diversity plays in RL, its influence on LLM reasoning remains largely underexplored. To bridge this gap, this work presents a systematic investigation into the impact of diversity in RL-based training for LLM reasoning, and proposes a novel diversity-aware policy optimization method. Across evaluations on 12 LLMs, we observe a strong positive correlation between the solution diversity and Potential at k (a novel metric quantifying an LLM's reasoning potential) in high-performing models. This finding motivates our method to explicitly promote diversity during RL training. Specifically, we design a token-level diversity and reformulate it into a practical objective, then we selectively apply it to positive samples. Integrated into the R1-zero training framework, our method achieves a 3.5 percent average improvement across four mathematical reasoning benchmarks, while generating more diverse and robust solutions. |
| 2025-05-29 | [Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models](http://arxiv.org/abs/2505.23404v1) | Mingyu Yu, Wei Wang et al. | Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release) |
| 2025-05-29 | [CF-DETR: Coarse-to-Fine Transformer for Real-Time Object Detection](http://arxiv.org/abs/2505.23317v1) | Woojin Shin, Donghwa Kang et al. | Detection Transformers (DETR) are increasingly adopted in autonomous vehicle (AV) perception systems due to their superior accuracy over convolutional networks. However, concurrently executing multiple DETR tasks presents significant challenges in meeting firm real-time deadlines (R1) and high accuracy requirements (R2), particularly for safety-critical objects, while navigating the inherent latency-accuracy trade-off under resource constraints. Existing real-time DNN scheduling approaches often treat models generically, failing to leverage Transformer-specific properties for efficient resource allocation. To address these challenges, we propose CF-DETR, an integrated system featuring a novel coarse-to-fine Transformer architecture and a dedicated real-time scheduling framework NPFP**. CF-DETR employs three key strategies (A1: coarse-to-fine inference, A2: selective fine inference, A3: multi-level batch inference) that exploit Transformer properties to dynamically adjust patch granularity and attention scope based on object criticality, aiming to satisfy R2. The NPFP** scheduling framework (A4) orchestrates these adaptive mechanisms A1-A3. It partitions each DETR task into a safety-critical coarse subtask for guaranteed critical object detection within its deadline (ensuring R1), and an optional fine subtask for enhanced overall accuracy (R2), while managing individual and batched execution. Our extensive evaluations on server, GPU-enabled embedded platforms, and actual AV platforms demonstrate that CF-DETR, under an NPFP** policy, successfully meets strict timing guarantees for critical operations and achieves significantly higher overall and critical object detection accuracy compared to existing baselines across diverse AV workloads. |

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



