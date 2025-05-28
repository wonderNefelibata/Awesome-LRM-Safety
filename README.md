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
| 2025-05-27 | [Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making](http://arxiv.org/abs/2505.21503v1) | Yihan Wang, Qiao Yan et al. | Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1. |
| 2025-05-27 | [Reinforcing General Reasoning without Verifiers](http://arxiv.org/abs/2505.21493v1) | Xiangxin Zhou, Zichen Liu et al. | The recent paradigm shift towards training large language models (LLMs) using DeepSeek-R1-Zero-style reinforcement learning (RL) on verifiable rewards has led to impressive advancements in code and mathematical reasoning. However, this methodology is limited to tasks where rule-based answer verification is possible and does not naturally extend to real-world domains such as chemistry, healthcare, engineering, law, biology, business, and economics. Current practical workarounds use an additional LLM as a model-based verifier; however, this introduces issues such as reliance on a strong verifier LLM, susceptibility to reward hacking, and the practical burden of maintaining the verifier model in memory during training. To address this and extend DeepSeek-R1-Zero-style training to general reasoning domains, we propose a verifier-free method (VeriFree) that bypasses answer verification and instead uses RL to directly maximize the probability of generating the reference answer. We compare VeriFree with verifier-based methods and demonstrate that, in addition to its significant practical benefits and reduced compute requirements, VeriFree matches and even surpasses verifier-based methods on extensive evaluations across MMLU-Pro, GPQA, SuperGPQA, and math-related benchmarks. Moreover, we provide insights into this method from multiple perspectives: as an elegant integration of training both the policy and implicit verifier in a unified model, and as a variational optimization approach. Code is available at https://github.com/sail-sg/VeriFree. |
| 2025-05-27 | [Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO](http://arxiv.org/abs/2505.21457v1) | Muzhi Zhu, Hao Zhong et al. | Active vision, also known as active perception, refers to the process of actively selecting where and how to look in order to gather task-relevant information. It is a critical component of efficient perception and decision-making in humans and advanced embodied agents. Recently, the use of Multimodal Large Language Models (MLLMs) as central planning and decision-making modules in robotic systems has gained extensive attention. However, despite the importance of active perception in embodied intelligence, there is little to no exploration of how MLLMs can be equipped with or learn active perception capabilities. In this paper, we first provide a systematic definition of MLLM-based active perception tasks. We point out that the recently proposed GPT-o3 model's zoom-in search strategy can be regarded as a special case of active perception; however, it still suffers from low search efficiency and inaccurate region selection. To address these issues, we propose ACTIVE-O3, a purely reinforcement learning based training framework built on top of GRPO, designed to equip MLLMs with active perception capabilities. We further establish a comprehensive benchmark suite to evaluate ACTIVE-O3 across both general open-world tasks, such as small-object and dense object grounding, and domain-specific scenarios, including small object detection in remote sensing and autonomous driving, as well as fine-grained interactive segmentation. In addition, ACTIVE-O3 also demonstrates strong zero-shot reasoning abilities on the V* Benchmark, without relying on any explicit reasoning data. We hope that our work can provide a simple codebase and evaluation protocol to facilitate future research on active perception in MLLMs. |
| 2025-05-27 | [Towards Better Instruction Following Retrieval Models](http://arxiv.org/abs/2505.21439v1) | Yuchen Zhuang, Aaron Trinh et al. | Modern information retrieval (IR) models, trained exclusively on standard <query, passage> pairs, struggle to effectively interpret and follow explicit user instructions. We introduce InF-IR, a large-scale, high-quality training corpus tailored for enhancing retrieval models in Instruction-Following IR. InF-IR expands traditional training pairs into over 38,000 expressive <instruction, query, passage> triplets as positive samples. In particular, for each positive triplet, we generate two additional hard negative examples by poisoning both instructions and queries, then rigorously validated by an advanced reasoning model (o3-mini) to ensure semantic plausibility while maintaining instructional incorrectness. Unlike existing corpora that primarily support computationally intensive reranking tasks for decoder-only language models, the highly contrastive positive-negative triplets in InF-IR further enable efficient representation learning for smaller encoder-only models, facilitating direct embedding-based retrieval. Using this corpus, we train InF-Embed, an instruction-aware Embedding model optimized through contrastive learning and instruction-query attention mechanisms to align retrieval outcomes precisely with user intents. Extensive experiments across five instruction-based retrieval benchmarks demonstrate that InF-Embed significantly surpasses competitive baselines by 8.1% in p-MRR, measuring the instruction-following capabilities. |
| 2025-05-27 | [Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery](http://arxiv.org/abs/2505.21418v1) | Lina Zhao, Jiaxing Bai et al. | Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability. We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5%, 82.5%, 87.5%, and 97.5% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains. |
| 2025-05-27 | [OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](http://arxiv.org/abs/2505.21347v1) | Ziheng Cheng, Yixiao Huang et al. | Text-to-Image (T2I) models have achieved remarkable success in generating visual content from text inputs. Although multiple safety alignment strategies have been proposed to prevent harmful outputs, they often lead to overly cautious behavior -- rejecting even benign prompts -- a phenomenon known as $\textit{over-refusal}$ that reduces the practical utility of T2I models. Despite over-refusal having been observed in practice, there is no large-scale benchmark that systematically evaluates this phenomenon for T2I models. In this paper, we present an automatic workflow to construct synthetic evaluation data, resulting in OVERT ($\textbf{OVE}$r-$\textbf{R}$efusal evaluation on $\textbf{T}$ext-to-image models), the first large-scale benchmark for assessing over-refusal behaviors in T2I models. OVERT includes 4,600 seemingly harmful but benign prompts across nine safety-related categories, along with 1,785 genuinely harmful prompts (OVERT-unsafe) to evaluate the safety-utility trade-off. Using OVERT, we evaluate several leading T2I models and find that over-refusal is a widespread issue across various categories (Figure 1), underscoring the need for further research to enhance the safety alignment of T2I models without compromising their functionality.As a preliminary attempt to reduce over-refusal, we explore prompt rewriting; however, we find it often compromises faithfulness to the meaning of the original prompts. Finally, we demonstrate the flexibility of our generation framework in accommodating diverse safety requirements by generating customized evaluation data adapting to user-defined policies. |
| 2025-05-27 | [The Multilingual Divide and Its Impact on Global AI Safety](http://arxiv.org/abs/2505.21344v1) | Aidan Peppin, Julia Kreutzer et al. | Despite advances in large language model capabilities in recent years, a large gap remains in their capabilities and safety performance for many languages beyond a relatively small handful of globally dominant languages. This paper provides researchers, policymakers and governance experts with an overview of key challenges to bridging the "language gap" in AI and minimizing safety risks across languages. We provide an analysis of why the language gap in AI exists and grows, and how it creates disparities in global AI safety. We identify barriers to address these challenges, and recommend how those working in policy and governance can help address safety concerns associated with the language gap by supporting multilingual dataset creation, transparency, and research. |
| 2025-05-27 | [Assured Autonomy with Neuro-Symbolic Perception](http://arxiv.org/abs/2505.21322v1) | R. Spencer Hallyburton, Miroslav Pajic | Many state-of-the-art AI models deployed in cyber-physical systems (CPS), while highly accurate, are simply pattern-matchers.~With limited security guarantees, there are concerns for their reliability in safety-critical and contested domains. To advance assured AI, we advocate for a paradigm shift that imbues data-driven perception models with symbolic structure, inspired by a human's ability to reason over low-level features and high-level context. We propose a neuro-symbolic paradigm for perception (NeuSPaPer) and illustrate how joint object detection and scene graph generation (SGG) yields deep scene understanding.~Powered by foundation models for offline knowledge extraction and specialized SGG algorithms for real-time deployment, we design a framework leveraging structured relational graphs that ensures the integrity of situational awareness in autonomy. Using physics-based simulators and real-world datasets, we demonstrate how SGG bridges the gap between low-level sensor perception and high-level reasoning, establishing a foundation for resilient, context-aware AI and advancing trusted autonomy in CPS. |
| 2025-05-27 | [rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset](http://arxiv.org/abs/2505.21297v1) | Yifei Liu, Li Lyna Zhang et al. | Advancing code reasoning in large language models (LLMs) is fundamentally limited by the scarcity of high-difficulty datasets, especially those with verifiable input-output test cases necessary for rigorous solution validation at scale. We introduce rStar-Coder, which significantly improves LLM code reasoning capabilities by constructing a large-scale, verified dataset of 418K competition-level code problems, 580K long-reasoning solutions along with rich test cases of varying difficulty. This is achieved through three core contributions: (1) we curate competitive programming code problems and oracle solutions to synthesize new, solvable problems; (2) we introduce a reliable input-output test case synthesis pipeline that decouples the generation into a three-step input generation method and a mutual verification mechanism for effective output labeling; (3) we augment problems with high-quality, test-case-verified long-reasoning solutions. Extensive experiments on Qwen models (1.5B-14B) across various code reasoning benchmarks demonstrate the superiority of rStar-Coder dataset, achieving leading performance comparable to frontier reasoning LLMs with much smaller model sizes. On LiveCodeBench, rStar-Coder improves Qwen2.5-7B from 17.4% to an impressive 57.3%, and Qwen2.5-14B from 23.3% to 62.5%, surpassing o3-mini (low) by3.1%. On the more challenging USA Computing Olympiad, our 7B model achieves an average pass@1 accuracy of 16.15%, outperforming the frontier-level QWQ-32B. Code and the dataset will be released at https://github.com/microsoft/rStar. |
| 2025-05-27 | [Complex System Diagnostics Using a Knowledge Graph-Informed and Large Language Model-Enhanced Framework](http://arxiv.org/abs/2505.21291v1) | Saman Marandi, Yu-Shu Hu et al. | In this paper, we present a novel diagnostic framework that integrates Knowledge Graphs (KGs) and Large Language Models (LLMs) to support system diagnostics in high-reliability systems such as nuclear power plants. Traditional diagnostic modeling struggles when systems become too complex, making functional modeling a more attractive approach. Our approach introduces a diagnostic framework grounded in the functional modeling principles of the Dynamic Master Logic (DML) model. It incorporates two coordinated LLM components, including an LLM-based workflow for automated construction of DML logic from system documentation and an LLM agent that facilitates interactive diagnostics. The generated logic is encoded into a structured KG, referred to as KG-DML, which supports hierarchical fault reasoning. Expert knowledge or operational data can also be incorporated to refine the model's precision and diagnostic depth. In the interaction phase, users submit natural language queries, which are interpreted by the LLM agent. The agent selects appropriate tools for structured reasoning, including upward and downward propagation across the KG-DML. Rather than embedding KG content into every prompt, the LLM agent distinguishes between diagnostic and interpretive tasks. For diagnostics, the agent selects and executes external tools that perform structured KG reasoning. For general queries, a Graph-based Retrieval-Augmented Generation (Graph-RAG) approach is used, retrieving relevant KG segments and embedding them into the prompt to generate natural explanations. A case study on an auxiliary feedwater system demonstrated the framework's effectiveness, with over 90% accuracy in key elements and consistent tool and argument extraction, supporting its use in safety-critical diagnostics. |
| 2025-05-27 | [Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space](http://arxiv.org/abs/2505.21277v1) | Yao Huang, Yitong Sun et al. | Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO. |
| 2025-05-27 | [A machine learning-enabled search for binary black hole mergers in LIGO-Virgo-KAGRAs third observing run](http://arxiv.org/abs/2505.21261v1) | Ethan Marx, William Benoit et al. | We conduct a search for stellar-mass binary black hole mergers in gravitational-wave data collected by the LIGO detectors during the LIGO-Virgo-KAGRA (LVK) third observing run (O3). Our search uses a machine learning (ML) based method, Aframe, an alternative to traditional matched filtering search techniques. The O3 observing run has been analyzed by the LVK collaboration, producing GWTC-3, the most recent catalog installment which has been made publicly available in 2021. Various groups outside the LVK have re-analyzed O3 data using both traditional and ML-based approaches. Here, we identify 38 candidates with probability of astrophysical origin ($p_\mathrm{astro}$) greater than 0.5, which were previously reported in GWTC-3. This is comparable to the number of candidates reported by individual matched-filter searches. In addition, we compare Aframe candidates with catalogs from research groups outside of the LVK, identifying three candidates with $p_\mathrm{astro} > 0.5$. No previously un-reported candidates are identified by Aframe. This work demonstrates that Aframe, and ML based searches more generally, are useful companions to matched filtering pipelines. |
| 2025-05-27 | [Active Learning-Enhanced Dual Control for Angle-Only Initial Relative Orbit Determination](http://arxiv.org/abs/2505.21248v1) | Kui Xie, Giovanni Romagnoli et al. | Accurate relative orbit determination is a key challenge in modern space operations, particularly when relying on angle-only measurements. The inherent observability limitations of this approach make initial state estimation difficult, impacting mission safety and performance. This work explores the use of active learning (AL) techniques to enhance observability by dynamically designing the input excitation signal offline and at runtime. Our approach leverages AL to design the input signal dynamically, enhancing the observability of the system without requiring additional hardware or predefined maneuvers. We incorporate a dual control technique to ensure target tracking while maintaining observability. The proposed method is validated through numerical simulations, demonstrating its effectiveness in estimating the initial relative state of the chaser and target spacecrafts and its robustness to various initial relative distances and observation periods. |
| 2025-05-27 | [PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing](http://arxiv.org/abs/2505.21184v1) | Yu Yan, Sheng Sun et al. | To construct responsible and secure AI applications, harmful information data is widely utilized for adversarial testing and the development of safeguards. Existing studies mainly leverage Large Language Models (LLMs) to synthesize data to obtain high-quality task datasets at scale, thereby avoiding costly human annotation. However, limited by the safety alignment mechanisms of LLMs, the synthesis of harmful data still faces challenges in generation reliability and content diversity. In this study, we propose a novel harmful information synthesis framework, PoisonSwarm, which applies the model crowdsourcing strategy to generate diverse harmful data while maintaining a high success rate. Specifically, we generate abundant benign data as the based templates in a counterfactual manner. Subsequently, we decompose each based template into multiple semantic units and perform unit-by-unit toxification and final refinement through dynamic model switching, thus ensuring the success of synthesis. Experimental results demonstrate that PoisonSwarm achieves state-of-the-art performance in synthesizing different categories of harmful data with high scalability and diversity. |
| 2025-05-27 | [Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning](http://arxiv.org/abs/2505.21178v1) | Mingyang Song, Mao Zheng | As test-time scaling becomes a pivotal research frontier in Large Language Models (LLMs) development, contemporary and advanced post-training methodologies increasingly focus on extending the generation length of long Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward DeepSeek R1-like performance. However, recent studies reveal a persistent overthinking phenomenon in state-of-the-art reasoning models, manifesting as excessive redundancy or repetitive thinking patterns in long CoT responses. To address this issue, in this paper, we propose a simple yet effective two-stage reinforcement learning framework for achieving concise reasoning in LLMs, named ConciseR. Specifically, the first stage, using more training steps, aims to incentivize the model's reasoning capabilities via Group Relative Policy Optimization with clip-higher and dynamic sampling components (GRPO++), and the second stage, using fewer training steps, explicitly enforces conciseness and improves efficiency via Length-aware Group Relative Policy Optimization (L-GRPO). Significantly, ConciseR only optimizes response length once all rollouts of a sample are correct, following the "walk before you run" principle. Extensive experimental results demonstrate that our ConciseR model, which generates more concise CoT reasoning responses, outperforms recent state-of-the-art reasoning models with zero RL paradigm across AIME 2024, MATH-500, AMC 2023, Minerva, and Olympiad benchmarks. |
| 2025-05-27 | [TAT-R1: Terminology-Aware Translation with Reinforcement Learning and Word Alignment](http://arxiv.org/abs/2505.21172v1) | Zheng Li, Mao Zheng et al. | Recently, deep reasoning large language models(LLMs) like DeepSeek-R1 have made significant progress in tasks such as mathematics and coding. Inspired by this, several studies have employed reinforcement learning(RL) to enhance models' deep reasoning capabilities and improve machine translation(MT) quality. However, the terminology translation, an essential task in MT, remains unexplored in deep reasoning LLMs. In this paper, we propose \textbf{TAT-R1}, a terminology-aware translation model trained with reinforcement learning and word alignment. Specifically, we first extract the keyword translation pairs using a word alignment model. Then we carefully design three types of rule-based alignment rewards with the extracted alignment relationships. With those alignment rewards, the RL-trained translation model can learn to focus on the accurate translation of key information, including terminology in the source text. Experimental results show the effectiveness of TAT-R1. Our model significantly improves terminology translation accuracy compared to the baseline models while maintaining comparable performance on general translation tasks. In addition, we conduct detailed ablation studies of the DeepSeek-R1-like training paradigm for machine translation and reveal several key findings. |
| 2025-05-27 | [Collision Probability Estimation for Optimization-based Vehicular Motion Planning](http://arxiv.org/abs/2505.21161v1) | Leon Tolksdorf, Arturo Tejada et al. | Many motion planning algorithms for automated driving require estimating the probability of collision (POC) to account for uncertainties in the measurement and estimation of the motion of road users. Common POC estimation techniques often utilize sampling-based methods that suffer from computational inefficiency and a non-deterministic estimation, i.e., each estimation result for the same inputs is slightly different. In contrast, optimization-based motion planning algorithms require computationally efficient POC estimation, ideally using deterministic estimation, such that typical optimization algorithms for motion planning retain feasibility. Estimating the POC analytically, however, is challenging because it depends on understanding the collision conditions (e.g., vehicle's shape) and characterizing the uncertainty in motion prediction. In this paper, we propose an approach in which we estimate the POC between two vehicles by over-approximating their shapes by a multi-circular shape approximation. The position and heading of the predicted vehicle are modelled as random variables, contrasting with the literature, where the heading angle is often neglected. We guarantee that the provided POC is an over-approximation, which is essential in providing safety guarantees, and present a computationally efficient algorithm for computing the POC estimate for Gaussian uncertainty in the position and heading. This algorithm is then used in a path-following stochastic model predictive controller (SMPC) for motion planning. With the proposed algorithm, the SMPC generates reproducible trajectories while the controller retains its feasibility in the presented test cases and demonstrates the ability to handle varying levels of uncertainty. |
| 2025-05-27 | [Thinker: Learning to Think Fast and Slow](http://arxiv.org/abs/2505.21097v1) | Stephen Chung, Wenyu Du et al. | Recent studies show that the reasoning capabilities of Large Language Models (LLMs) can be improved by applying Reinforcement Learning (RL) to question-answering (QA) tasks in areas such as math and coding. With a long context length, LLMs may learn to perform search, as indicated by the self-correction behavior observed in DeepSeek R1. However, this search behavior is often imprecise and lacks confidence, resulting in long, redundant responses and highlighting deficiencies in intuition and verification. Inspired by the Dual Process Theory in psychology, we introduce a simple modification to the QA task that includes four stages: Fast Thinking, where the LLM must answer within a strict token budget; Verification, where the model evaluates its initial response; Slow Thinking, where it refines the initial response with more deliberation; and Summarization, where it distills the refinement from the previous stage into precise steps. Our proposed task improves average accuracy from 24.9% to 27.9% for Qwen2.5-1.5B, and from 45.9% to 49.8% for DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone achieves 26.8% accuracy using fewer than 1000 tokens, demonstrating substantial inference efficiency gains. These findings suggest that intuition and deliberative reasoning are distinct, complementary systems benefiting from targeted training. |
| 2025-05-27 | [Stopping Criteria for Value Iteration on Concurrent Stochastic Reachability and Safety Games](http://arxiv.org/abs/2505.21087v1) | Marta Grobelna, Jan Křetínský et al. | We consider two-player zero-sum concurrent stochastic games (CSGs) played on graphs with reachability and safety objectives. These include degenerate classes such as Markov decision processes or turn-based stochastic games, which can be solved by linear or quadratic programming; however, in practice, value iteration (VI) outperforms the other approaches and is the most implemented method. Similarly, for CSGs, this practical performance makes VI an attractive alternative to the standard theoretical solution via the existential theory of reals.   VI starts with an under-approximation of the sought values for each state and iteratively updates them, traditionally terminating once two consecutive approximations are $\epsilon$-close. However, this stopping criterion lacks guarantees on the precision of the approximation, which is the goal of this work. We provide bounded (a.k.a. interval) VI for CSGs: it complements standard VI with a converging sequence of over-approximations and terminates once the over- and under-approximations are $\epsilon$-close. |
| 2025-05-27 | [Efficient Large Language Model Inference with Neural Block Linearization](http://arxiv.org/abs/2505.21077v1) | Mete Erdogan, Francesco Tonin et al. | The high inference demands of transformer-based Large Language Models (LLMs) pose substantial challenges in their deployment. To this end, we introduce Neural Block Linearization (NBL), a novel framework for accelerating transformer model inference by replacing self-attention layers with linear approximations derived from Linear Minimum Mean Squared Error estimators. NBL leverages Canonical Correlation Analysis to compute a theoretical upper bound on the approximation error. Then, we use this bound as a criterion for substitution, selecting the LLM layers with the lowest linearization error. NBL can be efficiently applied to pre-trained LLMs without the need for fine-tuning. In experiments, NBL achieves notable computational speed-ups while preserving competitive accuracy on multiple reasoning benchmarks. For instance, applying NBL to 12 self-attention layers in DeepSeek-R1-Distill-Llama-8B increases the inference speed by 32% with less than 1% accuracy trade-off, making it a flexible and promising solution to improve the inference efficiency of LLMs. |

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



