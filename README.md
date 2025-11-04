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
| 2025-10-31 | [Soft Gravitons, Hard Truths: Infrared Safety of Particle Processes in a Gravitational-Wave Background](http://arxiv.org/abs/2510.27690v1) | Wen-Yuan Ai, Sebastian A. R. Ellis et al. | Gravitational waves are thought to propagate unattenuated through matter due to a cancellation between graviton absorption and stimulated emission inferred from leading-order soft-graviton arguments. We revisit this reasoning and show that it fails for the converse problem: the effect of a gravitational-wave background on matter. For unstable particles, real graviton emission \emph{and} absorption appear to enhance decay rates. By extending the soft-graviton framework describing real and virtual processes in a gravitational wave background, and resumming them to all orders, we show that inclusive decay rates remain essentially unchanged. The mutual transparency between matter and gravitational radiation thus follows from infrared safety, and not from a fortuitous cancellation in the lowest-order approximation of exclusive rates. |
| 2025-10-31 | [Teaching competencies in physics for engineering education: A qualitative analysis from teaching practice](http://arxiv.org/abs/2510.27674v1) | Vanessa Cruz Molina, Daniel Sanchez Guzman et al. | Physics teaching in engineering programmes poses discipline-specific demands that intertwine conceptual modelling, experimental inquiry, and computational analysis. This study examines nine teaching competences for physics instruction derived from international and regional frameworks and interpreted within engineering contexts. Nineteen university instructors from the Technological Institute of Toluca completed an open-ended questionnaire; responses were analysed using a grounded theory approach (open and axial coding) complemented by descriptive frequencies. Results indicate stronger development in technical mastery, methodological/digital integration, technology-mediated communication, and innovation (C1, C2, C6, C9), while information literacy for digital content creation/adaptation and digital ethics/safety (C7, C8) remain underdeveloped. A recurrent understanding-application gap was identified, revealing uneven transfer from conceptual awareness to enacted classroom practice. We conclude that advancing physics education for engineers requires institutionally supported, discipline-specific professional development that aligns modelling, laboratory work, and computation with ethical and reproducible digital practices; such alignment can move instructors from adoption/adaptation toward sustained appropriation and innovation in multimodal settings. |
| 2025-10-31 | [Culture Cartography: Mapping the Landscape of Cultural Knowledge](http://arxiv.org/abs/2510.27672v1) | Caleb Ziems, William Held et al. | To serve global users safely and productively, LLMs need culture-specific knowledge that might not be learned during pre-training. How do we find such knowledge that is (1) salient to in-group users, but (2) unknown to LLMs? The most common solutions are single-initiative: either researchers define challenging questions that users passively answer (traditional annotation), or users actively produce data that researchers structure as benchmarks (knowledge extraction). The process would benefit from mixed-initiative collaboration, where users guide the process to meaningfully reflect their cultures, and LLMs steer the process towards more challenging questions that meet the researcher's goals. We propose a mixed-initiative methodology called CultureCartography. Here, an LLM initializes annotation with questions for which it has low-confidence answers, making explicit both its prior knowledge and the gaps therein. This allows a human respondent to fill these gaps and steer the model towards salient topics through direct edits. We implement this methodology as a tool called CultureExplorer. Compared to a baseline where humans answer LLM-proposed questions, we find that CultureExplorer more effectively produces knowledge that leading models like DeepSeek R1 and GPT-4o are missing, even with web search. Fine-tuning on this data boosts the accuracy of Llama-3.1-8B by up to 19.2% on related culture benchmarks. |
| 2025-10-31 | [Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models](http://arxiv.org/abs/2510.27629v1) | Boyi Wei, Zora Che et al. | Open-weight bio-foundation models present a dual-use dilemma. While holding great promise for accelerating scientific research and drug development, they could also enable bad actors to develop more deadly bioweapons. To mitigate the risk posed by these models, current approaches focus on filtering biohazardous data during pre-training. However, the effectiveness of such an approach remains unclear, particularly against determined actors who might fine-tune these models for malicious use. To address this gap, we propose \eval, a framework to evaluate the robustness of procedures that are intended to reduce the dual-use capabilities of bio-foundation models. \eval assesses models' virus understanding through three lenses, including sequence modeling, mutational effects prediction, and virulence prediction. Our results show that current filtering practices may not be particularly effective: Excluded knowledge can be rapidly recovered in some cases via fine-tuning, and exhibits broader generalizability in sequence modeling. Furthermore, dual-use signals may already reside in the pretrained representations, and can be elicited via simple linear probing. These findings highlight the challenges of data filtering as a standalone procedure, underscoring the need for further research into robust safety and security strategies for open-weight bio-foundation models. |
| 2025-10-31 | [Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models](http://arxiv.org/abs/2510.27629v2) | Boyi Wei, Zora Che et al. | Open-weight bio-foundation models present a dual-use dilemma. While holding great promise for accelerating scientific research and drug development, they could also enable bad actors to develop more deadly bioweapons. To mitigate the risk posed by these models, current approaches focus on filtering biohazardous data during pre-training. However, the effectiveness of such an approach remains unclear, particularly against determined actors who might fine-tune these models for malicious use. To address this gap, we propose \eval, a framework to evaluate the robustness of procedures that are intended to reduce the dual-use capabilities of bio-foundation models. \eval assesses models' virus understanding through three lenses, including sequence modeling, mutational effects prediction, and virulence prediction. Our results show that current filtering practices may not be particularly effective: Excluded knowledge can be rapidly recovered in some cases via fine-tuning, and exhibits broader generalizability in sequence modeling. Furthermore, dual-use signals may already reside in the pretrained representations, and can be elicited via simple linear probing. These findings highlight the challenges of data filtering as a standalone procedure, underscoring the need for further research into robust safety and security strategies for open-weight bio-foundation models. |
| 2025-10-31 | [Trends and Challenges in Next-Generation GNSS Interference Management](http://arxiv.org/abs/2510.27576v1) | Leatile Marata, Mariona Jaramillo-Civill et al. | The global navigation satellite system (GNSS) continues to evolve in order to meet the demands of emerging applications such as autonomous driving and smart environmental monitoring. However, these advancements are accompanied by a rise in interference threats, which can significantly compromise the reliability and safety of GNSS. Such interference problems are typically addressed through signal-processing techniques that rely on physics-based mathematical models. Unfortunately, solutions of this nature can often fail to fully capture the complex forms of interference. To address this, artificial intelligence (AI)-inspired solutions are expected to play a key role in future interference management solutions, thanks to their ability to exploit data in addition to physics-based models. This magazine paper discusses the main challenges and tasks required to secure GNSS and present a research vision on how AI can be leveraged towards achieving more robust GNSS-based positioning. |
| 2025-10-31 | [Learning Soft Robotic Dynamics with Active Exploration](http://arxiv.org/abs/2510.27428v1) | Hehui Zheng, Bhavya Sukhija et al. | Soft robots offer unmatched adaptability and safety in unstructured environments, yet their compliant, high-dimensional, and nonlinear dynamics make modeling for control notoriously difficult. Existing data-driven approaches often fail to generalize, constrained by narrowly focused task demonstrations or inefficient random exploration. We introduce SoftAE, an uncertainty-aware active exploration framework that autonomously learns task-agnostic and generalizable dynamics models of soft robotic systems. SoftAE employs probabilistic ensemble models to estimate epistemic uncertainty and actively guides exploration toward underrepresented regions of the state-action space, achieving efficient coverage of diverse behaviors without task-specific supervision. We evaluate SoftAE on three simulated soft robotic platforms -- a continuum arm, an articulated fish in fluid, and a musculoskeletal leg with hybrid actuation -- and on a pneumatically actuated continuum soft arm in the real world. Compared with random exploration and task-specific model-based reinforcement learning, SoftAE produces more accurate dynamics models, enables superior zero-shot control on unseen tasks, and maintains robustness under sensing noise, actuation delays, and nonlinear material effects. These results demonstrate that uncertainty-driven active exploration can yield scalable, reusable dynamics models across diverse soft robotic morphologies, representing a step toward more autonomous, adaptable, and data-efficient control in compliant robots. |
| 2025-10-31 | [Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity](http://arxiv.org/abs/2510.27378v1) | Austin Meek, Eitan Sprejer et al. | Chain-of-thought (CoT) outputs let us read a model's step-by-step reasoning. Since any long, serial reasoning process must pass through this textual trace, the quality of the CoT is a direct window into what the model is thinking. This visibility could help us spot unsafe or misaligned behavior (monitorability), but only if the CoT is transparent about its internal reasoning (faithfulness). Fully measuring faithfulness is difficult, so researchers often focus on examining the CoT in cases where the model changes its answer after adding a cue to the input. This proxy finds some instances of unfaithfulness but loses information when the model maintains its answer, and does not investigate aspects of reasoning not tied to the cue. We extend these results to a more holistic sense of monitorability by introducing verbosity: whether the CoT lists every factor needed to solve the task. We combine faithfulness and verbosity into a single monitorability score that shows how well the CoT serves as the model's external `working memory', a property that many safety schemes based on CoT monitoring depend on. We evaluate instruction-tuned and reasoning models on BBH, GPQA, and MMLU. Our results show that models can appear faithful yet remain hard to monitor when they leave out key factors, and that monitorability differs sharply across model families. We release our evaluation code using the Inspect library to support reproducible future work. |
| 2025-10-31 | [Modified-Emergency Index (MEI): A Criticality Metric for Autonomous Driving in Lateral Conflict](http://arxiv.org/abs/2510.27333v1) | Hao Cheng, Yanbo Jiang et al. | Effective, reliable, and efficient evaluation of autonomous driving safety is essential to demonstrate its trustworthiness. Criticality metrics provide an objective means of assessing safety. However, as existing metrics primarily target longitudinal conflicts, accurately quantifying the risks of lateral conflicts - prevalent in urban settings - remains challenging. This paper proposes the Modified-Emergency Index (MEI), a metric designed to quantify evasive effort in lateral conflicts. Compared to the original Emergency Index (EI), MEI refines the estimation of the time available for evasive maneuvers, enabling more precise risk quantification. We validate MEI on a public lateral conflict dataset based on Argoverse-2, from which we extract over 1,500 high-quality AV conflict cases, including more than 500 critical events. MEI is then compared with the well-established ACT and the widely used PET metrics. Results show that MEI consistently outperforms them in accurately quantifying criticality and capturing risk evolution. Overall, these findings highlight MEI as a promising metric for evaluating urban conflicts and enhancing the safety assessment framework for autonomous driving. The open-source implementation is available at https://github.com/AutoChengh/MEI. |
| 2025-10-31 | [Prevalence of Security and Privacy Risk-Inducing Usage of AI-based Conversational Agents](http://arxiv.org/abs/2510.27275v1) | Kathrin Grosse, Nico Ebert | Recent improvement gains in large language models (LLMs) have lead to everyday usage of AI-based Conversational Agents (CAs). At the same time, LLMs are vulnerable to an array of threats, including jailbreaks and, for example, causing remote code execution when fed specific inputs. As a result, users may unintentionally introduce risks, for example, by uploading malicious files or disclosing sensitive information. However, the extent to which such user behaviors occur and thus potentially facilitate exploits remains largely unclear. To shed light on this issue, we surveyed a representative sample of 3,270 UK adults in 2024 using Prolific. A third of these use CA services such as ChatGPT or Gemini at least once a week. Of these ``regular users'', up to a third exhibited behaviors that may enable attacks, and a fourth have tried jailbreaking (often out of understandable reasons such as curiosity, fun or information seeking). Half state that they sanitize data and most participants report not sharing sensitive data. However, few share very sensitive data such as passwords. The majority are unaware that their data can be used to train models and that they can opt-out. Our findings suggest that current academic threat models manifest in the wild, and mitigations or guidelines for the secure usage of CAs should be developed. In areas critical to security and privacy, CAs must be equipped with effective AI guardrails to prevent, for example, revealing sensitive information to curious employees. Vendors need to increase efforts to prevent the entry of sensitive data, and to create transparency with regard to data usage policies and settings. |
| 2025-10-31 | [MDAS-GNN: Multi-Dimensional Spatiotemporal GNN with Spatial Diffusion for Urban Traffic Risk Forecasting](http://arxiv.org/abs/2510.27197v1) | Ziyuan Gao | Traffic accidents represent a critical public health challenge, claiming over 1.35 million lives annually worldwide. Traditional accident prediction models treat road segments independently, failing to capture complex spatial relationships and temporal dependencies in urban transportation networks. This study develops MDAS-GNN, a Multi-Dimensional Attention-based Spatial-diffusion Graph Neural Network integrating three core risk dimensions: traffic safety, infrastructure, and environmental risk. The framework employs feature-specific spatial diffusion mechanisms and multi-head temporal attention to capture dependencies across different time horizons. Evaluated on UK Department for Transport accident data across Central London, South Manchester, and SE Birmingham, MDASGNN achieves superior performance compared to established baseline methods. The model maintains consistently low prediction errors across short, medium, and long-term periods, with particular strength in long-term forecasting. Ablation studies confirm that integrated multi-dimensional features outperform singlefeature approaches, reducing prediction errors by up to 40%. This framework provides civil engineers and urban planners with advanced predictive capabilities for transportation infrastructure design, enabling data-driven decisions for road network optimization, infrastructure resource improvements, and strategic safety interventions in urban development projects. |
| 2025-10-31 | [Adaptive Defense against Harmful Fine-Tuning for Large Language Models via Bayesian Data Scheduler](http://arxiv.org/abs/2510.27172v1) | Zixuan Hu, Li Shen et al. | Harmful fine-tuning poses critical safety risks to fine-tuning-as-a-service for large language models. Existing defense strategies preemptively build robustness via attack simulation but suffer from fundamental limitations: (i) the infeasibility of extending attack simulations beyond bounded threat models due to the inherent difficulty of anticipating unknown attacks, and (ii) limited adaptability to varying attack settings, as simulation fails to capture their variability and complexity. To address these challenges, we propose Bayesian Data Scheduler (BDS), an adaptive tuning-stage defense strategy with no need for attack simulation. BDS formulates harmful fine-tuning defense as a Bayesian inference problem, learning the posterior distribution of each data point's safety attribute, conditioned on the fine-tuning and alignment datasets. The fine-tuning process is then constrained by weighting data with their safety attributes sampled from the posterior, thus mitigating the influence of harmful data. By leveraging the post hoc nature of Bayesian inference, the posterior is conditioned on the fine-tuning dataset, enabling BDS to tailor its defense to the specific dataset, thereby achieving adaptive defense. Furthermore, we introduce a neural scheduler based on amortized Bayesian learning, enabling efficient transfer to new data without retraining. Comprehensive results across diverse attack and defense settings demonstrate the state-of-the-art performance of our approach. Code is available at https://github.com/Egg-Hu/Bayesian-Data-Scheduler. |
| 2025-10-31 | [A Hierarchical Deep Learning Model for Predicting Pedestrian-Level Urban Winds](http://arxiv.org/abs/2510.27101v1) | Reda Snaiki, Jiachen Lu et al. | Deep learning-based surrogate models offer a computationally efficient alternative to high-fidelity computational fluid dynamics (CFD) simulations for predicting urban wind flow. However, conventional approaches usually only yield low-frequency predictions (essentially averaging values from proximate pixels), missing critical high-frequency details such as sharp gradients and peak wind speeds. This study proposes a hierarchical approach for accurately predicting pedestrian-level urban winds, which adopts a two-stage predictor-refiner framework. In the first stage, a U-Net architecture generates a baseline prediction from urban geometry. In the second stage, a conditional Generative Adversarial Network (cGAN) refines this baseline by restoring the missing high-frequency content. The cGAN's generator incorporates a multi-scale architecture with stepwise kernel sizes, enabling simultaneous learning of global flow structures and fine-grained local features. Trained and validated on the UrbanTALES dataset with comprehensive urban configurations, the proposed hierarchical framework significantly outperforms the baseline predictor. With a marked qualitative improvement in resolving high-speed wind jets and complex turbulent wakes as well as wind statistics, the results yield quantitative enhancement in prediction accuracy (e.g., RMSE reduced by 76% for the training set and 60% for the validation set). This work presents an effective and robust methodology for enhancing the prediction fidelity of surrogate models in urban planning, pedestrian comfort assessment, and wind safety analysis. The proposed model will be integrated into an interactive web platform as Feilian Version 2. |
| 2025-10-31 | [Characterizing Selective Refusal Bias in Large Language Models](http://arxiv.org/abs/2510.27087v1) | Adel Khorramrouz, Sharon Levy | Safety guardrails in large language models(LLMs) are developed to prevent malicious users from generating toxic content at a large scale. However, these measures can inadvertently introduce or reflect new biases, as LLMs may refuse to generate harmful content targeting some demographic groups and not others. We explore this selective refusal bias in LLM guardrails through the lens of refusal rates of targeted individual and intersectional demographic groups, types of LLM responses, and length of generated refusals. Our results show evidence of selective refusal bias across gender, sexual orientation, nationality, and religion attributes. This leads us to investigate additional safety implications via an indirect attack, where we target previously refused groups. Our findings emphasize the need for more equitable and robust performance in safety guardrails across demographic groups. |
| 2025-10-31 | [Contrastive Knowledge Transfer and Robust Optimization for Secure Alignment of Large Language Models](http://arxiv.org/abs/2510.27077v1) | Jiasen Zheng, Huajun Zhang et al. | This paper addresses the limitations of large-scale language models in safety alignment and robustness by proposing a fine-tuning method that combines contrastive distillation with noise-robust training. The method freezes the backbone model and transfers the knowledge boundaries of the teacher model to the student model through distillation, thereby improving semantic consistency and alignment accuracy. At the same time, noise perturbations and robust optimization constraints are introduced during training to ensure that the model maintains stable predictive outputs under noisy and uncertain inputs. The overall framework consists of distillation loss, robustness loss, and a regularization term, forming a unified optimization objective that balances alignment ability with resistance to interference. To systematically validate its effectiveness, the study designs experiments from multiple perspectives, including distillation weight sensitivity, stability analysis under computation budgets and mixed-precision environments, and the impact of data noise and distribution shifts on model performance. Results show that the method significantly outperforms existing baselines in knowledge transfer, robustness, and overall safety, achieving the best performance across several key metrics. This work not only enriches the theoretical system of parameter-efficient fine-tuning but also provides a new solution for building safer and more trustworthy alignment mechanisms. |
| 2025-10-31 | [MLPerf Automotive](http://arxiv.org/abs/2510.27065v1) | Radoyeh Shojaei, Predrag Djurdjevic et al. | We present MLPerf Automotive, the first standardized public benchmark for evaluating Machine Learning systems that are deployed for AI acceleration in automotive systems. Developed through a collaborative partnership between MLCommons and the Autonomous Vehicle Computing Consortium, this benchmark addresses the need for standardized performance evaluation methodologies in automotive machine learning systems. Existing benchmark suites cannot be utilized for these systems since automotive workloads have unique constraints including safety and real-time processing that distinguish them from the domains that previously introduced benchmarks target. Our benchmarking framework provides latency and accuracy metrics along with evaluation protocols that enable consistent and reproducible performance comparisons across different hardware platforms and software implementations. The first iteration of the benchmark consists of automotive perception tasks in 2D object detection, 2D semantic segmentation, and 3D object detection. We describe the methodology behind the benchmark design including the task selection, reference models, and submission rules. We also discuss the first round of benchmark submissions and the challenges involved in acquiring the datasets and the engineering efforts to develop the reference implementations. Our benchmark code is available at https://github.com/mlcommons/mlperf_automotive. |
| 2025-10-31 | [Consistency Training Helps Stop Sycophancy and Jailbreaks](http://arxiv.org/abs/2510.27062v1) | Alex Irpan, Alexander Matt Turner et al. | An LLM's factuality and refusal training can be compromised by simple changes to a prompt. Models often adopt user beliefs (sycophancy) or satisfy inappropriate requests which are wrapped within special text (jailbreaking). We explore \emph{consistency training}, a self-supervised paradigm that teaches a model to be invariant to certain irrelevant cues in the prompt. Instead of teaching the model what exact response to give on a particular prompt, we aim to teach the model to behave identically across prompt data augmentations (like adding leading questions or jailbreak text). We try enforcing this invariance in two ways: over the model's external outputs (\emph{Bias-augmented Consistency Training} (BCT) from Chua et al. [2025]) and over its internal activations (\emph{Activation Consistency Training} (ACT), a method we introduce). Both methods reduce Gemini 2.5 Flash's susceptibility to irrelevant cues. Because consistency training uses responses from the model itself as training data, it avoids issues that arise from stale training data, such as degrading model capabilities or enforcing outdated response guidelines. While BCT and ACT reduce sycophancy equally well, BCT does better at jailbreak reduction. We think that BCT can simplify training pipelines by removing reliance on static datasets. We argue that some alignment problems are better viewed not in terms of optimal responses, but rather as consistency issues. |
| 2025-10-30 | [Fine-Grained Iterative Adversarial Attacks with Limited Computation Budget](http://arxiv.org/abs/2510.26981v1) | Zhichao Hou, Weizhi Gao et al. | This work tackles a critical challenge in AI safety research under limited compute: given a fixed computation budget, how can one maximize the strength of iterative adversarial attacks? Coarsely reducing the number of attack iterations lowers cost but substantially weakens effectiveness. To fulfill the attainable attack efficacy within a constrained budget, we propose a fine-grained control mechanism that selectively recomputes layer activations across both iteration-wise and layer-wise levels. Extensive experiments show that our method consistently outperforms existing baselines at equal cost. Moreover, when integrated into adversarial training, it attains comparable performance with only 30% of the original budget. |
| 2025-10-30 | [LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks](http://arxiv.org/abs/2510.26941v1) | Seif Ikbarieh, Maanak Gupta et al. | The Internet of Things has expanded rapidly, transforming communication and operations across industries but also increasing the attack surface and security breaches. Artificial Intelligence plays a key role in securing IoT, enabling attack detection, attack behavior analysis, and mitigation suggestion. Despite advancements, evaluations remain purely qualitative, and the lack of a standardized, objective benchmark for quantitatively measuring AI-based attack analysis and mitigation hinders consistent assessment of model effectiveness. In this work, we propose a hybrid framework combining Machine Learning (ML) for multi-class attack detection with Large Language Models (LLMs) for attack behavior analysis and mitigation suggestion. After benchmarking several ML and Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we applied structured role-play prompt engineering with Retrieval-Augmented Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed, context-aware responses. We introduce novel evaluation metrics for quantitative assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o, DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the responses. Results show that Random Forest has the best detection model, and ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation. |
| 2025-10-30 | [RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification](http://arxiv.org/abs/2510.26935v1) | Yunhao Yang, Neel P. Bhatt et al. | As AI systems migrate to safety-critical domains, verifying that their actions comply with well-defined rules remains a challenge. Formal methods provide provable guarantees but demand hand-crafted temporal-logic specifications, offering limited expressiveness and accessibility. Deep learning approaches enable evaluation of plans against natural-language constraints, yet their opaque decision process invites misclassifications with potentially severe consequences. We introduce RepV, a neurosymbolic verifier that unifies both views by learning a latent space where safe and unsafe plans are linearly separable. Starting from a modest seed set of plans labeled by an off-the-shelf model checker, RepV trains a lightweight projector that embeds each plan, together with a language model-generated rationale, into a low-dimensional space; a frozen linear boundary then verifies compliance for unseen natural-language rules in a single forward pass.   Beyond binary classification, RepV provides a probabilistic guarantee on the likelihood of correct verification based on its position in the latent space. This guarantee enables a guarantee-driven refinement of the planner, improving rule compliance without human annotations. Empirical evaluations show that RepV improves compliance prediction accuracy by up to 15% compared to baseline methods while adding fewer than 0.2M parameters. Furthermore, our refinement framework outperforms ordinary fine-tuning baselines across various planning domains. These results show that safety-separable latent spaces offer a scalable, plug-and-play primitive for reliable neurosymbolic plan verification. Code and data are available at: https://repv-project.github.io/. |

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



