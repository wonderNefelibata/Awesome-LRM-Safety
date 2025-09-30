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
| 2025-09-29 | [TemMed-Bench: Evaluating Temporal Medical Image Reasoning in Vision-Language Models](http://arxiv.org/abs/2509.25143v1) | Junyi Zhang, Jia-Chen Gu et al. | Existing medical reasoning benchmarks for vision-language models primarily focus on analyzing a patient's condition based on an image from a single visit. However, this setting deviates significantly from real-world clinical practice, where doctors typically refer to a patient's historical conditions to provide a comprehensive assessment by tracking their changes over time. In this paper, we introduce TemMed-Bench, the first benchmark designed for analyzing changes in patients' conditions between different clinical visits, which challenges large vision-language models (LVLMs) to reason over temporal medical images. TemMed-Bench consists of a test set comprising three tasks - visual question-answering (VQA), report generation, and image-pair selection - and a supplementary knowledge corpus of over 17,000 instances. With TemMed-Bench, we conduct an evaluation of six proprietary and six open-source LVLMs. Our results show that most LVLMs lack the ability to analyze patients' condition changes over temporal medical images, and a large proportion perform only at a random-guessing level in the closed-book setting. In contrast, GPT o3, o4-mini and Claude 3.5 Sonnet demonstrate comparatively decent performance, though they have yet to reach the desired level. Furthermore, we explore augmenting the input with both retrieved visual and textual modalities in the medical domain. We also show that multi-modal retrieval augmentation yields notably higher performance gains than no retrieval and textual retrieval alone across most models on our benchmark, with the VQA task showing an average improvement of 2.59%. Overall, we compose a benchmark grounded on real-world clinical practice, and it reveals LVLMs' limitations in temporal medical image reasoning, as well as highlighting the use of multi-modal retrieval augmentation as a potentially promising direction worth exploring to address this challenge. |
| 2025-09-29 | [Towards Trustworthy Lexical Simplification: Exploring Safety and Efficiency with Small LLMs](http://arxiv.org/abs/2509.25086v1) | Akio Hayakawa, Stefan Bott et al. | Despite their strong performance, large language models (LLMs) face challenges in real-world application of lexical simplification (LS), particularly in privacy-sensitive and resource-constrained environments. Moreover, since vulnerable user groups (e.g., people with disabilities) are one of the key target groups of this technology, it is crucial to ensure the safety and correctness of the output of LS systems. To address these issues, we propose an efficient framework for LS systems that utilizes small LLMs deployable in local environments. Within this framework, we explore knowledge distillation with synthesized data and in-context learning as baselines. Our experiments in five languages evaluate model outputs both automatically and manually. Our manual analysis reveals that while knowledge distillation boosts automatic metric scores, it also introduces a safety trade-off by increasing harmful simplifications. Importantly, we find that the model's output probability is a useful signal for detecting harmful simplifications. Leveraging this, we propose a filtering strategy that suppresses harmful simplifications while largely preserving beneficial ones. This work establishes a benchmark for efficient and safe LS with small LLMs. It highlights the key trade-offs between performance, efficiency, and safety, and demonstrates a promising approach for safe real-world deployment. |
| 2025-09-29 | [Effectiveness and Safety of Selective IL-23 Receptor Antagonists in Moderate to Severe Ulcerative Colitis: A Systematic Review, Meta-Analysis and Trial Sequential Analysis](http://arxiv.org/abs/2509.25069v1) | Wellgner Fernandes Oliveira Amador, Isabelle Castro Vitor et al. | Selective interleukin-23 receptor antagonists (IL-23RA) show promise for treating moderate to severe ulcerative colitis (UC) but their efficacy and safety are not fully understood. We performed a systematic review and meta-analysis of randomized controlled trials comparing IL-23RA with placebo in moderate to severe UC. Outcomes included clinical and endoscopic remission, response rates, and adverse events. Nine trials including 3808 patients in the induction phase and 1734 in the maintenance phase were analyzed. IL-23RA improved clinical remission (induction risk ratio 2.63, 95 percent confidence interval 2.05-3.36; maintenance 1.99, 95 percent confidence interval 1.63-2.44) and endoscopic remission (induction 2.36, 95 percent confidence interval 1.70-2.20; maintenance 1.96, 95 percent confidence interval 1.63-2.37). IL-23RA reduced serious adverse events in the induction phase (0.40, 95 percent confidence interval 0.27-0.69) with no difference during maintenance (0.75, 95 percent confidence interval 0.31-1.84). No significant differences were observed in overall adverse events or specific events such as headache or nasopharyngitis. Trial sequential analysis confirmed sufficient sample size for clinical endpoints. IL-23RA showed superior effectiveness and similar safety compared with placebo in moderate to severe UC. |
| 2025-09-29 | [Safety-Critical Input-Constrained Nonlinear Intercept Guidance in Multiple Engagement Zones](http://arxiv.org/abs/2509.25053v1) | Praveen Kumar Ranjan, Abhinav Sinha et al. | This paper presents an input-constrained nonlinear guidance law to address the problem of intercepting a stationary target in contested environments with multiple defending agents. Contrary to prior approaches that rely on explicit knowledge of defender strategies or utilize conservative safety conditions based on a defender's range, our work characterizes defender threats geometrically through engagement zones that delineate inevitable interception regions. Outside these engagement zones, the interceptor remains invulnerable. The proposed guidance law switches between a repulsive safety maneuver near these zones and a pursuit maneuver outside their influence. To deal with multiple engagement zones, we employ a smooth minimum function (log-sum-exponent approximation) that aggregates threats from all the zones while prioritizing the most critical threats. Input saturation is modeled and embedded in the non-holonomic vehicle dynamics so the controller respects actuator limits while maintaining stability. Numerical simulations with several defenders demonstrate the proposed method's ability to avoid engagement zones and achieve interception across diverse initial conditions. |
| 2025-09-29 | [CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation](http://arxiv.org/abs/2509.25016v1) | Max Curie, Paulo da Costa | We introduce CLASP (Clustering via Adaptive Spectral Processing), a lightweight framework for unsupervised image segmentation that operates without any labeled data or finetuning. CLASP first extracts per patch features using a self supervised ViT encoder (DINO); then, it builds an affinity matrix and applies spectral clustering. To avoid manual tuning, we select the segment count automatically with a eigengap silhouette search, and we sharpen the boundaries with a fully connected DenseCRF. Despite its simplicity and training free nature, CLASP attains competitive mIoU and pixel accuracy on COCO Stuff and ADE20K, matching recent unsupervised baselines. The zero training design makes CLASP a strong, easily reproducible baseline for large unannotated corpora especially common in digital advertising and marketing workflows such as brand safety screening, creative asset curation, and social media content moderation |
| 2025-09-29 | [Joyride: Rethinking Linux's network stack design for better performance, security, and reliability](http://arxiv.org/abs/2509.25015v1) | Yanlin Du, Ruslan Nikolaev | Contemporary distributed computing workloads, including scientific computation, data mining, and machine learning, increasingly demand OS networking with minimal latency as well as high throughput, security, and reliability. However, Linux's conventional TCP/IP stack becomes increasingly problematic for high-end NICs, particularly those operating at 100 Gbps and beyond.   These limitations come mainly from overheads associated with kernel space processing, mode switching, and data copying in the legacy architecture. Although kernel bypass techniques such as DPDK and RDMA offer alternatives, they introduce significant adoption barriers: both often require extensive application redesign, and RDMA is not universally available on commodity hardware.   This paper proposes Joyride, a high performance framework with a grand vision of replacing Linux's legacy network stack while providing compatibility with existing applications. Joyride aims to integrate kernel bypass ideas, specifically DPDK and a user-space TCP/IP stack, while designing a microkernel-style architecture for Linux networking. |
| 2025-09-29 | [World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](http://arxiv.org/abs/2509.24948v1) | Junjin Xiao, Yandan Yang et al. | Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose World-Env, an RL-based post-training framework that replaces physical interaction with a low-cost, world model-based virtual simulator. World-Env consists of two key components: (1) a video-based world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that World-Env effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. |
| 2025-09-29 | [BOE-XSUM: Extreme Summarization in Clear Language of Spanish Legal Decrees and Notifications](http://arxiv.org/abs/2509.24908v1) | Andrés Fernández García, Javier de la Rosa et al. | The ability to summarize long documents succinctly is increasingly important in daily life due to information overload, yet there is a notable lack of such summaries for Spanish documents in general, and in the legal domain in particular. In this work, we present BOE-XSUM, a curated dataset comprising 3,648 concise, plain-language summaries of documents sourced from Spain's ``Bolet\'{\i}n Oficial del Estado'' (BOE), the State Official Gazette. Each entry in the dataset includes a short summary, the original text, and its document type label. We evaluate the performance of medium-sized large language models (LLMs) fine-tuned on BOE-XSUM, comparing them to general-purpose generative models in a zero-shot setting. Results show that fine-tuned models significantly outperform their non-specialized counterparts. Notably, the best-performing model -- BERTIN GPT-J 6B (32-bit precision) -- achieves a 24\% performance gain over the top zero-shot model, DeepSeek-R1 (accuracies of 41.6\% vs.\ 33.5\%). |
| 2025-09-29 | [Vision At Night: Exploring Biologically Inspired Preprocessing For Improved Robustness Via Color And Contrast Transformations](http://arxiv.org/abs/2509.24863v1) | Lorena Stracke, Lia Nimmermann et al. | Inspired by the human visual system's mechanisms for contrast enhancement and color-opponency, we explore biologically motivated input preprocessing for robust semantic segmentation. By applying Difference-of-Gaussians (DoG) filtering to RGB, grayscale, and opponent-color channels, we enhance local contrast without modifying model architecture or training. Evaluations on Cityscapes, ACDC, and Dark Zurich show that such preprocessing maintains in-distribution performance while improving robustness to adverse conditions like night, fog, and snow. As this processing is model-agnostic and lightweight, it holds potential for integration into imaging pipelines, enabling imaging systems to deliver task-ready, robust inputs for downstream vision models in safety-critical environments. |
| 2025-09-29 | [KnowGuard: Knowledge-Driven Abstention for Multi-Round Clinical Reasoning](http://arxiv.org/abs/2509.24816v1) | Xilin Dang, Kexin Chen et al. | In clinical practice, physicians refrain from making decisions when patient information is insufficient. This behavior, known as abstention, is a critical safety mechanism preventing potentially harmful misdiagnoses. Recent investigations have reported the application of large language models (LLMs) in medical scenarios. However, existing LLMs struggle with the abstentions, frequently providing overconfident responses despite incomplete information. This limitation stems from conventional abstention methods relying solely on model self-assessments, which lack systematic strategies to identify knowledge boundaries with external medical evidences. To address this, we propose \textbf{KnowGuard}, a novel \textit{investigate-before-abstain} paradigm that integrates systematic knowledge graph exploration for clinical decision-making. Our approach consists of two key stages operating on a shared contextualized evidence pool: 1) an evidence discovery stage that systematically explores the medical knowledge space through graph expansion and direct retrieval, and 2) an evidence evaluation stage that ranks evidence using multiple factors to adapt exploration based on patient context and conversation history. This two-stage approach enables systematic knowledge graph exploration, allowing models to trace structured reasoning paths and recognize insufficient medical evidence. We evaluate our abstention approach using open-ended multi-round clinical benchmarks that mimic realistic diagnostic scenarios, assessing abstention quality through accuracy-efficiency trade-offs beyond existing closed-form evaluations. Experimental evidences clearly demonstrate that KnowGuard outperforms state-of-the-art abstention approaches, improving diagnostic accuracy by 3.93\% while reducing unnecessary interaction by 7.27 turns on average. |
| 2025-09-29 | [APREBot: Active Perception System for Reflexive Evasion Robot](http://arxiv.org/abs/2509.24733v1) | Zihao Xu, Kuankuan Sima et al. | Reliable onboard perception is critical for quadruped robots navigating dynamic environments, where obstacles can emerge from any direction under strict reaction-time constraints. Single-sensor systems face inherent limitations: LiDAR provides omnidirectional coverage but lacks rich texture information, while cameras capture high-resolution detail but suffer from restricted field of view. We introduce APREBot (Active Perception System for Reflexive Evasion Robot), a novel framework that integrates reflexive evasion with active hierarchical perception. APREBot strategically combines LiDAR-based omnidirectional scanning with camera-based active focusing, achieving comprehensive environmental awareness essential for agile obstacle avoidance in quadruped robots. We validate APREBot through extensive sim-to-real experiments on a quadruped platform, evaluating diverse obstacle types, trajectories, and approach directions. Our results demonstrate substantial improvements over state-of-the-art baselines in both safety metrics and operational efficiency, highlighting APREBot's potential for dependable autonomy in safety-critical scenarios. Videos are available at https://sites.google.com/view/aprebot/ |
| 2025-09-29 | [Diamonds in the rough: Transforming SPARCs of imagination into a game concept by leveraging medium sized LLMs](http://arxiv.org/abs/2509.24730v1) | Julian Geheeb, Farhan Abid Ivan et al. | Recent research has demonstrated that large language models (LLMs) can support experts across various domains, including game design. In this study, we examine the utility of medium-sized LLMs, models that operate on consumer-grade hardware typically available in small studios or home environments. We began by identifying ten key aspects that contribute to a strong game concept and used ChatGPT to generate thirty sample game ideas. Three medium-sized LLMs, LLaMA 3.1, Qwen 2.5, and DeepSeek-R1, were then prompted to evaluate these ideas according to the previously identified aspects. A qualitative assessment by two researchers compared the models' outputs, revealing that DeepSeek-R1 produced the most consistently useful feedback, despite some variability in quality. To explore real-world applicability, we ran a pilot study with ten students enrolled in a storytelling course for game development. At the early stages of their own projects, students used our prompt and DeepSeek-R1 to refine their game concepts. The results indicate a positive reception: most participants rated the output as high quality and expressed interest in using such tools in their workflows. These findings suggest that current medium-sized LLMs can provide valuable feedback in early game design, though further refinement of prompting methods could improve consistency and overall effectiveness. |
| 2025-09-29 | [GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training](http://arxiv.org/abs/2509.24494v1) | Hongcheng Wang, Yinuo Huang et al. | Recent progress, such as DeepSeek-R1, has shown that the GRPO algorithm, a Reinforcement Learning (RL) approach, can effectively train Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) and Vision-Language Models (VLMs). In this paper, we analyze three challenges of GRPO: gradient coupling between thoughts and answers, sparse reward signals caused by limited parallel sampling, and unstable advantage estimation. To mitigate these challenges, we propose GRPO-MA, a simple yet theoretically grounded method that leverages multi-answer generation from each thought process, enabling more robust and efficient optimization. Theoretically, we show that the variance of thought advantage decreases as the number of answers per thought increases. Empirically, our gradient analysis confirms this effect, showing that GRPO-MA reduces gradient spikes compared to GRPO. Experiments on math, code, and diverse multimodal tasks demonstrate that GRPO-MA substantially improves performance and training efficiency. Our ablation studies further reveal that increasing the number of answers per thought consistently enhances model performance. |
| 2025-09-29 | [Sanitize Your Responses: Mitigating Privacy Leakage in Large Language Models](http://arxiv.org/abs/2509.24488v1) | Wenjie Fu, Huandong Wang et al. | As Large Language Models (LLMs) achieve remarkable success across a wide range of applications, such as chatbots and code copilots, concerns surrounding the generation of harmful content have come increasingly into focus. Despite significant advances in aligning LLMs with safety and ethical standards, adversarial prompts can still be crafted to elicit undesirable responses. Existing mitigation strategies are predominantly based on post-hoc filtering, which introduces substantial latency or computational overhead, and is incompatible with token-level streaming generation. In this work, we introduce Self-Sanitize, a novel LLM-driven mitigation framework inspired by cognitive psychology, which emulates human self-monitor and self-repair behaviors during conversations. Self-Sanitize comprises a lightweight Self-Monitor module that continuously inspects high-level intentions within the LLM at the token level via representation engineering, and a Self-Repair module that performs in-place correction of harmful content without initiating separate review dialogues. This design allows for real-time streaming monitoring and seamless repair, with negligible impact on latency and resource utilization. Given that privacy-invasive content has often been insufficiently focused in previous studies, we perform extensive experiments on four LLMs across three privacy leakage scenarios. The results demonstrate that Self-Sanitize achieves superior mitigation performance with minimal overhead and without degrading the utility of LLMs, offering a practical and robust solution for safer LLM deployments. Our code is available at the following link: https://github.com/wjfu99/LLM_Self_Sanitize |
| 2025-09-29 | [GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners](http://arxiv.org/abs/2509.24418v1) | Haoran Li, Yulin Chen et al. | As large language models (LLMs) are increasingly integrated into numerous applications across various domains, LLMs' safety becomes a critical concern for both application developers and intended users. Currently, great efforts have been made to develop safety benchmarks with fine-grained taxonomies. However, these benchmarks' taxonomies are disparate with different safety policies. Thus, existing safeguards trained on these benchmarks are either coarse-grained to only distinguish between safe and unsafe, or constrained by the narrow risk taxonomies of a single benchmark. To leverage these fine-grained safety taxonomies across multiple safety benchmarks, in this paper, we propose GSPR, a Generalizable Safety Policy Reasoner to identify unsafe input prompts and LLMs' outputs with violated safety taxonomies through Group Relative Policy Optimization (GRPO). Unlike prior safeguards which only cover a fixed set of risk factors, our GSPR incentivizes its reasoning capability with varied safety taxonomies through our careful cold-start strategy and reward design. Consequently, our GSPR can be trained across multiple safety benchmarks with distinct taxonomies and naturally exhibits powerful generalization ability. We conduct extensive experiments to show that our GSPR significantly improves existing safety guardrails' reasoning capabilities for both safety and category prediction tasks. Moreover, our GSPR not only demonstrates powerful safety generalization abilities but also achieves the least inference token costs with explanations. |
| 2025-09-29 | [DynaMIC: Dynamic Multimodal In-Context Learning Enabled Embodied Robot Counterfactual Resistance Ability](http://arxiv.org/abs/2509.24413v1) | Tianqiang Yan, Ziqiao Lin et al. | The emergence of large pre-trained models based on natural language has breathed new life into robotics development. Extensive research has integrated large models with robots, utilizing the powerful semantic understanding and generation capabilities of large models to facilitate robot control through natural language instructions gradually. However, we found that robots that strictly adhere to human instructions, especially those containing misleading information, may encounter errors during task execution, potentially leading to safety hazards. This resembles the concept of counterfactuals in natural language processing (NLP), which has not yet attracted much attention in robotic research. In an effort to highlight this issue for future studies, this paper introduced directive counterfactuals (DCFs) arising from misleading human directives. We present DynaMIC, a framework for generating robot task flows to identify DCFs and relay feedback to humans proactively. This capability can help robots be sensitive to potential DCFs within a task, thus enhancing the reliability of the execution process. We conducted semantic-level experiments and ablation studies, showcasing the effectiveness of this framework. |
| 2025-09-29 | [Multilingual Text-to-SQL: Benchmarking the Limits of Language Models with Collaborative Language Agents](http://arxiv.org/abs/2509.24405v1) | Khanh Trinh Pham, Thu Huong Nguyen et al. | Text-to-SQL enables natural access to databases, yet most benchmarks are English-only, limiting multilingual progress. We introduce MultiSpider 2.0, extending Spider 2.0 to eight languages (English, German, French, Spanish, Portuguese, Japanese, Chinese, Vietnamese). It preserves Spider 2.0's structural difficulty while adding linguistic and dialectal variability, demanding deeper reasoning for complex SQL. On this benchmark, state-of-the-art LLMs (such as DeepSeek-R1 and OpenAI o1) reach only 4\% execution accuracy when relying on intrinsic reasoning, versus 60\% on MultiSpider 1.0. Therefore, we provide a collaboration-driven language agents baseline that iteratively refines queries, improving accuracy to 15\%. These results reveal a substantial multilingual gap and motivate methods that are robust across languages and ready for real-world enterprise deployment. Our benchmark is available at https://github.com/phkhanhtrinh23/Multilingual_Text_to_SQL. |
| 2025-09-29 | [Autonomous Detection and Coverage of Unknown Target Areas by Multi-Agent Systems](http://arxiv.org/abs/2509.24399v1) | Jie Song, Yang Bai et al. | This paper presents a novel coverage control algorithm for multi-agent systems, where each agent has no prior knowledge of the specific region to be covered. The proposed method enables agents to autonomously detect the target area and collaboratively achieve full coverage. Once an agent detects a part of the target region within its sensor range, a dynamically constructed density function is generated to attract nearby agents. By integrating this density-driven mechanism with Centroidal Voronoi Tessellation (CVT), the agents are guided to achieve optimal spatial distribution. Additionally, Control Barrier Functions (CBFs) are employed to ensure collision avoidance and maintain non-overlapping sensor coverage, enhancing both safety and efficiency. Simulation results verify that agents can independently locate and effectively cover the target area. |
| 2025-09-29 | [Assessing Roundabout Safety Perceptions under Heterogeneous Traffic: Socio-Demographic and Geometric Influences in Indian Urban Contexts](http://arxiv.org/abs/2509.24397v1) | Abhijnan Maji, Indrajit Ghosh | Evaluation of the safety perceptions of roundabout users is crucial for improving road safety in mixed-traffic environments. The crash- and conflict-based analyses do not incorporate the socio-demographic characteristics of the roundabout users, which can only be captured through questionnaire surveys on a larger scale. This research evaluated the relationship of roundabout safety perception with demographic factors, driving characteristics, and varying roundabout geometries using multiple correspondence analysis, cluster analysis, factor analysis, and multinomial logistic regression. The study analyzed data from 1,530 respondents across two Indian cities. The study identified three roundabout user clusters. Single-lane roundabouts were perceived as safer during entry and circulation, with a significant prominence among middle-aged users. In contrast, double- and multi-lane roundabouts presented higher perceived risks during exit maneuvers, especially among young, inexperienced, unemployed/self-employed users. Vulnerable road users reported significantly higher perceived risks, especially under suboptimal lighting conditions. Respondents with 10-20 years of driving experience, especially car users, perceived lower risk at single-lane roundabouts but acknowledged the higher risk linked to speed variations and complex maneuvers at multi-lane roundabouts. Driving experience, vehicle type, and geometric configurations were crucial in roundabout safety perception. The study highlighted the need to improve the built environment of roundabouts for vulnerable road users. The roundabout merging area was perceived as the most dangerous spot; however, exits were also perceived as dangerous for double- and multi-lane roundabouts. The findings can benefit policymakers, engineers, and urban planners by enabling them to deploy targeted safety interventions based on issues highlighted in the study. |
| 2025-09-29 | [The 2025 OpenAI Preparedness Framework does not guarantee any AI risk mitigation practices: a proof-of-concept for affordance analyses of AI safety policies](http://arxiv.org/abs/2509.24394v1) | Sam Coggins, Alex Saeri et al. | Prominent AI companies are producing 'safety frameworks' as a type of voluntary self-governance. These statements purport to establish risk thresholds and safety procedures for the development and deployment of highly capable AI. Understanding which AI risks are covered and what actions are allowed, refused, demanded, encouraged, or discouraged by these statements is vital for assessing how these frameworks actually govern AI development and deployment. We draw on affordance theory to analyse the OpenAI 'Preparedness Framework Version 2' (April 2025) using the Mechanisms & Conditions model of affordances and the MIT AI Risk Repository. We find that this safety policy requests evaluation of a small minority of AI risks, encourages deployment of systems with 'Medium' capabilities for what OpenAI itself defines as 'severe harm' (potential for >1000 deaths or >$100B in damages), and allows OpenAI's CEO to deploy even more dangerous capabilities. These findings suggest that effective mitigation of AI risks requires more robust governance interventions beyond current industry self-regulation. Our affordance analysis provides a replicable method for evaluating what safety frameworks actually permit versus what they claim. |

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



