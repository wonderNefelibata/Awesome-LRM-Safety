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
| 2025-07-28 | [A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence](http://arxiv.org/abs/2507.21046v1) | Huan-ang Gao, Jiayi Geng et al. | Large Language Models (LLMs) have demonstrated strong capabilities but remain fundamentally static, unable to adapt their internal parameters to novel tasks, evolving knowledge domains, or dynamic interaction contexts. As LLMs are increasingly deployed in open-ended, interactive environments, this static nature has become a critical bottleneck, necessitating agents that can adaptively reason, act, and evolve in real time. This paradigm shift -- from scaling static models to developing self-evolving agents -- has sparked growing interest in architectures and methods enabling continual learning and adaptation from data, interactions, and experiences. This survey provides the first systematic and comprehensive review of self-evolving agents, organized around three foundational dimensions -- what to evolve, when to evolve, and how to evolve. We examine evolutionary mechanisms across agent components (e.g., models, memory, tools, architecture), categorize adaptation methods by stages (e.g., intra-test-time, inter-test-time), and analyze the algorithmic and architectural designs that guide evolutionary adaptation (e.g., scalar rewards, textual feedback, single-agent and multi-agent systems). Additionally, we analyze evaluation metrics and benchmarks tailored for self-evolving agents, highlight applications in domains such as coding, education, and healthcare, and identify critical challenges and research directions in safety, scalability, and co-evolutionary dynamics. By providing a structured framework for understanding and designing self-evolving agents, this survey establishes a roadmap for advancing adaptive agentic systems in both research and real-world deployments, ultimately shedding lights to pave the way for the realization of Artificial Super Intelligence (ASI), where agents evolve autonomously, performing at or beyond human-level intelligence across a wide array of tasks. |
| 2025-07-28 | [LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning](http://arxiv.org/abs/2507.20999v1) | Yining Huang, Bin Li et al. | Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines. |
| 2025-07-28 | [VArsity: Can Large Language Models Keep Power Engineering Students in Phase?](http://arxiv.org/abs/2507.20995v1) | Samuel Talkington, Daniel K. Molzahn | This paper provides an educational case study regarding our experience in deploying ChatGPT Large Language Models (LLMs) in the Spring 2025 and Fall 2023 offerings of ECE 4320: Power System Analysis and Control at Georgia Tech. As part of course assessments, students were tasked with identifying, explaining, and correcting errors in the ChatGPT outputs corresponding to power factor correction problems. While most students successfully identified the errors in the outputs from the GPT-4 version of ChatGPT used in Fall 2023, students found the errors from the ChatGPT o1 version much more difficult to identify in Spring 2025. As shown in this case study, the role of LLMs in pedagogy, assessment, and learning in power engineering classrooms is an important topic deserving further investigation. |
| 2025-07-28 | [Security Tensors as a Cross-Modal Bridge: Extending Text-Aligned Safety to Vision in LVLM](http://arxiv.org/abs/2507.20994v1) | Shen Li, Liuyi Yao et al. | Large visual-language models (LVLMs) integrate aligned large language models (LLMs) with visual modules to process multimodal inputs. However, the safety mechanisms developed for text-based LLMs do not naturally extend to visual modalities, leaving LVLMs vulnerable to harmful image inputs. To address this cross-modal safety gap, we introduce security tensors - trainable input vectors applied during inference through either the textual or visual modality. These tensors transfer textual safety alignment to visual processing without modifying the model's parameters. They are optimized using a curated dataset containing (i) malicious image-text pairs requiring rejection, (ii) contrastive benign pairs with text structurally similar to malicious queries, with the purpose of being contrastive examples to guide visual reliance, and (iii) general benign samples preserving model functionality. Experimental results demonstrate that both textual and visual security tensors significantly enhance LVLMs' ability to reject diverse harmful visual inputs while maintaining near-identical performance on benign tasks. Further internal analysis towards hidden-layer representations reveals that security tensors successfully activate the language module's textual "safety layers" in visual inputs, thereby effectively extending text-based safety to the visual modality. |
| 2025-07-28 | [Core Safety Values for Provably Corrigible Agents](http://arxiv.org/abs/2507.20964v1) | Aran Nayebi | We introduce the first implementable framework for corrigibility, with provable guarantees in multi-step, partially observed environments. Our framework replaces a single opaque reward with five *structurally separate* utility heads -- deference, switch-access preservation, truthfulness, low-impact behavior via a belief-based extension of Attainable Utility Preservation, and bounded task reward -- combined lexicographically by strict weight gaps. Theorem 1 proves exact single-round corrigibility in the partially observable off-switch game; Theorem 3 extends the guarantee to multi-step, self-spawning agents, showing that even if each head is \emph{learned} to mean-squared error $\varepsilon$ and the planner is $\varepsilon$-sub-optimal, the probability of violating \emph{any} safety property is bounded while still ensuring net human benefit. In contrast to Constitutional AI or RLHF/RLAIF, which merge all norms into one learned scalar, our separation makes obedience and impact-limits dominate even when incentives conflict. For open-ended settings where adversaries can modify the agent, we prove that deciding whether an arbitrary post-hack agent will ever violate corrigibility is undecidable by reduction to the halting problem, then carve out a finite-horizon ``decidable island'' where safety can be certified in randomized polynomial time and verified with privacy-preserving, constant-round zero-knowledge proofs. Consequently, the remaining challenge is the ordinary ML task of data coverage and generalization: reward-hacking risk is pushed into evaluation quality rather than hidden incentive leak-through, giving clearer implementation guidance for today's LLM assistants and future autonomous systems. |
| 2025-07-28 | [FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models](http://arxiv.org/abs/2507.20930v1) | Likun Tan, Kuan-Wei Huang et al. | Hallucinations in large language models pose a critical challenge for applications requiring factual reliability, particularly in high-stakes domains such as finance. This work presents an effective approach for detecting and editing factually incorrect content in model-generated responses based on the provided context. Given a user-defined domain-specific error taxonomy, we construct a synthetic dataset by inserting tagged errors into financial question-answering corpora and then fine-tune four language models, Phi-4, Phi-4-mini, Qwen3-4B, and Qwen3-14B, to detect and edit these factual inaccuracies. Our best-performing model, fine-tuned Phi-4, achieves an 8% improvement in binary F1 score and a 30% gain in overall detection performance compared to OpenAI-o3. Notably, our fine-tuned Phi-4-mini model, despite having only 4 billion parameters, maintains competitive performance with just a 2% drop in binary detection and a 0.1% decline in overall detection compared to OpenAI-o3. Our work provides a practical solution for detecting and editing factual inconsistencies in financial text generation while introducing a generalizable framework that can enhance the trustworthiness and alignment of large language models across diverse applications beyond finance. Our code and data are available at https://github.com/pegasi-ai/fine-grained-editting. |
| 2025-07-28 | [Endoscopic Depth Estimation Based on Deep Learning: A Survey](http://arxiv.org/abs/2507.20881v1) | Ke Niu, Zeyun Liu et al. | Endoscopic depth estimation is a critical technology for improving the safety and precision of minimally invasive surgery. It has attracted considerable attention from researchers in medical imaging, computer vision, and robotics. Over the past decade, a large number of methods have been developed. Despite the existence of several related surveys, a comprehensive overview focusing on recent deep learning-based techniques is still limited. This paper endeavors to bridge this gap by systematically reviewing the state-of-the-art literature. Specifically, we provide a thorough survey of the field from three key perspectives: data, methods, and applications, covering a range of methods including both monocular and stereo approaches. We describe common performance evaluation metrics and summarize publicly available datasets. Furthermore, this review analyzes the specific challenges of endoscopic scenes and categorizes representative techniques based on their supervision strategies and network architectures. The application of endoscopic depth estimation in the important area of robot-assisted surgery is also reviewed. Finally, we outline potential directions for future research, such as domain adaptation, real-time implementation, and enhanced model generalization, thereby providing a valuable starting point for researchers to engage with and advance the field. |
| 2025-07-28 | [Leveraging Open-Source Large Language Models for Clinical Information Extraction in Resource-Constrained Settings](http://arxiv.org/abs/2507.20859v1) | Luc Builtjes, Joeran Bosma et al. | Medical reports contain rich clinical information but are often unstructured and written in domain-specific language, posing challenges for information extraction. While proprietary large language models (LLMs) have shown promise in clinical natural language processing, their lack of transparency and data privacy concerns limit their utility in healthcare. This study therefore evaluates nine open-source generative LLMs on the DRAGON benchmark, which includes 28 clinical information extraction tasks in Dutch. We developed \texttt{llm\_extractinator}, a publicly available framework for information extraction using open-source generative LLMs, and used it to assess model performance in a zero-shot setting. Several 14 billion parameter models, Phi-4-14B, Qwen-2.5-14B, and DeepSeek-R1-14B, achieved competitive results, while the bigger Llama-3.3-70B model achieved slightly higher performance at greater computational cost. Translation to English prior to inference consistently degraded performance, highlighting the need of native-language processing. These findings demonstrate that open-source LLMs, when used with our framework, offer effective, scalable, and privacy-conscious solutions for clinical information extraction in low-resource settings. |
| 2025-07-28 | [Free Energy-Inspired Cognitive Risk Integration for AV Navigation in Pedestrian-Rich Environments](http://arxiv.org/abs/2507.20850v1) | Meiting Dang, Yanping Wu et al. | Recent advances in autonomous vehicle (AV) behavior planning have shown impressive social interaction capabilities when interacting with other road users. However, achieving human-like prediction and decision-making in interactions with vulnerable road users remains a key challenge in complex multi-agent interactive environments. Existing research focuses primarily on crowd navigation for small mobile robots, which cannot be directly applied to AVs due to inherent differences in their decision-making strategies and dynamic boundaries. Moreover, pedestrians in these multi-agent simulations follow fixed behavior patterns that cannot dynamically respond to AV actions. To overcome these limitations, this paper proposes a novel framework for modeling interactions between the AV and multiple pedestrians. In this framework, a cognitive process modeling approach inspired by the Free Energy Principle is integrated into both the AV and pedestrian models to simulate more realistic interaction dynamics. Specifically, the proposed pedestrian Cognitive-Risk Social Force Model adjusts goal-directed and repulsive forces using a fused measure of cognitive uncertainty and physical risk to produce human-like trajectories. Meanwhile, the AV leverages this fused risk to construct a dynamic, risk-aware adjacency matrix for a Graph Convolutional Network within a Soft Actor-Critic architecture, allowing it to make more reasonable and informed decisions. Simulation results indicate that our proposed framework effectively improves safety, efficiency, and smoothness of AV navigation compared to the state-of-the-art method. |
| 2025-07-28 | [Towards Explainable Deep Clustering for Time Series Data](http://arxiv.org/abs/2507.20840v1) | Udo Schlegel, Gabriel Marques Tavares et al. | Deep clustering uncovers hidden patterns and groups in complex time series data, yet its opaque decision-making limits use in safety-critical settings. This survey offers a structured overview of explainable deep clustering for time series, collecting current methods and their real-world applications. We thoroughly discuss and compare peer-reviewed and preprint papers through application domains across healthcare, finance, IoT, and climate science. Our analysis reveals that most work relies on autoencoder and attention architectures, with limited support for streaming, irregularly sampled, or privacy-preserved series, and interpretability is still primarily treated as an add-on. To push the field forward, we outline six research opportunities: (1) combining complex networks with built-in interpretability; (2) setting up clear, faithfulness-focused evaluation metrics for unsupervised explanations; (3) building explainers that adapt to live data streams; (4) crafting explanations tailored to specific domains; (5) adding human-in-the-loop methods that refine clusters and explanations together; and (6) improving our understanding of how time series clustering models work internally. By making interpretability a primary design goal rather than an afterthought, we propose the groundwork for the next generation of trustworthy deep clustering time series analytics. |
| 2025-07-28 | [Comparison of Magnetic Field Characteristics among 3 Tesla MRI Scanners: An Experimental Measurement Study](http://arxiv.org/abs/2507.20818v1) | Francesco Girardello, Maria Antonietta D'Avanzo et al. | Magnetic resonance imaging (MRI) scanners have advanced significantly, with a growing use of highfield 3 T systems. This evolution gives rise to safety concerns for healthcare personnel working in proximity to MRI equipment. While manufacturers provide theoretical Gauss line projections, these are typically derived under ideal open-environment conditions and may not reflect real-world installations. For this reason, identical MRI models can produce markedly different fringe field distributions depending on shielding and room configurations. The present study proposes an experimental methodology for the mapping of the fringe magnetic field in the vicinity of three 3 T MRI scanners. Field measurements were interpolated to generate threedimensional magnetic field maps. A comparative analysis was conducted, which revealed notable differences among the scanners. These differences serve to highlight the influence of site-specific factors on magnetic field propagation. |
| 2025-07-28 | [Aligning Large Language Model Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach](http://arxiv.org/abs/2507.20796v1) | Wei Lu, Daniel L. Chen et al. | Understanding how large language model (LLM) agents behave in strategic interactions is essential as these systems increasingly participate autonomously in economically and morally consequential decisions. We evaluate LLM preferences using canonical economic games, finding substantial deviations from human behavior. Models like GPT-4o show excessive cooperation and limited incentive sensitivity, while reasoning models, such as o3-mini, align more consistently with payoff-maximizing strategies. We propose a supervised fine-tuning pipeline that uses synthetic datasets derived from economic reasoning to align LLM agents with economic preferences, focusing on two stylized preference structures. In the first, utility depends only on individual payoffs (homo economicus), while utility also depends on a notion of Kantian universalizability in the second preference structure (homo moralis). We find that fine-tuning based on small datasets shifts LLM agent behavior toward the corresponding economic agent. We further assess the fine-tuned agents' behavior in two applications: Moral dilemmas involving autonomous vehicles and algorithmic pricing in competitive markets. These examples illustrate how different normative objectives embedded via realizations from structured preference structures can influence market and moral outcomes. This work contributes a replicable, cost-efficient, and economically grounded pipeline to align AI preferences using moral-economic principles. |
| 2025-07-28 | [Automating Thematic Review of Prevention of Future Deaths Reports: Replicating the ONS Child Suicide Study using Large Language Models](http://arxiv.org/abs/2507.20786v1) | Sam Osian, Arpan Dutta et al. | Prevention of Future Deaths (PFD) reports, issued by coroners in England and Wales, flag systemic hazards that may lead to further loss of life. Analysis of these reports has previously been constrained by the manual effort required to identify and code relevant cases. In 2025, the Office for National Statistics (ONS) published a national thematic review of child-suicide PFD reports ($\leq$ 18 years), identifying 37 cases from January 2015 to November 2023 - a process based entirely on manual curation and coding. We evaluated whether a fully automated, open source "text-to-table" language-model pipeline (PFD Toolkit) could reproduce the ONS's identification and thematic analysis of child-suicide PFD reports, and assessed gains in efficiency and reliability. All 4,249 PFD reports published from July 2013 to November 2023 were processed via PFD Toolkit's large language model pipelines. Automated screening identified cases where the coroner attributed death to suicide in individuals aged 18 or younger, and eligible reports were coded for recipient category and 23 concern sub-themes, replicating the ONS coding frame. PFD Toolkit identified 72 child-suicide PFD reports - almost twice the ONS count. Three blinded clinicians adjudicated a stratified sample of 144 reports to validate the child-suicide screening. Against the post-consensus clinical annotations, the LLM-based workflow showed substantial to almost-perfect agreement (Cohen's $\kappa$ = 0.82, 95% CI: 0.66-0.98, raw agreement = 91%). The end-to-end script runtime was 8m 16s, transforming a process that previously took months into one that can be completed in minutes. This demonstrates that automated LLM analysis can reliably and efficiently replicate manual thematic reviews of coronial data, enabling scalable, reproducible, and timely insights for public health and safety. The PFD Toolkit is openly available for future research. |
| 2025-07-28 | [On The Role of Pretrained Language Models in General-Purpose Text Embeddings: A Survey](http://arxiv.org/abs/2507.20783v1) | Meishan Zhang, Xin Zhang et al. | Text embeddings have attracted growing interest due to their effectiveness across a wide range of natural language processing (NLP) tasks, such as retrieval, classification, clustering, bitext mining, and summarization. With the emergence of pretrained language models (PLMs), general-purpose text embeddings (GPTE) have gained significant traction for their ability to produce rich, transferable representations. The general architecture of GPTE typically leverages PLMs to derive dense text representations, which are then optimized through contrastive learning on large-scale pairwise datasets. In this survey, we provide a comprehensive overview of GPTE in the era of PLMs, focusing on the roles PLMs play in driving its development. We first examine the fundamental architecture and describe the basic roles of PLMs in GPTE, i.e., embedding extraction, expressivity enhancement, training strategies, learning objectives, and data construction. Then, we describe advanced roles enabled by PLMs, such as multilingual support, multimodal integration, code understanding, and scenario-specific adaptation. Finally, we highlight potential future research directions that move beyond traditional improvement goals, including ranking integration, safety considerations, bias mitigation, structural information incorporation, and the cognitive extension of embeddings. This survey aims to serve as a valuable reference for both newcomers and established researchers seeking to understand the current state and future potential of GPTE. |
| 2025-07-28 | [Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models](http://arxiv.org/abs/2507.20704v1) | Gabriel Downer, Sean Craven et al. | The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications. |
| 2025-07-28 | [What's Really Different with AI? -- A Behavior-based Perspective on System Safety for Automated Driving Systems](http://arxiv.org/abs/2507.20685v1) | Marcus Nolte, Nayel Fabian Salem et al. | Assuring safety for ``AI-based'' systems is one of the current challenges in safety engineering. For automated driving systems, in particular, further assurance challenges result from the open context that the systems need to operate in after deployment. The current standardization and regulation landscape for ``AI-based'' systems is becoming ever more complex, as standards and regulations are being released at high frequencies.   This position paper seeks to provide guidance for making qualified arguments which standards should meaningfully be applied to (``AI-based'') automated driving systems. Furthermore, we argue for clearly differentiating sources of risk between AI-specific and general uncertainties related to the open context. In our view, a clear conceptual separation can help to exploit commonalities that can close the gap between system-level and AI-specific safety analyses, while ensuring the required rigor for engineering safe ``AI-based'' systems. |
| 2025-07-28 | [Before the Outrage: Challenges and Advances in Predicting Online Antisocial Behavior](http://arxiv.org/abs/2507.20614v1) | Anaïs Ollagnier | Antisocial behavior (ASB) on social media-including hate speech, harassment, and trolling-poses growing challenges for platform safety and societal wellbeing. While prior work has primarily focused on detecting harmful content after it appears, predictive approaches aim to forecast future harmful behaviors-such as hate speech propagation, conversation derailment, or user recidivism-before they fully unfold. Despite increasing interest, the field remains fragmented, lacking a unified taxonomy or clear synthesis of existing methods. This paper presents a systematic review of over 49 studies on ASB prediction, offering a structured taxonomy of five core task types: early harm detection, harm emergence prediction, harm propagation prediction, behavioral risk prediction, and proactive moderation support. We analyze how these tasks differ by temporal framing, prediction granularity, and operational goals. In addition, we examine trends in modeling techniques-from classical machine learning to pre-trained language models-and assess the influence of dataset characteristics on task feasibility and generalization. Our review highlights methodological challenges, such as dataset scarcity, temporal drift, and limited benchmarks, while outlining emerging research directions including multilingual modeling, cross-platform generalization, and human-in-the-loop systems. By organizing the field around a coherent framework, this survey aims to guide future work toward more robust and socially responsible ASB prediction. |
| 2025-07-28 | [HJB-based online safety-embedded critic learning for uncertain systems with self-triggered mechanism](http://arxiv.org/abs/2507.20545v1) | Zhanglin Shangguan, Bo Yang et al. | This paper presents a learning-based optimal control framework for safety-critical systems with parametric uncertainties, addressing both time-triggered and self-triggered controller implementations. First, we develop a robust control barrier function (RCBF) incorporating Lyapunov-based compensation terms to rigorously guarantee safety despite parametric uncertainties. Building on this safety guarantee, we formulate the constrained optimal control problem as the minimization of a novel safety-embedded value function, where the RCBF is involved via a Lagrange multiplier that adaptively balances safety constraints against optimal stabilization objectives. To enhance computational efficiency, we propose a self-triggered implementation mechanism that reduces control updates while maintaining dual stability-safety guarantees. The resulting self-triggered constrained Hamilton-Jacobi-Bellman (HJB) equation is solved through an online safety-embedded critic learning framework, with the Lagrange multiplier computed in real time to ensure safety. Numerical simulations demonstrate the effectiveness of the proposed approach in achieving both safety and control performance. |
| 2025-07-28 | [Scaling Pedestrian Crossing Analysis to 100 U.S. Cities via AI-based Segmentation of Satellite Imagery](http://arxiv.org/abs/2507.20497v1) | Marcel Moran, Arunav Gupta et al. | Accurately measuring street dimensions is essential to evaluating how their design influences both travel behavior and safety. However, gathering street-level information at city scale with precision is difficult given the quantity and complexity of urban intersections. To address this challenge in the context of pedestrian crossings - a crucial component of walkability - we introduce a scalable and accurate method for automatically measuring crossing distance at both marked and unmarked crosswalks, applied to America's 100 largest cities. First, OpenStreetMap coordinates were used to retrieve satellite imagery of intersections throughout each city, totaling roughly three million images. Next, Meta's Segment Anything Model was trained on a manually-labelled subset of these images to differentiate drivable from non-drivable surfaces (i.e., roads vs. sidewalks). Third, all available crossing edges from OpenStreetMap were extracted. Finally, crossing edges were overlaid on the segmented intersection images, and a grow-cut algorithm was applied to connect each edge to its adjacent non-drivable surface (e.g., sidewalk, private property, etc.), thus enabling the calculation of crossing distance. This achieved 93 percent accuracy in measuring crossing distance, with a median absolute error of 2 feet 3 inches (0.69 meters), when compared to manually-verified data for an entire city. Across the 100 largest US cities, median crossing distance ranges from 32 feet to 78 feet (9.8 to 23.8m), with detectable regional patterns. Median crossing distance also displays a positive relationship with cities' year of incorporation, illustrating in a novel way how American cities increasingly emphasize wider (and more car-centric) streets. |
| 2025-07-28 | [STARN-GAT: A Multi-Modal Spatio-Temporal Graph Attention Network for Accident Severity Prediction](http://arxiv.org/abs/2507.20451v1) | Pritom Ray Nobin, Imran Ahammad Rifat | Accurate prediction of traffic accident severity is critical for improving road safety, optimizing emergency response strategies, and informing the design of safer transportation infrastructure. However, existing approaches often struggle to effectively model the intricate interdependencies among spatial, temporal, and contextual variables that govern accident outcomes. In this study, we introduce STARN-GAT, a Multi-Modal Spatio-Temporal Graph Attention Network, which leverages adaptive graph construction and modality-aware attention mechanisms to capture these complex relationships. Unlike conventional methods, STARN-GAT integrates road network topology, temporal traffic patterns, and environmental context within a unified attention-based framework. The model is evaluated on the Fatality Analysis Reporting System (FARS) dataset, achieving a Macro F1-score of 85 percent, ROC-AUC of 0.91, and recall of 81 percent for severe incidents. To ensure generalizability within the South Asian context, STARN-GAT is further validated on the ARI-BUET traffic accident dataset, where it attains a Macro F1-score of 0.84, recall of 0.78, and ROC-AUC of 0.89. These results demonstrate the model's effectiveness in identifying high-risk cases and its potential for deployment in real-time, safety-critical traffic management systems. Furthermore, the attention-based architecture enhances interpretability, offering insights into contributing factors and supporting trust in AI-assisted decision-making. Overall, STARN-GAT bridges the gap between advanced graph neural network techniques and practical applications in road safety analytics. |

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



