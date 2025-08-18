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
| 2025-08-15 | [Thyme: Think Beyond Images](http://arxiv.org/abs/2508.11630v1) | Yi-Fan Zhang, Xingyu Lu et al. | Following OpenAI's introduction of the ``thinking with images'' concept, recent efforts have explored stimulating the use of visual information in the reasoning process to enhance model performance in perception and reasoning tasks. However, to the best of our knowledge, no open-source work currently offers a feature set as rich as proprietary models (O3), which can perform diverse image manipulations and simultaneously enhance logical reasoning capabilities through code. In this paper, we make a preliminary attempt in this direction by introducing Thyme (Think Beyond Images), a novel paradigm for enabling MLLMs to transcend existing ``think with images'' approaches by autonomously generating and executing diverse image processing and computational operations via executable code. This approach not only facilitates a rich, on-the-fly set of image manipulations (e.g., cropping, rotation, contrast enhancement) but also allows for mathematical computations, all while maintaining high autonomy in deciding when and how to apply these operations. We activate this capability through a two-stage training strategy: an initial SFT on a curated dataset of 500K samples to teach code generation, followed by a RL phase to refine decision-making. For the RL stage, we manually collect and design high-resolution question-answer pairs to increase the learning difficulty, and we propose GRPO-ATS (Group Relative Policy Optimization with Adaptive Temperature Sampling), an algorithm that applies distinct temperatures to text and code generation to balance reasoning exploration with code execution precision. We conduct extensive experimental analysis and ablation studies. Comprehensive evaluations on nearly 20 benchmarks show that Thyme yields significant and consistent performance gains, particularly in challenging high-resolution perception and complex reasoning tasks. |
| 2025-08-15 | [Optimal CO2 storage management considering safety constraints in multi-stakeholder multi-site CCS projects: a game theoretic perspective](http://arxiv.org/abs/2508.11618v1) | Jungang Chen, Seyyed A. Hosseini | Carbon capture and storage (CCS) projects typically involve a diverse array of stakeholders or players from public, private, and regulatory sectors, each with different objectives and responsibilities. Given the complexity, scale, and long-term nature of CCS operations, determining whether individual stakeholders can independently maximize their interests or whether collaborative coalition agreements are needed remains a central question for effective CCS project planning and management. CCS projects are often implemented in geologically connected sites, where shared geological features such as pressure space and reservoir pore capacity can lead to competitive behavior among stakeholders. Furthermore, CO2 storage sites are often located in geologically mature basins that previously served as sites for hydrocarbon extraction or wastewater disposal in order to leverage existing infrastructures, which makes unilateral optimization even more complicated and unrealistic.   In this work, we propose a paradigm based on Markov games to quantitatively investigate how different coalition structures affect the goals of stakeholders. We frame this multi-stakeholder multi-site problem as a multi-agent reinforcement learning problem with safety constraints. Our approach enables agents to learn optimal strategies while compliant with safety regulations. We present an example where multiple operators are injecting CO2 into their respective project areas in a geologically connected basin. To address the high computational cost of repeated simulations of high-fidelity models, a previously developed surrogate model based on the Embed-to-Control (E2C) framework is employed. Our results demonstrate the effectiveness of the proposed framework in addressing optimal management of CO2 storage when multiple stakeholders with various objectives and goals are involved. |
| 2025-08-15 | [Intergenerational Support for Deepfake Scams Targeting Older Adults](http://arxiv.org/abs/2508.11579v1) | Karina LaRubbio, Alyssa Lanter et al. | AI-enhanced scams now employ deepfake technology to produce convincing audio and visual impersonations of trusted family members, often grandchildren, in real time. These attacks fabricate urgent scenarios, such as legal or medical emergencies, to socially engineer older adults into transferring money. The realism of these AI-generated impersonations undermines traditional cues used to detect fraud, making them a powerful tool for financial exploitation. In this study, we explore older adults' perceptions of these emerging threats and their responses, with a particular focus on the role of youth, who may also be impacted by having their identities exploited, in supporting older family members' online safety. We conducted focus groups with 37 older adults (ages 65+) to examine their understanding of deepfake impersonation scams and the value of intergenerational technology support. Findings suggest that older adults frequently rely on trusted relationships to detect scams and develop protective practices. Based on this, we identify opportunities to engage youth as active partners in enhancing resilience across generations. |
| 2025-08-15 | [DiCriTest: Testing Scenario Generation for Decision-Making Agents Considering Diversity and Criticality](http://arxiv.org/abs/2508.11514v1) | Qitong Chu, Yufeng Yue et al. | The growing deployment of decision-making agents in dynamic environments increases the demand for safety verification. While critical testing scenario generation has emerged as an appealing verification methodology, effectively balancing diversity and criticality remains a key challenge for existing methods, particularly due to local optima entrapment in high-dimensional scenario spaces. To address this limitation, we propose a dual-space guided testing framework that coordinates scenario parameter space and agent behavior space, aiming to generate testing scenarios considering diversity and criticality. Specifically, in the scenario parameter space, a hierarchical representation framework combines dimensionality reduction and multi-dimensional subspace evaluation to efficiently localize diverse and critical subspaces. This guides dynamic coordination between two generation modes: local perturbation and global exploration, optimizing critical scenario quantity and diversity. Complementarily, in the agent behavior space, agent-environment interaction data are leveraged to quantify behavioral criticality/diversity and adaptively support generation mode switching, forming a closed feedback loop that continuously enhances scenario characterization and exploration within the parameter space. Experiments show our framework improves critical scenario generation by an average of 56.23\% and demonstrates greater diversity under novel parameter-behavior co-driven metrics when tested on five decision-making agents, outperforming state-of-the-art baselines. |
| 2025-08-15 | [Predicting and Explaining Traffic Crash Severity Through Crash Feature Selection](http://arxiv.org/abs/2508.11504v1) | Andrea Castellani, Zacharias Papadovasilakis et al. | Motor vehicle crashes remain a leading cause of injury and death worldwide, necessitating data-driven approaches to understand and mitigate crash severity. This study introduces a curated dataset of more than 3 million people involved in accidents in Ohio over six years (2017-2022), aggregated to more than 2.3 million vehicle-level records for predictive analysis. The primary contribution is a transparent and reproducible methodology that combines Automated Machine Learning (AutoML) and explainable artificial intelligence (AI) to identify and interpret key risk factors associated with severe crashes. Using the JADBio AutoML platform, predictive models were constructed to distinguish between severe and non-severe crash outcomes. The models underwent rigorous feature selection across stratified training subsets, and their outputs were interpreted using SHapley Additive exPlanations (SHAP) to quantify the contribution of individual features. A final Ridge Logistic Regression model achieved an AUC-ROC of 85.6% on the training set and 84.9% on a hold-out test set, with 17 features consistently identified as the most influential predictors. Key features spanned demographic, environmental, vehicle, human, and operational categories, including location type, posted speed, minimum occupant age, and pre-crash action. Notably, certain traditionally emphasized factors, such as alcohol or drug impairment, were less influential in the final model compared to environmental and contextual variables. Emphasizing methodological rigor and interpretability over mere predictive performance, this study offers a scalable framework to support Vision Zero with aligned interventions and advanced data-informed traffic safety policy. |
| 2025-08-15 | [Tapas are free! Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](http://arxiv.org/abs/2508.11425v1) | Jinwei Hu, Yi Dong et al. | Autonomous agents in safety-critical applications must continuously adapt to dynamic conditions without compromising performance and reliability. This work introduces TAPA (Training-free Adaptation of Programmatic Agents), a novel framework that positions large language models (LLMs) as intelligent moderators of the symbolic action space. Unlike prior programmatic agents that typically generate a monolithic policy program or rely on fixed symbolic action sets, TAPA synthesizes and adapts modular programs for individual high-level actions, referred to as logical primitives. By decoupling strategic intent from execution, TAPA enables meta-agents to operate over an abstract, interpretable action space while the LLM dynamically generates, composes, and refines symbolic programs tailored to each primitive. Extensive experiments across cybersecurity and swarm intelligence domains validate TAPA's effectiveness. In autonomous DDoS defense scenarios, TAPA achieves 77.7% network uptime while maintaining near-perfect detection accuracy in unknown dynamic environments. In swarm intelligence formation control under environmental and adversarial disturbances, TAPA consistently preserves consensus at runtime where baseline methods fail completely. This work promotes a paradigm shift for autonomous system design in evolving environments, from policy adaptation to dynamic action adaptation. |
| 2025-08-15 | [An Exploratory Study on Crack Detection in Concrete through Human-Robot Collaboration](http://arxiv.org/abs/2508.11404v1) | Junyeon Kim, Tianshu Ruan et al. | Structural inspection in nuclear facilities is vital for maintaining operational safety and integrity. Traditional methods of manual inspection pose significant challenges, including safety risks, high cognitive demands, and potential inaccuracies due to human limitations. Recent advancements in Artificial Intelligence (AI) and robotic technologies have opened new possibilities for safer, more efficient, and accurate inspection methodologies. Specifically, Human-Robot Collaboration (HRC), leveraging robotic platforms equipped with advanced detection algorithms, promises significant improvements in inspection outcomes and reductions in human workload. This study explores the effectiveness of AI-assisted visual crack detection integrated into a mobile Jackal robot platform. The experiment results indicate that HRC enhances inspection accuracy and reduces operator workload, resulting in potential superior performance outcomes compared to traditional manual methods. |
| 2025-08-15 | [Retrieval-augmented reasoning with lean language models](http://arxiv.org/abs/2508.11386v1) | Ryan Sze-Yin Chan, Federico Nanni et al. | This technical report details a novel approach to combining reasoning and retrieval augmented generation (RAG) within a single, lean language model architecture. While existing RAG systems typically rely on large-scale models and external APIs, our work addresses the increasing demand for performant and privacy-preserving solutions deployable in resource-constrained or secure environments. Building on recent developments in test-time scaling and small-scale reasoning models, we develop a retrieval augmented conversational agent capable of interpreting complex, domain-specific queries using a lightweight backbone model. Our system integrates a dense retriever with fine-tuned Qwen2.5-Instruct models, using synthetic query generation and reasoning traces derived from frontier models (e.g., DeepSeek-R1) over a curated corpus, in this case, the NHS A-to-Z condition pages. We explore the impact of summarisation-based document compression, synthetic data design, and reasoning-aware fine-tuning on model performance. Evaluation against both non-reasoning and general-purpose lean models demonstrates that our domain-specific fine-tuning approach yields substantial gains in answer accuracy and consistency, approaching frontier-level performance while remaining feasible for local deployment. All implementation details and code are publicly released to support reproducibility and adaptation across domains. |
| 2025-08-15 | [Hyperspectral vs. RGB for Pedestrian Segmentation in Urban Driving Scenes: A Comparative Study](http://arxiv.org/abs/2508.11301v1) | Jiarong Li, Imad Ali Shah et al. | Pedestrian segmentation in automotive perception systems faces critical safety challenges due to metamerism in RGB imaging, where pedestrians and backgrounds appear visually indistinguishable.. This study investigates the potential of hyperspectral imaging (HSI) for enhanced pedestrian segmentation in urban driving scenarios using the Hyperspectral City v2 (H-City) dataset. We compared standard RGB against two dimensionality-reduction approaches by converting 128-channel HSI data into three-channel representations: Principal Component Analysis (PCA) and optimal band selection using Contrast Signal-to-Noise Ratio with Joint Mutual Information Maximization (CSNR-JMIM). Three semantic segmentation models were evaluated: U-Net, DeepLabV3+, and SegFormer. CSNR-JMIM consistently outperformed RGB with an average improvements of 1.44% in Intersection over Union (IoU) and 2.18% in F1-score for pedestrian segmentation. Rider segmentation showed similar gains with 1.43% IoU and 2.25% F1-score improvements. These improved performance results from enhanced spectral discrimination of optimally selected HSI bands effectively reducing false positives. This study demonstrates robust pedestrian segmentation through optimal HSI band selection, showing significant potential for safety-critical automotive applications. |
| 2025-08-15 | [SafeConstellations: Steering LLM Safety to Reduce Over-Refusals Through Task-Specific Trajectory](http://arxiv.org/abs/2508.11290v1) | Utsav Maskey, Sumit Yadav et al. | LLMs increasingly exhibit over-refusal behavior, where safety mechanisms cause models to reject benign instructions that superficially resemble harmful content. This phenomena diminishes utility in production applications that repeatedly rely on common prompt templates or applications that frequently rely on LLMs for specific tasks (e.g. sentiment analysis, language translation). Through comprehensive evaluation, we demonstrate that LLMs still tend to refuse responses to harmful instructions when those instructions are reframed to appear as benign tasks. Our mechanistic analysis reveal that LLMs follow distinct "constellation" patterns in embedding space as representations traverse layers, with each task maintaining consistent trajectories that shift predictably between refusal and non-refusal cases. We introduce SafeConstellations, an inference-time trajectory-shifting approach that tracks task-specific trajectory patterns and guides representations toward non-refusal pathways. By selectively guiding model behavior only on tasks prone to over-refusal, and by preserving general model behavior, our method reduces over-refusal rates by up to 73% with minimal impact on utility-offering a principled approach to mitigating over-refusals. |
| 2025-08-15 | [ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection](http://arxiv.org/abs/2508.11281v1) | Axel Delaval, Shujian Yang et al. | Detecting toxic content using language models is crucial yet challenging. While substantial progress has been made in English, toxicity detection in French remains underdeveloped, primarily due to the lack of culturally relevant, large-scale datasets. In this work, we introduce TOXIFRENCH, a new public benchmark of 53,622 French online comments, constructed via a semi-automated annotation pipeline that reduces manual labeling to only 10% through high-confidence LLM-based pre-annotation and human verification. Then, we benchmark a broad range of models and uncover a counterintuitive insight: Small Language Models (SLMs) outperform many larger models in robustness and generalization under the toxicity detection task. Motivated by this finding, we propose a novel Chain-of-Thought (CoT) fine-tuning strategy using a dynamic weighted loss that progressively emphasizes the model's final decision, significantly improving faithfulness. Our fine-tuned 4B model achieves state-of-the-art performance, improving its F1 score by 13% over its baseline and outperforming LLMs such as GPT-40 and Gemini-2.5. Further evaluation on a cross-lingual toxicity benchmark demonstrates strong multilingual ability, suggesting that our methodology can be effectively extended to other languages and safety-critical classification tasks. |
| 2025-08-15 | [LETToT: Label-Free Evaluation of Large Language Models On Tourism Using Expert Tree-of-Thought](http://arxiv.org/abs/2508.11280v1) | Ruiyan Qi, Congding Wen et al. | Evaluating large language models (LLMs) in specific domain like tourism remains challenging due to the prohibitive cost of annotated benchmarks and persistent issues like hallucinations. We propose $\textbf{L}$able-Free $\textbf{E}$valuation of LLM on $\textbf{T}$ourism using Expert $\textbf{T}$ree-$\textbf{o}$f-$\textbf{T}$hought (LETToT), a framework that leverages expert-derived reasoning structures-instead of labeled data-to access LLMs in tourism. First, we iteratively refine and validate hierarchical ToT components through alignment with generic quality dimensions and expert feedback. Results demonstrate the effectiveness of our systematically optimized expert ToT with 4.99-14.15\% relative quality gains over baselines. Second, we apply LETToT's optimized expert ToT to evaluate models of varying scales (32B-671B parameters), revealing: (1) Scaling laws persist in specialized domains (DeepSeek-V3 leads), yet reasoning-enhanced smaller models (e.g., DeepSeek-R1-Distill-Llama-70B) close this gap; (2) For sub-72B models, explicit reasoning architectures outperform counterparts in accuracy and conciseness ($p<0.05$). Our work established a scalable, label-free paradigm for domain-specific LLM evaluation, offering a robust alternative to conventional annotated benchmarks. |
| 2025-08-15 | [Hallucination in LLM-Based Code Generation: An Automotive Case Study](http://arxiv.org/abs/2508.11257v1) | Marc Pavel, Nenad Petrovic et al. | Large Language Models (LLMs) have shown significant potential in automating code generation tasks offering new opportunities across software engineering domains. However, their practical application remains limited due to hallucinations - outputs that appear plausible but are factually incorrect, unverifiable or nonsensical. This paper investigates hallucination phenomena in the context of code generation with a specific focus on the automotive domain. A case study is presented that evaluates multiple code LLMs for three different prompting complexities ranging from a minimal one-liner prompt to a prompt with Covesa Vehicle Signal Specifications (VSS) as additional context and finally to a prompt with an additional code skeleton. The evaluation reveals a high frequency of syntax violations, invalid reference errors and API knowledge conflicts in state-of-the-art models GPT-4.1, Codex and GPT-4o. Among the evaluated models, only GPT-4.1 and GPT-4o were able to produce a correct solution when given the most context-rich prompt. Simpler prompting strategies failed to yield a working result, even after multiple refinement iterations. These findings highlight the need for effective mitigation techniques to ensure the safe and reliable use of LLM generated code, especially in safety-critical domains such as automotive software systems. |
| 2025-08-15 | [ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal](http://arxiv.org/abs/2508.11222v1) | Haonan Zhang, Dongxia Wang et al. | Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems. |
| 2025-08-15 | [Visuomotor Grasping with World Models for Surgical Robots](http://arxiv.org/abs/2508.11200v1) | Hongbin Lin, Bin Li et al. | Grasping is a fundamental task in robot-assisted surgery (RAS), and automating it can reduce surgeon workload while enhancing efficiency, safety, and consistency beyond teleoperated systems. Most prior approaches rely on explicit object pose tracking or handcrafted visual features, limiting their generalization to novel objects, robustness to visual disturbances, and the ability to handle deformable objects. Visuomotor learning offers a promising alternative, but deploying it in RAS presents unique challenges, such as low signal-to-noise ratio in visual observations, demands for high safety and millimeter-level precision, as well as the complex surgical environment. This paper addresses three key challenges: (i) sim-to-real transfer of visuomotor policies to ex vivo surgical scenes, (ii) visuomotor learning using only a single stereo camera pair -- the standard RAS setup, and (iii) object-agnostic grasping with a single policy that generalizes to diverse, unseen surgical objects without retraining or task-specific models. We introduce Grasp Anything for Surgery V2 (GASv2), a visuomotor learning framework for surgical grasping. GASv2 leverages a world-model-based architecture and a surgical perception pipeline for visual observations, combined with a hybrid control system for safe execution. We train the policy in simulation using domain randomization for sim-to-real transfer and deploy it on a real robot in both phantom-based and ex vivo surgical settings, using only a single pair of endoscopic cameras. Extensive experiments show our policy achieves a 65% success rate in both settings, generalizes to unseen objects and grippers, and adapts to diverse disturbances, demonstrating strong performance, generality, and robustness. |
| 2025-08-15 | [Geometry-Aware Predictive Safety Filters on Humanoids: From Poisson Safety Functions to CBF Constrained MPC](http://arxiv.org/abs/2508.11129v1) | Ryan M. Bena, Gilbert Bahati et al. | Autonomous navigation through unstructured and dynamically-changing environments is a complex task that continues to present many challenges for modern roboticists. In particular, legged robots typically possess manipulable asymmetric geometries which must be considered during safety-critical trajectory planning. This work proposes a predictive safety filter: a nonlinear model predictive control (MPC) algorithm for online trajectory generation with geometry-aware safety constraints based on control barrier functions (CBFs). Critically, our method leverages Poisson safety functions to numerically synthesize CBF constraints directly from perception data. We extend the theoretical framework for Poisson safety functions to incorporate temporal changes in the domain by reformulating the static Dirichlet problem for Poisson's equation as a parameterized moving boundary value problem. Furthermore, we employ Minkowski set operations to lift the domain into a configuration space that accounts for robot geometry. Finally, we implement our real-time predictive safety filter on humanoid and quadruped robots in various safety-critical scenarios. The results highlight the versatility of Poisson safety functions, as well as the benefit of CBF constrained model predictive safety-critical controllers. |
| 2025-08-15 | [AI Agentic Programming: A Survey of Techniques, Challenges, and Opportunities](http://arxiv.org/abs/2508.11126v1) | Huanting Wang, Jingzhi Gong et al. | AI agentic programming is an emerging paradigm in which large language models (LLMs) autonomously plan, execute, and interact with external tools like compilers, debuggers, and version control systems to iteratively perform complex software development tasks. Unlike conventional code generation tools, agentic systems are capable of decomposing high-level goals, coordinating multi-step processes, and adapting their behavior based on intermediate feedback. These capabilities are transforming the software development practice. As this emerging field evolves rapidly, there is a need to define its scope, consolidate its technical foundations, and identify open research challenges. This survey provides a comprehensive and timely review of AI agentic programming. We introduce a taxonomy of agent behaviors and system architectures, and examine core techniques including planning, memory and context management, tool integration, and execution monitoring. We also analyze existing benchmarks and evaluation methodologies used to assess coding agent performance. Our study identifies several key challenges, including limitations in handling long context, a lack of persistent memory across tasks, and concerns around safety, alignment with user intent, and collaboration with human developers. We discuss emerging opportunities to improve the reliability, adaptability, and transparency of agentic systems. By synthesizing recent advances and outlining future directions, this survey aims to provide a foundation for research and development in building the next generation of intelligent and trustworthy AI coding agents. |
| 2025-08-14 | [Managing Risks from Large Digital Loads Using Coordinated Grid-Forming Storage Network](http://arxiv.org/abs/2508.11080v1) | Soumya Kundu, Kaustav Chatterjee et al. | Anticipated rapid growth of large digital load, driven by artificial intelligence (AI) data centers, is poised to increase uncertainty and large fluctuations in consumption, threatening the stability, reliability, and security of the energy infrastructure. Conventional measures taken by grid planners and operators to ensure stable and reliable integration of new resources are either cost-prohibitive (e.g., transmission upgrades) or ill-equipped (e.g., generation control) to resolve the unique challenges brought on by AI Data Centers (e.g., extreme load transients). In this work, we explore the feasibility of coordinating and managing available flexibility in the grid, in terms of grid-forming storage units, to ensure stable and reliable integration of AI Data Centers without the need for costly grid upgrades. Recently developed bi-layered coordinated control strategies -- involving fast-acting, local, autonomous, control at the storage to maintain transient safety in voltage and frequency at the point-of-interconnection, and a slower, coordinated (consensus) control to restore normal operating condition in the grid -- are used in the case studies. A comparison is drawn between broadly two scenarios: a network of coordinated, smaller, distributed storage vs. larger storage installations collocated with large digital loads. IEEE 68-bus network is used for the case studies, with large digital load profiles drawn from the MIT Supercloud Dataset. |
| 2025-08-14 | [Families' Vision of Generative AI Agents for Household Safety Against Digital and Physical Threats](http://arxiv.org/abs/2508.11030v1) | Zikai Wen, Lanjing Liu et al. | As families face increasingly complex safety challenges in digital and physical environments, generative AI (GenAI) presents new opportunities to support household safety through multiple specialized AI agents. Through a two-phase qualitative study consisting of individual interviews and collaborative sessions with 13 parent-child dyads, we explored families' conceptualizations of GenAI and their envisioned use of AI agents in daily family life. Our findings reveal that families preferred to distribute safety-related support across multiple AI agents, each embodying a familiar caregiving role: a household manager coordinating routine tasks and mitigating risks such as digital fraud and home accidents; a private tutor providing personalized educational support, including safety education; and a family therapist offering emotional support to address sensitive safety issues such as cyberbullying and digital harassment. Families emphasized the need for agent-specific privacy boundaries, recognized generational differences in trust toward AI agents, and stressed the importance of maintaining open family communication alongside the assistance of AI agents. Based on these findings, we propose a multi-agent system design featuring four privacy-preserving principles: memory segregation, conversational consent, selective data sharing, and progressive memory management to help balance safety, privacy, and autonomy within family contexts. |
| 2025-08-14 | [Can Multi-modal (reasoning) LLMs detect document manipulation?](http://arxiv.org/abs/2508.11021v1) | Zisheng Liang, Kidus Zewde et al. | Document fraud poses a significant threat to industries reliant on secure and verifiable documentation, necessitating robust detection mechanisms. This study investigates the efficacy of state-of-the-art multi-modal large language models (LLMs)-including OpenAI O1, OpenAI 4o, Gemini Flash (thinking), Deepseek Janus, Grok, Llama 3.2 and 4, Qwen 2 and 2.5 VL, Mistral Pixtral, and Claude 3.5 and 3.7 Sonnet-in detecting fraudulent documents. We benchmark these models against each other and prior work on document fraud detection techniques using a standard dataset with real transactional documents. Through prompt optimization and detailed analysis of the models' reasoning processes, we evaluate their ability to identify subtle indicators of fraud, such as tampered text, misaligned formatting, and inconsistent transactional sums. Our results reveal that top-performing multi-modal LLMs demonstrate superior zero-shot generalization, outperforming conventional methods on out-of-distribution datasets, while several vision LLMs exhibit inconsistent or subpar performance. Notably, model size and advanced reasoning capabilities show limited correlation with detection accuracy, suggesting task-specific fine-tuning is critical. This study underscores the potential of multi-modal LLMs in enhancing document fraud detection systems and provides a foundation for future research into interpretable and scalable fraud mitigation strategies. |

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



