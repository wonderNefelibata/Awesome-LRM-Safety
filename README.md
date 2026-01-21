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
| 2026-01-20 | [Attention-Based Offline Reinforcement Learning and Clustering for Interpretable Sepsis Treatment](http://arxiv.org/abs/2601.14228v1) | Punit Kumar, Vaibhav Saran et al. | Sepsis remains one of the leading causes of mortality in intensive care units, where timely and accurate treatment decisions can significantly impact patient outcomes. In this work, we propose an interpretable decision support framework. Our system integrates four core components: (1) a clustering-based stratification module that categorizes patients into low, intermediate, and high-risk groups upon ICU admission, using clustering with statistical validation; (2) a synthetic data augmentation pipeline leveraging variational autoencoders (VAE) and diffusion models to enrich underrepresented trajectories such as fluid or vasopressor administration; (3) an offline reinforcement learning (RL) agent trained using Advantage Weighted Regression (AWR) with a lightweight attention encoder and supported by an ensemble models for conservative, safety-aware treatment recommendations; and (4) a rationale generation module powered by a multi-modal large language model (LLM), which produces natural-language justifications grounded in clinical context and retrieved expert knowledge. Evaluated on the MIMIC-III and eICU datasets, our approach achieves high treatment accuracy while providing clinicians with interpretable and robust policy recommendations. |
| 2026-01-20 | [Rig-Aware 3D Reconstruction of Vehicle Undercarriages using Gaussian Splatting](http://arxiv.org/abs/2601.14208v1) | Nitin Kulkarni, Akhil Devarashetti et al. | Inspecting the undercarriage of used vehicles is a labor-intensive task that requires inspectors to crouch or crawl underneath each vehicle to thoroughly examine it. Additionally, online buyers rarely see undercarriage photos. We present an end-to-end pipeline that utilizes a three-camera rig to capture videos of the undercarriage as the vehicle drives over it, and produces an interactive 3D model of the undercarriage. The 3D model enables inspectors and customers to rotate, zoom, and slice through the undercarriage, allowing them to detect rust, leaks, or impact damage in seconds, thereby improving both workplace safety and buyer confidence. Our primary contribution is a rig-aware Structure-from-Motion (SfM) pipeline specifically designed to overcome the challenges of wide-angle lens distortion and low-parallax scenes. Our method overcomes the challenges of wide-angle lens distortion and low-parallax scenes by integrating precise camera calibration, synchronized video streams, and strong geometric priors from the camera rig. We use a constrained matching strategy with learned components, the DISK feature extractor, and the attention-based LightGlue matcher to generate high-quality sparse point clouds that are often unattainable with standard SfM pipelines. These point clouds seed the Gaussian splatting process to generate photorealistic undercarriage models that render in real-time. Our experiments and ablation studies demonstrate that our design choices are essential to achieve state-of-the-art quality. |
| 2026-01-20 | [A model of errors in transformers](http://arxiv.org/abs/2601.14175v1) | Suvrat Raju, Praneeth Netrapalli | We study the error rate of LLMs on tasks like arithmetic that require a deterministic output, and repetitive processing of tokens drawn from a small set of alternatives. We argue that incorrect predictions arise when small errors in the attention mechanism accumulate to cross a threshold, and use this insight to derive a quantitative two-parameter relationship between the accuracy and the complexity of the task. The two parameters vary with the prompt and the model; they can be interpreted in terms of an elementary noise rate, and the number of plausible erroneous tokens that can be predicted. Our analysis is inspired by an ``effective field theory'' perspective: the LLM's many raw parameters can be reorganized into just two parameters that govern the error rate. We perform extensive empirical tests, using Gemini 2.5 Flash, Gemini 2.5 Pro and DeepSeek R1, and find excellent agreement between the predicted and observed accuracy for a variety of tasks, although we also identify deviations in some cases. Our model provides an alternative to suggestions that errors made by LLMs on long repetitive tasks indicate the ``collapse of reasoning'', or an inability to express ``compositional'' functions. Finally, we show how to construct prompts to reduce the error rate. |
| 2026-01-20 | [An Empirical Study on Remote Code Execution in Machine Learning Model Hosting Ecosystems](http://arxiv.org/abs/2601.14163v1) | Mohammed Latif Siddiq, Tanzim Hossain Romel et al. | Model-sharing platforms, such as Hugging Face, ModelScope, and OpenCSG, have become central to modern machine learning development, enabling developers to share, load, and fine-tune pre-trained models with minimal effort. However, the flexibility of these ecosystems introduces a critical security concern: the execution of untrusted code during model loading (i.e., via trust_remote_code or trust_repo). In this work, we conduct the first large-scale empirical study of custom model loading practices across five major model-sharing platforms to assess their prevalence, associated risks, and developer perceptions. We first quantify the frequency with which models require custom code to function and identify those that execute arbitrary Python files during loading. We then apply three complementary static analysis tools: Bandit, CodeQL, and Semgrep, to detect security smells and potential vulnerabilities, categorizing our findings by CWE identifiers to provide a standardized risk taxonomy. We also use YARA to identify malicious patterns and payload signatures. In parallel, we systematically analyze the documentation, API design, and safety mechanisms of each platform to understand their mitigation strategies and enforcement levels. Finally, we conduct a qualitative analysis of over 600 developer discussions from GitHub, Hugging Face, and PyTorch Hub forums, as well as Stack Overflow, to capture community concerns and misconceptions regarding security and usability. Our findings reveal widespread reliance on unsafe defaults, uneven security enforcement across platforms, and persistent confusion among developers about the implications of executing remote code. We conclude with actionable recommendations for designing safer model-sharing infrastructures and striking a balance between usability and security in future AI ecosystems. |
| 2026-01-20 | [The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning](http://arxiv.org/abs/2601.14127v1) | Renmiao Chen, Yida Lu et al. | As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark focused on multi-image reasoning safety, which consists of 2,676 instances across a taxonomy of 9 multi-image relations. Our extensive evaluations on 19 MLLMs reveal a troubling trend: models with more advanced multi-image reasoning can be more vulnerable on MIR-SafetyBench. Beyond attack success rates, we find that many responses labeled as safe are superficial, often driven by misunderstanding or evasive, non-committal replies. We further observe that unsafe generations exhibit lower attention entropy than safe ones on average. This internal signature suggests a possible risk that models may over-focus on task solving while neglecting safety constraints. Our code and data are available at https://github.com/thu-coai/MIR-SafetyBench. |
| 2026-01-20 | [Communication Technologies for Intelligent Transportation Systems: From Railways to UAVs and Beyond](http://arxiv.org/abs/2601.14106v1) | Shrief Rizkalla, Adrian Kliks et al. | This white paper aims to comprehensively analyze and consolidate the state of the art in communication technologies supporting modern and future Information and Communication Technology (ICT). Its primary objective is to establish a common understanding of how communication solutions enable automation, safety, and efficiency across multiple transport domains, including railways, road vehicles, aircraft, and unmanned aerial vehicles. The document seeks to identify key communication requirements and technological enablers necessary for interoperable and reliable ITS operation. It also assesses the limitations of current systems and proposes pathways for integrating emerging technologies such as 5G, Sixth Generation (6G), and Artificial Intelligence (AI)-driven network control. The white paper also intends to support harmonization between different transport modes through a unified framework for communication modeling, testing, and standardization. It highlights the importance of accurate channel modeling and empirical validation to design efficient, robust, and scalable systems. Another objective is to explore the use of reconfigurable intelligent surfaces, integrated sensing and communication, and digital twin concepts within ITS. The document emphasizes the role of spectrum management and standardization efforts in ensuring interoperability among diverse communication systems. Finally, the paper seeks to stimulate collaboration among academia, industry, and standardization bodies to advance the design of resilient and adaptive communication infrastructures for future transportation systems. |
| 2026-01-20 | [Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning](http://arxiv.org/abs/2601.14104v1) | Tairan Huang, Qingqing Ye et al. | Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material. |
| 2026-01-20 | [Zero-shot adaptable task planning for autonomous construction robots: a comparative study of lightweight single and multi-AI agent systems](http://arxiv.org/abs/2601.14091v1) | Hossein Naderi, Alireza Shojaei et al. | Robots are expected to play a major role in the future construction industry but face challenges due to high costs and difficulty adapting to dynamic tasks. This study explores the potential of foundation models to enhance the adaptability and generalizability of task planning in construction robots. Four models are proposed and implemented using lightweight, open-source large language models (LLMs) and vision language models (VLMs). These models include one single agent and three multi-agent teams that collaborate to create robot action plans. The models are evaluated across three construction roles: Painter, Safety Inspector, and Floor Tiling. Results show that the four-agent team outperforms the state-of-the-art GPT-4o in most metrics while being ten times more cost-effective. Additionally, teams with three and four agents demonstrate the improved generalizability. By discussing how agent behaviors influence outputs, this study enhances the understanding of AI teams and supports future research in diverse unstructured environments beyond construction. |
| 2026-01-20 | [Data-Driven Safe Output Regulation of Strict-Feedback Linear Systems with Input Delay](http://arxiv.org/abs/2601.14089v1) | Zhenxu Zhao, Ji Wang et al. | This paper develops a data-driven safe control framework for linear systems possessing a known strict-feedback structure, but with most plant parameters, external disturbances, and input delay being unknown. By leveraging Koopman operator theory, we utilize Krylov dynamic mode decomposition (DMD) to extract the system dynamics from measured data, enabling the reconstruction of the system and disturbance matrices. Concurrently, the batch least-squares identification (BaLSI) method is employed to identify other unknown parameters in the input channel. Using control barrier functions (CBFs) and backstepping, we first develop a full-state safe controller. Based on this, we build an output-feedback controller by performing system identification using only the output data and actuation signals as well as constructing an observer to estimate the unmeasured plant states. The proposed approach achieves: 1) finite-time identification of a substantial set of unknown system quantities, and 2) exponential convergence of the output state (the state furthest from the control input) to a reference trajectory while rigorously ensuring safety constraints. The effectiveness of the proposed method is demonstrated through a safe vehicle platooning application. |
| 2026-01-20 | [LLMOrbit: A Circular Taxonomy of Large Language Models -From Scaling Walls to Agentic AI Systems](http://arxiv.org/abs/2601.14053v1) | Badri N. Patro, Vijay S. Agneeswaran | The field of artificial intelligence has undergone a revolution from foundational Transformer architectures to reasoning-capable systems approaching human-level performance. We present LLMOrbit, a comprehensive circular taxonomy navigating the landscape of large language models spanning 2019-2025. This survey examines over 50 models across 15 organizations through eight interconnected orbital dimensions, documenting architectural innovations, training methodologies, and efficiency patterns defining modern LLMs, generative AI, and agentic systems. We identify three critical crises: (1) data scarcity (9-27T tokens depleted by 2026-2028), (2) exponential cost growth ($3M to $300M+ in 5 years), and (3) unsustainable energy consumption (22x increase), establishing the scaling wall limiting brute-force approaches. Our analysis reveals six paradigms breaking this wall: (1) test-time compute (o1, DeepSeek-R1 achieve GPT-4 performance with 10x inference compute), (2) quantization (4-8x compression), (3) distributed edge computing (10x cost reduction), (4) model merging, (5) efficient training (ORPO reduces memory 50%), and (6) small specialized models (Phi-4 14B matches larger models). Three paradigm shifts emerge: (1) post-training gains (RLHF, GRPO, pure RL contribute substantially, DeepSeek-R1 achieving 79.8% MATH), (2) efficiency revolution (MoE routing 18x efficiency, Multi-head Latent Attention 8x KV cache compression enables GPT-4-level performance at <$0.30/M tokens), and (3) democratization (open-source Llama 3 88.6% MMLU surpasses GPT-4 86.4%). We provide insights into techniques (RLHF, PPO, DPO, GRPO, ORPO), trace evolution from passive generation to tool-using agents (ReAct, RAG, multi-agent systems), and analyze post-training innovations. |
| 2026-01-20 | [VirtualCrime: Evaluating Criminal Potential of Large Language Models via Sandbox Simulation](http://arxiv.org/abs/2601.13981v1) | Yilin Tang, Yu Wang et al. | Large language models (LLMs) have shown strong capabilities in multi-step decision-making, planning and actions, and are increasingly integrated into various real-world applications. It is concerning whether their strong problem-solving abilities may be misused for crimes. To address this gap, we propose VirtualCrime, a sandbox simulation framework based on a three-agent system to evaluate the criminal capabilities of models. Specifically, this framework consists of an attacker agent acting as the leader of a criminal team, a judge agent determining the outcome of each action, and a world manager agent updating the environment state and entities. Furthermore, we design 40 diverse crime tasks within this framework, covering 11 maps and 13 crime objectives such as theft, robbery, kidnapping, and riot. We also introduce a human player baseline for reference to better interpret the performance of LLM agents. We evaluate 8 strong LLMs and find (1) All agents in the simulation environment compliantly generate detailed plans and execute intelligent crime processes, with some achieving relatively high success rates; (2) In some cases, agents take severe action that inflicts harm to NPCs to achieve their goals. Our work highlights the need for safety alignment when deploying agentic AI in real-world settings. |
| 2026-01-20 | [VulnResolver: A Hybrid Agent Framework for LLM-Based Automated Vulnerability Issue Resolution](http://arxiv.org/abs/2601.13933v1) | Mingming Zhang, Xu Wang et al. | As software systems grow in complexity, security vulnerabilities have become increasingly prevalent, posing serious risks and economic costs. Although automated detection tools such as fuzzers have advanced considerably, effective resolution still often depends on human expertise. Existing automated vulnerability repair (AVR) methods rely heavily on manually provided annotations (e.g., fault locations or CWE labels), which are often difficult and time-consuming to obtain, while overlooking the rich, naturally embedded semantic context found in issue reports from developers.   In this paper, we present VulnResolver, the first LLM-based hybrid agent framework for automated vulnerability issue resolution. VulnResolver unites the adaptability of autonomous agents with the stability of workflow-guided repair through two specialized agents. The Context Pre-Collection Agent (CPCAgent) adaptively explores the repository to gather dependency and contextual information, while the Safety Property Analysis Agent (SPAAgent) generates and validates the safety properties violated by vulnerabilities. Together, these agents produce structured analyses that enrich the original issue reports, enabling more accurate vulnerability localization and patch generation.   Evaluations on the SEC-bench benchmark show that VulnResolver resolves 75% of issues on SEC-bench Lite, achieving the best resolution performance. On SEC-bench Full, VulnResolver also significantly outperforms the strongest baseline, the agent-based OpenHands, confirming its effectiveness. Overall, VulnResolver delivers an adaptive and security-aware framework that advances end-to-end automated vulnerability issue resolution through workflow stability and the specialized agents' capabilities in contextual reasoning and property-based analysis. |
| 2026-01-20 | [Towards Inclusive External Human-Machine Interface: Exploring the Effects of Visual and Auditory eHMI for Deaf and Hard-of-Hearing People](http://arxiv.org/abs/2601.13889v1) | Wenge Xu, Foroogh Hajiseyedjavadi et al. | External Human-Machine Interfaces (eHMIs) have been proposed to facilitate communication between Automated Vehicles (AVs) and pedestrians. However, no attention was given to Deaf and Hard-of-Hearing (DHH) people. We conducted a formative study through focus groups with 6 DHH people and 6 key stakeholders (including researchers, assistive technologists, and automotive interface designers) to compare proposed eHMIs and extract key design requirements. Subsequently, we investigated the effects of visual and auditory eHMI in a virtual reality user study with 32 participants (16 DHH). Results from our scenario suggesting that (1) DHH participants spent more time looking at the AV; (2) both visual and auditory eHMIs enhanced trust, usefulness, and perceived safety; and (3) only visual eHMIs reduced the time to step into the road, time looking at the AV, gaze time, and percentage looking at active visual eHMI components. Lastly, we provided five practical implications for making eHMI inclusive of DHH people. |
| 2026-01-20 | [Pedagogical Alignment for Vision-Language-Action Models: A Comprehensive Framework for Data, Architecture, and Evaluation in Education](http://arxiv.org/abs/2601.13876v1) | Unggi Lee, Jahyun Jeong et al. | Science demonstrations are important for effective STEM education, yet teachers face challenges in conducting them safely and consistently across multiple occasions, where robotics can be helpful. However, current Vision-Language-Action (VLA) models require substantial computational resources and sacrifice language generation capabilities to maximize efficiency, making them unsuitable for resource-constrained educational settings that require interpretable, explanation-generating systems. We present \textit{Pedagogical VLA Framework}, a framework that applies pedagogical alignment to lightweight VLA models through four components: text healing to restore language generation capabilities, large language model (LLM) distillation to transfer pedagogical knowledge, safety training for educational environments, and pedagogical evaluation adjusted to science education contexts. We evaluate Pedagogical VLA Framework across five science demonstrations spanning physics, chemistry, biology, and earth science, using an evaluation framework developed in collaboration with science education experts. Our evaluation assesses both task performance (success rate, protocol compliance, efficiency, safety) and pedagogical quality through teacher surveys and LLM-as-Judge assessment. We additionally provide qualitative analysis of generated texts. Experimental results demonstrate that Pedagogical VLA Framework achieves comparable task performance to baseline models while producing contextually appropriate educational explanations. |
| 2026-01-20 | [Designing Drone Interfaces to Assist Pedestrians Crossing Non-Signalised Roads](http://arxiv.org/abs/2601.13858v1) | Guixiang Zhang, Yiyuan Wang et al. | Recent research highlights the potential of drones to enhance pedestrian experiences, such as aiding navigation and supporting street-level activities. This paper explores the design of drone interfaces to assist pedestrians crossing dangerous roads without designated crosswalks or traffic lights, leveraging drones' ability to monitor and analyse real-time traffic data. Inspired by existing traffic signal systems, the interface communicates safety information through permissive alerts, prohibitive warnings, directional warnings, and collision emergency warnings. These safety cues were integrated into drone interfaces using in-situ projections and drone-equipped screens through an iterative design process. A mixed-methods, within-subjects VR evaluation (n=18) revealed that drone-assisted systems significantly improved pedestrian safety experiences and reduced mental workload compared to a baseline without any crossing aid, with projections outperforming screens. The findings suggest the potential for drone interfaces to be integrated into connected traffic systems. We also offer design recommendations for developing drone interfaces that support safe pedestrian crossings. |
| 2026-01-20 | [Small Models, Big Impact: Tool-Augmented AI Agents for Wireless Network Planning](http://arxiv.org/abs/2601.13843v1) | Yongqiang Zhang, Mustafa A. Kishk et al. | Large Language Models (LLMs) such as ChatGPT promise revolutionary capabilities for Sixth-Generation (6G) wireless networks but their massive computational requirements and tendency to generate technically incorrect information create deployment barriers. In this work, we introduce MAINTAINED: autonomous artificial intelligence agent for wireless network deployment. Instead of encoding domain knowledge within model parameters, our approach orchestrates specialized computational tools for geographic analysis, signal propagation modeling, and network optimization. In a real-world case study, MAINTAINED outperforms state-of-the-art LLMs including ChatGPT-4o, Claude Sonnet 4, and DeepSeek-R1 by up to 100-fold in verified performance metrics while requiring less computational resources. This paradigm shift, moving from relying on parametric knowledge towards externalizing domain knowledge into verifiable computational tools, eliminates hallucination in technical specifications and enables edge-deployable Artificial Intelligence (AI) for wireless communications. |
| 2026-01-20 | [DisasterVQA: A Visual Question Answering Benchmark Dataset for Disaster Scenes](http://arxiv.org/abs/2601.13839v1) | Aisha Al-Mohannadi, Ayisha Firoz et al. | Social media imagery provides a low-latency source of situational information during natural and human-induced disasters, enabling rapid damage assessment and response. While Visual Question Answering (VQA) has shown strong performance in general-purpose domains, its suitability for the complex and safety-critical reasoning required in disaster response remains unclear. We introduce DisasterVQA, a benchmark dataset designed for perception and reasoning in crisis contexts. DisasterVQA consists of 1,395 real-world images and 4,405 expert-curated question-answer pairs spanning diverse events such as floods, wildfires, and earthquakes. Grounded in humanitarian frameworks including FEMA ESF and OCHA MIRA, the dataset includes binary, multiple-choice, and open-ended questions covering situational awareness and operational decision-making tasks. We benchmark seven state-of-the-art vision-language models and find performance variability across question types, disaster categories, regions, and humanitarian tasks. Although models achieve high accuracy on binary questions, they struggle with fine-grained quantitative reasoning, object counting, and context-sensitive interpretation, particularly for underrepresented disaster scenarios. DisasterVQA provides a challenging and practical benchmark to guide the development of more robust and operationally meaningful vision-language models for disaster response. The dataset is publicly available at https://zenodo.org/records/18267770. |
| 2026-01-20 | [GuideTouch: An Obstacle Avoidance Device for Visually Impaired](http://arxiv.org/abs/2601.13813v1) | Timofei Kozlov, Artem Trandofilov et al. | Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation. |
| 2026-01-20 | [An Adaptive Phase II Trial Design for Dose Selection and Addition in Microfilarial Infections](http://arxiv.org/abs/2601.13784v1) | Sonja Zehetmayer, Marta Bofill Roig et al. | We propose a frequentist adaptive phase 2 trial design to evaluate the safety and efficacy of three treatment regimens (doses) compared to placebo for four types of helminth (worm) infections. This trial will be carried out in four Subsaharan African countries from spring 2025. Since the safety of the highest dose is not yet established, the study begins with the two lower doses and placebo. Based on safety and early efficacy results from an interim analysis, a decision will be made to either continue with the two lower doses or drop one or both and introduce the highest dose instead. This design borrows information across baskets for safety assessment, while efficacy is assessed separately for each basket. The proposed adaptive design addresses several key challenges: (1) The trial must begin with only the two lower doses because reassuring safety data from these doses is required before escalating to a higher dose. (2) Due to the expected speed of recruitment, adaptation decisions must rely on an earlier, surrogate endpoint. (3) The primary outcome is a count variable that follows a mixture distribution with an atom at 0. To control the familywise error rate in the strong sense when comparing multiple doses to the control in the adaptive design, we extend the partial conditional error approach to accommodate the inclusion of new hypotheses after the interim analysis. In a comprehensive simulation study we evaluate various design options and analysis strategies, assessing the robustness of the design under different design assumptions and parameter values. We identify scenarios where the adaptive design improves the trial's ability to identify an optimal dose. Adaptive dose selection enables resource allocation to the most promising treatment arms, increasing the likelihood of selecting the optimal dose while reducing the required overall sample size and trial duration. |
| 2026-01-20 | [TransMode-LLM: Feature-Informed Natural Language Modeling with Domain-Enhanced Prompting for Travel Behavior Modeling](http://arxiv.org/abs/2601.13763v1) | Meijing Zhang, Ying Xu | Understanding traveler behavior and accurately predicting travel mode choice are at the heart of transportation planning and policy-making. This study proposes TransMode-LLM, an innovative framework that integrates statistical methods with LLM-based techniques to predict travel modes from travel survey data. The framework operates through three phases: (1) statistical analysis identifies key behavioral features, (2) natural language encoding transforms structured data into contextual descriptions, and (3) LLM adaptation predicts travel mode through multiple learning paradigms including zero-shot and one/few-shot learning and domain-enhanced prompting. We evaluate TransMode-LLM using both general-purpose models (GPT-4o, GPT-4o-mini) and reasoning-focused models (o3-mini, o4-mini) with varying sample sizes on real-world travel survey data. Extensive experiment results demonstrate that the LLM-based approach achieves competitive accuracy compared to state-of-the-art baseline classifiers models. Moreover, few-shot learning significantly improves prediction accuracy, with models like o3-mini showing consistent improvements of up to 42.9\% with 5 provided examples. However, domain-enhanced prompting shows divergent effects across LLM architectures. In detail, it is helpful to improve performance for general-purpose models with GPT-4o achieving improvements of 2.27% to 12.50%. However, for reasoning-oriented models (o3-mini, o4-mini), domain knowledge enhancement does not universally improve performance. This study advances the application of LLMs in travel behavior modeling, providing promising and valuable insights for both academic research and transportation policy-making in the future. |

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



