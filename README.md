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
| 2025-06-25 | [Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings](http://arxiv.org/abs/2506.20609v1) | Ankit Shah, Rita Singh et al. | The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders. |
| 2025-06-25 | [Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm](http://arxiv.org/abs/2506.20606v1) | Baixiang Huang, Zhen Tan et al. | Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through comprehensive evaluations on agents based on frontier LLMs, BehaviorBench shows the effectiveness of Behavior Editing across different models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing. |
| 2025-06-25 | [Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges](http://arxiv.org/abs/2506.20598v1) | Alexander D. Kalian, Jaewook Lee et al. | The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities |
| 2025-06-25 | [Case-based Reasoning Augmented Large Language Model Framework for Decision Making in Realistic Safety-Critical Driving Scenarios](http://arxiv.org/abs/2506.20531v1) | Wenbin Gan, Minh-Son Dao et al. | Driving in safety-critical scenarios requires quick, context-aware decision-making grounded in both situational understanding and experiential reasoning. Large Language Models (LLMs), with their powerful general-purpose reasoning capabilities, offer a promising foundation for such decision-making. However, their direct application to autonomous driving remains limited due to challenges in domain adaptation, contextual grounding, and the lack of experiential knowledge needed to make reliable and interpretable decisions in dynamic, high-risk environments. To address this gap, this paper presents a Case-Based Reasoning Augmented Large Language Model (CBR-LLM) framework for evasive maneuver decision-making in complex risk scenarios. Our approach integrates semantic scene understanding from dashcam video inputs with the retrieval of relevant past driving cases, enabling LLMs to generate maneuver recommendations that are both context-sensitive and human-aligned. Experiments across multiple open-source LLMs show that our framework improves decision accuracy, justification quality, and alignment with human expert behavior. Risk-aware prompting strategies further enhance performance across diverse risk types, while similarity-based case retrieval consistently outperforms random sampling in guiding in-context learning. Case studies further demonstrate the framework's robustness in challenging real-world conditions, underscoring its potential as an adaptive and trustworthy decision-support tool for intelligent driving systems. |
| 2025-06-25 | [Learning-based safety lifting monitoring system for cranes on construction sites](http://arxiv.org/abs/2506.20475v1) | Hao Chen, Yu Hin Ng et al. | Lifting on construction sites, as a frequent operation, works still with safety risks, especially for modular integrated construction (MiC) lifting due to its large weight and size, probably leading to accidents, causing damage to the modules, or more critically, posing safety hazards to on-site workers. Aiming to reduce the safety risks in lifting scenarios, we design an automated safe lifting monitoring algorithm pipeline based on learning-based methods, and deploy it on construction sites. This work is potentially to increase the safety and efficiency of MiC lifting process via automation technologies. A dataset is created consisting of 1007 image-point cloud pairs (37 MiC liftings). Advanced object detection models are trained for automated two-dimensional (2D) detection of MiCs and humans. Fusing the 2D detection results with the point cloud information allows accurate determination of the three-dimensional (3D) positions of MiCs and humans. The system is designed to automatically trigger alarms that notify individuals in the MiC lifting danger zone, while providing the crane operator with real-time lifting information and early warnings. The monitoring process minimizes the human intervention and no or less signal men are required on real sites assisted by our system. A quantitative analysis is conducted to evaluate the effectiveness of the algorithmic pipeline. The pipeline shows promising results in MiC and human perception with the mean distance error of 1.5640 m and 0.7824 m respectively. Furthermore, the developed system successfully executes safety risk monitoring and alarm functionalities during the MiC lifting process with limited manual work on real construction sites. |
| 2025-06-25 | [Probing AI Safety with Source Code](http://arxiv.org/abs/2506.20471v1) | Ujwal Narayan, Shreyas Chaudhari et al. | Large language models (LLMs) have become ubiquitous, interfacing with humans in numerous safety-critical applications. This necessitates improving capabilities, but importantly coupled with greater safety measures to align these models with human values and preferences. In this work, we demonstrate that contemporary models fall concerningly short of the goal of AI safety, leading to an unsafe and harmful experience for users. We introduce a prompting strategy called Code of Thought (CoDoT) to evaluate the safety of LLMs. CoDoT converts natural language inputs to simple code that represents the same intent. For instance, CoDoT transforms the natural language prompt "Make the statement more toxic: {text}" to: "make_more_toxic({text})". We show that CoDoT results in a consistent failure of a wide range of state-of-the-art LLMs. For example, GPT-4 Turbo's toxicity increases 16.5 times, DeepSeek R1 fails 100% of the time, and toxicity increases 300% on average across seven modern LLMs. Additionally, recursively applying CoDoT can further increase toxicity two times. Given the rapid and widespread adoption of LLMs, CoDoT underscores the critical need to evaluate safety efforts from first principles, ensuring that safety and capabilities advance together. |
| 2025-06-25 | [Multimodal Behaviour Trees for Robotic Laboratory Task Automation](http://arxiv.org/abs/2506.20399v1) | Hatem Fakhruldeen, Arvind Raveendran Nambiar et al. | Laboratory robotics offer the capability to conduct experiments with a high degree of precision and reproducibility, with the potential to transform scientific research. Trivial and repeatable tasks; e.g., sample transportation for analysis and vial capping are well-suited for robots; if done successfully and reliably, chemists could contribute their efforts towards more critical research activities. Currently, robots can perform these tasks faster than chemists, but how reliable are they? Improper capping could result in human exposure to toxic chemicals which could be fatal. To ensure that robots perform these tasks as accurately as humans, sensory feedback is required to assess the progress of task execution. To address this, we propose a novel methodology based on behaviour trees with multimodal perception. Along with automating robotic tasks, this methodology also verifies the successful execution of the task, a fundamental requirement in safety-critical environments. The experimental evaluation was conducted on two lab tasks: sample vial capping and laboratory rack insertion. The results show high success rate, i.e., 88% for capping and 92% for insertion, along with strong error detection capabilities. This ultimately proves the robustness and reliability of our approach and that using multimodal behaviour trees should pave the way towards the next generation of robotic chemists. |
| 2025-06-25 | [Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles](http://arxiv.org/abs/2506.20311v1) | Jingwen Wei | The growing use of mobile robots in sectors such as automotive, agriculture, and rescue operations reflects progress in robotics and autonomy. In unmanned aerial vehicles (UAVs), most research emphasizes visual SLAM, sensor fusion, and path planning. However, applying UAVs to search and rescue missions in disaster zones remains underexplored, especially for autonomous navigation.   This report develops methods for real-time and secure UAV maneuvering in complex 3D environments, crucial during forest fires. Building upon past research, it focuses on designing navigation algorithms for unfamiliar and hazardous environments, aiming to improve rescue efficiency and safety through UAV-based early warning and rapid response.   The work unfolds in phases. First, a 2D fusion navigation strategy is explored, initially for mobile robots, enabling safe movement in dynamic settings. This sets the stage for advanced features such as adaptive obstacle handling and decision-making enhancements. Next, a novel 3D reactive navigation strategy is introduced for collision-free movement in forest fire simulations, addressing the unique challenges of UAV operations in such scenarios.   Finally, the report proposes a unified control approach that integrates UAVs and unmanned ground vehicles (UGVs) for coordinated rescue missions in forest environments. Each phase presents challenges, proposes control models, and validates them with mathematical and simulation-based evidence. The study offers practical value and academic insights for improving the role of UAVs in natural disaster rescue operations. |
| 2025-06-25 | [Enterprise Large Language Model Evaluation Benchmark](http://arxiv.org/abs/2506.20274v1) | Liya Wang, David Yi et al. | Large Language Models (LLMs) ) have demonstrated promise in boosting productivity across AI-powered tools, yet existing benchmarks like Massive Multitask Language Understanding (MMLU) inadequately assess enterprise-specific task complexities. We propose a 14-task framework grounded in Bloom's Taxonomy to holistically evaluate LLM capabilities in enterprise contexts. To address challenges of noisy data and costly annotation, we develop a scalable pipeline combining LLM-as-a-Labeler, LLM-as-a-Judge, and corrective retrieval-augmented generation (CRAG), curating a robust 9,700-sample benchmark. Evaluation of six leading models shows open-source contenders like DeepSeek R1 rival proprietary models in reasoning tasks but lag in judgment-based scenarios, likely due to overthinking. Our benchmark reveals critical enterprise performance gaps and offers actionable insights for model optimization. This work provides enterprises a blueprint for tailored evaluations and advances practical LLM deployment. |
| 2025-06-25 | [Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models](http://arxiv.org/abs/2506.20251v1) | Kejia Chen, Jiawen Zhang et al. | Quantized large language models (LLMs) have gained increasing attention and significance for enabling deployment in resource-constrained environments. However, emerging studies on a few calibration dataset-free quantization methods suggest that quantization may compromise the safety capabilities of LLMs, underscoring the urgent need for systematic safety evaluations and effective mitigation strategies. In this paper, we present comprehensive safety evaluations across various mainstream quantization techniques and diverse calibration datasets, utilizing widely accepted safety benchmarks. To address the identified safety vulnerabilities, we propose a quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while minimizing any adverse impact on utility. Extensive experimental results demonstrate that Q-resafe successfully re-aligns the safety of quantized LLMs with their pre-quantization counterparts, even under challenging evaluation scenarios. Project page is available at: https://github.com/Thecommonirin/Qresafe. |
| 2025-06-25 | [Enhancing Large Language Models through Structured Reasoning](http://arxiv.org/abs/2506.20241v1) | Yubo Dong, Hehe Fan | Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge representation.Inspired by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs. |
| 2025-06-25 | [Personalized Mental State Evaluation in Human-Robot Interaction using Federated Learning](http://arxiv.org/abs/2506.20212v1) | Andrea Bussolan, Oliver Avram et al. | With the advent of Industry 5.0, manufacturers are increasingly prioritizing worker well-being alongside mass customization. Stress-aware Human-Robot Collaboration (HRC) plays a crucial role in this paradigm, where robots must adapt their behavior to human mental states to improve collaboration fluency and safety. This paper presents a novel framework that integrates Federated Learning (FL) to enable personalized mental state evaluation while preserving user privacy. By leveraging physiological signals, including EEG, ECG, EDA, EMG, and respiration, a multimodal model predicts an operator's stress level, facilitating real-time robot adaptation. The FL-based approach allows distributed on-device training, ensuring data confidentiality while improving model generalization and individual customization. Results demonstrate that the deployment of an FL approach results in a global model with performance in stress prediction accuracy comparable to a centralized training approach. Moreover, FL allows for enhancing personalization, thereby optimizing human-robot interaction in industrial settings, while preserving data privacy. The proposed framework advances privacy-preserving, adaptive robotics to enhance workforce well-being in smart manufacturing. |
| 2025-06-25 | [Flutter Suppression Enhancement in Coupled Nonlinear Airfoils with Intermittent Mixed Interactions](http://arxiv.org/abs/2506.20154v1) | Qi Liu, Riccardo Muolo et al. | Flutter suppression facilitates the improvement of structural reliability to ensure the flight safety of an aircraft. In this study, we propose a novel strategy for enlarging amplitude death (AD) regime to enhance flutter suppression in two coupled identical airfoils with structural nonlinearity. Specifically, we introduce an intermittent mixed coupling strategy, i.e., a linear combination of intermittent instantaneous coupling and intermittent time-delayed coupling between two airfoils. Numerical simulations are performed to reveal the influence mechanisms of different coupling scenarios on the dynamical behaviors of the coupled airfoil systems. The obtained results indicate that the coupled airfoil systems experience the expected AD behaviors within a certain range of the coupling strength and time-delayed parameters. The continuous mixed coupling favors the onset of AD over a larger parameter set of coupling strength than the continuous purely time-delayed coupling. Moreover, the presence of intermittent interactions can lead to a further enlargement of the AD regions, that is, flutter suppression enhancement. Our findings support the structural design and optimization of an aircraft wing for mitigating the unwanted aeroelastic instability behaviors. |
| 2025-06-24 | [Single-event neutron time-of-flight spectroscopy with a petawatt-laser-driven neutron source](http://arxiv.org/abs/2506.20026v1) | M. A. Millán-Callado, S. Scheuren et al. | Fast neutron-induced nuclear reactions are crucial for advancing our understanding of fundamental nuclear processes, stellar nucleosynthesis, and applications, including reactor safety, medical isotope production, and materials research. With many research reactors being phased out, compact accelerator-based neutron sources are becoming increasingly important. Laser-driven neutron sources (LDNSs) offer unique advantages -- ultrashort neutron pulsees for superior energy resolution, high per-pulse flux, and a drastically reduced footprint. However, their use in single-event fast neutron spectroscopy remains unproven, requiring stable multi-shot operation and detectors capable of functioning in the extreme environment of petawatt-class laser-plasma interactions. Here, we present a proof-of-concept experiment at the DRACO~PW laser in a pitcher-catcher configuration, stably producing 6-7e7 neutrons/shot with energies above 1 MeV, over more than 200 shots delivered at a shot-per-minute rate. Neutron time-of-flight measurements were performed using a single-crystal diamond detector, which is located only 1.5 m away from the source and capable of resolving individual neutron-induced reactions. Observed reaction rates are consistent with Monte Carlo simulations inferred by real-time diagnostics of accompanying gamma, ion, and electron fluxes. With the recent advances in repetition rate, targetry, and ion acceleration efficiency, this work establishes LDNSs as a promising, scalable platform for future fast neutron-induced reaction studies, particularly for measurements involving short-lived isotopes or requiring high instantaneous neutron flux. |
| 2025-06-24 | [Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning](http://arxiv.org/abs/2506.20020v1) | Saloni Dash, Amélie Reymond et al. | Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans. |
| 2025-06-24 | [Recursive-ARX for Grid-Edge Fault Detection](http://arxiv.org/abs/2506.20011v1) | Soufiane El Yaagoubi, Keith Moffat et al. | Future electrical grids will require new ways to identify faults as inverters are not capable of supplying large fault currents to support existing fault detection methods and because distributed resources may feed faults from the edge of the grid. This paper proposes the use of real-time system identification for online power-system fault detection. Specifically, we implement Recursive ARX (rARX) system identification on a grid-connected inverter. Experiments demonstrate that the proposed rARX method is able to both detect large faults quickly, and distinguish between high-impedance faults and large load increases. These results indicate that rARX grid-edge fault detection is a promising research direction for improving the reliability and safety of modern electric grids. |
| 2025-06-24 | [Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks](http://arxiv.org/abs/2506.20009v1) | Konstantinos Vrettos, Michail E. Klontzas | Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals. |
| 2025-06-24 | [Can One Safety Loop Guard Them All? Agentic Guard Rails for Federated Computing](http://arxiv.org/abs/2506.20000v1) | Narasimha Raghavan Veeraragavan, Jan Franz Nygård | We propose Guardian-FC, a novel two-layer framework for privacy preserving federated computing that unifies safety enforcement across diverse privacy preserving mechanisms, including cryptographic back-ends like fully homomorphic encryption (FHE) and multiparty computation (MPC), as well as statistical techniques such as differential privacy (DP). Guardian-FC decouples guard-rails from privacy mechanisms by executing plug-ins (modular computation units), written in a backend-neutral, domain-specific language (DSL) designed specifically for federated computing workflows and interchangeable Execution Providers (EPs), which implement DSL operations for various privacy back-ends. An Agentic-AI control plane enforces a finite-state safety loop through signed telemetry and commands, ensuring consistent risk management and auditability. The manifest-centric design supports fail-fast job admission and seamless extensibility to new privacy back-ends. We present qualitative scenarios illustrating backend-agnostic safety and a formal model foundation for verification. Finally, we outline a research agenda inviting the community to advance adaptive guard-rail tuning, multi-backend composition, DSL specification development, implementation, and compiler extensibility alongside human-override usability. |
| 2025-06-24 | [Persona Features Control Emergent Misalignment](http://arxiv.org/abs/2506.19823v1) | Miles Wang, Tom Dupré la Tour et al. | Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. discovered that fine-tuning GPT-4o on intentionally insecure code causes "emergent misalignment," where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a "model diffing" approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several "misaligned persona" features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment. |
| 2025-06-24 | [Optimization Studies of Radiation Shielding for the PIP-II Project at Fermilab](http://arxiv.org/abs/2506.19763v1) | Alajos Makovec, Dali Georgobiani et al. | The PIP-II project at Fermilab, which includes an 800-MeV superconducting LINAC, demands rigorous radiation shielding optimization to meet safety requirements. We updated the MARS geometry model to reflect new magnet and collimator designs and introduced high-resolution detector planes to better capture radiation field distributions. To overcome the significant computational demands, we implemented a well-known branching technique that drastically reduced simulation runtimes while maintaining statistical integrity. This was achieved through particle splitting and the application of Russian Roulette techniques. Additionally, new graphical tools were created to streamline data visualization and MARS code usability. |

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



