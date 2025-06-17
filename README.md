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
| 2025-06-16 | [ExtendAttack: Attacking Servers of LRMs via Extending Reasoning](http://arxiv.org/abs/2506.13737v1) | Zhenhao Zhu, Yue Liu et al. | Large Reasoning Models (LRMs) have demonstrated promising performance in complex tasks. However, the resource-consuming reasoning processes may be exploited by attackers to maliciously occupy the resources of the servers, leading to a crash, like the DDoS attack in cyber. To this end, we propose a novel attack method on LRMs termed ExtendAttack to maliciously occupy the resources of servers by stealthily extending the reasoning processes of LRMs. Concretely, we systematically obfuscate characters within a benign prompt, transforming them into a complex, poly-base ASCII representation. This compels the model to perform a series of computationally intensive decoding sub-tasks that are deeply embedded within the semantic structure of the query itself. Extensive experiments demonstrate the effectiveness of our proposed ExtendAttack. Remarkably, it increases the length of the model's response by over 2.5 times for the o3 model on the HumanEval benchmark. Besides, it preserves the original meaning of the query and achieves comparable answer accuracy, showing the stealthiness. |
| 2025-06-16 | [Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs](http://arxiv.org/abs/2506.13727v1) | Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer et al. | Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at https://github.com/erfanhatefi/SparC3. |
| 2025-06-16 | [Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models](http://arxiv.org/abs/2506.13726v1) | Arjun Krishna, Aaditya Rastogi et al. | The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques. |
| 2025-06-16 | [We Should Identify and Mitigate Third-Party Safety Risks in MCP-Powered Agent Systems](http://arxiv.org/abs/2506.13666v1) | Junfeng Fang, Zijun Yao et al. | The development of large language models (LLMs) has entered in a experience-driven era, flagged by the emergence of environment feedback-driven learning via reinforcement learning and tool-using agents. This encourages the emergenece of model context protocol (MCP), which defines the standard on how should a LLM interact with external services, such as \api and data. However, as MCP becomes the de facto standard for LLM agent systems, it also introduces new safety risks. In particular, MCP introduces third-party services, which are not controlled by the LLM developers, into the agent systems. These third-party MCP services provider are potentially malicious and have the economic incentives to exploit vulnerabilities and sabotage user-agent interactions. In this position paper, we advocate the research community in LLM safety to pay close attention to the new safety risks issues introduced by MCP, and develop new techniques to build safe MCP-powered agent systems. To establish our position, we argue with three key parts. (1) We first construct \framework, a controlled framework to examine safety issues in MCP-powered agent systems. (2) We then conduct a series of pilot experiments to demonstrate the safety risks in MCP-powered agent systems is a real threat and its defense is not trivial. (3) Finally, we give our outlook by showing a roadmap to build safe MCP-powered agent systems. In particular, we would call for researchers to persue the following research directions: red teaming, MCP safe LLM development, MCP safety evaluation, MCP safety data accumulation, MCP service safeguard, and MCP safe ecosystem construction. We hope this position paper can raise the awareness of the research community in MCP safety and encourage more researchers to join this important research direction. Our code is available at https://github.com/littlelittlenine/SafeMCP.git. |
| 2025-06-16 | [Quantifying stored energy release in irradiated YBa$_2$Cu$_3$O$_7$ through molecular dynamics annealing simulations](http://arxiv.org/abs/2506.13625v1) | Lauryn Kortman, Alexis Devitre et al. | Over the lifetime of a fusion power plant, irradiation-induced defects will accumulate in the superconducting magnets compromising their ability to carry current without losses, generate high magnetic fields, and thus maintain plasma confinement. These defects also store potential energy within the crystalline lattice of materials, which can be released upon annealing. This phenomenon raises the question of whether the energy stored in defects may be sufficient to accelerate, or even trigger, a magnet quench? To provide an order of magnitude estimate, we used molecular dynamics simulations to generate defected YBCO supercells and conduct isothermal annealing simulations. Our results reveal that the maximum volumetric stored energy in a 4 mDPA defected single crystal of YBCO (240 $J/cm^3$) is 30 times greater than the experimental minimum quench energy values for YBCO tapes (8.1 $J/cm^3$). Our simulations also show that the amount of energy released increases as a function of annealing temperature or irradiation dose. This trend demonstrates that localized heating events in an irradiated fusion magnet have the potential to release significant amounts of defect energy. These findings underscore the critical need for experimental validation of the accumulation and release of defect stored energy, and highlight the importance of incorporating this contribution into quench detection systems, to enhance the operational safety of large-scale YBCO fusion magnets. |
| 2025-06-16 | [Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates](http://arxiv.org/abs/2506.13622v1) | Martino Gulisano, Matteo Masoni et al. | This paper presents a disturbance-aware framework that embeds robustness into minimum-lap-time trajectory optimization for motorsport. Two formulations are introduced. (i) Open-loop, horizon-based covariance propagation uses worst-case uncertainty growth over a finite window to tighten tire-friction and track-limit constraints. (ii) Closed-loop, covariance-aware planning incorporates a time-varying LQR feedback law in the optimizer, providing a feedback-consistent estimate of disturbance attenuation and enabling sharper yet reliable constraint tightening. Both methods yield reference trajectories for human or artificial drivers: in autonomous applications the modelled controller can replicate the on-board implementation, while for human driving accuracy increases with the extent to which the driver can be approximated by the assumed time-varying LQR policy. Computational tests on a representative Barcelona-Catalunya sector show that both schemes meet the prescribed safety probability, yet the closed-loop variant incurs smaller lap-time penalties than the more conservative open-loop solution, while the nominal (non-robust) trajectory remains infeasible under the same uncertainties. By accounting for uncertainty growth and feedback action during planning, the proposed framework delivers trajectories that are both performance-optimal and probabilistically safe, advancing minimum-time optimization toward real-world deployment in high-performance motorsport and autonomous racing. |
| 2025-06-16 | [Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs](http://arxiv.org/abs/2506.13593v1) | Hen Davidov, Gilad Freidkin et al. | We develop a framework to quantify the time-to-unsafe-sampling - the number of large language model (LLM) generations required to trigger an unsafe (e.g., toxic) response. Estimating this quantity is challenging, since unsafe responses are exceedingly rare in well-aligned LLMs, potentially occurring only once in thousands of generations. As a result, directly estimating time-to-unsafe-sampling would require collecting training data with a prohibitively large number of generations per prompt. However, with realistic sampling budgets, we often cannot generate enough responses to observe an unsafe outcome for every prompt, leaving the time-to-unsafe-sampling unobserved in many cases, making the estimation and evaluation tasks particularly challenging. To address this, we frame this estimation problem as one of survival analysis and develop a provably calibrated lower predictive bound (LPB) on the time-to-unsafe-sampling of a given prompt, leveraging recent advances in conformal prediction. Our key innovation is designing an adaptive, per-prompt sampling strategy, formulated as a convex optimization problem. The objective function guiding this optimized sampling allocation is designed to reduce the variance of the estimators used to construct the LPB, leading to improved statistical efficiency over naive methods that use a fixed sampling budget per prompt. Experiments on both synthetic and real data support our theoretical results and demonstrate the practical utility of our method for safety risk assessment in generative AI models. |
| 2025-06-16 | [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](http://arxiv.org/abs/2506.13585v1) | MiniMax, : et al. | We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-attention reasoning model. MiniMax-M1 is powered by a hybrid Mixture-of-Experts (MoE) architecture combined with a lightning attention mechanism. The model is developed based on our previous MiniMax-Text-01 model, which contains a total of 456 billion parameters with 45.9 billion parameters activated per token. The M1 model natively supports a context length of 1 million tokens, 8x the context size of DeepSeek R1. Furthermore, the lightning attention mechanism in MiniMax-M1 enables efficient scaling of test-time compute. These properties make M1 particularly suitable for complex tasks that require processing long inputs and thinking extensively. MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks. We publicly release MiniMax-M1 at https://github.com/MiniMax-AI/MiniMax-M1. |
| 2025-06-16 | [BattBee: Equivalent Circuit Modeling and Early Detection of Thermal Runaway Triggered by Internal Short Circuits for Lithium-Ion Batteries](http://arxiv.org/abs/2506.13577v1) | Sangwon Kang, Hao Tu et al. | Lithium-ion batteries are the enabling power source for transportation electrification. However, in real-world applications, they remain vulnerable to internal short circuits (ISCs) and the consequential risk of thermal runaway (TR). Toward addressing the challenge of ISCs and TR, we undertake a systematic study that extends from dynamic modeling to fault detection in this paper. First, we develop {\em BattBee}, the first equivalent circuit model to specifically describe the onset of ISCs and the evolution of subsequently induced TR. Drawing upon electrochemical modeling, the model can simulate ISCs at different severity levels and predict their impact on the initiation and progression of TR events. With the physics-inspired design, this model offers strong physical interpretability and predictive accuracy, while maintaining structural simplicity to allow fast computation. Then, building upon the BattBee model, we develop fault detection observers and derive detection criteria together with decision-making logics to identify the occurrence and emergence of ISC and TR events. This detection approach is principled in design and fast in computation, lending itself to practical applications. Validation based on simulations and experimental data demonstrates the effectiveness of both the BattBee model and the ISC/TR detection approach. The research outcomes underscore this study's potential for real-world battery safety risk management. |
| 2025-06-16 | [Safe-Child-LLM: A Developmental Benchmark for Evaluating LLM Safety in Child-AI Interactions](http://arxiv.org/abs/2506.13510v1) | Junfeng Jiao, Saleh Afroogh et al. | As Large Language Models (LLMs) increasingly power applications used by children and adolescents, ensuring safe and age-appropriate interactions has become an urgent ethical imperative. Despite progress in AI safety, current evaluations predominantly focus on adults, neglecting the unique vulnerabilities of minors engaging with generative AI. We introduce Safe-Child-LLM, a comprehensive benchmark and dataset for systematically assessing LLM safety across two developmental stages: children (7-12) and adolescents (13-17). Our framework includes a novel multi-part dataset of 200 adversarial prompts, curated from red-teaming corpora (e.g., SG-Bench, HarmBench), with human-annotated labels for jailbreak success and a standardized 0-5 ethical refusal scale. Evaluating leading LLMs -- including ChatGPT, Claude, Gemini, LLaMA, DeepSeek, Grok, Vicuna, and Mistral -- we uncover critical safety deficiencies in child-facing scenarios. This work highlights the need for community-driven benchmarks to protect young users in LLM interactions. To promote transparency and collaborative advancement in ethical AI development, we are publicly releasing both our benchmark datasets and evaluation codebase at https://github.com/The-Responsible-AI-Initiative/Safe_Child_LLM_Benchmark.git |
| 2025-06-16 | [UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data](http://arxiv.org/abs/2506.13505v1) | Vasiliki Balaska, Ioannis Tsampikos Papapetros et al. | The mining sector increasingly adopts digital tools to improve operational efficiency, safety, and data-driven decision-making. One of the key challenges remains the reliable acquisition of high-resolution, geo-referenced spatial information to support core activities such as extraction planning and on-site monitoring. This work presents an integrated system architecture that combines UAV-based sensing, LiDAR terrain modeling, and deep learning-based object detection to generate spatially accurate information for open-pit mining environments. The proposed pipeline includes geo-referencing, 3D reconstruction, and object localization, enabling structured spatial outputs to be integrated into an industrial digital twin platform. Unlike traditional static surveying methods, the system offers higher coverage and automation potential, with modular components suitable for deployment in real-world industrial contexts. While the current implementation operates in post-flight batch mode, it lays the foundation for real-time extensions. The system contributes to the development of AI-enhanced remote sensing in mining by demonstrating a scalable and field-validated geospatial data workflow that supports situational awareness and infrastructure safety. |
| 2025-06-16 | [Navigating through CS1: The Role of Self-Regulation and Supervision in Student Progress](http://arxiv.org/abs/2506.13461v1) | Ville Isomöttönen, Denis Zhidkikh | The need for students' self-regulation for fluent transitioning to university studies is known. Our aim was to integrate study-supportive activities with course supervision activities within CS1. We educated TAs to pay attention to students' study ability and self-regulation. An interview study ($N=14$) was undertaken to investigate this approach. A thematic analysis yielded rather mixed results in light of our aims. Self-regulation was underpinned by the influences external to our setting, including labor market-related needs, earlier crises in study habits, and personal characteristics such as passion, grit, creativity, and valuation of utility. Safety in one-to-one supervision was considered essential, while shyness, fear, and even altruism caused self-handicapping during the course. Students were aware of their learning styles and need for self-regulation, while did not always know how to self-regulate or preferred to externalize it. The results highlight that supporting self-regulation should be integrated with students' personal histories and experiences, and thereby calls attention to transformative learning pedagogies. The thematization can help to understand CS1 students' self-regulation processes and improve CS1 support practices. |
| 2025-06-16 | [Leveraging Vision-Language Pre-training for Human Activity Recognition in Still Images](http://arxiv.org/abs/2506.13458v1) | Cristina Mahanta, Gagan Bhatia | Recognising human activity in a single photo enables indexing, safety and assistive applications, yet lacks motion cues. Using 285 MSCOCO images labelled as walking, running, sitting, and standing, scratch CNNs scored 41% accuracy. Fine-tuning multimodal CLIP raised this to 76%, demonstrating that contrastive vision-language pre-training decisively improves still-image action recognition in real-world deployments. |
| 2025-06-16 | [Flavoured jet algorithms: a comparative study](http://arxiv.org/abs/2506.13449v1) | Arnd Behring, Simone Caletti et al. | The accurate identification of heavy-flavour jets, those which originate from bottom or charm quarks, is crucial for precision studies of the Standard Model and searches for new physics. However, assigning flavour to jets presents significant challenges, primarily due to issues with infrared and collinear (IRC) safety. This paper aims to address these challenges by evaluating recently-proposed jet algorithms designed to be IRC-safe and applicable in high-precision measurements. We compare these algorithms across benchmark heavy-flavour production processes and kinematic regimes that are relevant for LHC phenomenology. Exploiting both fixed-order calculations in QCD as well as parton shower simulations, we analyse the infrared sensitivity of these new algorithms at different stages of the event evolution and compare to flavour-labelling strategies currently adopted by LHC collaborations. The results highlight that, while all algorithms lead to more robust flavour-assignments compared to current techniques, they vary in performance depending on the observable and energy regime. The study lays groundwork for robust, flavour-aware jet analyses in current and future collider experiments to maximise the physics potential of experimental data by reducing discrepancies between theoretical and experimental methods. |
| 2025-06-16 | [Sparse Convolutional Recurrent Learning for Efficient Event-based Neuromorphic Object Detection](http://arxiv.org/abs/2506.13440v1) | Shenqi Wang, Yingfu Xu et al. | Leveraging the high temporal resolution and dynamic range, object detection with event cameras can enhance the performance and safety of automotive and robotics applications in real-world scenarios. However, processing sparse event data requires compute-intensive convolutional recurrent units, complicating their integration into resource-constrained edge applications. Here, we propose the Sparse Event-based Efficient Detector (SEED) for efficient event-based object detection on neuromorphic processors. We introduce sparse convolutional recurrent learning, which achieves over 92% activation sparsity in recurrent processing, vastly reducing the cost for spatiotemporal reasoning on sparse event data. We validated our method on Prophesee's 1 Mpx and Gen1 event-based object detection datasets. Notably, SEED sets a new benchmark in computational efficiency for event-based object detection which requires long-term temporal learning. Compared to state-of-the-art methods, SEED significantly reduces synaptic operations while delivering higher or same-level mAP. Our hardware simulations showcase the critical role of SEED's hardware-aware design in achieving energy-efficient and low-latency neuromorphic processing. |
| 2025-06-16 | [A data-driven analysis of the impact of non-compliant individuals on epidemic diffusion in urban settings](http://arxiv.org/abs/2506.13325v1) | Fabio Mazza, Marco Brambilla et al. | Individuals who do not comply with public health safety measures pose a significant challenge to effective epidemic control, as their risky behaviours can undermine public health interventions. This is particularly relevant in urban environments because of their high population density and complex social interactions. In this study, we employ detailed contact networks, built using a data-driven approach, to examine the impact of non-compliant individuals on epidemic dynamics in three major Italian cities: Torino, Milano, and Palermo. We use a heterogeneous extension of the Susceptible-Infected-Recovered model that distinguishes between ordinary and non-compliant individuals, who are more infectious and/or more susceptible. By combining electoral data with recent findings on vaccine hesitancy, we obtain spatially heterogeneous distributions of non-compliance. Epidemic simulations demonstrate that even a small proportion of non-compliant individuals in the population can substantially increase the number of infections and accelerate the timing of their peak. Furthermore, the impact of non-compliance is greatest when disease transmission rates are moderate. Including the heterogeneous, data-driven distribution of non-compliance in the simulations results in infection hotspots forming with varying intensity according to the disease transmission rate. Overall, these findings emphasise the importance of monitoring behavioural compliance and tailoring public health interventions to address localised risks. |
| 2025-06-16 | [Mitigating Safety Fallback in Editing-based Backdoor Injection on LLMs](http://arxiv.org/abs/2506.13285v1) | Houcheng Jiang, Zetong Zhao et al. | Large language models (LLMs) have shown strong performance across natural language tasks, but remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying parameters to map specific triggers to attacker-desired responses. However, these methods often suffer from safety fallback, where the model initially responds affirmatively but later reverts to refusals due to safety alignment. In this work, we propose DualEdit, a dual-objective model editing framework that jointly promotes affirmative outputs and suppresses refusal responses. To address two key challenges -- balancing the trade-off between affirmative promotion and refusal suppression, and handling the diversity of refusal expressions -- DualEdit introduces two complementary techniques. (1) Dynamic loss weighting calibrates the objective scale based on the pre-edited model to stabilize optimization. (2) Refusal value anchoring compresses the suppression target space by clustering representative refusal value vectors, reducing optimization conflict from overly diverse token sets. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 9.98\% and reduces safety fallback rate by 10.88\% over baselines. |
| 2025-06-16 | [Ai-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns](http://arxiv.org/abs/2506.13172v1) | Evgeny Markhasin | We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks: identifying unsubstantiated claims in summaries (informational integrity) and flagging ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. In a summary-only setting, however, ChatGPT achieved a perfect (100%) success rate, while Gemini's performance was substantially degraded. Our findings suggest that structured prompting is a viable methodology for complex textual analysis but show that prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing. |
| 2025-06-16 | [Crime Hotspot Prediction Using Deep Graph Convolutional Networks](http://arxiv.org/abs/2506.13116v1) | Tehreem Zubair, Syeda Kisaa Fatima et al. | Crime hotspot prediction is critical for ensuring urban safety and effective law enforcement, yet it remains challenging due to the complex spatial dependencies inherent in criminal activity. The previous approaches tended to use classical algorithms such as the KDE and SVM to model data distributions and decision boundaries. The methods often fail to capture these spatial relationships, treating crime events as independent and ignoring geographical interactions. To address this, we propose a novel framework based on Graph Convolutional Networks (GCNs), which explicitly model spatial dependencies by representing crime data as a graph. In this graph, nodes represent discrete geographic grid cells and edges capture proximity relationships. Using the Chicago Crime Dataset, we engineer spatial features and train a multi-layer GCN model to classify crime types and predict high-risk zones. Our approach achieves 88% classification accuracy, significantly outperforming traditional methods. Additionally, the model generates interpretable heat maps of crime hotspots, demonstrating the practical utility of graph-based learning for predictive policing and spatial criminology. |
| 2025-06-16 | [Rethinking Explainability in the Era of Multimodal AI](http://arxiv.org/abs/2506.13060v1) | Chirag Agarwal | While multimodal AI systems (models jointly trained on heterogeneous data types such as text, time series, graphs, and images) have become ubiquitous and achieved remarkable performance across high-stakes applications, transparent and accurate explanation algorithms are crucial for their safe deployment and ensure user trust. However, most existing explainability techniques remain unimodal, generating modality-specific feature attributions, concepts, or circuit traces in isolation and thus failing to capture cross-modal interactions. This paper argues that such unimodal explanations systematically misrepresent and fail to capture the cross-modal influence that drives multimodal model decisions, and the community should stop relying on them for interpreting multimodal models. To support our position, we outline key principles for multimodal explanations grounded in modality: Granger-style modality influence (controlled ablations to quantify how removing one modality changes the explanation for another), Synergistic faithfulness (explanations capture the model's predictive power when modalities are combined), and Unified stability (explanations remain consistent under small, cross-modal perturbations). This targeted shift to multimodal explanations will help the community uncover hidden shortcuts, mitigate modality bias, improve model reliability, and enhance safety in high-stakes settings where incomplete explanations can have serious consequences. |

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



