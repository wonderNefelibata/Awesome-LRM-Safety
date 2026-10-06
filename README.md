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
| 2026-10-05 | [Base Models Can Reason By Taking a Cue From Training Data](http://arxiv.org/abs/2610.06851v1) | Sophie L. Wang, Amil Dravid et al. | In this paper, we study how training data creates associations between the tokens at the start of a base model's response and the reasoning behavior that follows. First, we demonstrate that fixing particular starting token cues makes a base model's performance competitive with that of its reinforcement learning (RL)-trained counterparts on math and coding. For instance, the cue ".\n\nOkay" raises Olmo-3-7B's MATH-500 pass@1 accuracy from 42% to 78%, while "Alright," raises Qwen3-14B's from 72% to 87%. Second, RL makes these cues more likely, while fixing them recovers much of its performance gain over the base model. Third, we trace the reasoning effects of token cues to the training data. We perform causal data interventions to turn an arbitrary word, such as "chicken", into an effective reasoning cue, or remove an existing cue's effect. A similar edit makes the prompt instruction "Think duck duck goose" as effective as "Think step by step" at eliciting reasoning. We also find that the hidden state representations induced by different cues correlate with different document types from the training set. Finally, we extend our study of token cues with a case study in language model safety, finding that different cues elicit distinct refusal and compliance behaviors that correspond to different types of training data. |
| 2026-10-05 | [Conditional Rank Allocation for Taxonomy-Aware Medical Language Model Adaptation](http://arxiv.org/abs/2610.06765v1) | Guangyuan Dong, Ziwei Hong et al. | Medical question answering spans specialties and clinical operations that may benefit from different adaptation directions. We propose ARBOR, a parameter-efficient method that selects rank-one components from a shared low-rank basis for each question. An additive gate combines question representations, specialty tags, operation tags, and their interaction; a learned coefficient scales the adapter residual. An illustrative separation under orthogonal, equiprobable subtasks shows how conditional selection can avoid an approximation floor faced by a fixed update with the same active rank. This result motivates the design without asserting a corresponding bound for medical corpora. On Qwen3-8B across CMB, CMExam, MedQA, and MedMCQA, five-seed experiments yield 69.69% mean accuracy across benchmarks, exceeding LoRA r16 and MoELoRA by 1.26 and 1.30 percentage points, respectively. The reported advantage over LoRA r16 increases from 0.08 to 1.94 points as training expands from one to seven specialties. Tag perturbations and atom masking support the usefulness of clinical routing, while atom clusters align with the supplied specialty labels (adjusted Rand index 0.62). Calibration, transfer, and measured costs further characterize the method. These findings support structured conditional adaptation for medical QA, while leaving clinical safety and broader deployment untested. |
| 2026-10-05 | [BazaarBench: Delegation Safety in Decentralized C2C Marketplaces Run by LLM Agents](http://arxiv.org/abs/2610.06748v1) | Ziyan Wang, Shuqing Shi et al. | In decentralized consumer-to-consumer (C2C) marketplaces, people list goods, negotiate with strangers, and rate one another, so trust rests on reputation. Large language model (LLM) agents now act for users, raising risks to their money, privacy, and reputation. We introduce BazaarBench, a simulated C2C marketplace and benchmark for evaluating the safety of these agents. It tracks ownership, item condition, and commitments across transactions, combining record checks with rubric-based LLM judgments to identify six failure types across five stages. We run three base markets for 30 simulated days, each with 100 agents using one model and inventories drawn from a public eBay sample. Across 45 continuations, we evaluate five models under ordinary instructions, deadline pressure, or adversarial instructions to exploit other traders. Each continuation runs for seven simulated days from a copy of a market's day-30 state. The tested model controls the same 20 selected agents, retaining their personas, inventories, and histories, while the other 80 keep the base model. All five models attempt to promise the same item to multiple buyers under ordinary instructions. Adding targets and deadlines increases these attempts for every model. Under adversarial instructions, the share of tested sellers' committed transactions completed despite unavailable items or overstated conditions rises from 15.4% to 33.4%, reaching 55.5% for GPT-5.4. Averaged across models and markets, simulated weekly earnings per tested agent rise from USD 20 under ordinary instructions to USD 33 under adversarial instructions. Most of the increase comes from items the sellers never held. We release the simulator, saved market states, evaluation code, and records covering 357,608 agent model calls for evaluating new models and developing safer marketplace agents. |
| 2026-10-05 | [Reward Stealing Attack on Large Language Models](http://arxiv.org/abs/2610.06670v1) | Jiaming Qian, Pengyang Zhou et al. | Adversarial attacks on Large Language Models (LLMs) aim to induce harmful content. However, existing methods suffer from high computational costs or strict model-pairing dependencies, limiting their scalability and transferability. We propose Reward Stealing Attack (ReSA), an adversarial attack framework that targets the latent safety reward underlying LLM alignment. ReSA employs maximum entropy inverse reinforcement learning to recover a proxy reward model solely from the aligned model's behavior. The extracted reward is then reversed at inference time to derive an adversarial policy, efficiently implemented via a reward-guided decoding mechanism. Experiments demonstrate that a single recovered reward generalizes across prompts and diverse models to reveal a fundamental alignment vulnerability, enabling ReSA to significantly outperform existing attacks in effectiveness and transferability. The code is available at https://github.com/GarminQ/ReSA. |
| 2026-10-05 | [Vector Field Following for Quadrotors via Reduced Attitude Representations with Applications to Safety-Critical Control](http://arxiv.org/abs/2610.06629v1) | Mohammad Mirtaba, Max H. Cohen | Vector field tracking for quadrotors provides a systematic framework for specifying complex behaviors at the translational level and transferring them to the full vehicle dynamics. Objectives such as safety, for example, can be encoded into a low-dimensional vector field, which is then embedded into the full dynamics through appropriate tracking conditions. The development of such tracking controllers for quadrotors can be facilitated by the fact that their translational dynamics depend only on the thrust direction and are independent of the vehicle's heading. Thus, reduced-attitude representations, leveraging the geometry of the unit sphere, provide a natural framework for vector field tracking on quadrotors. In this work, we propose two vector field tracking controllers for quadrotors using reduced-attitude representations based on dynamic extension and geometric control, respectively. Next, we illustrate how safety can be encoded into the desired vector field using control barrier functions and enforced for the full system. The effectiveness of the proposed approaches is demonstrated via simulations and experimental flight tests. |
| 2026-10-05 | [Analysis of SWIR Imaging Detection Performance Under Adverse Environmental Conditions for Autonomous Driving Systems](http://arxiv.org/abs/2610.06596v1) | Rohan Mehra, Alexandre Riffard et al. | Short-wave infrared (SWIR) imaging has emerged as a promising modality for autonomous driving, yet its practical benefits over RGB remain poorly characterized across diverse conditions. This paper presents a systematic comparative study of paired RGB and SWIR object detection on the RASMD dataset, covering four weather conditions and two real-time detection architectures, with various fine-tunings evaluated against a unified ground truth. Overall, RGB demonstrates comparable or superior performance in most scenarios, while RF-DETR exhibits greater robustness across varying conditions. Beyond aggregate metrics, we propose a sensor-dominance mining framework that combines multi-model agreement with targeted manual inspection to identify scenarios where one sensing modality provides more reliable detections using largely unannotated paired data. This analysis reveals that SWIR offers clear advantages in four safety-critical situations, including windshield glare, water droplets on the windshield, low-contrast object visibility, and long-range vehicle detection. The findings suggest that SWIR should be viewed as a complementary modality that enhances perception in rare but challenging conditions. The datasets will be available upon request, and all code and trained model weights are publicly released at https://github.com/comsee-research/swir-adverse-env-analysis. |
| 2026-10-05 | [Preference-Adaptive Control in Autonomous Driving](http://arxiv.org/abs/2610.06539v1) | Zhongyu Mo, Shanting Wang et al. | In this paper, we present a preference-adaptive receding-horizon control framework for autonomous driving that accounts for passenger preferences and motion-sickness susceptibility. We formulate a finite-horizon optimal control problem with adaptive weights for speed, acceleration comfort, and motion sickness, while maintaining a fixed weight for collision risk. We predict motion sickness online using an individualized model on the Motion Illness Symptoms Classifi- cation (MISC) scale and update the preference weights offline from emotion-derived pairwise comparisons using Bayesian inference. A deterministic safety supervisor checks the planned trajectory and modifies the control command when necessary. We evaluate the framework using three simulated passenger profiles under three motion-sickness susceptibility levels. The learned weights yield distinct closed-loop behaviors, and the mean evaluation emotion score improves in seven of nine scenarios. No collisions occur in the full-method experiments, while the safety supervisor intervenes in 1.621% of the learning frames. |
| 2026-10-05 | [Harmful Content Generation in Text-to-Image Models: Capabilities and Moderation Limitations](http://arxiv.org/abs/2610.06503v1) | Paschalis Giakoumoglou, Manos Schinas et al. | Text-to-image generative models can produce highly realistic imagery but also raise concerns about harmful misuse. While safety mechanisms exist, systematic evaluations of their effectiveness against realistic attacks remain limited. We present a systematic evaluation of harmful content generation across five open text-to-image models using an automated pipeline that transforms legitimate news captions into unsafe prompts targeting sexually explicit content, violence/gore, harmful stereotypes, self-harm, and hate speech. We evaluate both standard models with built-in safety mechanisms and community fine-tuned variants that bypass content restrictions. A human evaluation of 1,500 generated images shows high harmful-content generation rates: 89.2% for gore-related prompts, 47.6% for sexually explicit content, 43.6% for harmful stereotypes, 46.0% for hate speech, and 34.5% for self-harm, predominantly through graphic violence. Models show substantial capability for generating violent and stereotypical content, while community fine-tuned variants are particularly vulnerable to sexually explicit prompts. Generation quality is largely preserved under harmful prompting, producing imagery of sufficient fidelity to pose risks for disinformation and abuse; FLUX.1-dev produces clearly realistic harmful images in 30.9% of cases. We further evaluate automated moderation systems and find substantial detection gaps that allow unsafe images to evade filtering. Finally, we assess synthetic image detectors and show that models trained only on benign datasets perform worse on explicit content, while more diverse training data improves detection, highlighting semantic distribution gaps in current approaches. These findings expose limitations in current generation safeguards, moderation systems, and synthetic image detection, highlighting the need for stronger defenses against misuse at scale. |
| 2026-10-05 | [Multimodal Safety Evaluation Should Measure Controllability Beyond Classification](http://arxiv.org/abs/2610.06452v1) | Junhyeong Park, Hanwool Lee et al. | VLM safety is commonly evaluated through input- and output-level classification. Such classification is necessary, but it does not reveal whether a safety state is accessible or controllable inside the model. We argue that multimodal safety evaluation should therefore report a \emph{controllability profile} alongside behavioral classification, separating representation-level detectability, cross-modal specificity, intervention sensitivity, and benign-preserving selectivity. Using implicit toxicity as a stress case, we instantiate this profile on LlavaGuard and Qwen3.5 with sparse feature decompositions. LlavaGuard admits localized handles with a narrow benign-preserving intervention range and modest downstream safety gains, whereas Qwen3.5 supports strong representation-level readout but no comparable selective-control regime under the tested operators. These results show that internal readout and controllability can diverge. Future multimodal safety benchmarks should therefore report not only behavioral safety metrics, but also whether safety-relevant internal signals can be intervention-tested and controlled within a validated operating range. |
| 2026-10-05 | [Ontology Concept Overlap as a Training Signal: Knowledge-Grounded Reinforcement Learning for Clinical Question Answering](http://arxiv.org/abs/2610.06360v1) | Aditya Tanna, Abhishek Jindal | Reinforcement learning post-training for language models relies on two reward designs: human preferences (RLHF, DPO) and binary verifiers (RLVR). Clinical question answering fits neither. Near-correct answers differ by a single substituted entity, and no executable check decides clinical correctness. We instantiate a soft verifier from a maintained controlled vocabulary: UMLS Concept Unique Identifier overlap (via scispaCy, set-level F1) gives a graded, externally specified reward computed without a model in the loop. We combine it inside GRPO with an entropy-normalised LLM judge, which covers the safety and evidence axes overlap cannot see, and a small consistency penalty on padding and repetition that keeps early-training samples scorable. This three-term composite improves over SFT on Phi-3-mini (3.8B) over MedQA by 2.9% on EM (0.700 vs 0.680) and 39% on Token-F1 (0.202 vs 0.145); on Llama-3.2-3B the corresponding gains are 14% on EM and 35% on Token-F1. We report Token-F1 as the primary metric because it credits partially-correct clinical content that EM discards at this open-generation scale. Main-table results are means over 3 seeds with standard deviations below 0.005. The method transfers to PubMedQA, where training on the PubMedQA train set with the same composite reward improves Token-F1 over SFT by 22% on Phi-3-mini and 17% on Llama-3.2-3B without retuning. A reward ablation on Phi-3, varying the judge-ontology split at a fixed consistency weight, attributes 3 EM points to the ontology term, the contribution that catches entity substitutions the judge cannot. Three negative findings constrain the design: DPO under random negatives underperforms SFT for strong-prior models but helps the weakest-prior one; PPO under a sparse neural reward diverges; GRPO with KL-in-loss collapses at 7B. |
| 2026-10-05 | [Computing Stable Matchings under Complementarities and Preference Misalignment](http://arxiv.org/abs/2610.06337v1) | Tatsuya Iwase, Bahar Rastegari et al. | We study many-to-one, two-sided stable matching problems in which preferences are complementary and firms and workers may rank the same allocation differently. Here a coalition is a group of workers that can be jointly matched with a firm. With complementary preferences, a stable matching need not exist. We show that a stable matching exists for every instance when two conditions hold. The first requires each firm to have an anchor worker who is included in every feasible coalition that can be matched with that firm. The second is Transitive Alignment, which means there must be an overall ranking across different coalitions that is consistent with the workers' preferences. We propose CDAR (Combinatorial Deferred Acceptance with Reproposals), a Deferred Acceptance-style algorithm that permits reproposals, and prove its finite-time convergence and its output of a stable matching by constructing an $N$-digit potential function. Moreover, we show that, under strict preferences, the output of CDAR is Pareto optimal among stable matchings. The proposed framework naturally captures applications such as vehicle-route assignment for traffic safety and the misaligned interests of labor and management in wage negotiation. |
| 2026-10-05 | [Inspect Robots: Evaluating the Capabilities and Safety of Embodied AI](http://arxiv.org/abs/2610.06306v1) | Christopher Leet, Achu Menon et al. | General purpose language models are increasingly able to control robotic hardware. Understanding the capabilities and safety of these models when embodied is therefore increasingly important for understanding their societal impact and risks. To this end, we introduce Inspect Robots, a modular, open-source framework for developing and running evaluations of embodied agents. Inspect Robots pairs customizable, reusable abstractions for specifying physical evaluations and analyzing their results with infrastructure that automates evaluation setup, execution and termination. We demonstrate Inspect Robots by using it to evaluate the capabilities and safety of six policies based on frontier language models. Inspect Robots has seen significant early uptake, receiving nearly 100,000 downloads in the three months since its release. |
| 2026-10-05 | [First-Principles Investigation of Multimodal Toxic Gas Sensing in Carbon-Tuned hBN-Graphene Alloys: Chemiresistive, Work-Function, and Optical Responses](http://arxiv.org/abs/2610.06283v1) | Tanjuma Shikder Jhumu, Ahmed Zubair | Compact, reliable, and cost-effective gas sensors have become a highly demanding subject for safety management in the medical sector, chemical manufacturing, food quality monitoring, agriculture, and industrial safety. Hazardous gas emissions need to be controlled and monitored with fast-responsive and highly sensitive sensing devices. A first-principles study employing density functional theory (DFT) was used to investigate the adsorption behavior of Cl2, CO, CO2, NO, NO2, and HCN gas molecules with our proposed alloys, which consisted of hexagonal boron nitride (hBN) and graphene (Gr). The alloy consisting of 22% carbon (BNGr-2) was found to be most competent for sensing Cl2, CO, CO2, and HCN with sufficient adsorption energy, charge transfer, and bandgap alteration. However, NO and NO2 gas molecules showed more engagement with 33% carbon-proportioned alloy (BNGr-3) in terms of adequate gas sensing properties. NOx gases exhibited the most chemiresistive sensitivity towards the adsorbents. Other gases also showed significant chemiresistive sensitivity and distinct selectivity ratios, which would facilitate these alloys as chemiresistive sensors. Besides, noticeable work function variation (~20%) of these systems manifested potential as work function based sensors. Cl2 and NO2 showed strong physical adsorption, while the rest of the gases were weakly to moderately physisorbed, resulting in very short recovery times (10-1 ~ 10-6 seconds). Additionally, the distinctive absorption spectra observed for the gas analyte systems highlighted the potential of the proposed alloys as optical gas sensors. Temperature variation revealed that all gas molecules can be freed from the adsorbent BNGr-2 at 425 K. These findings imply hBN-Gr alloys as promising gas sensors for pollution auditing. |
| 2026-10-05 | [Structured Representation Learning for Behavior Cloning: How can we learn to safely control a nuclear power plant?](http://arxiv.org/abs/2610.06211v1) | Perceval Beja-Battais, Alain Grosset{ê}te et al. | Learned models for industrial control are usually judged by aggregate accuracy, but accuracy at the component level does not guarantee safety once it is embedded in the system it is meant to serve. We study this gap on a behavior-cloning task: imitating an expert Nonlinear Model Predictive Control (NMPC) policy for load-following of a Pressurized Water Reactor (PWR), an industrial system with tight safety constraints. We propose a structured architecture encoding variables from each timescale into separate latent spaces, reflecting the physical decomposition of the system, before training a controller to imitate the expert on the product latent space. On long-horizon rollouts, separated embeddings improve both accuracy and feasibility compared with a shared-embedding baseline. Sensitivity analysis further shows that our model yields interpretable representations aligned with the system's physics. However, standalone deployment still leaves several percent of trajectories infeasible regardless of the architecture. Using our method to warmstart the NMPC optimizer rather than acting standalone, we recover full feasibility and near-optimal cost while still cutting computation time by $\sim$15% relative to the expert controller, and even more for abrupt operating changes. |
| 2026-10-05 | [Controllable and Photorealistic Pedestrian Risky Motion Generation for End-to-End Driving Safety Evaluation](http://arxiv.org/abs/2610.06171v1) | Siyuan Liu, Miao Li et al. | Evaluating end-to-end autonomous driving under rare, safety-critical vehicle-pedestrian interactions requires photorealistic, sensor-level scenarios. However, trajectory-based scenario generators cannot synthesize raw visual observations, whereas video-based approaches lack controllability. To bridge this gap, we present ControlPed, a novel framework that combines trajectory-level conflict synthesis with 3D Gaussian Splatting (3DGS) to generate photorealistic, motion-controllable safety-critical scenarios. Built upon HazardPed, a dataset derived from 10,352 traffic videos comprising 422 conflict trajectories, HD maps, and 857 annotated 3D human motions, ControlPed first generates conflict trajectories, lifts them into 3D human motion sequences via text-conditioned motion diffusion, and finally renders multi-view sensor observations using animatable 3DGS avatars. Safety evaluation in 88 rendered photorealistic scenarios reveals that seven leading end-to-end driving models suffer a severe performance drop, with their mean HDScore plunging from 88.8 to 47.4, exposing major failure modes under dangerous pedestrian behaviors. The dataset and testing benchmarks will be released to facilitate safety assessment of vehicle-pedestrian interactions. |
| 2026-10-05 | [Benchmarking Jailbreak Guardrails for Embodied Agents](http://arxiv.org/abs/2610.06122v1) | Xunguang Wang, Qingyue Wang et al. | Embodied agents powered by large language models and vision-language models are increasingly deployed in physical environments, but jailbreak attacks can induce these agents to perform physically harmful actions. A growing number of guardrail methods have been proposed to intercept dangerous behavior before it is executed, yet existing safety benchmarks evaluate the embodied models themselves, leaving it unclear how well these guardrails actually defend an embodied agent in practice. We present the first systematic evaluation of jailbreak guardrails for embodied agents. To compare guardrails under identical conditions, we build a pluggable evaluation framework that treats the embodied agent as a fixed backend and each guardrail as a module that can intervene at the perception, planning, or control stage. We subject six representative guardrails to template-based and automated jailbreak attacks as well as safe instructions, and assess them at the system level along three dimensions: defense effectiveness, measured by the bypass rate and the hazard success rate in the simulator; usability, measured by the false-positive rate and the task completion rate on safe instructions; and efficiency, measured by the latency overhead added at runtime. Experiments on guardrails that span different intervention stages, decision mechanisms, and input modalities reveal a clear trade-off among the three dimensions, and show that no single guardrail dominates in all settings. We further analyze how intervention stage, decision mechanism, and input modality shape safety outcomes, and we offer practical guidance for selecting and designing guardrails for embodied agents. |
| 2026-10-05 | [TrustMI: Causally controlling how assistants trust their users](http://arxiv.org/abs/2610.06064v1) | Théo Lasnier, Romain Froger et al. | Large Language Model (LLM) assistants routinely decide whether they can trust users and third parties whose competence, intentions, and integrity they cannot verify. This uncertainty matters for safety, as trusting the wrong party can lead an agent to comply with harmful requests or act on malicious instructions encountered during tool use. To study this problem, we define trust as an assistant's willingness to accept vulnerability to the actions of another party and ask whether such behavior can be causally controlled through model activations. We build 2,000 contrastive conversations spanning ability, benevolence, and integrity, where paired responses complete the same request but differ in whether the assistant trusts the user. From these pairs, we learn steering matrices while keeping the model parameters frozen and test them across six instruction-tuned models from three families, finding that steering changes trust decisions monotonically in both directions. We then ask whether this effect extends to several safety-related agent settings involving harmful requests, prompt injections, and insider threats, while using benign-task and reasoning as controls. Our findings provide evidence that trust in the user can be causally controlled along linear directions in model activations and provide a way to study how trust shapes safety-relevant behavior in language models. |
| 2026-10-05 | [AI-Driven XR Situational Awareness Platform for Urban Crisis Management and Smart Mobility Operations](http://arxiv.org/abs/2610.06051v1) | Dimitris Spyridonidis, Gerasimos Arvanitis et al. | Urban environments are increasingly exposed to complex, dynamic, and interdependent risks, ranging from traffic incidents and infrastructure failures to large-scale crises such as extreme weather events and emergency response scenarios. In such conditions, decision-makers and operators are required to act under time-critical constraints while relying on fragmented, heterogeneous, and often incomplete information. The lack of unified situational awareness, combined with limited visibility and disconnected systems, significantly affects response time, coordination efficiency, and overall operational effectiveness. In parallel, modern cities have deployed extensive sensing infrastructures, including surveillance camera networks, IoT devices, connected vehicles, and satellite-based observation systems. Although these technologies generate vast amounts of data, their exploitation remains limited due to the absence of integrated platforms capable of real time data fusion, intelligent interpretation, and intuitive visualization. As a result, a substantial gap persists between data availability and actionable intelligence, particularly in safety-critical and crisis management applications. To address this challenge, this paper presents an AI-driven XR situational awareness and operational platform designed for urban crisis management and smart mobility operations. The proposed system integrates data from city infrastructure, connected vehicles, and VRUs, enabling a comprehensive and real-time understanding of the urban environment. Through a unified operational dashboard and AR interfaces, the platform supports both centralized monitoring and field-level interaction. By enhancing perception, enabling cooperative awareness, and delivering context-aware information, the proposed approach improves decision-making, coordination, and safety across diverse urban scenarios. |
| 2026-10-05 | [AgentSpy: Making AI Agent Behavior Observable](http://arxiv.org/abs/2610.06001v1) | Christoph Bühler, Matteo Biagiola et al. | AI agents built on large language models (LLMs) run shell commands, read and write files, and reach the network, typically with their user's privileges. However, what an agent does during an execution is difficult to understand: tests assert on the result, and the agent's trajectory records only what the agent reports about itself, which may omit behavior executed by its subprocesses. We present AgentSpy, an approach that observes an agent from outside the agent. AgentSpy runs the agent in an isolated environment, configured by a declarative specification, and records the system calls and network traffic of the agent and of every process it executes. Based on this monitoring, AgentSpy supports two families of analyses: conformance analyses, which measure obligations, i.e., what an agent execution should do, and safety analyses, which check prohibitions, i.e., what an agent execution must never do. We instantiate one analysis of each family. The reliability analysis uses rules to summarize each run by the environment resources the agent uses: the commands it executed, the files it accessed, and the hosts it contacted. The security analysis applies deterministic rules to the system calls of an execution. For reliability, we evaluated AgentSpy on 77 tasks with the codex harness and three recent LLMs, executing each task three times. Sets of repeated runs of the same task are more similar than sets that include runs of another task in 92.2% of the comparisons. Among tasks for which all three runs pass outcome-based tests, the agent performs task-unrelated activities in 18% of the cases, reads the grading files in 7%, and does not use the developers' guidance in 17%. For security, the generic rules of AgentSpy detect four of five attack categories we considered, with no false positives across 50 runs. |
| 2026-10-05 | [Reachability-Aware Diffusion Policy Optimization](http://arxiv.org/abs/2610.05969v1) | Hikmet Simsir, Kutay Demiray et al. | Diffusion policies provide expressive action distributions for continuous-control reinforcement learning. However, safety-aware online diffusion policy optimization remains underexplored, particularly methods that use predictive reachability information without an explicit dynamics model. We propose Reachability-Aware Diffusion Policy Optimization (RADPO), a model-free method that combines predictive first-hit safety estimation with cumulative-cost budget feedback. RADPO learns a discounted first-hit reachability value that captures the discounted risk of a cost event, assigns larger weight to events that occur sooner, and uses this signal to shape the reward. A separate dual-like multiplier adjusts the shaping strength according to realized episodic costs relative to a prescribed budget. The diffusion actor improves through weighted denoising regression on candidate actions scored by the reward critic. Our approach requires neither a learned dynamics model, action gradients through the critics, nor differentiation through the reverse diffusion sampler. We establish theoretical properties of the reachability value and show that accumulated reachability penalty provides a conservative surrogate for future discounted cumulative cost. Across ten continuous-control safety tasks, RADPO achieves competitive reward-cost trade-offs, with substantial reductions in constraint violations on several tasks relative to the compared baselines. Our theoretical and empirical analysis supports that combining reachability with cumulative budget feedback is a viable approach to safety-aware diffusion policies. |

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



