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
| 2025-12-11 | [Asynchronous Reasoning: Training-Free Interactive Thinking LLMs](http://arxiv.org/abs/2512.10931v1) | George Yakushev, Nataliia Babina et al. | Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities and safety, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embedded assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of rotary embeddings to enable LLMs built for sequential interactions to simultaneously think, listen, and generate outputs. We evaluate our approach on math, commonsense, and safety reasoning and find that it can generate accurate thinking-augmented answers in real time, reducing time to first non-thinking token from minutes to <= 5s. and the overall real-time delays by 6-11x. |
| 2025-12-11 | [V-OCBF: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions](http://arxiv.org/abs/2512.10822v1) | Mumuksh Tayal, Manan Tayal et al. | Ensuring safety in autonomous systems requires controllers that satisfy hard, state-wise constraints without relying on online interaction. While existing Safe Offline RL methods typically enforce soft expected-cost constraints, they do not guarantee forward invariance. Conversely, Control Barrier Functions (CBFs) provide rigorous safety guarantees but usually depend on expert-designed barrier functions or full knowledge of the system dynamics. We introduce Value-Guided Offline Control Barrier Functions (V-OCBF), a framework that learns a neural CBF entirely from offline demonstrations. Unlike prior approaches, V-OCBF does not assume access to the dynamics model; instead, it derives a recursive finite-difference barrier update, enabling model-free learning of a barrier that propagates safety information over time. Moreover, V-OCBF incorporates an expectile-based objective that avoids querying the barrier on out-of-distribution actions and restricts updates to the dataset-supported action set. The learned barrier is then used with a Quadratic Program (QP) formulation to synthesize real-time safe control. Across multiple case studies, V-OCBF yields substantially fewer safety violations than baseline methods while maintaining strong task performance, highlighting its scalability for offline synthesis of safety-critical controllers without online interaction or hand-engineered barriers. |
| 2025-12-11 | [Zorya: Automated Concolic Execution of Single-Threaded Go Binaries](http://arxiv.org/abs/2512.10799v1) | Karolina Gorna, Nicolas Iooss et al. | Go's adoption in critical infrastructure intensifies the need for systematic vulnerability detection, yet existing symbolic execution tools struggle with Go binaries due to runtime complexity and scalability challenges. In this work, we build upon Zorya, a concolic execution framework that translates Go binaries to Ghidra's P-Code intermediate representation to address these challenges. We added the detection of bugs in concretely not taken paths and a multi-layer filtering mechanism to concentrate symbolic reasoning on panic-relevant paths. Evaluation on five Go vulnerabilities demonstrates that panic-reachability gating achieves 1.8-3.9x speedups when filtering 33-70% of branches, and that Zorya detects all panics while existing tools detect at most two. Function-mode analysis proved essential for complex programs, running roughly two orders of magnitude faster than starting from main. This work establishes that specialized concolic execution can achieve practical vulnerability detection in language ecosystems with runtime safety checks. |
| 2025-12-11 | [Natural Language Interface for Firewall Configuration](http://arxiv.org/abs/2512.10789v1) | F. Taghiyev, A. Aslanbayli | This paper presents the design and prototype implementation of a natural language interface for configuring enterprise firewalls. The framework allows administrators to express access control policies in plain language, which are then translated into vendor specific configurations. A compact schema bound intermediate representation separates human intent from device syntax and in the current prototype compiles to Palo Alto PAN OS command line configuration while remaining extensible to other platforms. Large language models are used only as assistive parsers that generate typed intermediate representation objects, while compilation and enforcement remain deterministic. The prototype integrates three validation layers, namely a static linter that checks structural and vendor specific constraints, a safety gate that blocks overly permissive rules such as any to any allows, and a Batfish based simulator that validates configuration syntax and referential integrity against a synthetic device model. The paper describes the architecture, implementation, and test methodology on synthetic network context datasets and discusses how this approach can evolve into a scalable auditable and human centered workflow for firewall policy management. |
| 2025-12-11 | [Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting](http://arxiv.org/abs/2512.10780v1) | Manurag Khullar, Utkarsh Desai et al. | Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably. |
| 2025-12-11 | [OPV: Outcome-based Process Verifier for Efficient Long Chain-of-Thought Verification](http://arxiv.org/abs/2512.10756v1) | Zijian Wu, Lingkai Kong et al. | Large language models (LLMs) have achieved significant progress in solving complex reasoning tasks by Reinforcement Learning with Verifiable Rewards (RLVR). This advancement is also inseparable from the oversight automated by reliable verifiers. However, current outcome-based verifiers (OVs) are unable to inspect the unreliable intermediate steps in the long reasoning chains of thought (CoTs). Meanwhile, current process-based verifiers (PVs) have difficulties in reliably detecting errors in the complex long CoTs, limited by the scarcity of high-quality annotations due to the prohibitive costs of human annotations. Therefore, we propose the Outcome-based Process Verifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation. To empower the proposed verifier, we adopt an iterative active learning framework with expert annotations to progressively improve the verification capability of OPV with fewer annotation costs. Specifically, in each iteration, the most uncertain cases of the current best OPV are annotated and then subsequently used to train a new OPV through Rejection Fine-Tuning (RFT) and RLVR for the next round. Extensive experiments demonstrate OPV's superior performance and broad applicability. It achieves new state-of-the-art results on our held-out OPV-Bench, outperforming much larger open-source models such as Qwen3-Max-Preview with an F1 score of 83.1 compared to 76.3. Furthermore, OPV effectively detects false positives within synthetic dataset, closely align with expert assessment. When collaborating with policy models, OPV consistently yields performance gains, e.g., raising the accuracy of DeepSeek-R1-Distill-Qwen-32B from 55.2% to 73.3% on AIME2025 as the compute budget scales. |
| 2025-12-11 | [Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving](http://arxiv.org/abs/2512.10739v1) | Songyang Gao, Yuzhe Gu et al. | Large language models (LLMs) have achieved significant progress in solving complex reasoning tasks by Reinforcement Learning with Verifiable Rewards (RLVR). This advancement is also inseparable from the oversight automated by reliable verifiers. However, current outcome-based verifiers (OVs) are unable to inspect the unreliable intermediate steps in the long reasoning chains of thought (CoTs). Meanwhile, current process-based verifiers (PVs) have difficulties in reliably detecting errors in the complex long CoTs, limited by the scarcity of high-quality annotations due to the prohibitive costs of human annotations. Therefore, we propose the \textbf{O}utcome-based \textbf{P}rocess \textbf{V}erifier (OPV), which verifies the rationale process of summarized outcomes from long CoTs to achieve both accurate and efficient verification and enable large-scale annotation. To empower the proposed verifier, we adopt an iterative active learning framework with expert annotations to progressively improve the verification capability of OPV with fewer annotation costs. Specifically, in each iteration, the most uncertain cases of the current best OPV are annotated and then subsequently used to train a new OPV through Rejection Fine-Tuning (RFT) and RLVR for the next round. Extensive experiments demonstrate OPV's superior performance and broad applicability. It achieves new state-of-the-art results on our held-out \textsc{\thisbench}, outperforming much larger open-source models such as Qwen3-Max-Preview with an F1 score of 83.1 compared to 76.3. Furthermore, OPV effectively detects false positives within synthetic dataset, closely align with expert assessment. When collaborating with policy models, OPV consistently yields performance gains, e.g., raising the accuracy of DeepSeek-R1-Distill-Qwen-32B from 55.2\% to 73.3\% on AIME2025 as the compute budget scales. |
| 2025-12-11 | [Constraints on the Population of Common Sources of Gravitational Waves and High-Energy Neutrinos with IceCube During the Third Observing Run of the LIGO and Virgo Detectors](http://arxiv.org/abs/2512.10707v1) | Doğa Veske, Zsuzsa Márka et al. | The discovery of joint sources of high-energy neutrinos and gravitational waves has been a primary target for the LIGO, Virgo, KAGRA, and IceCube observatories. The joint detection of high-energy neutrinos and gravitational waves would provide insight into cosmic processes, such as progenitor dynamics and outflows. The joint detection of multiple cosmic messengers can also elevate the significance of the observation when some or all of the constituent messengers are sub-threshold, not significant enough to declare their detection individually. Leveraging data from the LIGO, Virgo, and IceCube observatories, we conducted an archival investigation of sub-threshold multimessenger events. Complementing previous analyses, we used minimal assumptions to search for common sources of sub-threshold gravitational-wave and high-energy neutrino candidates during the third observing run (O3) of the Advanced LIGO and Advanced Virgo detectors. Our search did not identify significant joint sources. We therefore derive constraints on the rate density of joint sources for each compact binary merger population as a function of the energy emitted in neutrinos. Only a fraction of the gravitational-wave sources emit neutrinos, if the neutrino emission has high bolometric energy ($>10^{52}$ to $10^{54}$ erg). |
| 2025-12-11 | [How to Brake? Ethical Emergency Braking with Deep Reinforcement Learning](http://arxiv.org/abs/2512.10698v1) | Jianbo Wang, Galina Sidorenko et al. | Connected and automated vehicles (CAVs) have the potential to enhance driving safety, for example by enabling safe vehicle following and more efficient traffic scheduling. For such future deployments, safety requirements should be addressed, where the primary such are avoidance of vehicle collisions and substantial mitigating of harm when collisions are unavoidable. However, conservative worst-case-based control strategies come at the price of reduced flexibility and may compromise overall performance. In light of this, we investigate how Deep Reinforcement Learning (DRL) can be leveraged to improve safety in multi-vehicle-following scenarios involving emergency braking. Specifically, we investigate how DRL with vehicle-to-vehicle communication can be used to ethically select an emergency breaking profile in scenarios where overall, or collective, three-vehicle harm reduction or collision avoidance shall be obtained instead of single-vehicle such. As an algorithm, we provide a hybrid approach that combines DRL with a previously published method based on analytical expressions for selecting optimal constant deceleration. By combining DRL with the previous method, the proposed hybrid approach increases the reliability compared to standalone DRL, while achieving superior performance in terms of overall harm reduction and collision avoidance. |
| 2025-12-11 | [Challenges of Evaluating LLM Safety for User Welfare](http://arxiv.org/abs/2512.10687v1) | Manon Kempermann, Sai Suresh Macharla Vasu et al. | Safety evaluations of large language models (LLMs) typically focus on universal risks like dangerous capabilities or undesirable propensities. However, millions use LLMs for personal advice on high-stakes topics like finance and health, where harms are context-dependent rather than universal. While frameworks like the OECD's AI classification recognize the need to assess individual risks, user-welfare safety evaluations remain underdeveloped. We argue that developing such evaluations is non-trivial due to fundamental questions about accounting for user context in evaluation design. In this exploratory study, we evaluated advice on finance and health from GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across user profiles of varying vulnerability. First, we demonstrate that evaluators must have access to rich user context: identical LLM responses were rated significantly safer by context-blind evaluators than by those aware of user circumstances, with safety scores for high-vulnerability users dropping from safe (5/7) to somewhat unsafe (3/7). One might assume this gap could be addressed by creating realistic user prompts containing key contextual information. However, our second study challenges this: we rerun the evaluation on prompts containing context users report they would disclose, finding no significant improvement. Our work establishes that effective user-welfare safety evaluation requires evaluators to assess responses against diverse user profiles, as realistic user context disclosure alone proves insufficient, particularly for vulnerable populations. By demonstrating a methodology for context-aware evaluation, this study provides both a starting point for such assessments and foundational evidence that evaluating individual welfare demands approaches distinct from existing universal-risk frameworks. We publish our code and dataset to aid future developments. |
| 2025-12-11 | [Evaluating Gemini Robotics Policies in a Veo World Simulator](http://arxiv.org/abs/2512.10675v1) | Gemini Robotics Team, Coline Devin et al. | Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and general manner. However, the use of video models in robotics has been limited primarily to in-distribution evaluations, i.e., scenarios that are similar to ones used to train the policy or fine-tune the base video model. In this report, we demonstrate that video models can be used for the entire spectrum of policy evaluation use cases in robotics: from assessing nominal performance to out-of-distribution (OOD) generalization, and probing physical and semantic safety. We introduce a generative evaluation system built upon a frontier video foundation model (Veo). The system is optimized to support robot action conditioning and multi-view consistency, while integrating generative image-editing and multi-view completion to synthesize realistic variations of real-world scenes along multiple axes of generalization. We demonstrate that the system preserves the base capabilities of the video model to enable accurate simulation of scenes that have been edited to include novel interaction objects, novel visual backgrounds, and novel distractor objects. This fidelity enables accurately predicting the relative performance of different policies in both nominal and OOD conditions, determining the relative impact of different axes of generalization on policy performance, and performing red teaming of policies to expose behaviors that violate physical or semantic safety constraints. We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks for a bimanual manipulator. |
| 2025-12-11 | [NaviHydra: Controllable Navigation-guided End-to-end Autonomous Driving with Hydra-distillation](http://arxiv.org/abs/2512.10660v1) | Hanfeng Wu, Marlon Steiner et al. | The complexity of autonomous driving scenarios requires robust models that can interpret high-level navigation commands and generate safe trajectories. While traditional rule-based systems can react to these commands, they often struggle in dynamic environments, and end-to-end methods face challenges in complying with explicit navigation commands. To address this, we present NaviHydra, a controllable navigation-guided end-to-end model distilled from an existing rule-based simulator. Our framework accepts high-level navigation commands as control signals, generating trajectories that align with specified intentions. We utilize a Bird's Eye View (BEV) based trajectory gathering method to enhance the trajectory feature extraction. Additionally, we introduce a novel navigation compliance metric to evaluate adherence to intended route, improving controllability and navigation safety. To comprehensively assess our model's controllability, we design a test that evaluates its response to various navigation commands. Our method significantly outperforms baseline models, achieving state-of-the-art results in the NAVSIM benchmark, demonstrating its effectiveness in advancing autonomous driving. |
| 2025-12-11 | [Multi-Objective Reward and Preference Optimization: Theory and Algorithms](http://arxiv.org/abs/2512.10601v1) | Akhil Agnihotri | This thesis develops theoretical frameworks and algorithms that advance constrained reinforcement learning (RL) across control, preference learning, and alignment of large language models. The first contribution addresses constrained Markov Decision Processes (CMDPs) under the average-cost criterion through the Average-Constrained Policy Optimization (ACPO) algorithm. ACPO integrates sensitivity analysis with trust-region updates to ensure stable constraint handling, achieving state-of-the-art empirical performance with theoretical guarantees. Constrained RL is then extended to finite-horizon settings via e-COP, the first policy optimization method for episodic CMDPs. Built on an episodic policy difference lemma, e-COP offers provable performance, simplicity, and scalability in safety-critical environments. The thesis then investigates reinforcement learning from human preferences. warmPref-PS introduces a posterior sampling strategy for linear bandits that integrates offline preference data from heterogeneous raters into online learning. Explicit modeling of rater competence yields substantial regret reduction and more efficient data collection for RLHF. The PSPL algorithm further advances preference-based RL by jointly sampling reward models and transition dynamics from pairwise trajectory comparisons, providing Bayesian simple-regret guarantees and robust empirical identification of optimal policies. The final contribution applies these methods to large-scale model alignment. A multi-objective constrained optimization view yields MOPO, an iterative algorithm with closed-form updates that scales to multi-billion-parameter language models and remains robust across alignment settings. Collectively, the thesis unifies constrained RL across average-cost, episodic, and preference-driven paradigms, delivering theoretical advances and practical tools for safe and aligned decision-making. |
| 2025-12-11 | [Motion Planning for Safe Landing of a Human-Piloted Parafoil](http://arxiv.org/abs/2512.10595v1) | Maximillian Fainkich, Kiril Solovey et al. | Most skydiving accidents occur during the parafoil-piloting and landing stages and result from human lapses in judgment while piloting the parafoil. Training of novice pilots is protracted due to the lack of functional and easily accessible training simulators. Moreover, work on parafoil trajectory planning suitable for aiding human training remains limited. To bridge this gap, we study the problem of computing safe trajectories for human-piloted parafoil flight and examine how such trajectories fare against human-generated solutions. For the algorithmic part, we adapt the sampling-based motion planner Stable Sparse RRT (SST) by Li et al., to cope with the problem constraints while minimizing the bank angle (control effort) as a proxy for safety. We then compare the computer-generated solutions with data from human-generated parafoil flight, where the algorithm offers a relative cost improvement of 20\%-80\% over the performance of the human pilot. We observe that human pilots tend to, first, close the horizontal distance to the landing area, and then address the vertical gap by spiraling down to the suitable altitude for starting a landing maneuver. The algorithm considered here makes smoother and more gradual descents, arriving at the landing area at the precise altitude necessary for the final approach while maintaining safety constraints. Overall, the study demonstrates the potential of computer-generated guidelines, rather than traditional rules of thumb, which can be integrated into future simulators to train pilots for safer and more cost-effective flights. |
| 2025-12-11 | [From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection](http://arxiv.org/abs/2512.10485v1) | Chaomeng Lu, Bert Lagaisse | Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that both graph neural network (GNN)-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by consistent data distributions and heuristic or partially noisy labels. In this study, we systematically evaluate two representative DL models-ReVeal and LineVul-across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each respective dataset, and their code representations are analyzed using t-SNE to uncover vulnerability related patterns. To assess realistic applicability, we deploy these models along with four pretrained LLMs, Claude 3.5 Sonnet, GPT-o3-mini, GPT-4o, and GPT-5 on a curated dataset, VentiVul, comprising 20 recently (May 2025) fixed vulnerabilities from the Linux kernel. Our experiments reveal that current models struggle to distinguish vulnerable from non-vulnerable code in representation space and generalize poorly across datasets with differing distributions. When evaluated on VentiVul, our newly constructed time-wise out-of-distribution dataset, performance drops sharply, with most models failing to detect vulnerabilities reliably. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations and higher-quality datasets. |
| 2025-12-11 | [Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots](http://arxiv.org/abs/2512.10477v1) | Timur Ishuov, Michele Folgheraiter et al. | In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line. |
| 2025-12-11 | [T-SKM-Net: Trainable Neural Network Framework for Linear Constraint Satisfaction via Sampling Kaczmarz-Motzkin Method](http://arxiv.org/abs/2512.10461v1) | Haoyu Zhu, Yao Zhang et al. | Neural network constraint satisfaction is crucial for safety-critical applications such as power system optimization, robotic path planning, and autonomous driving. However, existing constraint satisfaction methods face efficiency-applicability trade-offs, with hard constraint methods suffering from either high computational complexity or restrictive assumptions on constraint structures. The Sampling Kaczmarz-Motzkin (SKM) method is a randomized iterative algorithm for solving large-scale linear inequality systems with favorable convergence properties, but its argmax operations introduce non-differentiability, posing challenges for neural network applications. This work proposes the Trainable Sampling Kaczmarz-Motzkin Network (T-SKM-Net) framework and, for the first time, systematically integrates SKM-type methods into neural network constraint satisfaction. The framework transforms mixed constraint problems into pure inequality problems through null space transformation, employs SKM for iterative solving, and maps solutions back to the original constraint space, efficiently handling both equality and inequality constraints. We provide theoretical proof of post-processing effectiveness in expectation and end-to-end trainability guarantees based on unbiased gradient estimators, demonstrating that despite non-differentiable operations, the framework supports standard backpropagation. On the DCOPF case118 benchmark, our method achieves 4.27ms/item GPU serial forward inference with 0.0025% max optimality gap with post-processing mode and 5.25ms/item with 0.0008% max optimality gap with joint training mode, delivering over 25$\times$ speedup compared to the pandapower solver while maintaining zero constraint violations under given tolerance. |
| 2025-12-11 | [When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection](http://arxiv.org/abs/2512.10449v1) | Devanshu Sahoo, Manish Prasad et al. | The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic. |
| 2025-12-11 | [Design and Implementation of a High-Precision Wind-Estimation UAV with Onboard Sensors](http://arxiv.org/abs/2512.10428v1) | Haowen Yu, Na Fan et al. | Accurate real-time wind vector estimation is essential for enhancing the safety, navigation accuracy, and energy efficiency of unmanned aerial vehicles (UAVs). Traditional approaches rely on external sensors or simplify vehicle dynamics, which limits their applicability during agile flight or in resource-constrained platforms. This paper proposes a real-time wind estimation method based solely on onboard sensors. The approach first estimates external aerodynamic forces using a disturbance observer (DOB), and then maps these forces to wind vectors using a thin-plate spline (TPS) model. A custom-designed wind barrel mounted on the UAV enhances aerodynamic sensitivity, further improving estimation accuracy. The system is validated through comprehensive experiments in wind tunnels, indoor and outdoor flights. Experimental results demonstrate that the proposed method achieves consistently high-accuracy wind estimation across controlled and real-world conditions, with speed RMSEs as low as \SI{0.06}{m/s} in wind tunnel tests, \SI{0.22}{m/s} during outdoor hover, and below \SI{0.38}{m/s} in indoor and outdoor dynamic flights, and direction RMSEs under \ang{7.3} across all scenarios, outperforming existing baselines. Moreover, the method provides vertical wind estimates -- unavailable in baselines -- with RMSEs below \SI{0.17}{m/s} even during fast indoor translations. |
| 2025-12-11 | [How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation](http://arxiv.org/abs/2512.10415v1) | Devanshu Sahoo, Vasudev Majhi et al. | The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment. |

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



