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
| 2026-02-25 | [RustyDL: A Program Logic for Rust](http://arxiv.org/abs/2602.22075v1) | Daniel Drodt, Reiner Hähnle | Rust is a modern programming language that guarantees memory safety and the absence of data races with a strong type system. We present RustyDL, a program logic for Rust, as a foundation for an auto-interactive, deductive verification tool for Rust. RustyDL reasons about Rust programs directly on the source code level, in contrast to other tools that are all based on translation to an intermediate language. A source-level program logic for Rust is crucial for a human-in-the-loop (HIL) style of verification that permits proving highly complex functional properties. We discuss specific Rust challenges in designing a program logic and calculus for HIL-style verification and propose a solution in each case. We provide a proof-of-concept of our ideas in the form of a prototype of a Rust instance of the deductive verification tool KeY. |
| 2026-02-25 | [Language Models Exhibit Inconsistent Biases Towards Algorithmic Agents and Human Experts](http://arxiv.org/abs/2602.22070v1) | Jessica Y. Bo, Lillio Mok et al. | Large language models are increasingly used in decision-making tasks that require them to process information from a variety of sources, including both human experts and other algorithmic agents. How do LLMs weigh the information provided by these different sources? We consider the well-studied phenomenon of algorithm aversion, in which human decision-makers exhibit bias against predictions from algorithms. Drawing upon experimental paradigms from behavioural economics, we evaluate how eightdifferent LLMs delegate decision-making tasks when the delegatee is framed as a human expert or an algorithmic agent. To be inclusive of different evaluation formats, we conduct our study with two task presentations: stated preferences, modeled through direct queries about trust towards either agent, and revealed preferences, modeled through providing in-context examples of the performance of both agents. When prompted to rate the trustworthiness of human experts and algorithms across diverse tasks, LLMs give higher ratings to the human expert, which correlates with prior results from human respondents. However, when shown the performance of a human expert and an algorithm and asked to place an incentivized bet between the two, LLMs disproportionately choose the algorithm, even when it performs demonstrably worse. These discrepant results suggest that LLMs may encode inconsistent biases towards humans and algorithms, which need to be carefully considered when they are deployed in high-stakes scenarios. Furthermore, we discuss the sensitivity of LLMs to task presentation formats that should be broadly scrutinized in evaluation robustness for AI safety. |
| 2026-02-25 | [Intrusive and Non-Intrusive Model Order Reduction for Airborne Contaminant Transport: Comparative Analysis and Uncertainty Quantification](http://arxiv.org/abs/2602.21996v1) | Lisa Kühn, Jacopo Bonari et al. | Numerical simulations of contaminant dispersion, as after a gas leakage incident on a chemical plant, can provide valuable insights for both emergency response and preparedness. Simulation approaches combine incompressible Navier-Stokes (INS) equations with advection-diffusion (AD) processes to model wind and concentration field. However, the computational cost of such high-fidelity simulations increases rapidly for complex geometries like urban environments, making them unfeasible in time-critical or multi-query "what-if" scenarios. Therefore, this study focuses on the application of model order reduction (MOR) techniques enabling fast yet accurate predictions. To this end, a thorough comparison of intrusive and non-intrusive MOR methods is performed for the computationally more demanding parametric INS problem with varying wind velocities. Based on these insights, a non-intrusive reduced-order model (ROM) is constructed accounting for both wind velocity and direction. The study is conducted on a two-dimensional domain derived from real-world building footprints, preserving key features for analyzing the dispersion of, for instance, denser contaminants. The resulting ROM enables faster than real-time predictions of spatio-temporal contaminant dispersion from an instantaneous source under varying wind conditions. This capability allows assessing wind measurement uncertainties through a Monte Carlo analysis. To demonstrate the practical applicability, an interactive dashboard provides intuitive access to simulation results. |
| 2026-02-25 | [Outpatient Appointment Scheduling Optimization with a Genetic Algorithm Approach](http://arxiv.org/abs/2602.21995v1) | Ana Rodrigues, Rui Rego | The optimization of complex medical appointment scheduling remains a significant operational challenge in multi-center healthcare environments, where clinical safety protocols and patient logistics must be reconciled. This study proposes and evaluates a Genetic Algorithm (GA) framework designed to automate the scheduling of multiple medical acts while adhering to rigorous inter-procedural incompatibility rules. Using a synthetic dataset encompassing 50 medical acts across four healthcare facilities, we compared two GA variants, Pre-Ordered and Unordered, against deterministic First-Come, First-Served (FCFS) and Random Choice baselines. Our results demonstrate that the GA framework achieved a 100% constraint fulfillment rate, effectively resolving temporal overlaps and clinical incompatibilities that the FCFS baseline failed to address in 60% and 40% of cases, respectively. Furthermore, the GA variants demonstrated statistically significant improvements (p < 0.001) in patient-centric metrics, achieving an Idle Time Ratio (ITR) frequently below 0.4 and reducing inter-healthcenter trips. While the GA (Ordered) variant provided a superior initial search locus, both evolutionary models converged to comparable global optima by the 100th generation. These findings suggest that transitioning from manual, human-mediated scheduling to an automated metaheuristic approach enhances clinical integrity, reduces administrative overhead, and significantly improves the patient experience by minimizing wait times and logistical burdens. |
| 2026-02-25 | [Aggressiveness-Aware Learning-based Control of Quadrotor UAVs with Safety Guarantees](http://arxiv.org/abs/2602.21936v1) | Leonardo Colombo, Thomas Beckers et al. | This paper presents an aggressiveness-aware control framework for quadrotor UAVs that integrates learning-based oracles to mitigate the effects of unknown disturbances. Starting from a nominal tracking controller on $\mathrm{SE}(3)$, unmodeled generalized forces and moments are estimated using a learning-based oracle and compensated in the control inputs. An aggressiveness-aware gain scheduling mechanism adapts the feedback gains based on probabilistic model-error bounds, enabling reduced feedback-induced aggressiveness while guaranteeing a prescribed practical exponential tracking performance. The proposed approach makes explicit the trade-off between model accuracy, robustness, and control aggressiveness, and provides a principled way to exploit learning for safer and less aggressive quadrotor maneuvers. |
| 2026-02-25 | [ProactiveMobile: A Comprehensive Benchmark for Boosting Proactive Intelligence on Mobile Devices](http://arxiv.org/abs/2602.21858v1) | Dezhi Kong, Zhengzhao Feng et al. | Multimodal large language models (MLLMs) have made significant progress in mobile agent development, yet their capabilities are predominantly confined to a reactive paradigm, where they merely execute explicit user commands. The emerging paradigm of proactive intelligence, where agents autonomously anticipate needs and initiate actions, represents the next frontier for mobile agents. However, its development is critically bottlenecked by the lack of benchmarks that can address real-world complexity and enable objective, executable evaluation. To overcome these challenges, we introduce ProactiveMobile, a comprehensive benchmark designed to systematically advance research in this domain. ProactiveMobile formalizes the proactive task as inferring latent user intent across four dimensions of on-device contextual signals and generating an executable function sequence from a comprehensive function pool of 63 APIs. The benchmark features over 3,660 instances of 14 scenarios that embrace real-world complexity through multi-answer annotations. To ensure quality, a team of 30 experts conducts a final audit of the benchmark, verifying factual accuracy, logical consistency, and action feasibility, and correcting any non-compliant entries. Extensive experiments demonstrate that our fine-tuned Qwen2.5-VL-7B-Instruct achieves a success rate of 19.15%, outperforming o1 (15.71%) and GPT-5 (7.39%). This result indicates that proactivity is a critical competency widely lacking in current MLLMs, yet it is learnable, emphasizing the importance of the proposed benchmark for proactivity evaluation. |
| 2026-02-25 | [A Multi-Turn Framework for Evaluating AI Misuse in Fraud and Cybercrime Scenarios](http://arxiv.org/abs/2602.21831v1) | Kimberly T. Mai, Anna Gausen et al. | AI is increasingly being used to assist fraud and cybercrime. However, it is unclear whether current large language models can assist complex criminal activity. Working with law enforcement and policy experts, we developed multi-turn evaluations for three fraud and cybercrime scenarios (romance scams, CEO impersonation, and identity theft). Our evaluations focused on text-to-text model capabilities. In each scenario, we measured model capabilities in ways designed to resemble real-world misuse, such as breaking down requests for fraud into a sequence of seemingly benign queries, and measuring whether models provide actionable information, relative to a standard web search baseline.   We found that (1) current large language models provide minimal practical assistance with complex criminal activity, (2) open-weight large language models fine-tuned to remove safety guardrails provided substantially more help, and (3) decomposing requests into benign-seeming queries elicited more assistance than explicitly malicious framing or system-level jailbreaks. Overall, the results suggest that current risks from text-generation models are relatively minimal. However, this work contributes a reproducible, expert-grounded framework for tracking how these risks may evolve with time as models grow more capable and adversaries adapt. |
| 2026-02-25 | [Heads Up!: Towards In Situ Photogrammetry Annotations and Augmented Reality Visualizations for Guided Backcountry Skiing](http://arxiv.org/abs/2602.21771v1) | Christoph Albert Johns, László Kopácsi et al. | Backcountry skiing is an activity where a group of skiers navigate challenging environmental conditions to ski outside of managed areas. This activity requires careful monitoring and effective communication around the current weather and terrain conditions to ensure skier safety. We aim to support and facilitate this communication by providing backcountry guides with a set of in situ spatial annotation tools to communicate hazards and appropriate speeds to the ski recreationalists. A guide can use a tablet application to annotate a photogrammetry-based map of a mountainside, for example, one collected using a commercial camera drone, with hazard points, slow-down zones, and safe zones. These annotations are communicated to the skiers via visual overlays in augmented reality heads-up displays. We present a prototype consisting of a web application and a virtual reality display that mirror the guide's and skier's perspectives, enabling participatory interaction design studies in a safe environment. |
| 2026-02-25 | [SurGo-R1: Benchmarking and Modeling Contextual Reasoning for Operative Zone in Surgical Video](http://arxiv.org/abs/2602.21706v1) | Guanyi Qin, Xiaozhen Wang et al. | Minimally invasive surgery has dramatically improved patient operative outcomes, yet identifying safe operative zones remains challenging in critical phases, requiring surgeons to integrate visual cues, procedural phase, and anatomical context under high cognitive load. Existing AI systems offer binary safety verification or static detection, ignoring the phase-dependent nature of intraoperative reasoning. We introduce ResGo, a benchmark of laparoscopic frames annotated with Go Zone bounding boxes and clinician-authored rationales covering phase, exposure quality reasoning, next action and risk reminder. We introduce evaluation metrics that treat correct grounding under incorrect phase as failures, revealing that most vision-language models cannot handle such tasks and perform poorly. We then present SurGo-R1, a model optimized via RLHF with a multi-turn phase-then-go architecture where the model first identifies the surgical phase, then generates reasoning and Go Zone coordinates conditioned on that context. On unseen procedures, SurGo-R1 achieves 76.6% phase accuracy, 32.7 mIoU, and 54.8% hardcore accuracy, a 6.6$\times$ improvement over the mainstream generalist VLMs. Code, model and benchmark will be available at https://github.com/jinlab-imvr/SurGo-R1 |
| 2026-02-25 | [Learning Complex Physical Regimes via Coverage-oriented Uncertainty Quantification: An application to the Critical Heat Flux](http://arxiv.org/abs/2602.21701v1) | Michele Cazzola, Alberto Ghione et al. | A central challenge in scientific machine learning (ML) is the correct representation of physical systems governed by multi-regime behaviours. In these scenarios, standard data analysis techniques often fail to capture the nature of the data, as the system's response varies significantly across the state space due to its stochasticity and the different physical regimes. Uncertainty quantification (UQ) should thus not be viewed merely as a safety assessment, but as a support to the learning task itself, guiding the model to internalise the behaviour of the data. We address this by focusing on the Critical Heat Flux (CHF) benchmark and dataset presented by the OECD/NEA Expert Group on Reactor Systems Multi-Physics. This case study represents a test for scientific ML due to the non-linear dependence of CHF on the inputs and the existence of distinct microscopic physical regimes. These regimes exhibit diverse statistical profiles, a complexity that requires UQ techniques to internalise the data behaviour and ensure reliable predictions. In this work, we conduct a comparative analysis of UQ methodologies to determine their impact on physical representation. We contrast post-hoc methods, specifically conformal prediction, against end-to-end coverage-oriented pipelines, including (Bayesian) heteroscedastic regression and quality-driven losses. These approaches treat uncertainty not as a final metric, but as an active component of the optimisation process, modelling the prediction and its behaviour simultaneously. We show that while post-hoc methods ensure statistical calibration, coverage-oriented learning effectively reshapes the model's representation to match the complex physical regimes. The result is a model that delivers not only high predictive accuracy but also a physically consistent uncertainty estimation that adapts dynamically to the intrinsic variability of the CHF. |
| 2026-02-25 | [SPOC: Safety-Aware Planning Under Partial Observability And Physical Constraints](http://arxiv.org/abs/2602.21595v1) | Hyungmin Kim, Hobeom Jeon et al. | Embodied Task Planning with large language models faces safety challenges in real-world environments, where partial observability and physical constraints must be respected. Existing benchmarks often overlook these critical factors, limiting their ability to evaluate both feasibility and safety. We introduce SPOC, a benchmark for safety-aware embodied task planning, which integrates strict partial observability, physical constraints, step-by-step planning, and goal-condition-based evaluation. Covering diverse household hazards such as fire, fluid, injury, object damage, and pollution, SPOC enables rigorous assessment through both state and constraint-based online metrics. Experiments with state-of-the-art LLMs reveal that current models struggle to ensure safety-aware planning, particularly under implicit constraints. Code and dataset are available at https://github.com/khm159/SPOC |
| 2026-02-25 | [Collisional-radiative data for tokamak disruption mitigation modeling](http://arxiv.org/abs/2602.21571v1) | Prashant Sharma, Christopher J. Fontes et al. | Effective tokamak disruption mitigation is crucial for ensuring the safety and integrity of fusion power reactors. Accurate collisional-radiative (CR) modeling of a radiative plasma is a critical component in predictive disruption mitigation design. In this paper, we focus on quasi-steady-state CR modeling applicable to the current quench phase of a tokamak disruption. We employ the ATOMIC collisional-radiative code from the Los Alamos suite and the newly developed Fusion Collisional-Radiative (FCR) code to model the atomic processes, providing high-fidelity data for radiative power loss, as well as average and effective charge states for hydrogen, helium, neon, and argon plasma species over a wide range of tokamak-relevant electron temperatures and electron densities. Fine-structure-resolved CR models are used for hydrogen and helium plasma species, while configuration-average CR models are implemented for neon and argon plasma species. The calculated values are compared with the superconfiguration CR model (FLYCHK) and the commonly used coronal equilibrium approximation to demonstrate the advantages and limitations of each model. To facilitate coupling of high-fidelity CR data to plasma simulation models, we represent the ATOMIC/FCR results over the relevant plasma parameter range using a smooth tensor product B-spline surface in electron temperature and electron density. This approach yields compact coefficient tables that can be evaluated efficiently while preserving spline smoothness across the domain. These data were previously used to examine ways to minimize runaway electrons in a tokamak current quench, and they are now made available in easy-to-use forms for community use and benchmarking. |
| 2026-02-25 | [Quantum Attacks Targeting Nuclear Power Plants: Threat Analysis, Defense and Mitigation Strategies](http://arxiv.org/abs/2602.21524v1) | Yaser Baseri, Edward Waller | The advent of Cryptographically Relevant Quantum Computers (CRQCs) presents a fundamental and existential threat to the forensic integrity and operational safety of Industrial Control Systems (ICS) and Operational Technology (OT) in critical infrastructure. This paper introduces a novel, forensics-first framework for achieving quantum resilience in high-consequence environments, with a specific focus on nuclear power plants. We systematically analyze the quantum threat landscape across the Purdue architecture (L0-L5), detailing how Harvest-Now, Decrypt-Later (HNDL) campaigns, enabled by algorithms like Shor's, can retroactively compromise cryptographic foundations, undermine evidence admissibility, and facilitate sophisticated sabotage. Through two detailed case studies, \textsc{Quantum~Scar} and \textsc{Quantum~Dawn}, we demonstrate multi-phase attack methodologies where state-level adversaries exploit cryptographic monoculture and extended OT lifecycles to degrade safety systems while creating unsolvable forensic paradoxes. Our probabilistic risk modeling reveals alarming success probabilities (up to 78\% for targeted facilities under current defenses), underscoring the criticality of immediate action. In response, we propose and validate a phased, defense-in-depth migration path to Post-Quantum Cryptography (PQC), integrating hybrid key exchange, cryptographic diversity, secure time synchronization, and side-channel resistant implementations aligned with ISA/IEC 62443 and NIST standards. The paper concludes that without urgent adoption of quantum-resilient controls, the integrity of both physical safety systems and digital forensic evidence remains at severe and irreversible risk. |
| 2026-02-25 | [Robust Electrocaloric Performance Enabled by Highly-Polar Frustrated Nanodomains in NaNbO3-Based Ferrodistortive Relaxor](http://arxiv.org/abs/2602.21520v1) | Feng Li, Changshun Dai et al. | Solid-state refrigeration technologies, represented by electrocaloric effect (ECE), are renowned for zero global-warming-potential and high cooling efficiency. Synergistically achieving high electrocaloric effect (ΔT) and wide temperature span (ΔTspan) for EC materials takes a leapfrog toward practical cooling applications, typical for integrated circuits. Guided by phase-field simulation, Ba(Ti, Hf)O3 dubbed as a polar wrench, establishes polar frustration by setting up local stress field and manipulating octahedral oxygen tilt (OOT) in NaNbO3-based relaxor. The resultant P4bm framework entails short-range and highly-polar ferrodistortive nanodomains, i.e., the abundant highly-polar nanodomains facilitate to increase entropy change and robust OOT enables to impede thermal perturbations. Consequently, a large ΔT of 0.85 K and 0.70 K with an ultrawide ΔTspan of 118 K and 130 K is obtained, contributing to an ultrahigh figure of merit of > 90 K2 in NaNbO3-Ba(Ti, Hf)O3, significantly outperforms its counterparts. The local structure responsible for robust EC performances are decrypted through 2D information from atomic-resolution scanning transmission electron microscope, 3D big-box model constructed from neutron total scattering and DFT calculations. These findings highlight that polar frustration strategy in ferrodistortive relaxor enables to pioneer emergent EC performances, and also unearth potential entropy-change-based ferroelectric and ferromagnetic materials beyond. |
| 2026-02-25 | [Beyond Refusal: Probing the Limits of Agentic Self-Correction for Semantic Sensitive Information](http://arxiv.org/abs/2602.21496v1) | Umid Suleymanov, Zaur Rajabov et al. | While defenses for structured PII are mature, Large Language Models (LLMs) pose a new threat: Semantic Sensitive Information (SemSI), where models infer sensitive identity attributes, generate reputation-harmful content, or hallucinate potentially wrong information. The capacity of LLMs to self-regulate these complex, context-dependent sensitive information leaks without destroying utility remains an open scientific question. To address this, we introduce SemSIEdit, an inference-time framework where an agentic "Editor" iteratively critiques and rewrites sensitive spans to preserve narrative flow rather than simply refusing to answer. Our analysis reveals a Privacy-Utility Pareto Frontier, where this agentic rewriting reduces leakage by 34.6% across all three SemSI categories while incurring a marginal utility loss of 9.8%. We also uncover a Scale-Dependent Safety Divergence: large reasoning models (e.g., GPT-5) achieve safety through constructive expansion (adding nuance), whereas capacity-constrained models revert to destructive truncation (deleting text). Finally, we identify a Reasoning Paradox: while inference-time reasoning increases baseline risk by enabling the model to make deeper sensitive inferences, it simultaneously empowers the defense to execute safe rewrites. |
| 2026-02-24 | [Provably Safe Generative Sampling with Constricting Barrier Functions](http://arxiv.org/abs/2602.21429v1) | Darshan Gadginmath, Ahmed Allibhoy et al. | Flow-based generative models, such as diffusion models and flow matching models, have achieved remarkable success in learning complex data distributions. However, a critical gap remains for their deployment in safety-critical domains: the lack of formal guarantees that generated samples will satisfy hard constraints. We address this by proposing a safety filtering framework that acts as an online shield for any pre-trained generative model. Our key insight is to cooperate with the generative process rather than override it. We define a constricting safety tube that is relaxed at the initial noise distribution and progressively tightens to the target safe set at the final data distribution, mirroring the coarse-to-fine structure of the generative process itself. By characterizing this tube via Control Barrier Functions (CBFs), we synthesize a feedback control input through a convex Quadratic Program (QP) at each sampling step. As the tube is loosest when noise is high and intervention is cheapest in terms of control energy, most constraint enforcement occurs when it least disrupts the model's learned structure. We prove that this mechanism guarantees safe sampling while minimizing the distributional shift from the original model at each sampling step, as quantified by the KL divergence. Our framework applies to any pre-trained flow-based generative scheme requiring no retraining or architectural modifications. We validate the approach across constrained image generation, physically-consistent trajectory sampling, and safe robotic manipulation policies, achieving 100% constraint satisfaction while preserving semantic fidelity. |
| 2026-02-24 | [Environment-Aware Learning of Smooth GNSS Covariance Dynamics for Autonomous Racing](http://arxiv.org/abs/2602.21366v1) | Y. Deemo Chen, Arion Zimmermann et al. | Ensuring accurate and stable state estimation is a challenging task crucial to safety-critical domains such as high-speed autonomous racing, where measurement uncertainty must be both adaptive to the environment and temporally smooth for control. In this work, we develop a learning-based framework, LACE, capable of directly modeling the temporal dynamics of GNSS measurement covariance. We model the covariance evolution as an exponentially stable dynamical system where a deep neural network (DNN) learns to predict the system's process noise from environmental features through an attention mechanism. By using contraction-based stability and systematically imposing spectral constraints, we formally provide guarantees of exponential stability and smoothness for the resulting covariance dynamics. We validate our approach on an AV-24 autonomous racecar, demonstrating improved localization performance and smoother covariance estimates in challenging, GNSS-degraded environments. Our results highlight the promise of dynamically modeling the perceived uncertainty in state estimation problems that are tightly coupled with control sensitivity. |
| 2026-02-24 | [Towards Controllable Video Synthesis of Routine and Rare OR Events](http://arxiv.org/abs/2602.21365v1) | Dominik Schneider, Lalithkumar Seenivasan et al. | Purpose: Curating large-scale datasets of operating room (OR) workflow, encompassing rare, safety-critical, or atypical events, remains operationally and ethically challenging. This data bottleneck complicates the development of ambient intelligence for detecting, understanding, and mitigating rare or safety-critical events in the OR.   Methods: This work presents an OR video diffusion framework that enables controlled synthesis of rare and safety-critical events. The framework integrates a geometric abstraction module, a conditioning module, and a fine-tuned diffusion model to first transform OR scenes into abstract geometric representations, then condition the synthesis process, and finally generate realistic OR event videos. Using this framework, we also curate a synthetic dataset to train and validate AI models for detecting near-misses of sterile-field violations.   Results: In synthesizing routine OR events, our method outperforms off-the-shelf video diffusion baselines, achieving lower FVD/LPIPS and higher SSIM/PSNR in both in- and out-of-domain datasets. Through qualitative results, we illustrate its ability for controlled video synthesis of counterfactual events. An AI model trained and validated on the generated synthetic data achieved a RECALL of 70.13% in detecting near safety-critical events. Finally, we conduct an ablation study to quantify performance gains from key design choices.   Conclusion: Our solution enables controlled synthesis of routine and rare OR events from abstract geometric representations. Beyond demonstrating its capability to generate rare and safety-critical scenarios, we show its potential to support the development of ambient intelligence models. |
| 2026-02-24 | [Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment](http://arxiv.org/abs/2602.21346v1) | Mengxuan Hu, Vivek V. Datla et al. | Recent advances in alignment techniques such as Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Direct Preference Optimization (DPO) have improved the safety of large language models (LLMs). However, these LLMs remain vulnerable to jailbreak attacks that disguise harmful intent through indirect or deceptive phrasing. Using causal intervention, we empirically demonstrate that this vulnerability stems from shallow alignment mechanisms that lack deep reasoning, often rejecting harmful prompts without truly understanding why they are harmful. To mitigate this vulnerability, we propose enhancing alignment through reasoning-aware post-training. We construct and release a novel Chain-of-Thought (CoT) fine-tuning dataset that includes both utility-oriented and safety-critical prompts with step-by-step rationales. Fine-tuning on this dataset encourages models to produce principled refusals grounded in reasoning, outperforming standard SFT baselines. Furthermore, inspired by failure patterns in CoT fine-tuning, we introduce Alignment-Weighted DPO, which targets the most problematic parts of an output by assigning different preference weights to the reasoning and final-answer segments. This produces finer-grained, targeted updates than vanilla DPO and improves robustness to diverse jailbreak strategies. Extensive experiments across multiple safety and utility benchmarks show that our method consistently improves alignment robustness while maintaining overall model utility. |
| 2026-02-24 | [Implications for PBH Dark Matter from a single Sub-Solar$\unicode{x2013}$GW Detection in LVK O1$\unicode{x2013}$O4](http://arxiv.org/abs/2602.21295v1) | Alberto Magaraggia, Nico Cappelluti | The detection of sub-solar mass black holes is a milestone of modern astrophysics as it would open a window either onto new stellar physics or could potentially unveil the nature of Dark Matter as Primordial Black Holes. On November 12, 2025, the LIGO-Virgo-KAGRA (LVK) collaboration reported the compact binary merger candidate S251112cm, a system with no obvious EM counterpart, consistent with binary black hole merger with a chirp mass in the range $0.1-0.87 \, M_\odot$. The probability that at least one component has mass $<$1 $M_{\odot}$ is $>99\%$. Inspired by this trigger, we tested if a population of PBHs formed at Quantum Chromodynamics epoch with a broad mass function could account for a signal of this type. Our results, corresponding to a predicted event rate of $0.8 \,\text{yr}^{-1}$ as seen by LVK O3b, suggest that the observed merger rate of $0.23^{+0.86}_{-0.218}\,\text{yr}^{-1}\;(95\%\;\text{C.L.})$ if the trigger is confirmed as an astrophysical event would be compatible with such a model. Our predicted detection rate is also in agreement with current LVK expectations for stellar-mass binaries, remaining consistent with a scenario in which a non-negligible fraction of the $3-200 \;M_\odot$ mergers observed by LVK originate from Primordial Black Holes. If confirmed, this detection would place a lower limit to the PBH abundance $f_{PBH}>0.04$ for our adopted model. |

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



