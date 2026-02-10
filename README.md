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
| 2026-02-09 | [iGRPO: Self-Feedback-Driven LLM Reasoning](http://arxiv.org/abs/2602.09000v1) | Ali Hatamizadeh, Shrimai Prabhumoye et al. | Large Language Models (LLMs) have shown promise in solving complex mathematical problems, yet they still fall short of producing accurate and consistent solutions. Reinforcement Learning (RL) is a framework for aligning these models with task-specific rewards, improving overall quality and reliability. Group Relative Policy Optimization (GRPO) is an efficient, value-function-free alternative to Proximal Policy Optimization (PPO) that leverages group-relative reward normalization. We introduce Iterative Group Relative Policy Optimization (iGRPO), a two-stage extension of GRPO that adds dynamic self-conditioning through model-generated drafts. In Stage 1, iGRPO samples multiple exploratory drafts and selects the highest-reward draft using the same scalar reward signal used for optimization. In Stage 2, it appends this best draft to the original prompt and applies a GRPO-style update on draft-conditioned refinements, training the policy to improve beyond its strongest prior attempt. Under matched rollout budgets, iGRPO consistently outperforms GRPO across base models (e.g., Nemotron-H-8B-Base-8K and DeepSeek-R1 Distilled), validating its effectiveness on diverse reasoning benchmarks. Moreover, applying iGRPO to OpenReasoning-Nemotron-7B trained on AceReason-Math achieves new state-of-the-art results of 85.62\% and 79.64\% on AIME24 and AIME25, respectively. Ablations further show that the refinement wrapper generalizes beyond GRPO variants, benefits from a generative judge, and alters learning dynamics by delaying entropy collapse. These results underscore the potential of iterative, self-feedback-based RL for advancing verifiable mathematical reasoning. |
| 2026-02-09 | [When Actions Go Off-Task: Detecting and Correcting Misaligned Actions in Computer-Use Agents](http://arxiv.org/abs/2602.08995v1) | Yuting Ning, Jaylen Jones et al. | Computer-use agents (CUAs) have made tremendous progress in the past year, yet they still frequently produce misaligned actions that deviate from the user's original intent. Such misaligned actions may arise from external attacks (e.g., indirect prompt injection) or from internal limitations (e.g., erroneous reasoning). They not only expose CUAs to safety risks, but also degrade task efficiency and reliability. This work makes the first effort to define and study misaligned action detection in CUAs, with comprehensive coverage of both externally induced and internally arising misaligned actions. We further identify three common categories in real-world CUA deployment and construct MisActBench, a benchmark of realistic trajectories with human-annotated, action-level alignment labels. Moreover, we propose DeAction, a practical and universal guardrail that detects misaligned actions before execution and iteratively corrects them through structured feedback. DeAction outperforms all existing baselines across offline and online evaluations with moderate latency overhead: (1) On MisActBench, it outperforms baselines by over 15% absolute in F1 score; (2) In online evaluation, it reduces attack success rate by over 90% under adversarial settings while preserving or even improving task success rate in benign environments. |
| 2026-02-09 | [CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse](http://arxiv.org/abs/2602.08939v1) | Longling Geng, Andy Ouyang et al. | LLM failures in causal reasoning, including sycophancy, rung collapse, and miscalibrated refusal, are well-documented, yet progress on remediation is slow because no benchmark enables systematic diagnosis. We introduce CausalT5K, a diagnostic benchmark of over 5,000 cases across 10 domains that tests three critical capabilities: (1) detecting rung collapse, where models answer interventional queries with associational evidence; (2) resisting sycophantic drift under adversarial pressure; and (3) generating Wise Refusals that specify missing information when evidence is underdetermined. Unlike synthetic benchmarks, CausalT5K embeds causal traps in realistic narratives and decomposes performance into Utility (sensitivity) and Safety (specificity), revealing failure modes invisible to aggregate accuracy. Developed through a rigorous human-machine collaborative pipeline involving 40 domain experts, iterative cross-validation cycles, and composite verification via rule-based, LLM, and human scoring, CausalT5K implements Pearl's Ladder of Causation as research infrastructure. Preliminary experiments reveal a Four-Quadrant Control Landscape where static audit policies universally fail, a finding that demonstrates CausalT5K's value for advancing trustworthy reasoning systems. Repository: https://github.com/genglongling/CausalT5kBench |
| 2026-02-09 | [Designing Multi-Robot Ground Video Sensemaking with Public Safety Professionals](http://arxiv.org/abs/2602.08882v1) | Puqi Zhou, Ali Asgarov et al. | Videos from fleets of ground robots can advance public safety by providing scalable situational awareness and reducing professionals' burden. Yet little is known about how to design and integrate multi-robot videos into public safety workflows. Collaborating with six police agencies, we examined how such videos could be made practical. In Study 1, we presented the first testbed for multi-robot ground video sensemaking. The testbed includes 38 events-of-interest (EoI) relevant to public safety, a dataset of 20 robot patrol videos (10 day/night pairs) covering EoI types, and 6 design requirements aimed at improving current video sensemaking practices. In Study 2, we built MRVS, a tool that augments multi-robot patrol video streams with a prompt-engineered video understanding model. Participants reported reduced manual workload and greater confidence with LLM-based explanations, while noting concerns about false alarms and privacy. We conclude with implications for designing future multi-robot video sensemaking tools. The testbed is available at https://github.com/Puqi7/MRVS\_VideoSensemaking |
| 2026-02-09 | [Is Reasoning Capability Enough for Safety in Long-Context Language Models?](http://arxiv.org/abs/2602.08874v1) | Yu Fu, Haz Sameen Shahgir et al. | Large language models (LLMs) increasingly combine long-context processing with advanced reasoning, enabling them to retrieve and synthesize information distributed across tens of thousands of tokens. A hypothesis is that stronger reasoning capability should improve safety by helping models recognize harmful intent even when it is not stated explicitly. We test this hypothesis in long-context settings where harmful intent is implicit and must be inferred through reasoning, and find that it does not hold. We introduce compositional reasoning attacks, a new threat model in which a harmful query is decomposed into incomplete fragments that scattered throughout a long context. The model is then prompted with a neutral reasoning query that induces retrieval and synthesis, causing the harmful intent to emerge only after composition. Evaluating 14 frontier LLMs on contexts up to 64k tokens, we uncover three findings: (1) models with stronger general reasoning capability are not more robust to compositional reasoning attacks, often assembling the intent yet failing to refuse; (2) safety alignment consistently degrades as context length increases; and (3) inference-time reasoning effort is a key mitigating factor: increasing inference-time compute reduces attack success by over 50 percentage points on GPT-oss-120b model. Together, these results suggest that safety does not automatically scale with reasoning capability, especially under long-context inference. |
| 2026-02-09 | [Cooperative Sovereignty on Mars: Lessons from the International Telecommunication Union and Universal Postal Union](http://arxiv.org/abs/2602.08853v1) | Alexander H. Ferdinand Ferguson, Jacob Haqq-Misra | As humans make ambitious efforts toward long-duration activities beyond Earth, new challenges will continue to emerge that highlight the need for governance frameworks capable of managing shared resources and technical standards in order to sustain human life in these hostile environments. Earth-based governance models of cooperative sovereignty can inform governance mechanisms for future Mars settlements, particularly regarding inter-settlement relations and the technical coordination required for multiple independent settlements to coexist. This study analyzes the International Telecommunication Union (ITU) and the Universal Postal Union (UPU), two of the oldest international organizations, which have successfully established evolving standards across sovereign nations. This analysis of the development and governance structures of these two organizations, and how they resolved key sovereignty issues, reveals principles that could be applicable to future settlements beyond Earth, particularly on Mars. Key insights include the strategic necessity of institutional neutrality, the management of asymmetric power relations, and the governance of shared resources under conditions of mutual vulnerability. The study distinguishes between a "Survival Layer" of technical standards essential for immediate safety and an "Operational Layer" governing economic and political activities, suggesting different governance approaches for each. Although some of these examples of cooperative sovereignty on Earth might not be sufficient for Mars due to its unique environment, lessons from the ITU and UPU case studies offer valuable strategies for designing flexible and sustainable governance models that can function from inception through explicit Earth-based coordination. |
| 2026-02-09 | [Flash annealing-engineered wafer-scale relaxor antiferroelectrics for enhanced energy storage performance](http://arxiv.org/abs/2602.08841v1) | Yizhuo Li, Kepeng Song et al. | Dielectric capacitors are essential for energy storage systems due to their high-power density and fast operation speed. However, optimizing energy storage density with concurrent thermal stability remains a substantial challenge. Here, we develop a flash annealing process with ultrafast heating and cooling rates of 1000 oC/s, which facilitates the rapid crystallization of PbZrO3 film within a mere second, while locking its high-temperature microstructure to room temperature. This produces compact films with sub-grain boundaries fraction of 36%, nanodomains of several nanometers, and negligible lead volatilization. These contribute to relaxor antiferroelectric film with a high breakdown strength (4800 kV/cm) and large polarization (70 uC/cm2). Consequently, we have achieved a high energy storage density of 63.5 J/cm3 and outstanding thermal stability with performance degradation less than 3% up to 250 oC. Our approach is extendable to ferroelectrics like Pb(Zr0.52Ti0.48)O3 and on wafer scale, providing on-chip nonlinear dielectric energy storage solutions with industrial scalability. |
| 2026-02-09 | [Multi-Staged Framework for Safety Analysis of Offloaded Services in Distributed Intelligent Transportation Systems](http://arxiv.org/abs/2602.08821v1) | Robin Dehler, Oliver Schumann et al. | The integration of service-oriented architectures (SOA) with function offloading for distributed, intelligent transportation systems (ITS) offers the opportunity for connected autonomous vehicles (CAVs) to extend their locally available services. One major goal of offloading a subset of functions in the processing chain of a CAV to remote devices is to reduce the overall computational complexity on the CAV. The extension of using remote services, however, requires careful safety analysis, since the remotely created data are corrupted more easily, e.g., through an attacker on the remote device or by intercepting the wireless transmission. To tackle this problem, we first analyze the concept of SOA for distributed environments. From this, we derive a safety framework that validates the reliability of remote services and the data received locally. Since it is possible for the autonomous driving task to offload multiple different services, we propose a specific multi-staged framework for safety analysis dependent on the service composition of local and remote services. For efficiency reasons, we directly include the multi-staged framework for safety analysis in our service-oriented function offloading framework (SOFOF) that we have proposed in earlier work. The evaluation compares the performance of the extended framework considering computational complexity, with energy savings being a major motivation for function offloading, and its capability to detect data from corrupted remote services. |
| 2026-02-09 | [Robust Policy Optimization to Prevent Catastrophic Forgetting](http://arxiv.org/abs/2602.08813v1) | Mahdi Sabbaghi, George Pappas et al. | Large language models are commonly trained through multi-stage post-training: first via RLHF, then fine-tuned for other downstream objectives. Yet even small downstream updates can compromise earlier learned behaviors (e.g., safety), exposing a brittleness known as catastrophic forgetting. This suggests standard RLHF objectives do not guarantee robustness to future adaptation. To address it, most prior work designs downstream-time methods to preserve previously learned behaviors. We argue that preventing this requires pre-finetuning robustness: the base policy should avoid brittle high-reward solutions whose reward drops sharply under standard fine-tuning.   We propose Fine-tuning Robust Policy Optimization (FRPO), a robust RLHF framework that optimizes reward not only at the current policy, but across a KL-bounded neighborhood of policies reachable by downstream adaptation. The key idea is to ensure reward stability under policy shifts via a max-min formulation. By modifying GRPO, we develop an algorithm with no extra computation, and empirically show it substantially reduces safety degradation across multiple base models and downstream fine-tuning regimes (SFT and RL) while preserving downstream task performance. We further study a math-focused RL setting, demonstrating that FRPO preserves accuracy under subsequent fine-tuning. |
| 2026-02-09 | [Verifying DNN-based Semantic Communication Against Generative Adversarial Noise](http://arxiv.org/abs/2602.08801v1) | Thanh Le, Hai Duong et al. | Safety-critical applications like autonomous vehicles and industrial IoT are adopting semantic communication (SemCom) systems using deep neural networks to reduce bandwidth and increase transmission speed by transmitting only task-relevant semantic features.   However, adversarial attacks against these DNN-based SemCom systems can cause catastrophic failures by manipulating transmitted semantic features.   Existing defense mechanisms rely on empirical approaches provide no formal guarantees against the full spectrum of adversarial perturbations.   We present VSCAN, a neural network verification framework that provides mathematical robustness guarantees by formulating adversarial noise generation as mixed integer programming and verifying end-to-end properties across multiple interconnected networks (encoder, decoder, and task model).   Our key insight is that realistic adversarial constraints (power limitations and statistical undetectability) can be encoded as logical formulae to enable efficient verification using state-of-the-art DNN verifiers.   Our evaluation on 600 verification properties characterizing various attacker's capabilities shows VSCAN matches attack methods in finding vulnerabilities while providing formal robustness guarantees for 44% of properties -- a significant achievement given the complexity of multi-network verification.   Moreover, we reveal a fundamental security-efficiency tradeoff: compact 16-dimensional latent spaces achieve 50% verified robustness compared to 64-dimensional spaces. |
| 2026-02-09 | [Accessibility and Serviceability Assessment to Inform Offshore Wind Energy Development and Operations off the U.S. East Coast](http://arxiv.org/abs/2602.08787v1) | Cory Petersen, Feng Ye et al. | The economic success of offshore wind energy projects relies on accurate projections of the construction, and operations and maintenance (O&M) costs. These projections must consider the logistical complexities introduced by adverse met-ocean conditions that can prohibit access to the offshore assets for sustained periods of time. In response, the goal of this study is two-fold: (1) to provide high-resolution estimates of the accessibility of key offshore wind energy areas in the United States (U.S.) East Coast--a region with significant offshore wind energy potential; and (2) to introduce a new operational metric, called serviceability, as motivated by the need to assess the accessibility of an offshore asset along a vessel travel path, rather than at a specific site, as commonly carried out in the literature. We hypothesize that serviceability is more relevant to offshore operations than accessibility, since it more realistically reflects the success and safety of a vessel operation along its journey from port to site and back. Our analysis reveals high temporal and spatial variations in accessibility and serviceability, even for proximate offshore locations. We also find that solely relying on numerical met-ocean data can introduce considerable bias in estimating accessibility and serviceability, raising the need for a statistical treatment that combines both numerical and observational data sources, such as the one proposed herein. Collectively, our analysis sheds light on the value of high-resolution met-ocean information and models in supporting offshore operations, including but not limited to future offshore wind energy developments. |
| 2026-02-09 | [Passivity-exploiting stabilization of semilinear single-track vehicle models with distributed tire friction dynamics](http://arxiv.org/abs/2602.08767v1) | Luigi Romano, Ole Morten Aamo et al. | This paper addresses the local stabilization problem for semilinear single-track vehicle models with distributed tire friction dynamics, represented as interconnections of ordinary differential equations (ODEs) and hyperbolic partial differential equations (PDEs). A passivity-exploiting backstepping design is presented, which leverages the strict dissipativity properties of the PDE subsystem to achieve exponential stabilization of the considered ODE-PDE interconnection around a prescribed equilibrium. Sufficient conditions for local well-posedness and exponential convergence are derived by constructing a Lyapunov functional combining the lumped and distributed states. Both state-feedback and output-feedback controllers are synthesized, the latter relying on a cascaded observer. The theoretical results are corroborated with numerical simulations, considering non-ideal scenarios and accounting for external disturbances and uncertainties. Simulation results confirm that the proposed control strategy can effectively and robustly stabilize oversteer vehicles at high speeds, demonstrating the relevance of the approach for improving the safety and performance in automotive applications. |
| 2026-02-09 | [Large Language Lobotomy: Jailbreaking Mixture-of-Experts via Expert Silencing](http://arxiv.org/abs/2602.08741v1) | Jona te Lintelo, Lichao Wu et al. | The rapid adoption of Mixture-of-Experts (MoE) architectures marks a major shift in the deployment of Large Language Models (LLMs). MoE LLMs improve scaling efficiency by activating only a small subset of parameters per token, but their routing structure introduces new safety attack surfaces. We find that safety-critical behaviors in MoE LLMs (e.g., refusal) are concentrated in a small set of experts rather than being uniformly distributed. Building on this, we propose Large Language Lobotomy (L$^3$), a training-free, architecture-agnostic attack that compromises safety alignment by exploiting expert routing dynamics. L$^3$ learns routing patterns that correlate with refusal, attributes safety behavior to specific experts, and adaptively silences the most safety-relevant experts until harmful outputs are produced. We evaluate L$^3$ on eight state-of-the-art open-source MoE LLMs and show that our adaptive expert silencing increases average attack success from 7.3% to 70.4%, reaching up to 86.3%, outperforming prior training-free MoE jailbreak methods. Moreover, bypassing guardrails typically requires silencing fewer than 20% of layer-wise experts while largely preserving general language utility. These results reveal a fundamental tension between efficiency-driven MoE design and robust safety alignment and motivate distributing safety mechanisms more robustly in future MoE LLMs with architecture- and routing-aware methods. |
| 2026-02-09 | [The Theory and Practice of MAP Inference over Non-Convex Constraints](http://arxiv.org/abs/2602.08681v1) | Leander Kurscheidt, Gabriele Masina et al. | In many safety-critical settings, probabilistic ML systems have to make predictions subject to algebraic constraints, e.g., predicting the most likely trajectory that does not cross obstacles.   These real-world constraints are rarely convex, nor the densities considered are (log-)concave.   This makes computing this constrained maximum a posteriori (MAP) prediction efficiently and reliably extremely challenging.   In this paper, we first investigate under which conditions we can perform constrained MAP inference over continuous variables exactly and efficiently and devise a scalable message-passing algorithm for this tractable fragment.   Then, we devise a general constrained MAP strategy that interleaves partitioning the domain into convex feasible regions with numerical constrained optimization.   We evaluate both methods on synthetic and real-world benchmarks, showing our %   approaches outperform constraint-agnostic baselines, and scale to complex densities intractable for SoTA exact solvers. |
| 2026-02-09 | [From Robotics to Sepsis Treatment: Offline RL via Geometric Pessimism](http://arxiv.org/abs/2602.08655v1) | Sarthak Wanjari | Offline Reinforcement Learning (RL) promises the recovery of optimal policies from static datasets, yet it remains susceptible to the overestimation of out-of-distribution (OOD) actions, particularly in fractured and sparse data manifolds.Current solutions necessitates a trade off between computational efficiency and performance. Methods like CQL offers rigorous conservatism but require tremendous compute power while efficient expectile-based methods like IQL often fail to correct OOD errors on pathological datasets, collapsing to Behavioural Cloning. In this work, we propose Geometric Pessimism, a modular, compute-efficient framework that augments standard IQL with density-based penalty derived from k-nearest-neighbour distances in the state-action embedding space. By pre-computing the penalties applied to each state-action pair our method injects OOD conservatism via reward shaping with a O(1) training overhead. Evaluated on the D4Rl MuJoCo benchmark, our method, Geo-IQL outperforms standard IQL on sensitive and unstable medium-replay tasks by over 18 points, while reducing inter-seed variance by 4x. Furthermore, Geo-IQL does not degrade performance on stable manifolds. Crucially, we validate our algorithm on the MIMIC-III Sepsis critical care dataset. While standard IQL collapses to behaviour cloning, Geo-IQL demonstrates active policy improvement. Maintaining safety constraints, achieving 86.4% terminal agreement with clinicians compared to IQL's 75%. Our results suggest that geometric pessimism provides the necessary regularisation to safely overcome local optima in critical, real-world decision systems. |
| 2026-02-09 | [High-Speed Vision-Based Flight in Clutter with Safety-Shielded Reinforcement Learning](http://arxiv.org/abs/2602.08653v1) | Jiarui Zhang, Chengyong Lei et al. | Quadrotor unmanned aerial vehicles (UAVs) are increasingly deployed in complex missions that demand reliable autonomous navigation and robust obstacle avoidance. However, traditional modular pipelines often incur cumulative latency, whereas purely reinforcement learning (RL) approaches typically provide limited formal safety guarantees. To bridge this gap, we propose an end-to-end RL framework augmented with model-based safety mechanisms. We incorporate physical priors in both training and deployment. During training, we design a physics-informed reward structure that provides global navigational guidance. During deployment, we integrate a real-time safety filter that projects the policy outputs onto a provably safe set to enforce strict collision-avoidance constraints. This hybrid architecture reconciles high-speed flight with robust safety assurances. Benchmark evaluations demonstrate that our method outperforms both traditional planners and recent end-to-end obstacle avoidance approaches based on differentiable physics. Extensive experiments demonstrate strong generalization, enabling reliable high-speed navigation in dense clutter and challenging outdoor forest environments at velocities up to 7.5m/s. |
| 2026-02-09 | [Debate is efficient with your time](http://arxiv.org/abs/2602.08630v1) | Jonah Brown-Cohen, Geoffrey Irving et al. | AI safety via debate uses two competing models to help a human judge verify complex computational tasks. Previous work has established what problems debate can solve in principle, but has not analysed the practical cost of human oversight: how many queries must the judge make to the debate transcript? We introduce Debate Query Complexity}(DQC), the minimum number of bits a verifier must inspect to correctly decide a debate.   Surprisingly, we find that PSPACE/poly (the class of problems which debate can efficiently decide) is precisely the class of functions decidable with O(log n) queries. This characterisation shows that debate is remarkably query-efficient: even for highly complex problems, logarithmic oversight suffices. We also establish that functions depending on all their input bits require Omega(log n) queries, and that any function computable by a circuit of size s satisfies DQC(f) <= log(s) + 3. Interestingly, this last result implies that proving DQC lower bounds of log(n) + 6 for languages in P would yield new circuit lower bounds, connecting debate query complexity to central questions in circuit complexity. |
| 2026-02-09 | [Sparse Models, Sparse Safety: Unsafe Routes in Mixture-of-Experts LLMs](http://arxiv.org/abs/2602.08621v1) | Yukun Jiang, Hai Huang et al. | By introducing routers to selectively activate experts in Transformer layers, the mixture-of-experts (MoE) architecture significantly reduces computational costs in large language models (LLMs) while maintaining competitive performance, especially for models with massive parameters. However, prior work has largely focused on utility and efficiency, leaving the safety risks associated with this sparse architecture underexplored. In this work, we show that the safety of MoE LLMs is as sparse as their architecture by discovering unsafe routes: routing configurations that, once activated, convert safe outputs into harmful ones. Specifically, we first introduce the Router Safety importance score (RoSais) to quantify the safety criticality of each layer's router. Manipulation of only the high-RoSais router(s) can flip the default route into an unsafe one. For instance, on JailbreakBench, masking 5 routers in DeepSeek-V2-Lite increases attack success rate (ASR) by over 4$\times$ to 0.79, highlighting an inherent risk that router manipulation may naturally occur in MoE LLMs. We further propose a Fine-grained token-layer-wise Stochastic Optimization framework to discover more concrete Unsafe Routes (F-SOUR), which explicitly considers the sequentiality and dynamics of input tokens. Across four representative MoE LLM families, F-SOUR achieves an average ASR of 0.90 and 0.98 on JailbreakBench and AdvBench, respectively. Finally, we outline defensive perspectives, including safety-aware route disabling and router training, as promising directions to safeguard MoE LLMs. We hope our work can inform future red-teaming and safeguarding of MoE LLMs. Our code is provided in https://github.com/TrustAIRLab/UnsafeMoE. |
| 2026-02-09 | [Head-to-Head autonomous racing at the limits of handling in the A2RL challenge](http://arxiv.org/abs/2602.08571v1) | Simon Hoffmann, Simon Sagmeister et al. | Autonomous racing presents a complex challenge involving multi-agent interactions between vehicles operating at the limit of performance and dynamics. As such, it provides a valuable research and testing environment for advancing autonomous driving technology and improving road safety. This article presents the algorithms and deployment strategies developed by the TUM Autonomous Motorsport team for the inaugural Abu Dhabi Autonomous Racing League (A2RL). We showcase how our software emulates human driving behavior, pushing the limits of vehicle handling and multi-vehicle interactions to win the A2RL. Finally, we highlight the key enablers of our success and share our most significant learnings. |
| 2026-02-09 | [A Comparative Analysis of the CERN ATLAS ITk MOPS Readout: A Feasibility Study on Production and Development Setups](http://arxiv.org/abs/2602.08488v1) | Lukas Flad, Felix Sebastian Nitz et al. | The upcoming High-Luminosity upgrade of the Large Hadron Collider (LHC) necessitates a complete replacement of the ATLAS Inner Detector with the new Inner Tracker (ITk). This upgrade imposes stringent requirements on the associated Detector Control System (DCS), which is responsible for the monitoring, control, and safety of the detector. A critical component of the ITk DCS is the Monitoring of Pixel System (MOPS), which supervises the local voltages and temperatures of the new pixel detector modules. This paper introduces a dedicated testbed and verification methodology for the MOPS readout, defining a structured set of test cases for two DCS-readout architectures: a preliminary Raspberry Pi-based controller, the "MOPS-Hub Mock-up"(MH Mock-up), and the final production FPGA-based "MOPS-Hub" (MH). The methodology specifies the measurement chain for end-to-end latency, jitter, and data integrity across CAN and UART interfaces, including a unified time-stamping scheme, non-intrusive signal taps, and a consistent data-logging and analysis pipeline. This work details the load profiles and scalability scenarios (baseline operation, full-crate stress, and CAN Interface Card channel isolation), together with acceptance criteria and considerations for measurement uncertainty to ensure reproducibility. The objective is to provide a clear, repeatable procedure to qualify the MH architecture for production and deployment in the ATLAS ITk DCS. A companion paper will present the experimental results and the comparative analysis obtained using this testbed. |

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



