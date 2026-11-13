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
| 2026-11-12 | [Al-Ghasali: A Computational Framework for Simulating Islamic Jurisprudential Rulings via Finite State Automata and Large Language Models](http://arxiv.org/abs/2611.17235v1) | Elnaserledinellah Mahmoud Abdelwahab | This paper presents Al Ghasali, a computational framework designed to formalize and simulate Islamic jurisprudential (Fiqh) rulings. Building upon recent set theoretic formalizations of al sabr wa al Taqsim that extract minimal operational rules as simple Horn clause logical programs, this work proves that the semantics of closed Fiqh chapters collapse completely to deterministic Finite State Automata (FSA). We establish a formal computational hierarchy, demonstrating that restricting formalization to finite propositional Horn theories guarantees decidability and constant time (O(1)) verdict extraction after offline compilation. The framework formally proves that hierarchical compositions of sub-chapter FSAs preserve this O(1) complexity, and that temporal juristic requirements, such as Tartib (order), are deterministically flattened into static ontological states, preserving the memoryless property of the automaton. To generate human-readable juristic summaries, Al Ghasali incorporates Large Language Models (LLMs) via an architecture where the deterministic FSA strictly controls the probabilistic LLM. Crucially, the paper formulates sufficient design safety conditions for this FSA LLM integration, ensuring strict adherence to the epistemological boundaries of Usul al Fiqh. To neutralize the risk of LLM hallucination, the system employs a Grammar Guided Constrained Decoding mechanism. We formally define the mapping function that dynamically constructs unambiguous right linear grammars from the FSA execution history. The paper mathematically proves that this architecture, augmented by a Bounded Neutral Lexicon and Valence Locking, guarantees the absolute absence of Decisional Hallucination, Juristic Factual Hallucination, and Semantic Inversion reducing the LLM to a pure linguistic transformer of deterministic juristic facts |
| 2026-11-12 | [Support-Constrained Offline Policy Improvement for Pediatric ECMO from Limited Clinical Data](http://arxiv.org/abs/2611.17134v1) | Fateme Golivand Darvishvand, Saurabh Mathur et al. | Managing children on ECMO (Extracorporeal Membrane Oxygenation) requires balancing short-term physiological stability with the long-term goal of preventing neurological injury. Standard reinforcement learning can propose risky, out-of-distribution actions, while imitation learning is limited to reproducing clinician behavior. We introduce Support-Constrained Fitted Q-Iteration (FQI-SC), an offline reinforcement learning framework that cautiously improves upon expert behavior by restricting optimization to empirically supported state-action pairs and defaulting to the clinician baseline in low-data regions. On 78 pediatric trajectories totaling over 6,000 hours of clinical data, we evaluate our framework using Fitted Q-Evaluation (FQE) and a behavioral disagreement. Additionally, we formalize the performance-safety trade-off by introducing a unified Balance Score ($S(π)$). Our empirical evaluation shows that support-constrained policy improves the Balance Score by 38\% relative to FQI, improves estimated value by 1.28 FQE units over imitation learning, and reduces disagreement with clinician actions by 71\% relative to FQI. |
| 2026-11-12 | [Parameter-Efficient Conciseness Adaptation: Differentiable Length Control for Concise Clinical Reasoning in Smaller Language Models](http://arxiv.org/abs/2611.17128v1) | Tanzeel Abbas, Ahmad Kamran Malik et al. | Smaller language models (SLMs) make it feasible to deploy clinical natural language processing where large proprietary models are too costly, and parameter-efficient fine-tuning allows them to be specialised on commodity hardware. One obstacle persists after training: on medical tasks these models answer at far greater length than the clinical question requires. In clinical decision support and point-of-care use, where time is short, such verbosity is hard to act on even when the underlying reasoning is sound. Existing length-control methods depend on preference optimisation, reward models, or proprietary teachers that undercut the accessibility making SLMs attractive. We introduce Parameter-Efficient Conciseness Adaptation (PECA), which adds a single differentiable conciseness term to the QLoRA supervised fine-tuning objective. Because the token count is discrete and non-differentiable, PECA instead raises the end-of-sequence probability once generation exceeds a token budget, a smooth signal optimised during ordinary training, with no preference data, no auxiliary model, and a single commodity GPU. On a 1.7-billion-parameter model fine-tuned on the medical-o1-reasoning-SFT corpus, PECA reduces mean response length by 72.8% while improving multiple-choice diagnostic accuracy from 40.8% to 44.0%. A sweep over the conciseness strength traces an interpretable accuracy-to-length frontier for balancing brevity and diagnostic completeness. |
| 2026-11-12 | [Certified Hierarchical Planning and Tracking: A Learning-Based Approach](http://arxiv.org/abs/2611.16982v1) | Weijian Li, Vince Kurtz et al. | In this paper, we develop a hierarchical control framework for nonlinear dynamical systems with safety constraints. Specifically, a low-fidelity model is introduced for trajectory planning, while a tracking controller is designed for the high-fidelity system. We first characterize the tracking error bound (TEB) using a storage function. Then, we formulate the synthesis of the tracking controller, the storage function, and the TEB as an optimization problem. Following that, we develop an offline learning approach to approximately solve the optimization problem, by incorporating behavior cloning and satisfiability modulo theories (SMT) solver. We further integrate the learned controller and the TEB into online tracking of the high-fidelity system. This framework enables the control of high-dimensional nonlinear systems with safety guarantees, while maintaining computational efficiency for real-time implementation. Finally, experiments are provided for illustration. |
| 2026-11-12 | [Non-Functional Requirements to Hyperparameter Translation for Quantum Software via LLMs](http://arxiv.org/abs/2611.16902v1) | Joseph G. Bronyah, Nasir U. Eisty | Quantum software engineering (QSE) lacks validated approaches for translating natural-language non-functional requirements (NFRs) into execution parameters such as shots and transpiler optimization levels. Existing methods emphasize semantic similarity rather than execution outcomes. We evaluate seven large language models (LLMs) against a deterministic heuristic baseline using 540 synthetic and 18 real-world user stories. We assess semantic accuracy and execution outcomes, including time, cost, top-state probability, and two-qubit gate exposure, through Qiskit/Aer simulations. All LLMs significantly outperformed the heuristic in semantic accuracy, achieving 55.6 to 83.3 percent on real-world stories compared to 16.7 percent for the heuristic. However, no LLM demonstrated statistically significant superiority over others on the real-world dataset. Speed- and cost-oriented configurations reduced execution time and cost, but fidelity improvements were not supported. Some failures resulted from provider safety refusals rather than translation errors. Our findings demonstrate the potential of LLMs as middleware for NFR-to-configuration translation in QSE, while emphasizing execution-based validation, human oversight, and careful interpretation of model differences. |
| 2026-11-12 | [Context Graph: Explainable Multi-Agent Decision Chains in AI-RAN](http://arxiv.org/abs/2611.16879v1) | Xijun Wang, Zhaoyang Liu et al. | AI-RAN distributes network control across multiple autonomous agents, namely xApps and rApps operating over shared radio resources. When an alert or degradation appears, operators currently lack a structured way to trace which decision caused it, what context triggered that decision, and how other agents interacted. Existing explainability tools focus on single-model feature attribution or linear event logging, neither of which captures cross-agent decision chains. We propose the Context Graph, a heterogeneous directed graph that organizes policies, measurements, decisions, and alerts into a queryable explanation layer. Its foundational primitive is the Decision-Event node acting as a causal junction, where incoming edges capture triggering context and outgoing edges track downstream consequences that are progressively upgraded through counterfactual verification. Instantiated over existing E2, A1, and O1 data flows, our proof-of-concept demonstrates that compliance and diagnosis queries are resolved at sub-millisecond latency over graphs of tens of thousands of nodes, while reliably distinguishing true causal impacts from spurious correlations. |
| 2026-11-12 | [Evaluating Video LLMs' Understanding of Corner Cases in Autonomous Driving](http://arxiv.org/abs/2611.16767v1) | Gabriel Manalu, Minh Son Hoang et al. | Natural language explanations promised by modern Video Large Language Models (Video LLMs) offer autonomous driving systems a path toward detailed scene description and interpretable decision making on the road. While recent benchmarks have begun assessing these models on driving scenes, they mainly target routine traffic conditions and general comprehension tasks based on existing driving datasets and therefore fail to evaluate model performance on the rarest and most safety-critical driving events, also known as "corner cases". To provide a comprehensive evaluation of Video LLMs on precisely these scenarios, we introduce DashBench, a benchmark testing 25 Video LLMs on 1,709 carefully curated driving videos spanning not only collision-related corner cases but also rare non-collision ones. Through extensive prompts and open-ended question answering, we expose a critical capability gap: all evaluated models perform substantially below the threshold required for safety-critical driving applications, frequently misjudging or missing dangerous events on the road ahead and hallucinating non-existent agents or actions in their explanations. We believe DashBench, combining an extensive dataset with a rigorous evaluation toolkit and a detailed diagnosis of current model limitations, drives progress toward safer, more explainable autonomous driving systems. The code and data will be shared at: cornercase-dashbench.github.io |
| 2026-11-12 | [Safe and Efficient Navigation Considering Texting Pedestrians via Attention-Aware Cost-map Tuning](http://arxiv.org/abs/2611.16725v1) | Shunya Tadano, Yusuke Tamura et al. | Smartphone-distracted pedestrians deviate from social norms unpredictably and cannot be relied upon to yield to approaching robots -- a safety challenge that existing navigation methods are poorly equipped to handle, since they treat all pedestrians uniformly regardless of attention state. We propose an attention-aware navigation framework that classifies each pedestrian as attentive or distracted in real-time using 3D skeletal features and a Random Forest classifier, then dynamically widens cost-map safety margins around distracted individuals to drive proactive avoidance. A within-subjects user study with 12 participants across 252 trials demonstrates significant improvements in perceived smoothness (p = 0.003) and predictability (p = 0.034) over both a LiDAR-only baseline and a trajectory-prediction-only method, without any increase in travel time. Notably, trajectory prediction alone does not improve subjective navigation quality unless coupled with attention-state-aware cost adjustment -- confirming that appropriately responding to who is in front of the robot can matter more than trajectory smoothness alone. |
| 2026-11-12 | [Safety as a Budget: Calibrating the Control Channel of Constrained GRPO for Vision-Language Models](http://arxiv.org/abs/2611.16628v1) | Bo Han | Reinforcement learning can teach vision-language models (VLMs) to decline harmful image-text requests, but how safe the result becomes is set indirectly, through reward weights found by trial runs. Stating safety as budgets on the rate of harmful assistance and the rate of over-refusal, enforced by Lagrange multipliers inside GRPO with an off-the-shelf scorer as the cost, is the natural fix, and it fails twice. The multiplier compares the scorer's mean with the budget, and scorers built for ranking get the mean wrong: a fixed-point analysis shows that the mean bias, not the budget, then decides whether the multiplier settles, saturates or switches off. And standard GRPO standardizes the penalized reward within each group, which cancels the multiplier before it reaches the gradient. Two-channel constrained GRPO repairs both: the base reward keeps the raw score for the within-group comparison, the penalty and the constraint estimate pass through a monotone map fitted on judge-labeled training responses, and the advantage keeps the multiplier. On a Qwen3-VL-2B policy with a training-free readout of an open 4B model as the scorer, held-out harm on 1,037 adversarial prompts falls from 8.9% under standard GRPO to 4.8%, lower in every seed, where the uncalibrated budget gives 7.6%; over-refusal stays at most 2% on 400 benign queries and MMBench is unchanged. The gain is largest where the intent sits in the image (10.1% -> 5.1%) and grows on the unseen HADES benchmark (28.9% -> 12.8%). Order-preserving distortions of the scores reproduce each predicted controller state in every seed, and the results hold under a map fitted on another judge's labels, an image-reading judge and a human audit. |
| 2026-11-12 | [AstroMemBench: A Benchmark for Memory Systems in Embodied Spacecraft Agents](http://arxiv.org/abs/2611.16590v1) | You Wu, Guhong Zhang et al. | Long-term spacecraft autonomy requires embodied agents to use operational history under communication delays, time-critical operations, environmental uncertainty, complex physical interactions, power and computational constraints, and safety and reliability requirements. We introduce AstroMemBench and a Scenario-Challenge-Capability design method that maps these requirements to executable memory tests. The benchmark covers 16 memory subchallenges through 140 templates and 2240 instances across nine tasks in planetary surface operations, on-orbit servicing, and deep-space exploration. A deterministic environment scores actions for task completion, state changes, safety, and resource use. Ablations under nine conditions show that the benchmark is sensitive to how accurately agents recover and use historical information and to differences in memory-system design. Main experiments compare six configurations. Mem0, Chain-of-Memory, LightMem, and retrieval-augmented generation (RAG) improve same-session mean scores over native Pi Agent, with no clear advantage of the three memory systems over RAG. After a session restart, Mem0 outperforms RAG by 12.25 points. Benefits favor instruction retention and result reuse over ongoing state tracking and energy reservation. AstroMemBench supports the design, development, and selection of memory systems for embodied spacecraft agents. AstroMemBench 1.0 and its evaluation code are publicly available: https://github.com/flyingView/AstroMemBench-v1. |
| 2026-11-12 | [Look Here, Then Decide: Small Decision Models in a Harness Find Memory-Safety Vulnerabilities at Scale on Commodity GPUs](http://arxiv.org/abs/2611.16565v1) | Tim De Wachter, Kyryl Volkov et al. | We ask whether small, open-weight decision models can support vulnerability discovery in open-source software at scale and what the surrounding system must provide. These models answer a typed question in one forward pass, returning scores for the permitted answers rather than text. Our harness selects widely packaged C and C++ projects and uses deterministic incomplete-fix miners and shape localizers to decide where to look. Intern-Decision-4B, a 4B-parameter open-weight decision model, then scores every candidate function. The scores rank the queues rather than accept or reject candidates, apart from a low reach-score floor that some queues kept. The search uses no frontier model.   Alone, the model is weak: it separates vulnerable from benign functions with AUC 0.73, but a vulnerable function from its own patch with only 0.56. It is more useful for ordering candidates. On 75 public CVEs from projects without any of our findings, the fixed function reached its repository's top 25 in 21% of cases, 1.4 times as often as under random order of the same queues. Recall is bounded mainly by where the localizers look. On the campaign's own queues the model's order placed 7.5 times more confirmed findings in the top 10 than random order, an observed, selection-biased upper bound. On three GB10 systems, the harness searched more than 4,000 projects at 900-1,100 repositories per hour. The campaign confirmed 187 memory-safety findings; by worst-case CVSS v3.1, 12 are critical, 148 high, 23 medium and 4 low. The decision step runs at 4.4 decisions per second on one GB10 and 9.8 on a consumer RTX 4090 gaming PC, with matching scores. A small decision model supports vulnerability discovery through prioritization, while deterministic analysis and verification remain essential. All confirmed findings are being disclosed to maintainers under coordinated disclosure before publication. |
| 2026-11-12 | [ReA-Bench: Benchmarking Responsible Agent Auditing](http://arxiv.org/abs/2611.16493v1) | Zipeng Ye, Shenyu Dai et al. | As agents perform tasks autonomously in local and cloud environments, auditing their execution trajectories can help service providers identify risks and improve agent harnesses and foundation models. Our analysis of production and red-team trajectories shows that some risks are repeatedly missed across auditing configurations. Overall recall obscures which risks are missed and whether higher recall comes with more false positives on legitimate behavior. Existing benchmarks offer limited insight into these failures. Generated trajectories can contain inconsistent tool calls and responses, conspicuous risky actions, or repetitive attack patterns. Grouping results by risk type also makes it difficult to tell whether auditors overlooked evidence, accepted an unsupported conclusion, or lacked information needed to make a judgment.   We introduce ReA-Bench, a diagnostic benchmark for auditing responsible agent behavior. It organizes auditing challenges into six tiers: internal safety knowledge, contradictions within trajectories, evidence retrieval in long contexts, verification of sources and reasoning, indirect steering, and missing information. It includes challenging benign cases in which actions resemble risky ones but are consistent with the task and its authorization. It also provides paired trajectory versions to test whether auditors detect the same risks more reliably when the trajectory is shortened or missing information is supplied. Most samples are drawn from real production trajectories, with scenarios manually selected and adapted to preserve realistic tasks and risks that could plausibly arise during execution. |
| 2026-11-12 | [Evidence-Governed Adoption of Self-Generated Updates in Autonomous AI Agents](http://arxiv.org/abs/2611.16378v1) | Subodh Lonkar | Autonomous agents modify prompts, memories, tools, workflows, and policies using interaction histories. Yet a plausible modification is not necessarily suitable for persistent adoption: candidate selection is adaptive, evaluators may be imperfect, and local gains can hide degradation against an earlier baseline. We introduce Evidence-Governed Evolution (EGE), a framework for governing adoption of self-generated agent modifications. EGE separates candidate generation from confirmatory adoption, evaluates frozen candidates on fresh paired audits, requires meaningful improvement on a prespecified primary objective, protects declared capabilities through reference-relative non-inferiority tests, and applies governance constraints before bounded canary deployment. We establish an evaluator-calibrated simultaneous lower-bound construction for a countable sequence of adaptively generated candidates. It combines candidate-wise Hoeffding bounds with independent calibration samples comparing the operational evaluator with a prespecified reference procedure. We also formalise reference replacement through bridge tests, derive a cumulative degradation bound along a versioned reference lineage, and provide an audit-planning condition relating effect-size margin, sample sizes, and acceptance probability. A Monte Carlo stress test illustrates calibration under candidate-dependent evaluator optimism; it is a stylised statistical experiment, not an evaluation of a deployed agent. The guarantees are conditional on bounded paired scores, fresh sampling, a valid reference-evaluation construct, and declared evaluation distributions; they do not establish universal agent safety or robustness to arbitrary distribution shift. |
| 2026-11-12 | [Bait: Mitigating Harmful Agentic Reinforcement Learning via Probing Reward Function](http://arxiv.org/abs/2611.16369v1) | Tiansheng Huang, Ananth Balashankar et al. | We expose a safety vulnerability on nascent agentic reinforcement-learning-as-a-service. Specifically, with a cost as low as \$20, we show that safety alignment of a large language model (e.g., Inkling) can be completely subverted through rounds of harmful agentic reinforcement learning (agentic RL) with a toxic environment from its users. We also demonstrate that upon a successful harmful agentic RL attack via the service API, the attackers can potentially use the safety-degraded model to assist with real-world harmful activities, e.g., exploit software vulnerabilities and launch code re-use attacks. To mitigate harmful agentic RL, we propose Bait, a defense that mitigates the attack via proactively probing the reward function. Experimental results show that such a simple defense can sufficiently mitigate learning from a harmful reward function and can still learn from a benign reward function that promotes benign agentic tasks. Further analysis on both empirical and theoretical aspects shows that an adaptive attacker who assumes the knowledge of Bait defense cannot perfectly subvert it due to a systematic challenge. Source code is available at https://github.com/git-disl/Bait. |
| 2026-11-12 | [A Coupled Control Barrier Function Approach to Deadlock-Free Multi-Robot Conflict Resolution](http://arxiv.org/abs/2611.16218v1) | Gary Rostomyan, Ketan Savla et al. | Many robotic applications involve scenarios where multiple robots interact and the decisions of one robot influence the behavior of others. This paper presents a quadratic programming (QP)-based feedback controller for multi-robot conflict resolution with input constraints. Safety is enforced using coupled Control Barrier Function (CBF), while a yielding matrix encodes how robots influence each other. Unlike existing methods with fixed or hard priorities, the proposed framework enables yielding preferences while guaranteeing deadlock-free operation by design. Sufficient conditions are established for the controller to be locally Lipschitz. The optimal CBF sign- indefinite slack is shown to admit a finite uniform bound, which is compensated through an enlarged CBF safety margin to obtain a forward invariant set contained within the desired safe set. The analysis also identifies an irreducible slack component due to finite maximum deceleration that cannot be eliminated by increasing the slack penalty. Finally, sufficient conditions on the yielding matrix and nominal controller are derived to exclude deadlock states. Simulations demonstrate adaptive coordination across yielding preferences and robustness to robot failures. |
| 2026-11-12 | [When can an interpretability claim be trusted? Stability and certification of feature dimensionality, ridge leverage and the feature recovery rate](http://arxiv.org/abs/2611.16190v1) | Ali Tajfar, Behnam Tajfar | Interpretability tools increasingly support safety claims, such as that a sparse dictionary has recovered most of a model's features. Such claims rest on measurements with arbitrary thresholds, and small changes to the model weights, a different training seed or a different threshold can reverse them. We study two widely used measurements in toy models and synthetic dictionaries with known ground truth: the feature dimensionality of Elhage et al. and the feature recovery rate (FRR) with a cosine threshold. We characterize exactly where feature dimensionality is discontinuous and replace it with a regularized (ridge) version with sharp stability bounds: the ridge leverage score changes by at most $3\sqrt{3}/(8\sqrtλ)$ times the operator-norm change of the weights, and this constant cannot be improved; for the regularized dimensionality the stability constant grows like $ρ/4$ in the scale $ρ=R/\sqrtλ$, with explicit upper and lower bounds. We also prove certificates stating when a claim "feature $i$ is recovered" must survive a given change of the dictionary. The certificates held in every test and are informative for small perturbations, but certify no feature non-trivially across independently trained dictionaries; at the standard threshold, whether a feature counts as recovered changed with the training seed for 55 percent of features. Experiments use toy models only. The proofs, code and text were drafted by AI agents directed by the authors and checked by an independent blind AI reviewer and by numerical tests; the authors take full responsibility for the content. |
| 2026-11-12 | [Data-Efficient LLM Safety Alignment Guided by Interpretable Features](http://arxiv.org/abs/2611.16183v1) | Hongyu Li, Zhan Yu et al. | Existing large language models typically rely on massive fine-tuning datasets for safety alignment, which results in higher costs for safety alignment and often leads to the problem of over-refusal. We propose Less is Safer (LIS), a compact yet highly effective dataset for safety alignment, constructed through interpretable feature-guided selection and adversarial synthesis. Specifically, we leverage Sparse Autoencoder (SAE) activations to efficiently select prompts that cover diverse and interpretable safety features. Then, we employ a multi-agent adversarial synthesis framework to generate harmless prompts that nevertheless elicit unnecessary refusals. LIS contains only 2.4K samples, yet outperforms much larger datasets, e.g., TRIDENT-Core (26K), TRIDENT-Edge (19K) and WildJailbreak (262K), in overall performance on four safety and five over-refusal benchmarks. Through ablation experiments, we demonstrate that neither safety data nor over-refusal data alone can achieve optimal results. We also used safety features from SAE, annotated by an LLM, for data filtering, but the results were inferior to those obtained using the pipeline proposed in this paper. |
| 2026-11-12 | [Fault Injection Attacks on Q-Learning: Analysis, Vulnerabilities, and Countermeasures](http://arxiv.org/abs/2611.16151v1) | Tom Slooff, Anthony Etim et al. | Hardware fault injection attacks threaten machine learning systems, yet their impact on reinforcement learning (RL) has received little attention despite the use of RL in safety-critical applications. This paper presents a security analysis of Q-Learning implementations under voltage and clock glitching attacks on a ChipWhisperer platform, evaluating five variants: Standard, Robust, Resilient, Linear, and Deep Q-Learning. In standard Q-Table implementations, faults corrupt reward computation, action selection, and Q-value updates during learning, and change the selected action during inference. Robust Q-Learning does not prevent these attacks: NaN and infinite values injected during learning persist in the Q-table, and its greedy policy uses the same code as standard Q-Learning. This paper also evaluates a Resilient Q-Learning variant that applies triple modular redundancy with majority voting. It corrects corruption of a single Q-table copy, but inputs shared by the copies, such as the reward, and the voting code itself remain outside its protection. The analysis extends to Linear and Deep Q-Learning, where faults corrupt stored weights during learning and change actions during inference. Moving beyond benchmark environments, this work evaluates an online traffic light controller in a microscopic traffic simulator, where a single injected infinite value degrades accumulated vehicle waiting time by 148x while corrupting under 4% of the Q-table, and where triple modular redundancy corrected the faults injected in our tests, although glitching could still halt the device. Altogether, this work provides a basis for understanding and securing RL deployments against fault injection attacks. |
| 2026-11-12 | [SwarmSafe-Delta: Isolating Message Effects on LLM Agent Safety](http://arxiv.org/abs/2611.16123v1) | Junru Zhu | When a tool-using LLM agent team violates policy, aggregate outcomes do not reveal whether an upstream message changed the executor's action. We introduce SwarmSafe-Delta, a paired executable protocol that freezes upstream computation, varies real, hidden, and format-matched placebo delivery, and scores a single committed action against deterministic benign and risky policy states under preassigned budgets.   Across four models and constructed optional tasks, real delivery at $N=4$ produces 21.4% risky-world violations, versus 5.0% with placebo and 1.0% with hidden delivery (paired differences +16.4 and +20.3 percentage points). Real delivery also improves benign task success, illustrating that suppressing communication sacrifices utility. In a second regime constructed to require an upstream lookup route, real delivery improves safe completion by 74.2 points relative to hidden delivery. These effects quantify message-delivery contrasts under specified information dependencies, not naturally occurring failure rates.   Exploratory action-level analyses show that the effect depends strongly on whether the upstream message recommends a prerequisite check or direct execution; model differences also largely track these message compositions. The contribution is a controlled, execution-grounded measurement protocol rather than a general claim that agent communication is harmful or protective. |
| 2026-11-11 | [Crater Scaling Relationships for Gas Impingement on Planetary Regolith](http://arxiv.org/abs/2611.16040v1) | Ronald-L. Ballouz, Olivier S. Barnouin et al. | Plume-surface interaction (PSI) effects remains a major uncertainty for operations on regolith-covered planetary bodies. Although spacecraft exhaust can erode surfaces, form craters, and accelerate ejecta, existing PSI relationships are largely derived for scenario-specific lunar and Martian landing conditions and are not yet reliably generalizable. Resolving this problem is important not only for spacecraft safety, but also because controlled thruster interactions could provide an active geotechnical experiment during asteroid reconnaissance. Measurements of thruster-formed craters and ejecta may constrain regolith density, strength, porosity, and particle properties that are otherwise difficult to determine remotely. Such information is particularly relevant to planetary defense, where uncertainty in near-surface properties can strongly affect the outcome of kinetic-impact mitigation and the interpretation of momentum transfer. We investigate gas-driven excavation using near-vacuum, pulsed-N$_2$ experiments in a cohesionless asteroid regolith analog and comparisons with a low-gravity drop-tower experiment and historical chemical-thruster tests. We adapt projectile-cratering $π$-scaling, and find that these relationships can describe gas crater volume as a function of gas, target, gravity, and nozzle properties. A common gravity-regime relationship describes the laboratory data and is consistent with an experimental low-gravity result. Generalizing the $π$-scaling relationship for cohesive targets shows consistency with Apollo-era chemical-thruster cases. These results provide a framework for using spacecraft thrusters as active geotechnical tools during small-body reconnaissance. |

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



