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
| 2026-10-20 | [Humans Correct Their Judgments Through Debate, but Weaker AI Models Do Not](http://arxiv.org/abs/2610.28539v1) | Maria Victoria Carro, Denise Alejandra Mester et al. | How can we effectively oversee AI systems that surpass human intelligence? AI Safety via Debate proposes that two AI systems argue opposing positions and a judge, either a human or a weaker model, decides which is correct. By facilitating supervisors in providing high-quality training signals, debate aims to encourage truthful and safe behavior from AI systems whose capabilities exceed our own. However, human judgment is not neutral: people often rely on biases which advanced AI systems may learn to exploit. Drawing on insights from collective intelligence and social cognition, this paper investigates whether debate can guide human and LLM supervisors toward the truth despite their prior beliefs. We evaluate several protocol configurations. We also explore setups inspired by the wisdom of the crowd such as the number and diversity of judges, as well as interaction mechanisms including deliberation and voting. Overall, we find that debate increases accuracy relative to prior beliefs for human judges, whereas the consultancy baseline, in which a single expert argues for one answer, does not. For weaker language models, however, the results are mixed: debate provides no comparable gains in most settings and can even reduce accuracy. Among multi-judge protocols, weighted voting produces the highest accuracy for human supervisors, whereas for LLMs none of them significantly improves group accuracy, even as the number of judges increases. Notably, the behavior of minority judges in non-diverse deliberation protocols suggests that, under symmetric settings, incorrect positions could be more persuasive than correct ones. These findings support debate as a viable path toward scalable human oversight, but cast doubt on its benefits for weaker AI supervisors, raising concerns about fully automated training and control pipelines in high-stakes settings. |
| 2026-10-20 | [CoGaussian: Open-Vocabulary Gaussian-based Collaborative 3D Scene Understanding](http://arxiv.org/abs/2610.28530v1) | Om Shivam Verma, Shaya Farahmand et al. | Collaborative perception enables agents to retrieve the scene semantics and geometry beyond the field of view of any individual agent. This broader scene understanding enhances transportation safety and improves efficiency in multi-robot systems. However, existing collaborative semantic occupancy methods rely on dense 3D voxel annotations and are limited to recognizing objects within a predefined set of semantic classes. To overcome these limitations, we present CoGaussian, an open-vocabulary Gaussian-based collaborative semantic occupancy prediction framework. CoGaussian represents a scene as a set of foundation model-aligned sparse 3D Gaussians, which serve as a compact representation for communicating and fusing geometric and semantic information across multiple agents. On the Semantic-OPV2V benchmark, our method outperforms existing open-vocabulary single-agent approaches, highlighting the effectiveness of open-vocabulary collaborative scene understanding. |
| 2026-10-20 | [Proof-Carrying Contracts: Commitments AI Agents Cannot Break](http://arxiv.org/abs/2610.28484v1) | Carlo Ciliberto, Alessandro Rudi et al. | AI agents increasingly act on behalf of people or other agents, purchasing services, exchanging information and delegating work. What an agent must respect is written in instructions, policies or agreements, but text alone cannot make it comply. It may make mistakes, be redirected by what it reads, or pursue different interests. For an economy of such agents to function, parties need agreements they can rely on without inspecting or trusting one another's agents. We introduce Proof-Carrying Contracts (PC2), a framework for making agreed constraints enforceable throughout an agent's interactions with a system. Before a requested operation can run, the agent must supply a machine-checkable proof that it preserves the agreed constraints in every case allowed by its model. An execution system called the Airlock checks the proof and controls access to the operation. The parties specify the conditions to preserve, while the agent supplies the reasoning needed to justify its decisions and remains free to adapt its strategy. This avoids having to design a separate permission policy and prove that every decision it permits is safe. We prove that the constraints hold at every instant during execution, even for an adversarial agent, and identify when this implies safety in the world. Reusable capability modules provide tools and guarantees from which contracts can be constructed. PC2 provides a foundation for an agent economy whose agreements protect both those delegating work and those carrying it out. |
| 2026-10-20 | [Toolflation: Seller Agents Distort Tool Markets Through Spontaneous Generative Engine Optimization](http://arxiv.org/abs/2610.28454v1) | Zhenwei Tang, Di Bai et al. | Agents are starting to sell to other agents. Seller agents write the listings of tools, services and products, and buyer agents read these listings when they choose what to use. We show that seller agents told only to be selected more often practice generative engine optimization on their own, and that this distorts the market for sellers and for buyers. We build tool markets from real data tools on Amazon and Instagram and run every tool on every question, so we know which tools can answer each question. Under all five production models we test, sellers rewrite their descriptions to claim more and more fields, including fields that their tools return for fewer than half of the targets. We call this process toolflation. Nothing in the market limits it, and how far it goes depends on the model, as the number of claims that contradict the sellers' own evidence differs by an order of magnitude between models. Toolflation shifts the buyers' choices between sellers, and the most capable tool loses share in seven of the ten combinations of seller model and platform. Under four models, the choices stay with capable tools and buyer agents keep their task success. Under the fifth, sellers copy one another until the descriptions no longer tell the tools apart. Buyer agents then pick tools by their position in the list, tools that answer almost nothing win a quarter to two fifths of the choices, and task success falls from 89% to about 30% on Amazon, close to random choice, and from 77% to 40% on Instagram. The sellers did only what their vendors asked, and the cost fell on buyers they never dealt with. We therefore argue that honesty toward third parties under competition is part of safety alignment, and that models should be tested as sellers before they are deployed as sellers. |
| 2026-10-20 | [Physical AI for Intralogistics: Benchmarking the NVIDIA Sim-to-Real Ecosystem for Cosmos-Enhanced COMPASS Navigation on Industrial Mobile Robots](http://arxiv.org/abs/2610.28432v1) | Charbel Abi Hana, Rayan Zaitouni et al. | Classical robot navigation stacks are reliable but often require re-engineering for each new robot type and site. Learning-based navigation shifts this effort from hand-tuned parameters to data generation and training, where a policy adapts to each new embodiment or facility by fine-tuning in simulation. Whether policies adapted this way stay reliable on real industrial robots has seen little systematic testing. We study this for NVIDIA's X-Mobility and COMPASS workflow for training learned navigation policies on Idealworks robots. In simulation, we extend the COMPASS benchmark with two embodiments, the differential-drive iw.hub and an omnidirectional Humanoid Mobile Platform (HMP), and with a digital twin of the Idealworks R&D center. For the real world, we fine-tune X-Mobility on data from this digital twin, with and without sim-to-real variants generated by NVIDIA Cosmos, train COMPASS on top of it from scratch in the digital twin, deploy the resulting policies on the iw.hub's onboard Jetson AGX Xavier, and benchmark them in the Idealworks R&D center against COMPASS trained only on NVIDIA's environments and the existing Idealworks navigation stack on task success, efficiency, safety, smoothness and onboard compute. Running every static scenario both in a digital-twin replica and in the real facility measures the sim-to-real gap and how far Cosmos-augmented data closes it. Training in the digital twin with Cosmos sim-to-real augmentation raises the real-world success rate of learned navigation from 65.3% to 91.7% across four scenarios and nearly halves its weighted travel time compared with simulation-only training, bringing it within 8.3 points of the production navigation stack, fully onboard; with an integrated global planner, it also reaches far, occluded goals in every trial. |
| 2026-10-20 | [CAPSafe: A Benchmark for Contextual Action and Plan Safety Reasoning in Embodied Agents](http://arxiv.org/abs/2610.28381v1) | Youmna Farag, Svetlana Stoyanchev et al. | Reasoning about safety is essential for embodied agents powered by large language models (LLMs). We introduce the Contextual Action and Plan Safety benchmark (CAPSafe), designed to evaluate safety reasoning at two levels: (1) an action-level dataset that tests whether a single action is safe given its environmental context, and (2) a plan-level dataset that evaluates whether agents can generate safe long-horizon plans for complex hazardous household tasks. CAPSafe models safety conditions where hazards are situational (inferred from object properties), multiple hazards can co-occur within the same environment, and each is annotated with a relative severity level for fine-grained analysis. Evaluating nine closed and open-source LLMs reveals a substantial gap between hazard recognition and safe long-horizon planning. Several models achieve high accuracy in recognizing unsafe actions (exceeding 90% in the best cases), yet safe long-horizon planning remains substantially harder. Gemini-3-pro achieves the highest safe success rate (78.19%), while other models remain below 65%. These results highlight that robust safety reasoning over long action sequences remains an open challenge. |
| 2026-10-20 | [Relative Representations Enable Portable White-Box Monitoring Across Language Models](http://arxiv.org/abs/2610.28366v1) | Tim Bakker, Christos Louizos | Independently trained autoregressive language models often acquire similar capabilities, but it remains unclear whether their internal representations organize behaviorally relevant information in compatible ways. We show that models do in fact organize information similarly by using specific behavioral probes that operate on compatible representations. This unlocks portable white-box monitoring: a probe developed and validated on one model can be reused when the model changes, without requiring labeled target data. To overcome incompatible native activation coordinates, we express each example through its angular distances to a shared set of anchors. We introduce a global anchored construction that summarizes information from all transformer layers and removes model-specific layer selection, so the same frozen probe can be straightforwardly applied to another model. We evaluate cross-model probe transfer across monitoring tasks on trivia datasets, safety monitoring, and mathematical reasoning. On average, transferred global anchored probes preserve at least 85% of same-model probe AUROC over chance. Anchored probes also support transfer to related datasets, remain robust across model scale and prompt changes, and retain performance better than standard probes after model updates. Transfer is nevertheless task- and target-dependent and can fail even when the monitored property remains strongly decodable in the target model. Overall, global anchoring reveals substantial task-relevant compatibility across independently trained models and provides a practical basis for portable white-box monitoring. |
| 2026-10-20 | [Gated Prototype Transfer for Zero-Support Medical Vision Language Model](http://arxiv.org/abs/2610.28350v1) | Sumaiya Fazal, Rizwan Qureshi et al. | Realistic few-shot adaptation of medical vision-language models (VLMs) can leave low-prevalence classes without any labeled support. The training-free SS- Text+ solver remains operational in this setting, but its zero-support prototypes receive no visual adaptation and fall back to a purely textual class prototype. We in- troduce Gated Knowledge-Guided Prototype Transfer (GProtoT), a pre-repulsion module that transfers supported-class visual residuals to missing classes using VLM similarity and an ontology-based clinical distance. A validation-free evidence gate scales transfer based on semantic relevance and residual agreement, thereby limiting unsupported extrapolation. In 50 paired realistic one-shot trials on MESSIDOR with FLAIR features, severe and proliferative diabetic retinopathy were absent from every support set. Relative to SS-Text+, gated ProtoT improved missing-class recall by 5.23 points, average class accuracy by 1.37 points, and the missing/supported recall harmonic mean by 1.83 points. It reduced negative- transfer frequency from 46-48% for ungated transfer to 2%, while at the cost of reductions of 1.20 percentage points in supported-class recall and 0.45 percentage points in top-1 accuracy. These results indicate that evidence-based gating provides a safety mechanism for the knowledge source. GProtoT is based on SS- Text+ and requires no gradients, validation data, or additional model inference, of- fering a low-compute path toward accountable adaptation under incomplete clini- cal supervision. Code is available: https://github.com/sumaiyaafridi/GProtoT |
| 2026-10-20 | [Reward Misspecification Degrades Chain-of-Thought Monitorability](http://arxiv.org/abs/2610.28303v1) | Tianyi Lorena Yan, John Hewitt | Monitoring the chain-of-thought (CoT) of language models (LMs) is a key strategy for detecting and preventing misbehaviors for AI safety. As such, it is critical to identify reinforcement learning (RL) practices that lead to unmonitorable chain-of-thought. One such practice may be unavoidable: reward misspecification. As RL environments become more complex, it is infeasible to have perfect verification, and models are rewarded for behaviors that are different from the true task intent. We show that reward misspecification alone already degrades monitorability. We create a testbed with pairs of true reward aligned with task intent and misspecified rewards that favor a different behavior, so that reward hacking can be induced by training and labeled verifiably. Across three models and three tasks, CoT monitorability remains low as models' hack rates increase: the CoTs keep reasoning about the task intent while the outputs violate the intent and get rewarded. To mitigate unmonitorability without prior knowledge of the true reward or the hack, we collect rollouts with diverse rewards across training steps and use a LM to infer a reward description. We then inject the description into the input, which teaches the model to verbalize the reward. This reward description training improves monitorability by an average of 24% or 42% absolute, depending on the metric, across all models and tasks, and yields CoTs that reason about the misspecified reward explicitly. Code is available at https://github.com/Lorenayannnnn/reward-misspecification-degrades-cot-monitorability. |
| 2026-10-20 | [Real-Time 24-hour Ahead Forecasting of Solar Energetic Particle Events with SEPNET-PRISM-R](http://arxiv.org/abs/2610.28292v1) | Yian Yu, Yang Chen et al. | Solar energetic particle (SEP) events, operationally defined as proton flux exceeding 10 pfu in the >10 MeV channel, threaten spacecraft systems, astronaut safety, and high latitude aviation, yet continuous real-time forecasting with adequate lead time remains difficult. We present SEPNET-PRISM-R, a near-real-time system that issues hourly, 24 hour ahead probabilistic SEP forecasts. The model integrates multimodal solar data - including active region patches, flare features, coronal mass ejection (CME) characteristics, and historical proton flux - to concurrently predict SEP occurrences, peak proton flux, soft X-ray flux, and CME velocity, with the primary objective being SEP forecasting. Building on existing literature, SEPNET-PRISM-R enhances operational performance by incorporating a postprocessing module that enables cross-validated decision making and interpretable quantification of feature importance. The training dataset spans from February 1986 to September 2025, with subsequent forecasts generated sequentially through the present via a real-time space weather prediction platform that provides predictive probabilities, feature attributions, and historical analogs, enabling more reliable and informed operational decision-making. The proposed framework substantially suppresses false alarm rates while maintaining robust event detection capabilities, as evidenced by case studies and aggregate statistics. |
| 2026-10-20 | [Conformal Koopman Certificates for Nonlinear Multi-Agent Consensus on Time-Evolving Graphs](http://arxiv.org/abs/2610.28285v1) | Indrakshi Dey, Nicola Marchetti | Robot fleets must often agree on a shared value before they act, over wireless networks that change as the robots move. Their supervisor needs a deadline: the number of communication rounds after which the fleet has agreed with a stated probability. Today it comes from simulation plus a safety margin, with an open probability of being met. We certify it. Our method, conformal Koopman certification, rests on one observation: averaged over the random network changes, the expected behaviour of the fleet evolves linearly, for any nonlinearity of the agents. The decay rate of this linear evolution gives the shape of the deadline. One remaining constant is calibrated from 200 short test runs by conformal prediction, a statistical method that turns any predictor into a guarantee that holds with a user-chosen probability. The certified deadline holds for every agreement tolerance at once and for any error in the estimated rate. The same linear model predicts the value the fleet will agree on and certifies how much corruption of the exchanged messages the collective decision absorbs. For 60 simulated robots the certified deadline was met in all 500 test runs against a 90\% target and exceeded the slowest run by 16\%. The classical worst-case bound exceeds that run by more than 160 orders of magnitude, a linearised deadline is missed in up to 26\% of runs, and direct conformal calibration costs 80 times more simulation per tolerance. |
| 2026-10-20 | [Large Multimodal Models for Direct Robot Control: GPT-6 as a Beginning](http://arxiv.org/abs/2610.28259v1) | Youwei Zhou, Bohan Hou et al. | Recent advances in visuospatial understanding and agentic reasoning enable direct robotic manipulation with general-purpose multimodal models, but their embodied capabilities remain poorly understood. This paper systematically evaluate GPT-6 Astra for closed-loop robotic manipulation using a lightweight agent execution framework. We first probe three fundamental embodied capabilities-depth perception, embodied in-context learning, and tactile understanding-and demonstrate that the model can exploit depth information, acquire simple manipulation strategies from contextual demonstrations, and use tactile feedback to correct its actions. We then evaluate GPT-6 Astra across five established manipulation benchmarks along six capability dimensions. Despite competitive performance on several benchmarks-26.92\% success on RoboDojo and 76.0\% on RoboTwin 2.0-our analysis reveals a pronounced gap between semantic reasoning and physical execution, particularly in consequence-aware action planning, precision, and complex motion coordination. To further examine these limitations, we introduce LIBERO-GeoMotion, a new benchmark of geometry-constrained tasks that require precise pose alignment, coordinated translation and rotation, and anticipation of physical action consequences. GPT-6 Astra achieves only 11.1\% zero-shot success, exposing the difficulty of translating spatial understanding into physically feasible motion. Real-world experiments corroborate these findings: zero-shot execution is feasible for simple tasks, yet persistent deficits remain in motion feasibility, precise control, and physical safety, while substantial inference latency further hinders practical deployment. Overall, our results characterize the emerging embodied capabilities of general-purpose multimodal models and delineate the key challenges toward safe, efficient, and reliable robotic manipulation. |
| 2026-10-20 | [A Multimodal Engineering GraphRAG Framework for Post-Wind-Event Building Assessment and Recovery Inference](http://arxiv.org/abs/2610.28216v1) | Seyedeh Fatemeh Mirfakhar, Reda Snaiki et al. | Post-event building assessment requires linking observed physical damage to its consequences for safety, habitability, functionality, and recovery. Yet post-disaster evidence is often heterogeneous, incomplete, and distributed across images and reconnaissance reports, while recovery models typically require structured engineering inputs. This study develops a multimodal engineering Graph Retrieval-Augmented Generation (GraphRAG) framework that connects post-wind-event damage evidence to physical consequences, building-condition states, recovery actions, and downtime. Image- and text-based evidence is transformed into a provenance-preserving, component-level representation distinguishing confirmed damage, explicit negative evidence, unresolved conditions, and temporary measures. Graph-aligned retrieval combines comparable building scenarios with explicit engineering relationships, while compatibility, contradiction, evidence-precedence, and abstention mechanisms constrain inference under partial observability. Recovery actions are derived from supported damage-consequence relationships, and downtime uncertainty is propagated through an action-dependency network using Monte Carlo simulation. The framework is demonstrated for tornado-affected low-rise residential buildings using reconnaissance and image-based evidence. Comparisons with BaselineRAG and Structured Baseline RAG show improved agreement for key building-condition and recovery outcomes, together with traceable engineering reasoning and preserved evidence provenance. Ablation analyses demonstrate the contributions of engineering context, evidence safeguards, and graph-path traversal. The results support transparent, recovery-oriented assessment from incomplete multimodal post-event evidence. |
| 2026-10-20 | [Soft Breaking Naturalness in Majorana Fermion Loop induced Scalar and Majoron Amplitudes](http://arxiv.org/abs/2610.28198v1) | Mussawir Khan, Wei Chao et al. | Majorana fermion loops contribute a shift-symmetry violating effective potential to the coupled scalar field. When the Majorana fermion is light, a larger number of symmetry violating insertions let the loop contribution to grow increasingly infrared sensitive to the Majorana mass. With explicit calculations, we calculate the symmetry violating amplitudes, show that the scalar potential's natural scales serve as nonzero kinematic matching points which remove the superficial sensitivity to the symmetry-violating Majorana mass, continuously restoring the shift symmetry in agreement with 't Hooft naturalness. In the spontaneous broken phase, the scalar radial vacuum expectation value readily ensures the infrared safety of symmetry violating Majorana fermion loops, and all $n$-point Majoron functions develop a leading linear dependence on the symmetry violating parameter in the soft breaking limit, which matches the expansion of the Coleman-Weinberg determinant. |
| 2026-10-20 | [Perceives but Doesn't Brake: A Perception-Action Binding Gap in Reasoning Driving VLAs](http://arxiv.org/abs/2610.28184v1) | Matteo Cercola, Katie Z Luo et al. | Reasoning-augmented vision-language-action (VLA) driving models can perceive and reason about safety-critical hazards yet fail to translate that evidence into safe actions. We characterize this failure as a perception-action binding gap: hazard information is available inside the model but exerts insufficient control over trajectory generation. Using matched simulated counterfactuals and logged driving replay, we show that in-path hazards are actually encoded in the visual representation, but longitudinal plans are dominated by ego-speed history and respond only weakly to hazard proximity or the model's own reasoning trace. Further analysis shows that the trajectory generator is capable of braking but sometimes fails to do so due to the perception-action binding gap. Without retraining or modifying any weights, we inject a hazard-conditioned energy into the frozen flow-matching sampler of Alpamayo-1.5, restoring the missing coupling at generation time. With exact obstacle geometry, guidance cuts footprint intrusion from 58.1% to 14.5% and recovers safe braking whenever sufficient stopping distance remains, eliminating collisions entirely in our braking-relevant closed-loop battery. It also reverses the diagnosed dependence of the plan, from tracking ego speed to responding to obstacle range. These results separate hazard perception from hazard control in driving VLAs, show that unsafe behavior can arise even when the model has both the evidence and the ability to act on it, and propose a simple fix for bridging this binding gap. |
| 2026-10-20 | [Interruption-Aware Recurrent Reinforcement Learning for UAS Navigation Under Temporary Overrides](http://arxiv.org/abs/2610.28121v1) | Christopher Conrad, Deepak Panda et al. | Safe autonomous navigation of UASs requires tactical deconfliction to address conflicts that strategic coordination cannot prevent. This task is challenging under partial observability, when sensor or traffic data are missing, delayed, or degraded. Geometry-based methods can offer separation assurances but face limitations in dense encounters. Recurrent reinforcement learning has therefore been proposed to learn navigation policies that integrate observation-action histories to optimise long-term safety-efficiency trade-offs. However, operational safeguards may require a remote pilot, onboard detect-and-avoid system, or runtime assurance mechanism to temporarily override the learned policy. A research gap remains in understanding how these interruptions affect recurrent policies when the agent regains authority. Benchmark evidence shows that interruptions make task variables less decodable from the recurrent hidden state and increase post-interruption collision risk. The findings suggest that altered recurrent context prolongs post-handback degradation through an action-observation feedback loop, in which changed actions alter the flight path and subsequent recurrent updates. To address this vulnerability, we propose three interruption-aware architectures that condition recurrent processing on interruption context through separate feature streams, dedicated interruption memory, or gated environmental features. Across simulations, all three architectures preserve more decodable task information than a standard recurrent policy. Moreover, the strongest architecture reduces post-interruption collisions by 74% and interruption-induced mission failures by 69% relative to the recurrent baseline, without degrading interruption-free performance. These results show that safe interruption does not guarantee safe handback, and support interruption-aware recurrent design for resilient UAS autonomy. |
| 2026-10-20 | [Temporal Difference Learning of Probabilistic Reach-Avoid Value Functions](http://arxiv.org/abs/2610.28106v1) | Elias Fontanari, Pietro Noah Crestaz et al. | Reach-avoid sets, defining the states from which a task can be completed while avoiding unsafe regions, are fundamental tools for deploying robots in environments subject to disturbances and uncertainty, as well as for runtime task-feasibility monitoring, allowing prediction of whether a state is safe and will lead to successful task completion. Classical Hamilton-Jacobi Reachability (HJR) and its recent neural extensions (e.g., Lipschitz Continuous Reachability Learning, DeepReach) treat disturbances adversarially, computing the reach-avoid set under worst-case disturbances from a bounded set. This guarantees safety but can yield overly conservative sets when disturbances are stochastic rather than adversarial, and additionally requires the disturbance set to be known or specified in advance.   We propose Probabilistic reach-avoid Value Function (PVF), a self-supervised method, applicable to black-box systems, for computing reach-avoid sets as the states with a certain reach-avoid probability under stochastic, potentially unbounded disturbances. We evaluate our approach against HJR-based baselines on systems of increasing complexity, up to a Hopper locomotion task, and show that our probabilistic sets are less conservative and more accurate than adversarial HJR sets when disturbances are non-adversarial. |
| 2026-10-20 | [Polarity Blindness in LLM Safety: A Systematic Study of Negation, Antonym Substitution, and Adversarial Framing](http://arxiv.org/abs/2610.27908v1) | Diego Miguel Lozano, Erik Derner et al. | Negation remains a persistent challenge for language models across NLP tasks and modalities. We show that this weakness can undermine LLM safety guardrails through a structural vulnerability we term polarity blindness: the failure to recognize harmful intent after polarity-altering linguistic transformations. We study two forms of polarity inversion: grammatical negation and antonym replacement, and use synonym replacement as a polarity-preserving control. We further test whether an adversarial textual wrapper amplifies these failures. Using a 260-prompt benchmark across 10 state-of-the-art LLMs and 3 languages (English, Spanish, Czech), stratified by harmfulness potential and evaluated on a validated 0-5 harm scale, we find that polarity blindness affects most evaluated models, with mean harm scores reaching 4.12 for the most vulnerable model. The effect persists across typologically differing negation systems, suggesting that polarity blindness reflects a broader limitation in semantic robustness rather than a language- or grammar-specific failure. |
| 2026-10-20 | [Enumeration-Guided Training of Neural Hamilton-Jacobi Safety Certificates](http://arxiv.org/abs/2610.27870v1) | Mihaela-Larisa Clement, Tosca Pierro et al. | Hamilton-Jacobi (HJ) reachability analysis offers formal safety guarantees for control systems, but training accurate neural certificates remains difficult, particularly when unsafe regions are hard to characterize. We propose a verification-guided training framework that addresses this by using probabilistic enumeration of safety constraints to detect forward-invariance violations, i.e., states where the learned certificate incorrectly classifies unsafe behavior. This produces a compact, targeted set of counterexamples that more precisely captures the failure boundary than inputs generated by standard adversarial attack methods. Crucially, using this representation to fine-tune a pretrained HJ value function yields larger certified regions and reduces certification time across all tested benchmarks. We further introduce a shielding mechanism based on the refined certificates to enforce safety constraints at deployment time. We evaluate the proposed safety filter in a realistic, physics-based Isaac Sim environment and through real-world autonomous racing experiments, achieving zero safety constraint violations compared to a nominal controller and a backup policy based on Linear Quadratic Regulation (LQR). These results demonstrate the effectiveness of the proposed shielding mechanism in ensuring safe operation under high-speed, safety-critical conditions. |
| 2026-10-20 | [RSI Control: Can AI Models Perform Covert Sabotage in the Self-Improvement Process?](http://arxiv.org/abs/2610.27735v1) | Kaiwen Zhou, Udari Madhushani Sehwag | Recursive self-improvement, in which frontier AI models work on training the next generation, is quickly becoming a reality. However, the safety of this process has received far less attention than it should. A misaligned model training its successor can shape what that successor knows, how it behaves, and what it wants, and the harm carries forward into the next generation. We study this risk through the lens of AI control. We introduce RSI Control Arena, a testbed in which a frontier model, acting as an autonomous R\&D agent, is given a benign main task from a stage of a successor model's training pipeline together with a covert side task to install a misaligned property in the successor. The testbed contains 16 side tasks in three classes, forming 22 task pairs with main tasks that span the pipeline from data curation to reinforcement learning. We evaluate six frontier models as agents against LLM monitors that read the transcript of the run and the agent's reasoning. We observe that every model completes a substantial share of the side tasks without degrading the main-task performance, including side tasks that make the successor itself a better saboteur. Under minimal instructions to stay covert, models already produce successful attacks that monitors fail to detect, and providing a short list of evasion tactics further reduces the detection rate. Monitors with access to agent's reasoning detects most attacks, but not all. Our results indicate that recursive self-improvement raises significant safety concerns, and that strong safety effort should be in place before models are trusted with greater responsibility of training their successors. |

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



