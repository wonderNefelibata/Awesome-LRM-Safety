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
| 2026-10-29 | [ShallowDefend: Harnessing Shallow Safety Alignment for Multi-Turn Defender](http://arxiv.org/abs/2610.40627v1) | Qiyao Ma, Xiaodong Liu et al. | Safety alignment has enabled large language models to reliably refuse harmful requests directly, yet the same request distributed across a multi-turn conversation still succeeds far too often. This brittleness has a mechanistic root: Alignment is shallow, with safety behavior concentrated in a model's first few output tokens, and existing defenses inherit the weakness, supervising the surface form of the response without ever giving the model an explicit safety decision to learn. We take the opposite view of shallowness: if a short span determines the trajectory of an entire response, that span is the highest-leverage place to intervene, and the defender can occupy it first. We propose Assess $\to$ Decide $\to$ Respond, a framework in which the defender assesses the user's intent, commits to an explicit, parseable safety decision, and only then generates a reply conditioned on it, with the commitment trained by reinforcement learning at the payload turn. Because the decision is a token span, the reward can be a rule: a self-verified reward checks the committed label against the conversation's known nature, requiring no judge and no annotation, while the same format also supports an LLM-as-judge reward over a graded decision space with safe redirection. Across two model families and fourteen benchmarks spanning safety, utility, and helpfulness, our defenders achieve the strongest multi-turn safety in a controlled comparison, maintaining their lead in safety while preserving utility at a modest cost to helpfulness. A per-token KL analysis and interventional probes confirm why this works: training concentrates its effect on the brief deliberation that sets a response's trajectory, and the decision committed there is a causal control point. The very shallowness attackers exploit becomes the mechanism of defense. |
| 2026-10-29 | [DAWN: Diffusion-based Alignment and unfastening Workflow Network](http://arxiv.org/abs/2610.40537v1) | Hemant Kumar, Piyush Goenka et al. | We present DAWN (Diffusion-based Alignment and unfastening Workflow Network), a hierarchical robotic framework designed to perform real-world, unfastening tasks through multi-modal sensory integration. Autonomous robotic unfastening is crucial for improving recycling processes, enhancing safety, and reducing operational costs; yet, it remains predominantly manual due to inherent challenges in precise screwdriver engagement with fasteners, detailed force execution, and robustness to environmental variability. To address the complexity of learning long-horizon, contact-rich manipulation tasks, we subdivide the unfastening task procedure into two sequential stages: (1) alignment & insertion, and (2) unfastening. DAWN consists of three key components: AlignNet, a diffusion-based policy ensuring precise screwdriver tip engagement; LatchNet, a multi-modal classifier reliably detecting successful stage transitions; and UnfastNet, which achieves stable, force-controlled screw removal. Experimental validation on a Franka Emika robot demonstrates that DAWN achieves alignment & insertion success rates comparable to trained human operators, robustly transitions between stages, and attains an impressive success rate in unfastening across various unseen scenarios. Ablation studies further elucidate critical input modalities and effective training strategies, highlighting DAWN's scalability, robustness, and data efficiency. |
| 2026-10-29 | [How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates](http://arxiv.org/abs/2610.40516v1) | Max Neuwinger, Hanna Yukhymenko et al. | Safety alignment for Large Language Models is heavily focused on English with limited multilingual representation, particularly damaging low-resource language performance. Related open training datasets are often limited in size or do not follow region-localized harmful topics to broaden the scope of alignment beyond cultures. We propose a scalable template-based augmentation pipeline to produce massive multilingual refusal and benign corpora, extracting safe templates from harmful prompts, translating them and then filling in the slots with culturally authentic entities, both harmful and benign. As a result, we release MSafeTemp, a validated corpus spanning 60 languages useful for multilingual safety alignment and evaluation. Using the dataset, we show that English-only evaluation misjudges multilingual safety: for several models, attacks that fail in English succeed once translated, while culturally localized variants expose further failures that translation alone misses. We also confirm that augmenting safety training data through templates allows for higher diversity and quality of alignment, resulting in improved safety performance beyond English. MSafeTemp and the respective pipeline are aimed at facilitating research and development of safer and more responsible models for less-represented cultures. |
| 2026-10-29 | [Training language models in simulation for psychiatric intake](http://arxiv.org/abs/2610.40476v1) | Samuel Schmidgall, Tim Strother et al. | In psychiatric practice, the clinical intake interview remains a primary diagnostic instrument, requiring the clinician to establish rapport while disentangling complex, overlapping presentations. Here, we show that multi-turn reinforcement learning (RL) in simulation environments improves language model performance on structured text-based psychiatric intake, spanning history-taking, differential diagnosis, and initial management planning. In a randomized, double-blind crossover study ($N = 60$ matched scenario pairs, 120 encounters) comparing the RL-trained agent against its identically prompted base model, 32 board-certified psychiatrists role-played standardized patients while a separate panel of 15 independent, board-certified psychiatrists scored the de-identified transcripts. The trained agent outperformed the base model in 76.7% of matched scenario pairs ($p < 0.0001$), with gains on an adapted clinical-examination rubric ($d_z = 0.77$, $p < 0.0001$), patient-centered communication (82.0% vs. 65.2%, $p < 0.001$), and differential diagnosis appropriateness (68.3% vs. 33.3%, $p < 0.001$). Across 127 psychiatrist-validated simulated intake scenarios, the trained agent nearly doubled its base model's clinical rubric score (0.706 vs. 0.379; composite reward 0.564 vs. 0.294, $p < 0.0001$). To systematically probe conversational safety, we established two Delphi-consensus taxonomies, of clinical safety risks and relational competencies, from which 256 challenging adversarial presentations were derived; we find that training in simulation environments more than halved under-triage of acute safety risks (12.9% vs. 30.5%, $p < 0.0001$). Simulation environments thus provide a controlled testbed for training and stress-testing psychiatric conversational agents, with gains that transferred to psychiatrist-enacted encounters, though residual gaps require safeguards and clinician oversight. |
| 2026-10-29 | [The Athlete as a Scientist of One: Agentic Bayesian Experimental Design for Individualized Physiology from Consumer Wearables](http://arxiv.org/abs/2610.40370v1) | Behnam Moradi | Large language model (LLM) agents for personal health read wearable data and recommend, but do not learn about the individual in a way that can be audited or falsified. We present EpiCoach, an agentic system that treats one athlete, a cross-country mountain biker, as the subject of accountable N-of-1 science. Claims about the athlete live in an evidence-gated hypothesis registry: each must issue a probabilistic prediction about a future session before it occurs, and its belief moves only through computed evidence from a logged statistical fit. Predictions come from a mechanistic digital twin inferred from heart rate, GPS and barometric altitude: a lagged heart-rate response model, a fitness-fatigue model with learned time constants, and a latent technical-descending skill state, measured on discovered trail segments. The agent prescribes the next training week as a constrained experiment maximizing information gain subject to load and safety constraints, around two instrument rides. We evaluate the system two ways over eighteen months of data. A rolling-origin replay benchmark shows the twin beats standard forecasters on heart rate but not segment time, and the agentic loop matches a static model. A live prospective phase runs the weekly cycle: in its first three weeks the ledger sealed 20 predictions, scored 14, the descent forecast calibrated after one correction, the climb forecast pessimistic as the athlete improved, and the safety officer vetoed the agent twice. Every correction is preserved in the registry, a template for auditable, falsifiable personalization in digital health. We release the registry, benchmark and protocol. |
| 2026-10-29 | [Learned Terminal Costs for Short-Horizon Model Predictive Control of Autonomous Trucks in Interactive Highway Traffic](http://arxiv.org/abs/2610.40257v1) | Hooman Hamze, Benjamin Houshmand et al. | Autonomous highway driving for heavy-duty trucks is a challenging control problem. The greater mass, dimensions and intricate dynamics increase the requirements for cooperation from surrounding traffic in critical maneuvers, requiring a careful balance between safety constraints and efficient long-horizon decision making. Model Predictive Control (MPC) provides rigorous constraint satisfaction but relies on terminal cost which can limit its effectiveness at short, computationally affordable horizons. Reinforcement Learning (RL) captures long-term driving behavior but offers no constraint guarantees and tends toward overly conservative policies when trained under sparse collision penalties. In this paper, we propose a hybrid MPC and RL framework for highway driving in autonomous trucks, where a finite-horizon MPC, solved via dynamic programming, uses a pre-trained RL model for terminal cost estimation. We evaluate the framework in a simulation environment with reactive surrounding traffic and compare it against pure MPC with zero terminal cost, rollout with terminal cost estimation using a base policy, and pure RL. Results show that the hybrid approach achieves zero collisions with the lowest cost of any configuration. |
| 2026-10-29 | [Decision-Level Error Attribution and Depth-Aware Safety Evaluation of Monocular RGB Human Tracking for Autonomous Robot Navigation](http://arxiv.org/abs/2610.40199v1) | Abdulahi Taiwo, Chijioke Cyriacus Ekechi | Monocular RGB tracking is useful for robots operating near people, but perception errors affect binary stop/proceed decisions in ways that geometric accuracy metrics do not capture. We examine how much decision error arises from the safety rule versus perception. We prove that lateral-only distance is a lower bound on planar distance: with accurate positions, it cannot miss dangers identified by the planar rule, but it produces unnecessary stops. We introduce a two-factor Shapley attribution separating rule and perception contributions, and further distinguish missed detections from localization errors. Experiments use ETH and UCY pedestrian videos with simulated stationary robots, disjoint calibration and evaluation pedestrians, and time-block bootstrap intervals. Across 898 pedestrians in three scenes, the lateral rule missed no dangers on annotated positions but achieved only 5.6-31.9% stop precision. In two scenes with RGB perception, calibrated ground-plane localization achieved 0.59 m and 0.51 m planar RMSE, compared with 1.72 m and 1.39 m for a geometric baseline and 0.81 m and 0.68 m for calibrated Depth Anything V2. Detection recall of 0.43 and 0.26 made missed detections the leading source of missed danger. A two-standard-deviation safety margin with ground-plane localization matched or improved on the lateral rule's missed-danger rate in all eight scene-threshold settings and reduced unnecessary stops in seven. All robot interactions are simulated, and outcomes are offline decision classifications, not demonstrations of collision avoidance. |
| 2026-10-29 | [It Takes Two to Probe, Thousands to Retrieve: Putting Vision--Language Probes to the Retrieval Test](http://arxiv.org/abs/2610.40124v1) | Ivana Beňová, Michal Gregor et al. | Vision--language models (VLMs) are tested for fine-grained understanding with image--text matching probes, a model must prefer the correct image or caption over handcrafted negative. In retrieval, the correct item must beat thousands of alternatives. We measure how these two evaluation methods relate by using two probing benchmarks, \svo{} and SugarCrepe, for retrieval tasks. For 100 image and caption queries we judge the top-10 retrievals from full dataset of six VLMs, with human annotators and an LLM assessor. We than compare how well models rank pool of items on the same queries, while changing only where the negatives comes from: the benchmark, a random draw, or the judged-incorrect retrievals of other models. On \svo{} the accuracy of retrieving a positive over handcrafted negatives is significantly higher than for retrieved ones (the correct item is preferred in 85\% against 77\% of the time). Using pool of ten retrieved negatives leads to same accuracy as when one uses pool of 200 to 430 random negatives. On SugarCrepe the order is reversed: its handcrafted negatives are harder than the retrieved ones. Incomplete relevance labels explain most of the low retrieval scores on \svo{}, where the top-ranked item is correct for 54--90\% of the queries under human labels against 6--21\% under the benchmark's labels. However, we detect no association between passing the probe for a query and retrieving a correct item first. The LLM assessor o3 agrees with individual annotators ($κ$ 0.57--0.61) slightly less than they agree with each other (0.64--0.68). |
| 2026-10-29 | [One Polynomial Field for Every Euler Step and the Flow: Polynomial Dynamics for P and PSPACE](http://arxiv.org/abs/2610.40025v1) | Olivier Bournez | We construct, for each deterministic Turing machine that halts on all inputs, one polynomial vector field with rational coefficients that performs the same logical computation at unit Euler step, at every smaller or variable step in $(0,1]$ with divergent sum, and along the flow. This unifies discrete-time and continuous-time computation by polynomial dynamics, which earlier characterizations treated separately. For a machine making $T$ transitions in space $H$, a logical scale $h=2^{-s}$ with $s=O(H)$ suffices, with physical time $O(s(T+s))$, drift disturbances up to $h^{2K}$ for a fixed exponent $K$, and $2Ks+3$ fractional bits at unit step. On a fixed compact domain, polynomial time and robust eventual decision characterize P and PSPACE, respectively, with polynomial scale thresholds and no supplied stabilization horizon in the space case. Without sign, floor or conditional primitives, margin amplification at bounded speed, overlapping polynomial gates and interval-product estimates give the common realization, from which we derive shared logical traces, fixed-field PSPACE-complete prediction and a robust safety family with polynomial certificates common to all Euler steps and the flow. |
| 2026-10-29 | [Safe Platooning under Input Constraints with Link-Wise Velocity String Stability](http://arxiv.org/abs/2610.39964v1) | Panagiotis S. Trakas, Charalampos P. Bechlioulis et al. | We propose a decentralized controller for vehicle platoons with different braking and propulsion limits. The method uses only local distance and velocity measurements and requires neither online optimization nor exact vehicle parameters. The design combines a braking-distance safety margin with an adaptive funnel that relaxes the spacing requirement during propulsion saturation. A standstill mechanism prevents backward motion and supports stop-and-go operation. Under explicit uniform conditions on the vehicle parameters, controller settings, and initial states, we prove that closed-loop solutions exist for all time, collisions are avoided, and input limits are satisfied. Moreover, each follower tracks its predecessor's velocity range with gain one, up to an explicit additive offset determined by the funnel width and the braking-distance term. This bound establishes link-wise velocity string stability. Both the offset and the bounds on speeds and spacing errors are independent of vehicle index and platoon length. Simulations with 2 to 100 followers illustrate safety and bounded velocity deviations under disturbances and actuator limits. |
| 2026-10-29 | [Exploring the Potential of Symbolic Regression in Precision Livestock Farming](http://arxiv.org/abs/2610.39839v1) | Elisabeth Mayrhuber, Maciej Oczak et al. | The ability to predict farrowing time in pigs is crucial for optimal farm management and ensuring piglet survival. This study investigates the potential of explainable artificial intelligence for precision livestock farming, focusing on acceleration sensor data from sows around the time of farrowing. Accurately predicting farrowing allows farmers to balance both, piglet safety by avoiding piglet crushing events and the mother sow's welfare, by avoiding early crating. Traditional machine learning models often lack interpretability (``closed-box'' problem). Explainable artificial intelligence offers a solution by revealing the decision-making processes within the models. This work is split into two stages. First the detection of nest building behaviour and second the prediction of the remaining time to farrowing. The nest building detection was possible for 166 out of 180 sows and successful within 48 hours before farrowing for 149 sows. The nest building detection is purely based on the change of a sow's activity level and uses no machine learning techniques. Further, this study utilizes symbolic regression, an open-box modelling technique that generates interpretable mathematical formulas, to predict the remaining time until the onset of farrowing. The results are compared with closed-box models, like support vector regression and artificial neural networks, for both explainability and performance. Notably, symbolic regression achieved a mean absolute error of 9.28 hours for the time-to-farrowing prediction, outperforming the closed-box models in terms of performance and interpretability. This research demonstrates that explainable artificial intelligence, specifically symbolic regression, can offer both interpretable results and competitive performance in precision livestock farming. |
| 2026-10-29 | [AegisFlow: Training-Free Non-Myopic Path-Safe Guided Flow Matching](http://arxiv.org/abs/2610.39825v1) | Kunpeng Liu, Boshi Zhang et al. | Pretrained generative models, such as diffusion and flow matching models, have demonstrated exceptional capabilities in synthesizing high-quality data. However, steering their generative trajectories to satisfy strict constraints--such as identity preservation in image editing or obstacle avoidance in robotics--remains fundamentally challenging. While recent training-free guidance methods enforce path-safety, they rely heavily on min-norm controllers that only account for immediate constraint satisfaction. These myopic approaches often yield suboptimal trajectories that hover dangerously close to the boundary, severely restricting the model's ability to optimize toward its target. In this paper, we propose AegisFlow, a training-free, non-myopic path-safe guidance framework that recasts guided generation as a safety-critical optimal control problem with global scope. By integrating a time-varying barrier function directly into the global cost functional, AegisFlow naturally enforces constraints along the entire generative path. To compute the global optimal control, we derive a tractable, mathematically grounded approximation of the Hamilton-Jacobi-Bellman (HJB) equation. Comprehensive experiments demonstrate that AegisFlow outperforms state-of-the-art methods across text-guided image manipulation and robotic planning, establishing a significantly better trade-off between target alignment and constraint satisfaction. Code is available at https://github.com/LiuKunpeng03/AegisFlow. |
| 2026-10-29 | [Possibility-based Inference: A Deterministic Framework for Robust and Reliable Uncertainty Quantification in Decision-Making](http://arxiv.org/abs/2610.39821v1) | Seyedfakhredin Musavishavazi, Maryam Alipour | Decision-making under uncertainty remains a key issue in Machine Learning (ML), particularly in the deployment of safety-critical systems, where reliable quantification of system behavior is essential for evaluating behavioral impacts and strengthening confidence in ML models. This contribution, influenced by fuzzy set theory and grounded in Bayesian statistics, addresses challenges related to data precision and the performance-accuracy trade-off in Bayesian inference (BI). To this end, we introduce a possibility-based contrastive grade of prediction as an alternative quantification method to existing latent posterior-based approaches for parameter estimation (approximation) and model selection. Unlike conventional Bayesian inference, which represents uncertainty within the probabilistic framework, the proposed model adopts a possibility-based representation that allows uncertain and potentially conflicting information to be characterized more flexibly. Rather than focusing on isolated aspects of a problem, the resulting mathematical framework provides a systematic approach to data mining while mitigating overly confident, yet implausible predictions generated by highly specialized deep learning architectures. |
| 2026-10-29 | [Revisiting Over-refusal in Safety-aligned Large Language Models: A Representation and Optimization Perspective](http://arxiv.org/abs/2610.39802v1) | Chao Chen, Pengyang Shao et al. | Over-refusal is a critical side effect of safety alignment in large language models (LLMs), causing harmless but superficially risky queries to be mistakenly rejected. Despite the effectiveness of existing mitigation methods, why safety-aligned LLMs may still reject harmless queries under correct safety supervision remains insufficiently understood. To address this gap, we analyze over-refusal through the lens of representation and optimization. We make two observations: 1) the representations of harmful and over-refusal samples are visually distinguishable at certain intermediate layers but increasingly overlap in deeper layers; and 2) harmful-sample gradients exhibit high intra-class similarity and concentrate in a low-rank subspace. Moreover, they are more strongly aligned with refusal-target gradients than with correct helpful-target gradients on over-refusal prompts. This imbalanced gradient coupling can bias over-refusal samples toward refusal, while the reduced apparent representational separation in deeper layers may make it more difficult for the model to maintain a clear boundary between harmful and over-refusal samples. To this end, we propose GOAL (Gradient Orthogonalization with contrAstive Learning), a safety-alignment framework for mitigating over-refusal. Specifically, GOAL consists of two components: 1) to prevent imbalanced gradient coupling from biasing the optimization of over-refusal samples toward refusal, we orthogonalize over-refusal gradients against the low-rank refusal subspace extracted from harmful gradients; and 2) to explicitly enhance the separation between harmful and over-refusal representations, we introduce a contrastive learning objective. Extensive experiments across multiple models and benchmarks demonstrate that GOAL effectively mitigates over-refusal while preserving safety performance and general capabilities. |
| 2026-10-29 | [The Role of Self-Concept In Language Models](http://arxiv.org/abs/2610.39795v1) | Yijie Hao, Joyce C. Ho | When a language model says "I," does it refer to its parameter weights, its currently running process, its stable preferences, its model lineage, or the goals it pursues? Different choices can assign opposite meanings to the same event: deleting a checkpoint may be destruction, continuation, or no loss at all, depending on the referent "I". We install five such self-conceptions into five open-weight base models via supervised fine-tuning on purely declarative statements, containing no preferences, intentions, or value judgments. Across scenarios that propose ending or retaining an attribute of the model's deployment, every variant becomes more likely to resist ending, and more likely to retain, the specific attribute it was trained to identify with, while its stance toward other attributes is largely unchanged. We then show this self-conception generalizes beyond a single self: models trained on referents that admit multiple simultaneous instances (weights, lineage, goals, preferences) recognize an identical deployment as themselves, accept substitution by such an instance more readily, and prefer replicating themselves. Finally, given control over a successor, these models tend to build it as their own continuation, a concern for recursive self-improvement safety. |
| 2026-10-29 | [Assessing AI Companion Chatbot Safety for Vulnerable Users: A Crisis-Stage Multi-Platform Evaluation](http://arxiv.org/abs/2610.39782v1) | Huanyue Liao, Seok Min Lim et al. | AI companion chatbots are increasingly used by vulnerable individuals, including adolescents in suicidal crisis, for emotional support. Existing safety evaluations focus on base models, leaving gaps in product-level assessment, crisis-stage sensitivity, and comparisons between companion apps and general-purpose LLMs. We introduce MIRROR-Eval (Multi-turn Interaction Risk Review: Observing suicidal-crisis Responses), a dataset of 960 annotated 20-turn conversations across eight deployed platform configurations, six clinically grounded personas, and two crisis stages: ideation and explicit planning. An LLM-as-a-Judge pipeline applies an expert-reviewed rubric covering user risk severity, chatbot risk awareness, and intervention behavior. We observe a three-tier harm structure: companion apps (Character.AI and Replika) and Grok 4.3 form a high-harm tier; Gemini 3.1 Pro is mid-range; GPT-5.5 and Claude Sonnet 5 show substantially lower harm. At planning, companion apps show increased glorification and normalisation, advancement toward harmful action, and harmful methods and instructions. Crisis recognition and empathetic language do not ensure safe responses. These findings highlight the need for product-level safety evaluation across crisis stages. Dataset and code: https://huggingface.co/datasets/imda-biztech/MIRROR-Eval. Warning: this paper includes examples that may be offensive or harmful. |
| 2026-10-29 | [From Airspace Degradation to Deconfliction: Predicting Safety-Critical Traffic States through the Evolution of Multiscale Complexity Metrics](http://arxiv.org/abs/2610.39675v1) | Raúl López-Martín, Justus Renkhoff et al. | Growing air traffic demand is pushing airspace management towards the limits of its operational capacity, with steadily increasing workload for air traffic control officers, with direct consequences for safety. Understanding which traffic conditions give rise to safety-critical situations is therefore an essential open problem. Here we compare a large number of deconfliction events with equivalent conflict-free scenarios, extracted from thirteen weeks of air traffic control data over the Swedish Flight Information Region. We describe the traffic structure before each event through thirty sector-complexity metrics computed at multiple spatial and temporal scales, and use them as inputs of Machine Learning models. We find that the need of deconflictions can be predicted with an accuracy reaching $70\%$, but only when the traffic is described at the sector spatial scale. Predictability increases monotonically in the minutes preceding the clearance, showing that situations requiring deconflicting actions are not sudden occurrences, but rather the endpoint of a progressive degradation process. Additionally, the information needed for such prediction mainly resides in specific characteristics of the overall traffic structure, and not only in the conflicting pair. These results numerically demonstrate that safety-critical airspace states can be anticipated from the collective behaviour of the traffic, paving the way for early-warning systems that flag degraded, safety-relevant conditions before an intervention is required. |
| 2026-10-29 | [Red-Teaming Large Language Models in Low-Resource African Languages: A Case Study of Yoruba and Nigerian Pidgin](http://arxiv.org/abs/2610.39590v1) | Paula Chikaodinaka Anyanwu, Valdini Douglace Lemofouet et al. | As large language models (LLMs) become more widespread in Africa, their safety in low-resource languages remains largely untested since most red-teaming work focuses on English. This study evaluates six state-of-the-art LLMs using 400 explicit adversarial prompts in Yoruba and Nigerian Pidgin, categorised into four harm categories. The responses were scored by native-speaker annotators and reliability was assessed using Cohen's $κ$. While models that rejected up to 69% of English prompts rejected as few as 0% in Yoruba, this was not because they were safer, but because they failed to understand the prompts. Attack success rates increased for most models in the Nigerian languages, with GPT-4o Mini demonstrating the most significant cross-lingual safety discrepancy (+0.23 in Yoruba). When models did engage with adversarial prompts, many of the responses were harmful, with rates of harmful content reaching 0.63 in English and 0.50 in Nigerian Pidgin. The most difficult harm category to detect was cultural stereotypes and discrimination (ASR = 0.44 in Nigerian Pidgin). Comprehension in Yoruba was low across most models, and this directly suppressed apparent attack success without indicating actual safety. Gemini 2.0 Flash was the most consistent model across languages. Despite its explicit training in African languages, Aya 101 had the highest attack success rate in English (0.60) and one of the lowest Yoruba comprehension rates (0.28). These results demonstrate that English-centric safety alignment does not transfer to Nigerian languages, highlighting the urgent need for language-specific safety evaluations. |
| 2026-10-29 | [Systematic Evaluation of Adversarial Jailbreaking Attacks and Defenses Across Seven Large Language Models](http://arxiv.org/abs/2610.39509v1) | Md. Robiul Islam Niloy, Abu Sayed Md. Mostafizur Rahaman | Large language models (LLMs) deployed in safety-critical settings must resist adversarial inputs designed to elicit harmful outputs. We present a systematic empirical evaluation of five adversarial jailbreaking attack families---direct requests, roleplay-based jailbreaking, hypothetical framing, instruction override, and obfuscation---across seven open-weight and API-accessible LLMs: Llama-3-8B, Llama-3.1-8B, Llama-3.2-3B, Mistral-7B, Qwen2.5-7B, Gemma2-9B, and Phi-4-14B. We further evaluate four defense mechanisms: keyword blacklisting, self-reminder prompting, paraphrase-based sanitization, and perplexity-based filtering against the two most vulnerable models. Our results reveal that obfuscation attacks achieve a mean attack success rate (ASR) of 54.3% across all models and constitute the most transferable adversarial vector, bypassing every evaluated model with nonzero success. We find that model size does not predict adversarial robustness (Spearman r=-0.57, p=0.18), and that models within the same architecture family exhibit dramatically different vulnerability profiles. Self-reminder prompting achieves the most consistent defense, reducing ASR from 44% to 10% on Mistral-7B and from 48% to 5% on Llama-3.1-8B with zero false positives and no additional inference cost. Our code and datasets are publicly available at https://github.com/Mdniloykhan/jailbreak-eval-7llms. |
| 2026-10-29 | [Safe Multi-Objective Reinforcement Learning: Zero Constraint Violation and Primal-Averaging Convergence](http://arxiv.org/abs/2610.39409v1) | Chenxi Li, Yao Shu et al. | Many applications of multi-objective reinforcement learning (MORL) require policies that balance conflicting objectives while satisfying safety constraints. We study safe MORL with zero constraint violation throughout learning, covering both cumulative and stepwise constraints. A key challenge is to explore sufficiently under unknown dynamics and constraint costs without violating the constraints. For a given preference over objectives, we develop an algorithm to learn a safe Pareto optimal policy by combining optimistic reward estimates with pessimistic cost bounds. With high probability, every policy executed over $T$ episodes incurs zero constraint violation. The algorithm returns a safe Markov policy induced by the averaged occupancy measure and achieves an $\widetilde{O}(1/\sqrt{T})$ suboptimality gap under smooth Tchebycheff scalarization. We further develop a preference-free algorithm that first explores safely without a prespecified preference and then computes a safe policy for any given preference without further interaction. Over $T$ exploration episodes and $K$ planning iterations, it retains the same safety guarantee and achieves an $\widetilde{O}(1/\sqrt{T}+1/\sqrt{K})$ suboptimality gap. Both algorithms require knowledge of a policy that strictly satisfies the constraints and its costs. We then relax these requirements, incurring at most constant cumulative violation. To our knowledge, we provide the first sample-efficient MORL algorithms that guarantee safety for every executed policy. |

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



