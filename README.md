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
| 2026-08-11 | [How to Verify Consistency of Probabilistic Claims](http://arxiv.org/abs/2608.11181v1) | Orr Paradise, Oliver Richardson et al. | When a probabilistic predictor answers many conditional-probability queries, are its answers self-consistent, and can this be verified in polynomial time? This problem is of interest for AI safety, where safety is derived from honesty about probabilistic predictions of unwanted outcomes potentially caused by an AI action. We construct an interactive PCP as follows. Let a predictive model be specified by a probability circuit P and a circuit Q which outputs confidence in predictions. Together, P and Q implicitly specify exponentially many probabilistic claims. We show a protocol in which a polynomial-time verifier can verify the approximate consistency of (P,Q). The verifier is given the pair of circuits (P,Q), which it evaluates at only a few points; alongside them it is given a proof oracle, an encoding of a witnessing probability distribution allegedly consistent with the predictions of (P,Q), which it reads at a few locations while interacting with a single untrusted prover. En route, we must ensure the existence of a sparse witnessing distribution consistent with the model's predictions. To do so, we first consider witness distributions for the consistency of explicit probabilistic claims, rather than claims specified by a predictor: say m claims, each of the form Pr[Y = 1 | X = x] = p, over n Boolean variables. Building on work initiated by Nilsson (Artif. Intell., 1986), we place l_2-approximate probabilistic consistency of explicit claims in NP, with certificates of length O(mn + log B) in the input bit-precision B; we further show how a small additive completeness-soundness gap removes the dependence on B. Together these results provide a complexity-theoretic foundation for certifying the self-consistency of probabilistic predictors. We view our interactive PCP as a first step toward training predictive models to prove their own consistency. |
| 2026-08-11 | [From Interpretability to Control: Insights from Six Years of the TrustNLP Workshop](http://arxiv.org/abs/2608.11171v1) | Rahul Gupta, Abhinav Mohanty et al. | The Workshop on Trustworthy Natural Language Processing (TrustNLP), co-located with major ACL conferences since 2021, has grown from 8 proceedings papers to 41 over six editions, documenting a field-wide transition from post-hoc interpretability of static models to mechanistic understanding and proactive control of generative systems. We synthesize insights from all 144 proceedings papers, classifying them along six trust dimensions grounded in established frameworks (TrustLLM, DecodingTrust). We observe co-occurrences with capability emergence. The release of the first high-impact chat models activated all trust dimensions simultaneously, while subsequent model generations shifted focus toward truthfulness and safety alignment. Analysis from the classification study reveals that truthfulness is the fastest-growing dimension (absent in 2021-2022, comprising 37% of papers by 2025-2026), fairness remains the most consistent theme, and explainability exhibits a U-shaped trajectory; declining as post-hoc methods lost relevance but resurging in 2026 through mechanistic interpretability. A cross-venue comparison with ACL, NAACL, EACL, and EMNLP (~2K papers) in the same period shows that TrustNLP's topical distribution closely follows the field average. We identify four structural insights and conclude with actionable directions for the research community. |
| 2026-08-11 | [The Illusion of Cross-Lingual Safety in Low-Resource Languages](http://arxiv.org/abs/2608.11146v1) | Abigail Oppong, P Sam Sahil et al. | Safety alignment in large language models (LLMs) is largely developed in English, assuming these safeguards generalize across multilingual settings. However, this assumption remains underexplored and exposes a vulnerability in low-resource languages. We investigate cross-lingual safety transfer in four African languages, Twi, Hausa, Amharic, and Swahili, using LoDNA, a new safety dataset that pairs literal translations with culturally localized prompts. To move beyond generation-based evaluation, we propose a latent geometric framework that probes hidden-state refusal representations in LLMs. Our experimental results show that cross-lingual safety transfer is severely limited; harmful prompts retain less than 10% of the English refusal signal across most language-model pairs. Literal and localized prompts are semantically aligned (cosine 0.95-0.996) but drift across layers, suggesting models encode the concepts without routing them to safety mechanisms. These findings demonstrate that current multilingual safety alignment is superficial, providing strong evidence against the assumption of a universal, language-agnostic harm manifold within the specific low-resource languages studied. Warning: This paper contains example data that may be offensive or harmful. |
| 2026-08-11 | [Data Attribution of Emergent Misalignment with Persona Features](http://arxiv.org/abs/2608.11025v1) | Clemens Vetter, David Kaczér et al. | Emergent misalignment (EM) is the phenomenon where fine-tuning a language model on a narrow task leads to harmful behavior in unrelated domains. A leading mechanistic account attributes EM to persona features: latent directions acquired during pre-training that misaligned fine-tuning amplifies. We ask where these features come from: which pre-training documents activate them, and whether naturally occurring human-written text suffices to induce EM. Using Sparse Autoencoder (SAE) based model diffing across four open-weight models, we find that features related to jailbreak personas, sarcasm, deception, and manipulation are amplified by misalignment fine-tuning, while safety-relevant and assistant-identity features are suppressed. Steering individual features controls EM in both directions: it induces misalignment rates of up to 62% in aligned models -- exceeding the 35% reached by misalignment fine-tuning itself -- and re-aligns misaligned models to near-baseline misalignment rates. Attributing the causal features to a corpus of one million pre-training web documents retrieves semantically relevant narratives about villainous characters, domination, and harmful agency. However, fine-tuning on these human-written documents does not reliably induce EM, even after reformatting into assistant-style responses, whereas synthetic instruction-response pairs derived from the same content do -- and transfer across model families. Semantic relevance alone is therefore not sufficient: response structure or model-generated phrasing plays an important role in inducing EM. |
| 2026-08-11 | [SafeCA: Safe Cross-Attention Localization and Regulation for Text-to-Video Jailbreak Defense](http://arxiv.org/abs/2608.10933v1) | Siyuan Liang, Yupeng Qiu et al. | Text-to-Video (T2V) generative models are vulnerable to jailbreak attacks in real-world deployment, leading them to produce harmful or inappropriate content. Existing defense approaches mainly rely on input filtering or reconstruction, which not only incur high computational latency but also tend to distort semantics. To address these issues, we experimentally and systematically analyze the differences between clean and jailbreak samples in the cross-attention feature space, revealing for the first time a cumulative separation effect and a progressively increasing trend of linear separability between the two during the diffusion process. Based on this insight, we propose SafeCA, a feature-level defense mechanism for safe cross-attention localization and regularization. Firstly, we identify key defensive regions and values through attention stability analysis using cross-attention features collected from clean prompts within a single inference. Secondly, SafeCA mitigates anomalous activations via attention masking with energy normalization and introduces a lightweight semantic-space adapter to redirect abnormal semantic flows. Furthermore, we detect and suppress potentially malicious tokens by back-propagating feature anomaly signals to the input cue words, thereby enhancing the deployability of the defense in commercial models. Experimental results show that SafeCA reduces the jailbreak success rate by about 20% on mainstream T2V models, adds almost no inference overhead (+0.1s), and maintains good text-video semantic consistency. Overall, SafeCA provides an architecture-level, deployable protection paradigm for T2V generation models. |
| 2026-08-11 | [ComBodied Agents: a New Paradigm of Human-Centric Agentic AI](http://arxiv.org/abs/2608.10915v1) | Qianggang Ding, Xingyao Wang et al. | After an older adult misses a medication dose, a software agent can send another reminder and an embodied agent can bring the medication. Yet neither explains whether the person forgot, is confused, has side effects, or deliberately refused, nor what support is appropriate. This reveals a structural gap in Agentic AI: Digital Agents primarily transform software states, while Embodied Agents transform physical states; neither makes a person's evolving state and agency the primary object of modeling, intervention, and evaluation. We introduce Combodied Agents, a human-centered paradigm that perceives, models, predicts, and supports individual human-state trajectories over time, using software tools, sensors, wearables, robots, and human services as action channels rather than end goals. We unify fragmented capabilities across personal assistants, health agents, AI companions, and adaptive human--AI systems into a closed loop: event-based multimodal perception reconstructs meaningful personal events; longitudinal, correctable memory provides temporal context; Personal World Models estimate future personal states and outcomes under alternative decisions and interventions; and an admissible intervention policy selects proportionate support under consent, uncertainty, safety, reversibility, and user control. Feedback from the person and environment updates the loop. Rather than requiring an exhaustive Human Digital Twin, the framework uses purpose-bounded, uncertainty-aware, user-correctable representations. We organize the design space by human-state targets, relational contexts, and agent roles, and propose scenario-centered evaluation, agency-preservation metrics, benchmark requirements, edge-native personal models, and governance directions. Combodied Agents shift Agentic AI from external task completion toward sustained human benefit. |
| 2026-08-11 | [VIDS-Seg: Towards Reliable Uncertainty Quantification in Pediatric Cardiac Ultrasound Segmentation](http://arxiv.org/abs/2608.10903v1) | Paul Fischer, Ece Ozkan | Reliable clinical deployment of machine learning requires models that know when they are likely to fail, particularly for subgroups underrepresented in training data. A common case is pediatric care, where models trained on adult cohorts can silently under-perform on children with no indication that something has gone wrong. As retraining with labeled pediatric data is often infeasible, detecting such failures at inference time is a critical clinical need. Building on the VIDS (Variational Inference under Distribution Shifts) framework, we introduce VIDS-Seg, which applies amortized variational inference over a lightweight prediction head to make this adaptive, OOD-aware prior tractable for dense image segmentation. We evaluate VIDS-Seg on left ventricular segmentation in echocardiography, a setting where pediatric anatomy differs systematically from the adult population most segmentation models are trained on, training on an adult cohort (EchoNet-Dynamic) and evaluating zero-shot on a pediatric cohort (EchoNet-Pediatric). Across all age strata, VIDS-Seg matches competitive baselines in segmentation accuracy while producing substantially higher spatial correspondence between predicted uncertainty and segmentation error, an advantage that persists even after applying temperature scaling to all baselines. Downstream, it yields more accurate and stable ejection fraction estimates and more reliable detection of cardiac malfunction in the infant subgroup. Our results indicate that OOD-aware uncertainty quantification can serve as a practical safety layer for deployed segmentation models, enabling detection of silent failures in underrepresented subgroups without retraining or additional labeled data. |
| 2026-08-11 | [ConfTriage: A Calibration-Aware LLM Triage Framework for Pulmonary Nodule Malignancy with Selective Specialist Deferral](http://arxiv.org/abs/2608.10885v1) | Md Rabiul Islam, Samir Abdaljalil et al. | Pulmonary nodule malignancy prediction typically depends on image-trained specialist deep learning (DL) models that require substantial annotated imaging data and task-specific training. We investigate whether a generalist large language model (LLM), reading only a faithful natural-language rendering of standard nodule attributes, can serve as a calibrated triage layer. We propose ConfTriage, a confidence-calibrated method built on three pillars: language as the modality, calibration as the safety mechanism, and a selective specialist DL backstop for low-confidence cases. We prove two guarantees: a finite-sample combined-error bound yielding an explicit per-threshold operational certificate, and an oracle inequality showing that excess risk over the Bayes-optimal deferral classifier is controlled by the L1 calibration error of the LLM probability. A controlled seven-way input ablation across five frontier LLMs on LIDC-IDRI shows that natural-language descriptions dominate the diagnostic signal, while low-level image statistics are essentially diagnostically vacuous. ConfTriage achieved an F1 score of 88.22% and an AUC of 0.92, resolving 76.5% of cases using zero-shot LLM inference alone and referring only uncertain cases to the specialist DL backstop. These results demonstrate that clinically meaningful diagnostic information can be captured through structured radiological descriptions and leveraged by calibrated LLMs for selective referral. The framework suggests a practical pathway for combining generalist LLM prediction with specialist AI models in medical decision-support systems. Source code is publicly available at https://github.com/rabiul-ai/ConfTriage. |
| 2026-08-11 | [Robust Safety Filtering for Input-Constrained Underactuated Linear Systems](http://arxiv.org/abs/2608.10872v1) | Muhamad Rausyan Fikri | We present a robust safety-filtering framework for input-constrained underactuated linear systems subject to unknown disturbances. A baseline H-$\infty$ input is derived from a zero-sum differential game, while a disturbance observer supplies an estimate and a transient error bound. The baseline input is adjusted using the disturbance estimate, while the estimate and its error bound are used to define robust high-order control barrier function constraints; forward invariance holds as long as the admissible-input set remains nonempty. For scalar-input systems, pointwise feasibility is determined from an exact input interval, and the interval width defines the feasibility margin. A finite-horizon H-$\infty$ performance balance accounts for the accumulated deviation of the applied input from the baseline H-$\infty$ policy. Simulations on a linearized two-wheeled balancing robot show how position and body-pitch constraints compete for the same bounded wheel-torque input. |
| 2026-08-11 | [Cross-modal topology decodes battery faults from sparse voltage snapshots](http://arxiv.org/abs/2608.10825v1) | Jinwen Li, Yunhong Che et al. | Battery safety remains the primary bottleneck for mass electric vehicle (EV) adoption, yet field monitoring is hamstrung by a fundamental asymmetry: complex electrochemical faults must be diagnosed via sparse, low-frequency voltage measurements. Existing methods struggle to resolve the signal ambiguity between overlapping fault modes without hardware upgrades. Here, we demonstrate that these distinct fault fingerprints are not lost, but topologically folded within voltage snapshots. We introduce DeFault, a cross-modal diagnostic framework that mathematically unfolds one-dimensional voltage sequences into multi-dimensional phase-space topologies. DeFault employs a bidirectional cross-attention mechanism that acts as an autonomous, physics-aligned filter, explicitly decoding compounded fault modes that remain fundamentally invisible to sequence-based methods. Validated on a field dataset of 16.4 million data records from 99 in-service EVs, our method achieves an average accuracy of 0.96 and an F1 score of 0.84 for four fault types using only 500-second snapshots (spanning <100 mV). This work proves that high-fidelity, interpretable electrochemical diagnosis is achievable on legacy fleets without new sensors, providing a scalable solution for the battery safety crisis. |
| 2026-08-11 | [Dual Stress: Runtime Safety Monitoring for Safety-Constrained MPC Navigation](http://arxiv.org/abs/2608.10791v1) | Jamil Chahine, Wenqi Cai et al. | Runtime hazard monitors for autonomous naviga- tion are conventionally built from geometric quantities: predicted clearance, time to collision, and required deceleration. A model-predictive controller that enforces safety through explicit con- straints computes, as a by-product of every control step, a second information channel that such monitors ignore: the Karush-Kuhn-Tucker multipliers of its constrained optimization, which measure the marginal control effort spent to maintain safety against each obstacle. This paper evaluates whether a horizon-weighted sum of those multipliers, a dual stress signal, provides a hazard monitor complementary to the geometric warnings the same state already supports. We compare it against a battery of fifteen geometric detectors tuned to a matched false-alarm budget, on preregistered held-out crossing scenarios driven through a physics simulator. The stress alarm actionably flags 4.7 times as many collisions missed by the entire geometric battery as the geometric battery flags in return (85 versus 18); combined, the two channels warn of three quarters of the collisions for which braking remained feasible, against under half for the geometric battery alone. |
| 2026-08-11 | [Rethinking LLM Verification: Evidence Structure, Uncertainty, and Selective Refinement](http://arxiv.org/abs/2608.10725v1) | Uma Ranjan, Kunal Tilaganji et al. | Large language models (LLMs) often rely on shortcuts rather than systematic reasoning, raising safety concerns in medical applications. Allowing models to abstain when uncertain improves reliability but introduces a coverage accuracy tradeoff. We propose a two-stage framework for medical hypothesis verification in multiple-choice settings that manages this tradeoff through targeted ontology grounding, applied only when the model abstains. We show that abstention is not random but reflects genuine uncertainty, with abstained predictions associated with lower confidence. Across two frontier models (GPT-5.5, accessed via the Azure OpenAI API, and DeepSeek-R1), the proposed framework improves question-level accuracy by 9.6 percentage points (82.9% to 92.5%) and hypothesis-level accuracy by 4.2 percentage points (92.0% to 96.2%). Our experiments conducted on MedReason and MedQA show that abstention can be repurposed as a control signal for selective reasoning refinement, achieving knowledge-graph-level performance without explicit knowledge graph construction. |
| 2026-08-11 | [Mixed Choice Multiparty Session Types, Precisely](http://arxiv.org/abs/2608.10704v1) | Jake Masters, Nobuko Yoshida | A precise (sound and complete) subtyping relation $\leq$ specifies that $T'$ is a subtype of $T$ if and only if a program of type $T'$ can always safely replace a program of type $T$ without compromising the safety of a larger program. This paper formulates and proves preciseness of subtyping for mixed choice multiparty session types with session delegation, creation, and interleaving. We prove soundness by developing the first general type system for a full mixed choice multiparty session $π$-calculus. To prove completeness, we introduce the three-party lock, which is a minimal and general form of liveness error for handling interleaved sessions, and we establish a new proof technique based on a construction of scheduler processes which enable exhaustive detection for all failures of the subtyping relation. We then extend the preciseness results to a family of mixed choice multiparty session types. Algorithms for checking (1) subtyping and (2) safety, deadlock-freedom, and liveness of typing contexts are fully implemented and optimised to run in quadratic time with respect to the size of the state space and typing context, and have been evaluated with (mixed choice) case studies from the literature. |
| 2026-08-11 | [Your LLM, Your Style: Behavioral Mode Axes for LLM Behavioral Control](http://arxiv.org/abs/2608.10703v1) | Haoze Liu, Run Liu et al. | Large language models (LLMs) increasingly act in interactive settings where their behavioral styles affect user experience, safety, and downstream decision making. Existing LLM personality studies largely rely on self-report questionnaires administered in first-person settings, making the resulting profiles sensitive to surface elicitation choices and poorly grounded in concrete model behavior. In this work, we introduce a situated behavioral-data (B-data) framework for studying and controlling LLM behavioral personality. We construct 3,200 contrastive behavioral scenarios spanning 20 behavioral patterns and four prompt registers, grounded in validated psychometric facets such as BFI-2, DOSPERT, and HEXACO. Using this framework, we find that LLMs exhibit stable and model-specific behavioral profiles, while also revealing register-dependent shifts across first-person decisions, advice-giving, and task execution. We then show that these behavioral patterns can be controlled through Behavioral Mode Axes (BMAs), activation-space directions derived from contrastive behavioral traces. Compared with response-derived BMAs, which are more prone to trait drift, thought-derived BMAs more faithfully capture the intended behavioral mechanism and provide cleaner control over situated behavioral styles. Our results suggest that LLM personality-like tendencies are better understood not as abstract self-report traits, but as measurable and controllable behavioral modes grounded in concrete interaction contexts. Our code and data are available at https://github.com/lhz191/LLM-Behavioral-Personality. |
| 2026-08-11 | [REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems](http://arxiv.org/abs/2608.10669v1) | Zixing Chen, Xingyuan Liu et al. | Large language model (LLM) agents combine language-based reasoning with external tools to perform complex tasks. Adversarial inputs can exploit interactions between the agent and its environment, causing the agent to violate safety policies during execution. Yet existing evaluations often reduce agent safety to a single attack success rate (ASR), collapsing exposure, execution, observation, and adjudication and potentially conflating actual violations with evidence visibility. We introduce REDAgentBench, an executable framework for autonomous red-teaming and faithful measurement. It derives attacks from explicit safety constraints and associated agent-system vulnerabilities, runs them in isolated service sandboxes, and verifies harmful effects from service receipts and final-state changes. The benchmark contains 1,661 cases across five service surfaces. Across six models and three agent harnesses, macro-average ASR is 65.69%; reported ASR varies with harness and evidence view, while evaluation-context disclosure changes execution behavior. In a state-grounded diagnostic cohort, almost one in five confirmed violations with resolved action anchors occurs after the agent states the relevant constraint or risk, revealing a Recognition--Execution Gap. Finally, a training-free policy reminder reduces confirmed violations by more than 70 percentage points in matched replay. These findings show that executable evaluation can improve safety measurement and identify actionable intervention points. |
| 2026-08-11 | [ProbGuard: Calibrated Safety Risk Estimation from LLM Output Distributions](http://arxiv.org/abs/2608.10621v1) | Xinzhe Huang, Biwu Yao et al. | Recent research on Large Language Model (LLM) safety has widely adopted guardrails to identify unsafe LLM outputs. Existing guardrails typically formulate safety assessment as a deterministic classification task, mapping a discrete token sequence to a discrete safety label. However, this paradigm has two limitations: First, safety assessment is inherently an uncertain problem, particularly during the early generation state. Second, relying solely on discrete token sequences discards the rich probabilistic information embedded in the LLM output distribution. To address these limitations, we propose the first completely probabilistic architecture-agnostic guardrail \textsc{ProbGuard} to leverage the LLM early output distributional signals for estimating and calibrating the safety probability, thereby enabling early stopping of unsafe ongoing outputs. Specifically, given an LLM's generated prefix distribution, we formulate the safety risk as the unsafe probability of its continued generation dynamics and estimate this risk by Monte-Carlo sampling. Through post-training on the distributional signals and calibrated safety risk, \textsc{ProbGuard} achieves the best calibration performance across all nine model--dataset combination settings, reducing the average Brier score and ECE by 79.6\% and 71.9\%, respectively, over the best baseline. \textsc{ProbGuard} further limits the attack success rate to at most 1\% across six representative jailbreak attacks after observing the LLM early output distributions from only the first ten decoding steps. |
| 2026-08-11 | [Toward the Cognitive--Physical Limits of Embodied Intelligence through a World-Model-Centric Autonomous Racing Agent](http://arxiv.org/abs/2608.10618v1) | Zitong Shan, Baichuan Lou et al. | Embodied artificial intelligence aims to develop agents that perceive, reason, and act through continuous interaction with the physical world. However, most embodied systems are still evaluated within conservative safety margins or moderate interaction regimes, leaving their capability boundaries under extreme conditions insufficiently understood. Autonomous racing provides a stringent testbed by combining high-frequency localization and perception, adversarial interaction, near-saturated vehicle dynamics, and strict safety constraints. Existing systems push high-speed performance but rarely model and refine cognitive and physical limits jointly. Here we show that a world-model-centric autonomous racing agent provides a concrete step toward exploring these coupled limits. The framework learns predictive world models from near-limit successes and failures to capture interaction evolution, ego dynamics, and feasible-motion boundaries, coupling world-state construction, future-aware reasoning, and near-limit control in a closed-loop refinement process. Training data were collected from real-vehicle autonomous racing, where the onboard system maintained robust localization and perception at speeds up to 256.3 km/h and peak lateral acceleration of 26.8 m/s$^2$. In full-scale simulated racing, the well trained world-model-centric agent achieves an 88.3% interaction success rate across various challenging simulated racing scenarios. Closed-loop refinement of the world model and policy further improved utilization of cognitive-physical limits, recovery from failure modes, and generalization across varying conditions and unseen circuits. These results suggest a boundary-aware methodology in which world models help embodied agents represent, predict, and continually refine their capability boundaries for safer real-world deployment. |
| 2026-08-11 | [Sensing in Low-altitude Wireless Networks: Systems, Techniques, and Developments](http://arxiv.org/abs/2608.10555v1) | Zihao Tao, Yiming Zhao et al. | The highly dynamic and safety-critical characteristics of low-altitude airspace render sensing an indispensable component of low-altitude wireless networks (LAWN). Although sensing techniques have been extensively studied under diverse paradigms, a prominent mismatch persists between state-of-the-art sensing schemes and the practical sensing demands of LAWN. To fill this research gap, this article systematically reviews LAWN-oriented sensing from the dimensions of system framework, core technologies, and research trends. Specifically, we first analyze the sensing system framework, covering concepts, services and tasks, nodes and targets, and scenarios for LAWN sensing. Next, we conduct a comparative analysis of existing sensing techniques from the perspectives of propagation medium, cooperation, methodology, and modality, analyzing their advantages and limitations. Then, we summarize promising future research directions for deployable LAWN sensing systems, covering non-cooperative and cooperative sensing, model-driven and data-driven sensing, and model-and-data-driven multi-modal sensing. Finally, we present a case study of a model-and-data-driven multi-modal method for real-time aerial target sensing. Compared with existing surveys on LAWN or sensing, this article delivers a more comprehensive, targeted review exclusively centered on LAWN sensing. |
| 2026-08-11 | [Measuring Semantic Abstractness of SAE Features via Nonlocality](http://arxiv.org/abs/2608.10537v1) | Chuqiao Lin, Shivaji Sondhi et al. | Sparse autoencoders (SAEs) have helped uncover mechanistic explanations for LLM behaviours such as reasoning, jailbreaking etc., via understanding the corresponding task-relevant and causally effective features. To evaluate such mechanistic explanations, downstream studies must distinguish surface lexical features from genuinely high-level ones. However, neither an autointerp-based semantic description nor causal steering utility fully resolves the abstraction level of a feature. To this end, we introduce \emph{Feature Nonlocality} (FNL), defined as the entropy of the normalized per-position influence on an SAE feature's activation. We report that FNL correlates with existing LLM-based proxy metrics of feature semantic abstractness, and successfully distinguishes context-dependent reasoning features from token-driven ones, correctly assigning the higher FNL to the contextual feature in $73$--$84\%$ of randomly drawn pairs that consist of one contextual and one token-level feature.   We demonstrate two downstream applications. We audit SAE-based features used for jailbreak mitigation and find surprisingly that most effective features are positional features with low FNL rather than genuinely recognizing harmful intents.   We report that steering high-FNL features in DeepSeek-R1-Distill-Llama-8B improves MATH-500 accuracy by $4.6$ points over the unsteered model and outperforms steering low-FNL features, though the gains are model-specific. We conclude that FNL provides an LLM-independent, label-free, correlational witness of the abstraction level of an SAE feature, with applications in evaluating mechanistic explanations as well as selecting features for downstream interventions. |
| 2026-08-11 | [On Understanding, Identifying, and Mitigating Vulnerabilities in Agentic Large Language Models](http://arxiv.org/abs/2608.10530v1) | Md Jafrin Hossain, Mohammad Arif Hossain et al. | Large Language Models (LLMs) have undergone a shift from stateless conversational interfaces to autonomous agents capable of multi-step planning, tool invocation, code execution, and maintaining persistent memory. When these agents operate with real-world privileges---calling APIs, modifying files, and querying databases---a compromised reasoning step can trigger unauthorized data access, irreversible state changes, or cascading failures, yet the security research community has not kept pace. To quantify the state of the field, we conducted a systematic literature review under PRISMA 2020 guidelines across six databases, screening 743 records and retaining 85 papers (2023--2025) on agentic LLM security. Attack research outpaces defense work by 3.9:1. Perception-layer vulnerabilities (prompt injection, jailbreaking, adversarial perturbations) dominate, accounting for 66\% of papers, while action-layer vulnerabilities (tool misuse, code injection, sandbox escape) appear in only 4.7\%, misaligned with real-world risk. Code execution security accounts for 3.5\%, and tool-augmented agents 12\%. We contribute a four-layer taxonomy mapping 13 vulnerability types across perception, brain, action, and interaction layers, and identify seven open problems centered on containment. Agentic LLM insecurity stems from architectural coupling, where weak isolation allows vulnerabilities to propagate across layers. |

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



