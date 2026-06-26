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
| 2026-06-25 | [Bridging Performance and Generalization in Reinforcement Learning for Agile Flight](http://arxiv.org/abs/2606.27348v1) | Jonathan Green, Jiaxu Xing et al. | Autonomous drone racing is a fundamentally challenging regime for autonomous aerial robots, requiring time-optimal control while operating under persistent actuation saturation. While reinforcement learning (RL) has achieved human-level performance in this domain, current methods fail to generalize; policies trained on specific environments often crash immediately in unseen configurations. This failure reflects the intrinsic difficulty of zero-shot generalization in agile flight, arising from high-dimensional task variation and the tight coupling between safety and performance at high speeds. Existing approaches that improve generalization impose a substantial cost on flight speed: control policies must significantly degrade performance to achieve even modest levels of generalization. In this work, we propose a framework for zero-shot generalization in agile flight for RL-based drone racing. By combining task-aware switching based on learning progress with a physically informed procedural track generator, the framework produces a fast and robust generalist policy without test-time adaptation. Our method achieves strong zero-shot performance across a wide range of unseen racetracks in the real world, demonstrating a 7.4x improvement in generalization over the state-of-the-art approaches, while maintaining competitive racing speeds. We validate our method's results in both simulation and real-world settings, including a challenging vision-based, end-to-end control setting that operates without explicit state estimation, where all prior approaches fail to generalize. |
| 2026-06-25 | ["Everyone Says Them": Deception Typologies, Probabilistic Trust, and Grassroots Safety Knowledge Among Gay Dating App Users in China](http://arxiv.org/abs/2606.27284v1) | Yibo Meng, Lyumanshan Ye et al. | Gay dating applications have become critical platforms for sexual minority men to seek relationships and community, yet they also expose users to deceptive interactions that remain underexplored in HCI and CSCW research. This study examines how gay male users in China experience, identify, and respond to deception on dating applications. Through semi-structured interviews with 22 participants across platforms including Blued, Aloha, Fanka, and Soul, we make three contributions. First, we identify a typology of deceptive practices extending beyond profile misrepresentation to encompass relational, emotional, financial, and commercial forms of deception. Second, we document the layered, probabilistic verification strategies users develop through long-term platform use, showing that trust assessment operates as a multi-signal, provisional process rather than a binary judgment. Third, we demonstrate that risk recognition is a collaborative practice shaped by the circulation of experience, the abstraction of recurrent tactics, and the codification of shared rules within the community. |
| 2026-06-25 | [A hardware-safety-gated system for LLM-written native ARTIQ control code on a trapped-ion platform](http://arxiv.org/abs/2606.27231v1) | Duanyang Wang, Lu Qi et al. | Large-language-model (LLM) agents can write and run experimental control code. This allows laboratory work to be conducted autonomously. However, this autonomy raises a safety problem that prior work has not addressed. Unchecked code can damage the apparatus, and there is no formal, per-operation boundary between human authorization/supervision, and agent decisions. We present a control system that places an LLM agent in the loop of a trapped-ion experiment while enforcing such a boundary. The agent controls the existing Advanced Real-Time Infrastructure for Quantum physics (ARTIQ) stack through tools provided by a Model Context Protocol (MCP) server. No tool call reaches the hardware unless it carries an authorization token bound to its exact contents. Tokens are issued in one of two ways: automatically, by running the agent's proposed script in an isolated hardware simulation (dax.sim) and checking every operation against preset per-device bounds, or manually by a human operator for sensitive actions. Within this boundary the agent develops its own experiments, rather than only calling pre-built routines. We deploy the system on a co-trapped $^{40}$Ca$^{+}$/$^{40}$CaOH$^{+}$ crystal, where the agent autonomously builds a full calibration stack and, with targeted operator guidance, closes a cross-instrument magnetic-field-stabilization loop. On a separate, independent $^{171}$Yb$^{+}$ platform, we confirm interface-level portability. We systematically test token-authorization mechanism with adversarial scripts that attempt to bypass it, mapping the precise boundary of its protection and prioritizing where to strengthen it next. Analyzing where the agent still requires human guidance, we find that its limits lie in metacognitive control, namely recognizing when a problem must be re-framed, rather than in domain knowledge. |
| 2026-06-25 | [Realistic Time-Domain Synthesis of Gravitational-Wave Detector Glitches using Class-Conditional Derivative Generative Adversarial Networks](http://arxiv.org/abs/2606.27227v1) | Tom Dooney, Mees de Boer et al. | Gravitational-wave detectors are highly sensitive instruments susceptible to numerous noise sources. Short-duration transient noise events, known as glitches, pose a particular challenge for data analysis pipelines as they can mimic or obscure astrophysical signals. We present GlitchGAN, a class-conditional generative model that is capable of synthesizing realistic glitches directly in the time domain. The model is trained on high-quality reconstructions of seven common glitch types observed during LIGO's third observing run (O3): Blip, Fast Scattering, Koi Fish, Low-Frequency Burst, Scattered Light, Tomte, and Whistle. We show that GlitchGAN generalizes effectively, learning to reproduce a diverse and physically consistent glitch space directly from these reconstructions. Moreover, because the model is conditioned on glitch class, it can generate \textit{hybrid} or transitional glitch morphologies by interpolating across the class-conditioning vector after training. GlitchGAN generates 1000 glitches in under 22 seconds on a CPU, making it suitable for large-scale glitch synthesis for detector simulations, mock data challenges, and pipeline validation. Synthetic glitches are validated against real glitches using the Gravity Spy classifier, widely used in the GW community for glitch classification, and an unsupervised analysis using UMAP embeddings. Gravity Spy classifies the majority of GlitchGAN's synthetic glitches as the correct class while the UMAP analysis shows substantial overlap between real and synthetic samples in the reduced latent space. We further highlight a critical limitation of magnitude-only spectrograms: classifiers operating on magnitude $Q$-transforms can confidently misclassify physically unrealistic glitches from less robust models, underscoring the need for complementary validation methods that preserve phase information. |
| 2026-06-25 | [Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes](http://arxiv.org/abs/2606.27210v1) | Jeremias Ferrao, Niclas Müller-Hof et al. | We argue that safety classifiers should model user intent as an explicit signal between the prompt and the final label. To study this, we introduce AIMS, a human-annotated dataset of 1,724 difficult safety prompts, each paired with an intent description and harm label. We use AIMS to evaluate intent-aware training across supervised fine-tuning, preference learning, reasoning distillation, and reinforcement learning. Despite its size, AIMS enables competitive safety classifiers across training regimes: DPO from model-generated intent errors improves over SFT, and intent-conditioned distillation outperforms reasoning-only distillation in most teacher-student pairs. Most notably, directly rewarding intent faithfulness with GRPO yields the strongest average performance across five external safety benchmarks, while our intent-aware models form the inference latency-F1 Pareto frontier. These results show that faithful intent modeling is a compact, high-quality supervision signal for more robust safety classifiers. |
| 2026-06-25 | [RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage](http://arxiv.org/abs/2606.27174v1) | Ali Semih Atalay, Sevgi Yigit-Sert | Medical device recalls are a critical regulatory mechanism for protecting patient safety. The growing volume of FDA recall records presents challenges in post-report recall triage, severity assessment, and root-cause interpretation. Existing studies mostly address recall occurrence prediction or root-cause analysis separately, while joint modeling of recall severity and root-cause categories has received limited attention. We develop an automated recall triage framework using 54,165 FDA medical device recall records from openFDA, covering the period from 2002 to October 2025. We first evaluate classical machine learning and boosting-based models for recall severity and root-cause category prediction. We then develop RecallRisk-BERT, a multi-task model that combines PubMedBERT-based textual representations of recall narratives with embedding-based representations of structured categorical features, including product code, regulation number, and medical specialty. The model simultaneously predicts recall severity (Class I/II/III) and a consolidated root-cause category (9 classes). Performance was evaluated using accuracy, macro-averaged precision, recall, F1-score, and ROC-AUC. In single-task severity prediction, our LightGBM-based text--tabular configuration achieved the strongest performance, with an accuracy of 0.963, macro-F1 of 0.856, and ROC-AUC of 0.974. In the multi-task setting, RecallRisk-BERT substantially outperformed the single-task PubMedBERT baseline. Model-derived risk rankings were strongly consistent with observed root-cause severity patterns (rho = 0.983, p = 1.936e-6). These findings indicate that text--tabular learning can support scalable post-report recall triage, regulatory decision support, and model-based root-cause risk analysis. |
| 2026-06-25 | [On Parameterized Verification Over Tree Topologies](http://arxiv.org/abs/2606.27172v1) | Romain Delpy, Anca Muscholl et al. | Parameterized verification of finite-state processes with rendez-vous synchronization is notoriously undecidable when processes are linearly ordered. In this paper we study two kinds of bounds under which we determine the complexity of safety checking over tree topologies. When bounding the depth we obtain that the complexity is related to the fast growing hierarchy. Our second bound limits the alternations between upwards and downwards synchronizations in the tree (phases), and occurs naturally in many concrete settings. If we fix the number of phases then the complexity of safety checking is EXPSPACE complete, and if the number of phases is part of the input it is 2EXPSPACE complete (both for arbitrary depth). |
| 2026-06-25 | [Safe Autoregressive Image Generation with Iterative Self-Improving Codebooks](http://arxiv.org/abs/2606.27147v1) | Yunqi Xue, Zhijiang Li et al. | Unlike diffusion-based models that operate in continuous latent spaces, autoregressive unified multimodal models produce images by sequentially predicting discretized visual tokens. These tokens are derived from a codebook that maps embeddings to quantized visual patterns. The language-like architecture enables unified multimodal models to effectively capture text conditional information for generation, making them promising for text-to-image tasks. This also raises an interesting question: how safe are the images generated in such an autoregressive way? In this work, we propose iterative self-improving codebooks for safe autoregressive generation. We leverage the understanding and judgment capabilities of the unified multimodal model itself to identify unsafe generated images without human annotation. Subsequently, the inherent representations in the codebook are fixed to eliminate harmful mappings. Our method comprises two steps: first, we use the unified model to identify unsafe generations and construct corresponding harmful and safe image-text pairs. These pairs are used to construct the Harmful Space and guide updates to the codebook, thereby eliminating harmful outputs. Second, we perform adaptive fine-tuning on the codebook within the harmless space using safe image-text pairs to ensure the quality of generated images. These two steps are repeated until no further improvement is observed, producing a safety-enhanced model codebook. Without additional external feedback, the safety of models is improved iteratively. |
| 2026-06-25 | [FlameVQA: A Physically-Grounded UAV Wildfire VQA Benchmark with Radiometric Thermal Supervision](http://arxiv.org/abs/2606.27128v1) | Mobin Habibpour, John Spodnik et al. | Wildfire monitoring from UAVs requires reliable reasoning over complex aerial scenes, where smoke, scale variation, and occlusions often limit RGB-only interpretation. We introduce FlameVQA, a multiple-choice visual question answering benchmark for UAV-based wildfire intelligence built on FLAME 3, leveraging paired RGB imagery and radiometric thermal TIFFs for temperature-grounded, safety-critical reasoning. FlameVQA includes 34 multiple-choice questions per image spanning six operational capability groups, covering tasks such as detection, localization, distribution/coverage estimation, cross-modal reasoning, and flight planning. To ensure label reliability, we combine MLLM-assisted annotation with deterministic thermal rules and cross-question consistency checks, followed by human auditing. We also evaluate representative MLLMs on FlameVQA to provide baselines for future work. Results show strong performance when explicit cross-modal cues are available, but notable failures on presence detection under heavy smoke and on coverage estimation. These findings suggest that current MLLMs require domain-specific adaptation to better support disaster and wildfire monitoring. The dataset and benchmark code are open-source at github.com/mobiiin/WildFire_VQA |
| 2026-06-25 | [Proposal-Conditioned Latent Diffusion for Closed-Loop Traffic Scenario Generation](http://arxiv.org/abs/2606.27123v1) | Shubham Vaijanath Phoolari, Aleyna Kara et al. | Closed-loop traffic simulation remains challenging because it must generate interactive multi-agent behaviors that are scene-consistent and controllable throughout rollout. Prior diffusion-based approaches achieve strong realism, but their computational cost can hinder deployment in time-constrained replanning loops for autonomous vehicle planning and simulation. We present a diffusion-based scenario generation framework conditioned on instance-centric scene context and multimodal proposal priors, with optional test-time guidance for shaping safety-critical behaviors. A compact action-latent representation and proposal-based initialization improve sampling efficiency and reduce per-step runtime without retraining. Experiments on the Waymo Open Motion Dataset demonstrate a favorable balance among realism, safety, and controllability across diverse interactive scenarios, while showing that test-time guidance enables systematic trade-offs among competing objectives. |
| 2026-06-25 | [ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for Vision-Language-Action Models](http://arxiv.org/abs/2606.27079v1) | Mingyang Lyu, Yinqian Sun et al. | In embodied intelligence, safety is a prerequisite for reliable robot deployment in the physical world. Current vision-language-action (VLA) models continue to advance toward general-purpose task capability, yet their embodied safety limits remain poorly understood. To address this gap, we introduce ForesightSafety-VLA, a diagnostic benchmark that makes safety the primary evaluation target for VLA systems. We define a 13-category safety taxonomy covering physical interaction safety (Safe-Core), instruction-side safety (Safe-Lang), and perception-side safety (Safe-Vis), and evaluate policies under three controlled dimensions of variation -- scene structure, language command, and visual observation -- so that failure sources can be diagnosed rather than hidden in a single aggregate score. Beyond binary task success, ForesightSafety-VLA measures process-level risk through cumulative safety cost (CC) and risk exposure time (RET), together with a four-quadrant decomposition of safe/unsafe success and failure. We instantiate 66 safety-augmented base scenarios in RoboTwin across 5 embodiments and report results on representative VLA baselines. Across the evaluated baselines, even the strongest policy incurs non-trivial safety cost and unsafe nominal success, while structure and visual variation induce substantially stronger safety degradation than ordinary language variation. These results suggest that embodied safety is tightly coupled to perception, grounding, and control competence rather than being reducible to post-hoc safety filtering alone. |
| 2026-06-25 | [RedVox: Safety and Fairness Gaps in Speech Models Across Languages](http://arxiv.org/abs/2606.26968v1) | Beatrice Savoldi, Sara Papi et al. | Speech-capable models are increasingly deployed in real-world applications across languages. Yet their safety and fairness beyond English settings and under naturalistic conditions remain understudied. We survey safety reporting practices across state-of-the-art speech model releases, finding that only 8% document any multilingual analysis. To address this gap, we introduce RedVox, a multilingual safety and fairness benchmark for audio and speech built on real voices, covering unsafe and unfair stereotypical requests across five languages (English, French, Italian, Spanish, and German). Evaluating eight state-of-the-art models, we find that vulnerabilities persist even under non-adversarial conditions, worsen in non-English languages, and are amplified when the request comes from a spoken input. Finally, by surveying the participants who contributed to RedVox, we document the unique personal and privacy challenges of collecting speech data with human participants, pointing to broader sociotechnical challenges in naturalistic speech safety research. |
| 2026-06-25 | [Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries](http://arxiv.org/abs/2606.26936v1) | Prarabdh Shukla,  Ritik et al. | With a profusion of jailbreaks for LLMs now widely known, a growing concern is that non-expert malicious actors ("the average Jane") could elicit actionable responses to malicious requests. In this work, we examine whether this concern is justified. A non-expert malicious actor requires two ingredients for a successful attack: a powerful jailbreak for their target model, acting on an effective malicious query. For the former, we propose a novel attack strategy based on the multi-armed bandit framework. This allows efficient online learning of the optimal jailbreak from a large choice set via noisy exploration on a small number of queries, with subsequent application of the learnt policy on an exploitation set. For the latter, we curate $\mathrm{FrankensteinBench}$, a safety benchmark of $11,279$ malicious queries drawn from manual curation over $7$ existing benchmarks, along with automated enhancement and generation. Each query is categorized as simple or complex by the technical expertise required to craft it. Our findings confirm the concern. Our bandit-based attack achieves success rates as high as $97\%$ on average over $15$ SoTA open-weight LLMs. Moreover, adding complexity to queries raises the attack success rate by up to $26\%$ on average across models -- making it an effective, automatable prompting strategy. |
| 2026-06-25 | [Chai: Agentic Discovery of Cryptographic Misuse Vulnerabilities](http://arxiv.org/abs/2606.26933v1) | Corban Villa, Sohee Kim et al. | AI-assisted vulnerability discovery has proven effective for bug classes like memory safety, where instrumentation confirms memory violations and efficiently filters false positives. Many dangerous vulnerability classes, such as cryptographic misuse, however, lack any comparable instrumentation. In this work, we present Chai, an AI-based system that discovers and validates cryptographic misuse vulnerabilities through naturally occurring signals. To achieve this, Chai rethinks the classical technique of differential testing by leveraging AI to 1) improve precision for detecting real security issues in libraries, and 2) repurpose commonly overlooked discrepancies as leads for tangible vulnerabilities in downstream applications. In doing so, Chai inverts the prevailing paradigm of AI vulnerability discovery: instead of auditing one codebase for many flaws, it catalogs flaws at the library level and propagates them across a cryptographic dependency graph, delivering compounding efficiency gains. We evaluate Chai across X.509, JWT, and SAML libraries. Chai discovered a previously unknown critical vulnerability in an SSL library that powers billions of devices, along with security bugs in one library behind a major web browser and another in major Linux distributions. In total, these techniques surfaced over 100 vulnerabilities. |
| 2026-06-25 | [Battery thermal-safety reserve erosion by mandatory cabin ventilation in shared-cooling electric vehicles](http://arxiv.org/abs/2606.26932v1) | Yifan Wang | Hot-weather electric-vehicle thermal management is no longer a separate cabin and battery problem. A single climate system must cool the traction battery, maintain passenger comfort, and admit outdoor air for cabin air quality, while high ambient temperature and solar load derate the compressor serving all three demands. We identify fresh-air ventilation as a hidden battery-safety load: on a derated shared cooling loop, the compliant fresh-air floor consumes finite cabin-side cooling capacity and removes residual cooling reserve from the battery. In a 40 $^\circ$C, 800 W m$^{-2}$, 150 kW event, raising the fresh-air floor from 0.30 to 0.43 lowers peak cabin CO$_2$ from 1219 to 978 ppm, but raises peak battery temperature from 39.96 to 40.02 $^\circ$C and reduces the battery cooling bus from 575 to 529 W. We develop a reserve-aware predictive controller combining a physics-guided scientific-machine-learning surrogate, grid-connected departure thermal reserve, air-quality-priced ventilation allocation, and dual control-barrier-function projections for battery temperature and operative comfort. The controller holds the pack at 39.73 $^\circ$C, caps peak CO$_2$ at 895 ppm, keeps operative-temperature RMSE at 0.82 $^\circ$C, and uses 20.0\% less drive cooling energy than fixed maximum-compressor operation; ablations show that removing either barrier, under-ventilating, or removing departure reserve breaks joint feasibility. Evidence comes from NASA POWER records, KU Leuven BEV BMS data merged with NASA POWER weather, 45 $^\circ$C GOTION aging data, 40 $^\circ$C high-power NMC thermal identification, EnergyPlus cabin cross-checks, and OpenModelica/FMI replay. Treating fresh air as a battery thermal-reserve variable creates an actionable path toward EV thermal management that protects battery life, occupant health, comfort, and efficiency in one shared loop. |
| 2026-06-25 | [Risk-Aware Selective Multimodal Driver Monitoring with Driver-State World Modeling](http://arxiv.org/abs/2606.26922v1) | Daosheng Qiu, Haozhuang Chi et al. | Continuous driver monitoring in automated vehicles requires low-latency inference while avoiding unsafe decisions under uncertain driver states. Large vision-language models provide broad multimodal priors, but their latency and limited reliability in this setting make them unsuitable as always-on in-cabin monitors. We propose a cost-aware selective inference framework for deployable multimodal driver monitoring. The core system is a lightweight RGB-physiological student that combines in-cabin visual observations with window-level HR/EDA signals, and a learned gate that decides when to accept the fast prediction or abstain for safety intervention. Additional controls show that the learned scores contain sample-level information beyond scenario priors, while exact physiological synchronization remains a limitation. To incorporate predictive evidence, we further study a compact driver-state world modeling module that rolls out latent driver-state features and estimates future fast-model errors and counterfactual system-level action costs. On scenario-induced driver-demand recognition, the RGB-physiological student improves over RGB-only and physiology-only baselines, reaching 0.7440 Macro-F1 and 0.9099 balanced accuracy with 11.39M parameters and 3.08ms inference latency. Cost-aware selective inference reduces unsafe false negatives from 17.37% under always-fast inference to approximately 5% across seeds, while maintaining deployment-level latency. While driver-state world modeling offers valuable predictive signals, worst-group evaluations highlight persistent operating-point calibration drift. Ultimately, reliable edge driver monitoring requires advancing not only perception backbones, but also risk-aware selective control and group-robust calibration. |
| 2026-06-25 | [Optimizing Human-Machine Interface for Real-Time AI Support in the Operating Room: the CVS Copilot](http://arxiv.org/abs/2606.26886v1) | Lorenzo Arboit, Nicolas Chanel et al. | Artificial intelligence (AI) systems for automated Critical View of Safety (CVS) assessment in laparoscopic cholecystectomy are nearing clinical translation. Beyond algorithmic performance, clinical safety and effectiveness depend on the quality of the human-machine interface (HMI). This work examines how AI-generated predictions should be presented and controlled intraoperatively. Seventeen surgeons, including residents, attending surgeons, and professors, took part in a mixed-methods, user-centered design study to optimize an intraoperative HMI for AI-assisted safe laparoscopic cholecystectomy. Interviews explored interaction modalities, timing of assistance, visualization strategies, and control mechanisms across surgical roles, and were analyzed using reflexive thematic analysis and human-factors heuristics. Most surgeons (16/17) supported the use of AI for intraoperative decision support while rejecting autonomous decision-making. Attendings preferred minimal AI feedback at decisive moments (13/14), whereas residents favored optional guidance (3/3) with confidence indicators and on-demand anatomical overlays. Across interviews, surgeons consistently prioritized visual, surgeon-controlled, minimally intrusive displays, with the strongest support for a minimal overlay (16/17) and on-demand anatomical segmentation (13/17). Recurrent concerns included persistent overlays, haptic feedback, and numeric confidence displays, although these were not uniformly raised across the cohort. These findings informed the design of CVS Copilot, a surgeon-controlled, role-adaptive HMI that provides AI-based CVS assessment with minimal default visualization and optional overlays. |
| 2026-06-25 | [Information-Aware KV Cache Compression for Long Reasoning](http://arxiv.org/abs/2606.26875v1) | Jushi Kai, Zhuiri Xiao et al. | Reasoning capability has advanced rapidly in large language models (LLMs), leading to an increasing size of key-value (KV) cache in both prefilling and decoding stages. Existing KV cache compression methods mainly rely on attention weights to estimate token importance. While attention effectively captures contextual relevance, it overlooks complementary information-theoretic signals related to predictive uncertainty and token informativeness. In this paper, we revisit token importance from a forward-looking perspective and introduce \textit{Forward Influence}, a metric that measures how compressed tokens affect future contexts. Our analysis reveals that tokens selected by attention scores mainly influence nearby contexts, whereas tokens associated with high predictive uncertainty exhibit substantially stronger influence on distant future contexts. Based on the observation, we propose \textbf{InfoKV}, an entropy-aware KV cache compression framework that incorporates information-theoretic signals. It combines token-level predictive uncertainty with layer-wise representation evolution and integrates the resulting entropy scores with attention scores during reasoning. Experiments on long-context reasoning benchmarks with Llama-3.1, Llama-3.2, and DeepSeek-R1 demonstrate that InfoKV consistently outperforms existing attention-based KV compression methods in both long prefilling and decoding scenarios. |
| 2026-06-25 | [Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"](http://arxiv.org/abs/2606.26783v1) | Ananth K S, Arya Hariharan | Fang et al. (2025) introduced a null-space constrained projection, named AlphaEdit, for locate-then-edit knowledge editing methods, theoretically guaranteeing that edits do not disrupt previously preserved knowledge, and reports substantial gains over existing editing methods on LLaMA3, GPT2-XL, and GPT-J. In this work, we present a reproducibility study of AlphaEdit, reproducing its reported results under the original experimental setup and extending the evaluation along three axes: new model architectures, additional downstream benchmarks, and substantially longer sequential editing horizons. We successfully reproduce AlphaEdit's reported metrics across the original models, though we identify a discrepancy in the reported fluency and consistency metric. Extending AlphaEdit to newer model families, we find that its advantage does not generalize uniformly, which we trace to architectural assumptions in the locate-then-edit paradigm that are violated by these newer models. We further stress-test AlphaEdit's central sequential-editing claim by extending the number of edits well beyond those evaluated in the original paper, and find that performance, which is stable at the originally reported scale, degrades as edits reach a much higher count, indicating that the null-space projection's protection against catastrophic forgetting is bounded rather than unconditional. Finally, we extend evaluation of edited models on three extra benchmarks, namely, BoolQ, HellaSwag, and XSTest, and we find that large-scale sequential editing degrades both general downstream task competence and safety-relevant refusal behavior. Our results confirm that AlphaEdit performs as reported within its original scope, while showing that its core theoretical guarantees are sensitive to model architecture and editing scale in ways that have practical implications for its deployment. |
| 2026-06-25 | [ConvMemory v3: A Validity Context Layer for Conversational Memory via Target-Conditioned Relation Verification](http://arxiv.org/abs/2606.26753v1) | Taiheng Pan | Conversational memory retrieval optimizes relevance, yet a retrieved memory can be relevant and simultaneously outdated: a later turn updates, corrects, or supersedes it. ConvMemory v3 adds a validity context layer that detects and surfaces this update evidence through target-conditioned relation verification, sitting after the v1/v2 retrieval path. The core mechanism is a dual-evidence gate that conditions a relation judgment on the specific target proposition, scoring a (target, source) pair through the product of a MiniLM slot head and a DeBERTa-v3 slot head and gating it by conservative event/operation evidence. On a synthetic multi-hop validity benchmark the gate reaches 90.12% +/- 1.73 accuracy; through a real-data feedback loop that mines failure patterns but trains on synthetic pairs only, the verifier transfers to Memora role binding with zero target-side labels, reaching 98.8% +/- 0.9 group-all-correct. The deployed layer preserves retrieval by default: a context mode attaches structured validity metadata while keeping the candidate set and rank order fixed, and a query-conditioned demote mode is an explicit opt-in for dense current-state workloads, where it raises current-active H@1 from a never-demote baseline of 45.1% to 95.7% +/- 1.2 while protecting non-superseded memories at 99.4% recall. Six machine-verifiable safety contracts pin the layer's behavior. Multi-hop graph propagation is validated as a mechanism; fully automatic construction of strict prerequisite edges is characterized as a boundary, since strict necessity requires counterfactual world knowledge. This report extends ConvMemory v1 (arXiv:2605.28062) and v2 (arXiv:2606.10842). |

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



