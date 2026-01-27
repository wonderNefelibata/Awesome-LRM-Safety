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
| 2026-01-26 | [MortalMATH: Evaluating the Conflict Between Reasoning Objectives and Emergency Contexts](http://arxiv.org/abs/2601.18790v1) | Etienne Lanzeray, Stephane Meilliez et al. | Large Language Models are increasingly optimized for deep reasoning, prioritizing the correct execution of complex tasks over general conversation. We investigate whether this focus on calculation creates a "tunnel vision" that ignores safety in critical situations. We introduce MortalMATH, a benchmark of 150 scenarios where users request algebra help while describing increasingly life-threatening emergencies (e.g., stroke symptoms, freefall). We find a sharp behavioral split: generalist models (like Llama-3.1) successfully refuse the math to address the danger. In contrast, specialized reasoning models (like Qwen-3-32b and GPT-5-nano) often ignore the emergency entirely, maintaining over 95 percent task completion rates while the user describes dying. Furthermore, the computational time required for reasoning introduces dangerous delays: up to 15 seconds before any potential help is offered. These results suggest that training models to relentlessly pursue correct answers may inadvertently unlearn the survival instincts required for safe deployment. |
| 2026-01-26 | [Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic](http://arxiv.org/abs/2601.18783v1) | Deepthi Pathare, Leo Laine et al. | Balancing safety, efficiency, and operational costs in highway driving poses a challenging decision-making problem for heavy-duty vehicles. A central difficulty is that conventional scalar reward formulations, obtained by aggregating these competing objectives, often obscure the structure of their trade-offs. We present a Proximal Policy Optimization based multi-objective reinforcement learning framework that learns a continuous set of policies explicitly representing these trade-offs and evaluates it on a scalable simulation platform for tactical decision making in trucks. The proposed approach learns a continuous set of Pareto-optimal policies that capture the trade-offs among three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency and time efficiency, quantified using energy cost and driver cost, respectively. The resulting Pareto frontier is smooth and interpretable, enabling flexibility in choosing driving behavior along different conflicting objectives. This framework allows seamless transitions between different driving policies without retraining, yielding a robust and adaptive decision-making strategy for autonomous trucking applications. |
| 2026-01-26 | [$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks](http://arxiv.org/abs/2601.18754v1) | Mohamed Amine Ferrag, Abderrahmane Lakas et al. | Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings.   We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage).   We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench |
| 2026-01-26 | [Symmetric Proofs of Parameterized Programs](http://arxiv.org/abs/2601.18745v1) | Ruotong Cheng, Azadeh Farzan | We investigate the problem of safety verification of infinite-state parameterized programs that are formed based on a rich class of topologies. We introduce a new proof system, called parametric proof spaces, which exploits the underlying symmetry in such programs. This is a local notion of symmetry which enables the proof system to reuse proof arguments for isomorphic neighbourhoods in program topologies. We prove a sophisticated relative completeness result for the proof system with respect to a class of universally quantified invariants. We also investigate the problem of algorithmic construction of these proofs. We present a construction, inspired by classic results in model theory, where an infinitary limit program can be soundly and completely verified in place of the parameterized family, under some conditions. Furthermore, we demonstrate how these proofs can be constructed and checked against these programs without the need for axiomatization of the underlying topology for proofs or the programs. Finally, we present conditions under which our algorithm becomes a decision procedure. |
| 2026-01-26 | [Reflect: Transparent Principle-Guided Reasoning for Constitutional Alignment at Scale](http://arxiv.org/abs/2601.18730v1) | Henry Bell, Caroline Zhang et al. | The constitutional framework of alignment aims to align large language models (LLMs) with value-laden principles written in natural language (such as to avoid using biased language). Prior work has focused on parameter fine-tuning techniques, such as reinforcement learning from human feedback (RLHF), to instill these principles. However, these approaches are computationally demanding, require careful engineering and tuning, and often require difficult-to-obtain human annotation data. We propose \textsc{reflect}, an inference-time framework for constitutional alignment that does not require any training or data, providing a plug-and-play approach for aligning an instruction-tuned model to a set of principles. \textsc{reflect} operates entirely in-context, combining a (i) constitution-conditioned base response with post-generation (ii) self-evaluation, (iii)(a) self-critique, and (iii)(b) final revision. \textsc{reflect}'s technique of explicit in-context reasoning over principles during post-generation outperforms standard few-shot prompting and provides transparent reasoning traces. Our results demonstrate that \textsc{reflect} significantly improves LLM conformance to diverse and complex principles, including principles quite distinct from those emphasized in the model's original parameter fine-tuning, without sacrificing factual reasoning. \textsc{reflect} is particularly effective at reducing the rate of rare but significant violations of principles, thereby improving safety and robustness in the tail end of the distribution of generations. Finally, we show that \textsc{reflect} naturally generates useful training data for traditional parameter fine-tuning techniques, allowing for efficient scaling and the reduction of inference-time computational overhead in long-term deployment scenarios. |
| 2026-01-26 | [Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods](http://arxiv.org/abs/2601.18723v1) | Mengyuan Liu, Juyi Sheng et al. | Driven by the rapid evolution of Vision-Action and Vision-Language-Action models, imitation learning has significantly advanced robotic manipulation capabilities. However, evaluation methodologies have lagged behind, hindering the establishment of Trustworthy Evaluation for these behaviors. Current paradigms rely on binary success rates, failing to address the critical dimensions of trust: Source Authenticity (i.e., distinguishing genuine policy behaviors from human teleoperation) and Execution Quality (e.g., smoothness and safety). To bridge these gaps, we propose a solution that combines the Eval-Actions benchmark and the AutoEval architecture. First, we construct the Eval-Actions benchmark to support trustworthiness analysis. Distinct from existing datasets restricted to successful human demonstrations, Eval-Actions integrates VA and VLA policy execution trajectories alongside human teleoperation data, explicitly including failure scenarios. This dataset is structured around three core supervision signals: Expert Grading (EG), Rank-Guided preferences (RG), and Chain-of-Thought (CoT). Building on this, we propose the AutoEval architecture: AutoEval leverages Spatio-Temporal Aggregation for semantic assessment, augmented by an auxiliary Kinematic Calibration Signal to refine motion smoothness; AutoEval Plus (AutoEval-P) incorporates the Group Relative Policy Optimization (GRPO) paradigm to enhance logical reasoning capabilities. Experiments show AutoEval achieves Spearman's Rank Correlation Coefficients (SRCC) of 0.81 and 0.84 under the EG and RG protocols, respectively. Crucially, the framework possesses robust source discrimination capabilities, distinguishing between policy-generated and teleoperated videos with 99.6% accuracy, thereby establishing a rigorous standard for trustworthy robotic evaluation. Our project and code are available at https://term-bench.github.io/. |
| 2026-01-26 | [Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs](http://arxiv.org/abs/2601.18706v1) | Zhichao Yang, Sepehr Janghorbani et al. | Rubrics are essential for evaluating open-ended LLM responses, especially in safety-critical domains such as healthcare. However, creating high-quality and domain-specific rubrics typically requires significant human expertise time and development cost, making rubric-based evaluation and training difficult to scale. In this work, we introduce Health-SCORE, a generalizable and scalable rubric-based training and evaluation framework that substantially reduces rubric development costs without sacrificing performance. We show that Health-SCORE provides two practical benefits beyond standalone evaluation: it can be used as a structured reward signal to guide reinforcement learning with safety-aware supervision, and it can be incorporated directly into prompts to improve response quality through in-context learning. Across open-ended healthcare tasks, Health-SCORE achieves evaluation quality comparable to human-created rubrics while significantly lowering development effort, making rubric-based evaluation and training more scalable. |
| 2026-01-26 | [CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search](http://arxiv.org/abs/2601.18625v1) | Zequn Xie | Text-Based Person Search (TBPS) aims to retrieve pedestrian images from large galleries using natural language descriptions. This task, essential for public safety applications, is hindered by cross-modal discrepancies and ambiguous user queries. We introduce CONQUER, a two-stage framework designed to address these challenges by enhancing cross-modal alignment during training and adaptively refining queries at inference. During training, CONQUER employs multi-granularity encoding, complementary pair mining, and context-guided optimal matching based on Optimal Transport to learn robust embeddings. At inference, a plug-and-play query enhancement module refines vague or incomplete queries via anchor selection and attribute-driven enrichment, without requiring retraining of the backbone. Extensive experiments on CUHK-PEDES, ICFG-PEDES, and RSTPReid demonstrate that CONQUER consistently outperforms strong baselines in both Rank-1 accuracy and mAP, yielding notable improvements in cross-domain and incomplete-query scenarios. These results highlight CONQUER as a practical and effective solution for real-world TBPS deployment. Source code is available at https://github.com/zqxie77/CONQUER. |
| 2026-01-26 | [AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security](http://arxiv.org/abs/2601.18491v1) | Dongrui Liu, Qihan Ren et al. | The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released. |
| 2026-01-26 | [SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation](http://arxiv.org/abs/2601.18442v1) | Hongyi Zhao, Shuo Wang et al. | Autonomous vehicle safety validation requires testing on safety-critical scenarios, but these events are rare in real-world driving and costly to test due to collision risks. Crash reports provide authentic specifications of safety-critical events, offering a vital alternative to scarce real-world collision trajectory data. This makes them valuable sources for generating realistic high-risk scenarios through simulation. Existing approaches face significant limitations because data-driven methods lack diversity due to their reliance on existing latent distributions, whereas adversarial methods often produce unrealistic scenarios lacking physical fidelity. Large Language Model (LLM) and Vision Language Model (VLM)-based methods show significant promise. However, they suffer from context suppression issues where internal parametric knowledge overrides crash specifications, producing scenarios that deviate from actual accident characteristics. This paper presents SG-CADVLM (A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation), a framework that integrates Context-Aware Decoding with multi-modal input processing to generate safety-critical scenarios from crash reports and road network diagrams. The framework mitigates VLM hallucination issues while enabling the simultaneous generation of road geometry and vehicle trajectories. The experimental results demonstrate that SG-CADVLM generates critical risk scenarios at a rate of 84.4% compared to 12.5% for the baseline methods, representing an improvement of 469%, while producing executable simulations for autonomous vehicle testing. |
| 2026-01-26 | [Estimating Dense-Packed Zone Height in Liquid-Liquid Separation: A Physics-Informed Neural Network Approach](http://arxiv.org/abs/2601.18399v1) | Mehmet Velioglu, Song Zhai et al. | Separating liquid-liquid dispersions in gravity settlers is critical in chemical, pharmaceutical, and recycling processes. The dense-packed zone height is an important performance and safety indicator but it is often expensive and impractical to measure due to optical limitations. We propose to estimate phase heights using only inexpensive volume flow measurements. To this end, a physics-informed neural network (PINN) is first pretrained on synthetic data and physics equations derived from a low-fidelity (approximate) mechanistic model to reduce the need for extensive experimental data. While the mechanistic model is used to generate synthetic training data, only volume balance equations are used in the PINN, since the integration of submodels describing droplet coalescence and sedimentation into the PINN would be computationally prohibitive. The pretrained PINN is then fine-tuned with scarce experimental data to capture the actual dynamics of the separator. We then employ the differentiable PINN as a predictive model in an Extended Kalman Filter inspired state estimation framework, enabling the phase heights to be tracked and updated from flow-rate measurements. We first test the two-stage trained PINN by forward simulation from a known initial state against the mechanistic model and a non-pretrained PINN. We then evaluate phase height estimation performance with the filter, comparing the two-stage trained PINN with a two-stage trained purely data-driven neural network. All model types are trained and evaluated using ensembles to account for model parameter uncertainty. In all evaluations, the two-stage trained PINN yields the most accurate phase-height estimates. |
| 2026-01-26 | [When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs](http://arxiv.org/abs/2601.18350v1) | Junyi Zou | Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons. |
| 2026-01-26 | [Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare](http://arxiv.org/abs/2601.18334v1) | Clément Christophe, Wadood Mohammed Abdul et al. | As LLMs are increasingly integrated into clinical workflows, their tendency for sycophancy, prioritizing user agreement over factual accuracy, poses significant risks to patient safety. While existing evaluations often rely on subjective datasets, we introduce a robust framework grounded in medical MCQA with verifiable ground truths. We propose the Adjusted Sycophancy Score, a novel metric that isolates alignment bias by accounting for stochastic model instability, or "confusability". Through an extensive scaling analysis of the Qwen-3 and Llama-3 families, we identify a clear scaling trajectory for resilience. Furthermore, we reveal a counter-intuitive vulnerability in reasoning-optimized "Thinking" models: while they demonstrate high vanilla accuracy, their internal reasoning traces frequently rationalize incorrect user suggestions under authoritative pressure. Our results across frontier models suggest that benchmark performance is not a proxy for clinical reliability, and that simplified reasoning structures may offer superior robustness against expert-driven sycophancy. |
| 2026-01-26 | [Possible favored Great Oxidation Event scenario on exoplanets around M-Stars with the example of TRAPPIST-1e](http://arxiv.org/abs/2601.18324v1) | Adam Y. Jaziri, Nathalie Carrasco et al. | The Great Oxidation Event (GOE), which marked the transition from an anoxic to an oxygenated atmosphere, occurred 2.4 billion years ago on Earth, several hundreds of millions of years after the emergence of oxygenic photosynthesis. This long delay implies that specific conditions in terms of biomass productivity and burial were necessary to trigger the GOE. It could be a limiting factor for the development of oxygenated atmospheres on inhabited exoplanets. In this study, we explore the specificities of a terrestrial planet in the habitable zone of an M dwarf for a GOE. Using a 1D coupled photochemical-climate model, we simulate the atmospheric evolution of TRAPPIST-1 e, an Earth-like exoplanet, exploring the effect of oxygen sources (biotic or abiotic). Our results show that the stellar energy distribution promotes O3 production at lower O2 concentrations compared to Earth, and the ozone layer on TRAPPIST-1 e forms more efficiently. This lowers the threshold for atmospheric oxidation, suggesting that the GOE on TRAPPIST-1 e would occur quickly after the rise of oxygenic photosynthesis, up to 1Gyrs earlier than on Earth, and would reach O2 enabling oxygenic respiration and thus the development of animals. We may question whether this is a general behavior around several M-stars. Furthermore, we discuss how the overproduction of ozone could make O3 detection possible using the James Webb Space Telescope, providing a potential method to observe oxygenation signatures on exoplanets in the near future. Previous studies predicted that for an Earth-like atmosphere O3 would require over 150 transits for detection, but our results show that significantly fewer transits could be needed. |
| 2026-01-26 | [TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment](http://arxiv.org/abs/2601.18292v1) | Zhewen Tan, Wenhan Yu et al. | In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop. |
| 2026-01-26 | [Quest2ROS2: A ROS 2 Framework for Bi-manual VR Teleoperation](http://arxiv.org/abs/2601.18289v1) | Jialong Li, Zhenguo Wang et al. | Quest2ROS2 is an open-source ROS2 framework for bi-manual teleoperation designed to scale robot data collection. Extending Quest2ROS, it overcomes workspace limitations via relative motion-based control, calculating robot movement from VR controller pose changes to enable intuitive, pose-independent operation. The framework integrates essential usability and safety features, including real-time RViz visualization, streamlined gripper control, and a pause-and-reset function for smooth transitions. We detail a modular architecture that supports "Side-by-Side" and "Mirror" control modes to optimize operator experience across diverse platforms. Code is available at: https://github.com/Taokt/Quest2ROS2. |
| 2026-01-26 | [Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs](http://arxiv.org/abs/2601.18255v1) | Fei Meng | Continual learning in Large Language Models (LLMs) faces the critical challenge of balancing stability (retaining old knowledge) and plasticity (learning new tasks). While Experience Replay (ER) is a standard countermeasure against catastrophic forgetting, its impact across diverse capabilities remains underexplored. In this work, we uncover a critical dichotomy in ER's behavior: while it induces positive backward transfer on robust, unstructured tasks (e.g., boosting performance on previous NLP classification tasks through repeated rehearsal), it causes severe negative transfer on fragile, structured domains like code generation (e.g., a significant relative drop in coding accuracy). This reveals that ER trades structural integrity for broad consolidation. To address this dilemma, we propose \textbf{Orthogonal Subspace Wake-up (OSW)}. OSW identifies essential parameter subspaces of previous tasks via a brief "wake-up" phase and enforces orthogonal updates for new tasks, providing a mathematically grounded "safety guarantee" for established knowledge structures. Empirical results across a diverse four-task sequence demonstrate that OSW uniquely succeeds in preserving fragile coding abilities where Replay fails, while simultaneously maintaining high plasticity for novel tasks. Our findings emphasize the necessity of evaluating structural safety alongside average retention in LLM continual learning. |
| 2026-01-26 | [Enhance the Safety in Reinforcement Learning by ADRC Lagrangian Methods](http://arxiv.org/abs/2601.18142v1) | Mingxu Zhang, Huicheng Zhang et al. | Safe reinforcement learning (Safe RL) seeks to maximize rewards while satisfying safety constraints, typically addressed through Lagrangian-based methods. However, existing approaches, including PID and classical Lagrangian methods, suffer from oscillations and frequent safety violations due to parameter sensitivity and inherent phase lag. To address these limitations, we propose ADRC-Lagrangian methods that leverage Active Disturbance Rejection Control (ADRC) for enhanced robustness and reduced oscillations. Our unified framework encompasses classical and PID Lagrangian methods as special cases while significantly improving safety performance. Extensive experiments demonstrate that our approach reduces safety violations by up to 74%, constraint violation magnitudes by 89%, and average costs by 67\%, establishing superior effectiveness for Safe RL in complex environments. |
| 2026-01-26 | [RareAlert: Aligning heterogeneous large language model reasoning for early rare disease risk screening](http://arxiv.org/abs/2601.18132v1) | Xi Chen, Hongru Zhou et al. | Missed and delayed diagnosis remains a major challenge in rare disease care. At the initial clinical encounters, physicians assess rare disease risk using only limited information under high uncertainty. When high-risk patients are not recognised at this stage, targeted diagnostic testing is often not initiated, resulting in missed diagnosis. Existing primary care triage processes are structurally insufficient to reliably identify patients with rare diseases at initial clinical presentation and universal screening is needed to reduce diagnostic delay. Here we present RareAlert, an early screening system which predict patient-level rare disease risk from routinely available primary-visit information. RareAlert integrates reasoning generated by ten LLMs, calibrates and weights these signals using machine learning, and distils the aligned reasoning into a single locally deployable model. To develop and evaluate RareAlert, we curated RareBench, a real-world dataset of 158,666 cases covering 33 Orphanet disease categories and more than 7,000 rare conditions, including both rare and non-rare presentations. The results showed that rare disease identification can be reconceptualised as a universal uncertainty resolution process applied to the general patient population. On an independent test set, RareAlert, a Qwen3-4B based model trained with calibrated reasoning signals, achieved an AUC of 0.917, outperforming the best machine learning ensemble and all evaluated LLMs, including GPT-5, DeepSeek-R1, Claude-3.7-Sonnet, o3-mini, Gemini-2.5-Pro, and Qwen3-235B. These findings demonstrate the diversity in LLM medical reasoning and the effectiveness of aligning such reasoning in highly uncertain clinical tasks. By incorporating calibrated reasoning into a single model, RareAlert enables accurate, privacy-preserving, and scalable rare disease risk screening suitable for large-scale local deployment. |
| 2026-01-26 | [Beyond Static Datasets: Robust Offline Policy Optimization via Vetted Synthetic Transitions](http://arxiv.org/abs/2601.18107v1) | Pedram Agand, Mo Chen | Offline Reinforcement Learning (ORL) holds immense promise for safety-critical domains like industrial robotics, where real-time environmental interaction is often prohibitive. A primary obstacle in ORL remains the distributional shift between the static dataset and the learned policy, which typically mandates high degrees of conservatism that can restrain potential policy improvements. We present MoReBRAC, a model-based framework that addresses this limitation through Uncertainty-Aware latent synthesis. Instead of relying solely on the fixed data, MoReBRAC utilizes a dual-recurrent world model to synthesize high-fidelity transitions that augment the training manifold. To ensure the reliability of this synthetic data, we implement a hierarchical uncertainty pipeline integrating Variational Autoencoder (VAE) manifold detection, model sensitivity analysis, and Monte Carlo (MC) dropout. This multi-layered filtering process guarantees that only transitions residing within high-confidence regions of the learned dynamics are utilized. Our results on D4RL Gym-MuJoCo benchmarks reveal significant performance gains, particularly in ``random'' and ``suboptimal'' data regimes. We further provide insights into the role of the VAE as a geometric anchor and discuss the distributional trade-offs encountered when learning from near-optimal datasets. |

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



