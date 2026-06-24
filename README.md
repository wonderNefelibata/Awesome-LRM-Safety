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
| 2026-06-23 | [HelpBench: Assessing the Ability of LLMs to Provide Privacy, Safety, and Security Advice](http://arxiv.org/abs/2606.24819v1) | Sarah Meiklejohn, Sunny Consolvo et al. | This paper introduces HelpBench, a benchmark for assessing whether LLMs are capable of providing accurate help in response to questions about digital privacy, safety, and security. We curated 450 questions representing authentic user situations and developed rubrics for each question to evaluate the factual accuracy and tone of a response. Example questions touch on how to regain access to lost or suspended accounts, how to balance the trade-offs of hardware security keys versus other forms of two-factor authentication, whether a suspicious email is likely a scam, or whether an abuser might be able to track an individual based on their device peripherals. We then developed and applied an auto-rater to evaluate responses from 18 state-of-the-art LLMs. Our results indicate that while models provide high-quality advice (with scores of 82% on average), one in ten responses from models scores less than 65%, reflecting inaccurate and even harmful advice. Addressing these failures is critical for models to serve as trustworthy sources of assistance for digital privacy, safety, and security needs. |
| 2026-06-23 | [DDStereo: Efficient Dual Decoder Transformers for Stereo 3D Road Anomaly Detection](http://arxiv.org/abs/2606.24805v1) | Shiyi Mu, Zichong Gu et al. | Stereo-based 3D object detection still faces two critical safety challenges: real-time performance and open-set generalization. Existing stereo 3D methods typically achieve twice the accuracy of monocular methods but suffer from significantly lower inference speeds, making them unsuitable for real-time applications. Meanwhile, recent advances in open-world detection have introduced open-set and open-vocabulary algorithms in monocular 2D and 3D settings, yet stereo-based open-set detection remains largely unexplored. To bridge this gap, we propose DDStereo, a novel Dual-Decoder Stereo Transformer for real-time open-set 3D object detection. DDStereo features two lightweight decoder branches: one for open-set foreground 2D detection and the other for 3D attribute regression. These decoders share object-level queries to achieve unified target-level alignment. To enhance inference efficiency, we designed a compact disparity feature extractor and a streamlined decoder architecture. Experiments on public stereo 3D benchmarks demonstrate that DDStereo achieves state-of-the-art accuracy under both closed-set and open-set protocols. Notably, our method surpasses existing stereo 3D detectors in inference speed and, for the first time, achieves real-time performance comparable to monocular approaches. |
| 2026-06-23 | [Grad Detect: Gradient-Based Hallucination Detection in LLMs](http://arxiv.org/abs/2606.24790v1) | Anand Kamat, Daniel Blake et al. | Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet they remain prone to generating hallucinations. Detecting these hallucinations is critical for deploying LLMs reliably in high-stakes applications. We present Grad Detect, a gradient-based approach for predicting hallucinations by analyzing layer-wise gradient patterns from a single forward-backward pass during inference. Our method shows that the internal gradient structure of a model carries rich information about the correctness of its output. This information is not accessible through output-level signals alone. We evaluate Grad Detect on several Q&A benchmarks across both hallucination detection and model abstention prediction, where it consistently outperforms confidence-based and sampling-based baselines. Through comprehensive layer ablation studies across all eleven models from four architectural families, we find that the final five layers concentrate over 97% of the discriminative gradient signal, enabling efficient deployment with minimal performance loss. Grad Detect provides a unified framework for predicting multiple dimensions of LLM reliability, offering strong predictive performance alongside interpretable insights into where and how model failures originate. |
| 2026-06-23 | [UniDrive: A Unified Vision-Language and Grounding Framework for Interpretable Risk Understanding in Autonomous Driving](http://arxiv.org/abs/2606.24759v1) | Xiaowei Gao, Pengxiang Li et al. | Recent multimodal large language models (MLLMs) have shown strong potential for autonomous driving scene understanding, yet existing methods still face a fundamental trade-off between temporal reasoning and spatial precision. Models that rely on single-frame or low-resolution inputs often miss small, distant, or partially occluded hazards, while language-centric driving models frequently provide limited grounded evidence for their explanations. To address this gap, we propose UniDrive, a unified visual-language and grounding framework for interpretable risk understanding in autonomous driving. UniDrive combines a temporal reasoning branch that models scene dynamics from multi-frame visual input with a high-resolution perception branch that preserves fine-grained spatial details from the latest frame. The two branches are integrated through a gated cross-attention fusion module, enabling dynamic context to be aligned with precise spatial evidence. Based on the fused representation, UniDrive jointly generates natural-language risk descriptions and grounded bounding-box outputs for risk objects. Experiments on the DRAMA-Reasoning benchmark show that UniDrive outperforms representative image-based and video-based baselines in both captioning and risk-object grounding. In particular, UniDrive achieves the best overall performance on the validation split and demonstrates clear advantages in small-object localization, zero-shot generalization to NuScenes and BDD100K, and human-rated interpretability and trustworthiness. These results suggest that explicitly combining temporal semantics and high-resolution perception provides a stronger foundation for interpretable and safety-oriented autonomous driving systems. The code is available at https://github.com/pixeli99/unidrive-dev. |
| 2026-06-23 | [SER: Learning to Ground Video Reasoning with Semantic Evidence Rewards](http://arxiv.org/abs/2606.24726v1) | Sheng Xia, Zhengqin Lai et al. | Video MLLMs often struggle with fine-grained spatio-temporal reasoning, sometimes generating correct answers based on irrelevant frames or objects. Although outputting spatio-temporal evidence during reasoning is a promising direction, existing RL frameworks typically rely on geometry-only (IoU) rewards, which can be sensitive to boundary perturbations and overlook semantic alignment. To address this, we propose Semantic Evidence Reward (SER), which reformulates spatio-temporal evidence grounding as a constrained verification task. Instead of computing pixel-level overlap, SER uses a referee VLM as a local checker to evaluate model-generated evidence claims across two dimensions: relevance and localization quality, combined with a temporal penalty. This design reduces the reliance on dense box annotations and enables training directly on standard video QA data. On the V-STAR benchmark, SER achieves 49.6% mLGM, improving by 3.0 points over the strong evidence-grounded baseline Open-o3-Video, demonstrating its potential in enhancing both answer accuracy and evidence grounding. |
| 2026-06-23 | [SAFARI: Scaling Long Horizon Agentic Fault Attribution via Active Investigation](http://arxiv.org/abs/2606.24626v1) | Chenyang Zhu, Jiayu Yao et al. | As autonomous agents tackle increasingly complex multi-step, multi-agent tasks, their execution trajectories have scaled beyond the constraints of even the largest context windows. Current methods for effectively diagnosing agent failures load the full trajectory into an LLM's context window, which suffers from attention dilution and fails when agentic traces inevitably exceed context limits. To address this, we introduce SAFARI (Scaling long-horizon Agentic Fault AttRibution via active Investigation), a framework that replaces linear context loading with a tool-augmented diagnostic loop. By equipping LLMs with a specialized toolbox to read and search trajectory segments alongside a persistent Short-Term Memory (STM) for cross-turn reasoning, SAFARI effectively decouples diagnostic accuracy from architectural context limits. Our experiments demonstrate that SAFARI outperforms state-of-the-art results by 20% on the Who&When dataset within a 1M token budget, and by 19% on TRAIL GAIA subset on a 25K token budget. Most significantly, SAFARI maintains a 0.58 precision even when the target fault resides 5x beyond the model's native context window, a scenario where traditional evaluators fail entirely. |
| 2026-06-23 | [Qwen-AgentWorld: Language World Models for General Agents](http://arxiv.org/abs/2606.24597v1) | Yuxin Zuo, Zikai Xiao et al. | A world model predicts environment dynamics based on current observations and actions, serving as a core cognitive mechanism for reasoning and planning. In this work, we investigate how world modeling based on language models can further push the boundaries of general agents. (i) We first focus on building foundation models for agentic environment simulation. We introduce Qwen-AgentWorld-35B-A3B and Qwen-AgentWorld-397B-A17B, the first language world models capable of simulating agentic environments covering 7 domains via long chain-of-thought reasoning. Leveraging more than 10M environment interaction trajectories of 7 domains in real-world environments, we develop Qwen-AgentWorld through a three-stage training pipeline: CPT injects general-purpose world modeling capabilities from the state transition dynamics and augmented professional corpora, SFT activates next-state-prediction reasoning, and RL sharpens simulation fidelity through a tailored framework with hybrid rubric-and-rule rewards. To evaluate language world models, we present AgentWorldBench, a comprehensive benchmark constructed from real-world interactions of 5 frontier models on 9 established benchmarks. Empirical results demonstrate that Qwen-AgentWorld significantly outperforms existing frontier models. (ii) Beyond foundation models, we further investigate two complementary paradigms through which world modeling enhances general agents. First, as a decoupled environment simulator, Qwen-AgentWorld supports scalable and controllable simulation of thousands of real-world environments for agentic RL, yielding gains that surpass real-environment training alone. Second, as a unified agent foundation model, world-model training acts as a highly effective warm-up that improves downstream performance across 7 agentic benchmarks. Code: https://github.com/QwenLM/Qwen-AgentWorld |
| 2026-06-23 | [LLMs Prompted for Legal Context Object More: Overrefusal from Small On-Premises LLMs in Criminal Legal Context](http://arxiv.org/abs/2606.24585v1) | Anastasiia Kucherenko, François Brouchoud et al. | While the validity of LLMs' use in the legal context remains subject to ethical and legal debate, legal professionals are already experimenting with personal LLMs, if only for translation and reformulation. However, even such a seemingly innocuous use can introduce biases through case processing speed if LLM assistants selectively refuse assistance on certain topics. To better anticipate such biases, we investigate several modern small LLMs that are most likely to be used as on-device assistants, to assess the impact of overrefusal on legal prompts.   Surprisingly, we find that authority-style prefixes (``you are acting as an assistant of the national supreme court'', ``[...] defense lawyer'') systematically increase refusal rates by 2--20x over the no-prefix baseline, while a known role-play jailbreak prefix shows mixed effects, sharply increasing refusals in some models and barely shifting them in others. The finding suggests that small on-prem deployable LLMs are unstable under contextual framings that a real institutional user might naturally introduce, and further investigation is essential to minimize opportunities for bias. |
| 2026-06-23 | [Quant Convergence: Bridging Classical Value Investing and Modern Factor Models for Systematic Equity Selection](http://arxiv.org/abs/2606.24575v1) | Augusto Eiji Yamazaki, Hugo Garrido-Lestache Belinchon | Modern finance relies heavily on complex machine learning models to find patterns in the stock market. However, as these AI models get more complicated, they often memorize short-term market noise instead of finding companies with real, lasting value. We designed this research to test if Benjamin Graham's classic value investing rules could act as a mathematical "low-pass filter" to keep these modern models in check. We built three different sets of features - pure Graham rules, modern market factors, and a mix of both - and tested them against highly complex models (XGBoost and AutoGluon) using 20 years of S&P 500 data. By applying a strict buy-and-hold strategy over a four-year test period (March 2022 to March 2026), the results showed that more complex algorithms do not always win. While the AutoGluon model captured high returns (222.68%), it suffered a substantial 39.78% drop because it bought volatile tech stocks right before the market crashed. On the other hand, the pure Graham Random Forest achieved the highest overall return (232.13%) with much less risk (1.38 Calmar Ratio). Furthermore, the Combined Random Forest successfully mixed momentum with Graham's rules, making a 202.91% return while keeping the lowest maximum drop (34.53%) of any model tested. Ultimately, this research proves that Graham's "margin of safety" isn't outdated; it is actually a highly effective way to prevent modern AI from taking on too much risk. |
| 2026-06-23 | [Hydrofluoric acid-free titanium etching for rare-event searches](http://arxiv.org/abs/2606.24524v1) | P. Knights, I. Manthos et al. | Rare-event search experiments require construction materials with high radiopurity to minimise background contributions. Thanks to its high mechanical strength, low density, machinability, and commercial availability in relatively radiopure forms, titanium is a suitable material for structural elements in rare-event searches. In such applications, a chemical etching stage is typically performed to remove surface contamination or to prepare the surface for further treatment. However, due to its chemical resistance, the etching of titanium conventionally requires hydrofluoric acid, posing serious health and safety concerns that are further exacerbated in deep underground laboratory settings. An alternative approach is proposed, which uses sulphuric acid. Grade 1 titanium samples were etched in 20\% and 40\% sulphuric acid solutions at 20$^\circ$C and 40$^\circ$C for up to 24\,h. The effects of etching were quantified through mass change measurements, surface roughness analysis, and scanning electron microscopy. Sulphuric acid effectively etches titanium, with up to $3.5\,\pm\,0.3$ mg/cm$^2$ of titanium removed for an unagitated solution of 40\% sulphuric acid at $40^\circ$C for 24\,h. Furthermore, sulphuric acid is shown to be effective at etching at lower concentration and temperature. The formation of a passivation layer during the etching may enable control of the total mass removed. |
| 2026-06-23 | [Poster: Exploring the Limits of Audio-Based Detection of Turkish Phone Call Scams](http://arxiv.org/abs/2606.24523v1) | Arda Eren, Micheal Cheung et al. | Scam phone calls exploit vulnerable communities worldwide, yet research on detection has focused almost exclusively on English and other high-resource languages. In low-resource settings such as Turkish, detection is especially difficult, as annotated data is scarce and technological defenses remain limited. This research investigates how large language models (LLMs) can support scam detection in Turkish by introducing the first public multi-modal dataset of 100 aligned audio-transcript pairs of scam and benign conversations. We evaluate seven LLMs spanning three model families: Gemini 2.5 (Flash, Flash-Lite, Pro), GPT-4o, and Qwen (Max, Plus, Turbo), under three input conditions: raw audio, automatic speech-to-text transcripts, and transcripts refined by a native speaker. Our results suggest that transcript-based inputs consistently outperform direct audio processing, while human-corrected and uncorrected transcripts perform comparably. By centering a low-resource language and real world threat, this work highlights the urgent need for culturally and linguistically inclusive AI safety research and more robust multi-modal systems for fraud prevention. |
| 2026-06-23 | [A specialized reasoning large language model for accelerating rare disease diagnosis: a randomized AI physician assistance trial](http://arxiv.org/abs/2606.24510v1) | Haichao Chen, Songchi Zhou et al. | Rare diseases affect millions of individuals worldwide, yet timely diagnosis remains a major public health challenge due to scarcity of specialized clinical expertise. While large language models (LLMs) show promise to support rare disease diagnosis, current models are constrained by insufficient clinical deployability, limited clinically grounded evidence, and scarcity of training data. Here we present RaDaR (Rare Disease navigatoR), an open-source, compact reasoning LLM (32B parameters) for rare disease diagnosis. RaDaR was trained with 49,170 publicly available free-text cases and 104,666 synthetic cases with reasoning-enhanced training. RaDaR showed the strongest performance among evaluated open-source models, including the 671B DeepSeek-R1, across public benchmarks and four external validation centers. In a retrospective cohort, RaDaR prioritized the final diagnosis before documented clinical suspicion in 61.06 percent of cases, corresponding to a potential lead time of 1.87 months and 50.18 percent of the within-center interval. In a randomized physician-assistance trial, RaDaR assistance improved physicians' rare-disease diagnostic accuracy by 21.44 percentage points compared with internet search alone. Synthetic-data ablations suggested that phenotype-anchored narratives provide useful training signal for long-tail rare diseases, with a monotonic scaling trend within the tested data range. Together, RaDaR and its development and validation framework provide a deployable rare-disease reasoning model and a reproducible development framework for diagnostic AI under data scarcity. |
| 2026-06-23 | [FT-WBC: Learning Fault-Tolerant Whole-Body Control for Legged Loco-Manipulation](http://arxiv.org/abs/2606.24466v1) | Yudong Zhong, Pengfei Mai et al. | Legged manipulators combine the mobility of legged platforms with the manipulation capability of robotic arms. However, arm-induced Center-of-Mass shifts and dynamic disturbances make the system more prone to instability under actuator failures, potentially leading to falls, task failures, or safety risks. Existing fault-tolerant control methods mainly focus on locomotion alone, leaving the coupled problem of whole-body stability and arm reachability in fault-tolerant loco-manipulation largely unaddressed. To bridge this gap, we propose FT-WBC, a fault-tolerant loco-manipulation framework for robust whole-body control of legged manipulators under actuator failures. FT-WBC adopts a decoupled upper- and lower-body policy architecture and introduces two key modules: a Fault Estimator (FE) and a Posture Adaptation Module (PAM). The FE predicts faulty joints from lower-body proprioceptive histories, while the PAM uses this fault information to adapt the base posture plan generated by the arm policy, converting potentially unstable posture requests into safe and executable base posture commands. Through this fault-aware posture adaptation mechanism, FT-WBC synthesizes compensatory gaits under actuator failures and preserves as much arm workspace as possible while maintaining whole-body stability. Simulation and real-world experiments show that FT-WBC significantly improves survival rate and workspace under weakening or locked failures, and transfers zero-shot to a real legged manipulator in the real world. |
| 2026-06-23 | [Explainable AI for Next-Generation Wireless Physical Layer: Basics, State-of-the-Art, and Open Challenges](http://arxiv.org/abs/2606.24424v1) | Bingnan Xiao, Shuyan Hu et al. | Next-generation wireless systems are expected to be ``AI-native," with neural networks (NNs) embedded throughout the physical (PHY) layer protocol stack to improve spectral efficiency, latency, and network autonomy. However, the opacity of deep learning (DL) models raises increasing concerns about system reliability, safety, and privacy, especially under complex and time-varying network environments. This survey studies explainable AI (XAI) in wireless PHY layers from the explainability perspective. We first formalize a series of responsibility-oriented goals for wireless XAI. Then, we develop a systematic taxonomy of explainability approaches and distill practical criteria for deploying explanations in communication scenarios. We provide a comprehensive review of where and how XAI can be applied throughout the PHY layer, connecting representative learning paradigms to appropriate explanation techniques, evaluation metrics, and deployment considerations. Open challenges and future directions are discussed, including explainability-performance tradeoffs, explainability-aware data processing, customized XAI for communication-specific structures, cross-layer explanation consistency, and emerging needs for explaining LLM- and Agentic-AI-driven PHY layers. |
| 2026-06-23 | [PHANTOM: A Large-Scale Dataset of Multimodal Adversarial Attacks for Vision-Language Models](http://arxiv.org/abs/2606.24388v1) | Simone Gallivanone, Hossein Khodadadi et al. | We introduce a large-scale, open-source dataset of pre-generated adversarial attacks for vision-language models (VLMs). The dataset is designed to be diverse, representative, and practical, extending existing benchmarks by covering 10 high-level categories and 55 subcategories of harmful intents. Our primary goal is to make adversarial data accessible to the research community, given the computational cost and complexity of generating large numbers of attacks. The dataset comprises 47 524 adversarial samples, generated using state-of-the-art attack strategies from recent literature. Our work complements existing efforts by consolidating and extending prior benchmarks from multiple established sources, resulting in 7 826 intents, and introduce an additional category to broaden coverage. This provides realistic evaluation resources for studying model robustness and alignment. Our dataset intends to enable researchers and practitioners to systematically evaluate the robustness and safety of VLMs, fine-tune attack-generation models, and develop or stress-test defensive guardrails under diverse adversarial conditions. By releasing this resource, we aim to lower the barrier to adversarial research and foster more reproducible, comprehensive, and comparable evaluations of VLM safety. |
| 2026-06-23 | [PDS Joint: A Parametric Double-Spiral Joint Tailored for Dexterous Hands](http://arxiv.org/abs/2606.24377v1) | Haoyang Li, Yibo Wen et al. | Compliant joints can embed safety and adaptability into dexterous hands, but achieving large-stroke anthropomorphic motion while maintaining joint-specific, directiondependent stiffness and reliable proprioception remains challenging. This paper presents the PDS joint, a parametric doublespiral (PDS) compliant joint that enables systematic shaping of directional stiffness across multiple deformation modes, including flexion/extension, abduction/adduction, and pronation/supination. We instantiate the joint using Archimedean and logarithmic spiral templates for different hand joints and introduce an asymmetry ratio to tailor stiffness distributions for both grasp stability and hyperextension resistance. To make the joint practically usable under large deformation, we co-design embedded inductive proprioception and propose a learningbased calibration pipeline that maps raw inductive signals to joint states using ArUco-marker tracking. Experiments characterize the stiffness landscapes across geometric parameters and demonstrate a non-monotonic dependence of lateral support on asymmetry, indicating the importance of principled parameter tuning. For joint-state estimation in the most challenging abduction/adduction motion, a learned multilayer-perceptron (MLP) mapping reduces the error compared with conventional curve fitting by 41.6%. Finally, we integrate the proposed joints into an open-source dexterous hand as a demonstration platform, on which the hand grasps a set of nine everyday objects and performs safe, contact-rich human-involved interactions. |
| 2026-06-23 | [Pigeonholing: Bad prompts hurt models to collapse and make mistakes](http://arxiv.org/abs/2606.24267v1) | Hyunji Nam, Keertana Chidambaram et al. | While in-context learning is generally shown to be effective in Large Language Models (LLMs), bad contexts can cause performance degradation and mode collapse, a phenomenon we call "pigeonholing." **Unintentionally bad** contexts can happen without malicious jailbreaking intents: For example, a user asks the model to justify an incorrect math theorem or fails to correct the model's buggy code. Specifically, we investigate ``pigeonholing" in two scenarios: (1) when the user suggests a solution, and (2) when the conversation context includes the assistant's previous (incorrect) responses. Our experiments across 10 verifiable and open-ended tasks with 10 different models show that pigeonholing manifests in several ways: (1) repeating the incorrect answers from context (leading to 38-40% performance drop), (2) converging on a narrow set of answers in coding and text generation without exploring alternatives, and (3) flipping stance on controversial topics to align with the user or the assistant's previous claims. We find that pigeonholing worsens almost monotonically with the number of conversation turns (performance drops by additional 14+% as repeated mistakes increase from 1 to 5), and pigeonholing-induced mode collapse can happen even when the provided example is correct. As a step toward mitigation, we propose RLVR with synthetic errors which improves models by 43-60% under bad contexts compared to vanilla RLVR baselines. |
| 2026-06-23 | [AutoSpec: Safety Rule Evolution for LLM Agents via Inductive Logic Programming](http://arxiv.org/abs/2606.24245v1) | Pingchuan Ma, Zhaoyu Wang et al. | Large language model (LLM) agents increasingly automate complex tasks by integrating language models with external tools and environments. However, their autonomy poses significant safety risks: agents may execute destructive commands, leak sensitive data, or violate domain constraints. Existing safety approaches face a fundamental tradeoff: hand-crafted rules are interpretable but brittle, with overly conservative rules blocking safe operations (high false positives) while permissive rules miss unsafe behaviors (high false negatives). Neural classifiers lack the interpretability required for safety-critical deployments.   We present AutoSpec, a framework that automatically evolves deployed expert-designed safety rules from user safe/unsafe annotations through counterexample-guided inductive synthesis (CEGIS) guided by inductive logic programming (ILP). Starting from the expert rules and a stream of annotated traces, AutoSpec iteratively evaluates rules, mines false-positive and false-negative counterexamples, uses ILP to learn which predicates discriminate them, generates candidate rule edits, and verifies candidates to select the best revision. The key insight is that ILP efficiently identifies predicates that appear frequently in false negatives but rarely in false positives (or vice versa), dramatically pruning the exponential search space of rule edits. This continues until convergence, producing interpretable rules that balance precision and recall.   We evaluate AutoSpec on 291 execution traces spanning code execution and embodied agent domains. AutoSpec raises rule F1 to 0.98 and 0.93 across the two domains, achieving up to 94% false positive reduction while maintaining high recall, and converges within 4-5 iterations. The ILP-guided approach achieves up to 4.8x higher F1 than heuristic CEGIS. The learned rules are human-readable, auditable, and generalize to unseen scenarios. |
| 2026-06-23 | [FlowR2A: Learning Reward-to-Action Distribution for Multimodal Driving Planning](http://arxiv.org/abs/2606.24231v1) | Xirui Li, Zhe Liu et al. | Multimodal driving planning faces a long-standing tension between two paradigms: scoring-based methods benefit from dense reward supervision but are confined to a fixed action vocabulary, while anchor-based methods generate proposals dynamically yet suffer from sparse supervision constrained to a single ground-truth trajectory. In this work, we propose FlowR2A, which resolves this tension by reframing simulation-based rewards from discriminative targets into generative conditions. By learning the reward-conditioned action distribution from dense trajectory-reward pairs with a flow-matching decoder, FlowR2A unifies the dense supervision of scoring-based methods with the proposal generation of anchor-based methods in a single generative model, forcing the model to internalize the correlation between an action and its outcomes in safety, progress, comfort, and rule compliance. To balance hard safety constraints against soft progress objectives, we introduce fine-grained per-timestep reward conditioning and reward noise augmentation. The generative formulation naturally supports controllable test-time sampling via reward guidance and anchored sampling, producing high-quality proposals. FlowR2A achieves state-of-the-art results on the NAVSIM v1 and v2 benchmarks, with multimodal proposals of substantially higher quality than prior methods. |
| 2026-06-23 | [Kops: Safely Extending the eBPF Compilation Pipeline with Native Operations](http://arxiv.org/abs/2606.24213v1) | Yusheng Zheng, Zhengjie Ji et al. | eBPF safely extends OS kernels in domains such as networking, observability, and security. The safety comes from an in-kernel compilation pipeline where a verifier checks every program, and a kernel just-in-time compiler (JIT) translates the verified bytecode to native code. The kernel keeps the JIT simple to stay trustworthy, translating one bytecode instruction at a time in a single pass. This single-pass design misses optimization opportunities, so eBPF runs up to twice as slow as natively compiled code in our characterization. Adding optimizations to the kernel JIT directly requires upstream acceptance and a long release cycle, enlarges the trusted computing base (TCB), and grows the per-architecture kernel code. To address this, we present Kops, an extension interface that lets userspace compilers and kernel modules introduce new operations without modifying the kernel core, while keeping a minimal trusted computing base (TCB). Each operation has two forms, a proof sequence of vanilla eBPF instructions that the existing verifier checks and a native emit of machine instructions that the JIT compiles. Because the verifier checks the proof sequence, the native emit is the only per-operation addition to the TCB. Hardware idioms are the lowest-hanging fruit for this interface. With Kops, we build EInsn, seven operations such as rotate and conditional select that CPUs execute as single instructions. Lean 4 proofs show that each native emit computes the same result as its proof sequence. On x86-64 and ARM64, EInsn speeds up eBPF microbenchmarks by up to 24% and production applications by up to 12%. The same interface also supports whole-program native replacement, reaching 2.358x at the cost of a larger TCB. |

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



