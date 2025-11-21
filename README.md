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
| 2025-11-20 | [Formal Abductive Latent Explanations for Prototype-Based Networks](http://arxiv.org/abs/2511.16588v1) | Jules Soria, Zakaria Chihani et al. | Case-based reasoning networks are machine-learning models that make predictions based on similarity between the input and prototypical parts of training samples, called prototypes. Such models are able to explain each decision by pointing to the prototypes that contributed the most to the final outcome. As the explanation is a core part of the prediction, they are often qualified as ``interpretable by design". While promising, we show that such explanations are sometimes misleading, which hampers their usefulness in safety-critical contexts. In particular, several instances may lead to different predictions and yet have the same explanation. Drawing inspiration from the field of formal eXplainable AI (FXAI), we propose Abductive Latent Explanations (ALEs), a formalism to express sufficient conditions on the intermediate (latent) representation of the instance that imply the prediction. Our approach combines the inherent interpretability of case-based reasoning models and the guarantees provided by formal XAI. We propose a solver-free and scalable algorithm for generating ALEs based on three distinct paradigms, compare them, and present the feasibility of our approach on diverse datasets for both standard and fine-grained image classification. The associated code can be found at https://github.com/julsoria/ale |
| 2025-11-20 | [Synthesis of Safety Specifications for Probabilistic Systems](http://arxiv.org/abs/2511.16579v1) | Gaspard Ohlmann, Edwin Hamel-De le Court et al. | Ensuring that agents satisfy safety specifications can be crucial in safety-critical environments. While methods exist for controller synthesis with safe temporal specifications, most existing methods restrict safe temporal specifications to probabilistic-avoidance constraints. Formal methods typically offer more expressive ways to express safety in probabilistic systems, such as Probabilistic Computation Tree Logic (PCTL) formulas. Thus, in this paper, we develop a new approach that supports more general temporal properties expressed in PCTL. Our contribution is twofold. First, we develop a theoretical framework for the Synthesis of safe-PCTL specifications. We show how the reducing global specification satisfaction to local constraints, and define CPCTL, a fragment of safe-PCTL. We demonstrate how the expressiveness of CPCTL makes it a relevant fragment for the Synthesis Problem. Second, we leverage these results and propose a new Value Iteration-based algorithm to solve the synthesis problem for these more general temporal properties, and we prove the soundness and completeness of our method. |
| 2025-11-20 | [WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue](http://arxiv.org/abs/2511.16544v1) | Zachary Ellis, Jared Joselowitz et al. | As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $κ$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue. |
| 2025-11-20 | [Graph Neural Networks for Surgical Scene Segmentation](http://arxiv.org/abs/2511.16430v1) | Yihan Li, Nikhil Churamani et al. | Purpose: Accurate identification of hepatocystic anatomy is critical to preventing surgical complications during laparoscopic cholecystectomy. Deep learning models often struggle with occlusions, long-range dependencies, and capturing the fine-scale geometry of rare structures. This work addresses these challenges by introducing graph-based segmentation approaches that enhance spatial and semantic understanding in surgical scene analyses.   Methods: We propose two segmentation models integrating Vision Transformer (ViT) feature encoders with Graph Neural Networks (GNNs) to explicitly model spatial relationships between anatomical regions. (1) A static k Nearest Neighbours (k-NN) graph with a Graph Convolutional Network with Initial Residual and Identity Mapping (GCNII) enables stable long-range information propagation. (2) A dynamic Differentiable Graph Generator (DGG) with a Graph Attention Network (GAT) supports adaptive topology learning. Both models are evaluated on the Endoscapes-Seg50 and CholecSeg8k benchmarks.   Results: The proposed approaches achieve up to 7-8% improvement in Mean Intersection over Union (mIoU) and 6% improvement in Mean Dice (mDice) scores over state-of-the-art baselines. It produces anatomically coherent predictions, particularly on thin, rare and safety-critical structures.   Conclusion: The proposed graph-based segmentation methods enhance both performance and anatomical consistency in surgical scene segmentation. By combining ViT-based global context with graph-based relational reasoning, the models improve interpretability and reliability, paving the way for safer laparoscopic and robot-assisted surgery through a precise identification of critical anatomical features. |
| 2025-11-20 | [Data Annotation Quality Problems in AI-Enabled Perception System Development](http://arxiv.org/abs/2511.16410v1) | Hina Saeeda, Tommy Johansson et al. | Data annotation is essential but highly error-prone in the development of AI-enabled perception systems (AIePS) for automated driving, and its quality directly influences model performance, safety, and reliability. However, the industry lacks empirical insights into how annotation errors emerge and spread across the multi-organisational automotive supply chain. This study addresses this gap through a multi-organisation case study involving six companies and four research institutes across Europe and the UK. Based on 19 semi-structured interviews with 20 experts (50 hours of transcripts) and a six-phase thematic analysis, we develop a taxonomy of 18 recurring annotation error types across three data-quality dimensions: completeness (e.g., attribute omission, missing feedback loops, edge-case omissions, selection bias), accuracy (e.g., mislabelling, bounding-box inaccuracies, granularity mismatches, bias-driven errors), and consistency (e.g., inter-annotator disagreement, ambiguous instructions, misaligned hand-offs, cross-modality inconsistencies). The taxonomy was validated with industry practitioners, who reported its usefulness for root-cause analysis, supplier quality reviews, onboarding, and improving annotation guidelines. They described it as a failure-mode catalogue similar to FMEA. By conceptualising annotation quality as a lifecycle and supply-chain issue, this study contributes to SE4AI by offering a shared vocabulary, diagnostic toolset, and actionable guidance for building trustworthy AI-enabled perception systems. |
| 2025-11-20 | [The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks](http://arxiv.org/abs/2511.16347v1) | Chunyang Li, Zifeng Kang et al. | The adoption of Vision-Language Models (VLMs) in embodied AI agents, while being effective, brings safety concerns such as jailbreaking. Prior work have explored the possibility of directly jailbreaking the embodied agents through elaborated multi-modal prompts. However, no prior work has studied or even reported indirect jailbreaks in embodied AI, where a black-box attacker induces a jailbreak without issuing direct prompts to the embodied agent. In this paper, we propose, for the first time, indirect environmental jailbreak (IEJ), a novel attack to jailbreak embodied AI via indirect prompt injected into the environment, such as malicious instructions written on a wall. Our key insight is that embodied AI does not ''think twice'' about the instructions provided by the environment -- a blind trust that attackers can exploit to jailbreak the embodied agent. We further design and implement open-source prototypes of two fully-automated frameworks: SHAWSHANK, the first automatic attack generation framework for the proposed attack IEJ; and SHAWSHANK-FORGE, the first automatic benchmark generation framework for IEJ. Then, using SHAWSHANK-FORGE, we automatically construct SHAWSHANK-BENCH, the first benchmark for indirectly jailbreaking embodied agents. Together, our two frameworks and one benchmark answer the questions of what content can be used for malicious IEJ instructions, where they should be placed, and how IEJ can be systematically evaluated. Evaluation results show that SHAWSHANK outperforms eleven existing methods across 3,957 task-scene combinations and compromises all six tested VLMs. Furthermore, current defenses only partially mitigate our attack, and we have responsibly disclosed our findings to all affected VLM vendors. |
| 2025-11-20 | [Beyond Generative AI: World Models for Clinical Prediction, Counterfactuals, and Planning](http://arxiv.org/abs/2511.16333v1) | Mohammad Areeb Qazi, Maryam Nadeem et al. | Healthcare requires AI that is predictive, reliable, and data-efficient. However, recent generative models lack physical foundation and temporal reasoning required for clinical decision support. As scaling language models show diminishing returns for grounded clinical reasoning, world models are gaining traction because they learn multimodal, temporally coherent, and action-conditioned representations that reflect the physical and causal structure of care. This paper reviews World Models for healthcare systems that learn predictive dynamics to enable multistep rollouts, counterfactual evaluation and planning. We survey recent work across three domains: (i) medical imaging and diagnostics (e.g., longitudinal tumor simulation, projection-transition modeling, and Joint Embedding Predictive Architecture i.e., JEPA-style predictive representation learning), (ii) disease progression modeling from electronic health records (generative event forecasting at scale), and (iii) robotic surgery and surgical planning (action-conditioned guidance and control). We also introduce a capability rubric: L1 temporal prediction, L2 action-conditioned prediction, L3 counterfactual rollouts for decision support, and L4 planning/control. Most reviewed systems achieve L1--L2, with fewer instances of L3 and rare L4. We identify cross-cutting gaps that limit clinical reliability; under-specified action spaces and safety constraints, weak interventional validation, incomplete multimodal state construction, and limited trajectory-level uncertainty calibration. This review outlines a research agenda for clinically robust prediction-first world models that integrate generative backbones (transformers, diffusion, VAE) with causal/mechanical foundation for safe decision support in healthcare. |
| 2025-11-20 | [Optimizing Operation Recipes with Reinforcement Learning for Safe and Interpretable Control of Chemical Processes](http://arxiv.org/abs/2511.16297v1) | Dean Brandner, Sergio Lucia | Optimal operation of chemical processes is vital for energy, resource, and cost savings in chemical engineering. The problem of optimal operation can be tackled with reinforcement learning, but traditional reinforcement learning methods face challenges due to hard constraints related to quality and safety that must be strictly satisfied, and the large amount of required training data. Chemical processes often cannot provide sufficient experimental data, and while detailed dynamic models can be an alternative, their complexity makes it computationally intractable to generate the needed data. Optimal control methods, such as model predictive control, also struggle with the complexity of the underlying dynamic models. Consequently, many chemical processes rely on manually defined operation recipes combined with simple linear controllers, leading to suboptimal performance and limited flexibility.   In this work, we propose a novel approach that leverages expert knowledge embedded in operation recipes. By using reinforcement learning to optimize the parameters of these recipes and their underlying linear controllers, we achieve an optimized operation recipe. This method requires significantly less data, handles constraints more effectively, and is more interpretable than traditional reinforcement learning methods due to the structured nature of the recipes. We demonstrate the potential of our approach through simulation results of an industrial batch polymerization reactor, showing that it can approach the performance of optimal controllers while addressing the limitations of existing methods. |
| 2025-11-20 | ["To Survive, I Must Defect": Jailbreaking LLMs via the Game-Theory Scenarios](http://arxiv.org/abs/2511.16278v1) | Zhen Sun, Zongmin Zhang et al. | As LLMs become more common, non-expert users can pose risks, prompting extensive research into jailbreak attacks. However, most existing black-box jailbreak attacks rely on hand-crafted heuristics or narrow search spaces, which limit scalability. Compared with prior attacks, we propose Game-Theory Attack (GTA), an scalable black-box jailbreak framework. Concretely, we formalize the attacker's interaction against safety-aligned LLMs as a finite-horizon, early-stoppable sequential stochastic game, and reparameterize the LLM's randomized outputs via quantal response. Building on this, we introduce a behavioral conjecture "template-over-safety flip": by reshaping the LLM's effective objective through game-theoretic scenarios, the originally safety preference may become maximizing scenario payoffs within the template, which weakens safety constraints in specific contexts. We validate this mechanism with classical game such as the disclosure variant of the Prisoner's Dilemma, and we further introduce an Attacker Agent that adaptively escalates pressure to increase the ASR. Experiments across multiple protocols and datasets show that GTA achieves over 95% ASR on LLMs such as Deepseek-R1, while maintaining efficiency. Ablations over components, decoding, multilingual settings, and the Agent's core model confirm effectiveness and generalization. Moreover, scenario scaling studies further establish scalability. GTA also attains high ASR on other game-theoretic scenarios, and one-shot LLM-generated variants that keep the model mechanism fixed while varying background achieve comparable ASR. Paired with a Harmful-Words Detection Agent that performs word-level insertions, GTA maintains high ASR while lowering detection under prompt-guard models. Beyond benchmarks, GTA jailbreaks real-world LLM applications and reports a longitudinal safety monitoring of popular HuggingFace LLMs. |
| 2025-11-20 | [SeSE: A Structural Information-Guided Uncertainty Quantification Framework for Hallucination Detection in LLMs](http://arxiv.org/abs/2511.16275v1) | Xingtao Zhao, Hao Peng et al. | Reliable uncertainty quantification (UQ) is essential for deploying large language models (LLMs) in safety-critical scenarios, as it enables them to abstain from responding when uncertain, thereby avoiding hallucinating falsehoods. However, state-of-the-art UQ methods primarily rely on semantic probability distributions or pairwise distances, overlooking latent semantic structural information that could enable more precise uncertainty estimates. This paper presents Semantic Structural Entropy (SeSE), a principled UQ framework that quantifies the inherent semantic uncertainty of LLMs from a structural information perspective for hallucination detection. Specifically, to effectively model semantic spaces, we first develop an adaptively sparsified directed semantic graph construction algorithm that captures directional semantic dependencies while automatically pruning unnecessary connections that introduce negative interference. We then exploit latent semantic structural information through hierarchical abstraction: SeSE is defined as the structural entropy of the optimal semantic encoding tree, formalizing intrinsic uncertainty within semantic spaces after optimal compression. A higher SeSE value corresponds to greater uncertainty, indicating that LLMs are highly likely to generate hallucinations. In addition, to enhance fine-grained UQ in long-form generation -- where existing methods often rely on heuristic sample-and-count techniques -- we extend SeSE to quantify the uncertainty of individual claims by modeling their random semantic interactions, providing theoretically explicable hallucination detection. Extensive experiments across 29 model-dataset combinations show that SeSE significantly outperforms advanced UQ baselines, including strong supervised methods and the recently proposed KLE. |
| 2025-11-20 | [A damage detection method of plate structure using fan-shaped sensor clusters](http://arxiv.org/abs/2511.16270v1) | Hand Yue, Ma Chenning et al. | Plate structures are widely used in large-scale engineering fields such as aerospace, hull manufacturing, and construction. However, the plate structure is easily damaged during long-term service or when it is impacted by foreign objects. Such a damage may lead to serious safety accidents. Beamforming and L-shaped sensor cluster (LSSC) localization method can be used to locate the damage on the plate. However, when using beamforming or LSSC localization method to locate the damages on plate-like structures, there exists blind area. In this paper, by combining the beamforming and LSSC localization method, a fan-shaped sensor cluster is proposed through arranging five sensors in a fan shape, which can effectively reduce the blind areas. The positions of damages on the plate can be accurately detected by using two groups of fan-shaped sensor clusters and one sensor for transmitting the excitation signal. The feasibility of the fan-shaped sensor cluster localization method is verified through numerical simulations and experiments, and the results are compared with those obtained by using the T-shaped sensor cluster. The results show that the fan-shaped sensor cluster can more accurately identify the damages at different positions. Both simulation and experimental results indicate that the fan-shaped sensor cluster can improve the accuracy of damage location. |
| 2025-11-20 | [Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security](http://arxiv.org/abs/2511.16229v1) | Wei Zhao, Zhe Li et al. | Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM. |
| 2025-11-20 | [Multi-Agent Collaborative Reward Design for Enhancing Reasoning in Reinforcement Learning](http://arxiv.org/abs/2511.16202v1) | Pei Yang, Ke Zhang et al. | We present CRM (Multi-Agent Collaborative Reward Model), a framework that replaces a single black-box reward model with a coordinated team of specialist evaluators to improve robustness and interpretability in RLHF. Conventional reward models struggle to jointly optimize multiple, sometimes conflicting, preference dimensions (e.g., factuality, helpfulness, safety) and offer limited transparency into why a score is assigned. CRM addresses these issues by decomposing preference evaluation into domain-specific agents that each produce partial signals, alongside global evaluators such as ranker-based and embedding-similarity rewards. A centralized aggregator fuses these signals at each timestep, balancing factors like step-wise correctness, multi-agent agreement, and repetition penalties, yielding a single training reward compatible with standard RL pipelines. The policy is optimized with advantage-based updates (e.g., GAE), while a value model regresses to the aggregated reward, enabling multi-perspective reward shaping without requiring additional human annotations beyond those used to train the evaluators. To support training and assessment, we introduce rewardBench, a benchmark and training suite aligned with the collaborative structure of CRM. Together, CRM and rewardBench provide a practical, modular path to more transparent reward modeling and more stable optimization. |
| 2025-11-20 | [Physics-informed Gaussian Processes as Linear Model Predictive Controller with Constraint Satisfaction](http://arxiv.org/abs/2511.16195v1) | Jörn Tebbe, Andreas Besginow et al. | Model Predictive Control evolved as the state of the art paradigm for safety critical control tasks. Control-as-Inference approaches thereof model the constrained optimization problem as a probabilistic inference problem. The constraints have to be implemented into the inference model. A recently introduced physics-informed Gaussian Process method uses Control-as-Inference with a Gaussian likelihood for state constraint modeling, but lacks guarantees of open-loop constraint satisfaction. We mitigate the lack of guarantees via an additional sampling step using Hamiltonian Monte Carlo sampling in order to obtain safe rollouts of the open-loop dynamics which are then used to obtain an approximation of the truncated normal distribution which has full probability mass in the safe area. We provide formal guarantees of constraint satisfaction while maintaining the ODE structure of the Gaussian Process on a discretized grid. Moreover, we show that we are able to perform optimization of a quadratic cost function by closed form Gaussian Process computations only and introduce the Matérn kernel into the inference model. |
| 2025-11-20 | [Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](http://arxiv.org/abs/2511.16110v1) | Yijun Yang, Lichao Wang et al. | The growing misuse of Vision-Language Models (VLMs) has led providers to deploy multiple safeguards, including alignment tuning, system prompts, and content moderation. However, the real-world robustness of these defenses against adversarial attacks remains underexplored. We introduce Multi-Faceted Attack (MFA), a framework that systematically exposes general safety vulnerabilities in leading defense-equipped VLMs such as GPT-4o, Gemini-Pro, and Llama-4. The core component of MFA is the Attention-Transfer Attack (ATA), which hides harmful instructions inside a meta task with competing objectives. We provide a theoretical perspective based on reward hacking to explain why this attack succeeds. To improve cross-model transferability, we further introduce a lightweight transfer-enhancement algorithm combined with a simple repetition strategy that jointly bypasses both input-level and output-level filters without model-specific fine-tuning. Empirically, we show that adversarial images optimized for one vision encoder transfer broadly to unseen VLMs, indicating that shared visual representations create a cross-model safety vulnerability. Overall, MFA achieves a 58.5% success rate and consistently outperforms existing methods. On state-of-the-art commercial models, MFA reaches a 52.8% success rate, surpassing the second-best attack by 34%. These results challenge the perceived robustness of current defense mechanisms and highlight persistent safety weaknesses in modern VLMs. Code: https://github.com/cure-lab/MultiFacetedAttack |
| 2025-11-20 | [Learning Tractable Distributions Of Language Model Continuations](http://arxiv.org/abs/2511.16054v1) | Gwen Yidou-Weng, Ian Li et al. | Controlled language generation conditions text on sequence-level constraints (for example, syntax, style, or safety). These constraints may depend on future tokens, which makes directly conditioning an autoregressive language model (LM) generally intractable. Prior work uses tractable surrogates such as hidden Markov models (HMMs) to approximate the distribution over continuations and adjust the model's next-token logits at decoding time. However, we find that these surrogates are often weakly context aware, which reduces query quality. We propose Learning to Look Ahead (LTLA), a hybrid approach that pairs the same base language model for rich prefix encoding with a fixed tractable surrogate model that computes exact continuation probabilities. Two efficiency pitfalls arise when adding neural context: (i) naively rescoring the prefix with every candidate next token requires a sweep over the entire vocabulary at each step, and (ii) predicting fresh surrogate parameters for each prefix, although tractable at a single step, forces recomputation of future probabilities for every new prefix and eliminates reuse. LTLA avoids both by using a single batched HMM update to account for all next-token candidates at once, and by conditioning only the surrogate's latent state prior on the LM's hidden representations while keeping the surrogate decoder fixed, so computations can be reused across prefixes. Empirically, LTLA attains higher conditional likelihood than an unconditional HMM, approximates continuation distributions for vision-language models where a standalone HMM cannot encode visual context, and improves constraint satisfaction at comparable fluency on controlled-generation tasks, with minimal inference overhead. |
| 2025-11-20 | [Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion](http://arxiv.org/abs/2511.16020v1) | Dingkun Zhou, Patrick P. K. Chan et al. | Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility. |
| 2025-11-20 | [Detecting Sleeper Agents in Large Language Models via Semantic Drift Analysis](http://arxiv.org/abs/2511.15992v1) | Shahin Zanbaghi, Ryan Rostampour et al. | Large Language Models (LLMs) can be backdoored to exhibit malicious behavior under specific deployment conditions while appearing safe during training a phenomenon known as "sleeper agents." Recent work by Hubinger et al. demonstrated that these backdoors persist through safety training, yet no practical detection methods exist. We present a novel dual-method detection system combining semantic drift analysis with canary baseline comparison to identify backdoored LLMs in real-time. Our approach uses Sentence-BERT embeddings to measure semantic deviation from safe baselines, complemented by injected canary questions that monitor response consistency. Evaluated on the official Cadenza-Labs dolphin-llama3-8B sleeper agent model, our system achieves 92.5% accuracy with 100% precision (zero false positives) and 85% recall. The combined detection method operates in real-time (<1s per query), requires no model modification, and provides the first practical solution to LLM backdoor detection. Our work addresses a critical security gap in AI deployment and demonstrates that embedding-based detection can effectively identify deceptive model behavior without sacrificing deployment efficiency. |
| 2025-11-20 | [KRAL: Knowledge and Reasoning Augmented Learning for LLM-assisted Clinical Antimicrobial Therapy](http://arxiv.org/abs/2511.15974v1) | Zhe Li, Yehan Qiu et al. | Clinical antimicrobial therapy requires the dynamic integration of pathogen profiles, host factors, pharmacological properties of antimicrobials, and the severity of infection.This complexity imposes fundamental limitations on the applicability of Large Language Models (LLMs) in high-stakes clinical decision-making including knowledge gaps, data privacy concerns, high deployment costs, and limited reasoning capabilities. To address these challenges, we propose KRAL (Knowledge and Reasoning Augmented Learning), a low-cost, scalable, privacy-preserving paradigm that leverages teacher-model reasoning to automatically distill knowledge and reasoning trajectories via answer-to-question reverse generation, employs heuristic learning for semi-supervised data augmentation (reducing manual annotation requirements by approximately 80%), and utilizes agentic reinforcement learning to jointly enhance medical knowledge and reasoning while optimizing computational and memory efficiency. A hierarchical evaluation employing diverse teacher-model proxies reduces assessment costs, while modular interface design facilitates seamless system updates. Experimental results demonstrate that KRAL significantly outperforms traditional Retrieval-Augmented Generation (RAG) and Supervised Fine-Tuning (SFT) methods. It improves knowledge question-answering capability (Accuracy@1 on the external open-source benchmark MEDQA increased by 1.8% vs. SFT and 3.6% vs. RAG) and reasoning capability (Pass@1 on the external benchmark PUMCH Antimicrobial increased by 27% vs. SFT and 27.2% vs. RAG), achieved at ~20% of SFT's long-term training costs. This establishes KRAL as an effective solution for enhancing local LLMs' clinical diagnostic capabilities, enabling low-cost, high-safety deployment in complex medical decision support. |
| 2025-11-19 | [Cyber-Resilient Data-Driven Event-Triggered Secure Control for Autonomous Vehicles Under False Data Injection Attacks](http://arxiv.org/abs/2511.15925v1) | Yashar Mousavi, Mahsa Tavasoli et al. | This paper proposes a cyber-resilient secure control framework for autonomous vehicles (AVs) subject to false data injection (FDI) threats as actuator attacks. The framework integrates data-driven modeling, event-triggered communication, and fractional-order sliding mode control (FSMC) to enhance the resilience against adversarial interventions. A dynamic model decomposition (DMD)-based methodology is employed to extract the lateral dynamics from real-world data, eliminating the reliance on conventional mechanistic modeling. To optimize communication efficiency, an event-triggered transmission scheme is designed to reduce the redundant transmissions while ensuring system stability. Furthermore, an extended state observer (ESO) is developed for real-time estimation and mitigation of actuator attack effects. Theoretical stability analysis, conducted using Lyapunov methods and linear matrix inequality (LMI) formulations, guarantees exponential error convergence. Extensive simulations validate the proposed event-triggered secure control framework, demonstrating substantial improvements in attack mitigation, communication efficiency, and lateral tracking performance. The results show that the framework effectively counteracts actuator attacks while optimizing communication-resource utilization, making it highly suitable for safety-critical AV applications. |

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



