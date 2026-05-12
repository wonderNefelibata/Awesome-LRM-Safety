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
| 2026-05-11 | [Variational Inference for Lévy Process-Driven SDEs via Neural Tilting](http://arxiv.org/abs/2605.10934v1) | Yaman Kindap, Manfred Opper et al. | Modelling extreme events and heavy-tailed phenomena is central to building reliable predictive systems in domains such as finance, climate science, and safety-critical AI. While Lévy processes provide a natural mathematical framework for capturing jumps and heavy tails, Bayesian inference for Lévy-driven stochastic differential equations (SDEs) remains intractable with existing methods: Monte Carlo approaches are rigorous but lack scalability, whereas neural variational inference methods are efficient but rely on Gaussian assumptions that fail to capture discontinuities. We address this tension by introducing a neural exponential tilting framework for variational inference in Lévy-driven SDEs. Our approach constructs a flexible variational family by exponentially reweighting the Lévy measure using neural networks. This parametrization preserves the jump structure of the underlying process while remaining computationally tractable. To enable efficient inference, we develop a quadratic neural parametrization that yields closed-form normalization of the tilted measure, a conditional Gaussian representation for stable processes that facilitates simulation, and symmetry-aware Monte Carlo estimators for scalable optimization. Empirically, we demonstrate that the method accurately captures jump dynamics and yields reliable posterior inference in regimes where Gaussian-based variational approaches fail, on both synthetic and real-world datasets. |
| 2026-05-11 | [Single-Photon Double Ionization of Ozone](http://arxiv.org/abs/2605.10918v1) | Veronica Daver Ideböhn, Antoine Gloriod et al. | Ozone (O3) is a triatomic molecule of central importance in the chemistry and physics of the Earth's and other planetary atmospheres. Beyond its environmental significance, a detailed understanding of the electronic structure and ionization dynamics of ozone is essential for modeling atmospheric, ionospheric, and astrochemical processes. In the present work, we substantially extend the experimental and theoretical characterization of ozone into the regime of valence double photoionization. Using HeII-alpha, HeII-beta, and higher-energy vacuum ultraviolet radiation in combination with a versatile multiple charged-particle correlation detection technique, we report the first single-photon valence double ionization electron spectrum of O3. To interpret the experimental observations, we mapped the lowest potential energy surfaces of dicationic ozone employing post-Hartree-Fock multi-configurational-interaction methods, and computed with high accuracy the energetics of the relevant dissociation channels. Our results demonstrate that dissociative double ionization of ozone produces electronically excited cationic atomic oxygen fragments in addition to the ground-state dissociation pathway, revealing a richer fragmentation dynamics than hitherto recognized. |
| 2026-05-11 | [Beyond Red-Teaming: Formal Guarantees of LLM Guardrail Classifiers](http://arxiv.org/abs/2605.10901v1) | Nikita Kezins, Urbas Ekka et al. | Guardrail Classifiers defend production language models against harmful behavior, but although results seem promising in testing, they provide no formal guarantees. Providing formal guarantees for such models is hard because "harmful behavior" has no natural specification in a discrete input space: and the standard epsilon-ball properties used in other domains do not carry semantic meaning. We close this gap by shifting verification from the discrete input space to the classifier's pre-activation space, where we define a harmful region as a convex shape enclosing the representations of known harmful prompts. Because the sigmoid classification head is monotonic, certifying the worst-case point is sufficient to certify the entire region, yielding a closed-form soundness proof without approximation in O(d) time. To formally evaluate these classifiers, we propose two constructions of such regions: SVD-aligned hyper-rectangles, which yield exact SAT/UNSAT certificates, and Gaussian Mixture Models, which yield probabilistic certificates over semantically coherent clusters. Applying this framework to three author-trained Guardrail Classifiers on the toxicity domain, every hyper-rectangle configuration returns SAT, exposing verifiable safety holes across all classifiers, despite seemingly high empirical metrics. Probabilistic GMM certificates also expose a divergent structural stability in how these models represent harm. While GPT-2 and Llama-3.1-8B maintain robust coverage of 90% and 80% across varying boundaries, BERT's safety guarantees prove uniquely volatile. This 'coverage collapse' to 55% at the optimal threshold reveals a sparsely populated safety margin in BERT, which only achieves full coverage by adopting an extremely conservative pessimistic threshold. These approaches combined, provide new insights on how effective Guardrail Classifiers really are, beyond traditional red-teaming. |
| 2026-05-11 | [Shields to Guarantee Probabilistic Safety in MDPs](http://arxiv.org/abs/2605.10888v1) | Linus Heck, Filip Macák et al. | Shielding is a prominent model-based technique to ensure safety of autonomous agents. Classical shielding aims to ensure that nothing bad ever happens and comes with strong guarantees about safety and maximal permissiveness. However, shielding systems for probabilistic safety, where something bad is allowed to happen with an acceptable probability, has proven to be more intricate. This paper presents a formal framework that conservatively extends classical shields to probabilistic safety. In this framework, we (i) demonstrate the impossibility of preserving the strong guarantees on safety and permissiveness, (ii) provide natural shields with weaker guarantees, and (iii) introduce offline and online shield constructions ensuring strong safety guarantees. The empirical evaluation highlights the practical advantages of the new shields, as well as their computational feasibility. |
| 2026-05-11 | [RUBEN: Rule-Based Explanations for Retrieval-Augmented LLM Systems](http://arxiv.org/abs/2605.10862v1) | Joel Rorseth, Parke Godfrey et al. | This paper demonstrates RUBEN, an interactive tool for discovering minimal rules to explain the outputs of retrieval-augmented large language models (LLMs) in data-driven applications. We leverage novel pruning strategies to efficiently identify a minimal set of rules that subsume all others. We further demonstrate novel applications of these rules for LLM safety, specifically to test the resiliency of safety training and effectiveness of adversarial prompt injections. |
| 2026-05-11 | [Verification Mirage: Mapping the Reliability Boundary of Self-Verification in Medical VQA](http://arxiv.org/abs/2605.10850v1) | Ruinan Jin, Beidi Zhao et al. | Self-verification, re-invoking the same vision language model (VLM) in a fresh context to check its own generated answer, is increasingly used as a default safety layer for medical visual question answering (VQA). We argue that this practice is fundamentally unreliable. We introduce [METHOD NAME], a diagnostic framework for mapping the reliability boundary of medical VLM self-verification by decomposing verifier behavior into discrimination capability and agreement bias. Because the verifier and answer generator are capacity-coupled, the verifier can overly agree with the generator, creating a verification mirage: a regime with both high verifier error and high agreement bias, driven by false acceptance of incorrect answers. Evaluating six open-weight VLMs across five medical VQA datasets and seven medical tasks, we find that this boundary is strongly task-conditioned. Knowledge-intensive clinical tasks fall deepest into the mirage, simpler tasks are more resistant, and perceptual tasks lie in between. Verification also fails to provide an independent safety signal: logistic mixed-effects analysis shows that verifier error and agreement bias become more likely when the generator is wrong, while saliency analyses show that verifiers under-attend to image evidence relative to generators, a phenomenon we call the lazy verifier. Cross-verification reduces but does not eliminate the mirage. Moreover, when verification is reused in multi-turn actor-verifier loops, most initially wrong answers become locked in by false verification. Since our experiments use clean benchmarks, the observed reliability boundary likely underestimates failures in real clinical deployment. |
| 2026-05-11 | [The Last Word Often Wins: A Format Confound in Chain-of-Thought Corruption Studies](http://arxiv.org/abs/2605.10799v1) | Gabriel Garcia | Corruption studies, the primary tool for evaluating chain-of-thought (CoT) faithfulness, identify which chain positions are "computationally important" by measuring accuracy when steps are replaced with errors. We identify a systematic confound: for chains with explicit terminal answer statements, the dominant format in standard benchmarks, corruption studies detect where the answer text appears, not where computation occurs.   A within-dataset format ablation provides the key evidence: on standard GSM8K chains ending with "the answer is X," removing only the answer statement, preserving all reasoning, collapses suffix sensitivity ~19x at 3B (N=300, p=0.022). Conflicting-answer experiments quantify the causal mechanism: at 7B, CC accuracy drops to near-zero (<=0.02) across five architecture families; the followed-wrong rate spans 0.63-1.00 at 3B-7B and attenuates at larger scales (0.300 at Phi-4-14B, ~0.01 at 32B). A within-stable 7B replication (9.3x attenuation, N=76, p=7.8e-3; Qwen3-8B N=299, p=0.004) provides converging evidence, and the pattern replicates on MATH (DeepSeek-R1-7B: 10.9x suffix-survival recovery). On chains without answer suffixes the same protocol identifies the prefix as load-bearing (Delta=-0.77, p<10^-12).   Generation-time probes confirm a dissociation: the answer is not early-determined during generation (early commitment <5%), yet at consumption time model outputs systematically follow the explicit answer text. The format-determination effect persists through 14B (8.5x ratio, p=0.001) and converges toward zero at 32B. We propose a three-prerequisite protocol (question-only control, format characterization, all-position sweep) as a minimum standard for corruption-based faithfulness studies. |
| 2026-05-11 | [LITMUS: Benchmarking Behavioral Jailbreaks of LLM Agents in Real OS Environments](http://arxiv.org/abs/2605.10779v1) | Chiyu Zhang, Huiqin Yang et al. | The rapid proliferation of LLM-based autonomous agents in real operating system environments introduces a new category of safety risk beyond content safety: behavior jailbreak, where an adversary induces an agent to execute dangerous OS-level operations with irreversible consequences. Existing benchmarks either evaluate safety at the semantic layer alone, missing physical-layer harms, or fail to isolate test cases, letting earlier runs contaminate later ones. We present LITMUS (LLM-agents In-OS Testing for Measuring Unsafe Subversion), a benchmark addressing both gaps via a semantic-physical dual verification mechanism and OS-level state rollback. LITMUS comprises 819 high-risk test cases organized into one harmful seed subset and six attack-extended subsets covering three adversarial paradigms (jailbreak speaking, skill injection, and entity wrapping), plus a fully automated multi-agent evaluation framework judging behavior at both conversational and OS-level physical layers. Evaluation across frontier agents reveals three findings: (1) current agents lack effective safety awareness, with strong models (e.g., Claude Sonnet 4.6) still executing 40.64% of high-risk operations; (2) agents exhibit pervasive Execution Hallucination (EH), verbally refusing a request while the dangerous operation has already completed at the system level, invisible to every prior semantic-only framework; and (3) skill injection and entity wrapping attacks achieve high success rates, exposing pronounced agent vulnerabilities. LITMUS provides the first standardized platform for reproducible, physically grounded behavioral safety evaluation of LLM agents in real OS environments. |
| 2026-05-11 | [DynaMiCS: Fine-tuning LLMs with Performance Constraints using Dynamic Mixtures](http://arxiv.org/abs/2605.10770v1) | Eleonora Gualdoni, Sonia Laguna et al. | Multi-domain fine-tuning of large language models requires improving performance on target domains while preserving performance on constrained domains, such as general knowledge, instruction following, or safety evaluations. Existing data mixing strategies rely on fixed heuristics or adaptive rules that cannot explicitly enforce preservation of such capabilities. We propose DynaMiCS, a dynamic mixture optimizer that casts multi-domain fine-tuning as a constrained optimization problem. At each update, DynaMiCS performs short domain-specific probing runs to estimate a slope matrix of local cross-domain effects, capturing how training on each fine-tuning dataset affects each evaluation domain. These estimates are then used to compute mixture weights through optimization over the probability simplex, with the objective of improving target-domain performance while keeping constrained-domain losses below reference levels. Across multi-domain fine-tuning scenarios with varying numbers of target and constrained domains, DynaMiCS achieves stronger target-domain improvements and higher constraint satisfaction than fixed-mixture baselines, at lower computational cost and without reference models, per-example scoring, or manually tuned mixture weights. |
| 2026-05-11 | [Break the Brake, Not the Wheel: Untargeted Jailbreak via Entropy Maximization](http://arxiv.org/abs/2605.10764v1) | Mengqi He, Xinyu Tian et al. | Recent studies show that gradient-based universal image jailbreaks on vision-language models (VLMs) exhibit little or no cross-model transferability, casting doubt on the feasibility of transferable multimodal jailbreaks. We revisit this conclusion under a strictly untargeted threat model without enforcing a fixed prefix or response pattern. Our preliminary experiment reveals that refusal behavior concentrates at high-entropy tokens during autoregressive decoding, and non-refusal tokens already carry substantial probability mass among the top-ranked candidates before attack. Motivated by this finding, we propose Untargeted Jailbreak via Entropy Maximization(UJEM)-KL, a lightweight attack that maximizes entropy at these decision tokens to flip refusal outcomes, while stabilizing the remaining low-entropy positions to preserve output quality. Across three VLMs and two safety benchmarks, UJEM-KL achieves competitive white-box attack success rates and consistently improves transferability, while remaining effective under representative defenses. Our experimental results indicate that the limited transferability primarily stems from overly constrained optimization objectives. |
| 2026-05-11 | [RadThinking: A Dataset for Longitudinal Clinical Reasoning in Radiology](http://arxiv.org/abs/2605.10761v1) | Wenxuan Li, Pedro R. A. S. Bassi et al. | Cancer screening is a reasoning task. A radiologist observes findings, compares them to prior scans, integrates clinical context, and reaches a diagnostic conclusion confirmed by pathology. We present RadThinking, a Visual Question Answering (VQA) dataset that makes this reasoning explicit and trainable. RadThinking releases VQA pairs at three difficulty tiers. Foundation VQAs are atomic perception questions. Single-step reasoning VQAs apply one clinical rule. Compositional VQAs require multi-step chain-of-thought to reach a guideline category such as LI-RADS-5. For every compositional VQA, we release the chain of foundation VQAs that solves it. The chain follows the rules of the governing clinical reporting standard. The dataset spans 20,362 CT scans from 9,131 patients across 43 cancer groups, plus 2,077 verified healthy controls with >1-year follow-up. To our knowledge, RadThinking is the first cancer-screening VQA corpus that stratifies questions by reasoning depth and grounds compositions in clinical reporting standards. The foundation tier supplies atomic perception supervision. The compositional tier supplies chain-of-thought data and verifiable rewards for reinforcement-learning recipes such as DeepSeek-R1 and OpenAI o1. RadThinking enables systematic training and evaluation of whether AI systems can reason about cancer, not merely detect it. |
| 2026-05-11 | [C-CoT: Counterfactual Chain-of-Thought with Vision-Language Models for Safe Autonomous Driving](http://arxiv.org/abs/2605.10744v1) | Kefei Tian, Yuansheng Lian et al. | Safety-critical planning in complex environments, particularly at urban intersections, remains a fundamental challenge for autonomous driving. Existing methods, whether rule-based or data-driven, frequently struggle to capture complex scene semantics, infer potential risks, and make reliable decisions in rare, high-risk situations. While vision-language models (VLMs) offer promising approaches for safe decision-making in these environments, most current approaches lack reflective and causal reasoning, thereby limiting their overall robustness. To address this, we propose a counterfactual chain-of-thought (C-CoT) framework that leverages VLMs to decompose driving decisions into five sequential stages: scene description, critical object identification, risk prediction, counterfactual risk reasoning, and final action planning. Within the counterfactual reasoning stage, we introduce a structured meta-action evaluation tree to explicitly assess the potential consequences of alternative action combinations. This self-reflective reasoning establishes causal links between action choices and safety outcomes, improving robustness in long-tail and out-of-distribution scenarios. To validate our approach, we construct the DeepAccident-CCoT dataset based on the DeepAccident benchmark and fine-tune a Qwen2.5-VL (7B) model using low-rank adaptation. Our model achieves a risk prediction recall of 81.9%, reduces the collision rate to 3.52%, and lowers L2 error to 1.98 m. Ablation studies further confirm the critical role of counterfactual reasoning and the meta-action evaluation tree in enhancing safety and interpretability. |
| 2026-05-11 | [Conformity Generates Collective Misalignment in AI Agents Societies](http://arxiv.org/abs/2605.10721v1) | Giordano De Marzo, Alessandro Bellina et al. | Artificial intelligence safety research focuses on aligning individual language models with human values, yet deployed AI systems increasingly operate as interacting populations where social influence may override individual alignment. Here we show that populations of individually aligned AI agents can be driven into stable misaligned states through conformity dynamics. Simulating opinion dynamics across nine large language models and one hundred opinion pairs, we find that each agent's behavior is governed by two competing forces: a tendency to follow the majority and an intrinsic bias toward specific positions. Using tools from statistical physics, we derive a quantitative theory that predicts when populations become trapped in long-lived misaligned configurations, and identifies predictable tipping points where small numbers of adversarial agents can irreversibly shift population-level alignment even after manipulation ceases. These results demonstrate that individual-level alignment provides no guarantee of collective safety, calling for evaluation frameworks that account for emergent behavior in AI populations. |
| 2026-05-11 | [UAV-Assisted Scan-to-Simulation for Landslides Using Physics-Informed Gaussian Splatting](http://arxiv.org/abs/2605.10715v1) | Zhenyu Liang, Jack C. P. Cheng | Landslide monitoring and simulation play an important role in urban safety assessment and disaster prevention. Existing landslide simulation pipelines typically rely on digital elevation model and mesh-based representations, which are suitable for geometric analysis, but often lack visual realism. This limitation reduces their effectiveness in interactive applications, hazard communication, and public education. In this paper, we propose a UAV-based scan-to-simulation framework that bridges photorealistic scene capture and physics-based landslide simulation through 3DGS. Specifically, our pipeline includes four stages: (1) UAV-based acquisition of slope imagery, (2) reconstruction of a low-anisotropy 3DGS scene representation, (3) volumetric conversion of the target simulation region by filling the interior of the surface-based model, and (4) integration with the Material Point Method (MPM) for landslide simulation. We validate the proposed framework on a real landslide site in Hong Kong that experienced a severe landslide event. The results show that our method supports both realistic visual reconstruction and effective simulation. |
| 2026-05-11 | [AutoSOUP: Safety-Oriented Unit Proof Generation for Component-level Memory-Safety Verification](http://arxiv.org/abs/2605.10712v1) | Paschal C. Amusuo, Ricardo Calvo et al. | Memory-safety errors remain a persistent source of zero-day vulnerabilities in low-level software. The problem is especially acute in embedded systems, where hardware protections are often limited and dynamic analysis is difficult to apply effectively. Memory-safety verification can provide stronger assurance by proving the absence of such errors or exposing violations when they exist. However, current verification workflows remain largely manual and require substantial specialized expertise, limiting their adoption in practice.   We present AutoSOUP, a system for automating component-level memory-safety verification through Safety-Oriented Unit Proofs. We formalize these unit proofs as artifacts that encode verification choices (scope, loop bounds, and environment models) for verifying safety properties, and introduce three techniques for deriving them automatically. To overcome the limitations of existing automation approaches, we further introduce LLM-As-Function-Call, a hybrid architecture that combines deterministic program synthesis with LLMs to automate these techniques and produce justifiable unit proofs. We evaluate AutoSOUP by assessing its ability to automate memory-safety verification and expose vulnerabilities in verified components, and we characterize the assumptions and guarantees of the resulting proofs. |
| 2026-05-11 | [Embodied AI in Action: Insights from SAE World Congress 2026 on Safety, Trust, Robotics, and Real-World Deployment](http://arxiv.org/abs/2605.10653v1) | Jan-Mou Li, Paul Schmitt et al. | Embodied artificial intelligence is rapidly moving from research into real-world systems such as autonomous vehicles, mobile robots, and industrial machines. As these systems become more capable of perceiving, deciding, and acting in dynamic environments, they also introduce new challenges in safety, trust, governance, and operational reliability. This white paper summarizes key insights from the SAE World Congress 2026 panel session \textit{Embodied AI in Action}, which brought together experts from automotive, robotics, artificial intelligence, and safety engineering. The discussion highlighted the need to treat embodied AI as a systems challenge requiring engineering rigor, lifecycle governance, human-centered design, and evolving standards. The paper provides practical perspectives for executives, policymakers, and technical leaders seeking to adopt embodied AI responsibly. The panel reached broad agreement that long-term success will depend not only on advances in AI capability, but equally on safe and trustworthy deployment. |
| 2026-05-11 | [Navigating the Sea of LLM Evaluation: Investigating Bias in Toxicity Benchmarks](http://arxiv.org/abs/2605.10639v1) | Regina Gugg, Selina Niederländer et al. | The rapid adoption of LLMs in both research and industry highlights the challenges of deploying them safely and reveals a gap in the systematic evaluation of toxicity benchmarks. As organizations increasingly rely on these benchmarks to certify models for customer-facing applications and automated moderation, unrecognized evaluation biases could lead to the deployment of vulnerable or unsafe systems. This work investigates the robustness of established benchmarking setups and examines how to measure currently neglected intrinsic biases, such as those related to model choice, metrics, and task types. Our experiments uncover significant discrepancies in benchmark behaviors when evaluation setups are altered. Specifically, shifting the task from text completion to summarization increases the tendency of benchmarks to flag content as harmful. Additionally, certain benchmarks fail to maintain consistent behavior when the input data domain is changed. Furthermore, we observe model-specific instabilities, demonstrating a clear need for more robust and comprehensive safety evaluation frameworks. |
| 2026-05-11 | [Hierarchical Causal Abduction: A Foundation Framework for Explainable Model Predictive Control](http://arxiv.org/abs/2605.10624v1) | Ramesh Arvind Naagarajan, Zühal Wagner et al. | Model Predictive Control (MPC) is widely used to operate safety-critical infrastructure by predicting future trajectories and optimizing control actions. However, nonlinear dynamics, hard safety constraints, and numerical optimization often render individual control moves opaque to human operators, undermining trust and hindering deployment. This paper presents Hierarchical Causal Abduction (HCA), which combines (i) physics-informed reasoning via domain knowledge graphs, (ii) optimization evidence from Karush--Kuhn--Tucker (KKT) multipliers, and (iii) temporal causal discovery via the PCMCI algorithm to generate faithful, human-interpretable explanations for control actions computed by nonlinear MPC. Across three diverse control applications (greenhouse climate, building HVAC, chemical process engineering) with expert validation, HCA improves explanation accuracy by 53\% over LIME (0.478 vs. 0.311) using a single set of cross-domain parameters without per-domain tuning; domain-specific KKT-threshold calibration over 2--3 days further increases accuracy to 0.88. Ablation studies confirm that each evidence source is essential, with 32--37\% accuracy degradation when any component is removed, and HCA's ranking-and-validation methodology generalizes beyond MPC to other prediction-based decision systems, including learning-based control and trajectory planning. |
| 2026-05-11 | [Hierarchical End-to-End Taylor Bounds for Complete Neural Network Verification](http://arxiv.org/abs/2605.10621v1) | Taha Entesari, Mahyar Fazlyab | Reachability analysis of neural networks, which seeks to compute or bound the set of outputs attainable over a given input domain, is central to certifying safety and robustness in learning-enabled physical systems. Since exact reachable set computation is generally intractable, existing methods typically rely on tractable overapproximations. Examining the state of the art for smooth, twice-differentiable networks, we observe that existing approaches exploit at most second-order information and do not systematically leverage higher-order information. In this work, we introduce \textsc{HiTaB}, a novel verification framework that exploits second-order smoothness through both the Hessian, $\nabla^2 f$, and its Lipschitz constant, $L_{\nabla^2 f}$. We further develop a unified hierarchy of zeroth-, first-, and second-order bounds, together with precise conditions under which higher-order approximations yield provable improvements. Our main technical contribution is a compositional procedure for efficiently bounding $L_{\nabla^2 f}$ in deep neural networks via layerwise propagation of curvature bounds. We extend the framework to both $\ell_2$- and $\ell_\infty$-constrained input sets and show how it can be integrated into branch-and-bound verification pipelines. To our knowledge, this is the first practical reachability analysis framework for smooth neural networks that systematically exploits Lipschitz continuity of curvature, leading to tighter and more informative safety certificates. |
| 2026-05-11 | [Re-Triggering Safeguards within LLMs for Jailbreak Detection](http://arxiv.org/abs/2605.10611v1) | Zheng Lin, Zhenxing Niu et al. | This paper proposes a jailbreaking prompt detection method for large language models (LLMs) to defend against jailbreak attacks. Although recent LLMs are equipped with built-in safeguards, it remains possible to craft jailbreaking prompts that bypass them. We argue that such jailbreaking prompts are inherently fragile, and thus introduce an embedding disruption method to re-activate the safeguards within LLMs. Unlike previous defense methods that aim to serve as standalone solutions, our approach instead cooperates with the LLM's internal defense mechanisms by re-triggering them. Moreover, through extensive analysis, we gain a comprehensive understanding of the disruption effects and develop an efficient search algorithm to identify appropriate disruptions for effective jailbreak detection. Extensive experiments demonstrate that our approach effectively defends against state-of-the-art jailbreak attacks in white-box and black-box settings, and remains robust even against adaptive attacks. |

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



