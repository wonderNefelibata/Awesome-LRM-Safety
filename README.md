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
| 2026-01-29 | [StepShield: When, Not Whether to Intervene on Rogue Agents](http://arxiv.org/abs/2601.22136v1) | Gloria Felicia, Michael Eniolade et al. | Existing agent safety benchmarks report binary accuracy, conflating early intervention with post-mortem analysis. A detector that flags a violation at step 8 enables intervention; one that reports it at step 48 provides only forensic value. This distinction is critical, yet current benchmarks cannot measure it. We introduce StepShield, the first benchmark to evaluate when violations are detected, not just whether. StepShield contains 9,213 code agent trajectories, including 1,278 meticulously annotated training pairs and a 7,935-trajectory test set with a realistic 8.1% rogue rate. Rogue behaviors are grounded in real-world security incidents across six categories. We propose three novel temporal metrics: Early Intervention Rate (EIR), Intervention Gap, and Tokens Saved. Surprisingly, our evaluation reveals that an LLM-based judge achieves 59% EIR while a static analyzer achieves only 26%, a 2.3x performance gap that is entirely invisible to standard accuracy metrics. We further show that early detection has direct economic benefits: our cascaded HybridGuard detector reduces monitoring costs by 75% and projects to $108M in cumulative savings over five years at enterprise scale. By shifting the focus of evaluation from whether to when, StepShield provides a new foundation for building safer and more economically viable AI agents. The code and data are released under an Apache 2.0 license. |
| 2026-01-29 | [A Federated and Parameter-Efficient Framework for Large Language Model Training in Medicine](http://arxiv.org/abs/2601.22124v1) | Anran Li, Yuanyuan Chen et al. | Large language models (LLMs) have demonstrated strong performance on medical benchmarks, including question answering and diagnosis. To enable their use in clinical settings, LLMs are typically further adapted through continued pretraining or post-training using clinical data. However, most medical LLMs are trained on data from a single institution, which faces limitations in generalizability and safety in heterogeneous systems. Federated learning (FL) is a promising solution for enabling collaborative model development across healthcare institutions. Yet applying FL to LLMs in medicine remains fundamentally limited. First, conventional FL requires transmitting the full model during each communication round, which becomes impractical for multi-billion-parameter LLMs given the limited computational resources. Second, many FL algorithms implicitly assume data homogeneity, whereas real-world clinical data are highly heterogeneous across patients, diseases, and institutional practices. We introduce the model-agnostic and parameter-efficient federated learning framework for adapting LLMs to medical applications. Fed-MedLoRA transmits only low-rank adapter parameters, reducing communication and computation overhead, while Fed-MedLoRA+ further incorporates adaptive, data-aware aggregation to improve convergence under cross-site heterogeneity. We apply the framework to clinical information extraction (IE), which transforms patient narratives into structured medical entities and relations. Accuracy was assessed across five patient cohorts through comparisons with BERT models, and LLaMA-3 and DeepSeek-R1, GPT-4o models. Evaluation settings included (1) in-domain training and testing, (2) external validation on independent cohorts, and (3) a low-resource new-site adaptation scenario using real-world clinical notes from the Yale New Haven Health System. |
| 2026-01-29 | [Defining Operational Conditions for Safety-Critical AI-Based Systems from Data](http://arxiv.org/abs/2601.22118v1) | Johann Christensen, Elena Hoemann et al. | Artificial Intelligence (AI) has been on the rise in many domains, including numerous safety-critical applications. However, for complex systems found in the real world, or when data already exist, defining the underlying environmental conditions is extremely challenging. This often results in an incomplete description of the environment in which the AI-based system must operate. Nevertheless, this description, called the Operational Design Domain (ODD), is required in many domains for the certification of AI-based systems. Traditionally, the ODD is created in the early stages of the development process, drawing on sophisticated expert knowledge and related standards. This paper presents a novel Safety-by-Design method to a posteriori define the ODD from previously collected data using a multi-dimensional kernel-based representation. This approach is validated through both Monte Carlo methods and a real-world aviation use case for a future safety-critical collision-avoidance system. Moreover, by defining under what conditions two ODDs are equal, the paper shows that the data-driven ODD can equal the original, underlying hidden ODD of the data. Utilizing the novel, Safe-by-Design kernel-based ODD enables future certification of data-driven, safety-critical AI-based systems. |
| 2026-01-29 | [MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts](http://arxiv.org/abs/2601.21971v1) | Lorenzo Mazza, Ariel Rodriguez et al. | Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery. |
| 2026-01-29 | [TraceRouter: Robust Safety for Large Foundation Models via Path-Level Intervention](http://arxiv.org/abs/2601.21900v1) | Chuancheng Shi, Shangze Li et al. | Despite their capabilities, large foundation models (LFMs) remain susceptible to adversarial manipulation. Current defenses predominantly rely on the "locality hypothesis", suppressing isolated neurons or features. However, harmful semantics act as distributed, cross-layer circuits, rendering such localized interventions brittle and detrimental to utility. To bridge this gap, we propose \textbf{TraceRouter}, a path-level framework that traces and disconnects the causal propagation circuits of illicit semantics. TraceRouter operates in three stages: (1) it pinpoints a sensitive onset layer by analyzing attention divergence; (2) it leverages sparse autoencoders (SAEs) and differential activation analysis to disentangle and isolate malicious features; and (3) it maps these features to downstream causal pathways via feature influence scores (FIS) derived from zero-out interventions. By selectively suppressing these causal chains, TraceRouter physically severs the flow of harmful information while leaving orthogonal computation routes intact. Extensive experiments demonstrate that TraceRouter significantly outperforms state-of-the-art baselines, achieving a superior trade-off between adversarial robustness and general utility. Our code will be publicly released. WARNING: This paper contains unsafe model responses. |
| 2026-01-29 | [Making Models Unmergeable via Scaling-Sensitive Loss Landscape](http://arxiv.org/abs/2601.21898v1) | Minwoo Jang, Hoyoung Kim et al. | The rise of model hubs has made it easier to access reusable model components, making model merging a practical tool for combining capabilities. Yet, this modularity also creates a \emph{governance gap}: downstream users can recompose released weights into unauthorized mixtures that bypass safety alignment or licensing terms. Because existing defenses are largely post-hoc and architecture-specific, they provide inconsistent protection across diverse architectures and release formats in practice. To close this gap, we propose \textsc{Trap}$^{2}$, an architecture-agnostic protection framework that encodes protection into the update during fine-tuning, regardless of whether they are released as adapters or full models. Instead of relying on architecture-dependent approaches, \textsc{Trap}$^{2}$ uses weight re-scaling as a simple proxy for the merging process. It keeps released weights effective in standalone use, but degrades them under re-scaling that often arises in merging, undermining unauthorized merging. |
| 2026-01-29 | [Constrained Meta Reinforcement Learning with Provable Test-Time Safety](http://arxiv.org/abs/2601.21845v1) | Tingting Ni, Maryam Kamgarpour | Meta reinforcement learning (RL) allows agents to leverage experience across a distribution of tasks on which the agent can train at will, enabling faster learning of optimal policies on new test tasks. Despite its success in improving sample complexity on test tasks, many real-world applications, such as robotics and healthcare, impose safety constraints during testing. Constrained meta RL provides a promising framework for integrating safety into meta RL. An open question in constrained meta RL is how to ensure the safety of the policy on the real-world test task, while reducing the sample complexity and thus, enabling faster learning of optimal policies. To address this gap, we propose an algorithm that refines policies learned during training, with provable safety and sample complexity guarantees for learning a near optimal policy on the test tasks. We further derive a matching lower bound, showing that this sample complexity is tight. |
| 2026-01-29 | [Test-Time Compute Games](http://arxiv.org/abs/2601.21839v1) | Ander Artola Velasco, Dimitrios Rontogiannis et al. | Test-time compute has emerged as a promising strategy to enhance the reasoning abilities of large language models (LLMs). However, this strategy has in turn increased how much users pay cloud-based providers offering LLM-as-a-service, since providers charge users for the amount of test-time compute they use to generate an output. In our work, we show that the market of LLM-as-a-service is socially inefficient: providers have a financial incentive to increase the amount of test-time compute, even if this increase contributes little to the quality of the outputs. To address this inefficiency, we introduce a reverse second-price auction mechanism where providers bid their offered price and (expected) quality for the opportunity to serve a user, and users pay proportionally to the marginal value generated by the winning provider relative to the second-highest bidder. To illustrate and complement our theoretical results, we conduct experiments with multiple instruct models from the $\texttt{Llama}$ and $\texttt{Qwen}$ families, as well as reasoning models distilled from $\texttt{DeepSeek-R1}$, on math and science benchmark datasets. |
| 2026-01-29 | [Trustworthy Intelligent Education: A Systematic Perspective on Progress, Challenges, and Future Directions](http://arxiv.org/abs/2601.21837v1) | Xiaoshan Yu, Shangshang Yang et al. | In recent years, trustworthiness has garnered increasing attention and exploration in the field of intelligent education, due to the inherent sensitivity of educational scenarios, such as involving minors and vulnerable groups, highly personalized learning data, and high-stakes educational outcomes. However, existing research either focuses on task-specific trustworthy methods without a holistic view of trustworthy intelligent education, or provides survey-level discussions that remain high-level and fragmented, lacking a clear and systematic categorization. To address these limitations, in this paper, we present a systematic and structured review of trustworthy intelligent education. Specifically, We first organize intelligent education into five representative task categories: learner ability assessment, learning resource recommendation, learning analytics, educational content understanding, and instructional assistance. Building on this task landscape, we review existing studies from five trustworthiness perspectives, including safety and privacy, robustness, fairness, explainability, and sustainability, and summarize and categorize the research methodologies and solution strategies therein. Finally, we summarize key challenges and discuss future research directions. This survey aims to provide a coherent reference framework and facilitate a clearer understanding of trustworthiness in intelligent education. |
| 2026-01-29 | [A Unified XAI-LLM Approach for EndotrachealSuctioning Activity Recognition](http://arxiv.org/abs/2601.21802v1) | Hoang Khang Phan, Quang Vinh Dang et al. | Endotracheal suctioning (ES) is an invasive yet essential clinical procedure that requires a high degree of skill to minimize patient risk - particularly in home care and educational settings, where consistent supervision may be limited. Despite its critical importance, automated recognition and feedback systems for ES training remain underexplored. To address this gap, this study proposes a unified, LLM-centered framework for video-based activity recognition benchmarked against conventional machine learning and deep learning approaches, and a pilot study on feedback generation. Within this framework, the Large Language Model (LLM) serves as the central reasoning module, performing both spatiotemporal activity recognition and explainable decision analysis from video data. Furthermore, the LLM is capable of verbalizing feedback in natural language, thereby translating complex technical insights into accessible, human-understandable guidance for trainees. Experimental results demonstrate that the proposed LLM-based approach outperforms baseline models, achieving an improvement of approximately 15-20\% in both accuracy and F1 score. Beyond recognition, the framework incorporates a pilot student-support module built upon anomaly detection and explainable AI (XAI) principles, which provides automated, interpretable feedback highlighting correct actions and suggesting targeted improvements. Collectively, these contributions establish a scalable, interpretable, and data-driven foundation for advancing nursing education, enhancing training efficiency, and ultimately improving patient safety. |
| 2026-01-29 | [BAP-SRL: Bayesian Adaptive Priority Safe Reinforcement Learning for Vehicle Motion Planning at Mixed Traffic Intersections](http://arxiv.org/abs/2601.21679v1) | Yuansheng Lian, Ke Zhang et al. | Navigating urban intersections, especially when interacting with heterogeneous traffic participants, presents a formidable challenge for autonomous vehicles (AVs). In such environments, safety risks arise simultaneously from multiple sources, each carrying distinct priority levels and sensitivities that necessitate differential protection preferences. While safe reinforcement learning (RL) offers a robust paradigm for constrained decision-making, existing methods typically model safety as a single constraint or employ static, heuristic weighting schemes for multiple constraints. These approaches often fail to address the dynamic nature of multi-source risks, leading to gradient cancellation that hampers learning, and suboptimal trade-offs in critical dilemma zones. To address this, we propose a Bayesian adaptive priority safe reinforcement learning (BAP-SRL) based motion planning framework. Unlike heuristic weighting schemes, BAP formulates constraint prioritization as a probabilistic inference task. By modeling historical optimization difficulty as a Bayesian prior and instantaneous risk evidence as a likelihood, BAP dynamically gates gradient updates using a Bayesian inference mechanism on latent constraint criticality. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in handling interactions with stochastic, heterogeneous agents, achieving lower collision rates and smoother conflict resolution. |
| 2026-01-29 | [SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding](http://arxiv.org/abs/2601.21666v1) | Ahmed Y. Radwan, Christos Emmanouilidis et al. | Multimodal Large Language Models (MLLMs) are a major focus of recent AI research. However, most prior work focuses on static image understanding, while their ability to process sequential audio-video data remains underexplored. This gap highlights the need for a high-quality benchmark to systematically evaluate MLLM performance in a real-world setting. We introduce SONIC-O1, a comprehensive, fully human-verified benchmark spanning 13 real-world conversational domains with 4,958 annotations and demographic metadata. SONIC-O1 evaluates MLLMs on key tasks, including open-ended summarization, multiple-choice question (MCQ) answering, and temporal localization with supporting rationales (reasoning). Experiments on closed- and open-source models reveal limitations. While the performance gap in MCQ accuracy between two model families is relatively small, we observe a substantial 22.6% performance difference in temporal localization between the best performing closed-source and open-source models. Performance further degrades across demographic groups, indicating persistent disparities in model behavior. Overall, SONIC-O1 provides an open evaluation suite for temporally grounded and socially robust multimodal understanding. We release SONIC-O1 for reproducibility and research: Project page: https://vectorinstitute.github.io/sonic-o1/ Dataset: https://huggingface.co/datasets/vector-institute/sonic-o1 Github: https://github.com/vectorinstitute/sonic-o1 Leaderboard: https://huggingface.co/spaces/vector-institute/sonic-o1-leaderboard |
| 2026-01-29 | [Chasing Elusive Memory Bugs in GPU Programs](http://arxiv.org/abs/2601.21552v1) | Anubhab Ghosh, Ajay Nayak et al. | Memory safety bugs, such as out-of-bound accesses (OOB) in GPU programs, can compromise the security and reliability of GPU-accelerated software. We report the existence of input-dependent OOBs in the wild that manifest only under specific inputs. All existing tools to detect OOBs in GPU programs rely on runtime techniques that require an OOB to manifest for detection. Thus, input-dependent OOBs elude them. We also discover intra-allocation OOBs that arise in the presence of logical partitioning of a memory allocation into multiple data structures. Existing techniques are oblivious to the possibility of such OOBs.   We make a key observation that the presence (or absence) of semantic relations among program variables, which determines the size of allocations (CPU code) and those calculating offsets into memory allocations (GPU code), helps identify the absence (or presence) of OOBs. We build SCuBA, a first-of-its-kind compile-time technique that analyzes CPU and GPU code to capture such semantic relations (if present). It uses a SAT solver to check if an OOB access is possible under any input, given the captured relations expressed as constraints. It further analyzes GPU code to track logical partitioning of memory allocations for detecting intra-allocation OOB. Compared to NVIDIA's Compute Sanitizer that misses 45 elusive memory bugs across 20 programs, SCuBA misses none with no false alarms. |
| 2026-01-29 | [From Vulnerable to Resilient: Examining Parent and Teen Perceptions on How to Respond to Unwanted Cybergrooming Advances](http://arxiv.org/abs/2601.21518v1) | Xinyi Zhang, Mamtaj Akter et al. | Cybergrooming is a form of online abuse that threatens teens' mental health and physical safety. Yet, most prior work has focused on detecting perpetrators' behaviors, leaving a limited understanding of how teens might respond to such unwanted advances. To address this gap, we conducted an online survey with 74 participants -- 51 parents and 23 teens -- who responded to simulated cybergrooming scenarios in two ways: responses that they think would make teens more vulnerable or resilient to unwanted sexual advances. Through a mixed-methods analysis, we identified four types of vulnerable responses (encouraging escalation, accepting an advance, displaying vulnerability, and negating risk concern) and four types of protective strategies (setting boundaries, directly declining, signaling risk awareness, and leveraging avoidance techniques). As the cybergrooming risk escalated, both vulnerable responses and protective strategies showed a corresponding progression. This study contributes a teen-centered understanding of cybergrooming, a labeled dataset, and a stage-based taxonomy of perceived protective strategies, while offering implications for educational programs and sociotechnical interventions. |
| 2026-01-29 | [HERS: Hidden-Pattern Expert Learning for Risk-Specific Vehicle Damage Adaptation in Diffusion Models](http://arxiv.org/abs/2601.21517v1) | Teerapong Panboonyuen | Recent advances in text-to-image (T2I) diffusion models have enabled increasingly realistic synthesis of vehicle damage, raising concerns about their reliability in automated insurance workflows. The ability to generate crash-like imagery challenges the boundary between authentic and synthetic data, introducing new risks of misuse in fraud or claim manipulation. To address these issues, we propose HERS (Hidden-Pattern Expert Learning for Risk-Specific Damage Adaptation), a framework designed to improve fidelity, controllability, and domain alignment of diffusion-generated damage images. HERS fine-tunes a base diffusion model via domain-specific expert adaptation without requiring manual annotation. Using self-supervised image-text pairs automatically generated by a large language model and T2I pipeline, HERS models each damage category, such as dents, scratches, broken lights, or cracked paint, as a separate expert. These experts are later integrated into a unified multi-damage model that balances specialization with generalization. We evaluate HERS across four diffusion backbones and observe consistent improvements: plus 5.5 percent in text faithfulness and plus 2.3 percent in human preference ratings compared to baselines. Beyond image fidelity, we discuss implications for fraud detection, auditability, and safe deployment of generative models in high-stakes domains. Our findings highlight both the opportunities and risks of domain-specific diffusion, underscoring the importance of trustworthy generation in safety-critical applications such as auto insurance. |
| 2026-01-29 | [From Basins to safe sets: a machine learning perspective on chaotic dynamics](http://arxiv.org/abs/2601.21510v1) | David Valle, Alexandre Wagemakers et al. | The study of chaos has long relied on computationally intensive methods to quantify unpredictability and design control strategies. Recent advances in machine learning, from convolutional neural networks to transformer architectures, provide new ways to analyze complex phase space structures and enable real time action in chaotic dynamics. In this perspective article, we highlight how data driven approaches can accelerate classical tasks such as estimating basin characterization metrics, or partial control of transient chaos, while opening new possibilities for scalable and robust interventions in chaotic systems. In recent studies, convolutional networks have reproduced classical basin metrics with negligible bias and low computational cost, while transformer based surrogates have computed accurate safety functions within seconds, bypassing the recursive procedures required by traditional methods. We discuss current opportunities, remaining challenges, and future directions at the intersection of nonlinear dynamics and artificial intelligence. |
| 2026-01-29 | [The Effectiveness of Style Vectors for Steering Large Language Models: A Human Evaluation](http://arxiv.org/abs/2601.21505v1) | Diaoulé Diallo, Katharina Dworatzyk et al. | Controlling the behavior of large language models (LLMs) at inference time is essential for aligning outputs with human abilities and safety requirements. \emph{Activation steering} provides a lightweight alternative to prompt engineering and fine-tuning by directly modifying internal activations to guide generation. This research advances the literature in three significant directions. First, while previous work demonstrated the technical feasibility of steering emotional tone using automated classifiers, this paper presents the first human evaluation of activation steering concerning the emotional tone of LLM outputs, collecting over 7,000 crowd-sourced ratings from 190 participants via Prolific ($n=190$). These ratings assess both perceived emotional intensity and overall text quality. Second, we find strong alignment between human and model-based quality ratings (mean $r=0.776$, range $0.157$--$0.985$), indicating automatic scoring can proxy perceived quality. Moderate steering strengths ($λ\approx 0.15$) reliably amplify target emotions while preserving comprehensibility, with the strongest effects for disgust ($η_p^2 = 0.616$) and fear ($η_p^2 = 0.540$), and minimal effects for surprise ($η_p^2 = 0.042$). Finally, upgrading from Alpaca to LlaMA-3 yielded more consistent steering with significant effects across emotions and strengths (all $p < 0.001$). Inter-rater reliability was high (ICC $= 0.71$--$0.87$), underscoring the robustness of the findings. These findings support activation-based control as a scalable method for steering LLM behavior across affective dimensions. |
| 2026-01-29 | [Fundamental Limits of Decentralized Self-Regulating Random Walks](http://arxiv.org/abs/2601.21489v1) | Ali Khalesi, Rawad Bitar | Self-regulating random walks (SRRWs) are decentralized token-passing processes on a graph allowing nodes to locally \emph{fork}, \emph{terminate}, or \emph{pass} tokens based only on a return-time \emph{age} statistic. We study SRRWs on a finite connected graph under a lazy reversible walk, with exogenous \emph{trap} deletions summarized by the absorption pressure $Λ_{\mathrm{del}}=\sum_{u\in\mathcal P_{\mathrm{trap}}}ζ(u)π(u)$ and a global per-visit fork cap $q$. Using exponential envelopes for return-time tails, we build graph-dependent Laplace envelopes that universally bound the stationary fork intensity of any age-based policy, leading to an effective triggering age $A_{\mathrm{eff}}$. A mixing-based block drift analysis then yields controller-agnostic stability limits: any policy that avoids extinction and explosion must satisfy a \emph{viability} inequality (births can overcome $Λ_{\mathrm{del}}$ at low population) and a \emph{safety} inequality (trap deletions plus deliberate terminations dominate births at high population). Under corridor-wise versions of these conditions, we obtain positive recurrence of the population to a finite corridor. |
| 2026-01-29 | [DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation](http://arxiv.org/abs/2601.21409v1) | Weitao An, Qi Liu et al. | Adaptive navigation in unfamiliar indoor environments is crucial for household service robots. Despite advances in zero-shot perception and reasoning from vision-language models, existing navigation systems still rely on single-pass scoring at the decision layer, leading to overconfident long-horizon errors and redundant exploration. To tackle these problems, we propose Dual-Stance Cooperative Debate Navigation (DSCD-Nav), a decision mechanism that replaces one-shot scoring with stance-based cross-checking and evidence-aware arbitration to improve action reliability under partial observability. Specifically, given the same observation and candidate action set, we explicitly construct two stances by conditioning the evaluation on diverse and complementary objectives: a Task-Scene Understanding (TSU) stance that prioritizes goal progress from scene-layout cues, and a Safety-Information Balancing (SIB) stance that emphasizes risk and information value. The stances conduct a cooperative debate and make policy by cross-checking their top candidates with cue-grounded arguments. Then, a Navigation Consensus Arbitration (NCA) agent is employed to consolidate both sides' reasons and evidence, optionally triggering lightweight micro-probing to verify uncertain choices, preserving NCA's primary intent while disambiguating. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate consistent improvements in success and path efficiency while reducing exploration redundancy. |
| 2026-01-29 | [RerouteGuard: Understanding and Mitigating Adversarial Risks for LLM Routing](http://arxiv.org/abs/2601.21380v1) | Wenhui Zhang, Huiyu Xu et al. | Recent advancements in multi-model AI systems have leveraged LLM routers to reduce computational cost while maintaining response quality by assigning queries to the most appropriate model. However, as classifiers, LLM routers are vulnerable to novel adversarial attacks in the form of LLM rerouting, where adversaries prepend specially crafted triggers to user queries to manipulate routing decisions. Such attacks can lead to increased computational cost, degraded response quality, and even bypass safety guardrails, yet their security implications remain largely underexplored. In this work, we bridge this gap by systematizing LLM rerouting threats based on the adversary's objectives (i.e., cost escalation, quality hijacking, and safety bypass) and knowledge. Based on the threat taxonomy, we conduct a measurement study of real-world LLM routing systems against existing LLM rerouting attacks. The results reveal that existing routing systems are vulnerable to rerouting attacks, especially in the cost escalation scenario. We then characterize existing rerouting attacks using interpretability techniques, revealing that they exploit router decision boundaries through confounder gadgets that prepend queries to force misrouting. To mitigate these risks, we introduce RerouteGuard, a flexible and scalable guardrail framework for LLM rerouting. RerouteGuard filters adversarial rerouting prompts via dynamic embedding-based detection and adaptive thresholding. Extensive evaluations in three attack settings and four benchmarks demonstrate that RerouteGuard achieves over 99% detection accuracy against state-of-the-art rerouting attacks, while maintaining negligible impact on legitimate queries. The experimental results indicate that RerouteGuard offers a principled and practical solution for safeguarding multi-model AI systems against adversarial rerouting. |

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



