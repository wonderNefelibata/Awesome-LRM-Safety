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
| 2025-10-08 | [Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts](http://arxiv.org/abs/2510.07239v1) | Christos Ziakas, Nicholas Loo et al. | Automated red-teaming has emerged as a scalable approach for auditing Large Language Models (LLMs) prior to deployment, yet existing approaches lack mechanisms to efficiently adapt to model-specific vulnerabilities at inference. We introduce Red-Bandit, a red-teaming framework that adapts online to identify and exploit model failure modes under distinct attack styles (e.g., manipulation, slang). Red-Bandit post-trains a set of parameter-efficient LoRA experts, each specialized for a particular attack style, using reinforcement learning that rewards the generation of unsafe prompts via a rule-based safety model. At inference, a multi-armed bandit policy dynamically selects among these attack-style experts based on the target model's response safety, balancing exploration and exploitation. Red-Bandit achieves state-of-the-art results on AdvBench under sufficient exploration (ASR@10), while producing more human-readable prompts (lower perplexity). Moreover, Red-Bandit's bandit policy serves as a diagnostic tool for uncovering model-specific vulnerabilities by indicating which attack styles most effectively elicit unsafe behaviors. |
| 2025-10-08 | [HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving](http://arxiv.org/abs/2510.07210v1) | Donald Pfaffmann, Matthias Klusch et al. | We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners. |
| 2025-10-08 | [EigenScore: OOD Detection using Covariance in Diffusion Models](http://arxiv.org/abs/2510.07206v1) | Shirin Shoushtari, Yi Wang et al. | Out-of-distribution (OOD) detection is critical for the safe deployment of machine learning systems in safety-sensitive domains. Diffusion models have recently emerged as powerful generative models, capable of capturing complex data distributions through iterative denoising. Building on this progress, recent work has explored their potential for OOD detection. We propose EigenScore, a new OOD detection method that leverages the eigenvalue spectrum of the posterior covariance induced by a diffusion model. We argue that posterior covariance provides a consistent signal of distribution shift, leading to larger trace and leading eigenvalues on OOD inputs, yielding a clear spectral signature. We further provide analysis explicitly linking posterior covariance to distribution mismatch, establishing it as a reliable signal for OOD detection. To ensure tractability, we adopt a Jacobian-free subspace iteration method to estimate the leading eigenvalues using only forward evaluations of the denoiser. Empirically, EigenScore achieves SOTA performance, with up to 5% AUROC improvement over the best baseline. Notably, it remains robust in near-OOD settings such as CIFAR-10 vs CIFAR-100, where existing diffusion-based methods often fail. |
| 2025-10-08 | [Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples](http://arxiv.org/abs/2510.07192v1) | Alexandra Souly, Javier Rando et al. | Poisoning attacks can compromise the safety of large language models (LLMs) by injecting malicious documents into their training data. Existing work has studied pretraining poisoning assuming adversaries control a percentage of the training corpus. However, for large models, even small percentages translate to impractically large amounts of data. This work demonstrates for the first time that poisoning attacks instead require a near-constant number of documents regardless of dataset size. We conduct the largest pretraining poisoning experiments to date, pretraining models from 600M to 13B parameters on chinchilla-optimal datasets (6B to 260B tokens). We find that 250 poisoned documents similarly compromise models across all model and dataset sizes, despite the largest models training on more than 20 times more clean data. We also run smaller-scale experiments to ablate factors that could influence attack success, including broader ratios of poisoned to clean data and non-random distributions of poisoned samples. Finally, we demonstrate the same dynamics for poisoning during fine-tuning. Altogether, our results suggest that injecting backdoors through data poisoning may be easier for large models than previously believed as the number of poisons required does not scale up with model size, highlighting the need for more research on defences to mitigate this risk in future models. |
| 2025-10-08 | [A multi-layered embedded intrusion detection framework for programmable logic controllers](http://arxiv.org/abs/2510.07171v1) | Rishabh Das. Aaron Werth, Tommy Morris | Industrial control system (ICS) operations use trusted endpoints like human machine interfaces (HMIs) and workstations to relay commands to programmable logic controllers (PLCs). Because most PLCs lack layered defenses, compromise of a trusted endpoint can drive unsafe actuator commands and risk safety-critical operation. This research presents an embedded intrusion detection system that runs inside the controller and uses header-level telemetry to detect and respond to network attacks. The system combines a semi-supervised anomaly detector and a supervised attack classifier. We evaluate the approach on a midstream oil-terminal testbed using three datasets collected during tanker-truck loading. The anomaly detector achieves zero missed attacks, corresponding to 0.998 Matthews correlation. The supervised stage attains 97.37 percent hold-out accuracy and 97.03 percent external accuracy. The embedded design adds a median of 2,031 microseconds of end-to-end latency and does not impact PLC's cycle time. The proposed architecture provides a multi-layer embedded security that meets the real-time requirements of an industrial system. |
| 2025-10-08 | [Stability Preserving Safe Control of a Bicopter](http://arxiv.org/abs/2510.07145v1) | Jhon Manuel Portella Delgado, Ankit Goel | This paper presents a control law for stabilization and trajectory tracking of a multicopter subject to safety constraints. The proposed approach guarantees forward invariance of a prescribed safety set while ensuring smooth tracking performance. Unlike conventional control barrier function methods, the constrained control problem is transformed into an unconstrained one using state-dependent mappings together with carefully constructed Lyapunov functions. This approach enables explicit synthesis of the control law, instead of requiring a solution of constrained optimization at each step. The transformation also enables the controller to enforce safety without sacrificing stability or performance. Simulation results for a polytopic reference trajectory confined within a designated safe region demonstrate the effectiveness of the proposed method. |
| 2025-10-08 | [A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model](http://arxiv.org/abs/2510.07133v1) | Tony Zhang, Burak Kantarci et al. | Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety. |
| 2025-10-08 | [Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages](http://arxiv.org/abs/2510.07000v1) | Neel Prabhanjan Rachamalla, Aravind Konakalla et al. | The effectiveness of Large Language Models (LLMs) depends heavily on the availability of high-quality post-training data, particularly instruction-tuning and preference-based examples. Existing open-source datasets, however, often lack multilingual coverage, cultural grounding, and suffer from task diversity gaps that are especially pronounced for Indian languages. We introduce a human-in-the-loop pipeline that combines translations with synthetic expansion to produce reliable and diverse Indic post-training data. Using this pipeline, we curate two datasets: Pragyaan-IT (22.5K) and Pragyaan-Align (100K) across 10 Indian languages covering 13 broad and 56 sub-categories, leveraging 57 diverse datasets. Our dataset protocol incorporates several often-overlooked dimensions and emphasize task diversity, multi-turn dialogue, instruction fidelity, safety alignment, and preservation of cultural nuance, providing a foundation for more inclusive and effective multilingual LLMs. |
| 2025-10-08 | [RedTWIZ: Diverse LLM Red Teaming via Adaptive Attack Planning](http://arxiv.org/abs/2510.06994v1) | Artur Horal, Daniel Pina et al. | This paper presents the vision, scientific contributions, and technical details of RedTWIZ: an adaptive and diverse multi-turn red teaming framework, to audit the robustness of Large Language Models (LLMs) in AI-assisted software development. Our work is driven by three major research streams: (1) robust and systematic assessment of LLM conversational jailbreaks; (2) a diverse generative multi-turn attack suite, supporting compositional, realistic and goal-oriented jailbreak conversational strategies; and (3) a hierarchical attack planner, which adaptively plans, serializes, and triggers attacks tailored to specific LLM's vulnerabilities. Together, these contributions form a unified framework -- combining assessment, attack generation, and strategic planning -- to comprehensively evaluate and expose weaknesses in LLMs' robustness. Extensive evaluation is conducted to systematically assess and analyze the performance of the overall system and each component. Experimental results demonstrate that our multi-turn adversarial attack strategies can successfully lead state-of-the-art LLMs to produce unsafe generations, highlighting the pressing need for more research into enhancing LLM's robustness. |
| 2025-10-08 | [OBJVanish: Physically Realizable Text-to-3D Adv. Generation of LiDAR-Invisible Objects](http://arxiv.org/abs/2510.06952v1) | Bing Li, Wuqi Wang et al. | LiDAR-based 3D object detectors are fundamental to autonomous driving, where failing to detect objects poses severe safety risks. Developing effective 3D adversarial attacks is essential for thoroughly testing these detection systems and exposing their vulnerabilities before real-world deployment. However, existing adversarial attacks that add optimized perturbations to 3D points have two critical limitations: they rarely cause complete object disappearance and prove difficult to implement in physical environments. We introduce the text-to-3D adversarial generation method, a novel approach enabling physically realizable attacks that can generate 3D models of objects truly invisible to LiDAR detectors and be easily realized in the real world. Specifically, we present the first empirical study that systematically investigates the factors influencing detection vulnerability by manipulating the topology, connectivity, and intensity of individual pedestrian 3D models and combining pedestrians with multiple objects within the CARLA simulation environment. Building on the insights, we propose the physically-informed text-to-3D adversarial generation (Phy3DAdvGen) that systematically optimizes text prompts by iteratively refining verbs, objects, and poses to produce LiDAR-invisible pedestrians. To ensure physical realizability, we construct a comprehensive object pool containing 13 3D models of real objects and constrain Phy3DAdvGen to generate 3D objects based on combinations of objects in this set. Extensive experiments demonstrate that our approach can generate 3D pedestrians that evade six state-of-the-art (SOTA) LiDAR 3D detectors in both CARLA simulation and physical environments, thereby highlighting vulnerabilities in safety-critical applications. |
| 2025-10-08 | [Dynamic Control Aware Semantic Communication Enabled Image Transmission for Lunar Landing](http://arxiv.org/abs/2510.06916v1) | Fangzhou Zhao, Yao Sun et al. | The primary challenge in autonomous lunar landing missions lies in the unreliable local control system, which has limited capacity to handle high-dynamic conditions, severely affecting landing precision and safety. Recent advancements in lunar satellite communication make it possible to establish a wireless link between lunar orbit satellites and the lunar lander. This enables satellites to run high-performance autonomous landing algorithms, improving landing accuracy while reducing the lander's computational and storage load. Nevertheless, traditional communication paradigms are not directly applicable due to significant temperature fluctuations on the lunar surface, intense solar radiation, and severe interference caused by lunar dust on hardware. The emerging technique of semantic communication (SemCom) offers significant advantages in robustness and resource efficiency, particularly under harsh channel conditions. In this paper, we introduce a novel SemCom framework for transmitting images from the lander to satellites operating the remote landing control system. The proposed encoder-decoder dynamically adjusts the transmission strategy based on real-time feedback from the lander's control algorithm, ensuring the accurate delivery of critical image features and enhancing control reliability. We provide a rigorous theoretical analysis of the conditions that improve the accuracy of the control algorithm and reduce end-to-end transmission time under the proposed framework. Simulation results demonstrate that our SemCom method significantly enhances autonomous landing performance compared to traditional communication methods. |
| 2025-10-08 | [LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling](http://arxiv.org/abs/2510.06915v1) | Zecheng Tang, Baibei Ji et al. | Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model. |
| 2025-10-08 | [SaFeR-VLM: Toward Safety-aware Fine-grained Reasoning in Multimodal Models](http://arxiv.org/abs/2510.06871v1) | Huahui Yi, Kun Wang et al. | Multimodal Large Reasoning Models (MLRMs) demonstrate impressive cross-modal reasoning but often amplify safety risks under adversarial or unsafe prompts, a phenomenon we call the \textit{Reasoning Tax}. Existing defenses mainly act at the output level and do not constrain the reasoning process, leaving models exposed to implicit risks. In this paper, we propose SaFeR-VLM, a safety-aligned reinforcement learning framework that embeds safety directly into multimodal reasoning. The framework integrates four components: (I) QI-Safe-10K, a curated dataset emphasizing safety-critical and reasoning-sensitive cases; (II) safety-aware rollout, where unsafe generations undergo reflection and correction instead of being discarded; (III) structured reward modeling with multi-dimensional weighted criteria and explicit penalties for hallucinations and contradictions; and (IV) GRPO optimization, which reinforces both safe and corrected trajectories. This unified design shifts safety from a passive safeguard to an active driver of reasoning, enabling scalable and generalizable safety-aware reasoning. SaFeR-VLM further demonstrates robustness against both explicit and implicit risks, supporting dynamic and interpretable safety decisions beyond surface-level filtering. SaFeR-VLM-3B achieves average performance $70.13$ and $78.97$ on safety and helpfulness across six benchmarks, surpassing both same-scale and $>10\times$ larger models such as Skywork-R1V3-38B, Qwen2.5VL-72B, and GLM4.5V-106B. Remarkably, SaFeR-VLM-7B benefits from its increased scale to surpass GPT-5-mini and Gemini-2.5-Flash by \num{6.47} and \num{16.76} points respectively on safety metrics, achieving this improvement without any degradation in helpfulness performance. Our codes are available at https://github.com/HarveyYi/SaFeR-VLM. |
| 2025-10-08 | [Decentralized CBF-based Safety Filters for Collision Avoidance of Cooperative Missile Systems with Input Constraints](http://arxiv.org/abs/2510.06846v1) | Johannes Autenrieb, Mark Spiller | This paper presents a decentralized safety filter for collision avoidance in multi-agent aerospace interception scenarios. The approach leverages robust control barrier functions (RCBFs) to guarantee forward invariance of safety sets under bounded inputs and high-relative-degree dynamics. Each effector executes its nominal cooperative guidance command, while a local quadratic program (QP) modifies the input only when necessary. Event-triggered activation based on range and zero-effort miss (ZEM) criteria ensures scalability by restricting active constraints to relevant neighbors. To resolve feasibility issues from simultaneous constraints, a slack-variable relaxation scheme is introduced that prioritizes critical agents in a Pareto-optimal manner. Simulation results in many-on-many interception scenarios demonstrate that the proposed framework maintains collision-free operation with minimal deviation from nominal guidance, providing a computationally efficient and scalable solution for safety-critical multi-agent aerospace systems. |
| 2025-10-08 | [Overview of the Plagiarism Detection Task at PAN 2025](http://arxiv.org/abs/2510.06805v1) | André Greiner-Petter, Maik Fröbe et al. | The generative plagiarism detection task at PAN 2025 aims at identifying automatically generated textual plagiarism in scientific articles and aligning them with their respective sources. We created a novel large-scale dataset of automatically generated plagiarism using three large language models: Llama, DeepSeek-R1, and Mistral. In this task overview paper, we outline the creation of this dataset, summarize and compare the results of all participants and four baselines, and evaluate the results on the last plagiarism detection task from PAN 2015 in order to interpret the robustness of the proposed approaches. We found that the current iteration does not invite a large variety of approaches as naive semantic similarity approaches based on embedding vectors provide promising results of up to 0.8 recall and 0.5 precision. In contrast, most of these approaches underperform significantly on the 2015 dataset, indicating a lack in generalizability. |
| 2025-10-08 | [FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline](http://arxiv.org/abs/2510.06800v1) | Haotian Wu, Shufan Jiang et al. | As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench. |
| 2025-10-08 | [Extreme Amodal Face Detection](http://arxiv.org/abs/2510.06791v1) | Changlin Song, Yunzhong Hou et al. | Extreme amodal detection is the task of inferring the 2D location of objects that are not fully visible in the input image but are visible within an expanded field-of-view. This differs from amodal detection, where the object is partially visible within the input image, but is occluded. In this paper, we consider the sub-problem of face detection, since this class provides motivating applications involving safety and privacy, but do not tailor our method specifically to this class. Existing approaches rely on image sequences so that missing detections may be interpolated from surrounding frames or make use of generative models to sample possible completions. In contrast, we consider the single-image task and propose a more efficient, sample-free approach that makes use of the contextual cues from the image to infer the presence of unseen faces. We design a heatmap-based extreme amodal object detector that addresses the problem of efficiently predicting a lot (the out-of-frame region) from a little (the image) with a selective coarse-to-fine decoder. Our method establishes strong results for this new task, even outperforming less efficient generative approaches. |
| 2025-10-08 | [Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness](http://arxiv.org/abs/2510.06790v1) | Tavish McDonald, Bo Lei et al. | Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization, while RL finetuning and protracted reasoning are not critical. For example, increasing emphasis on defensive specifications via prompting lowers the success rate of gradient-based multimodal attacks on VLMs robustified by adversarial pretraining, but this same intervention provides no such benefit to not-robustified models. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Accordingly, we advise layering train-time and test-time defenses to obtain their synergistic benefit. |
| 2025-10-08 | [Verifying Memoryless Sequential Decision-making of Large Language Models](http://arxiv.org/abs/2510.06756v1) | Dennis Gross, Helge Spieker et al. | We introduce a tool for rigorous and automated verification of large language model (LLM)- based policies in memoryless sequential decision-making tasks. Given a Markov decision process (MDP) representing the sequential decision-making task, an LLM policy, and a safety requirement expressed as a PCTL formula, our approach incrementally constructs only the reachable portion of the MDP guided by the LLM's chosen actions. Each state is encoded as a natural language prompt, the LLM's response is parsed into an action, and reachable successor states by the policy are expanded. The resulting formal model is checked with Storm to determine whether the policy satisfies the specified safety property. In experiments on standard grid world benchmarks, we show that open source LLMs accessed via Ollama can be verified when deterministically seeded, but generally underperform deep reinforcement learning baselines. Our tool natively integrates with Ollama and supports PRISM-specified tasks, enabling continuous benchmarking in user-specified sequential decision-making tasks and laying a practical foundation for formally verifying increasingly capable LLMs. |
| 2025-10-08 | [SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis](http://arxiv.org/abs/2510.06717v1) | Yuanfei Lin, Sebastian Illing et al. | Large language models have been widely applied to knowledge-driven decision-making for automated vehicles due to their strong generalization and reasoning capabilities. However, the safety of the resulting decisions cannot be ensured due to possible hallucinations and the lack of integrated vehicle dynamics. To address this issue, we propose SanDRA, the first safe large-language-model-based decision making framework for automated vehicles using reachability analysis. Our approach starts with a comprehensive description of the driving scenario to prompt large language models to generate and rank feasible driving actions. These actions are translated into temporal logic formulas that incorporate formalized traffic rules, and are subsequently integrated into reachability analysis to eliminate unsafe actions. We validate our approach in both open-loop and closed-loop driving environments using off-the-shelf and finetuned large language models, showing that it can provide provably safe and, where possible, legally compliant driving actions, even under high-density traffic conditions. To ensure transparency and facilitate future research, all code and experimental setups are publicly available at github.com/CommonRoad/SanDRA. |

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



