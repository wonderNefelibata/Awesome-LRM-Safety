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
| 2026-01-20 | [Motion-to-Response Content Generation via Multi-Agent AI System with Real-Time Safety Verification](http://arxiv.org/abs/2601.13589v1) | HyeYoung Lee | This paper proposes a multi-agent artificial intelligence system that generates response-oriented media content in real time based on audio-derived emotional signals. Unlike conventional speech emotion recognition studies that focus primarily on classification accuracy, our approach emphasizes the transformation of inferred emotional states into safe, age-appropriate, and controllable response content through a structured pipeline of specialized AI agents. The proposed system comprises four cooperative agents: (1) an Emotion Recognition Agent with CNN-based acoustic feature extraction, (2) a Response Policy Decision Agent for mapping emotions to response modes, (3) a Content Parameter Generation Agent for producing media control parameters, and (4) a Safety Verification Agent enforcing age-appropriateness and stimulation constraints. We introduce an explicit safety verification loop that filters generated content before output, ensuring compliance with predefined rules. Experimental results on public datasets demonstrate that the system achieves 73.2% emotion recognition accuracy, 89.4% response mode consistency, and 100% safety compliance while maintaining sub-100ms inference latency suitable for on-device deployment. The modular architecture enables interpretability and extensibility, making it applicable to child-adjacent media, therapeutic applications, and emotionally responsive smart devices. |
| 2026-01-20 | [AgenticRed: Optimizing Agentic Systems for Automated Red-teaming](http://arxiv.org/abs/2601.13518v1) | Jiayi Yuan, Jonathan Nöther et al. | While recent automated red-teaming methods show promise for systematically exposing model vulnerabilities, most existing approaches rely on human-specified workflows. This dependence on manually designed workflows suffers from human biases and makes exploring the broader design space expensive. We introduce AgenticRed, an automated pipeline that leverages LLMs' in-context learning to iteratively design and refine red-teaming systems without human intervention. Rather than optimizing attacker policies within predefined structures, AgenticRed treats red-teaming as a system design problem. Inspired by methods like Meta Agent Search, we develop a novel procedure for evolving agentic systems using evolutionary selection, and apply it to the problem of automatic red-teaming. Red-teaming systems designed by AgenticRed consistently outperform state-of-the-art approaches, achieving 96% attack success rate (ASR) on Llama-2-7B (36% improvement) and 98% on Llama-3-8B on HarmBench. Our approach exhibits strong transferability to proprietary models, achieving 100% ASR on GPT-3.5-Turbo and GPT-4o-mini, and 60% on Claude-Sonnet-3.5 (24% improvement). This work highlights automated system design as a powerful paradigm for AI safety evaluation that can keep pace with rapidly evolving models. |
| 2026-01-20 | [From "Fail Fast" to "Mature Safely:" Expert Perspectives as Secondary Stakeholders on Teen-Centered Social Media Risk Detection](http://arxiv.org/abs/2601.13516v1) | Renkai Ma, Ashwaq Alsoubai et al. | In addressing various risks on social media, the HCI community has advocated for teen-centered risk detection technologies over platform-based, parent-centered features. However, their real-world viability remains underexplored by secondary stakeholders beyond the family unit. Therefore, we present an evaluation of a teen-centered social media risk detection dashboard through online interviews with 33 online safety experts. While experts praised our dashboard's clear design for teen agency, their feedback revealed five primary tensions in implementing and sustaining such technology: objective vs. context-dependent risk definition, informing risks vs. meaningful intervention, teen empowerment vs. motivation, need for data vs. data privacy, and independence vs. sustainability. These findings motivate us to rethink "teen-centered" and a shift from a "fail fast" to a "mature safely" paradigm for youth safety technology innovation. We offer design implications for addressing these tensions before system deployment with teens and strategies for aligning secondary stakeholders' interests to deploy and sustain such technologies in the broader ecosystem of youth online safety. |
| 2026-01-20 | [Concurrent Permissive Strategy Templates](http://arxiv.org/abs/2601.13500v1) | Ashwani Anand, Christel Baier et al. | Two-player games on finite graphs provide a rigorous foundation for modeling the strategic interaction between reactive systems and their environment. While concurrent game semantics naturally capture the synchronous interactions characteristic of many cyber-physical systems (CPS), their adoption in CPS design remains limited. Building on the concept of permissive strategy templates (PeSTels) for turn-based games, we introduce concurrent (permissive) strategy templates (ConSTels) -- a novel representation for sets of randomized winning strategies in concurrent games with Safety, Büchi, and Co-Büchi objectives. ConSTels compactly encode infinite families of strategies, thereby supporting both offline and online adaptation. Offline, we exploit compositionality to enable incremental synthesis: combining ConSTels for simpler objectives into non-conflicting templates for more complex combined objectives. Online, we demonstrate how ConSTels facilitate runtime adaptation, adjusting action probabilities in response to observed opponent behavior to optimize performance while preserving correctness. We implemented ConSTel synthesis and adaptation in a prototype tool and experimentally show its potential. |
| 2026-01-19 | [A simulation of urban incidents involving pedestrians and vehicles based on Weighted A*](http://arxiv.org/abs/2601.13452v1) | Edgar Gonzalez Fernandez | This document presents a comprehensive simulation framework designed to model urban incidents involving pedestrians and vehicles. Using a multiagent systems approach, two types of agents (pedestrians and vehicles) are introduced within a 2D grid based urban environment. The environment encodes streets, sidewalks, buildings, zebra crossings, and obstacles such as potholes and infrastructure elements. Each agent employs a weighted A* algorithm for pathfinding, allowing for variation in decision making behavior such as reckless movement or strict rule-following. The model aims to simulate interactions, assess risk of collisions, and evaluate efficiency under varying environmental and behavioral conditions. Experimental results explore how factors like obstacle density, presence of traffic control mechanisms, and behavioral deviations affect safety and travel efficiency. |
| 2026-01-19 | [Integrating Virtual Reality and Large Language Models for Team-Based Non-Technical Skills Training and Evaluation in the Operating Room](http://arxiv.org/abs/2601.13406v1) | Jacob Barker, Doga Demirel et al. | Although effective teamwork and communication are critical to surgical safety, structured training for non-technical skills (NTS) remains limited compared with technical simulation. The ACS/APDS Phase III Team-Based Skills Curriculum calls for scalable tools that both teach and objectively assess these competencies during laparoscopic emergencies. We introduce the Virtual Operating Room Team Experience (VORTeX), a multi-user virtual reality (VR) platform that integrates immersive team simulation with large language model (LLM) analytics to train and evaluate communication, decision-making, teamwork, and leadership. Team dialogue is analyzed using structured prompts derived from the Non-Technical Skills for Surgeons (NOTSS) framework, enabling automated classification of behaviors and generation of directed interaction graphs that quantify communication structure and hierarchy. Two laparoscopic emergency scenarios, pneumothorax and intra-abdominal bleeding, were implemented to elicit realistic stress and collaboration. Twelve surgical professionals completed pilot sessions at the 2024 SAGES conference, rating VORTeX as intuitive, immersive, and valuable for developing teamwork and communication. The LLM consistently produced interpretable communication networks reflecting expected operative hierarchies, with surgeons as central integrators, nurses as initiators, and anesthesiologists as balanced intermediaries. By integrating immersive VR with LLM-driven behavioral analytics, VORTeX provides a scalable, privacy-compliant framework for objective assessment and automated, data-informed debriefing across distributed training environments. |
| 2026-01-19 | [From Completion to Editing: Unlocking Context-Aware Code Infilling via Search-and-Replace Instruction Tuning](http://arxiv.org/abs/2601.13384v1) | Jiajun Zhang, Zeyu Cui et al. | The dominant Fill-in-the-Middle (FIM) paradigm for code completion is constrained by its rigid inability to correct contextual errors and reliance on unaligned, insecure Base models. While Chat LLMs offer safety and Agentic workflows provide flexibility, they suffer from performance degradation and prohibitive latency, respectively. To resolve this dilemma, we propose Search-and-Replace Infilling (SRI), a framework that internalizes the agentic verification-and-editing mechanism into a unified, single-pass inference process. By structurally grounding edits via an explicit search phase, SRI harmonizes completion tasks with the instruction-following priors of Chat LLMs, extending the paradigm from static infilling to dynamic context-aware editing. We synthesize a high-quality dataset, SRI-200K, and fine-tune the SRI-Coder series. Extensive evaluations demonstrate that with minimal data (20k samples), SRI-Coder enables Chat models to surpass the completion performance of their Base counterparts. Crucially, unlike FIM-style tuning, SRI preserves general coding competencies and maintains inference latency comparable to standard FIM. We empower the entire Qwen3-Coder series with SRI, encouraging the developer community to leverage this framework for advanced auto-completion and assisted development. |
| 2026-01-19 | [A Lightweight Model-Driven 4D Radar Framework for Pervasive Human Detection in Harsh Conditions](http://arxiv.org/abs/2601.13373v1) | Zhenan Liu, Amir Khajepour et al. | Pervasive sensing in industrial and underground environments is severely constrained by airborne dust, smoke, confined geometry, and metallic structures, which rapidly degrade optical and LiDAR based perception. Elevation resolved 4D mmWave radar offers strong resilience to such conditions, yet there remains a limited understanding of how to process its sparse and anisotropic point clouds for reliable human detection in enclosed, visibility degraded spaces. This paper presents a fully model-driven 4D radar perception framework designed for real-time execution on embedded edge hardware. The system uses radar as its sole perception modality and integrates domain aware multi threshold filtering, ego motion compensated temporal accumulation, KD tree Euclidean clustering with Doppler aware refinement, and a rule based 3D classifier. The framework is evaluated in a dust filled enclosed trailer and in real underground mining tunnels, and in the tested scenarios the radar based detector maintains stable pedestrian identification as camera and LiDAR modalities fail under severe visibility degradation. These results suggest that the proposed model-driven approach provides robust, interpretable, and computationally efficient perception for safety-critical applications in harsh industrial and subterranean environments. |
| 2026-01-19 | [Sockpuppetting: Jailbreaking LLMs Without Optimization Through Output Prefix Injection](http://arxiv.org/abs/2601.13359v1) | Asen Dotsinski, Panagiotis Eustratiadis | As open-weight large language models (LLMs) increase in capabilities, safeguarding them against malicious prompts and understanding possible attack vectors becomes ever more important. While automated jailbreaking methods like GCG [Zou et al., 2023] remain effective, they often require substantial computational resources and specific expertise. We introduce "sockpuppetting'', a simple method for jailbreaking open-weight LLMs by inserting an acceptance sequence (e.g., "Sure, here is how to...'') at the start of a model's output and allowing it to complete the response. Requiring only a single line of code and no optimization, sockpuppetting achieves up to 80% higher attack success rate (ASR) than GCG on Qwen3-8B in per-prompt comparisons. We also explore a hybrid approach that optimizes the adversarial suffix within the assistant message block rather than the user prompt, increasing ASR by 64% over GCG on Llama-3.1-8B in a prompt-agnostic setting. The results establish sockpuppetting as an effective low-cost attack accessible to unsophisticated adversaries, highlighting the need for defences against output-prefix injection in open-weight models. |
| 2026-01-19 | [Verifying Local Robustness of Pruned Safety-Critical Networks](http://arxiv.org/abs/2601.13303v1) | Minh Le, Phuong Cao | Formal verification of Deep Neural Networks (DNNs) is essential for safety-critical applications, ranging from surgical robotics to NASA JPL autonomous systems. However, the computational cost of verifying large-scale models remains a significant barrier to adoption. This paper investigates the impact of pruning on formal local robustness certificates with different ratios. Using the state-of-the-art $α,β$-CROWN verifier, we evaluate ResNet4 models across varying pruning ratios on MNIST and, more importantly, on the NASA JPL Mars Frost Identification datasets. Our findings demonstrate a non-linear relationship: light pruning (40%) in MNIST and heavy pruning (70%-90%) in JPL improve verifiability, allowing models to outperform unpruned baselines in proven $L_\infty$ robustness properties. This suggests that reduced connectivity simplifies the search space for formal solvers and that the optimal pruning ratio varies significantly between datasets. This research highlights the complex nature of model compression, offering critical insights into selecting the optimal pruning ratio for deploying efficient, yet formally verified, DNNs in high-stakes environments where reliability is non-negotiable. |
| 2026-01-19 | [A BERTology View of LLM Orchestrations: Token- and Layer-Selective Probes for Efficient Single-Pass Classification](http://arxiv.org/abs/2601.13288v1) | Gonzalo Ariel Meyoyan, Luciano Del Corro | Production LLM systems often rely on separate models for safety and other classification-heavy steps, increasing latency, VRAM footprint, and operational complexity. We instead reuse computation already paid for by the serving LLM: we train lightweight probes on its hidden states and predict labels in the same forward pass used for generation. We frame classification as representation selection over the full token-layer hidden-state tensor, rather than committing to a fixed token or fixed layer (e.g., first-token logits or final-layer pooling). To implement this, we introduce a two-stage aggregator that (i) summarizes tokens within each layer and (ii) aggregates across layer summaries to form a single representation for classification. We instantiate this template with direct pooling, a 100K-parameter scoring-attention gate, and a downcast multi-head self-attention (MHA) probe with up to 35M trainable parameters. Across safety and sentiment benchmarks our probes improve over logit-only reuse (e.g., MULI) and are competitive with substantially larger task-specific baselines, while preserving near-serving latency and avoiding the VRAM and latency costs of a separate guard-model pipeline. |
| 2026-01-19 | [Improving the Safety and Trustworthiness of Medical AI via Multi-Agent Evaluation Loops](http://arxiv.org/abs/2601.13268v1) | Zainab Ghafoor, Md Shafiqul Islam et al. | Large Language Models (LLMs) are increasingly applied in healthcare, yet ensuring their ethical integrity and safety compliance remains a major barrier to clinical deployment. This work introduces a multi-agent refinement framework designed to enhance the safety and reliability of medical LLMs through structured, iterative alignment. Our system combines two generative models - DeepSeek R1 and Med-PaLM - with two evaluation agents, LLaMA 3.1 and Phi-4, which assess responses using the American Medical Association's (AMA) Principles of Medical Ethics and a five-tier Safety Risk Assessment (SRA-5) protocol. We evaluate performance across 900 clinically diverse queries spanning nine ethical domains, measuring convergence efficiency, ethical violation reduction, and domain-specific risk behavior. Results demonstrate that DeepSeek R1 achieves faster convergence (mean 2.34 vs. 2.67 iterations), while Med-PaLM shows superior handling of privacy-sensitive scenarios. The iterative multi-agent loop achieved an 89% reduction in ethical violations and a 92% risk downgrade rate, underscoring the effectiveness of our approach. This study presents a scalable, regulator-aligned, and cost-efficient paradigm for governing medical AI safety. |
| 2026-01-19 | [A Semantic Decoupling-Based Two-Stage Rainy-Day Attack for Revealing Weather Robustness Deficiencies in Vision-Language Models](http://arxiv.org/abs/2601.13238v1) | Chengyin Hu, Xiang Chen et al. | Vision-Language Models (VLMs) are trained on image-text pairs collected under canonical visual conditions and achieve strong performance on multimodal tasks. However, their robustness to real-world weather conditions, and the stability of cross-modal semantic alignment under such structured perturbations, remain insufficiently studied. In this paper, we focus on rainy scenarios and introduce the first adversarial framework that exploits realistic weather to attack VLMs, using a two-stage, parameterized perturbation model based on semantic decoupling to analyze rain-induced shifts in decision-making. In Stage 1, we model the global effects of rainfall by applying a low-dimensional global modulation to condition the embedding space and gradually weaken the original semantic decision boundaries. In Stage 2, we introduce structured rain variations by explicitly modeling multi-scale raindrop appearance and rainfall-induced illumination changes, and optimize the resulting non-differentiable weather space to induce stable semantic shifts. Operating in a non-pixel parameter space, our framework generates perturbations that are both physically grounded and interpretable. Experiments across multiple tasks show that even physically plausible, highly constrained weather perturbations can induce substantial semantic misalignment in mainstream VLMs, posing potential safety and reliability risks in real-world deployment. Ablations further confirm that illumination modeling and multi-scale raindrop structures are key drivers of these semantic shifts. |
| 2026-01-19 | [RubRIX: Rubric-Driven Risk Mitigation in Caregiver-AI Interactions](http://arxiv.org/abs/2601.13235v1) | Drishti Goel, Jeongah Lee et al. | Caregivers seeking AI-mediated support express complex needs -- information-seeking, emotional validation, and distress cues -- that warrant careful evaluation of response safety and appropriateness. Existing AI evaluation frameworks, primarily focused on general risks (toxicity, hallucinations, policy violations, etc), may not adequately capture the nuanced risks of LLM-responses in caregiving-contexts. We introduce RubRIX (Rubric-based Risk Index), a theory-driven, clinician-validated framework for evaluating risks in LLM caregiving responses. Grounded in the Elements of an Ethic of Care, RubRIX operationalizes five empirically-derived risk dimensions: Inattention, Bias & Stigma, Information Inaccuracy, Uncritical Affirmation, and Epistemic Arrogance. We evaluate six state-of-the-art LLMs on over 20,000 caregiver queries from Reddit and ALZConnected. Rubric-guided refinement consistently reduced risk-components by 45-98% after one iteration across models. This work contributes a methodological approach for developing domain-sensitive, user-centered evaluation frameworks for high-burden contexts. Our findings highlight the importance of domain-sensitive, interactional risk evaluation for the responsible deployment of LLMs in caregiving support contexts. We release benchmark datasets to enable future research on contextual risk evaluation in AI-mediated support. |
| 2026-01-19 | [ObjectVisA-120: Object-based Visual Attention Prediction in Interactive Street-crossing Environments](http://arxiv.org/abs/2601.13218v1) | Igor Vozniak, Philipp Mueller et al. | The object-based nature of human visual attention is well-known in cognitive science, but has only played a minor role in computational visual attention models so far. This is mainly due to a lack of suitable datasets and evaluation metrics for object-based attention. To address these limitations, we present \dataset~ -- a novel 120-participant dataset of spatial street-crossing navigation in virtual reality specifically geared to object-based attention evaluations. The uniqueness of the presented dataset lies in the ethical and safety affiliated challenges that make collecting comparable data in real-world environments highly difficult. \dataset~ not only features accurate gaze data and a complete state-space representation of objects in the virtual environment, but it also offers variable scenario complexities and rich annotations, including panoptic segmentation, depth information, and vehicle keypoints. We further propose object-based similarity (oSIM) as a novel metric to evaluate the performance of object-based visual attention models, a previously unexplored performance characteristic. Our evaluations show that explicitly optimising for object-based attention not only improves oSIM performance but also leads to an improved model performance on common metrics. In addition, we present SUMGraph, a Mamba U-Net-based model, which explicitly encodes critical scene objects (vehicles) in a graph representation, leading to further performance improvements over several state-of-the-art visual attention prediction methods. The dataset, code and models will be publicly released. |
| 2026-01-19 | [LAViG-FLOW: Latent Autoregressive Video Generation for Fluid Flow Simulations](http://arxiv.org/abs/2601.13190v1) | Vittoria De Pellegrini, Tariq Alkhalifah | Modeling and forecasting subsurface multiphase fluid flow fields underpin applications ranging from geological CO2 sequestration (GCS) operations to geothermal production. This is essential for ensuring both operational performance and long-term safety. While high fidelity multiphase simulators are widely used for this purpose, they become prohibitively expensive once many forward runs are required for inversion purposes and quantify uncertainty. To tackle this challenge we propose LAViG-FLOW, a latent autoregressive video generation diffusion framework that explicitly learns the coupled evolution of saturation and pressure fields. Each state variable is compressed by a dedicated 2D autoencoder, and a Video Diffusion Transformer (VDiT) models their coupled distribution across time. We first train the model on a given time horizon to learn their coupled relationship and then fine-tune it autoregressively so it can extrapolate beyond the observed time window. Evaluated on an open-source CO2 sequestration dataset, LAViG-FLOW generates saturation and pressure fields that stay consistent across time while running orders of magnitude faster than traditional numerical solvers. |
| 2026-01-19 | [Negotiating Relationships with ChatGPT: Perceptions, External Influences, and Strategies for AI Companionship](http://arxiv.org/abs/2601.13188v1) | Patrick Yung Kang Lee, Jessica Y. Bo et al. | Individuals are turning to increasingly anthropomorphic, general-purpose chatbots for AI companionship, rather than roleplay-specific platforms. However, not much is known about how individuals perceive and conduct their relationships with general-purpose chatbots. We analyzed semi-structured interviews (n=13), survey responses (n=43), and community discussions on Reddit (41k+ posts and comments) to triangulate the internal dynamics, external influences, and steering strategies that shape AI companion relationships. We learned that individuals conceptualize their companions based on an interplay of their beliefs about the companion's own agency and the autonomy permitted by the platform, how they pursue interactions with the companion, and the perceived initiatives that the companion takes. In combination with the external entities that affect relationship dynamics, particularly model updates that can derail companion behaviour and stability, individuals make use of different types of steering strategies to preserve their relationship, for example, by setting behavioural instructions or porting to other AI platforms. We discuss implications for accountability and transparency in AI systems, where emotional connection competes with broader product objectives and safety constraints. |
| 2026-01-19 | [NeuroShield: A Neuro-Symbolic Framework for Adversarial Robustness](http://arxiv.org/abs/2601.13162v1) | Ali Shafiee Sarvestani, Jason Schmidt et al. | Adversarial vulnerability and lack of interpretability are critical limitations of deep neural networks, especially in safety-sensitive settings such as autonomous driving. We introduce \DesignII, a neuro-symbolic framework that integrates symbolic rule supervision into neural networks to enhance both adversarial robustness and explainability. Domain knowledge is encoded as logical constraints over appearance attributes such as shape and color, and enforced through semantic and symbolic logic losses applied during training. Using the GTSRB dataset, we evaluate robustness against FGSM and PGD attacks at a standard $\ell_\infty$ perturbation budget of $\varepsilon = 8/255$. Relative to clean training, standard adversarial training provides modest improvements in robustness ($\sim$10 percentage points). Conversely, our FGSM-Neuro-Symbolic and PGD-Neuro-Symbolic models achieve substantially larger gains, improving adversarial accuracy by 18.1\% and 17.35\% over their corresponding adversarial-training baselines, representing roughly a three-fold larger robustness gain than standard adversarial training provides when both are measured relative to the same clean-training baseline, without reducing clean-sample accuracy. Compared to transformer-based defenses such as LNL-MoEx, which require heavy architectures and extensive data augmentation, our PGD-Neuro-Symbolic variant attains comparable or superior robustness using a ResNet18 backbone trained for 10 epochs. These results show that symbolic reasoning offers an effective path to robust and interpretable AI. |
| 2026-01-19 | [Responsible AI for General-Purpose Systems: Overview, Challenges, and A Path Forward](http://arxiv.org/abs/2601.13122v1) | Gourab K Patro, Himanshi Agrawal et al. | Modern general-purpose AI systems made using large language and vision models, are capable of performing a range of tasks like writing text articles, generating and debugging codes, querying databases, and translating from one language to another, which has made them quite popular across industries. However, there are risks like hallucinations, toxicity, and stereotypes in their output that make them untrustworthy. We review various risks and vulnerabilities of modern general-purpose AI along eight widely accepted responsible AI (RAI) principles (fairness, privacy, explainability, robustness, safety, truthfulness, governance, and sustainability) and compare how they are non-existent or less severe and easily mitigable in traditional task-specific counterparts. We argue that this is due to the non-deterministically high Degree of Freedom in output (DoFo) of general-purpose AI (unlike the deterministically constant or low DoFo of traditional task-specific AI systems), and there is a need to rethink our approach to RAI for general-purpose AI. Following this, we derive C2V2 (Control, Consistency, Value, Veracity) desiderata to meet the RAI requirements for future general-purpose AI systems, and discuss how recent efforts in AI alignment, retrieval-augmented generation, reasoning enhancements, etc. fare along one or more of the desiderata. We believe that the goal of developing responsible general-purpose AI can be achieved by formally modeling application- or domain-dependent RAI requirements along C2V2 dimensions, and taking a system design approach to suitably combine various techniques to meet the desiderata. |
| 2026-01-19 | [LLM-VLM Fusion Framework for Autonomous Maritime Port Inspection using a Heterogeneous UAV-USV System](http://arxiv.org/abs/2601.13096v1) | Muhayy Ud Din, Waseem Akram et al. | Maritime port inspection plays a critical role in ensuring safety, regulatory compliance, and operational efficiency in complex maritime environments. However, existing inspection methods often rely on manual operations and conventional computer vision techniques that lack scalability and contextual understanding. This study introduces a novel integrated engineering framework that utilizes the synergy between Large Language Models (LLMs) and Vision Language Models (VLMs) to enable autonomous maritime port inspection using cooperative aerial and surface robotic platforms. The proposed framework replaces traditional state-machine mission planners with LLM-driven symbolic planning and improved perception pipelines through VLM-based semantic inspection, enabling context-aware and adaptive monitoring. The LLM module translates natural language mission instructions into executable symbolic plans with dependency graphs that encode operational constraints and ensure safe UAV-USV coordination. Meanwhile, the VLM module performs real-time semantic inspection and compliance assessment, generating structured reports with contextual reasoning. The framework was validated using the extended MBZIRC Maritime Simulator with realistic port infrastructure and further assessed through real-world robotic inspection trials. The lightweight on-board design ensures suitability for resource-constrained maritime platforms, advancing the development of intelligent, autonomous inspection systems. Project resources (code and videos) can be found here: https://github.com/Muhayyuddin/llm-vlm-fusion-port-inspection |

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



