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
| 2025-09-15 | [Advancing Medical Artificial Intelligence Using a Century of Cases](http://arxiv.org/abs/2509.12194v1) | Thomas A. Buckley, Riccardo Conci et al. | BACKGROUND: For over a century, the New England Journal of Medicine Clinicopathological Conferences (CPCs) have tested the reasoning of expert physicians and, recently, artificial intelligence (AI). However, prior AI evaluations have focused on final diagnoses without addressing the multifaceted reasoning and presentation skills required of expert discussants.   METHODS: Using 7102 CPCs (1923-2025) and 1021 Image Challenges (2006-2025), we conducted extensive physician annotation and automated processing to create CPC-Bench, a physician-validated benchmark spanning 10 text-based and multimodal tasks, against which we evaluated leading large language models (LLMs). Then, we developed "Dr. CaBot," an AI discussant designed to produce written and slide-based video presentations using only the case presentation, modeling the role of the human expert in these cases.   RESULTS: When challenged with 377 contemporary CPCs, o3 (OpenAI) ranked the final diagnosis first in 60% of cases and within the top ten in 84% of cases, outperforming a 20-physician baseline; next-test selection accuracy reached 98%. Event-level physician annotations quantified AI diagnostic accuracy per unit of information. Performance was lower on literature search and image tasks; o3 and Gemini 2.5 Pro (Google) achieved 67% accuracy on image challenges. In blinded comparisons of CaBot vs. human expert-generated text, physicians misclassified the source of the differential in 46 of 62 (74%) of trials, and scored CaBot more favorably across quality dimensions. To promote research, we are releasing CaBot and CPC-Bench.   CONCLUSIONS: LLMs exceed physician performance on complex text-based differential diagnosis and convincingly emulate expert medical presentations, but image interpretation and literature retrieval remain weaker. CPC-Bench and CaBot may enable transparent and continued tracking of progress in medical AI. |
| 2025-09-15 | [A Converse Control Lyapunov Theorem for Joint Safety and Stability](http://arxiv.org/abs/2509.12182v1) | Thanin Quartz, Maxwell Fitzsimmons et al. | We show that the existence of a strictly compatible pair of control Lyapunov and control barrier functions is equivalent to the existence of a single smooth Lyapunov function that certifies both asymptotic stability and safety. This characterization complements existing literature on converse Lyapunov functions by establishing a partial differential equation (PDE) characterization with prescribed boundary conditions on the safe set, ensuring that the safe set is exactly certified by this Lyapunov function. The result also implies that if a safety and stability specification cannot be certified by a single Lyapunov function, then any pair of control Lyapunov and control barrier functions necessarily leads to a conflict and cannot be satisfied simultaneously in a robust sense. |
| 2025-09-15 | [Co-Alignment: Rethinking Alignment as Bidirectional Human-AI Cognitive Adaptation](http://arxiv.org/abs/2509.12179v1) | Yubo Li, Weiyi Song | Current AI alignment through RLHF follows a single directional paradigm that AI conforms to human preferences while treating human cognition as fixed. We propose a shift to co-alignment through Bidirectional Cognitive Alignment (BiCA), where humans and AI mutually adapt. BiCA uses learnable protocols, representation mapping, and KL-budget constraints for controlled co-evolution. In collaborative navigation, BiCA achieved 85.5% success versus 70.3% baseline, with 230% better mutual adaptation and 332% better protocol convergence. Emergent protocols outperformed handcrafted ones by 84%, while bidirectional adaptation unexpectedly improved safety (+23% out-of-distribution robustness). The 46% synergy improvement demonstrates optimal collaboration exists at the intersection, not union, of human and AI capabilities, validating the shift from single-directional to co-alignment paradigms. |
| 2025-09-15 | [Co-Alignment: Rethinking Alignment as Bidirectional Human-AI Cognitive Adaptation](http://arxiv.org/abs/2509.12179v2) | Yubo Li, Weiyi Song | Current AI alignment through RLHF follows a single directional paradigm that AI conforms to human preferences while treating human cognition as fixed. We propose a shift to co-alignment through Bidirectional Cognitive Alignment (BiCA), where humans and AI mutually adapt. BiCA uses learnable protocols, representation mapping, and KL-budget constraints for controlled co-evolution. In collaborative navigation, BiCA achieved 85.5% success versus 70.3% baseline, with 230% better mutual adaptation and 332% better protocol convergence. Emergent protocols outperformed handcrafted ones by 84%, while bidirectional adaptation unexpectedly improved safety (+23% out-of-distribution robustness). The 46% synergy improvement demonstrates optimal collaboration exists at the intersection, not union, of human and AI capabilities, validating the shift from single-directional to co-alignment paradigms. |
| 2025-09-15 | [Worker Discretion Advised: Co-designing Risk Disclosure in Crowdsourced Responsible AI (RAI) Content Work](http://arxiv.org/abs/2509.12140v1) | Alice Qian, Ziqi Yang et al. | Responsible AI (RAI) content work, such as annotation, moderation, or red teaming for AI safety, often exposes crowd workers to potentially harmful content. While prior work has underscored the importance of communicating well-being risk to employed content moderators, designing effective disclosure mechanisms for crowd workers while balancing worker protection with the needs of task designers and platforms remains largely unexamined. To address this gap, we conducted co-design sessions with 29 task designers, workers, and platform representatives. We investigated task designer preferences for support in disclosing tasks, worker preferences for receiving risk disclosure warnings, and how platform stakeholders envision their role in shaping risk disclosure practices. We identify design tensions and map the sociotechnical tradeoffs that shape disclosure practices. We contribute design recommendations and feature concepts for risk disclosure mechanisms in the context of RAI content work. |
| 2025-09-15 | [Control Analysis and Design for Autonomous Vehicles Subject to Imperfect AI-Based Perception](http://arxiv.org/abs/2509.12137v1) | Tao Yan, Zheyu Zhang et al. | Safety is a critical concern in autonomous vehicle (AV) systems, especially when AI-based sensing and perception modules are involved. However, due to the black box nature of AI algorithms, it makes closed-loop analysis and synthesis particularly challenging, for example, establishing closed-loop stability and ensuring performance, while they are fundamental to AV safety. To approach this difficulty, this paper aims to develop new modeling, analysis, and synthesis tools for AI-based AVs. Inspired by recent developments in perception error models (PEMs), the focus is shifted from directly modeling AI-based perception processes to characterizing the perception errors they produce. Two key classes of AI-induced perception errors are considered: misdetection and measurement noise. These error patterns are modeled using continuous-time Markov chains and Wiener processes, respectively. By means of that, a PEM-augmented driving model is proposed, with which we are able to establish the closed-loop stability for a class of AI-driven AV systems via stochastic calculus. Furthermore, a performance-guaranteed output feedback control synthesis method is presented, which ensures both stability and satisfactory performance. The method is formulated as a convex optimization problem, allowing for efficient numerical solutions. The results are then applied to an adaptive cruise control (ACC) scenario, demonstrating their effectiveness and robustness despite the corrupted and misleading perception. |
| 2025-09-15 | [XplaiNLP at CheckThat! 2025: Multilingual Subjectivity Detection with Finetuned Transformers and Prompt-Based Inference with Large Language Models](http://arxiv.org/abs/2509.12130v1) | Ariana Sahitaj, Jiaao Li et al. | This notebook reports the XplaiNLP submission to the CheckThat! 2025 shared task on multilingual subjectivity detection. We evaluate two approaches: (1) supervised fine-tuning of transformer encoders, EuroBERT, XLM-RoBERTa, and German-BERT, on monolingual and machine-translated training data; and (2) zero-shot prompting using two LLMs: o3-mini for Annotation (rule-based labelling) and gpt-4.1-mini for DoubleDown (contrastive rewriting) and Perspective (comparative reasoning). The Annotation Approach achieves 1st place in the Italian monolingual subtask with an F_1 score of 0.8104, outperforming the baseline of 0.6941. In the Romanian zero-shot setting, the fine-tuned XLM-RoBERTa model obtains an F_1 score of 0.7917, ranking 3rd and exceeding the baseline of 0.6461. The same model also performs reliably in the multilingual task and improves over the baseline in Greek. For German, a German-BERT model fine-tuned on translated training data from typologically related languages yields competitive performance over the baseline. In contrast, performance in the Ukrainian and Polish zero-shot settings falls slightly below the respective baselines, reflecting the challenge of generalization in low-resource cross-lingual scenarios. |
| 2025-09-15 | [RailSafeNet: Visual Scene Understanding for Tram Safety](http://arxiv.org/abs/2509.12125v1) | Ing. Ondrej Valach, Ing. Ivan Gruber | Tram-human interaction safety is an important challenge, given that trams frequently operate in densely populated areas, where collisions can range from minor injuries to fatal outcomes. This paper addresses the issue from the perspective of designing a solution leveraging digital image processing, deep learning, and artificial intelligence to improve the safety of pedestrians, drivers, cyclists, pets, and tram passengers. We present RailSafeNet, a real-time framework that fuses semantic segmentation, object detection and a rule-based Distance Assessor to highlight track intrusions. Using only monocular video, the system identifies rails, localises nearby objects and classifies their risk by comparing projected distances with the standard 1435mm rail gauge. Experiments on the diverse RailSem19 dataset show that a class-filtered SegFormer B3 model achieves 65% intersection-over-union (IoU), while a fine-tuned YOLOv8 attains 75.6% mean average precision (mAP) calculated at an intersection over union (IoU) threshold of 0.50. RailSafeNet therefore delivers accurate, annotation-light scene understanding that can warn drivers before dangerous situations escalate. Code available at https://github.com/oValach/RailSafeNet. |
| 2025-09-15 | [RailSafeNet: Visual Scene Understanding for Tram Safety](http://arxiv.org/abs/2509.12125v2) | Ondřej Valach, Ivan Gruber | Tram-human interaction safety is an important challenge, given that trams frequently operate in densely populated areas, where collisions can range from minor injuries to fatal outcomes. This paper addresses the issue from the perspective of designing a solution leveraging digital image processing, deep learning, and artificial intelligence to improve the safety of pedestrians, drivers, cyclists, pets, and tram passengers. We present RailSafeNet, a real-time framework that fuses semantic segmentation, object detection and a rule-based Distance Assessor to highlight track intrusions. Using only monocular video, the system identifies rails, localises nearby objects and classifies their risk by comparing projected distances with the standard 1435mm rail gauge. Experiments on the diverse RailSem19 dataset show that a class-filtered SegFormer B3 model achieves 65% intersection-over-union (IoU), while a fine-tuned YOLOv8 attains 75.6% mean average precision (mAP) calculated at an intersection over union (IoU) threshold of 0.50. RailSafeNet therefore delivers accurate, annotation-light scene understanding that can warn drivers before dangerous situations escalate. Code available at https://github.com/oValach/RailSafeNet. |
| 2025-09-15 | [In-domain SSL pre-training and streaming ASR](http://arxiv.org/abs/2509.12101v1) | Jarod Duret, Salima Mdhaffar et al. | In this study, we investigate the benefits of domain-specific self-supervised pre-training for both offline and streaming ASR in Air Traffic Control (ATC) environments. We train BEST-RQ models on 4.5k hours of unlabeled ATC data, then fine-tune on a smaller supervised ATC set. To enable real-time processing, we propose using chunked attention and dynamic convolutions, ensuring low-latency inference. We compare these in-domain SSL models against state-of-the-art, general-purpose speech encoders such as w2v-BERT 2.0 and HuBERT. Results show that domain-adapted pre-training substantially improves performance on standard ATC benchmarks, significantly reducing word error rates when compared to models trained on broad speech corpora. Furthermore, the proposed streaming approach further improves word error rate under tighter latency constraints, making it particularly suitable for safety-critical aviation applications. These findings highlight that specializing SSL representations for ATC data is a practical path toward more accurate and efficient ASR systems in real-world operational settings. |
| 2025-09-15 | [Compositional shield synthesis for safe reinforcement learning in partial observability](http://arxiv.org/abs/2509.12085v1) | Steven Carr, Georgios Bakirtzis et al. | Agents controlled by the output of reinforcement learning (RL) algorithms often transition to unsafe states, particularly in uncertain and partially observable environments. Partially observable Markov decision processes (POMDPs) provide a natural setting for studying such scenarios with limited sensing. Shields filter undesirable actions to ensure safe RL by preserving safety requirements in the agents' policy. However, synthesizing holistic shields is computationally expensive in complex deployment scenarios. We propose the compositional synthesis of shields by modeling safety requirements by parts, thereby improving scalability. In particular, problem formulations in the form of POMDPs using RL algorithms illustrate that an RL agent equipped with the resulting compositional shielding, beyond being safe, converges to higher values of expected reward. By using subproblem formulations, we preserve and improve the ability of shielded agents to require fewer training episodes than unshielded agents, especially in sparse-reward settings. Concretely, we find that compositional shield synthesis allows an RL agent to remain safe in environments two orders of magnitude larger than other state-of-the-art model-based approaches. |
| 2025-09-15 | [When Safe Unimodal Inputs Collide: Optimizing Reasoning Chains for Cross-Modal Safety in Multimodal Large Language Models](http://arxiv.org/abs/2509.12060v1) | Wei Cai, Shujuan Liu et al. | Multimodal Large Language Models (MLLMs) are susceptible to the implicit reasoning risk, wherein innocuous unimodal inputs synergistically assemble into risky multimodal data that produce harmful outputs. We attribute this vulnerability to the difficulty of MLLMs maintaining safety alignment through long-chain reasoning. To address this issue, we introduce Safe-Semantics-but-Unsafe-Interpretation (SSUI), the first dataset featuring interpretable reasoning paths tailored for such a cross-modal challenge. A novel training framework, Safety-aware Reasoning Path Optimization (SRPO), is also designed based on the SSUI dataset to align the MLLM's internal reasoning process with human safety values. Experimental results show that our SRPO-trained models achieve state-of-the-art results on key safety benchmarks, including the proposed Reasoning Path Benchmark (RSBench), significantly outperforming both open-source and top-tier commercial MLLMs. |
| 2025-09-15 | [When Safe Unimodal Inputs Collide: Optimizing Reasoning Chains for Cross-Modal Safety in Multimodal Large Language Models](http://arxiv.org/abs/2509.12060v2) | Wei Cai, Shujuan Liu et al. | Multimodal Large Language Models (MLLMs) are susceptible to the implicit reasoning risk, wherein innocuous unimodal inputs synergistically assemble into risky multimodal data that produce harmful outputs. We attribute this vulnerability to the difficulty of MLLMs maintaining safety alignment through long-chain reasoning. To address this issue, we introduce Safe-Semantics-but-Unsafe-Interpretation (SSUI), the first dataset featuring interpretable reasoning paths tailored for such a cross-modal challenge. A novel training framework, Safety-aware Reasoning Path Optimization (SRPO), is also designed based on the SSUI dataset to align the MLLM's internal reasoning process with human safety values. Experimental results show that our SRPO-trained models achieve state-of-the-art results on key safety benchmarks, including the proposed Reasoning Path Benchmark (RSBench), significantly outperforming both open-source and top-tier commercial MLLMs. |
| 2025-09-15 | [Travel Time and Weather-Aware Traffic Forecasting in a Conformal Graph Neural Network Framework](http://arxiv.org/abs/2509.12043v1) | Mayur Patil, Qadeer Ahmed et al. | Traffic flow forecasting is essential for managing congestion, improving safety, and optimizing various transportation systems. However, it remains a prevailing challenge due to the stochastic nature of urban traffic and environmental factors. Better predictions require models capable of accommodating the traffic variability influenced by multiple dynamic and complex interdependent factors. In this work, we propose a Graph Neural Network (GNN) framework to address the stochasticity by leveraging adaptive adjacency matrices using log-normal distributions and Coefficient of Variation (CV) values to reflect real-world travel time variability. Additionally, weather factors such as temperature, wind speed, and precipitation adjust edge weights and enable GNN to capture evolving spatio-temporal dependencies across traffic stations. This enhancement over the static adjacency matrix allows the model to adapt effectively to traffic stochasticity and changing environmental conditions. Furthermore, we utilize the Adaptive Conformal Prediction (ACP) framework to provide reliable uncertainty quantification, achieving target coverage while maintaining acceptable prediction intervals. Experimental results demonstrate that the proposed model, in comparison with baseline methods, showed better prediction accuracy and uncertainty bounds. We, then, validate this method by constructing traffic scenarios in SUMO and applying Monte-Carlo simulation to derive a travel time distribution for a Vehicle Under Test (VUT) to reflect real-world variability. The simulated mean travel time of the VUT falls within the intervals defined by INRIX historical data, verifying the model's robustness. |
| 2025-09-15 | [Bootstrapping Liquidity in BTC-Denominated Prediction Markets](http://arxiv.org/abs/2509.11990v1) | Fedor Shabashev | Prediction markets have gained adoption as on-chain mechanisms for aggregating information, with platforms such as Polymarket demonstrating demand for stablecoin-denominated markets. However, denominating in non-interest-bearing stablecoins introduces inefficiencies: participants face opportunity costs relative to the fiat risk-free rate, and Bitcoin holders in particular lose exposure to BTC appreciation when converting into stablecoins. This paper explores the case for prediction markets denominated in Bitcoin, treating BTC as a deflationary settlement asset analogous to gold under the classical gold standard. We analyse three methods of supplying liquidity to a newly created BTC-denominated prediction market: cross-market making against existing stablecoin venues, automated market making, and DeFi-based redirection of user trades. For each approach we evaluate execution mechanics, risks (slippage, exchange-rate risk, and liquidation risk), and capital efficiency. Our analysis shows that cross-market making provides the most user-friendly risk profile, though it requires active professional makers or platform-subsidised liquidity. DeFi redirection offers rapid bootstrapping and reuse of existing USDC liquidity, but exposes users to liquidation thresholds and exchange-rate volatility, reducing capital efficiency. Automated market making is simple to deploy but capital-inefficient and exposes liquidity providers to permanent loss. The results suggest that BTC-denominated prediction markets are feasible, but their success depends critically on the choice of liquidity provisioning mechanism and the trade-off between user safety and deployment convenience. |
| 2025-09-15 | [BREA-Depth: Bronchoscopy Realistic Airway-geometric Depth Estimation](http://arxiv.org/abs/2509.11885v1) | Francis Xiatian Zhang, Emile Mackute et al. | Monocular depth estimation in bronchoscopy can significantly improve real-time navigation accuracy and enhance the safety of interventions in complex, branching airways. Recent advances in depth foundation models have shown promise for endoscopic scenarios, yet these models often lack anatomical awareness in bronchoscopy, overfitting to local textures rather than capturing the global airway structure, particularly under ambiguous depth cues and poor lighting. To address this, we propose Brea-Depth, a novel framework that integrates airway-specific geometric priors into foundation model adaptation for bronchoscopic depth estimation. Our method introduces a depth-aware CycleGAN, refining the translation between real bronchoscopic images and airway geometries from anatomical data, effectively bridging the domain gap. In addition, we introduce an airway structure awareness loss to enforce depth consistency within the airway lumen while preserving smooth transitions and structural integrity. By incorporating anatomical priors, Brea-Depth enhances model generalization and yields more robust, accurate 3D airway reconstructions. To assess anatomical realism, we introduce Airway Depth Structure Evaluation, a new metric for structural consistency. We validate BREA-Depth on a collected ex vivo human lung dataset and an open bronchoscopic dataset, where it outperforms existing methods in anatomical depth preservation. |
| 2025-09-15 | [Letter of Intent: 100m Atom Interferometer Experiment at CERN](http://arxiv.org/abs/2509.11867v1) | Charles Baynham, Andrea Bertoldi et al. | We propose an O(100)m Atom Interferometer (AI) experiment to be installed against a wall of the PX46 access shaft to the LHC. This experiment would probe unexplored ranges of the possible couplings of bosonic ultralight dark matter (ULDM) to atomic constituents and undertake a pioneering search for gravitational waves (GWs) at frequencies intermediate between those to which existing and planned experiments are sensitive, among other fundamental physics studies. A conceptual feasibility study showed that this AI experiment could be isolated from the LHC by installing a shielding wall in the TX46 gallery, and surveyed issues related to the proximity of the LHC machine, finding no technical obstacles. A detailed technical implementation study has shown that the preparatory civil-engineering work, installation of bespoke radiation shielding, deployment of access-control systems and safety alarms, and installation of an elevator platform could be carried out during LS3, allowing installation and operation of the detector to proceed during Run 4 without impacting HL-LHC operation. These studies have established that PX46 is a uniquely promising location for an AI experiment. We foresee that, if the CERN management encourages this Letter of Intent, a significant fraction of the Terrestrial Very Long Baseline Atom Interferometer (TVLBAI) Proto-Collaboration may wish to contribute to such an AI experiment. |
| 2025-09-15 | [Letter of Intent: AICE - 100m Atom Interferometer Experiment at CERN](http://arxiv.org/abs/2509.11867v2) | Charles Baynham, Andrea Bertoldi et al. | We propose an O(100)m Atom Interferometer (AI) experiment - AICE - to be installed against a wall of the PX46 access shaft to the LHC. This experiment would probe unexplored ranges of the possible couplings of bosonic ultralight dark matter (ULDM) to atomic constituents and undertake a pioneering search for gravitational waves (GWs) at frequencies intermediate between those to which existing and planned experiments are sensitive, among other fundamental physics studies. A conceptual feasibility study showed that this AI experiment could be isolated from the LHC by installing a shielding wall in the TX46 gallery, and surveyed issues related to the proximity of the LHC machine, finding no technical obstacles. A detailed technical implementation study has shown that the preparatory civil-engineering work, installation of bespoke radiation shielding, deployment of access-control systems and safety alarms, and installation of an elevator platform could be carried out during LS3, allowing installation and operation of the AICE detector to proceed during Run 4 without impacting HL-LHC operation. These studies have established that PX46 is a uniquely promising location for an AI experiment. We foresee that, if the CERN management encourages this Letter of Intent, a significant fraction of the Terrestrial Very Long Baseline Atom Interferometer (TVLBAI) Proto-Collaboration may wish to contribute to AICE. |
| 2025-09-15 | [NeuroStrike: Neuron-Level Attacks on Aligned LLMs](http://arxiv.org/abs/2509.11864v1) | Lichao Wu, Sasha Behrouzi et al. | Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family. |
| 2025-09-15 | [Probabilistic Robustness Analysis in High Dimensional Space: Application to Semantic Segmentation Network](http://arxiv.org/abs/2509.11838v1) | Navid Hashemi, Samuel Sasaki et al. | Semantic segmentation networks (SSNs) play a critical role in domains such as medical imaging, autonomous driving, and environmental monitoring, where safety hinges on reliable model behavior under uncertainty. Yet, existing probabilistic verification approaches struggle to scale with the complexity and dimensionality of modern segmentation tasks, often yielding guarantees that are too conservative to be practical. We introduce a probabilistic verification framework that is both architecture-agnostic and scalable to high-dimensional outputs. Our approach combines sampling-based reachability analysis with conformal inference (CI) to deliver provable guarantees while avoiding the excessive conservatism of prior methods. To counteract CI's limitations in high-dimensional settings, we propose novel strategies that reduce conservatism without compromising rigor. Empirical evaluation on large-scale segmentation models across CamVid, OCTA-500, Lung Segmentation, and Cityscapes demonstrates that our method provides reliable safety guarantees while substantially tightening bounds compared to SOTA. We also provide a toolbox implementing this technique, available on Github. |

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



