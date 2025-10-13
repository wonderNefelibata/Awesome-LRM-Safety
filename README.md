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
| 2025-10-10 | [Autonomous Soft Robotic Guidewire Navigation via Imitation Learning](http://arxiv.org/abs/2510.09497v1) | Noah Barnes, Ji Woong Kim et al. | In endovascular surgery, endovascular interventionists push a thin tube called a catheter, guided by a thin wire to a treatment site inside the patient's blood vessels to treat various conditions such as blood clots, aneurysms, and malformations. Guidewires with robotic tips can enhance maneuverability, but they present challenges in modeling and control. Automation of soft robotic guidewire navigation has the potential to overcome these challenges, increasing the precision and safety of endovascular navigation. In other surgical domains, end-to-end imitation learning has shown promising results. Thus, we develop a transformer-based imitation learning framework with goal conditioning, relative action outputs, and automatic contrast dye injections to enable generalizable soft robotic guidewire navigation in an aneurysm targeting task. We train the model on 36 different modular bifurcated geometries, generating 647 total demonstrations under simulated fluoroscopy, and evaluate it on three previously unseen vascular geometries. The model can autonomously drive the tip of the robot to the aneurysm location with a success rate of 83% on the unseen geometries, outperforming several baselines. In addition, we present ablation and baseline studies to evaluate the effectiveness of each design and data collection choice. Project website: https://softrobotnavigation.github.io/ |
| 2025-10-10 | [Multimodal Policy Internalization for Conversational Agents](http://arxiv.org/abs/2510.09474v1) | Zhenhailong Wang, Jiateng Liu et al. | Modern conversational agents like ChatGPT and Alexa+ rely on predefined policies specifying metadata, response styles, and tool-usage rules. As these LLM-based systems expand to support diverse business and user queries, such policies, often implemented as in-context prompts, are becoming increasingly complex and lengthy, making faithful adherence difficult and imposing large fixed computational costs. With the rise of multimodal agents, policies that govern visual and multimodal behaviors are critical but remain understudied. Prior prompt-compression work mainly shortens task templates and demonstrations, while existing policy-alignment studies focus only on text-based safety rules. We introduce Multimodal Policy Internalization (MPI), a new task that internalizes reasoning-intensive multimodal policies into model parameters, enabling stronger policy-following without including the policy during inference. MPI poses unique data and algorithmic challenges. We build two datasets spanning synthetic and real-world decision-making and tool-using tasks and propose TriMPI, a three-stage training framework. TriMPI first injects policy knowledge via continual pretraining, then performs supervised finetuning, and finally applies PolicyRollout, a GRPO-style reinforcement learning extension that augments rollouts with policy-aware responses for grounded exploration. TriMPI achieves notable gains in end-to-end accuracy, generalization, and robustness to forgetting. As the first work on multimodal policy internalization, we provide datasets, training recipes, and comprehensive evaluations to foster future research. Project page: https://mikewangwzhl.github.io/TriMPI. |
| 2025-10-10 | [Getting Your Indices in a Row: Full-Text Search for LLM Training Data for Real World](http://arxiv.org/abs/2510.09471v1) | Ines Altemir Marinas, Anastasiia Kucherenko et al. | The performance of Large Language Models (LLMs) is determined by their training data. Despite the proliferation of open-weight LLMs, access to LLM training data has remained limited. Even for fully open LLMs, the scale of the data makes it all but inscrutable to the general scientific community, despite potentially containing critical data scraped from the internet.   In this paper, we present the full-text indexing pipeline for the Apertus LLM training data. Leveraging Elasticsearch parallel indices and the Alps infrastructure, a state-of-the-art, highly energy-efficient arm64 supercluster, we were able to index 8.6T tokens out of 15.2T used to train the Apertus LLM family, creating both a critical LLM safety tool and effectively an offline, curated, open web search engine. Our contribution is threefold. First, we demonstrate that Elasticsearch can be successfully ported onto next-generation arm64-based infrastructure. Second, we demonstrate that full-text indexing at the scale of modern LLM training datasets and the entire open web is feasible and accessible. Finally, we demonstrate that such indices can be used to ensure previously inaccessible jailbreak-agnostic LLM safety.   We hope that our findings will be useful to other teams attempting large-scale data indexing and facilitate the general transition towards greener computation. |
| 2025-10-10 | [Failure Prediction at Runtime for Generative Robot Policies](http://arxiv.org/abs/2510.09459v1) | Ralf Römer, Adrian Kobras et al. | Imitation learning (IL) with generative models, such as diffusion and flow matching, has enabled robots to perform complex, long-horizon tasks. However, distribution shifts from unseen environments or compounding action errors can still cause unpredictable and unsafe behavior, leading to task failure. Early failure prediction during runtime is therefore essential for deploying robots in human-centered and safety-critical environments. We propose FIPER, a general framework for Failure Prediction at Runtime for generative IL policies that does not require failure data. FIPER identifies two key indicators of impending failure: (i) out-of-distribution (OOD) observations detected via random network distillation in the policy's embedding space, and (ii) high uncertainty in generated actions measured by a novel action-chunk entropy score. Both failure prediction scores are calibrated using a small set of successful rollouts via conformal prediction. A failure alarm is triggered when both indicators, aggregated over short time windows, exceed their thresholds. We evaluate FIPER across five simulation and real-world environments involving diverse failure modes. Our results demonstrate that FIPER better distinguishes actual failures from benign OOD situations and predicts failures more accurately and earlier than existing methods. We thus consider this work an important step towards more interpretable and safer generative robot policies. Code, data and videos are available at https://tum-lsy.github.io/fiper_website. |
| 2025-10-10 | [Demystifying and Navigating AI Ethics in Power Electronics](http://arxiv.org/abs/2510.09439v1) | Fanfan Lin, Peter Wilson et al. | Artificial intelligence (AI) is rapidly transforming power electronics, with AI-related publications in IEEE Power Electronics Society selected journals increasing more than fourfold from 2020 to 2025. However, the ethical dimensions of this transformation have received limited attention. This article underscores the urgent need for an ethical framework to guide responsible AI integration in power electronics, not only to prevent AI-related incidents but also to comply with legal and regulatory responsibilities. In this context, this article identifies four core pillars of AI ethics in power electronics: Security & Safety, Explainability & Transparency, Energy Sustainability, and Evolving Roles of Engineers. Each pillar is supported by practical and actionable insights to ensure that ethical principles are embedded in algorithm design, system deployment, and workforce development. The authors advocate for power electronics engineers to lead the ethical discourse, given their deep technical understanding of both AI systems and power conversion technologies. The paper concludes by calling on the IEEE Power Electronics Society to spearhead the establishment of ethical standards and best practices that ensure AI innovations are not only technically advanced but also trustworthy, safe, and sustainable. |
| 2025-10-10 | [Domain-Adapted Pre-trained Language Models for Implicit Information Extraction in Crash Narratives](http://arxiv.org/abs/2510.09434v1) | Xixi Wang, Jordanka Kovaceva et al. | Free-text crash narratives recorded in real-world crash databases have been shown to play a significant role in improving traffic safety. However, large-scale analyses remain difficult to implement as there are no documented tools that can batch process the unstructured, non standardized text content written by various authors with diverse experience and attention to detail. In recent years, Transformer-based pre-trained language models (PLMs), such as Bidirectional Encoder Representations from Transformers (BERT) and large language models (LLMs), have demonstrated strong capabilities across various natural language processing tasks. These models can extract explicit facts from crash narratives, but their performance declines on inference-heavy tasks in, for example, Crash Type identification, which can involve nearly 100 categories. Moreover, relying on closed LLMs through external APIs raises privacy concerns for sensitive crash data. Additionally, these black-box tools often underperform due to limited domain knowledge. Motivated by these challenges, we study whether compact open-source PLMs can support reasoning-intensive extraction from crash narratives. We target two challenging objectives: 1) identifying the Manner of Collision for a crash, and 2) Crash Type for each vehicle involved in the crash event from real-world crash narratives. To bridge domain gaps, we apply fine-tuning techniques to inject task-specific knowledge to LLMs with Low-Rank Adaption (LoRA) and BERT. Experiments on the authoritative real-world dataset Crash Investigation Sampling System (CISS) demonstrate that our fine-tuned compact models outperform strong closed LLMs, such as GPT-4o, while requiring only minimal training resources. Further analysis reveals that the fine-tuned PLMs can capture richer narrative details and even correct some mislabeled annotations in the dataset. |
| 2025-10-10 | [BLINK-Twice: You see, but do you observe? A Reasoning Benchmark on Visual Perception](http://arxiv.org/abs/2510.09361v1) | Junyan Ye, Dongzhi Jiang et al. | Recently, Multimodal Large Language Models (MLLMs) have made rapid progress, particularly in enhancing their reasoning capabilities. However, existing reasoning benchmarks still primarily assess language-based reasoning, often treating visual input as replaceable context. To address this gap, we introduce BLINK-Twice, a vision-centric reasoning benchmark grounded in challenging perceptual tasks. Instead of relying on external knowledge, our tasks require models to reason from visual content alone, shifting the focus from language-based to image-grounded reasoning. Compared to prior perception benchmarks, it moves beyond shallow perception ("see") and requires fine-grained observation and analytical reasoning ("observe"). BLINK-Twice integrates three core components: seven types of visual challenges for testing visual reasoning, natural adversarial image pairs that enforce reliance on visual content, and annotated reasoning chains for fine-grained evaluation of the reasoning process rather than final answers alone. We evaluate 20 leading MLLMs, including 12 foundation models and 8 reasoning-enhanced models. BLINK-Twice poses a significant challenge to current models. While existing reasoning strategies in the language space-such as chain-of-thought or self-criticism can improve performance, they often result in unstable and redundant reasoning. We observe that repeated image observation improves performance across models, and active visual interaction, as demonstrated by models like o3, highlights the need for a new paradigm for vision reasoning. The dataset is publicly available at https://github.com/PicoTrex/BLINK-Twice |
| 2025-10-10 | [Safety Game: Balancing Safe and Informative Conversations with Blackbox Agentic AI using LP Solvers](http://arxiv.org/abs/2510.09330v1) | Tuan Nguyen, Long Tran-Thanh | Ensuring that large language models (LLMs) comply with safety requirements is a central challenge in AI deployment. Existing alignment approaches primarily operate during training, such as through fine-tuning or reinforcement learning from human feedback, but these methods are costly and inflexible, requiring retraining whenever new requirements arise. Recent efforts toward inference-time alignment mitigate some of these limitations but still assume access to model internals, which is impractical, and not suitable for third party stakeholders who do not have access to the models. In this work, we propose a model-independent, black-box framework for safety alignment that does not require retraining or access to the underlying LLM architecture. As a proof of concept, we address the problem of trading off between generating safe but uninformative answers versus helpful yet potentially risky ones. We formulate this dilemma as a two-player zero-sum game whose minimax equilibrium captures the optimal balance between safety and helpfulness. LLM agents operationalize this framework by leveraging a linear programming solver at inference time to compute equilibrium strategies. Our results demonstrate the feasibility of black-box safety alignment, offering a scalable and accessible pathway for stakeholders, including smaller organizations and entities in resource-constrained settings, to enforce safety across rapidly evolving LLM ecosystems. |
| 2025-10-10 | [Reliability Sensitivity with Response Gradient](http://arxiv.org/abs/2510.09315v1) | Siu-Kui Au, Zi-Jun Cao | Engineering risk is concerned with the likelihood of failure and the scenarios when it occurs. The sensitivity of failure probability to change in system parameters is relevant to risk-informed decision making. Computing sensitivity is at least one level more difficult than the probability itself, which is already challenged by a large number of input random variables, rare events and implicit nonlinear `black-box' response. Finite difference with Monte Carlo probability estimates is spurious, requiring the number of samples to grow with the reciprocal of step size to suppress estimation variance. Many existing works gain efficiency by exploiting a specific class of input variables, sensitivity parameters, or response in its exact or surrogate form. For general systems, this work presents a theory and associated Monte Carlo strategy for computing sensitivity using response values and gradients with respect to sensitivity parameters. It is shown that the sensitivity at a given response threshold can be expressed via the expectation of response gradient conditional on the threshold. Determining the expectation requires conditioning on the threshold that is a zero-probability event, but it can be resolved by the concept of kernel smoothing. The proposed method offers sensitivity estimates for all response thresholds generated in a single Monte Carlo run. It is investigated in a number of examples featuring sensitivity parameters of different nature. As response gradient becomes increasingly available, it is hoped that this work can provide the basis for embedding sensitivity calculations with reliability in the same Monte Carlo run. |
| 2025-10-10 | [CapGeo: A Caption-Assisted Approach to Geometric Reasoning](http://arxiv.org/abs/2510.09302v1) | Yuying Li, Siyi Qian et al. | Geometric reasoning remains a core challenge for Multimodal Large Language Models (MLLMs). Even the most advanced closed-source systems, such as GPT-O3 and Gemini-2.5-Pro, still struggle to solve geometry problems reliably, despite exhibiting strong textual reasoning abilities on tasks like the International Mathematical Olympiad (IMO). This gap suggests that the bottleneck lies in understanding geometric diagrams rather than reasoning itself. Since geometric figures can often be faithfully described in concise textual form, converting visual content into captions offers a promising direction. Motivated by this insight, we introduce CapGeo, a caption-assisted reasoning framework that bridges visual and textual modalities. Experiments show substantial improvements when models are equipped with captions: Qwen2.5-VL-72B improves from 8.6% (vision-only) to 59.0%, while Claude-Opus-4 rises from 44.8% to 73.0%. To systematically evaluate and identify high-quality geometric captioning models, we further propose CapGeo-Bench, a dataset of 4,641 curated figure-caption pairs. Crucially, CapGeo-Bench incorporates a keypoint-based evaluation metric that correlates strongly with downstream CapGeo performance, enabling reliable assessment of geometric captioning ability. Together, our framework and benchmark highlight a new pathway toward advancing geometric reasoning in MLLMs. |
| 2025-10-10 | [Safety Analysis of eVTOL Operations based on STPA](http://arxiv.org/abs/2510.09283v1) | Mariat James Elizebeth, Shufeng Chen et al. | Electric Vertical Take-Off and Landing (eVTOL) aircraft are expected to be quieter and more cost-effective than helicopters, offering major economic and social benefits through improved connectivity. Their adoption will require new ground infrastructure and airspace redesign, introducing risks involving multiple stakeholders (Regulators, eVTOL operators, Air navigation service providers, Vertiport operators, OEMs, Pilots, etc.). To assess these risks for the UK airspace, systems-thinking based System Theoretic Process Analysis (STPA) was conducted. To manage the large number of Unsafe Control Actions (UCAs) and requirements generated due to the complexity of the analysis, a novel extension to STPA for the prioritization of results was applied. 317 UCAs were identified in total out of which 110 high-priority UCAs were analyzed (Step-4), resulting in 377 causal factors and 432 requirements. These were prioritized to produce a targeted list of 124 distinct high-priority requirements, 56 of which were identified as gaps in existing aviation regulations, policies, or procedures.. These highlight opportunities for regulatory updates in areas such as organizational performance, certification processes, training, collision avoidance, energy management, and automation. The findings provide regulators with safety considerations that could shape new or updated regulations, compliance methods, and guidance materials for the safe deployment of eVTOLs. |
| 2025-10-10 | [Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization](http://arxiv.org/abs/2510.09204v1) | Simon Idoko, Arun Kumar Singh | Centralized trajectory optimization in the joint space of multiple robots allows access to a larger feasible space that can result in smoother trajectories, especially while planning in tight spaces. Unfortunately, it is often computationally intractable beyond a very small swarm size. In this paper, we propose Flow-Opt, a learning-based approach towards improving the computational tractability of centralized multi-robot trajectory optimization. Specifically, we reduce the problem to first learning a generative model to sample different candidate trajectories and then using a learned Safety-Filter(SF) to ensure fast inference-time constraint satisfaction. We propose a flow-matching model with a diffusion transformer (DiT) augmented with permutation invariant robot position and map encoders as the generative model. We develop a custom solver for our SF and equip it with a neural network that predicts context-specific initialization. The initialization network is trained in a self-supervised manner, taking advantage of the differentiability of the SF solver. We advance the state-of-the-art in the following respects. First, we show that we can generate trajectories of tens of robots in cluttered environments in a few tens of milliseconds. This is several times faster than existing centralized optimization approaches. Moreover, our approach also generates smoother trajectories orders of magnitude faster than competing baselines based on diffusion models. Second, each component of our approach can be batched, allowing us to solve a few tens of problem instances in a fraction of a second. We believe this is a first such result; no existing approach provides such capabilities. Finally, our approach can generate a diverse set of trajectories between a given set of start and goal locations, which can capture different collision-avoidance behaviors. |
| 2025-10-10 | [Towards Safer and Understandable Driver Intention Prediction](http://arxiv.org/abs/2510.09200v1) | Mukilan Karuppasamy, Shankar Gangisetty et al. | Autonomous driving (AD) systems are becoming increasingly capable of handling complex tasks, mainly due to recent advances in deep learning and AI. As interactions between autonomous systems and humans increase, the interpretability of decision-making processes in driving systems becomes increasingly crucial for ensuring safe driving operations. Successful human-machine interaction requires understanding the underlying representations of the environment and the driving task, which remains a significant challenge in deep learning-based systems. To address this, we introduce the task of interpretability in maneuver prediction before they occur for driver safety, i.e., driver intent prediction (DIP), which plays a critical role in AD systems. To foster research in interpretable DIP, we curate the eXplainable Driving Action Anticipation Dataset (DAAD-X), a new multimodal, ego-centric video dataset to provide hierarchical, high-level textual explanations as causal reasoning for the driver's decisions. These explanations are derived from both the driver's eye-gaze and the ego-vehicle's perspective. Next, we propose Video Concept Bottleneck Model (VCBM), a framework that generates spatio-temporally coherent explanations inherently, without relying on post-hoc techniques. Finally, through extensive evaluations of the proposed VCBM on the DAAD-X dataset, we demonstrate that transformer-based models exhibit greater interpretability than conventional CNN-based models. Additionally, we introduce a multilabel t-SNE visualization technique to illustrate the disentanglement and causal correlation among multiple explanations. Our data, code and models are available at: https://mukil07.github.io/VCBM.github.io/ |
| 2025-10-10 | [TARO: Toward Semantically Rich Open-World Object Detection](http://arxiv.org/abs/2510.09173v1) | Yuchen Zhang, Yao Lu et al. | Modern object detectors are largely confined to a "closed-world" assumption, limiting them to a predefined set of classes and posing risks when encountering novel objects in real-world scenarios. While open-set detection methods aim to address this by identifying such instances as 'Unknown', this is often insufficient. Rather than treating all unknowns as a single class, assigning them more descriptive subcategories can enhance decision-making in safety-critical contexts. For example, identifying an object as an 'Unknown Animal' (requiring an urgent stop) versus 'Unknown Debris' (requiring a safe lane change) is far more useful than just 'Unknown' in autonomous driving. To bridge this gap, we introduce TARO, a novel detection framework that not only identifies unknown objects but also classifies them into coarse parent categories within a semantic hierarchy. TARO employs a unique architecture with a sparsemax-based head for modeling objectness, a hierarchy-guided relabeling component that provides auxiliary supervision, and a classification module that learns hierarchical relationships. Experiments show TARO can categorize up to 29.9% of unknowns into meaningful coarse classes, significantly reduce confusion between unknown and known classes, and achieve competitive performance in both unknown recall and known mAP. Code will be made available. |
| 2025-10-10 | [A Semantic Framework for Patient Digital Twins in Chronic Care](http://arxiv.org/abs/2510.09134v1) | Amal Elgammal, Bernd J. Krämer et al. | Personalized chronic care requires the integration of multimodal health data to enable precise, adaptive, and preventive decision-making. Yet most current digital twin (DT) applications remain organ-specific or tied to isolated data types, lacking a unified and privacy-preserving foundation. This paper introduces the Patient Medical Digital Twin (PMDT), an ontology-driven in silico patient framework that integrates physiological, psychosocial, behavioral, and genomic information into a coherent, extensible model. Implemented in OWL 2.0, the PMDT ensures semantic interoperability, supports automated reasoning, and enables reuse across diverse clinical contexts. Its ontology is structured around modular Blueprints (patient, disease and diagnosis, treatment and follow-up, trajectories, safety, pathways, and adverse events), formalized through dedicated conceptual views. These were iteratively refined and validated through expert workshops, questionnaires, and a pilot study in the EU H2020 QUALITOP project with real-world immunotherapy patients. Evaluation confirmed ontology coverage, reasoning correctness, usability, and GDPR compliance. Results demonstrate the PMDT's ability to unify heterogeneous data, operationalize competency questions, and support descriptive, predictive, and prescriptive analytics in a federated, privacy-preserving manner. By bridging gaps in data fragmentation and semantic standardization, the PMDT provides a validated foundation for next-generation digital health ecosystems, transforming chronic care toward proactive, continuously optimized, and equitable management. |
| 2025-10-10 | [DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel Translation](http://arxiv.org/abs/2510.09116v1) | Enze Zhang, Jiaying Wang et al. | Large language models (LLMs) have substantially advanced machine translation (MT), yet their effectiveness in translating web novels remains unclear. Existing benchmarks rely on surface-level metrics that fail to capture the distinctive traits of this genre. To address these gaps, we introduce DITING, the first comprehensive evaluation framework for web novel translation, assessing narrative and cultural fidelity across six dimensions: idiom translation, lexical ambiguity, terminology localization, tense consistency, zero-pronoun resolution, and cultural safety, supported by over 18K expert-annotated Chinese-English sentence pairs. We further propose AgentEval, a reasoning-driven multi-agent evaluation framework that simulates expert deliberation to assess translation quality beyond lexical overlap, achieving the highest correlation with human judgments among seven tested automatic metrics. To enable metric comparison, we develop MetricAlign, a meta-evaluation dataset of 300 sentence pairs annotated with error labels and scalar quality scores. Comprehensive evaluation of fourteen open, closed, and commercial models reveals that Chinese-trained LLMs surpass larger foreign counterparts, and that DeepSeek-V3 delivers the most faithful and stylistically coherent translations. Our work establishes a new paradigm for exploring LLM-based web novel translation and provides public resources to advance future research. |
| 2025-10-10 | [When a Robot is More Capable than a Human: Learning from Constrained Demonstrators](http://arxiv.org/abs/2510.09096v1) | Xinhu Li, Ayush Jain et al. | Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on https://sites.google.com/view/constrainedexpert . |
| 2025-10-10 | [Visual Anomaly Detection for Reliable Robotic Implantation of Flexible Microelectrode Array](http://arxiv.org/abs/2510.09071v1) | Yitong Chen, Xinyao Xu et al. | Flexible microelectrode (FME) implantation into brain cortex is challenging due to the deformable fiber-like structure of FME probe and the interaction with critical bio-tissue. To ensure reliability and safety, the implantation process should be monitored carefully. This paper develops an image-based anomaly detection framework based on the microscopic cameras of the robotic FME implantation system. The unified framework is utilized at four checkpoints to check the micro-needle, FME probe, hooking result, and implantation point, respectively. Exploiting the existing object localization results, the aligned regions of interest (ROIs) are extracted from raw image and input to a pretrained vision transformer (ViT). Considering the task specifications, we propose a progressive granularity patch feature sampling method to address the sensitivity-tolerance trade-off issue at different locations. Moreover, we select a part of feature channels with higher signal-to-noise ratios from the raw general ViT features, to provide better descriptors for each specific scene. The effectiveness of the proposed methods is validated with the image datasets collected from our implantation system. |
| 2025-10-10 | [Sensing, Detection and Localization for Low Altitude UAV: A RF-Based Framework via Multiple BSs Collaboration](http://arxiv.org/abs/2510.09055v1) | Tianhao Liang, Mu Jia et al. | The rapid growth of the low-altitude economy has resulted in a significant increase in the number of Low, slow, and small (LLS) unmanned aerial vehicles (UAVs), raising critical challenges for secure airspace management and reliable trajectory planning. To address this, this paper proposes a cooperative radio-frequency (RF) detection and localization framework that leverages existing cellular base stations. The proposed approach features a robust scheme for LSS target identification, integrating a cell averaging-constant false alarm rate (CA-CFAR) detector with a micro-Doppler signature (MDS) based recognition method. Multi-station measurements are fused through a grid-based probabilistic algorithm combined with clustering techniques, effectively mitigating ghost targets and improving localization accuracy in multi-UAV scenarios. Furthermore, the Cramer-Rao lower bound (CRLB) is derived as a performance benchmark and reinforcement learning (RL)-based optimization is employed to balance localization accuracy against station resource usage. Simulations demonstrate that increasing from one to multiple BSs reduces the positioning error to near the CRLB, while practical experiments further verify the framework's effectiveness. Furthermore, our RL-based optimization can find solutions that maintain high accuracy while minimizing resource usage, highlighting its potential as a scalable solution for ensuring airspace safety in the emerging low-altitude economy. |
| 2025-10-10 | [Alif: Advancing Urdu Large Language Models via Multilingual Synthetic Data Distillation](http://arxiv.org/abs/2510.09051v1) | Muhammad Ali Shafique, Kanwal Mehreen et al. | Developing a high-performing large language models (LLMs) for low-resource languages such as Urdu, present several challenges. These challenges include the scarcity of high-quality datasets, multilingual inconsistencies, and safety concerns. Existing multilingual LLMs often address these issues by translating large volumes of available data. However, such translations often lack quality and cultural nuance while also incurring significant costs for data curation and training. To address these issues, we propose Alif-1.0-8B-Instruct, a multilingual Urdu-English model, that tackles these challenges with a unique approach. We train the model on a high-quality, multilingual synthetic dataset (Urdu-Instruct), developed using a modified self-instruct technique. By using unique prompts and seed values for each task along with a global task pool, this dataset incorporates Urdu-native chain-of-thought based reasoning, bilingual translation, cultural relevance, and ethical safety alignments. This technique significantly enhances the comprehension of Alif-1.0-8B-Instruct model for Urdu-specific tasks. As a result, Alif-1.0-8B-Instruct, built upon the pretrained Llama-3.1-8B, demonstrates superior performance compared to Llama-3.1-8B-Instruct for Urdu specific-tasks. It also outperformed leading multilingual LLMs, including Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, and Cohere-Aya-Expanse-8B, all within a training budget of under $100. Our results demonstrate that high-performance and low-resource language LLMs can be developed efficiently and culturally aligned using our modified self-instruct approach. All datasets, models, and code are publicly available at: https://github.com/traversaal-ai/alif-urdu-llm. |

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



