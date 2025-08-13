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
| 2025-08-12 | [A Review On Safe Reinforcement Learning Using Lyapunov and Barrier Functions](http://arxiv.org/abs/2508.09128v1) | Dhruv S. Kushwaha, Zoleikha A. Biron | Reinforcement learning (RL) has proven to be particularly effective in solving complex decision-making problems for a wide range of applications. From a control theory perspective, RL can be considered as an adaptive optimal control scheme. Lyapunov and barrier functions are the most commonly used certificates to guarantee system stability for a proposed/derived controller and constraint satisfaction guarantees, respectively, in control theoretic approaches. However, compared to theoretical guarantees available in control theoretic methods, RL lacks closed-loop stability of a computed policy and constraint satisfaction guarantees. Safe reinforcement learning refers to a class of constrained problems where the constraint violations lead to partial or complete system failure. The goal of this review is to provide an overview of safe RL techniques using Lyapunov and barrier functions to guarantee this notion of safety discussed (stability of the system in terms of a computed policy and constraint satisfaction during training and deployment). The different approaches employed are discussed in detail along with their shortcomings and benefits to provide critique and possible future research directions. Key motivation for this review is to discuss current theoretical approaches for safety and stability guarantees in RL similar to control theoretic approaches using Lyapunov and barrier functions. The review provides proven potential and promising scope of providing safety guarantees for complex dynamical systems with operational constraints using model-based and model-free RL. |
| 2025-08-12 | [Deep Neural Network Calibration by Reducing Classifier Shift with Stochastic Masking](http://arxiv.org/abs/2508.09116v1) | Jiani Ni, He Zhao et al. | In recent years, deep neural networks (DNNs) have shown competitive results in many fields. Despite this success, they often suffer from poor calibration, especially in safety-critical scenarios such as autonomous driving and healthcare, where unreliable confidence estimates can lead to serious consequences. Recent studies have focused on improving calibration by modifying the classifier, yet such efforts remain limited. Moreover, most existing approaches overlook calibration errors caused by underconfidence, which can be equally detrimental. To address these challenges, we propose MaC-Cal, a novel mask-based classifier calibration method that leverages stochastic sparsity to enhance the alignment between confidence and accuracy. MaC-Cal adopts a two-stage training scheme with adaptive sparsity, dynamically adjusting mask retention rates based on the deviation between confidence and accuracy. Extensive experiments show that MaC-Cal achieves superior calibration performance and robustness under data corruption, offering a practical and effective solution for reliable confidence estimation in DNNs. |
| 2025-08-12 | [Link Prediction for Event Logs in the Process Industry](http://arxiv.org/abs/2508.09096v1) | Anastasia Zhukova, Thomas Walton et al. | Knowledge management (KM) is vital in the process industry for optimizing operations, ensuring safety, and enabling continuous improvement through effective use of operational data and past insights. A key challenge in this domain is the fragmented nature of event logs in shift books, where related records, e.g., entries documenting issues related to equipment or processes and the corresponding solutions, may remain disconnected. This fragmentation hinders the recommendation of previous solutions to the users. To address this problem, we investigate record linking (RL) as link prediction, commonly studied in graph-based machine learning, by framing it as a cross-document coreference resolution (CDCR) task enhanced with natural language inference (NLI) and semantic text similarity (STS) by shifting it into the causal inference (CI). We adapt CDCR, traditionally applied in the news domain, into an RL model to operate at the passage level, similar to NLI and STS, while accommodating the process industry's specific text formats, which contain unstructured text and structured record attributes. Our RL model outperformed the best versions of NLI- and STS-driven baselines by 28% (11.43 points) and 27% (11.21 points), respectively. Our work demonstrates how domain adaptation of the state-of-the-art CDCR models, enhanced with reasoning capabilities, can be effectively tailored to the process industry, improving data quality and connectivity in shift logs. |
| 2025-08-12 | [Dynamic Uncertainty-aware Multimodal Fusion for Outdoor Health Monitoring](http://arxiv.org/abs/2508.09085v1) | Zihan Fang, Zheng Lin et al. | Outdoor health monitoring is essential to detect early abnormal health status for safeguarding human health and safety. Conventional outdoor monitoring relies on static multimodal deep learning frameworks, which requires extensive data training from scratch and fails to capture subtle health status changes. Multimodal large language models (MLLMs) emerge as a promising alternative, utilizing only small datasets to fine-tune pre-trained information-rich models for enabling powerful health status monitoring. Unfortunately, MLLM-based outdoor health monitoring also faces significant challenges: I) sensor data contains input noise stemming from sensor data acquisition and fluctuation noise caused by sudden changes in physiological signals due to dynamic outdoor environments, thus degrading the training performance; ii) current transformer based MLLMs struggle to achieve robust multimodal fusion, as they lack a design for fusing the noisy modality; iii) modalities with varying noise levels hinder accurate recovery of missing data from fluctuating distributions. To combat these challenges, we propose an uncertainty-aware multimodal fusion framework, named DUAL-Health, for outdoor health monitoring in dynamic and noisy environments. First, to assess the impact of noise, we accurately quantify modality uncertainty caused by input and fluctuation noise with current and temporal features. Second, to empower efficient muitimodal fusion with low-quality modalities,we customize the fusion weight for each modality based on quantified and calibrated uncertainty. Third, to enhance data recovery from fluctuating noisy modalities, we align modality distributions within a common semantic space. Extensive experiments demonstrate that our DUAL-Health outperforms state-of-the-art baselines in detection accuracy and robustness. |
| 2025-08-12 | [VLM-3D:End-to-End Vision-Language Models for Open-World 3D Perception](http://arxiv.org/abs/2508.09061v1) | Fuhao Chang, Shuxin Li et al. | Open-set perception in complex traffic environments poses a critical challenge for autonomous driving systems, particularly in identifying previously unseen object categories, which is vital for ensuring safety. Visual Language Models (VLMs), with their rich world knowledge and strong semantic reasoning capabilities, offer new possibilities for addressing this task. However, existing approaches typically leverage VLMs to extract visual features and couple them with traditional object detectors, resulting in multi-stage error propagation that hinders perception accuracy. To overcome this limitation, we propose VLM-3D, the first end-to-end framework that enables VLMs to perform 3D geometric perception in autonomous driving scenarios. VLM-3D incorporates Low-Rank Adaptation (LoRA) to efficiently adapt VLMs to driving tasks with minimal computational overhead, and introduces a joint semantic-geometric loss design: token-level semantic loss is applied during early training to ensure stable convergence, while 3D IoU loss is introduced in later stages to refine the accuracy of 3D bounding box predictions. Evaluations on the nuScenes dataset demonstrate that the proposed joint semantic-geometric loss in VLM-3D leads to a 12.8% improvement in perception accuracy, fully validating the effectiveness and advancement of our method. |
| 2025-08-12 | [CVCM Track Circuits Pre-emptive Failure Diagnostics for Predictive Maintenance Using Deep Neural Networks](http://arxiv.org/abs/2508.09054v1) | Debdeep Mukherjee, Eduardo Di Santi et al. | Track circuits are critical for railway operations, acting as the main signalling sub-system to locate trains. Continuous Variable Current Modulation (CVCM) is one such technology. Like any field-deployed, safety-critical asset, it can fail, triggering cascading disruptions. Many failures originate as subtle anomalies that evolve over time, often not visually apparent in monitored signals. Conventional approaches, which rely on clear signal changes, struggle to detect them early. Early identification of failure types is essential to improve maintenance planning, minimising downtime and revenue loss. Leveraging deep neural networks, we propose a predictive maintenance framework that classifies anomalies well before they escalate into failures. Validated on 10 CVCM failure cases across different installations, the method is ISO-17359 compliant and outperforms conventional techniques, achieving 99.31% overall accuracy with detection within 1% of anomaly onset. Through conformal prediction, we provide uncertainty estimates, reaching 99% confidence with consistent coverage across classes. Given CVCMs global deployment, the approach is scalable and adaptable to other track circuits and railway systems, enhancing operational reliability. |
| 2025-08-12 | [LLM-as-a-Supervisor: Mistaken Therapeutic Behaviors Trigger Targeted Supervisory Feedback](http://arxiv.org/abs/2508.09042v1) | Chen Xu, Zhenyu Lv et al. | Although large language models (LLMs) hold significant promise in psychotherapy, their direct application in patient-facing scenarios raises ethical and safety concerns. Therefore, this work shifts towards developing an LLM as a supervisor to train real therapists. In addition to the privacy of clinical therapist training data, a fundamental contradiction complicates the training of therapeutic behaviors: clear feedback standards are necessary to ensure a controlled training system, yet there is no absolute "gold standard" for appropriate therapeutic behaviors in practice. In contrast, many common therapeutic mistakes are universal and identifiable, making them effective triggers for targeted feedback that can serve as clearer evidence. Motivated by this, we create a novel therapist-training paradigm: (1) guidelines for mistaken behaviors and targeted correction strategies are first established as standards; (2) a human-in-the-loop dialogue-feedback dataset is then constructed, where a mistake-prone agent intentionally makes standard mistakes during interviews naturally, and a supervisor agent locates and identifies mistakes and provides targeted feedback; (3) after fine-tuning on this dataset, the final supervisor model is provided for real therapist training. The detailed experimental results of automated, human and downstream assessments demonstrate that models fine-tuned on our dataset MATE, can provide high-quality feedback according to the clinical guideline, showing significant potential for the therapist training scenario. |
| 2025-08-12 | [Multi-Energy and Multi-Sample Searches for Neutrinos from GW Events](http://arxiv.org/abs/2508.09034v1) | Aswathi Balagopal V., Sam Hori et al. | The IceCube Neutrino Observatory at the South Pole detects neutrinos of astrophysical origin via their interactions with ice. The main array is optimized for the detection of neutrinos with energies above 1 TeV. A much smaller infill array, known as IceCube DeepCore, extends the sensitivity down to a few GeV. Neutrinos observed in both parts of the detector are used for astrophysical-source searches with multiple messengers. We present two analyses that follow up archival gravitational wave (GW) events from runs O1 through O3 of LIGO/Virgo/KAGRA. The first analysis uses two neutrino datasets: one with high-energy tracks and another consisting of low-energy tracks and cascades. These two neutrino datasets were previously used independently to follow-up GW events. In the analysis presented here, a combined likelihood search is performed using both datasets to search for neutrinos coincident with the GW events across a wide energy range, from a few GeV to several PeV. The second analysis, for the first time, uses a neutrino-induced cascade sample with events of energy above ~1 TeV for searches of coincident neutrino-GW emission. We present results from both analyses and discuss prospects for conducting these analyses in real time. |
| 2025-08-12 | [Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs](http://arxiv.org/abs/2508.09019v1) | Shivam Dubey | As large language models (LLMs) become more integrated into societal systems, the risk of them perpetuating and amplifying harmful biases becomes a critical safety concern. Traditional methods for mitigating bias often rely on data filtering or post-hoc output moderation, which treat the model as an opaque black box. In this work, we introduce a complete, end-to-end system that uses techniques from mechanistic interpretability to both identify and actively mitigate bias directly within a model's internal workings. Our method involves two primary stages. First, we train linear "probes" on the internal activations of a model to detect the latent representations of various biases (e.g., gender, race, age). Our experiments on \texttt{gpt2-large} demonstrate that these probes can identify biased content with near-perfect accuracy, revealing that bias representations become most salient in the model's later layers. Second, we leverage these findings to compute "steering vectors" by contrasting the model's activation patterns for biased and neutral statements. By adding these vectors during inference, we can actively steer the model's generative process away from producing harmful, stereotypical, or biased content in real-time. We demonstrate the efficacy of this activation steering technique, showing that it successfully alters biased completions toward more neutral alternatives. We present our work as a robust and reproducible system that offers a more direct and interpretable approach to building safer and more accountable LLMs. |
| 2025-08-12 | [Large Scale Robotic Material Handling: Learning, Planning, and Control](http://arxiv.org/abs/2508.09003v1) | Filippo A. Spinelli, Yifan Zhai et al. | Bulk material handling involves the efficient and precise moving of large quantities of materials, a core operation in many industries, including cargo ship unloading, waste sorting, construction, and demolition. These repetitive, labor-intensive, and safety-critical operations are typically performed using large hydraulic material handlers equipped with underactuated grippers. In this work, we present a comprehensive framework for the autonomous execution of large-scale material handling tasks. The system integrates specialized modules for environment perception, pile attack point selection, path planning, and motion control. The main contributions of this work are two reinforcement learning-based modules: an attack point planner that selects optimal grasping locations on the material pile to maximize removal efficiency and minimize the number of scoops, and a robust trajectory following controller that addresses the precision and safety challenges associated with underactuated grippers in movement, while utilizing their free-swinging nature to release material through dynamic throwing. We validate our framework through real-world experiments on a 40 t material handler in a representative worksite, focusing on two key tasks: high-throughput bulk pile management and high-precision truck loading. Comparative evaluations against human operators demonstrate the system's effectiveness in terms of precision, repeatability, and operational safety. To the best of our knowledge, this is the first complete automation of material handling tasks on a full scale. |
| 2025-08-12 | [How Does a Virtual Agent Decide Where to Look? - Symbolic Cognitive Reasoning for Embodied Head Rotation](http://arxiv.org/abs/2508.08930v1) | Juyeong Hwang, Seong-Eun Hon et al. | Natural head rotation is critical for believable embodied virtual agents, yet this micro-level behavior remains largely underexplored. While head-rotation prediction algorithms could, in principle, reproduce this behavior, they typically focus on visually salient stimuli and overlook the cognitive motives that guide head rotation. This yields agents that look at conspicuous objects while overlooking obstacles or task-relevant cues, diminishing realism in a virtual environment. We introduce SCORE, a Symbolic Cognitive Reasoning framework for Embodied Head Rotation, a data-agnostic framework that produces context-aware head movements without task-specific training or hand-tuned heuristics. A controlled VR study (N=20) identifies five motivational drivers of human head movements: Interest, Information Seeking, Safety, Social Schema, and Habit. SCORE encodes these drivers as symbolic predicates, perceives the scene with a Vision-Language Model (VLM), and plans head poses with a Large Language Model (LLM). The framework employs a hybrid workflow: the VLM-LLM reasoning is executed offline, after which a lightweight FastVLM performs online validation to suppress hallucinations while maintaining responsiveness to scene dynamics. The result is an agent that predicts not only where to look but also why, generalizing to unseen scenes and multi-agent crowds while retaining behavioral plausibility. |
| 2025-08-12 | [Safe Semantics, Unsafe Interpretations: Tackling Implicit Reasoning Safety in Large Vision-Language Models](http://arxiv.org/abs/2508.08926v1) | Wei Cai, Jian Zhao et al. | Large Vision-Language Models face growing safety challenges with multimodal inputs. This paper introduces the concept of Implicit Reasoning Safety, a vulnerability in LVLMs. Benign combined inputs trigger unsafe LVLM outputs due to flawed or hidden reasoning. To showcase this, we developed Safe Semantics, Unsafe Interpretations, the first dataset for this critical issue. Our demonstrations show that even simple In-Context Learning with SSUI significantly mitigates these implicit multimodal threats, underscoring the urgent need to improve cross-modal implicit reasoning. |
| 2025-08-12 | [BiasGym: Fantastic Biases and How to Find (and Remove) Them](http://arxiv.org/abs/2508.08855v1) | Sekh Mainul Islam, Nadav Borenstein et al. | Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during training. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from a country being `reckless drivers') and in probing fictional associations (e.g., people from a country having `blue skin'), showing its utility for both safety interventions and interpretability research. |
| 2025-08-12 | [An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems](http://arxiv.org/abs/2508.08833v1) | Yuren Hao, Xiang Wan et al. | In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities. |
| 2025-08-12 | [TechOps: Technical Documentation Templates for the AI Act](http://arxiv.org/abs/2508.08804v1) | Laura Lucaj, Alex Loosley et al. | Operationalizing the EU AI Act requires clear technical documentation to ensure AI systems are transparent, traceable, and accountable. Existing documentation templates for AI systems do not fully cover the entire AI lifecycle while meeting the technical documentation requirements of the AI Act.   This paper addresses those shortcomings by introducing open-source templates and examples for documenting data, models, and applications to provide sufficient documentation for certifying compliance with the AI Act. These templates track the system status over the entire AI lifecycle, ensuring traceability, reproducibility, and compliance with the AI Act. They also promote discoverability and collaboration, reduce risks, and align with best practices in AI documentation and governance.   The templates are evaluated and refined based on user feedback to enable insights into their usability and implementability. We then validate the approach on real-world scenarios, providing examples that further guide their implementation: the data template is followed to document a skin tones dataset created to support fairness evaluations of downstream computer vision models and human-centric applications; the model template is followed to document a neural network for segmenting human silhouettes in photos. The application template is tested on a system deployed for construction site safety using real-time video analytics and sensor data. Our results show that TechOps can serve as a practical tool to enable oversight for regulatory compliance and responsible AI development. |
| 2025-08-12 | [Towards Safe Imitation Learning via Potential Field-Guided Flow Matching](http://arxiv.org/abs/2508.08707v1) | Haoran Ding, Anqing Duan et al. | Deep generative models, particularly diffusion and flow matching models, have recently shown remarkable potential in learning complex policies through imitation learning. However, the safety of generated motions remains overlooked, particularly in complex environments with inherent obstacles. In this work, we address this critical gap by proposing Potential Field-Guided Flow Matching Policy (PF2MP), a novel approach that simultaneously learns task policies and extracts obstacle-related information, represented as a potential field, from the same set of successful demonstrations. During inference, PF2MP modulates the flow matching vector field via the learned potential field, enabling safe motion generation. By leveraging these complementary fields, our approach achieves improved safety without compromising task success across diverse environments, such as navigation tasks and robotic manipulation scenarios. We evaluate PF2MP in both simulation and real-world settings, demonstrating its effectiveness in task space and joint space control. Experimental results demonstrate that PF2MP enhances safety, achieving a significant reduction of collisions compared to baseline policies. This work paves the way for safer motion generation in unstructured and obstaclerich environments. |
| 2025-08-12 | [Prompt-and-Check: Using Large Language Models to Evaluate Communication Protocol Compliance in Simulation-Based Training](http://arxiv.org/abs/2508.08652v1) | Vishakha Lall, Yisi Liu | Accurate evaluation of procedural communication compliance is essential in simulation-based training, particularly in safety-critical domains where adherence to compliance checklists reflects operational competence. This paper explores a lightweight, deployable approach using prompt-based inference with open-source large language models (LLMs) that can run efficiently on consumer-grade GPUs. We present Prompt-and-Check, a method that uses context-rich prompts to evaluate whether each checklist item in a protocol has been fulfilled, solely based on transcribed verbal exchanges. We perform a case study in the maritime domain with participants performing an identical simulation task, and experiment with models such as LLama 2 7B, LLaMA 3 8B and Mistral 7B, running locally on an RTX 4070 GPU. For each checklist item, a prompt incorporating relevant transcript excerpts is fed into the model, which outputs a compliance judgment. We assess model outputs against expert-annotated ground truth using classification accuracy and agreement scores. Our findings demonstrate that prompting enables effective context-aware reasoning without task-specific training. This study highlights the practical utility of LLMs in augmenting debriefing, performance feedback, and automated assessment in training environments. |
| 2025-08-12 | [Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment](http://arxiv.org/abs/2508.08629v1) | Farzana Zahid, Anjalika Sewwandi et al. | Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions. |
| 2025-08-11 | [When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World Lessons from LLM Co-Design in a Safety-Net Hospital](http://arxiv.org/abs/2508.08504v1) | Avni Kothari, Patrick Vossler et al. | Large language models (LLMs) have the potential to address social and behavioral determinants of health by transforming labor intensive workflows in resource-constrained settings. Creating LLM-based applications that serve the needs of underserved communities requires a deep understanding of their local context, but it is often the case that neither LLMs nor their developers possess this local expertise, and the experts in these communities often face severe time/resource constraints. This creates a disconnect: how can one engage in meaningful co-design of an LLM-based application for an under-resourced community when the communication channel between the LLM developer and domain expert is constrained? We explored this question through a real-world case study, in which our data science team sought to partner with social workers at a safety net hospital to build an LLM application that summarizes patients' social needs. Whereas prior works focus on the challenge of prompt tuning, we found that the most critical challenge in this setting is the careful and precise specification of \what information to surface to providers so that the LLM application is accurate, comprehensive, and verifiable. Here we present a novel co-design framework for settings with limited access to domain experts, in which the summary generation task is first decomposed into individually-optimizable attributes and then each attribute is efficiently refined and validated through a multi-tier cascading approach. |
| 2025-08-11 | [Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery](http://arxiv.org/abs/2508.08401v1) | Jiatong Li, Weida Wang et al. | Large language models (LLMs), especially Explicit Long Chain-of-Thought (CoT) reasoning models like DeepSeek-R1 and QWQ, have demonstrated powerful reasoning capabilities, achieving impressive performance in commonsense reasoning and mathematical inference. Despite their effectiveness, Long-CoT reasoning models are often criticized for their limited ability and low efficiency in knowledge-intensive domains such as molecule discovery. Success in this field requires a precise understanding of domain knowledge, including molecular structures and chemical principles, which is challenging due to the inherent complexity of molecular data and the scarcity of high-quality expert annotations. To bridge this gap, we introduce Mol-R1, a novel framework designed to improve explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation. Our approach begins with a high-quality reasoning dataset curated through Prior Regulation via In-context Distillation (PRID), a dedicated distillation strategy to effectively generate paired reasoning traces guided by prior regulations. Building upon this, we introduce MoIA, Molecular Iterative Adaptation, a sophisticated training strategy that iteratively combines Supervised Fine-tuning (SFT) with Reinforced Policy Optimization (RPO), tailored to boost the reasoning performance of R1-like reasoning models for molecule discovery. Finally, we examine the performance of Mol-R1 in the text-based molecule reasoning generation task, showing superior performance against existing baselines. |

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



