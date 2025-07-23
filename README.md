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
| 2025-07-22 | [Faithful, Interpretable Chest X-ray Diagnosis with Anti-Aliased B-cos Networks](http://arxiv.org/abs/2507.16761v1) | Marcel Kleinmann, Shashank Agnihotri et al. | Faithfulness and interpretability are essential for deploying deep neural networks (DNNs) in safety-critical domains such as medical imaging. B-cos networks offer a promising solution by replacing standard linear layers with a weight-input alignment mechanism, producing inherently interpretable, class-specific explanations without post-hoc methods. While maintaining diagnostic performance competitive with state-of-the-art DNNs, standard B-cos models suffer from severe aliasing artifacts in their explanation maps, making them unsuitable for clinical use where clarity is essential. Additionally, the original B-cos formulation is limited to multi-class settings, whereas chest X-ray analysis often requires multi-label classification due to co-occurring abnormalities. In this work, we address both limitations: (1) we introduce anti-aliasing strategies using FLCPooling (FLC) and BlurPool (BP) to significantly improve explanation quality, and (2) we extend B-cos networks to support multi-label classification. Our experiments on chest X-ray datasets demonstrate that the modified $\text{B-cos}_\text{FLC}$ and $\text{B-cos}_\text{BP}$ preserve strong predictive performance while providing faithful and artifact-free explanations suitable for clinical application in multi-label settings. Code available at: $\href{https://github.com/mkleinma/B-cos-medical-paper}{GitHub repository}$. |
| 2025-07-22 | [A Target-based Multi-LiDAR Multi-Camera Extrinsic Calibration System](http://arxiv.org/abs/2507.16621v1) | Lorenzo Gentilini, Pierpaolo Serio et al. | Extrinsic Calibration represents the cornerstone of autonomous driving. Its accuracy plays a crucial role in the perception pipeline, as any errors can have implications for the safety of the vehicle. Modern sensor systems collect different types of data from the environment, making it harder to align the data. To this end, we propose a target-based extrinsic calibration system tailored for a multi-LiDAR and multi-camera sensor suite. This system enables cross-calibration between LiDARs and cameras with limited prior knowledge using a custom ChArUco board and a tailored nonlinear optimization method. We test the system with real-world data gathered in a warehouse. Results demonstrated the effectiveness of the proposed method, highlighting the feasibility of a unique pipeline tailored for various types of sensors. |
| 2025-07-22 | [Optimization of DNN-based HSI Segmentation FPGA-based SoC for ADS: A Practical Approach](http://arxiv.org/abs/2507.16556v1) | Jon Gutiérrez-Zaballa, Koldo Basterretxea et al. | The use of HSI for autonomous navigation is a promising research field aimed at improving the accuracy and robustness of detection, tracking, and scene understanding systems based on vision sensors. Combining advanced computer algorithms, such as DNNs, with small-size snapshot HSI cameras enhances the reliability of these systems. HSI overcomes intrinsic limitations of greyscale and RGB imaging in depicting physical properties of targets, particularly regarding spectral reflectance and metamerism. Despite promising results in HSI-based vision developments, safety-critical systems like ADS demand strict constraints on latency, resource consumption, and security, motivating the shift of ML workloads to edge platforms. This involves a thorough software/hardware co-design scheme to distribute and optimize the tasks efficiently among the limited resources of computing platforms. With respect to inference, the over-parameterized nature of DNNs poses significant computational challenges for real-time on-the-edge deployment. In addition, the intensive data preprocessing required by HSI, which is frequently overlooked, must be carefully managed in terms of memory arrangement and inter-task communication to enable an efficient integrated pipeline design on a SoC. This work presents a set of optimization techniques for the practical co-design of a DNN-based HSI segmentation processor deployed on a FPGA-based SoC targeted at ADS, including key optimizations such as functional software/hardware task distribution, hardware-aware preprocessing, ML model compression, and a complete pipelined deployment. Applied compression techniques significantly reduce the complexity of the designed DNN to 24.34% of the original operations and to 1.02% of the original number of parameters, achieving a 2.86x speed-up in the inference task without noticeable degradation of the segmentation accuracy. |
| 2025-07-22 | [Probing Large $N_f$ Through Schemes](http://arxiv.org/abs/2507.16504v1) | Alan Pinoy, Shahram Vatani | We investigate the reliability of the large $N_f$ expansion of four-dimensional gauge-fermion quantum field theories, focusing on the structure and scheme dependence of the beta function. While the existence of a nontrivial UV fixed point at leading order in $1/N_f$ suggests the possibility of asymptotic safety, the absence of higher-order terms precludes robust conclusions. We analyze the impact of renormalization scheme transformations and show that higher-order corrections inevitably introduce increasingly singular contributions. We prove that at most one renormalization scheme can preserve the dominance of the leading contribution, rendering the truncation trustworthy; in all other schemes, higher-order terms dominate and the expansion becomes unreliable. This result places strong constraints on the physical interpretation of UV fixed points in large $N_f$ theories and emphasizes the need for resummation or non-perturbative control to establish asymptotic safety. |
| 2025-07-22 | [Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots](http://arxiv.org/abs/2507.16481v1) | Riccardo Bussola, Michele Focchi et al. | Jumping poses a significant challenge for quadruped robots, despite being crucial for many operational scenarios. While optimisation methods exist for controlling such motions, they are often time-consuming and demand extensive knowledge of robot and terrain parameters, making them less robust in real-world scenarios. Reinforcement learning (RL) is emerging as a viable alternative, yet conventional end-to-end approaches lack efficiency in terms of sample complexity, requiring extensive training in simulations, and predictability of the final motion, which makes it difficult to certify the safety of the final motion. To overcome these limitations, this paper introduces a novel guided reinforcement learning approach that leverages physical intuition for efficient and explainable jumping, by combining B\'ezier curves with a Uniformly Accelerated Rectilinear Motion (UARM) model. Extensive simulation and experimental results clearly demonstrate the advantages of our approach over existing alternatives. |
| 2025-07-22 | [Depth Gives a False Sense of Privacy: LLM Internal States Inversion](http://arxiv.org/abs/2507.16372v1) | Tian Dong, Yan Meng et al. | Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design. |
| 2025-07-22 | [DREAM: Scalable Red Teaming for Text-to-Image Generative Systems via Distribution Modeling](http://arxiv.org/abs/2507.16329v1) | Boheng Li, Junjie Wang et al. | Despite the integration of safety alignment and external filters, text-to-image (T2I) generative models are still susceptible to producing harmful content, such as sexual or violent imagery. This raises serious concerns about unintended exposure and potential misuse. Red teaming, which aims to proactively identify diverse prompts that can elicit unsafe outputs from the T2I system (including the core generative model as well as potential external safety filters and other processing components), is increasingly recognized as an essential method for assessing and improving safety before real-world deployment. Yet, existing automated red teaming approaches often treat prompt discovery as an isolated, prompt-level optimization task, which limits their scalability, diversity, and overall effectiveness. To bridge this gap, in this paper, we propose DREAM, a scalable red teaming framework to automatically uncover diverse problematic prompts from a given T2I system. Unlike most prior works that optimize prompts individually, DREAM directly models the probabilistic distribution of the target system's problematic prompts, which enables explicit optimization over both effectiveness and diversity, and allows efficient large-scale sampling after training. To achieve this without direct access to representative training samples, we draw inspiration from energy-based models and reformulate the objective into simple and tractable objectives. We further introduce GC-SPSA, an efficient optimization algorithm that provide stable gradient estimates through the long and potentially non-differentiable T2I pipeline. The effectiveness of DREAM is validated through extensive experiments, demonstrating that it surpasses 9 state-of-the-art baselines by a notable margin across a broad range of T2I models and safety filters in terms of prompt success rate and diversity. |
| 2025-07-22 | [Towards Resilient Safety-driven Unlearning for Diffusion Models against Downstream Fine-tuning](http://arxiv.org/abs/2507.16302v1) | Boheng Li, Renjie Gu et al. | Text-to-image (T2I) diffusion models have achieved impressive image generation quality and are increasingly fine-tuned for personalized applications. However, these models often inherit unsafe behaviors from toxic pretraining data, raising growing safety concerns. While recent safety-driven unlearning methods have made promising progress in suppressing model toxicity, they are identified to be fragile to downstream fine-tuning, where we reveal that state-of-the-art methods largely fail to retain their effectiveness even when fine-tuned on entirely benign datasets. To mitigate this problem, in this paper, we propose ResAlign, a safety-driven unlearning framework with enhanced resilience against downstream fine-tuning. By modeling downstream fine-tuning as an implicit optimization problem with a Moreau Envelope-based reformulation, ResAlign enables efficient gradient estimation to minimize the recovery of harmful behaviors. Additionally, a meta-learning strategy is proposed to simulate a diverse distribution of fine-tuning scenarios to improve generalization. Extensive experiments across a wide range of datasets, fine-tuning methods, and configurations demonstrate that ResAlign consistently outperforms prior unlearning approaches in retaining safety after downstream fine-tuning while preserving benign generation capability well. |
| 2025-07-22 | [PAC Off-Policy Prediction of Contextual Bandits](http://arxiv.org/abs/2507.16236v1) | Yilong Wan, Yuqiang Li et al. | This paper investigates off-policy evaluation in contextual bandits, aiming to quantify the performance of a target policy using data collected under a different and potentially unknown behavior policy. Recently, methods based on conformal prediction have been developed to construct reliable prediction intervals that guarantee marginal coverage in finite samples, making them particularly suited for safety-critical applications. To further achieve coverage conditional on a given offline data set, we propose a novel algorithm that constructs probably approximately correct prediction intervals. Our method builds upon a PAC-valid conformal prediction framework, and we strengthen its theoretical guarantees by establishing PAC-type bounds on coverage. We analyze both finite-sample and asymptotic properties of the proposed method, and compare its empirical performance with existing methods in simulations. |
| 2025-07-22 | [METER: Multi-modal Evidence-based Thinking and Explainable Reasoning -- Algorithm and Benchmark](http://arxiv.org/abs/2507.16206v1) | Xu Yang, Qi Zhang et al. | With the rapid advancement of generative AI, synthetic content across images, videos, and audio has become increasingly realistic, amplifying the risk of misinformation. Existing detection approaches predominantly focus on binary classification while lacking detailed and interpretable explanations of forgeries, which limits their applicability in safety-critical scenarios. Moreover, current methods often treat each modality separately, without a unified benchmark for cross-modal forgery detection and interpretation. To address these challenges, we introduce METER, a unified, multi-modal benchmark for interpretable forgery detection spanning images, videos, audio, and audio-visual content. Our dataset comprises four tracks, each requiring not only real-vs-fake classification but also evidence-chain-based explanations, including spatio-temporal localization, textual rationales, and forgery type tracing. Compared to prior benchmarks, METER offers broader modality coverage and richer interpretability metrics such as spatial/temporal IoU, multi-class tracing, and evidence consistency. We further propose a human-aligned, three-stage Chain-of-Thought (CoT) training strategy combining SFT, DPO, and a novel GRPO stage that integrates a human-aligned evaluator with CoT reasoning. We hope METER will serve as a standardized foundation for advancing generalizable and interpretable forgery detection in the era of generative media. |
| 2025-07-22 | [RealBench: Benchmarking Verilog Generation Models with Real-World IP Designs](http://arxiv.org/abs/2507.16200v1) | Pengwei Jin, Di Huang et al. | The automatic generation of Verilog code using Large Language Models (LLMs) has garnered significant interest in hardware design automation. However, existing benchmarks for evaluating LLMs in Verilog generation fall short in replicating real-world design workflows due to their designs' simplicity, inadequate design specifications, and less rigorous verification environments. To address these limitations, we present RealBench, the first benchmark aiming at real-world IP-level Verilog generation tasks. RealBench features complex, structured, real-world open-source IP designs, multi-modal and formatted design specifications, and rigorous verification environments, including 100% line coverage testbenches and a formal checker. It supports both module-level and system-level tasks, enabling comprehensive assessments of LLM capabilities. Evaluations on various LLMs and agents reveal that even one of the best-performing LLMs, o1-preview, achieves only a 13.3% pass@1 on module-level tasks and 0% on system-level tasks, highlighting the need for stronger Verilog generation models in the future. The benchmark is open-sourced at https://github.com/IPRC-DIP/RealBench. |
| 2025-07-22 | [Do Large Language Models Have a Planning Theory of Mind? Evidence from MindGames: a Multi-Step Persuasion Task](http://arxiv.org/abs/2507.16196v1) | Jared Moore, Ned Cooper et al. | Recent evidence suggests Large Language Models (LLMs) display Theory of Mind (ToM) abilities. Most ToM experiments place participants in a spectatorial role, wherein they predict and interpret other agents' behavior. However, human ToM also contributes to dynamically planning action and strategically intervening on others' mental states. We present MindGames: a novel `planning theory of mind' (PToM) task which requires agents to infer an interlocutor's beliefs and desires to persuade them to alter their behavior. Unlike previous evaluations, we explicitly evaluate use cases of ToM. We find that humans significantly outperform o1-preview (an LLM) at our PToM task (11% higher; $p=0.006$). We hypothesize this is because humans have an implicit causal model of other agents (e.g., they know, as our task requires, to ask about people's preferences). In contrast, o1-preview outperforms humans in a baseline condition which requires a similar amount of planning but minimal mental state inferences (e.g., o1-preview is better than humans at planning when already given someone's preferences). These results suggest a significant gap between human-like social reasoning and LLM abilities. |
| 2025-07-21 | [Interpreting CFD Surrogates through Sparse Autoencoders](http://arxiv.org/abs/2507.16069v1) | Yeping Hu, Shusen Liu | Learning-based surrogate models have become a practical alternative to high-fidelity CFD solvers, but their latent representations remain opaque and hinder adoption in safety-critical or regulation-bound settings. This work introduces a posthoc interpretability framework for graph-based surrogate models used in computational fluid dynamics (CFD) by leveraging sparse autoencoders (SAEs). By obtaining an overcomplete basis in the node embedding space of a pretrained surrogate, the method extracts a dictionary of interpretable latent features. The approach enables the identification of monosemantic concepts aligned with physical phenomena such as vorticity or flow structures, offering a model-agnostic pathway to enhance explainability and trustworthiness in CFD applications. |
| 2025-07-21 | ["Just a strange pic": Evaluating 'safety' in GenAI Image safety annotation tasks from diverse annotators' perspectives](http://arxiv.org/abs/2507.16033v1) | Ding Wang, Mark Díaz et al. | Understanding what constitutes safety in AI-generated content is complex. While developers often rely on predefined taxonomies, real-world safety judgments also involve personal, social, and cultural perceptions of harm. This paper examines how annotators evaluate the safety of AI-generated images, focusing on the qualitative reasoning behind their judgments. Analyzing 5,372 open-ended comments, we find that annotators consistently invoke moral, emotional, and contextual reasoning that extends beyond structured safety categories. Many reflect on potential harm to others more than to themselves, grounding their judgments in lived experience, collective risk, and sociocultural awareness. Beyond individual perceptions, we also find that the structure of the task itself -- including annotation guidelines -- shapes how annotators interpret and express harm. Guidelines influence not only which images are flagged, but also the moral judgment behind the justifications. Annotators frequently cite factors such as image quality, visual distortion, and mismatches between prompt and output as contributing to perceived harm dimensions, which are often overlooked in standard evaluation frameworks. Our findings reveal that existing safety pipelines miss critical forms of reasoning that annotators bring to the task. We argue for evaluation designs that scaffold moral reflection, differentiate types of harm, and make space for subjective, context-sensitive interpretations of AI-generated content. |
| 2025-07-21 | [Does More Inference-Time Compute Really Help Robustness?](http://arxiv.org/abs/2507.15974v1) | Tong Wu, Chong Xiang et al. | Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications. |
| 2025-07-21 | [The Impact of Language Mixing on Bilingual LLM Reasoning](http://arxiv.org/abs/2507.15849v1) | Yihao Li, Jiayi Xin et al. | Proficient multilingual speakers often intentionally switch languages in the middle of a conversation. Similarly, recent reasoning-focused bilingual large language models (LLMs) with strong capabilities in both languages exhibit language mixing--alternating languages within their chain of thought. Discouraging this behavior in DeepSeek-R1 was found to degrade accuracy, suggesting that language mixing may benefit reasoning. In this work, we study language switching in Chinese-English bilingual reasoning models. We identify reinforcement learning with verifiable rewards (RLVR) as the critical training stage that leads to language mixing. We demonstrate that language mixing can enhance reasoning: enforcing monolingual decoding reduces accuracy by 5.6 percentage points on math reasoning tasks. Additionally, a lightweight probe can be trained to predict whether a potential language switch would benefit or harm reasoning, and when used to guide decoding, increases accuracy by up to 6.25 percentage points. Our findings suggest that language mixing is not merely a byproduct of multilingual training, but is a strategic reasoning behavior. |
| 2025-07-21 | [Towards physician-centered oversight of conversational diagnostic AI](http://arxiv.org/abs/2507.15743v1) | Elahe Vedadi, David Barrett et al. | Recent work has demonstrated the promise of conversational AI systems for diagnostic dialogue. However, real-world assurance of patient safety means that providing individual diagnoses and treatment plans is considered a regulated activity by licensed professionals. Furthermore, physicians commonly oversee other team members in such activities, including nurse practitioners (NPs) or physician assistants/associates (PAs). Inspired by this, we propose a framework for effective, asynchronous oversight of the Articulate Medical Intelligence Explorer (AMIE) AI system. We propose guardrailed-AMIE (g-AMIE), a multi-agent system that performs history taking within guardrails, abstaining from individualized medical advice. Afterwards, g-AMIE conveys assessments to an overseeing primary care physician (PCP) in a clinician cockpit interface. The PCP provides oversight and retains accountability of the clinical decision. This effectively decouples oversight from intake and can thus happen asynchronously. In a randomized, blinded virtual Objective Structured Clinical Examination (OSCE) of text consultations with asynchronous oversight, we compared g-AMIE to NPs/PAs or a group of PCPs under the same guardrails. Across 60 scenarios, g-AMIE outperformed both groups in performing high-quality intake, summarizing cases, and proposing diagnoses and management plans for the overseeing PCP to review. This resulted in higher quality composite decisions. PCP oversight of g-AMIE was also more time-efficient than standalone PCP consultations in prior work. While our study does not replicate existing clinical practices and likely underestimates clinicians' capabilities, our results demonstrate the promise of asynchronous oversight as a feasible paradigm for diagnostic AI systems to operate under expert human oversight for enhancing real-world care. |
| 2025-07-21 | [BEnchmarking LLMs for Ophthalmology (BELO) for Ophthalmological Knowledge and Reasoning](http://arxiv.org/abs/2507.15717v1) | Sahana Srinivasan, Xuguang Ai et al. | Current benchmarks evaluating large language models (LLMs) in ophthalmology are limited in scope and disproportionately prioritise accuracy. We introduce BELO (BEnchmarking LLMs for Ophthalmology), a standardized and comprehensive evaluation benchmark developed through multiple rounds of expert checking by 13 ophthalmologists. BELO assesses ophthalmology-related clinical accuracy and reasoning quality. Using keyword matching and a fine-tuned PubMedBERT model, we curated ophthalmology-specific multiple-choice-questions (MCQs) from diverse medical datasets (BCSC, MedMCQA, MedQA, BioASQ, and PubMedQA). The dataset underwent multiple rounds of expert checking. Duplicate and substandard questions were systematically removed. Ten ophthalmologists refined the explanations of each MCQ's correct answer. This was further adjudicated by three senior ophthalmologists. To illustrate BELO's utility, we evaluated six LLMs (OpenAI o1, o3-mini, GPT-4o, DeepSeek-R1, Llama-3-8B, and Gemini 1.5 Pro) using accuracy, macro-F1, and five text-generation metrics (ROUGE-L, BERTScore, BARTScore, METEOR, and AlignScore). In a further evaluation involving human experts, two ophthalmologists qualitatively reviewed 50 randomly selected outputs for accuracy, comprehensiveness, and completeness. BELO consists of 900 high-quality, expert-reviewed questions aggregated from five sources: BCSC (260), BioASQ (10), MedMCQA (572), MedQA (40), and PubMedQA (18). A public leaderboard has been established to promote transparent evaluation and reporting. Importantly, the BELO dataset will remain a hold-out, evaluation-only benchmark to ensure fair and reproducible comparisons of future models. |
| 2025-07-21 | [Surfacing Variations to Calibrate Perceived Reliability of MLLM-generated Image Descriptions](http://arxiv.org/abs/2507.15692v1) | Meng Chen, Akhil Iyer et al. | Multimodal large language models (MLLMs) provide new opportunities for blind and low vision (BLV) people to access visual information in their daily lives. However, these models often produce errors that are difficult to detect without sight, posing safety and social risks in scenarios from medication identification to outfit selection. While BLV MLLM users use creative workarounds such as cross-checking between tools and consulting sighted individuals, these approaches are often time-consuming and impractical. We explore how systematically surfacing variations across multiple MLLM responses can support BLV users to detect unreliable information without visually inspecting the image. We contribute a design space for eliciting and presenting variations in MLLM descriptions, a prototype system implementing three variation presentation styles, and findings from a user study with 15 BLV participants. Our results demonstrate that presenting variations significantly increases users' ability to identify unreliable claims (by 4.9x using our approach compared to single descriptions) and significantly decreases perceived reliability of MLLM responses. 14 of 15 participants preferred seeing variations of MLLM responses over a single description, and all expressed interest in using our system for tasks from understanding a tornado's path to posting an image on social media. |
| 2025-07-21 | [EMP: Executable Motion Prior for Humanoid Robot Standing Upper-body Motion Imitation](http://arxiv.org/abs/2507.15649v1) | Haocheng Xu, Haodong Zhang et al. | To support humanoid robots in performing manipulation tasks, it is essential to study stable standing while accommodating upper-body motions. However, the limited controllable range of humanoid robots in a standing position affects the stability of the entire body. Thus we introduce a reinforcement learning based framework for humanoid robots to imitate human upper-body motions while maintaining overall stability. Our approach begins with designing a retargeting network that generates a large-scale upper-body motion dataset for training the reinforcement learning (RL) policy, which enables the humanoid robot to track upper-body motion targets, employing domain randomization for enhanced robustness. To avoid exceeding the robot's execution capability and ensure safety and stability, we propose an Executable Motion Prior (EMP) module, which adjusts the input target movements based on the robot's current state. This adjustment improves standing stability while minimizing changes to motion amplitude. We evaluate our framework through simulation and real-world tests, demonstrating its practical applicability. |

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



