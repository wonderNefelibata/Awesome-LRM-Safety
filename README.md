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
| 2025-04-18 | [DiffOG: Differentiable Policy Trajectory Optimization with Generalizability](http://arxiv.org/abs/2504.13807v1) | Zhengtong Xu, Zichen Miao et al. | Imitation learning-based visuomotor policies excel at manipulation tasks but often produce suboptimal action trajectories compared to model-based methods. Directly mapping camera data to actions via neural networks can result in jerky motions and difficulties in meeting critical constraints, compromising safety and robustness in real-world deployment. For tasks that require high robustness or strict adherence to constraints, ensuring trajectory quality is crucial. However, the lack of interpretability in neural networks makes it challenging to generate constraint-compliant actions in a controlled manner. This paper introduces differentiable policy trajectory optimization with generalizability (DiffOG), a learning-based trajectory optimization framework designed to enhance visuomotor policies. By leveraging the proposed differentiable formulation of trajectory optimization with transformer, DiffOG seamlessly integrates policies with a generalizable optimization layer. Visuomotor policies enhanced by DiffOG generate smoother, constraint-compliant action trajectories in a more interpretable way. DiffOG exhibits strong generalization capabilities and high flexibility. We evaluated DiffOG across 11 simulated tasks and 2 real-world tasks. The results demonstrate that DiffOG significantly enhances the trajectory quality of visuomotor policies while having minimal impact on policy performance, outperforming trajectory processing baselines such as greedy constraint clipping and penalty-based trajectory optimization. Furthermore, DiffOG achieves superior performance compared to existing constrained visuomotor policy. |
| 2025-04-18 | [Meta-Learning and Knowledge Discovery based Physics-Informed Neural Network for Remaining Useful Life Prediction](http://arxiv.org/abs/2504.13797v1) | Yu Wang, Shujie Liu et al. | Predicting the remaining useful life (RUL) of rotating machinery is critical for industrial safety and maintenance, but existing methods struggle with scarce target-domain data and unclear degradation dynamics. We propose a Meta-Learning and Knowledge Discovery-based Physics-Informed Neural Network (MKDPINN) to address these challenges. The method first maps noisy sensor data to a low-dimensional hidden state space via a Hidden State Mapper (HSM). A Physics-Guided Regulator (PGR) then learns unknown nonlinear PDEs governing degradation evolution, embedding these physical constraints into the PINN framework. This integrates data-driven and physics-based approaches. The framework uses meta-learning, optimizing across source-domain meta-tasks to enable few-shot adaptation to new target tasks. Experiments on industrial data and the C-MAPSS benchmark show MKDPINN outperforms baselines in generalization and accuracy, proving its effectiveness for RUL prediction under data scarcity |
| 2025-04-18 | [Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](http://arxiv.org/abs/2504.13677v1) | Andrea Santilli, Adam Golinski et al. | Uncertainty Quantification (UQ) in Language Models (LMs) is crucial for improving their safety and reliability. Evaluations often use performance metrics like AUROC to assess how well UQ methods (e.g., negative sequence probabilities) correlate with task correctness functions (e.g., ROUGE-L). In this paper, we show that commonly used correctness functions bias UQ evaluations by inflating the performance of certain UQ methods. We evaluate 7 correctness functions -- from lexical-based and embedding-based metrics to LLM-as-a-judge approaches -- across 4 datasets x 4 models x 6 UQ methods. Our analysis reveals that length biases in the errors of these correctness functions distort UQ assessments by interacting with length biases in UQ methods. We identify LLM-as-a-judge approaches as among the least length-biased choices and hence a potential solution to mitigate these biases. |
| 2025-04-18 | [Enhancing Pothole Detection and Characterization: Integrated Segmentation and Depth Estimation in Road Anomaly Systems](http://arxiv.org/abs/2504.13648v1) | Uthman Baroudi, Alala BaHamid et al. | Road anomaly detection plays a crucial role in road maintenance and in enhancing the safety of both drivers and vehicles. Recent machine learning approaches for road anomaly detection have overcome the tedious and time-consuming process of manual analysis and anomaly counting; however, they often fall short in providing a complete characterization of road potholes. In this paper, we leverage transfer learning by adopting a pre-trained YOLOv8-seg model for the automatic characterization of potholes using digital images captured from a dashboard-mounted camera. Our work includes the creation of a novel dataset, comprising both images and their corresponding depth maps, collected from diverse road environments in Al-Khobar city and the KFUPM campus in Saudi Arabia. Our approach performs pothole detection and segmentation to precisely localize potholes and calculate their area. Subsequently, the segmented image is merged with its depth map to extract detailed depth information about the potholes. This integration of segmentation and depth data offers a more comprehensive characterization compared to previous deep learning-based road anomaly detection systems. Overall, this method not only has the potential to significantly enhance autonomous vehicle navigation by improving the detection and characterization of road hazards but also assists road maintenance authorities in responding more effectively to road damage. |
| 2025-04-18 | [Thought Manipulation: External Thought Can Be Efficient for Large Reasoning Models](http://arxiv.org/abs/2504.13626v1) | Yule Liu, Jingyi Zheng et al. | Recent advancements in large reasoning models (LRMs) have demonstrated the effectiveness of scaling test-time computation to enhance reasoning capabilities in multiple tasks. However, LRMs typically suffer from "overthinking" problems, where models generate significantly redundant reasoning steps while bringing limited performance gains. Existing work relies on fine-tuning to mitigate overthinking, which requires additional data, unconventional training setups, risky safety misalignment, and poor generalization.   Through empirical analysis, we reveal an important characteristic of LRM behaviors that placing external CoTs generated by smaller models between the thinking token ($\texttt{<think>}$ and $\texttt{</think>)}$ can effectively manipulate the model to generate fewer thoughts. Building on these insights, we propose a simple yet efficient pipeline, ThoughtMani, to enable LRMs to bypass unnecessary intermediate steps and reduce computational costs significantly. We conduct extensive experiments to validate the utility and efficiency of ThoughtMani. For instance, when applied to QwQ-32B on the LiveBench/Code dataset, ThoughtMani keeps the original performance and reduces output token counts by approximately 30%, with little overhead from the CoT generator. Furthermore, we find that ThoughtMani enhances safety alignment by an average of 10%. Since model vendors typically serve models of different sizes simultaneously, ThoughtMani provides an effective way to construct more efficient and accessible LRMs for real-world applications. |
| 2025-04-18 | [Hysteresis-Aware Neural Network Modeling and Whole-Body Reinforcement Learning Control of Soft Robots](http://arxiv.org/abs/2504.13582v1) | Zongyuan Chen, Yan Xia et al. | Soft robots exhibit inherent compliance and safety, which makes them particularly suitable for applications requiring direct physical interaction with humans, such as surgical procedures. However, their nonlinear and hysteretic behavior, resulting from the properties of soft materials, presents substantial challenges for accurate modeling and control. In this study, we present a soft robotic system designed for surgical applications and propose a hysteresis-aware whole-body neural network model that accurately captures and predicts the soft robot's whole-body motion, including its hysteretic behavior. Building upon the high-precision dynamic model, we construct a highly parallel simulation environment for soft robot control and apply an on-policy reinforcement learning algorithm to efficiently train whole-body motion control strategies. Based on the trained control policy, we developed a soft robotic system for surgical applications and validated it through phantom-based laser ablation experiments in a physical environment. The results demonstrate that the hysteresis-aware modeling reduces the Mean Squared Error (MSE) by 84.95 percent compared to traditional modeling methods. The deployed control algorithm achieved a trajectory tracking error ranging from 0.126 to 0.250 mm on the real soft robot, highlighting its precision in real-world conditions. The proposed method showed strong performance in phantom-based surgical experiments and demonstrates its potential for complex scenarios, including future real-world clinical applications. |
| 2025-04-18 | [DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification](http://arxiv.org/abs/2504.13562v1) | Yu Li, Han Jiang et al. | With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance. |
| 2025-04-18 | [Risk-aware black-box portfolio construction using Bayesian optimization with adaptive weighted Lagrangian estimator](http://arxiv.org/abs/2504.13529v1) | Zinuo You, John Cartlidge et al. | Existing portfolio management approaches are often black-box models due to safety and commercial issues in the industry. However, their performance can vary considerably whenever market conditions or internal trading strategies change. Furthermore, evaluating these non-transparent systems is expensive, where certain budgets limit observations of the systems. Therefore, optimizing performance while controlling the potential risk of these financial systems has become a critical challenge. This work presents a novel Bayesian optimization framework to optimize black-box portfolio management models under limited observations. In conventional Bayesian optimization settings, the objective function is to maximize the expectation of performance metrics. However, simply maximizing performance expectations leads to erratic optimization trajectories, which exacerbate risk accumulation in portfolio management. Meanwhile, this can lead to misalignment between the target distribution and the actual distribution of the black-box model. To mitigate this problem, we propose an adaptive weight Lagrangian estimator considering dual objective, which incorporates maximizing model performance and minimizing variance of model observations. Extensive experiments demonstrate the superiority of our approach over five backtest settings with three black-box stock portfolio management models. Ablation studies further verify the effectiveness of the proposed estimator. |
| 2025-04-18 | [Monitor and Recover: A Paradigm for Future Research on Distribution Shift in Learning-Enabled Cyber-Physical Systems](http://arxiv.org/abs/2504.13484v1) | Vivian Lin, Insup Lee | With the known vulnerability of neural networks to distribution shift, maintaining reliability in learning-enabled cyber-physical systems poses a salient challenge. In response, many existing methods adopt a detect and abstain methodology, aiming to detect distribution shift at inference time so that the learning-enabled component can abstain from decision-making. This approach, however, has limited use in real-world applications. We instead propose a monitor and recover paradigm as a promising direction for future research. This philosophy emphasizes 1) robust safety monitoring instead of distribution shift detection and 2) distribution shift recovery instead of abstention. We discuss two examples from our recent work. |
| 2025-04-18 | [Safety Monitoring for Learning-Enabled Cyber-Physical Systems in Out-of-Distribution Scenarios](http://arxiv.org/abs/2504.13478v1) | Vivian Lin, Ramneet Kaur et al. | The safety of learning-enabled cyber-physical systems is compromised by the well-known vulnerabilities of deep neural networks to out-of-distribution (OOD) inputs. Existing literature has sought to monitor the safety of such systems by detecting OOD data. However, such approaches have limited utility, as the presence of an OOD input does not necessarily imply the violation of a desired safety property. We instead propose to directly monitor safety in a manner that is itself robust to OOD data. To this end, we predict violations of signal temporal logic safety specifications based on predicted future trajectories. Our safety monitor additionally uses a novel combination of adaptive conformal prediction and incremental learning. The former obtains probabilistic prediction guarantees even on OOD data, and the latter prevents overly conservative predictions. We evaluate the efficacy of the proposed approach in two case studies on safety monitoring: 1) predicting collisions of an F1Tenth car with static obstacles, and 2) predicting collisions of a race car with multiple dynamic obstacles. We find that adaptive conformal prediction obtains theoretical guarantees where other uncertainty quantification methods fail to do so. Additionally, combining adaptive conformal prediction and incremental learning for safety monitoring achieves high recall and timeliness while reducing loss in precision. We achieve these results even in OOD settings and outperform alternative methods. |
| 2025-04-18 | [Testing the Fault-Tolerance of Multi-Sensor Fusion Perception in Autonomous Driving Systems](http://arxiv.org/abs/2504.13420v1) | Haoxiang Tian, Wenqiang Ding et al. | High-level Autonomous Driving Systems (ADSs), such as Google Waymo and Baidu Apollo, typically rely on multi-sensor fusion (MSF) based approaches to perceive their surroundings. This strategy increases perception robustness by combining the respective strengths of the camera and LiDAR and directly affects the safety-critical driving decisions of autonomous vehicles (AVs). However, in real-world autonomous driving scenarios, cameras and LiDAR are subject to various faults, which can probably significantly impact the decision-making and behaviors of ADSs. Existing MSF testing approaches only discovered corner cases that the MSF-based perception cannot accurately detected by MSF-based perception, while lacking research on how sensor faults affect the system-level behaviors of ADSs.   To address this gap, we conduct the first exploration of the fault tolerance of MSF perception-based ADS for sensor faults. In this paper, we systematically and comprehensively build fault models for cameras and LiDAR in AVs and inject them into the MSF perception-based ADS to test its behaviors in test scenarios. To effectively and efficiently explore the parameter spaces of sensor fault models, we design a feedback-guided differential fuzzer to discover the safety violations of MSF perception-based ADS caused by the injected sensor faults. We evaluate FADE on the representative and practical industrial ADS, Baidu Apollo. Our evaluation results demonstrate the effectiveness and efficiency of FADE, and we conclude some useful findings from the experimental results. To validate the findings in the physical world, we use a real Baidu Apollo 6.0 EDU autonomous vehicle to conduct the physical experiments, and the results show the practical significance of our findings. |
| 2025-04-18 | [LangCoop: Collaborative Driving with Language](http://arxiv.org/abs/2504.13406v1) | Xiangbo Gao, Yuheng Wu et al. | Multi-agent collaboration holds great promise for enhancing the safety, reliability, and mobility of autonomous driving systems by enabling information sharing among multiple connected agents. However, existing multi-agent communication approaches are hindered by limitations of existing communication media, including high bandwidth demands, agent heterogeneity, and information loss. To address these challenges, we introduce LangCoop, a new paradigm for collaborative autonomous driving that leverages natural language as a compact yet expressive medium for inter-agent communication. LangCoop features two key innovations: Mixture Model Modular Chain-of-thought (M$^3$CoT) for structured zero-shot vision-language reasoning and Natural Language Information Packaging (LangPack) for efficiently packaging information into concise, language-based messages. Through extensive experiments conducted in the CARLA simulations, we demonstrate that LangCoop achieves a remarkable 96\% reduction in communication bandwidth (< 2KB per message) compared to image-based communication, while maintaining competitive driving performance in the closed-loop evaluation. |
| 2025-04-18 | [Towards a Multi-Agent Vision-Language System for Zero-Shot Novel Hazardous Object Detection for Autonomous Driving Safety](http://arxiv.org/abs/2504.13399v1) | Shashank Shriram, Srinivasa Perisetla et al. | Detecting anomalous hazards in visual data, particularly in video streams, is a critical challenge in autonomous driving. Existing models often struggle with unpredictable, out-of-label hazards due to their reliance on predefined object categories. In this paper, we propose a multimodal approach that integrates vision-language reasoning with zero-shot object detection to improve hazard identification and explanation. Our pipeline consists of a Vision-Language Model (VLM), a Large Language Model (LLM), in order to detect hazardous objects within a traffic scene. We refine object detection by incorporating OpenAI's CLIP model to match predicted hazards with bounding box annotations, improving localization accuracy. To assess model performance, we create a ground truth dataset by denoising and extending the foundational COOOL (Challenge-of-Out-of-Label) anomaly detection benchmark dataset with complete natural language descriptions for hazard annotations. We define a means of hazard detection and labeling evaluation on the extended dataset using cosine similarity. This evaluation considers the semantic similarity between the predicted hazard description and the annotated ground truth for each video. Additionally, we release a set of tools for structuring and managing large-scale hazard detection datasets. Our findings highlight the strengths and limitations of current vision-language-based approaches, offering insights into future improvements in autonomous hazard detection systems. Our models, scripts, and data can be found at https://github.com/mi3labucm/COOOLER.git |
| 2025-04-17 | [Leveraging Functional Encryption and Deep Learning for Privacy-Preserving Traffic Forecasting](http://arxiv.org/abs/2504.13267v1) | Isaac Adom, Mohammmad Iqbal Hossain et al. | Over the past few years, traffic congestion has continuously plagued the nation's transportation system creating several negative impacts including longer travel times, increased pollution rates, and higher collision risks. To overcome these challenges, Intelligent Transportation Systems (ITS) aim to improve mobility and vehicular systems, ensuring higher levels of safety by utilizing cutting-edge technologies, sophisticated sensing capabilities, and innovative algorithms. Drivers' participatory sensing, current/future location reporting, and machine learning algorithms have considerably improved real-time congestion monitoring and future traffic management. However, each driver's sensitive spatiotemporal location information can create serious privacy concerns. To address these challenges, we propose in this paper a secure, privacy-preserving location reporting and traffic forecasting system that guarantees privacy protection of driver data while maintaining high traffic forecasting accuracy. Our novel k-anonymity scheme utilizes functional encryption to aggregate encrypted location information submitted by drivers while ensuring the privacy of driver location data. Additionally, using the aggregated encrypted location information as input, this research proposes a deep learning model that incorporates a Convolutional-Long Short-Term Memory (Conv-LSTM) module to capture spatial and short-term temporal features and a Bidirectional Long Short-Term Memory (Bi-LSTM) module to recover long-term periodic patterns for traffic forecasting. With extensive evaluation on real datasets, we demonstrate the effectiveness of the proposed scheme with less than 10% mean absolute error for a 60-minute forecasting horizon, all while protecting driver privacy. |
| 2025-04-17 | [Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](http://arxiv.org/abs/2504.13169v1) | Tsung-Han Wu, Heekyung Lee et al. | Vision-Language Models (VLMs) excel at visual understanding but often suffer from visual hallucinations, where they generate descriptions of nonexistent objects, actions, or concepts, posing significant risks in safety-critical applications. Existing hallucination mitigation methods typically follow one of two paradigms: generation adjustment, which modifies decoding behavior to align text with visual inputs, and post-hoc verification, where external models assess and correct outputs. While effective, generation adjustment methods often rely on heuristics and lack correction mechanisms, while post-hoc verification is complicated, typically requiring multiple models and tending to reject outputs rather than refine them. In this work, we introduce REVERSE, a unified framework that integrates hallucination-aware training with on-the-fly self-verification. By leveraging a new hallucination-verification dataset containing over 1.3M semi-synthetic samples, along with a novel inference-time retrospective resampling technique, our approach enables VLMs to both detect hallucinations during generation and dynamically revise those hallucinations. Our evaluations show that REVERSE achieves state-of-the-art hallucination reduction, outperforming the best existing methods by up to 12% on CHAIR-MSCOCO and 28% on HaloQuest. Our dataset, model, and code are available at: https://reverse-vlm.github.io. |
| 2025-04-17 | [Energy-Based Reward Models for Robust Language Model Alignment](http://arxiv.org/abs/2504.13134v1) | Anamika Lochab, Ruqi Zhang | Reward models (RMs) are essential for aligning Large Language Models (LLMs) with human preferences. However, they often struggle with capturing complex human preferences and generalizing to unseen data. To address these challenges, we introduce Energy-Based Reward Model (EBRM), a lightweight post-hoc refinement framework that enhances RM robustness and generalization. EBRM models the reward distribution explicitly, capturing uncertainty in human preferences and mitigating the impact of noisy or misaligned annotations. It achieves this through conflict-aware data filtering, label-noise-aware contrastive training, and hybrid initialization. Notably, EBRM enhances RMs without retraining, making it computationally efficient and adaptable across different models and tasks. Empirical evaluations on RM benchmarks demonstrate significant improvements in both robustness and generalization, achieving up to a 5.97% improvement in safety-critical alignment tasks compared to standard RMs. Furthermore, reinforcement learning experiments confirm that our refined rewards enhance alignment quality, effectively delaying reward hacking. These results demonstrate our approach as a scalable and effective enhancement for existing RMs and alignment pipelines. The code is available at EBRM. |
| 2025-04-17 | [LLMs Meet Finance: Fine-Tuning Foundation Models for the Open FinLLM Leaderboard](http://arxiv.org/abs/2504.13125v1) | Varun Rao, Youran Sun et al. | This paper investigates the application of large language models (LLMs) to financial tasks. We fine-tuned foundation models using the Open FinLLM Leaderboard as a benchmark. Building on Qwen2.5 and Deepseek-R1, we employed techniques including supervised fine-tuning (SFT), direct preference optimization (DPO), and reinforcement learning (RL) to enhance their financial capabilities. The fine-tuned models demonstrated substantial performance gains across a wide range of financial tasks. Moreover, we measured the data scaling law in the financial domain. Our work demonstrates the potential of large language models (LLMs) in financial applications. |
| 2025-04-17 | [Accuracy is Not Agreement: Expert-Aligned Evaluation of Crash Narrative Classification Models](http://arxiv.org/abs/2504.13068v1) | Sudesh Ramesh Bhagat, Ibne Farabi Shihab et al. | This study explores the relationship between deep learning (DL) model accuracy and expert agreement in the classification of crash narratives. We evaluate five DL models -- including BERT variants, the Universal Sentence Encoder (USE), and a zero-shot classifier -- against expert-labeled data and narrative text. The analysis is further extended to four large language models (LLMs): GPT-4, LLaMA 3, Qwen, and Claude. Our results reveal a counterintuitive trend: models with higher technical accuracy often exhibit lower agreement with domain experts, whereas LLMs demonstrate greater expert alignment despite relatively lower accuracy scores. To quantify and interpret model-expert agreement, we employ Cohen's Kappa, Principal Component Analysis (PCA), and SHAP-based explainability techniques. Findings indicate that expert-aligned models tend to rely more on contextual and temporal language cues, rather than location-specific keywords. These results underscore that accuracy alone is insufficient for evaluating models in safety-critical NLP applications. We advocate for incorporating expert agreement as a complementary metric in model evaluation frameworks and highlight the promise of LLMs as interpretable, scalable tools for crash analysis pipelines. |
| 2025-04-17 | [GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms](http://arxiv.org/abs/2504.13052v1) | Sinan He, An Wang | Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms. |
| 2025-04-17 | [QI-MPC: A Hybrid Quantum-Inspired Model Predictive Control for Learning Optimal Policies](http://arxiv.org/abs/2504.13041v1) | Muhammad Al-Zafar Khan, Jamal Al-Karaki | In this paper, we present Quantum-Inspired Model Predictive Control (QIMPC), an approach that uses Variational Quantum Circuits (VQCs) to learn control polices in MPC problems. The viability of the approach is tested in five experiments: A target-tracking control strategy, energy-efficient building climate control, autonomous vehicular dynamics, the simple pendulum, and the compound pendulum. Three safety guarantees were established for the approach, and the experiments gave the motivation for two important theoretical results that, in essence, identify systems for which the approach works best. |

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



