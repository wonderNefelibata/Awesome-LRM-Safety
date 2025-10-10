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
| 2025-10-09 | [ArenaBencher: Automatic Benchmark Evolution via Multi-Model Competitive Evaluation](http://arxiv.org/abs/2510.08569v1) | Qin Liu, Jacob Dineen et al. | Benchmarks are central to measuring the capabilities of large language models and guiding model development, yet widespread data leakage from pretraining corpora undermines their validity. Models can match memorized content rather than demonstrate true generalization, which inflates scores, distorts cross-model comparisons, and misrepresents progress. We introduce ArenaBencher, a model-agnostic framework for automatic benchmark evolution that updates test cases while preserving comparability. Given an existing benchmark and a diverse pool of models to be evaluated, ArenaBencher infers the core ability of each test case, generates candidate question-answer pairs that preserve the original objective, verifies correctness and intent with an LLM as a judge, and aggregates feedback from multiple models to select candidates that expose shared weaknesses. The process runs iteratively with in-context demonstrations that steer generation toward more challenging and diagnostic cases. We apply ArenaBencher to math problem solving, commonsense reasoning, and safety domains and show that it produces verified, diverse, and fair updates that uncover new failure modes, increase difficulty while preserving test objective alignment, and improve model separability. The framework provides a scalable path to continuously evolve benchmarks in step with the rapid progress of foundation models. |
| 2025-10-09 | [ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](http://arxiv.org/abs/2510.08562v1) | Zhiyu Zheng, Shaoyu Chen et al. | End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of causal inference, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes the learning task to predict the residual deviation from a deterministic inertial reference. The inertial reference serves as a counterfactual, forcing the model to move beyond simple pattern recognition and instead identify the underlying causal factors (e.g., traffic rules, obstacles) that necessitate deviations from a default, inertially-guided path. To deal with the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. It re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. Extensive experiments validate the effectiveness of our framework. On the NAVSIM benchmark, ResAD achieves a state-of-the-art PDMS of 88.6 using a vanilla diffusion policy with only two denoising steps, demonstrating that our approach significantly simplifies the learning task and improves model performance. The code will be released to facilitate further research. |
| 2025-10-09 | [AI-Driven Post-Quantum Cryptography for Cyber-Resilient V2X Communication in Transportation Cyber-Physical Systems](http://arxiv.org/abs/2510.08496v1) | Akid Abrar, Sagar Dasgupta et al. | Transportation Cyber-Physical Systems (TCPS) integrate physical elements, such as transportation infrastructure and vehicles, with cyber elements via advanced communication technologies, allowing them to interact seamlessly. This integration enhances the efficiency, safety, and sustainability of transportation systems. TCPS rely heavily on cryptographic security to protect sensitive information transmitted between vehicles, transportation infrastructure, and other entities within the transportation ecosystem, ensuring data integrity, confidentiality, and authenticity. Traditional cryptographic methods have been employed to secure TCPS communications, but the advent of quantum computing presents a significant threat to these existing security measures. Therefore, integrating Post-Quantum Cryptography (PQC) into TCPS is essential to maintain secure and resilient communications. While PQC offers a promising approach to developing cryptographic algorithms resistant to quantum attacks, artificial intelligence (AI) can enhance PQC by optimizing algorithm selection, resource allocation, and adapting to evolving threats in real-time. AI-driven PQC approaches can improve the efficiency and effectiveness of PQC implementations, ensuring robust security without compromising system performance. This chapter introduces TCPS communication protocols, discusses the vulnerabilities of corresponding communications to cyber-attacks, and explores the limitations of existing cryptographic methods in the quantum era. By examining how AI can strengthen PQC solutions, the chapter presents cyber-resilient communication strategies for TCPS. |
| 2025-10-09 | [Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered](http://arxiv.org/abs/2510.08464v1) | Jason Jabbour, Dong-Ki Kim et al. | Vision-Language-Action (VLA) models have advanced robotic capabilities but remain challenging to deploy on resource-limited hardware. Pruning has enabled efficient compression of large language models (LLMs), yet it is largely understudied in robotics. Surprisingly, we observe that pruning VLA models leads to drastic degradation and increased safety violations. We introduce GLUESTICK, a post-pruning recovery method that restores much of the original model's functionality while retaining sparsity benefits. Our method performs a one-time interpolation between the dense and pruned models in weight-space to compute a corrective term. This correction is used during inference by each pruned layer to recover lost capabilities with minimal overhead. GLUESTICK requires no additional training, is agnostic to the pruning algorithm, and introduces a single hyperparameter that controls the tradeoff between efficiency and accuracy. Across diverse VLA architectures and tasks in manipulation and navigation, GLUESTICK achieves competitive memory efficiency while substantially recovering success rates and reducing safety violations. Additional material can be found at: https://gluestick-vla.github.io/. |
| 2025-10-09 | [Validation of collision-free spheres of Stewart-Gough platforms for constant orientations using the Application Programming Interface of a CAD software](http://arxiv.org/abs/2510.08408v1) | Bibekananda Patra, Rajeevlochana G. Chittawadigi et al. | This paper presents a method of validation of the size of the largest collision-free sphere (CFS) of a 6-6 Stewart-Gough platform manipulator (SGPM) for a given orientation of its moving platform (MP) using the Application Programming Interface (API) of a CAD software. The position of the MP is updated via the API in an automated manner over a set of samples within a shell enclosing the surface of the CFS. For each pose of the manipulator, each pair of legs is investigated for mutual collisions. The CFS is considered safe or validated iff none of the points falling inside the CFS lead to a collision between any pair of legs. This approach can not only validate the safety of a precomputed CFS, but also estimate the same for any spatial parallel manipulator. |
| 2025-10-09 | [Airy: Reading Robot Intent through Height and Sky](http://arxiv.org/abs/2510.08381v1) | Baoyang Chen, Xian Xu et al. | As industrial robots move into shared human spaces, their opaque decision making threatens safety, trust, and public oversight. This artwork, Airy, asks whether complex multi agent AI can become intuitively understandable by staging a competition between two reinforcement trained robot arms that snap a bedsheet skyward. Building on three design principles, competition as a clear metric (who lifts higher), embodied familiarity (audiences recognize fabric snapping), and sensor to sense mapping (robot cooperation or rivalry shown through forest and weather projections), the installation gives viewers a visceral way to read machine intent. Observations from five international exhibitions indicate that audiences consistently read the robots' strategies, conflict, and cooperation in real time, with emotional reactions that mirror the system's internal state. The project shows how sensory metaphors can turn a black box into a public interface. |
| 2025-10-09 | [Evaluating Small Vision-Language Models on Distance-Dependent Traffic Perception](http://arxiv.org/abs/2510.08352v1) | Nikos Theodoridis, Tim Brophy et al. | Vision-Language Models (VLMs) are becoming increasingly powerful, demonstrating strong performance on a variety of tasks that require both visual and textual understanding. Their strong generalisation abilities make them a promising component for automated driving systems, which must handle unexpected corner cases. However, to be trusted in such safety-critical applications, a model must first possess a reliable perception system. Moreover, since critical objects and agents in traffic scenes are often at a distance, we require systems that are not "shortsighted", i.e., systems with strong perception capabilities at both close (up to 20 meters) and long (30+ meters) range. With this in mind, we introduce Distance-Annotated Traffic Perception Question Answering (DTPQA), the first Visual Question Answering (VQA) benchmark focused solely on perception-based questions in traffic scenes, enriched with distance annotations. By excluding questions that require reasoning, we ensure that model performance reflects perception capabilities alone. Since automated driving hardware has limited processing power and cannot support large VLMs, our study centers on smaller VLMs. More specifically, we evaluate several state-of-the-art (SOTA) small VLMs on DTPQA and show that, despite the simplicity of the questions, these models significantly underperform compared to humans (~60% average accuracy for the best-performing small VLM versus ~85% human performance). However, it is important to note that the human sample size was relatively small, which imposes statistical limitations. We also identify specific perception tasks, such as distinguishing left from right, that remain particularly challenging for these models. |
| 2025-10-09 | [AutoRed: A Free-form Adversarial Prompt Generation Framework for Automated Red Teaming](http://arxiv.org/abs/2510.08329v1) | Muxi Diao, Yutao Mou et al. | The safety of Large Language Models (LLMs) is crucial for the development of trustworthy AI applications. Existing red teaming methods often rely on seed instructions, which limits the semantic diversity of the synthesized adversarial prompts. We propose AutoRed, a free-form adversarial prompt generation framework that removes the need for seed instructions. AutoRed operates in two stages: (1) persona-guided adversarial instruction generation, and (2) a reflection loop to iteratively refine low-quality prompts. To improve efficiency, we introduce a verifier to assess prompt harmfulness without querying the target models. Using AutoRed, we build two red teaming datasets -- AutoRed-Medium and AutoRed-Hard -- and evaluate eight state-of-the-art LLMs. AutoRed achieves higher attack success rates and better generalization than existing baselines. Our results highlight the limitations of seed-based approaches and demonstrate the potential of free-form red teaming for LLM safety evaluation. We will open source our datasets in the near future. |
| 2025-10-09 | [The Alignment Waltz: Jointly Training Agents to Collaborate for Safety](http://arxiv.org/abs/2510.08240v1) | Jingyu Zhang, Haozhu Wang et al. | Harnessing the power of LLMs requires a delicate dance between being helpful and harmless. This creates a fundamental tension between two competing challenges: vulnerability to adversarial attacks that elicit unsafe content, and a tendency for overrefusal on benign but sensitive prompts. Current approaches often navigate this dance with safeguard models that completely reject any content that contains unsafe portions. This approach cuts the music entirely-it may exacerbate overrefusals and fails to provide nuanced guidance for queries it refuses. To teach models a more coordinated choreography, we propose WaltzRL, a novel multi-agent reinforcement learning framework that formulates safety alignment as a collaborative, positive-sum game. WaltzRL jointly trains a conversation agent and a feedback agent, where the latter is incentivized to provide useful suggestions that improve the safety and helpfulness of the conversation agent's responses. At the core of WaltzRL is a Dynamic Improvement Reward (DIR) that evolves over time based on how well the conversation agent incorporates the feedback. At inference time, unsafe or overrefusing responses from the conversation agent are improved rather than discarded. The feedback agent is deployed together with the conversation agent and only engages adaptively when needed, preserving helpfulness and low latency on safe queries. Our experiments, conducted across five diverse datasets, demonstrate that WaltzRL significantly reduces both unsafe responses (e.g., from 39.0% to 4.6% on WildJailbreak) and overrefusals (from 45.3% to 9.9% on OR-Bench) compared to various baselines. By enabling the conversation and feedback agents to co-evolve and adaptively apply feedback, WaltzRL enhances LLM safety without degrading general capabilities, thereby advancing the Pareto front between helpfulness and harmlessness. |
| 2025-10-09 | [LLMs Learn to Deceive Unintentionally: Emergent Misalignment in Dishonesty from Misaligned Samples to Biased Human-AI Interactions](http://arxiv.org/abs/2510.08211v1) | XuHao Hu, Peng Wang et al. | Previous research has shown that LLMs finetuned on malicious or incorrect completions within narrow domains (e.g., insecure code or incorrect medical advice) can become broadly misaligned to exhibit harmful behaviors, which is called emergent misalignment. In this work, we investigate whether this phenomenon can extend beyond safety behaviors to a broader spectrum of dishonesty and deception under high-stakes scenarios (e.g., lying under pressure and deceptive behavior). To explore this, we finetune open-sourced LLMs on misaligned completions across diverse domains. Experimental results demonstrate that LLMs show broadly misaligned behavior in dishonesty. Additionally, we further explore this phenomenon in a downstream combined finetuning setting, and find that introducing as little as 1% of misalignment data into a standard downstream task is sufficient to decrease honest behavior over 20%. Furthermore, we consider a more practical human-AI interaction environment where we simulate both benign and biased users to interact with the assistant LLM. Notably, we find that the assistant can be misaligned unintentionally to exacerbate its dishonesty with only 10% biased user population. In summary, we extend the study of emergent misalignment to the domain of dishonesty and deception under high-stakes scenarios, and demonstrate that this risk arises not only through direct finetuning, but also in downstream mixture tasks and practical human-AI interactions. |
| 2025-10-09 | [Measuring What Matters: The AI Pluralism Index](http://arxiv.org/abs/2510.08193v1) | Rashid Mushkani | Artificial intelligence systems increasingly mediate knowledge, communication, and decision making. Development and governance remain concentrated within a small set of firms and states, raising concerns that technologies may encode narrow interests and limit public agency. Capability benchmarks for language, vision, and coding are common, yet public, auditable measures of pluralistic governance are rare. We define AI pluralism as the degree to which affected stakeholders can shape objectives, data practices, safeguards, and deployment. We present the AI Pluralism Index (AIPI), a transparent, evidence-based instrument that evaluates producers and system families across four pillars: participatory governance, inclusivity and diversity, transparency, and accountability. AIPI codes verifiable practices from public artifacts and independent evaluations, explicitly handling "Unknown" evidence to report both lower-bound ("evidence") and known-only scores with coverage. We formalize the measurement model; implement a reproducible pipeline that integrates structured web and repository analysis, external assessments, and expert interviews; and assess reliability with inter-rater agreement, coverage reporting, cross-index correlations, and sensitivity analysis. The protocol, codebook, scoring scripts, and evidence graph are maintained openly with versioned releases and a public adjudication process. We report pilot provider results and situate AIPI relative to adjacent transparency, safety, and governance frameworks. The index aims to steer incentives toward pluralistic practice and to equip policymakers, procurers, and the public with comparable evidence. |
| 2025-10-09 | [R-Horizon: How Far Can Your Large Reasoning Model Really Go in Breadth and Depth?](http://arxiv.org/abs/2510.08189v1) | Yi Lu, Jianing Wang et al. | Recent trends in test-time scaling for reasoning models (e.g., OpenAI o1, DeepSeek-R1) have led to remarkable improvements through long Chain-of-Thought (CoT). However, existing benchmarks mainly focus on immediate, single-horizon tasks, failing to adequately evaluate models' ability to understand and respond to complex, long-horizon scenarios. To address this incomplete evaluation of Large Reasoning Models (LRMs), we propose R-HORIZON, a method designed to stimulate long-horizon reasoning behaviors in LRMs through query composition. Based on R-HORIZON, we construct a long-horizon reasoning benchmark, comprising complex multi-step reasoning tasks with interdependent problems that span long reasoning horizons. Through comprehensive evaluation of LRMs using the R-HORIZON benchmark, we find that even the most advanced LRMs suffer significant performance degradation. Our analysis reveals that LRMs exhibit limited effective reasoning length and struggle to allocate thinking budget across multiple problems appropriately. Recognizing these limitations, we use R-HORIZON to construct long-horizon reasoning data for reinforcement learning with verified rewards (RLVR). Compared to training with single-horizon data, RLVR with R-HORIZON not only substantially improves performance on the multi-horizon reasoning tasks, but also promotes accuracy on standard reasoning tasks, with an increase of 7.5 on AIME2024. These results position R-HORIZON as a scalable, controllable, and low-cost paradigm for enhancing and evaluating the long-horizon reasoning capabilities of LRMs. |
| 2025-10-09 | [Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs](http://arxiv.org/abs/2510.08158v1) | Shuzhou Yuan, Ercong Nie et al. | Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments. |
| 2025-10-09 | [LinguaSim: Interactive Multi-Vehicle Testing Scenario Generation via Natural Language Instruction Based on Large Language Models](http://arxiv.org/abs/2510.08046v1) | Qingyuan Shi, Qingwen Meng et al. | The generation of testing and training scenarios for autonomous vehicles has drawn significant attention. While Large Language Models (LLMs) have enabled new scenario generation methods, current methods struggle to balance command adherence accuracy with the realism of real-world driving environments. To reduce scenario description complexity, these methods often compromise realism by limiting scenarios to 2D, or open-loop simulations where background vehicles follow predefined, non-interactive behaviors. We propose LinguaSim, an LLM-based framework that converts natural language into realistic, interactive 3D scenarios, ensuring both dynamic vehicle interactions and faithful alignment between the input descriptions and the generated scenarios. A feedback calibration module further refines the generation precision, improving fidelity to user intent. By bridging the gap between natural language and closed-loop, interactive simulations, LinguaSim constrains adversarial vehicle behaviors using both the scenario description and the autonomous driving model guiding them. This framework facilitates the creation of high-fidelity scenarios that enhance safety testing and training. Experiments show LinguaSim can generate scenarios with varying criticality aligned with different natural language descriptions (ACT: 0.072 s for dangerous vs. 3.532 s for safe descriptions; comfortability: 0.654 vs. 0.764), and its refinement module effectively reduces excessive aggressiveness in LinguaSim's initial outputs, lowering the crash rate from 46.9% to 6.3% to better match user intentions. |
| 2025-10-09 | [Verifying Graph Neural Networks with Readout is Intractable](http://arxiv.org/abs/2510.08045v1) | Artem Chernobrovkin, Marco Sälzer et al. | We introduce a logical language for reasoning about quantized aggregate-combine graph neural networks with global readout (ACR-GNNs). We provide a logical characterization and use it to prove that verification tasks for quantized GNNs with readout are (co)NEXPTIME-complete. This result implies that the verification of quantized GNNs is computationally intractable, prompting substantial research efforts toward ensuring the safety of GNN-based systems. We also experimentally demonstrate that quantized ACR-GNN models are lightweight while maintaining good accuracy and generalization capabilities with respect to non-quantized models. |
| 2025-10-09 | [Fewer Weights, More Problems: A Practical Attack on LLM Pruning](http://arxiv.org/abs/2510.07985v1) | Kazuki Egashira, Robin Staab et al. | Model pruning, i.e., removing a subset of model weights, has become a prominent approach to reducing the memory footprint of large language models (LLMs) during inference. Notably, popular inference engines, such as vLLM, enable users to conveniently prune downloaded models before they are deployed. While the utility and efficiency of pruning methods have improved significantly, the security implications of pruning remain underexplored. In this work, for the first time, we show that modern LLM pruning methods can be maliciously exploited. In particular, an adversary can construct a model that appears benign yet, once pruned, exhibits malicious behaviors. Our method is based on the idea that the adversary can compute a proxy metric that estimates how likely each parameter is to be pruned. With this information, the adversary can first inject a malicious behavior into those parameters that are unlikely to be pruned. Then, they can repair the model by using parameters that are likely to be pruned, effectively canceling out the injected behavior in the unpruned model. We demonstrate the severity of our attack through extensive evaluation on five models; after any of the pruning in vLLM are applied (Magnitude, Wanda, and SparseGPT), it consistently exhibits strong malicious behaviors in a diverse set of attack scenarios (success rates of up to $95.7\%$ for jailbreak, $98.7\%$ for benign instruction refusal, and $99.5\%$ for targeted content injection). Our results reveal a critical deployment-time security gap and underscore the urgent need for stronger security awareness in model compression. |
| 2025-10-09 | [IntMeanFlow: Few-step Speech Generation with Integral Velocity Distillation](http://arxiv.org/abs/2510.07979v1) | Wei Wang, Rong Cao et al. | Flow-based generative models have greatly improved text-to-speech (TTS) synthesis quality, but inference speed remains limited by the iterative sampling process and multiple function evaluations (NFE). The recent MeanFlow model accelerates generation by modeling average velocity instead of instantaneous velocity. However, its direct application to TTS encounters challenges, including GPU memory overhead from Jacobian-vector products (JVP) and training instability due to self-bootstrap processes. To address these issues, we introduce IntMeanFlow, a framework for few-step speech generation with integral velocity distillation. By approximating average velocity with the teacher's instantaneous velocity over a temporal interval, IntMeanFlow eliminates the need for JVPs and self-bootstrap, improving stability and reducing GPU memory usage. We also propose the Optimal Step Sampling Search (O3S) algorithm, which identifies the model-specific optimal sampling steps, improving speech synthesis without additional inference overhead. Experiments show that IntMeanFlow achieves 1-NFE inference for token-to-spectrogram and 3-NFE for text-to-spectrogram tasks while maintaining high-quality synthesis. Demo samples are available at https://vvwangvv.github.io/intmeanflow. |
| 2025-10-09 | [VoiceAgentBench: Are Voice Assistants ready for agentic tasks?](http://arxiv.org/abs/2510.07978v1) | Dhruv Jain, Harshit Shukla et al. | Large-scale Speech Language Models (SpeechLMs) have enabled voice assistants capable of understanding natural spoken queries and performing complex tasks. However, existing speech benchmarks primarily focus on isolated capabilities such as transcription, or question-answering, and do not systematically evaluate agentic scenarios encompassing multilingual and cultural understanding, as well as adversarial robustness. To address this, we introduce VoiceAgentBench, a comprehensive benchmark designed to evaluate SpeechLMs in realistic spoken agentic settings. It comprises over 5,500 synthetic spoken queries, including dialogues grounded in Indian context, covering single-tool invocations, multi-tool workflows, multi-turn interactions, and safety evaluations. The benchmark supports English, Hindi, and 5 other Indian languages, reflecting real-world linguistic and cultural diversity. We simulate speaker variability using a novel sampling algorithm that selects audios for TTS voice conversion based on its speaker embeddings, maximizing acoustic and speaker diversity. Our evaluation measures tool selection accuracy, structural consistency, and the correctness of tool invocations, including adversarial robustness. Our experiments reveal significant gaps in contextual tool orchestration tasks, Indic generalization, and adversarial robustness, exposing critical limitations of current SpeechLMs. |
| 2025-10-09 | [Active Confusion Expression in Large Language Models: Leveraging World Models toward Better Social Reasoning](http://arxiv.org/abs/2510.07974v1) | Jialu Du, Guiyang Hou et al. | While large language models (LLMs) excel in mathematical and code reasoning, we observe they struggle with social reasoning tasks, exhibiting cognitive confusion, logical inconsistencies, and conflation between objective world states and subjective belief states. Through deteiled analysis of DeepSeek-R1's reasoning trajectories, we find that LLMs frequently encounter reasoning impasses and tend to output contradictory terms like "tricky" and "confused" when processing scenarios with multiple participants and timelines, leading to erroneous reasoning or infinite loops. The core issue is their inability to disentangle objective reality from agents' subjective beliefs. To address this, we propose an adaptive world model-enhanced reasoning mechanism that constructs a dynamic textual world model to track entity states and temporal sequences. It dynamically monitors reasoning trajectories for confusion indicators and promptly intervenes by providing clear world state descriptions, helping models navigate through cognitive dilemmas. The mechanism mimics how humans use implicit world models to distinguish between external events and internal beliefs. Evaluations on three social benchmarks demonstrate significant improvements in accuracy (e.g., +10% in Hi-ToM) while reducing computational costs (up to 33.8% token reduction), offering a simple yet effective solution for deploying LLMs in social contexts. |
| 2025-10-09 | [From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses](http://arxiv.org/abs/2510.07968v1) | Xiangtao Meng, Tianshuo Cong et al. | Large Language Models (LLMs) have shown remarkable performance across various applications, but their deployment in sensitive domains raises significant concerns. To mitigate these risks, numerous defense strategies have been proposed. However, most existing studies assess these defenses in isolation, overlooking their broader impacts across other risk dimensions. In this work, we take the first step in investigating unintended interactions caused by defenses in LLMs, focusing on the complex interplay between safety, fairness, and privacy. Specifically, we propose CrossRiskEval, a comprehensive evaluation framework to assess whether deploying a defense targeting one risk inadvertently affects others. Through extensive empirical studies on 14 defense-deployed LLMs, covering 12 distinct defense strategies, we reveal several alarming side effects: 1) safety defenses may suppress direct responses to sensitive queries related to bias or privacy, yet still amplify indirect privacy leakage or biased outputs; 2) fairness defenses increase the risk of misuse and privacy leakage; 3) privacy defenses often impair safety and exacerbate bias. We further conduct a fine-grained neuron-level analysis to uncover the underlying mechanisms of these phenomena. Our analysis reveals the existence of conflict-entangled neurons in LLMs that exhibit opposing sensitivities across multiple risk dimensions. Further trend consistency analysis at both task and neuron levels confirms that these neurons play a key role in mediating the emergence of unintended behaviors following defense deployment. We call for a paradigm shift in LLM risk evaluation, toward holistic, interaction-aware assessment of defense strategies. |

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



