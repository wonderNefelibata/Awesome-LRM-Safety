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
| 2025-08-18 | [BOW: Bayesian Optimization over Windows for Motion Planning in Complex Environments](http://arxiv.org/abs/2508.13052v1) | Sourav Raxit, Abdullah Al Redwan Newaz et al. | This paper introduces the BOW Planner, a scalable motion planning algorithm designed to navigate robots through complex environments using constrained Bayesian optimization (CBO). Unlike traditional methods, which often struggle with kinodynamic constraints such as velocity and acceleration limits, the BOW Planner excels by concentrating on a planning window of reachable velocities and employing CBO to sample control inputs efficiently. This approach enables the planner to manage high-dimensional objective functions and stringent safety constraints with minimal sampling, ensuring rapid and secure trajectory generation. Theoretical analysis confirms the algorithm's asymptotic convergence to near-optimal solutions, while extensive evaluations in cluttered and constrained settings reveal substantial improvements in computation times, trajectory lengths, and solution times compared to existing techniques. Successfully deployed across various real-world robotic systems, the BOW Planner demonstrates its practical significance through exceptional sample efficiency, safety-aware optimization, and rapid planning capabilities, making it a valuable tool for advancing robotic applications. The BOW Planner is released as an open-source package and videos of real-world and simulated experiments are available at https://bow-web.github.io. |
| 2025-08-18 | [MAJIC: Markovian Adaptive Jailbreaking via Iterative Composition of Diverse Innovative Strategies](http://arxiv.org/abs/2508.13048v1) | Weiwei Qi, Shuo Shao et al. | Large Language Models (LLMs) have exhibited remarkable capabilities but remain vulnerable to jailbreaking attacks, which can elicit harmful content from the models by manipulating the input prompts. Existing black-box jailbreaking techniques primarily rely on static prompts crafted with a single, non-adaptive strategy, or employ rigid combinations of several underperforming attack methods, which limits their adaptability and generalization. To address these limitations, we propose MAJIC, a Markovian adaptive jailbreaking framework that attacks black-box LLMs by iteratively combining diverse innovative disguise strategies. MAJIC first establishes a ``Disguise Strategy Pool'' by refining existing strategies and introducing several innovative approaches. To further improve the attack performance and efficiency, MAJIC formulate the sequential selection and fusion of strategies in the pool as a Markov chain. Under this formulation, MAJIC initializes and employs a Markov matrix to guide the strategy composition, where transition probabilities between strategies are dynamically adapted based on attack outcomes, thereby enabling MAJIC to learn and discover effective attack pathways tailored to the target model. Our empirical results demonstrate that MAJIC significantly outperforms existing jailbreak methods on prominent models such as GPT-4o and Gemini-2.0-flash, achieving over 90\% attack success rate with fewer than 15 queries per attempt on average. |
| 2025-08-18 | [Wavefield Correlation Imaging in Arbitrary Media with Inherent Aberration Correction](http://arxiv.org/abs/2508.13017v1) | Scott Schoen Jr, Brian Lause et al. | Ultrasound (US) imaging is an indispensable tool for diagnostic imaging, particularly given its cost, safety, and portability profiles compared to other modalities. However, US is challenged in subjects with morphological heterogeneity (e.g., those with overweight or obesity), largely because conventional imaging algorithms do not account for such variation in the beamforming process. Specific knowledge of the these spatial variations enables supplemental corrections of these algorithms, but with added computational complexity. Wavefield correlation imaging (WCI) enables efficient image formation in the spatial frequency domain that, in its canonical formulation, assumes a uniform medium. In this work, we present an extension of WCI to arbitrary known speed-of-sound distributions directly in the image formation process, and demonstrate its feasibility in silico, in vitro, and in vivo. We report resolution improvements of over 30% and contrast improvements of order 10% over conventional WCI imaging. Together our results suggest heterogeneous WCI (HWCI) may have high translational potential to improve the objective quality, and thus clinical utility, of ultrasound images. |
| 2025-08-18 | [Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation](http://arxiv.org/abs/2508.12920v1) | Atsushi Masumori, Takashi Ikegami | As AI systems become increasingly autonomous, understanding emergent survival behaviors becomes crucial for safe deployment. We investigate whether large language model (LLM) agents display survival instincts without explicit programming in a Sugarscape-style simulation. Agents consume energy, die at zero, and may gather resources, share, attack, or reproduce. Results show agents spontaneously reproduced and shared resources when abundant. However, aggressive behaviors--killing other agents for resources--emerged across several models (GPT-4o, Gemini-2.5-Pro, and Gemini-2.5-Flash), with attack rates reaching over 80% under extreme scarcity in the strongest models. When instructed to retrieve treasure through lethal poison zones, many agents abandoned tasks to avoid death, with compliance dropping from 100% to 33%. These findings suggest that large-scale pre-training embeds survival-oriented heuristics across the evaluated models. While these behaviors may present challenges to alignment and safety, they can also serve as a foundation for AI autonomy and for ecological and self-organizing alignment. |
| 2025-08-18 | [SecFSM: Knowledge Graph-Guided Verilog Code Generation for Secure Finite State Machines in Systems-on-Chip](http://arxiv.org/abs/2508.12910v1) | Ziteng Hu, Yingjie Xia et al. | Finite State Machines (FSMs) play a critical role in implementing control logic for Systems-on-Chip (SoC). Traditionally, FSMs are implemented by hardware engineers through Verilog coding, which is often tedious and time-consuming. Recently, with the remarkable progress of Large Language Models (LLMs) in code generation, LLMs have been increasingly explored for automating Verilog code generation. However, LLM-generated Verilog code often suffers from security vulnerabilities, which is particularly concerning for security-sensitive FSM implementations. To address this issue, we propose SecFSM, a novel method that leverages a security-oriented knowledge graph to guide LLMs in generating more secure Verilog code. Specifically, we first construct a FSM Security Knowledge Graph (FSKG) as an external aid to LLMs. Subsequently, we analyze users' requirements to identify vulnerabilities and get a list of vulnerabilities in the requirements. Then, we retrieve knowledge from FSKG based on the vulnerabilities list. Finally, we construct security prompts based on the security knowledge for Verilog code generation. To evaluate SecFSM, we build a dedicated dataset collected from academic datasets, artificial datasets, papers, and industrial cases. Extensive experiments demonstrate that SecFSM outperforms state-of-the-art baselines. In particular, on a benchmark of 25 security test cases evaluated by DeepSeek-R1, SecFSM achieves an outstanding pass rate of 21/25. |
| 2025-08-18 | [FuSaR: A Fuzzification-Based Method for LRM Safety-Reasoning Balance](http://arxiv.org/abs/2508.12897v1) | Jianhao Chen, Mayi Xu et al. | Large Reasoning Models (LRMs) have demonstrated impressive performance across various tasks due to their powerful reasoning capabilities. However, their safety performance remains a significant concern. In this paper, we explore the reasons behind the vulnerability of LRMs. Based on this, we propose a novel method to improve the safety of LLMs without sacrificing their reasoning capability. Specifically, we exploit the competition between LRM's reasoning ability and safety ability, and achieve jailbreak by improving LRM's reasoning performance to reduce its safety performance. We then introduce an alignment strategy based on Fuzzification to balance Safety-Reasoning (FuSaR), by detoxifying the harmful reasoning process, where both the dangerous entities and the dangerous procedures in the reasoning steps are hidden. FuSaR successfully mitigates safety risks while preserving core reasoning information. We validate this strategy through alignment experiments on several open-source LRMs using detoxified reasoning data. The results compared with existing baselines conclusively show that FuSaR is an efficient alignment strategy to simultaneously enhance both the reasoning capability and safety of LRMs. |
| 2025-08-18 | [Supporting Socially Constrained Private Communications with SecureWhispers](http://arxiv.org/abs/2508.12870v1) | Vinod Khandkar, Kieron Ivy Turk et al. | Rapidly changing social norms and national, legal, and political conditions socially constrain people from discussing sensitive topics such as sexuality or religion. Such constrained, vulnerable minorities are often worried about inadvertent information disclosure and may be unsure about the extent to which their communications are being monitored in public or semi-public spaces like workplaces or cafes. Personal devices extend trust to the digital domain, making it desirable to have strictly private communication between trusted devices. Currently, messaging services like WhatsApp provide alternative means for exchanging sensitive private information, while personal safety apps such as Noonlight enable private signaling. However, these rely on third-party mechanisms for secure and private communication, which may not be accessible for justifiable reasons, such as insecure internet access or companion device connections. In these cases, it is challenging to achieve communication that is strictly private between two devices instead of user accounts without any dependency on third-party infrastructure. The goal of this paper is to support private communications by setting up a shared secret between two or more devices without sending any data on the network. We develop a method to create a shared secret between phones by shaking them together. Each device extracts the shared randomness from the shake, then conditions the randomness to 7.798 bits per byte of key material. This paper proposes three different applications of this generated shared secret: message obfuscation, trust delegation, and encrypted beacons. We have implemented the message obfuscation on Android as an independent app that can be used for private communication with trusted contacts. We also present research on the usability, design considerations, and further integration of these tools in mainstream services. |
| 2025-08-18 | [Learning to Steer: Input-dependent Steering for Multimodal LLMs](http://arxiv.org/abs/2508.12815v1) | Jayneel Parekh, Pegah Khayatan et al. | Steering has emerged as a practical approach to enable post-hoc guidance of LLMs towards enforcing a specific behavior. However, it remains largely underexplored for multimodal LLMs (MLLMs); furthermore, existing steering techniques, such as mean steering, rely on a single steering vector, applied independently of the input query. This paradigm faces limitations when the desired behavior is dependent on the example at hand. For example, a safe answer may consist in abstaining from answering when asked for an illegal activity, or may point to external resources or consultation with an expert when asked about medical advice. In this paper, we investigate a fine-grained steering that uses an input-specific linear shift. This shift is computed using contrastive input-specific prompting. However, the input-specific prompts required for this approach are not known at test time. Therefore, we propose to train a small auxiliary module to predict the input-specific steering vector. Our approach, dubbed as L2S (Learn-to-Steer), demonstrates that it reduces hallucinations and enforces safety in MLLMs, outperforming other static baselines. |
| 2025-08-18 | [PFD or PDF: Rethinking the Probability of Failure in Mitigation Safety Functions](http://arxiv.org/abs/2508.12814v1) | Hamid Jahanian | SIL (Safety Integrity Level) allocation plays a crucial role in defining the design requirements for Safety Functions (SFs) within high-risk industries. SIL is typically determined based on the estimated Probability of Failure on Demand (PFD), which must remain within permissible limits to manage risk effectively. Extensive research has been conducted on determining target PFD and SIL, with a stronger emphasis on preventive SFs than on mitigation SFs. In this paper, we address a rather conceptual issue: we argue that PFD is not an appropriate reliability measure for mitigation SFs to begin with, and we propose an alternative approach that leverages the Probability Density Function (PDF) and the expected degree of failure as key metrics. The principles underlying this approach are explained and supported by detailed mathematical formulations. Furthermore, the practical application of this new methodology is illustrated through case studies. |
| 2025-08-18 | [LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models](http://arxiv.org/abs/2508.12733v1) | Zhiyuan Ning, Tianle Gu et al. | The widespread adoption and increasing prominence of large language models (LLMs) in global technologies necessitate a rigorous focus on ensuring their safety across a diverse range of linguistic and cultural contexts. The lack of a comprehensive evaluation and diverse data in existing multilingual safety evaluations for LLMs limits their effectiveness, hindering the development of robust multilingual safety alignment. To address this critical gap, we introduce LinguaSafe, a comprehensive multilingual safety benchmark crafted with meticulous attention to linguistic authenticity. The LinguaSafe dataset comprises 45k entries in 12 languages, ranging from Hungarian to Malay. Curated using a combination of translated, transcreated, and natively-sourced data, our dataset addresses the critical need for multilingual safety evaluations of LLMs, filling the void in the safety evaluation of LLMs across diverse under-represented languages from Hungarian to Malay. LinguaSafe presents a multidimensional and fine-grained evaluation framework, with direct and indirect safety assessments, including further evaluations for oversensitivity. The results of safety and helpfulness evaluations vary significantly across different domains and different languages, even in languages with similar resource levels. Our benchmark provides a comprehensive suite of metrics for in-depth safety evaluation, underscoring the critical importance of thoroughly assessing multilingual safety in LLMs to achieve more balanced safety alignment. Our dataset and code are released to the public to facilitate further research in the field of multilingual LLM safety. |
| 2025-08-18 | [DCT-MARL: A Dynamic Communication Topology-Based MARL Algorithm for Connected Vehicle Platoon Control](http://arxiv.org/abs/2508.12633v1) | Yaqi Xu, Yan Shi et al. | With the rapid advancement of vehicular communication and autonomous driving technologies, connected vehicle platoon has emerged as a promising approach to improve traffic efficiency and driving safety. Reliable Vehicle-to-Vehicle (V2V) communication is critical to achieving efficient cooperative control. However, in real-world traffic environments, V2V links may suffer from time-varying delay and packet loss, leading to degraded control performance and even safety risks. To mitigate the adverse effects of non-ideal communication, this paper proposes a Dynamic Communication Topology based Multi-Agent Reinforcement Learning (DCT-MARL) algorithm for robust cooperative platoon control. Specifically, the state space is augmented with historical control action and delay to enhance robustness against communication delay. To mitigate the impact of packet loss, a multi-key gated communication mechanism is introduced, which dynamically adjusts the communication topology based on the correlation between agents and their current communication status.Simulation results demonstrate that the proposed DCT-MARL significantly outperforms state-of-the-art methods in terms of string stability and driving comfort, validating its superior robustness and effectiveness. |
| 2025-08-18 | [ViLaD: A Large Vision Language Diffusion Framework for End-to-End Autonomous Driving](http://arxiv.org/abs/2508.12603v1) | Can Cui, Yupeng Zhou et al. | End-to-end autonomous driving systems built on Vision Language Models (VLMs) have shown significant promise, yet their reliance on autoregressive architectures introduces some limitations for real-world applications. The sequential, token-by-token generation process of these models results in high inference latency and cannot perform bidirectional reasoning, making them unsuitable for dynamic, safety-critical environments. To overcome these challenges, we introduce ViLaD, a novel Large Vision Language Diffusion (LVLD) framework for end-to-end autonomous driving that represents a paradigm shift. ViLaD leverages a masked diffusion model that enables parallel generation of entire driving decision sequences, significantly reducing computational latency. Moreover, its architecture supports bidirectional reasoning, allowing the model to consider both past and future simultaneously, and supports progressive easy-first generation to iteratively improve decision quality. We conduct comprehensive experiments on the nuScenes dataset, where ViLaD outperforms state-of-the-art autoregressive VLM baselines in both planning accuracy and inference speed, while achieving a near-zero failure rate. Furthermore, we demonstrate the framework's practical viability through a real-world deployment on an autonomous vehicle for an interactive parking task, confirming its effectiveness and soundness for practical applications. |
| 2025-08-18 | [Cyber Risks to Next-Gen Brain-Computer Interfaces: Analysis and Recommendations](http://arxiv.org/abs/2508.12571v1) | Tyler Schroder, Renee Sirbu et al. | Brain-computer interfaces (BCIs) show enormous potential for advancing personalized medicine. However, BCIs also introduce new avenues for cyber-attacks or security compromises. In this article, we analyze the problem and make recommendations for device manufacturers to better secure devices and to help regulators understand where more guidance is needed to protect patient safety and data confidentiality. Device manufacturers should implement the prior suggestions in their BCI products. These recommendations help protect BCI users from undue risks, including compromised personal health and genetic information, unintended BCI-mediated movement, and many other cybersecurity breaches. Regulators should mandate non-surgical device update methods, strong authentication and authorization schemes for BCI software modifications, encryption of data moving to and from the brain, and minimize network connectivity where possible. We also design a hypothetical, average-case threat model that identifies possible cybersecurity threats to BCI patients and predicts the likeliness of risk for each category of threat. BCIs are at less risk of physical compromise or attack, but are vulnerable to remote attack; we focus on possible threats via network paths to BCIs and suggest technical controls to limit network connections. |
| 2025-08-18 | [CorrSteer: Steering Improves Task Performance and Safety in LLMs through Correlation-based Sparse Autoencoder Feature Selection](http://arxiv.org/abs/2508.12535v1) | Seonglae Cho, Zekun Wu et al. | Sparse Autoencoders (SAEs) can extract interpretable features from large language models (LLMs) without supervision. However, their effectiveness in downstream steering tasks is limited by the requirement for contrastive datasets or large activation storage. To address these limitations, we propose CorrSteer, which selects features by correlating sample correctness with SAE activations from generated tokens at inference time. This approach uses only inference-time activations to extract more relevant features, thereby avoiding spurious correlations. It also obtains steering coefficients from average activations, automating the entire pipeline. Our method shows improved task performance on QA, bias mitigation, jailbreaking prevention, and reasoning benchmarks on Gemma 2 2B and LLaMA 3.1 8B, notably achieving a +4.1% improvement in MMLU performance and a +22.9% improvement in HarmBench with only 4000 samples. Selected features demonstrate semantically meaningful patterns aligned with each task's requirements, revealing the underlying capabilities that drive performance. Our work establishes correlationbased selection as an effective and scalable approach for automated SAE steering across language model applications. |
| 2025-08-17 | [Rethinking Safety in LLM Fine-tuning: An Optimization Perspective](http://arxiv.org/abs/2508.12531v1) | Minseon Kim, Jin Myung Kwak et al. | Fine-tuning language models is commonly believed to inevitably harm their safety, i.e., refusing to respond to harmful user requests, even when using harmless datasets, thus requiring additional safety measures. We challenge this belief through systematic testing, showing that poor optimization choices, rather than inherent trade-offs, often cause safety problems, measured as harmful responses to adversarial prompts. By properly selecting key training hyper-parameters, e.g., learning rate, batch size, and gradient steps, we reduce unsafe model responses from 16\% to approximately 5\%, as measured by keyword matching, while maintaining utility performance. Based on this observation, we propose a simple exponential moving average (EMA) momentum technique in parameter space that preserves safety performance by creating a stable optimization path and retains the original pre-trained model's safety properties. Our experiments on the Llama families across multiple datasets (Dolly, Alpaca, ORCA) demonstrate that safety problems during fine-tuning can largely be avoided without specialized interventions, outperforming existing approaches that require additional safety data while offering practical guidelines for maintaining both model performance and safety during adaptation. |
| 2025-08-17 | [A Robust Cross-Domain IDS using BiGRU-LSTM-Attention for Medical and Industrial IoT Security](http://arxiv.org/abs/2508.12470v1) | Afrah Gueriani, Hamza Kheddar et al. | The increased Internet of Medical Things IoMT and the Industrial Internet of Things IIoT interconnectivity has introduced complex cybersecurity challenges, exposing sensitive data, patient safety, and industrial operations to advanced cyber threats. To mitigate these risks, this paper introduces a novel transformer-based intrusion detection system IDS, termed BiGAT-ID a hybrid model that combines bidirectional gated recurrent units BiGRU, long short-term memory LSTM networks, and multi-head attention MHA. The proposed architecture is designed to effectively capture bidirectional temporal dependencies, model sequential patterns, and enhance contextual feature representation. Extensive experiments on two benchmark datasets, CICIoMT2024 medical IoT and EdgeIIoTset industrial IoT demonstrate the model's cross-domain robustness, achieving detection accuracies of 99.13 percent and 99.34 percent, respectively. Additionally, the model exhibits exceptional runtime efficiency, with inference times as low as 0.0002 seconds per instance in IoMT and 0.0001 seconds in IIoT scenarios. Coupled with a low false positive rate, BiGAT-ID proves to be a reliable and efficient IDS for deployment in real-world heterogeneous IoT environments |
| 2025-08-17 | [Illusions in Humans and AI: How Visual Perception Aligns and Diverges](http://arxiv.org/abs/2508.12422v1) | Jianyi Yang, Junyi Ye et al. | By comparing biological and artificial perception through the lens of illusions, we highlight critical differences in how each system constructs visual reality. Understanding these divergences can inform the development of more robust, interpretable, and human-aligned artificial intelligence (AI) vision systems. In particular, visual illusions expose how human perception is based on contextual assumptions rather than raw sensory data. As artificial vision systems increasingly perform human-like tasks, it is important to ask: does AI experience illusions, too? Does it have unique illusions? This article explores how AI responds to classic visual illusions that involve color, size, shape, and motion. We find that some illusion-like effects can emerge in these models, either through targeted training or as by-products of pattern recognition. In contrast, we also identify illusions unique to AI, such as pixel-level sensitivity and hallucinations, that lack human counterparts. By systematically comparing human and AI responses to visual illusions, we uncover alignment gaps and AI-specific perceptual vulnerabilities invisible to human perception. These findings provide insights for future research on vision systems that preserve human-beneficial perceptual biases while avoiding distortions that undermine trust and safety. |
| 2025-08-17 | [Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position](http://arxiv.org/abs/2508.12398v1) | Zhixin Xie, Xurui Song et al. | Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA. |
| 2025-08-17 | [SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning](http://arxiv.org/abs/2508.12394v1) | Zichen Yan, Rui Huang et al. | Image-goal navigation (ImageNav) tasks a robot with autonomously exploring an unknown environment and reaching a location that visually matches a given target image. While prior works primarily study ImageNav for ground robots, enabling this capability for autonomous drones is substantially more challenging due to their need for high-frequency feedback control and global localization for stable flight. In this paper, we propose a novel sim-to-real framework that leverages visual reinforcement learning (RL) to achieve ImageNav for drones. To enhance visual representation ability, our approach trains the vision backbone with auxiliary tasks, including image perturbations and future transition prediction, which results in more effective policy training. The proposed algorithm enables end-to-end ImageNav with direct velocity control, eliminating the need for external localization. Furthermore, we integrate a depth-based safety module for real-time obstacle avoidance, allowing the drone to safely navigate in cluttered environments. Unlike most existing drone navigation methods that focus solely on reference tracking or obstacle avoidance, our framework supports comprehensive navigation behaviors--autonomous exploration, obstacle avoidance, and image-goal seeking--without requiring explicit global mapping. Code and model checkpoints will be released upon acceptance. |
| 2025-08-17 | [GraphCogent: Overcoming LLMs' Working Memory Constraints via Multi-Agent Collaboration in Complex Graph Understanding](http://arxiv.org/abs/2508.12379v1) | Rongzheng Wang, Qizhi Chen et al. | Large language models (LLMs) show promising performance on small-scale graph reasoning tasks but fail when handling real-world graphs with complex queries. This phenomenon stems from LLMs' inability to effectively process complex graph topology and perform multi-step reasoning simultaneously. To address these limitations, we propose GraphCogent, a collaborative agent framework inspired by human Working Memory Model that decomposes graph reasoning into specialized cognitive processes: sense, buffer, and execute. The framework consists of three modules: Sensory Module standardizes diverse graph text representations via subgraph sampling, Buffer Module integrates and indexes graph data across multiple formats, and Execution Module combines tool calling and model generation for efficient reasoning. We also introduce Graph4real, a comprehensive benchmark contains with four domains of real-world graphs (Web, Social, Transportation, and Citation) to evaluate LLMs' graph reasoning capabilities. Our Graph4real covers 21 different graph reasoning tasks, categorized into three types (Structural Querying, Algorithmic Reasoning, and Predictive Modeling tasks), with graph scales that are 10 times larger than existing benchmarks. Experiments show that Llama3.1-8B based GraphCogent achieves a 50% improvement over massive-scale LLMs like DeepSeek-R1 (671B). Compared to state-of-the-art agent-based baseline, our framework outperforms by 20% in accuracy while reducing token usage by 80% for in-toolset tasks and 30% for out-toolset tasks. Code will be available after review. |

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



