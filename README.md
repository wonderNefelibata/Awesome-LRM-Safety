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
| 2025-11-19 | [Optimus-Q: Utilizing Federated Learning in Adaptive Robots for Intelligent Nuclear Power Plant Operations through Quantum Cryptography](http://arxiv.org/abs/2511.15614v1) | Sai Puppala, Ismail Hossain et al. | The integration of advanced robotics in nuclear power plants (NPPs) presents a transformative opportunity to enhance safety, efficiency, and environmental monitoring in high-stakes environments. Our paper introduces the Optimus-Q robot, a sophisticated system designed to autonomously monitor air quality and detect contamination while leveraging adaptive learning techniques and secure quantum communication. Equipped with advanced infrared sensors, the Optimus-Q robot continuously streams real-time environmental data to predict hazardous gas emissions, including carbon dioxide (CO$_2$), carbon monoxide (CO), and methane (CH$_4$). Utilizing a federated learning approach, the robot collaborates with other systems across various NPPs to improve its predictive capabilities without compromising data privacy. Additionally, the implementation of Quantum Key Distribution (QKD) ensures secure data transmission, safeguarding sensitive operational information. Our methodology combines systematic navigation patterns with machine learning algorithms to facilitate efficient coverage of designated areas, thereby optimizing contamination monitoring processes. Through simulations and real-world experiments, we demonstrate the effectiveness of the Optimus-Q robot in enhancing operational safety and responsiveness in nuclear facilities. This research underscores the potential of integrating robotics, machine learning, and quantum technologies to revolutionize monitoring systems in hazardous environments. |
| 2025-11-19 | [EPSO: A Caching-Based Efficient Superoptimizer for BPF Bytecode](http://arxiv.org/abs/2511.15589v1) | Qian Zhu, Yuxuan Liu et al. | Extended Berkeley Packet Filter (eBPF) allows developers to extend Linux kernel functionality without modifying its source code. To ensure system safety, an in-kernel safety checker, the verifier, enforces strict safety constraints (for example, a limited program size) on eBPF programs loaded into the kernel. These constraints, combined with eBPF's performance-critical use cases, make effective optimization essential. However, existing compilers (such as Clang) offer limited optimization support, and many semantics-preserving transformations are rejected by the verifier, which makes handcrafted optimization rule design both challenging and limited in effectiveness. Superoptimization overcomes the limitations of rule-based methods by automatically discovering optimal transformations, but its high computational cost limits scalability. To address this, we propose EPSO, a caching-based superoptimizer that discovers rewrite rules via offline superoptimization and reuses them to achieve high-quality optimizations with minimal runtime overhead. We evaluate EPSO on benchmarks from the Linux kernel and several eBPF-based projects, including Cilium, Katran, hXDP, Sysdig, Tetragon, and Tracee. EPSO discovers 795 rewrite rules and achieves up to 68.87 percent (average 24.37 percent) reduction in program size compared to Clang's output, outperforming the state-of-the-art BPF optimizer K2 on all benchmarks and Merlin on 92.68 percent of them. Additionally, EPSO reduces program runtime by an average of 6.60 percent, improving throughput and lowering latency in network applications. |
| 2025-11-19 | [Real-Time Optimal Control via Transformer Networks and Bernstein Polynomials](http://arxiv.org/abs/2511.15588v1) | Gage MacLin, Venanzio Cichella et al. | In this paper, we propose a Transformer-based framework for approximating solutions to infinite-dimensional optimization problems: calculus of variations problems and optimal control problems. Our approach leverages offline training on data generated by solving a sample of infinite- dimensional optimization problems using composite Bernstein collocation. Once trained, the Transformer efficiently generates near-optimal, feasible trajectories, making it well-suited for real-time applications. In motion planning for autonomous vehicles, for instance, these trajectories can serve to warm- start optimal motion planners or undergo rigorous evaluation to ensure safety. We demonstrate the effectiveness of this method through numerical results on a classical control problem and an online obstacle avoidance task. This data-driven approach offers a promising solution for real-time optimal control of nonlinear, nonconvex systems. |
| 2025-11-19 | [A Review of Machine Learning for Cavitation Intensity Recognition in Complex Industrial Systems](http://arxiv.org/abs/2511.15497v1) | Yu Sha, Ningtao Liu et al. | Cavitation intensity recognition (CIR) is a critical technology for detecting and evaluating cavitation phenomena in hydraulic machinery, with significant implications for operational safety, performance optimization, and maintenance cost reduction in complex industrial systems. Despite substantial research progress, a comprehensive review that systematically traces the development trajectory and provides explicit guidance for future research is still lacking. To bridge this gap, this paper presents a thorough review and analysis of hundreds of publications on intelligent CIR across various types of mechanical equipment from 2002 to 2025, summarizing its technological evolution and offering insights for future development. The early stages are dominated by traditional machine learning approaches that relied on manually engineered features under the guidance of domain expert knowledge. The advent of deep learning has driven the development of end-to-end models capable of automatically extracting features from multi-source signals, thereby significantly improving recognition performance and robustness. Recently, physical informed diagnostic models have been proposed to embed domain knowledge into deep learning models, which can enhance interpretability and cross-condition generalization. In the future, transfer learning, multi-modal fusion, lightweight network architectures, and the deployment of industrial agents are expected to propel CIR technology into a new stage, addressing challenges in multi-source data acquisition, standardized evaluation, and industrial implementation. The paper aims to systematically outline the evolution of CIR technology and highlight the emerging trend of integrating deep learning with physical knowledge. This provides a significant reference for researchers and practitioners in the field of intelligent cavitation diagnosis in complex industrial systems. |
| 2025-11-19 | [Towards a Formal Verification of Secure Vehicle Software Updates](http://arxiv.org/abs/2511.15479v1) | Martin Slind Hagen, Emil Lundqvist et al. | With the rise of software-defined vehicles (SDVs), where software governs most vehicle functions alongside enhanced connectivity, the need for secure software updates has become increasingly critical. Software vulnerabilities can severely impact safety, the economy, and society. In response to this challenge, Strandberg et al. [escar Europe, 2021] introduced the Unified Software Update Framework (UniSUF), designed to provide a secure update framework that integrates seamlessly with existing vehicular infrastructures.   Although UniSUF has previously been evaluated regarding cybersecurity, these assessments have not employed formal verification methods. To bridge this gap, we perform a formal security analysis of UniSUF. We model UniSUF's architecture and assumptions to reflect real-world automotive systems and develop a ProVerif-based framework that formally verifies UniSUF's compliance with essential security requirements - confidentiality, integrity, authenticity, freshness, order, and liveness - demonstrating their satisfiability through symbolic execution. Our results demonstrate that UniSUF adheres to the specified security guarantees, ensuring the correctness and reliability of its security framework. |
| 2025-11-19 | [HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation](http://arxiv.org/abs/2511.15435v1) | Linyin Luo, Yujuan Ding et al. | Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance. |
| 2025-11-19 | [WarNav: An Autonomous Driving Benchmark for Segmentation of Navigable Zones in War Scenes](http://arxiv.org/abs/2511.15429v1) | Marc-Emmanuel Coupvent des Graviers, Hejer Ammar et al. | We introduce WarNav, a novel real-world dataset constructed from images of the open-source DATTALION repository, specifically tailored to enable the development and benchmarking of semantic segmentation models for autonomous ground vehicle navigation in unstructured, conflict-affected environments. This dataset addresses a critical gap between conventional urban driving resources and the unique operational scenarios encountered by unmanned systems in hazardous and damaged war-zones. We detail the methodological challenges encountered, ranging from data heterogeneity to ethical considerations, providing guidance for future efforts that target extreme operational contexts. To establish performance references, we report baseline results on WarNav using several state-of-the-art semantic segmentation models trained on structured urban scenes. We further analyse the impact of training data environments and propose a first step towards effective navigability in challenging environments with the constraint of having no annotation of the targeted images. Our goal is to foster impactful research that enhances the robustness and safety of autonomous vehicles in high-risk scenarios while being frugal in annotated data. |
| 2025-11-19 | [BlueBottle: Fast and Robust Blockchains through Subsystem Specialization](http://arxiv.org/abs/2511.15361v1) | Preston Vander Vos, Alberto Sonnino et al. | Blockchain consensus faces a trilemma of security, latency, and decentralization. High-throughput systems often require a reduction in decentralization or robustness against strong adversaries, while highly decentralized and secure systems tend to have lower performance. We present BlueBottle, a two-layer consensus architecture. The core layer, BB-Core, is an n=5f+1 protocol that trades some fault tolerance for a much lower finality latency with a medium-sized core validator set. Our experiments show that BB-Core reduces latency by 20-25% in comparison to Mysticeti. The guard layer, BB-Guard, provides decentralized timestamping, proactive misbehavior detection in BB-Core, and a synchronous recovery path. When it observes equivocations or liveness failures in the core -- while tolerating up to f<3n/5 faulty nodes in the primary layer -- guard validators disseminate evidence, agree on misbehaving parties for exclusion or slashing, and either restart the core protocol (for liveness violations) or select a canonical fork (for safety violations). Together, these layers enable optimistic sub-second finality at high throughput while maintaining strong safety and liveness under a mild synchrony assumption. |
| 2025-11-19 | [Platform-Agnostic Reinforcement Learning Framework for Safe Exploration of Cluttered Environments with Graph Attention](http://arxiv.org/abs/2511.15358v1) | Gabriele Calzolari, Vidya Sumathy et al. | Autonomous exploration of obstacle-rich spaces requires strategies that ensure efficiency while guaranteeing safety against collisions with obstacles. This paper investigates a novel platform-agnostic reinforcement learning framework that integrates a graph neural network-based policy for next-waypoint selection, with a safety filter ensuring safe mobility. Specifically, the neural network is trained using reinforcement learning through the Proximal Policy Optimization (PPO) algorithm to maximize exploration efficiency while minimizing safety filter interventions. Henceforth, when the policy proposes an infeasible action, the safety filter overrides it with the closest feasible alternative, ensuring consistent system behavior. In addition, this paper introduces a reward function shaped by a potential field that accounts for both the agent's proximity to unexplored regions and the expected information gain from reaching them. The proposed framework combines the adaptability of reinforcement learning-based exploration policies with the reliability provided by explicit safety mechanisms. This feature plays a key role in enabling the deployment of learning-based policies on robotic platforms operating in real-world environments. Extensive evaluations in both simulations and experiments performed in a lab environment demonstrate that the approach achieves efficient and safe exploration in cluttered spaces. |
| 2025-11-19 | [People readily follow personal advice from AI but it does not improve their well-being](http://arxiv.org/abs/2511.15352v1) | Lennart Luettgau, Vanessa Cheung et al. | People increasingly seek personal advice from large language models (LLMs), yet whether humans follow their advice, and its consequences for their well-being, remains unknown. In a longitudinal randomised controlled trial with a representative UK sample (N = 2,302), 75% of participants who had a 20-minute discussion with GPT-4o about health, careers or relationships subsequently reported following its advice. Based on autograder evaluations of chat transcripts, LLM advice rarely violated safety best practice. When queried 2-3 weeks later, participants who had interacted with personalised AI (with access to detailed user information) followed its advice more often in the real world and reported higher well-being than those advised by non-personalised AI. However, while receiving personal advice from AI temporarily reduced well-being, no differential long-term effects compared to a control emerged. Our results suggest that humans readily follow LLM advice about personal issues but doing so shows no additional well-being benefit over casual conversations. |
| 2025-11-19 | [Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models](http://arxiv.org/abs/2511.15304v1) | Piercosma Bisconti, Matteo Prandi et al. | We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for large language models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of open-weight judge models and a human-validated stratified subset (with double-annotations to measure agreement). Disagreements were manually resolved. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols. |
| 2025-11-19 | [GRPO-RM: Fine-Tuning Representation Models via GRPO-Driven Reinforcement Learning](http://arxiv.org/abs/2511.15256v1) | Yanchen Xu, Ziheng Jiao et al. | The Group Relative Policy Optimization (GRPO), a reinforcement learning method used to fine-tune large language models (LLMs), has proved its effectiveness in practical applications such as DeepSeek-R1. It raises a question whether GRPO can be generalized to representation learning models. In this paper, we propose Group Relative Policy Optimization for Representation Model (GRPO-RM), and investigate the performance of GRPO-like policy in post-training representation models. Specifically, our method establishes a predefined output set to functionally replace token sequence sampling in LLMs, thereby generating an output group, which is essential for the probability-driven optimization of GRPO. In addition, a specialized reward function is designed to accommodate the properties of representation models. Extensive experiments are conducted on various real-world datasets to validate the effectiveness of our proposed method. |
| 2025-11-19 | [Optimized scheduling of electricity-heat cooperative system considering wind energy consumption and peak shaving and valley filling](http://arxiv.org/abs/2511.15250v1) | Jin Ye, Lingmei Wang et al. | With the global energy transition and rapid development of renewable energy, the scheduling optimization challenge for combined power-heat systems under new energy integration and multiple uncertainties has become increasingly prominent. Addressing this challenge, this study proposes an intelligent scheduling method based on the improved Dual-Delay Deep Deterministic Policy Gradient (PVTD3) algorithm. System optimization is achieved by introducing a penalty term for grid power purchase variations. Simulation results demonstrate that under three typical scenarios (10%, 20%, and 30% renewable penetration), the PVTD3 algorithm reduces the system's comprehensive cost by 6.93%, 12.68%, and 13.59% respectively compared to the traditional TD3 algorithm. Concurrently, it reduces the average fluctuation amplitude of grid power purchases by 12.8%. Regarding energy storage management, the PVTD3 algorithm reduces the end-time state values of low-temperature thermal storage tanks by 7.67-17.67 units while maintaining high-temperature tanks within the 3.59-4.25 safety operating range. Multi-scenario comparative validation demonstrates that the proposed algorithm not only excels in economic efficiency and grid stability but also exhibits superior sustainable scheduling capabilities in energy storage device management. |
| 2025-11-19 | [Computing Sound and Accurate Upper and Lower Bounds on Hamilton-Jacobi Reachability Value Functions](http://arxiv.org/abs/2511.15238v1) | Ihab Tabbara, Eliya Badr et al. | Hamilton-Jacobi (HJ) reachability analysis is a fundamental tool for safety verification and control synthesis for nonlinear-control systems. Classical HJ reachability analysis methods discretize the continuous state space and solve the HJ partial differential equation over a grid, but these approaches do not account for discretization errors and can under-approximate backward reachable sets, which represent unsafe sets of states. We present a framework for computing sound upper and lower bounds on the HJ value functions via value iteration over grids. Additionally, we develop a refinement algorithm that splits cells that were not possible to classify as safe or unsafe given the computed bounds. This algorithm enables computing accurate over-approximations of backward reachable sets even when starting from coarse grids. Finally, we validate the effectiveness of our method in two case studies. |
| 2025-11-19 | [SafeRBench: A Comprehensive Benchmark for Safety Assessment in Large Reasoning Models](http://arxiv.org/abs/2511.15169v1) | Xin Gao, Shaohan Yu et al. | Large Reasoning Models (LRMs) improve answer quality through explicit chain-of-thought, yet this very capability introduces new safety risks: harmful content can be subtly injected, surface gradually, or be justified by misleading rationales within the reasoning trace. Existing safety evaluations, however, primarily focus on output-level judgments and rarely capture these dynamic risks along the reasoning process. In this paper, we present SafeRBench, the first benchmark that assesses LRM safety end-to-end -- from inputs and intermediate reasoning to final outputs. (1) Input Characterization: We pioneer the incorporation of risk categories and levels into input design, explicitly accounting for affected groups and severity, and thereby establish a balanced prompt suite reflecting diverse harm gradients. (2) Fine-Grained Output Analysis: We introduce a micro-thought chunking mechanism to segment long reasoning traces into semantically coherent units, enabling fine-grained evaluation across ten safety dimensions. (3) Human Safety Alignment: We validate LLM-based evaluations against human annotations specifically designed to capture safety judgments. Evaluations on 19 LRMs demonstrate that SafeRBench enables detailed, multidimensional safety assessment, offering insights into risks and protective mechanisms from multiple perspectives. |
| 2025-11-19 | [The Triple-C Paradigm: Cooperative, Complementary, and Competitive Modes for TBS-HAPS-LEO Integration](http://arxiv.org/abs/2511.15064v1) | Eros Kuikel, Sidrah Javed et al. | The growing demands of ubiquitous and resilient global coverage have pushed existing networks to their operational limits, making it increasingly difficult to meet all requirements on their own. Integrating \emph{Terrestrial Base Stations (TBS), High Altitude Platform Stations (HAPS)} and \emph{Low-Earth-Orbit (LEO)} satellites is envisioned as a promising solution, yet the coordination across these heterogeneous platforms remains an open challenge. This paper proposes a novel unifying \emph{Triple-C framework: Cooperation, Complementarity, and Competition}, that systematically defines the TBS-HAPS-LEO interaction to deliver seamless resilient and scalable connectivity. For each C, we detail the architectural methodology, required pre-requisites, and measurable deliverables that govern when and how the three layers should collaborate, complement each other, or contend. We further identify the enabling technologies across physical, logical, and cognitive layers to operationalize the proposed 3C paradigm. A rich portfolio of use cases and targeted applications demonstrates how this technological leap will make such integration both feasible and impactful. Comprehensive performance analysis and emulation results quantify the trade-offs of such integrated networks. In addition, we examine the economical, environmental, safety, privacy, standardization, and regulatory implications that shape the real-world implementation of the proposed framework. eventually, we provide the gap analysis, outline key technical/non-technical challenges, and a road-map of future research directions needed to unlock the full potential of Cooperation, Complementarity, and Competition operations in TBS-HAPS-LEO integrated networks. |
| 2025-11-19 | [Distributed primal-dual algorithm for constrained multi-agent reinforcement learning under coupled policies](http://arxiv.org/abs/2511.15053v1) | Pengcheng Dai, He Wang et al. | In this work, we investigate constrained multi-agent reinforcement learning (CMARL), where agents collaboratively maximize the sum of their local objectives while satisfying individual safety constraints. We propose a framework where agents adopt coupled policies that depend on both local states and parameters, as well as those of their $κ_p$-hop neighbors, with $κ_p>0$ denoting the coupling distance. A distributed primal-dual algorithm is further developed under this framework, wherein each agent has access only to state-action pairs within its $2κ_p$-hop neighborhood and to reward information within its $κ+ 2κ_p$-hop neighborhood, with $κ> 0$ representing the truncation distance. Moreover, agents are not permitted to directly share their true policy parameters or Lagrange multipliers. Instead, each agent constructs and maintains local estimates of these variables for other agents and employs such estimates to execute its policy. Additionally, these estimates are further updated and exchanged exclusively through an independent, time-varying networks, which enhances the overall system security. We establish that, with high probability, our algorithm can achieve an $ε$-first-order stationary convergence with an approximation error of $\mathcal{O}(γ^{\frac{κ+1}{κ_{p}}})$ for discount factor $γ\in(0,1)$. Finally, simulations in GridWorld environment are conducted to demonstrate the effectiveness of the proposed algorithm. |
| 2025-11-19 | [GeoShield: Byzantine Fault Detection and Recovery for Geo-Distributed Real-Time Cyber-Physical Systems](http://arxiv.org/abs/2511.15031v1) | Yifan Cai, Linh Thi Xuan Phan | Large-scale cyber-physical systems (CPS), such as railway control systems and smart grids, consist of geographically distributed subsystems that are connected via unreliable, asynchronous inter-region networks. Their scale and distribution make them especially vulnerable to faults and attacks. Unfortunately, existing fault-tolerant methods either consume excessive resources or provide only eventual guarantees, making them unsuitable for real-time resource-constrained CPS.   We present GeoShield, a resource-efficient solution for defending geo-distributed CPS against Byzantine faults. GeoShield leverages the property that CPS are designed to tolerate brief disruptions and maintain safety, as long as they recover (i.e., resume normal operations or transition to a safe mode) within a bounded amount of time following a fault. Instead of masking faults, it detects them and recovers the system within bounded time, thus guaranteeing safety with much fewer resources. GeoShield introduces protocols for Byzantine fault-resilient network measurement and inter-region omission fault detection that proactively detect malicious message delays, along with recovery mechanisms that guarantee timely recovery while maximizing operational robustness. It is the first bounded-time recovery solution that operates effectively under unreliable networks without relying on trusted hardware. Evaluations using real-world case studies show that it significantly outperforms existing methods in both effectiveness and resource efficiency. |
| 2025-11-18 | [SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification](http://arxiv.org/abs/2511.14977v1) | Xiangyu Li, Zhaomiao Guo | As more autonomous vehicles operate on public roads, understanding real-world behavior of autonomous vehicles is critical to analyzing traffic safety, making policies, and public acceptance. This paper proposes SVBRD-LLM, a framework that automatically discovers, verifies, and applies interpretable behavioral rules from real traffic videos through zero-shot prompt engineering. The framework extracts vehicle trajectories using YOLOv8 and ByteTrack, computes kinematic features, and employs GPT-5 zero-shot prompting to compare autonomous and human-driven vehicles, generating 35 structured behavioral rule hypotheses. These rules are tested on a validation set, iteratively refined based on failure cases to filter spurious correlations, and compiled into a high-confidence rule library. The framework is evaluated on an independent test set for speed change prediction, lane change prediction, and autonomous vehicle identification tasks. Experiments on over 1500 hours of real traffic videos show that the framework achieves 90.0% accuracy and 93.3% F1-score in autonomous vehicle identification. The discovered rules clearly reveal distinctive characteristics of autonomous vehicles in speed control smoothness, lane change conservativeness, and acceleration stability, with each rule accompanied by semantic description, applicable context, and validation confidence. |
| 2025-11-18 | [How Should the Law Treat Future AI Systems? Fictional Legal Personhood versus Legal Identity](http://arxiv.org/abs/2511.14964v1) | Heather J. Alexander, Jonathan A. Simon et al. | The law draws a sharp distinction between objects and persons, and between two kinds of persons, the ''fictional'' kind (i.e. corporations), and the ''non-fictional'' kind (individual or ''natural'' persons). This paper will assess whether we maximize overall long-term legal coherence by (A) maintaining an object classification for all future AI systems, (B) creating fictional legal persons associated with suitably advanced, individuated AI systems (giving these fictional legal persons derogable rights and duties associated with certified groups of existing persons, potentially including free speech, contract rights, and standing to sue ''on behalf of'' the AI system), or (C) recognizing non-fictional legal personhood through legal identity for suitably advanced, individuated AI systems (recognizing them as entities meriting legal standing with non-derogable rights which for the human case include life, due process, habeas corpus, freedom from slavery, and freedom of conscience). We will clarify the meaning and implications of each option along the way, considering liability, copyright, family law, fundamental rights, civil rights, citizenship, and AI safety regulation. We will tentatively find that the non-fictional personhood approach may be best from a coherence perspective, for at least some advanced AI systems. An object approach may prove untenable for sufficiently humanoid advanced systems, though we suggest that it is adequate for currently existing systems as of 2025. While fictional personhood would resolve some coherence issues for future systems, it would create others and provide solutions that are neither durable nor fit for purpose. Finally, our review will suggest that ''hybrid'' approaches are likely to fail and lead to further incoherence: the choice between object, fictional person and non-fictional person is unavoidable. |

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



