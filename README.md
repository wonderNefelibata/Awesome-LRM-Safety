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
| 2025-07-18 | [CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning](http://arxiv.org/abs/2507.14111v1) | Xiaoya Li, Xiaofei Sun et al. | The exponential growth in demand for GPU computing resources, driven by the rapid advancement of Large Language Models, has created an urgent need for automated CUDA optimization strategies. While recent advances in LLMs show promise for code generation, current SOTA models (e.g. R1, o1) achieve low success rates in improving CUDA speed. In this paper, we introduce CUDA-L1, an automated reinforcement learning framework for CUDA optimization.   CUDA-L1 achieves performance improvements on the CUDA optimization task: trained on NVIDIA A100, it delivers an average speedup of x17.7 across all 250 CUDA kernels of KernelBench, with peak speedups reaching x449. Furthermore, the model also demonstrates excellent portability across GPU architectures, achieving average speedups of x17.8 on H100, x19.0 on RTX 3090, x16.5 on L40, x14.7 on H800, and x13.9 on H20 despite being optimized specifically for A100. Beyond these benchmark results, CUDA-L1 demonstrates several remarkable properties: 1) Discovers a variety of CUDA optimization techniques and learns to combine them strategically to achieve optimal performance; 2) Uncovers fundamental principles of CUDA optimization; 3) Identifies non-obvious performance bottlenecks and rejects seemingly beneficial optimizations that harm performance.   The capabilities of CUDA-L1 demonstrate that reinforcement learning can transform an initially poor-performing LLM into an effective CUDA optimizer through speedup-based reward signals alone, without human expertise or domain knowledge. More importantly, the trained RL model extend the acquired reasoning abilities to new kernels. This paradigm opens possibilities for automated optimization of CUDA operations, and holds promise to substantially promote GPU efficiency and alleviate the rising pressure on GPU computing resources. |
| 2025-07-18 | [Automated Interpretation of Non-Destructive Evaluation Contour Maps Using Large Language Models for Bridge Condition Assessment](http://arxiv.org/abs/2507.14107v1) | Viraj Nishesh Darji, Callie C. Liao et al. | Bridge maintenance and safety are essential for transportation authorities, and Non-Destructive Evaluation (NDE) techniques are critical to assessing structural integrity. However, interpreting NDE data can be time-consuming and requires expertise, potentially delaying decision-making. Recent advancements in Large Language Models (LLMs) offer new ways to automate and improve this analysis. This pilot study introduces a holistic assessment of LLM capabilities for interpreting NDE contour maps and demonstrates the effectiveness of LLMs in providing detailed bridge condition analyses. It establishes a framework for integrating LLMs into bridge inspection workflows, indicating that LLM-assisted analysis can enhance efficiency without compromising accuracy. In this study, several LLMs are explored with prompts specifically designed to enhance the quality of image descriptions, which are applied to interpret five different NDE contour maps obtained through technologies for assessing bridge conditions. Each LLM model is evaluated based on its ability to produce detailed descriptions, identify defects, provide actionable recommendations, and demonstrate overall accuracy. The research indicates that four of the nine models provide better image descriptions, effectively covering a wide range of topics related to the bridge's condition. The outputs from these four models are summarized using five different LLMs to form a comprehensive overview of the bridge. Notably, LLMs ChatGPT-4 and Claude 3.5 Sonnet generate more effective summaries. The findings suggest that LLMs have the potential to significantly improve efficiency and accuracy. This pilot study presents an innovative approach that leverages LLMs for image captioning in parallel and summarization, enabling faster decision-making in bridge maintenance and enhancing infrastructure management and safety assessments. |
| 2025-07-18 | [Generative AI-Driven High-Fidelity Human Motion Simulation](http://arxiv.org/abs/2507.14097v1) | Hari Iyer, Neel Macwan et al. | Human motion simulation (HMS) supports cost-effective evaluation of worker behavior, safety, and productivity in industrial tasks. However, existing methods often suffer from low motion fidelity. This study introduces Generative-AI-Enabled HMS (G-AI-HMS), which integrates text-to-text and text-to-motion models to enhance simulation quality for physical tasks. G-AI-HMS tackles two key challenges: (1) translating task descriptions into motion-aware language using Large Language Models aligned with MotionGPT's training vocabulary, and (2) validating AI-enhanced motions against real human movements using computer vision. Posture estimation algorithms are applied to real-time videos to extract joint landmarks, and motion similarity metrics are used to compare them with AI-enhanced sequences. In a case study involving eight tasks, the AI-enhanced motions showed lower error than human created descriptions in most scenarios, performing better in six tasks based on spatial accuracy, four tasks based on alignment after pose normalization, and seven tasks based on overall temporal similarity. Statistical analysis showed that AI-enhanced prompts significantly (p $<$ 0.0001) reduced joint error and temporal misalignment while retaining comparable posture accuracy. |
| 2025-07-18 | [A multi-strategy improved snake optimizer for three-dimensional UAV path planning and engineering problems](http://arxiv.org/abs/2507.14043v1) | Genliang Li, Yaxin Cui et al. | Metaheuristic algorithms have gained widespread application across various fields owing to their ability to generate diverse solutions. One such algorithm is the Snake Optimizer (SO), a progressive optimization approach. However, SO suffers from the issues of slow convergence speed and susceptibility to local optima. In light of these shortcomings, we propose a novel Multi-strategy Improved Snake Optimizer (MISO). Firstly, we propose a new adaptive random disturbance strategy based on sine function to alleviate the risk of getting trapped in a local optimum. Secondly, we introduce adaptive Levy flight strategy based on scale factor and leader and endow the male snake leader with flight capability, which makes it easier for the algorithm to leap out of the local optimum and find the global optimum. More importantly, we put forward a position update strategy combining elite leadership and Brownian motion, effectively accelerating the convergence speed while ensuring precision. Finally, to demonstrate the performance of MISO, we utilize 30 CEC2017 test functions and the CEC2022 test suite, comparing it with 11 popular algorithms across different dimensions to validate its effectiveness. Moreover, Unmanned Aerial Vehicle (UAV) has been widely used in various fields due to its advantages of low cost, high mobility and easy operation. However, the UAV path planning problem is crucial for flight safety and efficiency, and there are still challenges in establishing and optimizing the path model. Therefore, we apply MISO to the UAV 3D path planning problem as well as 6 engineering design problems to assess its feasibility in practical applications. The experimental results demonstrate that MISO exceeds other competitive algorithms in terms of solution quality and stability, establishing its strong potential for application. |
| 2025-07-18 | [Automatic Classification and Segmentation of Tunnel Cracks Based on Deep Learning and Visual Explanations](http://arxiv.org/abs/2507.14010v1) | Yong Feng, Xiaolei Zhang et al. | Tunnel lining crack is a crucial indicator of tunnels' safety status. Aiming to classify and segment tunnel cracks with enhanced accuracy and efficiency, this study proposes a two-step deep learning-based method. An automatic tunnel image classification model is developed using the DenseNet-169 in the first step. The proposed crack segmentation model in the second step is based on the DeepLabV3+, whose internal logic is evaluated via a score-weighted visual explanation technique. Proposed method combines tunnel image classification and segmentation together, so that the selected images containing cracks from the first step are segmented in the second step to improve the detection accuracy and efficiency. The superior performances of the two-step method are validated by experiments. The results show that the accuracy and frames per second (FPS) of the tunnel crack classification model are 92.23% and 39.80, respectively, which are higher than other convolutional neural networks (CNN) based and Transformer based models. Also, the intersection over union (IoU) and F1 score of the tunnel crack segmentation model are 57.01% and 67.44%, respectively, outperforming other state-of-the-art models. Moreover, the provided visual explanations in this study are conducive to understanding the "black box" of deep learning-based models. The developed two-stage deep learning-based method integrating visual explanations provides a basis for fast and accurate quantitative assessment of tunnel health status. |
| 2025-07-18 | [NeHMO: Neural Hamilton-Jacobi Reachability Learning for Decentralized Safe Multi-Agent Motion Planning](http://arxiv.org/abs/2507.13940v1) | Qingyi Chen, Ahmed H. Qureshi | Safe Multi-Agent Motion Planning (MAMP) is a significant challenge in robotics. Despite substantial advancements, existing methods often face a dilemma. Decentralized algorithms typically rely on predicting the behavior of other agents, sharing contracts, or maintaining communication for safety, while centralized approaches struggle with scalability and real-time decision-making. To address these challenges, we introduce Neural Hamilton-Jacobi Reachability Learning (HJR) for Decentralized Multi-Agent Motion Planning. Our method provides scalable neural HJR modeling to tackle high-dimensional configuration spaces and capture worst-case collision and safety constraints between agents. We further propose a decentralized trajectory optimization framework that incorporates the learned HJR solutions to solve MAMP tasks in real-time. We demonstrate that our method is both scalable and data-efficient, enabling the solution of MAMP problems in higher-dimensional scenarios with complex collision constraints. Our approach generalizes across various dynamical systems, including a 12-dimensional dual-arm setup, and outperforms a range of state-of-the-art techniques in successfully addressing challenging MAMP tasks. Video demonstrations are available at https://youtu.be/IZiePX0p1Mc. |
| 2025-07-18 | [Fixed time convergence guarantees for Higher Order Control Barrier Functions](http://arxiv.org/abs/2507.13888v1) | Janani S K, Shishir Kolathaya | We present a novel method for designing higher-order Control Barrier Functions (CBFs) that guarantee convergence to a safe set within a user-specified finite. Traditional Higher Order CBFs (HOCBFs) ensure asymptotic safety but lack mechanisms for fixed-time convergence, which is critical in time-sensitive and safety-critical applications such as autonomous navigation. In contrast, our approach imposes a structured differential constraint using repeated roots in the characteristic polynomial, enabling closed-form polynomial solutions with exact convergence at a prescribed time. We derive conditions on the barrier function and its derivatives that ensure forward invariance and fixed-time reachability, and we provide an explicit formulation for second-order systems. Our method is evaluated on three robotic systems - a point-mass model, a unicycle, and a bicycle model and benchmarked against existing HOCBF approaches. Results demonstrate that our formulation reliably enforces convergence within the desired time, even when traditional methods fail. This work provides a tractable and robust framework for real-time control with provable finite-time safety guarantees. |
| 2025-07-18 | [Safe and Performant Controller Synthesis using Gradient-based Model Predictive Control and Control Barrier Functions](http://arxiv.org/abs/2507.13872v1) | Aditya Singh, Aastha Mishra et al. | Ensuring both performance and safety is critical for autonomous systems operating in real-world environments. While safety filters such as Control Barrier Functions (CBFs) enforce constraints by modifying nominal controllers in real time, they can become overly conservative when the nominal policy lacks safety awareness. Conversely, solving State-Constrained Optimal Control Problems (SC-OCPs) via dynamic programming offers formal guarantees but is intractable in high-dimensional systems. In this work, we propose a novel two-stage framework that combines gradient-based Model Predictive Control (MPC) with CBF-based safety filtering for co-optimizing safety and performance. In the first stage, we relax safety constraints as penalties in the cost function, enabling fast optimization via gradient-based methods. This step improves scalability and avoids feasibility issues associated with hard constraints. In the second stage, we modify the resulting controller using a CBF-based Quadratic Program (CBF-QP), which enforces hard safety constraints with minimal deviation from the reference. Our approach yields controllers that are both performant and provably safe. We validate the proposed framework on two case studies, showcasing its ability to synthesize scalable, safe, and high-performance controllers for complex, high-dimensional autonomous systems. |
| 2025-07-18 | [Safety Certification in the Latent space using Control Barrier Functions and World Models](http://arxiv.org/abs/2507.13871v1) | Mehul Anand, Shishir Kolathaya | Synthesising safe controllers from visual data typically requires extensive supervised labelling of safety-critical data, which is often impractical in real-world settings. Recent advances in world models enable reliable prediction in latent spaces, opening new avenues for scalable and data-efficient safe control. In this work, we introduce a semi-supervised framework that leverages control barrier certificates (CBCs) learned in the latent space of a world model to synthesise safe visuomotor policies. Our approach jointly learns a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive power of modern vision transformers for latent dynamics modelling. |
| 2025-07-18 | [Principles and Reasons Behind Automated Vehicle Decisions in Ethically Ambiguous Everyday Scenarios](http://arxiv.org/abs/2507.13837v1) | Lucas Elbert Suryana, Simeon Calvert et al. | Automated vehicles (AVs) increasingly encounter ethically ambiguous situations in everyday driving--scenarios involving conflicting human interests and lacking clearly optimal courses of action. While existing ethical models often focus on rare, high-stakes dilemmas (e.g., crash avoidance or trolley problems), routine decisions such as overtaking cyclists or navigating social interactions remain underexplored. This study addresses that gap by applying the tracking condition of Meaningful Human Control (MHC), which holds that AV behaviour should align with human reasons--defined as the values, intentions, and expectations that justify actions. We conducted qualitative interviews with 18 AV experts to identify the types of reasons that should inform AV manoeuvre planning. Thirteen categories of reasons emerged, organised across normative, strategic, tactical, and operational levels, and linked to the roles of relevant human agents. A case study on cyclist overtaking illustrates how these reasons interact in context, revealing a consistent prioritisation of safety, contextual flexibility regarding regulatory compliance, and nuanced trade-offs involving efficiency, comfort, and public acceptance. Based on these insights, we propose a principled conceptual framework for AV decision-making in routine, ethically ambiguous scenarios. The framework supports dynamic, human-aligned behaviour by prioritising safety, allowing pragmatic actions when strict legal adherence would undermine key values, and enabling constrained deviations when appropriately justified. This empirically grounded approach advances current guidance by offering actionable, context-sensitive design principles for ethically aligned AV systems. |
| 2025-07-18 | [Food safety trends across Europe: insights from the 392-million-entry CompreHensive European Food Safety (CHEFS) database](http://arxiv.org/abs/2507.13802v1) | Nehir Kizililsoley, Floor van Meer et al. | In the European Union, official food safety monitoring data collected by member states are submitted to the European Food Safety Authority (EFSA) and published on Zenodo. This data includes 392 million analytical results derived from over 15.2 million samples covering more than 4,000 different types of food products, offering great opportunities for artificial intelligence to analyze trends, predict hazards, and support early warning systems. However, the current format with data distributed across approximately 1000 files totaling several hundred gigabytes hinders accessibility and analysis. To address this, we introduce the CompreHensive European Food Safety (CHEFS) database, which consolidates EFSA monitoring data on pesticide residues, veterinary medicinal product residues, and chemical contaminants into a unified and structured dataset. We describe the creation and structure of the CHEFS database and demonstrate its potential by analyzing trends in European food safety monitoring data from 2000 to 2024. Our analyses explore changes in monitoring activities, the most frequently tested products, which products were most often non-compliant and which contaminants were most often found, and differences across countries. These findings highlight the CHEFS database as both a centralized data source and a strategic tool for guiding food safety policy, research, and regulation. |
| 2025-07-18 | [Innocence in the Crossfire: Roles of Skip Connections in Jailbreaking Visual Language Models](http://arxiv.org/abs/2507.13761v1) | Palash Nandi, Maithili Joshi et al. | Language models are highly sensitive to prompt formulations - small changes in input can drastically alter their output. This raises a critical question: To what extent can prompt sensitivity be exploited to generate inapt content? In this paper, we investigate how discrete components of prompt design influence the generation of inappropriate content in Visual Language Models (VLMs). Specifically, we analyze the impact of three key factors on successful jailbreaks: (a) the inclusion of detailed visual information, (b) the presence of adversarial examples, and (c) the use of positively framed beginning phrases. Our findings reveal that while a VLM can reliably distinguish between benign and harmful inputs in unimodal settings (text-only or image-only), this ability significantly degrades in multimodal contexts. Each of the three factors is independently capable of triggering a jailbreak, and we show that even a small number of in-context examples (as few as three) can push the model toward generating inappropriate outputs. Furthermore, we propose a framework that utilizes a skip-connection between two internal layers of the VLM, which substantially increases jailbreak success rates, even when using benign images. Finally, we demonstrate that memes, often perceived as humorous or harmless, can be as effective as toxic visuals in eliciting harmful content, underscoring the subtle and complex vulnerabilities of VLMs. |
| 2025-07-18 | [The Emperor's New Chain-of-Thought: Probing Reasoning Theater Bias in Large Reasoning Models](http://arxiv.org/abs/2507.13758v1) | Qian Wang, Yubo Fan et al. | Large Reasoning Models (LRMs) like DeepSeek-R1 and o1 are increasingly used as automated evaluators, raising critical questions about their vulnerability to the aesthetics of reasoning in LLM-as-a-judge settings. We introduce THEATER, a comprehensive benchmark to systematically evaluate this vulnerability-termed Reasoning Theater Bias (RTB)-by comparing LLMs and LRMs across subjective preference and objective factual datasets. Through investigation of six bias types including Simple Cues and Fake Chain-of-Thought, we uncover three key findings: (1) in a critical paradox, reasoning-specialized LRMs are consistently more susceptible to RTB than general-purpose LLMs, particularly in subjective tasks; (2) this creates a task-dependent trade-off, where LRMs show more robustness on factual tasks but less on subjective ones; and (3) we identify 'shallow reasoning'-plausible but flawed arguments-as the most potent form of RTB. To address this, we design and evaluate two prompting strategies: a targeted system prompt that improves accuracy by up to 12% on factual tasks but only 1-3% on subjective tasks, and a self-reflection mechanism that shows similarly limited effectiveness in the more vulnerable subjective domains. Our work reveals that RTB is a deep-seated challenge for LRM-based evaluation and provides a systematic framework for developing more genuinely robust and trustworthy LRMs. |
| 2025-07-18 | [Probing neutrino emission at GeV energies from compact binary mergers detected during O1-O4a with the IceCube Neutrino Observatory](http://arxiv.org/abs/2507.13698v1) | Karlijn Kruiswijk, Mathieu Lamoureux et al. | Compact binary mergers, detected in gravitational waves since 2015, are candidate sources for astrophysical neutrinos in the GeV regime from proton-proton and proton-neutron collisions. This contribution presents the results of the search for such a signal using mergers detected during the fourth observing run of the LIGO, Virgo, and KAGRA interferometers. We use the dense infill array at the center of the IceCube detector, IceCube-DeepCore, to select neutrino candidates in the 0.5-5 GeV energy range. The search for a statistically significant excess associated with an astrophysical signal is performed in a $\pm$ 500 s window around the gravitational wave detection time. We do not observe any statistically significant excess in the neutrino data, and set upper limits on the neutrino emission from these objects. Additionally, we search for subpopulations of neutrino-emitting sources, including merger events detected in previous observing runs; no significant signal has been identified yet. |
| 2025-07-18 | [Robust Probability Hypothesis Density Filtering: Theory and Algorithms](http://arxiv.org/abs/2507.13687v1) | Ming Lei, Shufan Wu | Multi-target tracking (MTT) serves as a cornerstone technology in information fusion, yet faces significant challenges in robustness and efficiency when dealing with model uncertainties, clutter interference, and target interactions. Conventional approaches like Gaussian Mixture PHD (GM-PHD) and Cardinalized PHD (CPHD) filters suffer from inherent limitations including combinatorial explosion, sensitivity to birth/death process parameters, and numerical instability. This study proposes an innovative minimax robust PHD filtering framework with four key contributions: (1) A theoretically derived robust GM-PHD recursion algorithm that achieves optimal worst-case error control under bounded uncertainties; (2) An adaptive real-time parameter adjustment mechanism ensuring stability and error bounds; (3) A generalized heavy-tailed measurement likelihood function maintaining polynomial computational complexity; (4) A novel partition-based credibility weighting method for extended targets. The research not only establishes rigorous convergence guarantees and proves the uniqueness of PHD solutions, but also verifies algorithmic equivalence with standard GM-PHD. Experimental results demonstrate that in high-clutter environments, this method achieves a remarkable 32.4% reduction in OSPA error and 25.3% lower cardinality RMSE compared to existing techniques, while maintaining real-time processing capability at 15.3 milliseconds per step. This breakthrough lays a crucial foundation for reliable MTT in safety-critical applications. |
| 2025-07-18 | [Spacecraft Safe Robust Control Using Implicit Neural Representation for Geometrically Complex Targets in Proximity Operations](http://arxiv.org/abs/2507.13672v1) | Hang Zhou, Tao Meng et al. | This study addresses the challenge of ensuring safe spacecraft proximity operations, focusing on collision avoidance between a chaser spacecraft and a complex-geometry target spacecraft under disturbances. To ensure safety in such scenarios, a safe robust control framework is proposed that leverages implicit neural representations. To handle arbitrary target geometries without explicit modeling, a neural signed distance function (SDF) is learned from point cloud data via a enhanced implicit geometric regularization method, which incorporates an over-apporximation strategy to create a conservative, safety-prioritized boundary. The target's surface is implicitly defined by the zero-level set of the learned neural SDF, while the values and gradients provide critical information for safety controller design. This neural SDF representation underpins a two-layer hierarchcial safe robust control framework: a safe velocity generation layer and a safe robust controller layer. In the first layer, a second-order cone program is formulated to generate safety-guaranteed reference velocity by explicitly incorporating the under-approximation error bound. Furthermore, a circulation inequality is introduced to mitigate the local minimum issues commonly encountered in control barrier function (CBF) methods. The second layer features an integrated disturbance observer and a smooth safety filter explicitly compensating for estimation error, bolstering robustness to external disturbances. Extensive numerical simulations and Monte Carlo analysis validate the proposed framework, demonstrating significantly improved safety margins and avoidance of local minima compared to conventional CBF approaches. |
| 2025-07-18 | [BifrostRAG: Bridging Dual Knowledge Graphs for Multi-Hop Question Answering in Construction Safety](http://arxiv.org/abs/2507.13625v1) | Yuxin Zhang, Xi Wang et al. | Information retrieval and question answering from safety regulations are essential for automated construction compliance checking but are hindered by the linguistic and structural complexity of regulatory text. Many compliance-related queries are multi-hop, requiring synthesis of information across interlinked clauses. This poses a challenge for traditional retrieval-augmented generation (RAG) systems. To overcome this, we introduce BifrostRAG: a dual-graph RAG-integrated system that explicitly models both linguistic relationships (via an Entity Network Graph) and document structure (via a Document Navigator Graph). This architecture powers a hybrid retrieval mechanism that combines graph traversal with vector-based semantic search, enabling large language models to reason over both the meaning and the structure of the text. Evaluation on a multi-hop question dataset shows that BifrostRAG achieves 92.8 percent precision, 85.5 percent recall, and an F1 score of 87.3 percent. These results significantly outperform vector-only and graph-only RAG baselines that represent current leading approaches. Error analysis further highlights the comparative advantages of our hybrid method over single-modality RAGs. These findings establish BifrostRAG as a robust knowledge engine for LLM-driven compliance checking. Its dual-graph, hybrid retrieval mechanism offers a transferable blueprint for navigating complex technical documents across knowledge-intensive engineering domains. |
| 2025-07-18 | [GIFT: Gradient-aware Immunization of diffusion models against malicious Fine-Tuning with safe concepts retention](http://arxiv.org/abs/2507.13598v1) | Amro Abdalla, Ismail Shaheen et al. | We present GIFT: a {G}radient-aware {I}mmunization technique to defend diffusion models against malicious {F}ine-{T}uning while preserving their ability to generate safe content. Existing safety mechanisms like safety checkers are easily bypassed, and concept erasure methods fail under adversarial fine-tuning. GIFT addresses this by framing immunization as a bi-level optimization problem: the upper-level objective degrades the model's ability to represent harmful concepts using representation noising and maximization, while the lower-level objective preserves performance on safe data. GIFT achieves robust resistance to malicious fine-tuning while maintaining safe generative quality. Experimental results show that our method significantly impairs the model's ability to re-learn harmful concepts while maintaining performance on safe content, offering a promising direction for creating inherently safer generative models resistant to adversarial fine-tuning attacks. |
| 2025-07-17 | [Space Shift Keying-Enabled ISAC for Efficient Debris Detection and Communication in LEO Satellite Networks](http://arxiv.org/abs/2507.13526v1) | Gedeon Ghislain Nkwewo Ngoufo, Khaled Humadi et al. | The proliferation of space debris in low Earth orbit (LEO) presents critical challenges for orbital safety, particularly for satellite constellations. Integrated sensing and communication (ISAC) systems provide a promising dual function solution by enabling both environmental sensing and data communication. This study explores the use of space shift keying (SSK) modulation within ISAC frameworks, evaluating its performance when combined with sinusoidal and chirp radar waveforms. SSK is particularly attractive due to its low hardware complexity and robust communication performance. Our results demonstrate that both waveforms achieve comparable bit error rate (BER) performance under SSK, validating its effectiveness for ISAC applications. However, waveform selection significantly affects sensing capability: while the sinusoidal waveform supports simpler implementation, its high ambiguity limits range detection. In contrast, the chirp waveform enables range estimation and provides a modest improvement in velocity detection accuracy. These findings highlight the strength of SSK as a modulation scheme for ISAC and emphasize the importance of selecting appropriate waveforms to optimize sensing accuracy without compromising communication performance. This insight supports the design of efficient and scalable ISAC systems for space applications, particularly in the context of orbital debris monitoring. |
| 2025-07-17 | [AI-Assisted Fixes to Code Review Comments at Scale](http://arxiv.org/abs/2507.13499v1) | Chandra Maddila, Negar Ghorbani et al. | Aim. There are 10s of thousands of code review comments each week at Meta. We developed Metamate for Code Review (MetaMateCR) that provides AI-assisted fixes for reviewer comments in production at scale.   Method. We developed an internal benchmark of 64k <review comment, patch> data points to fine-tune Llama models. Once our models achieve reasonable offline results, we roll them into production. To ensure that our AI-assisted fixes do not negatively impact the time it takes to do code reviews, we conduct randomized controlled safety trials as well as full production experiments.   Offline Results. As a baseline, we compare GPT-4o to our small and large Llama models. In offline results, our LargeLSFT model creates an exact match patch 68% of the time outperforming GPT-4o by 9 percentage points (pp). The internal models also use more modern Hack functions when compared to the PHP functions suggested by GPT-4o.   Safety Trial. When we roll MetaMateCR into production in a safety trial that compares no AI patches with AI patch suggestions, we see a large regression with reviewers taking over 5% longer to conduct reviews. After investigation, we modify the UX to only show authors the AI patches, and see no regressions in the time for reviews.   Production. When we roll LargeLSFT into production, we see an ActionableToApplied rate of 19.7%, which is a 9.2pp improvement over GPT-4o. Our results illustrate the importance of safety trials in ensuring that AI does not inadvertently slow down engineers, and a successful review comment to AI patch product running at scale. |

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



