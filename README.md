# Awesome Large Reasoning Model (LRM) Safety 🔥

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Auto Update](https://github.com/wonderNefelibata/Awesome-LRM-Safety/actions/workflows/arxiv-update.yml/badge.svg)

A curated list of **security and safety research** for Large Reasoning Models (LRMs) like DeepSeek-R1, OpenAI o1, and other cutting-edge models. Focused on identifying risks, mitigation strategies, and ethical implications.

---

## 📜 Table of Contents
- [Motivation](#-motivation)
- [Latest arXiv Papers](#-latest-arxiv-papers-auto-updated)
- [Key Safety Domains](#-key-safety-domains)
- [Research Papers](#-research-papers)
- [Projects & Tools](#-projects--tools)
- [Contributing](#-contributing)
- [License](#-license)
- [FAQ](#-faq)

---

## 🚀 Motivation
Large Reasoning Models (LRMs) are revolutionizing AI capabilities in complex decision-making scenarios. However, their deployment raises critical safety concerns:
- **Adversarial attacks** on reasoning pipelines
- **Privacy leakage** through multi-step reasoning
- **Ethical risks** in high-stakes domains (e.g., healthcare, finance)
- **Systemic failures** in autonomous systems
This repository aims to catalog research addressing these challenges and promote safer LRM development.

---

## 📰 Latest arXiv Papers (Auto-Updated)
<!-- ARXIV_PAPERS_START -->

| Date       | Title                                      | Authors           | Abstract                                      |
|------------|--------------------------------------------|-------------------|-----------------------------------------------|
| 2025-02-17 | [CriteoPrivateAd: A Real-World Bidding Dataset to Design Private Advertising Systems](http://arxiv.org/abs/2502.12103v1) | Mehdi Sebbar, Corentin Odic et al. | In the past years, many proposals have emerged in order to address online advertising use-cases without access to third-party cookies. All these propo... |
| 2025-02-17 | [FedEAT: A Robustness Optimization Framework for Federated LLMs](http://arxiv.org/abs/2502.11863v1) | Yahao Pang, Xingyuan Wu et al. | Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creatio... |
| 2025-02-17 | [Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives](http://arxiv.org/abs/2502.11858v1) | Zeliang Zhang, Susan Liang et al. | While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration al... |
| 2025-02-17 | [BackdoorDM: A Comprehensive Benchmark for Backdoor Learning in Diffusion Model](http://arxiv.org/abs/2502.11798v1) | Weilin Lin, Nanjun Zhou et al. | Backdoor learning is a critical research topic for understanding the vulnerabilities of deep neural networks. While it has been extensively studied in... |
| 2025-02-17 | [Lightweight Deepfake Detection Based on Multi-Feature Fusion](http://arxiv.org/abs/2502.11763v1) | Siddiqui Muhammad Yasir, Hyun Kim et al. | Deepfake technology utilizes deep learning based face manipulation techniques to seamlessly replace faces in videos creating highly realistic but arti... |
| 2025-02-17 | [Incomplete Modality Disentangled Representation for Ophthalmic Disease Grading and Diagnosis](http://arxiv.org/abs/2502.11724v1) | Chengzhi Liu, Zile Huang et al. | Ophthalmologists typically require multimodal data sources to improve diagnostic accuracy in clinical decisions. However, due to medical device shorta... |
| 2025-02-17 | [ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning](http://arxiv.org/abs/2502.11687v1) | Manaar Alam, Hithem Lamri et al. | Backdoor attacks embed hidden functionalities in deep neural networks (DNN), triggering malicious behavior with specific inputs. Advanced defenses mon... |
| 2025-02-17 | [Double Momentum and Error Feedback for Clipping with Fast Rates and Differential Privacy](http://arxiv.org/abs/2502.11682v1) | Rustem Islamov, Samuel Horvath et al. | Strong Differential Privacy (DP) and Optimization guarantees are two desirable properties for a method in Federated Learning (FL). However, existing a... |
| 2025-02-17 | ["I'm not for sale" -- Perceptions and limited awareness of privacy risks by digital natives about location data](http://arxiv.org/abs/2502.11658v1) | Antoine Boutet, Victor Morel et al. | Although mobile devices benefit users in their daily lives in numerous ways, they also raise several privacy concerns. For instance, they can reveal s... |
| 2025-02-17 | [DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing](http://arxiv.org/abs/2502.11647v1) | Yi Wang, Fenghua Weng et al. | Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users m... |
| 2025-02-17 | [Membership Inference Attacks for Face Images Against Fine-Tuned Latent Diffusion Models](http://arxiv.org/abs/2502.11619v1) | Lauritz Christian Holme, Anton Mosquera Storgaard et al. | The rise of generative image models leads to privacy concerns when it comes to the huge datasets used to train such models. This paper investigates th... |
| 2025-02-17 | [User-Centric Data Management in Decentralized Internet of Behaviors System](http://arxiv.org/abs/2502.11616v1) | Shiqi Zhang, Dapeng Wu et al. | The Internet of Behaviors (IoB) is an emerging concept that utilizes devices to collect human behavior and provide intelligent services. Although some... |
| 2025-02-17 | [InfiR : Crafting Effective Small Language Models and Multimodal Small Language Models in Reasoning](http://arxiv.org/abs/2502.11573v1) | Congkai Xie, Shuo Cai et al. | Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have made significant advancements in reasoning capabilities. However, they ... |
| 2025-02-17 | [Trinity: A Scalable and Forward-Secure DSSE for Spatio-Temporal Range Query](http://arxiv.org/abs/2502.11550v1) | Zhijun Li, Kuizhi Liu et al. | Cloud-based outsourced Location-based services have profound impacts on various aspects of people's lives but bring security concerns. Existing spatio... |
| 2025-02-17 | [Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of Stealing Privacy](http://arxiv.org/abs/2502.11533v1) | Zhenyuan Guo, Yi Shi et al. | Model merging is a widespread technology in large language models (LLMs) that integrates multiple task-specific LLMs into a unified one, enabling the ... |
| 2025-02-17 | [Adversary-Aware DPO: Enhancing Safety Alignment in Vision Language Models via Adversarial Training](http://arxiv.org/abs/2502.11455v1) | Fenghua Weng, Jian Lou et al. | Safety alignment is critical in pre-training large language models (LLMs) to generate responses aligned with human values and refuse harmful queries. ... |
| 2025-02-17 | [From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations](http://arxiv.org/abs/2502.11451v1) | Shenghan Wu, Yang Deng et al. | The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable so... |
| 2025-02-17 | [Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning](http://arxiv.org/abs/2502.11441v1) | Hwan Chang, Hwanhee Lee et al. | Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearn... |
| 2025-02-17 | ["An Image of Ourselves in Our Minds": How College-educated Online Dating Users Construct Profiles for Effective Self Presentation](http://arxiv.org/abs/2502.11430v1) | Fan Zhang, Yun Chen et al. | Online dating is frequently used by individuals looking for potential relationships and intimate connections. Central to dating apps is the creation a... |
| 2025-02-17 | [Detecting and Filtering Unsafe Training Data via Data Attribution](http://arxiv.org/abs/2502.11411v1) | Yijun Pan, Taiwei Shi et al. | Large language models (LLMs) are vulnerable to unsafe training data that even small amounts of unsafe data can lead to harmful model behaviors. Detect... |
| 2025-02-17 | [CCJA: Context-Coherent Jailbreak Attack for Aligned Large Language Models](http://arxiv.org/abs/2502.11379v1) | Guanghao Zhou, Panjia Qiu et al. | Despite explicit alignment efforts for large language models (LLMs), they can still be exploited to trigger unintended behaviors, a phenomenon known a... |
| 2025-02-17 | [VLDBench: Vision Language Models Disinformation Detection Benchmark](http://arxiv.org/abs/2502.11361v1) | Shaina Raza, Ashmal Vayani et al. | The rapid rise of AI-generated content has made detecting disinformation increasingly challenging. In particular, multimodal disinformation, i.e., onl... |
| 2025-02-17 | [Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System](http://arxiv.org/abs/2502.11358v1) | Ziyou Jiang, Mingyang Li et al. | Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands throu... |
| 2025-02-17 | [Evaluating the Performance of the DeepSeek Model in Confidential Computing Environment](http://arxiv.org/abs/2502.11347v1) | Ben Dong, Qian Wang et al. | The increasing adoption of Large Language Models (LLMs) in cloud environments raises critical security concerns, particularly regarding model confiden... |
| 2025-02-17 | [Differentially private fine-tuned NF-Net to predict GI cancer type](http://arxiv.org/abs/2502.11329v1) | Sai Venkatesh Chilukoti, Imran Hossen Md et al. | Based on global genomic status, the cancer tumor is classified as Microsatellite Instable (MSI) and Microsatellite Stable (MSS). Immunotherapy is used... |
| 2025-02-16 | [VLMs as GeoGuessr Masters: Exceptional Performance, Hidden Biases, and Privacy Risks](http://arxiv.org/abs/2502.11163v1) | Jingyuan Huang, Jen-tse Huang et al. | Visual-Language Models (VLMs) have shown remarkable performance across various tasks, particularly in recognizing geographic information from images. ... |
| 2025-02-16 | [G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems](http://arxiv.org/abs/2502.11127v1) | Shilong Wang, Guibin Zhang et al. | Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborat... |
| 2025-02-16 | [Ramp Up NTT in Record Time using GPU-Accelerated Algorithms and LLM-based Code Generation](http://arxiv.org/abs/2502.11110v1) | Yu Cui, Hang Fu et al. | Homomorphic encryption (HE) is a core building block in privacy-preserving machine learning (PPML), but HE is also widely known as its efficiency bott... |
| 2025-02-16 | [SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks](http://arxiv.org/abs/2502.11090v1) | Hongye Cao, Yanming Wang et al. | With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current bench... |
| 2025-02-16 | [Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction](http://arxiv.org/abs/2502.11084v1) | Yuting Huang, Chengyuan Liu et al. | As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful... |

<details><summary>View Older Papers</summary>

| Date       | Title                                      | Authors           | Abstract                                      |
|------------|--------------------------------------------|-------------------|-----------------------------------------------|
| 2025-02-16 | [DreamDDP: Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization](http://arxiv.org/abs/2502.11058v1) | Zhenheng Tang, Zichen Tang et al. | The growth of large language models (LLMs) increases challenges of accelerating distributed training across multiple GPUs in different data centers. M... |
| 2025-02-16 | [Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models](http://arxiv.org/abs/2502.11054v1) | Zonghao Ying, Deyue Zhang et al. | Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical... |
| 2025-02-16 | [HawkEye: Statically and Accurately Profiling the Communication Cost of Models in Multi-party Learning](http://arxiv.org/abs/2502.11029v1) | Wenqiang Ruan, Xin Lin et al. | Multi-party computation (MPC) based machine learning, referred to as multi-party learning (MPL), has become an important technology for utilizing data... |
| 2025-02-16 | [Computing Inconsistency Measures Under Differential Privacy](http://arxiv.org/abs/2502.11009v1) | Shubhankar Mohapatra, Amir Gilad et al. | Assessing data quality is crucial to knowing whether and how to use the data for different purposes. Specifically, given a collection of integrity con... |
| 2025-02-16 | [Prompt Inject Detection with Generative Explanation as an Investigative Tool](http://arxiv.org/abs/2502.11006v1) | Jonathan Pan, Swee Liang Wong et al. | Large Language Models (LLMs) are vulnerable to adversarial prompt based injects. These injects could jailbreak or exploit vulnerabilities within these... |
| 2025-02-16 | [New Rates in Stochastic Decision-Theoretic Online Learning under Differential Privacy](http://arxiv.org/abs/2502.10997v1) | Ruihan Wu, Yu-Xiang Wang et al. | Hu and Mehta (2024) posed an open problem: what is the optimal instance-dependent rate for the stochastic decision-theoretic online learning (with $K$... |
| 2025-02-15 | [FaceSwapGuard: Safeguarding Facial Privacy from DeepFake Threats through Identity Obfuscation](http://arxiv.org/abs/2502.10801v1) | Li Wang, Zheng Li et al. | DeepFakes pose a significant threat to our society. One representative DeepFake application is face-swapping, which replaces the identity in a facial ... |
| 2025-02-15 | [Distraction is All You Need for Multimodal Large Language Model Jailbreaking](http://arxiv.org/abs/2502.10794v1) | Zuopeng Yang, Jiluan Fan et al. | Multimodal Large Language Models (MLLMs) bridge the gap between visual and textual data, enabling a range of advanced applications. However, complex i... |
| 2025-02-15 | [ReReLRP - Remembering and Recognizing Tasks with LRP](http://arxiv.org/abs/2502.10789v1) | Karolina Bogacka, Maximilian Höfler et al. | Deep neural networks have revolutionized numerous research fields and applications. Despite their widespread success, a fundamental limitation known a... |
| 2025-02-15 | [Analyzing Privacy Dynamics within Groups using Gamified Auctions](http://arxiv.org/abs/2502.10788v1) | Hüseyin Aydın, Onuralp Ulusoy et al. | Online shared content, such as group pictures, often contains information about multiple users. Developing technical solutions to manage the privacy o... |
| 2025-02-15 | [Assessing the Trustworthiness of Electronic Identity Management Systems: Framework and Insights from Inception to Deployment](http://arxiv.org/abs/2502.10771v1) | Mirko Bottarelli, Gregory Epiphaniou et al. | The growing dependence on Electronic Identity Management Systems (EIDS) and recent advancements, such as non-human ID management, require a thorough e... |
| 2025-02-15 | [BASE-SQL: A powerful open source Text-To-SQL baseline approach](http://arxiv.org/abs/2502.10739v1) | Lei Sheng, Shuai-Shuai Xu et al. | The conversion of natural language into SQL language for querying databases (Text-to-SQL) has broad application prospects and has attracted widespread... |
| 2025-02-15 | [Unpacking the Layers: Exploring Self-Disclosure Norms, Engagement Dynamics, and Privacy Implications](http://arxiv.org/abs/2502.10701v1) | Ehsan-Ul Haq, Shalini Jangra et al. | This paper characterizes the self-disclosure behavior of Reddit users across 11 different types of self-disclosure. We find that at least half of the ... |
| 2025-02-15 | [Privacy Preservation through Practical Machine Unlearning](http://arxiv.org/abs/2502.10635v1) | Robert Dilworth et al. | Machine Learning models thrive on vast datasets, continuously adapting to provide accurate predictions and recommendations. However, in an era dominat... |
| 2025-02-14 | [Federated Learning-Driven Cybersecurity Framework for IoT Networks with Privacy-Preserving and Real-Time Threat Detection Capabilities](http://arxiv.org/abs/2502.10599v1) | Milad Rahmati et al. | The rapid expansion of the Internet of Things (IoT) ecosystem has transformed various sectors but has also introduced significant cybersecurity challe... |
| 2025-02-14 | [Small Loss Bounds for Online Learning Separated Function Classes: A Gaussian Process Perspective](http://arxiv.org/abs/2502.10292v1) | Adam Block, Abhishek Shetty et al. | In order to develop practical and efficient algorithms while circumventing overly pessimistic computational lower bounds, recent work has been interes... |
| 2025-02-14 | [Adversarial Mixup Unlearning](http://arxiv.org/abs/2502.10288v1) | Zhuoyi Peng, Yixuan Tang et al. | Machine unlearning is a critical area of research aimed at safeguarding data privacy by enabling the removal of sensitive information from machine lea... |
| 2025-02-14 | [A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems](http://arxiv.org/abs/2502.10284v1) | Binglei Zhao, Houying Qi et al. | Large-scale recommendation systems often adopt cascading architecture consisting of retrieval, pre-ranking, ranking, and re-ranking stages. With stric... |
| 2025-02-14 | [Efficient Zero-Order Federated Finetuning of Language Models for Resource-Constrained Devices](http://arxiv.org/abs/2502.10239v1) | Mohamed Aboelenien Ahmed, Kilian Pfeiffer et al. | Federated fine-tuning offers a promising approach for tuning Large Language Models (LLMs) on edge devices while preserving data privacy. However, fine... |
| 2025-02-14 | [RIPOST: Two-Phase Private Decomposition for Multidimensional Data](http://arxiv.org/abs/2502.10207v1) | Ala Eddine Laouir, Abdessamad Imine et al. | Differential privacy (DP) is considered as the gold standard for data privacy. While the problem of answering simple queries and functions under DP gu... |

</details>

---

## 🔑 Key Safety Domains(coming soon)
| Category               | Key Challenges                          | Related Topics                          |
|------------------------|-----------------------------------------|------------------------------------------|
| **Adversarial Robustness** | Prompt injection, Reasoning path poisoning | Red teaming, Formal verification        |
| **Privacy Preservation**  | Intermediate step memorization, Data leakage | Differential privacy, Federated learning|
| **Ethical Alignment**     | Value locking, Contextual moral reasoning | Constitutional AI, Value learning       |
| **System Safety**         | Cascading failures, Reward hacking       | Safe interruptibility, System monitoring|
| **Regulatory Compliance** | Audit trails, Explainability requirements | Model cards, Governance frameworks      |

---

## 📚 Research Papers(coming soon)
### Foundational Works
- [2023] [Towards Safer Large Reasoning Models: A Survey of Risks in Multistep Reasoning Systems](https://arxiv.org/abs/example)  
  *Comprehensive taxonomy of LRM safety risks*

### Attack Vectors
- [2024] [Hidden Triggers in Reasoning Chains: New Attack Surfaces for LRMs](https://arxiv.org/abs/example)  
  *Demonstrates adversarial manipulation of reasoning steps*

### Defense Mechanisms
- [2024] [Reasoning with Guardrails: Constrained Decoding for LRM Safety](https://arxiv.org/abs/example)  
  *Novel approach to step-wise constraint enforcement*

*(Add your collected papers here with proper categorization)*

---

## 🛠️ Projects & Tools(coming soon)
### Model-Specific Resources
- **DeepSeek-R1 Safety Kit**  
  Official safety evaluation toolkit for DeepSeek-R1 reasoning modules

- **OpenAI o1 Red Teaming Framework**  
  Adversarial testing framework for multi-turn reasoning tasks

### General Tools(coming soon)
- [ReasonGuard](https://github.com/example/reasonguard)  
  Real-time monitoring for reasoning chain anomalies

- [Ethos](https://github.com/example/ethos)  
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

> *"With great reasoning power comes great responsibility."* - Adapted from [AI Ethics Manifesto]
