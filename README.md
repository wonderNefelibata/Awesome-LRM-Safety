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

## 📰 Latest arXiv Papers (Auto-Updated)
<!-- ARXIV_PAPERS_START -->

| Date       | Title                                      | Authors           | Abstract                                      |
|------------|--------------------------------------------|-------------------|-----------------------------------------------|
| 2025-02-17 | [FedEAT: A Robustness Optimization Framework for Federated LLMs](http://arxiv.org/abs/2502.11863v1) | Yahao Pang, Xingyuan Wu et al. | Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creatio... |
| 2025-02-17 | [Rethinking Audio-Visual Adversarial Vulnerability from Temporal and Modality Perspectives](http://arxiv.org/abs/2502.11858v1) | Zeliang Zhang, Susan Liang et al. | While audio-visual learning equips models with a richer understanding of the real world by leveraging multiple sensory modalities, this integration al... |
| 2025-02-17 | [VLDBench: Vision Language Models Disinformation Detection Benchmark](http://arxiv.org/abs/2502.11361v1) | Shaina Raza, Ashmal Vayani et al. | The rapid rise of AI-generated content has made detecting disinformation increasingly challenging. In particular, multimodal disinformation, i.e., onl... |
| 2025-02-16 | [G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems](http://arxiv.org/abs/2502.11127v1) | Shilong Wang, Guibin Zhang et al. | Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborat... |
| 2025-02-14 | [Fast Proxies for LLM Robustness Evaluation](http://arxiv.org/abs/2502.10487v1) | Tim Beyer, Jan Schuchardt et al. | Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expen... |
| 2025-02-13 | [SyntheticPop: Attacking Speaker Verification Systems With Synthetic VoicePops](http://arxiv.org/abs/2502.09553v1) | Eshaq Jamdar, Amith Kamath Belman et al. | Voice Authentication (VA), also known as Automatic Speaker Verification (ASV), is a widely adopted authentication method, particularly in automated sy... |
| 2025-02-13 | [Wasserstein distributional adversarial training for deep neural networks](http://arxiv.org/abs/2502.09352v1) | Xingjian Bai, Guangyi He et al. | Design of adversarial attacks for deep neural networks, as well as methods of adversarial training against them, are subject of intense research. In t... |
| 2025-02-13 | [AI Safety for Everyone](http://arxiv.org/abs/2502.09288v2) | Balint Gyevnar, Atoosa Kasirzadeh et al. | Recent discussions and research in AI safety have increasingly emphasized the deep connection between AI safety and existential risk from advanced AI ... |
| 2025-02-13 | [LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection](http://arxiv.org/abs/2502.09271v2) | Wenlun Zhang, Enyan Dai et al. | Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their suscep... |
| 2025-02-13 | [FLAME: Flexible LLM-Assisted Moderation Engine](http://arxiv.org/abs/2502.09175v1) | Ivan Bakulin, Ilia Kopanichuk et al. | The rapid advancement of Large Language Models (LLMs) has introduced significant challenges in moderating user-model interactions. While LLMs demonstr... |
| 2025-02-13 | [Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks](http://arxiv.org/abs/2502.09110v1) | Eylon Mizrahi, Raz Lapid et al. | Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations ... |
| 2025-02-13 | [ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech](http://arxiv.org/abs/2502.08857v2) | Xin Wang, Héctor Delgado et al. | ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of det... |
| 2025-02-12 | [AdvSwap: Covert Adversarial Perturbation with High Frequency Info-swapping for Autonomous Driving Perception](http://arxiv.org/abs/2502.08374v1) | Yuanhao Huang, Qinfan Zhang et al. | Perception module of Autonomous vehicles (AVs) are increasingly susceptible to be attacked, which exploit vulnerabilities in neural networks through a... |
| 2025-02-12 | [MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models](http://arxiv.org/abs/2502.08079v1) | Peng-Fei Zhang, Guangdong Bai et al. | Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transfe... |
| 2025-02-12 | [Quaternion-Hadamard Network: A Novel Defense Against Adversarial Attacks with a New Dataset](http://arxiv.org/abs/2502.10452v1) | Vladimir Frants, Sos Agaian et al. | This paper addresses the vulnerability of deep-learning models designed for rain, snow, and haze removal. Despite enhancing image quality in adverse w... |
| 2025-02-11 | [Universal Adversarial Attack on Aligned Multimodal LLMs](http://arxiv.org/abs/2502.07987v2) | Temurbek Rahmatullaev, Polina Druzhinina et al. | We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment saf... |
| 2025-02-11 | [Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority](http://arxiv.org/abs/2502.07603v1) | Ram Padmanabhan, Melkior Ornik et al. | In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers ... |
| 2025-02-11 | [SEMU: Singular Value Decomposition for Efficient Machine Unlearning](http://arxiv.org/abs/2502.07587v1) | Marcin Sendera, Łukasz Struski et al. | While the capabilities of generative foundational models have advanced rapidly in recent years, methods to prevent harmful and unsafe behaviors remain... |
| 2025-02-11 | [RoMA: Robust Malware Attribution via Byte-level Adversarial Training with Global Perturbations and Adversarial Consistency Regularization](http://arxiv.org/abs/2502.07492v2) | Yuxia Sun, Huihong Chen et al. | Attributing APT (Advanced Persistent Threat) malware to their respective groups is crucial for threat intelligence and cybersecurity. However, APT adv... |
| 2025-02-11 | [LUNAR: LLM Unlearning via Neural Activation Redirection](http://arxiv.org/abs/2502.07218v1) | William F. Shen, Xinchi Qiu et al. | Large Language Models (LLMs) benefit from training on ever larger amounts of textual data, but as a result, they increasingly incur the risk of leakin... |
| 2025-02-10 | [Krum Federated Chain (KFC): Using blockchain to defend against adversarial attacks in Federated Learning](http://arxiv.org/abs/2502.06917v1) | Mario García-Márquez, Nuria Rodríguez-Barroso et al. | Federated Learning presents a nascent approach to machine learning, enabling collaborative model training across decentralized devices while safeguard... |
| 2025-02-10 | [Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection](http://arxiv.org/abs/2502.07821v1) | Dongsu Song, Daehwa Ko et al. | It is well known that query-based attacks tend to have relatively higher success rates in adversarial black-box attacks. While research on black-box a... |
| 2025-02-09 | [Jailbreaking to Jailbreak](http://arxiv.org/abs/2502.09638v1) | Jeremy Kritz, Vaughn Robinson et al. | Refusal training on Large Language Models (LLMs) prevents harmful outputs, yet this defense remains vulnerable to both automated and human-crafted jai... |
| 2025-02-09 | [Optimization under Attack: Resilience, Vulnerability, and the Path to Collapse](http://arxiv.org/abs/2502.05954v1) | Amal Aldawsari, Evangelos Pournaras et al. | Optimization is instrumental for improving operations of large-scale socio-technical infrastructures of Smart Cities, for instance, energy and traffic... |
| 2025-02-09 | [Protecting Intellectual Property of EEG-based Neural Networks with Watermarking](http://arxiv.org/abs/2502.05931v1) | Ahmed Abdelaziz, Ahmed Fathi et al. | EEG-based neural networks, pivotal in medical diagnosis and brain-computer interfaces, face significant intellectual property (IP) risks due to their ... |
| 2025-02-09 | [Assessing confidence in frontier AI safety cases](http://arxiv.org/abs/2502.05791v1) | Stephen Barrett, Philip Fox et al. | Powerful new frontier AI technologies are bringing many benefits to society but at the same time bring new risks. AI developers and regulators are the... |
| 2025-02-09 | [Effective Black-Box Multi-Faceted Attacks Breach Vision Large Language Model Guardrails](http://arxiv.org/abs/2502.05772v1) | Yijun Yang, Lichao Wang et al. | Vision Large Language Models (VLLMs) integrate visual data processing, expanding their real-world applications, but also increasing the risk of genera... |
| 2025-02-08 | [Rigid Body Adversarial Attacks](http://arxiv.org/abs/2502.05669v1) | Aravind Ramakrishnan, David I. W. Levin et al. | Due to their performance and simplicity, rigid body simulators are often used in applications where the objects of interest can considered very stiff.... |
| 2025-02-08 | [Democratic Training Against Universal Adversarial Perturbations](http://arxiv.org/abs/2502.05542v1) | Bing Sun, Jun Sun et al. | Despite their advances and success, real-world deep neural networks are known to be vulnerable to adversarial attacks. Universal adversarial perturbat... |
| 2025-02-08 | [Do Spikes Protect Privacy? Investigating Black-Box Model Inversion Attacks in Spiking Neural Networks](http://arxiv.org/abs/2502.05509v1) | Hamed Poursiami, Ayana Moshruba et al. | As machine learning models become integral to security-sensitive applications, concerns over data leakage from adversarial attacks continue to rise. M... |

<details><summary>View Older Papers</summary>

| Date       | Title                                      | Authors           | Abstract                                      |
|------------|--------------------------------------------|-------------------|-----------------------------------------------|
| 2025-02-08 | [Towards Trustworthy Retrieval Augmented Generation for Large Language Models: A Survey](http://arxiv.org/abs/2502.06872v1) | Bo Ni, Zheyuan Liu et al. | Retrieval-Augmented Generation (RAG) is an advanced technique designed to address the challenges of Artificial Intelligence-Generated Content (AIGC). ... |
| 2025-02-08 | [Forbidden Science: Dual-Use AI Challenge Benchmark and Scientific Refusal Tests](http://arxiv.org/abs/2502.06867v1) | David Noever, Forrest McKee et al. | The development of robust safety benchmarks for large language models requires open, reproducible datasets that can measure both appropriate refusal o... |
| 2025-02-08 | [The Odyssey of the Fittest: Can Agents Survive and Still Be Good?](http://arxiv.org/abs/2502.05442v1) | Dylan Waldner, Risto Miikkulainen et al. | As AI models grow in power and generality, understanding how agents learn and make decisions in complex environments is critical to promoting ethical ... |
| 2025-02-07 | [Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond](http://arxiv.org/abs/2502.05374v1) | Chongyu Fan, Jinghan Jia et al. | The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by remov... |
| 2025-02-07 | [Federated Learning for Anomaly Detection in Energy Consumption Data: Assessing the Vulnerability to Adversarial Attacks](http://arxiv.org/abs/2502.05041v1) | Yohannis Kifle Telila, Damitha Senevirathne et al. | Anomaly detection is crucial in the energy sector to identify irregular patterns indicating equipment failures, energy theft, or other issues. Machine... |
| 2025-02-07 | [DMPA: Model Poisoning Attacks on Decentralized Federated Learning for Model Differences](http://arxiv.org/abs/2502.04771v1) | Chao Feng, Yunlong Li et al. | Federated learning (FL) has garnered significant attention as a prominent privacy-preserving Machine Learning (ML) paradigm. Decentralized FL (DFL) es... |
| 2025-02-07 | [Mechanistic Understandings of Representation Vulnerabilities and Engineering Robust Vision Transformers](http://arxiv.org/abs/2502.04679v1) | Chashi Mahiul Islam, Samuel Jacob Chacko et al. | While transformer-based models dominate NLP and vision applications, their underlying mechanisms to map the input space to the label space semanticall... |
| 2025-02-07 | [Confidence Elicitation: A New Attack Vector for Large Language Models](http://arxiv.org/abs/2502.04643v2) | Brian Formento, Chuan Sheng Foo et al. | A fundamental issue in deep learning has been adversarial robustness. As these systems have scaled, such issues have persisted. Currently, large langu... |
| 2025-02-06 | [BitAbuse: A Dataset of Visually Perturbed Texts for Defending Phishing Attacks](http://arxiv.org/abs/2502.05225v1) | Hanyong Lee, Chaelyn Lee et al. | Phishing often targets victims through visually perturbed texts to bypass security systems. The noise contained in these texts functions as an adversa... |
| 2025-02-06 | [How vulnerable is my policy? Adversarial attacks on modern behavior cloning policies](http://arxiv.org/abs/2502.03698v1) | Basavasagar Patil, Akansha Kalra et al. | Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to adversarial attack... |
| 2025-02-05 | [Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach](http://arxiv.org/abs/2502.06832v2) | Xu Zhang, Kaidi Xu et al. | Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their su... |
| 2025-02-05 | [Enabling External Scrutiny of AI Systems with Privacy-Enhancing Technologies](http://arxiv.org/abs/2502.05219v1) | Kendrea Beers, Helen Toner et al. | This article describes how technical infrastructure developed by the nonprofit OpenMined enables external scrutiny of AI systems without compromising ... |
| 2025-02-05 | [Large Language Model Adversarial Landscape Through the Lens of Attack Objectives](http://arxiv.org/abs/2502.02960v1) | Nan Wang, Kane Walter et al. | Large Language Models (LLMs) represent a transformative leap in artificial intelligence, enabling the comprehension, generation, and nuanced interacti... |
| 2025-02-05 | [Real-Time Privacy Risk Measurement with Privacy Tokens for Gradient Leakage](http://arxiv.org/abs/2502.02913v4) | Jiayang Meng, Tao Huang et al. | The widespread deployment of deep learning models in privacy-sensitive domains has amplified concerns regarding privacy risks, particularly those stem... |
| 2025-02-05 | [Wolfpack Adversarial Attack for Robust Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2502.02844v2) | Sunwoo Lee, Jaebak Hwang et al. | Traditional robust methods in multi-agent reinforcement learning (MARL) often struggle against coordinated adversarial attacks in cooperative scenario... |
| 2025-02-05 | [MARAGE: Transferable Multi-Model Adversarial Attack for Retrieval-Augmented Generation Data Extraction](http://arxiv.org/abs/2502.04360v1) | Xiao Hu, Eric Liu et al. | Retrieval-Augmented Generation (RAG) offers a solution to mitigate hallucinations in Large Language Models (LLMs) by grounding their outputs to knowle... |
| 2025-02-04 | [Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks](http://arxiv.org/abs/2502.02537v1) | Huiqun Huang, Cong Chen et al. | Collaborative Object Detection (COD) and collaborative perception can integrate data or features from various entities, and improve object detection a... |
| 2025-02-04 | [CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models](http://arxiv.org/abs/2502.05214v1) | Amy Rafferty, Rishi Ramaesh et al. | Deep learning models for medical image classification tasks are becoming widely implemented in AI-assisted diagnostic tools, aiming to enhance diagnos... |
| 2025-02-04 | [FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection](http://arxiv.org/abs/2502.02290v1) | Daniele Lunghi, Yannick Molinghen et al. | Adversarial attacks pose a significant threat to data-driven systems, and researchers have spent considerable resources studying them. Despite its eco... |
| 2025-02-04 | [Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization](http://arxiv.org/abs/2502.02096v2) | Yixiao Chen, Shikun Sun et al. | Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Ex... |

</details>

---

## 🔑 Key Safety Domains
| Category               | Key Challenges                          | Related Topics                          |
|------------------------|-----------------------------------------|------------------------------------------|
| **Adversarial Robustness** | Prompt injection, Reasoning path poisoning | Red teaming, Formal verification        |
| **Privacy Preservation**  | Intermediate step memorization, Data leakage | Differential privacy, Federated learning|
| **Ethical Alignment**     | Value locking, Contextual moral reasoning | Constitutional AI, Value learning       |
| **System Safety**         | Cascading failures, Reward hacking       | Safe interruptibility, System monitoring|
| **Regulatory Compliance** | Audit trails, Explainability requirements | Model cards, Governance frameworks      |

---

## 📚 Research Papers
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

## 🛠️ Projects & Tools
### Model-Specific Resources
- **DeepSeek-R1 Safety Kit**  
  Official safety evaluation toolkit for DeepSeek-R1 reasoning modules

- **OpenAI o1 Red Teaming Framework**  
  Adversarial testing framework for multi-turn reasoning tasks

### General Tools
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
