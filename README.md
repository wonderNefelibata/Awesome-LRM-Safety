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
| 2025-04-11 | [BiFlex: A Passive Bimodal Stiffness Flexible Wrist for Manipulation in Unstructured Environments](http://arxiv.org/abs/2504.08706v1) | Gu-Cheol Jeong, Stefano Dalla Gasperina et al. | Robotic manipulation in unstructured, humancentric environments poses a dual challenge: achieving the precision need for delicate free-space operation while ensuring safety during unexpected contact events. Traditional wrists struggle to balance these demands, often relying on complex control schemes or complicated mechanical designs to mitigate potential damage from force overload. In response, we present BiFlex, a flexible robotic wrist that uses a soft buckling honeycomb structure to provides a natural bimodal stiffness response. The higher stiffness mode enables precise household object manipulation, while the lower stiffness mode provides the compliance needed to adapt to external forces. We design BiFlex to maintain a fingertip deflection of less than 1 cm while supporting loads up to 500g and create a BiFlex wrist for many grippers, including Panda, Robotiq, and BaRiFlex. We validate BiFlex under several real-world experimental evaluations, including surface wiping, precise pick-and-place, and grasping under environmental constraints. We demonstrate that BiFlex simplifies control while maintaining precise object manipulation and enhanced safety in real-world applications. |
| 2025-04-11 | [Offline Reinforcement Learning using Human-Aligned Reward Labeling for Autonomous Emergency Braking in Occluded Pedestrian Crossing](http://arxiv.org/abs/2504.08704v1) | Vinal Asodia, Zhenhua Feng et al. | Effective leveraging of real-world driving datasets is crucial for enhancing the training of autonomous driving systems. While Offline Reinforcement Learning enables the training of autonomous vehicles using such data, most available datasets lack meaningful reward labels. Reward labeling is essential as it provides feedback for the learning algorithm to distinguish between desirable and undesirable behaviors, thereby improving policy performance. This paper presents a novel pipeline for generating human-aligned reward labels. The proposed approach addresses the challenge of absent reward signals in real-world datasets by generating labels that reflect human judgment and safety considerations. The pipeline incorporates an adaptive safety component, activated by analyzing semantic segmentation maps, allowing the autonomous vehicle to prioritize safety over efficiency in potential collision scenarios. The proposed pipeline is applied to an occluded pedestrian crossing scenario with varying levels of pedestrian traffic, using synthetic and simulation data. The results indicate that the generated reward labels closely match the simulation reward labels. When used to train the driving policy using Behavior Proximal Policy Optimisation, the results are competitive with other baselines. This demonstrates the effectiveness of our method in producing reliable and human-aligned reward signals, facilitating the training of autonomous driving systems through Reinforcement Learning outside of simulation environments and in alignment with human values. |
| 2025-04-11 | [Safe Flow Matching: Robot Motion Planning with Control Barrier Functions](http://arxiv.org/abs/2504.08661v1) | Xiaobing Dai, Dian Yu et al. | Recent advances in generative modeling have led to promising results in robot motion planning, particularly through diffusion and flow-based models that capture complex, multimodal trajectory distributions. However, these methods are typically trained offline and remain limited when faced with unseen environments or dynamic constraints, often lacking explicit mechanisms to ensure safety during deployment. In this work, we propose, Safe Flow Matching (SafeFM), a motion planning approach for trajectory generation that integrates flow matching with safety guarantees. By incorporating the proposed flow matching barrier functions, SafeFM ensures that generated trajectories remain within safe regions throughout the planning horizon, even in the presence of previously unseen obstacles or state-action constraints. Unlike diffusion-based approaches, our method allows for direct, efficient sampling of constraint-satisfying trajectories, making it well-suited for real-time motion planning. We evaluate SafeFM on a diverse set of tasks, including planar robot navigation and 7-DoF manipulation, demonstrating superior safety, generalization, and planning performance compared to state-of-the-art generative planners. Comprehensive resources are available on the project website: https://safeflowmatching.github.io/SafeFM/ |
| 2025-04-11 | [TinyCenterSpeed: Efficient Center-Based Object Detection for Autonomous Racing](http://arxiv.org/abs/2504.08655v1) | Neil Reichlin, Nicolas Baumann et al. | Perception within autonomous driving is nearly synonymous with Neural Networks (NNs). Yet, the domain of autonomous racing is often characterized by scaled, computationally limited robots used for cost-effectiveness and safety. For this reason, opponent detection and tracking systems typically resort to traditional computer vision techniques due to computational constraints. This paper introduces TinyCenterSpeed, a streamlined adaptation of the seminal CenterPoint method, optimized for real-time performance on 1:10 scale autonomous racing platforms. This adaptation is viable even on OBCs powered solely by Central Processing Units (CPUs), as it incorporates the use of an external Tensor Processing Unit (TPU). We demonstrate that, compared to Adaptive Breakpoint Detector (ABD), the current State-of-the-Art (SotA) in scaled autonomous racing, TinyCenterSpeed not only improves detection and velocity estimation by up to 61.38% but also supports multi-opponent detection and estimation. It achieves real-time performance with an inference time of just 7.88 ms on the TPU, significantly reducing CPU utilization 8.3-fold. |
| 2025-04-11 | [Deep Learning Methods for Detecting Thermal Runaway Events in Battery Production Lines](http://arxiv.org/abs/2504.08632v1) | Athanasios Athanasopoulos, Matúš Mihalák et al. | One of the key safety considerations of battery manufacturing is thermal runaway, the uncontrolled increase in temperature which can lead to fires, explosions, and emissions of toxic gasses. As such, development of automated systems capable of detecting such events is of considerable importance in both academic and industrial contexts. In this work, we investigate the use of deep learning for detecting thermal runaway in the battery production line of VDL Nedcar, a Dutch automobile manufacturer. Specifically, we collect data from the production line to represent both baseline (non thermal runaway) and thermal runaway conditions. Thermal runaway was simulated through the use of external heat and smoke sources. The data consisted of both optical and thermal images which were then preprocessed and fused before serving as input to our models. In this regard, we evaluated three deep-learning models widely used in computer vision including shallow convolutional neural networks, residual neural networks, and vision transformers on two performance metrics. Furthermore, we evaluated these models using explainability methods to gain insight into their ability to capture the relevant feature information from their inputs. The obtained results indicate that the use of deep learning is a viable approach to thermal runaway detection in battery production lines. |
| 2025-04-11 | [Enabling Safety for Aerial Robots: Planning and Control Architectures](http://arxiv.org/abs/2504.08601v1) | Kaleb Ben Naveed, Devansh R. Agrawal et al. | Ensuring safe autonomy is crucial for deploying aerial robots in real-world applications. However, safety is a multifaceted challenge that must be addressed from multiple perspectives, including navigation in dynamic environments, operation under resource constraints, and robustness against adversarial attacks and uncertainties. In this paper, we present the authors' recent work that tackles some of these challenges and highlights key aspects that must be considered to enhance the safety and performance of autonomous aerial systems. All presented approaches are validated through hardware experiments. |
| 2025-04-11 | [Secondary Safety Control for Systems with Sector Bounded Nonlinearities](http://arxiv.org/abs/2504.08535v1) | Yankai Lin, Michelle S. Chong et al. | We consider the problem of safety verification and safety-aware controller synthesis for systems with sector bounded nonlinearities. We aim to keep the states of the system within a given safe set under potential actuator and sensor attacks. Specifically, we adopt the setup that a controller has already been designed to stabilize the plant. Using invariant sets and barrier certificate theory, we first give sufficient conditions to verify the safety of the closed-loop system under attacks. Furthermore, by using a subset of sensors that are assumed to be free of attacks, we provide a synthesis method for a secondary controller that enhances the safety of the system. The sufficient conditions to verify safety are derived using Lyapunov-based tools and the S-procedure. Using the projection lemma, the conditions are then formulated as linear matrix inequality (LMI) problems which can be solved efficiently. Lastly, our theoretical results are illustrated through numerical simulations. |
| 2025-04-11 | [Secondary Safety Control for Systems with Sector Bounded Nonlinearities [Extended Version]](http://arxiv.org/abs/2504.08535v2) | Yankai Lin, Michelle S. Chong et al. | We consider the problem of safety verification and safety-aware controller synthesis for systems with sector bounded nonlinearities. We aim to keep the states of the system within a given safe set under potential actuator and sensor attacks. Specifically, we adopt the setup that a controller has already been designed to stabilize the plant. Using invariant sets and barrier certificate theory, we first give sufficient conditions to verify the safety of the closed-loop system under attacks. Furthermore, by using a subset of sensors that are assumed to be free of attacks, we provide a synthesis method for a secondary controller that enhances the safety of the system. The sufficient conditions to verify safety are derived using Lyapunov-based tools and the S-procedure. Using the projection lemma, the conditions are then formulated as linear matrix inequality (LMI) problems which can be solved efficiently. Lastly, our theoretical results are illustrated through numerical simulations. |
| 2025-04-11 | [Physics-informed data-driven control without persistence of excitation](http://arxiv.org/abs/2504.08484v1) | Martina Vanelli, Julien M. Hendrickx | We show that data that is not sufficiently informative to allow for system re-identification can still provide meaningful information when combined with external or physical knowledge of the system, such as bounded system matrix norms. We then illustrate how this information can be leveraged for safety and energy minimization problems and to enhance predictions in unmodelled dynamics. This preliminary work outlines key ideas toward using limited data for effective control by integrating physical knowledge of the system and exploiting interpolation conditions. |
| 2025-04-11 | [Constrained Machine Learning Through Hyperspherical Representation](http://arxiv.org/abs/2504.08415v1) | Gaetano Signorelli, Michele Lombardi | The problem of ensuring constraints satisfaction on the output of machine learning models is critical for many applications, especially in safety-critical domains. Modern approaches rely on penalty-based methods at training time, which do not guarantee to avoid constraints violations; or constraint-specific model architectures (e.g., for monotonocity); or on output projection, which requires to solve an optimization problem that might be computationally demanding. We present the Hypersherical Constrained Representation, a novel method to enforce constraints in the output space for convex and bounded feasibility regions (generalizable to star domains). Our method operates on a different representation system, where Euclidean coordinates are converted into hyperspherical coordinates relative to the constrained region, which can only inherently represent feasible points. Experiments on a synthetic and a real-world dataset show that our method has predictive performance comparable to the other approaches, can guarantee 100% constraint satisfaction, and has a minimal computational cost at inference time. |
| 2025-04-11 | [Evaluating Pedestrian Risks in Shared Spaces Through Autonomous Vehicle Experiments on a Fixed Track](http://arxiv.org/abs/2504.08316v1) | Enrico Del Re, Novel Certad et al. | The majority of research on safety in autonomous vehicles has been conducted in structured and controlled environments. However, there is a scarcity of research on safety in unregulated pedestrian areas, especially when interacting with public transport vehicles like trams. This study investigates pedestrian responses to an alert system in this context by replicating this real-world scenario in an environment using an autonomous vehicle. The results show that safety measures from other contexts can be adapted to shared spaces with trams, where fixed tracks heighten risks in unregulated crossings. |
| 2025-04-11 | [SortBench: Benchmarking LLMs based on their ability to sort lists](http://arxiv.org/abs/2504.08312v1) | Steffen Herbold | Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models. |
| 2025-04-11 | [Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare](http://arxiv.org/abs/2504.08260v1) | Yonchanok Khaokaew, Flora D. Salim et al. | Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies. |
| 2025-04-11 | [Neural Network-assisted Interval Reachability for Systems with Control Barrier Function-Based Safe Controllers](http://arxiv.org/abs/2504.08249v1) | Damola Ajeyemi, Saber Jafarpour et al. | Control Barrier Functions (CBFs) have been widely utilized in the design of optimization-based controllers and filters for dynamical systems to ensure forward invariance of a given set of safe states. While CBF-based controllers offer safety guarantees, they can compromise the performance of the system, leading to undesirable behaviors such as unbounded trajectories and emergence of locally stable spurious equilibria. Computing reachable sets for systems with CBF-based controllers is an effective approach for runtime performance and stability verification, and can potentially serve as a tool for trajectory re-planning. In this paper, we propose a computationally efficient interval reachability method for performance verification of systems with optimization-based controllers by: (i) approximating the optimization-based controller by a pre-trained neural network to avoid solving optimization problems repeatedly, and (ii) using mixed monotone theory to construct an embedding system that leverages state-of-the-art neural network verification algorithms for bounding the output of the neural network. Results in terms of closeness of solutions of trajectories of the system with the optimization-based controller and the neural network are derived. Using a single trajectory of the embedding system along with our closeness of solutions result, we obtain an over-approximation of the reachable set of the system with optimization-based controllers. Numerical results are presented to corroborate the technical findings. |
| 2025-04-11 | [InSPE: Rapid Evaluation of Heterogeneous Multi-Modal Infrastructure Sensor Placement](http://arxiv.org/abs/2504.08240v1) | Zhaoliang Zheng, Yun Zhang et al. | Infrastructure sensing is vital for traffic monitoring at safety hotspots (e.g., intersections) and serves as the backbone of cooperative perception in autonomous driving. While vehicle sensing has been extensively studied, infrastructure sensing has received little attention, especially given the unique challenges of diverse intersection geometries, complex occlusions, varying traffic conditions, and ambient environments like lighting and weather. To address these issues and ensure cost-effective sensor placement, we propose Heterogeneous Multi-Modal Infrastructure Sensor Placement Evaluation (InSPE), a perception surrogate metric set that rapidly assesses perception effectiveness across diverse infrastructure and environmental scenarios with combinations of multi-modal sensors. InSPE systematically evaluates perception capabilities by integrating three carefully designed metrics, i.e., sensor coverage, perception occlusion, and information gain. To support large-scale evaluation, we develop a data generation tool within the CARLA simulator and also introduce Infra-Set, a dataset covering diverse intersection types and environmental conditions. Benchmarking experiments with state-of-the-art perception algorithms demonstrate that InSPE enables efficient and scalable sensor placement analysis, providing a robust solution for optimizing intelligent intersection infrastructure. |
| 2025-04-11 | [Advancing Autonomous Vehicle Safety: A Combined Fault Tree Analysis and Bayesian Network Approach](http://arxiv.org/abs/2504.08206v1) | Lansu Dai, Burak Kantarci | This paper integrates Fault Tree Analysis (FTA) and Bayesian Networks (BN) to assess collision risk and establish Automotive Safety Integrity Level (ASIL) B failure rate targets for critical autonomous vehicle (AV) components. The FTA-BN integration combines the systematic decomposition of failure events provided by FTA with the probabilistic reasoning capabilities of BN, which allow for dynamic updates in failure probabilities, enhancing the adaptability of risk assessment. A fault tree is constructed based on AV subsystem architecture, with collision as the top event, and failure rates are assigned while ensuring the total remains within 100 FIT. Bayesian inference is applied to update posterior probabilities, and the results indicate that perception system failures (46.06 FIT) are the most significant contributor, particularly failures to detect existing objects (PF5) and misclassification (PF6). Mitigation strategies are proposed for sensors, perception, decision-making, and motion control to reduce the collision risk. The FTA-BN integration approach provides dynamic risk quantification, offering system designers refined failure rate targets to improve AV safety. |
| 2025-04-11 | [EO-VLM: VLM-Guided Energy Overload Attacks on Vision Models](http://arxiv.org/abs/2504.08205v1) | Minjae Seo, Myoungsung You et al. | Vision models are increasingly deployed in critical applications such as autonomous driving and CCTV monitoring, yet they remain susceptible to resource-consuming attacks. In this paper, we introduce a novel energy-overloading attack that leverages vision language model (VLM) prompts to generate adversarial images targeting vision models. These images, though imperceptible to the human eye, significantly increase GPU energy consumption across various vision models, threatening the availability of these systems. Our framework, EO-VLM (Energy Overload via VLM), is model-agnostic, meaning it is not limited by the architecture or type of the target vision model. By exploiting the lack of safety filters in VLMs like DALL-E 3, we create adversarial noise images without requiring prior knowledge or internal structure of the target vision models. Our experiments demonstrate up to a 50% increase in energy consumption, revealing a critical vulnerability in current vision models. |
| 2025-04-11 | [SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs](http://arxiv.org/abs/2504.08192v1) | Aashiq Muhamed, Jacopo Bonato et al. | Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning. |
| 2025-04-11 | [Safe Data-Driven Predictive Control](http://arxiv.org/abs/2504.08188v1) | Amin Vahidi-Moghaddam, Kaian Chen et al. | In the realm of control systems, model predictive control (MPC) has exhibited remarkable potential; however, its reliance on accurate models and substantial computational resources has hindered its broader application, especially within real-time nonlinear systems. This study presents an innovative control framework to enhance the practical viability of the MPC. The developed safe data-driven predictive control aims to eliminate the requirement for precise models and alleviate computational burdens in the nonlinear MPC (NMPC). This is achieved by learning both the system dynamics and the control policy, enabling efficient data-driven predictive control while ensuring system safety. The methodology involves a spatial temporal filter (STF)-based concurrent learning for system identification, a robust control barrier function (RCBF) to ensure the system safety amid model uncertainties, and a RCBF-based NMPC policy approximation. An online policy correction mechanism is also introduced to counteract performance degradation caused by the existing model uncertainties. Demonstrated through simulations on two applications, the proposed approach offers comparable performance to existing benchmarks with significantly reduced computational costs. |
| 2025-04-10 | [Investigating Vision-Language Model for Point Cloud-based Vehicle Classification](http://arxiv.org/abs/2504.08154v1) | Yiqiao Li, Jie Wei et al. | Heavy-duty trucks pose significant safety challenges due to their large size and limited maneuverability compared to passenger vehicles. A deeper understanding of truck characteristics is essential for enhancing the safety perspective of cooperative autonomous driving. Traditional LiDAR-based truck classification methods rely on extensive manual annotations, which makes them labor-intensive and costly. The rapid advancement of large language models (LLMs) trained on massive datasets presents an opportunity to leverage their few-shot learning capabilities for truck classification. However, existing vision-language models (VLMs) are primarily trained on image datasets, which makes it challenging to directly process point cloud data. This study introduces a novel framework that integrates roadside LiDAR point cloud data with VLMs to facilitate efficient and accurate truck classification, which supports cooperative and safe driving environments. This study introduces three key innovations: (1) leveraging real-world LiDAR datasets for model development, (2) designing a preprocessing pipeline to adapt point cloud data for VLM input, including point cloud registration for dense 3D rendering and mathematical morphological techniques to enhance feature representation, and (3) utilizing in-context learning with few-shot prompting to enable vehicle classification with minimally labeled training data. Experimental results demonstrate encouraging performance of this method and present its potential to reduce annotation efforts while improving classification accuracy. |

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



