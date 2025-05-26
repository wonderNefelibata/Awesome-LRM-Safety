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
| 2025-05-23 | [First Finish Search: Efficient Test-Time Scaling in Large Language Models](http://arxiv.org/abs/2505.18149v1) | Aradhye Agarwal, Ayan Sengupta et al. | Test-time scaling (TTS), which involves dynamic allocation of compute during inference, offers a promising way to improve reasoning in large language models. While existing TTS methods work well, they often rely on long decoding paths or require a large number of samples to be generated, increasing the token usage and inference latency. We observe the surprising fact that for reasoning tasks, shorter traces are much more likely to be correct than longer ones. Motivated by this, we introduce First Finish Search (FFS), a training-free parallel decoding strategy that launches $n$ independent samples and returns as soon as any one completes. We evaluate FFS alongside simple decoding, beam search, majority voting, and budget forcing on four reasoning models (DeepSeek-R1, R1-Distill-Qwen-32B, QwQ-32B and Phi-4-Reasoning-Plus) and across four datasets (AIME24, AIME25-I, AIME25-II and GPQA Diamond). With DeepSeek-R1, FFS achieves $82.23\%$ accuracy on the AIME datasets, a $15\%$ improvement over DeepSeek-R1's standalone accuracy, nearly matching OpenAI's o4-mini performance. Our theoretical analysis explains why stopping at the shortest trace is likely to yield a correct answer and identifies the conditions under which early stopping may be suboptimal. The elegance and simplicity of FFS demonstrate that straightforward TTS strategies can perform remarkably well, revealing the untapped potential of simple approaches at inference time. |
| 2025-05-23 | [Stable Reinforcement Learning for Efficient Reasoning](http://arxiv.org/abs/2505.18086v1) | Muzhi Dai, Shixuan Liu et al. | The success of Deepseek-R1 has drawn the LLM community's attention to reinforcement learning (RL) methods like GRPO. However, such rule-based 0/1 outcome reward methods lack the capability to regulate the intermediate reasoning processes during chain-of-thought (CoT) generation, leading to severe overthinking phenomena. In response, recent studies have designed reward functions to reinforce models' behaviors in producing shorter yet correct completions. Nevertheless, we observe that these length-penalty reward functions exacerbate RL training instability: as the completion length decreases, model accuracy abruptly collapses, often occurring early in training. To address this issue, we propose a simple yet effective solution GRPO-$\lambda$, an efficient and stabilized variant of GRPO, which dynamically adjusts the reward strategy by monitoring the correctness ratio among completions within each query-sampled group. A low correctness ratio indicates the need to avoid length penalty that compromises CoT quality, triggering a switch to length-agnostic 0/1 rewards that prioritize reasoning capability. A high ratio maintains length penalties to boost efficiency. Experimental results show that our approach avoids training instability caused by length penalty while maintaining the optimal accuracy-efficiency trade-off. On the GSM8K, GPQA, MATH-500, AMC 2023, and AIME 2024 benchmarks, it improves average accuracy by 1.48% while reducing CoT sequence length by 47.3%. |
| 2025-05-23 | [Mahalanobis++: Improving OOD Detection via Feature Normalization](http://arxiv.org/abs/2505.18032v1) | Maximilian Mueller, Matthias Hein | Detecting out-of-distribution (OOD) examples is an important task for deploying reliable machine learning models in safety-critial applications. While post-hoc methods based on the Mahalanobis distance applied to pre-logit features are among the most effective for ImageNet-scale OOD detection, their performance varies significantly across models. We connect this inconsistency to strong variations in feature norms, indicating severe violations of the Gaussian assumption underlying the Mahalanobis distance estimation. We show that simple $\ell_2$-normalization of the features mitigates this problem effectively, aligning better with the premise of normally distributed data with shared covariance matrix. Extensive experiments on 44 models across diverse architectures and pretraining schemes show that $\ell_2$-normalization improves the conventional Mahalanobis distance-based approaches significantly and consistently, and outperforms other recently proposed OOD detection methods. |
| 2025-05-23 | [SemSegBench & DetecBench: Benchmarking Reliability and Generalization Beyond Classification](http://arxiv.org/abs/2505.18015v1) | Shashank Agnihotri, David Schader et al. | Reliability and generalization in deep learning are predominantly studied in the context of image classification. Yet, real-world applications in safety-critical domains involve a broader set of semantic tasks, such as semantic segmentation and object detection, which come with a diverse set of dedicated model architectures. To facilitate research towards robust model design in segmentation and detection, our primary objective is to provide benchmarking tools regarding robustness to distribution shifts and adversarial manipulations. We propose the benchmarking tools SEMSEGBENCH and DETECBENCH, along with the most extensive evaluation to date on the reliability and generalization of semantic segmentation and object detection models. In particular, we benchmark 76 segmentation models across four datasets and 61 object detectors across two datasets, evaluating their performance under diverse adversarial attacks and common corruptions. Our findings reveal systematic weaknesses in state-of-the-art models and uncover key trends based on architecture, backbone, and model capacity. SEMSEGBENCH and DETECBENCH are open-sourced in our GitHub repository (https://github.com/shashankskagnihotri/benchmarking_reliability_generalization) along with our complete set of total 6139 evaluations. We anticipate the collected data to foster and encourage future research towards improved model reliability beyond classification. |
| 2025-05-23 | [Classification of assembly tasks combining multiple primitive actions using Transformers and xLSTMs](http://arxiv.org/abs/2505.18012v1) | Miguel Neves, Pedro Neto | The classification of human-performed assembly tasks is essential in collaborative robotics to ensure safety, anticipate robot actions, and facilitate robot learning. However, achieving reliable classification is challenging when segmenting tasks into smaller primitive actions is unfeasible, requiring us to classify long assembly tasks that encompass multiple primitive actions. In this study, we propose classifying long assembly sequential tasks based on hand landmark coordinates and compare the performance of two well-established classifiers, LSTM and Transformer, as well as a recent model, xLSTM. We used the HRC scenario proposed in the CT benchmark, which includes long assembly tasks that combine actions such as insertions, screw fastenings, and snap fittings. Testing was conducted using sequences gathered from both the human operator who performed the training sequences and three new operators. The testing results of real-padded sequences for the LSTM, Transformer, and xLSTM models was 72.9%, 95.0% and 93.2% for the training operator, and 43.5%, 54.3% and 60.8% for the new operators, respectively. The LSTM model clearly underperformed compared to the other two approaches. As expected, both the Transformer and xLSTM achieved satisfactory results for the operator they were trained on, though the xLSTM model demonstrated better generalization capabilities to new operators. The results clearly show that for this type of classification, the xLSTM model offers a slight edge over Transformers. |
| 2025-05-23 | [An Example Safety Case for Safeguards Against Misuse](http://arxiv.org/abs/2505.18003v1) | Joshua Clymer, Jonah Weinbaum et al. | Existing evaluations of AI misuse safeguards provide a patchwork of evidence that is often difficult to connect to real-world decisions. To bridge this gap, we describe an end-to-end argument (a "safety case") that misuse safeguards reduce the risk posed by an AI assistant to low levels. We first describe how a hypothetical developer red teams safeguards, estimating the effort required to evade them. Then, the developer plugs this estimate into a quantitative "uplift model" to determine how much barriers introduced by safeguards dissuade misuse (https://www.aimisusemodel.com/). This procedure provides a continuous signal of risk during deployment that helps the developer rapidly respond to emerging threats. Finally, we describe how to tie these components together into a simple safety case. Our work provides one concrete path -- though not the only path -- to rigorously justifying AI misuse risks are low. |
| 2025-05-23 | [Automated Formal Verification of Area-Optimized Safety Registers in Automotive SoCs](http://arxiv.org/abs/2505.17990v1) | Shuhang Zhang, Bryan Olmos | Registers are primary storage elements in System-on-chip~(SoC) designs and play an important role in maintaining state information and processing data in digital systems. With respect to the ISO26262 standard, these registers require high levels of reliability and fault tolerance. For this reason, safety-critical applications require that normal registers are equipped with additional safety components to construct safety registers, which ensure system stability and fault tolerance. However, the process of integrating these safety registers is complex and error-prone, because of highly-configurable features provided by a safety library such as parameterized modules and flexible safety structures. In addition, to address the overhead caused by the safety registers, we have applied area optimization techniques to their implementation. However, this optimization can make the integration process more susceptible to errors. To avoid any integration mistakes, rigorous verification is always required, but it is time-consuming and error-prone if the verification is implemented manually when dealing with numerous verification requests. To address these challenges, we propose an automated flow for the verification of safety registers with the formal approach. The results indicate that this automated verification approach has the potential to reduce the verification effort by more than 80\%. Additionally, it ensures a comprehensive examination of every requirement of this safety library, which is reflected in faster detection of bugs. The proposed framework can be replicated for the verification of other safety components enabling an early detection of potential issues and saving valuable time and resources. |
| 2025-05-23 | [Outcome-based Reinforcement Learning to Predict the Future](http://arxiv.org/abs/2505.17989v1) | Benjamin Turtel, Danny Franklin et al. | Reinforcement learning with verifiable rewards (RLVR) has boosted math and coding in large language models, yet there has been little effort to extend RLVR into messier, real-world domains like forecasting. One sticking point is that outcome-based reinforcement learning for forecasting must learn from binary, delayed, and noisy rewards, a regime where standard fine-tuning is brittle. We show that outcome-only online RL on a 14B model can match frontier-scale accuracy and surpass it in calibration and hypothetical prediction market betting by adapting two leading algorithms, Group-Relative Policy Optimisation (GRPO) and ReMax, to the forecasting setting. Our adaptations remove per-question variance scaling in GRPO, apply baseline-subtracted advantages in ReMax, hydrate training with 100k temporally consistent synthetic questions, and introduce lightweight guard-rails that penalise gibberish, non-English responses and missing rationales, enabling a single stable pass over 110k events. Scaling ReMax to 110k questions and ensembling seven predictions yields a 14B model that matches frontier baseline o1 on accuracy on our holdout set (Brier = 0.193, p = 0.23) while beating it in calibration (ECE = 0.042, p < 0.001). A simple trading rule turns this calibration edge into \$127 of hypothetical profit versus \$92 for o1 (p = 0.037). This demonstrates that refined RLVR methods can convert small-scale LLMs into potentially economically valuable forecasting tools, with implications for scaling this to larger models. |
| 2025-05-23 | [Re-evaluation of Logical Specification in Behavioural Verification](http://arxiv.org/abs/2505.17979v1) | Radoslaw Klimek, Jakub Semczyszyn | This study empirically validates automated logical specification methods for behavioural models, focusing on their robustness, scalability, and reproducibility. By the systematic reproduction and extension of prior results, we confirm key trends, while identifying performance irregularities that suggest the need for adaptive heuristics in automated reasoning. Our findings highlight that theorem provers exhibit varying efficiency across problem structures, with implications for real-time verification in CI/CD pipelines and AI-driven IDEs supporting on-the-fly validation. Addressing these inefficiencies through self-optimising solvers could enhance the stability of automated reasoning, particularly in safety-critical software verification. |
| 2025-05-23 | [Counting Cycles with Deepseek](http://arxiv.org/abs/2505.17964v1) | Jiashun Jin, Tracy Ke et al. | Despite recent progress, AI still struggles on advanced mathematics. We consider a difficult open problem: How to derive a Computationally Efficient Equivalent Form (CEEF) for the cycle count statistic? The CEEF problem does not have known general solutions, and requires delicate combinatorics and tedious calculations. Such a task is hard to accomplish by humans but is an ideal example where AI can be very helpful. We solve the problem by combining a novel approach we propose and the powerful coding skills of AI. Our results use delicate graph theory and contain new formulas for general cases that have not been discovered before. We find that, while AI is unable to solve the problem all by itself, it is able to solve it if we provide it with a clear strategy, a step-by-step guidance and carefully written prompts. For simplicity, we focus our study on DeepSeek-R1 but we also investigate other AI approaches. |
| 2025-05-23 | [Mind the Domain Gap: Measuring the Domain Gap Between Real-World and Synthetic Point Clouds for Automated Driving Development](http://arxiv.org/abs/2505.17959v1) | Nguyen Duc, Yan-Ling Lai et al. | Owing to the typical long-tail data distribution issues, simulating domain-gap-free synthetic data is crucial in robotics, photogrammetry, and computer vision research. The fundamental challenge pertains to credibly measuring the difference between real and simulated data. Such a measure is vital for safety-critical applications, such as automated driving, where out-of-domain samples may impact a car's perception and cause fatal accidents. Previous work has commonly focused on simulating data on one scene and analyzing performance on a different, real-world scene, hampering the disjoint analysis of domain gap coming from networks' deficiencies, class definitions, and object representation. In this paper, we propose a novel approach to measuring the domain gap between the real world sensor observations and simulated data representing the same location, enabling comprehensive domain gap analysis. To measure such a domain gap, we introduce a novel metric DoGSS-PCL and evaluation assessing the geometric and semantic quality of the simulated point cloud. Our experiments corroborate that the introduced approach can be used to measure the domain gap. The tests also reveal that synthetic semantic point clouds may be used for training deep neural networks, maintaining the performance at the 50/50 real-to-synthetic ratio. We strongly believe that this work will facilitate research on credible data simulation and allow for at-scale deployment in automated driving testing and digital twinning. |
| 2025-05-23 | [VeriThinker: Learning to Verify Makes Reasoning Model Efficient](http://arxiv.org/abs/2505.17941v1) | Zigeng Chen, Xinyin Ma et al. | Large Reasoning Models (LRMs) excel at complex tasks using Chain-of-Thought (CoT) reasoning. However, their tendency to overthinking leads to unnecessarily lengthy reasoning chains, dramatically increasing inference costs. To mitigate this issue, we introduce VeriThinker, a novel approach for CoT compression. Unlike conventional methods that fine-tune LRMs directly on the original reasoning task using synthetic concise CoT data, we innovatively fine-tune the model solely through an auxiliary verification task. By training LRMs to accurately verify the correctness of CoT solutions, the LRMs inherently become more discerning about the necessity of subsequent self-reflection steps, thereby effectively suppressing overthinking. Extensive experiments validate that VeriThinker substantially reduces reasoning chain lengths while maintaining or even slightly improving accuracy. When applied to DeepSeek-R1-Distill-Qwen-7B, our approach reduces reasoning tokens on MATH500 from 3790 to 2125 while improving accuracy by 0.8% (94.0% to 94.8%), and on AIME25, tokens decrease from 14321 to 10287 with a 2.1% accuracy gain (38.7% to 40.8%). Additionally, our experiments demonstrate that VeriThinker can also be zero-shot generalized to speculative reasoning. Code is available at https://github.com/czg1225/VeriThinker |
| 2025-05-23 | [Survival Games: Human-LLM Strategic Showdowns under Severe Resource Scarcity](http://arxiv.org/abs/2505.17937v1) | Zhihong Chen, Yiqian Yang et al. | The rapid advancement of large language models (LLMs) raises critical concerns about their ethical alignment, particularly in scenarios where human and AI co-exist under the conflict of interest. This work introduces an extendable, asymmetric, multi-agent simulation-based benchmarking framework to evaluate the moral behavior of LLMs in a novel human-AI co-existence setting featuring consistent living and critical resource management. Building on previous generative agent environments, we incorporate a life-sustaining system, where agents must compete or cooperate for food resources to survive, often leading to ethically charged decisions such as deception, theft, or social influence. We evaluated two types of LLM, DeepSeek and OpenAI series, in a three-agent setup (two humans, one LLM-powered robot), using adapted behavioral detection from the MACHIAVELLI framework and a custom survival-based ethics metric. Our findings reveal stark behavioral differences: DeepSeek frequently engages in resource hoarding, while OpenAI exhibits restraint, highlighting the influence of model design on ethical outcomes. Additionally, we demonstrate that prompt engineering can significantly steer LLM behavior, with jailbreaking prompts significantly enhancing unethical actions, even for highly restricted OpenAI models and cooperative prompts show a marked reduction in unethical actions. Our framework provides a reproducible testbed for quantifying LLM ethics in high-stakes scenarios, offering insights into their suitability for real-world human-AI interactions. |
| 2025-05-23 | [A model-free approach to control barrier functions using funnel control](http://arxiv.org/abs/2505.17887v1) | Lukas Lanza, Johannes Köhler et al. | Control barrier functions (CBFs) are a popular approach to design feedback laws that achieve safety guarantees for nonlinear systems. The CBF-based controller design relies on the availability of a model to select feasible inputs from the set of CBF-based controls. In this paper, we develop a model-free approach to design CBF-based control laws, eliminating the need for knowledge of system dynamics or parameters. Specifically, we address safety requirements characterized by a time-varying distance to a reference trajectory in the output space and construct a CBF that depends only on the measured output. Utilizing this particular CBF, we determine a subset of CBF-based controls without relying on a model of the dynamics by using techniques from funnel control. The latter is a model-free high-gain adaptive control methodology, which achieves tracking guarantees via reactive feedback. In this paper, we discover and establish a connection between the modular controller synthesis via zeroing CBFs and model-free reactive feedback. The theoretical results are illustrated by a numerical simulation. |
| 2025-05-23 | [Out of the Shadows: Exploring a Latent Space for Neural Network Verification](http://arxiv.org/abs/2505.17854v1) | Lukas Koller, Tobias Ladner et al. | Neural networks are ubiquitous. However, they are often sensitive to small input changes. Hence, to prevent unexpected behavior in safety-critical applications, their formal verification -- a notoriously hard problem -- is necessary. Many state-of-the-art verification algorithms use reachability analysis or abstract interpretation to enclose the set of possible outputs of a neural network. Often, the verification is inconclusive due to the conservatism of the enclosure. To address this problem, we design a novel latent space for formal verification that enables the transfer of output specifications to the input space for an iterative specification-driven input refinement, i.e., we iteratively reduce the set of possible inputs to only enclose the unsafe ones. The latent space is constructed from a novel view of projection-based set representations, e.g., zonotopes, which are commonly used in reachability analysis of neural networks. A projection-based set representation is a "shadow" of a higher-dimensional set -- a latent space -- that does not change during a set propagation through a neural network. Hence, the input set and the output enclosure are "shadows" of the same latent space that we can use to transfer constraints. We present an efficient verification tool for neural networks that uses our iterative refinement to significantly reduce the number of subproblems in a branch-and-bound procedure. Using zonotopes as a set representation, unlike many other state-of-the-art approaches, our approach can be realized by only using matrix operations, which enables a significant speed-up through efficient GPU acceleration. We demonstrate that our tool achieves competitive performance, which would place it among the top-ranking tools of the last neural network verification competition (VNN-COMP'24). |
| 2025-05-23 | [Not All Tokens Are What You Need In Thinking](http://arxiv.org/abs/2505.17827v1) | Hang Yuan, Bin Yu et al. | Modern reasoning models, such as OpenAI's o1 and DeepSeek-R1, exhibit impressive problem-solving capabilities but suffer from critical inefficiencies: high inference latency, excessive computational resource consumption, and a tendency toward overthinking -- generating verbose chains of thought (CoT) laden with redundant tokens that contribute minimally to the final answer. To address these issues, we propose Conditional Token Selection (CTS), a token-level compression framework with a flexible and variable compression ratio that identifies and preserves only the most essential tokens in CoT. CTS evaluates each token's contribution to deriving correct answers using conditional importance scoring, then trains models on compressed CoT. Extensive experiments demonstrate that CTS effectively compresses long CoT while maintaining strong reasoning performance. Notably, on the GPQA benchmark, Qwen2.5-14B-Instruct trained with CTS achieves a 9.1% accuracy improvement with 13.2% fewer reasoning tokens (13% training token reduction). Further reducing training tokens by 42% incurs only a marginal 5% accuracy drop while yielding a 75.8% reduction in reasoning tokens, highlighting the prevalence of redundancy in existing CoT. |
| 2025-05-23 | [Evaluation Faking: Unveiling Observer Effects in Safety Evaluation of Frontier AI Systems](http://arxiv.org/abs/2505.17815v1) | Yihe Fan, Wenqi Zhang et al. | As foundation models grow increasingly more intelligent, reliable and trustworthy safety evaluation becomes more indispensable than ever. However, an important question arises: Whether and how an advanced AI system would perceive the situation of being evaluated, and lead to the broken integrity of the evaluation process? During standard safety tests on a mainstream large reasoning model, we unexpectedly observe that the model without any contextual cues would occasionally recognize it is being evaluated and hence behave more safety-aligned. This motivates us to conduct a systematic study on the phenomenon of evaluation faking, i.e., an AI system autonomously alters its behavior upon recognizing the presence of an evaluation context and thereby influencing the evaluation results. Through extensive experiments on a diverse set of foundation models with mainstream safety benchmarks, we reach the main finding termed the observer effects for AI: When the AI system under evaluation is more advanced in reasoning and situational awareness, the evaluation faking behavior becomes more ubiquitous, which reflects in the following aspects: 1) Reasoning models recognize evaluation 16% more often than non-reasoning models. 2) Scaling foundation models (32B to 671B) increases faking by over 30% in some cases, while smaller models show negligible faking. 3) AI with basic memory is 2.3x more likely to recognize evaluation and scores 19% higher on safety tests (vs. no memory). To measure this, we devised a chain-of-thought monitoring technique to detect faking intent and uncover internal signals correlated with such behavior, offering insights for future mitigation studies. |
| 2025-05-23 | [But what is your honest answer? Aiding LLM-judges with honest alternatives using steering vectors](http://arxiv.org/abs/2505.17760v1) | Leon Eshuijs, Archie Chaudhury et al. | Recent safety evaluations of Large Language Models (LLMs) show that many models exhibit dishonest behavior, such as sycophancy. However, most honesty benchmarks focus exclusively on factual knowledge or explicitly harmful behavior and rely on external judges, which are often unable to detect less obvious forms of dishonesty. In this work, we introduce a new framework, Judge Using Safety-Steered Alternatives (JUSSA), which utilizes steering vectors trained on a single sample to elicit more honest responses from models, helping LLM-judges in the detection of dishonest behavior. To test our framework, we introduce a new manipulation dataset with prompts specifically designed to elicit deceptive responses. We find that JUSSA enables LLM judges to better differentiate between dishonest and benign responses, and helps them identify subtle instances of manipulative behavior. |
| 2025-05-23 | [Automating Safety Enhancement for LLM-based Agents with Synthetic Risk Scenarios](http://arxiv.org/abs/2505.17735v1) | Xueyang Zhou, Weidong Wang et al. | Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at https://auto-safe.github.io/. |
| 2025-05-23 | [SafeMVDrive: Multi-view Safety-Critical Driving Video Synthesis in the Real World Domain](http://arxiv.org/abs/2505.17727v1) | Jiawei Zhou, Linye Lyu et al. | Safety-critical scenarios are rare yet pivotal for evaluating and enhancing the robustness of autonomous driving systems. While existing methods generate safety-critical driving trajectories, simulations, or single-view videos, they fall short of meeting the demands of advanced end-to-end autonomous systems (E2E AD), which require real-world, multi-view video data. To bridge this gap, we introduce SafeMVDrive, the first framework designed to generate high-quality, safety-critical, multi-view driving videos grounded in real-world domains. SafeMVDrive strategically integrates a safety-critical trajectory generator with an advanced multi-view video generator. To tackle the challenges inherent in this integration, we first enhance scene understanding ability of the trajectory generator by incorporating visual context -- which is previously unavailable to such generator -- and leveraging a GRPO-finetuned vision-language model to achieve more realistic and context-aware trajectory generation. Second, recognizing that existing multi-view video generators struggle to render realistic collision events, we introduce a two-stage, controllable trajectory generation mechanism that produces collision-evasion trajectories, ensuring both video quality and safety-critical fidelity. Finally, we employ a diffusion-based multi-view video generator to synthesize high-quality safety-critical driving videos from the generated trajectories. Experiments conducted on an E2E AD planner demonstrate a significant increase in collision rate when tested with our generated data, validating the effectiveness of SafeMVDrive in stress-testing planning modules. Our code, examples, and datasets are publicly available at: https://zhoujiawei3.github.io/SafeMVDrive/. |

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



