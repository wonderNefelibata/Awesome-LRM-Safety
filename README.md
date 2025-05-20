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
| 2025-05-19 | [AdaptThink: Reasoning Models Can Learn When to Think](http://arxiv.org/abs/2505.13417v1) | Jiajie Zhang, Nianyi Lin et al. | Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at https://github.com/THU-KEG/AdaptThink. |
| 2025-05-19 | [GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection](http://arxiv.org/abs/2505.13312v1) | Zhijie Deng, Chris Yuhao Liu et al. | Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility. |
| 2025-05-19 | [I'll believe it when I see it: Images increase misinformation sharing in Vision-Language Models](http://arxiv.org/abs/2505.13302v1) | Alice Plebe, Timothy Douglas et al. | Large language models are increasingly integrated into news recommendation systems, raising concerns about their role in spreading misinformation. In humans, visual content is known to boost credibility and shareability of information, yet its effect on vision-language models (VLMs) remains unclear. We present the first study examining how images influence VLMs' propensity to reshare news content, whether this effect varies across model families, and how persona conditioning and content attributes modulate this behavior. To support this analysis, we introduce two methodological contributions: a jailbreaking-inspired prompting strategy that elicits resharing decisions from VLMs while simulating users with antisocial traits and political alignments; and a multimodal dataset of fact-checked political news from PolitiFact, paired with corresponding images and ground-truth veracity labels. Experiments across model families reveal that image presence increases resharing rates by 4.8% for true news and 15.0% for false news. Persona conditioning further modulates this effect: Dark Triad traits amplify resharing of false news, whereas Republican-aligned profiles exhibit reduced veracity sensitivity. Of all the tested models, only Claude-3-Haiku demonstrates robustness to visual misinformation. These findings highlight emerging risks in multimodal model behavior and motivate the development of tailored evaluation frameworks and mitigation strategies for personalized AI systems. Code and dataset are available at: https://github.com/3lis/misinfo_vlm |
| 2025-05-19 | [Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability](http://arxiv.org/abs/2505.13258v1) | Jingyi Ren, Yekun Xu et al. | Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive domains. However, although RAG achieved successes across distinct domains, there are still some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on developing more powerful RAG retrievers, but how to enhance the generator's (LLM's) ability to utilize the retrieved information for reasoning and generation? 2) Transparency. Most RAG methods ignore which retrieved content actually contributes to the reasoning process, resulting in a lack of interpretability and visibility. To address this, we propose ARENA (Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator framework trained via reinforcement learning (RL) with our proposed rewards. Based on the structured generation and adaptive reward calculation, our RL-based training enables the model to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various RAG baselines demonstrate that our model achieves 10-30% improvements on all multi-hop QA datasets, which is comparable with the SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show that ARENA has strong flexibility to be adopted on new datasets without extra training. Our models and codes are publicly released. |
| 2025-05-19 | [Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities](http://arxiv.org/abs/2505.13195v1) | Lili Zhang, Haomiaomiao Wang et al. | As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research. |
| 2025-05-19 | [Interpretable Robotic Friction Learning via Symbolic Regression](http://arxiv.org/abs/2505.13186v1) | Philipp Scholl, Alexander Dietrich et al. | Accurately modeling the friction torque in robotic joints has long been challenging due to the request for a robust mathematical description. Traditional model-based approaches are often labor-intensive, requiring extensive experiments and expert knowledge, and they are difficult to adapt to new scenarios and dependencies. On the other hand, data-driven methods based on neural networks are easier to implement but often lack robustness, interpretability, and trustworthiness--key considerations for robotic hardware and safety-critical applications such as human-robot interaction. To address the limitations of both approaches, we propose the use of symbolic regression (SR) to estimate the friction torque. SR generates interpretable symbolic formulas similar to those produced by model-based methods while being flexible to accommodate various dynamic effects and dependencies. In this work, we apply SR algorithms to approximate the friction torque using collected data from a KUKA LWR-IV+ robot. Our results show that SR not only yields formulas with comparable complexity to model-based approaches but also achieves higher accuracy. Moreover, SR-derived formulas can be seamlessly extended to include load dependencies and other dynamic factors. |
| 2025-05-19 | [Information Science Principles of Machine Learning: A Causal Chain Meta-Framework Based on Formalized Information Mapping](http://arxiv.org/abs/2505.13182v1) | Jianfeng Xu | [Objective] This study focuses on addressing the current lack of a unified formal theoretical framework in machine learning, as well as the deficiencies in interpretability and ethical safety assurance. [Methods] A formal information model is first constructed, utilizing sets of well-formed formulas to explicitly define the ontological states and carrier mappings of typical components in machine learning. Learnable and processable predicates, along with learning and processing functions, are introduced to analyze the logical deduction and constraint rules of the causal chains within models. [Results] A meta-framework for machine learning theory (MLT-MF) is established. Based on this framework, universal definitions for model interpretability and ethical safety are proposed. Furthermore, three key theorems are proved: the equivalence of model interpretability and information recoverability, the assurance of ethical safety, and the estimation of generalization error. [Limitations] The current framework assumes ideal conditions with noiseless information-enabling mappings and primarily targets model learning and processing logic in static scenarios. It does not yet address information fusion and conflict resolution across ontological spaces in multimodal or multi-agent systems. [Conclusions] This work overcomes the limitations of fragmented research and provides a unified theoretical foundation for systematically addressing the critical challenges currently faced in machine learning. |
| 2025-05-19 | [Constraint-Aware Diffusion Guidance for Robotics: Real-Time Obstacle Avoidance for Autonomous Racing](http://arxiv.org/abs/2505.13131v1) | Hao Ma, Sabrina Bodmer et al. | Diffusion models hold great potential in robotics due to their ability to capture complex, high-dimensional data distributions. However, their lack of constraint-awareness limits their deployment in safety-critical applications. We propose Constraint-Aware Diffusion Guidance (CoDiG), a data-efficient and general-purpose framework that integrates barrier functions into the denoising process, guiding diffusion sampling toward constraint-satisfying outputs. CoDiG enables constraint satisfaction even with limited training data and generalizes across tasks. We evaluate our framework in the challenging setting of miniature autonomous racing, where real-time obstacle avoidance is essential. Real-world experiments show that CoDiG generates safe outputs efficiently under dynamic conditions, highlighting its potential for broader robotic applications. A demonstration video is available at https://youtu.be/KNYsTdtdxOU. |
| 2025-05-19 | [Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset](http://arxiv.org/abs/2505.13028v1) | Sayon Palit, Daniel Woods | Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics. |
| 2025-05-19 | [Evaluating the Performance of RAG Methods for Conversational AI in the Airport Domain](http://arxiv.org/abs/2505.13006v1) | Yuyang Li, Philip J. M. Kerbusch et al. | Airports from the top 20 in terms of annual passengers are highly dynamic environments with thousands of flights daily, and they aim to increase the degree of automation. To contribute to this, we implemented a Conversational AI system that enables staff in an airport to communicate with flight information systems. This system not only answers standard airport queries but also resolves airport terminology, jargon, abbreviations, and dynamic questions involving reasoning. In this paper, we built three different Retrieval-Augmented Generation (RAG) methods, including traditional RAG, SQL RAG, and Knowledge Graph-based RAG (Graph RAG). Experiments showed that traditional RAG achieved 84.84% accuracy using BM25 + GPT-4 but occasionally produced hallucinations, which is risky to airport safety. In contrast, SQL RAG and Graph RAG achieved 80.85% and 91.49% accuracy respectively, with significantly fewer hallucinations. Moreover, Graph RAG was especially effective for questions that involved reasoning. Based on our observations, we thus recommend SQL RAG and Graph RAG are better for airport environments, due to fewer hallucinations and the ability to handle dynamic questions. |
| 2025-05-19 | [EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code](http://arxiv.org/abs/2505.13004v1) | Yuhao Qing, Boyu Zhu et al. | Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python code is significantly more efficient than its Java code. These results highlight the critical need for research into LLM optimization techniques to improve code efficiency across diverse languages. The dataset and evaluation infrastructure are submitted and available at https://github.com/EffiBench/EffiBench-X.git and https://huggingface.co/datasets/EffiBench/effibench-x. |
| 2025-05-19 | [ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning](http://arxiv.org/abs/2505.12996v1) | Jiaan Wang, Fandong Meng et al. | In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance. |
| 2025-05-19 | [LoD: Loss-difference OOD Detection by Intentionally Label-Noisifying Unlabeled Wild Data](http://arxiv.org/abs/2505.12952v1) | Chuanxing Geng, Qifei Li et al. | Using unlabeled wild data containing both in-distribution (ID) and out-of-distribution (OOD) data to improve the safety and reliability of models has recently received increasing attention. Existing methods either design customized losses for labeled ID and unlabeled wild data then perform joint optimization, or first filter out OOD data from the latter then learn an OOD detector. While achieving varying degrees of success, two potential issues remain: (i) Labeled ID data typically dominates the learning of models, inevitably making models tend to fit OOD data as IDs; (ii) The selection of thresholds for identifying OOD data in unlabeled wild data usually faces dilemma due to the unavailability of pure OOD samples. To address these issues, we propose a novel loss-difference OOD detection framework (LoD) by \textit{intentionally label-noisifying} unlabeled wild data. Such operations not only enable labeled ID data and OOD data in unlabeled wild data to jointly dominate the models' learning but also ensure the distinguishability of the losses between ID and OOD samples in unlabeled wild data, allowing the classic clustering technique (e.g., K-means) to filter these OOD samples without requiring thresholds any longer. We also provide theoretical foundation for LoD's viability, and extensive experiments verify its superiority. |
| 2025-05-19 | [ORQA: A Benchmark and Foundation Model for Holistic Operating Room Modeling](http://arxiv.org/abs/2505.12890v1) | Ege Özsoy, Chantal Pellegrini et al. | The real-world complexity of surgeries necessitates surgeons to have deep and holistic comprehension to ensure precision, safety, and effective interventions. Computational systems are required to have a similar level of comprehension within the operating room. Prior works, limited to single-task efforts like phase recognition or scene graph generation, lack scope and generalizability. In this work, we introduce ORQA, a novel OR question answering benchmark and foundational multimodal model to advance OR intelligence. By unifying all four public OR datasets into a comprehensive benchmark, we enable our approach to concurrently address a diverse range of OR challenges. The proposed multimodal large language model fuses diverse OR signals such as visual, auditory, and structured data, for a holistic modeling of the OR. Finally, we propose a novel, progressive knowledge distillation paradigm, to generate a family of models optimized for different speed and memory requirements. We show the strong performance of ORQA on our proposed benchmark, and its zero-shot generalization, paving the way for scalable, unified OR modeling and significantly advancing multimodal surgical intelligence. We will release our code and data upon acceptance. |
| 2025-05-19 | [GAP: Graph-Assisted Prompts for Dialogue-based Medication Recommendation](http://arxiv.org/abs/2505.12888v1) | Jialun Zhong, Yanzeng Li et al. | Medication recommendations have become an important task in the healthcare domain, especially in measuring the accuracy and safety of medical dialogue systems (MDS). Different from the recommendation task based on electronic health records (EHRs), dialogue-based medication recommendations require research on the interaction details between patients and doctors, which is crucial but may not exist in EHRs. Recent advancements in large language models (LLM) have extended the medical dialogue domain. These LLMs can interpret patients' intent and provide medical suggestions including medication recommendations, but some challenges are still worth attention. During a multi-turn dialogue, LLMs may ignore the fine-grained medical information or connections across the dialogue turns, which is vital for providing accurate suggestions. Besides, LLMs may generate non-factual responses when there is a lack of domain-specific knowledge, which is more risky in the medical domain. To address these challenges, we propose a \textbf{G}raph-\textbf{A}ssisted \textbf{P}rompts (\textbf{GAP}) framework for dialogue-based medication recommendation. It extracts medical concepts and corresponding states from dialogue to construct an explicitly patient-centric graph, which can describe the neglected but important information. Further, combined with external medical knowledge graphs, GAP can generate abundant queries and prompts, thus retrieving information from multiple sources to reduce the non-factual responses. We evaluate GAP on a dialogue-based medication recommendation dataset and further explore its potential in a more difficult scenario, dynamically diagnostic interviewing. Extensive experiments demonstrate its competitive performance when compared with strong baselines. |
| 2025-05-19 | [Practical Equivalence Testing and Its Application in Synthetic Pre-Crash Scenario Validation](http://arxiv.org/abs/2505.12827v1) | Jian Wu, Ulrich Sander et al. | The use of representative pre-crash scenarios is critical for assessing the safety impact of driving automation systems through simulation. However, a gap remains in the robust evaluation of the similarity between synthetic and real-world pre-crash scenarios and their crash characteristics. Without proper validation, it cannot be ensured that the synthetic test scenarios adequately represent real-world driving behaviors and crash characteristics. One reason for this validation gap is the lack of focus on methods to confirm that the synthetic test scenarios are practically equivalent to real-world ones, given the assessment scope. Traditional statistical methods, like significance testing, focus on detecting differences rather than establishing equivalence; since failure to detect a difference does not imply equivalence, they are of limited applicability for validating synthetic pre-crash scenarios and crash characteristics. This study addresses this gap by proposing an equivalence testing method based on the Bayesian Region of Practical Equivalence (ROPE) framework. This method is designed to assess the practical equivalence of scenario characteristics that are most relevant for the intended assessment, making it particularly appropriate for the domain of virtual safety assessments. We first review existing equivalence testing methods. Then we propose and demonstrate the Bayesian ROPE-based method by testing the equivalence of two rear-end pre-crash datasets. Our approach focuses on the most relevant scenario characteristics. Our analysis provides insights into the practicalities and effectiveness of equivalence testing in synthetic test scenario validation and demonstrates the importance of testing for improving the credibility of synthetic data for automated vehicle safety assessment, as well as the credibility of subsequent safety impact assessments. |
| 2025-05-19 | [Language Models That Walk the Talk: A Framework for Formal Fairness Certificates](http://arxiv.org/abs/2505.12767v1) | Danqing Chen, Tobias Ladner et al. | As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation. |
| 2025-05-19 | [Bullying the Machine: How Personas Increase LLM Vulnerability](http://arxiv.org/abs/2505.12692v1) | Ziwei Xu, Udit Sanghi et al. | Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies. |
| 2025-05-19 | [CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models](http://arxiv.org/abs/2505.12677v1) | Shristi Das Biswas, Arani Roy et al. | As Text-to-Image models continue to evolve, so does the risk of generating unsafe, copyrighted, or privacy-violating content. Existing safety interventions - ranging from training data curation and model fine-tuning to inference-time filtering and guidance - often suffer from incomplete concept removal, susceptibility to jail-breaking, computational inefficiency, or collateral damage to unrelated capabilities. In this paper, we introduce CURE, a training-free concept unlearning framework that operates directly in the weight space of pre-trained diffusion models, enabling fast, interpretable, and highly specific suppression of undesired concepts. At the core of our method is the Spectral Eraser, a closed-form, orthogonal projection module that identifies discriminative subspaces using Singular Value Decomposition over token embeddings associated with the concepts to forget and retain. Intuitively, the Spectral Eraser identifies and isolates features unique to the undesired concept while preserving safe attributes. This operator is then applied in a single step update to yield an edited model in which the target concept is effectively unlearned - without retraining, supervision, or iterative optimization. To balance the trade-off between filtering toxicity and preserving unrelated concepts, we further introduce an Expansion Mechanism for spectral regularization which selectively modulates singular vectors based on their relative significance to control the strength of forgetting. All the processes above are in closed-form, guaranteeing extremely efficient erasure in only $2$ seconds. Benchmarking against prior approaches, CURE achieves a more efficient and thorough removal for targeted artistic styles, objects, identities, or explicit content, with minor damage to original generation ability and demonstrates enhanced robustness against red-teaming. |
| 2025-05-19 | [TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning](http://arxiv.org/abs/2505.12670v1) | Lihong Chen, Hossein Hassani et al. | Vision-Language Models (VLMs) have shown remarkable potential in advancing autonomous driving by leveraging multi-modal fusion in order to enhance scene perception, reasoning, and decision-making. Despite their potential, existing models suffer from computational overhead and inefficient integration of multi-view sensor data that make them impractical for real-time deployment in safety-critical autonomous driving applications. To address these shortcomings, this paper is devoted to designing a lightweight VLM called TS-VLM, which incorporates a novel Text-Guided SoftSort Pooling (TGSSP) module. By resorting to semantics of the input queries, TGSSP ranks and fuses visual features from multiple views, enabling dynamic and query-aware multi-view aggregation without reliance on costly attention mechanisms. This design ensures the query-adaptive prioritization of semantically related views, which leads to improved contextual accuracy in multi-view reasoning for autonomous driving. Extensive evaluations on the DriveLM benchmark demonstrate that, on the one hand, TS-VLM outperforms state-of-the-art models with a BLEU-4 score of 56.82, METEOR of 41.91, ROUGE-L of 74.64, and CIDEr of 3.39. On the other hand, TS-VLM reduces computational cost by up to 90%, where the smallest version contains only 20.1 million parameters, making it more practical for real-time deployment in autonomous vehicles. |

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



