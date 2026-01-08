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
| 2026-01-07 | [Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions](http://arxiv.org/abs/2601.04170v1) | Abhishek Rath | Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies).   We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements.   We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research. |
| 2026-01-07 | [From Abstract Threats to Institutional Realities: A Comparative Semantic Network Analysis of AI Securitisation in the US, EU, and China](http://arxiv.org/abs/2601.04107v1) | Ruiyi Guo, Bodong Zhang | Artificial intelligence governance exhibits a striking paradox: while major jurisdictions converge rhetorically around concepts such as safety, risk, and accountability, their regulatory frameworks remain fundamentally divergent and mutually unintelligible. This paper argues that this fragmentation cannot be explained solely by geopolitical rivalry, institutional complexity, or instrument selection. Instead, it stems from how AI is constituted as an object of governance through distinct institutional logics. Integrating securitisation theory with the concept of the dispositif, we demonstrate that jurisdictions govern ontologically different objects under the same vocabulary. Using semantic network analysis of official policy texts from the European Union, the United States, and China (2023-2025), we trace how concepts like safety are embedded within divergent semantic architectures. Our findings reveal that the EU juridifies AI as a certifiable product through legal-bureaucratic logic; the US operationalises AI as an optimisable system through market-liberal logic; and China governs AI as socio-technical infrastructure through holistic state logic. We introduce the concept of structural incommensurability to describe this condition of ontological divergence masked by terminological convergence. This reframing challenges ethics-by-principles approaches to global AI governance, suggesting that coordination failures arise not from disagreement over values but from the absence of a shared reference object. |
| 2026-01-07 | [SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks](http://arxiv.org/abs/2601.04093v1) | Yu Yan, Sheng Sun et al. | Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems. |
| 2026-01-07 | [When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](http://arxiv.org/abs/2601.04043v1) | Xinyue Lou, Jinan Xu et al. | As Multimodal Large Language Models (MLLMs) become an indispensable assistant in human life, the unsafe content generated by MLLMs poses a danger to human behavior, perpetually overhanging human society like a sword of Damocles. To investigate and evaluate the safety impact of MLLMs responses on human behavior in daily life, we introduce SaLAD, a multimodal safety benchmark which contains 2,013 real-world image-text samples across 10 common categories, with a balanced design covering both unsafe scenarios and cases of oversensitivity. It emphasizes realistic risk exposure, authentic visual inputs, and fine-grained cross-modal reasoning, ensuring that safety risks cannot be inferred from text alone. We further propose a safety-warning-based evaluation framework that encourages models to provide clear and informative safety warnings, rather than generic refusals. Results on 18 MLLMs demonstrate that the top-performing models achieve a safe response rate of only 57.2% on unsafe queries. Moreover, even popular safety alignment methods limit effectiveness of the models in our scenario, revealing the vulnerabilities of current MLLMs in identifying dangerous behaviors in daily life. Our dataset is available at https://github.com/xinyuelou/SaLAD. |
| 2026-01-07 | [HoneyTrap: Deceiving Large Language Model Attackers to Honeypot Traps with Resilient Multi-Agent Defense](http://arxiv.org/abs/2601.04034v1) | Siyuan Li, Xi Lin et al. | Jailbreak attacks pose significant threats to large language models (LLMs), enabling attackers to bypass safeguards. However, existing reactive defense approaches struggle to keep up with the rapidly evolving multi-turn jailbreaks, where attackers continuously deepen their attacks to exploit vulnerabilities. To address this critical challenge, we propose HoneyTrap, a novel deceptive LLM defense framework leveraging collaborative defenders to counter jailbreak attacks. It integrates four defensive agents, Threat Interceptor, Misdirection Controller, Forensic Tracker, and System Harmonizer, each performing a specialized security role and collaborating to complete a deceptive defense. To ensure a comprehensive evaluation, we introduce MTJ-Pro, a challenging multi-turn progressive jailbreak dataset that combines seven advanced jailbreak strategies designed to gradually deepen attack strategies across multi-turn attacks. Besides, we present two novel metrics: Mislead Success Rate (MSR) and Attack Resource Consumption (ARC), which provide more nuanced assessments of deceptive defense beyond conventional measures. Experimental results on GPT-4, GPT-3.5-turbo, Gemini-1.5-pro, and LLaMa-3.1 demonstrate that HoneyTrap achieves an average reduction of 68.77% in attack success rates compared to state-of-the-art baselines. Notably, even in a dedicated adaptive attacker setting with intensified conditions, HoneyTrap remains resilient, leveraging deceptive engagement to prolong interactions, significantly increasing the time and computational costs required for successful exploitation. Unlike simple rejection, HoneyTrap strategically wastes attacker resources without impacting benign queries, improving MSR and ARC by 118.11% and 149.16%, respectively. |
| 2026-01-07 | [Can Dynamic Spectrum Sharing Protect Passive Radio Sciences?](http://arxiv.org/abs/2601.03966v1) | Gregory Hellbourg | Dynamic Spectrum Sharing (DSS) is increasingly promoted as a key element of modern spectrum policy, driven by the rising demand from commercial wireless systems and advances in spectrum access technologies. Passive radio sciences, including radio astronomy, Earth remote sensing, and meteorology, operate under fundamentally different constraints. They rely on exceptionally low interference spectrum and are highly vulnerable to even brief radio frequency interference. We examine whether DSS can benefit passive services or whether it introduces new failure modes and enforcement challenges. We propose just-in-time quiet zones (JITQZ) as a mechanism for protecting high value observations and assess hybrid frameworks that preserve static protection for core passive bands while allowing constrained dynamic access in adjacent frequencies. We analyze the roles of propagation uncertainty, electromagnetic compatibility constraints, and limited spectrum awareness. Using a game theoretic framework, we show why non-cooperative sharing fails, identify conditions for sustained cooperation, and examine incentive mechanisms including pseudonymetry-enabled attribution that promote compliance. We conclude that DSS can support passive radio sciences only as a high-reliability, safety-critical system. Static allocations remain essential, and dynamic access is viable only with conservative safeguards and enforceable accountability. |
| 2026-01-07 | [Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](http://arxiv.org/abs/2601.03926v1) | Haeun Jang, Hwan Chang et al. | The deployment of Large Vision-Language Models (LVLMs) for real-world document question answering is often constrained by dynamic, user-defined policies that dictate information disclosure based on context. While ensuring adherence to these explicit constraints is critical, existing safety research primarily focuses on implicit social norms or text-only settings, overlooking the complexities of multimodal documents. In this paper, we introduce Doc-PP (Document Policy Preservation Benchmark), a novel benchmark constructed from real-world reports requiring reasoning across heterogeneous visual and textual elements under strict non-disclosure policies. Our evaluation highlights a systemic Reasoning-Induced Safety Gap: models frequently leak sensitive information when answers must be inferred through complex synthesis or aggregated across modalities, effectively circumventing existing safety constraints. Furthermore, we identify that providing extracted text improves perception but inadvertently facilitates leakage. To address these vulnerabilities, we propose DVA (Decompose-Verify-Aggregation), a structural inference framework that decouples reasoning from policy verification. Experimental results demonstrate that DVA significantly outperforms standard prompting defenses, offering a robust baseline for policy-compliant document understanding |
| 2026-01-07 | [Towards Safe Autonomous Driving: A Real-Time Motion Planning Algorithm on Embedded Hardware](http://arxiv.org/abs/2601.03904v1) | Korbinian Moller, Glenn Johannes Tungka et al. | Ensuring the functional safety of Autonomous Vehicles (AVs) requires motion planning modules that not only operate within strict real-time constraints but also maintain controllability in case of system faults. Existing safeguarding concepts, such as Online Verification (OV), provide safety layers that detect infeasible planning outputs. However, they lack an active mechanism to ensure safe operation in the event that the main planner fails. This paper presents a first step toward an active safety extension for fail-operational Autonomous Driving (AD). We deploy a lightweight sampling-based trajectory planner on an automotive-grade, embedded platform running a Real-Time Operating System (RTOS). The planner continuously computes trajectories under constrained computational resources, forming the foundation for future emergency planning architectures. Experimental results demonstrate deterministic timing behavior with bounded latency and minimal jitter, validating the feasibility of trajectory planning on safety-certifiable hardware. The study highlights both the potential and the remaining challenges of integrating active fallback mechanisms as an integral part of next-generation safeguarding frameworks. The code is available at: https://github.com/TUM-AVS/real-time-motion-planning |
| 2026-01-07 | [What Matters For Safety Alignment?](http://arxiv.org/abs/2601.03868v1) | Xing Li, Hui-Ling Zhen et al. | This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models. |
| 2026-01-07 | [Majorum: Ebb-and-Flow Consensus with Dynamic Quorums](http://arxiv.org/abs/2601.03862v1) | Francesco D'Amato, Roberto Saltini et al. | Dynamic availability is the ability of a consensus protocol to remain live despite honest participants going offline and later rejoining. A well-known limitation is that dynamically available protocols, on their own, cannot provide strong safety guarantees during network partitions or extended asynchrony. Ebb-and-flow protocols [SP21] address this by combining a dynamically available protocol with a partially synchronous finality protocol that irrevocably finalizes a prefix.   We present Majorum, an ebb-and-flow construction whose dynamically available component builds on a quorum-based protocol (TOB-SVD). Under optimistic conditions, Majorum finalizes blocks in as few as three slots while requiring only a single voting phase per slot. In particular, when conditions remain favourable, each slot finalizes the next block extending the previously finalized one. |
| 2026-01-07 | [PartisanLens: A Multilingual Dataset of Hyperpartisan and Conspiratorial Immigration Narratives in European Media](http://arxiv.org/abs/2601.03860v1) | Michele Joshua Maggini, Paloma Piot et al. | Detecting hyperpartisan narratives and Population Replacement Conspiracy Theories (PRCT) is essential to addressing the spread of misinformation. These complex narratives pose a significant threat, as hyperpartisanship drives political polarisation and institutional distrust, while PRCTs directly motivate real-world extremist violence, making their identification critical for social cohesion and public safety. However, existing resources are scarce, predominantly English-centric, and often analyse hyperpartisanship, stance, and rhetorical bias in isolation rather than as interrelated aspects of political discourse. To bridge this gap, we introduce \textsc{PartisanLens}, the first multilingual dataset of \num{1617} hyperpartisan news headlines in Spanish, Italian, and Portuguese, annotated in multiple political discourse aspects. We first evaluate the classification performance of widely used Large Language Models (LLMs) on this dataset, establishing robust baselines for the classification of hyperpartisan and PRCT narratives. In addition, we assess the viability of using LLMs as automatic annotators for this task, analysing their ability to approximate human annotation. Results highlight both their potential and current limitations. Next, moving beyond standard judgments, we explore whether LLMs can emulate human annotation patterns by conditioning them on socio-economic and ideological profiles that simulate annotator perspectives. At last, we provide our resources and evaluation, \textsc{PartisanLens} supports future research on detecting partisan and conspiratorial narratives in European contexts. |
| 2026-01-07 | [On Zeno-like Behaviors in the Event Calculus with Goal-directed Answer Set Programming](http://arxiv.org/abs/2601.03852v1) | Ondřej Vašíček, Joaquin Arias et al. | It has been argued that Event Calculus (EC) is suitable for modeling high-level specifications of safety-critical cyber-physical systems. The primary advantage lies in the rather small semantic gap between EC models and requirements expressed in a semi-formal natural language. Moreover, its use of continuous time and variables avoids imprecision that stems from discretization. In the past, we have shown that a goal-directed ASP system can be used for implementing these EC models. However, precise representation of time as an infinitesimally divisible continuous quantity leads to Zeno-like behaviors and to non-termination in such a system. In this work, we model a number of well-known example problems from the literature to systematically study various natural EC modeling patterns that yield these Zeno-like behaviors, and propose ways to deal with them. Moreover, we also propose a technique to automatically detect all such cases. |
| 2026-01-07 | [Formally Explaining Decision Tree Models with Answer Set Programming](http://arxiv.org/abs/2601.03845v1) | Akihiro Takemura, Masayuki Otani et al. | Decision tree models, including random forests and gradient-boosted decision trees, are widely used in machine learning due to their high predictive performance.  However, their complex structures often make them difficult to interpret, especially in safety-critical applications where model decisions require formal justification.  Recent work has demonstrated that logical and abductive explanations can be derived through automated reasoning techniques.  In this paper, we propose a method for generating various types of explanations, namely, sufficient, contrastive, majority, and tree-specific explanations, using Answer Set Programming (ASP).  Compared to SAT-based approaches, our ASP-based method offers greater flexibility in encoding user preferences and supports enumeration of all possible explanations.  We empirically evaluate the approach on a diverse set of datasets and demonstrate its effectiveness and limitations compared to existing methods. |
| 2026-01-07 | [Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning](http://arxiv.org/abs/2601.03823v1) | Fei Wu, Zhenrong Zhang et al. | Reinforcement Learning with Verifiable Rewards (RLVR) elicits long chain-of-thought reasoning in large language models (LLMs), but outcome-based rewards lead to coarse-grained advantage estimation. While existing approaches improve RLVR via token-level entropy or sequence-level length control, they lack a semantically grounded, step-level measure of reasoning progress. As a result, LLMs fail to distinguish necessary deduction from redundant verification: they may continue checking after reaching a correct solution and, in extreme cases, overturn a correct trajectory into an incorrect final answer. To remedy the lack of process supervision, we introduce a training-free probing mechanism that extracts intermediate confidence and correctness and combines them into a Step Potential signal that explicitly estimates the reasoning state at each step. Building on this signal, we propose Step Potential Advantage Estimation (SPAE), a fine-grained credit assignment method that amplifies potential gains, penalizes potential drops, and applies penalty after potential saturates to encourage timely termination. Experiments across multiple benchmarks show SPAE consistently improves accuracy while substantially reducing response length, outperforming strong RL baselines and recent efficient reasoning and token-level advantage estimation methods. The code is available at https://github.com/cii030/SPAE-RL. |
| 2026-01-07 | [HearSay Benchmark: Do Audio LLMs Leak What They Hear?](http://arxiv.org/abs/2601.03783v1) | Jin Wang, Liang Lin et al. | While Audio Large Language Models (ALLMs) have achieved remarkable progress in understanding and generation, their potential privacy implications remain largely unexplored. This paper takes the first step to investigate whether ALLMs inadvertently leak user privacy solely through acoustic voiceprints and introduces $\textit{HearSay}$, a comprehensive benchmark constructed from over 22,000 real-world audio clips. To ensure data quality, the benchmark is meticulously curated through a rigorous pipeline involving automated profiling and human verification, guaranteeing that all privacy labels are grounded in factual records. Extensive experiments on $\textit{HearSay}$ yield three critical findings: $\textbf{Significant Privacy Leakage}$: ALLMs inherently extract private attributes from voiceprints, reaching 92.89% accuracy on gender and effectively profiling social attributes. $\textbf{Insufficient Safety Mechanisms}$: Alarmingly, existing safeguards are severely inadequate; most models fail to refuse privacy-intruding requests, exhibiting near-zero refusal rates for physiological traits. $\textbf{Reasoning Amplifies Risk}$: Chain-of-Thought (CoT) reasoning exacerbates privacy risks in capable models by uncovering deeper acoustic correlations. These findings expose critical vulnerabilities in ALLMs, underscoring the urgent need for targeted privacy alignment. The codes and dataset are available at https://github.com/JinWang79/HearSay_Benchmark |
| 2026-01-07 | [ETR: Outcome-Guided Elastic Trust Regions for Policy Optimization](http://arxiv.org/abs/2601.03723v1) | Shijie Zhang, Kevin Zhang et al. | Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as an important paradigm for unlocking reasoning capabilities in large language models, exemplified by the success of OpenAI o1 and DeepSeek-R1. Currently, Group Relative Policy Optimization (GRPO) stands as the dominant algorithm in this domain due to its stable training and critic-free efficiency. However, we argue that GRPO suffers from a structural limitation: it imposes a uniform, static trust region constraint across all samples. This design implicitly assumes signal homogeneity, a premise misaligned with the heterogeneous nature of outcome-driven learning, where advantage magnitudes and variances fluctuate significantly. Consequently, static constraints fail to fully exploit high-quality signals while insufficiently suppressing noise, often precipitating rapid entropy collapse. To address this, we propose \textbf{E}lastic \textbf{T}rust \textbf{R}egions (\textbf{ETR}), a dynamic mechanism that aligns optimization constraints with signal quality. ETR constructs a signal-aware landscape through dual-level elasticity: at the micro level, it scales clipping boundaries based on advantage magnitude to accelerate learning from high-confidence paths; at the macro level, it leverages group variance to implicitly allocate larger update budgets to tasks in the optimal learning zone. Extensive experiments on AIME and MATH benchmarks demonstrate that ETR consistently outperforms GRPO, achieving superior accuracy while effectively mitigating policy entropy degradation to ensure sustained exploration. |
| 2026-01-07 | [RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models](http://arxiv.org/abs/2601.03699v1) | Quy-Anh Dang, Chris Ngo et al. | As large language models (LLMs) become integral to safety-critical applications, ensuring their robustness against adversarial prompts is paramount. However, existing red teaming datasets suffer from inconsistent risk categorizations, limited domain coverage, and outdated evaluations, hindering systematic vulnerability assessments. To address these challenges, we introduce RedBench, a universal dataset aggregating 37 benchmark datasets from leading conferences and repositories, comprising 29,362 samples across attack and refusal prompts. RedBench employs a standardized taxonomy with 22 risk categories and 19 domains, enabling consistent and comprehensive evaluations of LLM vulnerabilities. We provide a detailed analysis of existing datasets, establish baselines for modern LLMs, and open-source the dataset and evaluation code. Our contributions facilitate robust comparisons, foster future research, and promote the development of secure and reliable LLMs for real-world deployment. Code: https://github.com/knoveleng/redeval |
| 2026-01-07 | [How Does the Thinking Step Influence Model Safety? An Entropy-based Safety Reminder for LRMs](http://arxiv.org/abs/2601.03662v1) | Su-Hyeon Kim, Hyundong Jin et al. | Large Reasoning Models (LRMs) achieve remarkable success through explicit thinking steps, yet the thinking steps introduce a novel risk by potentially amplifying unsafe behaviors. Despite this vulnerability, conventional defense mechanisms remain ineffective as they overlook the unique reasoning dynamics of LRMs. In this work, we find that the emergence of safe-reminding phrases within thinking steps plays a pivotal role in ensuring LRM safety. Motivated by this finding, we propose SafeRemind, a decoding-time defense method that dynamically injects safe-reminding phrases into thinking steps. By leveraging entropy triggers to intervene at decision-locking points, SafeRemind redirects potentially harmful trajectories toward safer outcomes without requiring any parameter updates. Extensive evaluations across five LRMs and six benchmarks demonstrate that SafeRemind substantially enhances safety, achieving improvements of up to 45.5%p while preserving core reasoning utility. |
| 2026-01-07 | [SyncThink: A Training-Free Strategy to Align Inference Termination with Reasoning Saturation](http://arxiv.org/abs/2601.03649v1) | Gengyang Li, Wang Cai et al. | Chain-of-Thought (CoT) prompting improves reasoning but often produces long and redundant traces that substantially increase inference cost. We present SyncThink, a training-free and plug-and-play decoding method that reduces CoT overhead without modifying model weights. We find that answer tokens attend weakly to early reasoning and instead focus on the special token "/think", indicating an information bottleneck. Building on this observation, SyncThink monitors the model's own reasoning-transition signal and terminates reasoning. Experiments on GSM8K, MMLU, GPQA, and BBH across three DeepSeek-R1 distilled models show that SyncThink achieves 62.00 percent average Top-1 accuracy using 656 generated tokens and 28.68 s latency, compared to 61.22 percent, 2141 tokens, and 92.01 s for full CoT decoding. On long-horizon tasks such as GPQA, SyncThink can further yield up to +8.1 absolute accuracy by preventing over-thinking. |
| 2026-01-07 | [ALERT: Zero-shot LLM Jailbreak Detection via Internal Discrepancy Amplification](http://arxiv.org/abs/2601.03600v1) | Xiao Lin, Philip Li et al. | Despite rich safety alignment strategies, large language models (LLMs) remain highly susceptible to jailbreak attacks, which compromise safety guardrails and pose serious security risks. Existing detection methods mainly detect jailbreak status relying on jailbreak templates present in the training data. However, few studies address the more realistic and challenging zero-shot jailbreak detection setting, where no jailbreak templates are available during training. This setting better reflects real-world scenarios where new attacks continually emerge and evolve. To address this challenge, we propose a layer-wise, module-wise, and token-wise amplification framework that progressively magnifies internal feature discrepancies between benign and jailbreak prompts. We uncover safety-relevant layers, identify specific modules that inherently encode zero-shot discriminative signals, and localize informative safety tokens. Building upon these insights, we introduce ALERT (Amplification-based Jailbreak Detector), an efficient and effective zero-shot jailbreak detector that introduces two independent yet complementary classifiers on amplified representations. Extensive experiments on three safety benchmarks demonstrate that ALERT achieves consistently strong zero-shot detection performance. Specifically, (i) across all datasets and attack strategies, ALERT reliably ranks among the top two methods, and (ii) it outperforms the second-best baseline by at least 10% in average Accuracy and F1-score, and sometimes by up to 40%. |

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



