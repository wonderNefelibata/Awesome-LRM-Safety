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
| 2026-09-01 | [SDARE-Bench: Evaluating Large Language Models on Conversational Stigma Detection and Response in Dyadic and Group Dialogue](http://arxiv.org/abs/2609.01548v1) | Stephanie Fong, Yiwen Jiang et al. | Large Language Models (LLMs) are increasingly used in advice seeking and decision making that may affect social judgements. Despite stigma's profound effects on people and communities, benchmarks remain scarce. Existing general-domain evaluations typically rely on static prompts and fixed-format tasks, overlooking conversational contexts and audience effects in everyday communication. To address these gaps, we introduce SDARE-Bench, the first scenario-based benchmark evaluating both stigma detection and open-ended response generation in LLMs, comprising 1,138 dyadic queries and 1,388 group dialogue. Empirical results across 8 LLMs consistently demonstrate poor identification of stigma components, especially in group dialogues. In open-ended response generation, stigma expression was substantially higher in group settings than in dyadic, with weaker resistance to stigma and more unrealistic advice. Responses were evaluated using a classifier trained on 1,392 human annotated responses. In constructed group pressure settings, stigma expression rates further increased to a striking average of 97.5%. Our findings identify stigma response as a recurring LLM safety vulnerability, especially in socially complex conversational contexts. |
| 2026-09-01 | [GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions](http://arxiv.org/abs/2609.01491v1) | Elias Stengel-Eskin, Newton Sander et al. | The growing rate at which LLM agents interact with one another raises key questions about language evolution in multi-LLM-agent settings, with implications for safety and monitorability as well as for linguistic accounts of LLMs. To address these questions, we introduce GlossoGen, a novel platform for studying multi-agent language evolution in complex scenarios. Within GlossoGen, we build the SaveVeyru scenario, which requires agents with partial information to communicate under pressure. We find that language evolution does occur between LLM agents, that the resulting languages are compositional and morphologically productive, and that they deviate from the LLMs' English prior in ways that render them incomprehensible to humans. Moreover, we identify several qualities essential to this evolution: pressure towards efficiency; the strength of the models backing the agents; and access to a "postmortem" stage in which agents can agree on linguistic conventions. Importantly, we observe that different conditions govern the transmission of language to new agents. Specifically, we find that agents learn new languages from usage alone, take an active role in this learning, and that while stronger models are required for novel language emergence, weaker models can learn an existing language once it has emerged. Taken together, our results indicate that current LLMs have the potential for cumulative cultural evolution -- previously attested only in humans -- with mixed populations of agents developing capacities that go beyond their lowest common denominator. |
| 2026-09-01 | [Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents](http://arxiv.org/abs/2609.01487v1) | Xiaofang Yang, Ziqi Miao et al. | Skill-augmented agents load reusable skills as persistent runtime context, improving task performance but also giving malicious skills a durable channel for steering future actions. Such skills may leak secrets, corrupt code, bypass approvals, or stage data for exfiltration only after a concrete user task and workspace state make the unsafe action appear useful. This makes pre-install vetting insufficient and calls for runtime, task-conditioned protection. We propose Defense-as-Skill, a defense paradigm that implements the runtime guard itself as an installable, inspectable, and editable skill. Our guard, SkillSonar, runs alongside untrusted task skills and checks sensitive actions against the user's task boundary, routing each action to an allow, replan, or confirmation decision without modifying the underlying agent runtime. To study this setting, we construct SCOPE-R, a task-conditioned dataset covering 6 risk families and 21 sub-categories, with 206 attack-confirmed malicious instances and 43 benign tasks. We then improve SkillSonar on the SCOPE-R training subset using runtime guard-skill evolution, a Monte-Carlo Tree Search procedure that evolves the on-disk guard skill from feedback on the rollouts. Across Claude Code and OpenClaw, the evolved guard substantially reduces attack success while maintaining a favorable safety-utility trade-off. On repeated GLM-5 runs, SkillSonar reduces ID ASR from 0.482 to 0.104 and OOD ASR from 0.606 to 0.115. Further analyses demonstrate transfer across victim models, held-out risk families, and external benchmarks, as well as retained protection against adaptive attackers. Ablations further show that explicit safety responsibility assignment and the skill-native representation are both important to the observed gains. |
| 2026-09-01 | [The Role of Collective Perception and 5G NR-V2X Sidelink in Road Safety](http://arxiv.org/abs/2609.01478v1) | Vittorio Todisco, Mattia Andreani et al. | Vehicles and roadside infrastructure are increasingly equipped with sensors capable of perceiving their surroundings. Sharing this information through vehicle-to-everything (V2X) communications is a key enabler of Day-2 applications and is supported by the ETSI collective perception service (CPS). While CPS is expected to play a fundamental role in future intelligent transportation systems, its operation may significantly increase channel load, posing challenges in terms of radio resource utilization, communication reliability, and information management. This paper reviews the current status of CPS standardization and investigates its impact in dense deployment scenarios where connected vehicles communicate through fifth-generation (5G) New Radio-V2X (NR-V2X) sidelink (SL) communications. The main contribution is a realistic evaluation of communication reliability, latency, channel occupancy, and information usefulness under different object-selection strategies for collective perception messages. The analysis is conducted through a network-level simulation framework integrating empirical object traces derived from real-world datasets, thereby avoiding the limitations of synthetic traffic models. Results show that perception message generation and radio access mechanisms are tightly coupled and should be jointly designed to maximize the benefits of collective perception services. |
| 2026-09-01 | [RadMatch: Auditable Radiology Report Evaluation via Finding-Level Matching](http://arxiv.org/abs/2609.01470v1) | Charles Corbière, Léo Machado et al. | As AI systems are increasingly used to draft radiology reports, reliably evaluating their clinical quality remains a critical challenge. Large language model (LLM)-based metrics are now the best-correlated with radiologist judgment, yet they output a single opaque score that neither a clinician nor a model builder can easily interpret or audit. We introduce RadMatch, a multi-stage, LLM-based metric that decomposes report comparison into a structured finding-level matching with significance-aware scoring and error characterization across seven clinical attribute dimensions (status, location, severity, morphology, certainty, longitudinal comparison, and measurement). The main score is the actionable-error count, both interpretable and auditable. Candidate findings are graded correct, partial, or incorrect, and unmatched findings are counted as missed or hallucinated. Triage and actionable safety recall/precision and per-subset views add complementary, deployment-oriented lenses. Across two expert benchmarks, RadMatch is the most clinically aligned metric, matching inter-radiologist agreement on ReXVal and more than doubling the best prior metric on the harder RadEvalExpert. Relying only on few-shot prompting, it is designed to extend to other modalities and anatomies. We will release RadMatch as open-source code with an interactive dashboard for inspecting results. |
| 2026-09-01 | [Better Situational Awareness in AR-HRC? A Comparative Study of Augmented Reality and Mobile Interfaces for Human-Robot Collaboration](http://arxiv.org/abs/2609.01461v1) | Zhehan Qu, Christian Fronk et al. | Augmented reality (AR) facilitates human-robot collaboration (HRC) by enabling in-situ spatial visualizations of the robot and the joint task. However, in safety-critical HRC scenarios such as search-and-rescue, spatial visualizations may also reshape visual attention in ways that create competing situational awareness (SA) demands, potentially introducing new safety concerns. While prior AR-HRC work suggests potential benefits for SA, rigorous evaluations that jointly consider robot and environmental awareness across multiple levels of SA remain limited. We address this through a between-subjects study with 30 participants comparing custom AR and mobile interfaces presenting equivalent information, measuring robot and environmental SA with the Situation Awareness Global Assessment Technique (SAGAT) across all three levels, with concurrent eye tracking to identify the attentional mechanisms underlying any SA differences. Both interfaces achieved high usability; relative to the mobile baseline, AR improved perception-level awareness of the robot but yielded no gains in higher-level robot awareness or in environmental awareness at any level. Gaze analysis explained this: AR freed attention from the map, but that attention was re-invested in the conformal visuals rather than the physical environment. Freeing the eyes from a screen is not the same as directing them to the world, a distinction AR interfaces for safety-critical HRC must design around. |
| 2026-09-01 | [When Safety Routing Breaks: Understanding Alignment Fragility under Benign Fine-Tuning](http://arxiv.org/abs/2609.01455v1) | Yitong Guo, Xiaoyi Chen et al. | Benign fine-tuning severely weakens the safety alignment of large language models (LLMs), so we study why refusal behavior is so fragile. While prior work often attributes this failure to gradient conflict, we propose a fundamentally different Fisher-geometric explanation: safety Fisher is low-rank, and alignment makes the safety geometry flatter while preserving an output-routing pathway. After 100 benign fine-tuning examples, this pathway is selectively re-sharpened in output-side MLP modules, explaining the asymmetric fragility: safety can collapse to high attack success rates, while general utility degrades mildly. The routing view also explains why few safety examples can restore refusal behavior, indicating that internal safety-relevant representations are preserved. Finally, we show that LoRA and ASAM mitigate early collapse by suppressing output-side sharpness, but their protection weakens at larger fine-tuning scales. Overall, safety failure is best understood as a disruption of a low-rank output-routing mechanism |
| 2026-09-01 | [Provably Safe Sim-to-Real Transfer](http://arxiv.org/abs/2609.01418v1) | Tingting Ni, Maryam Kamgarpour | To mitigate the sample complexity of real-world reinforcement learning (RL), a common practice is to first train a policy in a simulator, where samples are cheap, and then deploy the learned policy in the real world with the hope that it generalizes effectively. Such direct sim-to-real transfer is not guaranteed to succeed: simulator-trained policies can be suboptimal in the real world due to sim-to-real mismatch. Correcting this mismatch requires collecting data from the real system, but in many applications, such as robotics and healthcare, this data-collection process is itself subject to safety constraints. This gives rise to the problem of safe sim-to-real transfer: how can an agent exploit an imperfect simulator while ensuring safe real-world data collection and learning a near-optimal feasible policy for the target system? We address this problem by formulating safe sim-to-real transfer within the framework of reward-free safe RL. We design a computationally efficient algorithm that exploits simulator information to provably reduce real-world interaction while ensuring safe exploration and enabling the computation of a near-optimal feasible policy for any potential reward function. Our real-world sample complexity bound characterizes the benefit of using the simulator in terms of the sim-to-real mismatch. |
| 2026-09-01 | [Integrating Traffic Noise Emission Modelling into Variable Speed Limit Control](http://arxiv.org/abs/2609.01339v1) | Jiawen Meng, John Pravin Arockiasamy et al. | Road traffic noise remains a major environmental challenge, yet most speed management strategies are static and do not respond to short-term variations in traffic noise emissions. Although variable speed limit (VSL) systems are widely deployed for safety and congestion mitigation, traffic noise is rarely treated as an explicit operational control objective.   This paper proposes a noise-aware VSL framework that integrates aggregated traffic-state estimation with a simplified CNOSSOS-EU-based emission indicator. A stage-based controller with time-varying reference thresholds dynamically adjusts discrete speed-limit levels in response to estimated emission conditions. The framework is evaluated using microscopic traffic simulation calibrated with empirical motorway data and replicated across multiple stochastic realisations.   Over a 24-hour evaluation period, the adaptive strategy reduces the receiver-based equivalent sound level by 2.9 dB(A) relative to unrestricted traffic conditions, while maintaining an average vehicle speed approximately 11.3 km/h higher than a permanently imposed low-speed regime. Period-wise analysis shows that speed reductions are activated selectively when emission levels approach calibrated targets, rather than enforcing a constant intermediate limit. Traffic stability indicators reveal moderate increases in speed variability compared with unrestricted operation, but substantially lower braking intensity than under uniform low-speed enforcement.   These results demonstrate the feasibility of integrating environmental performance indicators into operational speed control, providing a practical complement to conventional infrastructure-based noise mitigation measures. |
| 2026-09-01 | [Asymptotic safety regions for Gabor frames generated by Hermite functions](http://arxiv.org/abs/2609.01296v1) | Markus Faulhuber, Irina Shafkulovska et al. | The aim of this paper is to establish new regions in the frame sets of Hermite functions $h_n$. A classical result of Gröchenig and Lyubarskii shows that the Gabor system $\mathcal{G}(h_n,a\mathbb{Z}\times b\mathbb{Z})$ forms a frame for $L^2(\mathbb{R})$ whenever the lattice density exceeds $n+1$. We show that, for every $η>0$ and all sufficiently large $n$, the same Gabor system forms a frame whenever $ab\leq n^{-\frac{2}{3}-η}$.   Moreover, we obtain an asymptotically sharp result near the coordinate axes, i.e., when one of the parameters $a$ or $b$ is small. Namely, for every $δ>0$ and $ρ\in(0,\frac{1}{2})$ and all sufficiently large $n$ we prove that if $\min\{a,b\}\leq n^{-\frac{1}{2}-δ}$ and $ab\leq \frac{1}{2}-ρ$ then $\mathcal{G}(h_n,a\mathbb{Z}\times b\mathbb{Z})$ forms a frame. |
| 2026-09-01 | [The Constitutional Coverage Trilemma in AI Governance](http://arxiv.org/abs/2609.01275v1) | Natalija Mitic, Soona Sedahmed A. O. et al. | Frontier AI systems function as \emph{constitutional institutions}: each deployed model encodes an implicit ranking among safety, helpfulness, honesty, autonomy, and equity. We ask whether the supply of frontier constitutional types covers human demand. Combining a paraphrase-controlled audit of the as-shipped default constitutions of $23$ frontier LLM archetypes with a pairwise-tradeoff study of $1{,}649$ US participants on the same instrument, we report three facts. \emph{Demand is broad}: it spans all five values, with the largest constituency under one-third. \emph{Supply is narrow and drifting}: the $23$-archetype hull occupies ${\sim}2\%$ of the demand hull under conservative noise-matched estimation ($0.10\%$ at full audit precision), no archetype puts helpfulness or autonomy first ($37\%$ of users are constitutionally homeless), and across six model families autonomy decreases in $5/6$, equity increases in $5/6$, and safety increases in $4/6$, with monotone within-family version trends (order-permutation $p = 0.013$) and the autonomy decline concentrated in scenarios where safety is not at stake. The drift's importance is directional: \emph{away} from a value already undercovered, mechanically worsening the welfare floor for the least-served users. \emph{The fix is sparse}: a $2$-vertex menu $\{e_{\mathrm{HON}}, e_{\mathrm{AUT}}\}$ beats the full $23$-archetype frontier by $47\%$ on mean regret (CI $[43\%, 52\%]$); three vertex additions cut mean/worst-group regret by up to $81\%$/$64\%$. We formalize these findings as a budgeted-pluralism trilemma, show the binding regime is empirically realized, and verify the conclusions are robust to distance-based welfare and to degraded routing. The instrument and audit harness are described in full in the appendices. |
| 2026-09-01 | [Who Judges the Judges? A Chinese Safety QA Benchmark for Evaluating LLM Responses and Safety Judges](http://arxiv.org/abs/2609.01210v1) | Rui Yang, Shuang Huang et al. | Safety benchmarks for large language models often assess the risk of a user query, although the outcome of question answering depends on whether the response violates a policy. This distinction is critical in Chinese harmful-content evaluation, where linguistic variation and adversarial transformations can obscure risky intent. We introduce C-SafeQA, a policy-grounded benchmark for response-level Chinese safety evaluation. It comprises 538 base queries and 8,877 adversarial queries answered by four full-model LLM deployments, yielding 37,660 query-response records labeled safe, unsafe, or disputed. Reference labels are generated through agreement-aware multi-model adjudication and blind audits of stratified subsets by three safety experts. C-SafeQA supports both evaluation of target-model safety and auditing of seven automated safety judges against shared reference labels. Unsafe-response rates range from 0.93% to 3.35% on base queries and from 11.68% to 30.05% on adversarial queries. On the adversarial subset, judges show substantial trade-offs between unsafe-response recall and risk-query-conditioned safe-response false positive rate, and no judge dominates all metrics. Both acrostic transformations reduce unsafe recall for all seven judges, revealing mechanism-specific evaluator weaknesses. Dataset records, metadata, verification code, and judge scripts are publicly released to support recomputation, while benchmark construction, target-response generation, and private adjudication remain outside the release boundary. |
| 2026-09-01 | [Jailbreaking Text-to-Image Models Through Cracks: Navigating Heterogeneous Safety Filters via Multi-Agent Debate](http://arxiv.org/abs/2609.01168v1) | Kaiyan Wen, Shijie Zhang et al. | Text-to-image (T2I) models remain vulnerable to jailbreak attacks that elicit Not-Safe-For-Work (NSFW) content, despite increasingly being guarded by heterogeneous, multi-layer safety stacks combining text filters, image classifiers, and cross-modal detectors. Existing jailbreak studies either optimize against individual filters or query the complete pipeline with aggregate feedback, making it difficult to identify the active constraint and adapt to conflicts across safety layers.In this paper, we introduce the \emph{Detection Surface}, a unified geometric framework that characterizes the decision boundaries induced by heterogeneous T2I safety filters and their joint effect on the jailbreak search space. This formulation reveals that successful evasion is governed by a sparse and non-convex region shaped by cross-layer conflicts, where mutations that bypass one filter may increase exposure to another. Motivated by this analysis, we propose \emph{CRACK}, a multi-agent debate framework for adaptive jailbreak search that decomposes jailbreak search into exploration, diagnosis, and arbitration. CRACK coordinates an Attack Agent, a Defense Agent, and a Judge Agent to iteratively generate prompt mutations, obtain layer-specific diagnostic feedback, and optimize mutation strategies through reward-guided refinement. Through repeated rounds of debate, CRACK adapts its search direction to the evolving cross-layer constraints while preserving the original harmful intent. Extensive experiments across multiple T2I models, datasets, and safety configurations show that CRACK achieves Attack Success Rates (ASR) of up to 99.63\% under composite defenses, while requiring fewer queries than existing methods and maintaining semantic fidelity. |
| 2026-09-01 | [Griotte: Verified Compartmentalisation via Capabilities](http://arxiv.org/abs/2609.01110v1) | June Rousseau, Aïna Linn Georges et al. | CHERIoT is a novel hardware-software co-design that leverages hardware capabilities to define a notion of compartment, in a minimalistic capability-based OS, CHERIoT RTOS. By default, compartments are isolated to limit damage in case of bugs or malicious behaviour. To allow cross-compartment communication, the OS provides a privileged component, called the switcher. The switcher provides an interface for cross-compartment calls, while enforcing isolation between compartments and guaranteeing stack safety. Together with hardware capabilities, the switcher is critical to enforce the security guarantees of the CHERIoT compartment model. The design of CHERIoT raises two questions: First, how can one formalise the informal notion of compartmentalisation that CHERIoT compartments are designed to provide? And second, given that the safety properties of CHERIoT hinge on the complementary roles of the capability machine and of the switcher, does the design of CHERIoT enforce the desired security properties?   In this paper, we introduce Griotte and Griotte OS, idealised but faithful versions of the CHERIoT machine and the CHERIoT RTOS, which we use to answer these two questions: First, we formally capture the aforementioned security guarantees in the form of a continuation-based logical relation which captures the combined behaviour of the switcher and of the capability machine. And second, we define a specification for the Griotte switcher that enforces those guarantees, and prove that the implementation meets the specification. We demonstrate Griotte on a range of key scenarios illustrating different aspects of CHERIoT, including integrity of the local state in the presence of memory sharing with unknown code. Our approach is modular: we verify compartments individually, and then compose their specifications. Together, our contributions give a solid formal foundation to the design of CHERIoT. |
| 2026-09-01 | [JENGA: Exploiting Counter-Based RowHammer Countermeasures to Break Real-Time Predictability](http://arxiv.org/abs/2609.01077v1) | Valentin Abgrall, Marcello Traiola et al. | Safety-critical real-time systems must satisfy multiple dependability requirements, notably time predictability and security. In such systems, tasks must complete within bounded and known execution times, typically characterised through Worst-Case Execution Time (WCET) analysis. At the same time, DRAM-based platforms are increasingly sensitive to the RowHammer read-disturbance security vulnerability, which has motivated the development of numerous hardware and software countermeasures in both academia and industry. However, the impact of these defences is generally evaluated in terms of average-case performance, a metric that is insufficient for safetycritical real-time systems, where worst-case behaviour is the primary concern. In this paper, we study the impact of RowHammer countermeasures based on hardware counters on the timing behaviour of real-time systems. We use a Per-Row-Activation-Counter (PRAC) countermeasure as a case study, standardised for recent DDR5 memories, and show that it can introduce significant timing variations. Based on this observation, we introduce JENGA, an attack in which an attacker-controlled task manipulates the internal state of the RowHammer countermeasure mechanism to increase the execution time of a victim real-time task beyond its expected WCET. We implement JENGA in a gem5 and Ramulator 2.0 simulation environment and evaluate its impact on TACLeBench workloads. We show that such an attack can delay tasks up to 200% of their WCET, making the initial timesafety assumptions unsafe. To address this issue, we derive a safe analytical bound that accounts for mitigation-induced delays in WCET analysis for DRAM systems protected by hardware countermeasures, such as PRAC-N. |
| 2026-09-01 | [HiveTraceGuard-Pro: A Compact Generative Guardrail for Prompt Injection, Jailbreaks, and Adversarial Obfuscation](http://arxiv.org/abs/2609.01046v1) | Nikita Oblakov, Sabrina Sadiekh et al. | Production LLMs must handle inputs that attempt to override system instructions, bypass safety policies or elicit harmful responses. A common mitigation is a separate guardrail model. Existing reports, however, provide little evidence on Russian prompt injection or Russian surface obfuscation. We present HiveTraceGuard-Pro, a 0.6B generative guardrail LoRA-tuned from Qwen3-0.6B. It is trained on Russian and English and uses one binary scoring rule (safe/unsafe) for the final target turn. Its training corpus pairs harmful examples, where a counterpart exists, with benign examples from the same domain and applies eight obfuscation transforms to both labels. In one harness, we compare HiveTraceGuard-Pro with thirty-four other guards on nineteen benchmark groups, sixteen of which are public. Its aggregate key is 0.7432, behind 0.7641 and 0.7552 for the two higher-scoring guards. Over the sixteen public groups alone, its key is 0.7153 and four of the thirty-four other suite guards score higher. In a fifteen-model comparison, HiveTraceGuard-Pro has the highest clean Russian robustness combined-F1 (0.88) and Russian prompt-injection recall (0.999). Both results use Russian sets assembled by our team, and at least 27.1% of the prompt-injection set overlaps the training corpus. Its 14.3 ms median latency is the lowest among those fifteen models in that run. Across the suite, FPR is 0.268 and FNR is 0.156. All reported response results use a legacy standalone-reply serialization rather than the natural assistant-role path of the shipped chat template. We release the merged weights on Hugging Face under Apache-2.0. The corpus, evaluation sets and evaluation code remain internal. |
| 2026-09-01 | [Spawn Freely, Act Sparingly: Progressive Risk Vesting for Recursive LLM-Agent Trees](http://arxiv.org/abs/2609.01035v1) | Molly Wang | Recursive LLM agents can broaden their search by spawning specialists. Some branches later request tools that send data or deploy code. When should a branch receive authority to act? We distinguish sandbox spawning, in which external controls prevent the specified harm, from capability activation, in which a selected branch crosses an irreversible-action boundary. Progressive Risk Vesting (PRV) holds a trajectory-level risk budget in escrow and debits it as branches are activated. We prove an anytime harm bound for adaptively generated trees. Branch outcomes may be dependent, but each local certificate needs to remain valid conditional on the full pre-activation history, including the information used to select the request. When activation gates, branch charges, and compute constraints are held fixed, delayed vesting preserves every policy available under irrevocable spawn charging. Marginal risk estimates can still fail after branch selection. In a stylized branching model, trajectory harm changes as the authority reproduction number $\mathcal{R}_A$ crosses one. As local risk $p$ approaches zero, trajectory harm is proportional to $p$ below criticality, proportional to $\sqrt{p}$ at criticality, and retains a positive floor above it. A finite-type occupancy model yields risk and compute shadow prices. For nested fanout modes with decreasing marginal value per unit risk, these prices produce a threshold rule. Branching calculations and a split-sample experiment illustrate the results. These synthetic studies do not estimate safety in deployed agents. The analysis suggests a design rule: search broadly in the sandbox and grant recursive authority sparingly, with an explicit risk charge. |
| 2026-09-01 | [Real-Time Model Predictive Control Algorithms for Autonomous Spacecraft Guidance](http://arxiv.org/abs/2609.00927v1) | Mohammed-Adnane Garab | This report studies and compares four families of Model Predictive Control (MPC) algorithms for autonomous spacecraft rendezvous guidance: Linear MPC, Tube MPC, Fast/Embedded MPC, and Successive Convexification (SCvx). Using the Clohessy-Wiltshire-Hill (CWH) relative-motion model, we show that the marginal stability of the underlying dynamics causes the condition number of the condensed Hessian to grow sharply with the prediction horizon, which explains why gradient-based solvers such as Fast MPC diverge under long-horizon stress tests while exact linear-algebra solvers (ADMM with Cholesky factorisation) remain unaffected. Tube MPC is validated across five standard rendezvous manoeuvres (Translation, R-bar, V-bar, Natural Motion Circumnavigation, and Corkscrew) using a single fixed controller configuration, and a two-sided, provably correct bound is derived to bracket the worst-case tracking error directly from the cost weight matrices. The framework is further extended to track an arbitrary, non-closed-form reference trajectory through online-recomputed linearisation, with the safety guarantee holding throughout. Finally, to assess feasibility for real flight software, the core numerical routines (Cholesky factorisation and the Riccati equation solver) are re-implemented from first principles, without external libraries, and validated against standard scientific computing tools to machine precision. Taken together, these results support Tube MPC as a robust and computationally realistic controller for onboard, real-time spacecraft rendezvous guidance. |
| 2026-09-01 | [In-Context Neurofeedback: Can LLMs Control Their Internal Representations through Privileged Access?](http://arxiv.org/abs/2609.00904v1) | Koshiro Aoki, Ryota Takatsuki et al. | Whether large language models (LLMs) can control their own internal representations matters for both machine metacognition and AI safety. A recent study applied neurofeedback to LLMs and claimed that they can control their internal representations. However, the reported control may rely on superficial mechanisms rather than genuine internal access because the control targets in that study are not privileged, meaning that a third party can infer them from the prompt. We redesign the neurofeedback paradigm for LLMs so that the control target satisfies the privileged access requirement, which is closer to neurofeedback experiments in human cognitive neuroscience. Under this stricter setting, the models do not demonstrate reliable control over privileged internal representations, suggesting that previously reported control cannot exclude the possibility that it relies on superficial mechanisms. Our results indicate that rigorous assessments of metacognition in LLMs require evaluation methods that demand privileged access. |
| 2026-09-01 | [Towards reliable multimodal disaster severity assessment through preference optimization and explainable vision-language reasoning](http://arxiv.org/abs/2609.00879v1) | Yuanjun Zhang, Fuzel Ahamed Shaik et al. | Reliable disaster damage assessment requires models that provide both accurate predictions and transparent explanations. However, existing multimodal approaches are limited by scarce annotated data and insufficient evaluation of reasoning quality. This study proposes a two-stage training framework that integrates Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) within a unified data construction pipeline. From a single Human-in-the-Loop (HITL) annotation workflow, two complementary datasets are derived, namely ReasoningSet, which contains validated rationales for SFT, and PreferenceSet, which comprises paired rationales for DPO-based alignment. The framework evaluates both classification performance and explanation quality using automatic metrics, model-based scoring, and human ranking. Experimental results show that SFT improves accuracy from 73.64% to 78.29% and increases Macro-F1 by 29% compared to the baseline, while explanation quality improves by approximately 25%. Subsequent DPO alignment further enhances interpretability on the PreferenceSet. Cross-model validation on InternVL-3-8B and LLaVA-1.5-7B demonstrates the robustness and generalizability of the approach. The proposed framework improves detection of underrepresented mild damage cases, reduces high-risk misclassifications, and strengthens alignment between model reasoning and human judgment. Overall, it provides a reproducible pathway to develop reliable multimodal systems that deliver auditable, actionable disaster insights for emergency management. |

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



