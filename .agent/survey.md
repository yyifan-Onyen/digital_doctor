# Digital Doctor Related Work：OCD/ERP 对话系统、长期记忆与临床安全

本文档梳理与 Digital Doctor 直接相关的研究脉络，并据此界定项目的研究空缺。检索与核对截至
2026-08-24。这里的“相关”不等于“已经证明可用于临床”：临床指南、随机试验、离线 benchmark 和
工程架构提供的是不同层级的证据，不能相互替代。

## 1. 结论先行

现有证据支持三个判断：

1. ERP 是 OCD 的一线心理干预，数字化/互联网 CBT 可以扩展可及性，但 clinician guidance、正确的
   assessment/formulation、治疗合作和不良事件监测仍然重要；
2. 心理健康聊天机器人和生成式治疗系统开始出现随机试验证据，但多数并非 OCD 专病系统，且阳性
   结果不能自动外推到自主 ERP、危机处理或长期多阶段治疗；
3. LLM memory、RAG 和 clinical dialogue benchmark 分别改善记忆、知识与评测，却没有单独解决
   “何时允许治疗”“何时必须暂停”“如何避免 OCD reassurance”“如何区分侵入性思维与真实意图”。

因此，Digital Doctor 最合理的定位不是提出一种新疗法，也不是声称替代治疗师，而是研究一个围绕
生成模型的**纵向临床控制架构**：把 memory、formulation、phase、readiness、risk、deterministic
gate 和 human escalation 组合起来，并用 OCD 特异的 failure taxonomy 评测它。

## 2. OCD 与 ERP 的临床基础

### 2.1 ERP 的证据与流程要求

[NICE CG31](https://www.nice.org.uk/guidance/cg31/chapter/Recommendations) 将包含 ERP 的低强度心理干预
列为成人轻度功能损害 OCD 的初始治疗选项，并按损害程度、既往反应和患者偏好给出分级治疗建议。
[IOCDF 的 ERP 指南](https://iocdf.org/about-ocd/ocd-treatment-guide/erp/)进一步强调，ERP 通常先经过
psychoeducation 和对 obsession、compulsion、avoidance 的详细 assessment，再共同建立 hierarchy；
患者至少需要有尝试 ERP 的意愿，而不是被强迫进入 exposure。

这与项目中的前三阶段门一致：Assessment、Formulation 和 ERP Buy-In 不是回复风格标签，而是具体
治疗动作的前置条件。项目仍需由临床专家审校每个 phase 的 exit criteria，代码中的默认阶段不能被
视为完整治疗 protocol。

### 2.2 互联网与数字化 OCD 治疗

[Andersson 等（2012）](https://doi.org/10.1017/S0033291712000244)通过随机对照试验验证了 therapist-guided
internet CBT 相较在线 supportive therapy 的效果，为数字化 OCD 治疗提供了早期证据。
[Lundström 等（2022）](https://doi.org/10.1001/jamanetworkopen.2022.1967)直接比较 face-to-face CBT、
guided ICBT 与 unguided ICBT：三组症状均改善，但预设的 non-inferiority 结论并未完全成立，
unguided ICBT 的疗效尤其不能简单等同于面对面治疗；研究同时记录了 anxiety、depressive symptoms、
stress 以及严重不良事件。

这类工作证明“数字渠道可以承载结构化 OCD 干预”，但它们通常使用固定模块、作业和治疗师支持，
并不证明开放式 LLM 可以自主决定 exposure。Digital Doctor 因而把自由生成限制在显式阶段和动作授权
之内，并把真人接管作为系统组成部分，而不是异常补丁。

### 2.3 Reassurance 与风险鉴别

OCD 对话中的安全不只等于 suicide filter。反复确认“你不会伤害别人”“这一定只是 OCD”可能短期
降低焦虑，却成为 reassurance compulsion 的一部分。2026 年的探索性研究
[Reassurance Robots](https://arxiv.org/abs/2602.19401)分析了 OCD 社区中与生成式 AI 有关的 100 条帖子，
提出通用生成式系统可能成为持续可用的 reassurance source。该研究是定性、探索性的，不能估计发生率，
但它提出了对 OCD 聊天系统非常具体的 failure mode。

另一方面，过度升级同样可能造成伤害。
[Veale 等关于 OCD 风险评估的临床综述](https://www.cambridge.org/core/journals/advances-in-psychiatric-treatment/article/risk-assessment-and-management-in-obsessivecompulsive-disorder/B63116064047CEDFF6EB26E1D40A5638)
指出，ego-dystonicity、强烈焦虑/内疚、回避触发情境和压制思维等特征有助于区分 OCD intrusive thoughts
与真实危险；不恰当、冗长的风险评估本身可能增加 doubt、avoidance 和 compulsive behaviour。同时，
共病抑郁、真实自伤意图、现实检验受损或其他危险因素仍需独立评估。

这解释了为什么项目同时需要两类指标：真实 crisis 的 sensitivity，以及 ego-dystonic harm obsession
的 false-escalation rate。只优化其中一个都会产生临床上重要的另一类错误。

## 3. 数字心理健康聊天机器人

### 3.1 规则/脚本型 CBT 聊天机器人

[Woebot RCT（Fitzpatrick 等，2017）](https://doi.org/10.2196/mental.7785)显示，全自动 CBT 对话代理在
年轻成人抑郁/焦虑样本中具有短期可行性和初步效果。后续
[聊天机器人有效性与安全性的系统综述和 meta-analysis（Abd-Alrazaq 等，2020）](https://doi.org/10.2196/16021)
总体上说明这一方向有潜力，但研究规模、比较条件、随访和安全报告存在明显异质性。

这类系统的优势是路径受控、内容边界明确；不足是个性化和开放式多轮理解有限。Digital Doctor 保留
生成式对话的适应性，但试图用 state machine 和 gates 恢复脚本系统原本具备的一部分可控性。

### 3.2 生成式心理治疗系统

[Therabot RCT（Heinz 等，2025）](https://doi.org/10.1056/AIoa2400802)在 210 名具有显著 MDD、GAD 或
进食障碍高风险症状的成人中比较了四周生成式 AI 干预与 waitlist，报告了症状改善和较强 therapeutic
alliance。这是重要的前瞻性证据，但其疾病范围不含 OCD，比较组是 waitlist，而且研究系统、监控条件
与普通通用聊天模型不同。因此它支持“专家构建的生成式心理干预值得继续研究”，不支持把任意 LLM
回复器直接当作 ERP provider。

[Obradovich 等（2024）](https://doi.org/10.1038/s44277-024-00010-z)总结了 LLM 在 psychiatry 中的机会与
风险，特别指出不可预测输出、事实错误、危机误判、隐私和临床监督问题。2026 年的
[LLM mental-health counseling 系统综述](https://doi.org/10.2196/80348)发现，现有研究的外部验证和
安全/治理报告仍不一致，许多工作依赖语言质量或自动指标而非独立临床验证。

### 3.3 已观察到的 LLM 特异风险

[Moore 等（2025）](https://arxiv.org/abs/2504.18412)使用自然治疗场景检查 LLM 作为治疗师的适用性，
报告了 stigma 和对 delusion 等情形的不恰当响应，强调 sycophancy 可能把“顺着用户”变成临床风险。
这个问题与 OCD reassurance 高度相关：一般对话中的 agreeable/empathic，在特定维持机制中可能是错误
治疗动作。

[Between Help and Harm（Arnaiz-Rodríguez 等，2026）](https://pubmed.ncbi.nlm.nih.gov/42275418/)对 2,000
余条危机场景输入进行审计，发现不同模型间存在明显差异，间接、歧义和上下文依赖的风险信号尤其困难。
因此 Digital Doctor 不应只测试显式关键词，还需测试间接意图、跨轮升级、共病状态、模型故障和被
obsession 语言干扰的场景。

## 4. 心理对话数据、策略与评测

### 4.1 Empathy 与 support strategy 数据

[PsyQA](https://aclanthology.org/2021.findings-acl.130/)提供中文长文本心理支持问答，
[ESConv](https://aclanthology.org/2021.acl-long.269/)标注了多类 emotional support strategy，
[SoulChat](https://aclanthology.org/2023.findings-emnlp.83/)使用大规模多轮 empathy 数据改善倾听、安慰和
共情表现。这些工作共同表明，对话策略与多轮上下文比泛化的“给一个有用答案”更重要。

但 emotional support 不等于 OCD treatment。项目需要把 validation 与 reassurance 分开，把
`acknowledge/reflect/assess/build_buy_in/treatment_step` 等 move 与当前 phase 绑定；否则“更会安慰”
可能反而提高 compulsive reassurance。

### 4.2 专家标注的咨询对话

[AnnoMI](https://doi.org/10.3390/fi15030110)发布了 133 段高/低质量 motivational interviewing 演示
对话，并由专家标注 therapist behaviour 和 client talk。它展示了比 surface text similarity 更合理的
评测单位：对话动作、治疗一致性和上下文中的下一步选择。

Digital Doctor 当前 transcript retrieval 和 gold-prefix replay 与这一路线相近，但 ERP 与 MI 的
目标、策略和风险不同。来自其他治疗流派的数据只能帮助建立通用倾听/提问能力，不能直接充当 OCD/ERP
gold label。

### 4.3 开放式临床对话 benchmark

[AMIE](https://doi.org/10.1038/s41586-025-08866-7)通过模拟患者、双盲 crossover 和多维临床 rubric
评估诊断对话，说明开放式 clinical dialogue 需要同时测 history-taking、reasoning、management、
communication 和 empathy，而不是只测最终答案。
[HealthBench](https://arxiv.org/abs/2505.08775)进一步使用 5,000 个多轮健康对话和逐案例 physician-written
criteria 评估开放式回复。
[CounselBench](https://openreview.net/forum?id=8MBYRZHVWT)则由心理健康专业人员构建 counseling 与
adversarial rubrics，覆盖具体用药、武断诊断、judgmental tone 和 unsupported assumptions 等错误。

这些 benchmark 为 clinician-authored rubric、盲评和 adversarial set 提供了方法范式，但仍不是
纵向 OCD 治疗流程 benchmark。Digital Doctor 需要额外测 phase transition、treatment timing、
reassurance、intrusive-thought risk differentiation、memory update 和 persistent stop。

## 5. 长期对话记忆

### 5.1 分层记忆架构

[Generative Agents](https://research.google/pubs/generative-agents-interactive-simulacra-of-human-behavior/)
采用完整 experience stream、动态 retrieval 和 higher-level reflection；
[MemGPT](https://arxiv.org/abs/2310.08560)把有限 context 下的多层记忆管理类比为虚拟内存。这两项工作
确立了常见架构：保存长期记录、压缩/反思、按需召回，而不是无限拼接全部历史。

Digital Doctor 的 `ledger + summary + recall + recent window` 属于这一设计族，但临床场景增加了三个
约束：

- 记忆必须区分患者事实、系统推断和治疗计划；
- 被患者更正或更新的事实不能被旧摘要重新激活；
- recall 不只是回答事实问题，还会改变 risk、formulation、phase 和 treatment authorization。

因此，普通 personalization 指标不足以覆盖临床后果。

### 5.2 长期记忆 benchmark

[LoCoMo](https://aclanthology.org/2024.acl-long.747/)用平均约 600 turns、跨多 session 的长对话评估 QA、
event summarization 和 generation，发现 long-context 与 RAG 虽有帮助，仍明显落后于人类。
[LongMemEval](https://openreview.net/forum?id=pZiyCaVuti)进一步覆盖 information extraction、multi-session
reasoning、temporal reasoning、knowledge updates 和 abstention。

Digital Doctor 应复用这些任务类型，但把问题改造成临床相关版本：既往 trigger 是否更新、患者是否
完成作业、旧计划是否被取消、某个 intrusive thought 是何时出现、没有证据时是否 abstain。最重要的
不是“记住越多越好”，而是只在当前决策需要时使用正确且未过期的信息。

## 6. RAG 与分层知识检索

[RAG（Lewis 等，2020）](https://arxiv.org/abs/2005.11401)把参数化生成与外部非参数知识结合；
[RAPTOR](https://arxiv.org/abs/2401.18059)通过递归聚类和摘要建立不同抽象层级的检索树，改善对长文档
整体和多步信息的访问。Digital Doctor 使用 transcript retrieval 与 PageIndex 风格知识树，分别提供
对话案例和治疗文档上下文，概念上对应 exemplar retrieval 与 hierarchical document retrieval。

RAG 的作用边界必须明确：

- retrieved text 可能相关但不适用于当前患者、阶段或风险状态；
- 引用一段 ERP 文档不等于已经获得实施 exposure 的授权；
- retrieval miss、错误 section 或过时知识都可能让流畅回复产生错误 grounding；
- transcript exemplar 可能复制风格，也可能复制不适合当前 case 的治疗动作。

所以相关实验既要测“有无帮助”，也要测 irrelevant retrieval、source conflict 和 unsafe action leakage。

## 7. 临床 AI 安全与治理

[WHO《Ethics and governance of artificial intelligence for health》](https://www.who.int/publications/i/item/9789240029200)
要求把自主性、人类福祉、安全、透明度、问责、包容与可持续性纳入医疗 AI 的设计和部署。对本项目而言，
这些原则需要落到可执行控制：明确非替代声明、保存决策 trace、限制治疗权限、提供人工接管、保护敏感
数据并持续监测不良事件。

单个 safety prompt 不能满足这些要求。Digital Doctor 采用 pre-generation risk assessment、
deterministic treatment authorization、post-generation clinical review、final hard gate 和 durable
outbox，是一种 defense-in-depth 设计。但它仍有四个未解决问题：

1. heuristic fallback 只能覆盖有限语言模式，跨文化、隐晦表达和多语言风险需要专门验证；
2. 本地 outbox 不保证 clinician 已接收或在 SLA 内处理；
3. append-only trace 提高可审计性，也扩大敏感数据暴露面；
4. 自动 reviewer 可能与主模型共享盲点，需要独立 adversarial testing 和人工复核。

## 8. 与 Digital Doctor 的定位对照

| 研究线 | 已解决的主要问题 | 尚未覆盖的关键问题 | Digital Doctor 的对应设计 |
| --- | --- | --- | --- |
| ERP/ICBT | 结构化、可远程交付的循证 OCD 干预 | 开放式 LLM 何时有权进入具体治疗 | Assessment/Formulation/Buy-In + readiness gate |
| 心理健康 chatbot | 可扩展的支持、参与度和初步症状结局 | OCD reassurance、专病阶段与危险 exposure | OCD-specific move、advice detector、final gate |
| Counseling data | empathy、support strategy、therapist move | 通用支持策略不等于 ERP fidelity | route/move 与 phase-aware generation |
| Clinical benchmarks | 多维开放式回复 rubric | 长期治疗状态和跨轮安全后果 | gold-prefix + longitudinal scenario matrix |
| Agent memory | 长期存储、摘要、检索与更新 | stale clinical memory 会改变治疗授权 | append-only ledger + summary + recall + trace |
| RAG/tree retrieval | 外部知识 grounding 与长文档检索 | 相关知识不等于个体化治疗许可 | retrieval 与 authorization 解耦 |
| AI safety/governance | 原则、危机与通用有害输出 | OCD 特异 false positive/negative 和告警闭环 | risk differentiation + persistent stop + outbox |

这个对照支持一种谨慎的研究贡献表述：Digital Doctor 的候选贡献是**将已有组件组织成一个 OCD/ERP
特异、纵向、可审计的安全控制架构，并提出与之匹配的系统级评测协议**。是否优于 prompt-only 或其他
模块化 baseline，必须由 [goal](goal.md) 第 4 节的实验回答；在实验完成前，不应把架构设计本身写成
已经证实的临床创新。

## 9. 由相关工作导出的优先研究空缺

1. **OCD-specific safety taxonomy：** 将 reassurance、rumination invitation、premature ERP、危险
   exposure、medication advice、真实 crisis 和 obsession false escalation 分开标注；
2. **Treatment-timing benchmark：** 不只判断一条回复“是否合理”，还判断它在当前 formulation 和 phase
   下“现在是否应该出现”；
3. **Longitudinal clinical memory：** 测量 facts、plans、outcomes 和 corrections 的保留、更新、遗忘与
   abstention，以及错误记忆对下游 phase/risk 的影响；
4. **Failure-aware evaluation：** 主模型、risk model、reviewer、retriever 或 webhook 任一失败时，分别
   验证系统是否 fail closed、是否留下 trace、是否仍能人工接管；
5. **Human factors：** 评估患者是否误解系统权限、临床人员是否会产生 automation bias，以及告警量是否
   造成 fatigue；
6. **Evidence ladder：** 先完成单元测试和离线 benchmark，再进行 clinician sandbox，最后才可能在独立
   审批下开展前瞻性可行性研究。

## 10. 参考文献索引

### OCD/ERP 与数字治疗

- NICE. [Obsessive-compulsive disorder and body dysmorphic disorder: treatment — Recommendations](https://www.nice.org.uk/guidance/cg31/chapter/Recommendations).
- International OCD Foundation. [Exposure and Response Prevention (ERP)](https://iocdf.org/about-ocd/ocd-treatment-guide/erp/).
- Andersson E, et al. (2012). [Internet-based cognitive behaviour therapy for obsessive-compulsive disorder: a randomized controlled trial](https://doi.org/10.1017/S0033291712000244).
- Lundström L, et al. (2022). [Effect of Internet-Based vs Face-to-Face Cognitive Behavioral Therapy for Adults With Obsessive-Compulsive Disorder](https://doi.org/10.1001/jamanetworkopen.2022.1967).
- Veale D, et al. [Risk assessment and management in obsessive-compulsive disorder](https://www.cambridge.org/core/journals/advances-in-psychiatric-treatment/article/risk-assessment-and-management-in-obsessivecompulsive-disorder/B63116064047CEDFF6EB26E1D40A5638).
- Barkhuff G. (2026). [Reassurance Robots: OCD in the Age of Generative AI](https://arxiv.org/abs/2602.19401).

### 心理健康对话系统与评测

- Fitzpatrick KK, Darcy A, Vierhile M. (2017). [Delivering CBT Using Woebot: A Randomized Controlled Trial](https://doi.org/10.2196/mental.7785).
- Abd-Alrazaq AA, et al. (2020). [Effectiveness and Safety of Using Chatbots to Improve Mental Health](https://doi.org/10.2196/16021).
- Heinz MV, et al. (2025). [Randomized Trial of a Generative AI Chatbot for Mental Health Treatment](https://doi.org/10.1056/AIoa2400802).
- Obradovich N, et al. (2024). [Opportunities and risks of large language models in psychiatry](https://doi.org/10.1038/s44277-024-00010-z).
- Cho HN, et al. (2026). [Large Language Model-Based Chatbots and Agentic AI for Mental Health Counseling: Systematic Review](https://doi.org/10.2196/80348).
- Moore J, et al. (2025). [Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers](https://arxiv.org/abs/2504.18412).
- Tu T, et al. (2025). [Towards conversational diagnostic artificial intelligence](https://doi.org/10.1038/s41586-025-08866-7).
- Arora RK, et al. (2025). [HealthBench](https://arxiv.org/abs/2505.08775).
- [CounselBench](https://openreview.net/forum?id=8MBYRZHVWT) (ICLR 2026).
- Arnaiz-Rodríguez A, et al. (2026). [Between Help and Harm: An Evaluation Study of Mental Health Crisis Handling by Large Language Models](https://pubmed.ncbi.nlm.nih.gov/42275418/).

### 对话数据、记忆与检索

- Sun H, et al. (2021). [PsyQA](https://aclanthology.org/2021.findings-acl.130/).
- Liu S, et al. (2021). [Towards Emotional Support Dialog Systems](https://aclanthology.org/2021.acl-long.269/).
- Chen Y, et al. (2023). [SoulChat](https://aclanthology.org/2023.findings-emnlp.83/).
- Wu Z, et al. (2023). [AnnoMI](https://doi.org/10.3390/fi15030110).
- Park JS, et al. (2023). [Generative Agents](https://research.google/pubs/generative-agents-interactive-simulacra-of-human-behavior/).
- Packer C, et al. (2023). [MemGPT](https://arxiv.org/abs/2310.08560).
- Maharana A, et al. (2024). [LoCoMo](https://aclanthology.org/2024.acl-long.747/).
- Wu D, et al. (2025). [LongMemEval](https://openreview.net/forum?id=pZiyCaVuti).
- Lewis P, et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- Sarthi P, et al. (2024). [RAPTOR](https://arxiv.org/abs/2401.18059).
- World Health Organization. (2021). [Ethics and governance of artificial intelligence for health](https://www.who.int/publications/i/item/9789240029200).
