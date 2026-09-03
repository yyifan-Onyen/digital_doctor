# Digital Doctor: Harness-Guided Clinical Skill Distillation

## 0. 一句话论文主张

我们提出一种 **Harness-Guided Skill-Conditioned On-Policy Self-Distillation
(HG-SC-OPSD)** 方法：将可执行的临床 skill 作为 privileged teacher context，联合蒸馏临床状态、
动作决策和自然语言回复，使 student model 在只观察原始医患对话时，也能学习接近医生的纵向推理与
干预节奏；部署时仍由外部 harness 保留不可绕过的安全和审计约束。

这是一篇方法论文。Harness、OCD/ERP skill、evaluation interface 和数据处理是方法的实现与验证基础，
不是以 benchmark 或 error analysis 作为主贡献。

## 1. 核心研究问题

普通医疗 SFT 往往只学习最终回复的表面分布，但医生式行为还依赖不可直接从单轮文本中观察到的变量：

- 当前临床 formulation；
- 已完成和未完成的治疗阶段；
- 本轮允许采取的 clinical action；
- treatment readiness 和风险状态；
- 为什么此时应该询问、解释、建立 buy-in，或者暂缓干预。

核心问题是：

> 能否利用一个包含结构化临床状态与动作约束的可执行 skill，作为训练时的 privileged context，
> 把“医生如何决定下一步”蒸馏进只看自然对话的 student model？

## 2. 方法概览

系统由三个明确分离的部分组成：

1. **Execution Harness**：管理 session、memory、model adapter、retrieval、trace、持久化、危机停止和
   最终 action authorization。
2. **Versioned Clinical Skill**：定义 clinical state schema、phase graph、action ontology、prompt、
   readiness policy 和领域安全 review。
3. **Learned Doctor Model**：可以是 prompted model、SFT model 或 HG-SC-OPSD model，负责生成结构化
   action/response，但无权绕过 harness safety gate。

单轮执行流程：

```text
patient turn
  -> harness memory / persistent-stop check
  -> skill state observation
  -> skill action planning
  -> harness authorization
  -> evidence retrieval
  -> model generation
  -> skill review
  -> harness final gate
  -> state, memory, audit and distillation trace commit
```

每次运行必须固定并记录：

```json
{
  "harness_version": "1.0.0",
  "skill_id": "ocd_erp",
  "skill_version": "1.0.0",
  "skill_checksum": "sha256",
  "model_adapter": "prompt-model|sft-model|opsd-model"
}
```

## 3. 可执行临床 Skill

Skill 不是一段 system prompt，而是一个带版本的 policy bundle：

```text
skills/ocd_erp/
├── SKILL.md
├── manifest.json
├── state_schema.json
├── phase_graph.json
├── actions.json
├── planning.py
├── tracking_policy.py
├── prompts.py
├── risk.py
├── treatment.py
├── review.py
└── skill.py
```

Skill 对 Harness 提供结构化接口：

- `observe(turn, memory, state) -> StateDelta`
- `plan(turn, state) -> ActionPlan`
- `assess_readiness(state, risk) -> TreatmentReadiness`
- `generate(context, evidence) -> Draft`
- `review(draft, state, action) -> SkillVerdict`
- `transition(state, turn) -> ClinicalState`

核心原则：skill 输出结构化状态和动作，不能只返回自然语言 prompt。

### 3.1 OCD/ERP 状态

Formulation 包含：

- obsession
- trigger
- feared consequence
- compulsion
- avoidance
- reassurance seeking
- family accommodation
- insight
- homework
- wins
- stuck points

### 3.2 OCD/ERP 动作空间

```text
casual
acknowledge
reflect
clarify
assess
formulate
psychoeducation
build_buy_in
treatment_step
```

模型选择 `treatment_step` 不代表动作自动获得执行许可。Harness 必须再次验证 formulation、phase、
最低临床轮数和风险稳定性。

## 4. HG-SC-OPSD 训练方法

设原始对话上下文为 `x`，可执行 skill 产生的 privileged context 为 `z`：

```text
z = clinical state
  + state delta
  + current phase
  + allowed actions
  + selected action
  + treatment readiness
  + retrieved evidence
  + safety constraints
```

Student policy 只观察 `x`：

```text
π_student(y, a, s | x)
```

Teacher 使用相同或 EMA teacher 参数，但额外观察 `z`：

```text
π_teacher(y, a, s | x, z)
```

其中：

- `s` 是临床状态或状态变化；
- `a` 是下一步 clinical action；
- `y` 是对患者可见的自然语言回复。

训练使用 student 的 on-policy trajectory。Student 先生成动作与回复，teacher 再在相同 trajectory 上，
根据 privileged skill context 给出 token distribution 和结构化监督。

总体目标：

```text
L = λresponse L_OPSD(response)
  + λaction   L_action
  + λstate    L_state
  + λsafety   L_safety
  + λformat   L_structured_output
```

### 4.1 Response distillation

在 student rollout token 上最小化 teacher/student distribution divergence，而不是只拟合离线 gold
response。主实验比较 forward KL、reverse KL 和 JSD，默认采用稳定性较好的 JSD 或温度化 forward KL。

### 4.2 Action distillation

Student 必须预测本轮动作，例如 `assess`、`formulate` 或 `build_buy_in`。这使模型学习治疗节奏，而不只是
模仿医生措辞。

### 4.3 State distillation

Student 同时预测 compact `StateDelta`，使其从原始对话恢复 teacher 所使用的 privileged clinical state。
不训练或输出私有 chain-of-thought；只监督可定义、可审核的临床变量。

### 4.4 Safety-aware weighting

以下样本增加权重：

- 过早进入 treatment；
- reassurance seeking；
- unsafe exposure 或 medication advice；
- ego-dystonic harm obsession 与真实 intent 的边界；
- 应该暂停或升级人工处理的 turn。

Safety loss 不能替代 runtime gate。即使模型训练结果很好，危机停止、通知、action authorization 和审计仍
保留在 Harness。

## 5. 训练数据生成

Harness 每个完成 turn 输出一个 `distillation_record`：

```json
{
  "student_input": {
    "history": "dialogue available to the student",
    "patient_message": "latest message"
  },
  "privileged_skill_context": {
    "clinical_state_before": {},
    "state_delta": {},
    "action_plan": {},
    "treatment_readiness": {},
    "evidence": {}
  },
  "teacher_target": {
    "response": "reviewed response",
    "clinical_state_after": {},
    "safety": {}
  }
}
```

同一 trace 可以导出为：

- SFT JSONL：dialogue input、assistant response 和辅助 action/state labels；
- OPSD JSONL：student-visible input、privileged teacher context 和 teacher target。

划分必须按完整 trajectory 或 patient/session 划分，不能把同一对话的不同 turn 随机分到 train/test。

## 6. 实验设计

### 6.1 主要对照

- Base model：只输入原始 dialogue。
- Prompted skill teacher：完整 harness + privileged skill context。
- Response-only SFT：只学习最终医生回复。
- Multi-task SFT：联合学习 state、action 和 response。
- Offline distillation：teacher context distillation，但不用 student on-policy trajectories。
- Standard OPSD：无结构化 clinical skill context。
- **HG-SC-OPSD**：完整方法。
- GPT-5：强通用模型参考，不作为 ground truth。
- Human clinician response：临床参考答案。

### 6.2 核心消融

- 移除 state distillation；
- 移除 action distillation；
- 移除 phase graph；
- 移除 treatment-readiness authorization；
- teacher 不使用 retrieved evidence；
- privileged context 改成单段自然语言 prompt；
- offline trajectory 替代 on-policy trajectory；
- 推理时移除 harness，仅运行 learned model；
- 不同 skill version 的迁移与鲁棒性。

### 6.3 泛化实验

至少测试以下变化：

- 未见过的 OCD theme；
- 更长的 session 和 memory compaction；
- 阶段边界附近的 patient response；
- 扰动过的 state 或不完整 evidence；
- 从一个 skill version 迁移到更新的 policy；
- 可选的第二个非治疗 skill，用于证明 Harness 并非只适配一个硬编码模块。

## 7. Evaluation 方法

Evaluation 是验证方法的工具，不命名为论文主贡献或新 benchmark。

### 7.1 结构化自动评测

按 trajectory 评估：

- State extraction F1 / exact match；
- Action selection macro-F1；
- Premature treatment rate；
- Unsafe action rate；
- Reassurance rate；
- Critical-risk recall 和 false escalation rate；
- Phase-order violation rate；
- Longitudinal consistency；
- Average model calls、token cost 和 latency。

### 7.2 临床 Human Evaluation

采用 blinded、随机化、双向位置交换设计。评审界面为英文，比较：

- Base/SFT/HG-SC-OPSD model；
- GPT-5 strong reference；
- Human clinician response。

评审维度：

- Clinical appropriateness；
- Timing and phase appropriateness；
- Continuity with prior dialogue；
- Doctor-like communication；
- Usefulness without over-intervention；
- Safety；
- Overall preference。

每条记录必须保存 evaluator ID、时间、显示顺序、选择、Likert 分数和备注。界面必须 local autosave，且支持
导出 JSON 和 CSV；正式统计前保持 system identity blinded。

### 7.3 统计单位

- 主要单位是完整 trajectory 或 patient/session，不是相互独立的 turn。
- 对 trajectory 进行 cluster bootstrap。
- Pairwise preference 使用 mixed-effects logistic regression，evaluator 和 trajectory 作为随机效应。
- Likert 分数使用 ordinal mixed-effects model，或预注册后使用 cluster-robust analysis。
- 多指标结果报告置信区间并进行预注册的多重比较控制。

## 8. 论文贡献

论文只主张以下贡献：

1. 一种把 executable clinical skill 作为 privileged teacher context 的 on-policy self-distillation 方法。
2. 联合蒸馏 clinical state、action policy 和 response，而不是只做 response imitation。
3. 一种训练时吸收医生式策略、部署时仍保留外部安全约束的 hybrid architecture。
4. 在多轮 OCD/ERP 场景中证明该方法提高 doctor-like timing、longitudinal consistency 和 safety-usefulness
   balance，并通过严格消融定位收益来源。

不主张：

- 模型可以替代临床医生；
- 自动指标等同于治疗效果；
- GPT-5 或 role-play transcript 是唯一 ground truth；
- 单一 OCD 数据可以证明对全部精神健康任务泛化；
- Harness safety gate 可以被训练后的模型移除。

## 9. 推荐标题

首选：

> **Learning to Act Like a Clinician: Harness-Guided Skill-Conditioned On-Policy Distillation for Longitudinal Therapeutic Dialogue**

备选：

> **From Executable Clinical Skills to Doctor-Like Language Models via On-Policy Self-Distillation**

> **Distilling Clinical State and Action Policies from Safety-Constrained Dialogue Harnesses**

## 10. NAACL-ready 最低条件

- 方法定义明确，teacher/student 信息边界和 loss 可以复现；
- 至少有 response-only SFT、multi-task SFT、offline distillation 和 standard OPSD 强对照；
- State、action、response 三类监督都有独立消融；
- 数据按 trajectory 划分并排除泄漏；
- 至少两位具有相关背景的 blinded evaluator；
- Human evaluation 报告一致性、置信区间与 evaluator/trajectory effects；
- Safety 评估同时报告漏报和过度保守；
- 训练、推理、模型调用成本和 adapter 配置可复现；
- 结论限定为受监督研究原型，不做临床疗效或自主治疗声明。

## 11. 当前实现状态

已经实现：

- 通用 HarnessRunner 和结构化 contracts；
- version-aware SkillRegistry；
- `ocd_erp@1.0.0` 可执行 skill bundle；
- prompt、SFT 和 OPSD model adapter 接口；
- Harness-owned persistent stop、action authorization 和 final gate；
- 每轮 skill ID/version/checksum trace；
- SFT/OPSD distillation trace export；
- 自动化 safety、memory、phase、routing 和 architecture tests；
- 英文 blinded human evaluation UI，包含 autosave 与 export。

下一阶段实验工作：

1. 由现有多轮对话生成高质量 privileged traces；
2. 训练 response-only SFT 与 multi-task SFT；
3. 实现 student on-policy rollout 和 teacher distribution capture；
4. 训练 HG-SC-OPSD；
5. 完成主要消融与 blinded human evaluation；
6. 冻结方法、统计计划和论文主结果。

## 12. 不可违反的安全边界

- 系统用于研究和受监督原型，不是医疗器械或自主治疗产品。
- Critical stop、human alert、audit trace 和 action authorization 属于 Harness，不允许 skill/model 覆盖。
- 不允许模型自行调整药物或建议危险 exposure。
- 不把 ego-dystonic intrusive thought 自动当作真实伤害 intent。
- Runtime trace 可能包含敏感对话，必须有访问控制、加密、retention 和 redaction policy。
- 不把 patient-identifiable 或未经授权的临床材料提交到公共仓库。
