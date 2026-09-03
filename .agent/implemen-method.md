# Digital Doctor 实现与迁移记录

## 1. 目标与最终结果

本轮工作覆盖三个连续目标：

1. 为 milestone / phase planner 增加可审计日志和健康检查；
2. 修改 milestone prompt，使回复完成当前阶段后能够自然进入下一阶段；
3. 将项目整体更名为 Digital Doctor，并把本地 LLaMA-Factory 训练工程整合到顶层 `train/`。

最终目录为：

```text
/home/local/PARTNERS/yz646/digital_doctor/
├── digital_doctor/                # 对话系统 Python 包
├── train/                         # LLaMA-Factory 训练工程
├── tests/
├── data/
├── runtime/
├── pageindex/
├── README.md
└── run.py
```

外层仓库和内部 Python 包均已使用 `digital_doctor` 名称。主会话类由旧名称改为
`DigitalDoctorSession`，`MilestoneSession` 仍作为面向 planner 语义的别名保留。

## 2. 实施前审计

实施前先检查了工作区、Git remote、旧名称引用、默认路径和本地训练工程：

- 工作区已有未提交修改，因此整个迁移过程保留这些修改，没有执行 reset、checkout 或清理操作；
- Git 仅配置一个旧 `origin` remote；
- 运行时包、测试、README、环境名和入口中存在旧 package 名称；
- 本地 LLaMA-Factory 位于 `/home/local/PARTNERS/yz646/LLaMA-Factory`；
- 该训练目录没有独立 `.git`，但包含 `.env.local`、缓存、117 MB 的 `saves/`
  checkpoint，以及项目定制的 OCD 数据集与实验脚本。

基于审计结果，采取“复制并整合训练源码、保留原目录作为备份”的方式，没有删除原始
LLaMA-Factory 目录。

## 3. Milestone 日志实现

### 3.1 日志目标

原系统已有通用 debug log 和 JSONL trace，但 planner 内部模型调用不可直接审计：无法仅从
日志判断 formulation 是否成功解析、phase 模型是否返回完整结果、哪一个状态发生了迁移，
以及 planner 是否处于一致状态。

为此，在 `digital_doctor/tracking/milestones.py` 中为 `MilestoneTracker` 增加可选
`event_writer`。会话初始化时将 `DigitalDoctorSession._trace` 注入 tracker，使 planner
事件写入现有 `milestone_trace.jsonl`，避免创建另一套重复日志系统。

### 3.2 新增事件

每个临床轮次会根据实际执行路径写入以下事件：

| 事件 | 记录内容 |
| --- | --- |
| `milestone_formulation_inference` | formulation prompt、模型原始输出、解析状态、错误和候选更新 |
| `milestone_formulation_updated` | 实际应用的字段更新、filled count 和完整 formulation snapshot |
| `milestone_phase_inference` | phase prompt、模型原始输出、返回 phase IDs、完整性和解析状态 |
| `milestone_phase_floor_applied` | 确定性 structured floor 覆盖了哪些模型判断以及原因 |
| `milestone_state_transition` | 更新前后状态、状态变化、原 phase 和目标 phase |
| `milestone_health` | 结构约束、模型输出完整性、当前 focus 和最终健康状态 |

prompt 和 raw output 在写入 trace 前限制长度，避免异常响应导致日志无限膨胀。

### 3.3 健康状态

`MilestoneTracker.health()` 对以下不变量进行检查：

- phase ID 不重复；
- phase status 属于合法枚举；
- 已解决 phase 不会出现在更早未解决 phase 之后；
- 最多只有一个 `active` 或 `blocked` phase；
- focus phase 必须等于最早未解决 phase；
- phase 模型必须返回可解析 JSON；
- phase 模型必须恰好覆盖系统期望的全部 phase IDs。

健康状态定义：

- `not_run`：尚未执行临床 phase inference；
- `healthy`：模型输出完整且状态机不变量全部成立；
- `degraded`：JSON 损坏、phase 缺失或状态结构不一致。

每轮 update 和 API snapshot 都暴露 `milestone_health`。CLI 状态行同时显示
`planner: healthy|degraded|not_run`。简洁日志 `milestone_debug.log` 每轮增加一行
`[milestone]` 摘要，包括当前目标、是否推进、输出完整性和状态变化数量。

## 4. Milestone prompt 与下一阶段引导

### 4.1 原问题

原 planner context 只包含当前优先 phase 的描述和一个通用“move forward”约束。路由器没有
拿到 phase context，因此可能选择与阶段不匹配的 response move；生成模型也不知道当前
phase 的退出标准或下一 phase 是什么，容易出现重复 assessment 或提前跳到 exposure。

### 4.2 Context 改造

`MilestoneTracker.render_context()` 现在明确提供：

- 当前 priority phase 及 status；
- 当前 phase goals；
- 当前 phase exit criteria；
- 当前已记录 evidence；
- 完成后进入的下一 phase；
- 下一 phase 的 opening goal；
- blocked reason；
- 结构化病例 formulation；
- 不得跳过未解决阶段的 transition rule。

### 4.3 Router 改造

`decide_route()` 新增可选 `milestone_context` 参数。`DigitalDoctorSession` 在 routing 前渲染
planner context 并交给 router。Router prompt 要求：

- Assessment 优先 `assess/clarify`；
- Formulation 优先 `formulate/assess`；
- ERP Buy-In 优先 `psychoeducation/build_buy_in`；
- 后续 action phase 仅在 readiness 允许时选择 `treatment_step`；
- 如果最新患者证据已经满足当前 exit criteria，则选择“简短巩固 + 自然桥接”的 move；
- 如果 exit criteria 尚未满足，则不得进入 next-phase preview。

### 4.4 Draft、polish 和 helper 改造

三个生成层统一使用 transition contract：

1. 优先处理当前 phase 最小的未满足条件；
2. 不重复收集 structured formulation 中已有的信息；
3. 有证据支持完成时，先简短巩固，再自然打开下一 phase；
4. 不向患者说出内部 phase/milestone 标签；
5. 不宣称没有证据的完成状态；
6. 不绕过 treatment-readiness 和 safety guardrail。

这种设计让 milestone 负责顺序与方向，但 response move 仍负责对话自然度，安全门仍拥有最终
输出控制权。

## 5. 项目重命名

### 5.1 目录变化

```text
/home/local/PARTNERS/yz646/ocd_agent
    -> /home/local/PARTNERS/yz646/digital_doctor

<repo>/ocd_agent
    -> <repo>/digital_doctor
```

### 5.2 代码变化

统一更新了：

- Python imports 和 mock patch targets；
- 根入口 `run.py`；
- 包内 CLI fallback imports；
- FastAPI/uvicorn 启动路径；
- tests；
- workspace cleanup 脚本；
- README 和 `.agent/goal.md`；
- Conda 环境声明 `name: digital_doctor`；
- Web UI 的后端启动提示；
- 历史 evaluation summary 中指向本仓库的绝对路径。

旧的本机 Conda 环境仍名为 `ocd_agent`，仅用于本轮回归测试；以后执行
`conda env create -f environment.yml` 会创建新的 `digital_doctor` 环境。

### 5.3 GitHub 解绑

执行了旧 `origin` remote 删除。当前仓库保留完整本地 Git 历史，但不再配置 fetch/push URL。
未删除 `.git`，因为用户只要求更换 GitHub，而不是丢弃本地提交历史。

训练框架自带的 LICENSE、CITATION 和上游 README 链接予以保留。这些属于第三方归属和使用
文档，不是旧 Digital Doctor 仓库的 remote 或项目链接，不应为了“解绑旧 GitHub”而删除。

## 6. LLaMA-Factory 迁移

### 6.1 迁移内容

从本地 LLaMA-Factory 工作区复制以下内容到 `train/`：

- `src/llamafactory/` 完整训练源码；
- `pyproject.toml`、`setup.py`、requirements 和 Makefile；
- `examples/`、DeepSpeed/Accelerate 配置；
- `scripts/`、tests 和 Docker 配置；
- `data/` 中的 OCD、style、UGA 和官方 demo 数据；
- `dev/scripts/` 中的 Digital Doctor 定制训练与评估脚本；
- LICENSE、CITATION 和上游 README。

迁移后的训练目录约 19 MB，共 368 个文件；框架版本可导入为 `0.9.4.dev0`。

### 6.2 明确排除

以下内容没有迁入：

- `.env`、`.env.*`：防止密钥进入论文仓库；
- `.git/`：防止形成嵌套仓库；
- `.github/`：避免带入上游 GitHub workflow 和 issue automation；
- `saves/`：排除 checkpoint、optimizer state、tokenizer 副本和生成输出；
- `__pycache__`、`.pytest_cache`、`.mypy_cache` 和 `.pyc`。

顶层 `.gitignore` 额外保护 `train/.env*`、`train/saves/`、`train/output/`、W&B/SwanLab
日志和 Python 缓存。

### 6.3 路径可移植性

原定制脚本硬编码 `/workspace` 和 `/workspace/LLaMA-Factory`。所有这些路径已经移除。

每个 shell 实验脚本现在：

```bash
TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$TRAIN_DIR"
```

数据和输出分别使用 `$TRAIN_DIR/data` 与 `$TRAIN_DIR/saves`。因此脚本可以从任何工作目录启动。
同时为脚本增加 shebang、`set -euo pipefail` 和可执行权限，并通过 `bash -n` 检查。

RAG 示例文档改用 `train/` 内相对路径。`train/dev/expose_api.py` 和运行时 helper API 的默认
模型路径统一为：

```text
train/saves/gpt-20b/full/sft_ocd_v2
```

仍可通过 `HELPER_MODEL_DIR` 覆盖。

## 7. 验证方法与结果

### 7.1 静态验证

执行：

```bash
python -m compileall -q digital_doctor tests train/src train/dev
bash -n train/dev/scripts/*/*.sh
python run.py --help
```

结果：

- 运行时代码、测试和训练源码全部可编译；
- 所有定制训练 shell 脚本语法通过；
- 根 CLI 可从新 package namespace 加载并显示参数帮助；
- 搜索不到旧仓库绝对路径、旧 package namespace、旧 session class 或旧项目 remote URL。

### 7.2 LLaMA-Factory 集成验证

从 `train/src` 导入成功：

```text
llamafactory version: 0.9.4.dev0
registered datasets: 111
OCD datasets: ocd_train, ocd_test, ocd_train_v2, ocd_test_v2
```

新增 repository-layout 测试会持续检查：

- `digital_doctor` namespace 存在；
- OCD 数据集注册项及文件存在；
- `train/` 不包含 `.git`、`.env.local` 或迁入的 `saves/`；
- 定制实验脚本不再包含 `/workspace` 并定义 `TRAIN_DIR`。

### 7.3 单元测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：`35 tests passed`。

覆盖内容包括：

- memory compaction 和 recall；
- treatment readiness 与 deterministic buffer；
- mood/risk、危机停止和人工告警；
- chat/analysis routing 与 response move；
- formulation 更新和 phase progression；
- milestone 正常输出为 `healthy`；
- phase JSON 损坏时报告 `degraded`；
- current-to-next phase transition；
- 新 package namespace 和训练目录布局。

## 8. 当前边界

- 本轮没有启动真实 GPU fine-tuning，也没有下载模型权重；训练集成验证覆盖源码导入、数据注册、
  路径、脚本语法和仓库布局。
- milestone 的真实线上表现仍受所选模型输出质量影响；代码已能通过 health event 清楚区分正常、
  不完整和损坏的 planner 输出。
- 训练数据包含临床对话内容，论文公开前仍需完成数据授权、去标识化和发布范围审查。
- 新 GitHub 仓库尚未绑定。创建新仓库后再添加新的 remote 即可，本地历史目前完整保留。
