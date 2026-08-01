# Bauhinia-Agent Evo Agent 工作规范

> 适用范围：本仓库根目录及全部子目录
>
> 文档属性：项目治理文档，纳入版本控制
>
> 产品依据：Bauhinia_agent_PRD.md
>
> 任务依据：TASK.md

## 1. 使命与产品边界

本仓库的二次开发目标是把 Bauhinia-Agent 演进为 Bauhinia-Agent Evo：一个面向软件工程的、自我改进但受证据与安全门禁约束的 Coding Agent。

它的核心闭环是：

Goal -> Retrieve -> Plan -> Act -> Verify -> Diagnose/Reflect -> Compile -> Evaluate -> Promote/Rollback

核心能力：

- 可观察的分层 Planning 与安全 Replanning。
- 可审查的多步决策记录，而不是完整隐藏思维链。
- 有来源、作用域、置信度、时效和冲突治理的短期与长期记忆。
- 基于验证结果的 Experience Compiler。
- 版本化的 Skill、计划模板和策略候选。
- 用 held-out 评测、影子执行和人工门禁控制的演化。
- 透明的 Self Model，用于保守地增加规划或验证力度。
- 与现有 TUI、Tools、MCP、Skills、Session、Permission 和 Sub-agent 复用同一核心。

不要把项目改造成通用聊天助手、Deep Research Agent、渠道机器人、训练平台、无限制自修改系统或单纯代码生成器。每个功能都应回答：它是否让 Agent 更能计划、执行、验证、从证据中学习，且让改进本身可检查、可否决、可回滚？

MVP 明确不做：

- 模型权重训练或自动微调。
- 自动修改 Agent 核心循环、权限引擎、安全策略或生产配置。
- 未经评测与审批的自动晋升。
- 保存、索取、伪造或向用户暴露完整模型私有 CoT。
- 以 UI 内存、Prompt 文本或单次模型自评充当事实源。

## 2. 指令优先级与文档状态

执行顺序：

1. 当前用户的明确要求。
2. 系统与运行环境安全约束。
3. 本 AGENTS.md。
4. Bauhinia_agent_PRD.md 的产品边界与验收要求。
5. TASK.md 的任务顺序、状态与 Gate。
6. 仓库现有代码、测试和公开文档体现的约定。

如果这些内容冲突，不要静默选择。先收集代码证据，说明冲突、影响和建议，再请求必要决策。

如存在历史产品文档，只能作为迁移参考，不能覆盖或绕开 Bauhinia-Agent Evo PRD 的产品边界。

## 3. 项目治理文档

以下文件是版本化的项目治理文档：

- Bauhinia_agent_PRD.md
- TASK.md
- AGENTS.md

强制规则：

- 三份文档与代码一同版本控制；修改必须与当前实现范围一致，并在提交前复核。
- 不得用 .gitignore 或 .git/info/exclude 排除这三份文档。
- 每次提交前运行 git status --short 和 git diff --cached，确认其变更是有意的、范围明确且不含敏感信息或完整私有 CoT。
- 历史参考资料必须明确标识其状态，不能通过复制、改名、压缩包、生成物或文档引用绕开 PRD 的边界。

## 4. 开始、执行与完成任务

每次开始开发必须：

1. 完整阅读本文件。
2. 阅读 Bauhinia_agent_PRD.md 中与任务相关的需求、非目标和验收场景。
3. 阅读 TASK.md 中对应阶段、依赖、测试与 Gate。
4. 检查 git status --short --branch、当前分支、远端、相关目录、相关测试与用户未提交变更。
5. 在 TASK.md 中只将一个最小可交付任务标记为 [~]。
6. 用一句话说明范围、风险和验证方法。

执行中：

- 只修改当前任务所需内容；不要顺手格式化、重构或清理无关文件。
- 优先搜索已有接口、工厂、端口、事件和测试，再增加新模块。
- 发现邻近问题时记录为新任务，不混入当前提交。
- 以测试、可复现步骤、Trace、日志、Diff 或评测数据作为完成证据。
- 不能把代码看起来正确视为已验证。

完成时：

- 覆盖正常、失败、拒绝、取消和超时路径，范围与风险相称。
- 更新 TASK.md 中的状态、证据和后续发现。
- 检查 Trace、隐私、记忆和副作用影响。
- 在没有用户授权时，不提交、不推送、不创建 PR。

## 5. 现有架构边界

优先复用已有模块职责：

- bauhinia_agent/agent：Agent Loop、Session 交互与子 Agent 行为。
- bauhinia_agent/context：事实、上下文投影、预算和 L1-L4 压缩。
- bauhinia_agent/session：会话生命周期、索引、恢复和分叉。
- bauhinia_agent/tools：内置工具与 Tool Registry。
- bauhinia_agent/permissions：权限策略与确认。
- bauhinia_agent/providers：模型 Provider 适配。
- bauhinia_agent/mcp：MCP 连接与工具适配。
- bauhinia_agent/skills：Skills 发现与加载。
- bauhinia_agent/runtime：运行时与进程相关能力。
- bauhinia_agent/app：TUI、命令和应用装配。

新增 Evo 能力优先采用以下边界：

- bauhinia_agent/evolution：事件、Experience Compiler、Candidate 生命周期、Promotion。
- bauhinia_agent/memory：长期记忆领域、检索、作用域、冲突和索引投影。
- bauhinia_agent/evaluation：Case、Variant、Trial、Evaluator、保留集和报告。
- bauhinia_agent/planning：扩展现有计划能力，包括 PlanGraph、Decision Record、Replan。
- bauhinia_agent/self_model：能力画像与策略建议；不能成为权限层。

强制架构规则：

- 不创建第二套 Agent Loop、Tool Registry、Permission Engine、Session 真相源或 Context 压缩系统。
- 不把长期记忆实现为无约束 Prompt 拼接。
- 不让 UI、Provider、Tool 进程或 MCP 直接写 Evo 领域存储。
- 不让领域层依赖 Textual、React、FastAPI 或特定模型厂商。
- 应用装配通过现有工厂或 Bootstrap 集中完成，避免全局单例。
- 兼容旧会话、CLI 和 TUI 是默认要求。

## 6. Planning、Reasoning 与执行约束

PlanGraph 的节点至少应能表达：目标、依赖、前置条件、风险、预算、验证条件、状态、重试与关联证据。计划变化必须是可追溯版本，而非原地覆盖。

Decision Record 只保存对用户和系统有用的结构化决策摘要，例如目标、候选方案、选择、依据、假设、不确定性和预期证据。禁止把完整隐藏思维链当作持久化需求、评测真值或 UI 功能。

以下情况必须产生 Replan 事件或安全终止语义：

- 验证失败。
- 工具错误、环境错误或不可用。
- 权限拒绝。
- 前提或工作区状态不成立。
- 上下文冲突或记忆冲突。
- Token、时间或成本预算耗尽。
- 用户改变目标或取消任务。

Planner、Executor、Verifier、Critic 等角色只通过明确任务契约交接；没有证据的结论不可作为事实、记忆或晋升依据。

## 7. 事件、存储与数据契约

- 原始 Evo Event 是 append-only 事实源。
- SQLite、向量索引、搜索、统计、Self Model 和 UI 状态都是可重建投影。
- 投影不得反向覆盖或悄然修正事实；需要纠正时追加更正或失效事件。
- 每个持久化事件必须有 schema version、UTC 时间、稳定 ID、序号和关联关系。
- 稳定协议使用显式模型；禁止长期依赖未约束的字典。
- 新字段优先后向兼容；删除或改义必须提供迁移或降级读取。
- 未知事件或字段应保留和可展示，不能令整个 Run 无法打开。
- 关键元数据使用原子写入；事件写入应有尾部损坏诊断与恢复策略。
- 默认数据目录为项目 .bauhinia-agent；必须用跨平台路径 API，不能硬编码分隔符。
- 大对象使用引用、附件或内容寻址，不在多条事件中复制。

Recorder、Store 或投影失败不得改变 Agent 的原执行结果，但必须提供可发现的结构化诊断。

## 8. 记忆规则

记忆必须区分：

- Working/Context：现有会话上下文与 L1-L4 压缩。
- Task/Session：当前任务或会话内的可恢复事实。
- Episodic：已完成 Run 的事件性经验。
- Semantic：稳定项目事实、用户决策、惯例和已验证结论。
- Procedural：可复用的策略、计划模板、Skill 或反模式。
- Meta-memory：系统对经验可靠性、适用边界和失效风险的摘要。

每条长期记忆必须具备来源、时间、作用域、置信度、状态、时效或失效条件、敏感级别和冲突关系。推断、用户确认和验证事实必须明确区分。

强制规则：

- 默认按项目隔离；跨项目、跨用户或跨组织复用必须显式授权。
- 无来源、无证据、无作用域或仅由单次自评得到的内容不得直接进入长期记忆。
- 冲突与过期记忆必须降权、失效、形成待审查提案或请求确认；禁止静默拼接。
- 删除和失效不篡改历史 Run。
- 检索结果必须可说明为什么命中、来自何处、被截断了什么以及占用了多少预算。
- 记忆索引可重建，且无向量依赖时必须有确定性降级方案。

## 9. Experience Compiler 与自我演化

Experience Compiler 的输入是计划、决策摘要、工具与验证证据、结果分类和环境摘要。它的输出是候选经验，不是立即生效的规则。

每个 Candidate 必须含有：

- 类型：计划模板、Skill Draft、调用策略、Memory Rule 或反模式。
- 来源 Run 与 Evidence 引用。
- 适用条件、作用域、反例、风险、Effect、依赖和置信度。
- 版本、状态、创建时间、评测结果和审查记录。

生命周期严格为：

Candidate -> Shadow -> Validated -> Promoted -> Deprecated 或 Rejected

强制规则：

- 单次成功、单个模型输出或无独立验证的结论不能自动晋升。
- 先在 Shadow 或受控试验中比较，再在 held-out 集上评测。
- 同一比较必须固定任务输入、工作区基线、评测器版本与环境，并允许重复 Trial。
- 结论必须分开报告成功率、验证质量、成本、时延、风险事件、样本数和不确定性；禁止只用单一综合分数。
- 评测失败与任务失败必须区分。
- 支持按版本禁用、拒绝、降级和回滚，历史 Run 必须仍可读取。

绝对禁止让 Candidate 自动修改模型权重、核心 Agent Loop、权限策略、安全规则、生产部署、外部账号或未授权源码。任何真实副作用仍然由现有权限系统独立决定。

## 10. Self Model 约束

Self Model 用于表达系统在特定语言、任务类型、工具、仓库特征和风险等级下的可靠性与不确定性。

- 必须展示样本量、时间窗、来源、适用范围和不确定性。
- 低样本应输出不足以判断，不得伪造能力结论。
- 它可建议拆分任务、加强验证、选择保守模板、询问用户或降低并发。
- 它不得提升权限、绕过确认、扩大网络/文件/执行范围或替代用户决策。
- 不同项目、不同模型配置和不同评测器的统计不能未经声明地合并。

## 11. Tools、MCP、Skills 与 Sub-agent

- 内置 Tool、MCP Tool、Skill、Candidate Skill 与 Sub-agent 能力应通过统一 Capability 描述层展示，但保留来源、协议与信任差异。
- Skill 不是普通 Tool；尊重其触发条件、内容、依赖和加载语义。
- MCP 应保留原始协议诊断与给 Agent 的规范化结果；Schema 错误必须定位到字段路径。
- Tool 必须标记 read、write、execute、network、external 等 Effect；未知 Effect 按高风险处理。
- Tool 名称冲突必须命名空间化或报错，禁止静默覆盖。
- 远程 MCP、导入 Bundle、外部 Skill、子 Agent 输出和用户脚本默认不可信。
- 子 Agent 的任务契约须声明目标、输入快照、可用能力、预算、输出证据、置信度和取消语义。
- 并发执行须限制资源和写入竞争；父 Run 必须能追溯子 Run。
- 没有独立证据的多个相似子 Agent 结论不得被重复计票。

## 12. 权限、副作用、隐私与安全

权限系统是代码层强制机制，不能被 Prompt、模型输出、Candidate、记忆或 Self Model 绕过。

- 真实操作前必须经过现有 Permission Engine。
- 文件写入、Shell、Git、网络、外部系统和未知 Effect 都需要明确风险语义。
- 可写或试验性执行应使用隔离工作区、worktree 或临时目录。
- 取消、超时或失败后须回收子进程、临时目录和工作树。
- 不自动开启远程监听；本地 Web 服务仅监听 127.0.0.1 或 localhost。
- API Key、Token、Cookie、认证头、私钥和完整敏感环境变量不得落盘。
- 脱敏在事件和记忆持久化边界完成；界面遮罩不是替代方案。
- 导入或导出前验证路径穿越、压缩炸弹、敏感信息和自动脚本执行风险。
- Shell 记录 cwd、退出码和脱敏后的摘要；环境仅记录白名单摘要。

## 13. TUI 与 GUI

TUI 是 MVP 的主交互面，沿用 Textual 和现有命令体系。它应适合摘要、审查、控制和状态，不必复制复杂可视化。

TUI 必须提供计划、证据、记忆来源、Candidate 状态、评测结果、Self Model 摘要和安全禁用入口；同时保证键盘操作、小终端和无颜色环境可用。

GUI 是后续扩展：

- 与 TUI 共享 Evo 应用服务、Run 状态、权限和事件。
- GUI 不能直接操作 JSONL、SQLite、Provider 或工具进程。
- React 不复制 Python 领域规则；API 应经 Schema 校验。
- 大量事件使用虚拟列表和增量查询。
- 页面须有 Loading、Empty、Error、Disconnected 状态，且重要状态不能只依赖颜色。
- GUI 关闭不应取消正在运行的任务。

## 14. 工程与测试规范

Python：

- 支持 Python 3.11 及项目声明的更高版本。
- 公共接口必须有类型标注；跨边界数据使用显式校验。
- 保持函数单一职责；不要把存储、界面或演化逻辑堆入 Agent Loop。
- 仅在外部边界捕获宽泛异常，保留原因链和结构化诊断。
- 异步路径不得执行长时间阻塞 I/O；资源要有明确 close 或取消协议。
- 时间以带时区 UTC 存储；序列化保持确定性。
- 新增依赖须说明必要性、许可证、体积和可选安装策略。

TypeScript/React（仅 GUI 阶段）：

- 启用严格 TypeScript；禁止用 any 绕过核心事件和 API 类型。
- 服务端数据在边界校验；Server State 与 UI State 分离。
- 实时事件按 Run ID 与 sequence 幂等合并。
- 复杂时间线和图计算放入可测试纯函数或 Worker。

测试：

- 核心路径同时需要单元、契约、集成和评测层覆盖。
- 事件、Memory、Plan、Candidate、Promotion、API、MCP 和 Bundle 要有契约测试。
- 用 Golden Test 固定稳定事件序列、Context Pack 和导出物。
- Eval 使用版本化 Case、Variant、Trial 和 held-out 集；外部 Provider 测试与默认离线测试分组。
- 安全测试覆盖脱敏、作用域越权、权限绕过、恶意输入、未知 Effect 和导入风险。
- 涉及时间、并发和随机性的测试必须可控、可重复。
- Windows 专属行为应有自动测试或明确人工验收记录。
- 不得通过降低断言、跳过测试或延长无限超时解决失败。

## 15. Git、文档与交付

- 默认分支名使用 codex/topic，除非用户另有要求。
- 提交只包含单一逻辑主题；提交前检查 diff、测试结果和本地文件排除状态。
- 不使用破坏性 Git 命令清理用户变更。
- 未经用户授权不提交、不推送、不创建 PR。
- 用户授权提交或推送时，只处理明确范围，不捎带无关文件。
- 公共行为变化必须更新用户或开发者文档。
- 重大架构决定写 ADR，包含背景、选项、决策和后果。
- 事件、Memory、Candidate、Eval 和 Bundle 格式都应有版本和最小示例。
- 不声称尚未实现的功能已经可用；规划、实验和生产能力必须明确区分。

## 16. 完成定义

一个任务只有同时满足以下条件才可标记完成：

- 实现满足关联 Evo PRD 需求与非目标。
- 未复制既有核心系统或破坏模块边界。
- 正常、错误、拒绝、取消和超时路径按风险验证。
- 计划、事件、记忆、隐私和副作用影响已检查。
- 相关测试、评测或人工验收有实际命令和结果证据。
- 数据 Schema、迁移和兼容性影响已处理或明确记录。
- 用户可见行为有文档或可复现验收步骤。
- git diff 不含无关改动。
- Bauhinia_agent_PRD.md、TASK.md 与 AGENTS.md 的版本化变更已经复核，且仅包含当前任务所需内容。

## 17. 明确禁止

- 禁止以自我反思文本替代验证证据。
- 禁止持久化或要求完整隐藏 CoT。
- 禁止未经 held-out 评测与审批自动晋升 Candidate。
- 禁止让演化功能自行升级权限、执行真实副作用或修改核心安全逻辑。
- 禁止将临时对话、单次成功或跨项目噪声静默写入长期记忆。
- 禁止直接篡改历史 Event 或 Run 来修正显示、统计或评测。
- 禁止为了 GUI 复制 Agent 业务逻辑。
- 禁止默认记录真实密钥、Cookie、私钥或敏感环境变量。
- 禁止无迁移策略修改持久化 Schema。
- 禁止用单一评分掩盖成功、成本、时延和安全差异。
- 禁止无测试证据地宣称跨平台、可回滚或安全。
- 禁止未经授权提交、推送、发布或操作生产环境。
