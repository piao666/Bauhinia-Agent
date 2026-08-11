# ADR-008：Sub-agent 协作领域边界

- 状态：Accepted
- 日期：2026-08-09
- 关联：PRD FR-SUB、TASK D-007/P10-001/P10-002/P10-003

## 背景

仓库已有 `SubagentRunner`、`delegate` Tool、`BackgroundJobManager`、隔离 worktree 和统一 `PermissionManager`。P10 还需要六种协作角色、可追溯交接、资源冲突、父子 Run 聚合以及证据独立性。若为这些要求新建一套 Agent Loop、工具注册、权限或并发运行时，会造成二次真相源和权限绕过风险。

## 决策

1. 保留现有 `SubagentRunner` 作为唯一子 Agent 执行入口，保留 `BackgroundJobManager` 作为唯一本地并发/取消入口，保留现有权限与 worktree 隔离。
2. 扩展 Planning `TaskContract` 为 Planner、Researcher、Executor、Verifier、Critic、Curator 的统一领域契约，显式声明输入快照、能力、Effect、资源申请、预期证据、预算、最低置信度、deadline 和取消语义。
3. 新增纯领域协作聚合协议，用 append-only Evo Event 保存派发、结果、冲突与聚合事实；它不导入 Agent Loop、Tool Registry 或 Permission Engine。
4. 运行时适配层只将六种领域角色映射到已有 researcher/reviewer/tester/coder profile，并把已有运行结果转换为结构化协作结果。
5. 无证据、低于契约置信度、被取消/超时/拒绝的结果不得进入 Memory 或 Promotion。相同来源或复制的证据只形成一个独立支持组。

## 后果

- P10 可增加可审查协作语义，不改变现有工具权限和真实副作用路径。
- 角色能力映射是保守的；新角色不会自动获得新工具或更高权限。
- 冲突分支、子 Run 和证据去重具备 append-only 审计记录，Memory/Compiler 只消费通过门禁的聚合结果。

## 已否决方案

- 新建第二套 Multi-agent Loop/调度器：会复制取消、权限、Session 和工具语义。
- 只使用自由文本 prompt 交接：无法强制资源、证据、置信度与取消边界。
- 将相同结论数量当作独立验证数量：会对复制错误重复计票。
