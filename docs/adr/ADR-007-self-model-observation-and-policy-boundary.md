# ADR-007：Self Model Observation 与保守策略边界

- 状态：Accepted
- 日期：2026-08-09
- 关联：PRD FR-SELF、TASK P9-001/P9-002

## 背景

Self Model 需要展示特定项目、模型、环境和任务类型下的可靠性与不确定性，但统计投影不能成为新的事实源，也不能凭模型自评扩大权限。P5 已提供 Evidence/Outcome，P8 已提供固定且可复跑的 Evaluation Trial。

## 决策

1. 新增 append-only `SelfModelObservationRecorded` 事实事件，只接受带 Evidence 的 `OutcomeClassified` 或状态为 completed 的有效 `EvaluationTrialRecorded`。
2. Observation 显式保存项目、模型配置哈希、评测器版本、环境哈希、语言、仓库规模、任务类型、工具类别、风险、验证充分度和分项结果；同一项目内同一来源事件只计一次。
3. `SelfModelUpdated` 是可删除、可重建的画像快照。项目、模型、评测器和环境是不可省略的隔离边界；其他维度只有在 Selector 明确省略时才可聚合。
4. 有限样本成功率使用 Wilson 区间。少于 5 个样本只显示 `insufficient_data`，不输出擅长/不擅长的确定结论。
5. Policy Suggestion 是画像的确定性纯投影，只允许拆分任务、加强验证、保守模板、请求确认和降低并发。它不持有执行、网络、文件或权限接口，所有建议固定声明 `permission_effect=none`。

## 后果

- 用户可回到每个来源 Run、Evidence 或 Trial 检查画像依据，并可清空投影后重建。
- 不同项目、模型、评测器或环境不会静默合并。
- 建议可以关闭和回放，但不能直接改变计划、工具权限或真实副作用。
- P11 可在 TUI 中展示 Profile 和 Suggestion；无需复制统计或权限规则。

## 已否决方案

- 直接从模型自评文本计算能力：缺少独立证据。
- 只保存最新统计行：无法从原始事实重建，也会覆盖历史。
- 让 Self Model 自动修改 Permission Policy：违反最小权限和产品边界。
- 用单一综合能力分数：会掩盖成功率、验证质量、成本、时延与风险差异。
