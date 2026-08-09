# ADR-002：Decision Record 最小字段与隐私边界

- 状态：Accepted
- 日期：2026-08-02
- 决策范围：P2 的 PlanGraph、Decision Record 与 Replan 事件

## 背景

P2 需要让关键工程决策可审查、可回链到证据，同时不能把模型的完整私有推理、原始敏感提示或凭据变成持久化产品数据。

## 决策

Decision Record 仅持久化以下结构化摘要：`subgoal`、`evidence_refs`、`assumptions`、`options_considered`、`selected_action`、`rationale_summary`、`confidence`、`expected_observation`、`verification_method`、`outcome` 与 `next_decision`，并关联 `decision_id`、`plan_id`、`node_id`。

`rationale_summary` 是面向审查者的短摘要，最大 2,000 个字符；模型私有推理字段、未裁剪的 Provider reasoning、Prompt 原文、工具环境全文和凭据字段都不属于该 Schema，未知字段不会被静默接收为 Decision Record。

Replan 以 P1 的 `DecisionRecorded` 与 `PlanNodeUpdated` 追加事件表示：前者保存触发证据、候选安全结果和选择，后者保存新图版本、节点快照和触发/结果元数据。原始事件仍是事实源，SQLite 仍只是投影。

## 后果

- 用户可查看工程上有用的选择依据，而不能将此机制误解为完整 CoT 浏览器。
- 后续 P5 Evidence Adapter 和 P13 脱敏层必须在写入 Decision Record 前提供引用或脱敏摘要，而不是传递原始敏感文本。
- P2 的执行协调器只接收现有 PermissionManager/ToolExecutor 的回调；它不能绕过权限或成为第二套 Agent Loop。
