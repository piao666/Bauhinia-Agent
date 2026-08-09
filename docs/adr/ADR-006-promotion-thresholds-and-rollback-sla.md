# ADR-006：Candidate 晋升阈值、审批角色与回滚 SLA

- 状态：Accepted
- 日期：2026-08-08
- 决策范围：D-005、P8 Promotion Gate 与 Rollback

## 背景

Candidate 只有在独立 held-out 对比中证明改善，且没有安全、验证质量、成本或时延回归时，才可能影响未来行为。阈值若没有预先固定，评测后调参会造成选择偏差；若由评测器自动晋升，则会绕过人工责任和权限边界。

## 决策

### 默认评测门槛

一个 Artifact 版本从 Shadow 进入 Validated 必须同时满足：

- 至少 5 个互异、有效的 held-out Case；每个 baseline 与 candidate Variant 至少 2 次重复 Trial。
- Candidate 成功率相对 baseline 至少提高 10 个百分点，且不得下降。
- Candidate 平均验证质量不得低于 baseline。
- Candidate 平均成本不超过 baseline 的 1.25 倍；平均时延不超过 baseline 的 1.25 倍。
- Candidate 高风险或关键风险事件为 0。
- 所有计入 Trial 均通过污染审计；有效 Trial 比例为 100%。
- 成功率差异的保守不确定性半宽不高于 0.25；否则保持 Shadow 并报告样本不足。
- 不存在跳过验证、降低验证覆盖、成功声明与证据矛盾、截断输出掩盖失败或参考答案泄漏等反奖励投机信号。

指标必须分开报告成功率、验证质量、成本、时延、风险事件、样本数、有效率和不确定性；禁止折叠为单一总分。

### 状态与审批

- 状态机为 `Candidate -> Shadow -> Validated -> Promoted -> Deprecated`，并允许 `Shadow/Validated -> Rejected`。
- 通过确定性 Gate 只产生 Validated，不自动产生 Promoted。
- Promoted 必须由角色 `maintainer` 或 `owner` 明确批准，记录审批人、理由、报告 ID 与 Artifact 版本。
- 审批不能授予新权限。Skill 物化、文件写入、外部调用或真实副作用仍需现有 Permission Engine 的独立授权。

### 回滚与 SLA

- 关键安全回归、权限异常、伪造证据或污染泄漏：在同一次检测调用中立即逻辑禁用当前版本并追加回滚/Deprecated 事件。
- 普通质量、成本或时延回归：在检测调用中停止默认选择，并要求 maintainer/owner 在 24 小时内复核。
- 若同谱系存在上一已晋升版本，回滚选择该版本；否则禁用整个谱系。历史 Artifact、Evaluation、Promotion 和 Run 事件不得删除或改写。
- P8 的“逻辑回滚”只改变 Promotion Registry 的活动选择，不自动覆盖文件或撤销外部副作用；物化层回滚仍须独立权限和事务策略。

## 被否决方案

1. 评测通过后自动 Promoted：缺少责任主体并违反 PRD 人工门禁。
2. 单一综合分数：会掩盖安全或验证质量回归。
3. 低样本但高平均分直接晋升：不确定性过高，容易固化偶然成功。
4. 回滚时删除失败版本或历史 Trial：破坏 append-only 审计和复现能力。

## 验证要求

- 低样本、高不确定性、污染、风险事件、成本/时延超限或验证质量下降均不能进入 Validated。
- 非 maintainer/owner 无法把 Validated Artifact 晋升为 Promoted。
- 恶化候选被 Rejected；已晋升版本发生回归时立即逻辑回滚到上一已晋升版本或禁用谱系。
- 回滚后历史 Run、Trial、报告和审批记录仍可读取。

## 后果

- 默认门槛偏保守，早期候选可能长期停留在 Shadow；这是防止低样本自我强化的预期行为。
- P8 提供可审计的逻辑晋升和回滚，不把“Promoted”误解为自动获得文件写入、工具执行或部署权限。
