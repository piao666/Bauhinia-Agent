# Self Model 开发者说明

Self Model 是项目级、证据驱动的可靠性画像，不是权限引擎。它从 P5 Outcome 或 P8 Evaluation Trial 记录 Observation，再按显式 Selector 构建 Profile。

## 最小流程

```python
from bauhinia_agent.self_model import (
    PolicySuggestionEngine,
    ProfileSelector,
    SelfModelService,
    TaskClassification,
)

service = SelfModelService(store=evo_store, project_id="project_a")
classification = TaskClassification(
    project_id="project_a",
    model_config_hash=model_hash,
    evaluator_version="eval-v1",
    environment_hash=environment_hash,
    language="python",
    repository_scale="medium",
    task_type="bugfix",
    tool_category="pytest",
    risk_level="low",
)
service.record_observation(classification, source_event_id=outcome_or_trial_event_id)

selector = ProfileSelector(**classification.to_dict(), verification_level="strong")
published = service.publish_profile(selector)
suggestions = PolicySuggestionEngine().suggest(published.profile)
```

## 解释规则

- 项目、模型配置、评测器版本和环境哈希必须精确匹配，不会隐式合并。
- Selector 中语言、仓库规模、任务类型、工具类别、风险或验证充分度为 `None` 时，表示调用方明确请求跨该维度聚合。
- 少于 5 个样本时状态为 `insufficient_data`。
- `reliable`、`mixed`、`unreliable` 由成功率 Wilson 区间保守判定；Profile 同时保留成本、时延、验证质量、风险事件和失败类别，不提供单一能力总分。
- Policy Suggestion 可通过 `set_enabled(False)` 关闭；建议不会执行工具、写文件、开启网络或授予权限。

## 数据与兼容性

- `SelfModelObservationRecorded` 是 append-only 事实。
- `SelfModelUpdated` 是可重建快照，包含 Profile schema、Selector、样本窗口和来源引用。
- 未知事件和未来字段继续遵循 Evo Event 的兼容读取规则。
