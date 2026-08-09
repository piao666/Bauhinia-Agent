# ADR-005：离线 Eval Corpus 与 held-out 防污染治理

- 状态：Accepted
- 日期：2026-08-08
- 决策范围：D-004、P8 Eval Corpus、held-out 比较

## 背景

P8 必须用独立任务证明 Candidate 相比基线真实改善。若经验来源、开发调试样本和 held-out Case 混用，或 Candidate 能读取参考答案，评测结果会产生数据泄漏和奖励投机，不能作为晋升证据。Corpus 还必须满足本地优先、许可可审计、版本可复现和历史 Trial 可重放。

## 决策

1. 每个 Corpus 使用不可变的 `corpus_id + version` 标识，并记录 Schema 版本、SPDX 许可、来源、创建时间、Case 清单哈希和内容哈希。更新只能创建新版本，不得覆盖旧版本。
2. 仅允许仓库原创、用户明确授权或许可白名单内的离线数据进入 Corpus。默认白名单为 Apache-2.0、MIT、BSD-2-Clause、BSD-3-Clause、CC0-1.0 和 Public-Domain；其他许可必须人工批准并记录理由。
3. Case 明确属于 `source`、`development` 或 `held_out` split。同一个稳定 Case 指纹不得跨 split 重复；同一 Corpus 版本内 Case ID 和任务输入哈希必须唯一。
4. 持久化事件只保存可审计元数据和哈希。私有参考答案由 Evaluator 持有，不进入 Candidate 输入、Prompt、Artifact、Trial 公共视图或 Evo Event。
5. held-out 审计比较 Candidate 的来源 Run/Evidence、Case 的来源 Run/Evidence、任务输入哈希和访问资源哈希。任一重合、参考答案哈希被访问、split 不为 held-out、Manifest 哈希不一致或 Case 未注册时，Trial 标记为 `invalid`，不得计入晋升。
6. 同一比较固定 Corpus/Case 版本、任务输入、工作区基线、环境、模型配置、Variant、Evaluator 版本和随机种子。复跑产生新的 Trial/Run，但保持相同 trial key 并递增 attempt。
7. Corpus 内容建议存放在 `evals/corpus/<corpus-id>/<version>/`；Manifest 使用确定性 JSON，私有答案与大对象使用受访问控制的内容寻址附件。P8 的领域实现不自动下载外部语料，不引入网络依赖。

## 被否决方案

1. 直接复用产生 Candidate 的 Run 作为 held-out：不独立，无法证明泛化。
2. 把参考答案放入 Trial 或 Candidate 输入：会直接泄漏评测真值。
3. 原地更新 Corpus：历史比较无法复现，哈希与许可也会失去审计意义。
4. 只按文件名或 Case ID 去重：重命名即可绕过污染检查。

## 验证要求

- Manifest、Case 和 trial key 均可确定性序列化并计算哈希。
- 未许可 Corpus、安全边界不明的更新、跨 split 重复和来源重合均被拒绝。
- Candidate 公开输入不包含私有参考答案或其明文。
- 污染 Trial 明确标记为 `invalid`，与任务失败和评测器失败区分。
- 历史 Corpus 版本、Trial 和对应标准 Run 保持可读取。

## 后果

- P8 可以离线、确定性地复跑 held-out 比较，不需要新增运行时网络能力。
- 数据更新成本增加，但晋升证据可追溯且不会因静默改题失效。
- Corpus 的真实第三方数据仍需逐项许可审查；本 ADR 不授权下载或再分发任何外部数据。
