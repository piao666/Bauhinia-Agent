# ADR-001：Evo 数据事实源与投影边界

- 状态：Accepted
- 日期：2026-08-01
- 适用阶段：P1
- 关联任务：D-001、P1-002、P1-003、P1-004

## 背景

Evo 需要保存可追溯的 Run、Plan、Evidence、Memory、Candidate 和 Evaluation
事件。原始事件必须可审计、可迁移、可在索引损坏后重建；查询和统计又不能让每次
操作都扫描全部历史文件。现有 `SessionEvent`/`JsonlSessionStore` 服务于旧会话
恢复，不能被悄然改造成 Evo 的第二个事实源。

## 决策

### 1. 原始事实源：canonical JSONL

Evo 原始事件保存于项目数据目录：

```text
.bauhinia-agent/
└── evo/
    ├── events.jsonl
    └── events.lock
```

每行是一个完整的 `EvoEvent.to_json()` 结果，编码为 UTF-8、无 BOM、紧凑 JSON、
稳定排序键，并以单个换行结束。事件 envelope 必须携带：

- `event_id`、`event_type`、`schema_version`；
- UTC `occurred_at`；
- 持久化后为正整数的全局 `sequence`；
- `refs` 中的 Run/Session/Plan/Node/Memory/Candidate 等关联 ID；
- 显式 payload 或可保留的未知 payload。

JSONL 选择理由：追加写入简单、跨平台可读、便于诊断和迁移，且与 P1-001 的
Golden JSON 契约直接一致。原始事件不在 SQLite 中另存一份作为真相。

### 2. SQLite 只作为派生投影

投影保存于：

```text
.bauhinia-agent/evo/projection.sqlite3
```

SQLite 可以包含事件目录、Run/Plan 等查询索引、统计和重建游标，但不拥有比
`events.jsonl` 更多的事实。投影中的每一行必须能回指 `event_id` 和 `sequence`；
需要展示未知事件时保留其 canonical JSON 或可诊断引用。

投影规则：

1. 追加事件成功后再推进投影游标；投影失败不得改变原始事件写入结果。
2. 删除 `projection.sqlite3` 后，按 `events.jsonl` 的 sequence 顺序可重建等价查询结果。
3. 投影更新使用 SQLite transaction；重建写入临时数据库后原子替换，不能原地覆盖事实源。
4. 投影缺失时自动重建或返回结构化诊断；投影损坏不能阻断旧 Session、CLI 或 TUI。
5. 统计、搜索和 UI 状态只能来自投影或即时重放，不能反向改写 JSONL。

### 3. 追加、序号与锁

Store 以独立 `events.lock` 作为跨进程互斥锁，使用已有 `portalocker` 依赖并通过
路径 API 打开。写入流程必须在同一独占锁内完成：

1. 读取并校验事件文件尾部及最后一个合法 sequence；
2. 为待写事件分配 `last_sequence + 1`，禁止调用方伪造或重用 sequence；
3. 写入一条 canonical JSONL；
4. flush、`fsync` 后释放锁；
5. 再以 SQLite transaction 应用投影。

`event_id` 全局唯一；重复 ID、回退 sequence 或同一 sequence 对应不同事件均为
结构化 Store 错误。读者在扫描时使用共享锁或读取稳定快照，不能绕过写锁读取半条
追加记录。锁超时必须可发现，不得静默丢事件。

### 4. 尾部损坏与恢复

正常读取遇到无效 JSON、错误 schema、重复 ID 或 sequence 间隙时，必须报告文件名、
行号、byte offset、event_id（如可解析）和原因；不能自动删除或改写历史事件。

仅显式调用恢复流程时，才允许处理文件最后一个不完整的追加尾部。恢复前保存原文件
元数据并生成诊断记录；非尾部损坏、完整 JSON 但语义无效、sequence 冲突和未知事件
不得被自动修复。未知事件/字段保持在原始 JSONL 中，读取器应尽可能展示而不是令整次
Run 无法打开。

### 5. 旧 Session 与 Evo 并存

Evo Store 只读写 `.bauhinia-agent/evo/`，不迁移、不覆盖、不改变现有
`.bauhinia-agent/sessions/` 和 `SessionEvent` 协议。缺少 Evo 目录表示“尚无 Evo
数据”，不是旧 Session 损坏；旧 Session 恢复不依赖 SQLite 或 Evo 投影。

## 不采用的方案

- **SQLite 作为唯一事实源**：查询方便，但会把审计事实和可重建投影混在一起，且不利于
  人工诊断、尾部恢复和未来格式迁移。
- **把 Evo 事件追加到现有 Session JSONL**：会污染旧 Session 协议，扩大恢复回归面，
  也无法清晰表达跨 Session/Run 的 Evo 关系。
- **只维护内存或 Prompt 状态**：不可恢复、不可审计，不满足 PRD 的事实源要求。
- **每个事件单独一个文件**：跨事件序号和原子追加更复杂，目录扫描成本也更高。

## 后果

正面影响：原始事实可审计和迁移，投影可删除重建，旧 Session 保持兼容，未知事件可
前向读取，Windows 文件锁和原子替换有明确实现位置。

代价：P1-002 需要实现尾部扫描、跨进程锁、`fsync`、sequence 分配、SQLite transaction
和投影重建；P1-003 需要补充 schema 迁移、显式尾部恢复和导入诊断。

## 实现映射与验收

- P1-001：`bauhinia_agent/evolution/events.py` 提供 canonical envelope、schema version、
  sequence、关系引用和未知字段保留。
- P1-002：实现 `bauhinia_agent/evolution/store.py` 与 projection rebuild，严格遵循本 ADR。
- P1-003：补充版本探测、旧数据降级读取、尾部诊断/恢复和跨 Windows 路径夹具。
- 验收：删除 SQLite 投影后可由 JSONL 重建；Store/Recorder 失败不改变 Agent 原执行结果；
  旧 Session、未知事件和部分损坏记录均有可发现的处理结果。
