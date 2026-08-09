# ADR-004：Skill Draft 与现有运行时 Skill 的兼容边界

- 状态：Accepted
- 日期：2026-08-08
- 决策范围：D-006、P7 Skill Draft Candidate、后续 P8 晋升与回滚

## 背景

P7 需要把重复且有证据支持的做法表达为版本化 Skill Draft，但现有运行时已经有完整的 Skill 发现、冲突解析、加载与审计路径：

- 项目级 Markdown Skill：`skills/*.md`。
- 项目级 Agent Skill：`.agents/skills/<name>/SKILL.md`。
- 全局 Agent Skill：`~/.agents/skills/<name>/SKILL.md`、`~/.codex/skills/<name>/SKILL.md` 与 `~/.bauhinia-agent/skills/<name>/SKILL.md`。
- `SkillDefinition` 负责名称、路径、来源、根目录、描述和触发条件；`SkillLoader` 读取完整 Markdown，并记录内容哈希、字节数和必读文件。

任何文件一旦写入上述发现目录，就可能进入运行时 Catalog。若 P7 在生成 Candidate 时直接创建 `SKILL.md`，未经过 Shadow、held-out 评测、人工门禁和权限确认的草案就可能影响真实任务，违反 PRD 的 Candidate 隔离规则。为 Draft 新建独立加载器，则会复制现有 Skill 系统。

## 决策

Skill Draft 在 Candidate 生命周期中是不可执行的版本化领域 Artifact，不是运行时 `SkillDefinition`，也不是可发现目录中的文件。

P7 的 Skill Draft Artifact 至少包含：

- `artifact_schema_version`、稳定 Artifact ID、Candidate ID 和 Artifact 版本。
- `name`、`description`、`triggers` 与完整 `instructions_markdown`。
- 声明式输入、输出、依赖、Effect、权限要求、适用范围、风险与失效条件。
- 来源 Run、Evidence、反例、内容哈希和敏感信息扫描结果。
- 生命周期状态及后续评测、审查和替代关系的引用。

Candidate Store 或 append-only Evo Event 保存该 Artifact；未晋升 Artifact 不得写入 `skills/`、`.agents/skills/` 或任何全局 Skill 根目录，也不得被 `discover_all_skills` 或 `SkillLoader` 消费。

后续 P8 只有在 Artifact 达到 `Validated` 或 `Promoted` 所需门禁、通过敏感信息检查并取得目标路径写入授权后，才可调用单向适配器物化为现有格式：

```text
<skill-name>/SKILL.md
```

物化文件使用现有发现器已经支持的 YAML frontmatter：

```yaml
---
name: <skill-name>
description: <short description>
triggers:
  - <trigger>
---
```

正文来自 `instructions_markdown`。P7 专属的 Evidence、生命周期、Effect、风险和评测元数据继续留在 Evo Candidate/Promotion 事件中，不塞入运行时 `SkillDefinition`，也不要求现有 Loader 理解这些领域字段。

物化目标必须显式选择项目级或全局作用域，并复用现有 Catalog 的名称冲突优先级。覆盖已有 Skill、写入全局目录或启用真实副作用均是独立受权限控制的操作；晋升本身不隐含文件写入授权。回滚通过版本化 Promotion 记录选择前一已物化版本或禁用当前版本，不能改写历史 Candidate 事实。

## 被否决方案

1. **生成 Candidate 时直接写入 `.agents/skills`**：会让未评测草案被运行时发现，破坏 Shadow 隔离。
2. **为 Candidate Skill 新建第二套发现器和 Loader**：复制现有 Skill 系统，并使运行时行为、审计和冲突规则分叉。
3. **把全部 Evo 元数据写入 `SKILL.md` frontmatter**：现有运行时不需要这些字段，也会把领域生命周期与可移植 Skill 内容耦合。
4. **自动覆盖同名 Skill**：绕过现有来源优先级、用户选择和文件写入权限。

## 验证要求

P7-001 的契约测试必须证明：

- Skill Draft 可确定性序列化、计算内容哈希并后向兼容读取。
- 未知 Effect 按高风险处理，敏感字段不能进入可导出最小元数据。
- 创建、审查或分析 Skill Draft 不会改变现有 Skill Catalog。
- 适配器在内存中生成的 `SKILL.md` 可被现有发现器和 `SkillLoader` 正常读取，名称、描述、触发条件、正文与必读文件保持一致。
- 未通过生命周期门禁、缺少写入授权、存在路径穿越或同名覆盖未确认时，物化被拒绝且不产生部分文件。

## 后果

- P7 复用现有 Skill 运行时，不创建第二套 Capability 或 Loader。
- Candidate 的生成、评测和审查不会默认改变真实任务行为。
- P8 仍需实现受权限约束的物化、版本注册、禁用与回滚；本 ADR 不宣称这些能力已经可用。
- 若未来扩展 Skill Bundle 或资产文件，应在内容寻址清单和导入安全审计下增加版本化附件，不改变“Candidate Store 与运行时发现目录隔离”的原则。
