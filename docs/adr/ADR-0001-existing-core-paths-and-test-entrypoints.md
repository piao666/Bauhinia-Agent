# ADR-0001：现有核心路径与测试入口盘点

- 状态：已采纳
- 日期：2026-08-01
- 关联任务：P0-003

## 背景

Bauhinia-Agent Evo 必须扩展既有 Agent Loop、会话、上下文、工具和权限边界，不能新建平行核心系统。开始 Evo 领域开发前，需要确认当前代码的实际入口、事实存储、主要数据流和可运行测试。

## 决策

后续 Evo 功能复用下列边界；应用装配继续集中在 `bauhinia_agent.app.factory.create_bauhinia_agent_app`，领域代码不得直接依赖 Textual 或特定 Provider。

| 能力 | 当前入口与职责 | 主要数据流/存储 | 测试入口 | Evo 复用策略 |
| --- | --- | --- | --- | --- |
| CLI 与装配 | `cli.py:main`、`app/factory.py:create_bauhinia_agent_app` | CLI 创建 App；默认数据根为 `<project>/.bauhinia-agent` | `test_cli.py`、`test_app_factory.py` | 在 factory 注入 Evo 应用服务，不新增 CLI 或全局单例。 |
| Agent Loop | `agent/loop.py:AgentLoop`、`agent/session.py:AgentSession` | 用户消息追加到会话，`ContextBuilder` 投影后调用 Provider；工具结果再追加并继续本轮 | `test_agent_e2e.py`、`test_agent_tool_flow.py`、`test_agent_context_loop.py`、`test_agent_loop_limits.py` | 通过现有 Loop 的事件与服务边界接入 Plan/Evidence/Replan，不创建第二个 Loop。 |
| Session | `session/bootstrap.py:SessionBootstrap`、`session/new.py`、`session/resume.py:ResumeService` | `context/store.py:JsonlSessionStore` 与 `context/writer.py:SessionEventWriter` 保存并重建会话事实 | `test_session_*.py` | 旧会话仍由现有 Session Store 读取；Evo 事件使用独立、可关联的 append-only 事实源。 |
| Working Context（L1-L4） | `context/context_builder.py:ContextBuilder`、`context/manager.py:ContextWindowManager`、`context/compaction.py:CompactionPipeline` | 从 SessionView 投影 Provider 请求；checkpoint、压缩和归档不改写原始消息 | `test_context_*.py` | 长期记忆以有预算、可解释的检索结果进入既有 ContextBuilder，不以无限 Prompt 拼接替代它。 |
| Planning | `planning/models.py:TaskPlan`、`planning/service.py:TaskPlanService`、`planning/reducer.py`、`planning/projection.py` | session-scoped task 工具经 `TaskPlanService` 写入现有事件流 | `test_task_plan_*.py` | 扩展为可版本化 PlanGraph/Decision Record；保留既有任务计划和工具契约。 |
| Sub-agent | `agent/subagent.py:SubagentRunner`、`agent/worktree.py`、`tools/delegate.py` | delegate 创建隔离子会话/可选 worktree，并使用子权限管理器与 AgentLoop | `test_delegate_tool.py`、`test_worktree.py` | 在父 Run 关联子 Run；不将子代理输出直接当作长期事实。 |
| Tools | `tools/registry.py:ToolRegistry`、`tools/builtin.py`、`tools/session_registry.py:create_session_tool_registry` | 会话级注册表注入 task、archive、skill 等工具；Loop 的 ToolExecutor 结算调用 | `test_tools.py`、`test_read_tools.py`、`test_execution_tools.py` | Capability 描述层包裹既有 Tool，不复制 Registry。 |
| Permissions | `permissions/manager.py:PermissionManager`、`permissions/policy.py:DefaultPermissionPolicy`、`tools/permission_registry.py` | 所有会话工具可经 PermissionAwareToolRegistry 确认、授予或拒绝 | `test_permissions_*.py`、`test_permission_registry.py` | Evo Candidate、记忆和 Self Model 只能建议，不能绕过 PermissionManager。 |
| MCP | `mcp/config.py`、`mcp/manager.py:McpManager`、`mcp/adapter.py` | factory 连接 MCP，`McpToolProvider` 将发现的工具适配并命名空间化 | `test_mcp_*.py` | 保留原始 MCP 诊断；在 Capability 层标记来源与 Effect。 |
| Skills | `skills/discovery.py`、`skills/loader.py:SkillLoader`、`skills/catalog.py` | factory 发现项目 Skills；`load_skill` 由会话级注册表提供 | `test_skill_discovery.py`、`test_skill_loader.py`、`test_agent_skill_flow.py` | Candidate Skill 先作为版本化草案和 Shadow 输入，不直接替换加载机制。 |
| Providers | `providers/base.py`、`providers/factory.py:create_provider`、`providers/openai_compatible.py` | factory 根据配置创建 Provider；Loop 只依赖 ChatProvider 契约 | `test_providers.py`、`test_provider_errors.py`、`test_model_request_options.py` | Evo 领域不依赖某个厂商；Provider 错误被转化为可分类的执行证据。 |
| TUI 与运行时 | `app/tui.py:BauhiniaAgentApp`、`app/runtime.py:AgentChatRunner`、`app/router.py` | TUI 将命令路由到服务；Runner 创建或恢复 AgentLoop，UI 不直接写存储 | `test_app_tui.py`、`test_app_runtime.py`、`test_app_*_commands.py` | Evo TUI 复用命令、运行时、权限和事件服务；不让 UI 直写 Evo 存储。 |

## 当前主数据流

```text
CLI / Textual TUI
  -> create_bauhinia_agent_app
  -> SessionBootstrap + JsonlSessionStore + session ToolRegistry
  -> AgentChatRunner
  -> AgentLoop
  -> ContextBuilder / ContextWindowManager
  -> ChatProvider
  -> ToolExecutor -> PermissionAwareToolRegistry -> Tool / MCP / delegate
  -> SessionEventWriter -> JSONL session facts -> rebuilt SessionView
```

MCP 工具由 `McpManager` 经 `McpToolProvider` 适配到工具集；Skill 由 discovery/catalog/loader 通过会话工具加载；delegate 使用隔离子会话和现有 AgentLoop。它们均不应直接写入未来 Evo 的领域存储。

## 验证

以下命令验证跨越入口、Loop、Context、Planning、Session、权限、MCP、Skills、Provider 与 TUI 的现有回归面：

```powershell
python -m pytest -q tests/test_agent_e2e.py tests/test_agent_tool_flow.py tests/test_context_window_manager.py tests/test_task_plan_service.py tests/test_session_resume_service.py tests/test_permissions_manager.py tests/test_mcp_transport.py tests/test_skill_loader.py tests/test_providers.py tests/test_app_tui.py
python -m bauhinia_agent --help
```

2026-08-01 的执行结果为 260 passed、17 failed。17 个失败均来自 TUI 断言仍期待旧品牌 `BauhiniaAgent`，或紧凑欢迎页仍显示 `firstcoder`；该独立的用户可见品牌问题由 P0-008 修复。Windows 默认临时目录若不可访问，pytest 会在 fixture 建立阶段报 `WinError 5`；这是环境前置条件，应先修复 Temp 权限或在 CI 中配置可写的隔离临时目录，不能被误判为产品行为回归。

## 后果

- P1 之前不得实现 Evo 领域代码或变更既有 Agent Loop 的行为。
- P1 的 Event Store、P2 的 PlanGraph、P3 的 Memory 等能力必须以本 ADR 的既有边界为集成点。
- 后续架构变化应新增 ADR，而不是回写或修改本盘点的历史事实。
