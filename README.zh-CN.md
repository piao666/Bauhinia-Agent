# Bauhinia-Agent

Bauhinia-Agent 是一个面向软件工程的本地 Coding Agent。它将规划、工具执行、验证和会话上下文结合起来，并始终保留明确的权限边界。

## 开发

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
bauhinia-agent
```

项目正进入 Evo 实现阶段：计划、执行、验证、编译带证据的经验，并仅在明确门禁下评测、晋升或回滚。

## Provider 支持

运行时支持 OpenAI Chat Completions-compatible Provider 与 Anthropic Messages API，包含工具调用、流式输出、`tool_choice`、多模态输入和结构化 `PROMPT_TOO_LONG` 错误处理。目前未接入 OpenAI Responses API。

## 许可证

本仓库包含依据 MIT 许可证分发的衍生代码。所需的版权与许可声明见 [LICENSE](LICENSE)。
