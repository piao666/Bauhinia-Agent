# Bauhinia-Agent

Bauhinia-Agent is a local, evidence-gated coding agent for software engineering. It combines planning, tool execution, verification, and persistent session context while preserving explicit permission boundaries.

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
bauhinia-agent
```

The project is beginning its Evo implementation phase. Its intended lifecycle is: plan, act, verify, compile evidence-backed experience, evaluate, and promote or roll back only through explicit gates.

## Provider support

The runtime supports OpenAI Chat Completions-compatible providers and the Anthropic Messages API. Both providers support tool calling, streaming, `tool_choice`, multimodal inputs, and structured `PROMPT_TOO_LONG` error handling. OpenAI Responses API support is not currently included.

## License

This repository includes derivative code distributed under the MIT License. See [LICENSE](LICENSE) for the required copyright and license notice.
