"""BauhiniaAgent Textual/TUI 入口模块。"""

from bauhinia_agent.app.factory import create_bauhinia_agent_app
from bauhinia_agent.app.tui import BauhiniaAgentApp, BauhiniaAgentTuiConfig

__all__ = ["BauhiniaAgentApp", "BauhiniaAgentTuiConfig", "create_bauhinia_agent_app"]
