"""CLI for the fixed DeepSeek-backed P0 real-Agent baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bauhinia_agent.config import load_config
from bauhinia_agent.evaluation.agent_baseline import AgentBaselineConfig, run_agent_baseline
from bauhinia_agent.providers.factory import create_provider_from_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    app_config = load_config(args.provider, project_root=REPOSITORY_ROOT)
    provider = create_provider_from_config(app_config)
    if provider.name != "deepseek" or provider.model != "deepseek-v4-flash":
        raise SystemExit("P0 fixed baseline requires provider=deepseek and model=deepseek-v4-flash; " f"resolved {provider.name}/{provider.model}")
    report, report_path = run_agent_baseline(
        provider,
        AgentBaselineConfig(
            corpus_root=REPOSITORY_ROOT / "benchmarks" / "baseline_v0",
            output_root=REPOSITORY_ROOT / "benchmarks" / "agent_baseline_v0" / "runs",
            repeats=args.repeats,
            thinking=args.thinking,
            max_tokens=args.max_tokens,
        ),
    )
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "report_run_id": report["report_run_id"],
                "metrics": report["metrics"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
