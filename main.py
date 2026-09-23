#!/usr/bin/env python3
"""Command-line entry point for the DevOps Triage Agent."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from src.mcp_client import ReliableMCPClient
from src.orchestrator import triage_incident
from src.reliability import Budget, CircuitBreaker, LoopDetector, RetryPolicy
from src.tracing import Tracer

SCENARIO_TEXT = {
    "crashloop": "PagerDuty P1: payments-service failing readiness checks, customers report failed checkouts.",
    "latency": "PagerDuty P1: api-gateway p99 latency degraded sharply after this morning's deploy.",
    "db": "PagerDuty P1: orders-db returning connection errors under load, orders failing to write.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reliability-first multi-agent DevOps triage demo.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_TEXT),
        default="crashloop",
        help="Built-in incident scenario to run.",
    )
    parser.add_argument(
        "--flake-rate",
        type=float,
        default=0.25,
        help="Simulated transient failure rate for remediation tools (0.0-1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic simulation.")
    parser.add_argument(
        "--confirm-actions",
        action="store_true",
        help="Allow remediation tools to mutate simulator state. Default is dry-run.",
    )
    parser.add_argument("--max-tool-calls", type=int, default=20, help="Shared maximum MCP tool calls per run.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable live trace output. JSONL traces are still written.",
    )
    parser.add_argument(
        "--json-report",
        action="store_true",
        help="Emit the final triage report as machine-readable JSON on stdout.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if not 0.0 <= args.flake_rate <= 1.0:
        raise ValueError("--flake-rate must be between 0.0 and 1.0")

    tracer = Tracer(echo=not args.quiet)
    print(
        f"\n=== DevOps Triage Agent | run_id={tracer.run_id} | scenario={args.scenario} ===\n",
        file=sys.stderr,
    )

    server_env = {
        "TRIAGE_SCENARIO": args.scenario,
        "TRIAGE_SEED": str(args.seed),
        "TRIAGE_FLAKE_RATE": str(args.flake_rate),
        "TRIAGE_DB": f"infra_state_{tracer.run_id}.db",
    }
    confirm = args.confirm_actions or os.environ.get("CONFIRM_ACTIONS", "").lower() == "true"
    dry_run = not confirm

    async with ReliableMCPClient(
        tracer=tracer,
        server_env=server_env,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.4, max_delay_s=4.0),
        circuit_breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=10.0),
        loop_detector=LoopDetector(max_exact_repeats=2, max_cycle_repeats=3),
        budget=Budget(
            max_iterations=30,
            max_tool_calls=args.max_tool_calls,
            max_wall_time_s=60.0,
        ),
    ) as mcp:
        report = await triage_incident(
            SCENARIO_TEXT[args.scenario],
            mcp,
            tracer,
            dry_run=dry_run,
        )

        print("\n" + "=" * 70, file=sys.stderr)
        print("FINAL TRIAGE REPORT", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        if args.json_report:
            import json
            print(json.dumps(report.to_dict(), indent=2, default=str))
        else:
            print(report.summary(), file=sys.stderr)
        print(f"\nCircuit breaker status: {mcp.circuit_breaker.status()}", file=sys.stderr)
        print(f"Budget usage: {mcp.budget.snapshot()}", file=sys.stderr)
        print(f"Full trace written to: {tracer.path}", file=sys.stderr)

        if dry_run:
            print(
                "\n(DRY_RUN mode: no remediation actions were executed. "
                "Pass --confirm-actions to allow them.)",
                file=sys.stderr,
            )

    return 1 if report.escalated else 0


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
