"""
Integration tests: run the full planner -> specialists -> remediation flow
against the REAL MCP server (spawned as a subprocess) and the offline
deterministic LLM backend, so these pass with no ANTHROPIC_API_KEY and no
network access -- but every tool call is a genuine MCP round trip against
real (simulated) infra state, not a stub.

Run with: python3 -m pytest tests/test_orchestrator.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.mcp_client import ReliableMCPClient
from src.orchestrator import triage_incident
from src.reliability import Budget, CircuitBreaker, LoopDetector, RetryPolicy
from src.tracing import Tracer

os.environ.pop("ANTHROPIC_API_KEY", None)  # force offline playbooks for deterministic tests


async def _run_scenario(scenario: str, flake_rate: float = 0.0, dry_run: bool = True, tmp_suffix: str = ""):
    tracer = Tracer(out_dir="traces_test", echo=False)
    server_env = {
        "TRIAGE_SCENARIO": scenario,
        "TRIAGE_SEED": "42",
        "TRIAGE_FLAKE_RATE": str(flake_rate),
        "TRIAGE_DB": f"test_infra_{scenario}{tmp_suffix}.db",
    }
    incidents = {
        "crashloop": "payments-service is failing readiness checks",
        "latency": "api-gateway p99 latency degraded after deploy",
        "db": "orders-db connection errors under load",
    }
    async with ReliableMCPClient(
        tracer=tracer,
        server_env=server_env,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05),
        circuit_breaker=CircuitBreaker(failure_threshold=3, reset_timeout_s=1.0),
        loop_detector=LoopDetector(),
        budget=Budget(max_iterations=30, max_tool_calls=20, max_wall_time_s=30),
    ) as mcp:
        return await triage_incident(incidents[scenario], mcp, tracer, dry_run=dry_run)


@pytest.mark.asyncio
async def test_crashloop_scenario_diagnoses_and_remediates():
    report = await _run_scenario("crashloop", dry_run=False, tmp_suffix="_1")
    assert report.service == "payments-service"
    assert report.log_findings["root_cause_category"] == "crashloop"
    assert report.metrics_findings["severity"] == "critical"
    assert report.planner_decision["remediate"] is True
    assert report.remediation_result["result"] == "success"
    assert report.escalated is False


@pytest.mark.asyncio
async def test_latency_scenario_diagnoses_and_rolls_back():
    report = await _run_scenario("latency", dry_run=False, tmp_suffix="_1")
    assert report.service == "api-gateway"
    assert report.log_findings["root_cause_category"] == "latency"
    assert report.remediation_result["action_taken"] == "rollback_deployment"
    assert report.remediation_result["result"] == "success"


@pytest.mark.asyncio
async def test_db_scenario_escalates_instead_of_auto_remediating():
    """Policy test: DB connection exhaustion must NEVER trigger an automated
    restart/rollback/scale -- only human escalation via incident ticket."""
    report = await _run_scenario("db", dry_run=False, tmp_suffix="_1")
    assert report.log_findings["root_cause_category"] == "db"
    assert report.planner_decision["remediate"] is False
    assert report.planner_decision["escalate_to_human"] is True
    assert report.escalated is True
    assert report.remediation_result == {}  # remediation agent never even ran


@pytest.mark.asyncio
async def test_dry_run_default_never_mutates_state():
    """Safety-rail test: without --confirm-actions / dry_run=False, remediation
    tools must not actually execute even when the planner decides to remediate."""
    report = await _run_scenario("crashloop", dry_run=True, tmp_suffix="_2")
    assert report.planner_decision["remediate"] is True
    assert report.remediation_result["result"] == "skipped"


@pytest.mark.asyncio
async def test_high_flake_rate_still_reaches_a_terminal_outcome():
    """Reliability test: even with aggressive simulated infra flakiness,
    the run must terminate (not hang) and produce escalation rather than
    crashing or looping forever."""
    report = await _run_scenario("crashloop", flake_rate=0.95, dry_run=False, tmp_suffix="_3")
    assert report.remediation_result["result"] in ("success", "escalated")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
