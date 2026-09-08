"""
Real MCP server for the DevOps triage system, built on the official
`mcp` Python SDK's FastMCP interface (stdio transport).

This is a genuine MCP server -- not a mocked tool-calling shim. Any
MCP-compatible client (Claude Desktop, the Anthropic API's `mcp_servers`
field, this project's own mcp_client.py, or a different agent framework
entirely) can launch this process and call these tools over the real
protocol.

Run standalone for debugging with the MCP inspector:
    mcp dev src/mcp_server.py

Or let this project's orchestrator spawn it automatically (see
mcp_client.py), which is how `main.py` uses it.

The infra state is intentionally *simulated* (see infra_simulator.py) so
the project runs anywhere without cloud credentials. Only that module
would need to be swapped for real API clients (kubernetes-client,
prometheus-api-client, PagerDuty, etc.) to point this at real
infrastructure -- the MCP tool surface itself would not change.
"""

import os
import random
import sys

from mcp.server.fastmcp import FastMCP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.infra_simulator import InfraSimulator, ScenarioInjection, SERVICES  # noqa: E402

RUNBOOKS = {
    "crashloop": "Runbook RB-014 (CrashLoopBackOff): 1) check logs for OOM/panic. "
                 "2) if OOMKilled, restart_service (often self-heals via mem reset), "
                 "then consider raising memory limits. 3) if restart doesn't hold, escalate to owning team.",
    "latency": "Runbook RB-022 (Latency spike after deploy): 1) diff deployed_version against last known-good. "
               "2) if a canary/partial rollout expanded unexpectedly, rollback_deployment. "
               "3) check for N+1 query patterns in recent commits.",
    "db": "Runbook RB-031 (DB connection exhaustion): 1) check active connection count vs pool max. "
          "2) restart_service rarely helps here; consider scale_deployment down on the noisy client "
          "and/or raising pool size. 3) escalate to DB on-call if pool size can't be safely changed.",
}

_seed = int(os.environ.get("TRIAGE_SEED", "42"))
sim = InfraSimulator(db_path=os.environ.get("TRIAGE_DB", "infra_state.db"), seed=_seed)

_SCENARIOS = {
    "crashloop": ScenarioInjection("payments-service", "memory_leak_crashloop", "high"),
    "latency": ScenarioInjection("api-gateway", "bad_deploy_latency", "high"),
    "db": ScenarioInjection("orders-db", "db_connection_exhaustion", "critical"),
}
_scenario_key = os.environ.get("TRIAGE_SCENARIO", "crashloop")
sim.inject(_SCENARIOS[_scenario_key])

# Simulate real-world flakiness on write/remediation actions so the
# reliability layer (retry + circuit breaker) in the agent has something
# genuine to react to, not just a theoretical code path.
FLAKE_RATE = float(os.environ.get("TRIAGE_FLAKE_RATE", "0.25"))
_flake_rng = random.Random(_seed + 1)


def _maybe_flake(tool_name: str) -> None:
    if _flake_rng.random() < FLAKE_RATE:
        raise RuntimeError(f"transient infra error calling '{tool_name}' (timeout talking to orchestrator API)")


mcp = FastMCP("devops-triage-tools")


@mcp.tool()
def get_service_health(service: str) -> dict:
    """Get current health status, error rate, latency, replica count, and deployed version for a service.

    Args:
        service: One of the known service names, e.g. 'payments-service'.
    """
    return sim.get_service_health(service)


@mcp.tool()
def list_services() -> list[str]:
    """List all known services in the environment."""
    return sim.list_services()


@mcp.tool()
def get_recent_logs(service: str, limit: int = 20) -> list[dict]:
    """Fetch recent log lines for a service, most recent first.

    Args:
        service: Service name.
        limit: Max number of log lines to return.
    """
    return sim.get_recent_logs(service, limit)


@mcp.tool()
def get_metrics(service: str) -> dict:
    """Get key metrics (error_rate, latency_p99_ms, replicas) for a service."""
    return sim.get_metrics(service)


@mcp.tool()
def list_incidents() -> list[dict]:
    """List currently open and past incidents."""
    return sim.list_incidents()


@mcp.tool()
def search_runbook(query: str) -> list[str]:
    """Search internal runbooks by keyword (e.g. 'crashloop', 'latency', 'db')."""
    q = query.lower()
    matches = [v for k, v in RUNBOOKS.items() if k in q]
    return matches or list(RUNBOOKS.values())


@mcp.tool()
def restart_service(service: str) -> dict:
    """Restart a service. This is a REMEDIATION action that mutates real infra state.

    Args:
        service: Service name to restart.
    """
    _maybe_flake("restart_service")
    return sim.restart_service(service)


@mcp.tool()
def rollback_deployment(service: str) -> dict:
    """Roll a service back to its last known-good deployed version. REMEDIATION action.

    Args:
        service: Service name to roll back.
    """
    _maybe_flake("rollback_deployment")
    return sim.rollback_deployment(service)


@mcp.tool()
def scale_deployment(service: str, replicas: int) -> dict:
    """Change the replica count for a service. REMEDIATION action.

    Args:
        service: Service name to scale.
        replicas: Target replica count (1-20).
    """
    _maybe_flake("scale_deployment")
    return sim.scale_deployment(service, replicas)


@mcp.tool()
def create_incident_ticket(summary: str, severity: str) -> dict:
    """File an incident ticket for human follow-up / escalation.

    Args:
        summary: Short description of the incident.
        severity: One of 'low', 'medium', 'high', 'critical'.
    """
    return sim.create_incident_ticket(summary, severity)


if __name__ == "__main__":
    mcp.run()
