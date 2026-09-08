from __future__ import annotations

import json

from src.agents.base_agent import BaseAgent
from src.agents.offline_playbooks import extract_service, first_user_text, last_tool_result, stop, tool_call
from src.infra_simulator import SERVICES
from src.llm_client import LLMDecision, build_llm_client
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer


SYSTEM_PROMPT = """You are the Metrics Analyst, a specialist agent in a DevOps incident triage system.

Your job: given a service under investigation, check its current health and
key metrics (error rate, p99 latency, replica count) and classify severity.

Rules:
- Always call get_service_health first, then get_metrics for corroboration.
- Classify severity as one of: nominal, degraded, critical based on
  error_rate and latency thresholds (error_rate > 0.3 or latency > 2000ms
  is critical; error_rate > 0.05 or latency > 500ms is degraded; otherwise
  nominal).
- When confident, STOP with a final answer as compact JSON with keys:
  severity, error_rate, latency_p99_ms, replicas, needs_remediation (bool).
- Never call any tool other than get_service_health or get_metrics.
"""


def _offline_playbook(messages: list[dict], tools: list[dict], step: int) -> LLMDecision:
    task_text = first_user_text(messages)
    service = extract_service(task_text, SERVICES) or SERVICES[0]

    if step == 0:
        return tool_call("get_service_health", {"service": service})
    if step == 1:
        return tool_call("get_metrics", {"service": service})

    metrics = last_tool_result(messages) or {}
    error_rate = metrics.get("error_rate", 0.0)
    latency = metrics.get("latency_p99_ms", 0.0)
    replicas = metrics.get("replicas", 0)

    if error_rate > 0.3 or latency > 2000:
        severity = "critical"
    elif error_rate > 0.05 or latency > 500:
        severity = "degraded"
    else:
        severity = "nominal"

    return stop(json.dumps({
        "severity": severity,
        "error_rate": error_rate,
        "latency_p99_ms": latency,
        "replicas": replicas,
        "needs_remediation": severity in ("degraded", "critical"),
    }))


class MetricsAnalystAgent(BaseAgent):
    name = "metrics-analyst"
    system_prompt = SYSTEM_PROMPT
    allowed_tools = {"get_service_health", "get_metrics"}
    max_turns = 4

    @classmethod
    def create(cls, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True) -> "MetricsAnalystAgent":
        llm = build_llm_client(playbook=_offline_playbook)
        return cls(llm=llm, mcp=mcp, tracer=tracer, dry_run=dry_run)
