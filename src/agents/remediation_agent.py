from __future__ import annotations

import json

from src.agents.base_agent import BaseAgent
from src.agents.offline_playbooks import extract_service, first_user_text, last_tool_result, stop, tool_call
from src.infra_simulator import SERVICES
from src.llm_client import LLMDecision, build_llm_client
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer


SYSTEM_PROMPT = """You are the Remediation Agent, a specialist agent in a DevOps incident triage system.
You act ONLY after the Log Analyst and Metrics Analyst have provided findings; you do not
re-diagnose from scratch.

Category -> action mapping (apply the narrowest safe action for the diagnosed cause):
  - "crashloop"  -> restart_service (memory reset typically clears an OOM crashloop)
  - "latency"    -> rollback_deployment (bad canary/deploy is the usual cause)
  - "db"         -> do NOT restart or rollback (per runbook RB-031 these don't fix connection
                    exhaustion and are potentially disruptive); instead create_incident_ticket
                    to escalate to the database on-call team.
  - "unknown"    -> create_incident_ticket for human triage; do not guess with a remediation action.

Rules:
- Take at most ONE remediation action attempt. If the tool result says
  "no_effect" or continues to fail after the platform's own retries, do NOT
  keep trying different remediations yourself -- create_incident_ticket and
  stop. Repeated blind remediation attempts are exactly the failure mode
  this system is designed to prevent.
- When done, STOP with a final answer as compact JSON with keys:
  action_taken, result ("success"|"no_effect"|"escalated"|"skipped"), notes.
"""


_ACTION_FOR_CATEGORY = {
    "crashloop": "restart_service",
    "latency": "rollback_deployment",
}


def _offline_playbook(messages: list[dict], tools: list[dict], step: int) -> LLMDecision:
    task_text = first_user_text(messages)
    service = extract_service(task_text, SERVICES) or SERVICES[0]

    category = "unknown"
    for cat in ("crashloop", "latency", "db", "unknown"):
        if f'"{cat}"' in task_text or f"category: {cat}" in task_text.lower():
            category = cat
            break

    action = _ACTION_FOR_CATEGORY.get(category)

    if step == 0:
        if action:
            return tool_call(action, {"service": service})
        return tool_call("create_incident_ticket", {
            "summary": f"{service}: root cause '{category}' requires human review per runbook policy",
            "severity": "high",
        })

    result = last_tool_result(messages) or {}

    if step == 1 and action:
        outcome_result = result.get("result")
        if result.get("dry_run"):
            return stop(json.dumps({
                "action_taken": action, "result": "skipped",
                "notes": f"DRY_RUN: would have called {action} on {service}. "
                         f"Re-run with --confirm-actions to execute for real.",
            }))
        if outcome_result == "success":
            return stop(json.dumps({
                "action_taken": action, "result": "success",
                "notes": f"{action} resolved the issue for {service}; verified via updated health state.",
            }))
        if "error" in result or "error_type" in result:
            return tool_call("create_incident_ticket", {
                "summary": f"{service}: {action} failed after platform retries ({result.get('error', 'unknown error')})",
                "severity": "critical",
            })
        # no_effect
        return tool_call("create_incident_ticket", {
            "summary": f"{service}: {action} completed but did not resolve the issue; root cause likely deeper",
            "severity": "high",
        })

    # step 2, or step 1 when there was no direct action (db/unknown escalation path)
    ticket = result
    return stop(json.dumps({
        "action_taken": action or "create_incident_ticket",
        "result": "escalated",
        "notes": f"Escalated to human on-call. Ticket: {json.dumps(ticket)}",
    }))


class RemediationAgent(BaseAgent):
    name = "remediation"
    system_prompt = SYSTEM_PROMPT
    allowed_tools = {"restart_service", "rollback_deployment", "scale_deployment", "create_incident_ticket"}
    max_turns = 4
    requires_confirmation = True

    @classmethod
    def create(cls, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True) -> "RemediationAgent":
        llm = build_llm_client(playbook=_offline_playbook)
        return cls(llm=llm, mcp=mcp, tracer=tracer, dry_run=dry_run)
