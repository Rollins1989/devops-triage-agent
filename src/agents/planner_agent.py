from __future__ import annotations

import json

from src.agents.base_agent import BaseAgent
from src.agents.offline_playbooks import extract_service, first_user_text, stop
from src.infra_simulator import SERVICES
from src.llm_client import LLMDecision, build_llm_client
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer


SYSTEM_PROMPT = """You are the Planner, the orchestrating agent in a DevOps incident triage system.
You have no direct tool access -- you decompose incidents into subtasks for specialist agents
(Log Analyst, Metrics Analyst, Remediation) and later synthesize their findings.

When given "PLAN:" input: identify which service is affected and respond with JSON:
  {"service": <name>, "priority": "low"|"medium"|"high"|"critical", "investigate": true}

When given "SYNTHESIZE:" input containing specialist findings: decide whether an automated
remediation attempt is warranted, and respond with JSON:
  {"remediate": bool, "root_cause_category": <string>, "rationale": <string>,
   "escalate_to_human": bool}

Be conservative: if the log analyst's confidence is low, or metrics severity is "nominal"
while logs suggest a problem (inconsistent signals), or the diagnosed category is "db"
(policy requires human review), set escalate_to_human=true and remediate=false.
"""


def _offline_playbook(messages: list[dict], tools: list[dict], step: int) -> LLMDecision:
    task_text = first_user_text(messages)

    if task_text.startswith("TASK: PLAN:"):
        service = extract_service(task_text, SERVICES) or SERVICES[0]
        return stop(json.dumps({"service": service, "priority": "high", "investigate": True}))

    # SYNTHESIZE step: task_text contains embedded JSON findings from both specialists.
    try:
        payload = json.loads(task_text.split("CONTEXT: ", 1)[1])
    except (IndexError, json.JSONDecodeError):
        payload = {}

    log_findings = payload.get("log_findings", {})
    metrics_findings = payload.get("metrics_findings", {})

    category = log_findings.get("root_cause_category", "unknown")
    confidence = log_findings.get("confidence", 0.0)
    severity = metrics_findings.get("severity", "nominal")

    inconsistent = severity == "nominal" and category not in ("unknown",)
    low_confidence = confidence < 0.5
    policy_requires_human = category == "db"

    escalate = inconsistent or low_confidence or policy_requires_human or category == "unknown"
    remediate = not escalate and metrics_findings.get("needs_remediation", False)

    rationale_bits = []
    if policy_requires_human:
        rationale_bits.append("category 'db' requires human review per policy")
    if low_confidence:
        rationale_bits.append(f"log analyst confidence too low ({confidence})")
    if inconsistent:
        rationale_bits.append("log and metrics signals disagree")
    if not rationale_bits:
        rationale_bits.append(f"consistent '{category}' diagnosis with severity '{severity}'; safe to auto-remediate")

    return stop(json.dumps({
        "remediate": remediate,
        "root_cause_category": category,
        "rationale": "; ".join(rationale_bits),
        "escalate_to_human": escalate,
    }))


class PlannerAgent(BaseAgent):
    name = "planner"
    system_prompt = SYSTEM_PROMPT
    allowed_tools: set[str] = set()
    max_turns = 2

    @classmethod
    def create(cls, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True) -> "PlannerAgent":
        llm = build_llm_client(playbook=_offline_playbook)
        return cls(llm=llm, mcp=mcp, tracer=tracer, dry_run=dry_run)
