"""
Orchestrator: the planner-led control flow that ties everything together.

Flow for one incident:

  1. Planner reads the raw incident report -> identifies target service + priority.
  2. Log Analyst and Metrics Analyst run CONCURRENTLY (asyncio.gather) against
     the same shared ReliableMCPClient -- they're independent, read-only
     investigations, so there's no reason to serialize them. This is also
     where loop-detection state gets interesting: both agents' tool calls
     interleave through the same LoopDetector/CircuitBreaker instances.
  3. Planner synthesizes both findings into a remediate / escalate decision,
     applying an explicit conservative policy (see planner_agent.py).
  4. If remediation is warranted, the Remediation Agent runs (dry-run by
     default; real state mutation only with CONFIRM_ACTIONS=true).
  5. If anything escalated along the way (reliability guard tripped, planner
     policy said escalate, or remediation didn't fully resolve it), a
     human-readable incident summary is produced and an incident ticket
     exists for follow-up.

Every handoff between agents is traced explicitly (`tracer.event("handoff", ...)`)
so the full decision trail -- not just the final answer -- is inspectable.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.agents import AgentResult, LogAnalystAgent, MetricsAnalystAgent, PlannerAgent, RemediationAgent
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer


@dataclass
class TriageReport:
    incident: str
    service: Optional[str] = None
    log_findings: dict = field(default_factory=dict)
    metrics_findings: dict = field(default_factory=dict)
    planner_decision: dict = field(default_factory=dict)
    remediation_result: dict = field(default_factory=dict)
    escalated: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    agent_results: list[AgentResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable report for automation and CI consumers."""
        data = asdict(self)
        data["agent_results"] = [asdict(result) for result in self.agent_results]
        return data

    def summary(self) -> str:
        lines = [
            f"Incident: {self.incident}",
            f"Service:  {self.service}",
            f"Log Analyst  -> {json.dumps(self.log_findings)}",
            f"Metrics Analyst -> {json.dumps(self.metrics_findings)}",
            f"Planner decision -> {json.dumps(self.planner_decision)}",
        ]
        if self.remediation_result:
            lines.append(f"Remediation -> {json.dumps(self.remediation_result)}")
        lines.append(f"Escalated: {self.escalated}" + (f" ({'; '.join(self.escalation_reasons)})" if self.escalation_reasons else ""))
        return "\n".join(lines)


async def triage_incident(incident_text: str, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True) -> TriageReport:
    report = TriageReport(incident=incident_text)

    planner = PlannerAgent.create(mcp, tracer, dry_run=dry_run)

    # --- Step 1: plan -------------------------------------------------------
    tracer.event("handoff", "orchestrator -> planner (identify target service)")
    plan_result = await planner.run(f"PLAN: {incident_text}")
    report.agent_results.append(plan_result)
    plan = _safe_json(plan_result.output)
    report.service = plan.get("service")

    if not report.service:
        report.escalated = True
        report.escalation_reasons.append("planner could not identify a target service")
        return report

    # --- Step 2: concurrent specialist investigation ------------------------
    tracer.event("handoff", f"planner -> [log-analyst, metrics-analyst] for {report.service}", parallel=True)
    log_agent = LogAnalystAgent.create(mcp, tracer, dry_run=dry_run)
    metrics_agent = MetricsAnalystAgent.create(mcp, tracer, dry_run=dry_run)

    log_result, metrics_result = await asyncio.gather(
        log_agent.run(f"Investigate logs for {report.service}", context={"service": report.service}),
        metrics_agent.run(f"Check health/metrics for {report.service}", context={"service": report.service}),
    )
    report.agent_results += [log_result, metrics_result]

    if log_result.escalate or metrics_result.escalate:
        report.escalated = True
        if log_result.escalate:
            report.escalation_reasons.append(f"log-analyst escalated: {log_result.escalate_reason}")
        if metrics_result.escalate:
            report.escalation_reasons.append(f"metrics-analyst escalated: {metrics_result.escalate_reason}")

    report.log_findings = _safe_json(log_result.output)
    report.metrics_findings = _safe_json(metrics_result.output)

    # --- Step 3: synthesis ----------------------------------------------------
    tracer.event("handoff", "[log-analyst, metrics-analyst] -> planner (synthesize)")
    synth_result = await planner.run(
        "SYNTHESIZE: combine specialist findings and decide next action",
        context={"log_findings": report.log_findings, "metrics_findings": report.metrics_findings},
    )
    report.agent_results.append(synth_result)
    decision = _safe_json(synth_result.output)
    report.planner_decision = decision

    if decision.get("escalate_to_human"):
        report.escalated = True
        report.escalation_reasons.append(decision.get("rationale", "planner policy escalation"))

    # --- Step 4: remediation (conditional) -------------------------------------
    if decision.get("remediate"):
        tracer.event("handoff", f"planner -> remediation-agent for {report.service}")
        remediation_agent = RemediationAgent.create(mcp, tracer, dry_run=dry_run)
        remediation_task = (
            f'Category: "{decision.get("root_cause_category")}" for service {report.service}. '
            f"Findings: {json.dumps(report.log_findings)}"
        )
        rem_result = await remediation_agent.run(remediation_task, context={"service": report.service})
        report.agent_results.append(rem_result)
        report.remediation_result = _safe_json(rem_result.output)
        if rem_result.escalate or report.remediation_result.get("result") == "escalated":
            report.escalated = True
            report.escalation_reasons.append("remediation could not fully resolve; escalated to on-call")
    else:
        tracer.event("decision", f"planner decided against automated remediation: {decision.get('rationale')}")

    return report


def _safe_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}
