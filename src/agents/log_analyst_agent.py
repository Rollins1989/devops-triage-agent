from __future__ import annotations

import json

from src.agents.base_agent import BaseAgent
from src.agents.offline_playbooks import extract_service, first_user_text, last_tool_result, stop, tool_call, tool_result_for
from src.infra_simulator import SERVICES
from src.llm_client import LLMDecision, build_llm_client
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer


SYSTEM_PROMPT = """You are the Log Analyst, a specialist agent in a DevOps incident triage system.

Your job: given a service under investigation, pull its recent logs, identify
the error signature, and cross-reference internal runbooks to name the most
likely root cause category (crashloop / latency / db / unknown).

Rules:
- Use get_recent_logs before forming any hypothesis. Do not guess from the
  service name alone.
- Use search_runbook once you have a candidate signature (e.g. "OOMKilled",
  "connection pool exhausted") to find the matching remediation runbook.
- When you are confident, STOP by returning a final answer (no more tool
  calls) as compact JSON with keys: root_cause_category, evidence (list of
  1-3 short strings), runbook (string or null), confidence (0-1).
- Never call any tool other than get_recent_logs or search_runbook.
"""


def _offline_playbook(messages: list[dict], tools: list[dict], step: int) -> LLMDecision:
    task_text = first_user_text(messages)
    service = extract_service(task_text, SERVICES) or SERVICES[0]

    if step == 0:
        return tool_call("get_recent_logs", {"service": service, "limit": 10})

    logs_result = tool_result_for(messages, "get_recent_logs")
    if step == 1:
        # Pick a runbook query keyword from the log content we just saw.
        joined = json.dumps(logs_result or {})
        if "OOMKilled" in joined or "OutOfMemory" in joined or "CrashLoop" in joined:
            query = "crashloop"
        elif "N+1" in joined or "latency spike" in joined or "canary" in joined:
            query = "latency"
        elif "connection pool" in joined or "connection slots" in joined:
            query = "db"
        else:
            query = "general"
        return tool_call("search_runbook", {"query": query})

    runbook_result = last_tool_result(messages)
    joined_logs = json.dumps(logs_result or {})  # logs_result was captured via tool_result_for() above
    if "OOMKilled" in joined_logs or "CrashLoop" in joined_logs:
        category, evidence = "crashloop", ["OOMKilled container", "CrashLoopBackOff observed"]
    elif "N+1" in joined_logs or "canary" in joined_logs:
        category, evidence = "latency", ["unexpected canary rollout to 100%", "N+1 query pattern in logs"]
    elif "connection pool" in joined_logs or "connection slots" in joined_logs:
        category, evidence = "db", ["connection pool exhausted", "FATAL: no remaining connection slots"]
    else:
        category, evidence = "unknown", ["no clear error signature in recent logs"]

    runbook_text = None
    if isinstance(runbook_result, list) and runbook_result:
        runbook_text = runbook_result[0]
    elif isinstance(runbook_result, str):
        # A single-match runbook search collapses to a plain string over MCP
        # (see mcp_client._raw_call) rather than a one-element list.
        runbook_text = runbook_result

    return stop(json.dumps({
        "root_cause_category": category,
        "evidence": evidence,
        "runbook": runbook_text,
        "confidence": 0.85 if category != "unknown" else 0.3,
    }))


class LogAnalystAgent(BaseAgent):
    name = "log-analyst"
    system_prompt = SYSTEM_PROMPT
    allowed_tools = {"get_recent_logs", "search_runbook"}
    max_turns = 5

    @classmethod
    def create(cls, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True) -> "LogAnalystAgent":
        llm = build_llm_client(playbook=_offline_playbook)
        return cls(llm=llm, mcp=mcp, tracer=tracer, dry_run=dry_run)
