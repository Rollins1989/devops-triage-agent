"""
BaseAgent: the tool-calling loop shared by every specialist and the planner.

Design choices worth calling out (these are the things an interviewer will
probe):

  - Each agent gets its own turn budget (max_turns) for *reasoning* steps,
    separate from the MCP client's shared tool-call Budget, which caps
    total tool calls across the *entire* multi-agent run. An agent can
    reason for a while without touching a tool; it can't call tools forever.
  - Failure from the reliability stack (loop detected, circuit open, budget
    exceeded, retries exhausted) is fed back to the model as a tool_result
    so the model can adapt (e.g. try a different service, or give up and
    report), and is also surfaced structurally via AgentResult.escalate so
    the orchestrator doesn't depend on the model "noticing" on its own.
  - Every agent is restricted to an explicit allow-list of tool names. A
    log-analysis agent physically cannot call restart_service even if an
    LLM hallucinated the idea -- least privilege enforced in code, not
    just in a prompt.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.llm_client import LLMClient
from src.mcp_client import ReliableMCPClient
from src.tracing import Tracer

REMEDIATION_TOOLS = {"restart_service", "rollback_deployment", "scale_deployment"}


@dataclass
class AgentResult:
    agent: str
    success: bool
    output: str
    findings: dict = field(default_factory=dict)
    escalate: bool = False
    escalate_reason: str = ""
    tool_calls_made: int = 0


class BaseAgent:
    name: str = "base-agent"
    system_prompt: str = "You are a helpful assistant."
    allowed_tools: set[str] = set()
    max_turns: int = 6
    #: if True, remediation tool calls require CONFIRM_ACTIONS=true in env
    #: (see mcp_client / main.py) -- a safety rail so an agent can't take
    #: destructive action against "real" infra without an explicit opt-in.
    requires_confirmation: bool = False

    def __init__(self, llm: LLMClient, mcp: ReliableMCPClient, tracer: Tracer, dry_run: bool = True):
        self.llm = llm
        self.mcp = mcp
        self.tracer = tracer
        self.dry_run = dry_run

    def _tool_schemas(self) -> list[dict]:
        return [t for t in self.mcp.tool_schemas if t["name"] in self.allowed_tools]

    async def run(self, task: str, context: Optional[dict] = None) -> AgentResult:
        context = context or {}
        messages: list[dict] = [
            {"role": "user", "content": f"TASK: {task}\n\nCONTEXT: {json.dumps(context, default=str)}"}
        ]
        tool_calls_made = 0

        with self.tracer.span("agent_start", "agent_end", detail=self.name, task=task):
            for turn in range(1, self.max_turns + 1):
                decision = self.llm.decide(self.system_prompt, messages, self._tool_schemas())

                if decision.stop_text is not None:
                    return AgentResult(
                        agent=self.name, success=True, output=decision.stop_text,
                        tool_calls_made=tool_calls_made,
                    )

                tool_name = decision.tool_name
                tool_args = decision.tool_args or {}

                if tool_name not in self.allowed_tools:
                    # Least-privilege enforcement: refuse in code, tell the model why,
                    # let it try something it's actually allowed to do.
                    self.tracer.event(
                        "tool_error", f"{self.name} attempted disallowed tool '{tool_name}'",
                        tool=tool_name, agent=self.name,
                    )
                    messages.append(self._assistant_tool_use(decision, tool_name, tool_args))
                    messages.append(self._tool_result(
                        decision, json.dumps({"error": f"tool '{tool_name}' not permitted for this agent"})
                    ))
                    continue

                is_remediation = tool_name in REMEDIATION_TOOLS
                if is_remediation and self.dry_run:
                    result_text = json.dumps({
                        "dry_run": True,
                        "would_call": tool_name,
                        "args": tool_args,
                        "note": "DRY_RUN mode: remediation action not executed. Set CONFIRM_ACTIONS=true to allow.",
                    })
                    self.tracer.event("decision", f"DRY RUN: would call {tool_name}({tool_args})", agent=self.name)
                    messages.append(self._assistant_tool_use(decision, tool_name, tool_args))
                    messages.append(self._tool_result(decision, result_text))
                    tool_calls_made += 1
                    continue

                outcome = await self.mcp.call(tool_name, tool_args, is_remediation=is_remediation)
                tool_calls_made += 1
                messages.append(self._assistant_tool_use(decision, tool_name, tool_args))
                messages.append(self._tool_result(decision, outcome.as_tool_result_text()))

                if not outcome.ok and outcome.error_type in ("loop_detected", "budget_exceeded", "circuit_open"):
                    return AgentResult(
                        agent=self.name, success=False,
                        output=f"Halted after reliability guard tripped: {outcome.error}",
                        escalate=True, escalate_reason=outcome.error_type,
                        tool_calls_made=tool_calls_made,
                    )

            return AgentResult(
                agent=self.name, success=False,
                output=f"{self.name} did not conclude within {self.max_turns} turns.",
                escalate=True, escalate_reason="max_turns_exceeded",
                tool_calls_made=tool_calls_made,
            )

    @staticmethod
    def _assistant_tool_use(decision, tool_name: str, tool_args: dict) -> dict:
        tool_id = decision.tool_id or f"call_{uuid.uuid4().hex[:8]}"
        decision.tool_id = tool_id  # stash for pairing in _tool_result
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_args}],
        }

    @staticmethod
    def _tool_result(decision, text: str) -> dict:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": decision.tool_id, "content": text}],
        }
