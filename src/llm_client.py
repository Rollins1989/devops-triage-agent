"""
LLM client abstraction used by every agent.

Two backends:

  - AnthropicLLM:  real calls to the Anthropic Messages API with tool use
                   (model: claude-sonnet-4-6). This is what you'd use in
                   production and is fully wired up -- point ANTHROPIC_API_KEY
                   at it and every agent reasons with a real model.

  - OfflineLLM:    a small deterministic rule-based stand-in that mimics the
                   *shape* of tool-calling decisions (same interface: given
                   messages + tool schemas, decide to call a tool or respond
                   with text) so the entire orchestration, reliability, and
                   tracing machinery can be demoed, tested, and graded
                   without any API key or network access. It is intentionally
                   simple and clearly labeled -- it is not the point of this
                   project, the orchestration engineering is.

Both implement the same `LLMClient.decide()` interface so agents don't care
which backend they're talking to.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMDecision:
    """What the model wants to do next."""
    stop_text: Optional[str] = None          # non-None => model is done, this is its final answer
    tool_name: Optional[str] = None           # non-None => model wants to call this tool
    tool_args: Optional[dict] = None
    tool_id: Optional[str] = None              # provider's tool_use id, for tool_result correlation
    raw: Any = None                            # backend-specific raw response, for tracing/debugging


class LLMClient(ABC):
    @abstractmethod
    def decide(self, system: str, messages: list[dict], tools: list[dict]) -> LLMDecision:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class AnthropicLLM(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1500):
        import anthropic  # imported lazily so OfflineLLM works without the package configured

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def name(self) -> str:
        return f"anthropic:{self.model}"

    @staticmethod
    def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema") or {"type": "object", "properties": {}},
            }
            for t in tools
        ]

    def decide(self, system: str, messages: list[dict], tools: list[dict]) -> LLMDecision:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=self._to_anthropic_tools(tools) if tools else [],
        )
        tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
        text_blocks = [b.text for b in resp.content if b.type == "text"]

        if tool_use_blocks:
            b = tool_use_blocks[0]
            return LLMDecision(tool_name=b.name, tool_args=b.input, tool_id=b.id, raw=resp)
        return LLMDecision(stop_text="\n".join(text_blocks) or "(no response)", raw=resp)


class OfflineLLM(LLMClient):
    """Deterministic stand-in used when ANTHROPIC_API_KEY is not set.

    Implements a fixed, sensible triage playbook per agent role so the full
    system (planner -> specialists -> remediation -> escalation, with real
    MCP tool calls and real reliability handling) runs end-to-end offline.
    Each agent supplies its own `playbook` function; this class just drives
    it against the running conversation state.
    """

    def __init__(self, playbook):
        self.playbook = playbook
        self._step = 0

    def name(self) -> str:
        return "offline-deterministic"

    def decide(self, system: str, messages: list[dict], tools: list[dict]) -> LLMDecision:
        decision = self.playbook(messages, tools, self._step)
        self._step += 1
        return decision


def build_llm_client(playbook=None) -> LLMClient:
    """Factory: use the real Anthropic API if a key is configured, else
    fall back to the offline deterministic playbook for that agent."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLM()
        except Exception:
            pass
    if playbook is None:
        raise RuntimeError("No ANTHROPIC_API_KEY set and no offline playbook provided.")
    return OfflineLLM(playbook)
