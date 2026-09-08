"""
Helpers shared by each agent's offline deterministic playbook.

These playbooks exist purely so the whole orchestration (planner -> three
specialists -> aggregation -> possible remediation -> escalation), running
against the *real* MCP server and reliability stack, can be exercised and
graded without an ANTHROPIC_API_KEY. They are not meant to look like actual
model reasoning -- swap in AnthropicLLM (automatic when the key is present)
for that.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.llm_client import LLMDecision


def _parse_tool_content(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}


def last_tool_result(messages: list[dict]):
    """Result of the most recently completed tool call, regardless of which tool."""
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    return _parse_tool_content(block["content"])
    return None


def tool_result_for(messages: list[dict], tool_name: str):
    """Result of the most recent call to a *specific* tool by name.

    Needed because a playbook step often needs data from an earlier tool
    call (e.g. the original log dump) even after a later, different tool
    has been called -- `last_tool_result` alone would return the wrong
    call's data in that case.
    """
    # Map tool_use_id -> tool name from assistant messages, then find the
    # most recent tool_result whose id maps to the requested tool.
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        for block in msg["content"] if isinstance(msg["content"], list) else []:
            if block.get("type") == "tool_use":
                id_to_name[block["id"]] = block["name"]

    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        for block in msg["content"] if isinstance(msg["content"], list) else []:
            if block.get("type") == "tool_result" and id_to_name.get(block.get("tool_use_id")) == tool_name:
                return _parse_tool_content(block["content"])
    return None


def extract_service(task_and_context: str, known_services: list[str]) -> Optional[str]:
    for svc in known_services:
        if svc in task_and_context:
            return svc
    return None


def first_user_text(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            return msg["content"]
    return ""


def tool_call(name: str, args: dict) -> LLMDecision:
    return LLMDecision(tool_name=name, tool_args=args)


def stop(text: str) -> LLMDecision:
    return LLMDecision(stop_text=text)
