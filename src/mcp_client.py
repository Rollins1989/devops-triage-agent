"""
MCP client wrapper: connects to the real devops-triage MCP server (spawned
as a subprocess over stdio) and wraps every tool call with the reliability
stack (retry, circuit breaker, loop detection, budget) plus tracing.

This is the seam between "agent decides to call a tool" and "tool call
actually happens" -- every failure mode the reliability layer knows about
gets exercised here, against a real subprocess and real (if simulated)
infra state, not a stub.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.reliability import (
    Budget,
    CircuitBreaker,
    CircuitOpenError,
    LoopDetectedError,
    LoopDetector,
    RetryExhausted,
    RetryPolicy,
    TerminalError,
)
from src.tracing import Tracer


class ToolCallOutcome:
    """Normalized result of attempting a tool call, whatever happened."""

    def __init__(self, ok: bool, data: Any = None, error: Optional[str] = None, error_type: str = ""):
        self.ok = ok
        self.data = data
        self.error = error
        self.error_type = error_type  # "circuit_open" | "loop_detected" | "budget_exceeded" | "retry_exhausted"

    def as_tool_result_text(self) -> str:
        if self.ok:
            return json.dumps(self.data, default=str)
        return json.dumps({"error": self.error, "error_type": self.error_type}, default=str)


class ReliableMCPClient:
    """Wraps one MCP server subprocess connection with retry / circuit
    breaker / loop detection / budget enforcement, shared across whichever
    agent is currently using it (agents hand the same client instance
    around so loop detection and circuit state are visible run-wide)."""

    def __init__(
        self,
        tracer: Tracer,
        server_command: str = "python3",
        server_args: Optional[list[str]] = None,
        server_env: Optional[dict] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        loop_detector: Optional[LoopDetector] = None,
        budget: Optional[Budget] = None,
    ):
        self.tracer = tracer
        self.server_command = server_command
        self.server_args = server_args or ["-m", "src.mcp_server"]
        self.server_env = {**os.environ, **(server_env or {})}
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.loop_detector = loop_detector or LoopDetector()
        self.budget = budget or Budget()

        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None
        self.tool_schemas: list[dict] = []

    async def __aenter__(self) -> "ReliableMCPClient":
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self.server_command, args=self.server_args, env=self.server_env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        tools = await self._session.list_tools()
        self.tool_schemas = [
            {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
            for t in tools.tools
        ]
        self.tracer.event("mcp_connected", f"{len(self.tool_schemas)} tools available", tools=[t["name"] for t in self.tool_schemas])
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack:
            await self._stack.aclose()

    async def _raw_call(self, tool_name: str, args: dict) -> Any:
        assert self._session is not None
        result = await self._session.call_tool(tool_name, args)

        # FastMCP serializes non-string return values as JSON when there's a
        # single content block, but a tool returning a list of strings (e.g.
        # search_runbook -> list[str]) comes back as *multiple* TextContent
        # blocks, one per list element, each of which is plain text rather
        # than a JSON-encoded element. Handle both shapes explicitly instead
        # of assuming content[0].text is always a JSON document.
        if not result.content:
            parsed: Any = {}
        elif len(result.content) == 1:
            text = result.content[0].text
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text
        else:
            parsed = [block.text for block in result.content]

        if getattr(result, "isError", False):
            error_msg = parsed if isinstance(parsed, str) else json.dumps(parsed)
            raise RuntimeError(error_msg)

        if isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1:
            # Tool-level semantic error (e.g. unknown service) -> terminal, don't retry blindly.
            raise TerminalError(RuntimeError(parsed["error"]))
        return parsed

    async def call(self, tool_name: str, args: dict, is_remediation: bool = False) -> ToolCallOutcome:
        """Invoke a tool through the full reliability stack. Never raises --
        failures are returned as a ToolCallOutcome so the calling agent can
        reason about them (and so a bad tool never crashes the whole run)."""

        # 1. Budget check first: cheapest way to fail fast once a run is out of runway.
        try:
            self.budget.check_tool_call()
        except Exception as e:
            self.tracer.event("budget_exceeded", str(e), tool=tool_name)
            return ToolCallOutcome(ok=False, error=str(e), error_type="budget_exceeded")

        # 2. Loop detection: catch repeat/oscillating calls before we even try.
        try:
            self.loop_detector.record(tool_name, args)
        except LoopDetectedError as e:
            self.tracer.event("loop_detected", str(e), tool=tool_name, args=args)
            return ToolCallOutcome(ok=False, error=str(e), error_type="loop_detected")

        # 3. Circuit breaker: don't hammer a tool that's already tripped.
        try:
            self.circuit_breaker.before_call(tool_name)
        except CircuitOpenError as e:
            self.tracer.event("circuit_open", str(e), tool=tool_name)
            return ToolCallOutcome(ok=False, error=str(e), error_type="circuit_open")

        self.tracer.event(
            "tool_call", f"{tool_name}({json.dumps(args, default=str)})",
            tool=tool_name, args=args, remediation=is_remediation,
        )

        attempt_counter = {"n": 0}

        def sync_bridge_unused():
            # placeholder kept out of the hot path; real call is async below
            pass

        # RetryPolicy.run is synchronous-style (sleep-based); we adapt it to
        # async by driving attempts manually here so we can `await` the MCP
        # call while still reusing the exact same backoff/jitter math and
        # terminal-error semantics as the synchronous unit tests exercise.
        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            attempt_counter["n"] = attempt
            try:
                data = await self._raw_call(tool_name, args)
                self.circuit_breaker.on_success(tool_name)
                self.tracer.event("tool_result", f"{tool_name} -> ok", tool=tool_name, attempt=attempt)
                return ToolCallOutcome(ok=True, data=data)
            except TerminalError as e:
                self.circuit_breaker.on_failure(tool_name)
                self.tracer.event("tool_error", f"{tool_name} terminal: {e.original}", tool=tool_name, attempt=attempt)
                return ToolCallOutcome(ok=False, error=str(e.original), error_type="terminal")
            except Exception as e:  # noqa: BLE001 - transient/unknown, eligible for retry
                last_error = e
                self.circuit_breaker.on_failure(tool_name)
                if attempt < self.retry_policy.max_attempts:
                    delay = self.retry_policy._delay_for(attempt)
                    self.tracer.event(
                        "retry", f"{tool_name} attempt {attempt} failed ({e}); retrying in {delay:.2f}s",
                        tool=tool_name, attempt=attempt, delay_s=round(delay, 2),
                    )
                    await asyncio.sleep(delay)
                else:
                    self.tracer.event("tool_error", f"{tool_name} exhausted retries: {e}", tool=tool_name, attempt=attempt)

        return ToolCallOutcome(
            ok=False,
            error=f"'{tool_name}' failed after {self.retry_policy.max_attempts} attempt(s): {last_error!r}",
            error_type="retry_exhausted",
        )
