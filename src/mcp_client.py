"""Reliable MCP client with retries, circuit breaking, loop detection and budgets."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.reliability import (
    Budget, CircuitBreaker, CircuitOpenError, LoopDetectedError, LoopDetector,
    RetryPolicy, TerminalError,
)
from src.tracing import Tracer


class ToolCallOutcome:
    """Normalized result of a tool invocation."""

    def __init__(self, ok: bool, data: Any = None, error: Optional[str] = None, error_type: str = ""):
        self.ok = ok
        self.data = data
        self.error = error
        self.error_type = error_type

    def as_tool_result_text(self) -> str:
        if self.ok:
            return json.dumps(self.data, default=str)
        return json.dumps({"error": self.error, "error_type": self.error_type}, default=str)


class ReliableMCPClient:
    """One shared MCP connection with run-wide reliability controls."""

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
        tool_timeout_s: float = 15.0,
    ):
        self.tracer = tracer
        self.server_command = server_command
        self.server_args = server_args or ["-m", "src.mcp_server"]
        self.server_env = {**os.environ, **(server_env or {})}
        self.retry_policy = retry_policy or RetryPolicy()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.loop_detector = loop_detector or LoopDetector()
        self.budget = budget or Budget()
        self.tool_timeout_s = tool_timeout_s
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
        result = await asyncio.wait_for(self._session.call_tool(tool_name, args), timeout=self.tool_timeout_s)
        if not result.content:
            parsed: Any = {}
        elif len(result.content) == 1:
            raw = result.content[0].text
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
        else:
            parsed = [block.text for block in result.content]
        if getattr(result, "isError", False):
            raise RuntimeError(parsed if isinstance(parsed, str) else json.dumps(parsed))
        if isinstance(parsed, dict) and "error" in parsed and len(parsed) == 1:
            raise TerminalError(RuntimeError(parsed["error"]))
        return parsed

    async def call(self, tool_name: str, args: dict, is_remediation: bool = False) -> ToolCallOutcome:
        try:
            self.budget.check_tool_call()
        except Exception as e:
            self.tracer.event("budget_exceeded", str(e), tool=tool_name)
            return ToolCallOutcome(False, error=str(e), error_type="budget_exceeded")

        try:
            self.loop_detector.record(tool_name, args)
        except LoopDetectedError as e:
            self.tracer.event("loop_detected", str(e), tool=tool_name, args=args)
            return ToolCallOutcome(False, error=str(e), error_type="loop_detected")

        try:
            self.circuit_breaker.before_call(tool_name)
        except CircuitOpenError as e:
            self.tracer.event("circuit_open", str(e), tool=tool_name)
            return ToolCallOutcome(False, error=str(e), error_type="circuit_open")

        self.tracer.event("tool_call", f"{tool_name}({json.dumps(args, default=str)})", tool=tool_name, args=args, remediation=is_remediation)
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                data = await self._raw_call(tool_name, args)
                # A retry that eventually succeeds is one successful operation;
                # do not let earlier transient attempts trip the circuit breaker.
                self.circuit_breaker.on_success(tool_name)
                self.tracer.event("tool_result", f"{tool_name} -> ok", tool=tool_name, attempt=attempt)
                return ToolCallOutcome(True, data=data)
            except TerminalError as e:
                self.circuit_breaker.on_failure(tool_name)
                self.tracer.event("tool_error", f"{tool_name} terminal: {e.original}", tool=tool_name, attempt=attempt)
                return ToolCallOutcome(False, error=str(e.original), error_type="terminal")
            except Exception as e:
                last_error = e
                if attempt < self.retry_policy.max_attempts:
                    delay = self.retry_policy._delay_for(attempt)
                    self.tracer.event("retry", f"{tool_name} attempt {attempt} failed; retrying in {delay:.2f}s", tool=tool_name, attempt=attempt, delay_s=round(delay, 2))
                    await asyncio.sleep(delay)
                else:
                    self.circuit_breaker.on_failure(tool_name)
                    self.tracer.event("tool_error", f"{tool_name} exhausted retries: {e}", tool=tool_name, attempt=attempt)

        return ToolCallOutcome(
            False,
            error=f"'{tool_name}' failed after {self.retry_policy.max_attempts} attempt(s): {last_error!r}",
            error_type="retry_exhausted",
        )
