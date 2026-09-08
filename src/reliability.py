"""
Reliability primitives for agentic tool use.

These are the pieces that separate a "demo agent" from something you'd
actually put in front of production infrastructure:

  - RetryPolicy:      exponential backoff + jitter, bounded attempts,
                       distinguishes retryable vs. terminal errors.
  - CircuitBreaker:    per-tool breaker so a flaky/broken tool doesn't get
                       hammered forever and doesn't take the whole run down.
  - LoopDetector:      catches an agent calling the same tool with the same
                       arguments repeatedly (the #1 failure mode of naive
                       ReAct-style agents) and forces escalation instead of
                       spinning until the token/time budget is exhausted.
  - Budget:            hard ceiling on iterations, tool calls, and wall clock
                       time for a single agent run.

Everything here is deliberately dependency-light and unit-testable in
isolation (see tests/test_reliability.py) -- no LLM or network calls needed
to exercise the failure-handling logic.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# --------------------------------------------------------------------------
# Retry with exponential backoff + jitter
# --------------------------------------------------------------------------

class RetryExhausted(Exception):
    """Raised when a retryable operation fails on its final attempt."""

    def __init__(self, tool_name: str, attempts: int, last_error: Exception):
        self.tool_name = tool_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"'{tool_name}' failed after {attempts} attempt(s): {last_error!r}"
        )


class TerminalError(Exception):
    """Wrap an error to explicitly mark it as non-retryable (e.g. bad args,
    permission denied, resource not found). RetryPolicy will not retry these."""

    def __init__(self, original: Exception):
        self.original = original
        super().__init__(str(original))


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter_s: float = 0.25

    def _delay_for(self, attempt: int) -> float:
        # attempt is 1-indexed for the attempt that just failed
        backoff = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return backoff + random.uniform(0, self.jitter_s)

    def run(self, tool_name: str, fn: Callable[[], Any], sleep: Callable[[float], None] = time.sleep) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn()
            except TerminalError as e:
                # Don't waste retries on errors that will never succeed.
                raise RetryExhausted(tool_name, attempt, e.original) from e
            except Exception as e:  # noqa: BLE001 - intentionally broad, this is a retry boundary
                last_error = e
                if attempt == self.max_attempts:
                    break
                sleep(self._delay_for(attempt))
        raise RetryExhausted(tool_name, self.max_attempts, last_error)


# --------------------------------------------------------------------------
# Circuit breaker (per tool)
# --------------------------------------------------------------------------

class CircuitState(str, Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"             # tripped, calls short-circuit immediately
    HALF_OPEN = "half_open"   # trial period, one call allowed through


class CircuitOpenError(Exception):
    def __init__(self, tool_name: str, retry_after_s: float):
        self.tool_name = tool_name
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Circuit for '{tool_name}' is OPEN; retry after {retry_after_s:.1f}s"
        )


@dataclass
class _BreakerState:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    opened_at: float = 0.0


class CircuitBreaker:
    """One breaker instance is shared across all tools; state is tracked
    per tool_name internally so a failing tool can't starve healthy ones."""

    def __init__(self, failure_threshold: int = 3, reset_timeout_s: float = 15.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._breakers: dict[str, _BreakerState] = {}

    def _get(self, tool_name: str) -> _BreakerState:
        return self._breakers.setdefault(tool_name, _BreakerState())

    def before_call(self, tool_name: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        b = self._get(tool_name)
        if b.state == CircuitState.OPEN:
            elapsed = now - b.opened_at
            if elapsed >= self.reset_timeout_s:
                b.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(tool_name, self.reset_timeout_s - elapsed)

    def on_success(self, tool_name: str) -> None:
        b = self._get(tool_name)
        b.failure_count = 0
        b.state = CircuitState.CLOSED

    def on_failure(self, tool_name: str, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        b = self._get(tool_name)
        if b.state == CircuitState.HALF_OPEN:
            # Trial call failed -> re-open immediately.
            b.state = CircuitState.OPEN
            b.opened_at = now
            return
        b.failure_count += 1
        if b.failure_count >= self.failure_threshold:
            b.state = CircuitState.OPEN
            b.opened_at = now

    def status(self) -> dict[str, str]:
        return {name: b.state.value for name, b in self._breakers.items()}


# --------------------------------------------------------------------------
# Loop detection
# --------------------------------------------------------------------------

class LoopDetectedError(Exception):
    def __init__(self, signature: str, repeats: int):
        self.signature = signature
        self.repeats = repeats
        super().__init__(
            f"Loop detected: identical tool call repeated {repeats}x -> {signature}"
        )


class LoopDetector:
    """Detects two loop patterns commonly seen in tool-calling agents:

      1. Exact repeat: same tool + same args called N times in a row.
      2. Oscillation: a short cycle of >=2 distinct calls repeating
         (e.g. A, B, A, B, A, B).

    Call `record()` before each tool invocation; it raises LoopDetectedError
    if the new call would complete a disallowed pattern.
    """

    def __init__(self, max_exact_repeats: int = 2, max_cycle_repeats: int = 3, cycle_window: int = 6):
        self.max_exact_repeats = max_exact_repeats
        self.max_cycle_repeats = max_cycle_repeats
        self.cycle_window = cycle_window
        self._history: list[str] = []

    @staticmethod
    def _signature(tool_name: str, args: dict) -> str:
        payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def record(self, tool_name: str, args: dict) -> None:
        sig = self._signature(tool_name, args)

        # Pattern 1: exact consecutive repeats.
        run_len = 1
        for prev in reversed(self._history):
            if prev == sig:
                run_len += 1
            else:
                break
        if run_len > self.max_exact_repeats:
            self._history.append(sig)
            raise LoopDetectedError(f"{tool_name}({args})", run_len)

        # Pattern 2: short oscillating cycle within the recent window.
        window = self._history[-self.cycle_window:] + [sig]
        for cycle_len in (2, 3):
            if len(window) < cycle_len * self.max_cycle_repeats:
                continue
            tail = window[-cycle_len * self.max_cycle_repeats:]
            chunks = [tuple(tail[i:i + cycle_len]) for i in range(0, len(tail), cycle_len)]
            if len(set(chunks)) == 1:
                self._history.append(sig)
                raise LoopDetectedError(f"cycle of length {cycle_len} around {tool_name}", self.max_cycle_repeats)

        self._history.append(sig)

    def reset(self) -> None:
        self._history.clear()


# --------------------------------------------------------------------------
# Run budget
# --------------------------------------------------------------------------

class BudgetExceededError(Exception):
    pass


@dataclass
class Budget:
    max_iterations: int = 12
    max_tool_calls: int = 20
    max_wall_time_s: float = 90.0
    _iterations: int = field(default=0, init=False)
    _tool_calls: int = field(default=0, init=False)
    _started_at: float = field(default_factory=time.time, init=False)

    def check_iteration(self) -> None:
        self._iterations += 1
        if self._iterations > self.max_iterations:
            raise BudgetExceededError(f"Exceeded max_iterations={self.max_iterations}")
        if time.time() - self._started_at > self.max_wall_time_s:
            raise BudgetExceededError(f"Exceeded max_wall_time_s={self.max_wall_time_s}")

    def check_tool_call(self) -> None:
        self._tool_calls += 1
        if self._tool_calls > self.max_tool_calls:
            raise BudgetExceededError(f"Exceeded max_tool_calls={self.max_tool_calls}")

    def snapshot(self) -> dict:
        return {
            "iterations": self._iterations,
            "tool_calls": self._tool_calls,
            "elapsed_s": round(time.time() - self._started_at, 2),
        }
