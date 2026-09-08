"""
Structured tracing for the multi-agent run.

Every meaningful event (agent started, LLM call, tool call, retry, circuit
trip, loop detected, escalation, handoff between agents) is written as one
JSON object per line to a run-scoped trace file, plus mirrored to stdout in
a human-readable form. This is what you'd point a recruiter or an SRE at
after an incident to answer "what did the agent actually do and why."
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional


class Tracer:
    def __init__(self, run_id: Optional[str] = None, out_dir: str = "traces", echo: bool = True):
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
        self.echo = echo
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.path = Path(out_dir) / f"{self.run_id}.jsonl"
        self._t0 = time.time()
        self._depth = 0

    def _write(self, record: dict) -> None:
        record = {"ts": round(time.time() - self._t0, 3), "run_id": self.run_id, **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        if self.echo:
            self._echo(record)

    def _echo(self, record: dict) -> None:
        indent = "  " * self._depth
        kind = record.get("event", "?")
        label_map = {
            "agent_start": "▶ AGENT",
            "agent_end": "■ AGENT DONE",
            "llm_call": "🧠 LLM",
            "tool_call": "🔧 TOOL",
            "tool_result": "✅ RESULT",
            "tool_error": "⚠️  TOOL ERROR",
            "retry": "↻ RETRY",
            "circuit_open": "⛔ CIRCUIT OPEN",
            "loop_detected": "🔁 LOOP DETECTED",
            "budget_exceeded": "⏱  BUDGET EXCEEDED",
            "escalation": "🚨 ESCALATION",
            "handoff": "➡  HANDOFF",
            "decision": "🧭 DECISION",
        }
        label = label_map.get(kind, kind.upper())
        detail = record.get("detail", "")
        print(f"[{record['ts']:>6.2f}s] {indent}{label}: {detail}", file=sys.stderr)

    def event(self, event: str, detail: str = "", **fields: Any) -> None:
        self._write({"event": event, "detail": detail, **fields})

    @contextmanager
    def span(self, event_start: str, event_end: str, detail: str = "", **fields: Any):
        self.event(event_start, detail, **fields)
        self._depth += 1
        t0 = time.time()
        try:
            yield
        finally:
            self._depth -= 1
            self.event(event_end, detail, duration_s=round(time.time() - t0, 3), **fields)
