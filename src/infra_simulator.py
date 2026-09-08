"""
A small, deterministic (but seedable) infrastructure simulator.

Why simulate instead of hitting real Prometheus/Kubernetes/Datadog?
Because a portfolio piece needs to run for anyone who clones it, without
cloud credentials -- but the *tool layer* (mcp_server.py) is written
against a real MCP server and would work unchanged against a real backend;
only this module would be swapped for real API clients (kubernetes-client,
prometheus-api-client, etc). That boundary is intentional and called out
in the README.

State lives in SQLite so it's inspectable, persists across a run, and
tool calls genuinely mutate it (e.g. `restart_service` really does reset
the error rate and bump a restart counter) rather than returning canned
strings.
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SERVICES = ["api-gateway", "payments-service", "orders-db", "auth-service", "notifications-worker"]


@dataclass
class ScenarioInjection:
    """A scripted fault so the agent has something real to diagnose."""
    service: str
    fault: str  # "memory_leak_crashloop" | "bad_deploy_latency" | "db_connection_exhaustion"
    severity: str = "high"


class InfraSimulator:
    def __init__(self, db_path: str = "infra_state.db", seed: Optional[int] = None):
        self.db_path = db_path
        self.rng = random.Random(seed)
        Path(db_path).unlink(missing_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
        self._seed_services()

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE services (
                name TEXT PRIMARY KEY,
                status TEXT,
                error_rate REAL,
                latency_p99_ms REAL,
                replicas INTEGER,
                deployed_version TEXT,
                restart_count INTEGER DEFAULT 0,
                crashloop INTEGER DEFAULT 0
            );
            CREATE TABLE logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT,
                ts REAL,
                level TEXT,
                message TEXT
            );
            CREATE TABLE incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT,
                severity TEXT,
                created_ts REAL,
                status TEXT DEFAULT 'open'
            );
            CREATE TABLE actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                service TEXT,
                action TEXT,
                ts REAL,
                result TEXT
            );
            """
        )
        self.conn.commit()

    def _seed_services(self) -> None:
        for svc in SERVICES:
            self.conn.execute(
                "INSERT INTO services (name, status, error_rate, latency_p99_ms, replicas, deployed_version) "
                "VALUES (?, 'healthy', ?, ?, ?, ?)",
                (svc, round(self.rng.uniform(0.001, 0.01), 4), round(self.rng.uniform(80, 150), 1), 3, "v1.4.2"),
            )
            for _ in range(3):
                self._log(svc, "INFO", f"{svc} heartbeat ok", ts_offset=self.rng.uniform(-300, 0))
        self.conn.commit()

    def _log(self, service: str, level: str, message: str, ts_offset: float = 0.0) -> None:
        self.conn.execute(
            "INSERT INTO logs (service, ts, level, message) VALUES (?, ?, ?, ?)",
            (service, time.time() + ts_offset, level, message),
        )

    # -- scenario injection -------------------------------------------------

    def inject(self, s: ScenarioInjection) -> None:
        if s.fault == "memory_leak_crashloop":
            self.conn.execute(
                "UPDATE services SET status='degraded', error_rate=0.42, latency_p99_ms=2200, crashloop=1 "
                "WHERE name=?",
                (s.service,),
            )
            for msg in [
                "OOMKilled: container exceeded memory limit (512Mi)",
                "CrashLoopBackOff: back-off restarting failed container",
                "java.lang.OutOfMemoryError: Java heap space",
                "readiness probe failed: connection refused",
            ]:
                self._log(s.service, "ERROR", msg)
        elif s.fault == "bad_deploy_latency":
            self.conn.execute(
                "UPDATE services SET status='degraded', latency_p99_ms=4100, error_rate=0.08, "
                "deployed_version='v1.5.0-canary' WHERE name=?",
                (s.service,),
            )
            for msg in [
                "deploy v1.5.0-canary rolled out to 100% (expected 10%)",
                "p99 latency spike detected: 4100ms (baseline 120ms)",
                "N+1 query pattern detected in /checkout handler",
            ]:
                self._log(s.service, "WARN", msg)
        elif s.fault == "db_connection_exhaustion":
            self.conn.execute(
                "UPDATE services SET status='critical', error_rate=0.61, latency_p99_ms=6000 WHERE name=?",
                (s.service,),
            )
            for msg in [
                "FATAL: remaining connection slots reserved for superuser",
                "connection pool exhausted: 100/100 in use",
                "could not acquire connection within 5000ms",
            ]:
                self._log(s.service, "ERROR", msg)
        self.conn.execute(
            "INSERT INTO incidents (summary, severity, created_ts) VALUES (?, ?, ?)",
            (f"{s.service}: {s.fault}", s.severity, time.time()),
        )
        self.conn.commit()

    # -- read operations (used by MCP tools) --------------------------------

    def get_service_health(self, service: str) -> dict:
        row = self.conn.execute(
            "SELECT status, error_rate, latency_p99_ms, replicas, deployed_version, restart_count, crashloop "
            "FROM services WHERE name=?",
            (service,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown service: {service}")
        keys = ["status", "error_rate", "latency_p99_ms", "replicas", "deployed_version", "restart_count", "crashloop"]
        return dict(zip(keys, row))

    def list_services(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT name FROM services")]

    def get_recent_logs(self, service: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, level, message FROM logs WHERE service=? ORDER BY ts DESC LIMIT ?",
            (service, limit),
        ).fetchall()
        return [{"ts": ts, "level": lvl, "message": msg} for ts, lvl, msg in rows]

    def get_metrics(self, service: str) -> dict:
        h = self.get_service_health(service)
        return {
            "service": service,
            "error_rate": h["error_rate"],
            "latency_p99_ms": h["latency_p99_ms"],
            "replicas": h["replicas"],
        }

    def list_incidents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, summary, severity, status FROM incidents ORDER BY id DESC"
        ).fetchall()
        return [{"id": i, "summary": s, "severity": sev, "status": st} for i, s, sev, st in rows]

    # -- write operations (mutating remediation actions) ---------------------

    def restart_service(self, service: str) -> dict:
        h = self.get_service_health(service)
        # Restart fixes memory-leak crashloops reliably; does nothing for bad
        # deploys or DB exhaustion (realistic: restart isn't a silver bullet).
        if h["crashloop"]:
            self.conn.execute(
                "UPDATE services SET status='healthy', error_rate=0.005, latency_p99_ms=95, "
                "crashloop=0, restart_count=restart_count+1 WHERE name=?",
                (service,),
            )
            result = "success"
            self._log(service, "INFO", "container restarted, memory reset, readiness probe passing")
        else:
            self.conn.execute("UPDATE services SET restart_count=restart_count+1 WHERE name=?", (service,))
            result = "no_effect"
            self._log(service, "WARN", "restart completed but root cause persists")
        self.conn.execute(
            "INSERT INTO actions (service, action, ts, result) VALUES (?, 'restart_service', ?, ?)",
            (service, time.time(), result),
        )
        self.conn.commit()
        return {"service": service, "action": "restart_service", "result": result, **self.get_service_health(service)}

    def rollback_deployment(self, service: str) -> dict:
        h = self.get_service_health(service)
        if "canary" in h["deployed_version"] or h["latency_p99_ms"] > 1000:
            self.conn.execute(
                "UPDATE services SET status='healthy', latency_p99_ms=110, error_rate=0.005, "
                "deployed_version='v1.4.2' WHERE name=?",
                (service,),
            )
            result = "success"
            self._log(service, "INFO", "rolled back to v1.4.2, latency normalized")
        else:
            result = "no_effect"
        self.conn.execute(
            "INSERT INTO actions (service, action, ts, result) VALUES (?, 'rollback_deployment', ?, ?)",
            (service, time.time(), result),
        )
        self.conn.commit()
        return {"service": service, "action": "rollback_deployment", "result": result, **self.get_service_health(service)}

    def scale_deployment(self, service: str, replicas: int) -> dict:
        self.conn.execute("UPDATE services SET replicas=? WHERE name=?", (replicas, service))
        self.conn.execute(
            "INSERT INTO actions (service, action, ts, result) VALUES (?, 'scale_deployment', ?, 'success')",
            (service, time.time()),
        )
        self.conn.commit()
        return {"service": service, "action": "scale_deployment", "replicas": replicas, "result": "success"}

    def create_incident_ticket(self, summary: str, severity: str) -> dict:
        cur = self.conn.execute(
            "INSERT INTO incidents (summary, severity, created_ts) VALUES (?, ?, ?)",
            (summary, severity, time.time()),
        )
        self.conn.commit()
        return {"incident_id": cur.lastrowid, "summary": summary, "severity": severity, "status": "open"}
