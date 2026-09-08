"""
Unit tests for the reliability primitives, fully isolated from the LLM and
MCP layers. These exercise the exact failure modes a multi-agent tool-using
system needs to survive: transient failures, permanently broken tools,
infinite loops (exact-repeat and oscillating), and runaway resource use.

Run with: python3 -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.reliability import (
    Budget,
    BudgetExceededError,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    LoopDetectedError,
    LoopDetector,
    RetryExhausted,
    RetryPolicy,
    TerminalError,
)


# --------------------------------------------------------------------------
# RetryPolicy
# --------------------------------------------------------------------------

def test_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter_s=0.001)
    result = policy.run("test_tool", flaky, sleep=lambda s: None)
    assert result == "ok"
    assert calls["n"] == 2


def test_retry_exhausts_after_max_attempts():
    def always_fails():
        raise RuntimeError("permanently broken")

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.001, jitter_s=0.001)
    with pytest.raises(RetryExhausted) as exc_info:
        policy.run("test_tool", always_fails, sleep=lambda s: None)
    assert exc_info.value.attempts == 3


def test_terminal_error_does_not_retry():
    calls = {"n": 0}

    def bad_args():
        calls["n"] += 1
        raise TerminalError(ValueError("unknown service"))

    policy = RetryPolicy(max_attempts=5, base_delay_s=0.001)
    with pytest.raises(RetryExhausted):
        policy.run("test_tool", bad_args, sleep=lambda s: None)
    assert calls["n"] == 1, "terminal errors must not be retried"


def test_backoff_delay_grows_and_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=4.0, jitter_s=0.0)
    assert policy._delay_for(1) == 1.0
    assert policy._delay_for(2) == 2.0
    assert policy._delay_for(3) == 4.0
    assert policy._delay_for(4) == 4.0  # capped


# --------------------------------------------------------------------------
# CircuitBreaker
# --------------------------------------------------------------------------

def test_circuit_opens_after_threshold_failures():
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_s=10.0)
    for _ in range(3):
        cb.on_failure("flaky_tool", now=0.0)
    assert cb.status()["flaky_tool"] == CircuitState.OPEN.value
    with pytest.raises(CircuitOpenError):
        cb.before_call("flaky_tool", now=1.0)  # well within reset window


def test_circuit_half_opens_after_timeout_and_recloses_on_success():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=5.0)
    cb.on_failure("tool", now=0.0)
    cb.on_failure("tool", now=0.0)
    assert cb.status()["tool"] == CircuitState.OPEN.value

    # Before timeout: still open.
    with pytest.raises(CircuitOpenError):
        cb.before_call("tool", now=2.0)

    # After timeout: half-open trial allowed through.
    cb.before_call("tool", now=6.0)
    assert cb.status()["tool"] == CircuitState.HALF_OPEN.value

    cb.on_success("tool")
    assert cb.status()["tool"] == CircuitState.CLOSED.value


def test_circuit_reopens_if_half_open_trial_fails():
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=5.0)
    cb.on_failure("tool", now=0.0)
    cb.before_call("tool", now=6.0)  # half-open
    cb.on_failure("tool", now=6.0)
    assert cb.status()["tool"] == CircuitState.OPEN.value


def test_one_tools_failures_dont_affect_another_tool():
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=10.0)
    cb.on_failure("tool_a", now=0.0)
    cb.on_failure("tool_a", now=0.0)
    assert cb.status()["tool_a"] == CircuitState.OPEN.value
    cb.before_call("tool_b", now=0.0)  # must not raise


# --------------------------------------------------------------------------
# LoopDetector
# --------------------------------------------------------------------------

def test_loop_detector_allows_a_few_identical_calls():
    ld = LoopDetector(max_exact_repeats=2)
    ld.record("get_metrics", {"service": "x"})
    ld.record("get_metrics", {"service": "x"})  # 2nd repeat, still allowed


def test_loop_detector_catches_exact_repeat():
    ld = LoopDetector(max_exact_repeats=2)
    ld.record("restart_service", {"service": "x"})
    ld.record("restart_service", {"service": "x"})
    with pytest.raises(LoopDetectedError):
        ld.record("restart_service", {"service": "x"})  # 3rd in a row -> loop


def test_loop_detector_different_args_do_not_trigger():
    ld = LoopDetector(max_exact_repeats=2)
    ld.record("get_metrics", {"service": "a"})
    ld.record("get_metrics", {"service": "b"})
    ld.record("get_metrics", {"service": "c"})  # different args each time -> fine


def test_loop_detector_catches_oscillation():
    ld = LoopDetector(max_exact_repeats=5, max_cycle_repeats=3, cycle_window=8)
    # A, B, A, B, A -> on the 3rd 'B' the (A,B) cycle has repeated 3x
    ld.record("restart_service", {"service": "x"})
    ld.record("get_metrics", {"service": "x"})
    ld.record("restart_service", {"service": "x"})
    ld.record("get_metrics", {"service": "x"})
    ld.record("restart_service", {"service": "x"})
    with pytest.raises(LoopDetectedError):
        ld.record("get_metrics", {"service": "x"})


def test_loop_detector_reset_clears_history():
    ld = LoopDetector(max_exact_repeats=1)
    ld.record("tool", {})
    with pytest.raises(LoopDetectedError):
        ld.record("tool", {})
    ld.reset()
    ld.record("tool", {})  # no error after reset


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

def test_budget_enforces_max_iterations():
    b = Budget(max_iterations=3, max_tool_calls=100, max_wall_time_s=100)
    b.check_iteration()
    b.check_iteration()
    b.check_iteration()
    with pytest.raises(BudgetExceededError):
        b.check_iteration()


def test_budget_enforces_max_tool_calls():
    b = Budget(max_iterations=100, max_tool_calls=2, max_wall_time_s=100)
    b.check_tool_call()
    b.check_tool_call()
    with pytest.raises(BudgetExceededError):
        b.check_tool_call()


def test_budget_snapshot_reports_usage():
    b = Budget(max_iterations=10, max_tool_calls=10)
    b.check_iteration()
    b.check_tool_call()
    snap = b.snapshot()
    assert snap["iterations"] == 1
    assert snap["tool_calls"] == 1
    assert "elapsed_s" in snap


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
