import time

import pytest

from veeksha.core.dispatch_scheduler import DispatchScheduler
from veeksha.core.request_config import RequestConfig
from veeksha.metrics.metric_store import MetricStore
from veeksha.config.metrics import MetricsConfig
from veeksha.metrics.request_metrics import RequestMetrics


def wait_until(predicate, timeout_s=0.5, interval_s=0.005):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


@pytest.mark.unit
def test_scheduler_first_in_session_anchor_ready() -> None:
    scheduler = DispatchScheduler()

    req = RequestConfig(
        model="dummy",
        prompt=("", 0),
        dispatch_delay=0.0,
        id=1,
        session_id=10,
        session_sequence_index=0,
        anchor_at_s=0.05,
    )

    scheduler.add_request(req)

    # Not ready immediately
    assert scheduler.pop_ready() is None

    # Becomes ready around anchor time
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)


@pytest.mark.unit
def test_scheduler_in_session_waits_for_prev_completion_then_gap() -> None:
    scheduler = DispatchScheduler()

    first = RequestConfig(
        model="dummy",
        prompt=("", 0),
        dispatch_delay=0.0,
        id=1,
        session_id=42,
        session_sequence_index=0,
        anchor_at_s=0.01,
    )
    second = RequestConfig(
        model="dummy",
        prompt=("", 0),
        dispatch_delay=0.0,
        id=2,
        session_id=42,
        session_sequence_index=1,
        wait_after_prev_response_s=0.07,
    )

    scheduler.add_request(first)
    scheduler.add_request(second)

    # First should get ready around its anchor
    assert wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)

    # Before completion, second must not be ready
    assert scheduler.pop_ready() is None

    # Notify first completion now
    scheduler.notify_completion(request_id=1, completed_at_monotonic=time.monotonic(), success=True)

    # Second should become ready after wait gap from completion
    start = time.monotonic()
    got = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.5)
    assert got, "Second request did not become ready"
    elapsed = time.monotonic() - start
    # Allow small scheduling jitter
    assert elapsed >= 0.06, f"Second released too early: elapsed={elapsed:.3f}s"


@pytest.mark.unit
def test_scheduler_cancel_session_on_failure_drops_pending() -> None:
    scheduler = DispatchScheduler()

    first = RequestConfig(
        model="dummy",
        prompt=("", 0),
        dispatch_delay=0.0,
        id=11,
        session_id=77,
        session_sequence_index=0,
        anchor_at_s=0.0,
        cancel_session_on_failure=True,
    )
    second = RequestConfig(
        model="dummy",
        prompt=("", 0),
        dispatch_delay=0.0,
        id=12,
        session_id=77,
        session_sequence_index=1,
        wait_after_prev_response_s=0.02,
        cancel_session_on_failure=True,
    )

    scheduler.add_request(first)
    scheduler.add_request(second)

    # First should be ready immediately (anchor 0)
    ready_first = wait_until(lambda: scheduler.pop_ready() is not None, timeout_s=0.2)
    assert ready_first

    # Fail the first request -> should cancel session
    scheduler.notify_completion(request_id=11, completed_at_monotonic=time.monotonic(), success=False)

    # Second must never become ready
    assert scheduler.pop_ready() is None
    time.sleep(0.05)
    assert scheduler.pop_ready() is None


@pytest.mark.unit
def test_request_level_metrics_record_errored_and_successful_requests() -> None:
    # Ensure MetricStore persists request-level arrays for both success and error
    ms = MetricStore(timeout=10, max_requests=10, metrics_config=MetricsConfig())

    # One successful request
    ok = RequestMetrics(
        request_dispatched_at=0.01,
        inter_token_times=[0.02, 0.03],
        num_prompt_tokens=10,
        num_output_tokens=2,
        error_msg=None,
        error_code=None,
        request_id=0,
    )
    ms.add_request_metrics(ok)

    # One errored request; should still appear in request-level arrays
    err = RequestMetrics(
        request_dispatched_at=0.05,
        inter_token_times=[],
        num_prompt_tokens=8,
        num_output_tokens=0,
        error_msg="boom",
        error_code=500,
        request_id=1,
    )
    ms.add_request_metrics(err)

    rlm = ms.request_level_metrics
    assert len(rlm.request_dispatched_at) == 2
    assert rlm.request_dispatched_at[0] == 0.01
    assert rlm.request_dispatched_at[1] == 0.05
