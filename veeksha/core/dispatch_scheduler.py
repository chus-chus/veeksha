import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from veeksha.core.request_config import RequestConfig
from veeksha.logger import init_logger

logger = init_logger(__name__)


@dataclass(order=True)
class _ScheduledItem:
    ready_at: float
    request_id: int = field(compare=False)
    request: RequestConfig = field(compare=False)


class DispatchScheduler:
    """Thread-safe scheduler for dispatching requests with session-aware dependencies.

    Time base: seconds since scheduler creation (monotonic reference).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready_heap: List[_ScheduledItem] = []
        self._pending_by_session: Dict[int, Dict[int, RequestConfig]] = {}
        self._completed_seq_by_session: Dict[int, int] = {}
        self._last_completion_time_by_session: Dict[int, float] = {}
        self._canceled_sessions: Dict[int, bool] = {}
        self._cancel_policy_by_session: Dict[int, bool] = {}
        self._id_to_session_seq: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        self._start_monotonic = time.monotonic()
        self._non_session_ready_cursor: float = 0.0

    def _now(self) -> float:
        return time.monotonic() - self._start_monotonic

    def add_request(self, request: RequestConfig) -> None:
        with self._lock:
            req_id = request.id if (request.id is not None) else -1
            self._id_to_session_seq[req_id] = (
                request.session_id,
                request.session_sequence_index,
            )

            if (request.session_id is not None) and (
                request.session_sequence_index is not None
            ):
                if self._canceled_sessions.get(request.session_id, False):
                    return  # drop requests for canceled sessions
                # Remember cancel policy from any request in the session
                if request.cancel_session_on_failure is not None:
                    self._cancel_policy_by_session[request.session_id] = bool(
                        request.cancel_session_on_failure
                    )

                if request.session_sequence_index == 0:
                    # First-in-session: anchor by absolute if provided; else treat as normal delay
                    if request.anchor_at_s is not None:
                        ready_at = float(request.anchor_at_s)
                    else:
                        ready_at = self._now() + float(request.dispatch_delay)
                    heapq.heappush(
                        self._ready_heap,
                        _ScheduledItem(
                            ready_at=ready_at, request_id=int(req_id), request=request
                        ),
                    )
                else:
                    # Queue until prior is completed; then we can compute ready time
                    session_map = self._pending_by_session.setdefault(
                        request.session_id, {}
                    )
                    session_map[request.session_sequence_index] = request
                    self._maybe_release_next_locked(request.session_id)
            else:
                # Non-session request: schedule by dispatch_delay
                anchor_base = max(self._non_session_ready_cursor, self._now())
                ready_at = anchor_base + float(request.dispatch_delay)
                self._non_session_ready_cursor = ready_at
                heapq.heappush(
                    self._ready_heap,
                    _ScheduledItem(
                        ready_at=ready_at, request_id=int(req_id), request=request
                    ),
                )

    def _maybe_release_next_locked(self, session_id: int) -> None:
        # Release next-in-order pending request if its predecessor is completed
        completed_upto = self._completed_seq_by_session.get(session_id, -1)
        next_seq = completed_upto + 1
        session_map = self._pending_by_session.get(session_id)
        if not session_map:
            return
        if next_seq in session_map:
            req = session_map.pop(next_seq)
            # compute ready_at using last completion time + wait_after_prev_response_s
            last_completion = self._last_completion_time_by_session.get(session_id)
            wait = float(req.wait_after_prev_response_s or 0.0)
            if last_completion is None:
                # If predecessor completion is unknown, keep it pending
                session_map[next_seq] = req
                return
            ready_at = last_completion + wait
            next_req_id = req.id if (req.id is not None) else -1
            heapq.heappush(
                self._ready_heap,
                _ScheduledItem(
                    ready_at=ready_at, request_id=int(next_req_id), request=req
                ),
            )
            # Try to cascade only one step; caller may call again after completions

    def pop_ready(self) -> Optional[RequestConfig]:
        with self._lock:
            if not self._ready_heap:
                return None
            now = self._now()
            if self._ready_heap[0].ready_at <= now:
                item = heapq.heappop(self._ready_heap)
                return item.request
            return None

    def time_until_next_ready(self) -> Optional[float]:
        with self._lock:
            if not self._ready_heap:
                return None
            now = self._now()
            delta = self._ready_heap[0].ready_at - now
            return max(0.0, delta)

    def notify_completion(
        self, request_id: Optional[int], completed_at_monotonic: float, success: bool
    ) -> None:
        if request_id is None:
            return
        with self._lock:
            session_id, seq_idx = self._id_to_session_seq.get(request_id, (None, None))
            if session_id is None or seq_idx is None:
                return
            # Convert absolute monotonic to scheduler time base
            completed_at = completed_at_monotonic - self._start_monotonic

            if not success:
                if self._cancel_policy_by_session.get(session_id, False):
                    # Cancel remaining requests in this session
                    self._canceled_sessions[session_id] = True
                    self._pending_by_session.pop(session_id, None)
            # Mark completion
            prev_completed = self._completed_seq_by_session.get(session_id, -1)
            if seq_idx > prev_completed:
                self._completed_seq_by_session[session_id] = seq_idx
            self._last_completion_time_by_session[session_id] = completed_at
            # Try releasing next pending
            self._maybe_release_next_locked(session_id)

    def cancel_session(self, session_id: int) -> None:
        with self._lock:
            self._canceled_sessions[session_id] = True
            self._pending_by_session.pop(session_id, None)
