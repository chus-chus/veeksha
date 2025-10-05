import threading
import time
from typing import Dict, Optional

from veeksha.config.metrics import MetricsConfig
from veeksha.metrics.metric_store import MetricStore
from veeksha.metrics.request_metrics import RequestMetrics


class ServiceMetrics:
    def __init__(
        self,
        timeout: float,
        max_requests: int,
        metrics_config: MetricsConfig,
    ) -> None:
        self.timeout = timeout
        self.max_requests = max_requests
        self.start_time = None
        self.end_time = None
        self.output_dir = metrics_config.output_dir
        self._stop_event = threading.Event()
        self._error_lock = threading.Lock()
        self._error_event = threading.Event()
        self._error: Optional[BaseException] = None

        self.metric_store = MetricStore(
            timeout=timeout,
            max_requests=max_requests,
            metrics_config=metrics_config,
        )

    @property
    def num_requests(self) -> int:
        return self.metric_store.num_requests

    @property
    def num_completed_requests(self) -> int:
        return self.metric_store.num_completed_requests

    @property
    def num_errored_requests(self) -> int:
        return self.metric_store.num_errored_requests

    @property
    def duration(self):
        assert self.end_time is not None
        assert self.start_time is not None

        return self.end_time - self.start_time

    @property
    def completed_requests_per_min(self):
        return self.num_completed_requests / self.duration * 60

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()

    def should_stop(self):
        assert self.start_time is not None
        # never stop due to timeout if timeout is -1
        if self._stop_event.is_set() or self._error_event.is_set():
            return True
        if self.timeout == -1:
            return not (self.num_completed_requests < self.max_requests)
        return not (
            time.perf_counter() - self.start_time < self.timeout
            and self.num_completed_requests < self.max_requests
        )

    def register_launched_request(self):
        self.metric_store.register_launched_request()

    def add_request_metrics(self, request_metrics: RequestMetrics):
        self.metric_store.add_request_metrics(request_metrics)

    def get_duration_summary(self) -> Dict[str, float]:
        return {
            "Duration": self.duration,
            "Completed Requests Per Min": self.completed_requests_per_min,
        }

    def get_aggregated_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_aggregated_summary(),
            **self.get_duration_summary(),
        }

    def get_summary(self) -> Dict[str, float]:
        return {
            **self.metric_store.get_summary(),
            **self.get_duration_summary(),
        }

    def __str__(self) -> str:
        return "\n".join(
            [f"{k}: {v:.5f}" for k, v in self.get_aggregated_summary().items()]
            + [str(summary) for summary in self.metric_store.summaries.values()]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def store_output(self):
        self.metric_store.store_output(self.output_dir)
        
    def print_terminal_table(self):
        self.metric_store.print_terminal_table()

    def request_stop(self) -> None:
        """Signal main loop to stop as soon as possible."""
        self._stop_event.set()

    def notify_error(self, exc: BaseException) -> None:
        """Record an error and request stop; main loop can re-raise."""
        with self._error_lock:
            self._error = exc
        self._error_event.set()
        self._stop_event.set()

    @property
    def error(self) -> Optional[BaseException]:
        with self._error_lock:
            return self._error

    @property
    def stop_requested(self) -> bool:
        """Return True if a stop has been requested (thread-safe)."""
        return self._stop_event.is_set()
