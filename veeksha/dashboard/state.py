from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque, Union
from collections import deque
import time
import threading
from statistics import mean

from veeksha.dashboard.events import DashboardEvent, RequestStartedEvent, TokenBatchEvent, RequestCompletedEvent, RequestErrorEvent, CapacitySearchEvent, BenchmarkStatusEvent
from veeksha.metrics.request_metrics import RequestMetrics

@dataclass
class LiveRequestInfo:
    request_id: Union[str, int]  # Can be string or int depending on source
    start_timestamp: float
    input_tokens: int
    current_output_tokens: int = 0
    ttft_ms: Optional[float] = None
    current_tpot_ms: Optional[float] = None
    progress_pct: float = 0.0
    is_waiting_first_token: bool = True

@dataclass
class AggregateStats:
    completed_count: int = 0
    error_count: int = 0
    total_requests: int = 0

    recent_ttft_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_tpot_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_tbt_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    recent_latency_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def avg_ttft_ms(self) -> float:
        return mean(self.recent_ttft_ms) if self.recent_ttft_ms else 0.0

    @property
    def avg_tpot_ms(self) -> float:
        return mean(self.recent_tpot_ms) if self.recent_tpot_ms else 0.0

    @property
    def avg_tbt_ms(self) -> float:
        return mean(self.recent_tbt_ms) if self.recent_tbt_ms else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.recent_latency_ms) if self.recent_latency_ms else 0.0

@dataclass
class CapacitySearchState:
    current_qps: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    is_under_sla: bool = False
    slo_target: str = ""
    slo_metrics: Dict[str, float] = field(default_factory=dict)
    qps_history: List[tuple[float, bool]] = field(default_factory=list)  # list of (qps, under_sla) pairs

@dataclass
class SingleBenchmarkState:
    """State for a single benchmark run"""
    benchmark_id: str
    live_requests: Dict[str, LiveRequestInfo] = field(default_factory=dict)
    completed_requests: Dict[str, LiveRequestInfo] = field(default_factory=dict)
    aggregate_stats: AggregateStats = field(default_factory=AggregateStats)
    capacity_search: CapacitySearchState = field(default_factory=CapacitySearchState)
    benchmark_start_time: Optional[float] = None
    current_qps: float = 0.0

class DashboardState:
    def __init__(self, max_live_requests: int = 50):
        self._lock = threading.RLock()
        self.max_live_requests = max_live_requests

        # Track multiple benchmarks by ID
        self.benchmarks: Dict[str, SingleBenchmarkState] = {}
        self.active_benchmark_id: str = "default"  # Currently selected benchmark

    def _get_or_create_benchmark(self, benchmark_id: str) -> SingleBenchmarkState:
        """Get existing benchmark state or create a new one"""
        if benchmark_id not in self.benchmarks:
            self.benchmarks[benchmark_id] = SingleBenchmarkState(benchmark_id=benchmark_id)
            # Set as active if it's the first benchmark
            if len(self.benchmarks) == 1:
                self.active_benchmark_id = benchmark_id
        return self.benchmarks[benchmark_id]

    def apply(self, event: DashboardEvent) -> None:
        """Apply an event to update dashboard state with locking for thread safety"""
        with self._lock:
            match event:
                case RequestStartedEvent():
                    self._handle_request_started(event)
                case TokenBatchEvent():
                    self._handle_token_batch(event)
                case RequestCompletedEvent():
                    self._handle_request_completed(event)
                case CapacitySearchEvent():
                    self._handle_capacity_search(event)
                case BenchmarkStatusEvent():
                    self._handle_benchmark_status(event)
                case RequestErrorEvent():
                    self._handle_request_error(event)
    
    def _handle_request_started(self, event: RequestStartedEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update start time, live requests, and increment total requests
        if benchmark.benchmark_start_time is None:
            benchmark.benchmark_start_time = event.timestamp

        if len(benchmark.live_requests) >= self.max_live_requests: # throttle number of live requests
            oldest_id = min(benchmark.live_requests.keys(),
                          key=lambda k: benchmark.live_requests[k].start_timestamp)
            del benchmark.live_requests[oldest_id]

        benchmark.live_requests[event.request_id] = LiveRequestInfo(
            request_id=event.request_id,
            start_timestamp=event.timestamp,
            input_tokens=event.input_tokens,
        )

        benchmark.aggregate_stats.total_requests += 1
    
    def _handle_token_batch(self, event: TokenBatchEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update live request info
        if event.request_id in benchmark.live_requests:
            req = benchmark.live_requests[event.request_id]
            req.current_output_tokens = event.total_output_tokens
            req.ttft_ms = event.ttft_ms
            req.current_tpot_ms = event.current_tpot_ms
            req.is_waiting_first_token = not event.is_first_token and req.ttft_ms is None

        # Add live metrics to aggregate stats for graphing
        # Only add TTFT on first token to avoid duplicates
        if event.is_first_token and event.ttft_ms is not None and event.ttft_ms > 0:
            benchmark.aggregate_stats.recent_ttft_ms.append(event.ttft_ms)

        # For TPOT: we update continuously as it changes with more tokens
        # This gives live updates but will have duplicate request IDs - that's okay for visualization
        if event.current_tpot_ms is not None and event.current_tpot_ms > 0:
            benchmark.aggregate_stats.recent_tpot_ms.append(event.current_tpot_ms)

        # Add individual TBT values from this batch
        if event.recent_tbt_ms:
            for tbt in event.recent_tbt_ms:
                if tbt > 0:
                    benchmark.aggregate_stats.recent_tbt_ms.append(tbt)
    
    def _handle_request_completed(self, event: RequestCompletedEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Move from live requests to completed requests + update aggregate stats
        if event.request_id in benchmark.live_requests:
            # Move to completed requests with final state
            completed_req = benchmark.live_requests[event.request_id]
            completed_req.progress_pct = 100.0  # Mark as fully completed
            completed_req.is_waiting_first_token = False

            # Update with final metrics
            metrics = event.final_metrics
            if metrics.ttft > 0:
                completed_req.ttft_ms = metrics.ttft * 1000
            if metrics.tpot > 0:
                completed_req.current_tpot_ms = metrics.tpot * 1000

            # Move to completed collection
            benchmark.completed_requests[event.request_id] = completed_req
            del benchmark.live_requests[event.request_id]

        benchmark.aggregate_stats.completed_count += 1
        metrics = event.final_metrics

        # Note: TTFT, TPOT, and TBT are already added during TokenBatchEvent streaming
        # We only add end-to-end latency here since it's only available at completion
        if metrics.end_to_end_latency > 0:
            benchmark.aggregate_stats.recent_latency_ms.append(metrics.end_to_end_latency * 1000)
    
    def _handle_capacity_search(self, event: CapacitySearchEvent) -> None:
        self.capacity_search.current_qps = event.current_qps
        self.capacity_search.current_iteration = event.iteration
        self.capacity_search.total_iterations = event.total_iterations
        self.capacity_search.is_under_sla = event.is_under_sla
        self.capacity_search.slo_target = event.slo_target
        self.capacity_search.slo_metrics = event.slo_metrics.copy()
        self.capacity_search.qps_history.append((event.current_qps, event.is_under_sla))
    
    def _handle_benchmark_status(self, event: BenchmarkStatusEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        # Update current qps and aggregate stats
        benchmark.current_qps = event.current_qps
        benchmark.aggregate_stats.total_requests = event.total_requests
        benchmark.aggregate_stats.completed_count = event.completed_requests
        benchmark.aggregate_stats.error_count = event.errored_requests

    def _handle_request_error(self, event: RequestErrorEvent) -> None:
        benchmark = self._get_or_create_benchmark(event.benchmark_id)

        if event.request_id in benchmark.live_requests:
            del benchmark.live_requests[event.request_id]
        benchmark.aggregate_stats.error_count += 1
    
    # Getter methods for stats with locking for thread safety
    def get_benchmark_ids(self) -> List[str]:
        """Get list of all benchmark IDs"""
        with self._lock:
            return list(self.benchmarks.keys())

    def get_active_benchmark(self) -> Optional[SingleBenchmarkState]:
        """Get the currently active benchmark state"""
        with self._lock:
            return self.benchmarks.get(self.active_benchmark_id)

    def set_active_benchmark(self, benchmark_id: str) -> None:
        """Set the active benchmark by ID"""
        with self._lock:
            if benchmark_id in self.benchmarks:
                self.active_benchmark_id = benchmark_id

    def get_live_requests(self, benchmark_id: Optional[str] = None) -> List[LiveRequestInfo]:
        """Get live requests for a specific benchmark or the active one"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return list(self.benchmarks[bid].live_requests.values())
            return []

    def get_all_requests(self, benchmark_id: Optional[str] = None) -> List[LiveRequestInfo]:
        """Get list of all requests (live + completed) for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                benchmark = self.benchmarks[bid]
                all_requests = []
                all_requests.extend(benchmark.live_requests.values())
                all_requests.extend(benchmark.completed_requests.values())
                return all_requests
            return []

    def get_completed_requests(self, benchmark_id: Optional[str] = None) -> List[LiveRequestInfo]:
        """Get list of completed requests for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return list(self.benchmarks[bid].completed_requests.values())
            return []

    def get_aggregate_stats(self, benchmark_id: Optional[str] = None) -> AggregateStats:
        """Get aggregate stats for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return self.benchmarks[bid].aggregate_stats
            return AggregateStats()

    def get_capacity_search_state(self, benchmark_id: Optional[str] = None) -> CapacitySearchState:
        """Get capacity search state for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                return self.benchmarks[bid].capacity_search
            return CapacitySearchState()

    def get_benchmark_duration(self, benchmark_id: Optional[str] = None) -> float:
        """Get benchmark duration for a specific benchmark"""
        with self._lock:
            bid = benchmark_id or self.active_benchmark_id
            if bid in self.benchmarks:
                benchmark = self.benchmarks[bid]
                if benchmark.benchmark_start_time is None:
                    return 0.0
                return time.time() - benchmark.benchmark_start_time
            return 0.0