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
    recent_latency_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def avg_ttft_ms(self) -> float:
        return mean(self.recent_ttft_ms) if self.recent_ttft_ms else 0.0
    
    @property
    def avg_tpot_ms(self) -> float:
        return mean(self.recent_tpot_ms) if self.recent_tpot_ms else 0.0
        
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

class DashboardState:
    def __init__(self, max_live_requests: int = 50):
        self._lock = threading.RLock()
        self.max_live_requests = max_live_requests
        
        # State tracking:
        self.live_requests: Dict[str, LiveRequestInfo] = {}
        self.completed_requests: Dict[str, LiveRequestInfo] = {}
        self.aggregate_stats = AggregateStats()
        self.capacity_search = CapacitySearchState()
        self.benchmark_start_time: Optional[float] = None
        self.current_qps: float = 0.0
        
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
        # Update start time, live requests, and increment total requests
        if self.benchmark_start_time is None:
            self.benchmark_start_time = event.timestamp
            
        if len(self.live_requests) >= self.max_live_requests: # throttle number of live requests
            oldest_id = min(self.live_requests.keys(), 
                          key=lambda k: self.live_requests[k].start_timestamp)
            del self.live_requests[oldest_id]
        
        self.live_requests[event.request_id] = LiveRequestInfo(
            request_id=event.request_id,
            start_timestamp=event.timestamp,
            input_tokens=event.input_tokens,
        )
        
        self.aggregate_stats.total_requests += 1
    
    def _handle_token_batch(self, event: TokenBatchEvent) -> None:
        # Update live request info
        if event.request_id in self.live_requests:
            req = self.live_requests[event.request_id]
            req.current_output_tokens = event.total_output_tokens
            req.ttft_ms = event.ttft_ms
            req.current_tpot_ms = event.current_tpot_ms
            req.is_waiting_first_token = not event.is_first_token and req.ttft_ms is None
            
            # TODO: how to calculate progress?
    
    def _handle_request_completed(self, event: RequestCompletedEvent) -> None:
        # Move from live requests to completed requests + update aggregate stats
        if event.request_id in self.live_requests:
            # Move to completed requests with final state
            completed_req = self.live_requests[event.request_id]
            completed_req.progress_pct = 100.0  # Mark as fully completed
            completed_req.is_waiting_first_token = False
            
            # Update with final metrics
            metrics = event.final_metrics
            if metrics.ttft > 0:
                completed_req.ttft_ms = metrics.ttft * 1000
            if metrics.tpot > 0:
                completed_req.current_tpot_ms = metrics.tpot * 1000
            
            # Move to completed collection
            self.completed_requests[event.request_id] = completed_req
            del self.live_requests[event.request_id]
        
        self.aggregate_stats.completed_count += 1
        metrics = event.final_metrics
        
        if metrics.ttft > 0:
            self.aggregate_stats.recent_ttft_ms.append(metrics.ttft * 1000)
        if metrics.tpot > 0:
            self.aggregate_stats.recent_tpot_ms.append(metrics.tpot * 1000)
        if metrics.end_to_end_latency > 0:
            self.aggregate_stats.recent_latency_ms.append(metrics.end_to_end_latency * 1000)
    
    def _handle_capacity_search(self, event: CapacitySearchEvent) -> None:
        self.capacity_search.current_qps = event.current_qps
        self.capacity_search.current_iteration = event.iteration
        self.capacity_search.total_iterations = event.total_iterations
        self.capacity_search.is_under_sla = event.is_under_sla
        self.capacity_search.slo_target = event.slo_target
        self.capacity_search.slo_metrics = event.slo_metrics.copy()
        self.capacity_search.qps_history.append((event.current_qps, event.is_under_sla))
    
    def _handle_benchmark_status(self, event: BenchmarkStatusEvent) -> None:
        # Update current qps and aggregate stats
        self.current_qps = event.current_qps
        self.aggregate_stats.total_requests = event.total_requests
        self.aggregate_stats.completed_count = event.completed_requests
        self.aggregate_stats.error_count = event.errored_requests

    def _handle_request_error(self, event: RequestErrorEvent) -> None:
        if event.request_id in self.live_requests:
            del self.live_requests[event.request_id]
        self.aggregate_stats.error_count += 1
    
    # Getter methods for stats with locking for thread safety"
    def get_live_requests(self) -> List[LiveRequestInfo]:
        with self._lock:
            return list(self.live_requests.values())
    
    def get_all_requests(self) -> List[LiveRequestInfo]:
        """Get list of all requests (live + completed)"""
        with self._lock:
            all_requests = []
            all_requests.extend(self.live_requests.values())
            all_requests.extend(self.completed_requests.values())
            return all_requests
    
    def get_completed_requests(self) -> List[LiveRequestInfo]:
        """Get list of completed requests"""
        with self._lock:
            return list(self.completed_requests.values())
    
    def get_aggregate_stats(self) -> AggregateStats:
        with self._lock:
            return self.aggregate_stats
    
    def get_capacity_search_state(self) -> CapacitySearchState:
        with self._lock:
            return self.capacity_search
    
    def get_benchmark_duration(self) -> float:
        with self._lock:
            if self.benchmark_start_time is None:
                return 0.0
            return time.time() - self.benchmark_start_time