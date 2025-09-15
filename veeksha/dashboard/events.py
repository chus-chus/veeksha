from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import time
from veeksha.metrics.request_metrics import RequestMetrics

@dataclass
class RequestStartedEvent:
    request_id: str
    timestamp: float
    input_tokens: int
    expected_output_tokens: int

@dataclass  
class TokenBatchEvent:
    request_id: str
    timestamp: float
    tokens_received_this_batch: int
    total_output_tokens: int
    ttft_ms: Optional[float] = None
    current_tpot_ms: Optional[float] = None
    is_first_token: bool = False

@dataclass
class RequestCompletedEvent:
    request_id: str
    timestamp: float
    final_metrics: RequestMetrics

@dataclass
class CapacitySearchEvent:
    current_qps: float
    is_under_sla: bool
    slo_metrics: Dict[str, float]
    slo_target: str
    iteration: int
    total_iterations: int

@dataclass
class BenchmarkStatusEvent:
    total_requests: int
    completed_requests: int
    errored_requests: int
    active_requests: int
    current_qps: float
    elapsed_time: float

@dataclass
class RequestErrorEvent:
    request_id: str
    timestamp: float
    error_code: Optional[int]
    error_msg: str

DashboardEvent = Union[
    RequestStartedEvent, 
    TokenBatchEvent,
    RequestCompletedEvent,
    RequestErrorEvent,
    CapacitySearchEvent, 
    BenchmarkStatusEvent
]