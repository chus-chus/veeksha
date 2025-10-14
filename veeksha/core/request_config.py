from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt: The prompt to provide to the LLM API.
        dispatch_delay: The delay in seconds before dispatching the request to the LLM API.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions endpoint.
        llm_api: The name of the LLM API to send the request to.
    """

    model: str
    prompt: Tuple[str, int]
    dispatch_delay: float = 0.0
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    address_append_value: Optional[str] = None
    enqueued_at: Optional[float] = None
    scheduled_timestamp: Optional[float] = None
    id: Optional[int] = None

    # session/scheduling metadata
    session_id: Optional[int] = None
    session_sequence_index: Optional[int] = None
    anchor_at_s: Optional[float] = None  # absolute anchor for first-in-session requests
    wait_after_prev_response_s: Optional[float] = None
    cancel_session_on_failure: Optional[bool] = None

    def __str__(self) -> str:
        return f"RequestConfig(id={self.id})"
