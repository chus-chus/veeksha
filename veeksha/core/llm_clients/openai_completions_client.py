import asyncio
import os
import time
from typing import Dict, List, Optional, Tuple

import aiohttp

from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.core.llm_clients.streaming_mixin import StreamingMixin
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics
from veeksha.dashboard.handler import emit_dashboard_event
from veeksha.dashboard.events import TokenBatchEvent

logger = init_logger(__name__)


class OpenAICompletionsClient(BaseLLMClient, StreamingMixin):
    """Client for OpenAI Completions API.

    Behavior
    - Supports both streaming and non-streaming completions.
    - When streaming (default), yields SSE chunks and aggregates generated text.
    - When non-streaming (stream=False), expects a single JSON response.

    Logprobs handling
    - Streaming responses: collects per-chunk provider payloads and exposes them
      under Response.logprobs as {"chunks": [ {...}, ... ]}.
    - Non-streaming responses: when present, passes through the provider logprobs
      dict (e.g., {tokens, token_logprobs, top_logprobs, text_offset}).

    This format is consumed by the lmeval request generator
    """

    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        super().__init__(model_name, tokenizer_name)
        self.address = os.environ.get("OPENAI_API_BASE")
        if not self.address:
            self.address = "http://localhost:8000/v1"
            logger.warning(
                "Warning: OPENAI_API_BASE environment variable not set. Defaulting to localhost."
            )
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            self.key = ""
            logger.warning(
                "Warning: OPENAI_API_KEY environment variable not set. Defaulting to empty string."
            )
        self.start_time = time.monotonic()

    def _update_metrics_from_chunk(
        self,
        data: Dict,
        inter_token_times: list,
        previous_responses: list,
        previous_token_count: int,
        most_recent_received_token_time: float,
    ) -> Tuple[int, int, str, float, list]:
        """Update metrics and generated text from a single data chunk."""
        generated_text_chunk = ""
        tokens_received_chunk = 0
        logprobs = []

        choice = data.get("choices", [{}])[0]
        if text_chunk := choice.get("text"):
            (
                current_tokens_received,
                previous_token_count,
            ) = self.get_current_tokens_received(
                previous_responses=previous_responses,
                current_response=text_chunk,
                previous_token_count=previous_token_count,
            )

            tokens_received_chunk += current_tokens_received
            inter_token_times.append(time.monotonic() - most_recent_received_token_time)
            if current_tokens_received > 1:
                inter_token_times.extend([0] * (current_tokens_received - 1))

            most_recent_received_token_time = time.monotonic()
            generated_text_chunk = text_chunk
            if "logprobs" in choice:
                raw_logprobs = choice["logprobs"]
                if isinstance(raw_logprobs, list):
                    logprobs = [lp for lp in raw_logprobs if isinstance(lp, dict)]
                elif isinstance(raw_logprobs, dict):
                    logprobs = [raw_logprobs]
                else:
                    logprobs = []

        return (
            tokens_received_chunk,
            previous_token_count,
            generated_text_chunk,
            most_recent_received_token_time,
            logprobs,
        )

    async def send_llm_request(
        self, request_config: RequestConfig, session: aiohttp.ClientSession
    ) -> Tuple[RequestMetrics, Optional[Response]]:
        """Send a single completions request.

        Args:
            request_config: The request configuration, including prompt, model,
                and optional sampling params (e.g., stream, max_tokens, logprobs).
            session: The aiohttp client session to use.

        Returns:
            A tuple of (RequestMetrics, Optional[Response]). Response.logprobs is:
            - {"chunks": [...]} when streaming
            - provider logprobs dict when non-streaming
            - {} when unavailable
        """
        prompt, prompt_len = request_config.prompt

        # Completions API should only be used with lm_eval loglikelihood tasks.
        model = request_config.model
        body = {
            "model": model,
            "prompt": prompt,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})
        stream: bool = bool(body.get("stream", True))

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        address = self.address

        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        # Change the endpoint from "chat/completions" to "completions"
        address += request_config.address_append_value or "completions"

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        logprobs_chunks: List[Dict] = []
        non_stream_logprobs: Optional[Dict] = None
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()
        request_dispatched_at = time.monotonic() - self.start_time
        # Respect a local cap on tokens to avoid mismatches with server/tokenizer
        max_tokens_limit = None
        if isinstance(request_config.sampling_params, dict):
            max_tokens_limit = request_config.sampling_params.get("max_tokens")

        last_token_batch_emission = 0
        is_first_emission = True  # Track if this is the first dashboard event emission

        try:
            async with session.post(address, json=body, headers=headers) as response:
                response.raise_for_status()

                if not stream:
                    # expect standard OpenAI-compatible JSON
                    data = await response.json()
                    choice = (data.get("choices") or [{}])[0]
                    generated_text = choice.get("text", "") or ""
                    lp = choice.get("logprobs")
                    # logprobs expected to be a dict with
                    # tokens, token_logprobs, top_logprobs, text_offset
                    if isinstance(lp, dict):
                        non_stream_logprobs = lp
                else:
                    async for data in self._process_stream(response):
                        tokens_received_chunk = 0  # Track tokens received in this chunk
                        if "error" in data:
                            err = data.get("error") or {}
                            error_msg = err.get("message", "Unknown error")
                            code_value = err.get("code")
                            error_response_code = (
                                code_value if isinstance(code_value, int) else 400
                            )
                            break

                        text_chunk = data["choices"][0].get("text", "")
                        if text_chunk:
                            current_tokens_received, previous_token_count = (
                                self.get_current_tokens_received(
                                    previous_responses=previous_responses,
                                    current_response=text_chunk,
                                    previous_token_count=previous_token_count,
                                )
                            )
                            allowable_to_add = current_tokens_received
                            if isinstance(max_tokens_limit, int):
                                allowable_to_add = max(
                                    0,
                                    min(
                                        current_tokens_received,
                                        max_tokens_limit - tokens_received,
                                    ),
                                )
                            if allowable_to_add > 0:
                                tokens_received_chunk += allowable_to_add  # Track tokens for this chunk
                                inter_token_times.append(
                                    time.monotonic() - most_recent_received_token_time
                                )
                                if allowable_to_add > 1:
                                    inter_token_times.extend(
                                        [0] * (allowable_to_add - 1)
                                    )
                                tokens_received += allowable_to_add
                                most_recent_received_token_time = time.monotonic()
                                generated_text += text_chunk

                                # Truncate generated_text to exactly tokens_received tokens
                                if isinstance(max_tokens_limit, int):
                                    output_token_ids = self.tokenizer.encode(
                                        generated_text
                                    )
                                    if len(output_token_ids) > tokens_received:
                                        generated_text = self.tokenizer.decode(
                                            output_token_ids[:tokens_received]
                                        )

                            if (
                                isinstance(max_tokens_limit, int)
                                and tokens_received >= max_tokens_limit
                            ):
                                break

                        # Emit dashboard event every 10 tokens
                        if tokens_received_chunk > 0 and (tokens_received - last_token_batch_emission) >= 10:
                            current_ttft_ms = inter_token_times[0] * 1000 if inter_token_times else None
                            current_tpot_ms = None
                            recent_tbt_ms = None
                            if len(inter_token_times) > 1:
                                from statistics import mean
                                current_tpot_ms = mean(inter_token_times[1:]) * 1000
                                # Get recent TBT values (last 10 or all available)
                                recent_tbt_ms = [t * 1000 for t in inter_token_times[1:][-10:]]

                            emit_dashboard_event(TokenBatchEvent(
                                request_id=request_config.id,
                                timestamp=time.time(),
                                tokens_received_this_batch=tokens_received_chunk,
                                total_output_tokens=tokens_received,
                                ttft_ms=current_ttft_ms,
                                current_tpot_ms=current_tpot_ms,
                                is_first_token=is_first_emission,
                                recent_tbt_ms=recent_tbt_ms,
                                benchmark_id=request_config.benchmark_id
                            ))
                            last_token_batch_emission = tokens_received
                            is_first_emission = False  # After first emission, no longer first

                        if "logprobs" in data["choices"][0]:
                            raw_lp = data["choices"][0]["logprobs"]
                            if isinstance(raw_lp, list):
                                logprobs_chunks = [
                                    lp for lp in raw_lp if isinstance(lp, dict)
                                ]
                            elif isinstance(raw_lp, dict):
                                logprobs_chunks.append(raw_lp)

        except aiohttp.ClientResponseError as e:
            error_response_code = e.status
            error_msg = error_msg or (e.message if hasattr(e, "message") else str(e))
            logger.warning(f"HTTP Error: status={error_response_code} msg={error_msg}")
        except aiohttp.ClientConnectorError as e:
            error_response_code = 503
            error_msg = error_msg or str(e)
            logger.warning(f"Connection Error: ({error_response_code}) {error_msg}")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            error_response_code = 408
            error_msg = error_msg or "Request timed out"
            logger.warning(f"Timeout Error: ({error_response_code}) {error_msg}")
        except Exception as e:
            error_response_code = error_response_code or 520
            error_msg = error_msg or str(e)
            logger.exception(
                f"An unexpected error occurred: ({error_response_code}) {error_msg}"
            )

        metrics = RequestMetrics(
            request_id=request_config.id,
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
            benchmark_id=request_config.benchmark_id,
        )

        generated_response: Optional[Response]
        if error_msg or error_response_code:
            generated_response = None
        else:
            generated_response = Response(
                id=request_config.id,
                text=generated_text,
                logprobs=(
                    {"chunks": logprobs_chunks}
                    if stream
                    else (non_stream_logprobs or {})
                ),
            )

        return metrics, generated_response
