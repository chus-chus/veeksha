import asyncio
import os
import time
from typing import Dict, Optional, Tuple

import aiohttp

from veeksha.core.llm_clients.base_llm_client import BaseLLMClient
from veeksha.core.llm_clients.streaming_mixin import StreamingMixin
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.logger import init_logger
from veeksha.metrics.request_metrics import RequestMetrics

logger = init_logger(__name__)


class OpenAIChatCompletionsClient(BaseLLMClient, StreamingMixin):
    """Client for OpenAI Chat Completions API."""

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
    ) -> Tuple[int, int, str, float]:
        """Update metrics and generated text from a single data chunk."""
        generated_text_chunk = ""
        tokens_received_chunk = 0

        delta = data.get("choices", [{}])[0].get("delta", {})
        if content_chunk := delta.get("content"):
            (
                current_tokens_received,
                previous_token_count,
            ) = self.get_current_tokens_received(
                previous_responses=previous_responses,
                current_response=content_chunk,
                previous_token_count=previous_token_count,
            )

            tokens_received_chunk += current_tokens_received
            inter_token_times.append(time.monotonic() - most_recent_received_token_time)
            if current_tokens_received > 1:
                inter_token_times.extend([0] * (current_tokens_received - 1))

            most_recent_received_token_time = time.monotonic()
            generated_text_chunk = content_chunk

        return (
            tokens_received_chunk,
            previous_token_count,
            generated_text_chunk,
            most_recent_received_token_time,
        )

    async def send_llm_request(
        self, request_config: RequestConfig, session: aiohttp.ClientSession
    ) -> Tuple[RequestMetrics, Optional[Response]]:
        prompt, prompt_len = request_config.prompt

        message = [
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        body = {
            "model": model,
            "messages": message,
            "stream": True,
        }
        sampling_params = request_config.sampling_params
        body.update(sampling_params or {})

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        address = self.address

        if not address:
            raise ValueError("No host provided.")
        if not address.endswith("/"):
            address = address + "/"
        address += request_config.address_append_value or "chat/completions"

        inter_token_times = []
        error_msg = None
        error_response_code = None
        tokens_received = 0
        generated_text = ""
        previous_responses = []
        previous_token_count = 0

        most_recent_received_token_time = time.monotonic()
        request_dispatched_at = time.monotonic() - self.start_time
        logger.info(f"Request {request_config.id} dispatched at {request_dispatched_at}")
        # Respect a local cap on tokens to avoid mismatches with server/tokenizer
        max_tokens_limit = None
        if isinstance(request_config.sampling_params, dict):
            max_tokens_limit = request_config.sampling_params.get("max_tokens")

        try:
            async with session.post(address, json=body, headers=headers) as response:
                response.raise_for_status()

                async for data in self._process_stream(response):
                    if "error" in data:
                        err = data.get("error") or {}
                        error_msg = err.get("message", "Unknown error")
                        code_value = err.get("code")
                        error_response_code = (
                            code_value if isinstance(code_value, int) else 400
                        )
                        break  # Stop processing on error
                    delta = data["choices"][0]["delta"]
                    if delta.get("content", None):
                        (
                            current_tokens_received,
                            previous_token_count,
                        ) = self.get_current_tokens_received(
                            previous_responses=previous_responses,
                            current_response=delta["content"],
                            previous_token_count=previous_token_count,
                        )

                        # Apply local cap against requested max_tokens
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
                            inter_token_times.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                            if allowable_to_add > 1:
                                inter_token_times.extend([0] * (allowable_to_add - 1))
                            tokens_received += allowable_to_add
                            most_recent_received_token_time = time.monotonic()
                            # We still append full text; metrics cap governs counts/timings
                            generated_text += delta["content"]

                            # Truncate generated_text to exactly tokens_received tokens
                            if isinstance(max_tokens_limit, int):
                                output_token_ids = self.tokenizer.encode(generated_text)
                                if len(output_token_ids) > tokens_received:
                                    generated_text = self.tokenizer.decode(
                                        output_token_ids[:tokens_received]
                                    )

                        # If we've reached or exceeded the cap, stop processing further chunks
                        if (
                            isinstance(max_tokens_limit, int)
                            and tokens_received >= max_tokens_limit
                        ):
                            break

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
            request_dispatched_at=request_dispatched_at,
            inter_token_times=inter_token_times,
            num_prompt_tokens=prompt_len,
            num_output_tokens=tokens_received,
            error_code=error_response_code,
            error_msg=error_msg,
        )
        
        logger.info(f"Request metrics: dispatched at {request_dispatched_at}, inter token times {inter_token_times}, prompt length {prompt_len}, output length {tokens_received}")

        generated_response: Optional[Response]
        if error_msg or error_response_code:
            generated_response = None
        else:
            generated_response = Response(
                id=request_config.id,
                text=generated_text,
            )

        return metrics, generated_response
