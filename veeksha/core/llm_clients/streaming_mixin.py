import json
from typing import AsyncGenerator, Dict, List, Protocol, Tuple, cast
import time

import aiohttp

from veeksha.logger import init_logger

logger = init_logger(__name__)


class _HasTokenLength(Protocol):
    def get_token_length(self, text: str) -> int: ...


class StreamingMixin:
    """Shared streaming and token-count helpers for LLM clients.

    Requires the consumer class to implement `get_token_length(text: str) -> int`.
    """

    MAX_RESPONSES_ALLOWED_TO_STORE: int = 5

    async def _process_stream(
        self, response: aiohttp.ClientResponse
    ) -> AsyncGenerator[Tuple[Dict, float], None]:
        """Process Server-Sent Events (SSE) stream and yield parsed JSON dicts.

        Skips non-data lines and stops on "[DONE]" sentinel.
        """
        buffer = b""
        async for chunk_bytes in response.content.iter_any():
            buffer += chunk_bytes
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode("utf-8").strip()

                if not line or not line.startswith("data:"):
                    # Skip empty lines, comments, etc.
                    continue

                payload = line[len("data:") :].strip()
                arrival_ts = time.monotonic()
                if payload == "[DONE]":
                    return

                try:
                    yield json.loads(payload), arrival_ts
                except json.JSONDecodeError:
                    logger.exception(f"JSON decode error with chunk: {payload}")

    def total_tokens(self, response_list: List[str]) -> int:
        """Return token length of concatenated responses using client's tokenizer."""
        merged_content = "".join(response_list)
        provider = cast(_HasTokenLength, self)
        return provider.get_token_length(merged_content)

    def get_current_tokens_received(
        self,
        previous_responses: List[str],
        current_response: str,
        previous_token_count: int,
    ) -> Tuple[int, int]:
        """Update rolling token counters for streamed text.

        Returns a tuple of (new_tokens_in_this_chunk, updated_previous_token_count).
        """
        previous_responses.append(current_response)
        total_count_now = self.total_tokens(previous_responses)
        current_tokens_received = max(0, total_count_now - previous_token_count)
        if len(previous_responses) > self.MAX_RESPONSES_ALLOWED_TO_STORE:
            previous_responses.pop(0)
        # Accumulate total token count without re-tokenizing the window again
        previous_token_count = previous_token_count + current_tokens_received
        return current_tokens_received, previous_token_count
