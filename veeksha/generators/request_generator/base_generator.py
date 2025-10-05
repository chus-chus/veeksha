from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.response import Response
from veeksha.core.seeding import SeedManager
from veeksha.logger import init_logger

logger = init_logger(__name__)


class BaseRequestGenerator:
    def __init__(
        self,
        config: BaseRequestGeneratorConfig,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        seed_manager: Optional[SeedManager] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.seed_manager = seed_manager or SeedManager(0)

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def get_request(self) -> Tuple[RequestConfig, int, int, float]:
        """Return next request and metadata.

        Returns:
            A tuple of (request_config, prompt_token_count, output_token_count, next_timestamp_s).
        """
        raise NotImplementedError

    def get_responses(self, responses: List[Response]) -> None:
        pass

    def capacity(self) -> Optional[int]:
        """Optional: total number of requests producible if finite; None if unbounded."""
        return None

    def save_trace(self, file_path: str) -> None:
        """Optional: persist any internally generated trace. Default no-op."""
        return None
