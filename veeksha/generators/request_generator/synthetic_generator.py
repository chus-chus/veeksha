from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.seeding import SeedManager
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.generators.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import generate_random_prompt
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: SyntheticRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
        seed_manager: SeedManager,
        corpus_lines: Optional[List[str]] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.seed_manager = seed_manager

        self.client_config = client_config
        sm = self.seed_manager
        self.interval_rng_factory = sm.numpy_factory("interval")
        self.length_rng_factory = sm.numpy_factory("length")
        self.prompt_rng = sm.random("prompt")

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
            rng=self.length_rng_factory(),
        )
        self.requests_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
            rng=self.interval_rng_factory(),
        )
        self.corpus_lines = corpus_lines

        self.request_id = 0

        # Internal trace state
        self._trace: List[Dict[str, Any]] = []
        self._timestamp_accum_s: float = 0.0

    def get_request(self) -> RequestConfig:
        (
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        dispatch_delay = self.requests_interval_generator.get_next_inter_request_time()
        # graceful stop if any generator signals stop via sentinel values
        if num_prompt_tokens < 0 or num_output_tokens < 0 or dispatch_delay < 0:
            return RequestConfig(
                model=self.client_config.model,
                prompt=("", 0),
                dispatch_delay=-1,
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self.request_id,
            )
        num_prompt_tokens = int(num_prompt_tokens)
        num_output_tokens = int(num_output_tokens)
        # Use server-side min_tokens if available (probing already validated support)
        use_server_min_tokens = self.client_config.min_tokens_param is not None
        instruction = ""
        if not use_server_min_tokens:
            instruction = f"Generate at least {num_output_tokens} tokens repeating the following text:\n"
            instruction_token_count = len(self.tokenizer.encode(instruction))
            body_token_count = max(0, num_prompt_tokens - instruction_token_count)
        else:
            body_token_count = max(0, num_prompt_tokens)

        prompt_body, _ = generate_random_prompt(
            tokenizer=self.tokenizer,
            num_prompt_tokens=body_token_count,
            corpus_lines=self.corpus_lines,
            rng=self.prompt_rng,
        )

        prompt = (instruction + prompt_body) if instruction else prompt_body
        prompt_token_count = len(self.tokenizer.encode(prompt))

        default_sampling_params: Dict[str, Any] = {
            "max_tokens": num_output_tokens,
        }
        if use_server_min_tokens:
            min_token_value = num_output_tokens
            min_tokens_param_name = cast(str, self.client_config.min_tokens_param)
            default_sampling_params[min_tokens_param_name] = min_token_value
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, prompt_token_count),
            dispatch_delay=dispatch_delay,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=self.request_id,
        )

        # Accumulate trace row
        next_timestamp = self._timestamp_accum_s + float(dispatch_delay)
        self._trace.append(
            {
                "prompt": prompt,
                "input_length": int(prompt_token_count),
                "output_length": int(num_output_tokens),
                "timestamp": next_timestamp,
            }
        )
        self._timestamp_accum_s = next_timestamp

        self.request_id += 1

        return request_config

    def save_trace(self, file_path: str, timestamp_unit: str = "ms") -> None:
        """Save the internally generated trace to a file.

        The saved dataframe has columns: prompt, input_length, output_length, timestamp.

        Args:
            file_path: Destination path. Supports .jsonl and .csv extensions.
            timestamp_unit: Unit for timestamps in the saved file. "s" or "ms".
        """
        if not self._trace:
            logger.info("No generated requests to save; trace is empty.")
            return

        df = pd.DataFrame(self._trace)
        if timestamp_unit == "ms":
            df["timestamp"] = df["timestamp"] * 1000.0
        elif timestamp_unit != "s":
            raise ValueError(f"Unsupported timestamp_unit: {timestamp_unit}")

        # Ensure column order
        df = df[["prompt", "input_length", "output_length", "timestamp"]]

        if file_path.endswith(".jsonl"):
            df.to_json(file_path, orient="records", lines=True)
        elif file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            raise ValueError(
                f"Unsupported file extension for {file_path}; use .jsonl or .csv"
            )
