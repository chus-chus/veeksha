import os
from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.session_generator import (
    SessionGeneratorConfig,
)
from veeksha.config.utils import get_trace_file_path
from veeksha.constants.configuration_constants import (
    ALLOWED_EXHAUSTION_POLICIES,
    ALLOWED_TS_UNITS,
)
from veeksha.types.request_generator_type import RequestGeneratorType

_DATA_FILE_PATH = get_trace_file_path("swe_agent_trace_short.jsonl")

DEFAULT_TRACE_FILE = str(_DATA_FILE_PATH)


@frozen_dataclass(allow_from_file=True)
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    exhaustion_policy: str = field(
        default="stop",
        metadata={
            "help": "Behavior when the trace runs out: error | stop | wrap.",
        },
    )
    trace_file: str = field(
        default=DEFAULT_TRACE_FILE,
        metadata={"help": "Path to the trace file for request generation."},
    )
    input_length_column: str = field(
        default="num_prefill_tokens",
        metadata={"help": "Name of the column containing the input (prefill) length."},
    )
    output_length_column: str = field(
        default="num_decode_tokens",
        metadata={"help": "Name of the column containing the output (decode) length."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum number of tokens allowed in a request."},
    )
    prefill_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for prefill tokens."}
    )
    decode_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for decode tokens."}
    )
    block_size: int = field(
        default=512, metadata={"help": "Number of tokens per block."}
    )
    timestamp_column: str = field(
        default="timestamp",
        metadata={"help": "Name of the column containing request timestamps."},
    )
    timestamp_unit: str = field(
        default="ms",
        metadata={
            "help": f"Unit of the timestamps in the trace file. Must be in {sorted(ALLOWED_TS_UNITS)}."
        },
    )
    time_scale_factor: float = field(
        default=1, metadata={"help": "Scale factor for request dispatch intervals."}
    )
    use_trace_prefix_hash_ids: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, veeksha will use prefix hash IDs of requests to generate request prompts. Trace file specified by interval or/and length generator must include hash_ids: list[int]."
        },
    )
    remap_hash_ids: bool = field(
        default=False,
        metadata={
            "help": "If True and use_trace_prefix_hash_ids, randomly remap prefix hash IDs. Will also remap hash_ids when trace is exhausted and exhaustion_policy is set to 'wrap'.",
        },
    )
    use_trace_sessions: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, veeksha will use sessions provided in the trace file (session_id: int)."
        },
    )
    session_generator_config: Optional[SessionGeneratorConfig] = field(
        default=None,
        metadata={
            "help": "If not None, it will synthesize sessions based on the trace file and prefix hash IDs of requests (requires use_trace_prefix_hash_ids to be True)."
        },
    )

    def __post_init__(self):
        # Check if trace file exists, handling package resources
        if self.trace_file == DEFAULT_TRACE_FILE:
            # For the default path, use the is_file() method on the importlib.resources object
            if not _DATA_FILE_PATH.is_file():
                raise FileNotFoundError(
                    f"{self.__class__.__name__}: Default trace file resource not found."
                )
        else:
            # For user-provided paths, use os.path.exists
            if not os.path.exists(self.trace_file):
                raise FileNotFoundError(
                    f"{self.__class__.__name__}: Trace file not found: {self.trace_file}"
                )
        # factors cannot be negative
        if self.prefill_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: prefill_scale_factor cannot be negative"
            )
        if self.decode_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: decode_scale_factor cannot be negative"
            )
        if self.input_length_column == self.output_length_column:
            raise ValueError(
                f"{self.__class__.__name__}: input_length_column and output_length_column must differ"
            )
        if not (self.max_tokens == -1 or self.max_tokens > 0):
            raise ValueError(
                f"{self.__class__.__name__}: max_tokens must be -1 (no cap) or > 0; got {self.max_tokens}"
            )
        if self.time_scale_factor < 0:
            raise ValueError(
                f"{self.__class__.__name__}: time_scale_factor cannot be negative"
            )
        if self.timestamp_unit not in ALLOWED_TS_UNITS:
            raise ValueError(
                f"{self.__class__.__name__}: timestamp_unit must be in {sorted(ALLOWED_TS_UNITS)}"
            )
        # block_size must be > 0
        if self.block_size <= 0:
            raise ValueError(f"{self.__class__.__name__}: block_size must be positive")

        # session_generator_config and use_trace_sessions cannot both be provided
        if self.session_generator_config and self.use_trace_sessions:
            raise ValueError(
                f"{self.__class__.__name__}: session_generator_config and use_trace_sessions cannot both be provided"
            )
        # if session_generator_config is provided, use_trace_prefix_hash_ids must be True
        if self.session_generator_config and not self.use_trace_prefix_hash_ids:
            raise ValueError(
                f"{self.__class__.__name__}: session_generator_config requires use_trace_prefix_hash_ids to be True"
            )
        if self.exhaustion_policy not in ALLOWED_EXHAUSTION_POLICIES:
            raise ValueError(
                f"{self.__class__.__name__}: exhaustion_policy must be one of {sorted(ALLOWED_EXHAUSTION_POLICIES)}"
            )
        if self.remap_hash_ids and not self.use_trace_prefix_hash_ids:
            raise ValueError(
                f"{self.__class__.__name__}: remap_hash_ids requires use_trace_prefix_hash_ids to be True"
            )

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.TRACE
