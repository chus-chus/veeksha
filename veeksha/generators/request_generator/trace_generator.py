import ast
from typing import Any, Dict, List, Optional, Union, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.trace_generator import (
    TraceRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.seeding import SeedManager
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import (
    generate_random_prompt,
    load_trace,
    process_request_interval_trace,
    process_request_length_trace,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: TraceRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
        seed_manager: SeedManager,
        corpus_lines: Optional[List[str]] = None,
    ):
        from veeksha.generators.session_generator import (
            SessionGenerator,
        )

        self.config = config
        self.tokenizer = tokenizer
        self.seed_manager = seed_manager
        self.request_id = 0
        self.client_config = client_config
        self.past_prompts: Dict[int, str] = {}
        self.corpus_lines = corpus_lines
        self._remap_seed_for_save: Optional[int] = None
        self._epoch = 0
        self.prompt_rng = self.seed_manager.random("prompt")
        self.interval_rng_factory = self.seed_manager.numpy_factory("interval")
        self.session_rng_factory = self.seed_manager.numpy_factory("session")

        raw_trace_df = load_trace(self.config.trace_file)

        # canonical column names
        self.length_column_map = {
            self.config.input_length_column: "input_length",
            self.config.output_length_column: "output_length",
        }
        self.interval_column_map = {self.config.timestamp_column: "timestamp"}

        self.trace_df = raw_trace_df.pipe(
            process_request_length_trace,
            self.config.trace_file,
            self.length_column_map,
            self.config.prefill_scale_factor,
            self.config.decode_scale_factor,
            self.config.max_tokens,
        ).pipe(
            process_request_interval_trace,
            self.config.trace_file,
            self.interval_column_map,
            self.config.time_scale_factor,
            self.config.timestamp_unit,
        )

        logger.info(
            f"Loaded trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self._has_hash_ids = "hash_ids" in self.trace_df.columns

        # parse, or not, hash_ids
        if self.config.use_trace_prefix_hash_ids:
            if not self._has_hash_ids:
                raise ValueError("Trace file does not contain hash_ids")
            else:
                if not isinstance(self.trace_df["hash_ids"].iloc[0], list):
                    self.trace_df["hash_ids"] = self.trace_df["hash_ids"].apply(
                        ast.literal_eval
                    )
                if self.config.remap_hash_ids:
                    self._remap_trace_hash_ids()
        else:
            if self.corpus_lines is None:
                raise ValueError(
                    "A corpus file must be provided when not using trace prefix hash IDs."
                )

        if self.config.use_trace_sessions:
            if "session_id" not in self.trace_df.columns:
                raise ValueError("Trace file does not contain session_id of requests")
            self._annotate_trace_sessions()
        elif self.config.session_generator_config is not None:
            session_generator = SessionGenerator(
                self.config.session_generator_config,
                seed_manager=self.seed_manager.child("session"),
            )

            self.trace_df = self.trace_df.pipe(
                session_generator.generate_sessions,
            ).pipe(
                # get next request intervals again because session sampling shuffles sessions
                process_request_interval_trace,
                self.config.trace_file,
                None,  # colnames are already canonical
                self.config.time_scale_factor,
                "s",  # self.trace_df has already been converted to seconds
            )

            if self.config.session_generator_config.save_as_trace_file:
                # convert timestamps to milliseconds (default time units) before saving
                session_df_for_saving = self.trace_df.copy()
                session_df_for_saving["timestamp"] = (
                    session_df_for_saving["timestamp"] * 1000
                )
                save_suffix = f"_remapped" if self.config.remap_hash_ids else ""
                session_generator.save_requests_as_trace(
                    session_df_for_saving,
                    save_suffix=save_suffix,
                )

        self.request_idx = 0
        self._wrap_warning_logged = False

    def _annotate_trace_sessions(self) -> None:
        """Annotate trace-provided sessions with sequence index, intra-session wait, and anchor.

        Assumes `self.trace_df` has `session_id` and `timestamp` columns in seconds.
        """
        if (
            self.trace_df.empty
            or "session_id" not in self.trace_df.columns
            or "timestamp" not in self.trace_df.columns
        ):
            return

        def _annotate_group(g):
            g = g.sort_values("timestamp").copy()
            g["session_sequence_index"] = range(len(g))
            g["wait_after_prev_response_s"] = g["timestamp"].diff().fillna(0.0)
            g["anchor_at_s"] = None
            if not g.empty:
                g.loc[g.index[0], "anchor_at_s"] = float(g.iloc[0]["timestamp"])  # type: ignore
            return g

        self.trace_df = self.trace_df.groupby("session_id", group_keys=False).apply(
            _annotate_group
        )

    def _apply_session_fields(
        self,
        request_to_send,
        request_config,
        set_sequence_fields: bool,
        cancel_on_failure: Optional[bool] = None,
    ) -> None:
        """Apply session-related fields from a trace row to a RequestConfig.

        Args:
            request_to_send: Row-like object with session annotations.
            request_config: Mutable RequestConfig to populate.
            set_sequence_fields: Whether to set sequencing fields (anchor, sequence index, wait gap).
            cancel_on_failure: Optional cancel-on-failure policy to attach to the session.
        """
        session_id_val = request_to_send.get("session_id", None)
        if session_id_val is not None:
            request_config.session_id = int(session_id_val)
        if cancel_on_failure is not None:
            request_config.cancel_session_on_failure = bool(cancel_on_failure)

        if not set_sequence_fields:
            return

        seq_idx = int(request_to_send.get("session_sequence_index", 0))
        request_config.session_sequence_index = seq_idx

        anchor = request_to_send.get("anchor_at_s")
        if seq_idx == 0 and anchor is not None:
            request_config.anchor_at_s = float(anchor)

        wait_gap = float(request_to_send.get("wait_after_prev_response_s", 0.0))
        if seq_idx > 0:
            request_config.wait_after_prev_response_s = wait_gap

    def _attach_session_metadata(self, request_to_send, request_config) -> None:
        """Attach session metadata to `request_config` for both modes.

        - If `session_generator_config` is provided, use generated-session policy and
          `cancel_session_on_failure` from config and per-row annotations.
        - Else if `use_trace_sessions` is true, use only the per-row annotated
          fields (assumes `session_id` exists and `_annotate_trace_sessions` has run).
        """
        if self.config.session_generator_config is not None:
            session_policy = (
                self.config.session_generator_config.in_session_request_dispatch_policy  # type: ignore[union-attr]
            )
            cancel_on_failure = (
                self.config.session_generator_config.cancel_session_on_failure  # type: ignore[union-attr]
            )

            if session_policy == "absolute":
                # Only tag the session and cancel policy; do not set sequencing fields
                self._apply_session_fields(
                    request_to_send,
                    request_config,
                    set_sequence_fields=False,
                    cancel_on_failure=cancel_on_failure,
                )
                return

            # after_prev_response policy
            self._apply_session_fields(
                request_to_send,
                request_config,
                set_sequence_fields=True,
                cancel_on_failure=cancel_on_failure,
            )
        elif self.config.use_trace_sessions:
            # Always treat trace-provided sessions as after_prev_response
            self._apply_session_fields(
                request_to_send,
                request_config,
                set_sequence_fields=True,
                cancel_on_failure=None,
            )

    def is_stable_encoding(
        self,
        tokens: List[int],
    ) -> bool:
        return self.encode(self.decode(tokens)) == tokens

    def encode_value_as_base_52(self, value: int) -> List[int]:
        if value <= 0:
            raise ValueError(
                f"Value must be a positive integer for base-52 encoding, got: {value}"
            )

        base_52 = []
        while value > 0:
            mod = value % 52
            if mod < 26:
                base_52.append(chr(ord("a") + mod))
            else:
                base_52.append(chr(ord("A") + mod - 26))
            value = value // 52
        text_chunk = " " + "".join(base_52)
        encoding = self.encode(text_chunk)

        return encoding

    def encode_value_as_digits(self, value: int) -> List[int]:
        if value <= 0:
            raise ValueError(
                f"Value must be a positive integer for digits encoding, got: {value}"
            )

        digits = list(str(value))
        space_separated = " " + " ".join(digits)
        encoding = self.encode(space_separated)
        return encoding

    def pad_to_block_size(self, chunk: List[int], block_size: int) -> List[int]:
        final_chunk = chunk * (block_size // len(chunk) + 1)
        return final_chunk[:block_size]

    def generate_unique_encoding(self, value: int) -> List[int]:
        encoding = self.encode_value_as_base_52(value)
        if self.is_stable_encoding(encoding + encoding):
            return encoding

        encoding = self.encode_value_as_digits(value)
        if self.is_stable_encoding(encoding + encoding):
            return encoding

        raise Exception(f"Could not generate stable encoding for value {value}")

    def get_request(self) -> RequestConfig:
        if self.request_idx >= self.capacity():
            if self.config.exhaustion_policy == "error":
                raise StopIteration(
                    f"Trace exhausted for requests at index {self.request_idx}"
                )
            elif self.config.exhaustion_policy == "stop":
                # stop policy: return a sentinel request with negative dispatch delay
                logger.info(
                    f"Stop policy active: request trace exhausted at index {self.request_idx}."
                )
                return RequestConfig(
                    model=self.client_config.model,
                    prompt=("", 0),
                    dispatch_delay=-1,
                    llm_api=self.client_config.llm_api,
                    address_append_value=self.client_config.address_append_value,
                    id=self.request_idx,
                )
            elif self.config.exhaustion_policy == "wrap":
                if not self._wrap_warning_logged:
                    logger.warning(
                        f"Request trace exhausted at index {self.request_idx}; wrapping to start."
                    )
                    self._wrap_warning_logged = True
                self.request_idx = 0
                if self.config.remap_hash_ids:
                    self._epoch += 1
                    self._remap_trace_hash_ids()
                    self.past_prompts.clear()

        request_to_send = self.trace_df.iloc[self.request_idx]

        dispatch_delay = request_to_send["inter_request_time"]

        if self.config.use_trace_prefix_hash_ids:
            block_count = (
                request_to_send["input_length"] + self.config.block_size - 1
            ) // self.config.block_size

            assert (
                len(request_to_send["hash_ids"]) >= block_count
            ), f"Hash count {len(request_to_send['hash_ids'])} cannot be less than block count {block_count}"

        prompt = ""
        remaining_prompt_tokens = request_to_send["input_length"]
        use_server_min_tokens = self.client_config.min_tokens_param is not None
        instruction = ""
        if not use_server_min_tokens:
            instruction = f"Generate at least {int(request_to_send['output_length'])} tokens repeating the following text:\n"
            instruction_token_count = len(self.tokenizer.encode(instruction))
            remaining_prompt_tokens = max(
                0, remaining_prompt_tokens - instruction_token_count
            )

        if self.config.use_trace_prefix_hash_ids:
            for hash_id in request_to_send["hash_ids"]:
                if hash_id not in self.past_prompts:
                    chunk = self.generate_unique_encoding(hash_id)
                    block = self.pad_to_block_size(chunk, self.config.block_size)
                    prompt_segment = self.decode(block)
                    remaining_prompt_tokens -= self.config.block_size
                    self.past_prompts[hash_id] = prompt_segment
                prompt += self.past_prompts[hash_id]
        else:
            # generate input random text
            prompt, _ = generate_random_prompt(
                tokenizer=self.tokenizer,
                num_prompt_tokens=remaining_prompt_tokens,
                corpus_lines=self.corpus_lines,
                rng=self.prompt_rng,
            )

        prompt = (instruction + prompt) if instruction else prompt
        final_token_count = len(self.encode(prompt))

        default_sampling_params: Dict[str, Any] = {
            "max_tokens": int(request_to_send["output_length"]),
        }
        if use_server_min_tokens:
            min_token_value = int(request_to_send["output_length"])
            min_tokens_param_name = cast(str, self.client_config.min_tokens_param)
            default_sampling_params[min_tokens_param_name] = min_token_value
        # else prompt already includes instruction
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, final_token_count),
            dispatch_delay=dispatch_delay,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=self.request_idx,
        )

        # attach session scheduling metadata based on configuration
        self._attach_session_metadata(request_to_send, request_config)

        self.request_idx += 1

        return request_config

    def capacity(self) -> int:
        return len(self.trace_df)

    def _build_epoch_hash_id_map(self, unique_list: List[int]) -> Dict[int, int]:
        """Build a collision-free mapping for the current epoch."""
        rng = self.seed_manager.random("hash_remap", f"epoch_{self._epoch}")

        used: Dict[int, bool] = {}
        id_map: Dict[int, int] = {}
        for src in unique_list:
            dst = rng.getrandbits(32)
            _i = 0
            while dst == 0 or dst in used:
                dst = rng.getrandbits(32)
                _i += 1
                if _i > 1000:
                    raise RuntimeError(
                        f"Could not generate a non-colliding positive remapped ID for {src}"
                    )
            id_map[src] = int(dst)
            used[dst] = True

        return id_map

    def _remap_trace_hash_ids(self) -> None:
        """Remap prefix hash IDs in-place for the unified trace dataframe."""
        unique_ids = set()
        for ids in self.trace_df["hash_ids"]:
            unique_ids.update(ids)
        unique_list = sorted(unique_ids)
        if unique_list:
            id_map = self._build_epoch_hash_id_map(unique_list)
            logger.info("Remapping prefix hash IDs on wrap.")
            self.trace_df["hash_ids"] = self.trace_df["hash_ids"].apply(
                lambda lst: [id_map[x] for x in lst]
            )
