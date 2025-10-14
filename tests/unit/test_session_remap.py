import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.trace_generator import (
    TraceRequestGeneratorConfig,
)
from veeksha.config.generators.session_generator import SessionGeneratorConfig
from veeksha.config.generators.interval_generator.static_generator import (
    StaticRequestIntervalGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generators.request_generator.trace_generator import TraceRequestGenerator
from veeksha.logger import init_logger

logger = init_logger(__name__)

class _RoundTripTokenizer:
    """Minimal tokenizer with invertible encode/decode for ASCII text.

    This avoids depending on transformers in unit tests.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _write_trace_jsonl(tmp_path: Path) -> str:
    # Two requests with overlapping prefixes in hash_ids and increasing timestamps (ms)
    rows = [
        {
            "input_length": 64,
            "output_length": 16,
            "timestamp": 1000,
            "hash_ids": [1, 2, 1],
        },
        {
            "input_length": 64,
            "output_length": 16,
            "timestamp": 2000,
            "hash_ids": [1, 2, 2],
        },
    ]

    path = tmp_path / "trace.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _build_generator(
    trace_file: str,
    tmp_path: Path,
    *,
    remap_hash_ids: bool,
    exhaustion_policy: str = "wrap",
) -> TraceRequestGenerator:
    # Configure session generator so TraceRequestGenerator will synthesize sessions
    session_cfg = SessionGeneratorConfig(
        minimum_prefix_match=0.5,
        min_session_size=1,
        max_session_size=10,
        max_request_interval=60.0,
        save_as_trace_file=True,
        trace_file_save_dir=str(tmp_path),
        trace_file_name="session_trace_test",
        # Use static session interval to keep timestamps deterministic
        session_interval_generator_config=StaticRequestIntervalGeneratorConfig(),
    )

    cfg = TraceRequestGeneratorConfig(
        trace_file=trace_file,
        input_length_column="input_length",
        output_length_column="output_length",
        block_size=2048,  # ensure block_count = 1 for small inputs
        timestamp_column="timestamp",
        timestamp_unit="ms",
        time_scale_factor=1.0,
        use_trace_prefix_hash_ids=True,
        remap_hash_ids=remap_hash_ids,
        use_trace_sessions=False,
        session_generator_config=session_cfg,
        max_tokens=4096,
        exhaustion_policy=exhaustion_policy,
    )

    tokenizer = _RoundTripTokenizer()
    client_cfg = ClientConfig()
    sm = SeedManager(123)

    return TraceRequestGenerator(
        config=cfg,
        tokenizer=tokenizer,
        client_config=client_cfg,
        seed_manager=sm,
        corpus_lines=None,
    )


@pytest.mark.unit
def test_hash_ids_remap_on_wrap_with_sessions_and_save_suffix(tmp_path):
    trace_file = _write_trace_jsonl(tmp_path)

    gen = _build_generator(
        trace_file,
        tmp_path,
        remap_hash_ids=True,
        exhaustion_policy="wrap",
    )

    # Initial first-row hash_ids after session synthesis (running hashes) and initial remap
    initial_ids = list(gen.trace_df.iloc[0]["hash_ids"])  # type: ignore[index]

    # Drive generator to (and through) wrap: capacity is 2 after synthesis
    _ = gen.get_request()
    _ = gen.get_request()
    # Next call should trigger wrap + remap
    _ = gen.get_request()

    # After wrap, the in-place remap should change first-row hash_ids
    wrapped_ids = list(gen.trace_df.iloc[0]["hash_ids"])  # type: ignore[index]
    assert wrapped_ids != initial_ids, "Expected hash_ids to be remapped on wrap"
    assert all(i > 0 for i in wrapped_ids), "Remapped hash_ids must be positive"

    # Saved trace file should include the remapped suffix
    saved = list(Path(tmp_path).glob("**/*_remapped.jsonl"))
    assert saved, "Expected a saved trace file with '_remapped' suffix when remapping is enabled"


@pytest.mark.unit
def test_hash_ids_not_remapped_when_disabled_on_wrap(tmp_path):
    trace_file = _write_trace_jsonl(tmp_path)

    gen = _build_generator(
        trace_file,
        tmp_path,
        remap_hash_ids=False,
        exhaustion_policy="wrap",
    )

    initial_ids = list(gen.trace_df.iloc[0]["hash_ids"])  # type: ignore[index]

    _ = gen.get_request()
    _ = gen.get_request()
    _ = gen.get_request()  # triggers wrap without remap

    wrapped_ids = list(gen.trace_df.iloc[0]["hash_ids"])  # type: ignore[index]
    assert (
        wrapped_ids == initial_ids
    ), "hash_ids changed on wrap even though remap_hash_ids=False"

    # No remapped suffix should be saved when remapping is disabled
    saved_remapped = list(Path(tmp_path).glob("**/*_remapped.jsonl"))
    assert not saved_remapped, "Unexpected remapped file saved when remapping is disabled"


