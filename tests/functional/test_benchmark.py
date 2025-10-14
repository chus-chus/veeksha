"""Functional tests for veeksha benchmark functionality."""

import json
from pathlib import Path

import pytest

from .template_utils import create_benchmark_config
from .test_utils import BenchmarkTestRunner


@pytest.mark.functional
class TestBenchmarkFunctionality:
    """Test benchmark functionality with various configurations."""

    @pytest.mark.gpu
    def test_benchmark_with_poisson_interval_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Poisson request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="poisson",
            prefill_tokens=10,
            decode_tokens=5,
            qps=0.5,
        )

        runner.run_benchmark(config_content, "poisson_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_gamma_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Gamma request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="uniform",
            interval_generator_type="gamma",
            min_tokens=5,
            max_tokens=10,
            qps=0.5,
            cv=0.5,
        )

        runner.run_benchmark(config_content, "gamma_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_static_interval_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with static request interval generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            request_generator_type="synthetic",
            length_generator_type="fixed",
            interval_generator_type="static",
            prefill_tokens=10,
            decode_tokens=5,
            duration=0.05,
        )

        runner.run_benchmark(config_content, "static_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_zipf_length_generator(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with Zipf request length generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            request_generator_type="synthetic",
            length_generator_type="zipf",
            interval_generator_type="static",
            min_tokens=5,
            max_tokens=10,
            duration=0.05,
            theta=1.0,
            scramble=True,
        )

        runner.run_benchmark(config_content, "zipf_config.yml")

    @pytest.mark.gpu
    def test_benchmark_with_trace_length_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with trace request length generator."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="synthetic",
            length_generator_type="trace",
            interval_generator_type="static",
            trace_file=sample_trace_file,
            max_tokens=128,
            duration=0.05,
        )

        runner.run_benchmark(config_content, "trace_config.yml")

    def test_benchmark_config_validation(self, temp_output_dir: str) -> None:
        """Test benchmark configuration validation."""
        runner = BenchmarkTestRunner(temp_output_dir)
        # Create invalid config with missing required fields
        config_content = """
timeout: 30
max_completed_requests: 1
# Missing client_config and request_generator_config
"""
        # This should fail due to missing config
        runner.run_benchmark(
            config_content,
            "invalid_config.yml",
            timeout=10,
            expected_return_code=1,
            check_output_files=False
        )

    @pytest.mark.gpu
    def test_benchmark_lmeval_logit_task(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with lmeval request generator using a logit-based task (no extra deps)."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=60,  # will be overridden to -1 for lmeval
            request_generator_type="lmeval",
            interval_generator_type="static",
            duration=0.05,
            lmeval_tasks=["hellaswag"],
            lmeval_num_fewshot=0,
            lmeval_limit=1,
        )

        runner.run_benchmark(config_content, "lmeval_logit.yml")

        # Verify lmeval results presence and structure
        results_files = list(Path(temp_output_dir).glob("**/lmeval_results.json"))
        assert len(results_files) > 0, "lmeval_results.json not found in output"
        data = json.loads(results_files[0].read_text())
        for key in [
            "results",
            "configs",
            "versions",
            "n-shot",
            "higher_is_better",
            "n-samples",
        ]:
            assert key in data, f"Missing key in lmeval_results.json: {key}"
        assert isinstance(data["results"], dict), "results must be a dict"

    @pytest.mark.gpu
    def test_benchmark_lmeval_generation_task(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test benchmark with lmeval request generator using a generation-based task (no extra deps)."""
        runner = BenchmarkTestRunner(temp_output_dir)
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=60,
            request_generator_type="lmeval",
            interval_generator_type="static",
            duration=0.05,
            lmeval_tasks=["triviaqa"],
            lmeval_num_fewshot=0,
            lmeval_limit=1,
        )

        runner.run_benchmark(
            config_content,
            "lmeval_gen.yml",
            env={"HF_DATASETS_TRUST_REMOTE_CODE": "1"}
        )

        # Verify lmeval results presence and structure
        results_files = list(Path(temp_output_dir).glob("**/lmeval_results.json"))
        assert len(results_files) > 0, "lmeval_results.json not found in output"
        data = json.loads(results_files[0].read_text())
        for key in [
            "results",
            "configs",
            "versions",
            "n-shot",
            "higher_is_better",
            "n-samples",
        ]:
            assert key in data, f"Missing key in lmeval_results.json: {key}"
        assert isinstance(data["results"], dict), "results must be a dict"

    @pytest.mark.gpu
    def test_benchmark_with_trace_sessions_session_generator(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Trace generator using prefix hash ids and synthesized sessions."""
        runner = BenchmarkTestRunner(temp_output_dir)
        session_gen_cfg = {
            "seed": 123,
            "minimum_prefix_match": 0.5,
            "min_session_size": 1,
            "max_session_size": 5,
            "max_request_interval": 60.0,
            "session_interval_generator_config": {"type": "static", "duration": 0.1},
            # Enable trace saving to validate fields
            "save_as_trace_file": True,
            "trace_file_save_dir": temp_output_dir,
            "trace_file_name": "session_trace_test",
        }
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=30,
            request_generator_type="trace",
            trace_file="",
            max_tokens=512,
            trace_input_length_column="input_length",
            trace_output_length_column="output_length",
            trace_use_prefix_hash_ids=True,
            trace_remap_hash_ids=False,
            trace_use_sessions=False,
            # Use a large block size so required block_count <= len(hash_ids)
            trace_block_size=2048,
            session_generator_config=session_gen_cfg,
            interval_generator_type="static",
            duration=0.05,
        )

        runner.run_benchmark(config_content, "trace_sessions_session_gen.yml")

        # Validate saved session trace fields (including session scheduling annotations)
        jsonl_files = list(Path(temp_output_dir).glob("**/*.jsonl"))
        assert len(jsonl_files) > 0, "No saved session trace found"
        trace_path = jsonl_files[0]
        records = []
        with trace_path.open() as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        assert len(records) > 0, "Saved session trace is empty"

        required_keys = {
            "session_id",
            "prefix_match_n_hashes",
            "prefix_match_pct",
            "cummulative_prefix_match_pct",
            "hash_ids",
            "timestamp",
        }
        missing = required_keys - set(records[0].keys())
        assert not missing, f"Missing keys in saved session trace: {missing}"

        # At least one request has a non-zero prefix match
        assert any(r.get("prefix_match_n_hashes", 0) > 0 for r in records), (
            "Expected at least one request to have prefix cache match > 0"
        )

        # New session scheduling fields should exist (generated by TraceRequestGenerator)
        sched_keys = {"session_sequence_index", "wait_after_prev_response_s", "anchor_at_s"}
        missing_sched = sched_keys - set(records[0].keys())
        assert not missing_sched, f"Missing scheduling keys in saved session trace: {missing_sched}"

    @pytest.mark.gpu
    def test_benchmark_with_trace_sessions_from_trace(
        self, temp_output_dir: str, sample_trace_file: str, test_model: str, vllm_server
    ) -> None:
        """Trace generator using sessions already present in the trace file."""
        runner = BenchmarkTestRunner(temp_output_dir)
        # This test ensures config wiring; if default trace lacks session_id, generator should raise.
        config_content = create_benchmark_config(
            model=test_model,
            output_dir=temp_output_dir,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            request_generator_type="trace",
            trace_file="",
            max_tokens=256,
            trace_input_length_column="input_length",
            trace_output_length_column="output_length",
            trace_use_prefix_hash_ids=True,
            trace_use_sessions=True,
            interval_generator_type="static",
            duration=0.05,
        )

        runner.run_benchmark(
            config_content,
            "trace_sessions_from_trace.yml",
            expected_return_code=1,
            check_output_files=False,
        )

