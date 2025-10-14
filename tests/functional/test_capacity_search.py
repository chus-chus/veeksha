"""Functional tests for veeksha capacity search functionality."""

from pathlib import Path

import pytest

from .template_utils import create_capacity_search_config
from .test_utils import (
    CapacitySearchTestRunner,
    create_standard_slos,
    create_slo_config,
    verify_cache_behavior
)


@pytest.mark.functional
class TestCapacitySearch:
    """Test capacity search functionality with various SLO types."""

    @pytest.mark.gpu
    def test_capacity_search_tbt_ttft_slo(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with TBT-TTFT based SLO."""
        runner = CapacitySearchTestRunner(temp_output_dir)
        slos = create_standard_slos(ttft_value=1.0, tbt_value=0.1, percentile=0.9)

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/capacity_search_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            max_iterations=2,
            prompt_length=10,
            output_length=5,
        )

        runner.run_capacity_search(
            config_content,
            "capacity_search_config.yml",
            "capacity_search_results"
        )

    @pytest.mark.gpu
    def test_capacity_search_ttft_tpot_slo(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with TTFT-TPOT based SLO."""
        runner = CapacitySearchTestRunner(temp_output_dir)
        slos = create_standard_slos(ttft_value=1.0, tpot_value=0.1, percentile=0.9)

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/capacity_search_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            max_iterations=2,
            prompt_length=10,
            output_length=5,
        )

        runner.run_capacity_search(
            config_content,
            "capacity_search_config_tpot.yml",
            "capacity_search_results"
        )

    @pytest.mark.gpu
    def test_capacity_search_with_custom_slos(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test capacity search with custom SLO configurations."""
        runner = CapacitySearchTestRunner(temp_output_dir)
        slos = [
            create_slo_config("ttft", 0.5, 0.5, name="P50 TTFT"),
            create_slo_config("tbt", 0.05, 0.95, name="P95 TBT"),
            create_slo_config("tpot", 0.02, 0.99, name="P99 TPOT"),
        ]

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=f"{temp_output_dir}/custom_slos_results",
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=20,
            max_iterations=2,
            prompt_length=10,
            output_length=5,
        )

        runner.run_capacity_search(
            config_content,
            "custom_slos_config.yml",
            "custom_slos_results"
        )

    @pytest.mark.gpu
    def test_capacity_search_caching(
        self, temp_output_dir: str, test_model: str, vllm_server
    ) -> None:
        """Test that capacity search properly caches results between runs."""
        runner = CapacitySearchTestRunner(temp_output_dir)
        cache_dir = Path(temp_output_dir) / "cache_test_results"

        slos = create_standard_slos(ttft_value=1.0, percentile=0.9)

        config_content = create_capacity_search_config(
            model=test_model,
            output_dir=str(cache_dir),
            slos=slos,
            api_url=vllm_server.base_url,
            max_completed_requests=3,
            timeout=15,
            max_iterations=2,
            prompt_length=10,
            output_length=5,
        )

        verify_cache_behavior(runner, config_content, cache_dir)