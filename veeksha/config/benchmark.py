from dataclasses import field
from typing import Optional

from veeksha.config.client import ClientConfig
from veeksha.config.core.flat_dataclass import create_flat_dataclass
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.lmeval_generator import (
    LmevalRequestGeneratorConfig,
)
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.config.metrics import MetricsConfig
from veeksha.config.utils import dataclass_to_dict
from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.logger import init_logger
from veeksha.types import RequestGeneratorType

logger = init_logger(__name__)


@frozen_dataclass(allow_from_file=True)
class BenchmarkConfig:
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator."},
    )
    vajra_api_url: Optional[str] = field(
        default=None,
        metadata={"help": "The API URL for the Vajra benchmark."},
    )
    vajra_plots_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory for the Vajra plots."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run the load test for."},
    )
    max_completed_requests: int = field(
        default=10,
        metadata={
            "help": "The number of requests to complete before finishing the test. Note "
            "that its possible for the test to timeout first."
        },
    )
    api_url: Optional[str] = field(
        default="http://localhost:8000/v1",
        metadata={"help": "The API URL for the benchmark."},
    )
    api_key: Optional[str] = field(
        default="token-abc123",
        metadata={"help": "The API key for the benchmark."},
    )
    client_config: ClientConfig = field(
        default_factory=ClientConfig,
        metadata={"help": "The client configuration for the benchmark."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "The metrics configuration for the benchmark."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "The request generator configuration for the benchmark."},
    )

    def __post_init__(self):
        if self.request_generator_config.get_type() == RequestGeneratorType.LMEVAL:
            logger.warning("Removing timeout for LMEval.")
            self.timeout = -1
            assert isinstance(
                self.request_generator_config, LmevalRequestGeneratorConfig
            )

            # Import the utility function locally to avoid circular imports
            from veeksha.generators.request_generator.lmeval_generator import (
                detect_task_types,
            )

            if detect_task_types(self.request_generator_config.tasks):
                object.__setattr__(self.client_config, "llm_api", "openai_completions")
                object.__setattr__(
                    self.client_config, "address_append_value", "completions"
                )
            else:
                object.__setattr__(self.client_config, "llm_api", "openai_chat")
                object.__setattr__(
                    self.client_config, "address_append_value", "chat/completions"
                )

    @classmethod
    def create_from_cli_args(cls):
        """Create BenchmarkConfig instances from CLI

        Returns:
            List of BenchmarkConfig instances (single or
            multiple configs if YAML expands to multiple configurations)
        """
        flat_configs = create_flat_dataclass(cls).create_from_cli_args()
        instances = []
        for flat_config in flat_configs:
            instance = flat_config.reconstruct_original_dataclass()
            object.__setattr__(instance, "__flat_config__", flat_config)
            instances.append(instance)
        return instances

    def to_dict(self):
        flat_config = getattr(self, "__flat_config__", None)
        if flat_config is None:
            logger.debug("Flat config not found or is None. Using dataclass_to_dict.")
            return dataclass_to_dict(self)

        return flat_config.__dict__  # type: ignore
