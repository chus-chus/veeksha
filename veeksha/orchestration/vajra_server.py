from typing import List

from veeksha.config.server import VajraServerConfig
from veeksha.logger import init_logger
from veeksha.orchestration.server_manager import BaseServerManager

logger = init_logger(__name__)


class VajraServerManager(BaseServerManager):
    """Manager for Vajra inference servers."""

    def __init__(self, config: VajraServerConfig, output_dir: str):
        super().__init__(config, output_dir=output_dir)

        if config.engine.lower() != "vajra":
            logger.warning(
                f"VajraServerManager created with engine='{config.engine}'. "
                "Expected 'vajra'"
            )

    def _build_launch_command(self) -> List[str]:
        # Placeholder for Vajra command as it was not fully implemented in previous steps
        # Assuming similar structure
        command = [
            "python",
            "-m",
            "vajra_server.server",
            "--model",
            self.config.model,
            "--host",
            self.config.host,
            "--port",
            str(self.config.port),
        ]

        if self.config.tensor_parallel_size > 1:
            command.extend(
                ["--tensor-parallel-size", str(self.config.tensor_parallel_size)]
            )

        if self.config.api_key:
            command.extend(["--api-key", self.config.api_key])

        if self.config.max_model_len:
            command.extend(["--max-model-len", str(self.config.max_model_len)])

        additional_args_dict = self.get_additional_args_dict()
        for key, value in additional_args_dict.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            elif isinstance(value, list):
                for item in value:
                    command.extend([f"--{key}", str(item)])
            elif value is not None:
                command.extend([f"--{key}", str(value)])

        return command
