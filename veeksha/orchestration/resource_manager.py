"""
Resource manager for GPU allocation and management.

This module provides resource-aware scheduling of LLM inference servers,
enabling efficient utilization of GPU resources across multiple experiments.
"""

import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from veeksha.logger import init_logger

logger = init_logger(__name__)


ResourceMapping = List[Tuple[str, int]]  # list of (node_hostname, gpu_id)


@dataclass
class GPUInfo:
    """Information about a single GPU."""

    node_hostname: str
    gpu_id: int
    total_memory_mb: int
    is_free: bool = True


@dataclass
class NodeInfo:
    """Information about a compute node."""

    hostname: str
    num_gpus: int
    gpus: List[GPUInfo] = field(default_factory=list)
    is_fully_free: bool = True


class ResourceManager:
    """Manager for tracking and allocating GPU resources.

    This class provides resource-aware scheduling for LLM inference servers,
    enabling efficient utilization of GPUs across multiple experiments.

    Features:
    - Automatic GPU detection
    - Contiguous GPU allocation on single nodes
    - Multi-node allocation for large jobs
    - Resource tracking and cleanup
    """

    def __init__(self, detect_gpus: bool = True):
        """Initialize the resource manager.

        Args:
            detect_gpus: If True, automatically detect available GPUs using nvidia-smi
        """
        self.nodes: Dict[str, NodeInfo] = {}
        self.allocated_resources: Dict[str, ResourceMapping] = {}  # job_id -> resources
        self._lock = threading.RLock()

        if detect_gpus:
            self._detect_gpus()

    def _detect_gpus(self) -> None:
        """Detect available GPUs using nvidia-ml-py and check their memory availability."""
        try:
            # Get GPU memory info using nvidia-ml-py
            gpu_memory_info = self._get_gpu_memory_info()

            hostname = socket.gethostname()
            num_gpus = len(gpu_memory_info)

            if num_gpus > 0:
                gpus = []
                for i in range(num_gpus):
                    total_memory_mb = 0
                    is_free = True

                    if i in gpu_memory_info:
                        total_memory_mb = int(gpu_memory_info[i]["total"])
                        free_memory_mb = gpu_memory_info[i]["free"]
                        # Mark as free only if >= 90% of memory is available
                        is_free = (free_memory_mb / total_memory_mb) >= 0.90
                        if not is_free:
                            logger.warning(
                                f"GPU {i} on node {hostname} has only "
                                f"{free_memory_mb / total_memory_mb * 100:.1f}% free memory "
                                f"({free_memory_mb:.0f}/{total_memory_mb:.0f} MB), marking as unavailable"
                            )

                    gpus.append(
                        GPUInfo(
                            node_hostname=hostname,
                            gpu_id=i,
                            total_memory_mb=total_memory_mb,
                            is_free=is_free,
                        )
                    )

                self.nodes[hostname] = NodeInfo(
                    hostname=hostname,
                    num_gpus=num_gpus,
                    gpus=gpus,
                    is_fully_free=all(gpu.is_free for gpu in gpus),
                )
                free_gpus = [g for g in gpus if g.is_free]
                logger.info(
                    f"Detected {num_gpus} GPUs on node {hostname}, "
                    f"{len(free_gpus)} available (>=90% free): "
                    f"{[f'GPU{g.gpu_id}' for g in free_gpus]}"
                )
        except ImportError:
            logger.exception("nvidia-ml-py not installed. Cannot detect GPUs.")
        except Exception as e:
            logger.exception("Error detecting GPUs with nvidia-ml-py")

    def _get_gpu_memory_info(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory information using nvidia-ml-py.

        Returns:
            Dictionary mapping GPU ID to memory info (total, free, used in MB)
        """
        initialized = False
        pynvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            initialized = True
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_info = {}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_mb = int(mem_info.total) / (1024 * 1024)
                free_mb = int(mem_info.free) / (1024 * 1024)
                used_mb = int(mem_info.used) / (1024 * 1024)
                gpu_info[i] = {
                    "total": total_mb,
                    "free": free_mb,
                    "used": used_mb,
                }
            return gpu_info
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {}
        finally:
            if initialized and pynvml:
                pynvml.nvmlShutdown()

    def add_node(
        self, hostname: str, num_gpus: int, gpu_memory_mb: Optional[int] = None
    ) -> None:
        """Manually add a node with GPUs.

        Args:
            hostname: Hostname of the node
            num_gpus: Number of GPUs on the node
            gpu_memory_mb: Memory per GPU in MB (optional)
        """
        with self._lock:
            gpus = []
            for gpu_id in range(num_gpus):
                gpu_info = GPUInfo(
                    node_hostname=hostname,
                    gpu_id=gpu_id,
                    total_memory_mb=gpu_memory_mb or 0,
                    is_free=True,
                )
                gpus.append(gpu_info)

            if hostname in self.nodes:
                existing = self.nodes[hostname]
                logger.warning(
                    f"Overwriting existing node {hostname}: "
                    f"num_gpus={existing.num_gpus}, is_fully_free={existing.is_fully_free}"
                )

            self.nodes[hostname] = NodeInfo(
                hostname=hostname, num_gpus=num_gpus, gpus=gpus, is_fully_free=True
            )
            logger.info(f"Added node {hostname} with {num_gpus} GPUs")

    def get_total_gpus(self) -> int:
        """Get total number of GPUs across all nodes."""
        with self._lock:
            return sum(node.num_gpus for node in self.nodes.values())

    def get_free_gpus(self) -> int:
        """Get number of free GPUs across all nodes."""
        with self._lock:
            return sum(
                sum(1 for gpu in node.gpus if gpu.is_free)
                for node in self.nodes.values()
            )

    def allocate_resources(
        self, num_gpus: int, job_id: Optional[str] = None, contiguous: bool = True
    ) -> Optional[ResourceMapping]:
        """Allocate GPUs for a job.

        Args:
            num_gpus: Number of GPUs to allocate
            job_id: Unique identifier for the job (auto-generated if None)
            contiguous: If True, allocate contiguous GPUs on same node

        Returns:
            ResourceMapping of allocated (hostname, gpu_id) pairs, or None if allocation failed
        """
        with self._lock:
            if num_gpus <= 0:
                logger.error(f"Invalid num_gpus: {num_gpus}")
                return None

            if num_gpus > self.get_total_gpus():
                logger.error(
                    f"Requested {num_gpus} GPUs, but only {self.get_total_gpus()} available in cluster"
                )
                return None

            if num_gpus > self.get_free_gpus():
                logger.warning(
                    f"Requested {num_gpus} GPUs, but only {self.get_free_gpus()} currently free"
                )
                return None

            # try single node first
            for node in self.nodes.values():
                free_gpus = [gpu for gpu in node.gpus if gpu.is_free]

                if len(free_gpus) >= num_gpus:
                    # contiguous allocation if required
                    if contiguous:
                        allocated = self._allocate_contiguous(free_gpus, num_gpus)
                    else:
                        allocated = free_gpus[:num_gpus]

                    if allocated:
                        resource_mapping = [
                            (gpu.node_hostname, gpu.gpu_id) for gpu in allocated
                        ]

                        for gpu in allocated:
                            gpu.is_free = False

                        node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

                        if job_id is None:
                            job_id = f"job_{uuid.uuid4().hex}"
                        self.allocated_resources[job_id] = resource_mapping

                        logger.info(
                            f"Allocated {num_gpus} GPUs for {job_id}: "
                            f"{[(h, g) for h, g in resource_mapping]}"
                        )
                        return resource_mapping

            # multi-node allocation (if single-node failed and we have multiple nodes)
            if len(self.nodes) > 1 and not contiguous:
                allocated_gpus = []
                for node in self.nodes.values():
                    if len(allocated_gpus) >= num_gpus:
                        break

                    free_gpus = [gpu for gpu in node.gpus if gpu.is_free]
                    remaining_needed = num_gpus - len(allocated_gpus)
                    allocated_gpus.extend(free_gpus[:remaining_needed])

                if len(allocated_gpus) == num_gpus:
                    resource_mapping = [
                        (gpu.node_hostname, gpu.gpu_id) for gpu in allocated_gpus
                    ]

                    for gpu in allocated_gpus:
                        gpu.is_free = False

                    for node in self.nodes.values():
                        node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

                    if job_id is None:
                        job_id = f"job_{uuid.uuid4().hex}"
                    self.allocated_resources[job_id] = resource_mapping

                    logger.info(
                        f"Allocated {num_gpus} GPUs across multiple nodes for {job_id}"
                    )
                    return resource_mapping

            logger.warning(f"Could not allocate {num_gpus} GPUs")
            return None

    def get_gpu_memory_mb(self, resource_mapping: ResourceMapping) -> int:
        """Get total GPU memory for allocated resources.

        Args:
            resource_mapping: List of (hostname, gpu_id) tuples

        Returns:
            Total GPU memory in MB across all allocated GPUs
        """
        with self._lock:
            total_memory = 0
            for hostname, gpu_id in resource_mapping:
                if hostname in self.nodes:
                    node = self.nodes[hostname]
                    for gpu in node.gpus:
                        if gpu.gpu_id == gpu_id:
                            total_memory += gpu.total_memory_mb
                            break
            return total_memory

    def _allocate_contiguous(
        self, free_gpus: List[GPUInfo], num_gpus: int
    ) -> Optional[List[GPUInfo]]:
        """Try to allocate contiguous GPUs.

        Args:
            free_gpus: List of free GPUs to choose from
            num_gpus: Number of GPUs to allocate

        Returns:
            List of allocated GPUs if successful, None otherwise
        """
        sorted_gpus = sorted(free_gpus, key=lambda g: g.gpu_id)

        # find contiguous blocks
        for i in range(len(sorted_gpus) - num_gpus + 1):
            # does this starting position gives us contiguous GPUs?
            candidate = sorted_gpus[i : i + num_gpus]
            gpu_ids = [g.gpu_id for g in candidate]
            if gpu_ids == list(range(gpu_ids[0], gpu_ids[0] + num_gpus)):
                return candidate

        logger.debug(
            "Could not find contiguous GPUs, falling back to non-contiguous allocation"
        )
        return sorted_gpus[:num_gpus]

    def release_resources(self, job_id: str) -> bool:
        """Release resources allocated to a job.

        Args:
            job_id: Job identifier

        Returns:
            True if resources were released, False if job_id not found
        """
        with self._lock:
            if job_id not in self.allocated_resources:
                logger.warning(f"No allocation found for job_id: {job_id}")
                return False

            resource_mapping = self.allocated_resources[job_id]

            # free gpus
            for hostname, gpu_id in resource_mapping:
                if hostname in self.nodes:
                    node = self.nodes[hostname]
                    for gpu in node.gpus:
                        if gpu.gpu_id == gpu_id:
                            gpu.is_free = True
                            break

                    node.is_fully_free = all(gpu.is_free for gpu in node.gpus)

            del self.allocated_resources[job_id]
            return True

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status.

        Returns:
            Dictionary with resource information
        """
        with self._lock:
            status = {
                "total_nodes": len(self.nodes),
                "total_gpus": self.get_total_gpus(),
                "free_gpus": self.get_free_gpus(),
                "allocated_gpus": self.get_total_gpus() - self.get_free_gpus(),
                "active_jobs": len(self.allocated_resources),
                "nodes": {},
            }

            for hostname, node in self.nodes.items():
                node_status = {
                    "num_gpus": node.num_gpus,
                    "free_gpus": sum(1 for gpu in node.gpus if gpu.is_free),
                    "fully_free": node.is_fully_free,
                    "gpus": [
                        {
                            "gpu_id": gpu.gpu_id,
                            "free": gpu.is_free,
                            "memory_mb": gpu.total_memory_mb,
                        }
                        for gpu in node.gpus
                    ],
                }
                status["nodes"][hostname] = node_status

            return status

    def wait_for_resources(
        self,
        num_gpus: int,
        timeout: Optional[float] = None,
        poll_interval: float = 3.0,
        job_id: Optional[str] = None,
        contiguous: bool = True,
    ) -> Optional[ResourceMapping]:
        """Wait for resources to become available.

        Args:
            num_gpus: Number of GPUs needed
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            poll_interval: Time between checks in seconds
            job_id: Job identifier for allocation
            contiguous: Whether the job requires contiguous GPU IDs on a single node

        Returns:
            ResourceMapping if successful, None if timeout
        """
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time

            # check timeout before attempting allocation
            if timeout is not None and elapsed >= timeout:
                logger.warning(f"Timeout waiting for {num_gpus} GPUs after {timeout}s")
                return None

            total_gpus = self.get_total_gpus()
            free_gpus = self.get_free_gpus()

            # While waiting, avoid repeatedly logging allocation failures when the
            # cluster is too small or currently too busy to satisfy the request.
            if num_gpus <= total_gpus and num_gpus <= free_gpus:
                resource_mapping = self.allocate_resources(
                    num_gpus,
                    job_id=job_id,
                    contiguous=contiguous,
                )
                if resource_mapping:
                    return resource_mapping

            logger.debug(
                f"Waiting for {num_gpus} GPUs... (free: {free_gpus}, total: {total_gpus})"
            )

            sleep_duration = poll_interval
            if timeout is not None:
                remaining = timeout - elapsed
                sleep_duration = min(poll_interval, max(0.0, remaining))

            if sleep_duration > 0:
                time.sleep(sleep_duration)
