import time
import pytest  # type: ignore[import]
from veeksha.orchestration.resource_manager import ResourceManager, NodeInfo, GPUInfo

pytestmark = pytest.mark.unit

class TestResourceManager:
    def test_initialization_no_detect(self):
        """Test initialization without GPU detection."""
        rm = ResourceManager(detect_gpus=False)
        assert len(rm.nodes) == 0
        assert rm.get_total_gpus() == 0

    def test_add_node(self):
        """Test manually adding a node."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4, gpu_memory_mb=24000)
        
        assert "node1" in rm.nodes
        node = rm.nodes["node1"]
        assert node.num_gpus == 4
        assert len(node.gpus) == 4
        assert node.is_fully_free
        assert rm.get_total_gpus() == 4
        assert rm.get_free_gpus() == 4

    def test_allocate_single_node_contiguous(self):
        """Test allocating contiguous GPUs on a single node."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=8, gpu_memory_mb=24000)
        
        # Allocate 4 GPUs
        allocation = rm.allocate_resources(num_gpus=4, job_id="job1", contiguous=True)
        
        assert allocation is not None
        assert len(allocation) == 4
        # Should be 0, 1, 2, 3
        gpu_ids = [gpu_id for _, gpu_id in allocation]
        assert gpu_ids == [0, 1, 2, 3]
        assert all(hostname == "node1" for hostname, _ in allocation)
        
        assert rm.get_free_gpus() == 4
        assert not rm.nodes["node1"].is_fully_free

    def test_allocate_single_node_non_contiguous_fallback(self):
        """Test fallback to non-contiguous allocation on single node."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4, gpu_memory_mb=24000)
        
        # Manually occupy GPU 1
        rm.nodes["node1"].gpus[1].is_free = False
        
        # Try to allocate 3 GPUs (contiguous preferred)
        # Available: 0, 2, 3. Contiguous blocks of 3: None.
        allocation = rm.allocate_resources(num_gpus=3, job_id="job1", contiguous=True)
        
        # Should fall back to non-contiguous
        assert allocation is not None
        assert len(allocation) == 3
        gpu_ids = sorted([gpu_id for _, gpu_id in allocation])
        assert gpu_ids == [0, 2, 3]

    def test_allocate_multi_node(self):
        """Test allocation across multiple nodes."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=2, gpu_memory_mb=24000)
        rm.add_node("node2", num_gpus=2, gpu_memory_mb=24000)
        
        # Request 3 GPUs, allow non-contiguous (which implies multi-node allowed)
        allocation = rm.allocate_resources(num_gpus=3, job_id="job1", contiguous=False)
        
        assert allocation is not None
        assert len(allocation) == 3
        
        # Check distribution
        node1_count = sum(1 for h, _ in allocation if h == "node1")
        node2_count = sum(1 for h, _ in allocation if h == "node2")
        assert node1_count + node2_count == 3
        assert node1_count > 0
        assert node2_count > 0

    def test_allocation_failure_insufficient_resources(self):
        """Test allocation failure when not enough GPUs."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=2, gpu_memory_mb=24000)
        
        allocation = rm.allocate_resources(num_gpus=3, job_id="job1")
        assert allocation is None

    def test_release_resources(self):
        """Test releasing allocated resources."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4, gpu_memory_mb=24000)
        
        rm.allocate_resources(num_gpus=2, job_id="job1")
        assert rm.get_free_gpus() == 2
        
        success = rm.release_resources("job1")
        assert success
        assert rm.get_free_gpus() == 4
        assert rm.nodes["node1"].is_fully_free
        
        # Release non-existent job
        assert not rm.release_resources("job_fake")

    def test_get_gpu_memory_mb(self):
        """Test calculating total memory of allocated resources."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=2, gpu_memory_mb=1000)
        
        allocation = rm.allocate_resources(num_gpus=2, job_id="job1")
        assert allocation is not None
        total_mem = rm.get_gpu_memory_mb(allocation)
        assert total_mem == 2000

    def test_wait_for_resources_immediate(self):
        """Test wait_for_resources when resources are immediately available."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=4, gpu_memory_mb=24000)
        
        allocation = rm.wait_for_resources(num_gpus=2, timeout=1.0)
        assert allocation is not None
        assert len(allocation) == 2

    def test_wait_for_resources_timeout(self):
        """Test wait_for_resources timeout."""
        rm = ResourceManager(detect_gpus=False)
        rm.add_node("node1", num_gpus=1, gpu_memory_mb=24000)

        start = time.monotonic()
        allocation = rm.wait_for_resources(num_gpus=2, timeout=0.1, poll_interval=1.0)
        elapsed = time.monotonic() - start

        assert allocation is None
        assert elapsed < 0.3
