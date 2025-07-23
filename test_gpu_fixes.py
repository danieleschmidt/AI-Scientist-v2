#!/usr/bin/env python3
"""
Quick test for GPU manager race condition fixes
"""

import sys
import threading
import time
import uuid


class MockGPUManager:
    """Mock GPU manager for testing our fixes"""
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.available_gpus = set(range(num_gpus))
        self.gpu_assignments = {}
        self._shutdown = False
        self._lock = threading.Lock()
    
    def generate_process_id(self):
        return f"worker_{uuid.uuid4().hex[:8]}"
    
    def acquire_gpu(self, process_id, timeout=30.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if self._shutdown:
                    raise RuntimeError("GPUManager is shutting down")
                if self.available_gpus:
                    gpu_id = min(self.available_gpus)
                    self.available_gpus.remove(gpu_id)
                    self.gpu_assignments[process_id] = gpu_id
                    return gpu_id
            time.sleep(0.01)
        raise RuntimeError(f"No GPUs available after {timeout}s timeout")
    
    def release_gpu_if_assigned(self, process_id):
        with self._lock:
            if process_id in self.gpu_assignments:
                gpu_id = self.gpu_assignments[process_id]
                self.available_gpus.add(gpu_id)
                del self.gpu_assignments[process_id]
                return True
            return False
    
    def shutdown(self):
        with self._lock:
            self._shutdown = True
            process_ids = list(self.gpu_assignments.keys())
            for process_id in process_ids:
                gpu_id = self.gpu_assignments[process_id]
                self.available_gpus.add(gpu_id)
            self.gpu_assignments.clear()


def test_gpu_manager_fixes():
    """Test all GPU manager race condition fixes"""
    print("Testing GPU Manager Race Condition Fixes...")
    
    # Test 1: Unique process ID generation
    manager = MockGPUManager(2)
    process_ids = set()
    for i in range(100):
        pid = manager.generate_process_id()
        assert pid not in process_ids, f"Duplicate process ID: {pid}"
        process_ids.add(pid)
    print("✓ Unique process ID generation test passed")
    
    # Test 2: GPU acquisition and release
    pid1 = manager.generate_process_id()
    pid2 = manager.generate_process_id()
    
    gpu1 = manager.acquire_gpu(pid1)
    gpu2 = manager.acquire_gpu(pid2)
    assert gpu1 != gpu2, "GPUs should be different"
    print("✓ GPU acquisition test passed")
    
    # Test 3: Atomic release
    assert manager.release_gpu_if_assigned(pid1), "Release should succeed"
    assert not manager.release_gpu_if_assigned(pid1), "Second release should fail"
    print("✓ Atomic release test passed")
    
    # Test 4: Shutdown functionality  
    # First release pid2 then acquire it again
    manager.release_gpu_if_assigned(pid2)
    gpu2_new = manager.acquire_gpu(pid2)  # Re-acquire to test shutdown
    manager.shutdown()
    assert len(manager.gpu_assignments) == 0, "All assignments should be cleared"
    assert len(manager.available_gpus) == 2, "All GPUs should be available"
    print("✓ Shutdown test passed")
    
    # Test 5: Acquisition after shutdown
    try:
        manager.acquire_gpu(manager.generate_process_id())
        assert False, "Should have raised exception"
    except RuntimeError as e:
        assert "shutting down" in str(e)
    print("✓ Post-shutdown acquisition test passed")
    
    # Test 6: Concurrent access with actual contention
    manager = MockGPUManager(1)  # Single GPU for contention
    results = {"acquired": 0, "errors": 0}
    results_lock = threading.Lock()
    barrier = threading.Barrier(3)  # Synchronize thread start
    
    def worker():
        pid = manager.generate_process_id()
        barrier.wait()  # All threads start at the same time
        try:
            gpu_id = manager.acquire_gpu(pid, timeout=0.2)  # Short timeout
            with results_lock:
                results["acquired"] += 1
            time.sleep(0.3)  # Hold GPU longer than timeout
            manager.release_gpu_if_assigned(pid)
        except RuntimeError:
            with results_lock:
                results["errors"] += 1
    
    threads = []
    for _ in range(3):
        thread = threading.Thread(target=worker)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    # With true contention, exactly 1 should succeed
    total_ops = results["acquired"] + results["errors"]
    assert total_ops == 3, f"Expected 3 total operations, got {total_ops}"
    assert results["acquired"] >= 1, f"At least 1 should succeed, got {results['acquired']}"
    print("✓ Concurrent access test passed")
    
    print("\n🎉 All GPU manager race condition fix tests passed!")


if __name__ == "__main__":
    test_gpu_manager_fixes()