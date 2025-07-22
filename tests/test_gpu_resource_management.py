#!/usr/bin/env python3
"""
Test GPU Resource Management thread safety in AI Scientist v2
"""
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_scientist.treesearch.parallel_agent import GPUManager


class TestGPUResourceManagement(unittest.TestCase):
    """Test thread-safe GPU resource management"""

    def test_gpu_manager_initialization(self):
        """Test GPUManager initializes correctly"""
        manager = GPUManager(num_gpus=4)
        self.assertEqual(manager.num_gpus, 4)
        self.assertEqual(len(manager.available_gpus), 4)
        self.assertEqual(manager.available_gpus, {0, 1, 2, 3})
        self.assertEqual(len(manager.gpu_assignments), 0)
        self.assertIsNotNone(manager._lock)

    def test_single_gpu_acquire_release(self):
        """Test basic GPU acquisition and release"""
        manager = GPUManager(num_gpus=2)
        
        # Acquire GPU
        gpu_id = manager.acquire_gpu("process_1")
        self.assertIn(gpu_id, {0, 1})
        self.assertEqual(len(manager.available_gpus), 1)
        self.assertTrue(manager.has_gpu_assigned("process_1"))
        
        # Release GPU
        manager.release_gpu("process_1")
        self.assertEqual(len(manager.available_gpus), 2)
        self.assertFalse(manager.has_gpu_assigned("process_1"))

    def test_no_gpus_available_error(self):
        """Test error when no GPUs are available"""
        manager = GPUManager(num_gpus=1)
        
        # Acquire the only GPU
        manager.acquire_gpu("process_1")
        
        # Try to acquire another GPU
        with self.assertRaises(RuntimeError) as context:
            manager.acquire_gpu("process_2")
        self.assertEqual(str(context.exception), "No GPUs available")

    def test_concurrent_gpu_acquisition(self):
        """Test thread safety with concurrent GPU acquisitions"""
        manager = GPUManager(num_gpus=4)
        results = {}
        errors = []
        
        def acquire_gpu_worker(process_id):
            try:
                gpu_id = manager.acquire_gpu(process_id)
                results[process_id] = gpu_id
            except Exception as e:
                errors.append((process_id, str(e)))
        
        # Create multiple threads trying to acquire GPUs
        threads = []
        for i in range(6):  # More threads than GPUs
            thread = threading.Thread(
                target=acquire_gpu_worker, 
                args=(f"process_{i}",)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 4)  # Only 4 GPUs available
        self.assertEqual(len(errors), 2)   # 2 processes couldn't get GPUs
        
        # Verify all assigned GPUs are unique
        assigned_gpus = set(results.values())
        self.assertEqual(len(assigned_gpus), 4)
        self.assertEqual(assigned_gpus, {0, 1, 2, 3})

    def test_concurrent_acquire_release(self):
        """Test thread safety with concurrent acquire and release operations"""
        manager = GPUManager(num_gpus=2)
        operations_log = []
        
        def worker(worker_id):
            for i in range(5):
                try:
                    # Acquire GPU
                    gpu_id = manager.acquire_gpu(f"worker_{worker_id}")
                    operations_log.append(("acquire", worker_id, gpu_id))
                    
                    # Simulate work
                    time.sleep(0.001)
                    
                    # Release GPU
                    manager.release_gpu(f"worker_{worker_id}")
                    operations_log.append(("release", worker_id, gpu_id))
                except RuntimeError:
                    # Expected when no GPUs available
                    operations_log.append(("failed", worker_id, None))
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for future in futures:
                future.result()
        
        # Verify final state is consistent
        self.assertEqual(len(manager.available_gpus), 2)
        self.assertEqual(len(manager.gpu_assignments), 0)

    def test_get_all_assignments_thread_safety(self):
        """Test thread-safe retrieval of all GPU assignments"""
        manager = GPUManager(num_gpus=3)
        
        # Acquire some GPUs
        manager.acquire_gpu("process_1")
        manager.acquire_gpu("process_2")
        
        # Get assignments copy
        assignments = manager.get_all_assignments()
        self.assertEqual(len(assignments), 2)
        self.assertIn("process_1", assignments)
        self.assertIn("process_2", assignments)
        
        # Verify it's a copy, not the original
        assignments["fake_process"] = 99
        self.assertNotIn("fake_process", manager.gpu_assignments)

    def test_release_nonexistent_process(self):
        """Test releasing GPU for non-existent process is safe"""
        manager = GPUManager(num_gpus=2)
        
        # Should not raise error
        manager.release_gpu("non_existent_process")
        
        # State should remain unchanged
        self.assertEqual(len(manager.available_gpus), 2)

    def test_stress_concurrent_operations(self):
        """Stress test with many concurrent operations"""
        manager = GPUManager(num_gpus=8)
        completed_operations = []
        lock = threading.Lock()
        
        def stress_worker(worker_id):
            for _ in range(20):
                operation = "acquire" if len(completed_operations) % 2 == 0 else "release"
                
                if operation == "acquire":
                    try:
                        gpu_id = manager.acquire_gpu(f"worker_{worker_id}")
                        with lock:
                            completed_operations.append(("acquire", worker_id, gpu_id))
                        time.sleep(0.0001)  # Tiny delay
                        manager.release_gpu(f"worker_{worker_id}")
                        with lock:
                            completed_operations.append(("release", worker_id, gpu_id))
                    except RuntimeError:
                        # Expected when no GPUs available
                        pass
        
        # Run many workers concurrently
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(16)]
            for future in futures:
                future.result()
        
        # Verify final state
        self.assertEqual(len(manager.available_gpus), 8)
        self.assertEqual(len(manager.gpu_assignments), 0)
        
        # Verify no GPU was double-assigned
        active_gpus = {}
        for op, worker, gpu in completed_operations:
            if op == "acquire":
                self.assertNotIn(gpu, active_gpus, f"GPU {gpu} double-assigned")
                active_gpus[gpu] = worker
            elif op == "release":
                self.assertEqual(active_gpus.get(gpu), worker, 
                               f"Worker {worker} releasing GPU {gpu} it didn't own")
                del active_gpus[gpu]


if __name__ == '__main__':
    unittest.main()