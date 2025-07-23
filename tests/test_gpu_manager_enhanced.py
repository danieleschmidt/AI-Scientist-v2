"""
Enhanced GPU Resource Management Tests
Tests for the improved GPU manager with race condition fixes
"""

import unittest
import threading
import time
from unittest.mock import patch, MagicMock
import uuid
import sys
import os

# Add the ai_scientist directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_scientist.treesearch.parallel_agent import GPUManager


class TestEnhancedGPUManager(unittest.TestCase):
    """Test enhanced GPU manager with race condition fixes"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = GPUManager(num_gpus=2)

    def test_shutdown_functionality(self):
        """Test GPU manager shutdown releases all resources"""
        # Acquire both GPUs
        process_id1 = self.manager.generate_process_id()
        process_id2 = self.manager.generate_process_id()
        
        gpu1 = self.manager.acquire_gpu(process_id1)
        gpu2 = self.manager.acquire_gpu(process_id2)
        
        # Verify both GPUs are assigned
        self.assertEqual(len(self.manager.get_all_assignments()), 2)
        self.assertTrue(self.manager.has_gpu_assigned(process_id1))
        self.assertTrue(self.manager.has_gpu_assigned(process_id2))
        
        # Shutdown should release all GPUs
        self.manager.shutdown()
        
        # Verify all GPUs are released
        self.assertEqual(len(self.manager.get_all_assignments()), 0)
        self.assertFalse(self.manager.has_gpu_assigned(process_id1))
        self.assertFalse(self.manager.has_gpu_assigned(process_id2))
        
        # Verify GPUs are back in available pool
        self.assertEqual(len(self.manager.available_gpus), 2)

    def test_acquire_after_shutdown_fails(self):
        """Test that GPU acquisition fails after shutdown"""
        process_id = self.manager.generate_process_id()
        
        # Shutdown the manager
        self.manager.shutdown()
        
        # Attempting to acquire GPU should fail
        with self.assertRaises(RuntimeError) as cm:
            self.manager.acquire_gpu(process_id)
        
        self.assertIn("shutting down", str(cm.exception))

    def test_acquire_with_timeout(self):
        """Test GPU acquisition with timeout functionality"""
        # Acquire both GPUs first
        process_id1 = self.manager.generate_process_id()
        process_id2 = self.manager.generate_process_id()
        
        self.manager.acquire_gpu(process_id1)
        self.manager.acquire_gpu(process_id2)
        
        # Try to acquire third GPU with short timeout
        process_id3 = self.manager.generate_process_id()
        start_time = time.time()
        
        with self.assertRaises(RuntimeError) as cm:
            self.manager.acquire_gpu(process_id3, timeout=0.5)
        
        end_time = time.time()
        
        # Verify timeout was respected
        self.assertLessEqual(end_time - start_time, 1.0)  # Allow some margin
        self.assertIn("timeout", str(cm.exception))

    def test_release_gpu_if_assigned_atomic(self):
        """Test atomic release operation"""
        process_id = self.manager.generate_process_id()
        
        # Test release when not assigned
        result = self.manager.release_gpu_if_assigned(process_id)
        self.assertFalse(result)
        
        # Acquire GPU and test release
        gpu_id = self.manager.acquire_gpu(process_id)
        result = self.manager.release_gpu_if_assigned(process_id)
        self.assertTrue(result)
        
        # Verify GPU is released
        self.assertFalse(self.manager.has_gpu_assigned(process_id))
        self.assertIn(gpu_id, self.manager.available_gpus)

    def test_concurrent_acquire_with_timeout(self):
        """Test concurrent GPU acquisition with timeout under contention"""
        num_workers = 4
        acquire_results = {}
        error_results = {}
        
        def worker_acquire(worker_id):
            process_id = self.manager.generate_process_id()
            try:
                gpu_id = self.manager.acquire_gpu(process_id, timeout=2.0)
                acquire_results[worker_id] = (process_id, gpu_id)
                # Hold GPU for a bit
                time.sleep(0.2)
                self.manager.release_gpu_if_assigned(process_id)
            except RuntimeError as e:
                error_results[worker_id] = str(e)
        
        # Start all workers
        threads = []
        for i in range(num_workers):
            thread = threading.Thread(target=worker_acquire, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # With 2 GPUs and 4 workers, some should succeed and some should timeout
        total_results = len(acquire_results) + len(error_results)
        self.assertEqual(total_results, num_workers)
        
        # At least some should succeed
        self.assertGreater(len(acquire_results), 0)
        
        # Errors should be timeout errors
        for error in error_results.values():
            self.assertIn("timeout", error)

    def test_unique_process_id_generation(self):
        """Test that process IDs are unique to avoid race conditions"""
        process_ids = set()
        
        # Generate many process IDs
        for _ in range(1000):
            process_id = self.manager.generate_process_id()
            self.assertNotIn(process_id, process_ids, "Process ID should be unique")
            process_ids.add(process_id)
        
        # Verify format
        for process_id in process_ids:
            self.assertTrue(process_id.startswith("worker_"))
            # Extract UUID part and verify it's valid hex
            uuid_part = process_id.replace("worker_", "")
            self.assertEqual(len(uuid_part), 8)
            int(uuid_part, 16)  # Should not raise if valid hex

    def test_concurrent_shutdown_safety(self):
        """Test that shutdown is safe during concurrent operations"""
        results = {"acquired": [], "errors": []}
        
        def worker():
            try:
                process_id = self.manager.generate_process_id()
                gpu_id = self.manager.acquire_gpu(process_id, timeout=1.0)
                results["acquired"].append((process_id, gpu_id))
                time.sleep(0.1)  # Hold briefly
                self.manager.release_gpu_if_assigned(process_id)
            except RuntimeError as e:
                results["errors"].append(str(e))
        
        # Start workers
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Shutdown after a brief delay
        time.sleep(0.05)
        self.manager.shutdown()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All operations should either succeed or fail cleanly
        total_ops = len(results["acquired"]) + len(results["errors"])
        self.assertEqual(total_ops, 3)
        
        # Final state should be clean
        self.assertEqual(len(self.manager.get_all_assignments()), 0)

    def test_timeout_releases_no_resources(self):
        """Test that timeout doesn't leave resources in inconsistent state"""
        # Fill all GPU slots
        process_ids = []
        for i in range(2):
            process_id = self.manager.generate_process_id()
            self.manager.acquire_gpu(process_id)
            process_ids.append(process_id)
        
        # Record initial state
        initial_assignments = self.manager.get_all_assignments()
        initial_available = self.manager.available_gpus.copy()
        
        # Try to acquire with timeout
        process_id = self.manager.generate_process_id()
        with self.assertRaises(RuntimeError):
            self.manager.acquire_gpu(process_id, timeout=0.2)
        
        # State should be unchanged
        self.assertEqual(self.manager.get_all_assignments(), initial_assignments)
        self.assertEqual(self.manager.available_gpus, initial_available)
        
        # Cleanup
        for pid in process_ids:
            self.manager.release_gpu_if_assigned(pid)


class TestGPUManagerThreadSafety(unittest.TestCase):
    """Test thread safety of enhanced GPU manager"""

    def test_high_contention_acquire_release(self):
        """Test GPU manager under high contention"""
        manager = GPUManager(num_gpus=1)  # Single GPU for maximum contention
        results = {"acquired": 0, "released": 0, "errors": 0}
        results_lock = threading.Lock()
        
        def worker():
            for _ in range(10):
                process_id = manager.generate_process_id()
                try:
                    gpu_id = manager.acquire_gpu(process_id, timeout=1.0)
                    with results_lock:
                        results["acquired"] += 1
                    
                    # Brief hold
                    time.sleep(0.01)
                    
                    if manager.release_gpu_if_assigned(process_id):
                        with results_lock:
                            results["released"] += 1
                except RuntimeError:
                    with results_lock:
                        results["errors"] += 1
        
        # Start many workers
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify consistency
        self.assertEqual(results["acquired"], results["released"])
        self.assertEqual(len(manager.get_all_assignments()), 0)
        self.assertEqual(len(manager.available_gpus), 1)


if __name__ == '__main__':
    unittest.main()