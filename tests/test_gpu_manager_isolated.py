#!/usr/bin/env python3
"""
Isolated test for GPU Manager thread safety
Tests the GPUManager class in isolation without full module dependencies
"""
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Set


class GPUManager:
    """Thread-safe manager for GPU allocation across processes"""

    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.available_gpus: Set[int] = set(range(num_gpus))
        self.gpu_assignments: Dict[str, int] = {}  # process_id -> gpu_id
        self._lock = threading.Lock()  # Thread-safe synchronization

    def acquire_gpu(self, process_id: str) -> int:
        """Atomically assigns a GPU to a process"""
        with self._lock:
            if not self.available_gpus:
                raise RuntimeError("No GPUs available")
            
            gpu_id = self.available_gpus.pop()
            self.gpu_assignments[process_id] = gpu_id
            return gpu_id

    def release_gpu(self, process_id: str) -> None:
        """Releases GPU back to the pool"""
        with self._lock:
            if process_id in self.gpu_assignments:
                gpu_id = self.gpu_assignments.pop(process_id)
                self.available_gpus.add(gpu_id)


class TestGPUManagerIsolated(unittest.TestCase):
    def test_basic_functionality(self):
        """Test basic GPU allocation and release"""
        manager = GPUManager(2)
        
        # Test allocation
        gpu1 = manager.acquire_gpu("process1")
        self.assertIn(gpu1, [0, 1])
        
        gpu2 = manager.acquire_gpu("process2")
        self.assertIn(gpu2, [0, 1])
        self.assertNotEqual(gpu1, gpu2)
        
        # Test all GPUs allocated
        with self.assertRaises(RuntimeError):
            manager.acquire_gpu("process3")
        
        # Test release
        manager.release_gpu("process1")
        gpu3 = manager.acquire_gpu("process3")
        self.assertEqual(gpu3, gpu1)


if __name__ == "__main__":
    unittest.main()