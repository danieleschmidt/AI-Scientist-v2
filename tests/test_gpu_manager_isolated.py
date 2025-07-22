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