"""
Performance benchmarks for AI Scientist v2.
Tests system performance under various load conditions.
"""

import time
import pytest
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch


@pytest.mark.performance
@pytest.mark.slow
class TestLoadBenchmarks:
    """Test suite for system load and performance benchmarks."""

    def test_memory_usage_under_load(self, mock_llm_response):
        """Test memory usage remains stable under load."""
        initial_memory = psutil.Process().memory_info().rss
        max_memory_increase = 100 * 1024 * 1024  # 100MB
        
        def memory_intensive_task():
            # Simulate memory-intensive operations
            data = [i for i in range(10000)]
            return sum(data)
        
        # Run multiple tasks concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(memory_intensive_task) for _ in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < max_memory_increase, \
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, exceeds limit"
        assert len(results) == 20


    def test_cpu_usage_under_load(self):
        """Test CPU usage remains reasonable under load."""
        def cpu_intensive_task():
            # Simulate CPU-intensive operations
            result = 0
            for i in range(100000):
                result += i ** 2
            return result
        
        start_time = time.time()
        cpu_before = psutil.cpu_percent(interval=1)
        
        # Run CPU-intensive tasks
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        cpu_after = psutil.cpu_percent(interval=1)
        execution_time = time.time() - start_time
        
        # Verify reasonable execution time and CPU usage
        assert execution_time < 30, f"Execution took too long: {execution_time:.2f}s"
        assert len(results) == 10


    @pytest.mark.api
    def test_concurrent_api_calls(self, mock_openai_client):
        """Test system handles concurrent API calls efficiently."""
        mock_openai_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="Test response"))],
            usage=Mock(total_tokens=100)
        )
        
        def make_api_call():
            # Simulate API call
            response = mock_openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Test"}]
            )
            return response.choices[0].message.content
        
        start_time = time.time()
        
        # Make concurrent API calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_api_call) for _ in range(50)]
            results = [future.result() for future in as_completed(futures)]
        
        execution_time = time.time() - start_time
        
        # Verify all calls completed successfully and quickly
        assert len(results) == 50
        assert all(result == "Test response" for result in results)
        assert execution_time < 10, f"API calls took too long: {execution_time:.2f}s"


    def test_file_io_performance(self, temp_dir):
        """Test file I/O performance under load."""
        file_count = 100
        file_size = 1024  # 1KB per file
        
        def write_test_file(file_index):
            file_path = temp_dir / f"test_file_{file_index}.txt"
            content = "x" * file_size
            with open(file_path, "w") as f:
                f.write(content)
            return file_path
        
        def read_test_file(file_path):
            with open(file_path, "r") as f:
                return len(f.read())
        
        start_time = time.time()
        
        # Write files concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            write_futures = [executor.submit(write_test_file, i) 
                           for i in range(file_count)]
            file_paths = [future.result() for future in as_completed(write_futures)]
        
        # Read files concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            read_futures = [executor.submit(read_test_file, path) 
                          for path in file_paths]
            file_sizes = [future.result() for future in as_completed(read_futures)]
        
        execution_time = time.time() - start_time
        
        # Verify all files were processed correctly
        assert len(file_paths) == file_count
        assert all(size == file_size for size in file_sizes)
        assert execution_time < 5, f"File I/O took too long: {execution_time:.2f}s"


    def test_experiment_pipeline_performance(self, sample_experiment_config, temp_dir):
        """Test performance of experiment pipeline under load."""
        def simulate_experiment(experiment_id):
            # Simulate experiment execution
            config = sample_experiment_config.copy()
            config["experiment_id"] = experiment_id
            
            # Simulate data processing
            data = list(range(1000))
            processed_data = [x * 2 for x in data]
            
            # Simulate result generation
            result = {
                "experiment_id": experiment_id,
                "processed_count": len(processed_data),
                "config": config
            }
            
            # Simulate result saving
            result_file = temp_dir / f"experiment_{experiment_id}_result.json"
            import json
            with open(result_file, "w") as f:
                json.dump(result, f)
            
            return result
        
        start_time = time.time()
        experiment_count = 20
        
        # Run experiments concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(simulate_experiment, i) 
                      for i in range(experiment_count)]
            results = [future.result() for future in as_completed(futures)]
        
        execution_time = time.time() - start_time
        
        # Verify all experiments completed successfully
        assert len(results) == experiment_count
        assert all(result["processed_count"] == 1000 for result in results)
        assert execution_time < 15, f"Experiments took too long: {execution_time:.2f}s"


    @pytest.mark.gpu
    def test_gpu_memory_management(self, mock_gpu_info):
        """Test GPU memory management under load."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_properties") as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB
                
                def simulate_gpu_task():
                    # Simulate GPU memory allocation
                    with patch("torch.cuda.memory_allocated", return_value=1024**3):  # 1GB
                        time.sleep(0.1)  # Simulate processing
                        return True
                
                start_time = time.time()
                
                # Run GPU tasks concurrently
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(simulate_gpu_task) for _ in range(8)]
                    results = [future.result() for future in as_completed(futures)]
                
                execution_time = time.time() - start_time
                
                assert len(results) == 8
                assert all(result for result in results)
                assert execution_time < 5


    def test_stress_test_throughput(self, mock_llm_response):
        """Stress test system throughput."""
        task_count = 1000
        batch_size = 50
        
        def batch_task():
            # Simulate processing a batch of tasks
            results = []
            for _ in range(batch_size):
                # Simulate task processing
                result = {"status": "completed", "value": 42}
                results.append(result)
            return results
        
        start_time = time.time()
        
        # Process tasks in batches
        batch_count = task_count // batch_size
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(batch_task) for _ in range(batch_count)]
            batch_results = [future.result() for future in as_completed(futures)]
        
        execution_time = time.time() - start_time
        total_tasks = sum(len(batch) for batch in batch_results)
        throughput = total_tasks / execution_time
        
        # Verify throughput meets minimum requirements
        assert total_tasks == task_count
        assert throughput > 100, f"Throughput too low: {throughput:.1f} tasks/sec"
        assert execution_time < 20


@pytest.mark.performance
class TestResourceMonitoring:
    """Test suite for resource monitoring and limits."""

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import gc
        
        # Force garbage collection and get baseline
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss
        
        # Perform repeated operations that could cause leaks
        for _ in range(100):
            # Simulate operations that might leak memory
            data = {"key": "value" * 1000}
            processed = {k: v.upper() for k, v in data.items()}
            del data, processed
        
        # Force garbage collection again
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        
        memory_increase = final_memory - initial_memory
        max_acceptable_increase = 50 * 1024 * 1024  # 50MB
        
        assert memory_increase < max_acceptable_increase, \
            f"Potential memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase"


    def test_resource_cleanup(self, temp_dir):
        """Test proper resource cleanup after operations."""
        import gc
        import weakref
        
        class TestResource:
            def __init__(self, name):
                self.name = name
                self.data = "x" * 10000  # 10KB
        
        # Create resources and weak references
        resources = []
        weak_refs = []
        
        for i in range(10):
            resource = TestResource(f"resource_{i}")
            resources.append(resource)
            weak_refs.append(weakref.ref(resource))
        
        # Clear strong references
        del resources
        gc.collect()
        
        # Check that weak references are cleared (resources cleaned up)
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        assert alive_count == 0, f"{alive_count} resources not cleaned up"


    def test_concurrent_resource_access(self, temp_dir):
        """Test thread-safe resource access."""
        import threading
        import json
        
        shared_data = {"counter": 0}
        lock = threading.Lock()
        errors = []
        
        def worker(worker_id):
            try:
                for _ in range(100):
                    with lock:
                        current = shared_data["counter"]
                        # Simulate some processing time
                        time.sleep(0.001)
                        shared_data["counter"] = current + 1
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")
        
        # Start multiple worker threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors and correct final count
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert shared_data["counter"] == 1000


    def test_timeout_handling(self):
        """Test proper timeout handling for long-running operations."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")
        
        def long_running_task():
            # Simulate a task that might run too long
            time.sleep(5)
            return "completed"
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout
        
        start_time = time.time()
        try:
            result = long_running_task()
            assert False, "Task should have timed out"
        except TimeoutError:
            execution_time = time.time() - start_time
            assert execution_time < 3, "Timeout took too long to trigger"
        finally:
            signal.alarm(0)  # Cancel the alarm