"""
Performance benchmarks and stress tests for AI Scientist v2.
"""

import pytest
import time
import psutil
import os
import threading
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import memory_profiler


@pytest.mark.performance
class TestLLMPerformance:
    """Performance tests for LLM integration."""
    
    @pytest.mark.slow
    def test_llm_response_time_benchmark(self, mock_openai_client):
        """Benchmark LLM response times."""
        from ai_scientist.llm import get_llm_response
        
        response_times = []
        num_requests = 10
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = get_llm_response(
                    prompt=f"Test prompt {i}",
                    model="gpt-4",
                    max_tokens=100
                )
                end_time = time.time()
                response_times.append(end_time - start_time)
            except Exception as e:
                pytest.skip(f"LLM request failed: {e}")
        
        # Performance assertions
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 2.0, f"Average response time {avg_response_time:.2f}s exceeds 2s threshold"
        assert max_response_time < 5.0, f"Max response time {max_response_time:.2f}s exceeds 5s threshold"
        
        # Report performance metrics
        print(f"LLM Performance Metrics:")
        print(f"  Average response time: {avg_response_time:.2f}s")
        print(f"  Max response time: {max_response_time:.2f}s")
        print(f"  Min response time: {min(response_times):.2f}s")
    
    @pytest.mark.slow
    def test_concurrent_llm_requests(self, mock_openai_client):
        """Test performance under concurrent LLM requests."""
        from ai_scientist.llm import get_llm_response
        
        def make_request(request_id):
            start_time = time.time()
            try:
                response = get_llm_response(
                    prompt=f"Concurrent test request {request_id}",
                    model="gpt-4",
                    max_tokens=50
                )
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "response_time": end_time - start_time,
                    "success": True
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "error": str(e),
                    "success": False
                }
        
        num_concurrent = 5
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        successful_requests = [r for r in results if r["success"]]
        
        # Performance assertions
        assert len(successful_requests) >= num_concurrent * 0.8, "Too many failed requests"
        assert total_time < 10.0, f"Concurrent requests took {total_time:.2f}s, expected < 10s"
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            assert avg_response_time < 5.0, f"Average concurrent response time {avg_response_time:.2f}s too high"


@pytest.mark.performance
class TestMemoryPerformance:
    """Memory usage and leak detection tests."""
    
    @memory_profiler.profile
    @pytest.mark.slow
    def test_memory_usage_during_ideation(self, temp_dir):
        """Profile memory usage during research ideation."""
        from ai_scientist.perform_ideation_temp_free import generate_research_ideas
        
        # Create test topic file
        topic_file = temp_dir / "memory_test_topic.md"
        topic_content = """# Memory Test Topic
        
## Abstract
Testing memory usage during ideation process with a moderately sized research topic description.
We want to ensure that the system doesn't consume excessive memory during idea generation.
"""
        topic_file.write_text(topic_content)
        
        # Monitor memory before
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('ai_scientist.llm.get_llm_response') as mock_llm:
            # Generate multiple ideas to stress test memory
            mock_llm.return_value = json.dumps({
                "title": "Memory Test Idea",
                "abstract": "Testing memory usage patterns",
                "hypothesis": "Memory usage should remain stable",
                "methodology": "Monitor memory during execution"
            })
            
            for i in range(10):
                output_file = temp_dir / f"memory_test_ideas_{i}.json"
                try:
                    generate_research_ideas(
                        workshop_file=str(topic_file),
                        output_file=str(output_file),
                        model="gpt-4",
                        max_num_generations=5
                    )
                except Exception:
                    pass  # Focus on memory, ignore other errors
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory increase should be reasonable
        assert memory_increase_mb < 200, f"Memory increased by {memory_increase_mb:.2f} MB, expected < 200 MB"
        
        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"  Final: {final_memory / 1024 / 1024:.2f} MB")
        print(f"  Increase: {memory_increase_mb:.2f} MB")
    
    def test_memory_leak_detection(self, temp_dir):
        """Detect potential memory leaks in core functions."""
        import gc
        
        def run_cycle():
            """Run a complete cycle of operations that might leak memory."""
            from ai_scientist.utils.config import ConfigManager
            from ai_scientist.treesearch.utils.config import load_config
            
            # Create and destroy config managers
            for _ in range(10):
                config = ConfigManager()
                config.set("test_key", "test_value" * 1000)  # Large string
                del config
            
            # Force garbage collection
            gc.collect()
        
        # Measure memory before cycles
        process = psutil.Process(os.getpid())
        baseline_memory = []
        
        # Establish baseline
        for _ in range(3):
            run_cycle()
            baseline_memory.append(process.memory_info().rss)
            time.sleep(0.1)
        
        avg_baseline = sum(baseline_memory) / len(baseline_memory)
        
        # Run more cycles and measure memory
        test_memory = []
        for _ in range(10):
            run_cycle()
            test_memory.append(process.memory_info().rss)
            time.sleep(0.1)
        
        avg_test = sum(test_memory) / len(test_memory)
        memory_growth = avg_test - avg_baseline
        memory_growth_mb = memory_growth / (1024 * 1024)
        
        # Memory growth should be minimal
        assert memory_growth_mb < 50, f"Detected potential memory leak: {memory_growth_mb:.2f} MB growth"
        
        print(f"Memory Leak Test:")
        print(f"  Baseline average: {avg_baseline / 1024 / 1024:.2f} MB")
        print(f"  Test average: {avg_test / 1024 / 1024:.2f} MB")
        print(f"  Growth: {memory_growth_mb:.2f} MB")


@pytest.mark.performance
class TestCPUPerformance:
    """CPU usage and performance tests."""
    
    @pytest.mark.slow
    def test_cpu_usage_during_tree_search(self, temp_dir):
        """Monitor CPU usage during tree search operations."""
        from ai_scientist.treesearch.parallel_agent import ParallelAgent
        
        # Monitor CPU usage
        cpu_usage = []
        monitoring = True
        
        def monitor_cpu():
            while monitoring:
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        try:
            with patch('ai_scientist.treesearch.parallel_agent.ParallelAgent') as mock_agent:
                # Mock computationally intensive operations
                mock_instance = Mock()
                
                def cpu_intensive_mock(*args, **kwargs):
                    # Simulate CPU-intensive work
                    start_time = time.time()
                    while time.time() - start_time < 0.5:  # 500ms of work
                        _ = sum(i * i for i in range(1000))  # CPU work
                    return {"success": True, "results": {}}
                
                mock_instance.run.side_effect = cpu_intensive_mock
                mock_agent.return_value = mock_instance
                
                # Create mock configuration
                config = {
                    "agent": {"num_workers": 2, "steps": 5},
                    "search": {"max_debug_depth": 2}
                }
                
                # Run tree search simulation
                for _ in range(3):
                    mock_instance.run()
                
        finally:
            monitoring = False
            monitor_thread.join()
        
        # Analyze CPU usage
        if cpu_usage:
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            max_cpu = max(cpu_usage)
            
            # CPU usage should be reasonable
            assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}% is too high"
            assert max_cpu < 95, f"Peak CPU usage {max_cpu:.1f}% is too high"
            
            print(f"CPU Usage During Tree Search:")
            print(f"  Average: {avg_cpu:.1f}%")
            print(f"  Peak: {max_cpu:.1f}%")
            print(f"  Samples: {len(cpu_usage)}")
    
    def test_parallel_processing_efficiency(self, temp_dir):
        """Test efficiency of parallel processing."""
        def cpu_intensive_task(task_id, duration=0.1):
            """Simulate CPU-intensive task."""
            start_time = time.time()
            result = 0
            while time.time() - start_time < duration:
                result += sum(i * i for i in range(100))
            return {"task_id": task_id, "result": result}
        
        num_tasks = 8
        
        # Test sequential execution
        start_time = time.time()
        sequential_results = []
        for i in range(num_tasks):
            sequential_results.append(cpu_intensive_task(i))
        sequential_time = time.time() - start_time
        
        # Test parallel execution
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task, i) for i in range(num_tasks)]
            parallel_results = [future.result() for future in as_completed(futures)]
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / parallel_time
        efficiency = speedup / 4  # 4 workers
        
        # Parallel should be faster (allowing for overhead)
        assert parallel_time < sequential_time * 0.8, f"Parallel execution not faster: {parallel_time:.2f}s vs {sequential_time:.2f}s"
        assert speedup > 1.5, f"Insufficient speedup: {speedup:.2f}x"
        
        print(f"Parallel Processing Efficiency:")
        print(f"  Sequential time: {sequential_time:.2f}s")
        print(f"  Parallel time: {parallel_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {efficiency:.2f}")


@pytest.mark.performance
class TestIOPerformance:
    """File I/O and disk performance tests."""
    
    @pytest.mark.slow
    def test_large_file_processing(self, temp_dir):
        """Test performance with large research data files."""
        # Create large test data file
        large_data = {
            "papers": [
                {
                    "id": f"paper_{i}",
                    "title": f"Research Paper {i}",
                    "abstract": "This is a test abstract. " * 100,  # Large abstract
                    "authors": [f"Author {j}" for j in range(10)],
                    "keywords": [f"keyword_{k}" for k in range(20)],
                    "content": "Research content. " * 1000  # Large content
                }
                for i in range(100)  # 100 papers
            ]
        }
        
        large_file = temp_dir / "large_research_data.json"
        
        # Test write performance
        start_time = time.time()
        with open(large_file, 'w') as f:
            json.dump(large_data, f)
        write_time = time.time() - start_time
        
        file_size_mb = large_file.stat().st_size / (1024 * 1024)
        
        # Test read performance
        start_time = time.time()
        with open(large_file, 'r') as f:
            loaded_data = json.load(f)
        read_time = time.time() - start_time
        
        # Performance assertions
        write_speed_mbps = file_size_mb / write_time
        read_speed_mbps = file_size_mb / read_time
        
        assert write_speed_mbps > 10, f"Write speed {write_speed_mbps:.2f} MB/s is too slow"
        assert read_speed_mbps > 50, f"Read speed {read_speed_mbps:.2f} MB/s is too slow"
        assert len(loaded_data["papers"]) == 100, "Data integrity check failed"
        
        print(f"File I/O Performance:")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Write time: {write_time:.2f}s ({write_speed_mbps:.2f} MB/s)")
        print(f"  Read time: {read_time:.2f}s ({read_speed_mbps:.2f} MB/s)")
    
    def test_concurrent_file_operations(self, temp_dir):
        """Test performance of concurrent file operations."""
        def write_file(file_id):
            """Write a test file."""
            file_path = temp_dir / f"concurrent_test_{file_id}.json"
            data = {
                "file_id": file_id,
                "data": [f"item_{i}" for i in range(1000)],
                "timestamp": time.time()
            }
            
            start_time = time.time()
            with open(file_path, 'w') as f:
                json.dump(data, f)
            write_time = time.time() - start_time
            
            return {"file_id": file_id, "write_time": write_time}
        
        num_files = 10
        
        # Test sequential file writes
        start_time = time.time()
        sequential_results = []
        for i in range(num_files):
            sequential_results.append(write_file(i))
        sequential_time = time.time() - start_time
        
        # Clean up files
        for file_path in temp_dir.glob("concurrent_test_*.json"):
            file_path.unlink()
        
        # Test concurrent file writes
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_file, i + 100) for i in range(num_files)]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_time = time.time() - start_time
        
        # Analyze results
        avg_sequential_write = sum(r["write_time"] for r in sequential_results) / len(sequential_results)
        avg_concurrent_write = sum(r["write_time"] for r in concurrent_results) / len(concurrent_results)
        
        # Concurrent should not be significantly slower
        assert concurrent_time < sequential_time * 1.5, f"Concurrent I/O much slower: {concurrent_time:.2f}s vs {sequential_time:.2f}s"
        
        print(f"Concurrent I/O Performance:")
        print(f"  Sequential total: {sequential_time:.2f}s")
        print(f"  Concurrent total: {concurrent_time:.2f}s")
        print(f"  Avg sequential write: {avg_sequential_write:.3f}s")
        print(f"  Avg concurrent write: {avg_concurrent_write:.3f}s")


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Scalability and stress tests."""
    
    @pytest.mark.slow
    def test_scalability_with_increasing_load(self, temp_dir):
        """Test system performance under increasing load."""
        def process_batch(batch_size):
            """Process a batch of research ideas."""
            start_time = time.time()
            
            with patch('ai_scientist.llm.get_llm_response') as mock_llm:
                mock_llm.return_value = json.dumps({
                    "title": "Scalability Test",
                    "abstract": "Testing system scalability"
                })
                
                from ai_scientist.perform_ideation_temp_free import generate_research_ideas
                
                for i in range(batch_size):
                    topic_file = temp_dir / f"scalability_topic_{i}.md"
                    topic_file.write_text(f"# Scalability Test {i}\nTesting with batch size {batch_size}")
                    
                    output_file = temp_dir / f"scalability_ideas_{i}.json"
                    try:
                        generate_research_ideas(
                            workshop_file=str(topic_file),
                            output_file=str(output_file),
                            model="gpt-4",
                            max_num_generations=1
                        )
                    except Exception:
                        pass  # Focus on performance metrics
            
            processing_time = time.time() - start_time
            return {
                "batch_size": batch_size,
                "processing_time": processing_time,
                "throughput": batch_size / processing_time
            }
        
        # Test with increasing batch sizes
        batch_sizes = [1, 5, 10, 20]
        results = []
        
        for batch_size in batch_sizes:
            result = process_batch(batch_size)
            results.append(result)
            
            # Clean up files to avoid disk space issues
            for file_path in temp_dir.glob(f"scalability_*_{batch_size-1}*"):
                file_path.unlink(missing_ok=True)
        
        # Analyze scalability
        throughputs = [r["throughput"] for r in results]
        processing_times = [r["processing_time"] for r in results]
        
        # Throughput should not degrade significantly
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        throughput_degradation = (max_throughput - min_throughput) / max_throughput
        
        assert throughput_degradation < 0.5, f"Throughput degraded by {throughput_degradation:.1%}"
        
        print(f"Scalability Test Results:")
        for result in results:
            print(f"  Batch {result['batch_size']}: {result['processing_time']:.2f}s, {result['throughput']:.2f} items/s")
    
    @pytest.mark.slow
    def test_stress_test_concurrent_experiments(self, temp_dir):
        """Stress test with multiple concurrent experiments."""
        def run_mock_experiment(exp_id):
            """Run a mock experiment."""
            start_time = time.time()
            
            # Simulate experiment work
            time.sleep(0.1)  # Base processing time
            
            # Simulate some CPU work
            result = sum(i * i for i in range(1000))
            
            # Simulate file I/O
            exp_file = temp_dir / f"stress_experiment_{exp_id}.json"
            exp_data = {
                "experiment_id": exp_id,
                "result": result,
                "timestamp": time.time(),
                "data": [f"data_point_{i}" for i in range(100)]
            }
            
            with open(exp_file, 'w') as f:
                json.dump(exp_data, f)
            
            processing_time = time.time() - start_time
            return {
                "experiment_id": exp_id,
                "processing_time": processing_time,
                "success": True
            }
        
        num_experiments = 20
        max_workers = 5
        
        # Monitor system resources during stress test
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_mock_experiment, i) for i in range(num_experiments)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)
        
        # Analyze stress test results
        successful_experiments = [r for r in results if r["success"]]
        avg_processing_time = sum(r["processing_time"] for r in successful_experiments) / len(successful_experiments)
        
        # Performance assertions
        assert len(successful_experiments) >= num_experiments * 0.9, "Too many failed experiments under stress"
        assert total_time < 30, f"Stress test took too long: {total_time:.2f}s"
        assert memory_increase < 100, f"Memory increased too much: {memory_increase:.2f} MB"
        assert avg_processing_time < 1.0, f"Average processing time too high: {avg_processing_time:.2f}s"
        
        print(f"Stress Test Results:")
        print(f"  Total experiments: {num_experiments}")
        print(f"  Successful: {len(successful_experiments)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg processing time: {avg_processing_time:.2f}s")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Throughput: {len(successful_experiments) / total_time:.2f} experiments/s")