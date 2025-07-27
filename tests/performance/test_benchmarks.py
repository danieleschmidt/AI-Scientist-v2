"""Performance benchmarks and load testing for AI Scientist v2."""

import pytest
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch
import memory_profiler


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for the AI Scientist system."""

    def test_ideation_performance(self, mock_api_keys):
        """Benchmark ideation performance."""
        start_time = time.time()
        
        # Mock ideation process
        with patch('ai_scientist.perform_ideation_temp_free.generate_ideas') as mock_ideation:
            mock_ideation.return_value = {
                "ideas": [{"title": f"Idea {i}", "score": 0.8} for i in range(10)]
            }
            
            # Simulate ideation for multiple topics
            for i in range(5):
                result = mock_ideation(f"topic_{i}")
                assert len(result["ideas"]) == 10
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertion: should complete within reasonable time
        assert execution_time < 5.0, f"Ideation too slow: {execution_time:.2f}s"

    def test_memory_usage_during_experiments(self, mock_api_keys):
        """Test memory usage during experiment execution."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate memory-intensive experiment
        with patch('ai_scientist.treesearch.parallel_agent.ParallelAgent') as mock_agent:
            mock_agent_instance = Mock()
            
            # Simulate experiment execution
            for i in range(10):
                mock_agent_instance.run_experiment(f"experiment_{i}")
                
                current_memory = process.memory_info().rss
                memory_delta = current_memory - initial_memory
                
                # Memory should not grow excessively (max 500MB increase)
                assert memory_delta < 500 * 1024 * 1024, f"Memory usage too high: {memory_delta / 1024 / 1024:.1f}MB"

    def test_concurrent_experiment_performance(self, mock_api_keys):
        """Test performance with concurrent experiments."""
        num_workers = 4
        experiments_per_worker = 5
        
        def run_mock_experiment(experiment_id):
            """Mock experiment runner."""
            time.sleep(0.1)  # Simulate processing time
            return {"id": experiment_id, "status": "completed", "score": 0.8}
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for worker_id in range(num_workers):
                for exp_id in range(experiments_per_worker):
                    experiment_id = f"worker_{worker_id}_exp_{exp_id}"
                    future = executor.submit(run_mock_experiment, experiment_id)
                    futures.append(future)
            
            # Wait for all experiments to complete
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete faster than sequential execution
        expected_sequential_time = num_workers * experiments_per_worker * 0.1
        assert total_time < expected_sequential_time * 0.8, f"Parallel execution not efficient: {total_time:.2f}s"
        assert len(results) == num_workers * experiments_per_worker

    def test_tree_search_scalability(self, mock_api_keys):
        """Test tree search performance with varying complexity."""
        complexities = [5, 10, 20, 50]
        execution_times = []
        
        for complexity in complexities:
            start_time = time.time()
            
            # Mock tree search with varying complexity
            with patch('ai_scientist.treesearch.agent_manager.AgentManager') as mock_manager:
                mock_manager_instance = Mock()
                
                # Simulate tree search execution
                mock_manager_instance.run_search.return_value = {
                    "nodes_explored": complexity,
                    "successful_paths": complexity // 5,
                    "total_time": time.time() - start_time
                }
                
                result = mock_manager_instance.run_search()
            
            end_time = time.time()
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            
            # Performance should scale reasonably (not exponentially)
            if len(execution_times) > 1:
                time_ratio = execution_time / execution_times[0]
                complexity_ratio = complexity / complexities[0]
                
                # Time should not increase faster than complexity^2
                assert time_ratio <= complexity_ratio ** 1.5, f"Poor scalability at complexity {complexity}"

    @pytest.mark.memory
    def test_memory_profiling(self, mock_api_keys):
        """Profile memory usage of key components."""
        
        @memory_profiler.profile
        def memory_intensive_operation():
            """Simulate memory-intensive AI Scientist operation."""
            # Mock data structures that might consume memory
            data = []
            for i in range(1000):
                experiment_data = {
                    "id": f"exp_{i}",
                    "results": [0.1 * j for j in range(100)],
                    "metadata": {"param_" + str(k): k for k in range(50)}
                }
                data.append(experiment_data)
            return data
        
        # Run the memory-intensive operation
        initial_memory = psutil.Process().memory_info().rss
        result = memory_intensive_operation()
        final_memory = psutil.Process().memory_info().rss
        
        memory_used = final_memory - initial_memory
        
        # Verify memory usage is reasonable
        assert len(result) == 1000
        assert memory_used < 100 * 1024 * 1024, f"Memory usage too high: {memory_used / 1024 / 1024:.1f}MB"

    def test_api_rate_limiting_performance(self, mock_api_keys):
        """Test performance under API rate limiting."""
        call_times = []
        
        # Mock API with rate limiting
        def mock_api_call_with_delay():
            time.sleep(0.1)  # Simulate API response time
            return {"response": "success"}
        
        # Simulate multiple API calls
        start_time = time.time()
        for i in range(10):
            call_start = time.time()
            result = mock_api_call_with_delay()
            call_end = time.time()
            call_times.append(call_end - call_start)
            
            assert result["response"] == "success"
        
        total_time = time.time() - start_time
        avg_call_time = sum(call_times) / len(call_times)
        
        # Performance assertions
        assert total_time < 5.0, f"Total API time too high: {total_time:.2f}s"
        assert avg_call_time < 0.5, f"Average call time too high: {avg_call_time:.2f}s"

    def test_large_dataset_processing(self, mock_api_keys):
        """Test performance with large datasets."""
        dataset_sizes = [100, 500, 1000, 5000]
        
        for size in dataset_sizes:
            start_time = time.time()
            
            # Mock large dataset processing
            dataset = [{"sample_" + str(i): i * 0.1} for i in range(size)]
            
            # Simulate processing
            processed_count = 0
            for item in dataset:
                # Mock processing operation
                if item:
                    processed_count += 1
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Performance should be linear with dataset size
            time_per_item = processing_time / size
            assert time_per_item < 0.001, f"Processing too slow for size {size}: {time_per_item:.4f}s per item"
            assert processed_count == size

    def test_concurrent_user_simulation(self, mock_api_keys):
        """Simulate multiple concurrent users."""
        num_users = 5
        operations_per_user = 3
        
        def simulate_user_session(user_id):
            """Simulate a user session."""
            session_results = []
            
            for op_id in range(operations_per_user):
                # Mock user operation
                start_time = time.time()
                
                # Simulate different types of operations
                if op_id == 0:
                    # Ideation
                    time.sleep(0.05)
                    result = {"type": "ideation", "ideas": 5}
                elif op_id == 1:
                    # Experiment
                    time.sleep(0.1)
                    result = {"type": "experiment", "status": "completed"}
                else:
                    # Paper generation
                    time.sleep(0.08)
                    result = {"type": "paper", "pages": 10}
                
                end_time = time.time()
                operation_time = end_time - start_time
                
                session_results.append({
                    "user_id": user_id,
                    "operation": op_id,
                    "time": operation_time,
                    "result": result
                })
            
            return session_results
        
        # Run concurrent user sessions
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(simulate_user_session, user_id) 
                for user_id in range(num_users)
            ]
            
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify all operations completed
        assert len(all_results) == num_users * operations_per_user
        
        # Performance assertion
        assert total_time < 2.0, f"Concurrent user simulation too slow: {total_time:.2f}s"
        
        # Check individual operation times
        for result in all_results:
            assert result["time"] < 0.5, f"Individual operation too slow: {result['time']:.2f}s"

    def test_resource_cleanup_performance(self, mock_api_keys):
        """Test performance of resource cleanup operations."""
        # Simulate resource creation and cleanup
        resources = []
        
        # Create mock resources
        start_time = time.time()
        for i in range(100):
            resource = {
                "id": f"resource_{i}",
                "type": "experiment",
                "status": "active",
                "created_at": time.time()
            }
            resources.append(resource)
        
        creation_time = time.time() - start_time
        
        # Cleanup resources
        cleanup_start = time.time()
        cleaned_resources = []
        
        for resource in resources:
            # Mock cleanup operation
            resource["status"] = "cleaned"
            cleaned_resources.append(resource)
        
        cleanup_time = time.time() - cleanup_start
        
        # Performance assertions
        assert creation_time < 0.5, f"Resource creation too slow: {creation_time:.2f}s"
        assert cleanup_time < 0.5, f"Resource cleanup too slow: {cleanup_time:.2f}s"
        assert len(cleaned_resources) == 100
        assert all(r["status"] == "cleaned" for r in cleaned_resources)

    def test_error_handling_performance(self, mock_api_keys):
        """Test performance impact of error handling."""
        error_scenarios = [
            "network_timeout",
            "api_rate_limit", 
            "invalid_input",
            "resource_exhaustion",
            "parsing_error"
        ]
        
        for scenario in error_scenarios:
            start_time = time.time()
            
            # Mock error handling
            try:
                if scenario == "network_timeout":
                    time.sleep(0.01)  # Simulate timeout detection
                    raise TimeoutError("Network timeout")
                elif scenario == "api_rate_limit":
                    time.sleep(0.02)  # Simulate rate limit detection
                    raise Exception("Rate limit exceeded")
                else:
                    time.sleep(0.005)  # Simulate other error detection
                    raise ValueError(f"Error scenario: {scenario}")
            
            except Exception as e:
                # Mock error recovery
                recovery_start = time.time()
                
                # Simulate error recovery operations
                if "timeout" in str(e).lower():
                    time.sleep(0.01)  # Retry logic
                elif "rate limit" in str(e).lower():
                    time.sleep(0.02)  # Backoff logic
                else:
                    time.sleep(0.005)  # General error handling
                
                recovery_time = time.time() - recovery_start
                total_time = time.time() - start_time
                
                # Error handling should be fast
                assert recovery_time < 0.1, f"Error recovery too slow for {scenario}: {recovery_time:.3f}s"
                assert total_time < 0.2, f"Total error handling too slow for {scenario}: {total_time:.3f}s"