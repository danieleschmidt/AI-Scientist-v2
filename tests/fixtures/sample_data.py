"""
Sample test data and factories for AI Scientist v2 tests.
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class MockExperimentResult:
    """Mock experiment result for testing."""
    
    experiment_id: str
    title: str
    status: str
    accuracy: float
    loss: float
    training_time: float
    memory_usage: int
    gpu_utilization: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class MockResearchPaper:
    """Mock research paper for testing."""
    
    title: str
    abstract: str
    authors: List[str]
    keywords: List[str]
    sections: Dict[str, str]
    references: List[str]
    figures: List[str]
    tables: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class TestDataFactory:
    """Factory for generating test data."""
    
    @staticmethod
    def create_experiment_result(
        experiment_id: str = "test_exp_001",
        title: str = "Test Experiment",
        status: str = "completed",
        accuracy: float = 0.85,
        loss: float = 0.15,
        training_time: float = 120.5,
        memory_usage: int = 2048,
        gpu_utilization: float = 75.0,
    ) -> MockExperimentResult:
        """Create a mock experiment result."""
        return MockExperimentResult(
            experiment_id=experiment_id,
            title=title,
            status=status,
            accuracy=accuracy,
            loss=loss,
            training_time=training_time,
            memory_usage=memory_usage,
            gpu_utilization=gpu_utilization,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    
    @staticmethod
    def create_research_paper(
        title: str = "Test Research Paper",
        abstract: str = "This is a test abstract for a research paper.",
        authors: List[str] = None,
        keywords: List[str] = None,
    ) -> MockResearchPaper:
        """Create a mock research paper."""
        if authors is None:
            authors = ["Test Author 1", "Test Author 2"]
        if keywords is None:
            keywords = ["machine learning", "testing", "automation"]
        
        return MockResearchPaper(
            title=title,
            abstract=abstract,
            authors=authors,
            keywords=keywords,
            sections={
                "introduction": "This is the introduction section.",
                "methodology": "This is the methodology section.",
                "results": "This is the results section.",
                "conclusion": "This is the conclusion section.",
            },
            references=[
                "Reference 1: Important Paper (2023)",
                "Reference 2: Another Paper (2024)",
                "Reference 3: Third Paper (2024)",
            ],
            figures=["Figure 1: Test Figure", "Figure 2: Another Figure"],
            tables=["Table 1: Test Results", "Table 2: Comparison Data"],
        )
    
    @staticmethod
    def create_llm_config(
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        """Create LLM configuration for testing."""
        return {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "timeout": 30,
            "retry_attempts": 3,
        }
    
    @staticmethod
    def create_tree_search_config(
        num_workers: int = 3,
        steps: int = 21,
        max_debug_depth: int = 3,
        debug_prob: float = 0.8,
        num_drafts: int = 3,
    ) -> Dict[str, Any]:
        """Create tree search configuration for testing."""
        return {
            "agent": {
                "num_workers": num_workers,
                "steps": steps,
                "num_seeds": min(num_workers, 3),
                "k_fold_validation": False,
                "expose_prediction": False,
                "data_preview": False,
            },
            "search": {
                "max_debug_depth": max_debug_depth,
                "debug_prob": debug_prob,
                "num_drafts": num_drafts,
            },
        }
    
    @staticmethod
    def create_security_test_data() -> Dict[str, Any]:
        """Create security test data with various attack vectors."""
        return {
            "sql_injection": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'; SELECT * FROM secret_data; --",
            ],
            "xss_payloads": [
                "<script>alert('xss')</script>",
                "javascript:alert('xss')",
                "<img src=x onerror=alert('xss')>",
            ],
            "path_traversal": [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\config\\sam",
                "/etc/shadow",
            ],
            "command_injection": [
                "; ls -la",
                "| cat /etc/passwd",
                "`rm -rf /`",
                "$(whoami)",
            ],
            "code_injection": [
                "__import__('os').system('ls')",
                "eval('print(\"injected\")')",
                "exec('import subprocess; subprocess.call([\"ls\"])')",
            ],
        }
    
    @staticmethod
    def create_performance_test_data() -> Dict[str, Any]:
        """Create performance test data for benchmarking."""
        return {
            "small_dataset": list(range(100)),
            "medium_dataset": list(range(10000)),
            "large_dataset": list(range(1000000)),
            "memory_intensive_data": ["x" * 1000 for _ in range(1000)],
            "nested_structure": {
                f"level_{i}": {
                    f"sublevel_{j}": list(range(100))
                    for j in range(10)
                }
                for i in range(10)
            },
        }


# Sample JSON data for testing
SAMPLE_RESEARCH_IDEAS = [
    {
        "title": "Automated Hyperparameter Optimization for Deep Learning",
        "abstract": "This paper presents a novel approach to automate hyperparameter optimization...",
        "hypothesis": "Automated optimization can improve model performance by 15-20%",
        "methodology": "Use Bayesian optimization with Gaussian processes",
        "expected_results": "Significant improvement in model accuracy and training efficiency",
        "novelty_score": 0.8,
        "feasibility_score": 0.9,
        "impact_score": 0.7,
        "keywords": ["deep learning", "hyperparameter optimization", "automation"],
        "related_work": ["AutoML", "Bayesian Optimization", "Neural Architecture Search"],
    },
    {
        "title": "Federated Learning for Privacy-Preserving AI",
        "abstract": "We propose a federated learning framework that maintains privacy...",
        "hypothesis": "Federated learning can achieve comparable performance to centralized training",
        "methodology": "Implement differential privacy mechanisms",
        "expected_results": "Maintain privacy while achieving 95% of centralized performance",
        "novelty_score": 0.7,
        "feasibility_score": 0.8,
        "impact_score": 0.9,
        "keywords": ["federated learning", "privacy", "differential privacy"],
        "related_work": ["Differential Privacy", "Secure Aggregation", "Privacy-Preserving ML"],
    },
]

SAMPLE_EXPERIMENT_CONFIGS = [
    {
        "name": "baseline_experiment",
        "model": "gpt-4",
        "dataset": "test_dataset",
        "parameters": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "optimizer": "adam",
        },
        "metrics": ["accuracy", "loss", "f1_score"],
        "timeout": 3600,
    },
    {
        "name": "advanced_experiment",
        "model": "claude-3-5-sonnet-20241022",
        "dataset": "large_dataset",
        "parameters": {
            "learning_rate": 0.0001,
            "batch_size": 64,
            "epochs": 50,
            "optimizer": "adamw",
            "weight_decay": 0.01,
        },
        "metrics": ["accuracy", "precision", "recall", "auc"],
        "timeout": 7200,
    },
]

SAMPLE_API_RESPONSES = {
    "openai_chat_completion": {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1640995200,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from OpenAI GPT-4 model.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75,
        },
    },
    "anthropic_message": {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a test response from Anthropic Claude model.",
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 25,
        },
    },
    "semantic_scholar_search": {
        "total": 2,
        "offset": 0,
        "next": None,
        "data": [
            {
                "paperId": "123456789",
                "corpusId": 987654321,
                "title": "Deep Learning for Scientific Discovery",
                "abstract": "This paper explores the application of deep learning...",
                "venue": "Nature Machine Intelligence",
                "year": 2024,
                "referenceCount": 45,
                "citationCount": 12,
                "influentialCitationCount": 8,
                "isOpenAccess": True,
                "fieldsOfStudy": ["Computer Science", "Mathematics"],
                "authors": [
                    {"authorId": "1", "name": "Dr. AI Researcher"},
                    {"authorId": "2", "name": "Prof. ML Expert"},
                ],
            }
        ],
    },
}


def save_sample_data(filename: str, data: Any, output_dir: str = "/tmp") -> str:
    """Save sample data to JSON file for testing."""
    import os
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    return filepath


def load_sample_data(filename: str, data_dir: str = "/tmp") -> Any:
    """Load sample data from JSON file."""
    import os
    filepath = os.path.join(data_dir, filename)
    with open(filepath, 'r') as f:
        return json.load(f)