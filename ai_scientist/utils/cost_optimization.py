#!/usr/bin/env python3
"""
Cost Optimization Framework

Advanced cost tracking, optimization, and budgeting system for AI Scientist v2.
Monitors LLM API usage, GPU utilization, and provides intelligent cost controls.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Cost categories for tracking and optimization."""
    LLM_API = "llm_api"
    GPU_COMPUTE = "gpu_compute"
    STORAGE = "storage"
    NETWORK = "network"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_API = "external_api"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CostUsage:
    """Individual cost usage record."""
    timestamp: datetime
    category: CostCategory
    service: str
    operation: str
    cost: float
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.category, str):
            self.category = CostCategory(self.category)


@dataclass
class BudgetConfig:
    """Budget configuration for cost control."""
    daily_limit: float = 100.0
    weekly_limit: float = 500.0
    monthly_limit: float = 2000.0
    currency: str = "USD"
    
    # Category-specific limits
    category_limits: Dict[CostCategory, float] = field(default_factory=dict)
    
    # Alert thresholds (percentage of budget)
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%
    
    # Auto-actions on budget exceeded
    auto_throttle: bool = True
    auto_shutdown: bool = False
    emergency_contact: Optional[str] = None


@dataclass
class CostAlert:
    """Cost alert notification."""
    timestamp: datetime
    severity: AlertSeverity
    category: CostCategory
    message: str
    current_cost: float
    budget_limit: float
    budget_period: str
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation."""
    category: CostCategory
    description: str
    potential_savings: float
    implementation_effort: str  # "low", "medium", "high"
    priority: int  # 1-10, higher is more important
    actions: List[str] = field(default_factory=list)


class LLMPricingModel:
    """Pricing model for different LLM providers."""
    
    # Pricing per 1K tokens (as of 2025, subject to change)
    PRICING = {
        "gpt-4o": {"input": 0.0050, "output": 0.0150},
        "gpt-4o-mini": {"input": 0.0002, "output": 0.0006},
        "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.0100},
        "o1-preview-2024-09-12": {"input": 0.0150, "output": 0.0600},
        "o1-2024-12-17": {"input": 0.0150, "output": 0.0600},
        "o3-mini-2025-01-31": {"input": 0.0010, "output": 0.0040},
        "claude-3-5-sonnet-20240620": {"input": 0.0030, "output": 0.0150},
        "claude-3-5-sonnet-20241022-v2:0": {"input": 0.0030, "output": 0.0150},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
    }
    
    @classmethod
    def calculate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for LLM usage."""
        if model not in cls.PRICING:
            logger.warning(f"Unknown model pricing for {model}, using default rates")
            pricing = {"input": 0.0020, "output": 0.0060}  # Conservative default
        else:
            pricing = cls.PRICING[model]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    @classmethod
    def get_cheapest_model(cls, capability_class: str = "general") -> str:
        """Get the cheapest model for a capability class."""
        # Define capability classes
        capability_models = {
            "general": ["gpt-4o-mini", "claude-3-5-sonnet-20240620", "gemini-pro"],
            "reasoning": ["o3-mini-2025-01-31", "o1-2024-12-17", "gpt-4o"],
            "vision": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20240620"],
            "coding": ["claude-3-5-sonnet-20240620", "gpt-4o", "o1-2024-12-17"]
        }
        
        models = capability_models.get(capability_class, capability_models["general"])
        
        # Calculate average cost per 1K tokens for comparison
        cheapest_model = None
        lowest_avg_cost = float('inf')
        
        for model in models:
            if model in cls.PRICING:
                pricing = cls.PRICING[model]
                avg_cost = (pricing["input"] + pricing["output"]) / 2
                if avg_cost < lowest_avg_cost:
                    lowest_avg_cost = avg_cost
                    cheapest_model = model
        
        return cheapest_model or "gpt-4o-mini"


class CostTracker:
    """Advanced cost tracking and optimization system."""
    
    def __init__(self, budget_config: Optional[BudgetConfig] = None):
        """Initialize cost tracker."""
        self.budget_config = budget_config or BudgetConfig()
        self.usage_history: List[CostUsage] = []
        self.current_costs: Dict[str, float] = defaultdict(float)  # period -> cost
        self.alerts: List[CostAlert] = []
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Performance tracking
        self._cost_by_operation: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._throttle_status: Dict[CostCategory, bool] = defaultdict(bool)
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Cost tracker initialized with budget configuration")
    
    def track_llm_usage(self, model: str, input_tokens: int, output_tokens: int, 
                       operation: str = "inference") -> CostUsage:
        """Track LLM API usage and calculate cost."""
        cost = LLMPricingModel.calculate_cost(model, input_tokens, output_tokens)
        
        usage = CostUsage(
            timestamp=datetime.now(),
            category=CostCategory.LLM_API,
            service=model,
            operation=operation,
            cost=cost,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_per_token": cost / (input_tokens + output_tokens) if (input_tokens + output_tokens) > 0 else 0
            }
        )
        
        self._record_usage(usage)
        return usage
    
    def track_gpu_usage(self, gpu_type: str, duration_hours: float, 
                       cost_per_hour: float = 2.50) -> CostUsage:
        """Track GPU compute usage."""
        cost = duration_hours * cost_per_hour
        
        usage = CostUsage(
            timestamp=datetime.now(),
            category=CostCategory.GPU_COMPUTE,
            service=gpu_type,
            operation="compute",
            cost=cost,
            metadata={
                "duration_hours": duration_hours,
                "cost_per_hour": cost_per_hour,
                "gpu_type": gpu_type
            }
        )
        
        self._record_usage(usage)
        return usage
    
    def track_external_api_usage(self, service: str, operation: str, 
                                cost: float, metadata: Optional[Dict] = None) -> CostUsage:
        """Track external API usage (Semantic Scholar, etc.)."""
        usage = CostUsage(
            timestamp=datetime.now(),
            category=CostCategory.EXTERNAL_API,
            service=service,
            operation=operation,
            cost=cost,
            metadata=metadata or {}
        )
        
        self._record_usage(usage)
        return usage
    
    def _record_usage(self, usage: CostUsage):
        """Record usage and check budget limits."""
        with self._lock:
            self.usage_history.append(usage)
            
            # Update current costs
            today = datetime.now().strftime("%Y-%m-%d")
            week = datetime.now().strftime("%Y-W%U")
            month = datetime.now().strftime("%Y-%m")
            
            self.current_costs[f"daily-{today}"] += usage.cost
            self.current_costs[f"weekly-{week}"] += usage.cost
            self.current_costs[f"monthly-{month}"] += usage.cost
            
            # Track operation performance
            self._cost_by_operation[f"{usage.category.value}-{usage.operation}"].append(usage.cost)
            
            # Check budget limits
            self._check_budget_limits()
            
            # Generate optimization recommendations
            self._update_optimization_recommendations()
    
    def _check_budget_limits(self):
        """Check if budget limits are exceeded and generate alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        week = datetime.now().strftime("%Y-W%U")
        month = datetime.now().strftime("%Y-%m")
        
        checks = [
            ("daily", f"daily-{today}", self.budget_config.daily_limit),
            ("weekly", f"weekly-{week}", self.budget_config.weekly_limit),
            ("monthly", f"monthly-{month}", self.budget_config.monthly_limit)
        ]
        
        for period, key, limit in checks:
            current_cost = self.current_costs.get(key, 0.0)
            usage_ratio = current_cost / limit if limit > 0 else 0
            
            if usage_ratio >= self.budget_config.critical_threshold:
                self._create_alert(
                    AlertSeverity.CRITICAL,
                    CostCategory.INFRASTRUCTURE,
                    f"CRITICAL: {period} budget exceeded! ${current_cost:.2f} / ${limit:.2f}",
                    current_cost,
                    limit,
                    period
                )
                
                if self.budget_config.auto_shutdown:
                    self._trigger_emergency_shutdown()
                elif self.budget_config.auto_throttle:
                    self._enable_cost_throttling()
                    
            elif usage_ratio >= self.budget_config.warning_threshold:
                self._create_alert(
                    AlertSeverity.WARNING,
                    CostCategory.INFRASTRUCTURE,
                    f"WARNING: {period} budget at {usage_ratio:.1%}! ${current_cost:.2f} / ${limit:.2f}",
                    current_cost,
                    limit,
                    period
                )
    
    def _create_alert(self, severity: AlertSeverity, category: CostCategory, 
                     message: str, current_cost: float, budget_limit: float, 
                     budget_period: str):
        """Create and store a cost alert."""
        alert = CostAlert(
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            message=message,
            current_cost=current_cost,
            budget_limit=budget_limit,
            budget_period=budget_period,
            recommended_actions=self._get_emergency_actions(severity)
        )
        
        self.alerts.append(alert)
        logger.warning(f"Cost alert: {message}")
        
        # Send notification if configured
        if self.budget_config.emergency_contact:
            self._send_alert_notification(alert)
    
    def _get_emergency_actions(self, severity: AlertSeverity) -> List[str]:
        """Get recommended emergency actions based on alert severity."""
        if severity == AlertSeverity.CRITICAL:
            return [
                "Immediately review high-cost operations",
                "Consider switching to cheaper models",
                "Implement request throttling",
                "Pause non-essential experiments",
                "Contact budget administrator"
            ]
        elif severity == AlertSeverity.WARNING:
            return [
                "Monitor usage closely",
                "Review recent expensive operations",
                "Consider model optimization",
                "Check for runaway processes"
            ]
        else:
            return ["Continue monitoring"]
    
    def _enable_cost_throttling(self):
        """Enable cost throttling across all categories."""
        with self._lock:
            for category in CostCategory:
                self._throttle_status[category] = True
            logger.warning("Cost throttling enabled due to budget limits")
    
    def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown procedures."""
        logger.critical("EMERGENCY SHUTDOWN triggered due to budget exceeded")
        # Implementation would depend on deployment environment
        # For Kubernetes, this might involve scaling down deployments
    
    def _send_alert_notification(self, alert: CostAlert):
        """Send alert notification to configured endpoint."""
        # Implementation would depend on notification system (email, Slack, etc.)
        logger.info(f"Alert notification sent: {alert.message}")
    
    def _update_optimization_recommendations(self):
        """Update cost optimization recommendations based on usage patterns."""
        recommendations = []
        
        # Analyze LLM usage patterns
        llm_costs = [u for u in self.usage_history[-1000:] if u.category == CostCategory.LLM_API]
        if llm_costs:
            # Find most expensive models
            model_costs = defaultdict(float)
            model_usage = defaultdict(int)
            
            for usage in llm_costs:
                model_costs[usage.service] += usage.cost
                model_usage[usage.service] += 1
            
            # Recommend cheaper alternatives for high-usage expensive models
            for model, total_cost in model_costs.items():
                if total_cost > 10.0 and model_usage[model] > 10:  # Significant usage
                    cheaper_model = LLMPricingModel.get_cheapest_model("general")
                    if cheaper_model != model:
                        potential_savings = total_cost * 0.3  # Estimate 30% savings
                        recommendations.append(OptimizationRecommendation(
                            category=CostCategory.LLM_API,
                            description=f"Switch from {model} to {cheaper_model} for general tasks",
                            potential_savings=potential_savings,
                            implementation_effort="low",
                            priority=8,
                            actions=[
                                f"Replace {model} with {cheaper_model} in configuration",
                                "Test quality impact on non-critical operations",
                                "Implement model selection based on task complexity"
                            ]
                        ))
        
        # Analyze operation efficiency
        for operation_key, costs in self._cost_by_operation.items():
            if len(costs) >= 10:
                avg_cost = statistics.mean(costs)
                recent_cost = statistics.mean(list(costs)[-5:])  # Last 5 operations
                
                if recent_cost > avg_cost * 1.5:  # 50% increase in recent costs
                    recommendations.append(OptimizationRecommendation(
                        category=CostCategory.LLM_API,
                        description=f"Operation {operation_key} showing cost increase",
                        potential_savings=avg_cost * 0.2,
                        implementation_effort="medium",
                        priority=6,
                        actions=[
                            f"Investigate {operation_key} performance degradation",
                            "Review recent changes to operation logic",
                            "Consider caching or optimization strategies"
                        ]
                    ))
        
        # Update recommendations
        with self._lock:
            self.optimization_recommendations = sorted(
                recommendations, 
                key=lambda x: x.priority, 
                reverse=True
            )[:10]  # Keep top 10
    
    def get_cost_summary(self, period: str = "daily") -> Dict[str, Any]:
        """Get cost summary for specified period."""
        now = datetime.now()
        
        if period == "daily":
            key = f"daily-{now.strftime('%Y-%m-%d')}"
            limit = self.budget_config.daily_limit
        elif period == "weekly":
            key = f"weekly-{now.strftime('%Y-W%U')}"
            limit = self.budget_config.weekly_limit
        elif period == "monthly":
            key = f"monthly-{now.strftime('%Y-%m')}"
            limit = self.budget_config.monthly_limit
        else:
            raise ValueError(f"Invalid period: {period}")
        
        current_cost = self.current_costs.get(key, 0.0)
        
        # Break down by category
        category_costs = defaultdict(float)
        recent_usage = [
            u for u in self.usage_history 
            if u.timestamp >= (now - timedelta(days=1 if period == "daily" else 7 if period == "weekly" else 30))
        ]
        
        for usage in recent_usage:
            category_costs[usage.category.value] += usage.cost
        
        return {
            "period": period,
            "current_cost": current_cost,
            "budget_limit": limit,
            "remaining_budget": max(0, limit - current_cost),
            "usage_percentage": (current_cost / limit * 100) if limit > 0 else 0,
            "category_breakdown": dict(category_costs),
            "is_over_budget": current_cost > limit,
            "is_near_limit": current_cost >= (limit * self.budget_config.warning_threshold)
        }
    
    def get_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations."""
        with self._lock:
            return self.optimization_recommendations.copy()
    
    def get_recent_alerts(self, hours: int = 24) -> List[CostAlert]:
        """Get recent alerts within specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
    
    def is_throttled(self, category: CostCategory) -> bool:
        """Check if a category is currently throttled."""
        with self._lock:
            return self._throttle_status.get(category, False)
    
    def reset_throttling(self):
        """Reset all throttling status."""
        with self._lock:
            self._throttle_status.clear()
            logger.info("Cost throttling reset")
    
    def export_usage_data(self, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export usage data for analysis."""
        start_date = start_date or (datetime.now() - timedelta(days=30))
        end_date = end_date or datetime.now()
        
        filtered_usage = [
            usage for usage in self.usage_history
            if start_date <= usage.timestamp <= end_date
        ]
        
        return [asdict(usage) for usage in filtered_usage]


# Global cost tracker instance
_cost_tracker = None


def get_cost_tracker(budget_config: Optional[BudgetConfig] = None) -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(budget_config)
    return _cost_tracker


def init_cost_tracker(budget_config: BudgetConfig) -> CostTracker:
    """Initialize global cost tracker with configuration."""
    global _cost_tracker
    _cost_tracker = CostTracker(budget_config)
    return _cost_tracker


# Decorator for automatic cost tracking
def track_cost(category: CostCategory, service: str, operation: str = "default"):
    """Decorator for automatic cost tracking of function calls."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Estimate cost based on execution time (placeholder logic)
                duration = time.time() - start_time
                estimated_cost = duration * 0.01  # $0.01 per second (adjust as needed)
                
                # Track the cost
                tracker = get_cost_tracker()
                tracker.track_external_api_usage(
                    service=service,
                    operation=operation,
                    cost=estimated_cost,
                    metadata={"duration_seconds": duration, "function": func.__name__}
                )
                
                return result
                
            except Exception as e:
                # Still track failed operations for cost analysis
                duration = time.time() - start_time
                estimated_cost = duration * 0.005  # Lower cost for failed operations
                
                tracker = get_cost_tracker()
                tracker.track_external_api_usage(
                    service=service,
                    operation=f"{operation}_failed",
                    cost=estimated_cost,
                    metadata={"duration_seconds": duration, "function": func.__name__, "error": str(e)}
                )
                
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    budget_config = BudgetConfig(
        daily_limit=50.0,
        weekly_limit=300.0,
        monthly_limit=1200.0,
        warning_threshold=0.7,
        critical_threshold=0.9,
        auto_throttle=True
    )
    
    tracker = CostTracker(budget_config)
    
    # Simulate some usage
    tracker.track_llm_usage("gpt-4o", 1000, 500, "text_generation")
    tracker.track_llm_usage("claude-3-5-sonnet-20240620", 2000, 1000, "code_analysis")
    tracker.track_gpu_usage("A100", 0.5, 3.0)
    tracker.track_external_api_usage("semantic_scholar", "paper_search", 0.10)
    
    # Get cost summary
    summary = tracker.get_cost_summary("daily")
    print(f"Daily cost summary: {json.dumps(summary, indent=2)}")
    
    # Get optimization recommendations
    recommendations = tracker.get_optimization_recommendations()
    print(f"\nOptimization recommendations: {len(recommendations)} found")
    for rec in recommendations:
        print(f"- {rec.description} (Priority: {rec.priority}, Savings: ${rec.potential_savings:.2f})")
    
    # Test model recommendation
    cheapest = LLMPricingModel.get_cheapest_model("general")
    print(f"\nCheapest general model: {cheapest}")
    
    cost = LLMPricingModel.calculate_cost("gpt-4o", 1000, 500)
    print(f"Cost for GPT-4o (1000 input, 500 output tokens): ${cost:.4f}")