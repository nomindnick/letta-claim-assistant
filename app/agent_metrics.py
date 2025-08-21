"""
Agent Metrics - Performance monitoring for the stateful agent architecture.

Tracks performance metrics, tool usage patterns, and memory utilization
with CPU-appropriate expectations.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
import statistics

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ResponseMetrics:
    """Metrics for a single agent response."""
    
    matter_id: str
    message_id: str
    timestamp: datetime
    response_time: float  # seconds
    message_type: str  # 'simple', 'search', 'memory_recall'
    tools_used: List[str] = field(default_factory=list)
    search_performed: bool = False
    memory_updated: bool = False
    citations_count: int = 0
    token_estimate: int = 0
    error_occurred: bool = False
    error_message: Optional[str] = None


@dataclass
class MemoryMetrics:
    """Metrics for agent memory utilization."""
    
    matter_id: str
    agent_id: str
    timestamp: datetime
    memory_blocks: Dict[str, int]  # block_name -> token_count
    total_tokens: int
    capacity_percentage: float
    compression_triggered: bool = False


@dataclass
class ToolMetrics:
    """Metrics for tool usage."""
    
    tool_name: str
    invocation_count: int
    success_count: int
    failure_count: int
    total_execution_time: float
    average_execution_time: float
    last_used: datetime


class AgentMetricsCollector:
    """
    Collects and analyzes performance metrics for stateful agents.
    
    Adjusted for CPU-only inference with realistic expectations.
    """
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        """
        Initialize metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics files
        """
        self.metrics_dir = metrics_dir or Path.home() / ".letta-claim" / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage
        self.response_metrics: deque = deque(maxlen=1000)  # Keep last 1000 responses
        self.memory_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        
        # Performance thresholds (CPU-adjusted)
        self.PERFORMANCE_THRESHOLDS = {
            "simple_response": 300.0,  # 5 minutes
            "search_response": 360.0,  # 6 minutes
            "memory_recall": 240.0,  # 4 minutes
            "tool_execution": 60.0,  # 1 minute per tool
        }
        
        # Search reduction tracking
        self.search_patterns: Dict[str, List[bool]] = defaultdict(list)  # matter_id -> [performed_search]
        
        logger.info("AgentMetricsCollector initialized")
    
    def record_response(
        self,
        matter_id: str,
        message_id: str,
        response_time: float,
        tools_used: List[str],
        search_performed: bool,
        memory_updated: bool,
        citations_count: int,
        error: Optional[Exception] = None
    ) -> ResponseMetrics:
        """
        Record metrics for an agent response.
        
        Args:
            matter_id: Matter ID
            message_id: Unique message ID
            response_time: Time taken to generate response (seconds)
            tools_used: List of tools used
            search_performed: Whether search was performed
            memory_updated: Whether memory was updated
            citations_count: Number of citations in response
            error: Any error that occurred
            
        Returns:
            ResponseMetrics object
        """
        # Determine message type
        if search_performed:
            message_type = "search"
        elif tools_used and "recall_memory" in tools_used:
            message_type = "memory_recall"
        else:
            message_type = "simple"
        
        # Create metrics object
        metrics = ResponseMetrics(
            matter_id=matter_id,
            message_id=message_id,
            timestamp=datetime.now(),
            response_time=response_time,
            message_type=message_type,
            tools_used=tools_used,
            search_performed=search_performed,
            memory_updated=memory_updated,
            citations_count=citations_count,
            error_occurred=error is not None,
            error_message=str(error) if error else None
        )
        
        # Store metrics
        self.response_metrics.append(metrics)
        
        # Track search patterns for reduction analysis
        self.search_patterns[matter_id].append(search_performed)
        
        # Update tool metrics
        for tool in tools_used:
            self._update_tool_metrics(tool, response_time, error is None)
        
        # Log if response time exceeds threshold
        threshold = self.PERFORMANCE_THRESHOLDS.get(message_type, 600.0)
        if response_time > threshold * 1.5:  # 50% over threshold
            logger.warning(
                f"Response time exceeded threshold",
                matter_id=matter_id,
                message_type=message_type,
                response_time=response_time,
                threshold=threshold
            )
        
        return metrics
    
    def record_memory_state(
        self,
        matter_id: str,
        agent_id: str,
        memory_blocks: Dict[str, int],
        capacity_limit: int = 16000
    ) -> MemoryMetrics:
        """
        Record memory utilization metrics.
        
        Args:
            matter_id: Matter ID
            agent_id: Agent ID
            memory_blocks: Dictionary of block names to token counts
            capacity_limit: Total token capacity limit
            
        Returns:
            MemoryMetrics object
        """
        total_tokens = sum(memory_blocks.values())
        capacity_percentage = (total_tokens / capacity_limit) * 100
        
        metrics = MemoryMetrics(
            matter_id=matter_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            memory_blocks=memory_blocks,
            total_tokens=total_tokens,
            capacity_percentage=capacity_percentage,
            compression_triggered=capacity_percentage > 80
        )
        
        # Store metrics
        self.memory_metrics[matter_id].append(metrics)
        
        # Log if approaching capacity
        if capacity_percentage > 80:
            logger.warning(
                f"Memory approaching capacity",
                matter_id=matter_id,
                capacity_percentage=capacity_percentage,
                total_tokens=total_tokens
            )
        
        return metrics
    
    def _update_tool_metrics(self, tool_name: str, execution_time: float, success: bool):
        """Update metrics for a specific tool."""
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics(
                tool_name=tool_name,
                invocation_count=0,
                success_count=0,
                failure_count=0,
                total_execution_time=0.0,
                average_execution_time=0.0,
                last_used=datetime.now()
            )
        
        metrics = self.tool_metrics[tool_name]
        metrics.invocation_count += 1
        metrics.total_execution_time += execution_time
        metrics.average_execution_time = metrics.total_execution_time / metrics.invocation_count
        metrics.last_used = datetime.now()
        
        if success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
    
    def get_search_reduction_rate(self, matter_id: str, window: int = 10) -> float:
        """
        Calculate search reduction rate for a matter.
        
        Compares early interactions vs recent interactions to measure learning.
        
        Args:
            matter_id: Matter ID
            window: Number of interactions to compare
            
        Returns:
            Reduction percentage (0-100)
        """
        patterns = self.search_patterns.get(matter_id, [])
        
        if len(patterns) < window * 2:
            return 0.0  # Not enough data
        
        # Compare first window vs last window
        early_searches = sum(patterns[:window])
        recent_searches = sum(patterns[-window:])
        
        if early_searches == 0:
            return 0.0
        
        reduction = ((early_searches - recent_searches) / early_searches) * 100
        return max(0.0, reduction)  # Don't return negative reduction
    
    def get_performance_summary(self, matter_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Args:
            matter_id: Optional matter ID for matter-specific stats
            
        Returns:
            Dictionary of performance statistics
        """
        # Filter metrics by matter if specified
        if matter_id:
            metrics = [m for m in self.response_metrics if m.matter_id == matter_id]
        else:
            metrics = list(self.response_metrics)
        
        if not metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics by message type
        stats = {}
        for msg_type in ["simple", "search", "memory_recall"]:
            type_metrics = [m for m in metrics if m.message_type == msg_type]
            if type_metrics:
                response_times = [m.response_time for m in type_metrics]
                stats[msg_type] = {
                    "count": len(type_metrics),
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "within_threshold": sum(
                        1 for t in response_times 
                        if t <= self.PERFORMANCE_THRESHOLDS.get(msg_type, 600)
                    ) / len(response_times) * 100
                }
        
        # Overall statistics
        all_response_times = [m.response_time for m in metrics]
        error_rate = sum(1 for m in metrics if m.error_occurred) / len(metrics) * 100
        
        summary = {
            "total_interactions": len(metrics),
            "message_type_stats": stats,
            "overall": {
                "avg_response_time": statistics.mean(all_response_times),
                "median_response_time": statistics.median(all_response_times),
                "error_rate": error_rate,
                "search_performed_rate": sum(1 for m in metrics if m.search_performed) / len(metrics) * 100,
                "memory_updated_rate": sum(1 for m in metrics if m.memory_updated) / len(metrics) * 100,
                "avg_citations": statistics.mean([m.citations_count for m in metrics])
            },
            "tool_usage": {
                name: {
                    "invocations": tool.invocation_count,
                    "success_rate": (tool.success_count / tool.invocation_count * 100) if tool.invocation_count > 0 else 0,
                    "avg_execution_time": tool.average_execution_time
                }
                for name, tool in self.tool_metrics.items()
            }
        }
        
        # Add search reduction if matter-specific
        if matter_id:
            summary["search_reduction_rate"] = self.get_search_reduction_rate(matter_id)
        
        # Add performance note for CPU
        summary["performance_note"] = "Times shown are for CPU-only inference. GPU would be 10-100x faster."
        
        return summary
    
    def save_metrics(self, matter_id: Optional[str] = None):
        """
        Save metrics to disk for persistence.
        
        Args:
            matter_id: Optional matter ID to save specific metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if matter_id:
            filename = f"metrics_{matter_id}_{timestamp}.json"
            metrics_data = {
                "matter_id": matter_id,
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_performance_summary(matter_id),
                "search_reduction": self.get_search_reduction_rate(matter_id),
                "recent_responses": [
                    asdict(m) for m in self.response_metrics 
                    if m.matter_id == matter_id
                ][-50:]  # Last 50 responses
            }
        else:
            filename = f"metrics_global_{timestamp}.json"
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_performance_summary(),
                "tool_metrics": {
                    name: asdict(tool) for name, tool in self.tool_metrics.items()
                }
            }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        filepath = self.metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=convert_datetime)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def get_memory_trend(self, matter_id: str) -> Dict[str, Any]:
        """
        Analyze memory usage trends for a matter.
        
        Args:
            matter_id: Matter ID
            
        Returns:
            Dictionary with memory trend analysis
        """
        metrics = self.memory_metrics.get(matter_id, [])
        
        if not metrics:
            return {"error": "No memory metrics available"}
        
        # Get latest and earliest metrics
        latest = metrics[-1]
        earliest = metrics[0] if len(metrics) > 1 else latest
        
        # Calculate growth rate
        time_diff = (latest.timestamp - earliest.timestamp).total_seconds()
        token_growth = latest.total_tokens - earliest.total_tokens
        growth_rate = (token_growth / time_diff * 3600) if time_diff > 0 else 0  # Tokens per hour
        
        return {
            "current_usage": latest.total_tokens,
            "capacity_percentage": latest.capacity_percentage,
            "growth_rate_per_hour": growth_rate,
            "time_to_capacity": ((16000 - latest.total_tokens) / growth_rate) if growth_rate > 0 else float('inf'),
            "compression_needed": latest.compression_triggered,
            "block_usage": latest.memory_blocks,
            "measurement_period": time_diff / 3600  # hours
        }
    
    def generate_report(self, matter_id: Optional[str] = None) -> str:
        """
        Generate a human-readable performance report.
        
        Args:
            matter_id: Optional matter ID for matter-specific report
            
        Returns:
            Formatted report string
        """
        summary = self.get_performance_summary(matter_id)
        
        report = []
        report.append("=" * 60)
        report.append("AGENT PERFORMANCE REPORT")
        report.append("=" * 60)
        
        if matter_id:
            report.append(f"Matter ID: {matter_id}")
            memory_trend = self.get_memory_trend(matter_id)
            if "error" not in memory_trend:
                report.append(f"Memory Usage: {memory_trend['capacity_percentage']:.1f}% of capacity")
                report.append(f"Search Reduction: {summary.get('search_reduction_rate', 0):.1f}%")
        
        report.append(f"Total Interactions: {summary['total_interactions']}")
        report.append("")
        
        # Response time statistics
        report.append("RESPONSE TIMES (CPU-adjusted expectations):")
        for msg_type, stats in summary.get('message_type_stats', {}).items():
            report.append(f"  {msg_type.title()}:")
            report.append(f"    Count: {stats['count']}")
            report.append(f"    Average: {stats['avg_response_time']:.1f}s")
            report.append(f"    Median: {stats['median_response_time']:.1f}s")
            report.append(f"    Within Threshold: {stats['within_threshold']:.1f}%")
        
        # Tool usage
        if summary.get('tool_usage'):
            report.append("")
            report.append("TOOL USAGE:")
            for tool_name, tool_stats in summary['tool_usage'].items():
                report.append(f"  {tool_name}:")
                report.append(f"    Invocations: {tool_stats['invocations']}")
                report.append(f"    Success Rate: {tool_stats['success_rate']:.1f}%")
        
        # Overall metrics
        overall = summary.get('overall', {})
        report.append("")
        report.append("OVERALL METRICS:")
        report.append(f"  Error Rate: {overall.get('error_rate', 0):.1f}%")
        report.append(f"  Search Rate: {overall.get('search_performed_rate', 0):.1f}%")
        report.append(f"  Avg Citations: {overall.get('avg_citations', 0):.1f}")
        
        report.append("")
        report.append(summary.get('performance_note', ''))
        report.append("=" * 60)
        
        return "\n".join(report)


# Global metrics collector instance
metrics_collector = AgentMetricsCollector()