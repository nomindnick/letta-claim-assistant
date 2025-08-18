"""
Letta Cost Tracking System for External LLM Providers.

Tracks token usage, calculates costs, enforces spending limits,
and provides usage analytics for external API providers.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from enum import Enum

from .letta_provider_bridge import ProviderConfiguration
from .logging_conf import get_logger

logger = get_logger(__name__)


class CostPeriod(Enum):
    """Billing period types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    TOTAL = "total"


@dataclass
class UsageRecord:
    """Individual usage record for cost tracking."""
    
    timestamp: datetime
    matter_id: str
    provider_name: str
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    request_type: str = "generation"  # generation, embedding
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "matter_id": self.matter_id,
            "provider_name": self.provider_name,
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "request_type": self.request_type,
            "metadata": json.dumps(self.metadata)
        }


@dataclass
class SpendingLimit:
    """Spending limit configuration."""
    
    period: CostPeriod
    limit_usd: float
    warning_threshold: float = 0.8  # Warn at 80% of limit
    enabled: bool = True
    
    def check_limit(self, current_spend: float) -> Dict[str, Any]:
        """
        Check if spending is within limits.
        
        Returns:
            Dictionary with limit status
        """
        if not self.enabled:
            return {"within_limit": True, "warning": False}
        
        percentage = (current_spend / self.limit_usd) * 100 if self.limit_usd > 0 else 0
        
        return {
            "within_limit": current_spend <= self.limit_usd,
            "warning": current_spend >= (self.limit_usd * self.warning_threshold),
            "current_spend": current_spend,
            "limit": self.limit_usd,
            "percentage": percentage,
            "remaining": max(0, self.limit_usd - current_spend)
        }


class LettaCostTracker:
    """
    Cost tracking and management for Letta LLM providers.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize cost tracker.
        
        Args:
            db_path: Path to SQLite database for usage history
        """
        self.db_path = db_path or (Path.home() / ".letta-claim" / "usage_history.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Spending limits
        self.limits: Dict[str, SpendingLimit] = {}
        self.limits_file = self.db_path.parent / "spending_limits.json"
        
        # Initialize database
        self._init_database()
        self._load_limits()
        
        logger.info(f"Cost tracker initialized with database at {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database for usage tracking."""
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                matter_id TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                request_type TEXT DEFAULT 'generation',
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp 
            ON usage_records(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_matter 
            ON usage_records(matter_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_provider 
            ON usage_records(provider_name)
        """)
        
        # Create aggregated stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_stats (
                period TEXT NOT NULL,
                period_start TEXT NOT NULL,
                matter_id TEXT,
                provider_name TEXT,
                total_cost_usd REAL NOT NULL,
                total_input_tokens INTEGER NOT NULL,
                total_output_tokens INTEGER NOT NULL,
                request_count INTEGER NOT NULL,
                PRIMARY KEY (period, period_start, matter_id, provider_name)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_limits(self) -> None:
        """Load spending limits from disk."""
        if self.limits_file.exists():
            try:
                with open(self.limits_file, 'r') as f:
                    data = json.load(f)
                    
                for period_name, limit_data in data.items():
                    try:
                        period = CostPeriod(period_name)
                        self.limits[period_name] = SpendingLimit(
                            period=period,
                            limit_usd=limit_data["limit_usd"],
                            warning_threshold=limit_data.get("warning_threshold", 0.8),
                            enabled=limit_data.get("enabled", True)
                        )
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Invalid limit configuration for {period_name}: {e}")
                        
                logger.debug(f"Loaded {len(self.limits)} spending limits")
                
            except Exception as e:
                logger.warning(f"Could not load spending limits: {e}")
    
    def _save_limits(self) -> None:
        """Save spending limits to disk."""
        try:
            data = {}
            for period_name, limit in self.limits.items():
                data[period_name] = {
                    "limit_usd": limit.limit_usd,
                    "warning_threshold": limit.warning_threshold,
                    "enabled": limit.enabled
                }
            
            with open(self.limits_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save spending limits: {e}")
    
    def record_usage(
        self,
        matter_id: str,
        provider_config: ProviderConfiguration,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "generation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """
        Record token usage and calculate cost.
        
        Args:
            matter_id: Matter identifier
            provider_config: Provider configuration used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_type: Type of request (generation, embedding)
            metadata: Additional metadata to store
            
        Returns:
            Usage record with calculated cost
        """
        # Calculate cost
        cost = self._calculate_cost(provider_config, input_tokens, output_tokens)
        
        # Create usage record
        record = UsageRecord(
            timestamp=datetime.now(),
            matter_id=matter_id,
            provider_name=provider_config.provider_type,
            model_name=provider_config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            request_type=request_type,
            metadata=metadata or {}
        )
        
        # Store in database
        self._store_usage_record(record)
        
        # Check spending limits
        self._check_limits_after_usage(record)
        
        logger.debug(
            f"Recorded usage: {input_tokens} in, {output_tokens} out, ${cost:.4f}",
            matter_id=matter_id,
            model=provider_config.model_name
        )
        
        return record
    
    def _calculate_cost(
        self,
        provider_config: ProviderConfiguration,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost based on token counts and provider pricing."""
        input_cost = (input_tokens / 1000) * provider_config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * provider_config.cost_per_1k_output_tokens
        return round(input_cost + output_cost, 6)
    
    def _store_usage_record(self, record: UsageRecord) -> None:
        """Store usage record in database."""
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO usage_records (
                    timestamp, matter_id, provider_name, model_name,
                    input_tokens, output_tokens, cost_usd, request_type, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp.isoformat(),
                record.matter_id,
                record.provider_name,
                record.model_name,
                record.input_tokens,
                record.output_tokens,
                record.cost_usd,
                record.request_type,
                json.dumps(record.metadata)
            ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
        finally:
            conn.close()
    
    def _check_limits_after_usage(self, record: UsageRecord) -> None:
        """Check spending limits after recording usage."""
        for period_name, limit in self.limits.items():
            if not limit.enabled:
                continue
            
            current_spend = self.get_spending_for_period(limit.period)
            status = limit.check_limit(current_spend)
            
            if not status["within_limit"]:
                logger.error(
                    f"SPENDING LIMIT EXCEEDED: {period_name} limit of ${limit.limit_usd:.2f} exceeded",
                    current_spend=current_spend,
                    matter_id=record.matter_id
                )
            elif status["warning"]:
                logger.warning(
                    f"Spending warning: {status['percentage']:.1f}% of {period_name} limit used",
                    current_spend=current_spend,
                    limit=limit.limit_usd,
                    remaining=status["remaining"]
                )
    
    def set_spending_limit(
        self,
        period: CostPeriod,
        limit_usd: float,
        warning_threshold: float = 0.8
    ) -> None:
        """
        Set a spending limit for a period.
        
        Args:
            period: Period type (daily, weekly, monthly, total)
            limit_usd: Maximum spending in USD
            warning_threshold: Threshold for warning (0-1)
        """
        self.limits[period.value] = SpendingLimit(
            period=period,
            limit_usd=limit_usd,
            warning_threshold=warning_threshold,
            enabled=True
        )
        self._save_limits()
        
        logger.info(f"Set {period.value} spending limit to ${limit_usd:.2f}")
    
    def remove_spending_limit(self, period: CostPeriod) -> None:
        """Remove a spending limit."""
        if period.value in self.limits:
            del self.limits[period.value]
            self._save_limits()
            logger.info(f"Removed {period.value} spending limit")
    
    def get_spending_for_period(
        self,
        period: CostPeriod,
        matter_id: Optional[str] = None,
        provider_name: Optional[str] = None
    ) -> float:
        """
        Get total spending for a period.
        
        Args:
            period: Period to query
            matter_id: Optional matter filter
            provider_name: Optional provider filter
            
        Returns:
            Total spending in USD
        """
        # Ensure database exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._init_database()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Calculate date range based on period
            now = datetime.now()
            if period == CostPeriod.DAILY:
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == CostPeriod.WEEKLY:
                start_date = now - timedelta(days=now.weekday())
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == CostPeriod.MONTHLY:
                start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:  # TOTAL
                start_date = datetime.min
            
            # Build query
            query = """
                SELECT SUM(cost_usd) FROM usage_records
                WHERE timestamp >= ?
            """
            params = [start_date.isoformat()]
            
            if matter_id:
                query += " AND matter_id = ?"
                params.append(matter_id)
            
            if provider_name:
                query += " AND provider_name = ?"
                params.append(provider_name)
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            return result[0] if result[0] else 0.0
            
        finally:
            conn.close()
    
    def get_usage_summary(
        self,
        matter_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get usage summary for a matter or globally.
        
        Args:
            matter_id: Optional matter filter
            days: Number of days to include
            
        Returns:
            Usage summary with costs and token counts
        """
        # Ensure database exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._init_database()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            # Base query
            base_where = "WHERE timestamp >= ?"
            params = [start_date.isoformat()]
            
            if matter_id:
                base_where += " AND matter_id = ?"
                params.append(matter_id)
            
            # Get total costs and tokens
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM usage_records
                {base_where}
            """, params)
            
            totals = cursor.fetchone()
            
            # Get breakdown by provider
            cursor.execute(f"""
                SELECT 
                    provider_name,
                    model_name,
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM usage_records
                {base_where}
                GROUP BY provider_name, model_name
                ORDER BY total_cost DESC
            """, params)
            
            provider_breakdown = []
            for row in cursor.fetchall():
                provider_breakdown.append({
                    "provider": row[0],
                    "model": row[1],
                    "requests": row[2],
                    "cost": round(row[3], 4) if row[3] else 0,
                    "input_tokens": row[4] or 0,
                    "output_tokens": row[5] or 0
                })
            
            # Get daily costs for trend
            cursor.execute(f"""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(cost_usd) as daily_cost,
                    COUNT(*) as daily_requests
                FROM usage_records
                {base_where}
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            """, params)
            
            daily_trend = []
            for row in cursor.fetchall():
                daily_trend.append({
                    "date": row[0],
                    "cost": round(row[1], 4) if row[1] else 0,
                    "requests": row[2]
                })
            
            summary = {
                "period_days": days,
                "matter_id": matter_id,
                "total_requests": totals[0] or 0,
                "total_cost_usd": round(totals[1], 4) if totals[1] else 0,
                "total_input_tokens": totals[2] or 0,
                "total_output_tokens": totals[3] or 0,
                "average_cost_per_request": round(totals[1] / totals[0], 4) if totals[0] else 0,
                "provider_breakdown": provider_breakdown,
                "daily_trend": daily_trend,
                "spending_limits": self._get_limit_status()
            }
            
            return summary
            
        finally:
            conn.close()
    
    def _get_limit_status(self) -> Dict[str, Any]:
        """Get status of all spending limits."""
        status = {}
        
        for period_name, limit in self.limits.items():
            if limit.enabled:
                current_spend = self.get_spending_for_period(limit.period)
                status[period_name] = limit.check_limit(current_spend)
        
        return status
    
    def check_budget_available(
        self,
        provider_config: ProviderConfiguration,
        estimated_tokens: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if budget is available for a request.
        
        Args:
            provider_config: Provider to use
            estimated_tokens: Estimated total tokens
            
        Returns:
            Tuple of (allowed, reason)
        """
        # Estimate cost (assume 50/50 input/output split)
        estimated_input = estimated_tokens // 2
        estimated_output = estimated_tokens // 2
        estimated_cost = self._calculate_cost(provider_config, estimated_input, estimated_output)
        
        # Check each limit
        for period_name, limit in self.limits.items():
            if not limit.enabled:
                continue
            
            current_spend = self.get_spending_for_period(limit.period)
            projected_spend = current_spend + estimated_cost
            
            if projected_spend > limit.limit_usd:
                return False, f"{period_name} spending limit would be exceeded (${projected_spend:.2f} > ${limit.limit_usd:.2f})"
        
        return True, None
    
    def export_usage_data(
        self,
        output_path: Path,
        format: str = "csv",
        matter_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """
        Export usage data to file.
        
        Args:
            output_path: Path to output file
            format: Export format (csv, json)
            matter_id: Optional matter filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            True if export successful
        """
        # Ensure database exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._init_database()
            # If database was just created, no data to export
            logger.info("No usage data to export (database empty)")
            return True
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Build query
            query = "SELECT * FROM usage_records WHERE 1=1"
            params = []
            
            if matter_id:
                query += " AND matter_id = ?"
                params.append(matter_id)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            if format == "csv":
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(rows)
            
            elif format == "json":
                data = []
                for row in rows:
                    data.append(dict(zip(columns, row)))
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(rows)} usage records to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export usage data: {e}")
            return False
            
        finally:
            conn.close()


# Global cost tracker instance
cost_tracker = LettaCostTracker()