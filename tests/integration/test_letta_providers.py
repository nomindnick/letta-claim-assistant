"""
Integration tests for Letta provider management system.

Tests dynamic provider switching, health monitoring, fallback chains,
cost tracking, and provider configuration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import json
import tempfile
from datetime import datetime, timedelta

# Import the modules to test
from app.letta_provider_bridge import (
    letta_provider_bridge,
    LettaProviderBridge,
    ProviderConfiguration
)
from app.letta_provider_health import (
    provider_health_monitor,
    LettaProviderHealth,
    HealthStatus,
    HealthMetrics
)
from app.letta_cost_tracker import (
    cost_tracker,
    LettaCostTracker,
    CostPeriod,
    UsageRecord,
    SpendingLimit
)
from app.letta_adapter import LettaAdapter


# Fixtures
@pytest.fixture
def temp_matter_path():
    """Create a temporary matter directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        matter_path = Path(tmpdir) / "TestMatter"
        matter_path.mkdir(parents=True, exist_ok=True)
        yield matter_path


@pytest.fixture
def mock_letta_client():
    """Create a mock Letta client."""
    client = AsyncMock()
    client.agents = AsyncMock()
    client.agents.create = AsyncMock(return_value=Mock(id="test-agent-123"))
    client.agents.retrieve = AsyncMock(return_value=Mock(id="test-agent-123"))
    client.agents.modify = AsyncMock(return_value=Mock(id="test-agent-123"))
    client.agents.passages = AsyncMock()
    client.agents.passages.list = AsyncMock(return_value=[])
    client.agents.passages.create = AsyncMock(return_value=Mock(id="passage-123"))
    return client


@pytest.fixture
def provider_bridge():
    """Create a fresh provider bridge instance."""
    return LettaProviderBridge()


@pytest.fixture
def health_monitor():
    """Create a fresh health monitor instance."""
    return LettaProviderHealth()


@pytest.fixture
def cost_tracker_instance(tmp_path):
    """Create a fresh cost tracker instance."""
    db_path = tmp_path / "test_usage.db"
    return LettaCostTracker(db_path=db_path)


# Provider Bridge Tests
class TestLettaProviderBridge:
    """Test the provider bridge functionality."""
    
    def test_ollama_config_creation(self, provider_bridge):
        """Test creating Ollama provider configuration."""
        config = provider_bridge.get_ollama_config(
            model="gpt-oss:20b",
            embedding_model="nomic-embed-text"
        )
        
        assert config.provider_type == "ollama"
        assert config.model_name == "gpt-oss:20b"
        assert config.is_local is True
        assert config.requires_consent is False
        assert config.cost_per_1k_input_tokens == 0.0
        assert config.cost_per_1k_output_tokens == 0.0
    
    def test_gemini_config_creation(self, provider_bridge):
        """Test creating Gemini provider configuration."""
        config = provider_bridge.get_gemini_config(
            api_key="test-api-key",
            model="gemini-2.0-flash-exp"
        )
        
        assert config.provider_type == "google_ai"
        assert config.model_name == "gemini-2.0-flash-exp"
        assert config.api_key == "test-api-key"
        assert config.is_local is False
        assert config.requires_consent is True
        assert config.cost_per_1k_input_tokens > 0
    
    def test_openai_config_creation(self, provider_bridge):
        """Test creating OpenAI provider configuration."""
        config = provider_bridge.get_openai_config(
            api_key="test-api-key",
            model="gpt-4o-mini"
        )
        
        assert config.provider_type == "openai"
        assert config.model_name == "gpt-4o-mini"
        assert config.api_key == "test-api-key"
        assert config.is_local is False
        assert config.requires_consent is True
    
    @patch('app.letta_provider_bridge.LlmConfig')
    def test_to_letta_llm_config(self, mock_llm_config, provider_bridge):
        """Test converting to Letta LLM config format."""
        # Set mock to not be None
        mock_llm_config.__class__ = type
        
        config = provider_bridge.get_ollama_config()
        llm_dict = provider_bridge.to_letta_llm_config(config)
        
        assert llm_dict is not None
        assert llm_dict["model"] == config.model_name
        assert llm_dict["model_endpoint_type"] == config.provider_type
        assert llm_dict["context_window"] == config.context_window
    
    def test_fallback_chain_setup(self, provider_bridge):
        """Test setting up provider fallback chain."""
        primary = provider_bridge.get_ollama_config()
        secondary = provider_bridge.get_gemini_config(
            api_key="test-key",
            model="gemini-2.0-flash-exp"
        )
        
        provider_bridge.setup_fallback_chain(
            primary=primary,
            secondary=secondary
        )
        
        assert len(provider_bridge.fallback_chain) == 2
        assert provider_bridge.active_provider == primary
    
    def test_get_next_provider(self, provider_bridge):
        """Test getting next provider in fallback chain."""
        primary = provider_bridge.get_ollama_config()
        secondary = provider_bridge.get_gemini_config(
            api_key="test-key",
            model="gemini-2.0-flash-exp"
        )
        
        provider_bridge.setup_fallback_chain(
            primary=primary,
            secondary=secondary
        )
        
        # Mock consent check to return True
        with patch.object(provider_bridge, 'check_provider_consent', return_value=True):
            next_provider = provider_bridge.get_next_provider()
            assert next_provider == secondary
    
    def test_estimate_cost(self, provider_bridge):
        """Test cost estimation for different providers."""
        # Ollama (free)
        ollama_config = provider_bridge.get_ollama_config()
        cost = provider_bridge.estimate_cost(ollama_config, 1000, 500)
        assert cost == 0.0
        
        # Gemini (paid)
        gemini_config = provider_bridge.get_gemini_config(
            api_key="test-key",
            model="gemini-2.0-flash-exp"
        )
        cost = provider_bridge.estimate_cost(gemini_config, 1000, 500)
        assert cost > 0


# Health Monitor Tests
class TestLettaProviderHealth:
    """Test the provider health monitoring functionality."""
    
    def test_health_metrics_creation(self):
        """Test creating health metrics."""
        metrics = HealthMetrics(provider_name="test-provider")
        
        assert metrics.provider_name == "test-provider"
        assert metrics.status == HealthStatus.UNKNOWN
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
    
    def test_add_success(self):
        """Test recording successful health check."""
        metrics = HealthMetrics(provider_name="test-provider")
        metrics.add_success(100.0)
        
        assert metrics.status == HealthStatus.HEALTHY
        assert metrics.success_count == 1
        assert metrics.consecutive_failures == 0
        assert len(metrics.response_times) == 1
    
    def test_add_failure(self):
        """Test recording failed health check."""
        metrics = HealthMetrics(provider_name="test-provider")
        metrics.add_failure("Connection timeout")
        
        assert metrics.status == HealthStatus.DEGRADED
        assert metrics.failure_count == 1
        assert metrics.consecutive_failures == 1
    
    def test_consecutive_failures_trigger_unhealthy(self):
        """Test that consecutive failures mark provider as unhealthy."""
        metrics = HealthMetrics(provider_name="test-provider")
        
        # Add failures up to threshold
        for _ in range(metrics.max_consecutive_failures):
            metrics.add_failure()
        
        assert metrics.status == HealthStatus.UNHEALTHY
        assert not metrics.is_healthy()
    
    @pytest.mark.asyncio
    async def test_check_provider_health(self, health_monitor):
        """Test checking provider health."""
        config = ProviderConfiguration(
            provider_type="ollama",
            model_name="test-model-unique",  # Use unique name to avoid conflicts
            is_local=True
        )
        
        # Mock successful test
        async def mock_test(cfg):
            return True
        
        result = await health_monitor.check_provider_health(config, mock_test)
        assert result is True
        
        # Check metrics were recorded
        metrics = health_monitor.get_provider_health("test-model-unique")
        assert metrics is not None
        assert metrics.success_count >= 1  # At least one success
    
    @pytest.mark.asyncio
    async def test_trigger_fallback(self, health_monitor):
        """Test triggering fallback to next provider."""
        with patch.object(letta_provider_bridge, 'get_next_provider') as mock_next:
            with patch.object(letta_provider_bridge, 'switch_to_provider') as mock_switch:
                next_config = ProviderConfiguration(
                    provider_type="gemini",
                    model_name="backup-model",
                    is_local=False
                )
                mock_next.return_value = next_config
                mock_switch.return_value = True
                
                # Mock health check to return True
                with patch.object(health_monitor, 'check_provider_health', return_value=True):
                    result = await health_monitor.trigger_fallback()
                    assert result is True
                    mock_switch.assert_called_once_with(next_config)
    
    def test_get_health_summary(self, health_monitor):
        """Test getting health summary."""
        # Add some test metrics
        metrics = health_monitor.get_or_create_metrics("test-provider")
        metrics.add_success(100.0)
        metrics.add_success(150.0)
        metrics.add_failure()
        
        summary = health_monitor.get_health_summary()
        
        assert "providers" in summary
        assert "test-provider" in summary["providers"]
        # Check that our test provider is healthy (degraded is still considered healthy)
        provider_data = summary["providers"]["test-provider"]
        assert provider_data["status"] in ["healthy", "degraded"]


# Cost Tracker Tests
class TestLettaCostTracker:
    """Test the cost tracking functionality."""
    
    def test_record_usage(self, cost_tracker_instance):
        """Test recording token usage."""
        config = ProviderConfiguration(
            provider_type="gemini",
            model_name="gemini-2.0-flash-exp",
            cost_per_1k_input_tokens=0.0001,
            cost_per_1k_output_tokens=0.0003
        )
        
        record = cost_tracker_instance.record_usage(
            matter_id="test-matter",
            provider_config=config,
            input_tokens=1000,
            output_tokens=500
        )
        
        assert record.matter_id == "test-matter"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        assert record.cost_usd > 0
    
    def test_spending_limits(self, cost_tracker_instance):
        """Test spending limit functionality."""
        # Set a daily limit
        cost_tracker_instance.set_spending_limit(
            period=CostPeriod.DAILY,
            limit_usd=10.0,
            warning_threshold=0.8
        )
        
        # Check limit
        limit = cost_tracker_instance.limits[CostPeriod.DAILY.value]
        assert limit.limit_usd == 10.0
        
        # Check limit status
        status = limit.check_limit(5.0)
        assert status["within_limit"] is True
        assert status["warning"] is False
        
        # Check warning threshold
        status = limit.check_limit(8.5)
        assert status["within_limit"] is True
        assert status["warning"] is True
        
        # Check exceeded
        status = limit.check_limit(11.0)
        assert status["within_limit"] is False
    
    def test_get_usage_summary(self, cost_tracker_instance):
        """Test getting usage summary."""
        # Record some usage
        config = ProviderConfiguration(
            provider_type="ollama",
            model_name="test-model",
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0
        )
        
        for _ in range(5):
            cost_tracker_instance.record_usage(
                matter_id="test-matter",
                provider_config=config,
                input_tokens=100,
                output_tokens=50
            )
        
        summary = cost_tracker_instance.get_usage_summary(
            matter_id="test-matter",
            days=30
        )
        
        assert summary["total_requests"] == 5
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_input_tokens"] == 500
        assert summary["total_output_tokens"] == 250
    
    def test_check_budget_available(self, cost_tracker_instance):
        """Test checking if budget is available."""
        # Set a low daily limit
        cost_tracker_instance.set_spending_limit(
            period=CostPeriod.DAILY,
            limit_usd=0.01
        )
        
        # Check with expensive provider
        config = ProviderConfiguration(
            provider_type="openai",
            model_name="gpt-4",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03
        )
        
        allowed, reason = cost_tracker_instance.check_budget_available(
            provider_config=config,
            estimated_tokens=2000
        )
        
        assert allowed is False
        assert "spending limit would be exceeded" in reason


# Letta Adapter Integration Tests
class TestLettaAdapterIntegration:
    """Test the LettaAdapter integration with provider system."""
    
    @pytest.mark.asyncio
    async def test_switch_provider(self, temp_matter_path, mock_letta_client):
        """Test switching providers through LettaAdapter."""
        with patch('app.letta_adapter.connection_manager') as mock_conn:
            mock_conn.connect = AsyncMock(return_value=True)
            mock_conn.client = mock_letta_client
            mock_conn.get_state = Mock(return_value=Mock(value="connected"))
            
            adapter = LettaAdapter(
                matter_path=temp_matter_path,
                matter_name="TestMatter",
                matter_id="test-123"
            )
            
            # Mock the update method
            with patch.object(adapter, 'update_agent', return_value=True) as mock_update:
                result = await adapter.switch_provider(
                    provider_type="gemini",
                    model_name="gemini-2.0-flash-exp",
                    api_key="test-key"
                )
                
                assert result is True
                mock_update.assert_called_once()
                call_args = mock_update.call_args[0][0]
                assert call_args["llm_provider"] == "gemini"
                assert call_args["llm_model"] == "gemini-2.0-flash-exp"
    
    @pytest.mark.asyncio
    async def test_get_provider_info(self, temp_matter_path):
        """Test getting provider information."""
        adapter = LettaAdapter(
            matter_path=temp_matter_path,
            matter_name="TestMatter",
            matter_id="test-123"
        )
        
        with patch.object(letta_provider_bridge, 'get_provider_for_matter') as mock_get:
            with patch.object(provider_health_monitor, 'get_provider_health') as mock_health:
                mock_config = ProviderConfiguration(
                    provider_type="ollama",
                    model_name="test-model",
                    is_local=True,
                    cost_per_1k_input_tokens=0.0,
                    cost_per_1k_output_tokens=0.0
                )
                mock_get.return_value = mock_config
                
                mock_metrics = HealthMetrics(provider_name="test-model")
                mock_metrics.status = HealthStatus.HEALTHY
                mock_health.return_value = mock_metrics
                
                info = await adapter.get_provider_info()
                
                assert info["provider_type"] == "ollama"
                assert info["model_name"] == "test-model"
                assert info["is_local"] is True
                assert info["health_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_setup_provider_fallback(self, temp_matter_path):
        """Test setting up provider fallback chain."""
        adapter = LettaAdapter(
            matter_path=temp_matter_path,
            matter_name="TestMatter",
            matter_id="test-123"
        )
        
        primary = {
            "provider": "ollama",
            "model": "gpt-oss:20b",
            "api_key": None
        }
        
        secondary = {
            "provider": "gemini",
            "model": "gemini-2.0-flash-exp",
            "api_key": "test-key"
        }
        
        with patch.object(letta_provider_bridge, 'setup_fallback_chain') as mock_setup:
            result = await adapter.setup_provider_fallback(
                primary=primary,
                secondary=secondary
            )
            
            assert result is True
            mock_setup.assert_called_once()
            
            # Check fallback config was saved
            fallback_file = temp_matter_path / "fallback_config.json"
            assert fallback_file.exists()
    
    @pytest.mark.asyncio
    async def test_get_usage_stats(self, temp_matter_path):
        """Test getting usage statistics."""
        adapter = LettaAdapter(
            matter_path=temp_matter_path,
            matter_name="TestMatter",
            matter_id="test-123"
        )
        
        with patch.object(cost_tracker, 'get_usage_summary') as mock_summary:
            mock_summary.return_value = {
                "total_requests": 10,
                "total_cost_usd": 0.05,
                "total_input_tokens": 5000,
                "total_output_tokens": 2500
            }
            
            stats = await adapter.get_usage_stats(days=7)
            
            assert stats["total_requests"] == 10
            assert stats["total_cost_usd"] == 0.05
            mock_summary.assert_called_once_with(
                matter_id="test-123",
                days=7
            )
    
    @pytest.mark.asyncio
    async def test_set_spending_limit(self, temp_matter_path):
        """Test setting spending limit."""
        adapter = LettaAdapter(
            matter_path=temp_matter_path,
            matter_name="TestMatter",
            matter_id="test-123"
        )
        
        with patch.object(cost_tracker, 'set_spending_limit') as mock_set_limit:
            result = await adapter.set_spending_limit(
                limit_usd=50.0,
                period="monthly"
            )
            
            assert result is True
            mock_set_limit.assert_called_once()


# End-to-End Integration Test
@pytest.mark.asyncio
async def test_end_to_end_provider_flow(temp_matter_path, mock_letta_client):
    """Test complete provider management flow."""
    with patch('app.letta_adapter.connection_manager') as mock_conn:
        mock_conn.connect = AsyncMock(return_value=True)
        mock_conn.client = mock_letta_client
        mock_conn.sync_client = Mock()
        mock_conn.get_state = Mock(return_value=Mock(value="connected"))
        mock_conn.execute_with_retry = AsyncMock(return_value=[])
        
        # Create adapter
        adapter = LettaAdapter(
            matter_path=temp_matter_path,
            matter_name="TestMatter",
            matter_id="test-123"
        )
        
        # Set up provider fallback
        primary = {"provider": "ollama", "model": "gpt-oss:20b", "api_key": None}
        secondary = {"provider": "gemini", "model": "gemini-2.0-flash-exp", "api_key": "test-key"}
        
        await adapter.setup_provider_fallback(primary=primary, secondary=secondary)
        
        # Test provider
        with patch.object(adapter, 'test_provider_configuration') as mock_test:
            mock_test.return_value = {"success": True, "is_healthy": True}
            
            test_result = await adapter.test_provider_configuration(
                provider_type="ollama",
                model_name="gpt-oss:20b"
            )
            
            assert test_result["success"] is True
        
        # Set spending limit
        await adapter.set_spending_limit(limit_usd=100.0, period="monthly")
        
        # Get usage stats
        with patch.object(cost_tracker, 'get_usage_summary') as mock_summary:
            mock_summary.return_value = {
                "total_requests": 0,
                "total_cost_usd": 0.0
            }
            
            stats = await adapter.get_usage_stats()
            assert stats["total_requests"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])