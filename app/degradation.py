"""
Graceful degradation strategies for the Letta Claim Assistant.

Provides fallback mechanisms and reduced functionality modes when services
are unavailable or resources are constrained, ensuring the application
remains usable even under adverse conditions.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from .logging_conf import get_logger
from .error_handler import BaseApplicationError, ServiceUnavailableError, create_context
from .resource_monitor import ResourceStatus, ServiceInfo, resource_monitor

logger = get_logger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""
    FULL = "full"           # All features available
    REDUCED = "reduced"     # Some features disabled
    MINIMAL = "minimal"     # Only core features available
    EMERGENCY = "emergency" # Read-only mode
    OFFLINE = "offline"     # No external dependencies


class ServiceMode(Enum):
    """Service operation modes."""
    NORMAL = "normal"
    FALLBACK = "fallback"
    CACHED = "cached"
    DISABLED = "disabled"


@dataclass
class DegradationRule:
    """Rule for service degradation."""
    service_name: str
    trigger_condition: str  # e.g., "unavailable", "slow", "error_rate"
    degradation_level: DegradationLevel
    fallback_mode: ServiceMode
    message: str
    recovery_check_interval: int = 60  # seconds


@dataclass
class ServiceState:
    """Current state of a service."""
    name: str
    mode: ServiceMode
    degradation_level: DegradationLevel
    last_check: datetime
    error_count: int = 0
    fallback_active: bool = False
    user_notified: bool = False
    recovery_attempts: int = 0


class DegradationManager:
    """
    Manages graceful degradation of services and features.
    """
    
    def __init__(self):
        self.service_states: Dict[str, ServiceState] = {}
        self.degradation_rules: List[DegradationRule] = []
        self.global_degradation_level = DegradationLevel.FULL
        self.notification_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self.setup_default_rules()
    
    def setup_default_rules(self):
        """Setup default degradation rules."""
        
        # Letta service degradation
        self.degradation_rules.append(DegradationRule(
            service_name="letta",
            trigger_condition="unavailable",
            degradation_level=DegradationLevel.REDUCED,
            fallback_mode=ServiceMode.DISABLED,
            message="Agent memory is temporarily unavailable. The system will continue without persistent context."
        ))
        
        # Ollama service degradation
        self.degradation_rules.append(DegradationRule(
            service_name="ollama",
            trigger_condition="unavailable",
            degradation_level=DegradationLevel.REDUCED,
            fallback_mode=ServiceMode.FALLBACK,
            message="Local AI models are unavailable. Switching to external providers if configured."
        ))
        
        # ChromaDB degradation
        self.degradation_rules.append(DegradationRule(
            service_name="chromadb",
            trigger_condition="unavailable",
            degradation_level=DegradationLevel.MINIMAL,
            fallback_mode=ServiceMode.CACHED,
            message="Vector database is unavailable. Search will use cached results only."
        ))
        
        # Network degradation
        self.degradation_rules.append(DegradationRule(
            service_name="network",
            trigger_condition="unavailable",
            degradation_level=DegradationLevel.OFFLINE,
            fallback_mode=ServiceMode.CACHED,
            message="Network connectivity is unavailable. Operating in offline mode with cached data."
        ))
        
        # Disk space degradation
        self.degradation_rules.append(DegradationRule(
            service_name="disk",
            trigger_condition="critical",
            degradation_level=DegradationLevel.EMERGENCY,
            fallback_mode=ServiceMode.DISABLED,
            message="Critical disk space shortage. File operations are disabled to prevent system failure."
        ))
    
    def add_notification_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add callback for degradation notifications."""
        self.notification_callbacks.append(callback)
    
    def _notify_degradation(self, service_name: str, state: ServiceState):
        """Notify about service degradation."""
        for callback in self.notification_callbacks:
            try:
                callback(service_name, {
                    "mode": state.mode.value,
                    "degradation_level": state.degradation_level.value,
                    "fallback_active": state.fallback_active,
                    "error_count": state.error_count
                })
            except Exception as e:
                logger.warning(f"Degradation notification callback failed: {e}")
    
    async def evaluate_service_health(self, service_name: str) -> ServiceState:
        """
        Evaluate health of a specific service and apply degradation if needed.
        
        Args:
            service_name: Name of service to evaluate
            
        Returns:
            Current service state
        """
        current_time = datetime.utcnow()
        
        # Get or create service state
        if service_name not in self.service_states:
            self.service_states[service_name] = ServiceState(
                name=service_name,
                mode=ServiceMode.NORMAL,
                degradation_level=DegradationLevel.FULL,
                last_check=current_time
            )
        
        state = self.service_states[service_name]
        
        # Check service health
        service_healthy = await self._check_service_health(service_name)
        
        if not service_healthy:
            state.error_count += 1
            
            # Find applicable degradation rule
            rule = self._find_degradation_rule(service_name, "unavailable")
            if rule and not state.fallback_active:
                logger.warning(
                    f"Activating degradation for {service_name}",
                    degradation_level=rule.degradation_level.value,
                    fallback_mode=rule.fallback_mode.value
                )
                
                state.mode = rule.fallback_mode
                state.degradation_level = rule.degradation_level
                state.fallback_active = True
                
                self._notify_degradation(service_name, state)
                self._update_global_degradation_level()
        else:
            # Service is healthy, check for recovery
            if state.fallback_active:
                logger.info(f"Service {service_name} recovered, restoring normal operation")
                state.mode = ServiceMode.NORMAL
                state.degradation_level = DegradationLevel.FULL
                state.fallback_active = False
                state.error_count = 0
                state.recovery_attempts += 1
                
                self._notify_degradation(service_name, state)
                self._update_global_degradation_level()
        
        state.last_check = current_time
        return state
    
    async def _check_service_health(self, service_name: str) -> bool:
        """Check if a specific service is healthy."""
        try:
            if service_name == "ollama":
                service_info = await resource_monitor.check_ollama_service()
                return service_info.status == ResourceStatus.HEALTHY
            elif service_name == "chromadb":
                service_info = await resource_monitor.check_chromadb_health()
                return service_info.status == ResourceStatus.HEALTHY
            elif service_name == "network":
                network_info = await resource_monitor.check_network_connectivity()
                return network_info.status == ResourceStatus.HEALTHY
            elif service_name == "disk":
                disk_info = resource_monitor.check_disk_space()
                return disk_info.status != ResourceStatus.CRITICAL
            elif service_name == "letta":
                # Check if Letta is available by trying to import
                try:
                    import letta
                    return True
                except ImportError:
                    return False
            else:
                logger.warning(f"Unknown service for health check: {service_name}")
                return True
        except Exception as e:
            logger.debug(f"Health check failed for {service_name}: {e}")
            return False
    
    def _find_degradation_rule(self, service_name: str, condition: str) -> Optional[DegradationRule]:
        """Find degradation rule for service and condition."""
        for rule in self.degradation_rules:
            if rule.service_name == service_name and rule.trigger_condition == condition:
                return rule
        return None
    
    def _update_global_degradation_level(self):
        """Update global degradation level based on all service states."""
        if not self.service_states:
            self.global_degradation_level = DegradationLevel.FULL
            return
        
        # Find the most severe degradation level
        max_level = DegradationLevel.FULL
        for state in self.service_states.values():
            if state.degradation_level.value == "emergency":
                max_level = DegradationLevel.EMERGENCY
                break
            elif state.degradation_level.value == "offline" and max_level.value != "emergency":
                max_level = DegradationLevel.OFFLINE
            elif state.degradation_level.value == "minimal" and max_level.value not in ["emergency", "offline"]:
                max_level = DegradationLevel.MINIMAL
            elif state.degradation_level.value == "reduced" and max_level.value == "full":
                max_level = DegradationLevel.REDUCED
        
        if max_level != self.global_degradation_level:
            logger.info(
                f"Global degradation level changed",
                old_level=self.global_degradation_level.value,
                new_level=max_level.value
            )
            self.global_degradation_level = max_level
    
    async def evaluate_all_services(self) -> Dict[str, ServiceState]:
        """
        Evaluate health of all known services.
        
        Returns:
            Dictionary of service states
        """
        services = ["ollama", "chromadb", "network", "disk", "letta"]
        
        tasks = [
            self.evaluate_service_health(service)
            for service in services
        ]
        
        states = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for service, state in zip(services, states):
            if isinstance(state, Exception):
                logger.error(f"Failed to evaluate {service}: {state}")
                # Create error state
                result[service] = ServiceState(
                    name=service,
                    mode=ServiceMode.DISABLED,
                    degradation_level=DegradationLevel.EMERGENCY,
                    last_check=datetime.utcnow(),
                    error_count=1,
                    fallback_active=True
                )
            else:
                result[service] = state
        
        return result
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            "global_level": self.global_degradation_level.value,
            "services": {
                name: {
                    "mode": state.mode.value,
                    "degradation_level": state.degradation_level.value,
                    "fallback_active": state.fallback_active,
                    "error_count": state.error_count,
                    "last_check": state.last_check.isoformat()
                }
                for name, state in self.service_states.items()
            }
        }
    
    def is_feature_available(self, feature: str) -> bool:
        """
        Check if a specific feature is available in current degradation state.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available
        """
        # Define feature availability by degradation level
        feature_requirements = {
            "agent_memory": DegradationLevel.FULL,
            "local_llm": DegradationLevel.REDUCED,
            "external_llm": DegradationLevel.MINIMAL,
            "vector_search": DegradationLevel.REDUCED,
            "cached_search": DegradationLevel.MINIMAL,
            "pdf_upload": DegradationLevel.REDUCED,
            "file_operations": DegradationLevel.MINIMAL,
            "settings_save": DegradationLevel.MINIMAL,
            "matter_creation": DegradationLevel.REDUCED
        }
        
        required_level = feature_requirements.get(feature, DegradationLevel.FULL)
        
        # Check if current global level meets requirement
        level_order = [
            DegradationLevel.EMERGENCY,
            DegradationLevel.OFFLINE,
            DegradationLevel.MINIMAL,
            DegradationLevel.REDUCED,
            DegradationLevel.FULL
        ]
        
        current_index = level_order.index(self.global_degradation_level)
        required_index = level_order.index(required_level)
        
        return current_index >= required_index
    
    def get_fallback_provider(self, service_type: str) -> Optional[str]:
        """
        Get fallback provider for a service type.
        
        Args:
            service_type: Type of service (e.g., 'llm', 'embeddings')
            
        Returns:
            Fallback provider name or None
        """
        fallback_providers = {
            "llm": {
                "ollama": "gemini",
                "gemini": "cached_responses"
            },
            "embeddings": {
                "ollama": "cached_embeddings",
                "cached_embeddings": None
            },
            "vector_search": {
                "chromadb": "file_search",
                "file_search": None
            }
        }
        
        # Check current service state and find fallback
        if service_type in fallback_providers:
            # Find the primary service that's failing
            for service_name, state in self.service_states.items():
                if state.fallback_active and service_type in service_name:
                    providers = fallback_providers[service_type]
                    return providers.get(service_name)
        
        return None
    
    async def attempt_service_recovery(self, service_name: str) -> bool:
        """
        Attempt to recover a degraded service.
        
        Args:
            service_name: Service to attempt recovery
            
        Returns:
            True if recovery successful
        """
        if service_name not in self.service_states:
            return False
        
        state = self.service_states[service_name]
        if not state.fallback_active:
            return True  # Service is already healthy
        
        logger.info(f"Attempting recovery for service {service_name}")
        
        # Check if service is now healthy
        if await self._check_service_health(service_name):
            logger.info(f"Service {service_name} recovered successfully")
            state.mode = ServiceMode.NORMAL
            state.degradation_level = DegradationLevel.FULL
            state.fallback_active = False
            state.error_count = 0
            state.recovery_attempts += 1
            
            self._notify_degradation(service_name, state)
            self._update_global_degradation_level()
            return True
        
        return False
    
    def get_user_guidance(self) -> List[Dict[str, str]]:
        """
        Get user guidance messages for current degradation state.
        
        Returns:
            List of guidance messages
        """
        guidance = []
        
        for service_name, state in self.service_states.items():
            if state.fallback_active:
                rule = self._find_degradation_rule(service_name, "unavailable")
                if rule:
                    guidance.append({
                        "service": service_name,
                        "level": state.degradation_level.value,
                        "message": rule.message,
                        "suggestion": self._get_recovery_suggestion(service_name)
                    })
        
        return guidance
    
    def _get_recovery_suggestion(self, service_name: str) -> str:
        """Get recovery suggestion for a service."""
        suggestions = {
            "ollama": "Check if Ollama is running and models are available. Run 'ollama list' to verify.",
            "chromadb": "Verify ChromaDB installation and storage permissions.",
            "network": "Check network connectivity and firewall settings.",
            "disk": "Free up disk space by removing unnecessary files.",
            "letta": "Ensure Letta package is installed and properly configured."
        }
        
        return suggestions.get(service_name, "Check service configuration and try again.")


# Global degradation manager instance
degradation_manager = DegradationManager()


async def check_feature_availability(feature: str) -> bool:
    """
    Check if a feature is available, considering degradation state.
    
    Args:
        feature: Feature name to check
        
    Returns:
        True if feature is available
        
    Raises:
        ServiceUnavailableError: If feature is not available due to degradation
    """
    await degradation_manager.evaluate_all_services()
    
    if not degradation_manager.is_feature_available(feature):
        guidance = degradation_manager.get_user_guidance()
        
        # Find relevant guidance
        relevant_guidance = [g for g in guidance if feature in g.get("message", "")]
        if relevant_guidance:
            message = relevant_guidance[0]["message"]
            suggestion = relevant_guidance[0]["suggestion"]
        else:
            message = f"Feature '{feature}' is temporarily unavailable due to system constraints"
            suggestion = "Please check system status and try again later"
        
        raise ServiceUnavailableError(
            service=feature,
            user_message=message,
            suggestion=suggestion,
            context=create_context(operation="feature_check")
        )
    
    return True


def with_fallback(
    primary_func: Callable[..., Awaitable[Any]],
    fallback_func: Optional[Callable[..., Awaitable[Any]]] = None,
    service_name: str = "unknown"
):
    """
    Decorator for functions with fallback behavior.
    
    Args:
        primary_func: Primary function to execute
        fallback_func: Fallback function if primary fails
        service_name: Service name for degradation tracking
    """
    async def wrapper(*args, **kwargs):
        try:
            # Try primary function
            return await primary_func(*args, **kwargs)
        except Exception as e:
            # Record service failure
            await degradation_manager.evaluate_service_health(service_name)
            
            # Try fallback if available
            if fallback_func:
                logger.warning(
                    f"Primary function failed, using fallback",
                    service=service_name,
                    error=str(e)
                )
                return await fallback_func(*args, **kwargs)
            else:
                # No fallback available
                raise ServiceUnavailableError(
                    service=service_name,
                    cause=e,
                    context=create_context(operation="fallback_check")
                )
    
    return wrapper


class EmergencyMode:
    """
    Emergency mode operations when system is severely degraded.
    """
    
    @staticmethod
    def get_cached_response(query: str) -> Optional[str]:
        """Get cached response for common queries."""
        # Simple cache for emergency mode
        emergency_responses = {
            "help": "The system is operating in emergency mode. Only basic functions are available.",
            "status": "System resources are critically low. Please free up space and restart.",
            "error": "An error occurred. Please check system resources and try again."
        }
        
        # Simple keyword matching
        for keyword, response in emergency_responses.items():
            if keyword.lower() in query.lower():
                return response
        
        return "I'm operating in emergency mode with limited functionality. Please check system status."
    
    @staticmethod
    def get_minimal_features() -> List[str]:
        """Get list of features available in emergency mode."""
        return [
            "view_matters",
            "view_documents", 
            "basic_search",
            "export_data"
        ]