"""
Privacy consent management for external LLM providers.

Manages user consent for sending data to external services, tracks consent
status, and provides mechanisms for consent revocation and audit logging.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path
import json
from enum import Enum

from .logging_conf import get_logger
from .settings import settings

logger = get_logger(__name__)


class ConsentType(Enum):
    """Types of consent that can be requested."""
    EXTERNAL_LLM = "external_llm"
    DATA_SHARING = "data_sharing"
    ANALYTICS = "analytics"


class ConsentStatus(Enum):
    """Possible consent statuses."""
    NOT_REQUESTED = "not_requested"
    GRANTED = "granted" 
    DENIED = "denied"
    REVOKED = "revoked"


class ConsentRecord:
    """Individual consent record with metadata."""
    
    def __init__(
        self,
        consent_type: ConsentType,
        provider: str,
        status: ConsentStatus,
        granted_at: Optional[datetime] = None,
        revoked_at: Optional[datetime] = None,
        version: str = "1.0"
    ):
        self.consent_type = consent_type
        self.provider = provider
        self.status = status
        self.granted_at = granted_at
        self.revoked_at = revoked_at
        self.version = version
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "consent_type": self.consent_type.value,
            "provider": self.provider,
            "status": self.status.value,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "version": self.version,
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsentRecord':
        """Create from dictionary."""
        return cls(
            consent_type=ConsentType(data["consent_type"]),
            provider=data["provider"],
            status=ConsentStatus(data["status"]),
            granted_at=datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None,
            revoked_at=datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None,
            version=data.get("version", "1.0")
        )


class PrivacyConsentManager:
    """Manages privacy consent for external services."""
    
    def __init__(self):
        self.consent_file = Path.home() / ".letta-claim" / "consent.json"
        self.consent_file.parent.mkdir(parents=True, exist_ok=True)
        self._consents: Dict[str, ConsentRecord] = {}
        self._load_consents()
    
    def _consent_key(self, consent_type: ConsentType, provider: str) -> str:
        """Generate unique key for consent record."""
        return f"{consent_type.value}:{provider}"
    
    def _load_consents(self) -> None:
        """Load consent records from disk."""
        if self.consent_file.exists():
            try:
                with open(self.consent_file, 'r') as f:
                    data = json.load(f)
                
                for key, consent_data in data.items():
                    self._consents[key] = ConsentRecord.from_dict(consent_data)
                
                logger.debug("Loaded consent records", count=len(self._consents))
                
            except Exception as e:
                logger.error("Failed to load consent records", error=str(e))
                self._consents = {}
    
    def _save_consents(self) -> None:
        """Save consent records to disk."""
        try:
            consent_data = {
                key: record.to_dict() 
                for key, record in self._consents.items()
            }
            
            with open(self.consent_file, 'w') as f:
                json.dump(consent_data, f, indent=2)
            
            logger.debug("Saved consent records", count=len(self._consents))
            
        except Exception as e:
            logger.error("Failed to save consent records", error=str(e))
    
    def get_consent_status(
        self, 
        consent_type: ConsentType, 
        provider: str
    ) -> ConsentStatus:
        """Get current consent status for a provider."""
        key = self._consent_key(consent_type, provider)
        
        if key in self._consents:
            return self._consents[key].status
        else:
            return ConsentStatus.NOT_REQUESTED
    
    def is_consent_granted(
        self, 
        consent_type: ConsentType, 
        provider: str
    ) -> bool:
        """Check if consent is currently granted."""
        return self.get_consent_status(consent_type, provider) == ConsentStatus.GRANTED
    
    def grant_consent(
        self,
        consent_type: ConsentType,
        provider: str,
        version: str = "1.0"
    ) -> bool:
        """Grant consent for a provider."""
        try:
            key = self._consent_key(consent_type, provider)
            now = datetime.now(timezone.utc)
            
            self._consents[key] = ConsentRecord(
                consent_type=consent_type,
                provider=provider,
                status=ConsentStatus.GRANTED,
                granted_at=now,
                version=version
            )
            
            self._save_consents()
            
            logger.info(
                "Consent granted",
                consent_type=consent_type.value,
                provider=provider,
                version=version
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to grant consent",
                consent_type=consent_type.value,
                provider=provider,
                error=str(e)
            )
            return False
    
    def revoke_consent(
        self,
        consent_type: ConsentType,
        provider: str
    ) -> bool:
        """Revoke previously granted consent."""
        try:
            key = self._consent_key(consent_type, provider)
            now = datetime.now(timezone.utc)
            
            if key in self._consents:
                record = self._consents[key]
                record.status = ConsentStatus.REVOKED
                record.revoked_at = now
                record.updated_at = now
            else:
                # Create new record with revoked status
                self._consents[key] = ConsentRecord(
                    consent_type=consent_type,
                    provider=provider,
                    status=ConsentStatus.REVOKED,
                    revoked_at=now
                )
            
            self._save_consents()
            
            logger.info(
                "Consent revoked",
                consent_type=consent_type.value,
                provider=provider
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to revoke consent",
                consent_type=consent_type.value,
                provider=provider,
                error=str(e)
            )
            return False
    
    def deny_consent(
        self,
        consent_type: ConsentType,
        provider: str
    ) -> bool:
        """Explicitly deny consent for a provider."""
        try:
            key = self._consent_key(consent_type, provider)
            now = datetime.now(timezone.utc)
            
            self._consents[key] = ConsentRecord(
                consent_type=consent_type,
                provider=provider,
                status=ConsentStatus.DENIED,
            )
            
            self._save_consents()
            
            logger.info(
                "Consent denied",
                consent_type=consent_type.value,
                provider=provider
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to deny consent",
                consent_type=consent_type.value,
                provider=provider,
                error=str(e)
            )
            return False
    
    def get_all_consents(self) -> List[Dict[str, Any]]:
        """Get all consent records."""
        return [
            {
                "key": key,
                **record.to_dict()
            }
            for key, record in self._consents.items()
        ]
    
    def get_provider_consents(self, provider: str) -> List[Dict[str, Any]]:
        """Get all consent records for a specific provider."""
        return [
            {
                "key": key,
                **record.to_dict()
            }
            for key, record in self._consents.items()
            if record.provider == provider
        ]
    
    def clear_provider_consents(self, provider: str) -> bool:
        """Clear all consent records for a provider (for cleanup)."""
        try:
            keys_to_remove = [
                key for key, record in self._consents.items()
                if record.provider == provider
            ]
            
            for key in keys_to_remove:
                del self._consents[key]
            
            self._save_consents()
            
            logger.info("Cleared provider consents", provider=provider, count=len(keys_to_remove))
            return True
            
        except Exception as e:
            logger.error("Failed to clear provider consents", provider=provider, error=str(e))
            return False
    
    def ensure_external_llm_consent(self, provider: str) -> bool:
        """
        Check if consent is required and granted for external LLM usage.
        
        Returns True if consent is not required or is granted.
        Returns False if consent is required but not granted.
        """
        # Local providers don't require consent
        if provider.lower() in ["ollama", "local"]:
            return True
        
        # External providers require consent
        return self.is_consent_granted(ConsentType.EXTERNAL_LLM, provider)
    
    def get_consent_requirements(self, provider: str) -> Dict[str, Any]:
        """Get consent requirements and status for a provider."""
        is_external = provider.lower() not in ["ollama", "local"]
        
        if not is_external:
            return {
                "provider": provider,
                "requires_consent": False,
                "consent_granted": True,
                "consent_status": "not_required"
            }
        
        consent_status = self.get_consent_status(ConsentType.EXTERNAL_LLM, provider)
        
        return {
            "provider": provider,
            "requires_consent": True,
            "consent_granted": consent_status == ConsentStatus.GRANTED,
            "consent_status": consent_status.value,
            "data_usage_notice": self._get_data_usage_notice(provider)
        }
    
    def _get_data_usage_notice(self, provider: str) -> str:
        """Get data usage notice for a provider."""
        if provider.lower() == "gemini":
            return (
                "When using Google Gemini, your questions and document context "
                "will be sent to Google's servers for processing. This data may "
                "be used by Google to improve their services according to their "
                "privacy policy. Your original documents remain on your local machine."
            )
        else:
            return (
                f"When using {provider}, your questions and document context "
                "will be sent to external servers for processing. Please review "
                "the provider's privacy policy before proceeding."
            )


# Global consent manager instance
consent_manager = PrivacyConsentManager()