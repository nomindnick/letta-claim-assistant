"""
California Public Works Compliance Validator.

Validates claims against California statutory requirements, deadlines,
and procedural requirements.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re

from ..logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    valid: bool
    category: str
    item: str
    message: str
    severity: str  # "error", "warning", "info"
    statute: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class ClaimValidation:
    """Overall claim validation results."""
    
    claim_type: str
    is_valid: bool
    errors: List[ValidationResult] = field(default_factory=list)
    warnings: List[ValidationResult] = field(default_factory=list)
    info: List[ValidationResult] = field(default_factory=list)
    missing_items: List[str] = field(default_factory=list)
    deadline_risks: List[Dict[str, Any]] = field(default_factory=list)
    compliance_score: float = 0.0


class CaliforniaComplianceValidator:
    """Validates California public works claims for compliance."""
    
    def __init__(self):
        """Initialize the compliance validator."""
        self.claim_requirements = self._init_claim_requirements()
        self.statutory_deadlines = self._init_statutory_deadlines()
        self.notice_requirements = self._init_notice_requirements()
    
    def _init_claim_requirements(self) -> Dict[str, List[str]]:
        """Initialize requirements for each claim type."""
        return {
            "differing_site_conditions": [
                "Prompt written notice to owner",
                "Evidence of reliance on contract documents",
                "Documentation of actual conditions",
                "Cost and time impact analysis",
                "Compliance with contract notice provisions"
            ],
            "delay": [
                "Timely notice per contract",
                "Critical path impact demonstration",
                "Time impact analysis",
                "Proof of owner causation",
                "Actual damages documentation",
                "Compliance with scheduling specifications"
            ],
            "changes": [
                "Written change order or directive",
                "Notice of claim if no written order",
                "Cost proposal with backup",
                "Time extension request if applicable",
                "Compliance with changes clause"
            ],
            "payment": [
                "Proper payment application",
                "Lien release if required",
                "Certified payroll for public works",
                "Preliminary notice (if subcontractor)",
                "Compliance with prompt payment statutes"
            ],
            "termination": [
                "Termination notice",
                "Inventory of materials",
                "Demobilization costs",
                "Subcontractor settlements",
                "Final accounting"
            ]
        }
    
    def _init_statutory_deadlines(self) -> Dict[str, Dict[str, Any]]:
        """Initialize statutory deadline requirements."""
        return {
            "preliminary_notice": {
                "statute": "Civil Code §8200",
                "days": 20,
                "from": "first furnishing",
                "consequence": "Loss of lien/stop notice rights",
                "applies_to": ["subcontractor", "supplier"]
            },
            "stop_notice": {
                "statute": "Civil Code §8502",
                "days": 90,
                "from": "completion",
                "consequence": "Loss of stop notice rights",
                "applies_to": ["subcontractor", "supplier"]
            },
            "mechanics_lien": {
                "statute": "Civil Code §8412",
                "days": 90,
                "from": "completion",
                "consequence": "Loss of lien rights",
                "applies_to": ["contractor", "subcontractor"]
            },
            "government_claim": {
                "statute": "Gov. Code §911.2",
                "days": 365,
                "from": "accrual",
                "consequence": "Barred from suit",
                "applies_to": ["all"]
            },
            "bond_claim_notice": {
                "statute": "Civil Code §9350",
                "days": 90,
                "from": "completion",
                "consequence": "Loss of bond rights",
                "applies_to": ["subcontractor", "supplier"]
            },
            "bond_suit": {
                "statute": "Code Civ. Proc. §337",
                "days": 730,  # 2 years
                "from": "completion",
                "consequence": "Time-barred",
                "applies_to": ["subcontractor", "supplier"]
            }
        }
    
    def _init_notice_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize notice requirements."""
        return {
            "differing_site_conditions": {
                "timing": "promptly upon discovery",
                "form": "written",
                "content": ["location", "nature of condition", "impact"],
                "preserve": "condition for inspection"
            },
            "delay": {
                "timing": "per contract (typically 10-20 days)",
                "form": "written",
                "content": ["cause", "impact", "mitigation efforts"],
                "preserve": "schedule records"
            },
            "change": {
                "timing": "before performing work or per contract",
                "form": "written",
                "content": ["scope change", "cost impact", "time impact"],
                "preserve": "directive documentation"
            }
        }
    
    def validate_claim(
        self,
        claim_type: str,
        claim_data: Dict[str, Any],
        extracted_entities: Optional[Dict[str, Any]] = None
    ) -> ClaimValidation:
        """
        Validate a California construction claim.
        
        Args:
            claim_type: Type of claim
            claim_data: Claim information
            extracted_entities: Extracted California entities
            
        Returns:
            ClaimValidation object with results
        """
        validation = ClaimValidation(claim_type=claim_type, is_valid=True)
        
        # Validate basic requirements
        self._validate_basic_requirements(claim_type, claim_data, validation)
        
        # Validate deadlines
        self._validate_deadlines(claim_data, validation)
        
        # Validate notices
        self._validate_notices(claim_type, claim_data, validation)
        
        # Validate documentation
        self._validate_documentation(claim_type, claim_data, validation)
        
        # Validate statutory compliance
        self._validate_statutory_compliance(claim_data, extracted_entities, validation)
        
        # Calculate compliance score
        validation.compliance_score = self._calculate_compliance_score(validation)
        
        # Determine overall validity
        validation.is_valid = len(validation.errors) == 0
        
        return validation
    
    def _validate_basic_requirements(
        self,
        claim_type: str,
        claim_data: Dict[str, Any],
        validation: ClaimValidation
    ) -> None:
        """Validate basic claim requirements."""
        requirements = self.claim_requirements.get(claim_type, [])
        
        for requirement in requirements:
            # Check if requirement is mentioned or satisfied
            requirement_key = requirement.lower().replace(" ", "_")
            if not claim_data.get(requirement_key, False):
                validation.missing_items.append(requirement)
                validation.warnings.append(
                    ValidationResult(
                        valid=False,
                        category="Requirements",
                        item=requirement,
                        message=f"Missing or unclear: {requirement}",
                        severity="warning",
                        recommendation=f"Verify and document {requirement}"
                    )
                )
    
    def _validate_deadlines(
        self,
        claim_data: Dict[str, Any],
        validation: ClaimValidation
    ) -> None:
        """Validate deadline compliance."""
        current_date = datetime.now()
        
        for deadline_type, deadline_info in self.statutory_deadlines.items():
            # Check if this deadline applies
            trigger_date_key = f"{deadline_type}_trigger_date"
            if trigger_date_key in claim_data:
                trigger_date = claim_data[trigger_date_key]
                if isinstance(trigger_date, str):
                    try:
                        trigger_date = datetime.fromisoformat(trigger_date)
                    except:
                        continue
                
                deadline_date = trigger_date + timedelta(days=deadline_info["days"])
                days_remaining = (deadline_date - current_date).days
                
                if days_remaining < 0:
                    validation.errors.append(
                        ValidationResult(
                            valid=False,
                            category="Deadlines",
                            item=deadline_type,
                            message=f"Deadline expired {abs(days_remaining)} days ago",
                            severity="error",
                            statute=deadline_info["statute"],
                            recommendation=f"Check if {deadline_info['consequence']}"
                        )
                    )
                elif days_remaining < 30:
                    validation.warnings.append(
                        ValidationResult(
                            valid=False,
                            category="Deadlines",
                            item=deadline_type,
                            message=f"Deadline in {days_remaining} days",
                            severity="warning",
                            statute=deadline_info["statute"],
                            recommendation=f"Act promptly to avoid {deadline_info['consequence']}"
                        )
                    )
                    validation.deadline_risks.append({
                        "type": deadline_type,
                        "days_remaining": days_remaining,
                        "deadline_date": deadline_date.isoformat(),
                        "consequence": deadline_info["consequence"]
                    })
    
    def _validate_notices(
        self,
        claim_type: str,
        claim_data: Dict[str, Any],
        validation: ClaimValidation
    ) -> None:
        """Validate notice requirements."""
        notice_req = self.notice_requirements.get(claim_type)
        
        if notice_req:
            # Check notice timing
            if "notice_date" in claim_data and "discovery_date" in claim_data:
                notice_date = claim_data["notice_date"]
                discovery_date = claim_data["discovery_date"]
                
                if isinstance(notice_date, str):
                    notice_date = datetime.fromisoformat(notice_date)
                if isinstance(discovery_date, str):
                    discovery_date = datetime.fromisoformat(discovery_date)
                
                days_to_notice = (notice_date - discovery_date).days
                
                if "prompt" in notice_req["timing"] and days_to_notice > 7:
                    validation.warnings.append(
                        ValidationResult(
                            valid=False,
                            category="Notice",
                            item="Timing",
                            message=f"Notice given {days_to_notice} days after discovery may not be 'prompt'",
                            severity="warning",
                            recommendation="Document reasons for any delay in notice"
                        )
                    )
            
            # Check notice form
            if claim_data.get("notice_form") != notice_req["form"]:
                validation.warnings.append(
                    ValidationResult(
                        valid=False,
                        category="Notice",
                        item="Form",
                        message=f"Notice should be {notice_req['form']}",
                        severity="warning",
                        recommendation=f"Ensure notice is properly documented in {notice_req['form']} form"
                    )
                )
            
            # Check notice content
            for content_item in notice_req.get("content", []):
                if not claim_data.get(f"notice_includes_{content_item}", False):
                    validation.info.append(
                        ValidationResult(
                            valid=True,
                            category="Notice",
                            item="Content",
                            message=f"Verify notice includes {content_item}",
                            severity="info",
                            recommendation=f"Notice should clearly state {content_item}"
                        )
                    )
    
    def _validate_documentation(
        self,
        claim_type: str,
        claim_data: Dict[str, Any],
        validation: ClaimValidation
    ) -> None:
        """Validate documentation requirements."""
        required_docs = {
            "differing_site_conditions": [
                "photos", "daily_reports", "notice_letters", 
                "geotechnical_reports", "cost_records"
            ],
            "delay": [
                "baseline_schedule", "updated_schedules", "daily_reports",
                "correspondence", "time_impact_analysis"
            ],
            "changes": [
                "change_directives", "rfis", "cost_proposals",
                "correspondence", "approval_documents"
            ],
            "payment": [
                "payment_applications", "certified_payroll",
                "lien_releases", "preliminary_notices"
            ]
        }
        
        docs_needed = required_docs.get(claim_type, [])
        for doc in docs_needed:
            if not claim_data.get(f"has_{doc}", False):
                validation.missing_items.append(f"Documentation: {doc}")
                validation.info.append(
                    ValidationResult(
                        valid=True,
                        category="Documentation",
                        item=doc,
                        message=f"Verify {doc} is available",
                        severity="info",
                        recommendation=f"Gather and organize {doc}"
                    )
                )
    
    def _validate_statutory_compliance(
        self,
        claim_data: Dict[str, Any],
        extracted_entities: Optional[Dict[str, Any]],
        validation: ClaimValidation
    ) -> None:
        """Validate statutory compliance."""
        # Check prevailing wage compliance for public works
        if claim_data.get("is_public_works", True):
            if not claim_data.get("prevailing_wage_compliance"):
                validation.warnings.append(
                    ValidationResult(
                        valid=False,
                        category="Statutory",
                        item="Prevailing Wages",
                        message="Prevailing wage compliance not confirmed",
                        severity="warning",
                        statute="Labor Code §1770 et seq.",
                        recommendation="Ensure all certified payroll records are complete"
                    )
                )
        
        # Check government claim filing
        if claim_data.get("public_entity") and not claim_data.get("government_claim_filed"):
            validation.errors.append(
                ValidationResult(
                    valid=False,
                    category="Statutory",
                    item="Government Claim",
                    message="Government claim may be required",
                    severity="error",
                    statute="Gov. Code §910-911",
                    recommendation="File government claim before statute expires"
                )
            )
        
        # Check for stop notice rights
        if claim_data.get("role") in ["subcontractor", "supplier"]:
            if not claim_data.get("preliminary_notice_served"):
                validation.warnings.append(
                    ValidationResult(
                        valid=False,
                        category="Statutory",
                        item="Preliminary Notice",
                        message="Preliminary notice may affect payment rights",
                        severity="warning",
                        statute="Civil Code §8200",
                        recommendation="Verify preliminary notice was timely served"
                    )
                )
    
    def _calculate_compliance_score(self, validation: ClaimValidation) -> float:
        """Calculate overall compliance score."""
        score = 100.0
        
        # Deduct for errors (20 points each)
        score -= len(validation.errors) * 20
        
        # Deduct for warnings (10 points each)
        score -= len(validation.warnings) * 10
        
        # Deduct for missing items (5 points each)
        score -= len(validation.missing_items) * 5
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        return score / 100.0
    
    def validate_deadline(
        self,
        deadline_type: str,
        trigger_date: datetime,
        action_taken_date: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate a specific deadline.
        
        Args:
            deadline_type: Type of deadline
            trigger_date: Date triggering the deadline
            action_taken_date: Date action was taken (if any)
            
        Returns:
            ValidationResult
        """
        deadline_info = self.statutory_deadlines.get(deadline_type)
        
        if not deadline_info:
            return ValidationResult(
                valid=False,
                category="Deadlines",
                item=deadline_type,
                message=f"Unknown deadline type: {deadline_type}",
                severity="error"
            )
        
        deadline_date = trigger_date + timedelta(days=deadline_info["days"])
        
        if action_taken_date:
            if action_taken_date > deadline_date:
                days_late = (action_taken_date - deadline_date).days
                return ValidationResult(
                    valid=False,
                    category="Deadlines",
                    item=deadline_type,
                    message=f"Action taken {days_late} days after deadline",
                    severity="error",
                    statute=deadline_info["statute"],
                    recommendation=f"May have resulted in {deadline_info['consequence']}"
                )
            else:
                days_early = (deadline_date - action_taken_date).days
                return ValidationResult(
                    valid=True,
                    category="Deadlines",
                    item=deadline_type,
                    message=f"Action taken {days_early} days before deadline",
                    severity="info",
                    statute=deadline_info["statute"]
                )
        else:
            days_remaining = (deadline_date - datetime.now()).days
            
            if days_remaining < 0:
                return ValidationResult(
                    valid=False,
                    category="Deadlines",
                    item=deadline_type,
                    message=f"Deadline expired {abs(days_remaining)} days ago",
                    severity="error",
                    statute=deadline_info["statute"],
                    recommendation=f"May have resulted in {deadline_info['consequence']}"
                )
            elif days_remaining < 30:
                return ValidationResult(
                    valid=False,
                    category="Deadlines",
                    item=deadline_type,
                    message=f"Deadline in {days_remaining} days",
                    severity="warning",
                    statute=deadline_info["statute"],
                    recommendation=f"Act promptly to avoid {deadline_info['consequence']}"
                )
            else:
                return ValidationResult(
                    valid=True,
                    category="Deadlines",
                    item=deadline_type,
                    message=f"{days_remaining} days until deadline",
                    severity="info",
                    statute=deadline_info["statute"]
                )
    
    def get_claim_checklist(self, claim_type: str) -> List[str]:
        """
        Get a checklist for a specific claim type.
        
        Args:
            claim_type: Type of claim
            
        Returns:
            List of checklist items
        """
        base_checklist = [
            "Identify all applicable deadlines",
            "Review contract notice requirements",
            "Gather supporting documentation",
            "Calculate damages with backup",
            "Consider need for expert analysis"
        ]
        
        specific_items = self.claim_requirements.get(claim_type, [])
        
        return base_checklist + specific_items


# Singleton instance  
california_validator = CaliforniaComplianceValidator()