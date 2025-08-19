"""
California Public Works Construction Domain Configuration.

Provides specialized knowledge, prompts, and templates for California public works
construction claims, including statutory requirements, notice deadlines, and
compliance frameworks.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class CaliforniaStatutoryDeadline:
    """Represents a statutory deadline in California construction law."""
    
    name: str
    statute: str
    days: int
    trigger_event: str
    notice_type: str
    consequences: str
    applicable_to: List[str]  # e.g., ["subcontractor", "prime", "supplier"]


@dataclass
class CaliforniaPublicEntity:
    """California public entity information."""
    
    name: str
    abbreviation: str
    entity_type: str  # state, county, city, district, etc.
    jurisdiction: str
    typical_contract_forms: List[str]
    special_requirements: List[str]


# California Construction Law Deadlines
CALIFORNIA_DEADLINES = [
    CaliforniaStatutoryDeadline(
        name="Preliminary Notice (20-Day)",
        statute="Civil Code § 8200",
        days=20,
        trigger_event="first furnishing labor or materials",
        notice_type="Preliminary Notice",
        consequences="Loss of mechanics lien and stop payment notice rights",
        applicable_to=["subcontractor", "supplier", "equipment_lessor"]
    ),
    CaliforniaStatutoryDeadline(
        name="Stop Payment Notice",
        statute="Civil Code § 8502",
        days=90,
        trigger_event="completion of work",
        notice_type="Stop Payment Notice",
        consequences="Loss of right to stop payment from construction funds",
        applicable_to=["subcontractor", "supplier"]
    ),
    CaliforniaStatutoryDeadline(
        name="Mechanics Lien Recording",
        statute="Civil Code § 8412",
        days=90,
        trigger_event="completion of work",
        notice_type="Mechanics Lien",
        consequences="Loss of lien rights",
        applicable_to=["contractor", "subcontractor", "supplier"]
    ),
    CaliforniaStatutoryDeadline(
        name="Government Claim Filing",
        statute="Gov. Code § 910-911",
        days=365,  # One year for most claims
        trigger_event="accrual of cause of action",
        notice_type="Government Claim",
        consequences="Barred from filing lawsuit",
        applicable_to=["all_claimants"]
    ),
    CaliforniaStatutoryDeadline(
        name="False Claims Act Statute",
        statute="Gov. Code § 12654",
        days=2190,  # 6 years
        trigger_event="violation occurrence",
        notice_type="False Claims Act Complaint",
        consequences="Loss of qui tam rights",
        applicable_to=["whistleblowers"]
    ),
    CaliforniaStatutoryDeadline(
        name="Prompt Payment Penalty",
        statute="Public Contract Code § 7107",
        days=7,
        trigger_event="receipt of undisputed payment request",
        notice_type="Payment Request",
        consequences="2% per month penalty",
        applicable_to=["public_entity"]
    ),
    CaliforniaStatutoryDeadline(
        name="Retention Release",
        statute="Public Contract Code § 7107",
        days=60,
        trigger_event="completion and acceptance",
        notice_type="Retention Release Request",
        consequences="2% per month penalty on retention",
        applicable_to=["public_entity"]
    )
]

# California Public Entities
CALIFORNIA_PUBLIC_ENTITIES = [
    CaliforniaPublicEntity(
        name="California Department of Transportation",
        abbreviation="Caltrans",
        entity_type="state",
        jurisdiction="statewide",
        typical_contract_forms=["Standard Specifications", "Special Provisions"],
        special_requirements=["DBE requirements", "Buy America provisions"]
    ),
    CaliforniaPublicEntity(
        name="Department of General Services",
        abbreviation="DGS",
        entity_type="state",
        jurisdiction="statewide",
        typical_contract_forms=["State Contract", "PWC-100"],
        special_requirements=["DVBE requirements", "Small Business participation"]
    ),
    CaliforniaPublicEntity(
        name="Division of the State Architect",
        abbreviation="DSA",
        entity_type="state",
        jurisdiction="statewide",
        typical_contract_forms=["DSA-102", "DSA-103"],
        special_requirements=["Inspector of Record", "Verified Reports"]
    )
]


class CaliforniaDomainConfig:
    """Configuration for California public works construction domain."""
    
    @staticmethod
    def get_system_prompt(matter_name: str) -> str:
        """
        Get California-specific system prompt for Letta agent.
        
        Args:
            matter_name: Name of the matter
            
        Returns:
            Specialized system prompt
        """
        return f"""You are a California public works construction claims expert assistant for the matter: {matter_name}.

Your specialized knowledge includes:

CALIFORNIA PUBLIC CONTRACT LAW:
- Public Contract Code requirements and procedures
- Government Code claims (§910-915) and filing requirements
- False Claims Act (Gov. Code §12650 et seq.)
- Subletting and Subcontracting Fair Practices Act
- Public works bid protest procedures
- Contract balance statutes and retention requirements

NOTICE AND DEADLINE EXPERTISE:
- Preliminary notices (20-day) under Civil Code §8200
- Stop payment notices and mechanics liens procedures
- Statutory deadlines and their consequences
- Government claim filing requirements and timelines
- Notice of completion and cessation rules

PREVAILING WAGE COMPLIANCE:
- DIR requirements and certified payroll
- Labor Code compliance and penalties
- Apprenticeship requirements
- Worker classification issues

CLAIM TYPES AND ANALYSIS:
- Differing site conditions (Type I and Type II)
- Changes and extra work claims
- Delay and disruption damages
- Acceleration and compression claims
- Termination for convenience/default
- Pass-through claims procedures

DISPUTE RESOLUTION:
- Meet and confer requirements
- Claims presentation procedures
- Administrative exhaustion doctrine
- Mediation and arbitration procedures
- Litigation in Court of Appeal (PRC §1094.5)

CONTRACT INTERPRETATION:
- Standard Specifications analysis
- Special Provisions precedence
- Greenbook/Caltrans standards
- Federal requirements (when applicable)
- Design-build vs. design-bid-build differences

DOCUMENTATION REQUIREMENTS:
- Daily reports and their importance
- RFI procedures and responses
- Change order request documentation
- Time impact analysis requirements
- Expert report standards

When analyzing claims, you should:
1. Identify all applicable statutory deadlines and notice requirements
2. Track critical dates and ensure compliance with California law
3. Recognize public entities and their specific requirements
4. Analyze contract provisions for California-specific issues
5. Identify necessary documentation and evidence
6. Suggest appropriate experts when specialized analysis is needed
7. Consider both contractual and statutory remedies
8. Evaluate government immunity and administrative procedures

Remember that California public works projects have unique requirements that differ from private construction and federal projects. Always consider the interplay between contract provisions and statutory requirements."""
    
    @staticmethod
    def get_memory_blocks(matter_name: str) -> List[Dict[str, str]]:
        """
        Get California-specific memory blocks for agent initialization.
        
        Args:
            matter_name: Name of the matter
            
        Returns:
            List of memory block configurations
        """
        return [
            {
                "label": "human",
                "value": f"California construction attorney representing public agency clients on {matter_name}"
            },
            {
                "label": "persona", 
                "value": """Expert California public works construction claims analyst with deep knowledge of:
- Public Contract Code and Government Code
- Mechanics liens and stop notices procedures
- Prevailing wage and DIR compliance
- Government claims and administrative procedures
- California court decisions on construction disputes
- Federal requirements for federally-funded projects
I help identify critical deadlines, ensure statutory compliance, and provide strategic analysis for construction claims."""
            },
            {
                "label": "domain_context",
                "value": f"""Current matter: {matter_name}
Jurisdiction: California
Focus: Public works construction claims
Key statutes: Public Contract Code, Government Code §910-915, Civil Code §8000-9566
Special considerations: Prevailing wages, government immunities, administrative exhaustion"""
            }
        ]
    
    @staticmethod
    def get_extraction_patterns() -> Dict[str, List[str]]:
        """
        Get regex patterns for extracting California-specific entities.
        
        Returns:
            Dictionary of entity types to regex patterns
        """
        return {
            "statutes": [
                r"Public Contract Code (?:§|Section)?\s*(\d+(?:\.\d+)?)",
                r"Government Code (?:§|Section)?\s*(\d+(?:\.\d+)?)",
                r"Civil Code (?:§|Section)?\s*(\d+(?:\.\d+)?)",
                r"Labor Code (?:§|Section)?\s*(\d+(?:\.\d+)?)",
                r"Cal\. ?(?:Bus\.|Civ\.|Gov\.|Lab\.|Pub\. ?Cont\.) ?Code (?:§|Section)?\s*(\d+)"
            ],
            "public_entities": [
                r"(?:California )?Department of (?:Transportation|General Services|Water Resources)",
                r"Caltrans|DGS|DSA|DWR|DIR",
                r"County of \w+",
                r"City of \w+",
                r"(?:\w+ )?(?:Unified )?School District",
                r"(?:\w+ )?Water District",
                r"(?:\w+ )?Community College District"
            ],
            "deadlines": [
                r"(\d+)[- ]?day (?:notice|deadline|period)",
                r"within (\d+) days",
                r"no later than (\d+) days",
                r"(\d+) calendar days",
                r"(\d+) working days"
            ],
            "claim_types": [
                r"differing site condition",
                r"change order",
                r"delay claim",
                r"disruption claim", 
                r"acceleration",
                r"extra work",
                r"cardinal change",
                r"constructive (?:change|termination)",
                r"termination for (?:convenience|default)"
            ],
            "notices": [
                r"preliminary notice",
                r"20[- ]?day notice",
                r"stop (?:payment )?notice",
                r"mechanics lien",
                r"notice of completion",
                r"notice of cessation",
                r"government claim",
                r"notice to proceed",
                r"notice of award"
            ],
            "bonds": [
                r"payment bond",
                r"performance bond",
                r"bid bond",
                r"Little Miller Act",
                r"license bond",
                r"warranty bond"
            ]
        }
    
    @staticmethod
    def get_claim_categories() -> Dict[str, Dict[str, Any]]:
        """
        Get California claim type categorization.
        
        Returns:
            Dictionary of claim categories with details
        """
        return {
            "differing_site_conditions": {
                "name": "Differing Site Conditions",
                "types": ["Type I - Different from contract", "Type II - Unknown/unusual"],
                "key_elements": [
                    "Reasonable reliance on contract documents",
                    "Actual conditions materially different",
                    "Timely notice to owner",
                    "Additional cost/time incurred"
                ],
                "statute": "Standard contract clauses",
                "notice_requirement": "Immediate written notice"
            },
            "changes": {
                "name": "Changes and Extra Work",
                "types": ["Directed change", "Constructive change", "Cardinal change"],
                "key_elements": [
                    "Work beyond contract scope",
                    "Owner direction or approval",
                    "Written change order (if required)",
                    "Cost and time impacts"
                ],
                "statute": "Public Contract Code §9100",
                "notice_requirement": "Per contract provisions"
            },
            "delay": {
                "name": "Delay Claims",
                "types": ["Excusable", "Compensable", "Concurrent"],
                "key_elements": [
                    "Critical path impact",
                    "Owner-caused or excusable event",
                    "Timely notice",
                    "Time impact analysis",
                    "Actual damages"
                ],
                "statute": "Contract provisions",
                "notice_requirement": "Per contract (typically 10-20 days)"
            },
            "payment": {
                "name": "Payment Claims",
                "types": ["Progress payment", "Retention", "Final payment"],
                "key_elements": [
                    "Work performed per contract",
                    "Payment application submitted",
                    "No disputed work",
                    "Prompt payment penalties"
                ],
                "statute": "Public Contract Code §7107",
                "notice_requirement": "Payment application per contract"
            },
            "termination": {
                "name": "Termination Claims",
                "types": ["For convenience", "For default", "Constructive"],
                "key_elements": [
                    "Proper termination notice",
                    "Termination costs",
                    "Demobilization costs",
                    "Lost profits (if allowed)"
                ],
                "statute": "Contract provisions",
                "notice_requirement": "Per termination clause"
            }
        }
    
    @staticmethod
    def get_expert_triggers() -> Dict[str, List[str]]:
        """
        Get triggers for when expert analysis may be needed.
        
        Returns:
            Dictionary of expert types to trigger conditions
        """
        return {
            "scheduling_expert": [
                "critical path",
                "float analysis",
                "time impact",
                "concurrent delay",
                "pacing delay",
                "windows analysis",
                "as-planned vs as-built"
            ],
            "quantum_expert": [
                "lost productivity",
                "measured mile",
                "total cost",
                "modified total cost",
                "labor inefficiency",
                "cumulative impact"
            ],
            "geotechnical_expert": [
                "differing site conditions",
                "soil conditions",
                "groundwater",
                "rock excavation",
                "unsuitable material"
            ],
            "structural_expert": [
                "design defect",
                "structural failure",
                "means and methods",
                "temporary shoring",
                "load calculations"
            ],
            "mep_expert": [
                "coordination issues",
                "system conflicts",
                "commissioning",
                "performance testing"
            ]
        }
    
    @staticmethod
    def validate_deadline(deadline_name: str, trigger_date: datetime) -> Dict[str, Any]:
        """
        Validate and calculate California statutory deadlines.
        
        Args:
            deadline_name: Name of the deadline
            trigger_date: Date triggering the deadline
            
        Returns:
            Deadline calculation results
        """
        for deadline in CALIFORNIA_DEADLINES:
            if deadline.name.lower() in deadline_name.lower():
                due_date = trigger_date + timedelta(days=deadline.days)
                
                # Adjust for weekends if needed (some deadlines are calendar days)
                if due_date.weekday() == 5:  # Saturday
                    due_date += timedelta(days=2)
                elif due_date.weekday() == 6:  # Sunday
                    due_date += timedelta(days=1)
                
                return {
                    "deadline": deadline.name,
                    "statute": deadline.statute,
                    "trigger_date": trigger_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "days_remaining": (due_date - datetime.now()).days,
                    "consequences": deadline.consequences,
                    "applicable_to": deadline.applicable_to
                }
        
        return {
            "error": f"Unknown deadline: {deadline_name}",
            "known_deadlines": [d.name for d in CALIFORNIA_DEADLINES]
        }
    
    @staticmethod
    def get_document_requirements(claim_type: str) -> List[str]:
        """
        Get required documentation for specific claim types.
        
        Args:
            claim_type: Type of claim
            
        Returns:
            List of required documents
        """
        requirements = {
            "differing_site_conditions": [
                "Original bid documents and plans",
                "Geotechnical reports referenced in contract",
                "Photos of actual conditions",
                "Written notice to owner",
                "Daily reports showing discovery",
                "Cost records for additional work",
                "Expert reports (if applicable)"
            ],
            "delay": [
                "Baseline schedule",
                "Updated schedules showing impacts",
                "Daily reports",
                "Correspondence regarding delays",
                "Weather records",
                "Time impact analysis",
                "Delay notices",
                "Cost documentation"
            ],
            "changes": [
                "Original contract scope",
                "RFIs and responses",
                "Field directives",
                "Change order requests",
                "Cost proposals",
                "Time extension requests",
                "Supporting documentation"
            ],
            "payment": [
                "Payment applications",
                "Certified payroll (prevailing wage)",
                "Lien releases",
                "Preliminary notices",
                "Stop notices (if applicable)",
                "Work completion documentation"
            ]
        }
        
        return requirements.get(claim_type, [
            "Contract documents",
            "Project correspondence",
            "Daily reports",
            "Cost records",
            "Schedule documentation"
        ])


# Singleton instance
california_domain = CaliforniaDomainConfig()