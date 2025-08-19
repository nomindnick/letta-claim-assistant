"""
California Public Works Follow-up Templates.

Provides domain-specific follow-up questions for California construction claims,
organized by category and priority.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

from .logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class FollowUpTemplate:
    """Template for a follow-up question."""
    
    question: str
    category: str
    priority: float  # 0.0 to 1.0
    triggers: List[str]  # Keywords that trigger this question
    requires_expert: bool = False
    reasoning: str = ""


class CaliforniaFollowUpTemplates:
    """Manages California-specific follow-up question templates."""
    
    def __init__(self):
        """Initialize follow-up templates."""
        self.templates = self._initialize_templates()
        self.categories = self._get_categories()
    
    def _initialize_templates(self) -> List[FollowUpTemplate]:
        """Initialize all follow-up templates."""
        templates = []
        
        # Notice and Deadline Templates
        templates.extend([
            FollowUpTemplate(
                question="Have you served the required 20-day preliminary notice under Civil Code §8200?",
                category="Notice Compliance",
                priority=0.95,
                triggers=["preliminary notice", "20-day", "lien rights", "stop notice"],
                reasoning="Preliminary notice is prerequisite for mechanics lien and stop notice rights"
            ),
            FollowUpTemplate(
                question="When was the notice of completion recorded, and have you calculated the deadline for filing a mechanics lien?",
                category="Notice Compliance",
                priority=0.9,
                triggers=["completion", "mechanics lien", "recording"],
                reasoning="90-day deadline runs from notice of completion recording"
            ),
            FollowUpTemplate(
                question="Has a government claim been filed pursuant to Government Code §910-911, and when does the 6-month statute of limitations expire?",
                category="Notice Compliance",
                priority=0.95,
                triggers=["government claim", "statute of limitations", "public entity"],
                reasoning="Government claim is mandatory prerequisite to lawsuit"
            ),
            FollowUpTemplate(
                question="Were all contractual notice requirements met within the specified timeframes?",
                category="Notice Compliance",
                priority=0.85,
                triggers=["notice", "deadline", "contract requirement"],
                reasoning="Contractual notice often required for claims"
            ),
            FollowUpTemplate(
                question="Has the public entity responded to your government claim, and if rejected, when does the 6-month deadline to file suit expire?",
                category="Notice Compliance",
                priority=0.9,
                triggers=["government claim", "rejection", "lawsuit"],
                reasoning="6-month deadline from rejection is jurisdictional"
            )
        ])
        
        # Documentation Templates
        templates.extend([
            FollowUpTemplate(
                question="Do you have certified payroll records demonstrating prevailing wage compliance for all affected workers?",
                category="Documentation",
                priority=0.85,
                triggers=["prevailing wage", "certified payroll", "DIR", "labor compliance"],
                reasoning="Certified payroll critical for public works compliance"
            ),
            FollowUpTemplate(
                question="Are there daily reports documenting the discovery and impact of the differing site condition?",
                category="Documentation",
                priority=0.9,
                triggers=["differing site", "daily reports", "discovery", "DSC"],
                reasoning="Contemporaneous documentation essential for DSC claims"
            ),
            FollowUpTemplate(
                question="Do you have the baseline schedule and all updates showing the critical path impact?",
                category="Documentation",
                priority=0.85,
                triggers=["delay", "schedule", "critical path", "time impact"],
                reasoning="Schedule analysis required for delay claims"
            ),
            FollowUpTemplate(
                question="Are all RFIs, submittals, and responses documented to show the owner's involvement?",
                category="Documentation",
                priority=0.8,
                triggers=["RFI", "submittal", "approval", "direction"],
                reasoning="Owner direction evidence for change claims"
            ),
            FollowUpTemplate(
                question="Have you maintained separate cost codes for the disputed work to facilitate damage calculation?",
                category="Documentation",
                priority=0.85,
                triggers=["cost", "damages", "accounting", "segregation"],
                reasoning="Cost segregation crucial for damage proof"
            )
        ])
        
        # Procedural Templates
        templates.extend([
            FollowUpTemplate(
                question="Has the contractual dispute resolution process been followed, including any required meet and confer?",
                category="Procedural",
                priority=0.85,
                triggers=["dispute", "meet and confer", "mediation", "DRB"],
                reasoning="Exhaustion of contract procedures often required"
            ),
            FollowUpTemplate(
                question="Are there any Little Miller Act bond claim deadlines approaching?",
                category="Procedural",
                priority=0.9,
                triggers=["payment bond", "Miller Act", "bond claim"],
                reasoning="Bond claims have strict notice and suit deadlines"
            ),
            FollowUpTemplate(
                question="Has the contractor complied with all stop notice procedures under Civil Code §8500 et seq.?",
                category="Procedural",
                priority=0.85,
                triggers=["stop notice", "payment", "retention"],
                reasoning="Stop notice procedures are statutory requirements"
            ),
            FollowUpTemplate(
                question="Does the contract contain a 'no damages for delay' clause that might limit recovery?",
                category="Procedural",
                priority=0.8,
                triggers=["delay", "no damages", "limitation", "waiver"],
                reasoning="Contractual limitations may affect remedies"
            ),
            FollowUpTemplate(
                question="Have you verified compliance with the Subletting and Subcontracting Fair Practices Act?",
                category="Procedural",
                priority=0.75,
                triggers=["subcontractor", "listing", "substitution"],
                reasoning="SSFPA violations can affect payment rights"
            )
        ])
        
        # Evidence Templates
        templates.extend([
            FollowUpTemplate(
                question="Are there photographs or video documenting the conditions before, during, and after the issue arose?",
                category="Evidence",
                priority=0.85,
                triggers=["photo", "video", "documentation", "condition"],
                reasoning="Visual evidence powerful for claims"
            ),
            FollowUpTemplate(
                question="Do you have correspondence showing the owner's knowledge of the issue and response?",
                category="Evidence",
                priority=0.8,
                triggers=["correspondence", "email", "letter", "notice"],
                reasoning="Owner knowledge critical for many claims"
            ),
            FollowUpTemplate(
                question="Are there meeting minutes or other records documenting discussions about the claim?",
                category="Evidence",
                priority=0.75,
                triggers=["meeting", "minutes", "discussion", "conference"],
                reasoning="Meeting records provide claim timeline"
            ),
            FollowUpTemplate(
                question="Have you obtained weather data to support any weather-related delay claims?",
                category="Evidence",
                priority=0.7,
                triggers=["weather", "rain", "force majeure", "excusable delay"],
                reasoning="Weather data necessary for excusable delays"
            ),
            FollowUpTemplate(
                question="Do you have the original bid documents showing what was contemplated at bid time?",
                category="Evidence",
                priority=0.85,
                triggers=["bid", "estimate", "proposal", "contemplated"],
                reasoning="Bid basis essential for changed condition claims"
            )
        ])
        
        # Expert Analysis Templates
        templates.extend([
            FollowUpTemplate(
                question="Would a scheduling expert's CPM analysis strengthen your delay claim?",
                category="Expert Analysis",
                priority=0.85,
                triggers=["delay", "critical path", "float", "concurrent"],
                requires_expert=True,
                reasoning="Expert scheduling analysis often required for complex delays"
            ),
            FollowUpTemplate(
                question="Should a geotechnical expert evaluate the differing site conditions?",
                category="Expert Analysis",
                priority=0.85,
                triggers=["soil", "geotechnical", "differing site", "subsurface"],
                requires_expert=True,
                reasoning="Geotechnical expertise needed for DSC claims"
            ),
            FollowUpTemplate(
                question="Would a quantum expert help calculate lost productivity or cumulative impact damages?",
                category="Expert Analysis",
                priority=0.8,
                triggers=["productivity", "inefficiency", "cumulative impact", "measured mile"],
                requires_expert=True,
                reasoning="Productivity loss requires specialized analysis"
            ),
            FollowUpTemplate(
                question="Should a forensic accountant review the damage calculations and cost segregation?",
                category="Expert Analysis",
                priority=0.75,
                triggers=["damages", "costs", "accounting", "audit"],
                requires_expert=True,
                reasoning="Complex damages benefit from accounting expertise"
            ),
            FollowUpTemplate(
                question="Would a construction defect expert help establish the standard of care?",
                category="Expert Analysis",
                priority=0.8,
                triggers=["defect", "standard of care", "workmanship", "warranty"],
                requires_expert=True,
                reasoning="Standard of care requires expert testimony"
            )
        ])
        
        # Damages Templates
        templates.extend([
            FollowUpTemplate(
                question="Have you calculated the 2% monthly penalty for prompt payment violations under PCC §7107?",
                category="Damages",
                priority=0.8,
                triggers=["prompt payment", "penalty", "interest", "7107"],
                reasoning="Statutory penalties available for payment delays"
            ),
            FollowUpTemplate(
                question="Are there additional costs for acceleration or compression that should be included?",
                category="Damages",
                priority=0.75,
                triggers=["acceleration", "compression", "overtime", "premium"],
                reasoning="Acceleration costs often overlooked"
            ),
            FollowUpTemplate(
                question="Have you included home office overhead using an accepted methodology (Eichleay, etc.)?",
                category="Damages",
                priority=0.75,
                triggers=["overhead", "home office", "Eichleay", "markup"],
                reasoning="Home office overhead recoverable for delays"
            ),
            FollowUpTemplate(
                question="Are there subcontractor pass-through claims that need to be included?",
                category="Damages",
                priority=0.7,
                triggers=["subcontractor", "pass-through", "sponsor"],
                reasoning="Pass-through claims require special procedures"
            ),
            FollowUpTemplate(
                question="Have you considered claiming for extended general conditions and equipment costs?",
                category="Damages",
                priority=0.75,
                triggers=["general conditions", "extended", "equipment", "standby"],
                reasoning="Extended overhead often significant in delays"
            )
        ])
        
        # Strategic Templates
        templates.extend([
            FollowUpTemplate(
                question="Should you consider filing a False Claims Act complaint if fraud is suspected?",
                category="Strategic",
                priority=0.7,
                triggers=["fraud", "false claims", "whistleblower", "qui tam"],
                reasoning="FCA provides treble damages and attorney fees"
            ),
            FollowUpTemplate(
                question="Would pursuing a writ of mandate be appropriate for challenging the agency's decision?",
                category="Strategic",
                priority=0.7,
                triggers=["mandate", "writ", "administrative", "appeal"],
                reasoning="Writ may be required for certain agency actions"
            ),
            FollowUpTemplate(
                question="Should you consider requesting Public Records Act documents to support your claim?",
                category="Strategic",
                priority=0.75,
                triggers=["public records", "PRA", "CPRA", "documents"],
                reasoning="PRA can reveal helpful agency documents"
            ),
            FollowUpTemplate(
                question="Are there other contractors with similar claims that might support your position?",
                category="Strategic",
                priority=0.65,
                triggers=["other contractors", "similar claims", "pattern"],
                reasoning="Pattern evidence strengthens individual claims"
            ),
            FollowUpTemplate(
                question="Should the surety be notified and involved in the claim process?",
                category="Strategic",
                priority=0.75,
                triggers=["surety", "bond", "default", "termination"],
                reasoning="Surety involvement may be required or beneficial"
            )
        ])
        
        return templates
    
    def _get_categories(self) -> Dict[str, str]:
        """Get category descriptions."""
        return {
            "Notice Compliance": "Statutory and contractual notice requirements",
            "Documentation": "Required documentation and records",
            "Procedural": "Procedural requirements and compliance",
            "Evidence": "Evidence gathering and preservation",
            "Expert Analysis": "Expert witness and technical analysis",
            "Damages": "Damage calculation and recovery",
            "Strategic": "Strategic considerations and options"
        }
    
    def get_relevant_followups(
        self,
        query: str,
        answer: str,
        extracted_entities: Optional[Dict[str, Any]] = None,
        max_questions: int = 4,
        category_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get relevant follow-up questions based on context.
        
        Args:
            query: User's original query
            answer: Generated answer
            extracted_entities: Extracted California entities
            max_questions: Maximum number of questions to return
            category_filter: Optional list of categories to include
            
        Returns:
            List of follow-up questions with metadata
        """
        context = f"{query} {answer}".lower()
        
        # Score each template
        scored_templates = []
        for template in self.templates:
            # Skip if category filter applied
            if category_filter and template.category not in category_filter:
                continue
            
            # Calculate relevance score
            score = template.priority
            
            # Check for trigger words
            trigger_matches = sum(1 for trigger in template.triggers 
                                 if trigger.lower() in context)
            if trigger_matches > 0:
                score += 0.2 * min(trigger_matches, 3)  # Boost for triggers
            
            # Check extracted entities if provided
            if extracted_entities:
                # Boost for relevant claim types
                for claim in extracted_entities.get("claim_types", []):
                    if any(trigger in claim.get("type", "").lower() 
                          for trigger in template.triggers):
                        score += 0.1
                
                # Boost for deadline-related templates if deadlines found
                if extracted_entities.get("deadlines") and "deadline" in template.category.lower():
                    score += 0.15
                
                # Boost for agency-related if agencies found
                if extracted_entities.get("agencies") and any(
                    word in template.question.lower() 
                    for word in ["entity", "agency", "public", "government"]
                ):
                    score += 0.1
            
            if score > 0.3:  # Minimum threshold
                scored_templates.append((score, template))
        
        # Sort by score and diversify
        scored_templates.sort(key=lambda x: x[0], reverse=True)
        
        # Select diverse questions
        selected = []
        used_categories = set()
        
        for score, template in scored_templates:
            # Prioritize category diversity
            if template.category not in used_categories or len(selected) < 2:
                selected.append({
                    "question": template.question,
                    "category": template.category,
                    "priority": round(score, 2),
                    "reasoning": template.reasoning,
                    "requires_expert": template.requires_expert,
                    "related_entities": [t for t in template.triggers if t.lower() in context]
                })
                used_categories.add(template.category)
            
            if len(selected) >= max_questions:
                break
        
        # If we don't have enough, add high-priority general questions
        if len(selected) < max_questions:
            general_questions = [
                {
                    "question": "What is the total value of the claim and how was it calculated?",
                    "category": "Damages",
                    "priority": 0.7,
                    "reasoning": "Understanding claim value is fundamental",
                    "requires_expert": False,
                    "related_entities": []
                },
                {
                    "question": "What is the current status of the dispute and what are the next steps?",
                    "category": "Procedural",
                    "priority": 0.7,
                    "reasoning": "Current status determines strategy",
                    "requires_expert": False,
                    "related_entities": []
                },
                {
                    "question": "Are there any upcoming deadlines that need immediate attention?",
                    "category": "Notice Compliance",
                    "priority": 0.8,
                    "reasoning": "Deadline management is critical",
                    "requires_expert": False,
                    "related_entities": []
                }
            ]
            
            for q in general_questions:
                if len(selected) < max_questions and q["question"] not in [s["question"] for s in selected]:
                    selected.append(q)
        
        return selected
    
    def get_category_questions(self, category: str) -> List[FollowUpTemplate]:
        """
        Get all questions for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of templates in that category
        """
        return [t for t in self.templates if t.category == category]
    
    def get_expert_questions(self) -> List[FollowUpTemplate]:
        """Get all questions that require expert analysis."""
        return [t for t in self.templates if t.requires_expert]
    
    def get_high_priority_questions(self, threshold: float = 0.85) -> List[FollowUpTemplate]:
        """Get high-priority questions above threshold."""
        return [t for t in self.templates if t.priority >= threshold]


# Singleton instance
california_followup_templates = CaliforniaFollowUpTemplates()