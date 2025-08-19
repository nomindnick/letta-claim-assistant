"""
California Public Works Entity Extractor.

Extracts California-specific entities, deadlines, and legal references
from construction claim documents.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from ..logging_conf import get_logger
from ..models import KnowledgeItem

logger = get_logger(__name__)


class CaliforniaEntityExtractor:
    """Extracts California public works specific entities from text."""
    
    def __init__(self):
        """Initialize the California entity extractor."""
        self.statute_patterns = self._compile_statute_patterns()
        self.entity_patterns = self._compile_entity_patterns()
        self.deadline_patterns = self._compile_deadline_patterns()
        self.claim_patterns = self._compile_claim_patterns()
        
    def _compile_statute_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for California statutes."""
        return {
            "public_contract": re.compile(
                r"(?:Public Contract Code|Pub\.?\s*Cont\.?\s*Code|PCC)\s*(?:§|Section|Sec\.?)?\s*(\d+(?:\.\d+)?(?:\s*(?:et seq\.|through|[-–—]\s*\d+(?:\.\d+)?)?)?)",
                re.IGNORECASE
            ),
            "government": re.compile(
                r"(?:Government Code|Gov(?:\'t)?\.?\s*Code|GC)\s*(?:§|Section|Sec\.?)?\s*(\d+(?:\.\d+)?(?:\s*(?:et seq\.|through|[-–—]\s*\d+(?:\.\d+)?)?)?)",
                re.IGNORECASE
            ),
            "civil": re.compile(
                r"(?:Civil Code|Civ\.?\s*Code|CC)\s*(?:§|Section|Sec\.?)?\s*(\d+(?:\.\d+)?(?:\s*(?:et seq\.|through|[-–—]\s*\d+(?:\.\d+)?)?)?)",
                re.IGNORECASE
            ),
            "labor": re.compile(
                r"(?:Labor Code|Lab\.?\s*Code|LC)\s*(?:§|Section|Sec\.?)?\s*(\d+(?:\.\d+)?(?:\s*(?:et seq\.|through|[-–—]\s*\d+(?:\.\d+)?)?)?)",
                re.IGNORECASE
            ),
            "business_professions": re.compile(
                r"(?:Business (?:and|&) Professions Code|Bus\.?\s*(?:&|and)\s*Prof\.?\s*Code|BPC)\s*(?:§|Section|Sec\.?)?\s*(\d+(?:\.\d+)?)",
                re.IGNORECASE
            )
        }
    
    def _compile_entity_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for California entities."""
        return {
            "state_agencies": re.compile(
                r"(?:California\s+)?(?:Department\s+of\s+(?:Transportation|General\s+Services|Water\s+Resources)|"
                r"Caltrans|DGS|DSA|DWR|DIR|"
                r"Division\s+of\s+the\s+State\s+Architect|"
                r"Department\s+of\s+Industrial\s+Relations|"
                r"State\s+Water\s+Resources\s+Control\s+Board|"
                r"California\s+(?:State\s+)?University|CSU|UC|"
                r"University\s+of\s+California)",
                re.IGNORECASE
            ),
            "local_agencies": re.compile(
                r"(?:County\s+of\s+\w+(?:\s+\w+)?|"
                r"City\s+of\s+\w+(?:\s+\w+)?|"
                r"City\s+and\s+County\s+of\s+\w+|"
                r"\w+(?:\s+\w+)?\s+(?:Unified\s+)?School\s+District|"
                r"\w+(?:\s+\w+)?\s+Community\s+College\s+District|"
                r"\w+(?:\s+\w+)?\s+Water\s+District|"
                r"\w+(?:\s+\w+)?\s+(?:Sanitation|Sanitary)\s+District|"
                r"\w+(?:\s+\w+)?\s+Transportation\s+(?:Authority|Agency)|"
                r"(?:Los\s+Angeles|San\s+Francisco|San\s+Diego|Orange\s+County|Sacramento)\s+\w+\s+(?:Authority|Agency|District))",
                re.IGNORECASE
            ),
            "contractors": re.compile(
                r"(?:(?:General|Prime)\s+Contractor|GC|"
                r"Subcontractor|Sub|"
                r"(?:First|Second|Third)[\s-]Tier\s+(?:Subcontractor|Sub)|"
                r"Material\s+Supplier|Supplier|"
                r"Equipment\s+(?:Lessor|Supplier)|"
                r"Design[\s-]Build(?:er)?\s+(?:Contractor|Entity)|DB|"
                r"Construction\s+Manager(?:\s+at\s+Risk)?|CM(?:AR)?|"
                r"(?:Joint\s+Venture|JV)\s+(?:Partner|Contractor))",
                re.IGNORECASE
            )
        }
    
    def _compile_deadline_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for deadline extraction."""
        return {
            "day_notice": re.compile(
                r"(\d+)[\s-]?day\s+(?:preliminary\s+)?notice",
                re.IGNORECASE
            ),
            "within_days": re.compile(
                r"within\s+(\d+)\s+(?:calendar\s+|working\s+|business\s+)?days",
                re.IGNORECASE
            ),
            "no_later": re.compile(
                r"(?:no|not)\s+later\s+than\s+(\d+)\s+(?:calendar\s+|working\s+|business\s+)?days",
                re.IGNORECASE
            ),
            "days_after": re.compile(
                r"(\d+)\s+(?:calendar\s+|working\s+|business\s+)?days\s+(?:after|from|following)",
                re.IGNORECASE
            ),
            "specific_date": re.compile(
                r"(?:by|before|on)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
                re.IGNORECASE
            )
        }
    
    def _compile_claim_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for claim types."""
        return {
            "differing_conditions": re.compile(
                r"(?:differing\s+site\s+condition|DSC|"
                r"Type\s+(?:I|II|1|2)\s+(?:differing\s+site\s+condition|DSC)|"
                r"(?:latent|concealed)\s+(?:physical\s+)?condition|"
                r"unforeseen\s+(?:site\s+)?condition)",
                re.IGNORECASE
            ),
            "change_order": re.compile(
                r"(?:change\s+order|CO|"
                r"(?:contract\s+)?modification|"
                r"(?:field\s+)?directive|"
                r"extra\s+work|"
                r"additional\s+work|"
                r"supplemental\s+agreement|"
                r"cardinal\s+change|"
                r"constructive\s+change)",
                re.IGNORECASE
            ),
            "delay": re.compile(
                r"(?:(?:excusable|compensable|concurrent|critical\s+path)\s+delay|"
                r"time\s+(?:extension|impact)|"
                r"(?:project\s+)?delay|"
                r"schedule\s+(?:impact|delay)|"
                r"suspension\s+of\s+(?:work|the\s+work)|"
                r"(?:force\s+majeure|act\s+of\s+god))",
                re.IGNORECASE
            ),
            "payment": re.compile(
                r"(?:(?:progress|final|retention)\s+payment|"
                r"payment\s+(?:application|request|claim)|"
                r"prompt\s+payment|"
                r"withheld\s+(?:payment|retention)|"
                r"stop\s+(?:payment\s+)?notice|"
                r"mechanics\s+lien|"
                r"payment\s+bond\s+claim)",
                re.IGNORECASE
            )
        }
    
    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Extract all California-specific entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of extracted entities
        """
        return {
            "statutes": self.extract_statutes(text),
            "agencies": self.extract_agencies(text),
            "deadlines": self.extract_deadlines(text),
            "claim_types": self.extract_claim_types(text),
            "notices": self.extract_notices(text),
            "bonds": self.extract_bonds(text),
            "key_dates": self.extract_key_dates(text),
            "monetary_amounts": self.extract_monetary_amounts(text)
        }
    
    def extract_statutes(self, text: str) -> List[Dict[str, str]]:
        """Extract California statute references."""
        statutes = []
        
        for statute_type, pattern in self.statute_patterns.items():
            for match in pattern.finditer(text):
                section = match.group(1)
                full_citation = match.group(0)
                
                statute_info = {
                    "type": statute_type,
                    "section": section,
                    "citation": full_citation,
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                }
                
                # Add known interpretations
                if statute_type == "public_contract":
                    if "7107" in section:
                        statute_info["description"] = "Prompt payment requirements"
                    elif "9100" in section:
                        statute_info["description"] = "Changes and extra work"
                    elif "10240" in section:
                        statute_info["description"] = "Retention requirements"
                elif statute_type == "government":
                    if "910" in section or "911" in section:
                        statute_info["description"] = "Government claims filing"
                    elif "12650" in section:
                        statute_info["description"] = "False Claims Act"
                elif statute_type == "civil":
                    if "8200" in section:
                        statute_info["description"] = "Preliminary notice requirements"
                    elif "8400" in section or "8500" in section:
                        statute_info["description"] = "Mechanics lien/stop notice"
                
                statutes.append(statute_info)
        
        return statutes
    
    def extract_agencies(self, text: str) -> List[Dict[str, str]]:
        """Extract California public agency references."""
        agencies = []
        seen = set()
        
        for agency_type, pattern in self.entity_patterns.items():
            if agency_type in ["state_agencies", "local_agencies"]:
                for match in pattern.finditer(text):
                    agency_name = match.group(0).strip()
                    
                    # Normalize and deduplicate
                    normalized = agency_name.lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        agencies.append({
                            "name": agency_name,
                            "type": agency_type.replace("_", " "),
                            "context": text[max(0, match.start()-30):min(len(text), match.end()+30)]
                        })
        
        return agencies
    
    def extract_deadlines(self, text: str) -> List[Dict[str, Any]]:
        """Extract deadline references and requirements."""
        deadlines = []
        
        for deadline_type, pattern in self.deadline_patterns.items():
            for match in pattern.finditer(text):
                deadline_info = {
                    "type": deadline_type,
                    "text": match.group(0),
                    "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                }
                
                # Extract specific deadline information
                if deadline_type in ["day_notice", "within_days", "no_later", "days_after"]:
                    deadline_info["days"] = int(match.group(1))
                    
                    # Check for calendar vs working days
                    if "calendar" in match.group(0).lower():
                        deadline_info["day_type"] = "calendar"
                    elif "working" in match.group(0).lower() or "business" in match.group(0).lower():
                        deadline_info["day_type"] = "working"
                    else:
                        deadline_info["day_type"] = "unspecified"
                
                elif deadline_type == "specific_date":
                    deadline_info["date"] = match.group(1)
                
                deadlines.append(deadline_info)
        
        return deadlines
    
    def extract_claim_types(self, text: str) -> List[Dict[str, str]]:
        """Extract claim type references."""
        claims = []
        seen = set()
        
        for claim_type, pattern in self.claim_patterns.items():
            for match in pattern.finditer(text):
                claim_text = match.group(0)
                normalized = claim_text.lower()
                
                if normalized not in seen:
                    seen.add(normalized)
                    claims.append({
                        "type": claim_type.replace("_", " "),
                        "text": claim_text,
                        "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
                    })
        
        return claims
    
    def extract_notices(self, text: str) -> List[Dict[str, str]]:
        """Extract notice references."""
        notice_pattern = re.compile(
            r"(?:preliminary\s+notice|20[\s-]?day\s+notice|"
            r"stop\s+(?:payment\s+)?notice|"
            r"mechanics\s+lien|"
            r"notice\s+of\s+(?:completion|cessation|commencement)|"
            r"notice\s+to\s+(?:proceed|owner)|"
            r"(?:first|second|final)\s+notice|"
            r"government\s+claim|"
            r"claim\s+(?:for\s+)?(?:payment|damages)|"
            r"notice\s+of\s+(?:delay|change|claim))",
            re.IGNORECASE
        )
        
        notices = []
        for match in notice_pattern.finditer(text):
            notices.append({
                "type": match.group(0),
                "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        return notices
    
    def extract_bonds(self, text: str) -> List[Dict[str, str]]:
        """Extract bond references."""
        bond_pattern = re.compile(
            r"(?:payment\s+bond|performance\s+bond|"
            r"bid\s+bond|proposal\s+bond|"
            r"(?:Little\s+)?Miller\s+Act(?:\s+bond)?|"
            r"license\s+bond|"
            r"warranty\s+bond|"
            r"maintenance\s+bond|"
            r"faithful\s+performance\s+bond)",
            re.IGNORECASE
        )
        
        bonds = []
        for match in bond_pattern.finditer(text):
            bonds.append({
                "type": match.group(0),
                "context": text[max(0, match.start()-50):min(len(text), match.end()+50)]
            })
        
        return bonds
    
    def extract_key_dates(self, text: str) -> List[Dict[str, str]]:
        """Extract important dates from text."""
        date_pattern = re.compile(
            r"(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)"
            r"\s+\d{1,2},?\s+\d{4}|"
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2})",
            re.IGNORECASE
        )
        
        dates = []
        for match in date_pattern.finditer(text):
            date_context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
            
            # Try to identify what the date relates to
            date_type = "unspecified"
            if any(word in date_context.lower() for word in ["notice", "notify", "notification"]):
                date_type = "notice"
            elif any(word in date_context.lower() for word in ["complet", "finish"]):
                date_type = "completion"
            elif any(word in date_context.lower() for word in ["start", "commenc", "began"]):
                date_type = "commencement"
            elif any(word in date_context.lower() for word in ["contract", "agreement", "sign"]):
                date_type = "contract"
            elif any(word in date_context.lower() for word in ["payment", "paid", "invoice"]):
                date_type = "payment"
            
            dates.append({
                "date": match.group(0),
                "type": date_type,
                "context": date_context
            })
        
        return dates
    
    def extract_monetary_amounts(self, text: str) -> List[Dict[str, str]]:
        """Extract monetary amounts from text."""
        money_pattern = re.compile(
            r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|thousand|k|m))?",
            re.IGNORECASE
        )
        
        amounts = []
        for match in money_pattern.finditer(text):
            amount_context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
            
            # Try to identify what the amount relates to
            amount_type = "unspecified"
            if any(word in amount_context.lower() for word in ["contract", "price", "sum"]):
                amount_type = "contract value"
            elif any(word in amount_context.lower() for word in ["change", "modification", "extra"]):
                amount_type = "change order"
            elif any(word in amount_context.lower() for word in ["claim", "damage", "loss"]):
                amount_type = "claim amount"
            elif any(word in amount_context.lower() for word in ["payment", "paid", "invoice"]):
                amount_type = "payment"
            elif any(word in amount_context.lower() for word in ["retention", "withhold"]):
                amount_type = "retention"
            
            amounts.append({
                "amount": match.group(0),
                "type": amount_type,
                "context": amount_context
            })
        
        return amounts
    
    def create_knowledge_items(self, extracted_entities: Dict[str, Any]) -> List[KnowledgeItem]:
        """
        Convert extracted entities to KnowledgeItem objects.
        
        Args:
            extracted_entities: Dictionary of extracted entities
            
        Returns:
            List of KnowledgeItem objects
        """
        knowledge_items = []
        
        # Create items for statutes
        for statute in extracted_entities.get("statutes", []):
            snippet = statute.get("description", "")
            if not snippet and "context" in statute:
                snippet = statute["context"][:200]
            knowledge_items.append(KnowledgeItem(
                type="Fact",
                label=f"Legal Reference: {statute['citation']}",
                support_snippet=snippet
            ))
        
        # Create items for agencies
        for agency in extracted_entities.get("agencies", []):
            knowledge_items.append(KnowledgeItem(
                type="Entity",
                label=f"Agency: {agency['name']}",
                actors=[agency["name"]],
                support_snippet=agency["context"][:200]
            ))
        
        # Create items for deadlines
        for deadline in extracted_entities.get("deadlines", []):
            label = f"Deadline: {deadline['text']}"
            if "days" in deadline:
                label += f" ({deadline['days']} {deadline.get('day_type', 'days')})"
            
            knowledge_items.append(KnowledgeItem(
                type="Event",
                label=label,
                support_snippet=deadline["context"][:200]
            ))
        
        # Create items for claim types
        for claim in extracted_entities.get("claim_types", []):
            knowledge_items.append(KnowledgeItem(
                type="Issue",
                label=f"Claim Type: {claim['type']}",
                support_snippet=claim["context"][:200]
            ))
        
        # Create items for key dates
        for date_info in extracted_entities.get("key_dates", []):
            # Try to convert date to ISO format
            date_str = None
            try:
                from datetime import datetime
                # Try various date formats
                for fmt in ["%B %d, %Y", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d"]:
                    try:
                        parsed = datetime.strptime(date_info["date"], fmt)
                        date_str = parsed.isoformat()
                        break
                    except:
                        continue
            except:
                pass
            
            knowledge_items.append(KnowledgeItem(
                type="Event",
                label=f"{date_info['type'].title()} Date: {date_info['date']}",
                date=date_str,  # Use ISO format or None
                support_snippet=date_info["context"][:200]
            ))
        
        return knowledge_items


# Singleton instance
california_extractor = CaliforniaEntityExtractor()