"""
Unit tests for California public works domain optimization.

Tests domain configuration, entity extraction, follow-up templates,
and compliance validation for California construction claims.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.letta_domain_california import (
    CaliforniaDomainConfig,
    california_domain,
    CALIFORNIA_DEADLINES,
    CALIFORNIA_PUBLIC_ENTITIES
)
from app.extractors.california_extractor import (
    CaliforniaEntityExtractor,
    california_extractor
)
from app.followup_templates_california import (
    CaliforniaFollowUpTemplates,
    california_followup_templates,
    FollowUpTemplate
)
from app.validators.california_validator import (
    CaliforniaComplianceValidator,
    california_validator,
    ValidationResult,
    ClaimValidation
)
from app.models import KnowledgeItem


class TestCaliforniaDomainConfig:
    """Test California domain configuration."""
    
    def test_get_system_prompt(self):
        """Test system prompt generation."""
        prompt = california_domain.get_system_prompt("Test Matter")
        
        assert "California public works" in prompt
        assert "Test Matter" in prompt
        assert "Public Contract Code" in prompt
        assert "Government Code claims" in prompt
        assert "prevailing wage" in prompt.lower()
        assert "mechanics lien" in prompt.lower()
    
    def test_get_memory_blocks(self):
        """Test memory block generation."""
        blocks = california_domain.get_memory_blocks("Test Matter")
        
        assert len(blocks) >= 2
        assert any(b["label"] == "human" for b in blocks)
        assert any(b["label"] == "persona" for b in blocks)
        assert any("California" in b["value"] for b in blocks)
        assert any("Test Matter" in b["value"] for b in blocks)
    
    def test_get_extraction_patterns(self):
        """Test extraction pattern retrieval."""
        patterns = california_domain.get_extraction_patterns()
        
        assert "statutes" in patterns
        assert "public_entities" in patterns
        assert "deadlines" in patterns
        assert "claim_types" in patterns
        assert "notices" in patterns
        assert "bonds" in patterns
    
    def test_get_claim_categories(self):
        """Test claim categorization."""
        categories = california_domain.get_claim_categories()
        
        assert "differing_site_conditions" in categories
        assert "delay" in categories
        assert "changes" in categories
        assert "payment" in categories
        
        # Check category details
        dsc = categories["differing_site_conditions"]
        assert "key_elements" in dsc
        assert "notice_requirement" in dsc
    
    def test_validate_deadline(self):
        """Test deadline validation."""
        trigger_date = datetime.now()
        result = california_domain.validate_deadline(
            "Preliminary Notice (20-Day)",
            trigger_date
        )
        
        assert "deadline" in result
        assert "statute" in result
        assert "due_date" in result
        assert "days_remaining" in result
        assert result["statute"] == "Civil Code § 8200"
    
    def test_get_document_requirements(self):
        """Test document requirement retrieval."""
        docs = california_domain.get_document_requirements("delay")
        
        assert "Baseline schedule" in docs
        assert "Daily reports" in docs
        assert "Time impact analysis" in docs
        assert len(docs) > 5


class TestCaliforniaEntityExtractor:
    """Test California entity extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CaliforniaEntityExtractor()
    
    def test_extract_statutes(self):
        """Test statute extraction."""
        text = """
        The contractor must comply with Public Contract Code Section 7107
        for prompt payment. Also see Government Code §910 for claims.
        Civil Code §8200 requires preliminary notice.
        """
        
        statutes = self.extractor.extract_statutes(text)
        
        assert len(statutes) >= 3
        assert any("7107" in s["section"] for s in statutes)
        assert any("910" in s["section"] for s in statutes)
        assert any("8200" in s["section"] for s in statutes)
        assert any(s["type"] == "public_contract" for s in statutes)
    
    def test_extract_agencies(self):
        """Test agency extraction."""
        text = """
        Caltrans has issued the notice to proceed. The project is also
        overseen by DGS and the County of Los Angeles. The school district
        is also involved.
        """
        
        agencies = self.extractor.extract_agencies(text)
        
        assert len(agencies) >= 3
        assert any("Caltrans" in a["name"] for a in agencies)
        assert any("County of Los Angeles" in a["name"] for a in agencies)
        assert any("school district" in a["name"].lower() for a in agencies)
    
    def test_extract_deadlines(self):
        """Test deadline extraction."""
        text = """
        The preliminary notice must be served within 20 days.
        You have 90 days after completion to file a mechanics lien.
        The response is due no later than 30 days from notice.
        """
        
        deadlines = self.extractor.extract_deadlines(text)
        
        assert len(deadlines) >= 3
        assert any(d["days"] == 20 for d in deadlines if "days" in d)
        assert any(d["days"] == 90 for d in deadlines if "days" in d)
        assert any(d["days"] == 30 for d in deadlines if "days" in d)
    
    def test_extract_claim_types(self):
        """Test claim type extraction."""
        text = """
        We encountered a differing site condition when excavating.
        This caused a compensable delay to the project. We submitted
        a change order for the extra work required.
        """
        
        claims = self.extractor.extract_claim_types(text)
        
        assert len(claims) >= 3
        assert any("differing" in c["type"] for c in claims)
        assert any("delay" in c["type"] for c in claims)
        assert any("change" in c["type"] for c in claims)
    
    def test_extract_notices(self):
        """Test notice extraction."""
        text = """
        We served the preliminary notice last week. A stop payment notice
        will be filed if payment is not received. The mechanics lien was
        recorded yesterday.
        """
        
        notices = self.extractor.extract_notices(text)
        
        assert len(notices) >= 3
        assert any("preliminary notice" in n["type"].lower() for n in notices)
        assert any("stop" in n["type"].lower() for n in notices)
        assert any("mechanics lien" in n["type"].lower() for n in notices)
    
    def test_extract_monetary_amounts(self):
        """Test monetary amount extraction."""
        text = """
        The contract value is $5,000,000. The change order is for $250,000.
        We are claiming $75,000 in delay damages.
        """
        
        amounts = self.extractor.extract_monetary_amounts(text)
        
        assert len(amounts) >= 3
        assert any("$5,000,000" in a["amount"] for a in amounts)
        assert any("contract" in a["type"].lower() for a in amounts)
    
    def test_create_knowledge_items(self):
        """Test knowledge item creation."""
        extracted = {
            "statutes": [{"citation": "PCC 7107", "description": "Prompt payment"}],
            "agencies": [{"name": "Caltrans", "context": "Test context"}],
            "deadlines": [{"text": "20 days", "days": 20, "context": "Notice deadline"}]
        }
        
        items = self.extractor.create_knowledge_items(extracted)
        
        assert len(items) >= 3
        assert any(item.type == "Fact" for item in items)
        assert any(item.type == "Entity" for item in items)
        assert any(item.type == "Event" for item in items)


class TestCaliforniaFollowUpTemplates:
    """Test California follow-up templates."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.templates = CaliforniaFollowUpTemplates()
    
    def test_template_initialization(self):
        """Test template initialization."""
        assert len(self.templates.templates) > 30
        assert len(self.templates.categories) > 5
        
        # Check template structure
        for template in self.templates.templates:
            assert template.question
            assert template.category
            assert 0 <= template.priority <= 1
            assert isinstance(template.triggers, list)
    
    def test_get_relevant_followups_with_triggers(self):
        """Test follow-up generation with trigger words."""
        query = "We have a differing site condition claim"
        answer = "You need to provide prompt notice per the contract."
        
        followups = self.templates.get_relevant_followups(
            query=query,
            answer=answer,
            max_questions=4
        )
        
        assert len(followups) <= 4
        assert all("question" in f for f in followups)
        assert all("category" in f for f in followups)
        assert all("priority" in f for f in followups)
        
        # Should prioritize DSC-related questions
        assert any("differing site" in f["question"].lower() or 
                  "document" in f["question"].lower() for f in followups)
    
    def test_get_relevant_followups_with_entities(self):
        """Test follow-up generation with extracted entities."""
        query = "Caltrans rejected our claim"
        answer = "You may need to file a government claim."
        
        entities = {
            "agencies": [{"name": "Caltrans"}],
            "claim_types": [{"type": "claim"}]
        }
        
        followups = self.templates.get_relevant_followups(
            query=query,
            answer=answer,
            extracted_entities=entities,
            max_questions=3
        )
        
        assert len(followups) <= 3
        # Should include government claim questions
        assert any("government claim" in f["question"].lower() for f in followups)
    
    def test_get_category_questions(self):
        """Test category-specific question retrieval."""
        notice_questions = self.templates.get_category_questions("Notice Compliance")
        
        assert len(notice_questions) > 0
        assert all(q.category == "Notice Compliance" for q in notice_questions)
        assert any("preliminary notice" in q.question.lower() for q in notice_questions)
    
    def test_get_expert_questions(self):
        """Test expert question retrieval."""
        expert_questions = self.templates.get_expert_questions()
        
        assert len(expert_questions) > 0
        assert all(q.requires_expert for q in expert_questions)
        assert any("scheduling expert" in q.question.lower() for q in expert_questions)
    
    def test_get_high_priority_questions(self):
        """Test high-priority question retrieval."""
        high_priority = self.templates.get_high_priority_questions(threshold=0.9)
        
        assert len(high_priority) > 0
        assert all(q.priority >= 0.9 for q in high_priority)


class TestCaliforniaComplianceValidator:
    """Test California compliance validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CaliforniaComplianceValidator()
    
    def test_validate_claim_basic(self):
        """Test basic claim validation."""
        claim_data = {
            "prompt_written_notice_to_owner": True,
            "evidence_of_reliance_on_contract_documents": True,
            "documentation_of_actual_conditions": True,
            "cost_and_time_impact_analysis": True,
            "compliance_with_contract_notice_provisions": True
        }
        
        validation = self.validator.validate_claim(
            claim_type="differing_site_conditions",
            claim_data=claim_data
        )
        
        assert isinstance(validation, ClaimValidation)
        assert validation.claim_type == "differing_site_conditions"
        assert validation.compliance_score > 0
    
    def test_validate_claim_with_missing_requirements(self):
        """Test validation with missing requirements."""
        claim_data = {
            "prompt_written_notice_to_owner": False,
            "evidence_of_reliance_on_contract_documents": True
        }
        
        validation = self.validator.validate_claim(
            claim_type="differing_site_conditions",
            claim_data=claim_data
        )
        
        assert len(validation.missing_items) > 0
        assert len(validation.warnings) > 0
        assert validation.compliance_score < 1.0
    
    def test_validate_deadline(self):
        """Test deadline validation."""
        trigger_date = datetime.now() - timedelta(days=10)
        
        result = self.validator.validate_deadline(
            deadline_type="preliminary_notice",
            trigger_date=trigger_date
        )
        
        assert isinstance(result, ValidationResult)
        assert result.category == "Deadlines"
        assert result.statute == "Civil Code §8200"
    
    def test_validate_deadline_expired(self):
        """Test expired deadline validation."""
        trigger_date = datetime.now() - timedelta(days=100)
        
        result = self.validator.validate_deadline(
            deadline_type="mechanics_lien",
            trigger_date=trigger_date
        )
        
        assert not result.valid
        assert result.severity == "error"
        assert "expired" in result.message.lower()
    
    def test_validate_deadline_approaching(self):
        """Test approaching deadline validation."""
        trigger_date = datetime.now() - timedelta(days=70)
        
        result = self.validator.validate_deadline(
            deadline_type="mechanics_lien",
            trigger_date=trigger_date
        )
        
        assert not result.valid
        assert result.severity == "warning"
        assert "days" in result.message.lower()
    
    def test_get_claim_checklist(self):
        """Test claim checklist generation."""
        checklist = self.validator.get_claim_checklist("delay")
        
        assert len(checklist) > 5
        assert "Identify all applicable deadlines" in checklist
        assert any("schedule" in item.lower() for item in checklist)
        assert any("notice" in item.lower() for item in checklist)
    
    def test_validate_with_california_entities(self):
        """Test validation with extracted entities."""
        claim_data = {
            "public_entity": "Caltrans",
            "is_public_works": True,
            "prevailing_wage_compliance": False,
            "government_claim_filed": False
        }
        
        entities = {
            "agencies": [{"name": "Caltrans", "type": "state agency"}],
            "statutes": [{"citation": "PCC 7107", "type": "public_contract"}]
        }
        
        validation = self.validator.validate_claim(
            claim_type="payment",
            claim_data=claim_data,
            extracted_entities=entities
        )
        
        # Should have warnings about prevailing wage and government claim
        assert any("prevailing wage" in w.message.lower() for w in validation.warnings)
        assert any("government claim" in e.message.lower() for e in validation.errors)


@pytest.mark.asyncio
class TestIntegration:
    """Test integration of California domain features."""
    
    async def test_domain_config_integration(self):
        """Test domain configuration integration."""
        from app.letta_config import config_manager
        
        agent_config = config_manager.get_agent_config(
            matter_name="Test California Matter",
            use_california_domain=True
        )
        
        assert "California" in agent_config.system_prompt
        assert agent_config.metadata["domain"] == "california_public_works"
    
    async def test_extractor_integration(self):
        """Test entity extraction integration."""
        text = """
        Caltrans issued a stop work order on January 15, 2024.
        We filed a government claim per Gov Code 910 within 30 days.
        The contract value is $10 million under PCC 10240.
        """
        
        entities = california_extractor.extract_all(text)
        knowledge_items = california_extractor.create_knowledge_items(entities)
        
        assert len(entities["agencies"]) > 0
        assert len(entities["statutes"]) > 0
        assert len(entities["deadlines"]) > 0
        assert len(knowledge_items) > 0
        
        # Check knowledge item types
        assert any(k.type == "Entity" for k in knowledge_items)
        assert any(k.type == "Fact" for k in knowledge_items)
    
    async def test_followup_integration(self):
        """Test follow-up template integration."""
        query = "We have a delay claim against the County of Sacramento"
        answer = "You need to follow the contract dispute procedures."
        
        entities = california_extractor.extract_all(query + " " + answer)
        followups = california_followup_templates.get_relevant_followups(
            query=query,
            answer=answer,
            extracted_entities=entities,
            max_questions=4
        )
        
        assert len(followups) > 0
        assert len(followups) <= 4
        
        # Should have relevant follow-ups for delay claims
        questions = [f["question"] for f in followups]
        assert any("delay" in q.lower() or "schedule" in q.lower() or 
                  "notice" in q.lower() for q in questions)
    
    async def test_validator_integration(self):
        """Test compliance validation integration."""
        claim_text = """
        We encountered differing site conditions on the Caltrans project.
        Notice was provided within 5 days of discovery. The condition was
        different from what was shown in the geotechnical report.
        """
        
        entities = california_extractor.extract_all(claim_text)
        
        claim_data = {
            "text": claim_text,
            "discovery_date": datetime.now() - timedelta(days=10),
            "notice_date": datetime.now() - timedelta(days=5),
            "public_entity": "Caltrans",
            "is_public_works": True
        }
        
        validation = california_validator.validate_claim(
            claim_type="differing_site_conditions",
            claim_data=claim_data,
            extracted_entities=entities
        )
        
        assert validation.claim_type == "differing_site_conditions"
        assert validation.compliance_score > 0
        
        # Check for specific validations
        if validation.warnings:
            assert any("notice" in w.message.lower() or 
                      "document" in w.message.lower() for w in validation.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])