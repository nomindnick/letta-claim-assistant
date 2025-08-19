"""
Pydantic data models for the Letta Construction Claim Assistant.

Defines core data structures for Matters, configuration, and API communication
with proper validation and serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Union
from datetime import datetime
import uuid
from pydantic import BaseModel, Field, field_validator, model_serializer


class MatterPaths(BaseModel):
    """Matter directory paths structure."""
    
    root: Path
    docs: Path
    docs_ocr: Path
    parsed: Path
    vectors: Path
    knowledge: Path
    chat: Path
    logs: Path

    @model_serializer
    def serialize_model(self):
        """Serialize model with Path objects as strings."""
        return {
            field: str(value) if isinstance(value, Path) else value
            for field, value in self.__dict__.items()
        }

    @classmethod
    def from_root(cls, root_path: Path) -> 'MatterPaths':
        """Create MatterPaths from a root directory path."""
        return cls(
            root=root_path,
            docs=root_path / "docs",
            docs_ocr=root_path / "docs_ocr",
            parsed=root_path / "parsed",
            vectors=root_path / "vectors",
            knowledge=root_path / "knowledge",
            chat=root_path / "chat",
            logs=root_path / "logs"
        )


class Matter(BaseModel):
    """Matter representation with metadata and paths."""
    
    id: str = Field(..., description="Unique matter identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable matter name")
    slug: str = Field(..., description="URL-safe matter identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Matter creation timestamp")
    embedding_model: str = Field(default="nomic-embed-text", description="Embedding model used")
    generation_model: str = Field(default="gpt-oss:20b", description="Generation model used")
    paths: MatterPaths

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate matter name."""
        if not v or not v.strip():
            raise ValueError("Matter name cannot be empty")
        return v.strip()

    @field_validator('slug')
    @classmethod
    def validate_slug(cls, v):
        """Validate matter slug format."""
        import re
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError("Slug must contain only lowercase letters, numbers, and hyphens")
        return v
    
    @model_serializer
    def serialize_model(self):
        """Serialize model with proper datetime and Path formatting."""
        result = {}
        for field, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[field] = value.isoformat()
            elif isinstance(value, Path):
                result[field] = str(value)
            elif hasattr(value, 'model_dump'):
                result[field] = value.model_dump()
            else:
                result[field] = value
        return result

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to configuration dictionary for JSON storage."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "vector_path": "vectors/chroma",
            "letta_path": "knowledge/letta_state"
        }

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], root_path: Path) -> 'Matter':
        """Create Matter from configuration dictionary."""
        paths = MatterPaths.from_root(root_path)
        return cls(
            id=config_dict["id"],
            name=config_dict["name"],
            slug=config_dict["slug"],
            created_at=datetime.fromisoformat(config_dict["created_at"]),
            embedding_model=config_dict.get("embedding_model", "nomic-embed-text"),
            generation_model=config_dict.get("generation_model", "gpt-oss:20b"),
            paths=paths
        )


class SourceChunk(BaseModel):
    """Source chunk with document and page information."""
    
    doc: str = Field(..., description="Document name")
    page_start: int = Field(..., ge=1, description="Starting page number")
    page_end: int = Field(..., ge=1, description="Ending page number")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")

    @field_validator('page_end')
    @classmethod
    def validate_page_range(cls, v, info):
        """Ensure page_end >= page_start."""
        if hasattr(info, 'data') and info.data and 'page_start' in info.data:
            page_start = info.data.get('page_start')
            if page_start and v < page_start:
                raise ValueError("page_end must be >= page_start")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API compatibility."""
        return {
            "doc": self.doc,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "score": self.score
        }


class KnowledgeItem(BaseModel):
    """Agent knowledge item for Letta integration."""
    
    type: Literal["Entity", "Event", "Issue", "Fact"] = Field(..., description="Knowledge item type")
    label: str = Field(..., min_length=1, description="Knowledge item label")
    date: Optional[str] = Field(None, description="Associated date (ISO format)")
    actors: List[str] = Field(default_factory=list, description="Involved parties/actors")
    doc_refs: List[Dict[str, Any]] = Field(default_factory=list, description="Document references")
    support_snippet: Optional[str] = Field(None, max_length=300, description="Supporting text snippet")

    @field_validator('date')
    @classmethod 
    def validate_date_format(cls, v):
        """Validate date format if provided."""
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Date must be in ISO format")
        return v


class ChatRequest(BaseModel):
    """Chat request payload."""
    
    matter_id: str = Field(..., description="Matter ID for context")
    query: str = Field(..., min_length=1, description="User query")
    k: int = Field(default=8, ge=1, le=20, description="Number of chunks to retrieve")
    model: Optional[str] = Field(None, description="Model to use (defaults to active)")
    max_tokens: int = Field(default=900, ge=100, le=4000, description="Maximum tokens in response")


class CitationMetrics(BaseModel):
    """Citation quality metrics for API responses."""
    total_citations: int = Field(..., ge=0, description="Total number of citations")
    valid_citations: int = Field(..., ge=0, description="Number of valid citations")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="Citation coverage score")
    diversity_score: float = Field(..., ge=0.0, le=1.0, description="Citation diversity score")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Citation accuracy score")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Citation completeness score")


class QualityMetrics(BaseModel):
    """Response quality metrics for API responses."""
    citation_coverage: float = Field(..., ge=0.0, le=1.0, description="Citation coverage score")
    citation_accuracy: float = Field(..., ge=0.0, le=1.0, description="Citation accuracy score")
    citation_diversity: float = Field(..., ge=0.0, le=1.0, description="Citation diversity score")
    answer_completeness: float = Field(..., ge=0.0, le=1.0, description="Answer completeness score")
    content_coherence: float = Field(..., ge=0.0, le=1.0, description="Content coherence score")
    domain_specificity: float = Field(..., ge=0.0, le=1.0, description="Domain specificity score")
    source_diversity: float = Field(..., ge=0.0, le=1.0, description="Source diversity score")
    source_relevance: float = Field(..., ge=0.0, le=1.0, description="Source relevance score")
    source_recency: float = Field(..., ge=0.0, le=1.0, description="Source recency score")
    followup_relevance: float = Field(..., ge=0.0, le=1.0, description="Follow-up relevance score")
    followup_diversity: float = Field(..., ge=0.0, le=1.0, description="Follow-up diversity score")
    followup_actionability: float = Field(..., ge=0.0, le=1.0, description="Follow-up actionability score")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    response_length: int = Field(..., ge=0, description="Response length in characters")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    meets_minimum_standards: bool = Field(default=False, description="Whether response meets minimum quality standards")
    requires_regeneration: bool = Field(default=False, description="Whether response should be regenerated")
    quality_warnings: List[str] = Field(default_factory=list, description="Quality warning messages")


class FollowupSuggestion(BaseModel):
    """Enhanced follow-up suggestion with metadata."""
    question: str = Field(..., min_length=1, description="Follow-up question text")
    category: str = Field(..., description="Question category")
    priority: float = Field(..., ge=0.0, le=1.0, description="Priority score")
    reasoning: str = Field(..., description="Reasoning for suggestion")
    related_entities: List[str] = Field(default_factory=list, description="Related entities")
    requires_expert: bool = Field(default=False, description="Whether expert analysis is needed")


class ChatResponse(BaseModel):
    """Enhanced chat response payload with quality metrics."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    used_memory: List[KnowledgeItem] = Field(default_factory=list, description="Agent memory items used")
    
    # Advanced features (optional for backward compatibility)
    citation_metrics: Optional[CitationMetrics] = Field(None, description="Citation quality metrics")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Response quality metrics")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Processing time in seconds")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in response quality")
    quality_warnings: List[str] = Field(default_factory=list, description="Quality warning messages")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    
    # Enhanced follow-up suggestions
    enhanced_followups: List[FollowupSuggestion] = Field(default_factory=list, description="Enhanced follow-up suggestions")


class JobStatus(BaseModel):
    """Background job status."""
    
    id: str = Field(..., description="Job ID")
    status: Literal["queued", "running", "completed", "failed"] = Field(..., description="Job status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Progress (0-1)")
    detail: Optional[str] = Field(None, description="Status detail message")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentInfo(BaseModel):
    """Document information for UI display."""
    
    name: str = Field(..., description="Document filename")
    pages: int = Field(..., ge=0, description="Number of pages")
    chunks: int = Field(default=0, ge=0, description="Number of chunks created")
    ocr_status: Literal["none", "partial", "full"] = Field(default="none", description="OCR status")
    status: Literal["pending", "processing", "completed", "failed"] = Field(default="pending", description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class MatterSummary(BaseModel):
    """Matter summary for listing."""
    
    id: str = Field(..., description="Matter ID")
    name: str = Field(..., description="Matter name")
    slug: str = Field(..., description="Matter slug")
    created_at: datetime = Field(..., description="Creation timestamp")
    document_count: int = Field(default=0, ge=0, description="Number of documents")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CreateMatterRequest(BaseModel):
    """Request to create a new matter."""
    
    name: str = Field(..., min_length=1, max_length=200, description="Matter name")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate matter name."""
        if not v or not v.strip():
            raise ValueError("Matter name cannot be empty")
        return v.strip()


class CreateMatterResponse(BaseModel):
    """Response from creating a matter."""
    
    id: str = Field(..., description="Created matter ID")
    slug: str = Field(..., description="Created matter slug")
    paths: Dict[str, str] = Field(..., description="Matter directory paths")


class HistoricalQualityStats(BaseModel):
    """Historical quality statistics for a matter."""
    total_responses: int = Field(..., ge=0, description="Total number of responses")
    average_quality: float = Field(..., ge=0.0, le=1.0, description="Average quality score")
    quality_trend: float = Field(..., description="Quality trend (positive = improving)")
    best_quality_score: float = Field(..., ge=0.0, le=1.0, description="Best quality score achieved")
    worst_quality_score: float = Field(..., ge=0.0, le=1.0, description="Worst quality score")
    quality_consistency: float = Field(..., ge=0.0, description="Quality consistency (lower = more consistent)")
    citation_quality_avg: float = Field(..., ge=0.0, le=1.0, description="Average citation quality")
    content_quality_avg: float = Field(..., ge=0.0, le=1.0, description="Average content quality")
    source_quality_avg: float = Field(..., ge=0.0, le=1.0, description="Average source quality")
    followup_quality_avg: float = Field(..., ge=0.0, le=1.0, description="Average follow-up quality")
    first_response_date: datetime = Field(..., description="Date of first response")
    last_response_date: datetime = Field(..., description="Date of last response")
    quality_by_day: Dict[str, float] = Field(default_factory=dict, description="Quality scores by day")

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QualityInsights(BaseModel):
    """Quality insights response for a matter."""
    matter_id: str = Field(..., description="Matter ID")
    advanced_features_enabled: bool = Field(..., description="Whether advanced features are enabled")
    historical_stats: Optional[HistoricalQualityStats] = Field(None, description="Historical quality statistics")
    retrieval_stats: Dict[str, Any] = Field(default_factory=dict, description="Retrieval system statistics")
    quality_thresholds: Dict[str, float] = Field(default_factory=dict, description="Current quality thresholds")


class RetrievalWeights(BaseModel):
    """Configuration for hybrid retrieval scoring."""
    vector_similarity: float = Field(default=0.7, ge=0.0, le=1.0, description="Vector similarity weight")
    memory_relevance: float = Field(default=0.3, ge=0.0, le=1.0, description="Memory relevance weight")
    recency_boost: float = Field(default=0.1, ge=0.0, le=1.0, description="Recency boost weight")
    diversity_factor: float = Field(default=0.15, ge=0.0, le=1.0, description="Diversity factor weight")
    temporal_decay_days: int = Field(default=30, ge=1, le=365, description="Temporal decay period in days")


class CaliforniaStatute(BaseModel):
    """California statute reference."""
    code_type: Literal["PCC", "GC", "CC", "LC", "BPC"] = Field(..., description="Code type")
    section: str = Field(..., description="Section number")
    citation: str = Field(..., description="Full citation")
    description: Optional[str] = Field(None, description="Description of statute")
    context: str = Field(..., description="Context where found")


class CaliforniaDeadline(BaseModel):
    """California statutory deadline."""
    deadline_type: str = Field(..., description="Type of deadline")
    statute: str = Field(..., description="Governing statute")
    days: int = Field(..., ge=0, description="Days for deadline")
    trigger_event: str = Field(..., description="Event triggering deadline")
    deadline_date: Optional[datetime] = Field(None, description="Calculated deadline date")
    days_remaining: Optional[int] = Field(None, description="Days remaining")
    consequence: str = Field(..., description="Consequence of missing deadline")


class CaliforniaEntity(BaseModel):
    """California public entity."""
    name: str = Field(..., description="Entity name")
    entity_type: Literal["state", "county", "city", "district", "authority"] = Field(..., description="Entity type")
    abbreviation: Optional[str] = Field(None, description="Common abbreviation")
    special_requirements: List[str] = Field(default_factory=list, description="Special requirements")


class CaliforniaClaimData(BaseModel):
    """California construction claim data."""
    claim_type: Literal["differing_site_conditions", "delay", "changes", "payment", "termination"] = Field(..., description="Claim type")
    public_entity: Optional[CaliforniaEntity] = Field(None, description="Public entity involved")
    statutes: List[CaliforniaStatute] = Field(default_factory=list, description="Relevant statutes")
    deadlines: List[CaliforniaDeadline] = Field(default_factory=list, description="Applicable deadlines")
    notices_required: List[str] = Field(default_factory=list, description="Required notices")
    notices_served: List[Dict[str, Any]] = Field(default_factory=list, description="Notices already served")
    government_claim_filed: bool = Field(default=False, description="Whether government claim filed")
    government_claim_date: Optional[datetime] = Field(None, description="Date government claim filed")
    prevailing_wage_compliance: bool = Field(default=False, description="Prevailing wage compliance status")
    bond_claims: List[Dict[str, Any]] = Field(default_factory=list, description="Bond claims if any")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CaliforniaValidationResult(BaseModel):
    """California claim validation result."""
    is_valid: bool = Field(..., description="Whether claim is valid")
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Compliance score")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Validation errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Validation warnings")
    missing_items: List[str] = Field(default_factory=list, description="Missing required items")
    deadline_risks: List[CaliforniaDeadline] = Field(default_factory=list, description="Upcoming deadline risks")
    recommendations: List[str] = Field(default_factory=list, description="Compliance recommendations")