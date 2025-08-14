"""
Pydantic data models for the Letta Construction Claim Assistant.

Defines core data structures for Matters, configuration, and API communication
with proper validation and serialization.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
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


class ChatResponse(BaseModel):
    """Chat response payload."""
    
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks used")
    followups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    used_memory: List[KnowledgeItem] = Field(default_factory=list, description="Agent memory items used")


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