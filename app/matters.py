"""
Matter management and filesystem operations.

Handles creation, switching, and listing of Matters with proper filesystem
directory structure and configuration management.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import re
import json
import threading

from .models import Matter, MatterPaths, MatterSummary
from .logging_conf import get_logger, bind_matter_context
from .settings import settings

logger = get_logger(__name__)


# Models are now imported from models.py module


class MatterManager:
    """Manages Matter operations and filesystem structure."""
    
    def __init__(self):
        self.data_root = settings.global_config.data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._current_matter: Optional[Matter] = None
        self._lock = threading.Lock()  # Thread safety for matter switching
    
    def create_matter(self, name: str) -> Matter:
        """
        Create a new Matter with filesystem structure.
        
        Args:
            name: Human-readable matter name
            
        Returns:
            Created Matter instance
            
        Raises:
            ValueError: If matter name is invalid or already exists
        """
        if not name or not name.strip():
            raise ValueError("Matter name cannot be empty")
        
        name = name.strip()
        
        # Generate unique slug
        base_slug = self._create_matter_slug(name)
        slug = base_slug
        counter = 1
        
        # Ensure slug uniqueness
        while (self.data_root / f"Matter_{slug}").exists():
            slug = f"{base_slug}-{counter}"
            counter += 1
        
        # Create matter ID
        matter_id = str(uuid.uuid4())
        
        # Create filesystem structure
        matter_root = self.data_root / f"Matter_{slug}"
        paths = self._create_matter_filesystem(matter_root)
        
        # Create Matter instance
        matter = Matter(
            id=matter_id,
            name=name,
            slug=slug,
            created_at=datetime.now(),
            embedding_model=settings.global_config.embeddings_model,
            generation_model=settings.global_config.llm_model,
            paths=paths
        )
        
        # Save matter configuration
        config_dict = matter.to_config_dict()
        settings.save_matter_config(matter_root, config_dict)
        
        # Initialize Letta agent for the new matter
        try:
            from .letta_adapter import LettaAdapter
            letta_adapter = LettaAdapter(
                matter_path=matter_root,
                matter_name=matter.name,
                matter_id=matter.id
            )
            logger.info("Letta agent initialized for new matter", matter_id=matter.id)
        except Exception as e:
            logger.warning("Failed to initialize Letta agent for new matter", error=str(e))
        
        logger.info(
            "Matter created successfully",
            matter_id=matter.id,
            matter_name=matter.name,
            matter_slug=matter.slug,
            matter_path=str(matter_root)
        )
        
        return matter
    
    def list_matters(self) -> List[Matter]:
        """
        List all available Matters.
        
        Returns:
            List of Matter instances sorted by creation date (newest first)
        """
        matters = []
        
        if not self.data_root.exists():
            return matters
        
        # Find all matter directories
        for matter_dir in self.data_root.iterdir():
            if matter_dir.is_dir() and matter_dir.name.startswith("Matter_"):
                try:
                    config_dict = settings.load_matter_config(matter_dir)
                    if config_dict:  # Only include matters with valid config
                        matter = Matter.from_config_dict(config_dict, matter_dir)
                        matters.append(matter)
                except Exception as e:
                    logger.warning(
                        "Failed to load matter configuration",
                        matter_path=str(matter_dir),
                        error=str(e)
                    )
        
        # Sort by creation date (newest first)
        matters.sort(key=lambda m: m.created_at, reverse=True)
        
        logger.debug(f"Listed {len(matters)} matters")
        return matters
    
    def switch_matter(self, matter_id: str) -> Matter:
        """
        Switch active Matter context.
        
        Args:
            matter_id: Matter ID to switch to
            
        Returns:
            The activated Matter instance
            
        Raises:
            ValueError: If matter_id is not found
        """
        with self._lock:  # Thread-safe matter switching
            matters = self.list_matters()
            target_matter = None
            
            for matter in matters:
                if matter.id == matter_id:
                    target_matter = matter
                    break
            
            if target_matter is None:
                raise ValueError(f"Matter not found: {matter_id}")
            
            # Update current matter
            old_matter_id = self._current_matter.id if self._current_matter else None
            self._current_matter = target_matter
            
            logger.info(
                "Matter switched",
                old_matter_id=old_matter_id,
                new_matter_id=target_matter.id,
                new_matter_name=target_matter.name
            )
            
            return target_matter
    
    def get_active_matter(self) -> Optional[Matter]:
        """Get currently active Matter."""
        return self._current_matter
    
    def _create_matter_slug(self, name: str) -> str:
        """Create URL-safe slug from matter name."""
        # Convert to lowercase
        slug = name.lower()
        
        # Replace spaces and special characters with hyphens
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Collapse multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        
        # Ensure slug is not empty
        if not slug:
            slug = "matter"
        
        # Truncate if too long
        if len(slug) > 50:
            slug = slug[:50].rstrip('-')
        
        return slug
    
    def _create_matter_filesystem(self, matter_root: Path) -> MatterPaths:
        """Create complete Matter directory structure."""
        # Create MatterPaths instance
        paths = MatterPaths.from_root(matter_root)
        
        # Create all directories
        directories_to_create = [
            paths.root,
            paths.docs,
            paths.docs_ocr, 
            paths.parsed,
            paths.vectors,
            paths.vectors / "chroma",  # Chroma collection directory
            paths.knowledge,
            paths.knowledge / "letta_state",  # Letta agent state
            paths.chat,
            paths.logs
        ]
        
        for directory in directories_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.error(
                    "Failed to create directory",
                    directory=str(directory),
                    error=str(e)
                )
                raise
        
        return paths


    def list_matter_summaries(self) -> List[MatterSummary]:
        """
        List matter summaries for UI display.
        
        Returns:
            List of MatterSummary instances with basic info and stats
        """
        summaries = []
        
        for matter in self.list_matters():
            # Count documents
            doc_count = 0
            if matter.paths.docs.exists():
                doc_count = len([f for f in matter.paths.docs.iterdir() if f.is_file()])
            
            # Get last activity from chat history
            last_activity = None
            chat_history_file = matter.paths.chat / "history.jsonl"
            if chat_history_file.exists():
                try:
                    last_activity = datetime.fromtimestamp(chat_history_file.stat().st_mtime)
                except Exception:
                    pass
            
            summary = MatterSummary(
                id=matter.id,
                name=matter.name,
                slug=matter.slug,
                created_at=matter.created_at,
                document_count=doc_count,
                last_activity=last_activity
            )
            summaries.append(summary)
        
        return summaries
    
    def get_matter_by_id(self, matter_id: str) -> Optional[Matter]:
        """
        Get a specific matter by ID.
        
        Args:
            matter_id: Matter ID to retrieve
            
        Returns:
            Matter instance or None if not found
        """
        matters = self.list_matters()
        for matter in matters:
            if matter.id == matter_id:
                return matter
        return None
    
    def get_matter_documents(self, matter: Matter) -> List[Dict[str, Any]]:
        """
        Get document information for a matter.
        
        Args:
            matter: Matter to get documents for
            
        Returns:
            List of document info dictionaries
        """
        documents = []
        
        # Check docs directory for original files
        if matter.paths.docs.exists():
            for file_path in matter.paths.docs.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == '.pdf':
                    doc_info = self._get_document_info(matter, file_path)
                    documents.append(doc_info)
        
        return sorted(documents, key=lambda x: x['name'])
    
    def _get_document_info(self, matter: Matter, doc_path: Path) -> Dict[str, Any]:
        """
        Get detailed information about a document.
        
        Args:
            matter: Matter the document belongs to
            doc_path: Path to the document file
            
        Returns:
            Document information dictionary
        """
        import fitz  # PyMuPDF for page counting
        
        doc_name = doc_path.name
        doc_info = {
            "name": doc_name,
            "pages": 0,
            "chunks": 0,
            "ocr_status": "none",
            "status": "pending",
            "error_message": None,
            "file_size": doc_path.stat().st_size,
            "created_at": datetime.fromtimestamp(doc_path.stat().st_ctime).isoformat()
        }
        
        try:
            # Get page count
            with fitz.open(doc_path) as pdf_doc:
                doc_info["pages"] = pdf_doc.page_count
        except Exception as e:
            logger.warning("Could not read PDF for page count", error=str(e), doc=doc_name)
            doc_info["error_message"] = f"Could not read PDF: {str(e)}"
            doc_info["status"] = "failed"
            return doc_info
        
        # Check if OCR version exists
        ocr_path = matter.paths.docs_ocr / f"{doc_path.stem}.ocr.pdf"
        if ocr_path.exists():
            doc_info["ocr_status"] = "full"  # Simplified - would need analysis for partial
        
        # Check if parsed version exists
        parsed_path = matter.paths.parsed / f"{doc_path.stem}.jsonl"
        if parsed_path.exists():
            # Count chunks in parsed file
            try:
                chunk_count = 0
                with open(parsed_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            chunk_count += 1
                doc_info["chunks"] = chunk_count
                doc_info["status"] = "completed" if chunk_count > 0 else "processing"
            except Exception as e:
                logger.warning("Could not count chunks", error=str(e), doc=doc_name)
        
        # Check if embeddings exist in vector store
        try:
            from .vectors import VectorStore
            vector_store = VectorStore(matter.paths.vectors / "chroma")
            # This would require querying the vector store - simplified for now
            if doc_info["status"] == "completed":
                doc_info["status"] = "completed"
        except Exception:
            pass  # Vector store not available
        
        return doc_info


# Global matter manager instance
matter_manager = MatterManager()