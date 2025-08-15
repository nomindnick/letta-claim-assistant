"""
Unit tests for Matter management functionality.

Tests matter creation, listing, switching, filesystem operations,
and configuration management.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import json

from app.matters import MatterManager
from app.models import Matter, MatterPaths
from app.settings import settings


@pytest.fixture
def temp_data_root():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def matter_manager(temp_data_root):
    """Create MatterManager with temporary data root."""
    # Create a mock config object
    mock_config = MagicMock()
    mock_config.data_root = temp_data_root
    mock_config.embeddings_model = "test-embed-model"
    mock_config.llm_model = "test-llm-model"
    
    # Mock the global_config property
    with patch.object(type(settings), 'global_config', new_callable=lambda: property(lambda self: mock_config)):
        return MatterManager()


class TestMatterManager:
    """Test cases for MatterManager class."""
    
    def test_init_creates_data_root(self, temp_data_root):
        """Test that MatterManager creates data root directory."""
        mock_config = MagicMock()
        mock_config.data_root = temp_data_root / "test_root"
        
        with patch.object(type(settings), 'global_config', new_callable=lambda: property(lambda self: mock_config)):
            manager = MatterManager()
            assert (temp_data_root / "test_root").exists()
            assert manager.data_root == temp_data_root / "test_root"
    
    def test_create_matter_valid_name(self, matter_manager):
        """Test creating matter with valid name."""
        matter = matter_manager.create_matter("Test Construction Claim")
        
        # Verify matter properties
        assert matter.name == "Test Construction Claim"
        assert matter.slug == "test-construction-claim"
        assert isinstance(matter.id, str)
        assert len(matter.id) == 36  # UUID4 length
        assert isinstance(matter.created_at, datetime)
        assert matter.embedding_model == "nomic-embed-text"
        assert matter.generation_model == "gpt-oss:20b"
        
        # Verify filesystem structure
        expected_root = matter_manager.data_root / f"Matter_{matter.slug}"
        assert matter.paths.root == expected_root
        assert matter.paths.root.exists()
        assert matter.paths.docs.exists()
        assert matter.paths.docs_ocr.exists()
        assert matter.paths.parsed.exists()
        assert matter.paths.vectors.exists()
        assert (matter.paths.vectors / "chroma").exists()
        assert matter.paths.knowledge.exists()
        assert (matter.paths.knowledge / "letta_state").exists()
        assert matter.paths.chat.exists()
        assert matter.paths.logs.exists()
        
        # Verify config file exists
        config_file = matter.paths.root / "config.json"
        assert config_file.exists()
        
        with open(config_file) as f:
            config = json.load(f)
            assert config["id"] == matter.id
            assert config["name"] == matter.name
            assert config["slug"] == matter.slug
    
    def test_create_matter_invalid_names(self, matter_manager):
        """Test creating matter with invalid names."""
        with pytest.raises(ValueError, match="Matter name cannot be empty"):
            matter_manager.create_matter("")
        
        with pytest.raises(ValueError, match="Matter name cannot be empty"):
            matter_manager.create_matter("   ")
        
        with pytest.raises(ValueError, match="Matter name cannot be empty"):
            matter_manager.create_matter(None)
    
    def test_create_matter_duplicate_names(self, matter_manager):
        """Test creating matters with duplicate names generates unique slugs."""
        matter1 = matter_manager.create_matter("Duplicate Name")
        matter2 = matter_manager.create_matter("Duplicate Name")
        
        assert matter1.name == matter2.name
        assert matter1.slug == "duplicate-name"
        assert matter2.slug == "duplicate-name-1"
        assert matter1.id != matter2.id
        
        # Both directories should exist
        assert matter1.paths.root.exists()
        assert matter2.paths.root.exists()
    
    def test_slug_generation(self, matter_manager):
        """Test slug generation from various matter names."""
        test_cases = [
            ("Simple Name", "simple-name"),
            ("Name with CAPS", "name-with-caps"),
            ("Name-with-hyphens", "name-with-hyphens"),
            ("Name with 123 numbers", "name-with-123-numbers"),
            ("Name with !@#$ special chars", "name-with-special-chars"),
            ("   Whitespace   Around   ", "whitespace-around"),
            ("Multiple---Hyphens", "multiple-hyphens"),
            ("A" * 60, "a" * 50),  # Truncation test
            ("", "matter"),  # Empty fallback
        ]
        
        for name, expected_slug in test_cases:
            if name:  # Skip empty name as it will raise ValueError
                slug = matter_manager._create_matter_slug(name)
                assert slug == expected_slug
    
    def test_list_matters_empty(self, matter_manager):
        """Test listing matters when none exist."""
        matters = matter_manager.list_matters()
        assert matters == []
    
    def test_list_matters_multiple(self, matter_manager):
        """Test listing multiple matters."""
        # Create several matters
        matter1 = matter_manager.create_matter("First Matter")
        matter2 = matter_manager.create_matter("Second Matter")
        matter3 = matter_manager.create_matter("Third Matter")
        
        matters = matter_manager.list_matters()
        assert len(matters) == 3
        
        # Should be sorted by creation date (newest first)
        matter_ids = [m.id for m in matters]
        expected_ids = [matter3.id, matter2.id, matter1.id]  # Reverse chronological
        assert matter_ids == expected_ids
    
    def test_list_matters_with_invalid_config(self, matter_manager, temp_data_root):
        """Test listing matters handles invalid configuration files."""
        # Create valid matter first
        valid_matter = matter_manager.create_matter("Valid Matter")
        
        # Create directory with invalid config
        invalid_dir = temp_data_root / "Matter_invalid"
        invalid_dir.mkdir()
        
        # Write invalid JSON
        config_file = invalid_dir / "config.json"
        config_file.write_text("invalid json {")
        
        # Should only return valid matter
        matters = matter_manager.list_matters()
        assert len(matters) == 1
        assert matters[0].id == valid_matter.id
    
    def test_switch_matter_valid_id(self, matter_manager):
        """Test switching to a valid matter."""
        matter1 = matter_manager.create_matter("First Matter")
        matter2 = matter_manager.create_matter("Second Matter")
        
        # Initially no active matter
        assert matter_manager.get_active_matter() is None
        
        # Switch to first matter
        result = matter_manager.switch_matter(matter1.id)
        assert result.id == matter1.id
        assert matter_manager.get_active_matter().id == matter1.id
        
        # Switch to second matter
        result = matter_manager.switch_matter(matter2.id)
        assert result.id == matter2.id
        assert matter_manager.get_active_matter().id == matter2.id
    
    def test_switch_matter_invalid_id(self, matter_manager):
        """Test switching to invalid matter ID raises error."""
        with pytest.raises(ValueError, match="Matter not found"):
            matter_manager.switch_matter("nonexistent-id")
    
    def test_get_matter_by_id(self, matter_manager):
        """Test retrieving specific matter by ID."""
        matter = matter_manager.create_matter("Test Matter")
        
        retrieved = matter_manager.get_matter_by_id(matter.id)
        assert retrieved is not None
        assert retrieved.id == matter.id
        assert retrieved.name == matter.name
        
        # Test with invalid ID
        not_found = matter_manager.get_matter_by_id("invalid-id")
        assert not_found is None
    
    def test_list_matter_summaries(self, matter_manager):
        """Test listing matter summaries with stats."""
        matter = matter_manager.create_matter("Summary Test Matter")
        
        # Add some test documents to docs directory
        docs_dir = matter.paths.docs
        (docs_dir / "doc1.pdf").touch()
        (docs_dir / "doc2.pdf").touch()
        
        # Add test chat history
        chat_history = matter.paths.chat / "history.jsonl"
        chat_history.write_text('{"test": "data"}\n')
        
        summaries = matter_manager.list_matter_summaries()
        assert len(summaries) == 1
        
        summary = summaries[0]
        assert summary.id == matter.id
        assert summary.name == matter.name
        assert summary.slug == matter.slug
        assert summary.document_count == 2
        assert summary.last_activity is not None
        assert isinstance(summary.last_activity, datetime)
    
    def test_thread_safety(self, matter_manager):
        """Test thread safety of matter switching."""
        import threading
        import time
        
        matter1 = matter_manager.create_matter("Thread Test 1")
        matter2 = matter_manager.create_matter("Thread Test 2")
        
        results = []
        errors = []
        
        def switch_matter_worker(matter_id):
            try:
                time.sleep(0.01)  # Small delay to encourage race conditions
                result = matter_manager.switch_matter(matter_id)
                results.append(result.id)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads switching between matters
        threads = []
        for _ in range(10):
            t1 = threading.Thread(target=switch_matter_worker, args=(matter1.id,))
            t2 = threading.Thread(target=switch_matter_worker, args=(matter2.id,))
            threads.extend([t1, t2])
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have no errors and results should be valid matter IDs
        assert errors == []
        assert len(results) == 20
        assert all(r in [matter1.id, matter2.id] for r in results)
    
    def test_filesystem_creation_error_handling(self, matter_manager):
        """Test handling of filesystem creation errors."""
        # Mock a permission error during directory creation
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                matter_manager.create_matter("Permission Test")


class TestMatterPaths:
    """Test cases for MatterPaths model."""
    
    def test_from_root_creates_all_paths(self, temp_data_root):
        """Test MatterPaths.from_root creates all expected paths."""
        root = temp_data_root / "test_matter"
        paths = MatterPaths.from_root(root)
        
        assert paths.root == root
        assert paths.docs == root / "docs"
        assert paths.docs_ocr == root / "docs_ocr"
        assert paths.parsed == root / "parsed"
        assert paths.vectors == root / "vectors"
        assert paths.knowledge == root / "knowledge"
        assert paths.chat == root / "chat"
        assert paths.logs == root / "logs"
    
    def test_json_serialization(self, temp_data_root):
        """Test MatterPaths JSON serialization."""
        root = temp_data_root / "test_matter"
        paths = MatterPaths.from_root(root)
        
        # Should be able to serialize to JSON
        paths_dict = paths.model_dump()
        assert all(isinstance(v, str) for v in paths_dict.values())
        
        # JSON encoding should convert paths to strings
        import json
        json_str = paths.model_dump_json()
        loaded = json.loads(json_str)
        assert all(isinstance(v, str) for v in loaded.values())


class TestMatter:
    """Test cases for Matter model."""
    
    def test_matter_validation(self, temp_data_root):
        """Test Matter model validation."""
        paths = MatterPaths.from_root(temp_data_root)
        
        # Valid matter
        matter = Matter(
            id="test-id",
            name="Test Matter",
            slug="test-matter",
            paths=paths
        )
        assert matter.name == "Test Matter"
        assert matter.slug == "test-matter"
        
        # Invalid name
        with pytest.raises(ValueError):
            Matter(
                id="test-id",
                name="",
                slug="test-matter",
                paths=paths
            )
        
        # Invalid slug
        with pytest.raises(ValueError):
            Matter(
                id="test-id",
                name="Test Matter",
                slug="Invalid_Slug!",
                paths=paths
            )
    
    def test_config_serialization(self, temp_data_root):
        """Test Matter configuration serialization."""
        paths = MatterPaths.from_root(temp_data_root)
        matter = Matter(
            id="test-id",
            name="Test Matter",
            slug="test-matter",
            paths=paths,
            embedding_model="test-embed",
            generation_model="test-gen"
        )
        
        config_dict = matter.to_config_dict()
        assert config_dict["id"] == "test-id"
        assert config_dict["name"] == "Test Matter"
        assert config_dict["slug"] == "test-matter"
        assert config_dict["embedding_model"] == "test-embed"
        assert config_dict["generation_model"] == "test-gen"
        assert "created_at" in config_dict
        
        # Test round-trip
        restored = Matter.from_config_dict(config_dict, temp_data_root)
        assert restored.id == matter.id
        assert restored.name == matter.name
        assert restored.slug == matter.slug
        assert restored.embedding_model == matter.embedding_model
        assert restored.generation_model == matter.generation_model