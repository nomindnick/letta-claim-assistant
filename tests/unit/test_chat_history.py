"""
Unit tests for chat history functionality.

Tests message persistence and retrieval, validates matter-specific isolation,
tests timestamp formatting, and handles concurrent access scenarios.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json
import tempfile
from datetime import datetime
import threading
import time

from app.chat_history import ChatHistoryManager, ChatMessage, ChatHistoryError
from app.models import Matter, MatterPaths


class TestChatHistoryManager:
    """Test suite for ChatHistoryManager."""
    
    @pytest.fixture
    def temp_chat_dir(self):
        """Create temporary chat directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chat_dir = Path(temp_dir) / "chat"
            chat_dir.mkdir()
            yield chat_dir
    
    @pytest.fixture
    def test_matter(self, temp_chat_dir):
        """Create test matter with chat directory."""
        matter_root = temp_chat_dir.parent
        paths = MatterPaths.from_root(matter_root)
        
        return Matter(
            id="test-matter-001",
            name="Test Matter",
            slug="test-matter",
            embedding_model="nomic-embed-text",
            generation_model="gpt-oss:20b",
            paths=paths
        )
    
    @pytest.fixture
    def chat_manager(self, test_matter):
        """Create ChatHistoryManager instance."""
        return ChatHistoryManager(test_matter)
    
    @pytest.fixture
    def sample_messages(self):
        """Sample chat messages for testing."""
        return [
            ChatMessage(
                role="user",
                content="What caused the dry well failure?",
                timestamp=datetime(2025, 1, 21, 10, 0, 0),
                metadata={"query_id": "q1"}
            ),
            ChatMessage(
                role="assistant",
                content="The dry well failure was caused by improper installation according to [contract.pdf p.12].",
                timestamp=datetime(2025, 1, 21, 10, 0, 5),
                metadata={
                    "sources": [{"doc": "contract.pdf", "page": 12}],
                    "query_id": "q1"
                }
            ),
            ChatMessage(
                role="user",
                content="Who was responsible for the installation?",
                timestamp=datetime(2025, 1, 21, 10, 1, 0),
                metadata={"query_id": "q2"}
            )
        ]
    
    @pytest.mark.unit
    def test_chat_manager_initialization(self, temp_chat_dir):
        """Test ChatHistoryManager initialization."""
        manager = ChatHistoryManager(temp_chat_dir)
        assert manager.chat_dir == temp_chat_dir
        assert manager.history_file == temp_chat_dir / "history.jsonl"
    
    @pytest.mark.unit
    def test_add_message_creates_file(self, chat_manager, temp_chat_dir):
        """Test that adding first message creates history file."""
        message = ChatMessage(
            role="user",
            content="Test message",
            timestamp=datetime.now()
        )
        
        chat_manager.add_message(message)
        
        # History file should be created
        assert chat_manager.history_file.exists()
        
        # Should contain one message
        messages = chat_manager.get_history()
        assert len(messages) == 1
        assert messages[0].content == "Test message"
    
    @pytest.mark.unit
    def test_add_multiple_messages(self, chat_manager, sample_messages):
        """Test adding multiple messages."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Should have all messages
        messages = chat_manager.get_history()
        assert len(messages) == 3
        
        # Messages should be in order
        assert messages[0].content == "What caused the dry well failure?"
        assert messages[1].role == "assistant"
        assert messages[2].content == "Who was responsible for the installation?"
    
    @pytest.mark.unit
    def test_get_history_empty(self, chat_manager):
        """Test getting history when no messages exist."""
        messages = chat_manager.get_history()
        assert messages == []
    
    @pytest.mark.unit
    def test_get_history_with_limit(self, chat_manager, sample_messages):
        """Test getting history with message limit."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Get last 2 messages
        recent_messages = chat_manager.get_history(limit=2)
        assert len(recent_messages) == 2
        
        # Should be the most recent messages
        assert recent_messages[0].content == "The dry well failure was caused by improper installation according to [contract.pdf p.12]."
        assert recent_messages[1].content == "Who was responsible for the installation?"
    
    @pytest.mark.unit
    def test_get_history_with_since_timestamp(self, chat_manager, sample_messages):
        """Test getting history since a specific timestamp."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Get messages since 10:00:30
        since_time = datetime(2025, 1, 21, 10, 0, 30)
        recent_messages = chat_manager.get_history(since=since_time)
        
        # Should only get the last message
        assert len(recent_messages) == 1
        assert recent_messages[0].content == "Who was responsible for the installation?"
    
    @pytest.mark.unit
    def test_clear_history(self, chat_manager, sample_messages):
        """Test clearing chat history."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Verify messages exist
        assert len(chat_manager.get_history()) == 3
        
        # Clear history
        chat_manager.clear_history()
        
        # Should be empty
        assert len(chat_manager.get_history()) == 0
        
        # File should not exist
        assert not chat_manager.history_file.exists()
    
    @pytest.mark.unit
    def test_message_serialization(self, chat_manager):
        """Test message serialization to JSON."""
        message = ChatMessage(
            role="assistant",
            content="Response with metadata",
            timestamp=datetime(2025, 1, 21, 12, 0, 0),
            metadata={
                "sources": [{"doc": "spec.pdf", "page": 5}],
                "followups": ["What are the requirements?"],
                "processing_time": 2.5
            }
        )
        
        chat_manager.add_message(message)
        
        # Read back and verify serialization
        messages = chat_manager.get_history()
        restored_message = messages[0]
        
        assert restored_message.role == "assistant"
        assert restored_message.content == "Response with metadata"
        assert restored_message.metadata["sources"][0]["doc"] == "spec.pdf"
        assert restored_message.metadata["followups"][0] == "What are the requirements?"
        assert restored_message.metadata["processing_time"] == 2.5
    
    @pytest.mark.unit
    def test_timestamp_handling(self, chat_manager):
        """Test timestamp parsing and formatting."""
        # Message with specific timestamp
        timestamp = datetime(2025, 1, 21, 15, 30, 45)
        message = ChatMessage(
            role="user",
            content="Timestamp test",
            timestamp=timestamp
        )
        
        chat_manager.add_message(message)
        
        # Read back and verify timestamp
        messages = chat_manager.get_history()
        restored_message = messages[0]
        
        assert restored_message.timestamp == timestamp
    
    @pytest.mark.unit
    def test_concurrent_access(self, chat_manager):
        """Test concurrent access to chat history."""
        messages_added = []
        errors = []
        
        def add_messages_worker(worker_id):
            try:
                for i in range(5):
                    message = ChatMessage(
                        role="user",
                        content=f"Message {i} from worker {worker_id}",
                        timestamp=datetime.now()
                    )
                    chat_manager.add_message(message)
                    messages_added.append(f"worker_{worker_id}_msg_{i}")
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(target=add_messages_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have all messages
        assert len(messages_added) == 15
        
        # History should contain all messages
        history = chat_manager.get_history()
        assert len(history) == 15
    
    @pytest.mark.unit
    def test_corrupted_history_file_recovery(self, chat_manager, temp_chat_dir):
        """Test recovery from corrupted history file."""
        # Create corrupted history file
        history_file = temp_chat_dir / "history.jsonl"
        with open(history_file, 'w') as f:
            f.write("invalid json\n")
            f.write('{"valid": "json"}\n')
            f.write("another invalid line\n")
        
        # Should handle corruption gracefully
        with pytest.raises(ChatHistoryError):
            chat_manager.get_history()
    
    @pytest.mark.unit
    def test_backup_and_restore(self, chat_manager, sample_messages):
        """Test backup and restore functionality."""
        # Add messages
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Create backup
        backup_path = chat_manager.chat_dir / "backup.jsonl"
        chat_manager.backup_history(backup_path)
        
        assert backup_path.exists()
        
        # Clear history
        chat_manager.clear_history()
        assert len(chat_manager.get_history()) == 0
        
        # Restore from backup
        chat_manager.restore_history(backup_path)
        
        # Should have original messages
        restored_messages = chat_manager.get_history()
        assert len(restored_messages) == 3
        assert restored_messages[0].content == sample_messages[0].content
    
    @pytest.mark.unit
    def test_search_messages(self, chat_manager, sample_messages):
        """Test searching through message history."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Search for specific content
        results = chat_manager.search_messages("dry well")
        assert len(results) == 2  # User question and assistant answer
        
        # Search for role-specific messages
        user_messages = chat_manager.search_messages("", role="user")
        assert len(user_messages) == 2
        assert all(msg.role == "user" for msg in user_messages)
        
        # Search with no matches
        no_results = chat_manager.search_messages("nonexistent content")
        assert len(no_results) == 0
    
    @pytest.mark.unit
    def test_message_statistics(self, chat_manager, sample_messages):
        """Test getting message statistics."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        stats = chat_manager.get_statistics()
        
        assert stats["total_messages"] == 3
        assert stats["user_messages"] == 2
        assert stats["assistant_messages"] == 1
        assert "first_message_time" in stats
        assert "last_message_time" in stats
        assert "average_response_time" in stats
    
    @pytest.mark.unit
    def test_export_import_json(self, chat_manager, sample_messages, temp_chat_dir):
        """Test exporting and importing history as JSON."""
        for message in sample_messages:
            chat_manager.add_message(message)
        
        # Export to JSON
        export_path = temp_chat_dir / "export.json"
        chat_manager.export_history(export_path, format="json")
        
        assert export_path.exists()
        
        # Verify export content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data["messages"]) == 3
        assert exported_data["matter_info"]["id"] == "test-matter-001"
        
        # Clear and import
        chat_manager.clear_history()
        chat_manager.import_history(export_path, format="json")
        
        # Should have all messages back
        imported_messages = chat_manager.get_history()
        assert len(imported_messages) == 3
    
    @pytest.mark.unit
    def test_message_validation(self, chat_manager):
        """Test message validation."""
        # Valid message
        valid_message = ChatMessage(
            role="user",
            content="Valid message",
            timestamp=datetime.now()
        )
        chat_manager.add_message(valid_message)  # Should not raise
        
        # Invalid role
        with pytest.raises(ChatHistoryError):
            invalid_message = ChatMessage(
                role="invalid_role",
                content="Message with invalid role",
                timestamp=datetime.now()
            )
            chat_manager.add_message(invalid_message)
        
        # Empty content
        with pytest.raises(ChatHistoryError):
            empty_message = ChatMessage(
                role="user",
                content="",
                timestamp=datetime.now()
            )
            chat_manager.add_message(empty_message)
    
    @pytest.mark.unit
    def test_history_file_permissions(self, chat_manager, temp_chat_dir):
        """Test history file permissions and access."""
        message = ChatMessage(
            role="user",
            content="Permission test",
            timestamp=datetime.now()
        )
        
        chat_manager.add_message(message)
        
        # History file should be readable and writable
        history_file = chat_manager.history_file
        assert history_file.exists()
        assert history_file.is_file()
        
        # Should be able to read the file
        with open(history_file, 'r') as f:
            content = f.read()
            assert "Permission test" in content
    
    @pytest.mark.unit
    def test_matter_isolation(self, temp_chat_dir):
        """Test that different matters have isolated chat histories."""
        # Create two different matters
        matter1_root = temp_chat_dir / "matter1"
        matter1_root.mkdir()
        matter1_paths = MatterPaths.from_root(matter1_root)
        matter1_paths.chat.mkdir(parents=True)
        
        matter2_root = temp_chat_dir / "matter2"
        matter2_root.mkdir()
        matter2_paths = MatterPaths.from_root(matter2_root)
        matter2_paths.chat.mkdir(parents=True)
        
        manager1 = ChatHistoryManager(matter1_paths.chat)
        manager2 = ChatHistoryManager(matter2_paths.chat)
        
        # Add different messages to each
        manager1.add_message(ChatMessage(
            role="user",
            content="Matter 1 message",
            timestamp=datetime.now()
        ))
        
        manager2.add_message(ChatMessage(
            role="user",
            content="Matter 2 message",
            timestamp=datetime.now()
        ))
        
        # Histories should be separate
        history1 = manager1.get_history()
        history2 = manager2.get_history()
        
        assert len(history1) == 1
        assert len(history2) == 1
        assert history1[0].content == "Matter 1 message"
        assert history2[0].content == "Matter 2 message"
    
    @pytest.mark.unit
    def test_large_history_performance(self, chat_manager):
        """Test performance with large chat history."""
        import time
        
        # Add many messages
        start_time = time.time()
        for i in range(1000):
            message = ChatMessage(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i} with some content to make it realistic",
                timestamp=datetime.now()
            )
            chat_manager.add_message(message)
        
        add_time = time.time() - start_time
        
        # Retrieve all messages
        start_time = time.time()
        all_messages = chat_manager.get_history()
        retrieve_time = time.time() - start_time
        
        assert len(all_messages) == 1000
        
        # Performance should be reasonable
        assert add_time < 10.0  # Less than 10 seconds to add 1000 messages
        assert retrieve_time < 2.0  # Less than 2 seconds to retrieve 1000 messages
    
    @pytest.mark.unit
    def test_chat_message_model(self):
        """Test ChatMessage model validation and serialization."""
        # Valid message
        message = ChatMessage(
            role="assistant",
            content="Test response",
            timestamp=datetime(2025, 1, 21, 10, 0, 0),
            metadata={"key": "value"}
        )
        
        assert message.role == "assistant"
        assert message.content == "Test response"
        assert message.metadata["key"] == "value"
        
        # Convert to dict
        message_dict = message.to_dict()
        assert message_dict["role"] == "assistant"
        assert message_dict["content"] == "Test response"
        assert "timestamp" in message_dict
        
        # Create from dict
        restored_message = ChatMessage.from_dict(message_dict)
        assert restored_message.role == message.role
        assert restored_message.content == message.content
        assert restored_message.timestamp == message.timestamp