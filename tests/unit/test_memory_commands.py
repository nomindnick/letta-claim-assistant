"""
Unit tests for memory command parsing and processing.

Tests natural language command detection, parsing accuracy,
and command execution logic.
"""

import pytest
from app.memory_commands import (
    MemoryCommand, MemoryAction, MemoryCommandParser,
    parse_memory_command
)


class TestMemoryCommandParser:
    """Test memory command parsing functionality."""
    
    def test_parse_remember_commands(self):
        """Test parsing of remember/note commands."""
        test_cases = [
            ("remember that the deadline is March 15th", MemoryAction.REMEMBER, "the deadline is March 15th"),
            ("Please remember the contractor is ABC Corp", MemoryAction.REMEMBER, "the contractor is ABC Corp"),
            ("keep in mind that payment is due monthly", MemoryAction.REMEMBER, "payment is due monthly"),
            ("don't forget the meeting is at 3pm", MemoryAction.REMEMBER, "the meeting is at 3pm"),
            ("note that the budget is $500,000", MemoryAction.REMEMBER, "the budget is $500,000"),
            ("for future reference, the case number is 12345", MemoryAction.REMEMBER, "the case number is 12345"),
        ]
        
        for input_text, expected_action, expected_content in test_cases:
            command = parse_memory_command(input_text)
            assert command is not None, f"Failed to parse: {input_text}"
            assert command.action == expected_action
            assert command.content == expected_content
            assert command.confidence >= 0.7
    
    def test_parse_forget_commands(self):
        """Test parsing of forget/delete commands."""
        test_cases = [
            ("forget about the old deadline", MemoryAction.FORGET, "the old deadline"),
            ("please forget what I said about the contractor", MemoryAction.FORGET, "the contractor"),
            ("delete the memory about ABC Corp", MemoryAction.FORGET, "ABC Corp"),
            ("remove from memory what I said about the budget", MemoryAction.FORGET, "the budget"),
            ("erase the fact that payment was late", MemoryAction.FORGET, "payment was late"),
        ]
        
        for input_text, expected_action, expected_content in test_cases:
            command = parse_memory_command(input_text)
            assert command is not None, f"Failed to parse: {input_text}"
            assert command.action == expected_action
            assert command.content == expected_content
            assert command.confidence >= 0.7
    
    def test_parse_update_commands(self):
        """Test parsing of update/correct commands."""
        test_cases = [
            ("update the deadline to April 1st", MemoryAction.UPDATE, "April 1st", "the deadline"),
            ("correct the budget to $600,000", MemoryAction.UPDATE, "$600,000", "the budget"),
            ("change the contractor name to XYZ Inc", MemoryAction.UPDATE, "XYZ Inc", "the contractor name"),
            ("actually, the meeting is at 4pm", MemoryAction.UPDATE, "at 4pm", "the meeting"),
            ("correction: the case number is 54321", MemoryAction.UPDATE, "54321", "the case number"),
        ]
        
        for input_text, expected_action, expected_content, expected_target in test_cases:
            command = parse_memory_command(input_text)
            assert command is not None, f"Failed to parse: {input_text}"
            assert command.action == expected_action
            assert command.content == expected_content
            assert command.target == expected_target
            assert command.confidence >= 0.7
    
    def test_parse_query_commands(self):
        """Test parsing of query/recall commands."""
        test_cases = [
            ("what do you remember about the deadline", MemoryAction.QUERY, "the deadline"),
            ("show me what you know about ABC Corp", MemoryAction.QUERY, "ABC Corp"),
            ("what have you learned about the project", MemoryAction.QUERY, "the project"),
            ("recall what you know about payment terms", MemoryAction.QUERY, "payment terms"),
            ("search your memory for contract details", MemoryAction.QUERY, "contract details"),
        ]
        
        for input_text, expected_action, expected_content in test_cases:
            command = parse_memory_command(input_text)
            assert command is not None, f"Failed to parse: {input_text}"
            assert command.action == expected_action
            assert command.content == expected_content
            assert command.confidence >= 0.7
    
    def test_non_commands_not_parsed(self):
        """Test that regular queries are not parsed as commands."""
        non_commands = [
            "What is the status of the project?",
            "How much is the budget?",
            "When is the deadline?",
            "Tell me about the contractor",
            "I need information about payments",
            "The contractor is working on the project",  # Statement, not command
        ]
        
        for text in non_commands:
            command = parse_memory_command(text)
            # These should either not parse or have low confidence
            if command:
                assert command.confidence < 0.7, f"Incorrectly parsed as command: {text}"
    
    def test_confidence_adjustments(self):
        """Test confidence score adjustments."""
        # Polite command should have higher confidence
        polite = parse_memory_command("please remember that the deadline is tomorrow")
        assert polite is not None
        
        # Direct command should also parse well
        direct = parse_memory_command("remember that the deadline is tomorrow")
        assert direct is not None
        
        # Both should have good confidence
        assert polite.confidence >= 0.85
        assert direct.confidence >= 0.80
        
        # Very short command might have lower confidence
        short = parse_memory_command("remember x")
        if short:
            assert short.confidence < 0.9
    
    def test_command_normalization(self):
        """Test text normalization in parsing."""
        # Test with extra spaces
        command = parse_memory_command("  remember   that   the   deadline   is   tomorrow  ")
        assert command is not None
        assert command.content == "the   deadline   is   tomorrow"  # Content preserves original spacing
        
        # Test with punctuation
        command = parse_memory_command("Remember that the deadline is tomorrow!!!")
        assert command is not None
        assert command.action == MemoryAction.REMEMBER
    
    def test_case_insensitivity(self):
        """Test that parsing is case-insensitive."""
        test_cases = [
            "REMEMBER THAT THE DEADLINE IS TOMORROW",
            "Remember That The Deadline Is Tomorrow",
            "rEmEmBeR tHaT tHe DeAdLiNe Is ToMoRrOw",
        ]
        
        for text in test_cases:
            command = parse_memory_command(text)
            assert command is not None
            assert command.action == MemoryAction.REMEMBER
    
    def test_suggest_command(self):
        """Test command suggestion for unclear input."""
        # Test deadline-related suggestion
        suggestion = MemoryCommandParser.suggest_command("the deadline is March 15")
        assert suggestion == "Remember that the deadline is March 15"
        
        # Test error correction suggestion
        suggestion = MemoryCommandParser.suggest_command("that's wrong, it's actually April")
        assert suggestion == "Update the memory about [topic] to that's wrong, it's actually April"
        
        # Test deletion suggestion
        suggestion = MemoryCommandParser.suggest_command("remove the old information")
        assert suggestion == "Forget about remove the old information"
        
        # Test query suggestion
        suggestion = MemoryCommandParser.suggest_command("what about the contractor?")
        assert suggestion == "What do you remember about what about the contractor?"
    
    def test_format_confirmation(self):
        """Test confirmation message formatting."""
        # Test remember confirmation
        command = MemoryCommand(
            action=MemoryAction.REMEMBER,
            content="the deadline is March 15th",
            confidence=0.95,
            raw_input="remember that the deadline is March 15th"
        )
        confirmation = MemoryCommandParser.format_confirmation(command)
        assert "I'll remember" in confirmation
        assert "the deadline is March 15th" in confirmation
        
        # Test forget confirmation
        command = MemoryCommand(
            action=MemoryAction.FORGET,
            content="the old deadline",
            confidence=0.90,
            raw_input="forget about the old deadline"
        )
        confirmation = MemoryCommandParser.format_confirmation(command)
        assert "forgotten" in confirmation
        assert "the old deadline" in confirmation
        
        # Test update confirmation
        command = MemoryCommand(
            action=MemoryAction.UPDATE,
            content="April 1st",
            target="the deadline",
            confidence=0.85,
            raw_input="update the deadline to April 1st"
        )
        confirmation = MemoryCommandParser.format_confirmation(command)
        assert "updated" in confirmation
        assert "the deadline" in confirmation
        assert "April 1st" in confirmation
    
    def test_command_to_dict(self):
        """Test command serialization to dictionary."""
        command = MemoryCommand(
            action=MemoryAction.REMEMBER,
            content="test content",
            confidence=0.95,
            raw_input="remember that test content",
            target=None
        )
        
        result = command.to_dict()
        assert result["action"] == "remember"
        assert result["content"] == "test content"
        assert result["confidence"] == 0.95
        assert result["raw_input"] == "remember that test content"
        assert result["target"] is None


class TestMemoryCommandEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test parsing empty or whitespace input."""
        assert parse_memory_command("") is None
        assert parse_memory_command("   ") is None
        assert parse_memory_command("\n\t") is None
    
    def test_very_long_input(self):
        """Test parsing very long input."""
        long_content = "x" * 500
        command = parse_memory_command(f"remember that {long_content}")
        
        if command:
            # Should parse but might have reduced confidence
            assert command.action == MemoryAction.REMEMBER
            assert len(command.content) > 400
    
    def test_special_characters(self):
        """Test parsing with special characters."""
        command = parse_memory_command("remember that the cost is $1,234.56 (excluding tax)")
        assert command is not None
        assert "$1,234.56" in command.content
        assert "(excluding tax)" in command.content
    
    def test_ambiguous_commands(self):
        """Test handling of ambiguous commands."""
        # This could be a question or a command
        ambiguous = "do you remember the meeting time?"
        command = parse_memory_command(ambiguous)
        
        if command:
            # If parsed, should be a query
            assert command.action == MemoryAction.QUERY
            # Confidence might be lower due to question mark
            assert command.confidence < 0.95