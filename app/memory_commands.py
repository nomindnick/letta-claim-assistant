"""
Natural language memory command parser and handlers.

Provides detection and parsing of memory management commands in natural language,
allowing users to manage agent memory through conversational interfaces.
"""

import re
from typing import Optional, Literal, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from .logging_conf import get_logger

logger = get_logger(__name__)


class MemoryAction(str, Enum):
    """Types of memory management actions."""
    REMEMBER = "remember"
    FORGET = "forget"
    UPDATE = "update"
    QUERY = "query"


@dataclass
class MemoryCommand:
    """Parsed memory command with action and content."""
    action: MemoryAction
    content: str
    confidence: float
    raw_input: str
    target: Optional[str] = None  # For updates/deletions - what to target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "action": self.action.value,
            "content": self.content,
            "confidence": self.confidence,
            "raw_input": self.raw_input,
            "target": self.target
        }


class MemoryCommandParser:
    """Parser for natural language memory commands."""
    
    # Command patterns with their confidence weights
    PATTERNS = {
        MemoryAction.REMEMBER: [
            (r"(?:please\s+)?remember\s+(?:that\s+)?(.+)", 0.95),
            (r"(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+)?(.+)", 0.90),
            (r"(?:please\s+)?don'?t\s+forget\s+(?:that\s+)?(.+)", 0.90),
            (r"(?:please\s+)?note\s+(?:that\s+)?(.+)", 0.85),
            (r"(?:please\s+)?store\s+(?:the\s+)?(?:fact\s+)?(?:that\s+)?(.+)", 0.85),
            (r"(?:please\s+)?memorize\s+(?:that\s+)?(.+)", 0.85),
            (r"(?:please\s+)?add\s+to\s+(?:your\s+)?memory[::\s]+(.+)", 0.95),
            (r"for\s+future\s+reference[,:\s]+(.+)", 0.80),
            (r"(?:please\s+)?save\s+(?:the\s+)?(?:fact\s+)?(?:that\s+)?(.+)", 0.80),
        ],
        MemoryAction.FORGET: [
            (r"(?:please\s+)?forget\s+(?:about\s+)?(.+)", 0.95),
            (r"(?:please\s+)?remove\s+(?:from\s+memory\s+)?(?:what\s+(?:I|you)\s+said\s+about\s+)?(.+)", 0.90),
            (r"(?:please\s+)?delete\s+(?:the\s+)?(?:memory\s+)?(?:about\s+)?(.+)", 0.90),
            (r"(?:please\s+)?disregard\s+(?:what\s+I\s+said\s+about\s+)?(.+)", 0.85),
            (r"(?:please\s+)?ignore\s+(?:what\s+I\s+said\s+about\s+)?(.+)", 0.80),
            (r"(?:please\s+)?erase\s+(?:from\s+memory\s+)?(?:the\s+)?(?:fact\s+)?(?:that\s+)?(.+)", 0.85),
            (r"(?:please\s+)?clear\s+(?:your\s+)?memory\s+(?:about\s+)?(.+)", 0.85),
            (r"don'?t\s+remember\s+(?:that\s+)?(.+)", 0.85),
        ],
        MemoryAction.UPDATE: [
            (r"(?:please\s+)?update\s+(?:the\s+)?(?:memory\s+)?(?:about\s+)?(.+?)\s+(?:to|with)\s+(.+)", 0.95),
            (r"(?:please\s+)?correct\s+(?:the\s+)?(?:fact\s+)?(?:that\s+)?(.+?)\s+(?:to|with)\s+(.+)", 0.90),
            (r"(?:please\s+)?change\s+(?:the\s+)?(?:memory\s+)?(?:about\s+)?(.+?)\s+(?:to|with)\s+(.+)", 0.90),
            (r"(?:please\s+)?revise\s+(?:the\s+)?(?:memory\s+)?(?:about\s+)?(.+?)\s+(?:to|with)\s+(.+)", 0.85),
            (r"actually[,\s]+(.+?)\s+(?:is|should\s+be)\s+(.+)", 0.80),
            (r"correction:\s+(.+?)\s+(?:is|should\s+be)\s+(.+)", 0.85),
            (r"(?:please\s+)?replace\s+(?:the\s+)?(?:memory\s+)?(?:about\s+)?(.+?)\s+with\s+(.+)", 0.90),
        ],
        MemoryAction.QUERY: [
            (r"what\s+do\s+you\s+(?:remember|know|recall)\s+about\s+(.+)", 0.95),
            (r"(?:please\s+)?(?:show|tell)\s+me\s+what\s+you\s+(?:remember|know)\s+about\s+(.+)", 0.90),
            (r"(?:please\s+)?(?:list|show)\s+(?:your\s+)?memor(?:y|ies)\s+(?:about\s+)?(.+)", 0.90),
            (r"what\s+have\s+you\s+learned\s+about\s+(.+)", 0.85),
            (r"what\s+facts\s+do\s+you\s+have\s+about\s+(.+)", 0.85),
            (r"(?:please\s+)?recall\s+(?:what\s+you\s+know\s+about\s+)?(.+)", 0.85),
            (r"do\s+you\s+remember\s+(?:anything\s+)?(?:about\s+)?(.+)", 0.85),
            (r"(?:please\s+)?search\s+(?:your\s+)?memory\s+for\s+(.+)", 0.90),
        ]
    }
    
    # Confidence adjustment factors
    CONFIDENCE_ADJUSTMENTS = {
        "contains_please": 0.05,  # Polite commands are more likely intentional
        "contains_question": -0.05,  # Questions might not be commands
        "starts_with_command": 0.05,  # Direct commands at start
        "all_caps": -0.10,  # All caps might be emphasis, not command
        "too_short": -0.15,  # Very short inputs are less likely commands
        "too_long": -0.10,  # Very long inputs might be explanations
    }
    
    # Minimum confidence threshold for executing commands
    MIN_CONFIDENCE_THRESHOLD = 0.70
    
    @classmethod
    def parse(cls, text: str) -> Optional[MemoryCommand]:
        """
        Parse natural language text to detect memory commands.
        
        Args:
            text: The input text to parse
            
        Returns:
            MemoryCommand if detected with sufficient confidence, None otherwise
        """
        if not text or not text.strip():
            return None
        
        # Normalize text for parsing
        normalized = cls._normalize_text(text)
        
        # Try to match against each action's patterns
        best_match: Optional[Tuple[MemoryAction, str, float, Optional[str]]] = None
        
        for action, patterns in cls.PATTERNS.items():
            for pattern, base_confidence in patterns:
                match = re.match(pattern, normalized, re.IGNORECASE)
                if match:
                    # Extract content based on action type
                    if action == MemoryAction.UPDATE:
                        # Update has two capture groups: target and new content
                        if len(match.groups()) >= 2:
                            target = match.group(1).strip()
                            content = match.group(2).strip()
                        else:
                            continue
                    else:
                        # Other actions have single capture group
                        content = match.group(1).strip() if match.groups() else ""
                        target = None
                    
                    # Calculate adjusted confidence
                    confidence = cls._calculate_confidence(
                        text, normalized, base_confidence
                    )
                    
                    # Track best match
                    if not best_match or confidence > best_match[2]:
                        best_match = (action, content, confidence, target)
        
        # Return best match if above threshold
        if best_match and best_match[2] >= cls.MIN_CONFIDENCE_THRESHOLD:
            action, content, confidence, target = best_match
            
            command = MemoryCommand(
                action=action,
                content=content,
                confidence=confidence,
                raw_input=text,
                target=target
            )
            
            logger.info(
                "Parsed memory command",
                action=action.value,
                confidence=confidence,
                content_preview=content[:50]
            )
            
            return command
        
        return None
    
    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for pattern matching."""
        # Remove extra whitespace
        normalized = " ".join(text.split())
        
        # Remove leading/trailing punctuation
        normalized = normalized.strip(".,!?;:")
        
        # Standardize contractions
        normalized = normalized.replace("don't", "do not")
        normalized = normalized.replace("won't", "will not")
        normalized = normalized.replace("can't", "cannot")
        
        return normalized
    
    @classmethod
    def _calculate_confidence(
        cls, 
        original: str, 
        normalized: str, 
        base_confidence: float
    ) -> float:
        """Calculate adjusted confidence score for a match."""
        confidence = base_confidence
        
        # Apply adjustments
        if "please" in normalized.lower():
            confidence += cls.CONFIDENCE_ADJUSTMENTS["contains_please"]
        
        if "?" in original:
            confidence += cls.CONFIDENCE_ADJUSTMENTS["contains_question"]
        
        # Check if starts with command verb
        command_verbs = ["remember", "forget", "update", "correct", "delete", "remove"]
        if any(normalized.lower().startswith(verb) for verb in command_verbs):
            confidence += cls.CONFIDENCE_ADJUSTMENTS["starts_with_command"]
        
        # Check for all caps (might be emphasis rather than command)
        if original.isupper() and len(original) > 3:
            confidence += cls.CONFIDENCE_ADJUSTMENTS["all_caps"]
        
        # Length adjustments
        if len(normalized) < 10:
            confidence += cls.CONFIDENCE_ADJUSTMENTS["too_short"]
        elif len(normalized) > 200:
            confidence += cls.CONFIDENCE_ADJUSTMENTS["too_long"]
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    @classmethod
    def suggest_command(cls, text: str) -> Optional[str]:
        """
        Suggest a properly formatted command based on unclear input.
        
        Args:
            text: The input text that might be a command
            
        Returns:
            Suggested command format or None
        """
        normalized = cls._normalize_text(text).lower()
        
        # Look for keywords that suggest intent
        if any(word in normalized for word in ["deadline", "due", "date", "when"]):
            return f"Remember that {text}"
        
        if any(word in normalized for word in ["wrong", "incorrect", "mistake"]):
            return f"Update the memory about [topic] to {text}"
        
        if any(word in normalized for word in ["delete", "remove", "clear"]):
            return f"Forget about {text}"
        
        if any(word in normalized for word in ["what", "show", "tell"]):
            return f"What do you remember about {text}?"
        
        return None
    
    @classmethod
    def format_confirmation(cls, command: MemoryCommand) -> str:
        """
        Format a user-friendly confirmation message for a command.
        
        Args:
            command: The parsed memory command
            
        Returns:
            Formatted confirmation message
        """
        if command.action == MemoryAction.REMEMBER:
            return f"✓ I'll remember: {command.content}"
        
        elif command.action == MemoryAction.FORGET:
            return f"✓ I've forgotten about: {command.content}"
        
        elif command.action == MemoryAction.UPDATE:
            return f"✓ I've updated the memory about '{command.target}' to: {command.content}"
        
        elif command.action == MemoryAction.QUERY:
            return f"Searching my memory for: {command.content}"
        
        return "✓ Memory operation completed"


# Export main parser function for convenience
def parse_memory_command(text: str) -> Optional[MemoryCommand]:
    """
    Parse natural language text to detect memory commands.
    
    Args:
        text: The input text to parse
        
    Returns:
        MemoryCommand if detected with sufficient confidence, None otherwise
    """
    return MemoryCommandParser.parse(text)