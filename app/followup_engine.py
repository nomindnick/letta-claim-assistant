"""
Advanced Follow-up Question Generation Engine.

Generates contextually relevant, domain-specific follow-up questions
for construction claims analysis based on conversation history and memory.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from .logging_conf import get_logger
from .models import KnowledgeItem
from .llm.base import LLMProvider

logger = get_logger(__name__)


class FollowupCategory(Enum):
    """Categories of follow-up questions for construction claims."""
    EVIDENCE = "evidence"
    TIMELINE = "timeline"
    TECHNICAL = "technical"
    LEGAL = "legal"
    DAMAGES = "damages"
    CAUSATION = "causation"
    RESPONSIBILITY = "responsibility"


@dataclass
class FollowupSuggestion:
    """A single follow-up suggestion with metadata."""
    question: str
    category: FollowupCategory
    priority: float  # 0.0 to 1.0
    reasoning: str
    related_entities: List[str]
    requires_expert: bool = False


@dataclass
class FollowupContext:
    """Context for generating follow-up suggestions."""
    user_query: str
    assistant_answer: str
    memory_items: List[KnowledgeItem]
    conversation_history: List[str] = None
    matter_context: Dict = None


class FollowupEngine:
    """Advanced follow-up question generation with domain expertise."""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider
        self.template_library = self._build_template_library()
        self.asked_questions_cache = set()  # Prevent duplicates
        
    def _build_template_library(self) -> Dict[FollowupCategory, List[str]]:
        """Build library of domain-specific follow-up templates."""
        return {
            FollowupCategory.EVIDENCE: [
                "What additional documentation exists for {topic}?",
                "Are there photos or videos of {issue}?",
                "Do daily logs mention {event}?",
                "What correspondence addresses {matter}?",
                "Are there inspection reports for {area}?",
                "Does the contract specify requirements for {aspect}?",
                "What expert testimony supports {claim}?"
            ],
            FollowupCategory.TIMELINE: [
                "When was {party} first notified of {issue}?",
                "What was the sequence of events leading to {failure}?",
                "How does this impact the project schedule?",
                "What delays resulted from {event}?",
                "When was {work} originally scheduled to complete?",
                "Are there critical path implications?",
                "What notice requirements were triggered?"
            ],
            FollowupCategory.TECHNICAL: [
                "What technical standards apply to {work}?",
                "Should we engage a {expert_type} expert?",
                "What testing was performed on {material}?",
                "Are there design deficiencies in {system}?",
                "What industry practices govern {activity}?",
                "How was {installation} supposed to be performed?",
                "What quality control measures were required?"
            ],
            FollowupCategory.LEGAL: [
                "What contract provisions address {issue}?",
                "Are there warranty implications?",
                "What insurance coverage applies?",
                "Are there notice requirements for {claim_type}?",
                "What legal precedents are relevant?",
                "How does this affect liquidated damages?",
                "Are there indemnification provisions?"
            ],
            FollowupCategory.DAMAGES: [
                "What are the cost implications of {issue}?",
                "How should damages be calculated?",
                "What mitigation efforts were made?",
                "Are there consequential damages?",
                "What is the cost of remediation?",
                "How does this impact project completion?",
                "What lost profits are attributable to {delay}?"
            ],
            FollowupCategory.CAUSATION: [
                "What was the root cause of {failure}?",
                "How did {condition} contribute to {problem}?",
                "Was {issue} foreseeable?",
                "What other factors could have caused {damage}?",
                "How do we establish proximate cause?",
                "What evidence links {action} to {result}?",
                "Are there multiple contributing causes?"
            ],
            FollowupCategory.RESPONSIBILITY: [
                "Who was responsible for {work}?",
                "What were {party}'s contractual obligations?",
                "Did {contractor} breach any duties?",
                "How should liability be allocated?",
                "What standard of care applies?",
                "Were proper procedures followed?",
                "Who had control over {area}?"
            ]
        }
    
    async def generate_followups(
        self, 
        context: FollowupContext,
        max_suggestions: int = 4,
        min_priority: float = 0.3
    ) -> List[FollowupSuggestion]:
        """Generate prioritized follow-up suggestions."""
        
        logger.debug(
            "Generating follow-up suggestions",
            user_query_preview=context.user_query[:100],
            memory_items=len(context.memory_items),
            max_suggestions=max_suggestions
        )
        
        # Generate suggestions from multiple sources
        all_suggestions = []
        
        # 1. Template-based suggestions
        template_suggestions = self._generate_template_suggestions(context)
        all_suggestions.extend(template_suggestions)
        
        # 2. Memory-driven suggestions
        memory_suggestions = self._generate_memory_suggestions(context)
        all_suggestions.extend(memory_suggestions)
        
        # 3. LLM-generated suggestions (if available)
        if self.llm_provider:
            llm_suggestions = await self._generate_llm_suggestions(context)
            all_suggestions.extend(llm_suggestions)
        
        # Filter and prioritize
        filtered_suggestions = self._filter_and_prioritize(
            all_suggestions, min_priority, max_suggestions
        )
        
        # Update cache to avoid duplicates in future calls
        for suggestion in filtered_suggestions:
            self.asked_questions_cache.add(suggestion.question.lower())
        
        logger.debug(
            "Follow-up generation completed",
            total_generated=len(all_suggestions),
            filtered_count=len(filtered_suggestions)
        )
        
        return filtered_suggestions
    
    def _generate_template_suggestions(self, context: FollowupContext) -> List[FollowupSuggestion]:
        """Generate suggestions using domain-specific templates."""
        suggestions = []
        
        # Extract key entities and topics from query and answer
        entities = self._extract_entities(context.user_query, context.assistant_answer)
        topics = self._extract_topics(context.user_query, context.assistant_answer)
        
        # Generate suggestions for each category
        for category, templates in self.template_library.items():
            category_suggestions = self._apply_templates_to_context(
                templates, entities, topics, category, context
            )
            suggestions.extend(category_suggestions)
        
        return suggestions
    
    def _apply_templates_to_context(
        self, 
        templates: List[str], 
        entities: Dict[str, List[str]], 
        topics: List[str],
        category: FollowupCategory,
        context: FollowupContext
    ) -> List[FollowupSuggestion]:
        """Apply templates to extracted entities and topics."""
        suggestions = []
        
        for template in templates:
            # Try to fill template with relevant entities/topics
            filled_questions = self._fill_template(template, entities, topics)
            
            for question in filled_questions:
                if self._is_relevant_to_context(question, context):
                    priority = self._calculate_priority(question, category, context)
                    
                    suggestion = FollowupSuggestion(
                        question=question,
                        category=category,
                        priority=priority,
                        reasoning=f"Template-based suggestion for {category.value}",
                        related_entities=list(entities.get('organizations', [])) + list(entities.get('people', [])),
                        requires_expert=self._requires_expert_analysis(question)
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _extract_entities(self, query: str, answer: str) -> Dict[str, List[str]]:
        """Extract relevant entities from query and answer."""
        text = f"{query} {answer}".lower()
        
        entities = {
            'organizations': [],
            'people': [],
            'documents': [],
            'dates': [],
            'technical_terms': [],
            'locations': []
        }
        
        # Common construction entities
        org_patterns = [
            r'\b(\w+\s+(?:construction|contracting|engineering|inc|llc|corp))\b',
            r'\b(owner|contractor|subcontractor|architect|engineer)\b',
            r'\b(\w+\s+company)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['organizations'].extend([m.title() for m in matches])
        
        # Document references
        doc_pattern = r'\b(\w+\.pdf|\w+\s+report|\w+\s+log|\w+\s+spec)\b'
        doc_matches = re.findall(doc_pattern, text, re.IGNORECASE)
        entities['documents'].extend([m.title() for m in doc_matches])
        
        # Technical terms
        tech_patterns = [
            r'\b(concrete|steel|rebar|foundation|excavation|grading|structural)\b',
            r'\b(rfi|change\s+order|punch\s+list|inspection|testing)\b',
            r'\b(delay|schedule|critical\s+path|milestone)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['technical_terms'].extend([m.title() for m in matches])
        
        # Remove duplicates and empty strings
        for key in entities:
            entities[key] = list(set(filter(None, entities[key])))
        
        return entities
    
    def _extract_topics(self, query: str, answer: str) -> List[str]:
        """Extract main topics from query and answer."""
        text = f"{query} {answer}".lower()
        
        # Common construction claim topics
        topic_patterns = [
            r'\b(delay\w*|schedule\w*)\b',
            r'\b(damage\w*|repair\w*|defect\w*)\b',
            r'\b(contract\w*|specification\w*)\b',
            r'\b(failure\w*|breach\w*|non-conformance)\b',
            r'\b(change\s+order\w*|extra\s+work)\b',
            r'\b(inspection\w*|testing\w*|quality)\b',
            r'\b(notice\w*|claim\w*|dispute\w*)\b',
            r'\b(cost\w*|payment\w*|billing)\b'
        ]
        
        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            topics.extend([m.strip() for m in matches])
        
        return list(set(topics))
    
    def _fill_template(self, template: str, entities: Dict[str, List[str]], topics: List[str]) -> List[str]:
        """Fill template placeholders with relevant entities and topics."""
        filled_questions = []
        
        # Define placeholder mappings
        placeholder_mappings = {
            '{topic}': topics[:3],  # Limit to top 3 topics
            '{issue}': entities['technical_terms'][:2],
            '{event}': topics[:2],
            '{matter}': topics[:2],
            '{area}': entities['locations'][:2] if entities['locations'] else ['work area', 'site'],
            '{aspect}': entities['technical_terms'][:2],
            '{claim}': ['this claim', 'the issue'],
            '{party}': entities['organizations'][:2] if entities['organizations'] else ['the contractor', 'owner'],
            '{failure}': ['failure', 'defect', 'problem'],
            '{work}': entities['technical_terms'][:2] if entities['technical_terms'] else ['work', 'construction'],
            '{expert_type}': ['structural', 'geotechnical', 'scheduling', 'forensic'],
            '{material}': ['concrete', 'steel', 'materials'],
            '{system}': ['system', 'installation'],
            '{activity}': ['activity', 'operation'],
            '{installation}': ['installation', 'construction'],
            '{claim_type}': ['delay claim', 'damage claim', 'extra work claim'],
            '{delay}': ['delay', 'schedule impact'],
            '{problem}': ['problem', 'issue', 'defect'],
            '{condition}': ['condition', 'circumstance'],
            '{damage}': ['damage', 'defect'],
            '{action}': ['action', 'work', 'decision'],
            '{result}': ['result', 'consequence', 'impact'],
            '{contractor}': entities['organizations'][:1] if entities['organizations'] else ['contractor']
        }
        
        # Find placeholders in template
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        if not placeholders:
            # No placeholders, return as-is
            return [template]
        
        # Generate combinations
        for placeholder in placeholders:
            if f'{{{placeholder}}}' in placeholder_mappings:
                values = placeholder_mappings[f'{{{placeholder}}}']
                if values:
                    for value in values[:2]:  # Limit to 2 variations per placeholder
                        filled = template.replace(f'{{{placeholder}}}', value)
                        if len(filled) <= 150:  # Reasonable length limit
                            filled_questions.append(filled)
                    break
        
        # If no successful fills, try with generic values
        if not filled_questions:
            generic_filled = template
            for placeholder in placeholders:
                generic_filled = generic_filled.replace(f'{{{placeholder}}}', 'this matter')
            if len(generic_filled) <= 150:
                filled_questions.append(generic_filled)
        
        return filled_questions[:3]  # Limit variations
    
    def _generate_memory_suggestions(self, context: FollowupContext) -> List[FollowupSuggestion]:
        """Generate suggestions based on agent memory context."""
        suggestions = []
        
        if not context.memory_items:
            return suggestions
        
        # Group memory items by type
        memory_by_type = {}
        for item in context.memory_items:
            if item.type not in memory_by_type:
                memory_by_type[item.type] = []
            memory_by_type[item.type].append(item)
        
        # Generate type-specific suggestions
        for memory_type, items in memory_by_type.items():
            type_suggestions = self._generate_memory_type_suggestions(memory_type, items, context)
            suggestions.extend(type_suggestions)
        
        return suggestions
    
    def _generate_memory_type_suggestions(
        self, 
        memory_type: str, 
        items: List[KnowledgeItem], 
        context: FollowupContext
    ) -> List[FollowupSuggestion]:
        """Generate suggestions specific to memory item type."""
        suggestions = []
        
        if memory_type == "Event":
            for item in items[:2]:  # Limit to prevent too many suggestions
                suggestions.extend([
                    FollowupSuggestion(
                        question=f"What documentation exists for {item.label}?",
                        category=FollowupCategory.EVIDENCE,
                        priority=0.7,
                        reasoning=f"Follow-up on documented event: {item.label}",
                        related_entities=item.actors,
                        requires_expert=False
                    ),
                    FollowupSuggestion(
                        question=f"What were the consequences of {item.label}?",
                        category=FollowupCategory.DAMAGES,
                        priority=0.6,
                        reasoning=f"Assess impact of event: {item.label}",
                        related_entities=item.actors,
                        requires_expert=False
                    )
                ])
        
        elif memory_type == "Issue":
            for item in items[:2]:
                suggestions.append(
                    FollowupSuggestion(
                        question=f"How should we address the {item.label} issue?",
                        category=FollowupCategory.LEGAL,
                        priority=0.8,
                        reasoning=f"Legal strategy for issue: {item.label}",
                        related_entities=item.actors,
                        requires_expert=True
                    )
                )
        
        elif memory_type == "Entity":
            for item in items[:2]:
                if any(role in item.label.lower() for role in ['contractor', 'owner', 'engineer']):
                    suggestions.append(
                        FollowupSuggestion(
                            question=f"What are {item.label}'s contractual obligations?",
                            category=FollowupCategory.RESPONSIBILITY,
                            priority=0.7,
                            reasoning=f"Review obligations of party: {item.label}",
                            related_entities=[item.label],
                            requires_expert=False
                        )
                    )
        
        return suggestions
    
    async def _generate_llm_suggestions(self, context: FollowupContext) -> List[FollowupSuggestion]:
        """Generate suggestions using LLM with enhanced prompts."""
        if not self.llm_provider:
            return []
        
        try:
            # Build enhanced context for LLM
            memory_summary = self._summarize_memory_for_llm(context.memory_items)
            
            prompt = f"""Based on this construction claims analysis:

USER QUESTION: {context.user_query}

ASSISTANT ANSWER: {context.assistant_answer[:800]}...

RELEVANT MEMORY: {memory_summary}

Generate 3-4 specific follow-up questions that would help a construction claims attorney:

1. Identify additional evidence or documentation needed
2. Clarify legal issues or responsibilities
3. Assess damages or technical aspects
4. Explore causation and timeline issues

Each question should be:
- Specific and actionable (≤18 words)
- Grounded in construction law and industry practice
- Designed to advance the legal analysis
- Different from obvious next steps

Return only the questions, one per line."""

            messages = [{"role": "user", "content": prompt}]
            
            response = await self.llm_provider.generate(
                system="You are a construction claims legal assistant specializing in follow-up question generation.",
                messages=messages,
                max_tokens=300,
                temperature=0.4
            )
            
            # Parse response into suggestions
            suggestions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering and formatting
                line = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                
                if line and len(line) <= 150:
                    suggestion = FollowupSuggestion(
                        question=line,
                        category=self._classify_question_category(line),
                        priority=0.6,  # Medium priority for LLM suggestions
                        reasoning="LLM-generated contextual suggestion",
                        related_entities=[],
                        requires_expert=self._requires_expert_analysis(line)
                    )
                    suggestions.append(suggestion)
            
            return suggestions[:4]  # Limit to 4 suggestions
            
        except Exception as e:
            logger.warning("LLM follow-up generation failed", error=str(e))
            return []
    
    def _summarize_memory_for_llm(self, memory_items: List[KnowledgeItem]) -> str:
        """Create concise memory summary for LLM context."""
        if not memory_items:
            return "No relevant memory available."
        
        summary_parts = []
        for item in memory_items[:5]:  # Limit to top 5 items
            actors_str = ", ".join(item.actors) if item.actors else "N/A"
            summary_parts.append(f"{item.type}: {item.label} (Actors: {actors_str})")
        
        return " | ".join(summary_parts)
    
    def _classify_question_category(self, question: str) -> FollowupCategory:
        """Classify question into appropriate category."""
        question_lower = question.lower()
        
        # Classification based on keywords
        if any(word in question_lower for word in ['document', 'evidence', 'photo', 'report', 'record']):
            return FollowupCategory.EVIDENCE
        elif any(word in question_lower for word in ['when', 'timeline', 'schedule', 'sequence', 'date']):
            return FollowupCategory.TIMELINE
        elif any(word in question_lower for word in ['expert', 'technical', 'standard', 'testing', 'specification']):
            return FollowupCategory.TECHNICAL
        elif any(word in question_lower for word in ['contract', 'legal', 'liability', 'responsibility', 'breach']):
            return FollowupCategory.LEGAL
        elif any(word in question_lower for word in ['damage', 'cost', 'money', 'repair', 'impact']):
            return FollowupCategory.DAMAGES
        elif any(word in question_lower for word in ['cause', 'why', 'reason', 'factor', 'led to']):
            return FollowupCategory.CAUSATION
        else:
            return FollowupCategory.RESPONSIBILITY  # Default
    
    def _requires_expert_analysis(self, question: str) -> bool:
        """Determine if question requires expert analysis."""
        expert_indicators = [
            'technical', 'structural', 'engineering', 'geotechnical', 
            'forensic', 'testing', 'analysis', 'expert', 'specialist'
        ]
        return any(indicator in question.lower() for indicator in expert_indicators)
    
    def _is_relevant_to_context(self, question: str, context: FollowupContext) -> bool:
        """Check if question is relevant to current context."""
        # Avoid questions that are too generic or already covered
        if question.lower() in self.asked_questions_cache:
            return False
        
        # Check if question relates to any content in query or answer
        question_words = set(question.lower().split())
        context_words = set((context.user_query + " " + context.assistant_answer).lower().split())
        
        # Require some overlap in content
        overlap = len(question_words.intersection(context_words))
        return overlap >= 2  # At least 2 words in common
    
    def _calculate_priority(
        self, 
        question: str, 
        category: FollowupCategory, 
        context: FollowupContext
    ) -> float:
        """Calculate priority score for suggestion."""
        base_priority = 0.5
        
        # Category-based adjustments
        category_weights = {
            FollowupCategory.LEGAL: 0.3,
            FollowupCategory.DAMAGES: 0.25,
            FollowupCategory.CAUSATION: 0.2,
            FollowupCategory.EVIDENCE: 0.15,
            FollowupCategory.TIMELINE: 0.1,
            FollowupCategory.TECHNICAL: 0.1,
            FollowupCategory.RESPONSIBILITY: 0.15
        }
        
        base_priority += category_weights.get(category, 0.1)
        
        # Context relevance boost
        question_words = set(question.lower().split())
        query_words = set(context.user_query.lower().split())
        
        relevance_overlap = len(question_words.intersection(query_words))
        if relevance_overlap >= 3:
            base_priority += 0.2
        elif relevance_overlap >= 2:
            base_priority += 0.1
        
        # Memory connection boost
        if context.memory_items:
            memory_text = " ".join([item.label.lower() for item in context.memory_items])
            memory_words = set(memory_text.split())
            if question_words.intersection(memory_words):
                base_priority += 0.15
        
        return min(base_priority, 1.0)  # Cap at 1.0
    
    def _filter_and_prioritize(
        self, 
        suggestions: List[FollowupSuggestion],
        min_priority: float,
        max_suggestions: int
    ) -> List[FollowupSuggestion]:
        """Filter and prioritize suggestions."""
        
        # Filter by minimum priority
        filtered = [s for s in suggestions if s.priority >= min_priority]
        
        # Remove duplicates based on question similarity
        unique_suggestions = self._remove_duplicate_questions(filtered)
        
        # Sort by priority (descending)
        sorted_suggestions = sorted(unique_suggestions, key=lambda x: x.priority, reverse=True)
        
        # Ensure category diversity in top suggestions
        diverse_suggestions = self._ensure_category_diversity(sorted_suggestions, max_suggestions)
        
        return diverse_suggestions[:max_suggestions]
    
    def _remove_duplicate_questions(self, suggestions: List[FollowupSuggestion]) -> List[FollowupSuggestion]:
        """Remove questions that are too similar."""
        unique_suggestions = []
        seen_questions = set()
        
        for suggestion in suggestions:
            # Normalize question for comparison
            normalized = re.sub(r'[^\w\s]', '', suggestion.question.lower())
            normalized = ' '.join(normalized.split())  # Normalize whitespace
            
            # Check for substantial overlap with existing questions
            is_duplicate = False
            for seen in seen_questions:
                # Calculate word overlap
                words1 = set(normalized.split())
                words2 = set(seen.split())
                overlap = len(words1.intersection(words2))
                total = len(words1.union(words2))
                
                if total > 0 and overlap / total > 0.7:  # 70% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_suggestions.append(suggestion)
                seen_questions.add(normalized)
        
        return unique_suggestions
    
    def _ensure_category_diversity(
        self, 
        suggestions: List[FollowupSuggestion], 
        max_suggestions: int
    ) -> List[FollowupSuggestion]:
        """Ensure diverse categories in final suggestions."""
        if len(suggestions) <= max_suggestions:
            return suggestions
        
        # Try to include at least one from each major category
        priority_categories = [
            FollowupCategory.LEGAL,
            FollowupCategory.EVIDENCE,
            FollowupCategory.DAMAGES,
            FollowupCategory.CAUSATION
        ]
        
        diverse_suggestions = []
        used_categories = set()
        
        # First pass: one from each priority category
        for suggestion in suggestions:
            if suggestion.category in priority_categories and suggestion.category not in used_categories:
                diverse_suggestions.append(suggestion)
                used_categories.add(suggestion.category)
                
                if len(diverse_suggestions) >= max_suggestions:
                    break
        
        # Second pass: fill remaining slots with highest priority
        remaining_slots = max_suggestions - len(diverse_suggestions)
        remaining_suggestions = [s for s in suggestions if s not in diverse_suggestions]
        
        diverse_suggestions.extend(remaining_suggestions[:remaining_slots])
        
        return diverse_suggestions