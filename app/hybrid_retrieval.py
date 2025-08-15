"""
Hybrid Retrieval System for Enhanced RAG.

Combines vector similarity search with agent memory recall,
temporal scoring, and diversity optimization for improved
context selection in construction claims analysis.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

from .logging_conf import get_logger
from .vectors import SearchResult, VectorStore
from .models import KnowledgeItem

logger = get_logger(__name__)


@dataclass
class RetrievalWeights:
    """Configuration for hybrid retrieval scoring."""
    vector_similarity: float = 0.7
    memory_relevance: float = 0.3
    recency_boost: float = 0.1
    diversity_factor: float = 0.15
    temporal_decay_days: int = 30


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional scoring metadata."""
    # Original search result data
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str
    
    # Enhanced scoring information
    vector_score: float
    memory_score: float = 0.0
    recency_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0
    
    # Additional metadata
    last_accessed: Optional[datetime] = None
    document_type: Optional[str] = None
    related_memory_items: List[str] = field(default_factory=list)
    
    # Compatibility with original SearchResult
    @property
    def similarity_score(self) -> float:
        """Compatibility property for original interface."""
        return self.final_score


@dataclass
class HybridRetrievalContext:
    """Context for hybrid retrieval operations."""
    query: str
    matter_id: str
    conversation_history: List[str] = field(default_factory=list)
    recent_documents: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class HybridRetrieval:
    """Advanced retrieval system combining multiple signals."""
    
    def __init__(
        self, 
        vector_store: VectorStore,
        weights: Optional[RetrievalWeights] = None
    ):
        self.vector_store = vector_store
        self.weights = weights or RetrievalWeights()
        self.document_access_history = {}  # Track document access for recency
        self.query_history = []  # Track recent queries for diversity
        
    async def hybrid_search(
        self,
        context: HybridRetrievalContext,
        memory_items: List[KnowledgeItem],
        k: int = 8,
        enable_mmr: bool = True
    ) -> List[EnhancedSearchResult]:
        """
        Perform hybrid search combining vector similarity and memory relevance.
        
        Args:
            context: Retrieval context with query and metadata
            memory_items: Relevant memory items from agent
            k: Number of results to return
            enable_mmr: Whether to apply Maximal Marginal Relevance
            
        Returns:
            List of enhanced search results with hybrid scoring
        """
        logger.debug(
            "Starting hybrid retrieval",
            query_preview=context.query[:100],
            memory_items_count=len(memory_items),
            k=k,
            enable_mmr=enable_mmr
        )
        
        # Step 1: Get vector search results (retrieve more for reranking)
        vector_k = min(k * 3, 50)  # Get more candidates for better reranking
        vector_results = await self.vector_store.search(
            context.query, 
            k=vector_k
        )
        
        if not vector_results:
            logger.warning("No vector results found for query")
            return []
        
        # Step 2: Convert to enhanced results and calculate hybrid scores
        enhanced_results = self._create_enhanced_results(
            vector_results, context, memory_items
        )
        
        # Step 3: Apply hybrid scoring
        scored_results = self._apply_hybrid_scoring(
            enhanced_results, context, memory_items
        )
        
        # Step 4: Apply diversity optimization (MMR)
        if enable_mmr and len(scored_results) > k:
            diverse_results = self._apply_mmr(scored_results, context, k)
        else:
            diverse_results = sorted(scored_results, key=lambda x: x.final_score, reverse=True)[:k]
        
        # Step 5: Update access history for future recency scoring
        self._update_access_history(diverse_results, context)
        
        logger.debug(
            "Hybrid retrieval completed",
            vector_results_count=len(vector_results),
            enhanced_results_count=len(enhanced_results),
            final_results_count=len(diverse_results),
            avg_final_score=sum(r.final_score for r in diverse_results) / len(diverse_results) if diverse_results else 0
        )
        
        return diverse_results
    
    def _create_enhanced_results(
        self,
        vector_results: List[SearchResult],
        context: HybridRetrievalContext,
        memory_items: List[KnowledgeItem]
    ) -> List[EnhancedSearchResult]:
        """Convert vector results to enhanced results with metadata."""
        enhanced_results = []
        
        for result in vector_results:
            enhanced = EnhancedSearchResult(
                chunk_id=result.chunk_id,
                doc_name=result.doc_name,
                page_start=result.page_start,
                page_end=result.page_end,
                text=result.text,
                vector_score=result.similarity_score,
                document_type=self._infer_document_type(result.doc_name),
                related_memory_items=self._find_related_memory_items(result, memory_items)
            )
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def _apply_hybrid_scoring(
        self,
        results: List[EnhancedSearchResult],
        context: HybridRetrievalContext,
        memory_items: List[KnowledgeItem]
    ) -> List[EnhancedSearchResult]:
        """Apply hybrid scoring combining multiple signals."""
        
        for result in results:
            # 1. Memory relevance score
            result.memory_score = self._calculate_memory_score(result, memory_items, context.query)
            
            # 2. Recency score
            result.recency_score = self._calculate_recency_score(result, context)
            
            # 3. Document type preference score
            type_score = self._calculate_document_type_score(result, context)
            
            # 4. Calculate final hybrid score
            result.final_score = (
                self.weights.vector_similarity * result.vector_score +
                self.weights.memory_relevance * result.memory_score +
                self.weights.recency_boost * result.recency_score +
                0.1 * type_score  # Small boost for preferred document types
            )
        
        return results
    
    def _calculate_memory_score(
        self, 
        result: EnhancedSearchResult, 
        memory_items: List[KnowledgeItem],
        query: str
    ) -> float:
        """Calculate relevance score based on agent memory."""
        if not memory_items:
            return 0.0
        
        memory_score = 0.0
        result_text_lower = result.text.lower()
        query_lower = query.lower()
        
        for item in memory_items:
            item_score = 0.0
            
            # Check if memory item is mentioned in result text
            if item.label.lower() in result_text_lower:
                item_score += 0.4
            
            # Check for actor mentions
            for actor in item.actors:
                if actor.lower() in result_text_lower:
                    item_score += 0.2
            
            # Check for document reference matches
            for doc_ref in item.doc_refs:
                if doc_ref.get('doc', '').lower() in result.doc_name.lower():
                    # Check page overlap
                    ref_page = doc_ref.get('page', 0)
                    if result.page_start <= ref_page <= result.page_end:
                        item_score += 0.3
                    else:
                        item_score += 0.1
            
            # Check query relevance to memory item
            if any(word in item.label.lower() for word in query_lower.split()):
                item_score += 0.2
            
            # Type-specific bonuses
            if item.type == "Event" and "event" in query_lower:
                item_score += 0.1
            elif item.type == "Issue" and any(word in query_lower for word in ["issue", "problem", "claim"]):
                item_score += 0.1
            
            memory_score = max(memory_score, item_score)  # Take best match
        
        return min(memory_score, 1.0)  # Cap at 1.0
    
    def _calculate_recency_score(
        self, 
        result: EnhancedSearchResult, 
        context: HybridRetrievalContext
    ) -> float:
        """Calculate recency bonus based on recent access patterns."""
        base_recency = 0.0
        
        # Check if document was recently accessed
        doc_key = result.doc_name.lower()
        if doc_key in self.document_access_history:
            last_access = self.document_access_history[doc_key]
            days_since_access = (datetime.now() - last_access).days
            
            if days_since_access <= self.weights.temporal_decay_days:
                # Exponential decay over time
                decay_factor = math.exp(-days_since_access / (self.weights.temporal_decay_days / 3))
                base_recency += 0.5 * decay_factor
        
        # Check if document is in recent documents list
        if result.doc_name in context.recent_documents:
            base_recency += 0.3
        
        # Boost for recently queried content
        query_words = set(context.query.lower().split())
        for recent_query in self.query_history[-5:]:  # Check last 5 queries
            recent_words = set(recent_query.lower().split())
            overlap = len(query_words.intersection(recent_words))
            if overlap >= 2:
                base_recency += 0.2 * (overlap / len(query_words))
                break
        
        return min(base_recency, 1.0)
    
    def _calculate_document_type_score(
        self, 
        result: EnhancedSearchResult, 
        context: HybridRetrievalContext
    ) -> float:
        """Calculate document type preference score."""
        if not result.document_type:
            return 0.0
        
        # Document type preferences based on query content
        query_lower = context.query.lower()
        type_preferences = {
            'contract': 0.3 if any(word in query_lower for word in ['contract', 'agreement', 'provision']) else 0.0,
            'specification': 0.3 if any(word in query_lower for word in ['spec', 'requirement', 'standard']) else 0.0,
            'daily_log': 0.2 if any(word in query_lower for word in ['daily', 'log', 'timeline', 'when']) else 0.0,
            'email': 0.2 if any(word in query_lower for word in ['communication', 'notice', 'correspondence']) else 0.0,
            'report': 0.25 if any(word in query_lower for word in ['report', 'analysis', 'findings']) else 0.0,
            'drawing': 0.15 if any(word in query_lower for word in ['drawing', 'plan', 'diagram']) else 0.0,
            'other': 0.1
        }
        
        return type_preferences.get(result.document_type, type_preferences['other'])
    
    def _infer_document_type(self, doc_name: str) -> str:
        """Infer document type from filename."""
        doc_lower = doc_name.lower()
        
        if any(term in doc_lower for term in ['contract', 'agreement']):
            return 'contract'
        elif any(term in doc_lower for term in ['spec', 'specification']):
            return 'specification'
        elif any(term in doc_lower for term in ['daily', 'log']):
            return 'daily_log'
        elif any(term in doc_lower for term in ['email', 'correspondence']):
            return 'email'
        elif any(term in doc_lower for term in ['report', 'analysis']):
            return 'report'
        elif any(term in doc_lower for term in ['drawing', 'plan', 'dwg']):
            return 'drawing'
        else:
            return 'other'
    
    def _find_related_memory_items(
        self, 
        result: EnhancedSearchResult, 
        memory_items: List[KnowledgeItem]
    ) -> List[str]:
        """Find memory items related to this search result."""
        related = []
        result_text_lower = result.text.lower()
        
        for item in memory_items:
            # Check for direct mentions
            if item.label.lower() in result_text_lower:
                related.append(item.label)
                continue
            
            # Check for actor mentions
            for actor in item.actors:
                if actor.lower() in result_text_lower:
                    related.append(item.label)
                    break
            
            # Check document references
            for doc_ref in item.doc_refs:
                if (doc_ref.get('doc', '').lower() in result.doc_name.lower() and 
                    result.page_start <= doc_ref.get('page', 0) <= result.page_end):
                    related.append(item.label)
                    break
        
        return list(set(related))  # Remove duplicates
    
    def _apply_mmr(
        self, 
        results: List[EnhancedSearchResult], 
        context: HybridRetrievalContext,
        k: int,
        diversity_lambda: float = 0.7
    ) -> List[EnhancedSearchResult]:
        """Apply Maximal Marginal Relevance for diversity optimization."""
        if len(results) <= k:
            return results
        
        # Sort by score initially
        sorted_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        
        # MMR algorithm
        selected = [sorted_results[0]]  # Start with highest scoring
        candidates = sorted_results[1:]
        
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for candidate in candidates:
                # Relevance score (normalized)
                relevance = candidate.final_score
                
                # Maximum similarity to already selected items
                max_similarity = 0.0
                for selected_item in selected:
                    similarity = self._calculate_content_similarity(candidate, selected_item)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                mmr_scores.append((candidate, mmr_score))
            
            # Select best MMR candidate
            best_candidate, best_mmr_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        
        # Update diversity scores for logging
        for i, result in enumerate(selected):
            result.diversity_score = 1.0 - (i * 0.1)  # Decreasing diversity bonus
        
        logger.debug(
            "MMR diversity optimization applied",
            original_count=len(results),
            selected_count=len(selected),
            diversity_lambda=diversity_lambda
        )
        
        return selected
    
    def _calculate_content_similarity(
        self, 
        result1: EnhancedSearchResult, 
        result2: EnhancedSearchResult
    ) -> float:
        """Calculate content similarity between two results."""
        
        # Document similarity
        doc_similarity = 1.0 if result1.doc_name == result2.doc_name else 0.0
        
        # Page proximity similarity
        page_distance = abs(result1.page_start - result2.page_start)
        page_similarity = max(0.0, 1.0 - page_distance / 20.0)  # Decay over 20 pages
        
        # Text content similarity (simplified word overlap)
        words1 = set(result1.text.lower().split())
        words2 = set(result2.text.lower().split())
        
        if not words1 or not words2:
            text_similarity = 0.0
        else:
            overlap = len(words1.intersection(words2))
            union = len(words1.union(words2))
            text_similarity = overlap / union if union > 0 else 0.0
        
        # Combined similarity
        combined_similarity = (
            0.3 * doc_similarity +
            0.3 * page_similarity +
            0.4 * text_similarity
        )
        
        return combined_similarity
    
    def _update_access_history(
        self, 
        results: List[EnhancedSearchResult], 
        context: HybridRetrievalContext
    ):
        """Update access history for future recency calculations."""
        current_time = datetime.now()
        
        # Update document access times
        for result in results:
            doc_key = result.doc_name.lower()
            self.document_access_history[doc_key] = current_time
            result.last_accessed = current_time
        
        # Update query history (keep last 10 queries)
        self.query_history.append(context.query)
        if len(self.query_history) > 10:
            self.query_history = self.query_history[-10:]
        
        # Clean old access history (older than 90 days)
        cutoff_date = current_time - timedelta(days=90)
        self.document_access_history = {
            doc: access_time for doc, access_time in self.document_access_history.items()
            if access_time > cutoff_date
        }
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        return {
            'weights': {
                'vector_similarity': self.weights.vector_similarity,
                'memory_relevance': self.weights.memory_relevance,
                'recency_boost': self.weights.recency_boost,
                'diversity_factor': self.weights.diversity_factor
            },
            'access_history_size': len(self.document_access_history),
            'query_history_size': len(self.query_history),
            'temporal_decay_days': self.weights.temporal_decay_days
        }
    
    def update_weights(self, new_weights: RetrievalWeights):
        """Update retrieval weights for tuning."""
        self.weights = new_weights
        logger.info("Retrieval weights updated", weights=new_weights)
    
    def clear_history(self):
        """Clear access and query history."""
        self.document_access_history.clear()
        self.query_history.clear()
        logger.info("Retrieval history cleared")


def create_retrieval_context(
    query: str,
    matter_id: str,
    conversation_history: Optional[List[str]] = None,
    recent_documents: Optional[List[str]] = None,
    user_preferences: Optional[Dict[str, Any]] = None
) -> HybridRetrievalContext:
    """Convenience function to create retrieval context."""
    return HybridRetrievalContext(
        query=query,
        matter_id=matter_id,
        conversation_history=conversation_history or [],
        recent_documents=recent_documents or [],
        user_preferences=user_preferences or {}
    )