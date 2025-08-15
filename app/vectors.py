"""
ChromaDB vector operations and search functionality.

Provides isolated vector storage per Matter with embedding generation
and similarity search capabilities.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import chromadb
from chromadb.config import Settings as ChromaSettings
import hashlib
import uuid
import time
from collections import OrderedDict

from .logging_conf import get_logger
from .chunking import Chunk
from .llm.embeddings import embedding_manager

logger = get_logger(__name__)


class SearchCache:
    """LRU cache for search results with TTL."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def _make_key(self, query: str, k: int, filter_metadata: Optional[Dict] = None) -> str:
        """Create cache key from search parameters."""
        filter_str = str(sorted(filter_metadata.items())) if filter_metadata else ""
        return f"{query}:{k}:{filter_str}"
    
    def get(self, query: str, k: int, filter_metadata: Optional[Dict] = None) -> Optional[List]:
        """Get cached results if valid."""
        key = self._make_key(query, k, filter_metadata)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, query: str, k: int, results: List, filter_metadata: Optional[Dict] = None):
        """Cache search results."""
        key = self._make_key(query, k, filter_metadata)
        
        # Remove oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        self.cache[key] = results
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.timestamps.clear()


@dataclass
class SearchResult:
    """Vector search result with metadata."""
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    text: str
    similarity_score: float
    metadata: Dict[str, Any]


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, recoverable: bool = True):
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


class VectorStore:
    """ChromaDB-based vector storage for a specific Matter with performance optimizations."""
    
    def __init__(self, matter_path: Path, collection_name: Optional[str] = None):
        self.matter_path = matter_path
        self.collection_path = matter_path / "vectors" / "chroma"
        self.collection_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize search cache
        self.search_cache = SearchCache(max_size=100, ttl_seconds=300)  # 5 minute TTL
        
        # Generate collection name from matter path if not provided
        if collection_name is None:
            matter_name = matter_path.name
            # Sanitize collection name for ChromaDB
            collection_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in matter_name.lower())
            if not collection_name or collection_name[0].isdigit():
                collection_name = f"matter_{collection_name}"
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistent storage
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.collection_path),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Get or create collection with cosine similarity
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(
                "VectorStore initialized",
                matter_path=str(matter_path),
                collection_name=self.collection_name,
                collection_path=str(self.collection_path),
                existing_count=self.collection.count()
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg, matter_path=str(matter_path))
            raise VectorStoreError(error_msg, recoverable=False)
    
    async def upsert_chunks(self, chunks: List[Chunk], batch_size: int = 250) -> None:
        """
        Upsert chunks with their embeddings into the vector store using optimized batching.
        
        Args:
            chunks: List of Chunk objects to store
            batch_size: Size of batches for processing (increased from 100 to 250)
        """
        if not chunks:
            logger.debug("No chunks provided for upserting")
            return
        
        logger.info(
            "Starting optimized chunk upsert",
            matter_path=str(self.matter_path),
            chunk_count=len(chunks),
            batch_size=batch_size
        )
        
        try:
            # Check for existing chunks to avoid duplicates (optimized)
            existing_ids = set()
            if self.collection.count() > 0:
                try:
                    # Get only IDs for faster duplicate checking
                    existing_data = self.collection.get(include=[])
                    existing_ids = set(existing_data.get("ids", []))
                except Exception as e:
                    logger.warning("Failed to get existing chunk IDs", error=str(e))
            
            # Filter out duplicates first
            new_chunks = []
            skipped_count = 0
            
            for chunk in chunks:
                chunk_vector_id = f"chunk_{chunk.md5}"
                if chunk_vector_id not in existing_ids:
                    new_chunks.append((chunk, chunk_vector_id))
                else:
                    skipped_count += 1
            
            if not new_chunks:
                logger.info("All chunks already exist, skipping upsert", skipped_count=skipped_count)
                return
            
            logger.info(f"Processing {len(new_chunks)} new chunks (skipped {skipped_count} duplicates)")
            
            # Process in optimized batches
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                await self._upsert_batch(batch, i // batch_size + 1, len(new_chunks) // batch_size + 1)
                
                # Small delay to prevent overwhelming the system
                if len(new_chunks) > batch_size:
                    await asyncio.sleep(0.1)
            
            logger.info(
                "Chunk upsert completed",
                total_processed=len(new_chunks),
                total_skipped=skipped_count
            )
            
        except Exception as e:
            error_msg = f"Failed to upsert chunks: {str(e)}"
            logger.error(error_msg, matter_path=str(self.matter_path))
            raise VectorStoreError(error_msg)
    
    async def _upsert_batch(self, batch: List[tuple], batch_num: int, total_batches: int) -> None:
        """Upsert a single batch of chunks with optimized processing."""
        chunk_ids = []
        chunk_texts = []
        metadatas = []
        documents = []
        
        for chunk, chunk_vector_id in batch:
            chunk_ids.append(chunk_vector_id)
            chunk_texts.append(chunk.text)
            documents.append(chunk.text)
            
            # Prepare metadata for ChromaDB
            chunk_metadata = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "token_count": chunk.token_count,
                "char_count": getattr(chunk, 'char_count', len(chunk.text)),
                "md5": chunk.md5,
                "section_title": getattr(chunk, 'section_title', '') or "",
                "chunk_index": getattr(chunk, 'chunk_index', 0),
                "has_overlap": getattr(chunk, 'overlap_info', {}).get("has_overlap", False),
                "overlap_sentences": getattr(chunk, 'overlap_info', {}).get("overlap_sentences", 0)
            }
            
            # Add custom metadata fields
            if hasattr(chunk, 'metadata') and chunk.metadata:
                for key, value in chunk.metadata.items():
                    # ChromaDB metadata values must be strings, numbers, or bools
                    if isinstance(value, (str, int, float, bool)):
                        chunk_metadata[f"meta_{key}"] = value
                    else:
                        chunk_metadata[f"meta_{key}"] = str(value)
            
            metadatas.append(chunk_metadata)
        
        # Generate embeddings for batch
        logger.debug(f"Generating embeddings for batch {batch_num}/{total_batches}", count=len(chunk_texts))
        embeddings = await embedding_manager.embed(chunk_texts)
        
        if len(embeddings) != len(chunk_texts):
            raise VectorStoreError(
                f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunk_texts)}"
            )
        
        # Validate embeddings
        if not all(isinstance(emb, list) and len(emb) > 0 for emb in embeddings):
            raise VectorStoreError("Invalid embeddings received from embedding provider")
        
        # Upsert to ChromaDB
        logger.debug(f"Upserting batch {batch_num}/{total_batches} to ChromaDB", batch_size=len(chunk_ids))
        
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        
        logger.debug(f"Batch {batch_num}/{total_batches} upserted successfully")
    
    async def search(
        self,
        query: str,
        k: int = 8,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity with caching optimization.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            use_cache: Whether to use search cache
            
        Returns:
            List of SearchResult objects ordered by similarity
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []
        
        # Check cache first
        if use_cache:
            cached_results = self.search_cache.get(query, k, filter_metadata)
            if cached_results is not None:
                logger.debug("Returning cached search results", query=query[:50])
                return cached_results
        
        search_start = time.time()
        
        try:
            # Generate query embedding
            logger.debug("Generating query embedding", query_preview=query[:100])
            query_embedding = await embedding_manager.embed_single(query.strip())
            
            if not query_embedding:
                raise VectorStoreError("Failed to generate query embedding")
            
            # Prepare ChromaDB query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": min(k, 50),  # Cap at 50 for performance
                "include": ["documents", "metadatas", "distances"]
            }
            
            # Add metadata filters if provided
            if filter_metadata:
                # Convert filters to ChromaDB format
                chroma_filters = {}
                for key, value in filter_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_filters[key] = value
                    elif isinstance(value, list):
                        chroma_filters[key] = {"$in": value}
                
                if chroma_filters:
                    query_params["where"] = chroma_filters
            
            # Query ChromaDB
            logger.debug(
                "Querying ChromaDB",
                collection=self.collection_name,
                k=k,
                filters=bool(filter_metadata)
            )
            
            results = self.collection.query(**query_params)
            
            if not results or not results.get("ids") or not results["ids"][0]:
                logger.debug("No search results found")
                return []
            
            # Convert ChromaDB results to SearchResult objects
            search_results = []
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i, (result_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances)
            ):
                try:
                    # Convert distance to similarity score (cosine: 0=identical, 2=opposite)
                    similarity_score = max(0.0, 1.0 - (distance / 2.0))
                    
                    # Extract metadata
                    chunk_id = metadata.get("chunk_id", result_id)
                    doc_name = metadata.get("doc_name", "unknown")
                    page_start = metadata.get("page_start", 1)
                    page_end = metadata.get("page_end", 1)
                    
                    # Truncate text for display (keep full text for context)
                    display_text = document
                    if len(display_text) > 600:
                        display_text = display_text[:600] + "..."
                    
                    search_result = SearchResult(
                        chunk_id=chunk_id,
                        doc_name=doc_name,
                        page_start=int(page_start),
                        page_end=int(page_end),
                        text=display_text,
                        similarity_score=float(similarity_score),
                        metadata=metadata
                    )
                    
                    search_results.append(search_result)
                    
                except Exception as e:
                    logger.warning(
                        "Failed to parse search result",
                        index=i,
                        error=str(e),
                        result_id=result_id
                    )
                    continue
            
            logger.debug(
                "Search completed",
                query_preview=query[:100],
                results_count=len(search_results),
                avg_similarity=sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0
            )
            
            return search_results
            
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logger.error(
                error_msg,
                matter_path=str(self.matter_path),
                query_preview=query[:100] if query else "None"
            )
            raise VectorStoreError(error_msg, recoverable=True)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "collection_path": str(self.collection_path),
                "matter_path": str(self.matter_path)
            }
            
            if count > 0:
                # Get sample of data for statistics
                sample_size = min(100, count)
                sample_data = self.collection.get(
                    limit=sample_size,
                    include=["metadatas"]
                )
                
                if sample_data and sample_data.get("metadatas"):
                    metadatas = sample_data["metadatas"]
                    
                    # Document statistics
                    doc_names = set()
                    page_ranges = []
                    token_counts = []
                    sections = set()
                    
                    for metadata in metadatas:
                        if metadata.get("doc_name"):
                            doc_names.add(metadata["doc_name"])
                        
                        if metadata.get("page_start") and metadata.get("page_end"):
                            page_ranges.append((metadata["page_start"], metadata["page_end"]))
                        
                        if metadata.get("token_count"):
                            try:
                                token_counts.append(int(metadata["token_count"]))
                            except (ValueError, TypeError):
                                pass
                        
                        if metadata.get("section_title"):
                            sections.add(metadata["section_title"])
                    
                    stats.update({
                        "unique_documents": len(doc_names),
                        "document_names": list(doc_names),
                        "unique_sections": len(sections),
                        "page_range": {
                            "min_page": min(p[0] for p in page_ranges) if page_ranges else None,
                            "max_page": max(p[1] for p in page_ranges) if page_ranges else None
                        },
                        "token_stats": {
                            "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
                            "min_tokens": min(token_counts) if token_counts else 0,
                            "max_tokens": max(token_counts) if token_counts else 0
                        }
                    })
            
            logger.debug("Collection stats calculated", stats=stats)
            return stats
            
        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            logger.error(error_msg, matter_path=str(self.matter_path))
            raise VectorStoreError(error_msg, recoverable=True)
    
    async def delete_chunks(self, chunk_ids: List[str]) -> int:
        """
        Delete chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0
        
        try:
            # Convert chunk IDs to vector IDs if needed
            vector_ids = []
            for chunk_id in chunk_ids:
                if chunk_id.startswith("chunk_"):
                    vector_ids.append(chunk_id)
                else:
                    # Assume it's a regular chunk ID, need to find corresponding vector ID
                    # This requires a query to find the right vector ID by metadata
                    results = self.collection.get(
                        where={"chunk_id": chunk_id},
                        include=["metadatas"]
                    )
                    if results and results.get("ids"):
                        vector_ids.extend(results["ids"])
            
            if not vector_ids:
                logger.warning("No valid vector IDs found for deletion", chunk_ids=chunk_ids)
                return 0
            
            # Delete from ChromaDB
            self.collection.delete(ids=vector_ids)
            
            logger.info(
                "Chunks deleted",
                matter_path=str(self.matter_path),
                deleted_count=len(vector_ids)
            )
            
            return len(vector_ids)
            
        except Exception as e:
            error_msg = f"Failed to delete chunks: {str(e)}"
            logger.error(error_msg, matter_path=str(self.matter_path))
            raise VectorStoreError(error_msg, recoverable=True)
    
    async def reset_collection(self) -> None:
        """Reset (clear) the entire collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            
            logger.info(
                "Collection reset completed",
                matter_path=str(self.matter_path),
                collection_name=self.collection_name
            )
            
        except Exception as e:
            error_msg = f"Failed to reset collection: {str(e)}"
            logger.error(error_msg, matter_path=str(self.matter_path))
            raise VectorStoreError(error_msg, recoverable=True)