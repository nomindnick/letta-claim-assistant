"""
Additional job type implementations for the enhanced job queue.

Provides implementations for batch processing, large model operations,
and matter analysis jobs.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from .logging_conf import get_logger
from .models import Matter
from .matters import matter_manager

logger = get_logger(__name__)


async def execute_batch_document_processing(
    job_id: str,
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """
    Execute batch document processing job.
    
    Processes multiple PDF files across one or more matters with
    optimized resource usage.
    """
    try:
        # Extract parameters
        file_batches = params.get("file_batches", [])
        force_ocr = params.get("force_ocr", False)
        ocr_language = params.get("ocr_language", "eng")
        max_concurrent_files = params.get("max_concurrent_files", 3)
        
        if not file_batches:
            raise ValueError("No file batches provided for processing")
        
        progress_callback(0.05, "Starting batch document processing")
        
        total_files = sum(len(batch["files"]) for batch in file_batches)
        processed_files = 0
        batch_results = []
        
        # Process each batch
        for batch_idx, batch in enumerate(file_batches):
            matter_id = batch["matter_id"]
            files = batch["files"]
            
            matter = matter_manager.get_matter_by_id(matter_id)
            if not matter:
                logger.warning("Matter not found, skipping batch", matter_id=matter_id)
                continue
            
            progress_callback(
                0.05 + (batch_idx / len(file_batches)) * 0.9,
                f"Processing matter {matter.name} ({len(files)} files)"
            )
            
            # Import here to avoid circular imports
            from .ingest import IngestionPipeline
            pipeline = IngestionPipeline(matter)
            
            # Process files with concurrency limit
            batch_results_detail = {}
            semaphore = asyncio.Semaphore(max_concurrent_files)
            
            async def process_file(file_path: Path):
                async with semaphore:
                    try:
                        def file_progress(progress: float, message: str):
                            nonlocal processed_files
                            file_complete = progress >= 1.0
                            if file_complete:
                                processed_files += 1
                            
                            overall_progress = 0.05 + (processed_files / total_files) * 0.9
                            progress_callback(
                                overall_progress,
                                f"Processing {file_path.name}: {message}"
                            )
                        
                        result = await pipeline.ingest_pdfs(
                            pdf_files=[file_path],
                            force_ocr=force_ocr,
                            ocr_language=ocr_language,
                            progress_callback=file_progress
                        )
                        
                        batch_results_detail[str(file_path)] = result.get(file_path, {})
                        
                    except Exception as e:
                        logger.error("Failed to process file in batch", file=str(file_path), error=str(e))
                        batch_results_detail[str(file_path)] = {
                            "success": False,
                            "error": str(e)
                        }
            
            # Process all files in current batch concurrently
            file_paths = [Path(f) for f in files]
            await asyncio.gather(*[process_file(fp) for fp in file_paths], return_exceptions=True)
            
            batch_results.append({
                "matter_id": matter_id,
                "matter_name": matter.name,
                "files_processed": len(batch_results_detail),
                "files_successful": sum(1 for r in batch_results_detail.values() 
                                      if r.get("success", False)),
                "results": batch_results_detail
            })
        
        # Calculate final statistics
        total_successful = sum(b["files_successful"] for b in batch_results)
        total_failed = total_files - total_successful
        
        progress_callback(1.0, f"Batch processing complete: {total_successful}/{total_files} files successful")
        
        return {
            "total_files": total_files,
            "successful_files": total_successful,
            "failed_files": total_failed,
            "batch_count": len(file_batches),
            "batch_results": batch_results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Batch document processing failed", job_id=job_id, error=str(e))
        raise


async def execute_large_model_operation(
    job_id: str,
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """
    Execute large model operations like bulk re-embedding or batch Q&A.
    """
    try:
        operation_type = params.get("operation_type")
        matter_id = params.get("matter_id")
        
        if not operation_type:
            raise ValueError("operation_type is required")
        
        matter = matter_manager.get_matter_by_id(matter_id) if matter_id else None
        
        progress_callback(0.05, f"Starting {operation_type} operation")
        
        if operation_type == "bulk_re_embedding":
            return await _execute_bulk_re_embedding(matter, params, progress_callback)
        elif operation_type == "batch_qa":
            return await _execute_batch_qa(matter, params, progress_callback)
        elif operation_type == "model_update":
            return await _execute_model_update(matter, params, progress_callback)
        else:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
    except Exception as e:
        logger.error("Large model operation failed", job_id=job_id, error=str(e))
        raise


async def _execute_bulk_re_embedding(
    matter: Optional[Matter],
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Re-embed all chunks in a matter with a new embedding model."""
    if not matter:
        raise ValueError("Matter is required for bulk re-embedding")
    
    new_embedding_model = params.get("new_embedding_model")
    if not new_embedding_model:
        raise ValueError("new_embedding_model is required")
    
    from .vectors import VectorStore
    from .llm.embeddings import get_embedding_manager
    
    # Get current vector store and embedding manager
    vector_store = VectorStore(matter.paths.vectors)
    embedding_manager = get_embedding_manager()
    
    progress_callback(0.1, "Loading existing chunks")
    
    # Get all chunks (this is a simplified approach - in real implementation
    # you'd want to batch this for large collections)
    all_chunks = await vector_store.get_all_chunks()
    total_chunks = len(all_chunks)
    
    if total_chunks == 0:
        progress_callback(1.0, "No chunks found to re-embed")
        return {"chunks_processed": 0, "embedding_model": new_embedding_model}
    
    progress_callback(0.2, f"Re-embedding {total_chunks} chunks with {new_embedding_model}")
    
    # Process chunks in batches
    batch_size = 50
    processed_chunks = 0
    
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        
        # Extract texts
        texts = [chunk.get("text", "") for chunk in batch]
        
        # Generate new embeddings
        embeddings = await embedding_manager.embed_texts(texts)
        
        # Update vector store with new embeddings
        for chunk, embedding in zip(batch, embeddings):
            await vector_store.update_chunk_embedding(chunk["id"], embedding)
        
        processed_chunks += len(batch)
        
        progress = 0.2 + (processed_chunks / total_chunks) * 0.7
        progress_callback(progress, f"Re-embedded {processed_chunks}/{total_chunks} chunks")
        
        # Brief pause to prevent overwhelming the system
        await asyncio.sleep(0.1)
    
    # Update matter configuration
    matter_config = matter.model_dump()
    matter_config["embedding_model"] = new_embedding_model
    
    # Save updated config (simplified - would use proper matter management)
    config_path = matter.paths.root / "config.json"
    import json
    with open(config_path, "w") as f:
        json.dump(matter_config, f, indent=2, default=str)
    
    progress_callback(1.0, f"Bulk re-embedding complete: {processed_chunks} chunks updated")
    
    return {
        "chunks_processed": processed_chunks,
        "embedding_model": new_embedding_model,
        "matter_id": matter.id,
        "completed_at": datetime.utcnow().isoformat()
    }


async def _execute_batch_qa(
    matter: Optional[Matter],
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Execute multiple Q&A queries in batch."""
    if not matter:
        raise ValueError("Matter is required for batch Q&A")
    
    questions = params.get("questions", [])
    if not questions:
        raise ValueError("questions list is required")
    
    from .rag import RAGEngine
    
    rag_engine = RAGEngine()
    total_questions = len(questions)
    results = []
    
    progress_callback(0.05, f"Processing {total_questions} questions")
    
    for i, question in enumerate(questions):
        try:
            # Generate answer
            response = await rag_engine.generate_answer(
                query=question,
                matter=matter,
                k=8
            )
            
            results.append({
                "question": question,
                "answer": response.answer,
                "sources_count": len(response.sources),
                "success": True
            })
            
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
        
        progress = 0.05 + ((i + 1) / total_questions) * 0.95
        progress_callback(progress, f"Processed {i + 1}/{total_questions} questions")
        
        # Brief pause between questions
        await asyncio.sleep(0.5)
    
    successful_answers = sum(1 for r in results if r["success"])
    
    progress_callback(1.0, f"Batch Q&A complete: {successful_answers}/{total_questions} successful")
    
    return {
        "total_questions": total_questions,
        "successful_answers": successful_answers,
        "results": results,
        "matter_id": matter.id,
        "completed_at": datetime.utcnow().isoformat()
    }


async def _execute_model_update(
    matter: Optional[Matter],
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Update the LLM model for a matter and test connectivity."""
    new_model = params.get("new_model")
    provider = params.get("provider", "ollama")
    
    if not new_model:
        raise ValueError("new_model is required")
    
    progress_callback(0.1, f"Testing connectivity to {new_model}")
    
    try:
        from .llm.provider_manager import provider_manager
        
        # Test model connectivity
        test_successful = await provider_manager.test_provider_model(provider, new_model)
        
        if not test_successful:
            raise Exception(f"Failed to connect to model {new_model}")
        
        progress_callback(0.5, "Model connectivity confirmed")
        
        # Update matter configuration if matter is specified
        if matter:
            matter_config = matter.model_dump()
            matter_config["generation_model"] = new_model
            
            config_path = matter.paths.root / "config.json"
            import json
            with open(config_path, "w") as f:
                json.dump(matter_config, f, indent=2, default=str)
            
            progress_callback(0.8, "Matter configuration updated")
        
        progress_callback(1.0, f"Model update complete: {new_model}")
        
        return {
            "new_model": new_model,
            "provider": provider,
            "test_successful": test_successful,
            "matter_id": matter.id if matter else None,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Model update failed", new_model=new_model, error=str(e))
        raise


async def execute_matter_analysis(
    job_id: str,
    params: Dict[str, Any],
    progress_callback: Callable[[float, str], None]
) -> Dict[str, Any]:
    """
    Analyze all documents in a matter and generate comprehensive insights.
    """
    try:
        matter_id = params.get("matter_id")
        analysis_types = params.get("analysis_types", ["overview", "timeline", "entities"])
        
        if not matter_id:
            raise ValueError("matter_id is required")
        
        matter = matter_manager.get_matter_by_id(matter_id)
        if not matter:
            raise ValueError(f"Matter not found: {matter_id}")
        
        progress_callback(0.05, f"Starting analysis of matter: {matter.name}")
        
        from .rag import RAGEngine
        from .letta_adapter import LettaAdapter
        
        rag_engine = RAGEngine()
        results = {}
        
        # Overview analysis
        if "overview" in analysis_types:
            progress_callback(0.2, "Generating matter overview")
            
            overview_query = "Provide a comprehensive overview of this construction claim, including key parties, project details, and main issues."
            overview_response = await rag_engine.generate_answer(overview_query, matter, k=12)
            
            results["overview"] = {
                "summary": overview_response.answer,
                "sources_count": len(overview_response.sources),
                "key_entities": overview_response.knowledge_items[:10] if overview_response.knowledge_items else []
            }
        
        # Timeline analysis
        if "timeline" in analysis_types:
            progress_callback(0.5, "Building project timeline")
            
            timeline_query = "Create a chronological timeline of key events, milestones, and issues in this construction project."
            timeline_response = await rag_engine.generate_answer(timeline_query, matter, k=15)
            
            # Extract dates and events from knowledge items
            events = []
            if timeline_response.knowledge_items:
                for item in timeline_response.knowledge_items:
                    if item.type == "Event" and item.date:
                        events.append({
                            "date": item.date,
                            "event": item.label,
                            "actors": item.actors,
                            "source": item.doc_refs[0] if item.doc_refs else None
                        })
            
            results["timeline"] = {
                "narrative": timeline_response.answer,
                "events": sorted(events, key=lambda x: x["date"]) if events else [],
                "sources_count": len(timeline_response.sources)
            }
        
        # Entity analysis
        if "entities" in analysis_types:
            progress_callback(0.8, "Identifying key entities and relationships")
            
            # Get all knowledge items from Letta
            letta_adapter = LettaAdapter(matter.paths.root)
            knowledge_items = await letta_adapter.recall("", top_k=100)  # Get all items
            
            entities_by_type = {}
            for item in knowledge_items:
                if item.type not in entities_by_type:
                    entities_by_type[item.type] = []
                
                entities_by_type[item.type].append({
                    "label": item.label,
                    "date": item.date,
                    "actors": item.actors,
                    "doc_refs_count": len(item.doc_refs) if item.doc_refs else 0
                })
            
            results["entities"] = entities_by_type
        
        # Calculate final statistics
        total_docs = len(list((matter.paths.docs).glob("*.pdf")))
        
        progress_callback(1.0, f"Matter analysis complete for {matter.name}")
        
        return {
            "matter_id": matter.id,
            "matter_name": matter.name,
            "analysis_types": analysis_types,
            "total_documents": total_docs,
            "results": results,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Matter analysis failed", job_id=job_id, matter_id=matter_id, error=str(e))
        raise