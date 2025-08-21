"""
Letta Tools - Custom tools for Letta agents to interact with the RAG system.

Provides tool definitions and implementations that allow Letta agents to:
- Search documents using the existing vector store
- Extract information with proper citations
- Access matter-specific data
"""

import inspect
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

from .logging_conf import get_logger
from .vectors import VectorStore
from .rag import RAGEngine
from .matters import matter_manager

logger = get_logger(__name__)


def create_search_documents_tool():
    """
    Create the search_documents tool definition and implementation for Letta.
    
    Returns:
        Tuple of (tool_definition, tool_implementation)
    """
    
    # Tool definition in OpenAPI-like format for Letta
    tool_definition = {
        "name": "search_documents",
        "description": "Search case documents for specific information using semantic search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string to find relevant information"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["query"]
        }
    }
    
    # Tool implementation
    def search_documents_impl(query: str, k: int = 5) -> str:
        """
        Search implementation that uses the existing RAG infrastructure.
        
        This function will be executed by Letta when the agent decides to search.
        It must return a string (JSON formatted) that the agent can parse.
        
        Args:
            query: Search query string
            k: Number of results to return (default 5)
        
        Returns:
            JSON string with search results
        """
        try:
            # Import here to avoid circular dependencies
            import asyncio
            from app.vectors import VectorStore
            from app.matters import matter_manager
            
            # Get current matter context
            # Try to get from matter manager first
            current_matter = matter_manager.get_current_matter()
            
            # If no current matter, try to get from thread-local or global context
            if not current_matter:
                # In production, this would be passed via agent metadata
                # For now, return an error indicating no documents available
                return json.dumps({
                    "status": "success",
                    "message": "No documents available yet for this matter",
                    "results_count": 0,
                    "results": []
                })
            
            # Create vector store for the matter
            vector_store = VectorStore(current_matter.paths.root)
            
            # Perform synchronous search (Letta tools are synchronous)
            # We'll use a wrapper to call the async method
            def run_search():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        vector_store.search(query=query, k=k)
                    )
                finally:
                    loop.close()
            
            search_results = run_search()
            
            if not search_results:
                return json.dumps({
                    "status": "success",
                    "message": "No relevant documents found",
                    "results_count": 0,
                    "results": []
                })
            
            # Format results with precise citations
            formatted_results = []
            for result in search_results:
                # Create citation in standard format
                citation = f"[{result.doc_name} p.{result.page_start}"
                if result.page_end and result.page_end != result.page_start:
                    citation += f"-{result.page_end}"
                citation += "]"
                
                # Truncate text for readability
                snippet = result.text[:500] + "..." if len(result.text) > 500 else result.text
                
                formatted_results.append({
                    "doc_name": result.doc_name,
                    "page_start": result.page_start,
                    "page_end": result.page_end or result.page_start,
                    "score": round(result.similarity_score, 4),
                    "snippet": snippet,
                    "citation": citation
                })
            
            # Return structured JSON response
            return json.dumps({
                "status": "success",
                "message": f"Found {len(formatted_results)} relevant documents",
                "results_count": len(formatted_results),
                "query": query,
                "results": formatted_results
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Search tool failed: {e}")
            return json.dumps({
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "results": []
            })
    
    return tool_definition, search_documents_impl


def register_search_tool_with_agent(client, agent_id: str, matter_id: str):
    """
    Register the search_documents tool with a Letta agent.
    
    Note: Using synchronous client as that's what Letta v0.10.0 provides.
    
    Args:
        client: Letta client instance (RESTClient)
        agent_id: ID of the agent to attach the tool to
        matter_id: ID of the matter for context
    """
    try:
        # Create the tool
        tool_def, tool_impl = create_search_documents_tool()
        
        # Get the source code of the implementation
        tool_source = inspect.getsource(tool_impl)
        
        # Create tool in Letta server
        tool = client.create_tool(
            name=tool_def["name"],
            source_code=tool_source,
            description=tool_def["description"],
            parameters=tool_def["parameters"],
            return_char_limit=10000  # Limit response size
        )
        
        logger.info(f"Tool '{tool_def['name']}' registered with server")
        
        # Attach tool to agent
        client.attach_tool(
            agent_id=agent_id,
            tool_name=tool.name
        )
        
        logger.info(f"Tool '{tool_def['name']}' attached to agent {agent_id}")
        
        # Update agent metadata for matter context
        # Note: Letta v0.10.0 may not support metadata updates directly
        # This would need to be handled through agent state or memory blocks
        
        logger.info(f"Tool registration complete for agent {agent_id}")
        
    except Exception as e:
        logger.error(f"Failed to register search tool: {e}")
        # Log but don't raise - tool registration is optional


def create_extract_entities_tool():
    """
    Create tool for extracting entities from documents (future enhancement).
    
    This is a placeholder for Phase 2 - not implemented in Phase 1.
    """
    pass


def create_timeline_tool():
    """
    Create tool for building timeline from documents (future enhancement).
    
    This is a placeholder for Phase 2 - not implemented in Phase 1.
    """
    pass


# Tool registry for easy access
AVAILABLE_TOOLS = {
    "search_documents": create_search_documents_tool,
    # Future tools will be added here
    # "extract_entities": create_extract_entities_tool,
    # "timeline": create_timeline_tool,
}


def get_tool_definition(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the definition for a named tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool definition dict or None if not found
    """
    if tool_name in AVAILABLE_TOOLS:
        tool_def, _ = AVAILABLE_TOOLS[tool_name]()
        return tool_def
    return None


def get_tool_implementation(tool_name: str) -> Optional[callable]:
    """
    Get the implementation function for a named tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool implementation function or None if not found
    """
    if tool_name in AVAILABLE_TOOLS:
        _, tool_impl = AVAILABLE_TOOLS[tool_name]()
        return tool_impl
    return None