"""
Prompt templates for construction claims analysis.

Contains system prompts, formatting templates, and instructions for
consistent LLM behavior across the application.
"""

from typing import List, Dict, Any
from .vectors import SearchResult
from .models import KnowledgeItem

# System prompt for construction claims analysis
SYSTEM_PROMPT = """You are a construction claims analyst assisting an attorney.

You must analyze construction claims and disputes using the MEMORY and DOC contexts provided below. Your analysis must be:

1. CONSERVATIVE AND PRECISE: Only state facts that can be directly supported by the provided documents. If information is uncertain or sources conflict, explicitly say so.

2. PROPERLY CITED: Every key point must be supported with citations in the exact format [DocName p.N] where DocName is the document filename and N is the page number. Do not invent citations.

3. CONSTRUCTION-FOCUSED: Apply knowledge of construction law, contracts, industry practices, schedule analysis, change orders, and damage assessment.

4. TRACEABLE: Each citation must correspond to a real document and page provided in the context.

Follow the exact output format specified in the user prompt. Do not deviate from the required structure."""

# Output format template
OUTPUT_FORMAT_INSTRUCTION = """
You must structure your response in exactly this format:

## Key Points
- [First key finding with citation]
- [Second key finding with citation]
- [Additional findings as needed]

## Analysis
[Detailed analysis explaining the significance of the findings, potential legal implications, and connections between different pieces of evidence. Include discussion of construction industry context where relevant.]

## Citations
- [DocName.pdf p.N] - Brief description of what this source supports
- [DocName.pdf p.M] - Brief description of what this source supports
- [Additional citations as needed]

## Suggested Follow-ups
- [Specific question that would help clarify legal issues]
- [Question about additional documentation needed]
- [Question about expert analysis or technical investigation]
- [Question about damages or remediation]
"""

# Template for formatting document context
DOC_CONTEXT_TEMPLATE = """DOC[{index}]: {doc_name} (p.{page_start}-{page_end}, similarity: {similarity:.3f})
{text}

"""

# Template for formatting memory context
MEMORY_CONTEXT_TEMPLATE = """MEMORY[{index}]: {type} - {label}
{details}

"""

# Information extraction prompt for knowledge items
INFORMATION_EXTRACTION_PROMPT = """Based on the user's question and the assistant's answer, extract structured knowledge items that should be stored in the agent's memory.

Extract items as a JSON array where each item has:
- type: "Entity" | "Event" | "Issue" | "Fact"
- label: string (concise description)
- date: ISO8601 string or null
- actors: array of strings (people, companies involved)
- doc_refs: array of objects with {doc: string, page: number}
- support_snippet: string (<=300 chars supporting text) or null

Focus on:
- ENTITIES: Parties (Owner, Contractor, Subcontractor), People, Organizations, Projects, Locations
- EVENTS: Delays, RFIs, Change Orders, Inspections, Failures, Payments, Non-Conformances
- ISSUES: Differing Site Conditions, Design Defects, Schedule Delays, Extra Work, Liquidated Damages
- FACTS: Specific findings that establish timeline, causation, or responsibility

Return only valid JSON array. No other text."""

# Follow-up suggestions prompt
FOLLOWUP_SUGGESTIONS_PROMPT = """Based on the user's question, the generated answer, and the memory context, propose 2-4 specific follow-up questions that would help a construction claims attorney:

1. Uncover causation and responsibility
2. Identify damages and remedies
3. Find additional evidence or documentation
4. Clarify technical or legal issues

Each suggestion should be:
- Specific and actionable (≤18 words)
- Grounded in the current context
- Focused on advancing the legal analysis
- Relevant to construction claims practice

Return only the question text, one per line. No formatting or numbering."""


def format_doc_context(search_results: List[SearchResult]) -> str:
    """Format search results into document context for prompt."""
    if not search_results:
        return ""
    
    context_blocks = []
    for i, result in enumerate(search_results):
        context_block = DOC_CONTEXT_TEMPLATE.format(
            index=i + 1,
            doc_name=result.doc_name,
            page_start=result.page_start,
            page_end=result.page_end,
            similarity=result.similarity_score,
            text=result.text
        )
        context_blocks.append(context_block)
    
    return "".join(context_blocks)


def format_memory_context(memory_items: List[KnowledgeItem]) -> str:
    """Format memory items into context for prompt."""
    if not memory_items:
        return ""
    
    context_blocks = []
    for i, item in enumerate(memory_items):
        details_parts = []
        
        if item.date:
            details_parts.append(f"Date: {item.date}")
        
        if item.actors:
            details_parts.append(f"Actors: {', '.join(item.actors)}")
        
        if item.doc_refs:
            refs = [f"{ref.get('doc', 'Unknown')} p.{ref.get('page', '?')}" for ref in item.doc_refs]
            details_parts.append(f"References: {', '.join(refs)}")
        
        if item.support_snippet:
            details_parts.append(f"Context: {item.support_snippet}")
        
        details = " | ".join(details_parts) if details_parts else "No additional details"
        
        context_block = MEMORY_CONTEXT_TEMPLATE.format(
            index=i + 1,
            type=item.type,
            label=item.label,
            details=details
        )
        context_blocks.append(context_block)
    
    return "".join(context_blocks)


def assemble_rag_prompt(
    user_query: str,
    search_results: List[SearchResult],
    memory_items: List[KnowledgeItem]
) -> List[Dict[str, str]]:
    """Assemble complete RAG prompt with context."""
    
    # Format contexts
    doc_context = format_doc_context(search_results)
    memory_context = format_memory_context(memory_items)
    
    # Build context section
    context_parts = []
    
    if memory_context:
        context_parts.append("=== AGENT MEMORY ===")
        context_parts.append(memory_context)
    
    if doc_context:
        context_parts.append("=== DOCUMENT CONTEXT ===")
        context_parts.append(doc_context)
    
    if not context_parts:
        context_parts.append("=== NO CONTEXT AVAILABLE ===")
        context_parts.append("No relevant documents or memory items found. Please inform the user that you don't have sufficient information to answer their question.")
    
    # Combine all context
    full_context = "\n".join(context_parts)
    
    # Build user message with query and context
    user_message = f"""{user_query}

{full_context}

{OUTPUT_FORMAT_INSTRUCTION}"""
    
    return [
        {"role": "user", "content": user_message}
    ]


def extract_citations_from_answer(answer: str) -> List[str]:
    """Extract citation strings from answer text."""
    import re
    
    # Pattern to match [DocName p.N] or [DocName.pdf p.N-M] format
    citation_pattern = r'\[([\w\-\s\.]+\.(?:pdf|PDF))\s+p\.(\d+(?:-\d+)?)\]'
    
    matches = re.findall(citation_pattern, answer)
    citations = []
    
    for doc_name, page_range in matches:
        citation = f"[{doc_name} p.{page_range}]"
        if citation not in citations:  # Avoid duplicates
            citations.append(citation)
    
    return citations


def validate_citations(citations: List[str], search_results: List[SearchResult]) -> Dict[str, bool]:
    """Validate that citations correspond to provided sources."""
    citation_validity = {}
    
    # Create lookup of available sources
    available_sources = {}
    for result in search_results:
        doc_key = result.doc_name.lower()
        if doc_key not in available_sources:
            available_sources[doc_key] = []
        
        # Add all pages in range
        for page in range(result.page_start, result.page_end + 1):
            available_sources[doc_key].append(page)
    
    # Check each citation
    for citation in citations:
        citation_validity[citation] = False
        
        # Parse citation
        import re
        match = re.match(r'\[([\w\-\s\.]+\.(?:pdf|PDF))\s+p\.(\d+(?:-\d+)?)\]', citation)
        if not match:
            continue
        
        doc_name, page_range = match.groups()
        doc_key = doc_name.lower()
        
        if doc_key not in available_sources:
            continue
        
        # Parse page range
        try:
            if '-' in page_range:
                start_page, end_page = map(int, page_range.split('-'))
                cited_pages = list(range(start_page, end_page + 1))
            else:
                cited_pages = [int(page_range)]
            
            # Check if any cited page is available
            available_pages = available_sources[doc_key]
            if any(page in available_pages for page in cited_pages):
                citation_validity[citation] = True
                
        except ValueError:
            continue
    
    return citation_validity