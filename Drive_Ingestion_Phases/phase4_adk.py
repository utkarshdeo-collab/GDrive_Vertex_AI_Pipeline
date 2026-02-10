"""
Phase 4 (ADK): Document Q&A using Google ADK with retrieval as a Tool.

Retrieval (Vector Search) is exposed as a tool. Gemini decides when to call it
and answers from the returned context. Run this script and type your question.

Usage:
  pip install google-adk   # if not already installed
  python phase4_adk.py
  Then type your question and press Enter.
"""
import asyncio
import json
import sys

import config
from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint,
    Namespace,
)

import vertexai
from vertexai.language_models import TextEmbeddingModel

# Lazy-loaded state for the retrieval tool
_chunk_cache = None
_embedding_model = None
_endpoint = None

MAX_CONTEXT_CHARS = 80_000
DEFAULT_TOP_K = 80


def _load_chunks():
    """Load chunk id -> text from GCS (cached)."""
    global _chunk_cache
    if _chunk_cache is not None:
        return _chunk_cache
    chunks_bucket = getattr(config, "CHUNKS_BUCKET", config.GCS_BUCKET_NAME)
    bucket = storage.Client(project=config.PROJECT_ID).bucket(chunks_bucket)
    blob = bucket.blob(f"{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl")
    if not blob.exists():
        return {}
    data = blob.download_as_text()
    _chunk_cache = {}
    for line in data.strip().split("\n"):
        if not line:
            continue
        obj = json.loads(line)
        _chunk_cache[obj["id"]] = obj.get("text", "")
    return _chunk_cache


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
        _embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
    return _embedding_model


def _get_endpoint():
    """Get the index endpoint by display name (case-insensitive)."""
    global _endpoint
    if _endpoint is None:
        aiplatform.init(project=config.PROJECT_ID, location=config.LOCATION)
        endpoints = list(matching_engine.MatchingEngineIndexEndpoint.list())
        target = (config.INDEX_ENDPOINT_DISPLAY_NAME or "").strip()
        resource_name = None
        for ep in endpoints:
            name = getattr(getattr(ep, "_gca_resource", None), "display_name", None) or ""
            if name.strip().lower() == target.lower():
                resource_name = ep.resource_name
                break
        if resource_name is None:
            available = [getattr(getattr(ep, "_gca_resource", None), "display_name", None) for ep in endpoints]
            available = [n for n in available if n]
            raise RuntimeError(
                f"No index endpoint with display name '{config.INDEX_ENDPOINT_DISPLAY_NAME}' found. "
                f"Available in {config.LOCATION}: {available or '(none)'}. "
                "Set INDEX_ENDPOINT_DISPLAY_NAME in config.py to one of these, or create an endpoint in Console (Vector Search → Index Endpoints)."
            )
        _endpoint = MatchingEngineIndexEndpoint(resource_name)
    return _endpoint


def search_document(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Search the indexed document and return relevant passages. Call with the user's
    question or with section/table names from the document. Useful search terms:
    Detailed Financial Benefits Realization, Reduced Readmissions; Phase 3 Key
    Activities, Phase 3 Milestones; Technology Stack, Comprehensive technology
    stack and cost allocation, Telehealth, Zoom for Healthcare; Integration
    Platform, MuleSoft, Epic EHR, Resolution Strategy; Department-Level Performance
    Analysis, Monthly Performance Trajectory; Lessons Learned, Clinical Advisory
    Board. Returns a dict with "context" and "num_passages".
    """
    print(f"\n>>> SEARCH_DOCUMENT CALLED: query={query[:80]!r}...", flush=True)
    try:
        return _search_document_impl(query, top_k)
    except Exception as e:
        import traceback
        print(f"\n>>> SEARCH_DOCUMENT EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        return {"context": "", "error": f"Exception in search_document: {e}"}


def _search_document_impl(query: str, top_k: int) -> dict:
    """Actual implementation of search_document."""
    print(">>> [1] Loading chunks...", flush=True)
    id_to_text = _load_chunks()
    if not id_to_text:
        print(f">>> [1] No chunks loaded from gs://{getattr(config, 'CHUNKS_BUCKET', config.GCS_BUCKET_NAME)}/{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl", flush=True)
        return {"context": "", "error": "No chunks loaded. Run Phase 2 first."}
    print(f">>> [1] Loaded {len(id_to_text)} chunks", flush=True)

    print(">>> [2] Getting embedding model...", flush=True)
    model = _get_embedding_model()
    print(">>> [2] Embedding query...", flush=True)
    embeddings = model.get_embeddings([query])
    query_embedding = list(embeddings[0].values)
    print(f">>> [2] Query embedded, dim={len(query_embedding)}", flush=True)

    print(">>> [3] Getting endpoint...", flush=True)
    endpoint = _get_endpoint()
    endpoint._sync_gca_resource()
    deployed_ids = [d.id for d in (getattr(endpoint, "deployed_indexes", None) or [])]
    print(f">>> [3] Endpoint has deployed_ids={deployed_ids}", flush=True)
    if deployed_ids and config.DEPLOYED_INDEX_ID not in deployed_ids:
        print(f">>> [3] DEPLOYED_INDEX_ID '{config.DEPLOYED_INDEX_ID}' not in {deployed_ids}", flush=True)
        return {
            "context": "",
            "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found on endpoint. Available: {deployed_ids}. Set config.DEPLOYED_INDEX_ID to one of these (e.g. from Console: Vector Search → Index Endpoints → your endpoint → Deployed indexes).",
        }

    print(">>> [4] Calling find_neighbors (with filter)...", flush=True)
    filter_ns = [Namespace(name="source", allow_tokens=["doc-pipeline"], deny_tokens=[])]
    try:
        results = endpoint.find_neighbors(
            deployed_index_id=config.DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=top_k,
            filter=filter_ns,
        )
    except NotFound as e:
        print(f">>> [4] NotFound exception: {e}", flush=True)
        return {
            "context": "",
            "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found on endpoint. In Console go to Vertex AI → Vector Search → Index Endpoints → your endpoint, and check the exact 'Deployed index' ID. Set config.DEPLOYED_INDEX_ID to that value. Details: {e}",
        }
    neighbors = results[0] if results else []
    print(f">>> [4] find_neighbors returned {len(neighbors)} neighbors", flush=True)
    if not neighbors:
        print(">>> [4] Retrying without filter...", flush=True)
        try:
            results = endpoint.find_neighbors(
                deployed_index_id=config.DEPLOYED_INDEX_ID,
                queries=[query_embedding],
                num_neighbors=top_k,
            )
            neighbors = results[0] if results else []
            print(f">>> [4] Retry returned {len(neighbors)} neighbors", flush=True)
        except Exception as retry_e:
            print(f">>> [4] Retry exception: {retry_e}", flush=True)

    print(">>> [5] Building context from neighbors...", flush=True)
    parts = []
    total = 0
    for neighbor in neighbors:
        nid = neighbor.id if isinstance(neighbor.id, str) else str(neighbor.id)
        text = id_to_text.get(nid, "")
        if not text:
            continue
        if total + len(text) > MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        total += len(text)
    context = "\n\n---\n\n".join(parts) if parts else ""

    print(f">>> [5] RESULT: chunks={len(id_to_text)} neighbors={len(neighbors)} passages={len(parts)} context_len={len(context)}", flush=True)
    if not context and neighbors:
        sample_ids = [getattr(n, "id", n) for n in neighbors[:5]]
        print(f"--- RETRIEVAL DEBUG: ID mismatch, sample neighbor IDs: {sample_ids} ---\n", flush=True)
        return {
            "context": "",
            "num_passages": 0,
            "error": f"Vector search returned {len(neighbors)} neighbors but none matched chunks in GCS. Chunk cache size: {len(id_to_text)}. Sample neighbor IDs: {sample_ids}. Check that chunks at gs://{getattr(config, 'CHUNKS_BUCKET', config.GCS_BUCKET_NAME)}/{config.CHUNK_OUTPUT_PREFIX}/chunks.jsonl match the indexed index (same Phase 2 run as Phase 3).",
        }
    if not context and not neighbors:
        print(f"--- RETRIEVAL DEBUG: Vector search returned 0 neighbors ---\n", flush=True)
        return {
            "context": "",
            "num_passages": 0,
            "error": "Vector search returned 0 neighbors. Check that the deployed index has embeddings (same index as Phase 3) and that the filter (source=doc-pipeline) matches the index restricts.",
        }
    return {"context": context, "num_passages": len(parts)}


def _extract_search_terms(question: str) -> list:
    """Extract multiple search terms from a question for targeted retrieval."""
    terms = [question]  # Always include the full question
    
    # Keyword to table/section mapping for targeted searches
    keyword_mappings = {
        "feature": ["Key features and functionality", "Symphony for Insurance features", "Preferred channels Virtual number Contact onboarding"],
        "functionality": ["Key features and functionality", "Symphony for Insurance"],
        "overtime": ["Monthly Performance Trajectory", "staff overtime hours baseline", "overtime reduction goal"],
        "telehealth": ["Technology Stack", "Telehealth Platform cost", "Zoom for Healthcare"],
        "staff resistance": ["Implementation Challenges and Resolutions", "Staff Resistance resolution strategy"],
        "technology": ["Technology Stack", "Comprehensive technology stack and cost allocation"],
        "cost": ["Technology Stack cost", "Total Implementation Cost", "Financial Benefits"],
        "resolution": ["Implementation Challenges and Resolutions", "Resolution Strategy"],
        "challenge": ["Implementation Challenges and Resolutions"],
        "financial": ["Detailed Financial Benefits Realization", "Financial Benefits"],
        "performance": ["Monthly Performance Trajectory", "Department-Level Performance Analysis"],
        "satisfaction": ["Patient Satisfaction Score", "Monthly Performance Trajectory"],
        "readmission": ["Reduced Readmissions", "Readmission rates"],
        "migration": ["Data migration", "migration success rate"],
        "data quality": ["Legacy Data Quality", "data quality activities"],
        "timeline": ["Implementation Timeline", "Phase timeline"],
        "lessons": ["Lessons Learned"],
        "guarantee": ["job security guarantees", "staff guarantees"],
    }
    
    question_lower = question.lower()
    for keyword, searches in keyword_mappings.items():
        if keyword in question_lower:
            terms.extend(searches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_terms.append(t)
    
    return unique_terms[:8]  # Limit to 8 searches max


def comprehensive_search(question: str) -> str:
    """Run multiple targeted searches and combine results."""
    search_terms = _extract_search_terms(question)
    print(f"\n>>> COMPREHENSIVE SEARCH: {len(search_terms)} search terms extracted", flush=True)
    
    all_context_parts = []
    seen_passages = set()
    
    for term in search_terms:
        result = search_document(term, top_k=60)
        context = result.get("context", "")
        if context:
            # Add unique passages only
            for passage in context.split("\n\n---\n\n"):
                passage_hash = hash(passage[:200]) if len(passage) > 200 else hash(passage)
                if passage_hash not in seen_passages:
                    seen_passages.add(passage_hash)
                    all_context_parts.append(passage)
    
    # Limit total context
    combined = []
    total_len = 0
    for p in all_context_parts:
        if total_len + len(p) > MAX_CONTEXT_CHARS:
            break
        combined.append(p)
        total_len += len(p)
    
    print(f">>> COMPREHENSIVE SEARCH RESULT: {len(combined)} unique passages, {total_len} chars", flush=True)
    return "\n\n---\n\n".join(combined)


def _create_agent():
    """Create ADK Agent with search_document tool."""
    from google.adk.agents.llm_agent import Agent

    return Agent(
        model=config.GEMINI_MODEL,
        name="doc_agent",
        description="Answers questions about the document by searching it and using the retrieved passages.",
        instruction="""You answer questions about a document. The search has already been performed and the relevant passages are provided in the conversation.

RULES:
1. Answer ONLY from the provided passages - never make up information
2. Quote specific numbers, percentages, dollar amounts, and facts directly from the text
3. If the passages don't contain the answer, say "The document does not contain this information"
4. For multi-part questions, address each part separately
5. When you find numerical data in tables, state the exact values (e.g., "8,400 hours/month baseline", "$0.08M")
6. Show your calculations when computing percentages or comparisons""",
        tools=[search_document],
    )


async def _run_once(runner, session_id: str, user_id: str, question: str) -> str:
    """Run the agent with one user message and return the final text response."""
    from google.genai import types

    out_parts = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part.from_text(text=question)]),
    ):
        # Event may be ADK Event with .content.parts[].text or similar
        content = getattr(event, "content", None)
        if content and getattr(content, "parts", None):
            for part in content.parts:
                text = getattr(part, "text", None)
                if text:
                    out_parts.append(text)
        elif content and getattr(content, "text", None):
            out_parts.append(content.text)
        if hasattr(event, "text") and event.text:
            out_parts.append(event.text)
    return "\n".join(out_parts).strip() if out_parts else "(No response)"


async def main():
    print("\n" + "=" * 60)
    print("  PHASE 4: ADK — Document Q&A (retrieval as Tool)")
    print("=" * 60)

    from google.adk.runners import InMemoryRunner

    root_agent = _create_agent()
    runner = InMemoryRunner(
        agent=root_agent,
        app_name="doc-pipeline-adk",
    )

    user_id = "user"
    session = await runner.session_service.create_session(
        user_id=user_id,
        app_name="doc-pipeline-adk",
    )
    session_id = session.id

    question = input("\nEnter your question: ").strip()
    if not question:
        print("No question provided.")
        sys.exit(1)

    print("\nPre-fetching relevant passages...")
    context = comprehensive_search(question)
    
    # Build the full prompt with pre-fetched context
    full_prompt = f"""QUESTION: {question}

RELEVANT PASSAGES FROM THE DOCUMENT:
{context}

Based on the passages above, please answer the question. Quote specific numbers and facts from the text."""

    print("\nRunning agent with pre-fetched context...")
    answer = await _run_once(runner, session_id, user_id, full_prompt)

    print("\n" + "-" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(answer)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    vertexai.init(project=config.PROJECT_ID, location=config.LOCATION)
    asyncio.run(main())
