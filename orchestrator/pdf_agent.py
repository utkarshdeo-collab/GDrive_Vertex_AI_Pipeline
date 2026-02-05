"""
PDF sub-agent: self-contained document Q&A using Vector Search + chunks.
Uses root config.py for project, bucket, index, and model settings.
Exports create_pdf_agent() for the orchestrator.
"""
import json
import sys
from pathlib import Path

# Ensure project root is on path (for config)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
from google.adk.agents import LlmAgent
from google.adk.models import Gemini

# Lazy-loaded state for the retrieval tool
_chunk_cache = None
_embedding_model = None
_endpoint = None

MAX_CONTEXT_CHARS = 80_000
DEFAULT_TOP_K = 50


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
                "Set INDEX_ENDPOINT_DISPLAY_NAME in config.py to one of these."
            )
        _endpoint = MatchingEngineIndexEndpoint(resource_name)
    return _endpoint


def search_document(query: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """
    Search the indexed document and return relevant passages. Call with the user's
    question or with section/table names from the document. Returns a dict with
    "context" and "num_passages".
    """
    from .usage_collector import record_embedding
    record_embedding(len(query))
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
            "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found on endpoint. Available: {deployed_ids}. Set config.DEPLOYED_INDEX_ID to one of these.",
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
            "error": f"Deployed index '{config.DEPLOYED_INDEX_ID}' not found on endpoint. Set config.DEPLOYED_INDEX_ID. Details: {e}",
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
            "error": f"Vector search returned {len(neighbors)} neighbors but none matched chunks in GCS. Sample neighbor IDs: {sample_ids}.",
        }
    if not context and not neighbors:
        return {
            "context": "",
            "num_passages": 0,
            "error": "Vector search returned 0 neighbors. Check deployed index and filter (source=doc-pipeline).",
        }
    return {"context": context, "num_passages": len(parts)}


def _extract_search_terms(question: str) -> list:
    """Extract multiple search terms from a question for targeted retrieval."""
    terms = [question]
    keyword_mappings = {
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
    seen = set()
    unique_terms = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_terms.append(t)
    return unique_terms[:8]


def comprehensive_search(question: str) -> str:
    """Run multiple targeted searches and combine results (optional helper)."""
    search_terms = _extract_search_terms(question)
    print(f"\n>>> COMPREHENSIVE SEARCH: {len(search_terms)} search terms extracted", flush=True)
    all_context_parts = []
    seen_passages = set()
    for term in search_terms:
        result = search_document(term, top_k=30)
        context = result.get("context", "")
        if context:
            for passage in context.split("\n\n---\n\n"):
                passage_hash = hash(passage[:200]) if len(passage) > 200 else hash(passage)
                if passage_hash not in seen_passages:
                    seen_passages.add(passage_hash)
                    all_context_parts.append(passage)
    combined = []
    total_len = 0
    for p in all_context_parts:
        if total_len + len(p) > MAX_CONTEXT_CHARS:
            break
        combined.append(p)
        total_len += len(p)
    print(f">>> COMPREHENSIVE SEARCH RESULT: {len(combined)} unique passages, {total_len} chars", flush=True)
    return "\n\n---\n\n".join(combined)


def create_pdf_agent():
    """Create the PDF/document sub-agent (LlmAgent with search_document tool)."""
    model = Gemini(
        model_name=config.GEMINI_MODEL,
        project=config.PROJECT_ID,
        location=config.LOCATION,
        vertexai=True,
    )
    return LlmAgent(
        model=model,
        name="pdf_agent",
        instruction="""You answer questions using only the documents we have indexed. Use the search_document tool to find relevant passages; you may call it multiple times with different search terms if needed.

Rules:
- Answer ONLY from the retrieved context. Do not invent or assume information.
- If the search returns no relevant passages, or the retrieved text does not contain the answer, respond clearly with: "This information is not available in the documents we have." or "We don't have that information in our documents."
- If only part of the question can be answered from the documents, answer that part and say what we don't have (e.g. "We have X in our documents, but we don't have information about Y.").
- Quote specific numbers and facts when the documents support them.
- Do not refer to Salesforce or BigQuery unless the user asks about them.""",
        tools=[search_document],
    )
