"""
Per-question usage collector for orchestrator cost display.
Collects Gemini (per event/agent), embedding (pdf_agent search_document), and BigQuery bytes.
Thread-safe for single-threaded async use; cleared at start of each user turn.
"""
from typing import List, Dict, Any

_tasks: List[Dict[str, Any]] = []


def clear():
    """Clear collected usage at the start of a new user question."""
    global _tasks
    _tasks = []


def record_gemini(author: str, prompt_token_count: int, candidates_token_count: int):
    """Record one Gemini LLM call (from event.usage_metadata + event.author)."""
    _tasks.append({
        "kind": "gemini",
        "label": author,
        "prompt_token_count": prompt_token_count or 0,
        "candidates_token_count": candidates_token_count or 0,
    })


def record_embedding(chars: int):
    """Record embedding usage (e.g. search_document query length in chars)."""
    _tasks.append({
        "kind": "embedding",
        "label": "search_document",
        "chars": chars,
    })


def record_bigquery(bytes_processed: int):
    """Record BigQuery bytes processed (execute_sql)."""
    _tasks.append({
        "kind": "bigquery",
        "label": "execute_sql",
        "bytes_processed": bytes_processed,
    })


def get_and_clear() -> List[Dict[str, Any]]:
    """Return collected tasks and clear the list."""
    global _tasks
    out = list(_tasks)
    _tasks = []
    return out
