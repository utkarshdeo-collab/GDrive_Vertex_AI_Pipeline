"""
Audit context for orchestrator: holds session_id, turn_id, user_question, routing_hints,
turn_start_utc, and a list of tool-invocation entries. Tools append to this list;
orchestrator writes one BigQuery row per entry at end of turn.
Uses contextvars so async flow has one context per turn.
"""
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, List, Optional

# One dict per turn: session_id, turn_id, user_question, routing_hints, turn_start_utc, audit_entries (list)
_audit_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar("audit_context", default=None)


def set_turn_context(
    session_id: str,
    turn_id: str,
    user_question: str,
    routing_hints: Optional[str] = None,
    turn_start_utc: Optional[datetime] = None,
) -> None:
    """Set the current turn context and clear any previous audit entries."""
    from datetime import timezone
    _audit_context.set({
        "session_id": session_id,
        "turn_id": turn_id,
        "user_question": user_question,
        "routing_hints": routing_hints,
        "turn_start_utc": turn_start_utc or datetime.now(timezone.utc),
        "audit_entries": [],
    })


def get_turn_context() -> Optional[Dict[str, Any]]:
    """Return current turn context or None if not in an audited turn."""
    return _audit_context.get()


def append_audit_entry(
    tool_call: str,
    sql_generated: Optional[str] = None,
    bigquery_bytes_processed: Optional[int] = None,
    error_messages: Optional[str] = None,
) -> None:
    """Append one tool-invocation entry. No-op if no turn context is set."""
    ctx = _audit_context.get()
    if ctx is None:
        return
    entries = ctx.get("audit_entries")
    if entries is not None:
        entries.append({
            "tool_call": tool_call,
            "sql_generated": sql_generated,
            "bigquery_bytes_processed": bigquery_bytes_processed,
            "error_messages": error_messages,
        })


def get_and_clear_turn_context() -> Optional[Dict[str, Any]]:
    """Return full turn context (session_id, turn_id, user_question, routing_hints, audit_entries) and clear.
    Orchestrator uses this at end of turn to build audit rows then clear for next turn."""
    ctx = _audit_context.get()
    _audit_context.set(None)
    return ctx
