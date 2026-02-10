"""
Audit callbacks for ADK web path: set turn context before agent run and write
audit rows after. Keeps CLI logic unchanged; only adds behavior when running
via adk web (where the main loop in run_orchestrator.py is not used).
"""
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from google.adk.agents.callback_context import CallbackContext


def _text_from_content(content) -> str:
    """Extract plain text from genai Content (user_content or event content)."""
    if not content or not getattr(content, "parts", None):
        return ""
    parts = content.parts
    if not parts:
        return ""
    return "".join(
        getattr(p, "text", "") or ""
        for p in parts
    ).strip()


def _assistant_response_from_session(session, invocation_id: str) -> str:
    """Collect assistant text from session events for this invocation."""
    if not getattr(session, "events", None):
        return ""
    texts = []
    for event in session.events:
        if getattr(event, "invocation_id", None) != invocation_id:
            continue
        if getattr(event, "author", None) == "user":
            continue
        if not getattr(event, "content", None) or not getattr(event.content, "parts", None):
            continue
        for part in event.content.parts:
            if getattr(part, "text", None):
                texts.append(part.text)
    return "".join(texts).strip()


async def before_agent_audit_callback(*, callback_context: "CallbackContext") -> Optional[object]:
    """Set audit turn context when running from ADK web (no main loop).
    Only sets context if none is set yet (CLI already sets it in main loop)."""
    from orchestrator.audit_context import get_turn_context, set_turn_context

    try:
        import config
    except ImportError:
        return None
    if not getattr(config, "AUDIT_ENABLED", False):
        return None
    if get_turn_context() is not None:
        return None  # CLI path already set context
    user_content = getattr(callback_context, "user_content", None)
    user_question = _text_from_content(user_content) if user_content else ""
    session = getattr(callback_context, "session", None)
    session_id = getattr(session, "id", "") or ""
    invocation_id = getattr(callback_context, "invocation_id", "") or ""
    turn_start_utc = datetime.now(timezone.utc)
    set_turn_context(session_id, invocation_id, user_question, None, turn_start_utc)
    return None


async def after_agent_audit_callback(*, callback_context: "CallbackContext") -> Optional[object]:
    """Write audit rows after agent run (used by both CLI and ADK web).
    Clears turn context and writes to BigQuery when audit is enabled."""
    from orchestrator.audit_context import get_and_clear_turn_context
    from orchestrator.audit_logger import build_audit_rows, write_audit_rows

    try:
        import config
    except ImportError:
        return None
    if not getattr(config, "AUDIT_ENABLED", False):
        return None
    ctx = get_and_clear_turn_context()
    if ctx is None:
        return None
    session = getattr(callback_context, "session", None)
    invocation_id = getattr(callback_context, "invocation_id", "") or ""
    assistant_response = _assistant_response_from_session(session, invocation_id) if session else ""
    entries = ctx.get("audit_entries") or []
    turn_start_utc = ctx.get("turn_start_utc") or datetime.now(timezone.utc)
    rows = build_audit_rows(
        turn_start_utc,
        ctx.get("user_question", ""),
        assistant_response,
        ctx.get("turn_id", invocation_id),
        ctx.get("session_id"),
        ctx.get("routing_hints"),
        entries,
    )
    if not rows:
        return None
    try:
        import google.auth
        from google.auth.transport import requests as google_requests
        credentials, _ = google.auth.default()
        if not credentials.valid:
            credentials.refresh(google_requests.Request())
    except Exception:
        return None
    write_audit_rows(
        rows,
        config.PROJECT_ID,
        config.AUDIT_DATASET,
        config.AUDIT_TABLE,
        credentials,
    )
    return None
