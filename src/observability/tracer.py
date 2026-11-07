"""Opik tracing, opt-in: active only when OPIK_URL_OVERRIDE is set.

Without Opik everything here is a no-op, so the app works as before.
"""

import logging

from src.core import config

log = logging.getLogger("rag")
ENABLED = bool(config.OPIK_URL_OVERRIDE)


def track(name: str, **kwargs):
    """`@opik.track` when enabled, otherwise the function is returned unchanged."""
    if not ENABLED:
        return lambda fn: fn
    import opik

    return opik.track(name=name, project_name=config.OPIK_PROJECT_NAME, **kwargs)


def llm_config() -> dict:
    """LangChain callback config that shows model calls as spans in the trace."""
    if not ENABLED:
        return {}
    from opik.integrations.langchain import OpikTracer

    return {"callbacks": [OpikTracer(project_name=config.OPIK_PROJECT_NAME)]}


def tag_trace(tags: list[str], metadata: dict):
    """Add tags and metadata (prompt versions) to the current trace."""
    if not ENABLED:
        return
    try:
        from opik import opik_context

        opik_context.update_current_trace(tags=tags, metadata=metadata)
    except Exception as e:  # tracing must never break an answer
        log.warning("could not tag trace: %s", e)


def client():
    """Opik client, or None when disabled or unreachable."""
    if not ENABLED:
        return None
    try:
        import opik

        return opik.Opik(project_name=config.OPIK_PROJECT_NAME)
    except Exception as e:
        log.warning("Opik unavailable: %s", e)
        return None
