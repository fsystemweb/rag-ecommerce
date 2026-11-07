"""Opik tracing, opt-in: active only when OPIK_URL_OVERRIDE is set.

Without Opik everything here is a no-op, so the app works as before.
"""

import logging

from app import config

log = logging.getLogger("rag")
ENABLED = bool(config.OPIK_URL_OVERRIDE)


def track(name: str, **kwargs):
    """`@opik.track` when enabled, otherwise the function is returned unchanged."""
    if not ENABLED:
        return lambda fn: fn
    import opik

    return opik.track(name=name, **kwargs)


def llm_config() -> dict:
    """LangChain callback config that shows model calls as spans in the trace."""
    if not ENABLED:
        return {}
    from opik.integrations.langchain import OpikTracer

    return {"callbacks": [OpikTracer()]}


def tag_trace(tags: list[str], metadata: dict):
    """Add filterable tags (grounded, attempts, prompt versions) to the current trace."""
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

        return opik.Opik()
    except Exception as e:
        log.warning("Opik unavailable: %s", e)
        return None
