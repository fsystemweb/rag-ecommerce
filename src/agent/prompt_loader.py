"""Load a prompt template: Comet Opik first, local .txt file as fallback.

The local files in app/prompts/ are the examples that seed Opik and the
fallback when Opik is off, unreachable, or does not have the prompt yet.
"""

import logging
from pathlib import Path

from src.observability import tracer as observability

log = logging.getLogger("rag")
PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_FILES = {  # Opik prompt name -> local file
    "rag-ecommerce-generator": "generator_prompt.txt",
    "rag-ecommerce-judge": "judge_prompt.txt",
    "rag-ecommerce-rewriter": "rewriter_prompt.txt",
}
VERSIONS: dict[str, str] = {}  # name -> "opik:<commit>" or "local", for traces
_cache: dict[str, tuple[str, str]] = {}


def load_prompt(name: str, local_file: str) -> tuple[str, str]:
    """Return (text, source) where source is "opik" or "local". Cached per process."""
    if name in _cache:
        return _cache[name]
    text, source = None, "local"
    client = observability.client()
    if client is not None:
        try:
            prompt = client.get_prompt(name)
            if prompt is not None:
                text, source = prompt.prompt, "opik"
                VERSIONS[name] = f"opik:{prompt.commit}"
        except Exception as e:
            log.warning("could not read prompt %s from Opik: %s", name, e)
    if text is None:
        text = (PROMPTS_DIR / local_file).read_text(encoding="utf-8")
        VERSIONS[name] = "local"
    log.info("prompt %s source=%s version=%s", name, source, VERSIONS[name])
    _cache[name] = (text, source)
    return text, source


def render(template: str, **values) -> str:
    """Fill {{name}} placeholders (the same syntax Opik uses). No escaping."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
