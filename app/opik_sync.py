"""On startup, push the local prompts to Opik if they are missing there."""

import logging

from app import observability
from app.prompt_loader import PROMPT_FILES, PROMPTS_DIR

log = logging.getLogger("rag")


def seed_prompts():
    client = observability.client()
    if client is None:
        return
    for name, file in PROMPT_FILES.items():
        try:
            if client.get_prompt(name) is None:
                client.create_prompt(name=name, prompt=(PROMPTS_DIR / file).read_text(encoding="utf-8"))
                log.info("seeded prompt %s in Opik", name)
        except Exception as e:  # Opik down must never stop the app
            log.warning("could not seed prompt %s: %s", name, e)
            return
