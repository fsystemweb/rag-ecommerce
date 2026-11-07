"""Load a prompt template. Local .txt files in app/prompts/ for now."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str, local_file: str) -> tuple[str, str]:
    """Return (text, source)."""
    return (PROMPTS_DIR / local_file).read_text(encoding="utf-8"), "local"


def render(template: str, **values) -> str:
    """Fill {{name}} placeholders (the same syntax Opik uses). No escaping."""
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template
