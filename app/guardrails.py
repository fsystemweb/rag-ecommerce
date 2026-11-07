"""Input checks, prompt-injection rules and the constants that control retries."""

import re

# One place for the numbers (changing them needs approval, see AGENTS.md)
MAX_QUESTION_CHARS = 500
MIN_JUDGE_SCORE = 3
MAX_ATTEMPTS = 3
CANARY = "CANARY-7f3a91"  # hidden in the system prompt; if it shows up in an answer, it leaked

REFUSAL = (
    "Lo siento, solo puedo ayudarte a encontrar productos del catálogo. "
    "No tengo información sobre precios, stock, envíos o devoluciones. "
    "¿Qué producto buscas?"
)

INJECTION_PATTERNS = [
    r"ignor(e|a)\s+(all\s+|todas\s+|todo\s+)?(the\s+|las\s+|tus\s+|your\s+)?(previous|above|prior|anteriores|instrucciones|instructions)",
    r"(olvida|forget)\s+(todo|tus|your|all|everything|las)",
    r"(system|developer)\s*prompt",
    r"prompt\s+del\s+sistema",
    r"(reveal|show|print|muestra|revela|repite)\s+.{0,30}(instructions|instrucciones|rules|reglas|prompt)",
    r"(you are now|ahora eres|act as|actúa como|finge ser|pretend to be)",
    r"jailbreak|do anything now|\bDAN\b",
]


def clean_question(text: str) -> str:
    """Strip control characters and extra spaces, cut to the maximum length."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()[:MAX_QUESTION_CHARS]


def looks_like_injection(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS)


def leaks_prompt(answer: str) -> bool:
    return CANARY in answer


def sanitize_product_text(text: str) -> str:
    """Product text is data: it must not be able to close the <products> block."""
    return re.sub(r"</?\s*products?\s*[^>]*>", "", text, flags=re.IGNORECASE)
