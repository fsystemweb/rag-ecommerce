from src.agent import prompt_loader
from src.agent.prompt_loader import load_prompt, render
from src.observability import tracer as observability


def test_render_replaces_placeholders_without_escaping():
    assert render("Q: {{q}}", q="<b> & 'x'") == "Q: <b> & 'x'"


def test_local_fallback_when_opik_is_off(monkeypatch):
    prompt_loader._cache.clear()
    monkeypatch.setattr(observability, "client", lambda: None)
    text, source = load_prompt("rag-ecommerce-classifier", "classifier_prompt.txt")
    assert source == "local" and "{{question}}" in text
    assert prompt_loader.VERSIONS["rag-ecommerce-classifier"] == "local"


def test_local_fallback_when_opik_raises(monkeypatch):
    class Broken:
        def get_prompt(self, name):
            raise ConnectionError("opik is down")

    prompt_loader._cache.clear()
    monkeypatch.setattr(observability, "client", lambda: Broken())
    assert load_prompt("rag-ecommerce-classifier", "classifier_prompt.txt")[1] == "local"


def test_prompt_from_opik_wins(monkeypatch):
    class P:
        prompt, commit = "from opik {{question}}", "abc123"

    class Client:
        def get_prompt(self, name):
            return P()

    prompt_loader._cache.clear()
    monkeypatch.setattr(observability, "client", lambda: Client())
    assert load_prompt("rag-ecommerce-generator", "generator_prompt.txt") == ("from opik {{question}}", "opik")
    assert prompt_loader.VERSIONS["rag-ecommerce-generator"] == "opik:abc123"
