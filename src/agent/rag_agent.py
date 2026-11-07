"""The RAG pipeline: classify -> rewrite -> retrieve -> generate -> answer.

`Pipeline.run()` yields events so the same code serves /ask and /ask/stream:
  {"type": "status", "step": "..."}   progress
  {"type": "token", "text": "..."}    generated text (streaming)
  {"type": "final", "response": AskResponse}
"""

import logging
import re

from src.agent import guardrails
from src.agent.prompt_loader import VERSIONS, load_prompt, render
from src.core import config
from src.core.llm import make_llm
from src.core.models import AskRequest, AskResponse, ProductRef
from src.observability import tracer as observability
from src.observability.tracer import track

log = logging.getLogger("rag")
HISTORY_TURNS = 6  # last turns used for the rewrite


def format_context(chunks) -> str:
    return "\n".join(
        f'<product id="{c.metadata["product_id"]}">\n{guardrails.sanitize_product_text(c.text)}\n</product>' for c in chunks
    )


class Pipeline:
    def __init__(self, retriever, generator=None, classifier=None):
        self.retriever = retriever
        self.generator = generator or make_llm(config.GENERATOR_LLM)
        self.classifier = classifier or make_llm(config.EVALUATOR_LLM)

    # --- steps -------------------------------------------------------
    @track("classify", type="llm")
    def is_off_topic(self, question, history) -> bool:
        text = "\n".join(f"{t.role}: {t.content}" for t in history[-HISTORY_TURNS:]) or "(none)"
        template, _ = load_prompt("rag-ecommerce-classifier", "classifier_prompt.txt")
        reply = self.classifier.invoke(
            render(template, history=text, question=question), config=observability.llm_config()
        ).content
        return guardrails.is_off_topic(reply)

    @track("rewrite", type="llm")
    def rewrite(self, question, history) -> str:
        if not history:
            return question
        text = "\n".join(f"{t.role}: {t.content}" for t in history[-HISTORY_TURNS:])
        template, _ = load_prompt("rag-ecommerce-rewriter", "rewriter_prompt.txt")
        out = self.generator.invoke(render(template, history=text, question=question), config=observability.llm_config()).content
        return out.strip() or question

    @track("generate", type="llm", generations_aggregator="".join)
    def generate_stream(self, question, context):
        template, _ = load_prompt("rag-ecommerce-generator", "generator_prompt.txt")
        prompt = render(template, question=question, context=context)
        for piece in self.generator.stream(prompt, config=observability.llm_config()):
            yield piece.content

    # --- main flow ---------------------------------------------------
    @track("ask", generations_aggregator=lambda events: events[-1])
    def run(self, req: AskRequest):
        refusal = AskResponse(answer=guardrails.REFUSAL, attempts=0)
        no_match = AskResponse(answer=guardrails.NO_MATCH, attempts=0)
        question = guardrails.clean_question(req.question)
        if not question or guardrails.looks_like_injection(question):
            log.warning("question blocked by input guardrail")
            yield self._final(refusal)
            return

        yield {"type": "status", "step": "classify"}
        if self.is_off_topic(question, req.history):
            log.warning("question blocked by topic classifier")
            yield self._final(refusal)
            return

        yield {"type": "status", "step": "rewrite"}
        standalone = self.rewrite(question, req.history)
        yield {"type": "status", "step": "retrieve"}
        chunks = self.retriever.retrieve(standalone)
        if not chunks:
            yield self._final(no_match)
            return

        yield {"type": "status", "step": "generate"}
        parts = []
        for piece in self.generate_stream(question, format_context(chunks)):
            parts.append(piece)
            yield {"type": "token", "text": piece}
        answer = "".join(parts).strip()

        if guardrails.leaks_prompt(answer):
            yield self._final(refusal)
        elif answer.startswith("NO_ANSWER"):
            yield self._final(no_match)
        else:
            yield self._final(AskResponse(answer=answer, products=self._products_in(answer, chunks)))

    @staticmethod
    def _final(response: AskResponse) -> dict:
        observability.tag_trace([], {"prompt_versions": dict(VERSIONS)})
        return {"type": "final", "response": response}

    @staticmethod
    def _products_in(answer, chunks) -> list[ProductRef]:
        mentioned = set(re.findall(r"\[([^\]\s]+)\]", answer))
        seen, out = set(), []
        for c in chunks:
            pid = c.metadata["product_id"]
            if pid in mentioned and pid not in seen:
                seen.add(pid)
                out.append(ProductRef(product_id=pid, title=c.metadata["title"], brand=c.metadata.get("brand", "")))
        return out

    def ask(self, req: AskRequest) -> AskResponse:
        for event in self.run(req):
            if event["type"] == "final":
                return event["response"]
