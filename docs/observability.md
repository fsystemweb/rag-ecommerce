# Observability

With [Comet Opik](https://github.com/comet-ml/opik) running (`scripts/opik.sh up`, UI on port 5173) and `OPIK_URL_OVERRIDE` set, each question is a **trace**: topic classification, rewrite, retrieval (with dense score and BM25 rank), and generation, with the prompt versions in the metadata. Prompts are read from Opik first (`rag-ecommerce-generator`, `-rewriter`, `-classifier`), with `src/agent/prompts/*.txt` as fallback and seed. Upload the golden set with `python -m src.evaluation.dataset --upload`; evaluation runs appear as **experiments**.

## Traces

Every `/ask` call is a trace. The list shows the question, the answer, the duration and the errors.

![Traces list](assets/screens/opik-traces.png)

Open one to see the steps (`classify`, `rewrite`, `retrieve`, `generate`), the model used in each LLM call, the token usage and the prompt versions (`prompt_versions` in the metadata).

![Trace detail](assets/screens/opik-trace-detail.png)

## Prompts

The three prompts are versioned in the prompt library. Edit one in Opik to change the behaviour without a deploy; the trace metadata shows which version answered.

![Prompt library](assets/screens/opik-prompts.png)

## Dataset

The golden dataset (120 items) is uploaded as `rag-ecommerce-golden`. Opik reserves `id` and `source`, so they are stored as `item_id` and `golden_source`.

![Dataset items](assets/screens/opik-dataset-items.png)

## Experiments

Each evaluation run is an experiment with the average of every metric. The last one is the base of the [evaluation report](report.md).

![Experiments](assets/screens/opik-experiments.png)

![Experiment detail](assets/screens/opik-experiment-detail.png)

> Two older experiments show "Deleted dataset": they ran against a first upload that was removed and uploaded again. They have no scores.
