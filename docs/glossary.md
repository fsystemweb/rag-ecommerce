# Glossary

- **RAG**: retrieve relevant text first, then let a model answer using it.
- **Embedding**: a list of numbers representing meaning; similar texts get close numbers.
- **Chunk**: a piece of a document that is embedded and retrieved.
- **Vector store**: database of embeddings (Chroma here).
- **Hybrid search**: embeddings search plus keyword search, merged.
- **BM25**: classic keyword ranking by term frequency.
- **RRF (Reciprocal Rank Fusion)**: merges ranked lists by summing `1 / (60 + rank)`; see [how hybrid search works](architecture.md#how-hybrid-search-works).
- **Grounding**: every claim in the answer comes from the retrieved data.
- **Hallucination**: the model states something not supported by the data.
- **Golden dataset**: fixed questions with known good answers, used to measure.
- **Precision / Recall**: how much of what we returned is right / how much of what was right we returned.
- **Trace / Span**: one request recorded end to end / one step inside it.
- **Experiment**: one evaluation run saved so runs can be compared.
