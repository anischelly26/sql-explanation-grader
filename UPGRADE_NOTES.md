# Benchmark v3 Upgrade

## Benchmark fairness

- Explicit Model A, Model B and shared judge CLI parameters.
- Same SQL, schema, prompt and generation settings for both generators.
- Explicit temperature, seed, context size, top-p, top-k, repeat penalty and output limit.
- Warmup before measured generation.
- Alternating A-first and B-first order.
- Default single worker for clean latency comparison.
- Per-run benchmark fairness hash and per-query case hash.

## Evaluation quality

- Judge receives schema + SQL + actual operation inventory.
- Judge explicitly cannot demand absent SQL operations.
- Detector now covers window functions, CTEs, set operations, CASE, EXISTS, IN, BETWEEN, LIKE and NULL filtering.
- Comments and quoted literals/identifiers are masked before feature detection.
- Overall pass is derived from correctness + completeness + hallucination-free + clarity.
- Judge grade consistency is retained as an audit field.
- Infrastructure/model errors are separated from wrong answers.

## Comparison quality

- Paired bootstrap confidence interval for B minus A.
- Exact McNemar test on discordant pairs.
- Category-level comparison.
- Optional external human/golden label agreement.
- Explicit terminology: judge pass rate is not true accuracy without external validation.

## Dashboard and reporting

- Side-by-side A/B cards and dimensions.
- Pair outcomes and category performance.
- Search/filter per-query cases.
- Side-by-side explanations and judge reasons.
- PDF export endpoint and standalone PDF report.


## Lightweight RAG (v4)

- Added a checked-in SQL knowledge base at knowledge/sql_rag.jsonl.
- Added deterministic local TF-IDF + cosine retrieval in harness/rag.py.
- Both generators and the shared judge receive the same retrieved chunks.
- RAG retrieval is independent of model output and therefore does not create an A/B fairness difference.
- Result rows store the knowledge-base hash, retrieved chunk IDs/scores, and context hash.
- Added --rag-k and --no-rag for simple RAG ablation testing.
