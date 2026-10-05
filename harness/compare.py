"""Fair, reproducible A/B benchmark for two local Ollama models."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from llm import call_json
from metrics import valid_rows
from rag import LocalRAG
from schema import Explanation, Verdict
from sql_features import detect_categories, operation_inventory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two Ollama models fairly on SQL explanations.")
    p.add_argument("split", choices=("tiny", "dev", "test", "all"))
    p.add_argument("prompt_file", type=pathlib.Path)
    p.add_argument("run_id")
    p.add_argument("--model-a", required=True, help="First generator Ollama model.")
    p.add_argument("--model-b", required=True, help="Second generator Ollama model.")
    p.add_argument("--judge-model", required=True, help="Single shared judge Ollama model.")
    p.add_argument("--workers", type=int, default=1, help="Concurrent items. Keep at 1 for clean latency comparisons.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-predict", type=int, default=3000)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repeat-penalty", type=float, default=1.0)
    p.add_argument("--judge-num-predict", type=int, default=1200)
    p.add_argument("--judge-prompt", type=pathlib.Path, default=pathlib.Path("prompts/judge.md"))
    p.add_argument("--ollama-host", default=None, help="Optional Ollama host, e.g. http://127.0.0.1:11434")
    p.add_argument("--keep-alive", default="10m")
    p.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                   help="Warm both generators and the judge before measured calls (default: true).")
    p.add_argument("--rag-file", type=pathlib.Path, default=pathlib.Path("knowledge/sql_rag.jsonl"),
                   help="Local JSONL knowledge base used by the lightweight RAG retriever.")
    p.add_argument("--rag-k", type=int, default=3, help="Number of local RAG chunks retrieved per query.")
    p.add_argument("--no-rag", action="store_true", help="Disable local RAG for a clean baseline/ablation run.")
    args = p.parse_args()

    for path, label in ((args.prompt_file, "prompt file"), (args.judge_prompt, "judge prompt")):
        if not path.exists():
            p.error(f"{label} does not exist: {path}")
    if args.workers < 1:
        p.error("--workers must be >= 1")
    if args.rag_k < 0:
        p.error("--rag-k must be >= 0")
    if not args.no_rag and not args.rag_file.exists():
        p.error(f"RAG file does not exist: {args.rag_file}")
    if args.model_a == args.model_b:
        logger.warning("Model A and Model B are identical; this is a sanity check, not a model comparison.")
    return args


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def outcome(score_a: int | None, score_b: int | None) -> str:
    if score_a is None or score_b is None:
        return "evaluation_error"
    if score_a and score_b:
        return "both_correct"
    if score_a:
        return "only_a_correct"
    if score_b:
        return "only_b_correct"
    return "both_wrong"


def dimensions(v: Verdict) -> dict[str, bool]:
    return {
        "correctness": v.correctness,
        "completeness": v.completeness,
        "hallucination_free": v.hallucination_free,
        "clarity": v.clarity,
    }


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def is_complete(row: dict[str, Any]) -> bool:
    return (
        row.get("benchmark_version") == 4
        and not row.get("error")
        and row.get("model_a", {}).get("score") is not None
        and row.get("model_b", {}).get("score") is not None
    )


def main() -> int:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    judge_prompt = args.judge_prompt.read_text(encoding="utf-8")
    input_path = pathlib.Path(f"data/{args.split}.jsonl")
    if not input_path.exists():
        raise SystemExit(f"dataset does not exist: {input_path}")
    items = load_jsonl(input_path)
    if not items:
        raise SystemExit("dataset is empty")

    rag = None if args.no_rag else LocalRAG(args.rag_file)
    rag_file_sha = rag.file_sha256 if rag else None
    rag_config = {
        "enabled": rag is not None,
        "file": str(args.rag_file) if rag else None,
        "file_sha256": rag_file_sha,
        "top_k": args.rag_k if rag else 0,
        "retrieval": "tfidf_cosine" if rag else None,
    }

    settings = {
        "temperature": args.temperature,
        "seed": args.seed,
        "num_predict": args.num_predict,
        "num_ctx": args.num_ctx,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
    }
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    judge_prompt_sha = hashlib.sha256(judge_prompt.encode()).hexdigest()
    benchmark_fairness_sha = sha256_text(
        prompt_sha,
        judge_prompt_sha,
        json.dumps(settings, sort_keys=True),
        json.dumps(rag_config, sort_keys=True),
        args.model_a,
        args.model_b,
        args.judge_model,
    )

    out_path = pathlib.Path(f"results/{args.run_id}_{args.split}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for row in load_jsonl(out_path):
            existing[row["id"]] = row
        existing = {k: v for k, v in existing.items() if is_complete(v)}
        logger.info("Resume state: %d/%d complete v4 items", len(existing), len(items))

    def retrieve(item: dict[str, Any]) -> tuple[list[Any], str]:
        if not rag:
            return [], "No local RAG context was used. Use the supplied SQL and schema only."
        return rag.retrieve_for_case(
            schema=item["schema"],
            sql=item["sql"],
            categories=detect_categories(item["sql"]),
            top_k=args.rag_k,
        )

    if args.warmup:
        warm_item = items[0]
        _, warm_rag_context = retrieve(warm_item)
        warm_user = (
            f"Schema:\n{warm_item['schema']}\n\nSQL:\n{warm_item['sql']}\n\n"
            f"Retrieved local guidance:\n{warm_rag_context}"
        )
        for model in (args.model_a, args.model_b):
            try:
                call_json(model, prompt, warm_user, Explanation, temperature=args.temperature, seed=args.seed,
                          num_predict=args.num_predict, num_ctx=args.num_ctx, top_p=args.top_p, top_k=args.top_k,
                          repeat_penalty=args.repeat_penalty, ollama_host=args.ollama_host, keep_alive=args.keep_alive)
            except Exception as exc:
                logger.warning("Warmup failed for %s: %s", model, exc)
        try:
            call_json(
                args.judge_model,
                judge_prompt,
                warm_user + "\n\nExplanation to evaluate:\nWarmup only.",
                Verdict,
                temperature=0.0,
                seed=args.seed,
                num_predict=args.judge_num_predict,
                num_ctx=args.num_ctx,
                top_p=1.0,
                top_k=0,
                repeat_penalty=1.0,
                ollama_host=args.ollama_host,
                keep_alive=args.keep_alive,
            )
        except Exception as exc:
            logger.warning("Judge warmup failed: %s", exc)

    remaining = [(idx, item) for idx, item in enumerate(items) if item["id"] not in existing]
    logger.info(
        "A/B: %s vs %s | judge=%s | split=%s | workers=%d | warmup=%s | rag=%s(k=%d)",
        args.model_a,
        args.model_b,
        args.judge_model,
        args.split,
        args.workers,
        args.warmup,
        rag is not None,
        args.rag_k if rag else 0,
    )

    def generate(model: str, item: dict[str, Any], rag_context: str) -> tuple[Explanation, dict[str, Any]]:
        user = (
            f"Schema:\n{item['schema']}\n\n"
            f"SQL:\n{item['sql']}\n\n"
            f"Retrieved local guidance:\n{rag_context}\n\n"
            "Use the guidance only to clarify SQL semantics. The exact SQL and schema are authoritative."
        )
        return call_json(model, prompt, user, Explanation, temperature=args.temperature, seed=args.seed,
                         num_predict=args.num_predict, num_ctx=args.num_ctx, top_p=args.top_p, top_k=args.top_k,
                         repeat_penalty=args.repeat_penalty, ollama_host=args.ollama_host, keep_alive=args.keep_alive)

    def judge(item: dict[str, Any], explanation: str, rag_context: str) -> tuple[Verdict, dict[str, Any]]:
        user = (
            f"Schema:\n{item['schema']}\n\n"
            f"SQL:\n{item['sql']}\n\n"
            f"{operation_inventory(item['sql'])}\n\n"
            f"Retrieved local guidance:\n{rag_context}\n\n"
            f"Candidate explanation:\n{explanation}"
        )
        return call_json(args.judge_model, judge_prompt, user, Verdict, temperature=0.0, seed=args.seed,
                         num_predict=args.judge_num_predict, num_ctx=args.num_ctx, top_p=1.0, top_k=0,
                         repeat_penalty=1.0, ollama_host=args.ollama_host, keep_alive=args.keep_alive)

    def process(index: int, item: dict[str, Any]) -> dict[str, Any]:
        case_sha = sha256_text(item["schema"], item["sql"])
        rag_results, rag_context = retrieve(item)
        rag_rows = [
            {
                "id": result.chunk.id,
                "title": result.chunk.title,
                "score": result.score,
                "source": result.chunk.source,
            }
            for result in rag_results
        ]
        base: dict[str, Any] = {
            "benchmark_version": 4,
            "id": item["id"],
            "domain": item.get("domain", "unknown"),
            "dataset_feature": item.get("feature", "unknown"),
            "categories": detect_categories(item["sql"]),
            "schema": item["schema"],
            "sql": item["sql"],
            "case_sha256": case_sha,
            "benchmark_fairness_sha256": benchmark_fairness_sha,
            "prompt_file": str(args.prompt_file),
            "prompt_sha256": prompt_sha,
            "judge_prompt_file": str(args.judge_prompt),
            "judge_prompt_sha256": judge_prompt_sha,
            "settings": settings,
            "model_a_name": args.model_a,
            "model_b_name": args.model_b,
            "judge_model": args.judge_model,
            "generation_order": "A_FIRST" if index % 2 == 0 else "B_FIRST",
            "rag": {
                **rag_config,
                "retrieved_chunks": rag_rows,
                "context_sha256": hashlib.sha256(rag_context.encode()).hexdigest(),
            },
            "model_a": {"score": None, "error": None},
            "model_b": {"score": None, "error": None},
            "error": None,
        }

        generated: dict[str, tuple[Explanation, dict[str, Any]]] = {}
        order = (("a", args.model_a), ("b", args.model_b)) if base["generation_order"] == "A_FIRST" else (("b", args.model_b), ("a", args.model_a))
        for label, model in order:
            try:
                generated[label] = generate(model, item, rag_context)
            except Exception as exc:
                base[f"model_{label}"]["error"] = str(exc)

        for label in ("a", "b"):
            if label not in generated:
                continue
            output, gen_cost = generated[label]
            record = base[f"model_{label}"]
            record["explanation"] = output.explanation
            record["generation_cost"] = gen_cost
            record["output_chars"] = len(output.explanation)
            record["output_words"] = len(output.explanation.split())
            try:
                verdict, judge_cost = judge(item, output.explanation, rag_context)
                dims = dimensions(verdict)
                score = int(all(dims.values()))
                record.update({
                    "score": score,
                    "dimensions": dims,
                    "judge_reported_grade": verdict.grade,
                    "judge_grade_consistent": (verdict.grade == "good") == bool(score),
                    "judge_reason": verdict.reason,
                    "judge_cost": judge_cost,
                })
            except Exception as exc:
                record["error"] = f"judge: {exc}"

        score_a = base["model_a"].get("score")
        score_b = base["model_b"].get("score")
        base["outcome"] = outcome(score_a, score_b)
        base["disagreement"] = score_a is not None and score_b is not None and score_a != score_b
        if base["outcome"] == "evaluation_error":
            errors = []
            for label in ("a", "b"):
                if base[f"model_{label}"].get("error"):
                    errors.append(f"{label.upper()}: {base[f'model_{label}']['error']}")
            base["error"] = "; ".join(errors) or "incomplete evaluation"
        return base

    completed = dict(existing)
    with out_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, idx, item): item for idx, item in remaining}
            with tqdm(total=len(remaining), unit="item", desc=f"{args.run_id}/{args.split}") as bar:
                for future in as_completed(futures):
                    row = future.result()
                    completed[row["id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    bar.set_postfix(id=row["id"], outcome=row.get("outcome"))
                    bar.update(1)

    with out_path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = completed.get(item["id"])
            if row:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = [completed[item["id"]] for item in items if item["id"] in completed]
    valid = valid_rows(rows)
    logger.info(
        "Completed %d/%d | valid A/B comparisons: %d | evaluation errors: %d",
        len(rows), len(items), len(valid), len(rows) - len(valid),
    )
    if valid:
        a = sum(r["model_a"]["score"] for r in valid)
        b = sum(r["model_b"]["score"] for r in valid)
        logger.info("Judge pass rate: A %.1f%% | B %.1f%%", 100 * a / len(valid), 100 * b / len(valid))
    return 0 if len(valid) == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
