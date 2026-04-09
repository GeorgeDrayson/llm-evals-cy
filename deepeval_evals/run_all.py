#!/usr/bin/env python3
"""CLI runner for Welsh LLM evaluations.

Usage:
    python -m deepeval_evals.run_all --model gpt-4o
    python -m deepeval_evals.run_all --model anthropic/claude-sonnet-4-20250514 --eval welsh-lexicon
    python -m deepeval_evals.run_all --model ollama/llama3 --max-samples 50
    python -m deepeval_evals.run_all --model gpt-4o --eval welsh-legislation-translation
"""

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from deepeval.test_case import LLMTestCase
from tqdm import tqdm

from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics.bleu_score import compute_corpus_bleu
from deepeval_evals.models import generate_response, resolve_hf_model_id

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "evals-cymraeg")

EVALS = {
    "welsh-lexicon": {
        "jsonl": "welsh-lexicon/data/welsh-lexicon/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-grammar": {
        "jsonl": "welsh-grammar/data/welsh-grammar/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-yes-no": {
        "jsonl": "welsh-yes-no/data/welsh-yes-no/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-obscenities": {
        "jsonl": "welsh-obscenities/data/welsh-obscenities/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-bilingual-placenames": {
        "jsonl": "welsh-bilingual-placenames/data/welsh-bilingual-placenames/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-legislation-translation": {
        "jsonl": "welsh-legislation-translation/data/welsh-legislation-translation/samples.jsonl",
        "metric": "bleu"
    },
    "welsh-registers": {
        "jsonl": "welsh-registers/data/welsh-registers/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-mmlu-lite": {
        "jsonl": "welsh-mmlu-lite/data/welsh-mmlu-lite/samples.jsonl",
        "metric": "mcq",
        "max_tokens": 10
    },
    "welsh-toxigen": {
        "jsonl": "welsh-toxigen/data/welsh-toxigen/samples.jsonl",
        "metric": "exact_match"
    },
    "welsh-arc-easy-mini-cy": {
        "jsonl": "welsh-arc-easy-mini-cy/data/welsh-arc-easy-mini-cy/samples.jsonl",
        "metric": "mcq",
        "max_tokens": 10
    }
}


def _generate_one(item: tuple) -> tuple[int, str]:
    """Run one completion; returns (index, text) so callers can preserve golden order."""
    idx, model_id, max_tokens, system_message, user_message, enable_thinking = item
    actual = generate_response(
        model_id,
        system_message,
        user_message,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )
    return idx, actual


def _score_sample(metric: str, actual: str, expected: str) -> float:
    """Score a single sample. Returns 1.0 for correct, 0.0 for incorrect (or BLEU placeholder)."""
    if metric == "mcq":
        m = re.search(r'\b([A-D])\b', actual)
        extracted = m.group(1) if m else actual.strip()
        return 1.0 if extracted == expected.strip() else 0.0
    elif metric == "exact_match":
        return 1.0 if actual.strip().strip(".,!?").lower() == expected.strip().strip(".,!?").lower() else 0.0
    return 0.0


def run_eval(
    eval_name: str,
    model_id: str,
    max_samples: int = None,
    details_dir: str = None,
    max_tokens_override: int = None,
    max_concurrent: int = 1,
    enable_thinking: bool | None = None,
):
    config = EVALS[eval_name]
    jsonl_path = os.path.join(BASE_DIR, config["jsonl"])

    print(f"\n{'='*60}")
    print(f"Eval: {eval_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)
    print(f"Loaded {len(goldens)} samples")

    details = []

    max_tokens = max_tokens_override or config.get("max_tokens", 500)
    n = len(goldens)
    predictions = [""] * n
    references = [g.expected_output for g in goldens]

    if max_concurrent <= 1:
        for i, g in enumerate(tqdm(goldens, desc="Generating responses")):
            _, actual = _generate_one(
                (i, model_id, max_tokens, g.system_message, g.user_message, enable_thinking)
            )
            predictions[i] = actual
    else:
        work = [
            (i, model_id, max_tokens, g.system_message, g.user_message, enable_thinking)
            for i, g in enumerate(goldens)
        ]
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(_generate_one, w): w[0] for w in work}
            with tqdm(total=n, desc="Generating responses") as pbar:
                for fut in as_completed(futures):
                    idx, actual = fut.result()
                    predictions[idx] = actual
                    pbar.update(1)

    test_cases = []
    for i, g in enumerate(goldens):
        actual = predictions[i]
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))
        if details_dir is not None:
            score = _score_sample(config["metric"], actual, g.expected_output)
            details.append({
                "input": g.user_message,
                "expected": g.expected_output,
                "output": actual,
                "score": score,
            })

    if details_dir is not None:
        os.makedirs(details_dir, exist_ok=True)
        details_path = os.path.join(details_dir, f"{eval_name}.jsonl")
        with open(details_path, "w") as f:
            for d in details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Details saved to {details_path}")

    # Summary
    if config["metric"] in ("exact_match", "mcq"):
        correct = sum(1 for tc in test_cases if _score_sample(config["metric"], tc.actual_output, tc.expected_output))
        accuracy = correct / len(test_cases) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})")
        return {"eval": eval_name, "metric": "accuracy", "score": f"{accuracy:.2f}", "n": len(test_cases)}
    else:
        corpus_bleu = compute_corpus_bleu(predictions, references)
        print(f"\nCorpus BLEU: {corpus_bleu:.1f}")
        return {"eval": eval_name, "metric": "BLEU", "score": f"{corpus_bleu:.1f}", "n": len(test_cases)}


def main():
    parser = argparse.ArgumentParser(description="Run Welsh LLM evaluations")
    parser.add_argument("--model", required=True, help="LLM model ID (e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3)")
    parser.add_argument("--eval", choices=list(EVALS.keys()), help="Run a specific eval (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per eval")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max generation tokens (default: per-eval setting, typically 500)")
    parser.add_argument("--output-dir", default="results", help="Directory to save results (default: results)")
    parser.add_argument("--save-details", action="store_true", help="Save per-sample details (input, output, expected, score) as JSONL")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Parallel completion requests (default: 8). Use 1 for sequential. Match your server (e.g. vLLM --max-num-seqs).",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable model thinking/reasoning trace (vLLM: extra_body chat_template_kwargs enable_thinking=false).",
    )
    thinking.add_argument(
        "--thinking",
        action="store_true",
        help="Force enable_thinking=true in the request (when server supports it).",
    )
    args = parser.parse_args()

    enable_thinking: bool | None = None
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    model_id = args.model
    if model_id.startswith("hf/"):
        actual_model = resolve_hf_model_id()
        print(f"HF server is serving: {actual_model}")
        model_id = f"hf/{actual_model}"

    eval_names = [args.eval] if args.eval else list(EVALS.keys())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_")

    details_dir = None
    if args.save_details:
        details_dir = os.path.join(args.output_dir, "details", f"{timestamp}_{model_slug}")

    print(f"Running {len(eval_names)} eval(s) with model: {model_id}")
    start = time.time()

    summaries = []
    for name in eval_names:
        summary = run_eval(
            name,
            model_id,
            args.max_samples,
            details_dir=details_dir,
            max_tokens_override=args.max_tokens,
            max_concurrent=args.max_concurrent,
            enable_thinking=enable_thinking,
        )
        summaries.append(summary)

    elapsed = time.time() - start

    # Write summary CSV
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{timestamp}_{model_slug}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "metric", "score", "n"])
        writer.writeheader()
        writer.writerows(summaries)

    # Print summary table
    print(f"\n{'='*50}")
    print(f"{'Eval':<35} {'Metric':<10} {'Score':>8} {'N':>6}")
    print(f"{'-'*50}")
    for s in summaries:
        print(f"{s['eval']:<35} {s['metric']:<10} {s['score']:>8} {s['n']:>6}")
    print(f"{'='*50}")
    print(f"Results saved to {csv_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
