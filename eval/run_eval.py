"""Eval harness for the research assistant.

Runs every entry in eval_set.json through the agent pipeline and scores:
  - retrieval_hit:    did any retrieved chunk come from the expected source?
  - mention_hit:      does the final answer contain any of the expected keywords?
  - critic_supported: did the critic think the answer was grounded?

Exits non-zero if any metric's pass-rate drops below its threshold — suitable
to wire into CI as a regression gate.

Run from the repo root:
    python -m eval.run_eval
"""

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from agents import run_query

# Gemini free tier is ~10 RPM per model; each query fires 3-5 calls.
# Sleep between items to stay comfortably under the limit.
SLEEP_BETWEEN_ITEMS_S = 15

load_dotenv()

EVAL_PATH = Path(__file__).parent / "eval_set.json"

# Regression thresholds — tune as the corpus grows
THRESHOLDS = {
    "retrieval_hit": 0.75,
    "mention_hit": 0.50,
    "critic_supported": 0.75,
}


def score_one(item: dict) -> dict:
    question = item["question"]
    expected_src = item["expected_source_substring"].lower()
    must_mention = [m.lower() for m in item.get("must_mention_any", [])]

    result = run_query(question)

    chunks = result.get("chunks", [])
    answer = result.get("answer", "").lower()
    critic = result.get("critic", {})

    retrieval_hit = any(expected_src in c["source"].lower() for c in chunks)
    mention_hit = any(m in answer for m in must_mention) if must_mention else True
    critic_supported = bool(critic.get("supported", False))

    return {
        "question": question,
        "type": item.get("type"),
        "retrieval_hit": retrieval_hit,
        "mention_hit": mention_hit,
        "critic_supported": critic_supported,
        "latency_ms": result.get("latency_ms"),
        "attempts": result.get("attempts"),
    }


def main() -> int:
    items = json.loads(EVAL_PATH.read_text())
    print(f"Running {len(items)} eval items...\n")

    results = []
    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['question'][:70]}")
        r = score_one(item)
        results.append(r)
        flags = "".join(
            [
                "R" if r["retrieval_hit"] else ".",
                "M" if r["mention_hit"] else ".",
                "C" if r["critic_supported"] else ".",
            ]
        )
        print(f"    {flags}  {r['latency_ms']}ms  attempts={r['attempts']}\n")
        if i < len(items):
            time.sleep(SLEEP_BETWEEN_ITEMS_S)

    # Aggregate
    n = len(results)
    pass_rates = {
        metric: sum(r[metric] for r in results) / n for metric in THRESHOLDS
    }

    print("=" * 60)
    print(f"{'Metric':<20} {'Pass rate':>10} {'Threshold':>12} {'Status':>10}")
    print("-" * 60)
    any_failed = False
    for metric, threshold in THRESHOLDS.items():
        rate = pass_rates[metric]
        ok = rate >= threshold
        any_failed = any_failed or not ok
        status = "PASS" if ok else "FAIL"
        print(f"{metric:<20} {rate:>10.1%} {threshold:>12.1%} {status:>10}")
    print("=" * 60)

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
