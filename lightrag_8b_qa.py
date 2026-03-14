"""
Run 8B QA on all 3 datasets using cached LightRAG contexts.
Contexts are already cached from 70B runs — no LightRAG instance needed.

Usage:
    python -u lightrag_8b_qa.py
"""

import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set.")
    sys.exit(1)

_groq_sync = Groq(api_key=GROQ_API_KEY)

BASE = Path(__file__).resolve().parent
QA_MODEL = "llama-3.1-8b-instant"

# Import from qa_pipeline
sys.path.insert(0, str(BASE))
from qa_pipeline import (
    answer_with_context,
    answer_with_sparql_cot,
    eval_once,
    _match_question_entities,
    _bfs_entities,
    _expand_via_text,
    _estimate_words,
)

# Import parsers/compressors from existing scripts
from lightrag_musique import parse_lightrag_context, compress_lightrag_context

# ── Dataset configs ──────────────────────────────────────────────

DATASETS = {
    "hotpotqa": {
        "qa_path": BASE / "experiments" / "hotpotqa" / "large_scale" / "qa-pairs" / "qa-pairs.json",
        "context_cache": BASE / "experiments" / "hotpotqa" / "large_scale_lightrag" / "lightrag_contexts.json",
        "results_dir": BASE / "experiments" / "hotpotqa" / "large_scale_lightrag" / "results",
    },
    "musique": {
        "qa_path": BASE / "experiments" / "musique" / "large_scale" / "qa-pairs" / "qa-pairs.json",
        "context_cache": BASE / "experiments" / "musique" / "large_scale_lightrag" / "lightrag_contexts.json",
        "results_dir": BASE / "experiments" / "musique" / "large_scale_lightrag" / "results",
    },
    "2wikimultihopqa": {
        "qa_path": BASE / "experiments" / "2wikimultihopqa" / "large_scale" / "qa-pairs" / "qa-pairs.json",
        "context_cache": BASE / "experiments" / "2wikimultihopqa" / "large_scale_lightrag" / "lightrag_contexts.json",
        "results_dir": BASE / "experiments" / "2wikimultihopqa" / "large_scale_lightrag" / "results",
    },
}

QA_CONFIGS = [
    ("Baseline_8B",    False, False),
    ("Baseline_GW_8B", False, True),
    ("CoT_8B",         True,  False),
    ("CoT_GW_8B",      True,  True),
]


def run_dataset(ds_name, ds_cfg):
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name} (8B model)")
    print(f"{'='*60}")

    qa_list = json.loads(ds_cfg["qa_path"].read_text(encoding="utf-8"))
    contexts = json.loads(ds_cfg["context_cache"].read_text(encoding="utf-8"))
    results_dir = ds_cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {len(qa_list)} questions, {len(contexts)} cached contexts")

    # Build GW-compressed contexts
    print("  Building graph-walk compressed contexts...")
    gw_contexts = {}
    gw_stats = {"compressed": 0, "fallback": 0}
    t0 = time.time()
    for i, q in enumerate(qa_list):
        qid = str(q["id"])
        ctx = contexts.get(qid, "")
        compressed, meta = compress_lightrag_context(q["question"], ctx)
        gw_contexts[qid] = compressed
        if meta["mode"] == "compressed":
            gw_stats["compressed"] += 1
        else:
            gw_stats["fallback"] += 1
    print(f"    GW done: {gw_stats['compressed']} compressed, {gw_stats['fallback']} fallback ({time.time()-t0:.0f}s)")

    results = {}
    for config_name, use_cot, use_gw in QA_CONFIGS:
        checkpoint_path = results_dir / f"{config_name}.jsonl"

        # Load checkpoint
        done_rows = []
        done_ids = set()
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done_rows.append(row)
                    done_ids.add(str(row["id"]))
            print(f"\n  [{config_name}] Resuming: {len(done_rows)} already done")
        else:
            print(f"\n  [{config_name}] Starting fresh")

        rows = list(done_rows)
        t0 = time.time()
        total = len(qa_list)

        for i, q in enumerate(qa_list):
            qid = str(q["id"])
            if qid in done_ids:
                continue

            ctx = gw_contexts[qid] if use_gw else contexts.get(qid, "")
            question = q["question"]
            gold = q.get("answer", "")
            gold_list = q.get("answers", [gold])

            try:
                if use_cot:
                    pred, raw = answer_with_sparql_cot(
                        _groq_sync, QA_MODEL, question, ctx,
                        max_tokens=512, temperature=0.3,
                    )
                else:
                    pred = answer_with_context(
                        _groq_sync, QA_MODEL, question, ctx,
                        max_tokens=160, temperature=0.3,
                    )
                    raw = pred
            except Exception as e:
                pred = f"ERROR: {e}"
                raw = pred

            # Evaluate
            try:
                verdict = eval_once(
                    _groq_sync, "llama-3.1-8b-instant",
                    question, gold_list, pred,
                )
            except Exception as e:
                verdict = {"verdict": "unknown", "reason": str(e)[:200]}

            row = {
                "id": qid,
                "question": question,
                "gold": gold,
                "final_pred": pred,
                "eval_verdict": verdict["verdict"],
                "eval_reason": verdict.get("reason", ""),
                "config": config_name,
            }
            rows.append(row)
            done_ids.add(qid)

            # Append checkpoint
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            n_done = len(rows)
            if n_done % 25 == 0 or i == total - 1:
                acc = sum(1 for r in rows if r["eval_verdict"] == "correct") / len(rows) * 100
                q_safe = question[:40].encode("ascii", "replace").decode()
                print(f"    [{config_name}] {n_done}/{total} "
                      f"acc={acc:.1f}% elapsed={elapsed:.0f}s q={q_safe}")

        results[config_name] = rows
    return results


def print_summary(ds_name, results):
    print(f"\n  {'='*50}")
    print(f"  {ds_name} — LightRAG 8B Results")
    print(f"  {'='*50}")
    base_acc = None
    for config_name in ["Baseline_8B", "Baseline_GW_8B", "CoT_8B", "CoT_GW_8B"]:
        rows = results.get(config_name, [])
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["eval_verdict"] == "correct")
        acc = correct / n * 100 if n else 0
        if config_name == "Baseline_8B":
            base_acc = acc
        delta = f"  ({acc - base_acc:+.1f})" if base_acc is not None and config_name != "Baseline_8B" else ""
        print(f"    {config_name:18s}  {acc:5.1f}%  ({correct}/{n}){delta}")
    print(f"  {'='*50}")


if __name__ == "__main__":
    for ds_name, ds_cfg in DATASETS.items():
        results = run_dataset(ds_name, ds_cfg)
        print_summary(ds_name, results)

    print("\n\nAll 8B experiments complete!")
