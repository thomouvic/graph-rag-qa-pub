"""
run.py -- Run KET-RAG QA experiments (baseline + hybrid + evaluation).

Expects setup.py to have already created:
  experiments/{dataset}/{split}/
    qa-pairs/qa-pairs.json
    output/{split}-{strategy}-{theta}.json

Usage:
  python run.py --dataset hotpotqa --split small_scale
  python run.py --dataset hotpotqa --split small_scale --limit 5
  python run.py --dataset hotpotqa --split small_scale --skip-baseline
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SPLIT_CONFIGS = {
    "small_scale": {"n_queries": 50},
    "medium_100":  {"n_queries": 100},
    "medium_200":  {"n_queries": 200},
    "large_scale": {"n_queries": 500},
    "full_1000":   {"n_queries": 1000},
}

DEFAULTS = {
    "voter_models":       ["llama-3.1-8b-instant"] * 3,
    "arbiter_model":      "llama-3.1-8b-instant",
    "eval_model":         "llama-3.1-8b-instant",
    "baseline_model":     "llama-3.1-8b-instant",
    "voter_temp":         0.3,
    "baseline_temp":      0.3,
    "semantic_threshold": 0.82,
}


def build_paths(work_dir, dataset, split):
    return {
        "work_dir":     work_dir,
        "project_root": work_dir / "experiments" / dataset / split,
        "summaries":    work_dir / "summaries",
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run KET-RAG QA experiments")
    p.add_argument("--dataset",  required=True,
                   choices=["hotpotqa", "musique", "2wikimultihopqa"])
    p.add_argument("--split",    required=True,
                   help="Split name (e.g. large_scale, large_scale_70b)")
    p.add_argument("--strategy", default="keyword",
                   choices=["keyword", "text", "skeleton"])
    p.add_argument("--theta",    type=float, default=0.5)
    p.add_argument("--work-dir", type=Path, default=Path.cwd())
    p.add_argument("--limit",    type=int, default=None,
                   help="Process only first N QA pairs (for quick tests)")
    p.add_argument("--voter-models", nargs="+",
                   default=DEFAULTS["voter_models"])
    p.add_argument("--arbiter-model", default=DEFAULTS["arbiter_model"])
    p.add_argument("--eval-model",    default=DEFAULTS["eval_model"])
    p.add_argument("--baseline-model", default=DEFAULTS["baseline_model"])
    p.add_argument("--voter-temp",    type=float,
                   default=DEFAULTS["voter_temp"])
    p.add_argument("--baseline-temp", type=float,
                   default=DEFAULTS["baseline_temp"])
    p.add_argument("--semantic-threshold", type=float,
                   default=DEFAULTS["semantic_threshold"])
    p.add_argument("--graph-compress", action="store_true",
                   help="Compress context via graph-walk (keep chain-relevant entities/chunks)")
    p.add_argument("--graph-compress-hops", type=int, default=3,
                   help="Max BFS hops from question entities (default 3)")
    p.add_argument("--graph-compress-budget", type=int, default=4000,
                   help="Token budget for graph-compressed context (default 4000)")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-hybrid",   action="store_true")
    p.add_argument("--skip-sparql",   action="store_true")
    p.add_argument("--skip-gated",    action="store_true")
    p.add_argument("--skip-generic-cot", action="store_true")
    p.add_argument("--skip-compare",    action="store_true")
    p.add_argument("--clean-checkpoints", action="store_true",
                   help="Delete checkpoint files before starting (fresh run)")
    return p.parse_args()


def load_env(work_dir):
    try:
        from dotenv import load_dotenv
        env_path = work_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Put it in .env or export it.")
        sys.exit(1)


def sanity_check(root, split, strategy, theta):
    checks = {
        "settings.yaml":   root / "settings.yaml",
        "parquet index":    root / "output" / "create_final_nodes.parquet",
        "KET context JSON": root / "output" / f"{split}-{strategy}-{theta}.json",
        "qa-pairs.json":    root / "qa-pairs" / "qa-pairs.json",
    }
    ok = True
    for label, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {label:25s} {status}")
        if not exists:
            ok = False
    if not ok:
        print("\nRun setup.py first to generate missing files.")
        sys.exit(1)

    input_dir = root / "input"
    n_txt = len(list(input_dir.glob("*.txt"))) if input_dir.exists() else 0
    n_qa = len(json.loads(
        (root / "qa-pairs" / "qa-pairs.json").read_text(encoding="utf-8")
    ))
    print(f"  docs={n_txt}  qa_pairs={n_qa}")


def acc(series):
    series = series.astype(str)
    return float((series == "correct").mean()) if len(series) else 0.0


def main():
    args = parse_args()
    paths = build_paths(args.work_dir, args.dataset, args.split)
    root = paths["project_root"]
    key = f"{args.dataset}/{args.split}"
    safe_key = key.replace("/", "_")

    s = args.strategy
    t = args.theta
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hybrid_tag = f"hybrid_vote{len(args.voter_models)}_vtemp{args.voter_temp}_{s}_theta{t}"
    baseline_tag = f"baseline_{args.baseline_model}_temp{args.baseline_temp}_{s}_theta{t}"

    print(f"=== KET-RAG Run ===")
    print(f"Experiment: {key}")
    print(f"Strategy:   {s}  theta={t}")
    print(f"Limit:      {args.limit or 'all'}")
    print()

    # --- Sanity check ---
    print("Sanity check:")
    sanity_check(root, args.split, s, t)
    print()

    # --- Load env ---
    load_env(args.work_dir)

    # --- Build context lookup ---
    from qa_pipeline import (
        load_ket_records, record_to_qid_ctx,
        run_baseline_single_model, run_hybrid, run_sparql, run_generic_cot,
        run_gated, add_eval_column, error_analysis, call_groq_chat,
        build_graph_compressed_context_lookup,
    )
    from groq import Groq
    from sentence_transformers import SentenceTransformer

    print("Loading KET context records...")
    recs = load_ket_records(root, args.split, s, t)
    context_lookup = {}
    for rec in recs:
        qid, ctx = record_to_qid_ctx(rec)
        context_lookup[qid] = ctx
    nonempty = sum(1 for v in context_lookup.values() if v.strip())
    print(f"  records={len(context_lookup)}  nonempty={nonempty}")

    # --- Init clients ---
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Groq client + embedder ready.")

    # --- Model availability test ---
    needed = sorted(set(
        args.voter_models + [args.arbiter_model, args.eval_model, args.baseline_model]
    ))
    available = []
    for m in needed:
        try:
            out = call_groq_chat(
                groq_client, m,
                [{"role": "user", "content": "Reply with only: OK"}],
                max_tokens=5, temperature=0.0,
            )
            print(f"  {m} -> {out.strip()}")
            available.append(m)
        except Exception as e:
            print(f"  {m} FAILED: {repr(e)}")
    missing = set(needed) - set(available)
    if missing:
        print(f"ERROR: Models not available: {missing}")
        sys.exit(1)

    # --- Load QA pairs ---
    qa_path = root / "qa-pairs" / "qa-pairs.json"
    qa_list = json.loads(qa_path.read_text(encoding="utf-8"))
    print(f"\nLoaded {len(qa_list)} QA pairs")

    missing_ctx = [str(q["id"]) for q in qa_list
                   if not context_lookup.get(str(q["id"]), "").strip()]
    if missing_ctx:
        print(f"  WARNING: {len(missing_ctx)}/{len(qa_list)} QA pairs have no context")
    print()

    paths["summaries"].mkdir(parents=True, exist_ok=True)

    # --- Graph-walk context compression ---
    if args.graph_compress:
        print("=== GRAPH-WALK CONTEXT COMPRESSION ===")
        context_lookup, gw_stats = build_graph_compressed_context_lookup(
            qa_list, context_lookup,
            max_hops=args.graph_compress_hops,
            budget_tokens=args.graph_compress_budget,
        )
        print(f"  avg words: {gw_stats['avg_orig']:.0f} -> {gw_stats['avg_new']:.0f}"
              f"  (compressed={gw_stats['n_compressed']}, fallback={gw_stats['n_fallback']})")
        print(f"  avg seeds={gw_stats['avg_seeds']:.1f}  avg chain={gw_stats['avg_chain']:.1f}")
        print()

    # --- Checkpoint setup ---
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "baseline": ckpt_dir / "baseline.jsonl",
        "hybrid":   ckpt_dir / "hybrid.jsonl",
        "sparql":   ckpt_dir / "sparql.jsonl",
        "gated":    ckpt_dir / "gated.jsonl",
    }
    if args.clean_checkpoints:
        for p in ckpt.values():
            if p.exists():
                p.unlink()
        print("Cleared checkpoint files.")

    # --- Baseline ---
    df_baseline = None
    if not args.skip_baseline:
        print(f"=== BASELINE: {args.baseline_model} ===")
        results_dir = root / "results_baseline_groq"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{baseline_tag}_{safe_key}_{run_ts}.csv"

        df_baseline = run_baseline_single_model(
            groq_client, args.baseline_model, qa_list,
            context_lookup, temperature=args.baseline_temp,
            limit=args.limit, checkpoint_path=ckpt["baseline"],
        )
        df_baseline["experiment"] = key
        df_baseline = add_eval_column(groq_client, args.eval_model, df_baseline)
        df_baseline.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        base_acc = acc(df_baseline["eval_verdict"])
        base_abstain = float(df_baseline["final_abstain"].astype(bool).mean())
        print(f"  Accuracy: {base_acc:.3f}  Abstain rate: {base_abstain:.3f}")
        print(f"  n={len(df_baseline)}")

        summary_path = paths["summaries"] / f"summary_{baseline_tag}_{run_ts}.csv"
        pd.DataFrame([{
            "experiment": key, "n": len(df_baseline),
            "baseline_acc": base_acc, "abstain_rate": base_abstain,
        }]).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    else:
        print("[skip] Baseline")

    print()

    # --- Hybrid ---
    df_hybrid = None
    if not args.skip_hybrid:
        print(f"=== HYBRID: voters={args.voter_models} ===")
        results_dir = root / "results_hybrid_groq"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"{hybrid_tag}_{safe_key}_{run_ts}.csv"

        df_hybrid = run_hybrid(
            groq_client, qa_list, context_lookup,
            args.voter_models, args.arbiter_model,
            embedder=embedder, use_semantic_vote=True,
            voter_temp=args.voter_temp,
            semantic_threshold=args.semantic_threshold,
            limit=args.limit, checkpoint_path=ckpt["hybrid"],
        )
        df_hybrid["experiment"] = key
        df_hybrid = add_eval_column(groq_client, args.eval_model, df_hybrid)
        df_hybrid.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        hyb_acc = acc(df_hybrid["eval_verdict"])
        hyb_abstain = float(df_hybrid["final_abstain"].astype(bool).mean())
        hyb_escalated = float(df_hybrid["escalated_to_level2"].astype(bool).mean())
        print(f"  Accuracy: {hyb_acc:.3f}  Abstain: {hyb_abstain:.3f}  Escalated: {hyb_escalated:.3f}")

        summary_path = paths["summaries"] / f"summary_{hybrid_tag}_{run_ts}.csv"
        pd.DataFrame([{
            "experiment": key, "n": len(df_hybrid),
            "final_acc": hyb_acc, "abstain_rate": hyb_abstain,
            "escalated_rate": hyb_escalated,
        }]).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

        # Error analysis
        print("\n--- Error Analysis (hybrid) ---")
        error_dir = results_dir / f"wrong_groups_{safe_key}_{run_ts}"
        error_analysis(df_hybrid, error_dir)
    else:
        print("[skip] Hybrid")

    print()

    # --- SPARQL ---
    df_sparql = None
    if not args.skip_sparql:
        print(f"=== SPARQL: {args.baseline_model} ===")
        results_dir = root / "results_sparql_groq"
        results_dir.mkdir(parents=True, exist_ok=True)
        sparql_tag = f"sparql_{args.baseline_model}_temp{args.baseline_temp}_{s}_theta{t}"
        out_path = results_dir / f"{sparql_tag}_{safe_key}_{run_ts}.csv"

        df_sparql = run_sparql(
            groq_client, args.baseline_model, qa_list,
            context_lookup, temperature=args.baseline_temp,
            limit=args.limit, checkpoint_path=ckpt["sparql"],
        )
        df_sparql["experiment"] = key
        df_sparql = add_eval_column(groq_client, args.eval_model, df_sparql)
        df_sparql.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        sparql_acc = acc(df_sparql["eval_verdict"])
        sparql_abstain = float(df_sparql["final_abstain"].astype(bool).mean())
        print(f"  Accuracy: {sparql_acc:.3f}  Abstain rate: {sparql_abstain:.3f}")
        print(f"  n={len(df_sparql)}")

        summary_path = paths["summaries"] / f"summary_{sparql_tag}_{run_ts}.csv"
        pd.DataFrame([{
            "experiment": key, "n": len(df_sparql),
            "sparql_acc": sparql_acc, "abstain_rate": sparql_abstain,
        }]).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    else:
        print("[skip] SPARQL")

    print()

    # --- Generic CoT ---
    df_generic_cot = None
    if not args.skip_generic_cot:
        print(f"=== GENERIC COT: {args.baseline_model} ===")
        results_dir = root / "results_generic_cot_groq"
        results_dir.mkdir(parents=True, exist_ok=True)
        generic_tag = f"generic_cot_{args.baseline_model}_temp{args.baseline_temp}_{s}_theta{t}"
        out_path = results_dir / f"{generic_tag}_{safe_key}_{run_ts}.csv"

        ckpt["generic_cot"] = ckpt_dir / "generic_cot.jsonl"
        if args.clean_checkpoints and ckpt["generic_cot"].exists():
            ckpt["generic_cot"].unlink()

        df_generic_cot = run_generic_cot(
            groq_client, args.baseline_model, qa_list,
            context_lookup, temperature=args.baseline_temp,
            limit=args.limit, checkpoint_path=ckpt["generic_cot"],
        )
        df_generic_cot["experiment"] = key
        df_generic_cot = add_eval_column(groq_client, args.eval_model, df_generic_cot)
        df_generic_cot.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        gcot_acc = acc(df_generic_cot["eval_verdict"])
        gcot_abstain = float(df_generic_cot["final_abstain"].astype(bool).mean())
        print(f"  Accuracy: {gcot_acc:.3f}  Abstain rate: {gcot_abstain:.3f}")
        print(f"  n={len(df_generic_cot)}")
    else:
        print("[skip] Generic CoT")

    print()

    # --- Gated ---
    df_gated = None
    if not args.skip_gated:
        print(f"=== GATED (comparison->SPARQL, bridge->baseline): {args.baseline_model} ===")
        results_dir = root / "results_gated_groq"
        results_dir.mkdir(parents=True, exist_ok=True)
        gated_tag = f"gated_{args.baseline_model}_temp{args.baseline_temp}_{s}_theta{t}"
        out_path = results_dir / f"{gated_tag}_{safe_key}_{run_ts}.csv"

        df_gated = run_gated(
            groq_client, args.baseline_model, qa_list,
            context_lookup, temperature=args.baseline_temp,
            limit=args.limit, checkpoint_path=ckpt["gated"],
        )
        df_gated["experiment"] = key
        df_gated = add_eval_column(groq_client, args.eval_model, df_gated)
        df_gated.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        gated_acc = acc(df_gated["eval_verdict"])
        gated_abstain = float(df_gated["final_abstain"].astype(bool).mean())
        print(f"  Accuracy: {gated_acc:.3f}  Abstain rate: {gated_abstain:.3f}")
        print(f"  n={len(df_gated)}")

        summary_path = paths["summaries"] / f"summary_{gated_tag}_{run_ts}.csv"
        pd.DataFrame([{
            "experiment": key, "n": len(df_gated),
            "gated_acc": gated_acc, "abstain_rate": gated_abstain,
        }]).to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")
    else:
        print("[skip] Gated")

    print()

    # --- Compare ---
    if not args.skip_compare:
        results = {}
        if df_baseline is not None:
            results["Baseline"] = df_baseline
        if df_hybrid is not None:
            results["Hybrid"] = df_hybrid
        if df_sparql is not None:
            results["SPARQL"] = df_sparql
        if df_gated is not None:
            results["Gated"] = df_gated

        if "Baseline" in results and len(results) > 1:
            print("=== COMPARISON ===")
            base_acc_val = acc(df_baseline["eval_verdict"])
            print(f"  Baseline accuracy: {base_acc_val:.3f}")
            b_wrong_ids = set(df_baseline[df_baseline["eval_verdict"] != "correct"]["id"])

            for name, df in results.items():
                if name == "Baseline":
                    continue
                a = acc(df["eval_verdict"])
                delta = a - base_acc_val
                w_ids = set(df[df["eval_verdict"] != "correct"]["id"])
                fixed = b_wrong_ids - w_ids
                regressed = w_ids - b_wrong_ids
                print(f"  {name:8s} accuracy: {a:.3f}  (delta {delta:+.3f})"
                      f"  fixed={len(fixed)} regressed={len(regressed)}")
        elif not results or ("Baseline" not in results):
            print("[skip] Compare (need baseline + at least one other method)")

    print(f"\n{'='*60}")
    print(f"Done. Results in: {root}")


if __name__ == "__main__":
    main()
