"""
setup.py -- One-time setup for a KET-RAG experiment.

Creates:
  experiments/{dataset}/{split}/
    input/*.txt, qa-pairs/qa-pairs.json, settings.yaml,
    output/ (graphrag index artifacts + KET context JSON)

Usage:
  python setup.py --dataset hotpotqa --split small_scale
  python setup.py --dataset hotpotqa --split small_scale --skip-index
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SPLIT_CONFIGS = {
    "small_scale": {"n_queries": 50},
    "medium_100":  {"n_queries": 100},
    "medium_200":  {"n_queries": 200},
    "large_scale": {"n_queries": 500},
    "full_1000":   {"n_queries": 1000},
}

GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_INDEX_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL   = "all-MiniLM-L6-v2"


def build_paths(work_dir, dataset, split):
    return {
        "work_dir":     work_dir,
        "ketrag_dir":   work_dir / "KET-RAG",
        "hippo_data":   work_dir / "datasets",
        "project_root": work_dir / "experiments" / dataset / split,
        "summaries":    work_dir / "summaries",
    }


def load_env(work_dir):
    try:
        from dotenv import load_dotenv
        env_path = work_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"Loaded .env from {env_path}")
    except ImportError:
        pass

    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Put it in .env or export it.")
        sys.exit(1)
    os.environ["GRAPHRAG_API_KEY"] = os.environ["GROQ_API_KEY"]


def parse_args():
    p = argparse.ArgumentParser(description="KET-RAG experiment setup")
    p.add_argument("--dataset",  default="hotpotqa",
                   choices=["hotpotqa", "musique", "2wikimultihopqa"])
    p.add_argument("--split",    default="small_scale",
                   choices=list(SPLIT_CONFIGS))
    p.add_argument("--strategy", default="keyword",
                   choices=["keyword", "text", "skeleton"])
    p.add_argument("--theta",    type=float, default=0.5)
    p.add_argument("--work-dir", type=Path, default=Path.cwd())
    p.add_argument("--port",     type=int, default=8000)
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--skip-data",    action="store_true")
    p.add_argument("--skip-init",    action="store_true")
    p.add_argument("--skip-index",   action="store_true")
    p.add_argument("--skip-context", action="store_true")
    p.add_argument("--index-model",  default=DEFAULT_INDEX_MODEL,
                   help="LLM for GraphRAG indexing (default: llama-3.1-8b-instant)")
    p.add_argument("--index-suffix", default="",
                   help="Suffix appended to split dir name (e.g. '_70b')")
    return p.parse_args()


def main():
    args = parse_args()
    split_dir = args.split + args.index_suffix
    paths = build_paths(args.work_dir, args.dataset, split_dir)
    root = paths["project_root"]

    print(f"=== KET-RAG Setup ===")
    print(f"Dataset:  {args.dataset}")
    print(f"Split:    {args.split} (dir: {split_dir})")
    print(f"Index LLM: {args.index_model}")
    print(f"Strategy: {args.strategy}")
    print(f"Theta:    {args.theta}")
    print(f"Project:  {root}")
    print()

    # --- Step 1: Patch KET-RAG source ---
    from setup_utils import patch_ketrag
    patch_ketrag(paths["ketrag_dir"])

    # --- Step 2: Install dependencies ---
    if not args.skip_install:
        from setup_utils import install_deps
        install_deps(paths["ketrag_dir"])
    else:
        print("[skip] Install deps")

    # --- Step 3: Prepare experiment data ---
    if not args.skip_data:
        from data_prep import prepare_experiment
        prepare_experiment(
            root, paths["hippo_data"],
            args.dataset, args.split, SPLIT_CONFIGS,
        )
    else:
        print("[skip] Data prep")

    # --- Step 4: Start embedding server ---
    from setup_utils import start_embedding_server, set_ket_env_vars

    _emb_proc = start_embedding_server(args.work_dir, args.port)
    set_ket_env_vars(
        GROQ_API_BASE,
        f"http://localhost:{args.port}/v1",
        args.index_model,
        EMBED_MODEL,
    )

    # --- Step 5: Load API keys ---
    load_env(args.work_dir)

    # --- Step 6: GraphRAG init ---
    if not args.skip_init:
        from setup_utils import run_cmd, patch_settings_yaml
        if not (root / "settings.yaml").exists():
            run_cmd([sys.executable, "-m", "graphrag", "init", "--root", str(root)],
                    cwd=paths["ketrag_dir"])
        patch_settings_yaml(root)
        print(f"GraphRAG init done")
    else:
        from setup_utils import patch_settings_yaml
        if (root / "settings.yaml").exists():
            patch_settings_yaml(root)
        print("[skip] GraphRAG init")

    # --- Step 7: GraphRAG index ---
    if not args.skip_index:
        from setup_utils import run_cmd
        print(f"\n=== INDEXING ===")
        try:
            run_cmd(
                [sys.executable, "-m", "graphrag", "index", "--root", str(root), "--reporter", "print"],
                cwd=paths["ketrag_dir"],
            )
        except RuntimeError:
            # GraphRAG may return non-zero exit code due to transient rate-limit
            # errors that were actually retried successfully. Check if output exists.
            if (root / "output" / "create_final_entities.parquet").exists():
                print("WARNING: graphrag index reported errors but output exists — continuing.")
            else:
                raise
    else:
        print("[skip] GraphRAG index")

    # --- Step 8: KET create_context ---
    if not args.skip_context:
        import subprocess as _sp
        print(f"\n=== CREATE CONTEXT: {args.strategy} theta={args.theta} ===")
        ctx_cmd = [sys.executable, "indexing_sket/create_context.py",
                   str(root), args.strategy, str(args.theta)]
        print("$", " ".join(str(c) for c in ctx_cmd))
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        rc = _sp.call(ctx_cmd, cwd=str(paths["ketrag_dir"]), env=env)
        if rc != 0:
            raise RuntimeError(f"create_context failed (rc={rc})")
    else:
        print("[skip] KET create_context")

    # --- Done ---
    print(f"\n{'='*60}")
    print(f"Setup complete for {args.dataset}/{args.split}")
    print(f"Project root: {root}")
    print(f"Next: python run.py --dataset {args.dataset} --split {args.split} "
          f"--strategy {args.strategy} --theta {args.theta}")


if __name__ == "__main__":
    main()
