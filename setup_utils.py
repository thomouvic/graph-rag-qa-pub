"""
Setup utilities for KET-RAG experiments.

Handles: repo cloning, KET-RAG patching, dependency installation,
embedding-server lifecycle, and settings.yaml configuration.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
import urllib.request

import yaml


# ── subprocess helpers ─────────────────────────────────────────────

def _run(cmd, **kwargs):
    """Simple subprocess wrapper for setup commands."""
    print("$", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], text=True, capture_output=True, **kwargs)
    if r.stdout:
        print(r.stdout[-2000:])
    if r.returncode != 0 and r.stderr:
        print(r.stderr[-2000:])
    return r


def run_cmd(cmd, cwd=None):
    """Run a shell command with nice printing + fail fast."""
    print("\n$", " ".join(str(c) for c in cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        env=env,
        encoding="utf-8",
        errors="replace",
    )
    if r.stdout:
        print(r.stdout[-3000:].encode("ascii", "replace").decode())
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr[-3000:].encode("ascii", "replace").decode())
        raise RuntimeError(
            f"Command failed (rc={r.returncode}): {' '.join(str(c) for c in cmd)}"
        )
    return r


# ── repo cloning ───────────────────────────────────────────────────

def clone_repos(repo_dir: Path, ketrag_dir: Path, hippo_dir: Path):
    """Clone KET-RAG and HippoRAG repos if not already present."""
    hippo_dataset_dir = hippo_dir / "reproduce" / "dataset"

    if not (ketrag_dir / "pyproject.toml").exists():
        if ketrag_dir.exists():
            shutil.rmtree(ketrag_dir)
        _run(["git", "clone", "https://github.com/waetr/KET-RAG.git", str(ketrag_dir)])
        print("Cloned KET-RAG")
    else:
        print(f"KET-RAG already present at {ketrag_dir}")

    if not (hippo_dataset_dir / "hotpotqa.json").exists():
        if hippo_dir.exists():
            shutil.rmtree(hippo_dir)
        _run(["git", "clone", "https://github.com/OSU-NLP-Group/HippoRAG.git", str(hippo_dir)])
        print("Cloned HippoRAG")
    else:
        print(f"HippoRAG already present at {hippo_dir}")


# ── KET-RAG patching ──────────────────────────────────────────────

def _patch_file(path: Path):
    """Patch a single KET-RAG source file to read API bases from env vars."""
    import re
    text = path.read_text(encoding="utf-8")
    if "KET_LLM_API_BASE" in text:
        print(f"{path.name} already patched")
        return

    # Patch llm_model and embedding_model defaults
    text = text.replace(
        'llm_model = "gpt-4o-mini"',
        'llm_model = os.environ.get("KET_LLM_MODEL", "gpt-4o-mini")',
    )
    text = text.replace(
        'embedding_model = "text-embedding-3-small"',
        'embedding_model = os.environ.get("KET_EMBEDDING_MODEL", "text-embedding-3-small")',
    )
    # Patch ChatOpenAI: add api_base before api_type (handles trailing comments)
    text = re.sub(
        r'(llm = ChatOpenAI\(\s*api_key=api_key,\s*model=llm_model,\s*)(api_type=OpenaiApiType\.OpenAI)',
        r'\1api_base=os.environ.get("KET_LLM_API_BASE"),\n        \2',
        text,
    )
    # Patch OpenAIEmbedding: replace api_base=None with env var
    text = text.replace(
        "api_base=None,",
        'api_base=os.environ.get("KET_EMBEDDING_API_BASE"),',
    )
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path.name} for custom API bases")


def patch_ketrag(ketrag_dir: Path):
    """Patch KET-RAG source files to support custom API bases via env vars."""
    _patch_file(ketrag_dir / "indexing_sket" / "create_context.py")
    _patch_file(ketrag_dir / "indexing_sket" / "llm_answer.py")


# ── dependency install ─────────────────────────────────────────────

def install_deps(ketrag_dir: Path):
    """Install KET-RAG + graphrag via pip editable install, NLTK data, and extra pip packages."""
    _run([sys.executable, "-m", "pip", "install", "-e", str(ketrag_dir)])

    _run([
        sys.executable, "-c",
        "import nltk; [nltk.download(x) for x in ['stopwords','punkt','punkt_tab']]",
    ])

    r = _run([sys.executable, "-c", "import graphrag; print('graphrag OK')"])
    assert r.returncode == 0, "graphrag import failed"

    for pkg in ["groq", "sentence-transformers", "scikit-learn", "python-dotenv"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            _run([sys.executable, "-m", "pip", "install", "-q", pkg])

    print("\nSetup complete.")


# ── settings.yaml patching ─────────────────────────────────────────

def patch_settings_yaml(project_root: Path):
    """
    Patch settings.yaml to use:
    - Groq for LLM (via OpenAI-compatible API)
    - Local embedding server for embeddings
    - LanceDB in the project root
    """
    settings_path = project_root / "settings.yaml"
    if not settings_path.exists():
        return

    cfg = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

    cfg.setdefault("llm", {})
    cfg["llm"]["api_key"] = "${GRAPHRAG_API_KEY}"
    cfg["llm"]["type"] = "openai_chat"
    cfg["llm"]["model"] = os.environ.get("KET_LLM_MODEL", "llama-3.1-8b-instant")
    cfg["llm"]["api_base"] = os.environ.get(
        "KET_LLM_API_BASE", "https://api.groq.com/openai/v1"
    )
    # Rate limiting for Groq paid tier (env-overridable for larger models)
    cfg["llm"]["requests_per_minute"] = int(os.environ.get("KET_RPM", 800))
    cfg["llm"]["tokens_per_minute"] = int(os.environ.get("KET_TPM", 200000))
    cfg["llm"]["concurrent_requests"] = 25
    cfg["llm"]["max_retries"] = 30
    cfg["llm"]["max_retry_wait"] = 30.0

    cfg.setdefault("embeddings", {}).setdefault("llm", {})
    # KET-RAG needs entity description embeddings too
    cfg["embeddings"]["target"] = "all"
    cfg["embeddings"]["llm"]["api_key"] = "dummy"
    cfg["embeddings"]["llm"]["type"] = "openai_embedding"
    cfg["embeddings"]["llm"]["model"] = os.environ.get(
        "KET_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )
    cfg["embeddings"]["llm"]["api_base"] = os.environ.get(
        "KET_EMBEDDING_API_BASE", "http://localhost:8000/v1"
    )
    cfg["embeddings"]["llm"]["concurrent_requests"] = 5
    cfg["embeddings"]["llm"]["max_retries"] = 30
    cfg["embeddings"]["llm"]["max_retry_wait"] = 30.0

    db_uri = str(project_root / "lancedb-new").replace("\\", "/")
    cfg["embeddings"].setdefault("vector_store", {})
    cfg["embeddings"]["vector_store"]["type"] = "lancedb"
    cfg["embeddings"]["vector_store"]["db_uri"] = db_uri
    cfg["embeddings"]["vector_store"]["overwrite"] = True

    settings_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"Patched settings.yaml: LLM -> Groq, embeddings -> local, db_uri -> {db_uri}")


# ── embedding server lifecycle ─────────────────────────────────────

def start_embedding_server(work_dir: Path, port: int = 8000):
    """
    Launch embedding_server.py in the background and wait for it to be ready.
    Returns the subprocess.Popen handle (or None if already running).
    """
    # Check if already running
    try:
        urllib.request.urlopen(f"http://localhost:{port}/v1/embeddings", timeout=2)
        print(f"Embedding server already running on port {port}")
        return None
    except Exception:
        pass

    server_script = work_dir / "embedding_server.py"
    assert server_script.exists(), f"Missing: {server_script}"

    proc = subprocess.Popen(
        [sys.executable, str(server_script), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started embedding server (PID {proc.pid}) on port {port}")

    for i in range(60):
        time.sleep(2)
        try:
            req = urllib.request.Request(
                f"http://localhost:{port}/v1/embeddings",
                data=json.dumps({"input": "test", "model": "all-MiniLM-L6-v2"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            if resp.status == 200:
                print(f"Embedding server ready after {(i + 1) * 2}s")
                return proc
        except Exception:
            pass

    raise RuntimeError("Embedding server did not start in time")


def set_ket_env_vars(
    groq_api_base: str, embedding_base: str, llm_model: str, embedding_model: str
):
    """Set environment variables consumed by the patched KET-RAG code."""
    os.environ["KET_LLM_API_BASE"] = groq_api_base
    os.environ["KET_LLM_MODEL"] = llm_model
    os.environ["KET_EMBEDDING_API_BASE"] = embedding_base
    os.environ["KET_EMBEDDING_MODEL"] = embedding_model
    print(f"KET_LLM_API_BASE: {groq_api_base}")
    print(f"KET_LLM_MODEL: {llm_model}")
    print(f"KET_EMBEDDING_API_BASE: {embedding_base}")
    print(f"KET_EMBEDDING_MODEL: {embedding_model}")
