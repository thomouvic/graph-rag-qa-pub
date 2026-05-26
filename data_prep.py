"""
Data preparation utilities for KET-RAG experiments.

Loads HippoRAG benchmark data, selects experiment splits,
converts to KET-RAG format, and writes to disk.
"""

import json
import random
from pathlib import Path


def load_hipporag_dataset(dataset_dir: Path, dataset_name: str):
    """
    Load corpus and queries from HippoRAG reproduce/dataset/.
    Returns (corpus, queries) in HippoRAG's native format.
    """
    corpus_path = dataset_dir / f"{dataset_name}_corpus.json"
    queries_path = dataset_dir / f"{dataset_name}.json"

    assert corpus_path.exists(), f"Missing: {corpus_path}"
    assert queries_path.exists(), f"Missing: {queries_path}"

    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))

    print(f"  {dataset_name}: {len(corpus)} corpus docs, {len(queries)} queries")
    return corpus, queries


def select_split(queries, corpus, n_queries, seed=42):
    """
    Select n_queries queries and relevant corpus subset.
    Uses per-question context paragraphs (gold + benchmark distractors),
    pooled and deduplicated — matching the KET-RAG paper methodology.
    For large splits (>=500): full corpus.
    """
    rng = random.Random(seed)

    if n_queries >= len(queries):
        selected_queries = queries
    else:
        selected_queries = rng.sample(queries, n_queries)

    # Pool all per-question context paragraph titles (gold + distractors)
    needed_titles = set()
    for q in selected_queries:
        if "context" in q:  # HotpotQA / 2Wiki: list of [title, sentences]
            for title, _sentences in q["context"]:
                needed_titles.add(title)
        elif "paragraphs" in q:  # MuSiQue
            for p in q["paragraphs"]:
                needed_titles.add(p["title"])

    selected_corpus = [d for d in corpus if d.get("title") in needed_titles]
    print(f"  Pooled context: {len(needed_titles)} unique titles -> {len(selected_corpus)} corpus docs")
    return selected_queries, selected_corpus


def convert_queries_to_qa_pairs(queries: list) -> list:
    """
    Normalize HippoRAG query format to KET-RAG qa-pairs format.
    Handles both _id (HotpotQA/2Wiki) and id (MuSiQue).
    """
    qa_pairs = []
    for q in queries:
        qid = str(q.get("id") or q.get("_id"))
        answer = str(q.get("answer", ""))
        aliases = q.get("answer_aliases", [])
        answers_list = [answer] + [str(a) for a in aliases if str(a) != answer]

        qa_pairs.append({
            "id": qid,
            "question": q["question"],
            "answer": answer,
            "answers": answers_list,
        })
    return qa_pairs


def write_txt_inputs(project_root: Path, corpus: list) -> Path:
    """Write corpus docs as input/*.txt for GraphRAG."""
    input_dir = project_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for i, doc in enumerate(corpus):
        doc_id = str(doc.get("idx", i))
        title = str(doc.get("title", ""))
        text = str(doc.get("text", ""))
        (input_dir / f"{doc_id}.txt").write_text(f"{title}\n\n{text}", encoding="utf-8")
    return input_dir


def write_qa_pairs(project_root: Path, qa_pairs: list) -> Path:
    """Write qa-pairs.json in KET-RAG format."""
    out_path = project_root / "qa-pairs" / "qa-pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(qa_pairs, indent=2), encoding="utf-8")
    return out_path


def prepare_experiment(
    project_root: Path,
    dataset_dir: Path,
    dataset_name: str,
    split_name: str,
    split_configs: dict,
):
    """
    Full pipeline: load HippoRAG data -> select split -> convert -> write.
    Skips if already prepared.
    """
    n_queries = split_configs[split_name]["n_queries"]
    key = f"{dataset_name}/{split_name}"

    qa_path = project_root / "qa-pairs" / "qa-pairs.json"
    input_dir = project_root / "input"
    if qa_path.exists() and input_dir.exists() and any(input_dir.iterdir()):
        n_txt = len(list(input_dir.glob("*.txt")))
        n_qa = len(json.loads(qa_path.read_text(encoding="utf-8")))
        print(f"{key}: already prepared ({n_txt} docs, {n_qa} queries) -- skipping")
        return

    print(f"\nPreparing {key} ...")
    corpus, queries = load_hipporag_dataset(dataset_dir, dataset_name)
    sel_queries, sel_corpus = select_split(queries, corpus, n_queries)
    qa_pairs = convert_queries_to_qa_pairs(sel_queries)

    project_root.mkdir(parents=True, exist_ok=True)
    write_txt_inputs(project_root, sel_corpus)
    write_qa_pairs(project_root, qa_pairs)

    print(f"  -> {len(sel_corpus)} docs, {len(qa_pairs)} queries written to {project_root}")
