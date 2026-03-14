# SPARQL CoT + Graph-Walk Compression for Graph-RAG QA

Reproduction code for the paper *"Structured Reasoning over Retrieval: SPARQL-Guided Graph-Walk Augmentation for Graph-RAG"*.

We evaluate two augmentations for multi-hop question answering over Graph-RAG systems:
1. **SPARQL Chain-of-Thought prompting** -- decomposes questions into triple-pattern queries aligned with entity-relationship context
2. **Graph-walk context compression** -- reduces context by ~60% via knowledge-graph traversal (no LLM calls)

Evaluated on three benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) using [KET-RAG](https://github.com/waetr/KET-RAG) and [LightRAG](https://github.com/HKUDS/LightRAG), with Llama-3.1-8B and Llama-3.3-70B via [Groq](https://groq.com/).

## Requirements

- Python 3.10-3.12
- [Groq API key](https://console.groq.com/) (paid tier recommended for indexing speed)
- ~4 GB disk space
- No GPU needed

## Quick Start

```bash
git clone https://github.com/anonymous/graph-rag-qa.git
cd graph-rag-qa
cp .env.example .env
# Edit .env: add your Groq API key

# One-time setup: install deps, patch KET-RAG, build index, create context
python setup.py --dataset hotpotqa --split large_scale

# Run QA experiments
python run.py --dataset hotpotqa --split large_scale
```

## Repo Structure

```
setup.py                 One-time setup: patch, install, index, create context
run.py                   Run QA experiments (baseline + SPARQL CoT + GW)
qa_pipeline.py           Core QA logic: Groq API calls, evaluation, error analysis
data_prep.py             Load benchmark data, select splits, convert formats
setup_utils.py           Setup helpers: patching, settings.yaml, embedding server
embedding_server.py      Local OpenAI-compatible embedding server (MiniLM-L6-v2)

compute_f1.py            Compute F1/EM metrics with SQuAD normalization
normalize_answers.py     LLM-based answer normalization for 8B outputs
find_coverage.py         Measure context coverage (gold answer in retrieved context)
compare_results.py       Compare results across configurations

lightrag_experiment.py   LightRAG replication experiment (2WikiMHQA)
lightrag_hotpotqa.py     LightRAG experiment for HotpotQA
lightrag_musique.py      LightRAG experiment for MuSiQue
lightrag_8b_qa.py        LightRAG QA with 8B model across all datasets

KET-RAG/                 Bundled KET-RAG (includes GraphRAG v0.4.1)
datasets/                Benchmark datasets (from HippoRAG)
```

## Datasets

The `datasets/` directory contains benchmark data from [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG):

| Dataset | Questions | Hops | Source |
|---------|-----------|------|--------|
| HotpotQA | 7,405 | 2 | Wikipedia paragraphs |
| MuSiQue | 2,417 | 2-4 | Composed single-hop questions |
| 2WikiMultiHopQA | 12,576 | 2 | Wikidata + Wikipedia |

We sample N=500 questions per dataset (seed=42) for all experiments.

## Model Stack

| Component | Model | Cost |
|-----------|-------|------|
| Embeddings | MiniLM-L6-v2 (local) | Free |
| Index LLM | Llama-3.1-8B (Groq) | $0.05/M |
| QA (budget) | Llama-3.1-8B (Groq) | $0.05/M |
| QA (standard) | Llama-3.3-70B (Groq) | $0.59/M |

The graph index is model-independent: build once with 8B, then run QA with any model.

## Experiment Configurations

Each dataset is evaluated with 4 methods x 2 models = 8 configurations:

1. **Baseline** -- direct QA over full KET-RAG context (~10,000 tokens)
2. **Baseline + GW** -- direct QA over graph-walk compressed context (~4,000 tokens)
3. **SPARQL CoT** -- SPARQL chain-of-thought prompting over full context
4. **SPARQL CoT + GW** -- SPARQL CoT over compressed context

## CLI Reference

### setup.py

```bash
python setup.py --dataset hotpotqa --split large_scale [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | hotpotqa | hotpotqa, musique, 2wikimultihopqa |
| `--split` | small_scale | small_scale, medium_100, large_scale, full_1000 |
| `--strategy` | keyword | keyword, text, skeleton |
| `--theta` | 0.5 | Entity-keyword balance |
| `--skip-install` | | Skip dependency installation |
| `--skip-index` | | Skip GraphRAG indexing |

### run.py

```bash
python run.py --dataset hotpotqa --split large_scale [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | required | hotpotqa, musique, 2wikimultihopqa |
| `--split` | required | Must match setup.py |
| `--baseline-model` | llama-3.1-8b-instant | QA model |
| `--baseline-temp` | 0.3 | Sampling temperature |
| `--limit` | all | Process first N questions only |

## LightRAG Replication

The LightRAG experiments require installing [LightRAG](https://github.com/HKUDS/LightRAG) separately. Run per-dataset scripts after building LightRAG indexes:

```bash
python lightrag_hotpotqa.py    # Index + retrieve for HotpotQA
python lightrag_musique.py     # Index + retrieve for MuSiQue
python lightrag_experiment.py  # Index + retrieve for 2WikiMHQA
python lightrag_8b_qa.py       # Run QA across all datasets
```

## Known Issues

- **Groq rate limiting during indexing**: GraphRAG makes many concurrent LLM calls. With the free tier (30 RPM), expect frequent retries. Paid tier (1K RPM) is much faster.
- **Entity embeddings**: GraphRAG may not generate `embeddings.entity.description.parquet` even with `target: all`. Generate manually using the embedding server if missing.
- **Windows console encoding**: Unicode in question text may show as `?` in progress output; the underlying data is unaffected.

## License

The experiment code in this repository is released under the MIT License. The bundled KET-RAG code retains its original license. The benchmark datasets are from HippoRAG and retain their original licenses.

## Citation

```bibtex
@inproceedings{anonymous2025sparql,
  title={Structured Reasoning over Retrieval: SPARQL-Guided Graph-Walk Augmentation for Graph-RAG},
  author={Anonymous},
  year={2025}
}
```
