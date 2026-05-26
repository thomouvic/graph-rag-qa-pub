# SPARQL CoT, Graph-Walk Compression, and Question-Type Routing for Graph-RAG QA

Reproduction code for the paper *"The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA"*.

We evaluate three inference-time augmentations for multi-hop question answering over Graph-RAG systems:

1. **SPARQL Chain-of-Thought prompting** decomposes questions into triple-pattern queries aligned with entity-relationship context.
2. **Graph-walk context compression (GW)** reduces context by approximately 60% via knowledge-graph BFS traversal, with no LLM calls.
3. **Question-type routing** dispatches questions to SPARQL CoT or generic CoT based on a lightweight classifier, with retry on abstain.

Evaluated on three benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA) using [KET-RAG](https://github.com/waetr/KET-RAG) and [LightRAG](https://github.com/HKUDS/LightRAG), with Llama-3.1-8B and Llama-3.3-70B via [Groq](https://groq.com/).

## Requirements

- Python 3.10 to 3.12
- [Groq API key](https://console.groq.com/) (paid tier recommended for indexing speed)
- Approximately 4 GB disk space
- No GPU needed

## Quick Start

```bash
git clone https://github.com/anonymous/graph-rag-qa.git
cd graph-rag-qa
cp .env.example .env
# Edit .env: add your Groq API key

# One-time setup: install deps, patch KET-RAG, build index, create context
python setup.py --dataset hotpotqa --split large_scale

# Run QA experiments (baseline + SPARQL CoT + generic CoT, with and without GW)
python run.py --dataset hotpotqa --split large_scale --graph-compress

# Run with Self-Ask baseline added
python run.py --dataset hotpotqa --split large_scale --run-self-ask
```

## Repo Structure

```
setup.py                       One-time setup: patch, install, index, create context
run.py                         Run QA experiments (configurations below)
qa_pipeline.py                 Core QA logic: prompting, GW compression, routing, abstain, evaluation
data_prep.py                   Load benchmark data, select splits, convert formats
setup_utils.py                 Setup helpers: patching, settings.yaml, embedding server
embedding_server.py            Local OpenAI-compatible embedding server (MiniLM-L6-v2)

compute_f1.py                  F1/EM metrics with SQuAD normalization
normalize_answers.py           LLM-based answer normalization for 8B outputs
find_coverage.py               Substring-based context coverage (gold answer in context)
supporting_fact_coverage.py    Bridging-entity coverage (M3 metric) using native annotations
verify_coverage.py             Cross-check coverage outputs
compare_results.py             Compare results across configurations
embed_watchdog.py              Local embedding server health check

lightrag_experiment.py         LightRAG replication (2WikiMHQA)
lightrag_hotpotqa.py           LightRAG experiment (HotpotQA)
lightrag_musique.py            LightRAG experiment (MuSiQue)
lightrag_8b_qa.py              LightRAG QA with 8B across all datasets
normalize_lightrag_answers.py  Answer normalization for LightRAG outputs

revision/                      Revision-cycle analysis scripts and outputs (see below)

KET-RAG/                       Bundled KET-RAG (includes GraphRAG v0.4.1)
datasets/                      Benchmark datasets from HippoRAG
```

## The revision/ directory

Scripts and outputs supporting the analyses added in the May 2026 revision cycle.

```
revision/
  smoke_self_ask.py                      Run Self-Ask baseline (Press et al. 2023 prompts)
  extended_table5.py                     Build extended Table 5 (SPARQL vs Generic CoT, 3 datasets)
  run_classifier.py                      Two-way (bridge / non-bridge) routing classifier
  run_classifier_3way.py                 Three-way (bridge / comparison / inference) routing classifier
  routing_analysis.py                    Routing aggregate accuracy (no GW)
  routing_analysis_gw.py                 Routing aggregate accuracy (with GW)
  sparql_syntax_validity.py              SPARQL parse-rate analysis
  sparql_vs_generic_per_type.py          Per-native-type SPARQL vs Generic (no GW)
  sparql_vs_generic_per_type_gw.py       Per-native-type SPARQL vs Generic (with GW)
  diagnose_abstain.py                    Abstain-rate diagnosis across configurations
  sanity_check_m3.py                     Cross-check bridging-entity coverage outputs

  coverage/                              Bridging-entity (M3) coverage outputs per dataset
  classifier_predictions_*.json          Cached routing classifier outputs (per dataset, 2-way and 3-way)

  compression_baselines.md               Documentation of truncation and top-k embedding baselines
  data_inventory.md                      Mapping from result tables to source CSVs
  routing_validation.md                  Routing analysis writeup
  self_ask_results.md                    Self-Ask baseline comparison writeup
  sparql_syntax_validity.md              SPARQL parse-rate analysis writeup
  sparql_vs_generic_per_type.md          Per-type breakdown writeup
```

## Datasets

The `datasets/` directory contains benchmark data from [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG):

| Dataset         | Questions | Hops | Source                          |
|-----------------|-----------|------|---------------------------------|
| HotpotQA        | 7,405     | 2    | Wikipedia paragraphs            |
| MuSiQue         | 2,417     | 2-4  | Composed single-hop questions   |
| 2WikiMultiHopQA | 12,576    | 2    | Wikidata + Wikipedia            |

We sample N=500 questions per dataset (seed=42) for all main experiments.

## Model Stack

| Component     | Model                | Cost      |
|---------------|----------------------|-----------|
| Embeddings    | MiniLM-L6-v2 (local) | Free      |
| Index LLM     | Llama-3.1-8B (Groq)  | $0.05/M   |
| QA (budget)   | Llama-3.1-8B (Groq)  | $0.05/M   |
| QA (standard) | Llama-3.3-70B (Groq) | $0.59/M   |

The graph index is model-independent. Build once with 8B, then run QA with any model without re-indexing.

## Experiment Configurations

Each dataset is evaluated with multiple methods across two models:

1. **Baseline** runs direct QA over full KET-RAG context (approximately 10,000 tokens).
2. **Baseline + GW** runs direct QA over graph-walk compressed context (approximately 4,000 tokens).
3. **Generic CoT** runs natural-language chain-of-thought prompting over full context.
4. **Generic CoT + GW** runs generic CoT over compressed context.
5. **SPARQL CoT** runs SPARQL chain-of-thought prompting over full context.
6. **SPARQL CoT + GW** runs SPARQL CoT over compressed context.
7. **Routing + GW** dispatches via the question-type classifier and retries on abstain.
8. **Self-Ask** runs Press et al. 2023 Self-Ask as a multi-hop CoT baseline (optional flag).

## CLI Reference

### setup.py

```bash
python setup.py --dataset hotpotqa --split large_scale [OPTIONS]
```

| Option           | Default     | Description                                |
|------------------|-------------|--------------------------------------------|
| `--dataset`      | hotpotqa    | hotpotqa, musique, 2wikimultihopqa         |
| `--split`        | small_scale | small_scale, medium_100, large_scale       |
| `--strategy`     | keyword     | keyword, text, skeleton                    |
| `--theta`        | 0.5         | Entity-keyword balance                     |
| `--skip-install` |             | Skip dependency installation               |
| `--skip-index`   |             | Skip GraphRAG indexing                     |

### run.py

```bash
python run.py --dataset hotpotqa --split large_scale [OPTIONS]
```

Core options:

| Option            | Default               | Description                                |
|-------------------|-----------------------|--------------------------------------------|
| `--dataset`       | required              | hotpotqa, musique, 2wikimultihopqa         |
| `--split`         | required              | Must match setup.py                        |
| `--baseline-model`| llama-3.1-8b-instant  | QA model                                   |
| `--baseline-temp` | 0.3                   | Sampling temperature                       |
| `--limit`         | all                   | Process first N questions only             |

Graph-walk compression:

| Option                     | Default | Description                                |
|----------------------------|---------|--------------------------------------------|
| `--graph-compress`         |         | Enable graph-walk compression              |
| `--graph-compress-hops`    | 3       | BFS depth                                  |
| `--graph-compress-budget`  | 4000    | Token budget for compressed context        |

SPARQL CoT variants:

| Option                       | Default | Description                                |
|------------------------------|---------|--------------------------------------------|
| `--sparql-prompt-variant`    | generic | SPARQL prompt template variant             |

Self-Ask baseline:

| Option                    | Default | Description                                |
|---------------------------|---------|--------------------------------------------|
| `--run-self-ask`          |         | Run Self-Ask baseline                      |
| `--self-ask-demos-from`   | auto    | Few-shot demo source (Press et al. tables) |
| `--self-ask-max-tokens`   | 2048    | Max tokens for Self-Ask responses          |

Skip flags for selective re-runs:

| Option                | Description                                |
|-----------------------|--------------------------------------------|
| `--skip-baseline`     | Skip the baseline configuration            |
| `--skip-sparql`       | Skip SPARQL CoT                            |
| `--skip-generic-cot`  | Skip generic CoT                           |

## Reproducing the revision experiments

Once the main `run.py` outputs are available for a dataset and model, the analyses in `revision/` reuse those CSVs.

**SPARQL parse-rate analysis** (Appendix A.6):
```bash
python revision/sparql_syntax_validity.py
```

**Per-native-type SPARQL vs Generic CoT** (Appendix A.7):
```bash
python revision/sparql_vs_generic_per_type.py        # no GW
python revision/sparql_vs_generic_per_type_gw.py     # with GW
```

**Routing classifier** (3-way, used for the deployed configuration):
```bash
python revision/run_classifier_3way.py --dataset hotpotqa
python revision/routing_analysis_gw.py
```

**Bridging-entity (M3) coverage** (Appendix A.5):
```bash
python supporting_fact_coverage.py --dataset hotpotqa
```

**Self-Ask comparison** (Appendix A.4):
```bash
python run.py --dataset hotpotqa --split large_scale --run-self-ask
```

**Compression baselines** (Appendix A.9): see `revision/compression_baselines.md` for runner commands for truncation and top-k embedding similarity at the same 4k budget.

## LightRAG replication

The LightRAG experiments require installing [LightRAG](https://github.com/HKUDS/LightRAG) separately. Run per-dataset scripts after building LightRAG indexes:

```bash
python lightrag_hotpotqa.py    # Index + retrieve for HotpotQA
python lightrag_musique.py     # Index + retrieve for MuSiQue
python lightrag_experiment.py  # Index + retrieve for 2WikiMHQA
python lightrag_8b_qa.py       # Run QA across all datasets
```

## Known issues

- **Groq rate limiting during indexing.** GraphRAG makes many concurrent LLM calls. Free tier (30 RPM) leads to frequent retries. Paid tier (1K RPM) is much faster.
- **Entity embeddings.** GraphRAG sometimes does not generate `embeddings.entity.description.parquet` even with `target: all`. Generate manually using the embedding server if missing.
- **Windows console encoding.** Unicode in question text may show as `?` in progress output. The underlying data is unaffected.

## License

The experiment code in this repository is released under the MIT License. The bundled KET-RAG code retains its original license. The benchmark datasets are from HippoRAG and retain their original licenses.

## Citation

```bibtex
@inproceedings{anonymous2026reasoning,
  title={The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA},
  author={Anonymous},
  year={2026}
}
```
