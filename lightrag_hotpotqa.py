"""
LightRAG generality experiment — HotpotQA.

Demonstrates that SPARQL CoT and graph-walk compression help beyond KET-RAG
by applying them to LightRAG's entity-relationship context.

Stages:
    1. Index HotpotQA corpus (4,927 docs) with LightRAG
    2. Retrieve context for 500 questions
    3. Run 4 QA configurations (Base/CoT × ±GW) with 70B model
    4. Evaluate and print results

Prerequisites:
    python embedding_server.py   # must be running on port 8000
    GROQ_API_KEY in .env
"""

import asyncio
import json
import os
import re
import sys
import time
import random
import unicodedata
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set.")
    sys.exit(1)

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

# ── Paths ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
INPUT_DIR = BASE / "experiments" / "hotpotqa" / "large_scale" / "input"
QA_PATH = BASE / "experiments" / "hotpotqa" / "large_scale" / "qa-pairs" / "qa-pairs.json"
WORK_DIR = BASE / "experiments" / "hotpotqa" / "large_scale_lightrag"
RESULTS_DIR = BASE / "experiments" / "hotpotqa" / "large_scale_lightrag" / "results"
CONTEXT_CACHE = WORK_DIR / "lightrag_contexts.json"

EMBED_BASE_URL = "http://localhost:8000/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
INDEX_MODEL = "llama-3.1-8b-instant"
QA_MODEL = "llama-3.3-70b-versatile"

# ── Custom LLM/embedding wrappers ────────────────────────────────

_groq_async = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
_embed_async = AsyncOpenAI(api_key="not-needed", base_url=EMBED_BASE_URL)
_groq_sync = Groq(api_key=GROQ_API_KEY)


async def groq_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    resp = await _groq_async.chat.completions.create(
        model=INDEX_MODEL,
        messages=messages,
        max_tokens=kwargs.get("max_tokens", 1024),
        temperature=kwargs.get("temperature", 0.0),
    )
    return resp.choices[0].message.content.strip()


async def local_embed_func(texts):
    resp = await _embed_async.embeddings.create(
        input=texts, model="all-MiniLM-L6-v2",
    )
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# ── Reuse from qa_pipeline.py ────────────────────────────────────

sys.path.insert(0, str(BASE))
from qa_pipeline import (
    call_groq_chat,
    answer_with_context,
    answer_with_sparql_cot,
    eval_once,
    extract_first_json_object,
    normalize_answer,
    alias_equivalent,
    _match_question_entities,
    _bfs_entities,
    _expand_via_text,
    _estimate_words,
)


# ── LightRAG context parser ──────────────────────────────────────

def parse_lightrag_context(context):
    """Parse LightRAG context into KET-RAG-compatible dict."""
    result = {
        "entities": {},
        "relationships": [],
        "sources": [],
        "chunks": [],
        "adj": {},
    }
    if not context:
        return result

    blocks = re.findall(r'```json\s*\n(.*?)```', context, re.DOTALL)
    sections = re.split(r'```json\s*\n.*?```', context, flags=re.DOTALL)

    for i, block in enumerate(blocks):
        header = sections[i] if i < len(sections) else ""
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]

        if "Entity" in header:
            for line in lines:
                try:
                    obj = json.loads(line)
                    name = (obj.get("entity") or obj.get("entity_name") or "").upper()
                    desc = obj.get("description", "")
                    if name:
                        result["entities"][name] = desc
                except json.JSONDecodeError:
                    continue

        elif "Relationship" in header:
            for line in lines:
                try:
                    obj = json.loads(line)
                    src = (obj.get("entity1") or obj.get("src_id") or "").upper()
                    tgt = (obj.get("entity2") or obj.get("tgt_id") or "").upper()
                    desc = obj.get("description", "")
                    weight = float(obj.get("weight", 1.0))
                    if src and tgt:
                        result["relationships"].append((src, tgt, desc, weight))
                        result["adj"].setdefault(src, set()).add(tgt)
                        result["adj"].setdefault(tgt, set()).add(src)
                except (json.JSONDecodeError, ValueError):
                    continue

        elif "Chunk" in header or "Document" in header:
            for j, line in enumerate(lines):
                try:
                    obj = json.loads(line)
                    cid = obj.get("reference_id", f"chunk_{j}")
                    content = obj.get("content", "")
                    if content:
                        result["chunks"].append((cid, content))
                except json.JSONDecodeError:
                    continue
    return result


# ── Graph-walk compression (adapted for parsed LightRAG context) ──

def compress_lightrag_context(question, context, max_hops=3, budget_tokens=4000):
    """Apply graph-walk compression to LightRAG context."""
    parsed = parse_lightrag_context(context)
    orig_words = _estimate_words(context)

    if not parsed["entities"]:
        return context, {"mode": "no_entities", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    seeds = _match_question_entities(question, list(parsed["entities"].keys()))
    if not seeds:
        return context, {"mode": "no_seeds", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    # BFS
    chain = _bfs_entities(parsed["adj"], seeds, max_hops)
    text_expanded = _expand_via_text(chain, set(parsed["entities"].keys()),
                                      parsed["chunks"])
    chain.update(text_expanded)
    chain_set = set(chain.keys())

    # Assemble compressed context by hop level
    budget_words = int(budget_tokens / 1.3)
    max_hop_seen = max(chain.values()) if chain else 0

    hop_entities = {}
    for ent, hop in chain.items():
        hop_entities.setdefault(hop, []).append(ent)
    for h in hop_entities:
        hop_entities[h].sort()

    rel_by_src = {}
    for src, tgt, desc, _ in parsed["relationships"]:
        if src in chain_set and tgt in chain_set:
            rel_by_src.setdefault(src, []).append((tgt, desc))

    q_content = {w for w in re.findall(r"[a-z0-9]+", (question or "").lower())
                 if len(w) > 2}

    chunk_hop_assignment = {}
    tier2 = []
    chunk_map = {cid: text for cid, text in parsed["chunks"]}
    for cid, text in parsed["chunks"]:
        text_low = text.lower()
        best_hop, chain_score = None, 0
        for e in chain_set:
            if e.lower() in text_low:
                chain_score += 1
                h = chain[e]
                if best_hop is None or h < best_hop:
                    best_hop = h
        if chain_score > 0:
            chunk_hop_assignment[cid] = (best_hop, chain_score)
        else:
            chunk_words = set(re.findall(r"[a-z0-9]+", text_low))
            kw_score = len(q_content & chunk_words)
            if kw_score > 0:
                tier2.append((kw_score, cid, text))
    tier2.sort(key=lambda x: -x[0])

    parts = []
    used_words = 0
    used_cids = set()
    chunk_parts_count = 0
    rel_lines_count = 0

    for hop in range(max_hop_seen + 1):
        ents_at_hop = hop_entities.get(hop, [])
        if not ents_at_hop:
            continue
        hop_lines = []
        hop_label = "from question" if hop == 0 else f"hop {hop}"
        hop_lines.append(f"=== Step {hop} ({hop_label}) ===")

        for ent in ents_at_hop:
            desc = parsed["entities"].get(ent, "")
            hop_lines.append(f"  {ent}: {desc}" if desc else f"  {ent}")
            for tgt, rdesc in rel_by_src.get(ent, []):
                hop_lines.append(f"    -> {tgt}: {rdesc}")
                rel_lines_count += 1

        hop_chunks = [(cid, chunk_hop_assignment[cid][1])
                      for cid in chunk_hop_assignment
                      if chunk_hop_assignment[cid][0] == hop and cid not in used_cids]
        hop_chunks.sort(key=lambda x: -x[1])

        hop_block = "\n".join(hop_lines)
        block_words = _estimate_words(hop_block)
        parts.append(hop_block)
        used_words += block_words

        if used_words < budget_words:
            for cid, score in hop_chunks:
                text = chunk_map.get(cid, "")
                w = _estimate_words(text)
                if used_words + w > budget_words:
                    continue
                parts.append(f"  [{cid}] {text}")
                used_words += w
                used_cids.add(cid)
                chunk_parts_count += 1

    if tier2:
        t2_parts = []
        for score, cid, text in tier2:
            if cid in used_cids:
                continue
            w = _estimate_words(text)
            if used_words + w > budget_words:
                continue
            t2_parts.append(f"  [{cid}] {text}")
            used_words += w
            used_cids.add(cid)
            chunk_parts_count += 1
        if t2_parts:
            parts.append("=== Additional context ===")
            parts.extend(t2_parts)

    current = "\n\n".join(parts)
    new_words = _estimate_words(current)
    return current, {
        "mode": "compressed",
        "orig_words": orig_words,
        "new_words": new_words,
        "seeds": len(seeds),
        "chain": len(chain),
        "n_chunks_kept": chunk_parts_count,
        "n_rels_kept": rel_lines_count,
    }


# ── Stage 1: Index corpus ────────────────────────────────────────

async def index_corpus():
    """Index all HotpotQA documents with LightRAG."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    embed_func = EmbeddingFunc(
        embedding_dim=384,
        func=local_embed_func,
        max_token_size=512,
        model_name="all-MiniLM-L6-v2",
    )
    rag = LightRAG(
        working_dir=str(WORK_DIR),
        llm_model_func=groq_llm_func,
        llm_model_name=INDEX_MODEL,
        embedding_func=embed_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        llm_model_max_async=8,
        embedding_func_max_async=4,
        entity_extract_max_gleaning=0,   # skip gleaning pass (halves LLM calls)
        force_llm_summary_on_merge=999,  # skip LLM entity-merge calls
        max_parallel_insert=4,           # process 4 docs concurrently (default 2)
    )
    await rag.initialize_storages()

    # ── Throttle disk persistence (200 MB per persist is the bottleneck) ──
    PERSIST_EVERY = 25          # persist every 25 docs instead of every 1
    _persist_counter = {"n": 0}
    _real_insert_done = rag._insert_done

    async def _throttled_insert_done(pipeline_status=None, pipeline_status_lock=None):
        _persist_counter["n"] += 1
        if _persist_counter["n"] % PERSIST_EVERY == 0:
            await _real_insert_done(pipeline_status, pipeline_status_lock)

    rag._insert_done = _throttled_insert_done

    doc_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"\n=== Stage 1: Index {len(doc_files)} documents ===")
    print(f"  Working dir: {WORK_DIR}")
    print(f"  Persist throttle: every {PERSIST_EVERY} docs")

    # Check how many are already indexed
    status_file = WORK_DIR / "doc_status.json"
    n_existing = 0
    if status_file.exists():
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            n_existing = sum(1 for v in status.values()
                            if isinstance(v, dict) and v.get("status") == "PROCESSED")
        except Exception:
            pass

    if n_existing >= len(doc_files):
        print(f"  Already indexed {n_existing}/{len(doc_files)} — skipping.")
        await rag.finalize_storages()
        return rag

    print(f"  Already indexed: {n_existing}/{len(doc_files)}")

    # Batch insert for efficiency
    BATCH_SIZE = 50
    docs_text = []
    for f in doc_files:
        docs_text.append(f.read_text(encoding="utf-8").strip())

    t0 = time.time()
    for start in range(0, len(docs_text), BATCH_SIZE):
        batch = docs_text[start:start + BATCH_SIZE]
        await rag.ainsert(batch)
        elapsed = time.time() - t0
        done = min(start + BATCH_SIZE, len(docs_text))
        rate = done / elapsed * 60 if elapsed > 0 else 0
        print(f"  Indexed {done}/{len(docs_text)} ({elapsed:.0f}s, {rate:.1f} docs/min)")

    # Final persist to make sure everything is saved
    await _real_insert_done()
    await rag.finalize_storages()
    print(f"  Indexing complete ({time.time() - t0:.0f}s)")

    # Summarize long entity descriptions
    await summarize_long_entities()

    return rag


# ── Stage 1b: Post-indexing entity summarization ─────────────────

SUMMARY_TOKEN_THRESHOLD = 300   # summarize entities longer than this
GRAPH_FIELD_SEP = "<SEP>"       # LightRAG's default description separator

async def summarize_long_entities():
    """One-time pass: summarize entity descriptions that got too long."""
    import networkx as nx

    graph_path = WORK_DIR / "graph_chunk_entity_relation.graphml"
    if not graph_path.exists():
        print("  No graph found, skipping entity summarization.")
        return

    G = nx.read_graphml(str(graph_path))
    print(f"\n=== Stage 1b: Summarize long entity descriptions ===")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Find entities with long descriptions
    long_entities = []
    for node, data in G.nodes(data=True):
        desc = data.get("description", "")
        est_tokens = len(desc.split()) / 0.75
        if est_tokens > SUMMARY_TOKEN_THRESHOLD and GRAPH_FIELD_SEP in desc:
            long_entities.append((node, desc, int(est_tokens)))

    if not long_entities:
        print("  No entities need summarization.")
        return

    long_entities.sort(key=lambda x: -x[2])
    print(f"  Found {len(long_entities)} entities above {SUMMARY_TOKEN_THRESHOLD} tokens")
    print(f"  Longest: '{long_entities[0][0]}' ({long_entities[0][2]} tokens)")

    summarized = 0
    t0 = time.time()
    for i, (name, desc, tok) in enumerate(long_entities):
        prompt = (
            f"Below are multiple description fragments for the entity '{name}', "
            f"separated by '{GRAPH_FIELD_SEP}'. Merge them into a single concise "
            f"description (2-4 sentences). Keep all key facts. Output ONLY the merged description.\n\n"
            f"{desc}"
        )
        try:
            summary = await groq_llm_func(prompt, max_tokens=256, temperature=0.0)
            G.nodes[name]["description"] = summary
            summarized += 1
        except Exception as e:
            print(f"  WARNING: Failed to summarize '{name}': {e}")

        if (i + 1) % 50 == 0 or i == len(long_entities) - 1:
            print(f"  Summarized {i+1}/{len(long_entities)} ({time.time()-t0:.0f}s)")

    # Write updated graph
    nx.write_graphml(G, str(graph_path))
    print(f"  Done: summarized {summarized}/{len(long_entities)} entities ({time.time()-t0:.0f}s)")


# ── Stage 2: Retrieve contexts ───────────────────────────────────

async def retrieve_contexts(rag):
    """Retrieve LightRAG context for each of 500 questions."""
    if CONTEXT_CACHE.exists():
        print(f"\n=== Stage 2: Loading cached contexts from {CONTEXT_CACHE.name} ===")
        contexts = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
        print(f"  Loaded {len(contexts)} cached contexts")
        return contexts

    qa_list = json.loads(QA_PATH.read_text(encoding="utf-8"))
    print(f"\n=== Stage 2: Retrieve context for {len(qa_list)} questions ===")

    contexts = {}
    t0 = time.time()
    for i, q in enumerate(qa_list):
        qid = str(q["id"])
        question = q["question"]
        try:
            ctx = await rag.aquery(
                question,
                QueryParam(mode="hybrid", only_need_context=True),
            )
            contexts[qid] = ctx if ctx else ""
        except Exception as e:
            print(f"  WARNING: Query failed for q{i}: {e}")
            contexts[qid] = ""

        if (i + 1) % 50 == 0 or i == len(qa_list) - 1:
            elapsed = time.time() - t0
            print(f"  Retrieved {i+1}/{len(qa_list)} ({elapsed:.0f}s)")

    # Save cache
    CONTEXT_CACHE.write_text(json.dumps(contexts, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print(f"  Saved contexts to {CONTEXT_CACHE.name}")
    return contexts


# ── Stage 3: QA pipeline ─────────────────────────────────────────

def run_qa_configs(qa_list, contexts):
    """Run 4 QA configurations and return results dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build GW-compressed contexts
    print("\n  Building graph-walk compressed contexts...")
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
        if (i + 1) % 100 == 0:
            print(f"    GW: {i+1}/{len(qa_list)} ({time.time()-t0:.0f}s)")
    print(f"    GW done: {gw_stats['compressed']} compressed, {gw_stats['fallback']} fallback")

    configs = [
        ("Baseline",    False, False),
        ("Baseline+GW", False, True),
        ("CoT",         True,  False),
        ("CoT+GW",      True,  True),
    ]

    results = {}
    for config_name, use_cot, use_gw in configs:
        checkpoint_path = RESULTS_DIR / f"{config_name.replace('+','_')}.jsonl"
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


# ── Stage 4: Summary ──────────────────────────────────────────────

def print_summary(results):
    print("\n" + "=" * 60)
    print("LightRAG Generality Experiment — HotpotQA (70B)")
    print("=" * 60)

    base_acc = None
    for config_name in ["Baseline", "Baseline+GW", "CoT", "CoT+GW"]:
        rows = results.get(config_name, [])
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["eval_verdict"] == "correct")
        acc = correct / n * 100 if n else 0
        if config_name == "Baseline":
            base_acc = acc
        delta = f"  ({acc - base_acc:+.1f})" if base_acc is not None and config_name != "Baseline" else ""
        print(f"  {config_name:15s}  {acc:5.1f}%  ({correct}/{n}){delta}")

    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────

async def main():
    # Stage 1: Index
    rag = await index_corpus()

    # Reopen for querying if needed
    if not isinstance(rag, LightRAG):
        embed_func = EmbeddingFunc(
            embedding_dim=384,
            func=local_embed_func,
            max_token_size=512,
            model_name="all-MiniLM-L6-v2",
        )
        rag = LightRAG(
            working_dir=str(WORK_DIR),
            llm_model_func=groq_llm_func,
            llm_model_name=INDEX_MODEL,
            embedding_func=embed_func,
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            llm_model_max_async=4,
            embedding_func_max_async=4,
            force_llm_summary_on_merge=999,
        )
        await rag.initialize_storages()

    # Stage 2: Retrieve
    contexts = await retrieve_contexts(rag)
    await rag.finalize_storages()

    # Stage 3: QA
    qa_list = json.loads(QA_PATH.read_text(encoding="utf-8"))
    print(f"\n=== Stage 3: Run 4 QA configurations ({len(qa_list)} questions × 4) ===")
    results = run_qa_configs(qa_list, contexts)

    # Stage 4: Summary
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
