"""
QA pipeline for KET-RAG hybrid majority-voting experiments.

Contains: Groq API wrapper, answer normalization, semantic/alias voting,
Level-2 arbiter, hybrid and baseline runners, LLM-judge evaluation,
and error-grouping analysis.

All functions take explicit parameters (client, embedder, etc.) —
no module-level globals.
"""

import json
import re
import time
import random
import unicodedata

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ── Checkpointing helpers ─────────────────────────────────────────

def _load_checkpoint(path):
    """Load completed rows from a JSONL checkpoint file. Returns (list[dict], set[str])."""
    rows = []
    done_ids = set()
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                row = json.loads(line)
                rows.append(row)
                done_ids.add(str(row["id"]))
        print(f"  [checkpoint] Loaded {len(rows)} completed items from {path}")
    return rows, done_ids


def _append_checkpoint(path, row):
    """Append a single result row to a JSONL checkpoint file."""
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Groq API wrapper ──────────────────────────────────────────────

def call_groq_chat(
    client,
    model: str,
    messages,
    max_tokens=256,
    temperature=0.3,
    retries=10,
    base_sleep=1.0,
):
    """Call Groq chat API with exponential-backoff retry on transient errors."""
    last_err = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            msg = repr(e).lower()
            transient = any(
                k in msg
                for k in [
                    "429", "rate", "timeout", "timed out",
                    "498", "capacity",
                    "502", "503", "504", "gateway", "overloaded", "temporarily",
                ]
            )
            if (not transient) or attempt == retries - 1:
                raise
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
            print(f"Transient error, retry {attempt + 1}/{retries} in {sleep_s:.1f}s")
            time.sleep(sleep_s)
    raise last_err


# ── answer helpers ─────────────────────────────────────────────────

def answer_with_context(
    client, model: str, question: str, context: str,
    max_tokens=160, temperature=0.0,
):
    """Ask a single QA question using the given context."""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using ONLY the provided context. "
                "Reply with ONLY the final answer (few words). "
                "Do NOT include explanation. "
                "If the answer is not clearly in the context, reply exactly: I don't know"
            ),
        },
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
    ]
    return call_groq_chat(
        client, model, messages, max_tokens=max_tokens, temperature=temperature,
    )


def classify_question_type(question: str) -> str:
    """Heuristic classifier: 'comparison' or 'bridge'.

    Comparison questions compare two entities (dates, populations, attributes).
    Everything else is treated as bridge (entity chain-following).
    """
    q = question.lower().strip()
    # "both" anywhere — strong comparison signal
    if re.search(r'\bboth\b', q):
        return 'comparison'
    # "share" / "in common" — shared-property questions
    if re.search(r'\b(share|shares|shared|in common)\b', q):
        return 'comparison'
    # "same" — "from the same country", "the same profession"
    if re.search(r'\bsame\b', q):
        return 'comparison'
    # "more ... than" — explicit comparison
    if re.search(r'\bmore\b.*\bthan\b', q):
        return 'comparison'
    # Question word + "or" — two-entity choice
    if re.search(r'\bor\b', q) and re.match(r'^(which|who|what|is|are|does|did|do|was|were|has|have|whose)\b', q):
        return 'comparison'
    # Comparative phrases without "or"
    if re.search(r'\b(born first|born earlier|born before|died first|lived longer)\b', q):
        return 'comparison'
    # "Between X and Y ..."
    if re.match(r'^(in between|between)\b', q):
        return 'comparison'
    # "either" — "are either X or Y"
    if re.search(r'\beither\b', q):
        return 'comparison'
    # "these two" / "the two"
    if re.search(r'\b(these two|the two)\b', q):
        return 'comparison'
    return 'bridge'


def answer_with_sparql_cot(client, model, question, context,
                           max_tokens=512, temperature=0.0):
    """Answer a question by first reformulating as SPARQL (chain-of-thought).

    Single LLM call: the model writes a SPARQL query to trace relationships,
    then extracts the final answer. Returns (answer, sparql) tuple.
    """
    prompt = (
        "You are answering a multi-hop question using ONLY the provided context.\n\n"
        "Step 1: Write a simple SPARQL query (max 4 triple patterns, plain English\n"
        "predicates, NO URIs, NO FILTER, NO subqueries). Example:\n"
        "  SELECT ?answer WHERE {\n"
        '    ?x name "Paradise Creek" .\n'
        "    ?x tributaryOf ?y .\n"
        "    ?y tributaryOf ?answer .\n"
        "  }\n"
        "Step 2: Follow the SPARQL chain step by step through the context.\n"
        "Step 3: Write FINAL ANSWER: <your answer in a few words>\n\n"
        "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )
    raw = call_groq_chat(client, model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)
    # Extract the last "FINAL ANSWER: <text>" from the response
    answer = raw
    matches = re.findall(r'(?i)FINAL\s*ANSWER\s*:\s*(.+)', raw)
    if matches:
        # Take the last match (model may echo template early, real answer comes last)
        candidate = matches[-1].strip()
        # Ignore template placeholders like "<your answer in a few words>"
        if candidate and not candidate.startswith("<"):
            answer = candidate
    return answer, raw



def answer_with_generic_cot(client, model, question, context,
                            max_tokens=512, temperature=0.0):
    """Answer a question using generic chain-of-thought decomposition.

    Single LLM call: the model breaks the question into sub-questions,
    finds answers in context, then combines. Returns (answer, raw_cot) tuple.
    Same output format as SPARQL CoT (FINAL ANSWER:) for fair comparison.
    """
    prompt = (
        "You are answering a multi-hop question using ONLY the provided context.\n\n"
        "Step 1: Break down the question into simpler sub-questions that need\n"
        "to be answered one at a time.\n"
        "Step 2: For each sub-question, find the relevant information in the\n"
        "context and write the intermediate answer.\n"
        "Step 3: Combine the intermediate answers to reach the final answer.\n"
        "Step 4: Write FINAL ANSWER: <your answer in a few words>\n\n"
        "If the answer is not in the context, write FINAL ANSWER: I don't know\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )
    raw = call_groq_chat(client, model,
                         [{"role": "user", "content": prompt}],
                         max_tokens=max_tokens, temperature=temperature)
    answer = raw
    matches = re.findall(r'(?i)FINAL\s*ANSWER\s*:\s*(.+)', raw)
    if matches:
        candidate = matches[-1].strip()
        if candidate and not candidate.startswith("<"):
            answer = candidate
    return answer, raw


def is_abstain(pred: str) -> bool:
    """Check if a prediction is an abstention."""
    if pred is None:
        return True
    p = str(pred).strip().lower()
    p = re.sub(r"\s+", " ", p)
    p = p.strip(" \t\n\r\f\v.?!:;\"'`~()[]{}")
    return p in {
        "i don't know", "i do not know", "unknown",
        "not sure", "cannot determine", "can't tell",
    }


def normalize_answer(a: str) -> str:
    """Normalize an answer string for comparison."""
    if a is None:
        return "i don't know"
    a = unicodedata.normalize("NFKD", str(a)).lower()
    a = re.sub(r"\([^)]*\)", "", a)
    a = a.replace("\u2019", "'")
    a = re.sub(r"[^a-z0-9\s]", "", a)
    a = re.sub(r"\s+", " ", a).strip()
    if a in {"i dont know", "unknown", "not sure", "cannot determine", "cant tell"}:
        return "i don't know"
    return a


def _last_token(s: str) -> str:
    toks = normalize_answer(s).split()
    return toks[-1] if toks else ""


def alias_equivalent(a: str, b: str) -> bool:
    """Check if two answers are equivalent (exact or last-token match)."""
    a_n, b_n = normalize_answer(a), normalize_answer(b)
    if a_n == b_n:
        return True
    if a_n == "i don't know" or b_n == "i don't know":
        return False
    la, lb = _last_token(a), _last_token(b)
    return len(la) >= 4 and la == lb


def get_gold(q: dict):
    """Extract gold answer from a QA pair dict."""
    return q.get("answer") or q.get("answers")


# ── graph-walk context compression ────────────────────────────────
# Pure parsing + BFS — zero LLM calls, zero embedding calls.
# Parses the structured KET-RAG context (entities/relationships/chunks)
# and keeps only the subgraph reachable from question entities.

from collections import deque

_GW_SECTION_HEADERS = [
    "-----Entities-----",
    "-----Relationships-----",
    "-----Sources-----",
    "-----Text source that may be relevant-----",
]


def _parse_ket_context(context: str) -> dict:
    """Parse a KET-RAG context string into structured sections."""
    c = context or ""
    result = {
        "entities": {},       # {UPPER_NAME: description}
        "relationships": [],  # [(src, tgt, desc, weight)]
        "sources": [],        # [(id, text)]
        "chunks": [],         # [(id, text)]
        "adj": {},            # {entity: set(neighbors)}  bidirectional
    }

    # Find section boundaries
    ent_start = c.find("-----Entities-----")
    rel_start = c.find("-----Relationships-----")
    src_start = c.find("-----Sources-----")
    txt_start = c.find("-----Text source that may be relevant-----")

    if ent_start < 0:
        return result

    # --- Entities ---
    ent_end = rel_start if rel_start > ent_start else len(c)
    for line in c[ent_start:ent_end].split("\n")[2:]:  # skip header + column row
        parts = line.split("|")
        if len(parts) >= 3:
            name = parts[1].strip()
            desc = parts[2].strip()
            if name:
                result["entities"][name] = desc

    # --- Relationships ---
    if rel_start >= 0:
        rel_end = src_start if src_start > rel_start else len(c)
        for line in c[rel_start:rel_end].split("\n")[2:]:
            parts = line.split("|")
            if len(parts) >= 4:
                src = parts[1].strip()
                tgt = parts[2].strip()
                desc = parts[3].strip()
                try:
                    weight = float(parts[4]) if len(parts) > 4 and parts[4].strip() else 1.0
                except ValueError:
                    weight = 1.0
                if src and tgt:
                    result["relationships"].append((src, tgt, desc, weight))
                    result["adj"].setdefault(src, set()).add(tgt)
                    result["adj"].setdefault(tgt, set()).add(src)

    # --- Sources (community reports) ---
    if src_start >= 0:
        src_end = txt_start if txt_start > src_start else len(c)
        current_id, current_lines = None, []
        for line in c[src_start:src_end].split("\n")[2:]:
            m = re.match(r"^(\d+)\|(.*)$", line)
            if m:
                if current_id is not None:
                    result["sources"].append((current_id, "\n".join(current_lines)))
                current_id = m.group(1)
                current_lines = [m.group(2)]
            elif current_id is not None:
                current_lines.append(line)
        if current_id is not None:
            result["sources"].append((current_id, "\n".join(current_lines)))

    # --- Text chunks ---
    if txt_start >= 0:
        current_id, current_lines = None, []
        for line in c[txt_start:].split("\n")[2:]:
            m = re.match(r"^(chunk_\d+)\|(.*)$", line)
            if m:
                if current_id is not None:
                    result["chunks"].append((current_id, "\n".join(current_lines)))
                current_id = m.group(1)
                current_lines = [m.group(2)]
            elif current_id is not None:
                current_lines.append(line)
        if current_id is not None:
            result["chunks"].append((current_id, "\n".join(current_lines)))

    return result


_GW_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "for", "to", "of", "in",
    "on", "at", "by", "with", "from", "as", "is", "are", "was", "were",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "will", "that",
    "this", "these", "those", "it", "its", "be", "been", "being", "have",
    "has", "had", "not", "no", "same", "part", "also", "called", "known",
}


def _match_question_entities(question: str, entity_names: list) -> list:
    """Find which graph entities are mentioned in the question."""
    q_low = (question or "").lower()
    q_words = set(re.findall(r"[a-z0-9]+", q_low))
    # Content words from question (for partial matching)
    q_content = {w for w in q_words if len(w) > 2 and w not in _GW_STOPWORDS}
    matched = []
    for name in entity_names:
        n_low = name.lower()
        # Exact substring match
        if n_low in q_low:
            matched.append(name)
            continue
        # Multi-word: all content words of entity appear in question
        words = [w for w in re.findall(r"[a-z0-9]+", n_low) if len(w) > 2]
        if len(words) >= 2 and all(w in q_words for w in words):
            matched.append(name)
            continue
        # Partial: a significant question word appears as part of entity name
        # (catches "navigator" matching "PRINCE HENRY THE NAVIGATOR")
        ent_words = set(re.findall(r"[a-z0-9]+", n_low))
        overlap = q_content & ent_words
        if overlap and any(len(w) >= 5 for w in overlap):
            matched.append(name)
    return matched


def _bfs_entities(adj: dict, seeds: list, max_hops: int = 3) -> dict:
    """BFS from seed entities. Returns {entity: hop_distance}."""
    visited = {}
    queue = deque()
    for s in seeds:
        if s not in visited:
            visited[s] = 0
            queue.append((s, 0))
    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = depth + 1
                queue.append((neighbor, depth + 1))
    return visited


def _expand_via_text(chain_entities: dict, all_entity_names: set,
                     chunks: list) -> dict:
    """Expand chain via text co-occurrence: if a chunk mentions a chain entity
    AND a non-chain entity, add the new entity at max_hop + 1."""
    max_hop = max(chain_entities.values()) if chain_entities else 0
    new_entities = {}
    for _, text in chunks:
        text_low = text.lower()
        has_chain = any(e.lower() in text_low for e in chain_entities)
        if not has_chain:
            continue
        for ent in all_entity_names:
            if ent not in chain_entities and ent not in new_entities:
                if ent.lower() in text_low:
                    new_entities[ent] = max_hop + 1
    return new_entities


def _estimate_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def compress_context_graph_walk(question: str, context: str,
                                max_hops: int = 3,
                                budget_tokens: int = 4000) -> tuple:
    """Compress KET-RAG context by keeping only the graph-reachable subgraph.

    Returns (compressed_text, metadata_dict).
    """
    parsed = _parse_ket_context(context)
    orig_words = _estimate_words(context)

    # Fallback: if no entities parsed, return original
    if not parsed["entities"]:
        return context, {"mode": "no_entities", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    # Step 1: Match question to seed entities
    seeds = _match_question_entities(question, list(parsed["entities"].keys()))

    if not seeds:
        return context, {"mode": "no_seeds", "orig_words": orig_words,
                         "new_words": orig_words, "seeds": 0, "chain": 0}

    # Step 2: BFS
    chain = _bfs_entities(parsed["adj"], seeds, max_hops)

    # Step 3: Expand via text co-occurrence
    text_expanded = _expand_via_text(chain, set(parsed["entities"].keys()),
                                     parsed["chunks"])
    chain.update(text_expanded)

    chain_set = set(chain.keys())

    # Step 4: Assemble compressed context — chain-ordered by hop distance
    # Instead of flat sections, group entities+relationships+chunks per hop
    # so the model reads the reasoning chain sequentially.
    budget_words = int(budget_tokens / 1.3)  # rough tokens → words

    # Pre-compute: which entities live at each hop?
    max_hop_seen = max(chain.values()) if chain else 0
    hop_entities = {}  # hop -> sorted list of entity names
    for ent, hop in chain.items():
        hop_entities.setdefault(hop, []).append(ent)
    for h in hop_entities:
        hop_entities[h].sort()

    # Pre-compute: relationships indexed by source entity
    rel_by_src = {}
    for src, tgt, desc, weight in parsed["relationships"]:
        if src in chain_set and tgt in chain_set:
            rel_by_src.setdefault(src, []).append((tgt, desc))

    # Pre-compute: chunks scored by chain entity mentions (for later assignment)
    q_content = {w for w in re.findall(r"[a-z0-9]+", (question or "").lower())
                 if len(w) > 2 and w not in _GW_STOPWORDS}

    chunk_hop_assignment = {}  # cid -> (best_hop, chain_score)
    tier2 = []
    for cid, text in parsed["chunks"]:
        text_low = text.lower()
        # Find best (lowest) hop of any chain entity mentioned in this chunk
        best_hop = None
        chain_score = 0
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

    # Build chain-ordered output
    parts = []
    used_words = 0
    used_cids = set()
    chunk_parts_count = 0
    rel_lines_count = 0

    # Build chunks lookup for quick access
    chunk_map = {cid: text for cid, text in parsed["chunks"]}

    for hop in range(max_hop_seen + 1):
        ents_at_hop = hop_entities.get(hop, [])
        if not ents_at_hop:
            continue

        hop_lines = []
        hop_label = "from question" if hop == 0 else f"hop {hop}"
        hop_lines.append(f"=== Step {hop} ({hop_label}) ===")

        for ent in ents_at_hop:
            desc = parsed["entities"].get(ent, "")
            if desc:
                hop_lines.append(f"  {ent}: {desc}")
            else:
                hop_lines.append(f"  {ent}")
            # Relationships FROM this entity
            for tgt, rdesc in rel_by_src.get(ent, []):
                hop_lines.append(f"    -> {tgt}: {rdesc}")
                rel_lines_count += 1

        # Add tier-1 chunks assigned to this hop
        hop_chunks = [(cid, chunk_hop_assignment[cid][1])
                      for cid in chunk_hop_assignment
                      if chunk_hop_assignment[cid][0] == hop and cid not in used_cids]
        hop_chunks.sort(key=lambda x: -x[1])  # highest chain_score first

        hop_block = "\n".join(hop_lines)
        block_words = _estimate_words(hop_block)
        if used_words + block_words > budget_words:
            # Still add the entity/rel lines (they're small), skip chunks
            parts.append(hop_block)
            used_words += block_words
            continue

        parts.append(hop_block)
        used_words += block_words

        for cid, score in hop_chunks:
            text = chunk_map.get(cid, "")
            w = _estimate_words(text)
            if used_words + w > budget_words:
                continue
            parts.append(f"  [{cid}] {text}")
            used_words += w
            used_cids.add(cid)
            chunk_parts_count += 1

    # Fill remaining budget with tier-2 keyword-relevant chunks
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

    # Source reports if budget allows
    source_parts = []
    for sid, text in parsed["sources"]:
        text_low = text.lower()
        score = sum(1 for e in chain_set if e.lower() in text_low)
        if score > 0:
            w = _estimate_words(text)
            if used_words + w > budget_words:
                continue
            source_parts.append(text)
            used_words += w

    if source_parts:
        parts.append("=== Sources ===\n" + "\n".join(source_parts))

    current = "\n\n".join(parts)

    new_words = _estimate_words(current)
    return current, {
        "mode": "compressed",
        "orig_words": orig_words,
        "new_words": new_words,
        "seeds": len(seeds),
        "seed_names": seeds,
        "chain": len(chain),
        "chain_entities": sorted(chain.keys()),
        "text_expanded": len(text_expanded),
        "n_chunks_kept": chunk_parts_count,
        "n_sources_kept": len(source_parts),
        "n_chunks_total": len(parsed["chunks"]),
        "n_rels_kept": rel_lines_count,
    }


def build_graph_compressed_context_lookup(qa_list: list, context_lookup: dict,
                                          max_hops: int = 3,
                                          budget_tokens: int = 4000) -> tuple:
    """Build per-question compressed contexts via graph-walk.
    Returns (new_context_lookup, stats_dict).
    """
    out = {}
    t0 = time.time()
    total = len(qa_list)
    orig_sum, new_sum = 0, 0
    seed_sum, chain_sum = 0, 0
    n_compressed, n_fallback = 0, 0

    for i, q in enumerate(qa_list):
        qid = str(q["id"])
        question = str(q["question"])
        context = context_lookup.get(qid, "")

        compressed, meta = compress_context_graph_walk(
            question, context, max_hops=max_hops, budget_tokens=budget_tokens,
        )
        out[qid] = compressed

        orig_sum += meta.get("orig_words", 0)
        new_sum += meta.get("new_words", 0)
        seed_sum += meta.get("seeds", 0)
        chain_sum += meta.get("chain", 0)
        if meta.get("mode") == "compressed":
            n_compressed += 1
        else:
            n_fallback += 1

        elapsed = time.time() - t0
        line = (
            f"  [graph-walk] {i+1}/{total} elapsed={elapsed:.1f}s "
            f"mode={meta['mode']} words={meta.get('orig_words',0)}->{meta.get('new_words',0)} "
            f"seeds={meta.get('seeds',0)} chain={meta.get('chain',0)} "
            f"chunks={meta.get('n_chunks_kept',0)}/{meta.get('n_chunks_total',0)}"
        )
        print(line.encode("ascii", "replace").decode())

    stats = {
        "n_total": total,
        "n_compressed": n_compressed,
        "n_fallback": n_fallback,
        "avg_orig": orig_sum / total if total else 0,
        "avg_new": new_sum / total if total else 0,
        "avg_seeds": seed_sum / total if total else 0,
        "avg_chain": chain_sum / total if total else 0,
    }
    return out, stats


# ── voting ─────────────────────────────────────────────────────────

def _normalize_for_semantic(s: str) -> str:
    if s is None:
        return "i don't know"
    s = unicodedata.normalize("NFKD", str(s))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def semantic_vote_winner(voter_answers, threshold: float, embedder):
    """
    Cluster voter answers by cosine similarity and pick the majority winner.
    Returns (winner_index | None, debug_dict).
    """
    norm = [_normalize_for_semantic(a) for a in voter_answers]
    vecs = embedder.encode(norm, normalize_embeddings=True)
    sim = cosine_similarity(vecs)

    used = set()
    clusters = []
    for i in range(len(norm)):
        if i in used:
            continue
        grp = [i]
        used.add(i)
        for j in range(i + 1, len(norm)):
            if sim[i, j] >= threshold:
                grp.append(j)
                used.add(j)
        clusters.append(grp)

    clusters.sort(key=len, reverse=True)
    top = clusters[0]

    debug = {
        "threshold": threshold,
        "norm": norm,
        "clusters": clusters,
        "sim_matrix": [[float(f"{x:.3f}") for x in row] for row in sim],
    }

    if len(top) >= 2:
        winner = top[0]
        if is_abstain(voter_answers[winner]):
            return None, debug
        return winner, debug

    return None, debug


def alias_vote_winner(voter_answers):
    """
    Group voter answers by alias equivalence and pick the majority winner.
    Returns (winner_index | None, debug_dict).
    """
    groups = []
    reps = []

    for i, ans in enumerate(voter_answers):
        placed = False
        for g_idx, rep_i in enumerate(reps):
            if alias_equivalent(ans, voter_answers[rep_i]):
                groups[g_idx].append(i)
                placed = True
                break
        if not placed:
            reps.append(i)
            groups.append([i])

    debug = {voter_answers[rep_i]: len(members) for rep_i, members in zip(reps, groups)}

    best = None
    best_size = 0
    for rep_i, members in zip(reps, groups):
        if is_abstain(voter_answers[rep_i]):
            continue
        if len(members) > best_size:
            best_size = len(members)
            best = rep_i

    if best is not None and best_size >= 2:
        return best, debug

    return None, debug


# ── JSON extraction ────────────────────────────────────────────────

def extract_first_json_object(txt: str):
    """Robustly extract the first JSON object from a string."""
    if txt is None:
        return None
    txt = str(txt).strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(txt):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(txt[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


# ── Level-2 arbiter ────────────────────────────────────────────────

def arbiter_choose(
    client, model: str, question: str, context: str,
    candidates: list, max_tokens=220,
):
    """Ask the arbiter to pick among candidate answers using the context."""
    prompt = f"""
You are a strict arbiter for retrieval-augmented QA.

Rules:
- Use ONLY the CONTEXT.
- Choose ONE of the candidate answers EXACTLY as written if it directly answers the question.
- If none clearly answer the question, output exactly: I don't know
- Provide an evidence_quote copied verbatim from the context.
- If final is I don't know, evidence_quote must be empty.

Return ONLY JSON:
{{"final":"<exact candidate OR I don't know>","evidence_quote":"<verbatim quote or empty>","reason":"short"}}

QUESTION:
{question}

CONTEXT:
{context}

CANDIDATES:
1) {candidates[0]}
2) {candidates[1]}
3) {candidates[2]}
""".strip()

    txt = call_groq_chat(
        client, model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    obj = extract_first_json_object(txt)
    if not isinstance(obj, dict):
        return {"final": "I don't know", "evidence_quote": "", "reason": (txt or "")[:160]}

    final = str(obj.get("final", "I don't know")).strip()
    quote = str(obj.get("evidence_quote", "")).strip()
    reason = str(obj.get("reason", ""))[:220]

    if final != "I don't know":
        if final not in candidates:
            final, quote = "I don't know", ""
        elif (not quote) or (quote not in context):
            final, quote = "I don't know", ""

    if final == "I don't know":
        quote = ""

    return {"final": final, "evidence_quote": quote, "reason": reason}


# ── context loading ────────────────────────────────────────────────

def load_ket_records(project_root: Path, split: str, strategy: str, theta: float):
    """Load context records from a KET-RAG output JSON."""
    p = project_root / "output" / f"{split}-{strategy}-{theta}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["data", "records", "results", "items", "contexts"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(f"Unexpected format in {p}: {type(obj)}")


def record_to_qid_ctx(rec: dict):
    """Extract (question_id, context_text) from a single KET record."""
    qid = None
    for k in ["id", "query_id", "qid", "question_id"]:
        if k in rec:
            qid = str(rec[k])
            break
    if qid is None:
        raise ValueError(f"No qid key in record keys={list(rec.keys())[:30]}")

    ctx = ""
    for k in ["context", "context_text", "ctx", "retrieved_context", "final_context"]:
        if k in rec and rec[k] is not None:
            ctx = rec[k]
            break

    if isinstance(ctx, list):
        ctx = "\n\n".join(str(x) for x in ctx)
    if ctx is None:
        ctx = ""

    return qid, str(ctx)


# ── hybrid runner ──────────────────────────────────────────────────

def run_hybrid(
    client,
    qa_list: list,
    context_lookup: dict,
    voter_models: list,
    arbiter_model: str,
    embedder=None,
    use_semantic_vote: bool = True,
    voter_temp: float = 0.3,
    semantic_threshold: float = 0.82,
    limit: int = None,
    checkpoint_path=None,
):
    """
    Run the hybrid majority-voting pipeline on a list of QA pairs.
    Returns a DataFrame with predictions and vote debug info.
    """
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()

    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break

        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")

        # fire voter calls sequentially
        voter_answers = [
            answer_with_context(client, m, question, context, 160, voter_temp)
            for m in voter_models
        ]

        if use_semantic_vote and embedder is not None:
            winner_i, vote_debug = semantic_vote_winner(
                voter_answers, semantic_threshold, embedder,
            )
        else:
            winner_i, vote_debug = alias_vote_winner(voter_answers)

        escalated = False
        arb = None
        if winner_i is not None:
            final = voter_answers[winner_i]
        else:
            escalated = True
            arb = arbiter_choose(client, arbiter_model, question, context, voter_answers)
            final = arb.get("final", "I don't know")

        row = {
            "id": qid, "question": question, "gold": gold,
            "context_chars": len(context),
            "voter_models": json.dumps(voter_models),
            "voter_answers": json.dumps(voter_answers),
            "vote_debug": json.dumps(vote_debug),
            "escalated_to_level2": escalated,
            "arbiter_json": json.dumps(arb) if arb else "",
            "evidence_quote": (arb.get("evidence_quote", "") if arb else ""),
            "final_pred": final, "final_abstain": is_abstain(final),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        esc_tag = " [ESC]" if escalated else ""
        pred_safe = final[:40].encode("ascii", "replace").decode()
        q_safe = question[:50].encode("ascii", "replace").decode()
        print(f"  [hybrid] {i+1}/{total}  elapsed={elapsed:.1f}s{esc_tag}  pred={pred_safe}  q={q_safe}")

    return pd.DataFrame(rows)


# ── baseline runner ────────────────────────────────────────────────

def run_baseline_single_model(
    client,
    model: str,
    qa_list: list,
    context_lookup: dict,
    temperature: float = 0.3,
    limit: int = None,
    checkpoint_path=None,
):
    """Run a single-model baseline on a list of QA pairs. Returns a DataFrame."""
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()
    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break
        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")
        pred = answer_with_context(client, model, question, context, temperature=temperature)
        row = {
            "qa_model": model, "id": qid, "question": question, "gold": gold,
            "context_chars": len(context), "final_pred": pred, "final_abstain": is_abstain(pred),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        q_safe = question[:50].encode("ascii", "replace").decode()
        print(f"  [baseline] {i+1}/{total}  elapsed={elapsed:.1f}s  q={q_safe}")
    return pd.DataFrame(rows)


# ── SPARQL reformulation runner ───────────────────────────────────

def run_sparql(
    client,
    model: str,
    qa_list: list,
    context_lookup: dict,
    temperature: float = 0.0,
    limit: int = None,
    checkpoint_path=None,
):
    """Answer via SPARQL chain-of-thought reformulation. Returns a DataFrame."""
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()
    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break
        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")
        pred, raw_cot = answer_with_sparql_cot(client, model, question, context,
                                               temperature=temperature)
        row = {
            "qa_model": model, "id": qid, "question": question, "gold": gold,
            "context_chars": len(context), "sparql_cot": raw_cot,
            "final_pred": pred, "final_abstain": is_abstain(pred),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        q_safe = question[:50].encode("ascii", "replace").decode()
        print(f"  [sparql] {i+1}/{total}  elapsed={elapsed:.1f}s  q={q_safe}")
    return pd.DataFrame(rows)



def run_generic_cot(
    client,
    model: str,
    qa_list: list,
    context_lookup: dict,
    temperature: float = 0.0,
    limit: int = None,
    checkpoint_path=None,
):
    """Answer via generic chain-of-thought decomposition. Returns a DataFrame."""
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()
    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break
        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")
        pred, raw_cot = answer_with_generic_cot(client, model, question, context,
                                                temperature=temperature)
        row = {
            "qa_model": model, "id": qid, "question": question, "gold": gold,
            "context_chars": len(context), "generic_cot": raw_cot,
            "final_pred": pred, "final_abstain": is_abstain(pred),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        q_safe = question[:50].encode("ascii", "replace").decode()
        print(f"  [generic_cot] {i+1}/{total}  elapsed={elapsed:.1f}s  q={q_safe}")
    return pd.DataFrame(rows)


def llm_classify_question_type(client, model, question):
    """Use an LLM to classify a question as 'comparison' or 'bridge'."""
    prompt = (
        'Classify this question as either "comparison" or "bridge".\n'
        'A comparison question compares two entities (their properties, dates, sizes, etc.).\n'
        'A bridge question follows a chain of entities to find an answer.\n'
        'Reply with exactly one word: comparison or bridge\n\n'
        f'Question: {question}'
    )
    resp = call_groq_chat(client, model,
                          [{"role": "user", "content": prompt}],
                          max_tokens=5, temperature=0.0)
    label = resp.strip().lower().rstrip(".")
    if label not in ("comparison", "bridge"):
        label = "bridge"
    return label


def run_gated(
    client,
    model: str,
    qa_list: list,
    context_lookup: dict,
    temperature: float = 0.0,
    limit: int = None,
    checkpoint_path=None,
):
    """Gated runner: comparison questions use SPARQL CoT, bridge uses baseline."""
    rows, done_ids = _load_checkpoint(checkpoint_path)
    total = min(len(qa_list), limit) if limit else len(qa_list)
    t0 = time.time()
    n_sparql = sum(1 for r in rows if r.get("qtype") == "comparison")
    for i, q in enumerate(qa_list):
        if limit is not None and i >= limit:
            break
        qid = str(q["id"])
        if qid in done_ids:
            continue
        question = q["question"]
        gold = get_gold(q)
        context = context_lookup.get(qid, "")
        qtype = llm_classify_question_type(client, model, question)
        sparql_cot = ""
        if qtype == "comparison":
            pred, sparql_cot = answer_with_sparql_cot(
                client, model, question, context, temperature=temperature)
            n_sparql += 1
            route_tag = "[SPARQL]"
        else:
            pred = answer_with_context(
                client, model, question, context, temperature=temperature)
            route_tag = "[base]"
        row = {
            "qa_model": model, "id": qid, "question": question, "gold": gold,
            "context_chars": len(context), "qtype": qtype, "sparql_cot": sparql_cot,
            "final_pred": pred, "final_abstain": is_abstain(pred),
        }
        rows.append(row)
        _append_checkpoint(checkpoint_path, row)
        elapsed = time.time() - t0
        q_safe = question[:50].encode("ascii", "replace").decode()
        print(f"  [gated] {i+1}/{total}  {route_tag}  elapsed={elapsed:.1f}s  q={q_safe}")
    print(f"  Routed: {n_sparql} SPARQL, {total - n_sparql} baseline")
    return pd.DataFrame(rows)


# ── evaluation (LLM judge) ────────────────────────────────────────

def _norm(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r\f\v.?!:;\"'`~()[]{}")
    return s


def _gold_candidates(gold):
    if gold is None:
        return []
    if isinstance(gold, list):
        return [g for g in gold if g is not None and str(g).strip()]
    return [gold]


def _gold_to_text(gold):
    if gold is None:
        return ""
    if isinstance(gold, str):
        return gold
    if isinstance(gold, list):
        return " | ".join(str(x) for x in gold)
    return str(gold)


def eval_once(
    client, eval_model: str, question: str, gold, pred: str, max_tokens=160,
):
    """Evaluate a single prediction against gold using heuristics + LLM fallback."""
    pred_n = _norm(pred)

    if pred_n in {
        "i don't know", "i do not know", "unknown",
        "not sure", "cannot determine", "can't tell",
    }:
        gold_text = _gold_to_text(gold).strip()
        if gold_text:
            return {"verdict": "incorrect", "reason": "predicted abstain but gold exists"}
        return {"verdict": "correct", "reason": "both abstain/empty gold"}

    for g in _gold_candidates(gold):
        g_n = _norm(g)
        if g_n and (g_n in pred_n or pred_n in g_n):
            return {"verdict": "correct", "reason": "normalized match"}

    for g in _gold_candidates(gold):
        if alias_equivalent(pred, g):
            return {"verdict": "correct", "reason": "alias-equivalent match"}

    # LLM judge fallback
    prompt = f"""
You are an evaluator.

Decide if the PREDICTED ANSWER matches the GOLD ANSWER.
Return ONLY valid JSON:
{{"verdict":"correct|incorrect|unknown","reason":"short"}}

QUESTION:
{question}

GOLD ANSWER:
{_gold_to_text(gold)}

PREDICTED ANSWER:
{pred}
""".strip()

    txt = call_groq_chat(
        client, eval_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )

    obj = extract_first_json_object(txt)
    if not isinstance(obj, dict):
        return {"verdict": "unknown", "reason": (txt or "")[:160]}

    v = obj.get("verdict", "unknown")
    if v not in {"correct", "incorrect", "unknown"}:
        v = "unknown"

    return {"verdict": v, "reason": str(obj.get("reason", ""))[:200]}


def add_eval_column(client, eval_model: str, df: pd.DataFrame) -> pd.DataFrame:
    """Add eval_verdict and eval_reason columns to a results DataFrame.
    Runs sequentially — most evals are heuristic matches (no API call needed)."""
    rows = []
    total = len(df)
    t0 = time.time()
    for idx, (_, r) in enumerate(df.iterrows()):
        out = eval_once(client, eval_model, r["question"], r["gold"], r["final_pred"])
        row = r.to_dict()
        row["eval_verdict"] = out["verdict"]
        row["eval_reason"] = out.get("reason", "")
        rows.append(row)
        elapsed = time.time() - t0
        print(f"  [eval] {idx+1}/{total}  elapsed={elapsed:.1f}s  verdict={out['verdict']}")
    return pd.DataFrame(rows)


# ── error analysis ─────────────────────────────────────────────────

def _extract_year(s):
    if s is None:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", str(s))
    return int(m.group(0)) if m else None


def categorize_error(row) -> str:
    """Categorize a wrong prediction into an error group."""
    q = str(row["question"]).lower()
    gold = str(row["gold"]).strip()
    pred = str(row["final_pred"]).strip()

    if is_abstain(pred):
        return "Final=IDK (arbiter/escalation)"

    gy, py = _extract_year(gold), _extract_year(pred)
    if (gy is not None) and (py is not None) and (py != gy):
        return "Wrong year"

    if q.startswith(("is ", "are ", "do ", "did ", "was ", "were ", "can ")):
        return "Boolean (yes/no) wrong"

    if "continent" in q:
        return "Attribute mismatch (continent vs realm/region)"

    if "in relation to" in q or re.search(r"\b(mi|miles|km)\b", gold.lower()):
        return "Relational descriptor vs entity mismatch"

    if any(k in q for k in ["genre", "style"]):
        return "Category/genre mismatch"

    if any(k in q for k in ["village", "district"]):
        return "Location granularity mismatch"

    if ("nickname" in q) or ("called" in q):
        return "Nickname/descriptor mismatch"

    if any(k in q for k in ["song", "film", "movie", "manga", "star of", "album"]):
        return "Wrong title (media mix-up)"

    if q.startswith("between ") or any(
        k in q for k in ["lived longer", "came out ahead", "older"]
    ):
        return "Comparative / choose-one wrong"

    if re.search(r"\b(us\$|\$|usd|price)\b", gold, re.I):
        return "Answer-type mismatch (date/price vs entity)"

    return "Other (manual)"


def error_analysis(df: pd.DataFrame, output_dir: Path):
    """
    Group wrong predictions by error category and save per-group CSVs.
    Returns the summary DataFrame.
    """
    wrong = df[df["eval_verdict"].astype(str) != "correct"].copy()
    print(f"Total rows: {len(df)}")
    print(f"Wrong rows: {len(wrong)}")

    wrong["error_group"] = wrong.apply(categorize_error, axis=1)

    summary_wrong = (
        wrong.groupby("error_group")
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    print("\n--- Wrong rows grouped ---")
    print(summary_wrong.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)

    wrong.to_csv(output_dir / "wrong_rows_with_groups.csv", index=False)

    for grp, sub in wrong.groupby("error_group"):
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", grp)[:80]
        sub.to_csv(output_dir / f"{safe}.csv", index=False)

    print(f"Saved per-group CSVs in: {output_dir}")
    return summary_wrong
