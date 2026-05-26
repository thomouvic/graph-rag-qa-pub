"""
Bridging-entity coverage analysis.

Addresses reviewer eQxk's concern that substring matching on the gold answer
overcounts coverage (final answer can appear for unrelated reasons in long
retrieved contexts) and undercounts (paraphrased answers). Uses dataset-
provided annotations to check whether the *bridging entities required for
multi-hop reasoning* are present in the retrieved context.

This is the principled fix to the substring metric. eQxk's example: question
"Who directed the 1994 Best Picture winner?" gold "Robert Zemeckis". The model
needs both (Forrest Gump won 1994 Best Picture) and (Forrest Gump was directed
by Robert Zemeckis) in context. Substring matching on "Robert Zemeckis" can
hit unrelated text; bridging-entity coverage requires the actual bridge entity
("Forrest Gump") to be retrieved.

Required-entity definitions per dataset:
  HotpotQA      Supporting-fact article titles (with parenthetical
                disambiguators stripped) + final answer.
  2WikiMHQA     Subjects + objects of evidence triples (covers both
                intermediate entities and the final answer).
  MuSiQue       Decomposition-step answers (intermediate + final).

Two metrics reported side by side:
  M1 substring  Original paper metric: gold answer (with aliases) appears as
                substring of the lowercased context. Documented limitations.
  M3 entities   All required entities (above) appear as substring of the
                context, normalized for whitespace/punctuation. Principled.

Outputs:
  revision/coverage/{dataset}_coverage.csv   per-question rows
  revision/coverage/summary.txt              aggregate + error decomposition
"""
import json
import os
import re
import string
import csv

BASE = os.path.dirname(os.path.abspath(__file__))
HIPPORAG_DATASET = os.path.join(BASE, 'HippoRAG', 'reproduce', 'dataset')
EXPERIMENTS = os.path.join(BASE, 'experiments')
OUT_DIR = os.path.join(BASE, 'revision', 'coverage')

os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Text normalization ----------

def squad_norm(s):
    """SQuAD-style: lowercase, drop articles + punctuation, collapse ws."""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s


# ---------- Required-entity extraction per dataset ----------

def hotpotqa_required_entities(item):
    """Supporting-fact article titles + final answer. Strips parenthetical
    disambiguators like '(film)', '(actor)' from titles."""
    titles = set()
    for title, _ in item['supporting_facts']:
        clean = re.sub(r'\s*\([^)]*\)\s*$', '', title).strip()
        if clean:
            titles.add(clean)
    return list(titles) + [item['answer']]


def twiki_required_entities(item):
    """Subjects + objects of evidence triples (intermediate + final)."""
    out = set()
    for s, _, o in item.get('evidences', []):
        if s:
            out.add(s)
        if o:
            out.add(o)
    return list(out)


def musique_required_entities(item):
    """All decomposition-step answers (intermediate + final)."""
    return [step['answer'] for step in item['question_decomposition']
            if step.get('answer')]


# ---------- Coverage checks ----------

def entity_present(entity, lower_ctx, norm_ctx):
    """Is the entity name in context? Tries lowercase substring, then
    squad_norm substring. Entities are typically proper nouns; substring
    matching is reliable."""
    if not entity:
        return False
    if entity.lower().strip() in lower_ctx:
        return True
    e_norm = squad_norm(entity)
    if e_norm and e_norm in norm_ctx:
        return True
    return False


def gold_answer_substring(answer, lower_ctx, aliases=None):
    """Original paper metric: gold answer (with aliases) as substring of
    lowercased context."""
    candidates = [answer] + (aliases or [])
    for a in candidates:
        if a and a.lower().strip() in lower_ctx:
            return True
    return False


# ---------- Per-dataset pipeline ----------

def load_dataset(name):
    path = os.path.join(HIPPORAG_DATASET, f'{name}.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_retrieved_contexts(ds_dir_name):
    """Map id -> retrieved context string. Uses the keyword-0.5 strategy file
    (matches what was used for baseline evaluation in the paper)."""
    path = os.path.join(EXPERIMENTS, ds_dir_name, 'large_scale', 'output',
                        'large_scale-keyword-0.5.json')
    with open(path, 'r', encoding='utf-8') as f:
        ctxs = json.load(f)
    return {c['id']: c['context'] for c in ctxs}


def load_qa_ids(ds_dir_name):
    path = os.path.join(EXPERIMENTS, ds_dir_name, 'large_scale', 'qa-pairs',
                        'qa-pairs.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_dataset(name, ds_dir_name, entity_extractor):
    print(f'\n=== {name} ===')

    full = {item.get('_id', item.get('id')): item
            for item in load_dataset(name)}
    qa_pairs = load_qa_ids(ds_dir_name)
    retrieved = load_retrieved_contexts(ds_dir_name)

    rows = []
    for qa in qa_pairs:
        qid = qa['id']
        item = full.get(qid)
        if item is None:
            continue
        ctx_raw = retrieved.get(qid, '')
        norm_ctx = squad_norm(ctx_raw)
        lower_ctx = ' '.join(ctx_raw.lower().split())

        gold = qa['answer']
        aliases = qa.get('answers', [])
        substring_covered = gold_answer_substring(gold, lower_ctx, aliases)

        entities = entity_extractor(item)
        n_present = sum(1 for e in entities if entity_present(e, lower_ctx, norm_ctx))
        all_entities_present = (len(entities) > 0 and n_present == len(entities))

        rows.append({
            'id': qid,
            'question': qa['question'],
            'gold': gold,
            'n_entities': len(entities),
            'n_entities_present': n_present,
            'all_entities_present': all_entities_present,
            'substring_covered': substring_covered,
            'context_chars': len(ctx_raw),
        })
    return rows


# ---------- Error decomposition ----------

def load_baseline_csv(ds_dir_name, model_substr):
    """Find the baseline CSV with the largest avg context (full keyword-0.5,
    matching the paper Table 8 row)."""
    rdir = os.path.join(EXPERIMENTS, ds_dir_name, 'large_scale',
                        'results_baseline_groq')
    if not os.path.isdir(rdir):
        return None
    best = None
    best_avg = -1
    for fn in sorted(os.listdir(rdir)):
        if model_substr not in fn or not fn.endswith('.csv'):
            continue
        if '_norm' in fn:
            continue
        with open(os.path.join(rdir, fn), 'r', encoding='utf-8', newline='') as f:
            data = list(csv.DictReader(f))
        if len(data) != 500:
            continue
        avg = sum(int(r['context_chars']) for r in data) / len(data)
        if avg > best_avg:
            best_avg = avg
            best = (fn, data)
    return best


def cross_tab(rows, baseline_data, label):
    """Error decomposition under M1 and M3, plus disagreement matrix."""
    eval_by_id = {r['id']: r['eval_verdict'].strip().upper() for r in baseline_data}
    cov_by_id = {r['id']: r for r in rows}

    def pct(k, n):
        return f'{100*k/n:.1f}%' if n else 'n/a'

    def decompose(metric_key, metric_name):
        cov_n = cov_c = ncov_n = ncov_c = 0
        for r in baseline_data:
            cov = cov_by_id.get(r['id'])
            if cov is None:
                continue
            verdict = eval_by_id[r['id']]
            correct = int(verdict == 'CORRECT')
            if cov[metric_key]:
                cov_n += 1; cov_c += correct
            else:
                ncov_n += 1; ncov_c += correct
        rf = cov_n - cov_c
        retf = ncov_n - ncov_c
        total_wrong = rf + retf
        lines = [f'  [{metric_name}]']
        lines.append(f'    coverage rate:    {pct(cov_n, cov_n+ncov_n)} ({cov_n}/{cov_n+ncov_n})')
        lines.append(f'    Cov accuracy:     {pct(cov_c, cov_n)} ({cov_c}/{cov_n})')
        lines.append(f'    NCov accuracy:    {pct(ncov_c, ncov_n)} ({ncov_c}/{ncov_n})')
        lines.append(f'    reasoning errors: {rf}    retrieval errors: {retf}    lucky: {ncov_c}')
        if total_wrong:
            lines.append(f'    reasoning share of errors: {pct(rf, total_wrong)}    '
                         f'retrieval share: {pct(retf, total_wrong)}')
        return '\n'.join(lines)

    out = [f'\n--- {label} ---']
    out.append(decompose('substring_covered',
                         'M1: substring of final answer (original paper)'))
    out.append(decompose('all_entities_present',
                         'M3: all bridging entities present (principled metric)'))

    # Disagreement matrix
    sub_only = sub_only_c = ent_only = ent_only_c = both = both_c = neither = neither_c = 0
    for r in baseline_data:
        cov = cov_by_id.get(r['id'])
        if cov is None:
            continue
        correct = int(eval_by_id[r['id']] == 'CORRECT')
        sub = cov['substring_covered']
        ent = cov['all_entities_present']
        if sub and ent:
            both += 1; both_c += correct
        elif sub and not ent:
            sub_only += 1; sub_only_c += correct
        elif not sub and ent:
            ent_only += 1; ent_only_c += correct
        else:
            neither += 1; neither_c += correct
    out.append(f'  Disagreement matrix (M1 substring x M3 entities):')
    out.append(f'    Both Y:        {both:4d}  acc={pct(both_c, both)}')
    out.append(f"    M1-only:       {sub_only:4d}  acc={pct(sub_only_c, sub_only)}"
               f"   (gold appears, bridging entities missing -- eQxk's false-Y case)")
    out.append(f'    M3-only:       {ent_only:4d}  acc={pct(ent_only_c, ent_only)}'
               f'   (entities present, gold paraphrased -- M1 undercount)')
    out.append(f'    Neither:       {neither:4d}  acc={pct(neither_c, neither)}')
    return '\n'.join(out)


# ---------- Run ----------

def main():
    summary = []
    summary.append('Bridging-entity coverage analysis')
    summary.append('=================================')
    summary.append('')

    configs = [
        ('hotpotqa', 'hotpotqa', hotpotqa_required_entities),
        ('2wikimultihopqa', '2wikimultihopqa', twiki_required_entities),
        ('musique', 'musique', musique_required_entities),
    ]

    for name, ds_dir, ent_extractor in configs:
        rows = analyze_dataset(name, ds_dir, ent_extractor)

        out_csv = os.path.join(OUT_DIR, f'{ds_dir}_coverage.csv')
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f'  wrote {out_csv} ({len(rows)} rows)')

        n = len(rows)
        sub_cov = sum(1 for r in rows if r['substring_covered'])
        ent_cov = sum(1 for r in rows if r['all_entities_present'])
        avg_ents = sum(r['n_entities'] for r in rows) / max(1, n)
        avg_present = sum(r['n_entities_present'] for r in rows) / max(1, n)

        summary.append(f'\n## {name}')
        summary.append(f'  questions: {n}')
        summary.append(f'  avg required entities per question: {avg_ents:.1f}')
        summary.append(f'  avg entities present per question:  {avg_present:.1f}')
        summary.append(f'  Coverage:')
        summary.append(f'    M1 substring of gold:    {sub_cov}/{n} ({100*sub_cov/n:.1f}%)')
        summary.append(f'    M3 bridging entities:    {ent_cov}/{n} ({100*ent_cov/n:.1f}%)')

        for model_substr, label in [('8b-instant', '8B baseline'),
                                    ('70b-versatile', '70B baseline')]:
            res = load_baseline_csv(ds_dir, model_substr)
            if res:
                fn, data = res
                summary.append(cross_tab(rows, data, f'{label} ({fn})'))

    summary_path = os.path.join(OUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    print(f'\nWrote summary to {summary_path}')
    print('\n' + '\n'.join(summary))


if __name__ == '__main__':
    main()
