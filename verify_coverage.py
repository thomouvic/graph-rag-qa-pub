"""Verify coverage method against paper Table 8 values."""
import json, re, csv, string, os

def squad_norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def is_covered(gold, aliases, ctx):
    ctx_n = squad_norm(ctx)
    if squad_norm(gold) in ctx_n:
        return True
    for a in aliases:
        if squad_norm(a) in ctx_n:
            return True
    return False

BASE = os.path.dirname(os.path.abspath(__file__))

for ds in ['hotpotqa', 'musique', '2wikimultihopqa']:
    ds_path = os.path.join(BASE, 'experiments', ds, 'large_scale')
    with open(os.path.join(ds_path, 'qa-pairs', 'qa-pairs.json'), 'r', encoding='utf-8') as f:
        qa = json.load(f)
    with open(os.path.join(ds_path, 'output', 'large_scale-keyword-0.5.json'), 'r', encoding='utf-8') as f:
        ctxs = json.load(f)
    ctx_by_id = {c['id']: c['context'] for c in ctxs}

    cov_set = set()
    for q in qa:
        ctx = ctx_by_id.get(q['id'], '')
        if is_covered(q['answer'], q.get('answers', []), ctx):
            cov_set.add(q['id'])

    n_cov = len(cov_set)
    n_ncov = 500 - n_cov
    print(f"\n{ds}: {n_cov}/{500} covered")

    # Verify with Baseline 8B Full
    bdir = os.path.join(ds_path, 'results_baseline_groq')
    if not os.path.isdir(bdir):
        continue
    for cf in sorted(os.listdir(bdir)):
        if '8b-instant' not in cf or not cf.endswith('.csv'):
            continue
        with open(os.path.join(bdir, cf), 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 500:
            continue
        avg_ctx = sum(int(r['context_chars']) for r in rows) / 500
        if avg_ctx > 30000:
            cov_correct = sum(1 for r in rows if r['id'] in cov_set and r['eval_verdict'].strip().upper() == 'CORRECT')
            ncov_correct = sum(1 for r in rows if r['id'] not in cov_set and r['eval_verdict'].strip().upper() == 'CORRECT')
            cov_n = sum(1 for r in rows if r['id'] in cov_set)
            ncov_n = sum(1 for r in rows if r['id'] not in cov_set)
            print(f"  Baseline 8B Full: Cov={100*cov_correct/cov_n:.1f}% ({cov_correct}/{cov_n}), NCov={100*ncov_correct/ncov_n:.1f}% ({ncov_correct}/{ncov_n})")
            break
