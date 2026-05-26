"""Find the exact coverage method matching paper Table 8 values."""
import json, re, csv, string, os

BASE = os.path.dirname(os.path.abspath(__file__))

def squad_norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def load_qa_ctx(ds):
    ds_path = os.path.join(BASE, 'experiments', ds, 'large_scale')
    with open(os.path.join(ds_path, 'qa-pairs', 'qa-pairs.json'), 'r', encoding='utf-8') as f:
        qa = json.load(f)
    with open(os.path.join(ds_path, 'output', 'large_scale-keyword-0.5.json'), 'r', encoding='utf-8') as f:
        ctxs = json.load(f)
    ctx_by_id = {c['id']: c['context'] for c in ctxs}
    return qa, ctx_by_id

# For 2wiki: try lowercase primary only (gives 405)
print("=== 2wikimultihopqa ===")
qa, ctx_by_id = load_qa_ctx('2wikimultihopqa')

# Method: lowercase primary
cov_2wiki = set()
for q in qa:
    ctx = ctx_by_id.get(q['id'], '').lower()
    if q['answer'].lower() in ctx:
        cov_2wiki.add(q['id'])
print(f"lowercase primary: {len(cov_2wiki)}")

# Verify with Table 8
bdir = os.path.join(BASE, 'experiments', '2wikimultihopqa', 'large_scale', 'results_baseline_groq')
for cf in sorted(os.listdir(bdir)):
    if '8b-instant' not in cf or not cf.endswith('.csv'):
        continue
    with open(os.path.join(bdir, cf), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 500:
        continue
    avg_ctx = sum(int(r['context_chars']) for r in rows) / 500
    if avg_ctx > 30000:
        cov_c = sum(1 for r in rows if r['id'] in cov_2wiki and r['eval_verdict'].strip().upper() == 'CORRECT')
        cov_n = sum(1 for r in rows if r['id'] in cov_2wiki)
        ncov_c = sum(1 for r in rows if r['id'] not in cov_2wiki and r['eval_verdict'].strip().upper() == 'CORRECT')
        ncov_n = sum(1 for r in rows if r['id'] not in cov_2wiki)
        print(f"  Baseline 8B Full: Cov={100*cov_c/cov_n:.1f}% ({cov_c}/{cov_n}), NCov={100*ncov_c/ncov_n:.1f}% ({ncov_c}/{ncov_n})")
        print(f"  Paper: Cov=32.1% (405), NCov=28.4% (95)")
        break

# For musique: we need 386. squad_norm+aliases gives 385. Let me try various combos.
print("\n=== musique ===")
qa, ctx_by_id = load_qa_ctx('musique')

# Method A: squad_norm + aliases = 385
# Method B: try with 'answers' including primary
cov_mus_sq_alias = set()
for q in qa:
    ctx_n = squad_norm(ctx_by_id.get(q['id'], ''))
    all_answers = [q['answer']] + q.get('answers', [])
    for a in all_answers:
        if squad_norm(a) in ctx_n:
            cov_mus_sq_alias.add(q['id'])
            break
print(f"squad_norm + all answers: {len(cov_mus_sq_alias)}")

# Method C: lowercase + aliases
cov_mus_low_alias = set()
for q in qa:
    ctx = ctx_by_id.get(q['id'], '').lower()
    all_answers = [q['answer']] + q.get('answers', [])
    for a in all_answers:
        if a.lower() in ctx:
            cov_mus_low_alias.add(q['id'])
            break
print(f"lowercase + all answers: {len(cov_mus_low_alias)}")

# Method D: strip extra whitespace in context
cov_mus_d = set()
for q in qa:
    ctx = ' '.join(ctx_by_id.get(q['id'], '').lower().split())
    all_answers = [q['answer']] + q.get('answers', [])
    for a in all_answers:
        if a.lower().strip() in ctx:
            cov_mus_d.add(q['id'])
            break
print(f"lowercase+ws_collapse + all answers: {len(cov_mus_d)}")

# Method E: just lowercase on answer, no ws collapse on context
cov_mus_e = set()
for q in qa:
    ctx = ctx_by_id.get(q['id'], '').lower()
    all_answers = [q['answer']] + q.get('answers', [])
    for a in all_answers:
        if a.lower().strip() in ctx:
            cov_mus_e.add(q['id'])
            break
print(f"lowercase_strip + all answers: {len(cov_mus_e)}")

# Method F: normalize just the answer (not context), lowercase context
cov_mus_f = set()
for q in qa:
    ctx = ctx_by_id.get(q['id'], '').lower()
    all_answers = [q['answer']] + q.get('answers', [])
    for a in all_answers:
        norm_a = squad_norm(a)
        if norm_a in ctx:
            cov_mus_f.add(q['id'])
            break
print(f"squad_norm(answer) in lower(ctx): {len(cov_mus_f)}")

# Verify the 385 method
bdir = os.path.join(BASE, 'experiments', 'musique', 'large_scale', 'results_baseline_groq')
for cf in sorted(os.listdir(bdir)):
    if '8b-instant' not in cf or not cf.endswith('.csv'):
        continue
    with open(os.path.join(bdir, cf), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 500:
        continue
    avg_ctx = sum(int(r['context_chars']) for r in rows) / 500
    if avg_ctx > 30000:
        for name, cov_set in [('sq+alias', cov_mus_sq_alias), ('low+alias', cov_mus_low_alias)]:
            cov_c = sum(1 for r in rows if r['id'] in cov_set and r['eval_verdict'].strip().upper() == 'CORRECT')
            cov_n = sum(1 for r in rows if r['id'] in cov_set)
            ncov_c = sum(1 for r in rows if r['id'] not in cov_set and r['eval_verdict'].strip().upper() == 'CORRECT')
            ncov_n = sum(1 for r in rows if r['id'] not in cov_set)
            print(f"  {name}: Cov={100*cov_c/cov_n:.1f}% ({cov_c}/{cov_n}), NCov={100*ncov_c/ncov_n:.1f}% ({ncov_c}/{ncov_n})")
        print(f"  Paper: Cov=27.5% (386), NCov=10.5% (114)")
        break
