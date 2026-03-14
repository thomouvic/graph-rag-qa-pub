"""Compute F1/EM for all 24 configs, overall + conditioned on coverage.
Coverage methods verified against paper Table 8:
  HotpotQA: squad_norm + aliases → 454 (exact)
  2WikiMHQA: lowercase primary → 405 (exact)
  MuSiQue: lowercase + aliases → 385 (paper: 386, off by 1)
"""
import os, re, csv, json, string, collections

BASE = os.path.dirname(os.path.abspath(__file__))

def squad_norm(s):
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = ' '.join(s.split())
    return s

def extract_answer(pred, is_sparql):
    if not is_sparql:
        return pred.strip()
    parts = re.split(r'FINAL ANSWER:\s*', pred, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return pred.strip()

def compute_f1(pred, gold):
    pt = squad_norm(pred).split()
    gt = squad_norm(gold).split()
    if not gt: return float(pt == [])
    if not pt: return 0.0
    common = collections.Counter(pt) & collections.Counter(gt)
    ns = sum(common.values())
    if ns == 0: return 0.0
    p = ns / len(pt); r = ns / len(gt)
    return 2 * p * r / (p + r)

def compute_em(pred, gold):
    return float(squad_norm(pred) == squad_norm(gold))

def load_coverage(ds_name):
    ds_path = os.path.join(BASE, 'experiments', ds_name, 'large_scale')
    with open(os.path.join(ds_path, 'qa-pairs', 'qa-pairs.json'), 'r', encoding='utf-8') as f:
        qa = json.load(f)
    with open(os.path.join(ds_path, 'output', 'large_scale-keyword-0.5.json'), 'r', encoding='utf-8') as f:
        ctxs = json.load(f)
    ctx_by_id = {c['id']: c['context'] for c in ctxs}

    cov = set()
    for q in qa:
        ctx = ctx_by_id.get(q['id'], '')
        if ds_name == 'hotpotqa':
            # squad_norm + aliases
            ctx_n = squad_norm(ctx)
            if squad_norm(q['answer']) in ctx_n:
                cov.add(q['id'])
            elif any(squad_norm(a) in ctx_n for a in q.get('answers', [])):
                cov.add(q['id'])
        elif ds_name == '2wikimultihopqa':
            # lowercase primary only
            if q['answer'].lower() in ctx.lower():
                cov.add(q['id'])
        elif ds_name == 'musique':
            # lowercase + aliases
            ctx_l = ctx.lower()
            if q['answer'].lower() in ctx_l:
                cov.add(q['id'])
            elif any(a.lower() in ctx_l for a in q.get('answers', [])):
                cov.add(q['id'])
    return cov

datasets_info = {
    'hotpotqa': 'experiments/hotpotqa/large_scale',
    'musique': 'experiments/musique/large_scale',
    '2wikimultihopqa': 'experiments/2wikimultihopqa/large_scale',
}

coverage = {}
for ds_name in datasets_info:
    coverage[ds_name] = load_coverage(ds_name)
    n_cov = len(coverage[ds_name])
    print(f"Coverage {ds_name}: {n_cov}/{500}")

print()

results = []

for ds_name, ds_rel in datasets_info.items():
    ds_path = os.path.join(BASE, ds_rel)
    cov_set = coverage[ds_name]
    n_cov = len(cov_set)
    n_ncov = 500 - n_cov

    for prompt_dir in ['results_baseline_groq', 'results_sparql_groq']:
        prompt_type = 'baseline' if 'baseline' in prompt_dir else 'sparql'
        full_path = os.path.join(ds_path, prompt_dir)
        if not os.path.isdir(full_path):
            continue
        csvs = sorted([f for f in os.listdir(full_path) if f.endswith('.csv')])
        for csv_file in csvs:
            fpath = os.path.join(full_path, csv_file)
            if '8b-instant' in csv_file:
                model = '8B'
            elif '70b-versatile' in csv_file:
                model = '70B'
            else:
                continue

            with open(fpath, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 500:
                continue

            avg_ctx = sum(int(r.get('context_chars', 0)) for r in rows) / 500
            gw = 'GW' if avg_ctx < 30000 else 'Full'
            is_sparql = (prompt_type == 'sparql')

            f1_all, em_all = [], []
            f1_cov, em_cov, acc_cov = [], [], []
            f1_ncov, em_ncov = [], []

            for row in rows:
                pred = extract_answer(row.get('final_pred', ''), is_sparql)
                gold = row.get('gold', '')
                f1_v = compute_f1(pred, gold)
                em_v = compute_em(pred, gold)
                correct = row.get('eval_verdict', '').strip().upper() == 'CORRECT'
                f1_all.append(f1_v)
                em_all.append(em_v)

                if row['id'] in cov_set:
                    f1_cov.append(f1_v)
                    em_cov.append(em_v)
                    acc_cov.append(float(correct))
                else:
                    f1_ncov.append(f1_v)
                    em_ncov.append(em_v)

            def avg(lst):
                return 100 * sum(lst) / len(lst) if lst else 0.0

            acc = 100 * sum(1 for r in rows if r['eval_verdict'].strip().upper() == 'CORRECT') / 500
            abs_rate = 100 * sum(1 for r in rows if r.get('final_abstain', '').strip().upper() in ('TRUE', '1')) / 500

            results.append({
                'dataset': ds_name,
                'model': model,
                'prompt': prompt_type,
                'gw': gw,
                'acc': acc,
                'abs': abs_rate,
                'f1': avg(f1_all),
                'em': avg(em_all),
                'acc_cov': avg(acc_cov),
                'f1_cov': avg(f1_cov),
                'em_cov': avg(em_cov),
            })

results.sort(key=lambda r: (r['dataset'], r['model'], r['prompt'], r['gw']))

# Print in Table 3 format (ordered like paper)
print("=" * 120)
print("TABLE 3: Add F1 and EM columns")
print(f"{'Dataset':<18} {'Model':<6} {'Prompt':<10} {'GW':<5} {'Acc':>6} {'Abs':>6} {'F1':>6} {'EM':>6}")
print('-' * 75)
for r in results:
    print(f"{r['dataset']:<18} {r['model']:<6} {r['prompt']:<10} {r['gw']:<5} {r['acc']:6.1f} {r['abs']:6.1f} {r['f1']:6.1f} {r['em']:6.1f}")

# Print in Table 8 format (Acc|Cov, F1|Cov, EM|Cov — drop NCov)
print()
print("=" * 120)
print("TABLE 8: Cov columns only (Acc|Cov, F1|Cov, EM|Cov)")
print(f"{'Dataset':<18} {'Model':<6} {'Prompt':<10} {'GW':<5} {'AccCov':>7} {'F1Cov':>7} {'EMCov':>7}")
print('-' * 75)
for r in results:
    print(f"{r['dataset']:<18} {r['model']:<6} {r['prompt']:<10} {r['gw']:<5} {r['acc_cov']:7.1f} {r['f1_cov']:7.1f} {r['em_cov']:7.1f}")
