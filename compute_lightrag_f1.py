"""Compute Acc/F1/EM for all 24 LightRAG configs, overall + conditioned on coverage.
Coverage = gold answer substring present in LightRAG cached context.

Outputs table matching Table 3 (tab:main) structure from results.tex.
"""
import os, re, json, string, collections

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

# ── Coverage computation using LightRAG cached contexts ──

def load_lightrag_coverage(ds_name):
    """Check if gold answer is substring of LightRAG context."""
    qa_path = os.path.join(BASE, 'experiments', ds_name, 'large_scale', 'qa-pairs', 'qa-pairs.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)

    # Context cache paths
    ctx_paths = {
        'hotpotqa': os.path.join(BASE, 'experiments', 'hotpotqa', 'large_scale_lightrag', 'lightrag_contexts.json'),
        'musique': os.path.join(BASE, 'experiments', 'musique', 'large_scale_lightrag', 'lightrag_contexts.json'),
        '2wikimultihopqa': os.path.join(BASE, 'experiments', '2wikimultihopqa', 'large_scale_lightrag', 'lightrag_contexts.json'),
    }
    with open(ctx_paths[ds_name], 'r', encoding='utf-8') as f:
        ctxs = json.load(f)

    cov = set()
    for q in qa:
        qid = str(q['id'])
        ctx = ctxs.get(qid, '')
        if ds_name == 'hotpotqa':
            ctx_n = squad_norm(ctx)
            if squad_norm(q['answer']) in ctx_n:
                cov.add(qid)
            elif any(squad_norm(a) in ctx_n for a in q.get('answers', [])):
                cov.add(qid)
        elif ds_name == '2wikimultihopqa':
            if q['answer'].lower() in ctx.lower():
                cov.add(qid)
        elif ds_name == 'musique':
            ctx_l = ctx.lower()
            if q['answer'].lower() in ctx_l:
                cov.add(qid)
            elif any(a.lower() in ctx_l for a in q.get('answers', [])):
                cov.add(qid)
    return cov

# ── Load results ──

datasets = ['hotpotqa', 'musique', '2wikimultihopqa']
configs = [
    ('Baseline',    '70B', False),
    ('Baseline_GW', '70B', False),
    ('CoT',         '70B', True),
    ('CoT_GW',      '70B', True),
    ('Baseline_8B',    '8B', False),
    ('Baseline_GW_8B', '8B', False),
    ('CoT_8B',         '8B', True),
    ('CoT_GW_8B',      '8B', True),
]

# Load coverage
coverage = {}
for ds_name in datasets:
    coverage[ds_name] = load_lightrag_coverage(ds_name)
    print(f"Coverage {ds_name}: {len(coverage[ds_name])}/500")

print()

# Build QA lookup for gold answers
qa_lookup = {}
for ds_name in datasets:
    qa_path = os.path.join(BASE, 'experiments', ds_name, 'large_scale', 'qa-pairs', 'qa-pairs.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    qa_lookup[ds_name] = {str(q['id']): q for q in qa}

# Compute metrics
results = {}  # (ds_name, config_name) -> {acc, f1, em, acc_cov, f1_cov, em_cov}

for ds_name in datasets:
    cov_set = coverage[ds_name]
    for config_name, model, is_sparql in configs:
        results_dir = os.path.join(BASE, 'experiments', ds_name, 'large_scale_lightrag', 'results')
        fpath = os.path.join(results_dir, f'{config_name}.jsonl')
        if not os.path.exists(fpath):
            print(f"  MISSING: {fpath}")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if len(rows) != 500:
            print(f"  WARNING: {ds_name}/{config_name} has {len(rows)} rows (expected 500)")

        f1_all, em_all = [], []
        f1_cov, em_cov, acc_cov_list = [], [], []
        correct_all = 0

        for row in rows:
            qid = str(row['id'])
            # Use normalized_pred for 8B configs (LLM-extracted short answers)
            if 'normalized_pred' in row:
                pred = row['normalized_pred'].strip()
            else:
                pred = extract_answer(row.get('final_pred', ''), is_sparql)
            gold = row.get('gold', '')

            f1_v = compute_f1(pred, gold)
            em_v = compute_em(pred, gold)
            correct = row.get('eval_verdict', '').strip().lower() == 'correct'

            f1_all.append(f1_v)
            em_all.append(em_v)
            if correct:
                correct_all += 1

            if qid in cov_set:
                f1_cov.append(f1_v)
                em_cov.append(em_v)
                acc_cov_list.append(float(correct))

        def avg(lst):
            return 100 * sum(lst) / len(lst) if lst else 0.0

        acc = 100 * correct_all / len(rows)
        n_cov = len(cov_set)

        results[(ds_name, config_name)] = {
            'acc': acc,
            'f1': avg(f1_all),
            'em': avg(em_all),
            'acc_cov': avg(acc_cov_list),
            'f1_cov': avg(f1_cov),
            'em_cov': avg(em_cov),
            'n': len(rows),
            'n_cov': n_cov,
            'model': model,
        }

# ── Print Table 3-style output ──

# Row order: Base 8B, +GW 8B, SPARQL 8B, +GW 8B, Base 70B, +GW 70B, SPARQL 70B, +GW 70B
row_order = [
    ('Baseline_8B',    'Base',   '8B'),
    ('Baseline_GW_8B', '~+GW',  '8B'),
    ('CoT_8B',         'SPARQL', '8B'),
    ('CoT_GW_8B',      '~+GW',  '8B'),
    ('Baseline',       'Base',   '70B'),
    ('Baseline_GW',    '~+GW',  '70B'),
    ('CoT',            'SPARQL', '70B'),
    ('CoT_GW',         '~+GW',  '70B'),
]

ds_order = ['hotpotqa', 'musique', '2wikimultihopqa']
ds_labels = {'hotpotqa': 'Hotpot', 'musique': 'MuSiQue', '2wikimultihopqa': '2Wiki'}

# Print coverage counts
for ds in ds_order:
    n_cov = len(coverage[ds])
    print(f"  {ds_labels[ds]} coverage: {n_cov}/500")
print()

# Print header
print("=" * 140)
print("LIGHTRAG TABLE 3: All Questions + Covered Questions")
print("=" * 140)

header = f"{'Method':<8} {'Model':<5}"
for ds in ds_order:
    header += f" | {'Acc':>5} {'F1':>5} {'EM':>5}"
header += " ||"
for ds in ds_order:
    n_cov = len(coverage[ds])
    header += f" | {'Acc':>5} {'F1':>5} {'EM':>5}"

print(f"{'':13s}", end="")
print(f" | {'All Questions':^53s} || {'Covered Questions':^53s}")

print(f"{'':13s}", end="")
for ds in ds_order:
    print(f" | {ds_labels[ds]:^17s}", end="")
print(" ||", end="")
for ds in ds_order:
    n_cov = len(coverage[ds])
    print(f" | {ds_labels[ds]+f'({n_cov})':^17s}", end="")
print()

print(f"{'Method':<8} {'Model':<4}", end="")
for _ in range(6):
    print(f" | {'Acc':>5} {'F1':>5} {'EM':>5}", end="")
print()
print("-" * 140)

for config_name, method, model in row_order:
    print(f"{method:<8} {model:<4}", end="")
    for ds in ds_order:
        r = results.get((ds, config_name))
        if r:
            print(f" | {r['acc']:5.1f} {r['f1']:5.1f} {r['em']:5.1f}", end="")
        else:
            print(f" | {'---':>5} {'---':>5} {'---':>5}", end="")
    print(" ||", end="")
    for ds in ds_order:
        r = results.get((ds, config_name))
        if r:
            print(f" | {r['acc_cov']:5.1f} {r['f1_cov']:5.1f} {r['em_cov']:5.1f}", end="")
        else:
            print(f" | {'---':>5} {'---':>5} {'---':>5}", end="")
    print()
    if config_name == 'CoT_GW_8B':
        print("-" * 140)

print("=" * 140)

# ── Print LaTeX table ──
print()
print("% LaTeX table for appendix.tex")
print(r"\begin{table*}[t]")
print(r"\centering")
print(r"\small")
print(r"\setlength{\tabcolsep}{2.5pt}")
print(r"\caption{LightRAG generality experiment (\%) on 500 paired questions per dataset. ``Covered'' = gold answer present in LightRAG context (count in parentheses). \textbf{Bold} = best per column. GW = graph-walk compression.}")
print(r"\label{tab:lightrag_full}")
print(r"\begin{tabular}{ll|ccc|ccc|ccc||ccc|ccc|ccc}")
print(r"\toprule")

# Sub-headers
print(r"& & \multicolumn{9}{c||}{\textbf{All Questions}} & \multicolumn{9}{c}{\textbf{Covered Questions}} \\")

cov_counts = {ds: len(coverage[ds]) for ds in ds_order}
print(f"& & \\multicolumn{{3}}{{c|}}{{Hotpot}} & \\multicolumn{{3}}{{c|}}{{MuSiQue}} & \\multicolumn{{3}}{{c||}}{{2Wiki}}"
      f" & \\multicolumn{{3}}{{c|}}{{Hotpot\\,({cov_counts['hotpotqa']})}}"
      f" & \\multicolumn{{3}}{{c|}}{{MuSiQue\\,({cov_counts['musique']})}}"
      f" & \\multicolumn{{3}}{{c}}{{2Wiki\\,({cov_counts['2wikimultihopqa']})}} \\\\")
print(r"\textbf{Method} & \textbf{Model} & Acc & F1 & EM & Acc & F1 & EM & Acc & F1 & EM & Acc & F1 & EM & Acc & F1 & EM & Acc & F1 & EM \\")
print(r"\midrule")

# Find best per column for bolding
# Columns: for each (ds, metric, scope) find the best value
best = {}
for ds in ds_order:
    for metric in ['acc', 'f1', 'em', 'acc_cov', 'f1_cov', 'em_cov']:
        best_val = -1
        for config_name, _, _ in row_order:
            r = results.get((ds, config_name))
            if r and r[metric] > best_val:
                best_val = r[metric]
        best[(ds, metric)] = best_val

def fmt(val, ds, metric):
    """Format value, bold if best in column."""
    if abs(val - best[(ds, metric)]) < 0.05:
        return f"\\textbf{{{val:.1f}}}"
    return f"{val:.1f}"

for config_name, method, model in row_order:
    parts = [f"{method} & {model}"]
    for ds in ds_order:
        r = results[(ds, config_name)]
        parts.append(f"{fmt(r['acc'], ds, 'acc')} & {fmt(r['f1'], ds, 'f1')} & {fmt(r['em'], ds, 'em')}")
    for ds in ds_order:
        r = results[(ds, config_name)]
        parts.append(f"{fmt(r['acc_cov'], ds, 'acc_cov')} & {fmt(r['f1_cov'], ds, 'f1_cov')} & {fmt(r['em_cov'], ds, 'em_cov')}")

    line = " & ".join(parts) + " \\\\"
    if config_name == 'CoT_GW_8B':
        line += "\n\\cmidrule{1-20}"
    print(line)

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")
