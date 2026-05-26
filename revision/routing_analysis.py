"""Routing-rule validation analysis for eQxk #5.

Three independent analyses:
  1. Classifier accuracy: LLM classifier predictions vs gold types per dataset.
  2. Per-gold-type rule validity: SPARQL no-GW vs Generic CoT broken down by
     gold type, per dataset, per model.
  3. Deployed routing simulation: use classifier predictions to pick method
     per question (bridge -> SPARQL, comparison -> Generic), aggregate.

All comparisons use no-GW SPARQL CoT (paper-era timestamps from
data_inventory.md) and the latest Generic CoT runs (just completed for
HotpotQA + MuSiQue, Feb 21 for 2WikiMHQA).
"""
import csv
import json
import os
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Paper-era no-GW SPARQL CSVs (per data_inventory.md)
SPARQL_CSV = {
    ('hotpotqa', '8B'):
        'experiments/hotpotqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260218_103802.csv',
    ('hotpotqa', '70B'):
        'experiments/hotpotqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260218_111824.csv',
    ('2wikimultihopqa', '8B'):
        'experiments/2wikimultihopqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_200659.csv',
    ('2wikimultihopqa', '70B'):
        'experiments/2wikimultihopqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_211014.csv',
    ('musique', '8B'):
        'experiments/musique/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_musique_large_scale_20260220_092744.csv',
    ('musique', '70B'):
        'experiments/musique/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_musique_large_scale_20260220_080936.csv',
}


def latest_csv(d, model_substr):
    """Pick the latest CSV in dir d matching model_substr (used for generic_cot)."""
    files = sorted([f for f in os.listdir(d)
                    if f.endswith('.csv') and model_substr in f and '_norm' not in f])
    return os.path.join(d, files[-1]) if files else None


GENERIC_CSV = {
    (ds, model): latest_csv(
        os.path.join(BASE, 'experiments', ds, 'large_scale', 'results_generic_cot_groq'),
        '8b-instant' if model == '8B' else '70b-versatile')
    for ds in ['hotpotqa', '2wikimultihopqa', 'musique']
    for model in ['8B', '70B']
}


def load_csv_eval(path):
    """Returns dict: id -> 'CORRECT'/'INCORRECT'."""
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['id']: r['eval_verdict'].strip().upper() for r in rows}


def load_gold_types(ds):
    """Returns dict: id -> binary type 'bridge' or 'comparison'.
    HotpotQA: native type is bridge/comparison.
    2WikiMHQA: collapse {bridge_comparison, compositional} -> bridge,
                       {comparison, inference} -> comparison.
    MuSiQue: 2hop -> bridge, 3hop+/4hop+ -> comparison (compositional)."""
    path = os.path.join(BASE, 'HippoRAG', 'reproduce', 'dataset', f'{ds}.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for item in data:
        qid = item.get('_id', item.get('id'))
        if ds == 'hotpotqa':
            out[qid] = item['type']  # bridge / comparison
        elif ds == '2wikimultihopqa':
            t = item['type']
            if t in ('bridge_comparison', 'compositional'):
                out[qid] = 'bridge'
            elif t in ('comparison', 'inference'):
                out[qid] = 'comparison'
            else:
                out[qid] = t
        elif ds == 'musique':
            prefix = qid.split('_')[0]
            out[qid] = 'bridge' if prefix == '2hop' else 'comparison'
    return out


def load_pred_types(ds):
    path = os.path.join(BASE, 'revision', f'classifier_predictions_{ds}.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pct(k, n):
    return f'{100*k/n:.1f}%' if n else 'n/a'


# ---------- Run all analyses ----------

datasets = ['hotpotqa', '2wikimultihopqa', 'musique']
models = ['8B', '70B']

print('=' * 70)
print('  1. CLASSIFIER ACCURACY (predicted vs gold, binary)')
print('=' * 70)
classifier_summary = {}
for ds in datasets:
    gold = load_gold_types(ds)
    pred = load_pred_types(ds)
    n = sum(1 for qid in pred if qid in gold)
    correct = sum(1 for qid in pred if qid in gold and pred[qid] == gold[qid])
    print(f'  {ds:18s}: {pct(correct, n)} ({correct}/{n})')
    # Confusion matrix
    cm = Counter()
    for qid in pred:
        if qid in gold:
            cm[(gold[qid], pred[qid])] += 1
    print(f'    confusion (gold, pred): {dict(cm)}')
    classifier_summary[ds] = (correct, n, dict(cm))

print()
print('=' * 70)
print('  2. PER-GOLD-TYPE RULE VALIDITY (SPARQL no-GW vs Generic CoT)')
print('=' * 70)
rule_table = {}
for ds in datasets:
    gold = load_gold_types(ds)
    print(f'\n  --- {ds} ---')
    for model in models:
        sparql = load_csv_eval(os.path.join(BASE, SPARQL_CSV[(ds, model)]))
        generic = load_csv_eval(GENERIC_CSV[(ds, model)])
        # Bucket questions by gold type
        per_type = defaultdict(lambda: {'n': 0, 'sparql_c': 0, 'generic_c': 0})
        for qid, gt in gold.items():
            if qid not in sparql or qid not in generic:
                continue
            per_type[gt]['n'] += 1
            if sparql[qid] == 'CORRECT':
                per_type[gt]['sparql_c'] += 1
            if generic[qid] == 'CORRECT':
                per_type[gt]['generic_c'] += 1
        print(f'    {model}:')
        for gt in sorted(per_type.keys()):
            d = per_type[gt]
            sparql_acc = 100 * d['sparql_c'] / d['n'] if d['n'] else 0
            generic_acc = 100 * d['generic_c'] / d['n'] if d['n'] else 0
            winner = 'SPARQL' if sparql_acc > generic_acc else 'Generic' if generic_acc > sparql_acc else 'tie'
            print(f'      {gt:12s} n={d["n"]:3d}   SPARQL={sparql_acc:5.1f}%   Generic={generic_acc:5.1f}%   '
                  f'gap={sparql_acc-generic_acc:+5.1f} pp ({winner})')
            rule_table.setdefault((ds, model), {})[gt] = (sparql_acc, generic_acc, d['n'])

print()
print('=' * 70)
print('  3. DEPLOYED ROUTING SIMULATION (classifier picks method per question)')
print('=' * 70)
print('  Routing rule: bridge -> SPARQL CoT, comparison -> Generic CoT')
print('  Compared against: SPARQL alone, Generic alone, Oracle (gold-type routing)')
print()
deploy_summary = {}
for ds in datasets:
    gold = load_gold_types(ds)
    pred = load_pred_types(ds)
    print(f'\n  --- {ds} ---')
    for model in models:
        sparql = load_csv_eval(os.path.join(BASE, SPARQL_CSV[(ds, model)]))
        generic = load_csv_eval(GENERIC_CSV[(ds, model)])

        n = 0
        sparql_only = generic_only = routed = oracle = 0
        for qid in pred:
            if qid not in sparql or qid not in generic or qid not in gold:
                continue
            n += 1
            sc = sparql[qid] == 'CORRECT'
            gc = generic[qid] == 'CORRECT'
            if sc:
                sparql_only += 1
            if gc:
                generic_only += 1
            # Routed: classifier says bridge -> SPARQL, else Generic
            routed_ans = sc if pred[qid] == 'bridge' else gc
            if routed_ans:
                routed += 1
            # Oracle: gold type tells us to pick SPARQL on bridge, Generic on comparison
            oracle_ans = sc if gold[qid] == 'bridge' else gc
            if oracle_ans:
                oracle += 1

        print(f'    {model} (n={n}):')
        print(f'      SPARQL alone:    {pct(sparql_only, n)} ({sparql_only}/{n})')
        print(f'      Generic alone:   {pct(generic_only, n)} ({generic_only}/{n})')
        print(f'      Routed (classifier): {pct(routed, n)} ({routed}/{n})')
        print(f'      Oracle (gold types): {pct(oracle, n)} ({oracle}/{n})')
        deploy_summary[(ds, model)] = {
            'n': n, 'sparql': sparql_only, 'generic': generic_only,
            'routed': routed, 'oracle': oracle,
        }

print()
print('=' * 70)
print('  HEADLINE: routed accuracy delta vs alternatives')
print('=' * 70)
print(f'{"Dataset":<18} {"Model":<5} {"SPARQL":>7} {"Generic":>8} {"Routed":>7} {"Oracle":>7} {"R-S":>6} {"R-G":>6} {"O-R":>6}')
for ds in datasets:
    for model in models:
        d = deploy_summary[(ds, model)]
        sa = 100 * d['sparql'] / d['n']
        ga = 100 * d['generic'] / d['n']
        ra = 100 * d['routed'] / d['n']
        oa = 100 * d['oracle'] / d['n']
        print(f'{ds:<18} {model:<5} {sa:>6.1f}% {ga:>7.1f}% {ra:>6.1f}% {oa:>6.1f}% '
              f'{ra-sa:>+5.1f} {ra-ga:>+5.1f} {oa-ra:>+5.1f}')
