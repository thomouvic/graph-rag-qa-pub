"""+GW routing analysis (paper's headline configuration).

Mirror of routing_analysis.py but using +GW data for both SPARQL and Generic.
Compares simulated 8B+routing+GW against the paper's claim that this matches
the unaugmented 70B baseline.
"""
import csv
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# +GW SPARQL CSVs (from data_inventory.md, Feb 20 +GW timestamps)
SPARQL_GW_CSV = {
    ('hotpotqa', '8B'):
        'experiments/hotpotqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260220_171837.csv',
    ('hotpotqa', '70B'):
        'experiments/hotpotqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260220_172852.csv',
    ('2wikimultihopqa', '8B'):
        'experiments/2wikimultihopqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_203018.csv',
    ('2wikimultihopqa', '70B'):
        'experiments/2wikimultihopqa/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_212846.csv',
    ('musique', '8B'):
        'experiments/musique/large_scale/results_sparql_groq/'
        'sparql_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_musique_large_scale_20260220_095057.csv',
    ('musique', '70B'):
        'experiments/musique/large_scale/results_sparql_groq/'
        'sparql_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_musique_large_scale_20260220_082837.csv',
}

# +GW Generic CoT CSVs (the runs we just completed; ctx_avg < 25000)
GENERIC_GW_CSV = {
    ('hotpotqa', '8B'):
        'experiments/hotpotqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260501_184943.csv',
    ('hotpotqa', '70B'):
        'experiments/hotpotqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260501_185206.csv',
    ('2wikimultihopqa', '8B'):
        'experiments/2wikimultihopqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260501_190039.csv',
    ('2wikimultihopqa', '70B'):
        'experiments/2wikimultihopqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260501_191138.csv',
    ('musique', '8B'):
        'experiments/musique/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_musique_large_scale_20260501_192032.csv',
    ('musique', '70B'):
        'experiments/musique/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_musique_large_scale_20260501_193157.csv',
}

# 70B no-GW baseline reference (paper's Table 2)
BASELINE_70B = {'hotpotqa': 78.0, '2wikimultihopqa': 48.8, 'musique': 35.2}


def load_csv_eval(path):
    with open(path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['id']: r['eval_verdict'].strip().upper() for r in rows}


def load_gold_types(ds):
    path = os.path.join(BASE, 'HippoRAG', 'reproduce', 'dataset', f'{ds}.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for item in data:
        qid = item.get('_id', item.get('id'))
        if ds == 'hotpotqa':
            out[qid] = item['type']
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


datasets = ['hotpotqa', '2wikimultihopqa', 'musique']
models = ['8B', '70B']

print('=' * 70)
print('  +GW PER-GOLD-TYPE RULE VALIDITY (SPARQL+GW vs Generic+GW)')
print('=' * 70)
for ds in datasets:
    gold = load_gold_types(ds)
    print(f'\n  --- {ds} ---')
    for model in models:
        sparql = load_csv_eval(os.path.join(BASE, SPARQL_GW_CSV[(ds, model)]))
        generic = load_csv_eval(os.path.join(BASE, GENERIC_GW_CSV[(ds, model)]))
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
            sparql_acc = 100 * d['sparql_c'] / d['n']
            generic_acc = 100 * d['generic_c'] / d['n']
            winner = 'SPARQL' if sparql_acc > generic_acc else 'Generic' if generic_acc > sparql_acc else 'tie'
            print(f'      {gt:12s} n={d["n"]:3d}   SPARQL+GW={sparql_acc:5.1f}%   Generic+GW={generic_acc:5.1f}%   '
                  f'gap={sparql_acc-generic_acc:+5.1f} pp ({winner})')

print()
print('=' * 70)
print('  +GW DEPLOYED ROUTING SIMULATION (paper headline configuration)')
print('=' * 70)
print('  Routing rule: bridge -> SPARQL+GW, comparison -> Generic+GW')
print('  Compared against: SPARQL+GW alone, Generic+GW alone, Oracle, 70B baseline (no-GW)')
print()
deploy = {}
for ds in datasets:
    gold = load_gold_types(ds)
    pred = load_pred_types(ds)
    print(f'\n  --- {ds} (70B no-GW baseline = {BASELINE_70B[ds]:.1f}%) ---')
    for model in models:
        sparql = load_csv_eval(os.path.join(BASE, SPARQL_GW_CSV[(ds, model)]))
        generic = load_csv_eval(os.path.join(BASE, GENERIC_GW_CSV[(ds, model)]))
        n = sparql_only = generic_only = routed = oracle = 0
        for qid in pred:
            if qid not in sparql or qid not in generic or qid not in gold:
                continue
            n += 1
            sc = sparql[qid] == 'CORRECT'
            gc = generic[qid] == 'CORRECT'
            if sc: sparql_only += 1
            if gc: generic_only += 1
            routed_ans = sc if pred[qid] == 'bridge' else gc
            if routed_ans: routed += 1
            oracle_ans = sc if gold[qid] == 'bridge' else gc
            if oracle_ans: oracle += 1
        sa, ga, ra, oa = (100*x/n for x in [sparql_only, generic_only, routed, oracle])
        delta_70b = ra - BASELINE_70B[ds]
        print(f'    {model} (n={n}):')
        print(f'      SPARQL+GW alone:    {sa:5.1f}%')
        print(f'      Generic+GW alone:   {ga:5.1f}%')
        print(f'      Routed (classifier): {ra:5.1f}%   [vs 70B no-GW baseline: {delta_70b:+5.1f} pp]')
        print(f'      Oracle (gold types): {oa:5.1f}%')
        deploy[(ds, model)] = (sa, ga, ra, oa, delta_70b)

print()
print('=' * 70)
print('  HEADLINE: 8B+routing+GW vs 70B no-GW baseline (paper claim)')
print('=' * 70)
print(f'{"Dataset":<18} {"Model":<5} {"SPARQL+GW":>10} {"Generic+GW":>11} {"Routed+GW":>10} {"70B base":>9} {"vs 70B":>7}')
for ds in datasets:
    for model in models:
        sa, ga, ra, oa, d70 = deploy[(ds, model)]
        print(f'{ds:<18} {model:<5} {sa:>9.1f}% {ga:>10.1f}% {ra:>9.1f}% {BASELINE_70B[ds]:>8.1f}% {d70:>+6.1f}')
