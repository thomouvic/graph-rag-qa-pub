"""Per-type SPARQL CoT vs Generic CoT, WITH graph-walk compression on both methods.

Mirror of sparql_vs_generic_per_type.py but uses the +GW CSVs identified by
their later timestamps per data_inventory.md conventions.
"""
import csv
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# +GW CSVs (later timestamp per dataset/model)
SPARQL_GW = {
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

GENERIC_GW = {
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


def load_csv(path):
    out = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            out[r['id']] = {'verdict': r['eval_verdict'].strip().upper()}
    return out


def native_types(ds):
    path = os.path.join(BASE, 'HippoRAG', 'reproduce', 'dataset', f'{ds}.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for item in data:
        qid = item.get('_id', item.get('id'))
        if ds == 'hotpotqa':
            out[qid] = item['type']
        elif ds == '2wikimultihopqa':
            out[qid] = item['type']
        elif ds == 'musique':
            out[qid] = qid.split('_')[0]
    return out


def per_type(ds, model):
    sparql = load_csv(os.path.join(BASE, SPARQL_GW[(ds, model)]))
    generic = load_csv(os.path.join(BASE, GENERIC_GW[(ds, model)]))
    types = native_types(ds)
    buckets = defaultdict(lambda: {'n': 0, 's': 0, 'g': 0})
    for qid, t in types.items():
        if qid not in sparql or qid not in generic:
            continue
        buckets[t]['n'] += 1
        if sparql[qid]['verdict'] == 'CORRECT':
            buckets[t]['s'] += 1
        if generic[qid]['verdict'] == 'CORRECT':
            buckets[t]['g'] += 1
    return buckets


for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
    print(f'\n=== {ds} (WITH GW) ===')
    for model in ['8B', '70B']:
        buckets = per_type(ds, model)
        print(f'\n  {model}')
        print(f'  {"Type":<22} {"n":>5} {"S+GW":>7} {"G+GW":>7} {"Gap":>8}')
        print(f'  {"-"*55}')
        for t in sorted(buckets, key=lambda k: -buckets[k]['n']):
            b = buckets[t]
            sa = 100 * b['s'] / b['n']
            ga = 100 * b['g'] / b['n']
            gap = sa - ga
            print(f'  {t:<22} {b["n"]:>5} {sa:>6.1f}% {ga:>6.1f}% {gap:>+7.1f}')
