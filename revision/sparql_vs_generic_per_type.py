"""Per-type SPARQL CoT vs Generic CoT comparison on all three datasets,
using each dataset's native type structure (no cross-dataset collapsing).

Addresses the second half of eQxk #5: "Tables 9 and 10 decompose results by
question type ... but they compare SPARQL CoT to baseline and GW to non-GW
— not SPARQL CoT to generic CoT." So this is the SPARQL-vs-Generic version
of those per-type tables.
"""
import csv
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reuse paths from extended_table5.py
import sys
sys.path.insert(0, os.path.join(BASE, 'revision'))
from extended_table5 import SPARQL_NO_GW, GENERIC_NO_GW, load_csv


def native_types(ds):
    """Return dict: id -> native type label, using each dataset's own taxonomy."""
    path = os.path.join(BASE, 'HippoRAG', 'reproduce', 'dataset', f'{ds}.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    for item in data:
        qid = item.get('_id', item.get('id'))
        if ds == 'hotpotqa':
            out[qid] = item['type']  # bridge / comparison
        elif ds == '2wikimultihopqa':
            out[qid] = item['type']  # bridge_comparison / compositional / comparison / inference
        elif ds == 'musique':
            out[qid] = qid.split('_')[0]  # 2hop / 3hop1 / 3hop2 / 4hop1 / 4hop2 / 4hop3
    return out


def per_type_table(ds, model):
    sparql = load_csv(os.path.join(BASE, SPARQL_NO_GW[(ds, model)]))
    generic = load_csv(os.path.join(BASE, GENERIC_NO_GW[(ds, model)]))
    types = native_types(ds)
    buckets = defaultdict(lambda: {'n': 0, 'sparql': 0, 'generic': 0})
    for qid, t in types.items():
        if qid not in sparql or qid not in generic:
            continue
        buckets[t]['n'] += 1
        if sparql[qid]['verdict'] == 'CORRECT':
            buckets[t]['sparql'] += 1
        if generic[qid]['verdict'] == 'CORRECT':
            buckets[t]['generic'] += 1
    return buckets


for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
    print(f'\n=== {ds} ===')
    for model in ['8B', '70B']:
        buckets = per_type_table(ds, model)
        print(f'\n  {model}')
        print(f'  {"Type":<22} {"n":>5} {"SPARQL":>8} {"Generic":>9} {"Gap (S-G)":>11} {"Winner":>10}')
        print(f'  {"-"*70}')
        # Sort by n descending so largest types come first
        for t in sorted(buckets, key=lambda k: -buckets[k]['n']):
            b = buckets[t]
            sa = 100 * b['sparql'] / b['n']
            ga = 100 * b['generic'] / b['n']
            gap = sa - ga
            winner = 'SPARQL' if gap > 0.5 else ('Generic' if gap < -0.5 else 'tie')
            print(f'  {t:<22} {b["n"]:>5} {sa:>7.1f}% {ga:>8.1f}% {gap:>+10.1f} {winner:>10}')
