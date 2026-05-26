"""Extend Table 5 (cot_ablation, no-GW) to HotpotQA and MuSiQue.

Computes the Routing row via post-processing using:
  - existing classifier predictions (revision/classifier_predictions_*.json)
  - existing no-GW SPARQL CoT and Generic CoT CSVs
  - abstain-fallback rule: if classifier-routed method abstains, use the
    other method's answer.
"""
import csv
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Paper-era no-GW CSVs
SPARQL_NO_GW = {
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

GENERIC_NO_GW = {
    ('hotpotqa', '8B'):
        'experiments/hotpotqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260501_170833.csv',
    ('hotpotqa', '70B'):
        'experiments/hotpotqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260501_173051.csv',
    ('2wikimultihopqa', '8B'):
        'experiments/2wikimultihopqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260221_140052.csv',
    ('2wikimultihopqa', '70B'):
        'experiments/2wikimultihopqa/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260221_142306.csv',
    ('musique', '8B'):
        'experiments/musique/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_musique_large_scale_20260501_174920.csv',
    ('musique', '70B'):
        'experiments/musique/large_scale/results_generic_cot_groq/'
        'generic_cot_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_musique_large_scale_20260501_181201.csv',
}

# Paper-era no-GW BASELINE CSVs (for the Baseline row)
BASELINE_NO_GW = {
    ('hotpotqa', '8B'):
        'experiments/hotpotqa/large_scale/results_baseline_groq/'
        'baseline_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260218_101626.csv',
    ('hotpotqa', '70B'):
        'experiments/hotpotqa/large_scale/results_baseline_groq/'
        'baseline_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_hotpotqa_large_scale_20260218_110034.csv',
    ('2wikimultihopqa', '8B'):
        'experiments/2wikimultihopqa/large_scale/results_baseline_groq/'
        'baseline_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_193514.csv',
    ('2wikimultihopqa', '70B'):
        'experiments/2wikimultihopqa/large_scale/results_baseline_groq/'
        'baseline_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_2wikimultihopqa_large_scale_20260220_204350.csv',
    ('musique', '8B'):
        'experiments/musique/large_scale/results_baseline_groq/'
        'baseline_llama-3.1-8b-instant_temp0.3_keyword_theta0.5_musique_large_scale_20260220_085621.csv',
    ('musique', '70B'):
        'experiments/musique/large_scale/results_baseline_groq/'
        'baseline_llama-3.3-70b-versatile_temp0.3_keyword_theta0.5_musique_large_scale_20260220_073938.csv',
}


def load_csv(path):
    """Returns dict: id -> {'verdict': 'CORRECT'/'INCORRECT', 'abstain': bool}."""
    out = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            out[r['id']] = {
                'verdict': r['eval_verdict'].strip().upper(),
                'abstain': str(r.get('final_abstain','')).lower() == 'true',
            }
    return out


def load_classifier(ds, three_way=True):
    """Load classifier predictions. 3-way (paper's appendix prompt) by default;
    binary as legacy fallback."""
    fname = f'classifier_predictions_3way_{ds}.json' if three_way else f'classifier_predictions_{ds}.json'
    with open(os.path.join(BASE, 'revision', fname), 'r', encoding='utf-8') as f:
        return json.load(f)


def acc_abs(d):
    n = len(d)
    n_correct = sum(1 for r in d.values() if r['verdict'] == 'CORRECT')
    n_abstain = sum(1 for r in d.values() if r['abstain'])
    return 100*n_correct/n, 100*n_abstain/n


def routing_with_fallback(sparql, generic, classifier_pred):
    """Paper's routing rule: bridge -> SPARQL, comparison or inference -> Generic,
    fallback on abstain to the other method. 3-way classifier from paper appendix.
    Returns (acc%, abs%)."""
    n_correct = n_abstain = n_total = 0
    for qid, pred in classifier_pred.items():
        if qid not in sparql or qid not in generic:
            continue
        n_total += 1
        # Route per paper rule (bridge -> SPARQL; comparison/inference -> Generic)
        if pred == 'bridge':
            primary, secondary = sparql[qid], generic[qid]
        else:  # 'comparison' or 'inference'
            primary, secondary = generic[qid], sparql[qid]
        # Fallback on abstain
        chosen = secondary if primary['abstain'] else primary
        if chosen['verdict'] == 'CORRECT':
            n_correct += 1
        if chosen['abstain']:
            n_abstain += 1
    return 100*n_correct/n_total, 100*n_abstain/n_total


print('Extended Table 5: SPARQL CoT vs Generic CoT vs Routing (without GW)')
print('=' * 70)
for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
    print(f'\n  --- {ds} ---')
    classifier = load_classifier(ds)
    print(f'  {"Method":<14} {"8B Acc":>8} {"8B Abs":>8} {"70B Acc":>9} {"70B Abs":>9}')
    print(f'  {"-"*54}')
    for method, csv_map in [('Baseline', BASELINE_NO_GW),
                             ('Generic CoT', GENERIC_NO_GW),
                             ('SPARQL CoT', SPARQL_NO_GW)]:
        row = [method]
        for model in ['8B', '70B']:
            d = load_csv(os.path.join(BASE, csv_map[(ds, model)]))
            a, ab = acc_abs(d)
            row += [f'{a:.1f}', f'{ab:.1f}']
        print(f'  {row[0]:<14} {row[1]:>8} {row[2]:>8} {row[3]:>9} {row[4]:>9}')
    # Routing row
    row = ['Routing']
    for model in ['8B', '70B']:
        sparql = load_csv(os.path.join(BASE, SPARQL_NO_GW[(ds, model)]))
        generic = load_csv(os.path.join(BASE, GENERIC_NO_GW[(ds, model)]))
        a, ab = routing_with_fallback(sparql, generic, classifier)
        row += [f'{a:.1f}', f'{ab:.1f}']
    print(f'  {row[0]:<14} {row[1]:>8} {row[2]:>8} {row[3]:>9} {row[4]:>9}')
