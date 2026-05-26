"""3-way classifier matching the paper's routing classifier (Appendix A.routing_prompt).

Classifies each question as bridge / comparison / inference using the paper's
exact prompt. Saves predictions to revision/classifier_predictions_3way_*.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_pipeline import call_groq_chat
from groq import Groq
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE, '.env'))
client = Groq(api_key=os.environ['GROQ_API_KEY'])

CLASSIFIER_MODEL = 'llama-3.1-8b-instant'

# Verbatim from paper appendix A.routing_prompt
PROMPT_TEMPLATE = (
    'Classify this question as exactly one of: "bridge", "comparison", "inference".\n'
    'Definitions:\n'
    '- bridge: answer requires following an entity chain across facts\n'
    '- comparison: answer compares two entities/values\n'
    '- inference: answer requires implicit reasoning not a clean entity chain\n'
    'Reply with exactly one word: bridge or comparison or inference\n\n'
    'Question: {question}'
)


def classify_3way(question):
    prompt = PROMPT_TEMPLATE.format(question=question)
    resp = call_groq_chat(client, CLASSIFIER_MODEL,
                          [{"role": "user", "content": prompt}],
                          max_tokens=5, temperature=0.0)
    label = resp.strip().lower().rstrip('.').strip('"')
    if label not in ('bridge', 'comparison', 'inference'):
        label = 'bridge'  # default per paper
    return label


for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
    out_path = os.path.join(BASE, 'revision', f'classifier_predictions_3way_{ds}.json')
    qa_path = os.path.join(BASE, 'experiments', ds, 'large_scale',
                           'qa-pairs', 'qa-pairs.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa = json.load(f)
    print(f'\n=== {ds}: {len(qa)} questions ===')

    preds = {}
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            preds = json.load(f)
        print(f'  resuming from {len(preds)} existing predictions')

    t0 = time.time()
    for i, q in enumerate(qa):
        qid = str(q['id'])
        if qid in preds:
            continue
        try:
            preds[qid] = classify_3way(q['question'])
        except Exception as e:
            preds[qid] = f'ERROR: {e}'
        if (i + 1) % 100 == 0 or i + 1 == len(qa):
            elapsed = time.time() - t0
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(preds, f, indent=2)
            print(f'  [{i+1}/{len(qa)}] elapsed={elapsed:.1f}s')
    print(f'  saved -> {out_path}')
