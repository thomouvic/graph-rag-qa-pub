"""Run the LLM classifier on every question in each dataset's qa-pairs.json.
Saves predictions to revision/classifier_predictions_{dataset}.json so we can
do post-hoc routing simulation and classifier-accuracy analysis without
re-running QA experiments."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_pipeline import llm_classify_question_type
from groq import Groq
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE, '.env'))
client = Groq(api_key=os.environ['GROQ_API_KEY'])

CLASSIFIER_MODEL = 'llama-3.1-8b-instant'

for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
    out_path = os.path.join(BASE, 'revision', f'classifier_predictions_{ds}.json')
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
            pred = llm_classify_question_type(client, CLASSIFIER_MODEL, q['question'])
        except Exception as e:
            pred = f'ERROR: {e}'
        preds[qid] = pred
        if (i + 1) % 50 == 0 or i + 1 == len(qa):
            elapsed = time.time() - t0
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(preds, f, indent=2)
            print(f'  [{i+1}/{len(qa)}] elapsed={elapsed:.1f}s')

    print(f'  saved -> {out_path}')
