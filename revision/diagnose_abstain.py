"""Re-run the MuSiQue 70B abstain cases with max_tokens=2048 to diagnose
truncation (Hypothesis A) vs genuine abstain (Hypothesis B).

Same 5 MuSiQue questions as the Stage C smoke. Prints the LAST 600 chars of
raw output so we can see how the chain ended.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_pipeline import answer_with_self_ask
from groq import Groq
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(BASE, '.env'))
client = Groq(api_key=os.environ['GROQ_API_KEY'])

# Load first 5 MuSiQue questions (same as the Stage C smoke)
with open(os.path.join(BASE, 'experiments', 'musique', 'large_scale',
                      'qa-pairs', 'qa-pairs.json'), 'r', encoding='utf-8') as f:
    qa = json.load(f)
with open(os.path.join(BASE, 'experiments', 'musique', 'large_scale',
                      'output', 'large_scale-keyword-0.5.json'),
          'r', encoding='utf-8') as f:
    ctxs = json.load(f)
ctx_by_id = {c['id']: c['context'] for c in ctxs}

MODEL = 'llama-3.3-70b-versatile'

for i, q in enumerate(qa[:5], 1):
    print('=' * 70)
    print(f'  [{i}] q: {q["question"][:120]}')
    print(f'      gold: {q["answer"]!r}')
    answer, raw = answer_with_self_ask(
        client, MODEL, q['question'], ctx_by_id.get(q['id'], ''),
        'musique', max_tokens=2048, temperature=0.3,
    )
    print(f'      extracted answer: {answer!r}')
    print(f'      raw output length: {len(raw)} chars, '
          f'finishes with: ...{raw[-300:]!r}')
    has_final = 'so the final answer is' in raw.lower()
    print(f'      "so the final answer is" present in raw: {has_final}')
    print()
