"""Post-process LightRAG 8B predictions: LLM-extract short answers for cleaner F1/EM.

Reads each 8B JSONL, calls Groq to extract a 1-5 word answer from each prediction,
adds 'normalized_pred' field, and overwrites the JSONL.

Usage:
    python -u normalize_lightrag_answers.py
"""

import os, json, re, time, random, concurrent.futures, threading

BASE = os.path.dirname(os.path.abspath(__file__))

from groq import Groq
from dotenv import load_dotenv
load_dotenv(os.path.join(BASE, ".env"))

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"

NORM_PROMPT = (
    "Given the question and a model's response, extract only the short factual answer "
    "in 1-5 words (an entity, name, date, number, or yes/no). "
    "Do not explain, just output the answer.\n\n"
    "Question: {question}\n"
    "Response: {pred}\n\n"
    "Short answer:"
)

# ── Rate-limited API call ────────────────────────────────────────────

_lock = threading.Lock()
_call_times = []
RPM_LIMIT = 900

def call_groq(question, pred_text, retries=8, base_sleep=1.0):
    pred_text = pred_text[:500]
    messages = [{"role": "user", "content": NORM_PROMPT.format(question=question, pred=pred_text)}]

    for attempt in range(retries):
        with _lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60]
            if len(_call_times) >= RPM_LIMIT:
                wait = 60 - (now - _call_times[0]) + 0.1
                if wait > 0:
                    time.sleep(wait)
            _call_times.append(time.time())

        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=32,
                temperature=0,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            msg = repr(e).lower()
            transient = any(k in msg for k in [
                "429", "rate", "timeout", "timed out",
                "502", "503", "504", "gateway", "overloaded", "capacity",
            ])
            if (not transient) or attempt == retries - 1:
                print(f"  FAIL after {attempt+1} attempts: {e}")
                return pred_text
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    return pred_text

# ── Main ─────────────────────────────────────────────────────────────

datasets = ['hotpotqa', 'musique', '2wikimultihopqa']
configs_8b = ['Baseline_8B', 'Baseline_GW_8B', 'CoT_8B', 'CoT_GW_8B']

total_calls = 0
total_changed = 0

for ds_name in datasets:
    results_dir = os.path.join(BASE, 'experiments', ds_name, 'large_scale_lightrag', 'results')
    for cfg in configs_8b:
        fpath = os.path.join(results_dir, f'{cfg}.jsonl')
        if not os.path.exists(fpath):
            print(f"SKIP {ds_name}/{cfg}: file not found")
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            rows = [json.loads(line) for line in f if line.strip()]

        if len(rows) != 500:
            print(f"SKIP {ds_name}/{cfg}: {len(rows)} rows (expected 500)")
            continue

        # Check if already normalized
        if all('normalized_pred' in r for r in rows):
            print(f"SKIP {ds_name}/{cfg} (already normalized)")
            continue

        is_sparql = 'CoT' in cfg
        label = f"{ds_name}/{cfg}"
        print(f"Processing {label} ...")
        t0 = time.time()

        # Pre-extract FINAL ANSWER for SPARQL configs
        preds = []
        for r in rows:
            raw = r.get('final_pred', '')
            if is_sparql:
                parts = re.split(r'FINAL ANSWER:\s*', raw, flags=re.IGNORECASE)
                extracted = parts[-1].strip() if len(parts) > 1 else raw.strip()
            else:
                extracted = raw.strip()
            preds.append(extracted)

        # Normalize via LLM (concurrent)
        normalized = [None] * 500
        questions = [r.get('question', '') for r in rows]

        def process(idx):
            normalized[idx] = call_groq(questions[idx], preds[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            futures = {ex.submit(process, i): i for i in range(500)}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
                done += 1
                if done % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  {done}/500 ({elapsed:.0f}s)")

        # Update rows with normalized_pred
        for i, r in enumerate(rows):
            r['normalized_pred'] = normalized[i]

        # Write back
        with open(fpath, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        changed = sum(1 for i in range(500) if normalized[i] != preds[i])
        elapsed = time.time() - t0
        total_calls += 500
        total_changed += changed
        print(f"  Done {label}: {changed}/500 changed, {elapsed:.0f}s")

print(f"\nTotal: {total_calls} calls, {total_changed} changed")
