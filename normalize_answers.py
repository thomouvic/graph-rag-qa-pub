"""Post-process 8B predictions: LLM-extract short answers for cleaner F1/EM.

Reads each 8B CSV, calls Groq to extract a 1-5 word answer from each prediction,
writes a new column 'normalized_pred' alongside the original, and saves to
a parallel *_normalized.csv file.

Usage:
    python -u normalize_answers.py
"""

import os, csv, re, time, random, concurrent.futures, threading

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Groq client ──────────────────────────────────────────────────────

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
RPM_LIMIT = 900  # stay under 1K

def call_groq(question, pred_text, retries=8, base_sleep=1.0):
    """Call Groq with rate limiting and retry."""
    # Truncate very long predictions
    pred_text = pred_text[:500]

    messages = [{"role": "user", "content": NORM_PROMPT.format(question=question, pred=pred_text)}]

    for attempt in range(retries):
        # Simple RPM throttle
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
                return pred_text  # fallback to original
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)
    return pred_text

# ── Main ─────────────────────────────────────────────────────────────

datasets_info = {
    'hotpotqa': 'experiments/hotpotqa/large_scale',
    'musique': 'experiments/musique/large_scale',
    '2wikimultihopqa': 'experiments/2wikimultihopqa/large_scale',
}

total_calls = 0
total_changed = 0

for ds_name, ds_rel in datasets_info.items():
    ds_path = os.path.join(BASE, ds_rel)
    for prompt_dir in ['results_baseline_groq', 'results_sparql_groq']:
        full_path = os.path.join(ds_path, prompt_dir)
        if not os.path.isdir(full_path):
            continue
        csvs = sorted([f for f in os.listdir(full_path) if f.endswith('.csv') and '_norm' not in f])
        for csv_file in csvs:
            if '8b-instant' not in csv_file:
                continue
            fpath = os.path.join(full_path, csv_file)
            out_path = os.path.join(full_path, csv_file.replace('.csv', '_norm2.csv'))

            with open(fpath, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != 500:
                continue

            avg_ctx = sum(int(r.get('context_chars', 0)) for r in rows) / 500
            gw = 'GW' if avg_ctx < 30000 else 'Full'
            prompt_type = 'baseline' if 'baseline' in prompt_dir else 'sparql'
            label = f"{ds_name} 8B {prompt_type} {gw}"

            # Check if already done
            if os.path.exists(out_path):
                with open(out_path, 'r', encoding='utf-8') as f:
                    existing = list(csv.DictReader(f))
                if len(existing) == 500 and 'normalized_pred' in existing[0]:
                    print(f"SKIP {label} (already normalized)")
                    continue

            print(f"Processing {label} ...")
            t0 = time.time()

            # Extract answer first (for SPARQL, strip before FINAL ANSWER:)
            is_sparql = prompt_type == 'sparql'
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
                    fut.result()  # raise if error
                    done += 1
                    if done % 100 == 0:
                        elapsed = time.time() - t0
                        print(f"  {done}/500 ({elapsed:.0f}s)")

            # Write output
            fieldnames = list(rows[0].keys()) + ['extracted_pred', 'normalized_pred']
            with open(out_path, 'w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for i, r in enumerate(rows):
                    r['extracted_pred'] = preds[i]
                    r['normalized_pred'] = normalized[i]
                    w.writerow(r)

            changed = sum(1 for i in range(500) if normalized[i] != preds[i])
            elapsed = time.time() - t0
            total_calls += 500
            total_changed += changed
            print(f"  Done {label}: {changed}/500 changed, {elapsed:.0f}s")

print(f"\nTotal: {total_calls} calls, {total_changed} changed")
