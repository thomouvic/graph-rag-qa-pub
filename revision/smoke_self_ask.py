"""Smoke-test Self-Ask prompt assembly and (optionally) actual LLM calls.

Stage A: --print-prompt-only assembles and prints prompts without LLM calls.
Stage B: default behavior runs ONE actual call per dataset on Llama-8B.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_pipeline import (
    answer_with_self_ask, get_self_ask_demos, SELF_ASK_2WIKI_HOTPOT,
    SELF_ASK_MUSIQUE,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_n_questions_per_dataset(n=5):
    out = {}
    for ds in ['hotpotqa', '2wikimultihopqa', 'musique']:
        qa_path = os.path.join(BASE, 'experiments', ds, 'large_scale',
                               'qa-pairs', 'qa-pairs.json')
        ctx_path = os.path.join(BASE, 'experiments', ds, 'large_scale',
                                'output', 'large_scale-keyword-0.5.json')
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa = json.load(f)
        with open(ctx_path, 'r', encoding='utf-8') as f:
            ctxs = json.load(f)
        ctx_by_id = {c['id']: c['context'] for c in ctxs}
        out[ds] = []
        for q in qa[:n]:
            out[ds].append({
                'qid': q['id'],
                'question': q['question'],
                'gold': q['answer'],
                'aliases': q.get('answers', []),
                'context': ctx_by_id.get(q['id'], ''),
            })
    return out


def load_one_question_per_dataset():
    """Stage A helper - one sample per dataset for prompt-shape inspection."""
    samples = load_n_questions_per_dataset(n=1)
    return {ds: items[0] for ds, items in samples.items()}


def assemble_prompt(question, context, dataset):
    """Mirror what answer_with_self_ask builds."""
    demos = get_self_ask_demos(dataset)
    return (
        f"{demos}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Are follow up questions needed here: Yes.\n"
        f"Follow up:"
    )


def stage_a_print_prompts(samples):
    for ds, s in samples.items():
        print('=' * 70)
        print(f'  DATASET: {ds}')
        print(f'  question: {s["question"][:120]}')
        print(f'  gold:     {s["gold"]!r}')
        prompt = assemble_prompt(s['question'], s['context'], ds)
        # We only need to verify the SHAPE — show start (demos) and end
        # (post-context, test question, ending). Skip the long context middle.
        head_lines = 30
        tail_lines = 20
        lines = prompt.split('\n')
        print(f'  total prompt chars: {len(prompt)}')
        print(f'  total prompt lines: {len(lines)}')
        print(f'\n  --- HEAD (first {head_lines} lines, demonstrations) ---')
        for line in lines[:head_lines]:
            print(f'    {line}')
        print(f'\n  --- TAIL (last {tail_lines} lines, context end + question + ending) ---')
        for line in lines[-tail_lines:]:
            print(f'    {line[:200]}')
        print()


def stage_b_run_one(samples, model='llama-3.1-8b-instant'):
    """Run ONE actual Self-Ask call per dataset on the given model.
    `samples` is a flat dict of dataset -> single-question dict."""
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, '.env'))
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    for ds, s in samples.items():
        print('=' * 70)
        print(f'  DATASET: {ds}  MODEL: {model}')
        print(f'  question: {s["question"][:120]}')
        print(f'  gold:     {s["gold"]!r}')
        try:
            answer, raw = answer_with_self_ask(
                client, model, s['question'], s['context'], ds,
                temperature=0.3,
            )
        except Exception as e:
            print(f'  ERROR: {e}')
            continue
        print(f'\n  --- RAW OUTPUT ---')
        for line in raw.split('\n'):
            print(f'    {line[:200]}')
        print(f'\n  --- EXTRACTED ANSWER ---')
        print(f'    {answer!r}')
        print()


def stage_c_smoke(n_per_dataset=5):
    """Run Self-Ask on N questions per dataset on both 8B and 70B.
    Reports accuracy + abstain rate per (dataset, model)."""
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE, '.env'))
    client = Groq(api_key=os.environ['GROQ_API_KEY'])

    samples = load_n_questions_per_dataset(n=n_per_dataset)
    models = ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile']

    summary_rows = []
    for ds, items in samples.items():
        for model in models:
            print('=' * 70)
            print(f'  DATASET: {ds}  MODEL: {model}  N: {len(items)}')
            n_correct_lex = 0
            n_abstain = 0
            for i, s in enumerate(items, 1):
                try:
                    answer, raw = answer_with_self_ask(
                        client, model, s['question'], s['context'], ds,
                        temperature=0.3,
                    )
                except Exception as e:
                    print(f'    [{i}] ERROR: {e}')
                    continue
                # Quick lexical match (final eval will use LLM judge)
                norm_ans = answer.lower().strip()
                gold_candidates = [s['gold']] + s.get('aliases', [])
                is_correct = any(g and g.lower().strip() in norm_ans
                                  for g in gold_candidates) or \
                             any(g and norm_ans in g.lower().strip()
                                  for g in gold_candidates if g)
                is_abstain = norm_ans in ("i don't know", "i do not know",
                                          "unknown")
                if is_correct:
                    n_correct_lex += 1
                if is_abstain:
                    n_abstain += 1
                tag = 'OK' if is_correct else ('ABS' if is_abstain else 'X')
                print(f'    [{i}] {tag:3s} q={s["question"][:60]!r}')
                print(f'         gold={s["gold"]!r}  pred={answer[:80]!r}')
            n = len(items)
            print(f'  -> lexical correct: {n_correct_lex}/{n} '
                  f'({100*n_correct_lex/n:.0f}%)   '
                  f'abstain: {n_abstain}/{n}')
            summary_rows.append({
                'dataset': ds, 'model': model, 'n': n,
                'lex_correct': n_correct_lex, 'abstain': n_abstain,
            })
            print()

    print('\n' + '=' * 70)
    print('  SUMMARY')
    print('=' * 70)
    for r in summary_rows:
        print(f"  {r['dataset']:18s}  {r['model']:28s}  "
              f"lex={r['lex_correct']}/{r['n']}  abs={r['abstain']}/{r['n']}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--print-prompt-only', action='store_true',
                   help='Stage A: assemble and print prompts; no LLM calls.')
    p.add_argument('--stage-c', action='store_true',
                   help='Stage C: full smoke (N questions x 3 datasets x 2 models).')
    p.add_argument('--n', type=int, default=5,
                   help='Questions per dataset for stage C.')
    args = p.parse_args()

    if args.print_prompt_only:
        samples = load_one_question_per_dataset()
        stage_a_print_prompts(samples)
    elif args.stage_c:
        stage_c_smoke(n_per_dataset=args.n)
    else:
        samples = load_one_question_per_dataset()
        stage_b_run_one(samples)
