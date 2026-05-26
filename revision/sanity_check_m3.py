"""Sanity-check Metric 3 (bridging-entity coverage) on a few samples.

We want to verify:
  - Substring-only cells (gold present, entities missing): do these match
    eQxk's predicted false-Y pattern?
  - Entities-only cells (entities present, gold paraphrased): are these
    genuine paraphrase / format cases?
"""
import json, os, sys, csv, random
import io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from supporting_fact_coverage import (
    load_dataset, load_qa_ids, load_retrieved_contexts,
    hotpotqa_required_entities, twiki_required_entities, musique_required_entities,
    entity_present, squad_norm,
)

random.seed(7)

def show_excerpt(ctx, needle, span=200):
    """Show a window of context around the first occurrence of `needle`."""
    pos = ctx.lower().find(needle.lower())
    if pos == -1:
        return f'[{needle!r} not found in context]'
    start = max(0, pos - span // 2)
    end = min(len(ctx), pos + len(needle) + span // 2)
    return ctx[start:end].replace('\n', ' ')

CONFIGS = [
    ('2wikimultihopqa', '2wikimultihopqa', twiki_required_entities),
    ('musique', 'musique', musique_required_entities),
    ('hotpotqa', 'hotpotqa', hotpotqa_required_entities),
]

for name, ds_dir, ent_extractor in CONFIGS:
    print(f'\n{"="*70}')
    print(f'  {name}')
    print('='*70)

    csv_path = os.path.join(BASE, 'revision', 'coverage', f'{ds_dir}_coverage.csv')
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))

    full = {item.get('_id', item.get('id')): item for item in load_dataset(name)}
    retrieved = load_retrieved_contexts(ds_dir)

    # Bucket the 4 cells of substring x entities
    sub_only = [r for r in rows if r['substring_covered']=='True' and r['all_entities_present']=='False']
    ent_only = [r for r in rows if r['substring_covered']=='False' and r['all_entities_present']=='True']

    for label, bucket in [('SUBSTRING-ONLY (gold present, entities missing — eQxk predicted)', sub_only),
                          ('ENTITIES-ONLY (entities present, gold paraphrased)', ent_only)]:
        print(f'\n  --- {label} ---')
        print(f'  total in this cell: {len(bucket)}')
        if not bucket:
            continue
        sample = random.sample(bucket, min(3, len(bucket)))
        for r in sample:
            qid = r['id']
            item = full[qid]
            ctx = retrieved.get(qid, '')
            entities = ent_extractor(item)
            print(f'\n    Q ({qid}): {r["question"][:120]}')
            print(f'    Gold: {r["gold"]!r}')
            print(f'    Required entities ({len(entities)}):')
            for e in entities:
                hit = entity_present(e, ' '.join(ctx.lower().split()), squad_norm(ctx))
                marker = '[Y]' if hit else '[N]'
                print(f'      {marker} {e!r}')
            # Show excerpt around gold answer if present
            gold = r['gold']
            if gold.lower() in ctx.lower():
                print(f'    Context excerpt around gold:')
                excerpt = show_excerpt(ctx, gold, 250)
                excerpt = ' '.join(excerpt.split())
                print(f'      "{excerpt[:300]}"')
