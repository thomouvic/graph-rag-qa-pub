"""Compare N=200 vs N=500 results with bridge/comparison breakdown."""
import pandas as pd, json, glob

dfs = {}
for split, n in [('medium_200', 200), ('large_scale', 500)]:
    pfx = 'experiments/hotpotqa/' + split
    for mt, mf in [('8b','llama-3.1-8b-instant'),('70b','llama-3.3-70b-versatile')]:
        for method in ['baseline','sparql']:
            files = glob.glob(pfx+'/results_'+method+'_groq/'+method+'_'+mf+'*.csv')
            if files:
                dfs[(n,mt,method)] = pd.read_csv(files[0])

hippo = json.loads(open('HippoRAG/reproduce/dataset/hotpotqa.json', encoding='utf-8').read())
type_map = {str(q.get('_id',q.get('id',''))): q.get('type','unknown') for q in hippo}

def get_ids(n, qtype):
    f = 'experiments/hotpotqa/medium_200/qa-pairs/qa-pairs.json' if n==200 else 'experiments/hotpotqa/large_scale/qa-pairs/qa-pairs.json'
    qa = json.loads(open(f, encoding='utf-8').read())
    return set(str(q['id']) for q in qa if type_map.get(str(q['id']))==qtype)

def acc(df, ids=None):
    if ids is not None:
        df = df[df['id'].astype(str).isin(ids)]
    if len(df)==0:
        return float('nan'), 0
    return (df['eval_verdict']=='correct').mean(), len(df)

ne = lambda a, b: a != b

# ===== Table 1: Overall =====
print('='*70)
print('Table 1: Overall Accuracy')
print('='*70)
print('{:<20s} {:>10s} {:>10s}'.format('Method','N=200','N=500'))
print('-'*42)
for mt in ['8b','70b']:
    for method in ['baseline','sparql']:
        lbl = ('SPARQL CoT ' if method=='sparql' else 'Baseline ')+mt
        a2,_ = acc(dfs[(200,mt,method)])
        a5,_ = acc(dfs[(500,mt,method)])
        print('{:<20s} {:>9.1%} {:>9.1%}'.format(lbl,a2,a5))
print()

# ===== Table 2: By question type =====
print('='*70)
print('Table 2: Accuracy by Question Type')
print('='*70)
for qtype in ['bridge','comparison']:
    print()
    print('  --- {} ---'.format(qtype.upper()))
    print('  {:<20s} {:>10s} {:>10s}'.format('Method','N=200','N=500'))
    print('  '+'-'*42)
    for mt in ['8b','70b']:
        for method in ['baseline','sparql']:
            lbl = ('SPARQL CoT ' if method=='sparql' else 'Baseline ')+mt
            a2, c2 = acc(dfs[(200,mt,method)], get_ids(200,qtype))
            a5, c5 = acc(dfs[(500,mt,method)], get_ids(500,qtype))
            print('  {:<20s} {:>5.1%} ({:>3d}/{:<3d}) {:>5.1%} ({:>3d}/{:<3d})'.format(
                lbl, a2, int(a2*c2), c2, a5, int(a5*c5), c5))
print()

# ===== Table 3: Paired =====
print('='*70)
print('Table 3: Paired SPARQL CoT vs Baseline (net = sparql_only - base_only)')
print('='*70)
print('{:<12s} {:>6s} {:>6s} {:>8s}  {:>10s} {:>12s}'.format(
    'Config','B-only','S-only','Net(All)','Net(Bridge)','Net(Compare)'))
print('-'*60)
for nn in [200,500]:
    for mt in ['8b','70b']:
        b = dfs[(nn,mt,'baseline')]
        s = dfs[(nn,mt,'sparql')]
        m = b[['id','eval_verdict']].merge(s[['id','eval_verdict']],on='id',suffixes=('_b','_s'))
        base_ok = m.eval_verdict_b=='correct'
        sparql_ok = m.eval_verdict_s=='correct'
        bo = (base_ok & ~sparql_ok).sum()
        so = (~base_ok & sparql_ok).sum()
        net = so - bo

        bids = get_ids(nn,'bridge')
        cids = get_ids(nn,'comparison')
        mb = m[m['id'].astype(str).isin(bids)]
        mc = m[m['id'].astype(str).isin(cids)]

        bo_br = (  (mb.eval_verdict_b=='correct') & (mb.eval_verdict_s != 'correct')).sum()
        so_br = (  (mb.eval_verdict_b != 'correct') & (mb.eval_verdict_s=='correct')).sum()
        bo_co = (  (mc.eval_verdict_b=='correct') & (mc.eval_verdict_s != 'correct')).sum()
        so_co = (  (mc.eval_verdict_b != 'correct') & (mc.eval_verdict_s=='correct')).sum()

        lbl = 'N={} {}'.format(nn, mt)
        print('{:<12s} {:>6d} {:>6d} {:>+4d}/{:<3d}  {:>+4d}/{:<5d} {:>+4d}/{:<5d}'.format(
            lbl, bo, so, net, len(m), so_br-bo_br, len(mb), so_co-bo_co, len(mc)))
print()

# ===== Table 4: Abstain =====
print('='*70)
print('Table 4: Abstain Rates')
print('='*70)
print('{:<20s} {:>10s} {:>10s}'.format('Method','N=200','N=500'))
print('-'*42)
for mt in ['8b','70b']:
    for method in ['baseline','sparql']:
        lbl = ('SPARQL CoT ' if method=='sparql' else 'Baseline ')+mt
        d2 = dfs[(200,mt,method)]
        d5 = dfs[(500,mt,method)]
        a2 = d2['final_abstain'].sum()/len(d2) if 'final_abstain' in d2.columns else 0
        a5 = d5['final_abstain'].sum()/len(d5) if 'final_abstain' in d5.columns else 0
        print('{:<20s} {:>9.1%} {:>9.1%}'.format(lbl,a2,a5))
