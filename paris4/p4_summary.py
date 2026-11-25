"""Per-tile table and pooled totals of the Paris 4 scores (frozen rule, amended rule, repair variant).
usage: p4_summary.py DIR_WITH_atlas_tile_*  [OUT.json]"""
import glob, json, os, sys
root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else None
rows, pool = [], {}
for d in sorted(glob.glob(f'{root}/atlas_tile_*')):
    t = os.path.basename(d)[6:]; row = {'tile': t}
    for name in ('score', 'score_rel_r48_t5', 'score_rel_repair'):
        f = f'{d}/{name}.json'
        if not os.path.exists(f):
            continue
        r = json.load(open(f)); P = pool.setdefault(name, {})
        for k in ('switch', 'ctrl_across', 'ctrl_inside', 'ctrl_far'):
            if k in r and name != 'score_rel_repair':
                lab, eq = r[k]['labelled'], r[k]['diff_hist']['0']
                p = P.setdefault(k, {'pairs': 0, 'labelled': 0, 'diff': 0, 'hist': {}})
                p['pairs'] += r[k]['pairs']; p['labelled'] += lab; p['diff'] += lab - eq
                for h, v in r[k]['diff_hist'].items():
                    p['hist'][h] = p['hist'].get(h, 0) + v
                row[f'{name}:{k}'] = (lab, round((lab - eq) / lab, 3) if lab else None)
        for k, v in r.get('variant', {}).items():
            p = P.setdefault(k, {})
            for a, b in v.items():
                p[a] = p.get(a, 0) + b
            row[f'repair:{k}'] = (v['scored'], v['fixed'], v['broken'])
    rows.append(row)
for name, P in pool.items():
    for k, p in P.items():
        if 'labelled' in p:
            p['D'] = p['diff'] / p['labelled'] if p['labelled'] else None
res = {'tiles': [r['tile'] for r in rows], 'n_tiles': len(rows), 'pool': pool, 'per_tile': rows}
print(json.dumps(pool, indent=1))
for r in rows:
    print(r['tile'], {k: v for k, v in r.items() if k.startswith('score_rel_r48_t5:switch') or k.startswith('repair:switch')})
if out:
    json.dump(res, open(out, 'w'), indent=1)
