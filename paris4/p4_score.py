"""Score seam_audit switches on PHerc. Paris 4 against the official spiral's winding crossings
(rule: README section 5; the frozen rule is the default, the amended rule is --r-side 48 --tol 5 with the
reliable-cube reference from ref_filter.py).

usage: p4_score.py SURF.tifxyz SEAMS.csv CROSS.npz OUT.json [--variant REPAIRED.tifxyz] [--n-ctrl 80000]
  SURF: seam_audit's surface.tifxyz (with charts.tif) in the REFERENCE frame (unshifted).
  CROSS: npz with P (n, 3) zyx reference-frame crossing positions and W (n,) absolute windings (seed + level).
Side label of node n (chart c): chart-c nodes within 24 vx (3D) of n; each node takes the nearest crossing within
3 vx; label = most common winding over the distinct crossings if >= 3 and share >= 0.6, else none.
"""
import argparse, csv, json, os, sys
import numpy as np, tifffile
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import read_tifxyz, normals

ap = argparse.ArgumentParser()
ap.add_argument('surf'); ap.add_argument('seams'); ap.add_argument('cross'); ap.add_argument('out')
ap.add_argument('--variant'); ap.add_argument('--n-ctrl', type=int, default=80000)
ap.add_argument('--r-side', type=float, default=24.0); ap.add_argument('--tol', type=float, default=3.0)
ap.add_argument('--min-cross', type=int, default=3); ap.add_argument('--min-share', type=float, default=0.6)
ap.add_argument('--sets', default='switch,ctrl_across,ctrl_inside,ctrl_far', help='pair sets to label (coverage checks: controls only)')
a = ap.parse_args()

X = read_tifxyz(a.surf)[0]; ch = tifffile.imread(f'{a.surf}/charts.tif').astype(np.int64)
H, W_ = ch.shape; ok = np.isfinite(X).all(-1)
C = np.load(a.cross); P = C['P'][:, ::-1].astype(np.float64); Wc = C['W'].astype(np.int64)   # -> x, y, z like X
T = cKDTree(P)
WIN = int(np.ceil(a.r_side / 2.0)) + 2       # grid nodes are ~4 vx apart; generous window, 3D test below


def nearest_crossing(Xg, okg):
    near = np.full(Xg.shape[:2], -1, np.int64)
    idx = np.nonzero(okg)
    d, j = T.query(Xg[idx], k=1, distance_upper_bound=a.tol)
    j[~np.isfinite(d)] = -1; near[idx] = j
    return near


def side_label(Xg, near, r, c):
    if not np.isfinite(Xg[r, c]).all():
        return None
    r0, r1, c0, c1 = max(0, r - WIN), min(H, r + WIN + 1), max(0, c - WIN), min(W_, c + WIN + 1)
    blk = Xg[r0:r1, c0:c1]; nb = near[r0:r1, c0:c1]; cb = ch[r0:r1, c0:c1]
    m = (cb == ch[r, c]) & (nb >= 0) & np.isfinite(blk).all(-1)
    m &= np.linalg.norm(np.where(m[..., None], blk, 0) - Xg[r, c], axis=-1) <= a.r_side
    js = np.unique(nb[m])
    if len(js) < a.min_cross:
        return None
    v, n = np.unique(Wc[js], return_counts=True); k = n.argmax()
    return int(v[k]) if n[k] / n.sum() >= a.min_share else None


near0 = nearest_crossing(X, ok)
print('nodes', int(ok.sum()), 'nodes with a crossing <= tol', int((near0 >= 0).sum()), flush=True)

# pairs
sw = []
with open(a.seams) as f:
    for row in csv.DictReader(f):
        if row['class'] == 'switch':
            sw.append((int(row['row']), int(row['col']), int(row['row2']), int(row['col2'])))
sw = np.array(sw, np.int64).reshape(-1, 4)
n0 = normals(X, smooth=2.0); rng = np.random.default_rng(0)


def neighbour_pairs(across):
    out = []
    for dr, dc in ((1, 0), (0, 1)):
        A = (slice(0, H - dr), slice(0, W_ - dc)); B = (slice(dr, H), slice(dc, W_))
        m = ok[A] & ok[B] & ((ch[A] != ch[B]) if across else (ch[A] == ch[B]))
        if across:
            off = np.abs(((X[B] - X[A]) * n0[A]).sum(-1)); m &= np.isfinite(off) & (off <= 1.0)
        r, c = np.nonzero(m); out.append(np.stack([r, c, r + dr, c + dc], 1))
    out = np.concatenate(out)
    return out[rng.choice(len(out), min(len(out), a.n_ctrl), replace=False)]




def far_pairs():
    """Same-chart pairs >= 2 r_side apart (3D) along a grid row/column: windows do not overlap (non-tautological)."""
    k = int(np.ceil(2 * a.r_side / 4.0)) + 1; out = []
    for dr, dc in ((k, 0), (0, k)):
        A = (slice(0, H - dr), slice(0, W_ - dc)); B = (slice(dr, H), slice(dc, W_))
        m = ok[A] & ok[B] & (ch[A] == ch[B])
        m &= np.linalg.norm(np.where(m[..., None], X[B] - X[A], 0), axis=-1) >= 2 * a.r_side
        r, c = np.nonzero(m); out.append(np.stack([r, c, r + dr, c + dc], 1))
    out = np.concatenate(out)
    return out[rng.choice(len(out), min(len(out), a.n_ctrl), replace=False)]


sets = {'switch': sw, 'ctrl_across': neighbour_pairs(True), 'ctrl_inside': neighbour_pairs(False), 'ctrl_far': far_pairs()}
sets = {k: v for k, v in sets.items() if k in a.sets.split(',')}
res = {'params': vars(a), 'nodes': int(ok.sum()), 'nodes_near_crossing': int((near0 >= 0).sum())}
if 'ctrl_across' in sets and len(sets['ctrl_across']):
    sel = near0 >= 0; cc = {}
    for cid, j in zip(ch[sel], near0[sel]):
        cc.setdefault(int(cid), set()).add(int(j))
    pr = sets['ctrl_across']; n_side = np.array([len(cc.get(int(ch[r, c]), ())) for r, c in np.concatenate([pr[:, :2], pr[:, 2:]])])
    res['across_side_chart_crossings_pct10_50_90'] = np.percentile(n_side, [10, 50, 90]).tolist()
    res['across_side_chart_crossings_ge3'] = float((n_side >= a.min_cross).mean())
    print('chart-wide crossings per across side: pct10/50/90', res['across_side_chart_crossings_pct10_50_90'], 'share>=3', res['across_side_chart_crossings_ge3'], flush=True)
labels = {}
for name, pr in sets.items():
    L = np.array([[side_label(X, near0, r, c) if True else None, side_label(X, near0, r2, c2)] for r, c, r2, c2 in pr], object)
    lab = np.array([x is not None and y is not None for x, y in L], bool) if len(L) else np.zeros(0, bool)
    dif = np.array([abs(x - y) for x, y in L[lab]], np.int64) if lab.any() else np.zeros(0, np.int64)
    hist = {str(k): int((dif == k).sum()) for k in range(4)}; hist['>=4'] = int((dif >= 4).sum())
    res[name] = dict(pairs=int(len(pr)), labelled=int(lab.sum()), D=float((dif > 0).mean()) if len(dif) else None, diff_hist=hist)
    labels[name] = (pr, L, lab)
    print(name, json.dumps(res[name]), flush=True)

if a.variant:
    V = read_tifxyz(a.variant)[0]; okv = np.isfinite(V).all(-1); nearv = nearest_crossing(V, okv)
    tr = {}
    for name in ('switch', 'ctrl_across'):
        pr, L, lab = labels[name]; t = {'equal_before': 0, 'equal_after': 0, 'scored': 0, 'fixed': 0, 'broken': 0}
        for (r, c, r2, c2), (x, y), l in zip(pr, L, lab):
            u, w = side_label(V, nearv, r, c), side_label(V, nearv, r2, c2)
            if not l or u is None or w is None:
                continue
            t['scored'] += 1; t['equal_before'] += x == y; t['equal_after'] += u == w
            t['fixed'] += (x != y) and (u == w); t['broken'] += (x == y) and (u != w)
        tr[name] = t; print('variant', name, t, flush=True)
    # touched pairs: all grid-neighbour pairs (any charts) with a node moved > 2 vx; only these can change
    mv = ok & okv & (np.linalg.norm(np.nan_to_num(V - X), axis=-1) > 2); tp = []
    for dr, dc in ((1, 0), (0, 1)):
        A = (slice(0, H - dr), slice(0, W_ - dc)); B = (slice(dr, H), slice(dc, W_))
        m = ok[A] & ok[B] & okv[A] & okv[B] & (mv[A] | mv[B])
        r, c = np.nonzero(m); tp.append(np.stack([r, c, r + dr, c + dc], 1))
    tp = np.concatenate(tp); n_touched = len(tp)
    tp = tp[rng.choice(len(tp), min(len(tp), a.n_ctrl), replace=False)]
    t = {'pairs': int(n_touched), 'sampled': int(len(tp)), 'scored': 0, 'equal_before': 0, 'equal_after': 0, 'fixed': 0, 'broken': 0}
    for r, c, r2, c2 in tp:
        x, y = side_label(X, near0, r, c), side_label(X, near0, r2, c2)
        u, w = side_label(V, nearv, r, c), side_label(V, nearv, r2, c2)
        if None in (x, y, u, w):
            continue
        t['scored'] += 1; t['equal_before'] += x == y; t['equal_after'] += u == w
        t['fixed'] += (x != y) and (u == w); t['broken'] += (x == y) and (u != w)
    tr['touched'] = t; print('variant touched', t, flush=True)
    res['variant'] = tr
json.dump(res, open(a.out, 'w'), indent=1)
print('wrote', a.out)
