#!/usr/bin/env python3
"""Check a seam audit and a seam repair against independent reference segments (e.g. the official, human-reviewed
segments of the same scroll, registered to the same volume).

Each atlas node is matched to the nearest reference segment node (references upsampled to ~4 vx): 'on' a reference
if within --tol vx of its plane and 2 * --tol in 3D. The winding of each reference segment is read from its name
(--winding-regex, default '-w(\\d+)_').
 - seam pairs of the ORIGINAL (across charts, 4 < |offset| <= 25 vx): winding difference of the two nodes'
   segments, split by 3D gap (<= 30 vx: local switches; > 30: layout tears).
 - transition table original -> variant per pair: continuous (one segment holds both nodes <= --grid vx apart on its
   grid), broken (both on references, not continuous), one side, none.
 - controls: inside-chart and across-chart pairs with |offset| <= 1 vx (a chart moved onto a wrong wrap shows here);
   'all': every neighbour pair (net census).
 - nodes the variant moved > 2 vx: on references before / after.
usage: seam_check.py 'REF_GLOB' ORIG.tifxyz VARIANT.tifxyz OUT.json [--tol 3] [--grid 12]
  REF_GLOB: tifxyz dirs of the reference segments, e.g.
    'PHerc0139/segments/*/mesh/*-on-20250728140407-9.362um.tifxyz' (quoted; expanded by the script)
  ORIG needs charts.tif (obj_to_tifxyz.py --charts / seam_audit.py)."""
import argparse, os, sys, glob, json, re, numpy as np, tifffile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import read_tifxyz, normals
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('refs'); ap.add_argument('orig'); ap.add_argument('variant'); ap.add_argument('out')
ap.add_argument('--tol', type=float, default=3.0); ap.add_argument('--grid', type=float, default=12.0)
ap.add_argument('--winding-regex', default=r'-w(\d+)_')
a = ap.parse_args(); f0, f1, out = a.orig, a.variant, a.out
TOL, GRID, FAR = a.tol, a.grid, 40.0
X0 = read_tifxyz(f0)[0]; X1 = read_tifxyz(f1)[0]; ch = tifffile.imread(f'{f0}/charts.tif')
ok0 = np.isfinite(X0).all(-1); n0 = normals(X0, smooth=2.0)
lo = np.nanmin(X0[ok0], 0) - 60; hi = np.nanmax(X0[ok0], 0) + 60
P_, N_, S_, RC_, W_, names = [], [], [], [], [], []
for d in sorted(glob.glob(a.refs)):
    if not os.path.exists(f'{d}/meta.json'): continue
    S = read_tifxyz(d)[0]; sc = json.load(open(f'{d}/meta.json'))['scale'][0]; up = max(1, int(round((1 / sc) / 4)))
    okS = np.isfinite(S).all(-1)
    if not okS.any(): continue
    Sf = np.where(okS[..., None], S, 0.0)
    U = np.stack([ndi.zoom(Sf[..., k], up, order=1) for k in range(3)], -1)
    okU = ndi.binary_erosion(ndi.zoom(okS.astype(np.uint8), up, order=0) > 0, iterations=up)
    nU = normals(np.where(okU[..., None], U, np.nan), smooth=1.0)
    inb = okU & np.all((U >= lo) & (U <= hi), -1) & np.isfinite(nU).all(-1); rr, cc = np.nonzero(inb)
    if not len(rr): continue
    name = d.split('/segments/')[1].split('/')[0] if '/segments/' in d else os.path.basename(d.rstrip('/')); m = re.search(a.winding_regex, name); w = int(m.group(1)) if m else -1
    si = len(names); names.append(name)
    P_.append(U[rr, cc]); N_.append(nU[rr, cc]); S_.append(np.full(len(rr), si)); RC_.append(np.stack([rr, cc], 1)); W_.append(np.full(len(rr), w))
P_, N_, S_, RC_, W_ = map(np.concatenate, (P_, N_, S_, RC_, W_)); T = cKDTree(P_)
print(f'{len(names)} reference segments, {len(P_)} nodes', flush=True)


def nodes(X):
    """per valid node: seg, winding, rc, dpl, d3 (nearest official node within FAR; NaN/-1 beyond)."""
    ok = np.isfinite(X).all(-1); P = X[ok]; d, i = T.query(P, distance_upper_bound=FAR); f = np.isfinite(d); i2 = np.where(f, i, 0)
    dpl = np.where(f, np.abs(((P - P_[i2]) * N_[i2]).sum(-1)), np.nan); d = np.where(f, d, np.nan)
    G = lambda v, fill: (lambda A: (A.__setitem__(ok, v), A)[1])(np.full(ok.shape + v.shape[1:], fill, v.dtype))
    return dict(ok=ok, seg=G(np.where(f, S_[i2], -1), -1), w=G(np.where(f, W_[i2], -1), -1), rc=G(RC_[i2], -1),
                dpl=G(dpl, np.nan), d3=G(d, np.nan), on=G(f & (dpl <= TOL) & (d <= 2 * TOL), False))


A0, A1 = nodes(X0), nodes(X1)
for k in ('on',): print('on references: orig', A0[k].sum(), 'variant', A1[k].sum(), flush=True)
R = {'segments': names}
# pairs
pairs = {'seam': [], 'seam_local': [], 'ctrl_inside': [], 'ctrl_across': [], 'all': []}
rng = np.random.default_rng(0)
for ax in (0, 1):
    a = [slice(None)] * 2; b = [slice(None)] * 2; a[ax] = slice(0, -1); b[ax] = slice(1, None); a, b = tuple(a), tuple(b)
    m = ok0[a] & ok0[b]; dn = np.abs(((X0[b] - X0[a]) * (0.5 * (n0[a] + n0[b]))).sum(-1)); same = ch[a] == ch[b]
    gg = np.linalg.norm(X0[b] - X0[a], axis=-1)
    for k, sel in (('seam', m & ~same & (dn > 4) & (dn <= 25)), ('seam_local', m & ~same & (dn > 4) & (dn <= 25) & (gg <= 30)),
                   ('ctrl_inside', m & same & (dn <= 1)), ('ctrl_across', m & ~same & (dn <= 1)), ('all', m)):
        r, c = np.nonzero(sel); Pp = np.stack([r, c, r + (ax == 0), c + (ax == 1)], 1)
        if k in ('ctrl_inside', 'ctrl_across') and len(Pp) > 40000: Pp = Pp[rng.choice(len(Pp), 40000, replace=False)]
        pairs[k].append(Pp)
pairs = {k: np.concatenate(v) for k, v in pairs.items()}


def state(A, Pp):
    """per pair: 0 continuous, 1 broken (both on, not continuous), 2 one side on, 3 none."""
    ia, ib = (Pp[:, 0], Pp[:, 1]), (Pp[:, 2], Pp[:, 3])
    oa, ob = A['on'][ia], A['on'][ib]
    cont = oa & ob & (A['seg'][ia] == A['seg'][ib]) & (np.linalg.norm((A['rc'][ia] - A['rc'][ib]).astype(float), axis=1) * 4.0 <= GRID)
    s = np.full(len(Pp), 3); s[oa ^ ob] = 2; s[oa & ob] = 1; s[cont] = 0; return s


L = ['continuous', 'broken', 'one_side', 'none']
for k, Pp in pairs.items():
    ia, ib = (Pp[:, 0], Pp[:, 1]), (Pp[:, 2], Pp[:, 3])
    gap = np.linalg.norm(X0[ia] - X0[ib], axis=1); s0 = state(A0, Pp); s1 = state(A1, Pp)
    both = A0['on'][ia] & A0['on'][ib]; wd = np.abs(A0['w'][ia] - A0['w'][ib])
    rk = {'n': int(len(Pp))}
    if k == 'all':
        rk['continuous_before'] = int((s0 == 0).sum()); rk['continuous_after'] = int((s1 == 0).sum())
        rk['scorable_before'] = int((s0 <= 1).sum()); rk['scorable_after'] = int((s1 <= 1).sum())
        rk['transition_orig_to_variant'] = {L[i]: {L[j]: int(((s0 == i) & (s1 == j)).sum()) for j in range(4)} for i in range(4)}
        R[k] = rk; print(k, json.dumps(rk), flush=True); continue
    for g, gs in (('gap<=30', gap <= 30), ('gap>30', gap > 30)):
        sel = both & gs
        rk[g] = dict(pairs=int(gs.sum()), both_on=int(sel.sum()), continuous=int((sel & (s0 == 0)).sum()),
                     winding_diff={str(v): int((wd[sel] == v).sum()) for v in range(0, 5)}, winding_diff_ge5=int((wd[sel] >= 5).sum()),
                     same_seg_far_on_grid=int((sel & (s0 != 0) & (A0['seg'][ia] == A0['seg'][ib])).sum()))
    rk['transition_orig_to_variant'] = {L[i]: {L[j]: int(((s0 == i) & (s1 == j)).sum()) for j in range(4)} for i in range(4)}
    R[k] = rk; print(k, json.dumps(rk), flush=True)
# moved nodes
mv = ok0 & np.isfinite(X1).all(-1) & (np.linalg.norm(np.nan_to_num(X1 - X0), axis=-1) > 2.0)
left = mv & A0['on'] & ~A1['on']; came = mv & ~A0['on'] & A1['on']
dd = A1['dpl'][left]; d3 = A1['d3'][left]
R['moved'] = dict(n=int(mv.sum()), on_before=int((mv & A0['on']).sum()), on_after=int((mv & A1['on']).sum()), left=int(left.sum()), came=int(came.sum()),
                  left_dpl_hist=dict(zip(['3-5', '5-8', '8-12', '12-20', '20-40', 'none<=40'],
                                         [int(((dd > 3) & (dd <= 5)).sum()), int(((dd > 5) & (dd <= 8)).sum()), int(((dd > 8) & (dd <= 12)).sum()),
                                          int(((dd > 12) & (dd <= 20)).sum()), int(((dd > 20) & (dd <= 40)).sum()), int(np.isnan(dd).sum())])),
                  left_but_dpl_le3_d3_gt6=int((left & (A1['dpl'] <= TOL)).sum()),
                  stayed_on_same_winding=int((mv & A0['on'] & A1['on'] & (A0['w'] == A1['w'])).sum()),
                  on_other_winding=int((mv & A0['on'] & A1['on'] & (A0['w'] != A1['w'])).sum()))
print('moved', json.dumps(R['moved']), flush=True)
json.dump(R, open(out, 'w'), indent=1); print('wrote', out)
