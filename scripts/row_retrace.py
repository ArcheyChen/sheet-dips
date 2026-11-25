#!/usr/bin/env python3
"""Re-trace a flattened surface row by row in its CT cross-sections (sheetdips.retrace; for grids whose rows are
constant-z curves, e.g. ScrollFiesta atlases converted by obj_to_tifxyz.py, best after seam_fix.py).
usage: row_retrace.py SURFACE.tifxyz PRED OUT.tifxyz [--propagate | --track] [--search 6] [--max-step 1] [--lam 0.5] [--no-unloop]
  --propagate: slice by slice (retrace.retrace_propagate): rows predicted from the traced row before them, then
           snapped by dynamic programming along the row; the recommended mode.
  --track: trace each row afresh from an anchor node (in the row's largest seam-free piece) along the predicted sheet
           (retrace.track_row) instead of offsetting the existing nodes; rows that jump across a seam are replaced
           by a trace that stays on one sheet. Nodes the trace cannot reach are dropped.
Writes OUT.tifxyz (charts.tif copied when present) and OUT.tifxyz/row_retrace.json."""
import argparse, json, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, write_tifxyz
from sheetdips.retrace import retrace_rows, track_row, retrace_propagate, project_rows, project_nearest
from sheetdips.io import Dense, normals
from sheetdips.fix import seam_components

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('pred'); ap.add_argument('out')
ap.add_argument('--search', type=float, default=6.0, help='offsets searched along the in-slice normal (+- vx)')
ap.add_argument('--max-step', type=float, default=1.0, help='largest offset change between neighbouring nodes (vx)')
ap.add_argument('--lam', type=float, default=0.5, help='cost per vx of offset change between neighbours')
ap.add_argument('--no-unloop', action='store_true')
ap.add_argument('--track', action='store_true')
ap.add_argument('--propagate', action='store_true', help='slice-by-slice: each row predicted from the traced row before it')
ap.add_argument('--drop-unreached', action='store_true', help='drop nodes no traced curve reaches (default: keep them as they were)')
ap.add_argument('--nearest', action='store_true', help='place nodes at the nearest point of their own traced curve (not along their normals)')
a = ap.parse_args()
t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
pred = Zarr2(a.pred); X, gs, meta = read_tifxyz(a.surface)
dz = np.nanmax(X[..., 2], 1) - np.nanmin(X[..., 2], 1)
log(f'{a.surface}: {X.shape[0]} rows; z spread within a row: median {np.nanmedian(dz):.2f} vx, max {np.nanmax(dz):.2f}')
if a.propagate:
    v = np.isfinite(X).all(-1); q = X[v]; lo = np.floor(q.min(0)).astype(int) - 12; hi = np.ceil(q.max(0)).astype(int) + 12
    vol = Dense(pred, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]); log(f'prediction box {vol.a.shape} in memory')
    lab, sz, _ = seam_components(X, normals(X, smooth=2.0), 4.0)
    C = retrace_propagate(vol, X, lab, sz, log=log); st = []
    P = project_nearest(X, C) if a.nearest else project_rows(X, C)   # nodes keep their place along the row
    n_proj = int(np.isfinite(P).all(-1).sum())
    if not a.drop_unreached:
        # nodes no traced curve reached follow the moves of the traced nodes around them (normalised Gaussian average of
        # the displacement, so traced and untraced parts meet without a step); far from any traced node they stay put
        from scipy import ndimage as ndi
        tr = np.isfinite(P).all(-1); keep = ~tr & np.isfinite(X).all(-1)
        Dv = np.where(tr[..., None], P - X, 0.0); w = tr.astype(np.float64); fillD = np.zeros_like(Dv); got = np.zeros(w.shape, bool)
        for sg in (2.0, 6.0, 16.0):
            ws = ndi.gaussian_filter(w, sg); Ds = np.stack([ndi.gaussian_filter(Dv[..., k], sg) for k in range(3)], -1)
            m = keep & ~got & (ws > 0.05)
            fillD[m] = Ds[m] / ws[m, None]; got |= m
        P[keep] = X[keep] + fillD[keep]
    log(f'projected onto the traced curves: {n_proj} nodes; {int(np.isfinite(P).all(-1).sum()) - n_proj} kept as they were')
elif a.track:
    v = np.isfinite(X).all(-1); q = X[v]; lo = np.floor(q.min(0)).astype(int) - 12; hi = np.ceil(q.max(0)).astype(int) + 12
    vol = Dense(pred, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]); log(f'prediction box {vol.a.shape} in memory')
    lab, sz, _ = seam_components(X, normals(X, smooth=2.0), 4.0)
    P = np.full(X.shape, np.nan); lost = 0
    for r in range(X.shape[0]):
        R = X[r]; ok = np.isfinite(R).all(1)
        if ok.sum() < 3: continue
        runs = np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1)
        for run in runs:
            if len(run) < 3: continue
            L = lab[r, run]; big = L[np.argmax(sz[L])]                        # the largest piece present in the run
            m = (L == big).astype(int); best, cur, bs, be = 0, 0, 0, 0     # its longest contiguous stretch
            for i, x in enumerate(m):
                cur = cur + 1 if x else 0
                if cur > best: best, be = cur, i
            anc = run[be - best // 2]
            T = track_row(vol, R[run[0]:run[-1] + 1], anc - run[0])
            P[r, run[0]:run[-1] + 1] = T; lost += int(np.isnan(T[:, 0]).sum())
        if r % 20 == 0: log(f'row {r}/{X.shape[0]}')
    log(f'track: {lost} nodes not reached')
    st = []
else:
    P, st = retrace_rows(pred, X, a.search, max_step=a.max_step, lam=a.lam, unloop=not a.no_unloop, log=log)
d = np.linalg.norm(P - X, axis=-1); v = np.isfinite(d)
log(f'moved: median {np.median(d[v]):.2f} vx, p90 {np.percentile(d[v], 90):.2f}; nodes kept {np.isfinite(P).all(-1).sum()} of {np.isfinite(X).all(-1).sum()}')
write_tifxyz(a.out, P, 1, os.path.basename(a.out.rstrip('/')), {'row_retrace': {**vars(a), 'source': os.path.abspath(a.surface)}})
m = json.load(open(f'{a.out}/meta.json')); m['scale'] = meta['scale']; json.dump(m, open(f'{a.out}/meta.json', 'w'), indent=1)
if os.path.exists(f'{a.surface}/charts.tif'): shutil.copy(f'{a.surface}/charts.tif', f'{a.out}/charts.tif')
json.dump(dict(moved_median=float(np.median(d[v])), moved_p90=float(np.percentile(d[v], 90))), open(f'{a.out}/row_retrace.json', 'w'), indent=1)
log(f'wrote {a.out}')
