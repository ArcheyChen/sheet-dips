#!/usr/bin/env python3
"""Repair seam switches in a flattened surface: pieces of the uv grid that were placed on another wrap than their
neighbours (an offset of about one sheet spacing across a seam) are moved back onto the neighbours' sheet
(sheetdips.fix.seam_repair). Works on the tifxyz grid nodes directly, so the grid, its uv layout and its scale are
kept; only the positions of the moved pieces change.
usage: seam_fix.py SURFACE.tifxyz PRED OUT.tifxyz [--thr 4] [--min-edges 3] [--grow [--max-offset 25] [--snap 2.5]]
  PRED: the scroll's surface prediction (level matching the surface's coordinates), local zarr level or open-data key
Writes OUT.tifxyz (with charts.tif copied when present) and OUT.tifxyz/seam_fix.json (per-piece moves, seam pairs
before and after)."""
import argparse, json, os, shutil, sys, time
import numpy as np, tifffile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, write_tifxyz, normals
from sheetdips.fix import seam_repair, seam_repair_grow, seam_repair_icm, seam_components

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('pred'); ap.add_argument('out')
ap.add_argument('--thr', type=float, default=4.0, help='seam: neighbour offset along the normal above this (vx)')
ap.add_argument('--min-edges', type=int, default=3, help='pieces with fewer seam pairs to the fixed set are left alone')
ap.add_argument('--grow', action='store_true', help='grow the move over each piece from its seam (snapped to the predicted line within --snap vx node by node) instead of one rigid shift per piece; seams > --max-offset are ignored (layout tears)')
ap.add_argument('--snap', type=float, default=2.5)
ap.add_argument('--icm', action='store_true', help='joint labelling of chart units (seam_repair_icm) instead of the greedy order; implies chart units')
ap.add_argument('--charts', action='store_true', help='--grow: repair units = the atlas charts (charts.tif) split at their own seams, instead of seam-free pieces')
ap.add_argument('--min-snapped', type=float, default=0.5, help='--grow: leave a piece where it is if fewer of its nodes snap to a line')
ap.add_argument('--max-offset', type=float, default=25.0)
a = ap.parse_args()
t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
pred = Zarr2(a.pred); X, gs, meta = read_tifxyz(a.surface)


def seams(P):
    lab, sz, D = seam_components(P, normals(P, smooth=2.0), a.thr)
    k = sum(int((np.abs(d[np.isfinite(d)]) > a.thr).sum()) for d in D)
    return k, sz[sz > 0]


k0, sz0 = seams(X)
log(f'{a.surface}: {np.isfinite(X).all(-1).sum()} nodes; {k0} seam pairs (offset > {a.thr:g} vx); {len(sz0)} pieces, largest {sz0.max()}')
P, moves = (seam_repair_icm(pred, X, tifffile.imread(f'{a.surface}/charts.tif').astype(np.int64), a.thr, a.max_offset, min_snapped=a.min_snapped, snap=a.snap, log=log) if a.icm else
            seam_repair_grow(pred, X, a.thr, a.min_edges, a.max_offset, a.snap, min_snapped=a.min_snapped, units=(tifffile.imread(f'{a.surface}/charts.tif').astype(np.int64) if a.charts else None), log=log) if a.grow else seam_repair(pred, X, a.thr, a.min_edges, log=log))
k1, sz1 = seams(P)
log(f'after: {k1} seam pairs; {len(sz1)} pieces, largest {sz1.max()}')
write_tifxyz(a.out, P, 1, os.path.basename(a.out.rstrip('/')), {'seam_fix': {**vars(a), 'source': os.path.abspath(a.surface)}})
m = json.load(open(f'{a.out}/meta.json')); m['scale'] = meta['scale']; json.dump(m, open(f'{a.out}/meta.json', 'w'), indent=1)
if os.path.exists(f'{a.surface}/charts.tif'): shutil.copy(f'{a.surface}/charts.tif', f'{a.out}/charts.tif')
json.dump(dict(seam_pairs_before=k0, seam_pairs_after=k1, pieces_before=int(len(sz0)), largest_before=int(sz0.max()),
               pieces_after=int(len(sz1)), largest_after=int(sz1.max()), moves=moves), open(f'{a.out}/seam_fix.json', 'w'), indent=1)
log(f'wrote {a.out}')
