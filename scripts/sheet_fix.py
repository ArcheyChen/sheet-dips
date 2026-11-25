#!/usr/bin/env python3
"""Repair sheet switches in a segmented surface (tifxyz) by dip integration on the surface prediction.

1. The surface (optionally a crop, in canvas px) is resampled to canvas resolution.
2. On-sheet map from the surface prediction; anchor = the largest on-sheet region away from off-sheet cells.
3. Each pass: the prediction is sampled along the normals (LAYERS), local layer dips are measured by
   cross-correlation, the offset field through the anchor is integrated, and the surface moves along its normals.
4. On-sheet map again; the repaired surface is written as tifxyz (VC3D can open and render it).

usage: sheet_fix.py SURFACE.tifxyz PRED OUT_DIR [--rows R0 R1] [--cols C0 C1] [--passes 3] [--layers 81]
                    [--margin 6] [--step 4] [--ds 2] [--sample-every 1]
  PRED: the scroll's surface prediction, level 0: a local zarr v2 level directory or an open-data key such as
  PHerc0306B/representations/predictions/surfaces/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr/0
Writes OUT/surface.tifxyz, OUT/points3d.npy (canvas points), OUT/onsheet_before.png, OUT/onsheet_after.png and
OUT/summary.json."""
import argparse, json, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, write_tifxyz, to_canvas
from sheetdips.dips import anchor_from_map
from sheetdips.fix import repair
from sheetdips.offsheet import offsheet_map, map_image

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('pred'); ap.add_argument('out')
ap.add_argument('--rows', nargs=2, type=int); ap.add_argument('--cols', nargs=2, type=int)
ap.add_argument('--passes', type=int, default=3); ap.add_argument('--layers', type=int, default=81)
ap.add_argument('--margin', type=float, default=6); ap.add_argument('--step', type=int, default=4)
ap.add_argument('--ds', type=int, default=2, help='canvas px per dip-grid px')
ap.add_argument('--sample-every', type=int, default=1, help='sample the prediction on every n-th canvas px (a divisor of --ds; less memory)')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); t0 = time.time()
log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
pred = Zarr2(a.pred)
X, gs, meta = read_tifxyz(a.surface)
P = to_canvas(X, gs, a.rows, a.cols); H, W = P.shape[:2]
log(f'{a.surface}: canvas {H} x {W} px ({np.isfinite(P).all(-1).mean() * 100:.0f} % valid)')
on0, off0, ok0 = offsheet_map(pred, P)
cv2.imwrite(f'{a.out}/onsheet_before.png', map_image(on0, off0, ok0))
log(f'on-sheet before: {on0[ok0].mean() * 100:.1f} %')
anc_map = anchor_from_map(on0, ok0, a.margin)
log(f'anchor: {anc_map[ok0].mean() * 100:.0f} % of the surface')
P, hist = repair(pred, P, anc_map, a.passes, a.layers, a.ds, log, a.sample_every)
on1, off1, ok1 = offsheet_map(pred, P)
cv2.imwrite(f'{a.out}/onsheet_after.png', map_image(on1, off1, ok1))
log(f'on-sheet after: {on1[ok1].mean() * 100:.1f} %')
np.save(f'{a.out}/points3d.npy', P.astype(np.float32))
write_tifxyz(f'{a.out}/surface.tifxyz', P, a.step, 'sheet_fix', {'sheet_fix': {**vars(a), 'source_meta': meta.get('name')}})
summary = dict(surface=a.surface, rows=a.rows, cols=a.cols, canvas=[H, W], onsheet_before=float(on0[ok0].mean()),
               offsheet_before=float((ok0 & ~on0 & np.isfinite(off0))[ok0].mean()), onsheet_after=float(on1[ok1].mean()),
               offsheet_after=float((ok1 & ~on1 & np.isfinite(off1))[ok1].mean()), anchor_share=float(anc_map[ok0].mean()), passes=hist)
json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=1)
log(f'wrote {a.out}/surface.tifxyz; on-sheet {summary["onsheet_before"] * 100:.1f} % -> {summary["onsheet_after"] * 100:.1f} %, '
    f'off-sheet {summary["offsheet_before"] * 100:.1f} % -> {summary["offsheet_after"] * 100:.1f} %')
