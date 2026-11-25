#!/usr/bin/env python3
"""Controlled test with a known answer: take a good surface, push part of it off its sheet along the normals (0 on
the anchor side, a smooth ramp to AMP voxels over RAMP canvas px, plus a +-WOB voxel wobble), repair it with the
anchor on the untouched part, and measure how far each version lies from the original along its normals, per
column band (anchor, ramp, displaced).
usage: synth_test.py SURFACE.tifxyz PRED OUT_DIR --rows R0 R1 --cols C0 C1 [--anchor-cols 700] [--amp 24]
                     [--ramp 600] [--wob 6] [--lambda 350] [--passes 3]
Also runs the usual local alternative as a baseline: snapping each point to the nearest predicted sheet line along its
normal (as the snapping steps of current pipelines do).
Writes OUT/base.tifxyz, OUT/perturbed.tifxyz, OUT/snapped.tifxyz (baseline), OUT/repaired.tifxyz (render them and run an ink model to compare the
text, as in the README) and OUT/synth.json."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, write_tifxyz, to_canvas, normals
from sheetdips.fix import repair
from sheetdips.offsheet import offsheet_map
from scipy import ndimage as ndi
import cv2

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('pred'); ap.add_argument('out')
ap.add_argument('--rows', nargs=2, type=int, required=True); ap.add_argument('--cols', nargs=2, type=int, required=True)
ap.add_argument('--anchor-cols', type=int, default=700); ap.add_argument('--amp', type=float, default=24)
ap.add_argument('--ramp', type=int, default=600); ap.add_argument('--wob', type=float, default=6)
ap.add_argument('--lambda', dest='lam', type=float, default=350); ap.add_argument('--passes', type=int, default=3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); t0 = time.time()
log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
pred = Zarr2(a.pred); X, gs, _ = read_tifxyz(a.surface)
B = to_canvas(X, gs, a.rows, a.cols); H, W = B.shape[:2]; nB = normals(B)
r, c = np.arange(H)[:, None], np.arange(W)[None, :]
t = np.clip((c - a.anchor_cols) / a.ramp, 0, 1); t = t * t * (3 - 2 * t)
delta = t * (a.amp + a.wob * np.sin(2 * np.pi * r / a.lam) * np.cos(2 * np.pi * c / (1.3 * a.lam)))
Pp = B + delta[..., None] * nB
anchor = np.zeros((H, W), bool); anchor[:, :a.anchor_cols] = True
Pr, hist = repair(pred, Pp, anchor, a.passes, log=log)
# baseline: local snap to the nearest predicted sheet line along the normal (what snapping steps in current pipelines
# do, e.g. villa snap / the 343 winners' snap_meshes: offset to the nearest sheet, lightly regularised)
on, off, ok = offsheet_map(pred, Pp, every=2, r=20)
off = np.where(np.isfinite(off), off, 0).astype(np.float32); off = ndi.median_filter(off, 5)
Ps = Pp + cv2.resize(off, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * np.nan_to_num(normals(Pp))
res = {}
for name, (c0, c1) in {'anchor': (0, a.anchor_cols), 'ramp': (a.anchor_cols, a.anchor_cols + a.ramp), 'displaced': (a.anchor_cols + a.ramp, W)}.items():
    dp = np.abs(((Pp - B) * nB).sum(-1))[:, c0:c1]; dr = np.abs(((Pr - B) * nB).sum(-1))[:, c0:c1]; ds_ = np.abs(((Ps - B) * nB).sum(-1))[:, c0:c1]
    ok = np.isfinite(dp) & np.isfinite(dr) & np.isfinite(ds_)
    res[name] = dict(perturbed_median=float(np.median(dp[ok])), repaired_median=float(np.median(dr[ok])),
                     repaired_within_2=float(np.mean(dr[ok] <= 2)), repaired_within_4=float(np.mean(dr[ok] <= 4)),
                     snap_median=float(np.median(ds_[ok])), snap_within_4=float(np.mean(ds_[ok] <= 4)))
    log(f"{name:9s}: off the original along its normals, perturbed median {res[name]['perturbed_median']:.1f} vx; nearest-sheet snap "
        f"{res[name]['snap_median']:.1f} vx (<= 4 vx: {res[name]['snap_within_4'] * 100:.0f} %); sheet-dips {res[name]['repaired_median']:.1f} vx "
        f"(<= 2 vx: {res[name]['repaired_within_2'] * 100:.0f} %, <= 4 vx: {res[name]['repaired_within_4'] * 100:.0f} %)")
for P, nm in ((B, 'base'), (Pp, 'perturbed'), (Ps, 'snapped'), (Pr, 'repaired')): write_tifxyz(f'{a.out}/{nm}.tifxyz', P, 4, nm)
json.dump(dict(args=vars(a), bands=res, passes=hist), open(f'{a.out}/synth.json', 'w'), indent=1)
log(f'wrote {a.out}')
