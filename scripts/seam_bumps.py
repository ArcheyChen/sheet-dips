#!/usr/bin/env python3
"""Flatten right-angle detours ('box bumps') left on a repaired surface grid (sheetdips.bumps.bump_pass).

seam_audit.py --repair runs this after the ICM seam repair by default (pieces up to 300 nodes, then bump regions).
This script runs the same cleanup on any tifxyz grid, e.g. one repaired earlier.
usage: seam_bumps.py SURFACE.tifxyz OUT.tifxyz [--max-piece 300] [--thr 4] [--kmax 40] [--agree 0.6]
Writes OUT.tifxyz (charts.tif copied) and OUT.tifxyz/seam_bumps.json (per round: bumps, regions, tried, kept)."""
import argparse, json, os, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from sheetdips.io import read_tifxyz, write_tifxyz, normals
from sheetdips.bumps import bump_pass, bump_mask

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('out')
ap.add_argument('--max-piece', type=int, default=300, help='pieces larger than this are anchors and do not move')
ap.add_argument('--thr', type=float, default=4.0); ap.add_argument('--kmax', type=int, default=40)
ap.add_argument('--agree', type=float, default=0.6)
a = ap.parse_args()
t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
X, gs, meta = read_tifxyz(a.surface)
P, r1 = bump_pass(X, 'pieces', max_piece=a.max_piece, thr=a.thr, kmax=a.kmax, agree=a.agree, rounds=4, log=log)
P, r2 = bump_pass(P, 'bumps', thr=a.thr, kmax=a.kmax, agree=a.agree, rounds=3, log=log)
_, nb = bump_mask(P, normals(P, smooth=2.0), thr=a.thr, kmax=a.kmax); log(f'after: {nb} bumps')
write_tifxyz(a.out, P, 1, 'seam_bumps', {'seam_bumps': {'source': os.path.abspath(a.surface), **vars(a)}, 'scale': meta['scale']})
if os.path.exists(f'{a.surface}/charts.tif'): shutil.copy(f'{a.surface}/charts.tif', f'{a.out}/charts.tif')
json.dump(dict(params=vars(a), pieces=r1, bumps=r2, final_bumps=nb), open(f'{a.out}/seam_bumps.json', 'w'), indent=1)
log('done')
