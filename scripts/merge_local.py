#!/usr/bin/env python3
"""Keep a repair only where it is needed: the repaired surface replaces the original within GROW nodes of the
original's seam jumps and over the pieces the seam repair moved; elsewhere the original stays, with a smooth
transition (the displacement is blended over FEATHER nodes), so a re-trace cannot spoil parts that were right.
usage: merge_local.py ORIGINAL.tifxyz SEAM_FIXED.tifxyz REPAIRED.tifxyz OUT.tifxyz [--grow 10] [--feather 3] [--thr 4]"""
import argparse, json, os, shutil, sys
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import read_tifxyz, write_tifxyz, normals
from sheetdips.fix import seam_components

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('original'); ap.add_argument('seam_fixed'); ap.add_argument('repaired'); ap.add_argument('out')
ap.add_argument('--grow', type=int, default=10); ap.add_argument('--feather', type=float, default=3.0)
ap.add_argument('--thr', type=float, default=4.0)
a = ap.parse_args()
X0, gs, meta = read_tifxyz(a.original); XF, _, _ = read_tifxyz(a.seam_fixed); XR, _, _ = read_tifxyz(a.repaired)
v = np.isfinite(X0).all(-1)
_, _, D = seam_components(X0, normals(X0, smooth=2.0), a.thr)
J = np.zeros(v.shape, bool)
for ax, d in enumerate(D):
    m = np.isfinite(d) & (np.abs(d) > a.thr)
    if ax == 0: J[:-1] |= m; J[1:] |= m
    else: J[:, :-1] |= m; J[:, 1:] |= m
moved = v & (np.linalg.norm(np.nan_to_num(XF - X0), axis=-1) > 2.0)
mask = (ndi.binary_dilation(J, iterations=a.grow) | moved) & v
w = np.clip(ndi.gaussian_filter(mask.astype(np.float64), a.feather) * 2, 0, 1); w[mask] = 1.0
okR = np.isfinite(XR).all(-1)
P = X0.copy(); m = v & okR; P[m] = X0[m] + w[m, None] * (XR[m] - X0[m])
write_tifxyz(a.out, P, 1, os.path.basename(a.out.rstrip('/')), {'merge_local': {**vars(a)}})
mm = json.load(open(f'{a.out}/meta.json')); mm['scale'] = meta['scale']; json.dump(mm, open(f'{a.out}/meta.json', 'w'), indent=1)
if os.path.exists(f'{a.original}/charts.tif'): shutil.copy(f'{a.original}/charts.tif', f'{a.out}/charts.tif')
print(f'repair applied on {mask.sum()} of {v.sum()} nodes ({mask.sum() / v.sum() * 100:.1f} %), blended on {int(((w > 0) & (w < 1) & v).sum())} more')
