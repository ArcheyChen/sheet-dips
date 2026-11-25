#!/usr/bin/env python3
"""Independent visual checks of a repaired surface against the CT (no surface prediction involved):
1. surface images: CT sampled on the surface (mean of layers -1..+1 along the normals), before | after: papyrus fibres
   should run continuously across repaired patches;
2. cross-sections: the CT slice at each z given, with the before (red) and after (yellow) surface curves drawn: on one
   sheet the curve follows a single bright layer.
usage: qa.py BEFORE AFTER CT OUT_PREFIX [--z Z1,Z2,..] [--half 600] [--rows R0 R1] [--cols C0 C1]
  BEFORE / AFTER: tifxyz directories or points3d.npy files (canvas points, e.g. OUT/points3d.npy of sheet_fix.py);
  a tifxyz BEFORE is resampled to canvas px with --rows / --cols as given to sheet_fix.py.
  CT: level 0 of the scroll volume (local zarr v2 level directory or open-data key)."""
import argparse, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, to_canvas, normals, thick_sample

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('before'); ap.add_argument('after'); ap.add_argument('ct'); ap.add_argument('out')
ap.add_argument('--z', default=''); ap.add_argument('--half', type=int, default=600)
ap.add_argument('--rows', nargs=2, type=int); ap.add_argument('--cols', nargs=2, type=int)
a = ap.parse_args()
ct = Zarr2(a.ct)


def load(p):
    if p.endswith('.npy'): return np.load(p).astype(np.float64)
    X, gs, _ = read_tifxyz(p); return to_canvas(X, gs, a.rows, a.cols)


Pb, Pa = load(a.before), load(a.after)
cl = cv2.createCLAHE(2.0, (8, 8))
imgs = []
for P, lab in ((Pb, 'before'), (Pa, 'after')):
    V, ok = thick_sample(ct, P, normals(P), 3, 1)
    im = cl.apply(V.mean(0).astype(np.uint8)); im[~ok] = 0
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); cv2.putText(im, f'CT on the surface, {lab}', (10, 32), 0, 1.0, (0, 0, 255), 2); imgs.append(im)
cv2.imwrite(f'{a.out}_surface.jpg', np.vstack([imgs[0], np.full((8, imgs[0].shape[1], 3), 90, np.uint8), imgs[1]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f'wrote {a.out}_surface.jpg')
for z in [int(v) for v in a.z.split(',') if v]:
    sel = [P[np.abs(P[..., 2] - z) < 0.5] for P in (Pb, Pa)]
    if not len(sel[0]) and not len(sel[1]): print(f'z{z}: the surfaces do not cross this slice'); continue
    Q = np.concatenate([s for s in sel if len(s)]); cx, cy = np.median(Q[:, 0]), np.median(Q[:, 1]); h = a.half
    gx, gy = np.meshgrid(np.arange(cx - h, cx + h), np.arange(cy - h, cy + h))
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1)
    im = cv2.cvtColor(cl.apply(ct.sample(pts).reshape(gx.shape)), cv2.COLOR_GRAY2BGR)
    for s, col in zip(sel, ((0, 0, 255), (0, 220, 255))):
        q = np.round(s[:, :2] - [cx - h, cy - h]).astype(int); m = (q >= 0).all(1) & (q < 2 * h).all(1); im[q[m, 1], q[m, 0]] = col
    cv2.putText(im, f'z{z}: before red, after yellow (1 px = 1 voxel)', (10, 32), 0, 0.9, (255, 255, 255), 2)
    cv2.imwrite(f'{a.out}_z{z}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(f'wrote {a.out}_z{z}.jpg')
