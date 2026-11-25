#!/usr/bin/env python3
"""Convert a flattened triangle mesh (OBJ with world vertices and texture coordinates, e.g. a ScrollFiesta atlas or
any UV-unwrapped segment) into a tifxyz surface, so that scan_segments.py / sheet_fix.py can work on it.
The UV plane is rasterised on a regular grid (one node every STEP uv units) and each node gets the world position
interpolated barycentrically inside the triangle that covers it; nodes outside the mesh are invalid (-1).
usage: obj_to_tifxyz.py MESH.obj OUT.tifxyz [--step 4] [--order zyx|xyz] [--uv-scale 1] [--name NAME] [--charts]
  --order: column order of the OBJ vertices (ScrollFiesta writes z, y, x); tifxyz stores x, y, z
  --uv-scale: world voxels per uv unit (1 for ScrollFiesta atlases, whose uv is in voxels)
Where several triangles cover a node (uv overlaps), the first one drawn wins; the share of such nodes is reported.
--charts also writes OUT/charts.tif: per node, the connected mesh component (chart) it comes from (-1 outside), so
that seams between charts placed next to each other in uv can be checked."""
import argparse, json, os, sys
import numpy as np, cv2, tifffile

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('obj'); ap.add_argument('out')
ap.add_argument('--step', type=float, default=4.0); ap.add_argument('--order', default='zyx', choices=['zyx', 'xyz'])
ap.add_argument('--uv-scale', type=float, default=1.0); ap.add_argument('--name')
ap.add_argument('--charts', action='store_true', help='write OUT/charts.tif (mesh component per node)')
a = ap.parse_args()

V, T, F = [], [], []
for line in open(a.obj):
    if line.startswith('v '): V.append(line.split()[1:4])
    elif line.startswith('vt '): T.append(line.split()[1:3])
    elif line.startswith('f '):
        p = [w.split('/') for w in line.split()[1:4]]
        F.append([int(q[0]) - 1 for q in p] + [int(q[1]) - 1 if len(q) > 1 and q[1] else int(q[0]) - 1 for q in p])
V = np.asarray(V, np.float64); T = np.asarray(T, np.float64); F = np.asarray(F, np.int64)
if a.order == 'zyx': V = V[:, ::-1]                               # -> x, y, z
print(f'{a.obj}: {len(V)} vertices, {len(T)} uv, {len(F)} faces', flush=True)
fv, ft = F[:, :3], F[:, 3:]
uv = T * a.uv_scale / a.step                                       # uv in grid nodes
u0, v0 = np.floor(uv.min(0)) - 1
uvg = uv - [u0, v0]
W, H = int(np.ceil(uvg[:, 0].max())) + 2, int(np.ceil(uvg[:, 1].max())) + 2
print(f'grid {H} x {W} nodes (step {a.step} uv units)', flush=True)
# triangle id raster: fill each triangle with id + 1 (int32); count overdraw with a second raster
tid = np.zeros((H, W), np.int32)
pts = np.round(uvg[ft] * 16).astype(np.int32)                      # 4 fractional bits (shift=4 below)
for k in range(len(F)):
    cv2.fillConvexPoly(tid, pts[k], int(k + 1), lineType=cv2.LINE_8, shift=4)
# overlaps: redraw in reverse order; a node covered by two different triangles is a real uv overlap only if it lies
# strictly inside both (neighbouring triangles share their edges)
tid2 = np.zeros((H, W), np.int32)
for k in range(len(F) - 1, -1, -1):
    cv2.fillConvexPoly(tid2, pts[k], int(k + 1), lineType=cv2.LINE_8, shift=4)


def bary(kk, P):
    A, B, C = uvg[ft[kk, 0]], uvg[ft[kk, 1]], uvg[ft[kk, 2]]
    d = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    l1 = ((B[:, 1] - C[:, 1]) * (P[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (P[:, 1] - C[:, 1])) / d
    l2 = ((C[:, 1] - A[:, 1]) * (P[:, 0] - C[:, 0]) + (A[:, 0] - C[:, 0]) * (P[:, 1] - C[:, 1])) / d
    return l1, l2, 1 - l1 - l2


both = (tid > 0) & (tid2 > 0) & (tid != tid2); br, bc = np.nonzero(both); Pb = np.stack([bc, br], 1).astype(np.float64)
inside = lambda kk: np.min(np.stack(bary(kk, Pb)), 0) > 0.05
over_n = int((inside(tid[both] - 1) & inside(tid2[both] - 1)).sum())
tid = np.where(tid > 0, tid, tid2)                                 # first-drawn wins where covered
ok = tid > 0
rr, cc = np.nonzero(ok); k = tid[ok] - 1
l1, l2, l3 = bary(k, np.stack([cc, rr], 1).astype(np.float64))
l1, l2, l3 = np.clip(l1, 0, 1), np.clip(l2, 0, 1), np.clip(l3, 0, 1); s = l1 + l2 + l3
X = (l1[:, None] * V[fv[k, 0]] + l2[:, None] * V[fv[k, 1]] + l3[:, None] * V[fv[k, 2]]) / s[:, None]
os.makedirs(a.out, exist_ok=True)
for j, c in enumerate('xyz'):
    g = np.full((H, W), -1, np.float32); g[rr, cc] = X[:, j]; tifffile.imwrite(f'{a.out}/{c}.tif', g)
q = X
meta = {'uuid': a.name or os.path.basename(a.out.rstrip('/')), 'name': a.name or os.path.basename(a.out.rstrip('/')),
        'type': 'seg', 'format': 'tifxyz', 'scale': [1 / (a.step * a.uv_scale)] * 2,
        'bbox': [q.min(0).tolist(), q.max(0).tolist()],
        'source': {'obj': os.path.abspath(a.obj), 'uv_origin_nodes': [float(u0), float(v0)], 'step_uv': a.step, 'order': a.order}}
json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
if a.charts:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, wid = np.unique(np.round(V, 3), axis=0, return_inverse=True); wid = wid.ravel()   # vertices duplicated at uv seams
    fw = wid[fv]; e = np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]]]); nv = int(wid.max()) + 1
    nc, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
    ch = np.full((H, W), -1, np.int32); ch[rr, cc] = lab[fw[k, 0]]; tifffile.imwrite(f'{a.out}/charts.tif', ch)
    print(f'{nc} mesh components (charts)')
print(f'wrote {a.out}: {ok.sum()} valid nodes ({ok.mean() * 100:.1f} % of the grid), {over_n} nodes inside two triangles '
      f'(uv overlaps, {over_n / max(ok.sum(), 1) * 100:.2f} %)')
