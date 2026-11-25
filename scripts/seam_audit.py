#!/usr/bin/env python3
"""Audit a flattened scroll surface for wrap switches at its seams, and optionally repair them.

A flattened atlas (e.g. a ScrollFiesta atlas_bake.obj) is a uv layout of many mesh charts. Where two charts meet in uv
but were taken from different wraps of the scroll, the surface steps from one sheet to the next across the seam.
This script finds every such seam on the uv grid and writes:

  OUT/surface.tifxyz        the surface as a tifxyz grid with charts.tif (converted when the input is an OBJ)
  OUT/seams.csv             one row per neighbour pair across a chart boundary with |offset| > --thr:
                            grid row/col of both nodes, world xyz of the first, signed offset along the normal (vx),
                            3D gap (vx), class: 'switch' (|offset| <= --max-offset and gap <= --max-gap: neighbours in
                            the scroll on different sheets) or 'tear' (uv neighbours that are not neighbours in the scroll)
  OUT/seam_map.png          the uv grid: grey = surface, red = switch, blue = tear (columns max-pooled to <= 8000 px)
  OUT/summary.json          counts, and the parameters used
  OUT/repaired.tifxyz       with --repair PRED: switches moved back onto the neighbours' sheet
                            (sheetdips.fix.seam_repair_icm with the defaults validated on PHerc0139, then the
                            piece / bump cleanup of sheetdips.bumps unless --no-cleanup), and
  OUT/moves.json            per labelled unit: offset, moved or why it was left in place

usage: seam_audit.py SURFACE(.obj|.tifxyz) OUT_DIR [--repair PRED] [--step 4] [--order zyx] [--thr 4]
                     [--max-offset 25] [--max-gap 30] [--min-snapped 0.7] [--no-cleanup] [--max-piece 300]
  PRED: the scroll's surface prediction in the surface's voxel frame, a local zarr level or an open-data key, e.g.
        PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr/0"""
import argparse, csv, json, os, subprocess, sys, time
import numpy as np, cv2, tifffile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
from sheetdips.io import Zarr2, read_tifxyz, write_tifxyz, normals

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('surface'); ap.add_argument('out')
ap.add_argument('--repair', metavar='PRED', help='also repair the switches, using this surface prediction')
ap.add_argument('--step', type=float, default=4.0, help='OBJ input: uv units per grid node')
ap.add_argument('--order', default='zyx', choices=['zyx', 'xyz'], help='OBJ input: vertex column order (ScrollFiesta: zyx)')
ap.add_argument('--thr', type=float, default=4.0, help='a seam pair has |offset along the normal| above this (vx)')
ap.add_argument('--max-offset', type=float, default=25.0, help='switch: |offset| at most this (vx); larger is a tear')
ap.add_argument('--max-gap', type=float, default=30.0, help='switch: 3D distance of the pair at most this (vx); larger is a tear')
ap.add_argument('--min-snapped', type=float, default=0.7, help='--repair: leave a unit in place if fewer of its nodes snap to a line')
ap.add_argument('--no-cleanup', action='store_true', help='--repair: stop after the ICM repair (skip the piece / bump cleanup of sheetdips.bumps)')
ap.add_argument('--max-piece', type=int, default=300, help='--repair cleanup: seam-free pieces up to this many nodes may move')
a = ap.parse_args()
t0 = time.time(); log = lambda s: print(f'[{time.time() - t0:6.0f} s] {s}', flush=True)
os.makedirs(a.out, exist_ok=True)

src = a.surface.rstrip('/')
if src.lower().endswith('.obj'):
    surf = os.path.join(a.out, 'surface.tifxyz')
    if not os.path.isdir(surf):
        log(f'converting {src} -> {surf}')
        subprocess.run([sys.executable, os.path.join(HERE, 'obj_to_tifxyz.py'), src, surf, '--step', str(a.step), '--order', a.order, '--charts'], check=True)
else:
    surf = src
if not os.path.exists(f'{surf}/charts.tif'):
    sys.exit(f'{surf}/charts.tif is missing: convert the OBJ with obj_to_tifxyz.py --charts (seams are chart boundaries)')
X = read_tifxyz(surf)[0]; ch = tifffile.imread(f'{surf}/charts.tif').astype(np.int64)
ok = np.isfinite(X).all(-1); n = normals(X, smooth=2.0); H, W = ok.shape
log(f'{surf}: {H} x {W} grid, {ok.sum()} nodes, {len(np.unique(ch[ok]))} charts')

rows = []; cls_map = np.zeros((H, W), np.uint8)                      # 1 surface, 2 tear, 3 switch
cls_map[ok] = 1
counts = dict(switch=0, tear=0)
for ax in (0, 1):
    sa = [slice(None)] * 2; sb = [slice(None)] * 2; sa[ax] = slice(0, -1); sb[ax] = slice(1, None); sa, sb = tuple(sa), tuple(sb)
    m = ok[sa] & ok[sb] & (ch[sa] != ch[sb])
    d = ((X[sb] - X[sa]) * (0.5 * (n[sa] + n[sb]))).sum(-1); g = np.linalg.norm(X[sb] - X[sa], axis=-1)
    sel = m & np.isfinite(d) & (np.abs(d) > a.thr)
    r, c = np.nonzero(sel)
    for i, j in zip(r, c):
        r2, c2 = i + (ax == 0), j + (ax == 1); dd, gg = float(d[i, j]), float(g[i, j])
        k = 'switch' if abs(dd) <= a.max_offset and gg <= a.max_gap else 'tear'
        counts[k] += 1; v = 3 if k == 'switch' else 2
        cls_map[i, j] = max(cls_map[i, j], v); cls_map[r2, c2] = max(cls_map[r2, c2], v)
        x, y, z = X[i, j]
        rows.append((int(i), int(j), int(r2), int(c2), round(float(x), 1), round(float(y), 1), round(float(z), 1), round(dd, 2), round(gg, 2), k))
with open(os.path.join(a.out, 'seams.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['row', 'col', 'row2', 'col2', 'x', 'y', 'z', 'offset_vx', 'gap_vx', 'class']); w.writerows(rows)
log(f'seam pairs: {counts["switch"]} switch, {counts["tear"]} tear -> {a.out}/seams.csv')

f = int(np.ceil(W / 8000))                                           # max-pool columns so seams stay visible
Wp = int(np.ceil(W / f)); M = np.zeros((H, Wp * f), np.uint8); M[:, :W] = cls_map; M = M.reshape(H, Wp, f).max(-1)
lut = np.array([[0, 0, 0], [110, 110, 110], [255, 120, 40], [40, 40, 255]], np.uint8)   # BGR: bg, surface, tear, switch
img = lut[M]
if H < 400: img = cv2.resize(img, (Wp, H * int(np.ceil(400 / H))), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(os.path.join(a.out, 'seam_map.png'), img)

summary = dict(surface=surf, grid=[H, W], nodes=int(ok.sum()), charts=int(len(np.unique(ch[ok]))), seam_pairs=counts,
               nodes_at_switches=int((cls_map == 3).sum()), params=dict(thr=a.thr, max_offset=a.max_offset, max_gap=a.max_gap))
if a.repair:
    from sheetdips.fix import seam_repair_icm
    P, moves = seam_repair_icm(Zarr2(a.repair), X, ch, a.thr, a.max_offset, max_gap=a.max_gap, min_snapped=a.min_snapped, log=log)
    cleanup = None
    if not a.no_cleanup:                                             # flatten the right-angle detours ICM leaves
        from sheetdips.bumps import bump_pass, bump_mask
        _, b0 = bump_mask(P, normals(P, smooth=2.0), thr=a.thr)
        P, r1 = bump_pass(P, 'pieces', max_piece=a.max_piece, thr=a.thr, max_step=a.max_offset, rounds=4, log=log)
        P, r2 = bump_pass(P, 'bumps', thr=a.thr, max_step=a.max_offset, rounds=3, log=log)
        _, b1 = bump_mask(P, normals(P, smooth=2.0), thr=a.thr); log(f'cleanup: bumps {b0} -> {b1}')
        cleanup = dict(max_piece=a.max_piece, bumps_before=b0, bumps_after=b1, pieces=r1, bumps=r2)
    rep = os.path.join(a.out, 'repaired.tifxyz')
    write_tifxyz(rep, P, 1, 'repaired', {'seam_audit': {**vars(a), 'source': os.path.abspath(surf)}}); tifffile.imwrite(f'{rep}/charts.tif', ch.astype(np.int32))
    moved = ok & (np.linalg.norm(np.nan_to_num(P - X), axis=-1) > 2)
    summary['repair'] = dict(pred=a.repair, min_snapped=a.min_snapped, nodes_moved=int(moved.sum()), units_labelled=len(moves),
                          units_moved=sum(1 for m in moves if m['moved']),
                          units_left={w: sum(1 for m in moves if m.get('why') == w) for w in ('seeds', 'not reached', 'snapped', 'drift', 'cycle')},
                          cleanup=cleanup)
    with open(os.path.join(a.out, 'moves.json'), 'w') as f: json.dump(moves, f)
    log(f'repair: {moved.sum()} nodes moved > 2 vx -> {rep}')
json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
log('done')
