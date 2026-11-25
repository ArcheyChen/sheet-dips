#!/usr/bin/env python3
"""Survey segments for sheet switches: for each tifxyz, the on-sheet map on its grid nodes (cheap: one sample per
node), and the islands, i.e. on-sheet regions cut off from the main sheet by off-sheet rings.
usage: scan_segments.py PRED OUT_DIR SURFACE.tifxyz [SURFACE.tifxyz ...] [--every 1] [--margin 3] [--min-island 30]
                        [--coord-scale 1] [--tol 1] [--lateral-sigma 1]
  SURFACE may also be an open-data key of a tifxyz directory (fetched to the cache), e.g.
  PHerc0139/segments/20250108000000-w025_2025010863/mesh/20250108000000-on-20250728140407-9.362um.tifxyz
  --coord-scale maps the surface into PRED's voxels when PRED is a coarser pyramid level of the surface's volume
  (e.g. 0.25 for level 2).
Writes OUT/<name>_onsheet.png and OUT/<name>_islands.npz per surface and OUT/scan.json (one row per surface:
grid nodes, on-sheet / off-sheet shares, main-sheet share, islands with their node counts and grid bboxes, and the
jump islands of offsheet.jump_islands, the better switch test: the +-1 vx islands also fire where a surface runs a
few voxels off its predicted line, e.g. a recto-face prediction)."""
import argparse, json, os, sys, time
import numpy as np, cv2, requests
from scipy import ndimage as ndi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sheetdips.io import Zarr2, read_tifxyz, OPEN_DATA
from sheetdips.offsheet import offsheet_map, map_image, jump_islands

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('pred'); ap.add_argument('out'); ap.add_argument('surfaces', nargs='+')
ap.add_argument('--every', type=int, default=1, help='grid nodes per map cell'); ap.add_argument('--margin', type=float, default=3)
ap.add_argument('--min-island', type=int, default=30, help='islands smaller than this many cells are ignored')
ap.add_argument('--coord-scale', type=float, default=1.0, help='multiply surface coordinates by this (PRED at a coarser level)')
ap.add_argument('--tol', type=int, default=1, help='on-sheet if a predicted line is within +-TOL vx')
ap.add_argument('--lateral-sigma', type=float, default=1.0, help='profile smoothing across cells (cells)')
ap.add_argument('--jump', type=float, default=3.0, help='jump islands: neighbours whose nearest-line offsets differ by more are cut apart')
ap.add_argument('--cache', default=os.environ.get('SHEETDIPS_CACHE', os.path.expanduser('~/.cache/sheetdips')))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); t0 = time.time(); pred = Zarr2(a.pred)


def fetch_tifxyz(key):
    if os.path.isdir(key): return key
    d = os.path.join(a.cache, key.strip('/'))
    os.makedirs(d, exist_ok=True)
    for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
        p = os.path.join(d, f)
        if not os.path.exists(p):
            r = requests.get(OPEN_DATA + key.strip('/') + '/' + f, timeout=120); r.raise_for_status(); open(p, 'wb').write(r.content)
    return d


rows = []
out_json = os.path.join(a.out, 'scan.json')
if os.path.exists(out_json): rows = json.load(open(out_json))
done = {r['surface'] for r in rows}
for s in a.surfaces:
    if s in done: continue
    try:
        X, gs, meta = read_tifxyz(fetch_tifxyz(s))
    except Exception as e:
        print(f'{s}: cannot read ({e})', flush=True); continue
    P = X[::a.every, ::a.every] * a.coord_scale; Hc, Wc = P.shape[:2]
    on = np.zeros((Hc, Wc), bool); ok = np.zeros((Hc, Wc), bool); off = np.full((Hc, Wc), np.nan, np.float32)
    T, ov = 384, 16                                     # tiles of the grid, with an overlap for the smoothed normals
    for r0 in range(0, Hc, T):
        for c0 in range(0, Wc, T):
            ra, rb, ca, cb = max(0, r0 - ov), min(Hc, r0 + T + ov), max(0, c0 - ov), min(Wc, c0 + T + ov)
            Pt = P[ra:rb, ca:cb]
            if not np.isfinite(Pt).all(-1).any(): continue
            o_, f_, k_ = offsheet_map(pred, Pt, every=1, tol=a.tol, lateral_sigma=a.lateral_sigma)
            sr, sc = slice(r0 - ra, r0 - ra + min(T, Hc - r0)), slice(c0 - ca, c0 - ca + min(T, Wc - c0))
            on[r0:r0 + T, c0:c0 + T] = o_[sr, sc]; off[r0:r0 + T, c0:c0 + T] = f_[sr, sc]; ok[r0:r0 + T, c0:c0 + T] = k_[sr, sc]
    offs = ok & ~on & np.isfinite(off)
    core = on & (ndi.distance_transform_edt(on) > a.margin)
    lab, n = ndi.label(core); sizes = np.bincount(lab.ravel()); sizes[0] = 0
    main = sizes.argmax() if n else 0
    # islands: on-sheet components (of the full on-sheet mask) that do not contain the main core
    lab_on, n_on = ndi.label(on)
    main_on = np.unique(lab_on[lab == main]) if n else []
    isl = []
    for i, sl in enumerate(ndi.find_objects(lab_on)):
        k = i + 1
        if k in main_on: continue
        m = lab_on[sl] == k; c = int(m.sum())
        if c < a.min_island: continue
        isl.append(dict(cells=c, rows=[sl[0].start, sl[0].stop], cols=[sl[1].start, sl[1].stop]))
    isl.sort(key=lambda d: -d['cells'])
    jl, jm, ji = jump_islands(off, ok, a.jump, 5); jv = jl >= 0
    jisl = [dict(cells=c, rows=[int(np.nonzero(jl == k)[0].min()), int(np.nonzero(jl == k)[0].max()) + 1],
                 cols=[int(np.nonzero(jl == k)[1].min()), int(np.nonzero(jl == k)[1].max()) + 1]) for c, k in ji[:20]]
    name = os.path.basename(s.rstrip('/'))
    cv2.imwrite(os.path.join(a.out, f'{name}_onsheet.png'), map_image(on, off, ok, scale=1))
    np.savez_compressed(os.path.join(a.out, f'{name}_islands.npz'), on=on, off=off, ok=ok, lab_on=lab_on, main=np.isin(lab_on, main_on),
                        jump_lab=jl, jump_main=jm)
    row = dict(surface=s, name=name, grid=list(X.shape[:2]), canvas_px_per_cell=gs * a.every, cells=int(ok.sum()),
               onsheet=float(on[ok].mean()) if ok.any() else None, offsheet=float(offs[ok].mean()) if ok.any() else None,
               main_share=float(np.isin(lab_on, main_on)[ok].mean()) if ok.any() else None,
               island_cells=int(sum(d['cells'] for d in isl)), islands=isl[:20],
               jump=a.jump, jump_main_share=float((jl == jm)[jv].mean()) if jv.any() else None,
               jump_island_cells=int(sum(c for c, _ in ji)), jump_islands=jisl)
    rows.append(row); json.dump(rows, open(out_json, 'w'), indent=1)
    print(f"[{time.time() - t0:6.0f} s] {name}: {row['cells']} cells, on-sheet {row['onsheet'] * 100:.1f} %, off-sheet {row['offsheet'] * 100:.1f} %, "
          f"{len(isl)} islands ({row['island_cells']} cells, {row['island_cells'] / max(row['cells'], 1) * 100:.1f} %); "
          f"jump > {a.jump:g} vx: main {(row['jump_main_share'] or 0) * 100:.1f} %, {len(ji)} islands", flush=True)
print(f'wrote {out_json}')
