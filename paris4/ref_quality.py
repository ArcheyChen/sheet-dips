"""Per-cube quality of the reference crossings, from the reference and the m7 prediction only (no atlas, no seams).

For every 128^3 cube of the carved grid: median distance (vx) from the reference crossings in the cube to the
nearest predicted voxel, the same for uniform random points, and the crossing count. A cube is 'reliable' when
crossings sit on the prediction clearly better than chance. usage: ref_quality.py GRID CROSS.npz OUT.json [--workers N]
"""
import argparse, json, multiprocessing as mp, os, numpy as np, tifffile
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import distance_transform_edt
ap = argparse.ArgumentParser(); ap.add_argument('grid'); ap.add_argument('cross'); ap.add_argument('out')
ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
sh = json.load(open(f'{a.grid}/shift.json')); dz, dy, dx = sh['shift_zyx']; b = sh['bbox_reference']
P = np.load(a.cross)['P']


def one(o):
    z, y, x = o
    f = f'{a.grid}/cubes_PRED/z{z+dz:05d}_y{y+dy:05d}_x{x+dx:05d}.tif'
    lo = np.array(o); m = ((P >= lo + 0.5) & (P < lo + 127.5)).all(1); Q = P[m] - lo
    if not os.path.exists(f) or len(Q) == 0:
        return dict(origin_ref=list(o), n=int(len(Q)), d_med=None, rnd_med=None)
    E = distance_transform_edt(~(tifffile.imread(f) > 0))
    d = E[tuple(np.round(Q).astype(int).T)]
    R = np.random.default_rng(z * 7 + y * 3 + x).random((4000, 3)) * 127; dr = E[tuple(np.round(R).astype(int).T)]
    return dict(origin_ref=list(o), n=int(len(Q)), d_med=float(np.median(d)), d_p75=float(np.percentile(d, 75)),
                rnd_med=float(np.median(dr)), on_pred=float((d <= 1).mean()), rnd_on_pred=float((dr <= 1).mean()))


cubes = [(z, y, x) for z in range(b[0], b[1], 128) for y in range(b[2], b[3], 128) for x in range(b[4], b[5], 128)]
with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('fork')) as ex:
    res = list(ex.map(one, cubes, chunksize=8))
json.dump(res, open(a.out, 'w'))
print('cubes', len(res))
