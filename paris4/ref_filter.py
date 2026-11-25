"""Keep only reference crossings in cubes where they sit on the m7 prediction clearly better than chance
(ref_quality.py: on_pred - rnd_on_pred >= MIN_GAIN). Uses the reference and the prediction only.
usage: ref_filter.py QUALITY.json CROSS.npz OUT.npz [--min-gain 0.15]"""
import argparse, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('q'); ap.add_argument('cross'); ap.add_argument('out')
ap.add_argument('--min-gain', type=float, default=0.15); a = ap.parse_args()
good = {tuple(r['origin_ref']) for r in json.load(open(a.q)) if r['d_med'] is not None and r['on_pred'] - r['rnd_on_pred'] >= a.min_gain}
C = dict(np.load(a.cross)); key = [tuple(v) for v in (np.floor(C['P']) // 128 * 128).astype(int)]
m = np.array([k in good for k in key])
np.savez(a.out, **{k: v[m] for k, v in C.items()}, min_gain=a.min_gain)
print('cubes', len(good), 'crossings kept', int(m.sum()), 'of', len(m))
