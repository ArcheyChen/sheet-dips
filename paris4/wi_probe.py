"""Verify the winding_inference download and list reference crossings inside a reference-frame box
(W = absolute winding = seed_winding + crossing_level).
usage: wi_probe.py WI_DIR OUT.npz z0 z1 y0 y1 x0 x1"""
import json, hashlib, sys, numpy as np
wi, out = sys.argv[1], sys.argv[2]; z0, z1, y0, y1, x0, x1 = map(int, sys.argv[3:9])
m = json.load(open(f'{wi}/manifest.json')); bad = 0
for sh in m['shards']:
    for k, a in sh['arrays'].items():
        bad += hashlib.sha256(open(f"{wi}/{sh['name']}/{a['file']}", 'rb').read()).hexdigest() != a['sha256']
print('sha mismatches', bad)
P, L, R, Wabs = [], [], [], []
for s in range(len(m['shards'])):
    d = f'{wi}/shard_{s}'
    o = np.load(f'{d}/ray_origin_zyx.npy'); st = np.load(f'{d}/ray_step_zyx.npy'); t = np.load(f'{d}/crossing_t.npy')
    lv = np.load(f'{d}/crossing_level.npy'); off = np.load(f'{d}/crossing_offsets.npy')
    sw = np.load(f'{d}/seed_winding.npy').astype(np.int64)
    ray = np.repeat(np.arange(len(o)), np.diff(off)); p = o[ray] + t[:, None] * st[ray]
    k = (p[:, 0] >= z0) & (p[:, 0] < z1) & (p[:, 1] >= y0) & (p[:, 1] < y1) & (p[:, 2] >= x0) & (p[:, 2] < x1)
    P.append(p[k]); L.append(lv[k]); R.append(ray[k] + s * 10**7); Wabs.append(sw[ray[k]] + lv[k].astype(np.int64))
    if s == 0:
        print('step norms', np.round(np.linalg.norm(st[:3], axis=1), 3), 'step[0]', st[0], 'origin z', o[:, 0].min(), o[:, 0].max(),
              'levels', lv.min(), lv.max())
P, L, R, Wabs = map(np.concatenate, (P, L, R, Wabs))
print('crossings in box', len(P), 'rays', len(np.unique(R)), 'levels', np.unique(L))
print('distinct z (rounded)', len(np.unique(np.round(P[:, 0]))))
print('absolute windings', Wabs.min(), Wabs.max())
np.savez(out, P=P, L=L, R=R, W=Wabs)   # W = seed_winding + crossing_level (absolute winding)
