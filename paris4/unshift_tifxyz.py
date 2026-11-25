"""Move a tifxyz from ScrollFiesta's shifted world frame back to the Paris 4 reference frame (in place).
reference = world - shift (shift.json written by carve.py, zyx). usage: unshift_tifxyz.py SURF.tifxyz SHIFT.json"""
import json, sys, numpy as np, tifffile
d, sj = sys.argv[1], sys.argv[2]
dz, dy, dx = json.load(open(sj))['shift_zyx']
m = json.load(open(f'{d}/meta.json'))
if m.get('p4_unshifted'):
    sys.exit('already unshifted')
X = {c: tifffile.imread(f'{d}/{c}.tif').astype(np.float32) for c in 'xyz'}
ok = X['z'] > 0
for c, s in (('x', dx), ('y', dy), ('z', dz)):
    X[c][ok] -= s; tifffile.imwrite(f'{d}/{c}.tif', X[c])
if 'bbox' in m:
    m['bbox'] = [[b[0] - dx, b[1] - dy, b[2] - dz] for b in m['bbox']]
m['p4_unshifted'] = {'shift_zyx': [dz, dy, dx]}
json.dump(m, open(f'{d}/meta.json', 'w'), indent=1)
print('unshifted', int(ok.sum()), 'nodes by', (dz, dy, dx))
