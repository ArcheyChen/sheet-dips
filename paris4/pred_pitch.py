"""Radial spacing of predicted sheets in a carved grid (no reference data used).

Rays from the umbilicus (axis table, shifted frame) at 72 angles in a few z slices; along each ray count rising
edges of the binary prediction; spacing = radial length / edges, per radial bin. Compare with ScrollFiesta's
auto pitch. usage: pred_pitch.py GRID AXIS.csv z1 [z2 ...]
"""
import sys, glob, re, numpy as np, tifffile
grid, axis = sys.argv[1], sys.argv[2]; zs = [int(v) for v in sys.argv[3:]]
A = np.array([[float(x) for x in l.split(',')[:3]] for l in open(axis) if l[0].isdigit()])
cubes = {tuple(map(int, re.findall(r'\d+', f.split('/')[-1]))): f for f in glob.glob(f'{grid}/cubes_PRED/*.tif')}
Z0 = sorted({k[0] for k in cubes}); Y = sorted({k[1] for k in cubes}); X = sorted({k[2] for k in cubes})
bins = [(50, 300), (300, 600), (600, 900), (900, 1200), (1200, 1500)]
for z in zs:
    cz = max(v for v in Z0 if v <= z); k = z - cz
    S = np.zeros((len(Y) * 128, len(X) * 128), np.uint8)
    for i, y in enumerate(Y):
        for j, x in enumerate(X):
            f = cubes.get((cz, y, x))
            if f:
                S[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = tifffile.imread(f, key=k) > 0
    uy, ux = np.interp(z, A[:, 0], A[:, 1]) - Y[0], np.interp(z, A[:, 0], A[:, 2]) - X[0]
    res = {b: [0.0, 0] for b in bins}
    for th in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        r = np.arange(50, 1500, 0.5); yy = uy + r * np.sin(th); xx = ux + r * np.cos(th)
        ok = (yy >= 0) & (yy < S.shape[0] - 1) & (xx >= 0) & (xx < S.shape[1] - 1)
        v = S[yy[ok].round().astype(int), xx[ok].round().astype(int)]; rr = r[ok]
        if len(v) < 10:
            continue
        rise = rr[1:][(v[1:] == 1) & (v[:-1] == 0)]
        for b in bins:
            m = (rr >= b[0]) & (rr < b[1])
            if m.sum() > 100:
                res[b][0] += (rr[m].max() - rr[m].min()); res[b][1] += ((rise >= b[0]) & (rise < b[1])).sum()
    print(f'z {z} umbilicus (shifted) y {uy + Y[0]:.0f} x {ux + X[0]:.0f}: ' +
          ', '.join(f'r{b[0]}-{b[1]}: {res[b][0] / max(res[b][1], 1):.2f} vx/edge (n={res[b][1]})' for b in bins))
