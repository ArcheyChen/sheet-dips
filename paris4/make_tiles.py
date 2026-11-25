"""Split one global ScrollFiesta placement into bounded tiles for atlas_overlap_fix (the canonical runs used
bounded tiles of <= 100 cubes; their tile driver is not public). Each tile dir holds symlinks to the cubes' placed
files and a placed_index.json filtered to those cubes; placement (u, windings) stays the global one.

usage: make_tiles.py PLACED_DIR OUT_ROOT [--ny 5 --nx 5] [--chunk 128]
Writes OUT_ROOT/tile_YY_XX/ and OUT_ROOT/tiles.json (cubes and faces per tile).
"""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument('placed'); ap.add_argument('out')
ap.add_argument('--ny', type=int, default=5); ap.add_argument('--nx', type=int, default=5)
ap.add_argument('--chunk', type=int, default=128)
a = ap.parse_args()
idx = json.load(open(f'{a.placed}/placed_index.json'))
cubes = idx['cubes']
ys = sorted({c['origin'][1] for c in cubes}); xs = sorted({c['origin'][2] for c in cubes})
files = os.listdir(a.placed)
tiles = {}
for c in cubes:
    ty = (c['origin'][1] - ys[0]) // a.chunk // a.ny; tx = (c['origin'][2] - xs[0]) // a.chunk // a.nx
    tiles.setdefault((ty, tx), []).append(c)
summary = {}
for (ty, tx), cs in sorted(tiles.items()):
    d = f'{a.out}/tile_{ty:02d}_{tx:02d}'; os.makedirs(d, exist_ok=True)
    ids = {c['id'] for c in cs}
    for f in files:
        if f[:20] in ids:
            dst = f'{d}/{f}'
            if not os.path.lexists(dst):
                os.symlink(os.path.abspath(f'{a.placed}/{f}'), dst)
    sub = dict(idx); sub['cubes'] = cs; sub['n_cubes'] = len(cs); sub['n_ok'] = sum(c['status'] == 0 for c in cs)
    sub['tile'] = {'ty': ty, 'tx': tx, 'source_placed': os.path.abspath(a.placed)}
    json.dump(sub, open(f'{d}/placed_index.json', 'w'), indent=1)
    summary[f'tile_{ty:02d}_{tx:02d}'] = {'cubes': len(cs), 'faces': int(sum(c.get('nf', 0) for c in cs))}
json.dump(summary, open(f'{a.out}/tiles.json', 'w'), indent=1)
for k, v in summary.items():
    print(k, v)
