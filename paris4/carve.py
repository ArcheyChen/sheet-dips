"""Carve a ScrollFiesta cube grid (cubes_PRED/, cubes_RAW/) for PHerc. Paris 4 from the public OME-Zarrs.

Frame: the Paris 4 reference frame = m7 surface prediction level 0 = RAW 2.4 um volume level 2 (9.6 um), the
frame of the spiral dataset (umbilicus.json, winding_inference). Cubes are written under SHIFTED names
(origin + shift), so that ScrollFiesta's hard-coded PHerc0139 umbilicus (y 3405, x 2878) lies near the Paris 4
umbilicus. shift.json records the shift: reference = ScrollFiesta world - shift.

usage: carve.py OUT z0 z1 y0 y1 x0 x1 [--shift dz dy dx] [--threads N]
"""
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, requests, tifffile
from numcodecs import Blosc

HOST = 'https://data.aws.ash2txt.org'
PRED = 'PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0'
RAW = 'PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2'
CUBE = 128
sess = requests.Session()
sess.mount('https://', requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


def meta(path):
    return sess.get(f'{HOST}/{path}/.zarray', timeout=60).json()


def get_chunk(path, m, idx):
    url = f'{HOST}/{path}/' + '/'.join(map(str, idx))
    for attempt in range(6):
        try:
            r = sess.get(url, timeout=120)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            b = r.content
            if m['compressor'] is not None:
                b = Blosc().decode(b)
            return np.frombuffer(b, np.uint8).reshape(m['chunks'])
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(2 + 3 * attempt)


def read_box(path, box, threads):
    m = meta(path); c = m['chunks']; sh = m['shape']
    (z0, z1), (y0, y1), (x0, x1) = box
    out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
    idx = [(a, b, d) for a in range(z0 // c[0], (z1 - 1) // c[0] + 1)
           for b in range(y0 // c[1], (y1 - 1) // c[1] + 1) for d in range(x0 // c[2], (x1 - 1) // c[2] + 1)]

    def one(i):
        a, b, d = i
        ch = get_chunk(path, m, i)
        if ch is None:
            return
        cz, cy, cx = a * c[0], b * c[1], d * c[2]
        sz = slice(max(z0, cz), min(z1, cz + c[0])); sy = slice(max(y0, cy), min(y1, cy + c[1]))
        sx = slice(max(x0, cx), min(x1, cx + c[2]))
        out[sz.start - z0:sz.stop - z0, sy.start - y0:sy.stop - y0, sx.start - x0:sx.stop - x0] = \
            ch[sz.start - cz:sz.stop - cz, sy.start - cy:sy.stop - cy, sx.start - cx:sx.stop - cx]

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, idx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('bbox', nargs=6, type=int)
    ap.add_argument('--shift', nargs=3, type=int, default=[0, 0, 0])
    ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--pred', default=PRED, help='open-data key of the prediction level (default: Paris 4 m7)')
    ap.add_argument('--raw', default=RAW, help='open-data key of the RAW level in the same frame')
    a = ap.parse_args()
    z0, z1, y0, y1, x0, x1 = a.bbox
    assert all(v % CUBE == 0 for v in a.bbox + a.shift)
    (a.out / 'cubes_PRED').mkdir(parents=True, exist_ok=True); (a.out / 'cubes_RAW').mkdir(parents=True, exist_ok=True)
    dz, dy, dx = a.shift
    n_pred = n_raw = n_empty = 0; vals = set()
    for bz in range(z0, z1, CUBE):
        t = time.time()
        box = ((bz, bz + CUBE), (y0, y1), (x0, x1))
        P = read_box(a.pred, box, a.threads); Rw = read_box(a.raw, box, a.threads)
        vals |= set(np.unique(P[::4, ::4, ::4]).tolist())
        for by in range(y0, y1, CUBE):
            for bx in range(x0, x1, CUBE):
                name = f'z{bz + dz:05d}_y{by + dy:05d}_x{bx + dx:05d}.tif'
                sl = (slice(None), slice(by - y0, by - y0 + CUBE), slice(bx - x0, bx - x0 + CUBE))
                p = P[sl]
                tifffile.imwrite(a.out / 'cubes_RAW' / name, np.ascontiguousarray(Rw[sl])); n_raw += 1
                if p.any():
                    tifffile.imwrite(a.out / 'cubes_PRED' / name, np.ascontiguousarray((p != 0).astype(np.uint8) * 255)); n_pred += 1
                else:
                    n_empty += 1
        print(f'z {bz}: {time.time() - t:.0f} s, pred {n_pred} empty {n_empty} raw {n_raw}', flush=True)
    json.dump(dict(bbox_reference=a.bbox, shift_zyx=a.shift, pred=a.pred, raw=a.raw, pred_values_seen=sorted(vals),
                   note='reference = ScrollFiesta world - shift'), open(a.out / 'shift.json', 'w'), indent=1)
    print('done', n_pred, n_empty, n_raw, 'pred values', sorted(vals))


if __name__ == '__main__':
    main()
