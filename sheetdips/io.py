"""Data access and surface geometry: chunked zarr v2 arrays (local or over HTTPS with a disk cache), tifxyz meshes,
canvas resampling and surface normals."""
import json, os, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, tifffile
from scipy import ndimage as ndi
import numcodecs, requests

OPEN_DATA = 'https://data.aws.ash2txt.org/'


class Zarr2:
    """Read-only access to one level of a zarr v2 array (uint8, 'C' order), chunk by chunk.
    SRC is a local directory holding .zarray and the chunk files, or an open-data key / https URL of that level;
    remote chunks are cached under CACHE (default $SHEETDIPS_CACHE or ~/.cache/sheetdips, then <key>)."""

    def __init__(self, src, cache=None, threads=32, keep=150):
        if os.path.isdir(src):
            self.local, self.url = src, None
        else:
            url = src if src.startswith('http') else OPEN_DATA + src.strip('/')
            self.url = url.rstrip('/')
            key = self.url.split('://', 1)[1].split('/', 1)[1]
            root = os.environ.get('SHEETDIPS_CACHE', os.path.expanduser('~/.cache/sheetdips'))
            self.local = cache or os.path.join(root, key)
            os.makedirs(self.local, exist_ok=True)
        self.sess = requests.Session(); self.sess.mount('https://', requests.adapters.HTTPAdapter(pool_maxsize=threads))
        self.threads = threads
        meta = self._get('.zarray'); z = json.loads(meta)
        self.shape, self.chunks = z['shape'], z['chunks']; self.sep = z.get('dimension_separator', '.')
        self.codec = numcodecs.get_codec(z['compressor']) if z['compressor'] else None
        self.cs = self.chunks[0]; self._mem = {}; self._lock = threading.Lock(); self.keep = keep   # decoded chunks kept in RAM

    def _get(self, name):
        p = os.path.join(self.local, name)
        if os.path.exists(p): return open(p, 'rb').read()
        if os.path.exists(p + '.absent') or self.url is None: return None
        for _ in range(4):
            try:
                r = self.sess.get(f'{self.url}/{name}', timeout=60)
                if r.status_code in (403, 404):
                    os.makedirs(os.path.dirname(p), exist_ok=True); open(p + '.absent', 'w').close(); return None
                r.raise_for_status(); os.makedirs(os.path.dirname(p), exist_ok=True)
                tmp = f'{p}.{os.getpid()}.{threading.get_ident()}.part'; open(tmp, 'wb').write(r.content); os.replace(tmp, p)
                return r.content
            except requests.RequestException:
                continue
        raise RuntimeError(f'cannot fetch {self.url}/{name}')

    def chunk(self, k):
        """chunk (z, y, x) as a (cs, cs, cs) uint8 array, or None where the array is empty there."""
        with self._lock:
            if k in self._mem: return self._mem[k]
        name = self.sep.join(map(str, k)); p = os.path.join(self.local, name)
        b = open(p, 'rb').read() if os.path.exists(p) else self._get(name)
        d = None
        if b is not None:
            b = self.codec.decode(b) if self.codec else b
            if len(b) == self.cs ** 3: d = np.frombuffer(b, np.uint8).reshape(self.cs, self.cs, self.cs)
        with self._lock:
            self._mem[k] = d
            while len(self._mem) > self.keep: self._mem.pop(next(iter(self._mem)))
        return d

    def _download(self, k):
        """make sure chunk k is on disk (remote arrays); nothing is decoded or kept in RAM here."""
        if self.url is None: return
        name = self.sep.join(map(str, k)); p = os.path.join(self.local, name)
        if not os.path.exists(p) and not os.path.exists(p + '.absent'): self._get(name)

    def prefetch(self, keys):
        """download missing chunks in parallel (decoding happens later, one chunk at a time, so memory stays bounded
        however many chunks a call touches)."""
        with ThreadPoolExecutor(self.threads) as ex:
            for _ in ex.map(self._download, keys): pass

    def sample(self, Q):
        """nearest-voxel values at points Q (N, 3 as x, y, z); 0 outside the array or in empty chunks."""
        iz, iy, ix = (np.round(Q[:, 2]).astype(np.int64), np.round(Q[:, 1]).astype(np.int64), np.round(Q[:, 0]).astype(np.int64))
        out = np.zeros(len(Q), np.uint8); cs = self.cs
        ok = (iz >= 0) & (iy >= 0) & (ix >= 0) & (iz < self.shape[0]) & (iy < self.shape[1]) & (ix < self.shape[2])
        idx = np.flatnonzero(ok)
        if not len(idx): return out
        key = (iz[idx] // cs) * 10 ** 8 + (iy[idx] // cs) * 10 ** 4 + ix[idx] // cs
        order = np.argsort(key, kind='stable'); groups = np.split(order, np.flatnonzero(np.diff(key[order])) + 1)
        keys = [(int(key[g[0]] // 10 ** 8), int(key[g[0]] // 10 ** 4 % 10 ** 4), int(key[g[0]] % 10 ** 4)) for g in groups]
        self.prefetch([k for k in keys if k not in self._mem])
        for g, k in zip(groups, keys):
            d = self.chunk(k)
            if d is not None:
                j = idx[g]; out[j] = d[iz[j] % cs, iy[j] % cs, ix[j] % cs]
        return out


class Dense:
    """a box of a Zarr2 array held in memory, with trilinear sampling (for samplers called many times, e.g. tracking)."""
    def __init__(self, arr, x0, x1, y0, y1, z0, z1):
        cs = arr.cs; self.o = np.array([z0 // cs * cs, y0 // cs * cs, x0 // cs * cs])
        ks = [range(a // cs, (b - 1) // cs + 1) for a, b in ((z0, z1), (y0, y1), (x0, x1))]
        self.a = np.zeros((len(ks[0]) * cs, len(ks[1]) * cs, len(ks[2]) * cs), np.uint8)
        keys = [(i, j, k) for i in ks[0] for j in ks[1] for k in ks[2]]; arr.prefetch(keys)
        for (i, j, k) in keys:
            d = arr.chunk((i, j, k))
            if d is not None:
                zi, yi, xi = i * cs - self.o[0], j * cs - self.o[1], k * cs - self.o[2]
                self.a[zi:zi + cs, yi:yi + cs, xi:xi + cs] = d

    def sample(self, Q, order=1):
        """values at points Q (N, 3 as x, y, z), trilinear (ORDER 1) or nearest (0); 0 outside the box."""
        from scipy.ndimage import map_coordinates
        c = np.stack([Q[:, 2] - self.o[0], Q[:, 1] - self.o[1], Q[:, 0] - self.o[2]])
        return map_coordinates(self.a, c, order=order, mode='constant', cval=0.0)


def read_tifxyz(d):
    """(grid points (Hg, Wg, 3) as x, y, z with NaN at invalid nodes, canvas px per grid node, meta dict)."""
    meta = json.load(open(f'{d}/meta.json'))
    X = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    X[~(X[..., 2] > 0)] = np.nan
    return X, 1 / meta['scale'][0], meta


def write_tifxyz(d, P, step, name, extra=None):
    """write canvas points P (H, W, 3) as a tifxyz mesh with one node every STEP canvas px (-1 at invalid nodes)."""
    os.makedirs(d, exist_ok=True)
    g = P[::step, ::step]; bad = ~np.isfinite(g).all(-1)
    for k, c in enumerate('xyz'):
        X = g[..., k].astype(np.float32).copy(); X[bad] = -1; tifffile.imwrite(f'{d}/{c}.tif', X)
    q = g[~bad]
    meta = {'uuid': name, 'name': name, 'type': 'seg', 'format': 'tifxyz', 'scale': [1 / step, 1 / step],
            'bbox': [q.min(0).tolist(), q.max(0).tolist()]}
    if extra: meta.update(extra)
    json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)


def to_canvas(X, gs, rows=None, cols=None):
    """bilinear upsampling of grid points X (Hg, Wg, 3) to canvas px (rows / cols ranges in canvas px), NaN where any of
    the four surrounding nodes is invalid."""
    Hc, Wc = int((X.shape[0] - 1) * gs) + 1, int((X.shape[1] - 1) * gs) + 1
    (r0, r1), (c0, c1) = rows or (0, Hc), cols or (0, Wc)
    rr = (np.arange(r0, min(r1, Hc)) / gs).astype(np.float32); cc = (np.arange(c0, min(c1, Wc)) / gs).astype(np.float32)
    mx, my = np.meshgrid(cc, rr)
    ok = np.isfinite(X).all(-1)
    Xf = np.where(ok[..., None], X, 0).astype(np.float32)
    P = np.stack([cv2.remap(Xf[..., k], mx, my, cv2.INTER_LINEAR) for k in range(3)], -1).astype(np.float64)
    v = cv2.remap(ok.astype(np.float32), mx, my, cv2.INTER_LINEAR) > 0.999
    P[~v] = np.nan
    return P


def normals(P, smooth=6.0):
    """unit surface normals of canvas points P (H, W, 3): cross product of the column and row tangents, Gaussian-
    smoothed over SMOOTH px. Segment meshes upsampled from coarse grids are piecewise bilinear: their raw normals jump
    at the grid lines, and moving the surface along them tears it there, so the normals are smoothed. Invalid (NaN)
    points are filled from their nearest valid neighbour first, so a hole does not spread NaN normals along its row
    and column; the normals stay NaN at the holes only."""
    bad = ~np.isfinite(P).all(-1)
    if bad.any() and not bad.all():
        _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True); Q = P[iy, ix]
    else:
        Q = P
    du = np.gradient(Q, axis=1); dz = np.gradient(Q, axis=0)
    n = np.cross(du, dz); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    if smooth: n = ndi.gaussian_filter(n, (smooth, smooth, 0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n[bad] = np.nan; return n


def thick_sample(arr, P, n, layers, every=1):
    """values of ARR (Zarr2) at P + (k - layers // 2) * n for k = 0..layers-1 on the canvas subsampled by EVERY:
    (layers, H / every, W / every) uint8."""
    Ps, ns = P[::every, ::every], n[::every, ::every]; h, w = Ps.shape[:2]
    ok = np.isfinite(Ps).all(-1) & np.isfinite(ns).all(-1); flat = np.flatnonzero(ok.ravel())
    out = np.zeros((layers, h, w), np.uint8); c = layers // 2
    Pf, nf = Ps.reshape(-1, 3)[flat], ns.reshape(-1, 3)[flat]
    for k in range(layers):
        v = out[k].reshape(-1); v[flat] = arr.sample(Pf + (k - c) * nf); out[k] = v.reshape(h, w)
    return out, ok
