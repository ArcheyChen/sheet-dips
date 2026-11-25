"""Where does a surface leave its sheet? The surface prediction is sampled along the surface normals (offsets -R..R);
a canvas cell is on-sheet if a predicted sheet line passes within +-1 voxel of the surface. Patches where the surface
has moved onto a neighbouring sheet show up as on-sheet islands ringed by off-sheet cells."""
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .io import normals, thick_sample


def offsheet_map(pred, P, every=4, r=20, thr=100.0, tol=1, lateral_sigma=1.0):
    """(on, off, ok) on the canvas subsampled by EVERY: on = prediction line within +-TOL vx; off = signed offset of the
    nearest line (vx, NaN if none within R); ok = valid surface cells. The profiles are smoothed with sigma 1 vx along
    the normals and LATERAL_SIGMA cells across (keep it small when cells are far apart, e.g. on a grid-node scan)."""
    n = normals(P)
    prof, ok = thick_sample(pred, P, n, 2 * r + 1, every)
    S = ndi.gaussian_filter(prof.astype(np.float32), (1.0, lateral_sigma, lateral_sigma)); c = r
    on = (S[c - tol:c + tol + 1].max(0) > thr) & ok
    peak = (S > thr) & (S >= np.roll(S, 1, 0)) & (S >= np.roll(S, -1, 0)); peak[0] = peak[-1] = False
    offs = np.arange(-r, r + 1); dist = np.where(peak, np.abs(offs)[:, None, None], 999)
    has = dist.min(0) < 999; off = np.where(has, offs[dist.argmin(0)], np.nan).astype(np.float32); off[on] = 0
    return on, off, ok


def map_image(on, off, ok, r=20, scale=2):
    """green = on-sheet; red / blue = nearest line in front / behind (brighter = farther); grey = no line within R."""
    img = np.full(ok.shape + (3,), 70, np.uint8); f = np.clip(np.abs(np.nan_to_num(off)) / r, 0, 1)
    m = ok & ~on & np.isfinite(off)
    img[m & (np.nan_to_num(off) > 0)] = np.stack([40 + 0 * f, 40 + 0 * f, 120 + 135 * f], -1)[m & (np.nan_to_num(off) > 0)].astype(np.uint8)
    img[m & (np.nan_to_num(off) < 0)] = np.stack([120 + 135 * f, 60 + 0 * f, 40 + 0 * f], -1)[m & (np.nan_to_num(off) < 0)].astype(np.uint8)
    img[on] = (60, 190, 60); img[~ok] = 0
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def jump_islands(off, ok, jump=3.0, min_cells=5):
    """Sheet identity by continuity: neighbouring cells stay on the same sheet while the offset of their nearest
    predicted line changes by <= JUMP vx; a switch shows as a jump of about half the sheet spacing where the surface
    passes between sheets. Unlike the +-1 vx on-sheet test this tolerates a surface that runs a few voxels off its
    line, and unlike a wider tolerance it does not merge tightly packed sheets.
    Returns (labels (-1 = no line within R), main label, [(cells, label), ...] of the other components >= MIN_CELLS)."""
    v = ok & np.isfinite(off); H, W = off.shape; idx = np.arange(H * W).reshape(H, W); i, j = [], []
    for a, b in (((slice(None), slice(0, -1)), (slice(None), slice(1, None))), ((slice(0, -1), slice(None)), (slice(1, None), slice(None)))):
        m = v[a] & v[b] & (np.abs(off[a] - off[b]) <= jump); i.append(idx[a][m]); j.append(idx[b][m])
    i, j = np.concatenate(i), np.concatenate(j)
    n, lab = connected_components(coo_matrix((np.ones(i.size), (i, j)), shape=(H * W, H * W)), directed=False)
    lab = lab.reshape(H, W); lab[~v] = -1
    sz = np.bincount(lab[v], minlength=n); main = int(sz.argmax()) if v.any() else -1
    isl = sorted(((int(sz[k]), k) for k in range(n) if k != main and sz[k] >= min_cells), reverse=True)
    return lab, main, isl
