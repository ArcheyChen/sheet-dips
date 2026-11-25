"""Sheet following by dip integration (the seismic "horizon from dips" idea: Lomask et al. 2006, Fomel 2010,
Wu & Hale 2015). In a thick sampling V(o, z, u) of a volume along a surface's normals, the sheets are layers; where
the surface crosses them obliquely the layers dip. Local dips are measured by cross-correlating normal profiles of
neighbouring columns, and the offset field o(z, u) that follows one layer through a trusted (anchored) region is the
weighted least-squares integral of the dips. A whole 2-D field is solved at once, so one ambiguous spot cannot
derail everything after it, as in slice-by-slice tracing."""
import numpy as np, cv2
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def xcorr_dips(V, lag=8, win=20, smax=3.0, step=0.25, agg=15.0):
    """Dips of the layers of V (layers, h, w) at its middle layer.
    For each axis (z = rows, u = cols) the normal profile at a pixel (layers c-WIN..c+WIN) is compared with the profile
    LAG px further along the axis, shifted by -SMAX..SMAX layers in sub-layer STEPs (linear interpolation); the
    normalised correlation maps are averaged over a Gaussian of AGG px before taking the best shift (parabolic
    refinement). Returns pz, pu (layer shift per px along z and u) and a weight (geometric mean of the peak
    correlations, clipped at 0)."""
    L = V.shape[0]; c = L // 2
    Vs = ndi.gaussian_filter(V.astype(np.float32), (1.0, 0.7, 0.7))
    o = np.arange(c - win, c + win + 1)
    A0 = Vs[o]; A0 = A0 - A0.mean(0); A0 /= np.sqrt((A0 ** 2).sum(0)) + 1e-6
    shifts = np.arange(-smax, smax + 1e-9, step); res = []
    for axis in (1, 2):
        C = []
        for sh in shifts:
            k = o + sh; k0 = np.floor(k).astype(int); f = (k - k0)[:, None, None]
            B = (1 - f) * Vs[np.clip(k0, 0, L - 1)] + f * Vs[np.clip(k0 + 1, 0, L - 1)]
            B = np.roll(B, -lag, axis=axis); B = B - B.mean(0); B /= np.sqrt((B ** 2).sum(0)) + 1e-6
            C.append(ndi.gaussian_filter((A0 * B).sum(0), agg))
        C = np.stack(C); i = np.clip(C.argmax(0), 1, len(shifts) - 2); ii = np.indices(i.shape)
        cm, c0, cp = C[i - 1, ii[0], ii[1]], C[i, ii[0], ii[1]], C[i + 1, ii[0], ii[1]]
        den = cm - 2 * c0 + cp; frac = np.divide(0.5 * (cm - cp), den, out=np.zeros_like(den), where=np.abs(den) > 1e-6)
        dip = (shifts[i] + np.clip(frac, -1, 1) * step) / lag; peak = C.max(0)
        edge = [slice(None)] * 2; edge[axis - 1] = slice(V.shape[axis] - lag, None); dip[tuple(edge)] = 0; peak[tuple(edge)] = 0
        res.append((dip.astype(np.float32), np.clip(peak, 0, 1).astype(np.float32)))
    (pz, wz), (pu, wu) = res
    return pz, pu, np.sqrt(wz * wu)


def pool(x, f):
    if f == 1: return x
    h, w = x.shape[0] // f * f, x.shape[1] // f * f
    return x[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))


def integrate(pz, pu, w, anchor, valid, coarse=2, beta=100.0, omax=36.0):
    """Offset field o (same grid as pz) with o = 0 on ANCHOR and do/dz = pz, do/du = pu elsewhere, weighted least
    squares (edge weight sqrt(w_i w_j)), solved directly on a grid coarsened by COARSE and upsampled."""
    H, W = pz.shape; f = coarse
    PZ, PU, WW = pool(pz, f) * f, pool(pu, f) * f, pool(w * valid, f); ac = pool(anchor.astype(np.float64), f) > 0.5
    h, ww = PZ.shape; n = h * ww; idx = np.arange(n).reshape(h, ww)
    rows, cols, vals, rhs, wts, e = [], [], [], [], [], 0
    for (i1, i0, b, wt) in ((idx[1:], idx[:-1], 0.5 * (PZ[1:] + PZ[:-1]), np.sqrt(WW[1:] * WW[:-1])),
                            (idx[:, 1:], idx[:, :-1], 0.5 * (PU[:, 1:] + PU[:, :-1]), np.sqrt(WW[:, 1:] * WW[:, :-1]))):
        m = i1.size; rows += [e + np.arange(m)] * 2; cols += [i1.ravel(), i0.ravel()]; vals += [np.ones(m), -np.ones(m)]
        rhs.append(b.ravel()); wts.append(wt.ravel() + 1e-4); e += m
    ka = idx[ac]; m = len(ka)
    rows.append(e + np.arange(m)); cols.append(ka); vals.append(np.ones(m)); rhs.append(np.zeros(m)); wts.append(np.full(m, beta)); e += m
    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(e, n)).tocsr()
    AtW = A.T.multiply(np.concatenate(wts))
    M = (AtW @ A).tocsc() + 1e-6 * coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)).tocsc()
    x = spsolve(M, AtW @ np.concatenate(rhs)).reshape(h, ww)
    x = cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) if f > 1 else x
    return np.clip(x, -omax, omax)


def anchor_from_map(on, ok, margin=6):
    """The main sheet: the largest connected set of on-sheet cells farther than MARGIN cells from any off-sheet
    cell (cuts the thin bridges into islands, i.e. patches where the surface sits on another sheet)."""
    core = on & ok & (ndi.distance_transform_edt(on & ok) > margin)
    lab, nl = ndi.label(core)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if nl else core
