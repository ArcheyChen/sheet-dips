"""Row-by-row re-trace of a flattened surface whose grid rows are (near) constant-z curves, as in ScrollFiesta atlases
(v = z): in each row the surface is a curve in one CT cross-section, and the re-trace makes it follow a single sheet
there, the way an annotator checks a trace in VC3D's slice views.
Each node moves along the row's in-slice normal; the offsets of a whole row are chosen together by dynamic
programming: the data term rewards a predicted sheet line at the offset, the smoothness term limits the change between
neighbouring nodes, so the row keeps to one line instead of dropping into the gap between wraps or hopping between
the plies of one delaminated sheet. Fold-backs (the row running back over itself) are cut out and the row is
resampled at its node spacing along the traced curve."""
import numpy as np
from scipy import ndimage as ndi


def row_normals(R):
    """in-slice (xy) unit normals of a row of points R (n, 3), from the smoothed tangent; NaN where invalid."""
    ok = np.isfinite(R).all(1); T = np.full(R.shape, np.nan)
    if ok.sum() < 3: return T
    idx = np.flatnonzero(ok); Q = R[idx]
    Qs = ndi.gaussian_filter1d(Q, 2.0, axis=0, mode='nearest'); t = np.gradient(Qs, axis=0)
    t[:, 2] = 0; t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    n = np.stack([-t[:, 1], t[:, 0], np.zeros(len(t))], 1); T[idx] = n
    return T


def dp_row(prof, offs, max_step, lam):
    """offsets (index into OFFS) for one row: maximise sum prof[i, k_i] - lam * |offs[k_i] - offs[k_(i-1)]|,
    |offs change| <= MAX_STEP between neighbours. PROF (n, K), no NaN."""
    n, K = prof.shape; d = np.abs(offs[:, None] - offs[None, :]); allowed = d <= max_step + 1e-9
    pen = np.where(allowed, -lam * d, -np.inf)
    S = prof[0].copy(); back = np.zeros((n, K), np.int32)
    for i in range(1, n):
        M = S[None, :] + pen                                   # M[k, j]: from state j to state k
        back[i] = M.argmax(1); S = M.max(1) + prof[i]
    k = np.zeros(n, np.int32); k[-1] = int(S.argmax())
    for i in range(n - 1, 0, -1): k[i - 1] = back[i, k[i]]
    return k


def unloop_row(R, step):
    """drop nodes where the row runs back over itself (direction reversal) and resample the remaining curve at STEP vx
    along it, keeping the number of nodes and the first valid node's position; R (n, 3) with NaN gaps kept as gaps."""
    out = np.full(R.shape, np.nan); ok = np.isfinite(R).all(1)
    runs = np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1) if ok.any() else []
    for run in runs:
        if len(run) < 3: out[run] = R[run]; continue
        Q = R[run]; keep = [0]
        for i in range(1, len(Q)):
            v = Q[i] - Q[keep[-1]]
            if len(keep) >= 2:
                u = Q[keep[-1]] - Q[keep[-2]]
                if np.dot(u[:2], v[:2]) < 0: continue           # backwards along the curve: a fold-back
            if np.linalg.norm(v) < 0.5: continue
            keep.append(i)
        C = Q[keep]; s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        target = np.arange(len(Q)) * step
        m = target <= s[-1] + 1e-6
        res = np.stack([np.interp(target[m], s, C[:, j]) for j in range(3)], 1)
        o = np.full((len(Q), 3), np.nan); o[m] = res; out[run] = o
    return out


def retrace_rows(pred, P, search=6.0, st=0.25, max_step=1.0, lam=0.5, sigma=1.0, unloop=True, step=None, log=print):
    """Re-trace every row of the grid P (H, W, 3) on the prediction PRED (Zarr2). Returns the new points and per-row
    stats (median |offset|)."""
    H, W = P.shape[:2]; offs = np.arange(-search, search + 1e-9, st); out = P.copy(); stats = []
    if step is None:
        d = np.linalg.norm(np.diff(P, axis=1), axis=-1); step = float(np.nanmedian(d))
    for r in range(H):
        R = P[r]; n = row_normals(R); ok = np.isfinite(R).all(1) & np.isfinite(n).all(1)
        if ok.sum() < 3: continue
        runs = np.split(np.flatnonzero(ok), np.flatnonzero(np.diff(np.flatnonzero(ok)) > 1) + 1)
        for run in runs:
            if len(run) < 3: continue
            Q = R[run]; N = n[run]
            pts = Q[:, None, :] + offs[None, :, None] * N[:, None, :]
            prof = pred.sample(pts.reshape(-1, 3)).reshape(len(run), len(offs)).astype(np.float32) / 255.0
            prof = ndi.gaussian_filter1d(prof, sigma / st, axis=1)
            k = dp_row(prof, offs, max_step, lam)
            out[r, run] = Q + offs[k][:, None] * N
            stats.append(float(np.median(np.abs(offs[k]))))
        if unloop: out[r] = unloop_row(out[r], step)
        if r % 20 == 0: log(f'row {r}/{H}')
    return out, stats


def track_row(vol, R, a, search=2.0, st=0.25, momentum=0.6, thr=0.35, max_coast=3, sigma=1.0):
    """Trace one row in its cross-section from node A along the sheet both ways: each step goes one node spacing (taken
    from the input row, clipped to 2-6 vx) along the current direction, then moves to the strongest predicted line
    within +-SEARCH vx along the in-slice normal (a small pull towards 0 keeps it on the nearest of two close lines);
    the direction follows the path with MOMENTUM. Where no line above THR is found for more than MAX_COAST steps the
    trace stops and the remaining nodes are left out (NaN). VOL: a sampler with .sample(Q) (values 0-255)."""
    n = len(R); out = np.full((n, 3), np.nan); out[a] = R[a]
    offs = np.arange(-search, search + 1e-9, st)
    for sgn in (1, -1):
        rng = range(a + 1, n) if sgn > 0 else range(a - 1, -1, -1)
        j = [i for i in (a + sgn, a + 2 * sgn, a + 3 * sgn) if 0 <= i < n and np.isfinite(R[i]).all()]
        if not j: continue
        t = R[j[-1]] - R[a]; t[2] = 0; t /= np.linalg.norm(t) + 1e-9
        p = R[a].copy(); coast = 0; prev = a
        for i in rng:
            if not np.isfinite(R[i]).all(): break
            stp = float(np.clip(np.linalg.norm(R[i] - R[prev]), 2.0, 6.0)); prev = i
            q = p + t * stp; q[2] = R[i, 2]
            nn = np.array([-t[1], t[0], 0.0])
            prof = vol.sample(q[None, :] + offs[:, None] * nn[None, :]).astype(np.float32) / 255.0
            prof = ndi.gaussian_filter1d(prof, sigma / st) - 0.02 * np.abs(offs)
            k = int(prof.argmax())
            if prof[k] + 0.02 * abs(offs[k]) < thr:
                coast += 1
                if coast > max_coast: break
                pn = q
            else:
                coast = 0; pn = q + offs[k] * nn
            d = pn - p; d[2] = 0; dn = np.linalg.norm(d)
            if dn > 1e-6: t = momentum * t + (1 - momentum) * d / dn; t /= np.linalg.norm(t) + 1e-9
            p = pn; out[i] = p
    return out


def _runs(ok):
    i = np.flatnonzero(ok)
    return np.split(i, np.flatnonzero(np.diff(i) > 1) + 1) if len(i) else []


def retrace_propagate(vol, X, lab, sz, r0=None, search=4.0, st=0.25, max_step=0.5, lam=1.0, sigma=1.5,
                      max_pred=6.0, smooth=1.5, log=print):
    """Slice-by-slice re-trace of a grid whose rows are cross-section curves. Row R0 (default: the row with most
    nodes) is traced from anchors (track_row); every next row, up and down, is predicted from the row before it
    (its traced point at each column, moved by the input grid's step between the two rows where that step is under
    MAX_PRED vx, else just to the new row's z), and the predicted row snaps to the predicted sheet line by dynamic
    programming along the row (dp_row: within +-SEARCH vx of the prediction, offset changes <= MAX_STEP between
    neighbours; the prediction is near-binary, so the profile is smoothed by SIGMA vx to peak at the middle of a sheet,
    and the chosen offsets are smoothed by SMOOTH nodes along the row). Columns the row before did not reach are
    traced on from the nearest solved node (track_row); runs
    with no solved node at all start from their anchor (the longest stretch of the largest seam piece LAB / SZ).
    Returns the new grid (NaN where nothing was traced)."""
    H, W = X.shape[:2]; okX = np.isfinite(X).all(-1); out = np.full(X.shape, np.nan)
    offs = np.arange(-search, search + 1e-9, st)
    if r0 is None: r0 = int(okX.sum(1).argmax())

    def anchor_of(run, r):
        L = lab[r, run]; big = L[np.argmax(sz[L])]; m = L == big; best = cur = be = 0
        for i, x in enumerate(m):
            cur = cur + 1 if x else 0
            if cur > best: best, be = cur, i
        return run[be - best // 2]

    def fill(r, T):
        """trace on from solved nodes over the unsolved nodes of each run of row r (in place on T)."""
        for run in _runs(okX[r]):
            solved = np.isfinite(T[run]).all(1)
            if not solved.any():
                if len(run) < 3: continue
                a = anchor_of(run, r); seg = slice(run[0], run[-1] + 1)
                T[seg] = track_row(vol, X[r, seg], a - run[0]); continue
            for gap in _runs(~solved):
                g0, g1 = run[gap[0]], run[gap[-1]]
                if gap[0] > 0:                                         # solved node on the left: trace rightwards
                    a = run[gap[0] - 1]; R = X[r, a:g1 + 1].copy(); R[0] = T[a]
                    T[a:g1 + 1] = np.where(np.isfinite(T[a:g1 + 1]), T[a:g1 + 1], track_row(vol, R, 0))
                else:                                                  # only on the right: trace leftwards
                    a = run[gap[-1] + 1]; R = X[r, g0:a + 1].copy(); R[-1] = T[a]
                    tr = track_row(vol, R, len(R) - 1)
                    T[g0:a + 1] = np.where(np.isfinite(T[g0:a + 1]), T[g0:a + 1], tr)
        return T

    out[r0] = fill(r0, out[r0].copy())
    for rng in (range(r0 + 1, H), range(r0 - 1, -1, -1)):
        prev = r0
        for r in rng:
            Pv = out[prev]; ref = np.full((W, 3), np.nan)
            both = np.isfinite(Pv).all(1) & okX[r]
            stepv = X[r] - X[prev]; small = both & okX[prev] & (np.linalg.norm(stepv, axis=1) < max_pred)
            ref[small] = Pv[small] + stepv[small]
            other = both & ~small; ref[other] = Pv[other]; ref[other, 2] = X[r, other, 2]
            T = np.full((W, 3), np.nan)
            for run in _runs(np.isfinite(ref).all(1)):
                if len(run) < 3: continue
                Q = ref[run]; N = row_normals(Q)
                if not np.isfinite(N).all(): continue
                pts = Q[:, None, :] + offs[None, :, None] * N[:, None, :]
                prof = vol.sample(pts.reshape(-1, 3)).reshape(len(run), len(offs)).astype(np.float32) / 255.0
                prof = ndi.gaussian_filter1d(prof, sigma / st, axis=1) - 0.01 * np.abs(offs)[None, :]
                k = dp_row(prof, offs, max_step, lam)
                o = ndi.gaussian_filter1d(offs[k], smooth, mode='nearest') if smooth > 0 else offs[k]
                T[run] = Q + o[:, None] * N
            out[r] = fill(r, T); prev = r
            if r % 20 == 0: log(f'row {r}/{H}')
    return out


def clean_curve(R, ds=0.5, smooth=2.0):
    """a traced row R (n, 3) as a clean dense polyline: fold-backs (steps against the row's overall direction) are
    dropped, the rest is resampled every DS vx along it and smoothed by SMOOTH vx; one polyline per run of valid
    nodes. Returns a list of (m, 3) arrays."""
    out = []
    for run in _runs(np.isfinite(R).all(1)):
        if len(run) < 3: continue
        Q = R[run]; g = ndi.gaussian_filter1d(Q, 3.0, axis=0, mode='nearest'); dirn = np.gradient(g, axis=0)
        keep = [0]
        for i in range(1, len(Q)):
            if np.dot((Q[i] - Q[keep[-1]])[:2], dirn[i][:2]) > 0.25: keep.append(i)
        C = Q[keep]
        if len(C) < 2: continue
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        t = np.arange(0, s[-1] + 1e-9, ds)
        D = np.stack([np.interp(t, s, C[:, j]) for j in range(3)], 1)
        out.append(ndi.gaussian_filter1d(D, smooth / ds, axis=0, mode='nearest'))
    return out


def project_rows(X, C, reach=10.0, st=0.25, tol=1.0, max_step=1.0, lam=0.5, pull=0.1):
    """move every node of X along its row's in-slice normal onto the cleaned traced curve of its row (C: the traced
    grid); the node keeps its place along the row, so the grid's uv layout is unchanged. Nodes whose normal does not
    meet the curve within +-REACH vx (closest approach above TOL vx) are dropped (NaN). PULL (per vx) favours the
    crossing nearest the node, so nodes already on the right sheet keep their ply."""
    from scipy.spatial import cKDTree
    out = np.full(X.shape, np.nan); offs = np.arange(-reach, reach + 1e-9, st)
    for r in range(X.shape[0]):
        curves = clean_curve(C[r])
        if not curves: continue
        D = np.concatenate(curves); tree = cKDTree(D[:, :2])
        R = X[r]; N = row_normals(R); ok = np.isfinite(R).all(1) & np.isfinite(N).all(1)
        if not ok.any(): continue
        idx = np.flatnonzero(ok); P = R[idx]; Nn = N[idx]
        cand = P[:, None, :2] + offs[None, :, None] * Nn[:, None, :2]
        dist, j = tree.query(cand.reshape(-1, 2)); dist = dist.reshape(len(idx), len(offs)); j = j.reshape(len(idx), len(offs))
        # crossings chosen together along each run (dp_row): where the curve offers two crossings (two plies of one
        # sheet drawn as two lines) the row keeps to one instead of flipping between them node by node
        k = np.zeros(len(idx), np.int32)
        for run in np.split(np.arange(len(idx)), np.flatnonzero(np.diff(idx) > 1) + 1):
            sc = np.where(dist[run] <= tol, -dist[run], -10.0) - pull * np.abs(offs)[None, :]
            k[run] = dp_row(sc.astype(np.float32), offs, max_step, lam)
        good = dist[np.arange(len(idx)), k] <= tol
        z = D[j[np.arange(len(idx)), k], 2]
        newp = P + offs[k][:, None] * Nn; newp[:, 2] = np.where(good, z, np.nan)
        out[r, idx[good]] = newp[good]
    return out


def project_nearest(X, C, reach=25.0, ds=0.5, smooth=2.0):
    """move every node of X to the nearest point of its own traced curve: the cleaned curve (clean_curve) of the traced
    grid C over the same run of valid nodes of the same row, so a node never lands on a curve traced for another stretch
    of the row (which may lie on another wrap). Independent of the input row's normals, which are broken where the
    input jumps or folds back; nodes farther than REACH vx from their curve are dropped (NaN)."""
    from scipy.spatial import cKDTree
    out = np.full(X.shape, np.nan)
    for r in range(X.shape[0]):
        for run in _runs(np.isfinite(X[r]).all(1)):
            curves = clean_curve(C[r, run], ds, smooth)
            if not curves: continue
            D = np.concatenate(curves); tree = cKDTree(D)
            d, j = tree.query(X[r, run], distance_upper_bound=reach)
            good = np.isfinite(d); out[r, run[good]] = D[j[good]]
    return out
