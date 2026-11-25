"""The repair loop shared by the scripts: sample the surface prediction along the normals, measure layer dips,
integrate the offset field through the anchor, move the surface; repeat."""
import numpy as np, cv2
from scipy import ndimage as ndi
from .io import normals, thick_sample
from .dips import xcorr_dips, integrate


def repair(pred, P, anchor, passes=3, layers=81, ds=2, log=print, sample_every=1):
    """P: canvas points (H, W, 3); ANCHOR: bool mask (any resolution, resized to the dip grid) of the trusted part,
    which stays in place. SAMPLE_EVERY (a divisor of DS) samples the prediction on every n-th canvas px only, which
    cuts memory by n^2 on large canvases. Returns the moved points and per-pass offset statistics."""
    H, W = P.shape[:2]; hist = []
    for k in range(passes):
        n = normals(P)
        V, okv = thick_sample(pred, P, n, layers, sample_every); f = ds // sample_every
        V = ndi.uniform_filter(V, (3, 1, 1)); V[:, okv] = np.maximum(V[:, okv], 1)
        Vd = ndi.uniform_filter(V.astype(np.float32), (1, f, f))[:, ::f, ::f]; valid = okv[::f, ::f]; del V
        pz, pu, w = xcorr_dips(Vd)
        anc = cv2.resize(anchor.astype(np.uint8), (pz.shape[1], pz.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        o = integrate(pz, pu, w, anc, valid)
        O = cv2.resize(o.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        P = P + O[..., None] * np.nan_to_num(n)                       # holes keep their NaN points
        free = ~anc & valid
        hist.append(dict(pass_=k + 1, offset_median=float(np.median(np.abs(o[free]))), offset_p90=float(np.percentile(np.abs(o[free]), 90))))
        log(f'pass {k + 1}: |offset| outside the anchor median {hist[-1]["offset_median"]:.1f}, p90 {hist[-1]["offset_p90"]:.1f} vx')
    return P, hist


def seam_components(P, n, thr=4.0):
    """Pieces of a surface grid separated by seams: neighbouring valid nodes stay in one piece while their offset
    along the local normal is <= THR vx. In a flattened atlas, pieces placed on different wraps meet at seams with
    offsets of about one sheet spacing. Returns (labels (-1 invalid), sizes, signed offsets across axis-0 and axis-1
    neighbour pairs (NaN where not both valid))."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ok = np.isfinite(P).all(-1) & np.isfinite(n).all(-1); H, W = ok.shape; idx = np.arange(H * W).reshape(H, W)
    I, J, D = [], [], []
    for ax in (0, 1):
        a = [slice(None)] * 2; b = [slice(None)] * 2; a[ax] = slice(0, -1); b[ax] = slice(1, None); a, b = tuple(a), tuple(b)
        d = ((P[b] - P[a]) * (0.5 * (n[a] + n[b]))).sum(-1); m = ok[a] & ok[b]; d = np.where(m, d, np.nan); D.append(d)
        k = m & (np.abs(d) <= thr); I.append(idx[a][k]); J.append(idx[b][k])
    I, J = np.concatenate(I), np.concatenate(J)
    nc, lab = connected_components(coo_matrix((np.ones(I.size), (I, J)), shape=(H * W, H * W)), directed=False)
    lab = lab.reshape(H, W); lab[~ok] = -1
    return lab, np.bincount(lab[ok], minlength=nc), D


def seam_repair(pred, P, thr=4.0, min_edges=3, max_shift=40.0, log=print):
    """Move the pieces of a surface that sit on another wrap than their neighbours back onto the neighbours' sheet.
    The largest piece is the anchor. Pieces are taken in turn, most seam contacts with the fixed set first: each moves
    along its normals by the median offset across its seam with the fixed set, then by the median offset to the nearest
    predicted sheet line, so it lands on that sheet. Returns the moved points and a list of per-piece moves."""
    from .offsheet import offsheet_map
    n = normals(P, smooth=2.0)
    lab, sz, (D0, D1) = seam_components(P, n, thr)
    if sz.size == 0: return P, []
    P = P.copy(); moves = []
    H, W = lab.shape
    # seam pairs: (piece a, piece b, node a, node b, signed offset a->b along the normal) for both axes
    pairs = []
    for ax, D in ((0, D0), (1, D1)):
        r, c = np.nonzero(np.isfinite(D) & (np.abs(D) > thr))
        r2, c2 = (r + 1, c) if ax == 0 else (r, c + 1)
        pairs.append(np.stack([lab[r, c], lab[r2, c2], r, c, r2, c2], 1))
    pairs = np.concatenate(pairs); pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] != pairs[:, 1])]
    # pieces joined by seams form clusters; each cluster is repaired from its largest piece (pieces that touch no other
    # piece across a seam are left alone: nothing says which wrap they belong to)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nc_all = len(sz); _, clu = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                                                               shape=(nc_all, nc_all)), directed=False)
    todo = set(np.unique(pairs[:, :2]).tolist()); fixed = set()
    while todo:
        cand = [p for p in todo if p not in fixed]
        best, bn = None, 0
        for p in cand:                                              # piece with the most seam contacts with the fixed set
            m = ((pairs[:, 0] == p) & np.isin(pairs[:, 1], list(fixed))) | ((pairs[:, 1] == p) & np.isin(pairs[:, 0], list(fixed)))
            if m.sum() > bn: best, bn = p, int(m.sum())
        if best is None or bn < min_edges:                          # start the next cluster from its largest piece
            root = max(cand, key=lambda p: sz[p]); fixed.add(root); todo.discard(root)
            log(f'cluster {clu[root]}: anchor piece {root} ({sz[root]} nodes)')
            continue
        p = best; todo.discard(p)
        m1 = (pairs[:, 0] == p) & np.isin(pairs[:, 1], list(fixed)); m2 = (pairs[:, 1] == p) & np.isin(pairs[:, 0], list(fixed))
        e = np.concatenate([pairs[m1], pairs[m2][:, [1, 0, 4, 5, 2, 3]]])   # rows: (p, fixed, node in p, node in fixed)
        s = ((P[e[:, 4], e[:, 5]] - P[e[:, 2], e[:, 3]]) * n[e[:, 2], e[:, 3]]).sum(-1)
        shift = float(np.median(s))
        if abs(shift) > max_shift: fixed.add(p); continue
        mk = lab == p
        Q = P.copy(); Q[mk] = P[mk] + shift * n[mk]
        # land on the predicted sheet: median offset of the nearest line over the piece
        rr, cc = np.nonzero(mk); pad = 4                             # a margin keeps thin pieces measurable
        r0, r1, c0, c1 = max(0, rr.min() - pad), min(H, rr.max() + 1 + pad), max(0, cc.min() - pad), min(W, cc.max() + 1 + pad)
        sub = Q[r0:r1, c0:c1].copy(); sub[~mk[r0:r1, c0:c1]] = np.nan
        on, off, okm = offsheet_map(pred, sub, every=1, r=6, tol=1, lateral_sigma=0)
        snap = float(np.nanmedian(off[okm & mk[r0:r1, c0:c1]])) if np.isfinite(off[okm & mk[r0:r1, c0:c1]]).any() else 0.0
        P[mk] = Q[mk] + snap * n[mk]
        fixed.add(p)
        moves.append(dict(piece=p, nodes=int(mk.sum()), seam_pairs=bn, shift=shift, snap=snap))
        log(f'piece {p}: {mk.sum()} nodes, {bn} seam pairs, moved {shift:+.1f} vx then snapped {snap:+.1f} vx')
    return P, moves



def _line_offsets(pred, Q, n, R, thr):
    """for nodes Q (M, 3) with normals n: offsets (vx, -R..R, sub-voxel by a parabola through the peak) of the
    predicted lines along the normal (peaks of the profile smoothed with sigma 1 vx, above THR), list of arrays."""
    o = np.arange(-R, R + 1); V = np.empty((len(o), len(Q)), np.float32)
    for k, ok in enumerate(o): V[k] = pred.sample(Q + ok * n)
    S = ndi.gaussian_filter1d(V, 1.0, axis=0)
    pk = (S > thr) & (S >= np.roll(S, 1, 0)) & (S >= np.roll(S, -1, 0)); pk[0] = pk[-1] = False
    out = []
    for i in range(len(Q)):
        j = np.flatnonzero(pk[:, i])
        if len(j):
            a, b, c = S[j - 1, i], S[j, i], S[j + 1, i]; den = a - 2 * b + c
            d = np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / np.where(np.abs(den) > 1e-6, den, 1), 0.0)
            out.append(o[j] + np.clip(d, -0.5, 0.5))
        else: out.append(np.zeros(0))
    return out


def seam_repair_grow(pred, P, thr=4.0, min_edges=3, max_offset=25.0, snap=2.5, R=None, line_thr=100.0, min_snapped=0.5,
                     units=None, max_gap=30.0, log=print):
    """seam_repair with the move grown across the piece instead of one rigid shift: at the piece's seam with the fixed
    set each node starts from its offset to the fixed neighbours (the wrap they are on), snapped to the predicted line
    within SNAP vx; the offset then spreads over the piece's grid, node by node outwards from the seam (prediction =
    median offset of the already placed neighbours, snapped to the nearest line within SNAP vx, kept unsnapped if
    there is none), so it follows the sheet spacing as it changes over a large piece. Only neighbour pairs with
    |offset| <= MAX_OFFSET and a 3D gap <= MAX_GAP count (larger ones are layout tears, not wrap switches). Pieces are
    left where they are if fewer than MIN_EDGES seam nodes land on a line, if fewer than MIN_SNAPPED of the nodes snap,
    or if the median move exceeds MAX_OFFSET (so one wrong move cannot carry the next pieces off).
    UNITS (H, W int labels, -1 invalid; e.g. the atlas charts split at their own seams): repair units instead of the
    seam-free pieces. Then every neighbour pair between units counts (also those that join smoothly): a unit whose
    median offset to the fixed units is <= THR stays (and becomes a reference); otherwise it moves, seeded from the
    contact nodes within 3 vx of that median."""
    import heapq
    from collections import defaultdict
    n = normals(P, smooth=2.0)
    lab, sz, (D0, D1) = seam_components(P, n, thr)
    if units is not None:
        lab = np.where(lab >= 0, units.astype(np.int64) * (int(lab.max()) + 1) + lab, -1)   # chart x seam-free piece
        u, inv = np.unique(lab[lab >= 0], return_inverse=True)
        L2 = -np.ones_like(lab); L2[lab >= 0] = inv; lab = L2; sz = np.bincount(inv)
    if sz.size == 0: return P, []
    R = int(R or max_offset + 8)
    P = P.copy(); moves = []; H, W = lab.shape
    pairs = []
    for ax, D in ((0, D0), (1, D1)):
        sel = np.isfinite(D) & (np.abs(D) <= max_offset)
        if units is None: sel &= np.abs(D) > thr
        r, c = np.nonzero(sel)
        r2, c2 = (r + 1, c) if ax == 0 else (r, c + 1)
        g = np.linalg.norm(P[r2, c2] - P[r, c], axis=-1); k = g <= max_gap; r, c, r2, c2 = r[k], c[k], r2[k], c2[k]
        pairs.append(np.stack([lab[r, c], lab[r2, c2], r, c, r2, c2], 1))
    pairs = np.concatenate(pairs); pairs = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] != pairs[:, 1])]
    if units is not None:                                           # only units that have a seam somewhere need work
        Dp = np.abs(((P[pairs[:, 4], pairs[:, 5]] - P[pairs[:, 2], pairs[:, 3]]) * n[pairs[:, 2], pairs[:, 3]]).sum(-1))
        seamy = np.zeros(len(sz), bool); seamy[pairs[Dp > thr, 0]] = True; seamy[pairs[Dp > thr, 1]] = True
    nc = len(sz)
    by = defaultdict(list)
    for i, (a, b) in enumerate(pairs[:, :2]): by[int(a)].append(i); by[int(b)].append(i)
    flat = lab.ravel(); order = np.argsort(flat, kind='stable'); starts = np.searchsorted(flat[order], np.arange(nc + 1))
    nodes_of = lambda p: np.unravel_index(order[starts[p]:starts[p + 1]], (H, W))
    todo = np.zeros(nc, bool); todo[np.unique(pairs[:, :2])] = True
    fixed = np.zeros(nc, bool); cnt = np.zeros(nc, np.int64); heap = []
    by_size = [int(p) for p in np.argsort(sz)[::-1]]; bs = [0]
    NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def fix(p):
        fixed[p] = True; todo[p] = False
        for i in by[p]:
            q = pairs[i, 1] if pairs[i, 0] == p else pairs[i, 0]
            if not fixed[q]: cnt[q] += 1; heapq.heappush(heap, (-int(cnt[q]), int(q)))

    def near(lines, x):
        if len(lines) == 0: return np.nan
        j = int(np.argmin(np.abs(lines - x))); return float(lines[j]) if abs(lines[j] - x) <= snap else np.nan

    def leave(p, nb, why, ref=True):
        fix(p); moves.append(dict(piece=int(p), nodes=int(sz[p]), seam_pairs=nb, moved=False, why=why))
        if sz[p] >= 1000: log(f'piece {p}: {sz[p]} nodes, {nb} contacts: left as it is ({why})')
    while todo.any():
        while heap and (not todo[heap[0][1]] or -heap[0][0] != cnt[heap[0][1]]): heapq.heappop(heap)
        if not heap or -heap[0][0] < min_edges:
            while not todo[by_size[bs[0]]]: bs[0] += 1
            fix(by_size[bs[0]]); continue
        p = heapq.heappop(heap)[1]
        e = pairs[by[p]]; e = e[(e[:, 0] == p) & fixed[e[:, 1]] | (e[:, 1] == p) & fixed[e[:, 0]]]
        e = np.where((e[:, 0] == p)[:, None], e, e[:, [1, 0, 4, 5, 2, 3]])          # (p, fixed, node in p, node in fixed)
        s = ((P[e[:, 4], e[:, 5]] - P[e[:, 2], e[:, 3]]) * n[e[:, 2], e[:, 3]]).sum(-1)
        nb = int(cnt[p]); keep = np.abs(s) <= max_offset; e, s = e[keep], s[keep]    # offsets now (fixed pieces may have moved)
        if len(s) < min_edges: leave(p, nb, 'seam > max_offset now'); continue
        if units is not None:
            med = float(np.median(s))
            if abs(med) <= thr or not seamy[p]: fix(p); continue          # agrees with the fixed units: a reference
            k = np.abs(s - med) <= 3.0; e, s = e[k], s[k]
            if len(s) < min_edges: leave(p, nb, 'contacts disagree'); continue
        rr, cc = nodes_of(p)
        L = _line_offsets(pred, P[rr, cc], n[rr, cc], R, line_thr)
        loc = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rr, cc))}
        off = np.full(len(rr), np.nan); snapped = np.zeros(len(rr), bool)
        start = defaultdict(list)
        for (ra, ca), sv in zip(e[:, 2:4], s): start[loc[(int(ra), int(ca))]].append(sv)
        hq = []
        for i, sv in list(start.items()):                           # seeds: seam nodes whose offset lands on a line
            x = float(np.median(sv)); y = near(L[i], x)
            if np.isfinite(y): off[i] = y; snapped[i] = True; heapq.heappush(hq, (0, i))
            else: del start[i]
        if len(start) < min_edges: leave(p, nb, 'seam offsets not on a line'); continue
        seen = np.zeros(len(rr), bool); seen[list(start)] = True
        while hq:
            dd, i = heapq.heappop(hq)
            for dr, dc in NB:
                j = loc.get((int(rr[i]) + dr, int(cc[i]) + dc))
                if j is None or seen[j]: continue
                seen[j] = True
                vals = [off[loc[k]] for k in ((int(rr[j]) + a, int(cc[j]) + b) for a, b in NB) if k in loc and np.isfinite(off[loc[k]])]
                x = float(np.median(vals)); y = near(L[j], x)
                off[j] = y if np.isfinite(y) else x; snapped[j] = np.isfinite(y)
                heapq.heappush(hq, (dd + 1, j))
        g = np.isfinite(off)
        why = ('no move' if not g.any() or abs(float(np.median(off[g]))) < 1.0 else
               f'snapped on {snapped[g].mean() * 100:.0f} % < {min_snapped * 100:.0f} %' if snapped[g].mean() < min_snapped else
               'move > max_offset' if abs(float(np.median(off[g]))) > max_offset else None)
        if why: leave(p, nb, why); continue
        P[rr[g], cc[g]] = P[rr[g], cc[g]] + off[g][:, None] * n[rr[g], cc[g]]
        fix(p)
        moves.append(dict(piece=int(p), nodes=int(sz[p]), seam_pairs=nb, start_nodes=len(start), reached=round(float(g.mean()), 3),
                          snapped=round(float(snapped[g].mean()), 3), off_median=round(float(np.median(off[g])), 2),
                          off_p10=round(float(np.percentile(off[g], 10)), 2), off_p90=round(float(np.percentile(off[g], 90)), 2), moved=True))
        if sz[p] >= 100 or units is None:
            log(f'piece {p}: {sz[p]} nodes, {nb} contacts from {len(start)} nodes; reached {g.mean() * 100:.0f} %, snapped '
                f'{snapped[g].mean() * 100:.0f} %; move median {np.median(off[g]):+.1f} vx (p10 {np.percentile(off[g], 10):+.1f}, p90 {np.percentile(off[g], 90):+.1f})')
    log(f'{sum(1 for m in moves if m.get("moved"))} units moved, {sum(1 for m in moves if not m.get("moved"))} left')
    return P, moves


def _unit_normals(P, rr, cc, smooth=2.0, pad=4):
    """normals of one unit's nodes computed from the unit alone (normals smoothed across a seam into a chart on
    another sheet are tilted, and a move along them lands between sheets)."""
    r0, r1, c0, c1 = max(0, rr.min() - pad), min(P.shape[0], rr.max() + 1 + pad), max(0, cc.min() - pad), min(P.shape[1], cc.max() + 1 + pad)
    sub = np.full((r1 - r0, c1 - c0, 3), np.nan); sub[rr - r0, cc - c0] = P[rr, cc]
    return normals(sub, smooth=smooth)[rr - r0, cc - c0]


def _grow(pred, P, n, rr, cc, seeds, R, snap, line_thr):
    """offsets along the normals for the nodes (rr, cc) of one unit: start at SEEDS {node index: target offset} (snapped
    to a predicted line within SNAP vx; seeds that do not land on a line are dropped), then spread node by node over
    the unit's grid (median of placed 8-neighbours, snapped within SNAP, kept unsnapped if no line). Returns
    (offsets (NaN = not reached), snapped mask, number of seeds used)."""
    import heapq
    nn = n if n.ndim == 2 else n[rr, cc]                            # per-node normals (M, 3) or the grid's
    L = _line_offsets(pred, P[rr, cc], nn, R, line_thr)
    loc = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(rr, cc))}
    off = np.full(len(rr), np.nan); snapped = np.zeros(len(rr), bool); hq = []

    def near(lines, x):
        if len(lines) == 0: return np.nan
        j = int(np.argmin(np.abs(lines - x))); return float(lines[j]) if abs(lines[j] - x) <= snap else np.nan
    for i, x in seeds.items():
        y = near(L[i], x)
        if np.isfinite(y): off[i] = y; snapped[i] = True; heapq.heappush(hq, (0, i))
    ns = len(hq); seen = np.isfinite(off)
    NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while hq:
        dd, i = heapq.heappop(hq)
        for dr, dc in NB:
            j = loc.get((int(rr[i]) + dr, int(cc[i]) + dc))
            if j is None or seen[j]: continue
            seen[j] = True
            vals = [off[loc[k]] for k in ((int(rr[j]) + a, int(cc[j]) + b) for a, b in NB) if k in loc and np.isfinite(off[loc[k]])]
            x = float(np.median(vals)); y = near(L[j], x)
            off[j] = y if np.isfinite(y) else x; snapped[j] = np.isfinite(y)
            heapq.heappush(hq, (dd + 1, j))
    return off, snapped, ns


def seam_repair_icm(pred, P, units, thr=4.0, max_offset=25.0, max_gap=30.0, tau=6.0, gain=0.1, snap=2.5, min_snapped=0.7,
                    min_seeds=3, sweeps=30, R=None, line_thr=100.0, log=print):
    """Seam repair by a joint labelling instead of a greedy order. Units = UNITS (e.g. atlas charts) split at their own
    seams. Every neighbour pair between two units (|normal offset| <= MAX_OFFSET, 3D gap <= MAX_GAP, normals agreeing)
    is a contact with offset s (from the node in u to the node in v along u's normal). Each unit gets one offset o_u;
    a contact's residual after the shifts is s - o_u + o_v, its cost min(|residual|, TAU). ICM: each unit in turn takes
    the offset (0 or the median target s + o_v of one neighbouring unit) that minimises its summed contact cost, but
    only if that beats its current cost by GAIN x (its number of contacts) (a unit that agrees with most of its
    neighbours stays); sweeps until nothing changes. Then units with o_u != 0 move by seam_repair_grow's per-node
    growth, seeded from the contact nodes whose target lies within 3 vx of o_u, snapped to the predicted line; a unit
    stays if < MIN_SEEDS seeds land on a line, < MIN_SNAPPED of it snaps, or its median move differs from o_u by > 4 vx."""
    from collections import defaultdict
    n = normals(P, smooth=2.0)
    lab, sz, (D0, D1) = seam_components(P, n, thr)
    lab = np.where(lab >= 0, units.astype(np.int64) * (int(lab.max()) + 1) + lab, -1)
    u_, inv = np.unique(lab[lab >= 0], return_inverse=True); L2 = -np.ones_like(lab); L2[lab >= 0] = inv; lab = L2; sz = np.bincount(inv)
    H, W = lab.shape; R = int(R or max_offset + 8); P = P.copy()
    E = []
    for ax, D in ((0, D0), (1, D1)):
        r, c = np.nonzero(np.isfinite(D) & (np.abs(D) <= max_offset)); r2, c2 = (r + 1, c) if ax == 0 else (r, c + 1)
        k = (lab[r, c] != lab[r2, c2]) & (np.linalg.norm(P[r2, c2] - P[r, c], axis=-1) <= max_gap) & ((n[r, c] * n[r2, c2]).sum(-1) > 0.5)
        E.append(np.stack([r[k], c[k], r2[k], c2[k]], 1))
    E = np.concatenate(E); A = lab[E[:, 0], E[:, 1]]; B = lab[E[:, 2], E[:, 3]]
    s = ((P[E[:, 2], E[:, 3]] - P[E[:, 0], E[:, 1]]) * n[E[:, 0], E[:, 1]]).sum(-1)
    # directed contacts: (unit, other unit, offset from unit to other, node in unit)
    U = np.concatenate([A, B]); V = np.concatenate([B, A]); S = np.concatenate([s, -s]); NODE = np.concatenate([E[:, :2], E[:, 2:]])
    order = np.argsort(U, kind='stable'); U, V, S, NODE = U[order], V[order], S[order], NODE[order]
    st = np.searchsorted(U, np.arange(len(sz) + 1))
    nc = len(sz); o = np.zeros(nc); active = np.zeros(nc, bool)
    seamy = np.zeros(nc, bool); seamy[U[np.abs(S) > thr]] = True
    active[:] = seamy
    log(f'{nc} units, {len(E)} contacts, {seamy.sum()} units with a seam')
    for sw in range(sweeps):
        changed = 0; nxt = np.zeros(nc, bool)
        for u in np.flatnonzero(active):
            a, b = st[u], st[u + 1]
            if a == b: continue
            t = S[a:b] + o[V[a:b]]                                  # where each contact wants u
            cand = [0.0, o[u]] + [float(np.median(t[V[a:b] == v])) for v in np.unique(V[a:b])]
            cost = [np.minimum(np.abs(t - x), tau).sum() for x in cand]
            j = int(np.argmin(cost)); cur = np.minimum(np.abs(t - o[u]), tau).sum()
            if cost[j] < cur - gain * (b - a) and abs(cand[j] - o[u]) > 1.0 and abs(cand[j]) <= max_offset:
                o[u] = cand[j]; changed += 1; nxt[V[a:b]] = True
        log(f'sweep {sw + 1}: {changed} units changed offset')
        if not changed: break
        active = nxt
    mv = np.flatnonzero(np.abs(o) > thr); moves = []
    flat = lab.ravel(); od = np.argsort(flat, kind='stable'); starts = np.searchsorted(flat[od], np.arange(nc + 1))
    final = np.abs(o) <= thr; pending = set(int(u) for u in mv)
    # apply in passes: a unit moves once enough of its contacts are with final units (unlabelled ones, or units already
    # moved or given up), seeded from the neighbours' ACTUAL positions (a neighbour that could not move is not where
    # the labelling put it)
    while pending:
        progress = False
        for u in sorted(pending, key=lambda u: -sz[u]):
            a, b = st[u], st[u + 1]; fv = final[V[a:b]]
            if fv.sum() < min_seeds and any(not final[v] for v in V[a:b]): continue          # wait for neighbours
            rr, cc = np.unravel_index(od[starts[u]:starts[u + 1]], (H, W))
            nod = NODE[a:b]; oth = np.concatenate([E[:, 2:], E[:, :2]])[order][a:b]          # the neighbour node of each contact
            nu = _unit_normals(P, rr, cc); bad = ~np.isfinite(nu).all(1); nu[bad] = n[rr[bad], cc[bad]]
            loc = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(rr, cc))}
            ix = np.array([loc[(int(x), int(y))] for x, y in nod])
            t = ((P[oth[:, 0], oth[:, 1]] - P[nod[:, 0], nod[:, 1]]) * nu[ix]).sum(-1)
            k = fv & (np.abs(t - o[u]) <= 4.0)
            seeds = defaultdict(list)
            for (x, y), tv in zip(nod[k], t[k]): seeds[loc[(int(x), int(y))]].append(tv)
            seeds = {i: float(np.median(v)) for i, v in seeds.items()}
            off, snapped, ns = _grow(pred, P, nu, rr, cc, seeds, R, snap, line_thr) if len(seeds) >= min_seeds else (None, None, 0)
            g = np.isfinite(off) if off is not None else None
            why = ('seeds' if ns < min_seeds else 'not reached' if not g.any() else
                   'snapped' if snapped[g].mean() < min_snapped else 'drift' if abs(np.median(off[g]) - o[u]) > 6.0 else None)
            moves.append(dict(unit=int(u), nodes=int(sz[u]), o=round(float(o[u]), 2), moved=why is None, why=why, contacts=int(b - a),
                              agree=int(k.sum()), seeds=len(seeds), seeds_on_line=int(ns),
                              snapped=None if g is None or not g.any() else round(float(snapped[g].mean()), 3)))
            if why is None: P[rr[g], cc[g]] = P[rr[g], cc[g]] + off[g][:, None] * nu[g]
            final[u] = True; pending.discard(u); progress = True
        if not progress:                                            # cycles of pending units: release the largest
            u = max(pending, key=lambda u: sz[u]); final[u] = True; pending.discard(u)
            moves.append(dict(unit=int(u), nodes=int(sz[u]), o=round(float(o[u]), 2), moved=False, why='cycle'))
    m = sum(x['moved'] for x in moves)
    log(f'labelling: {len(mv)} units with |o| > {thr:g} vx; moved {m} ({sum(x["nodes"] for x in moves if x["moved"])} nodes), left '
        f'{len(mv) - m}: ' + str({w: sum(1 for x in moves if x['why'] == w) for w in ('seeds', 'not reached', 'snapped', 'drift', 'cycle')}))
    return P, moves
