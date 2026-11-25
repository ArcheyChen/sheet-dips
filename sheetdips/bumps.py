"""Cleanup after the seam repair: flatten right-angle detours ('box bumps') left on a surface grid.

A bump, along a grid row or column, is a step of more than THR vx along the normal, a flat run of at most KMAX nodes, and
a step back that cancels the first within BACK vx: both ends of the line lie on one sheet and a short stretch on
another. On a flattened atlas these are small pieces placed on the wrong wrap whose neighbours all agree where they
belong, but which a one-unit-at-a-time labelling (seam_repair_icm) leaves in place, because they agree with each other.

bump_pass moves whole regions:
  units='pieces': the seam-free pieces of the grid (neighbours joined while their normal offset is <= THR), up to
                  MAX_PIECE nodes (larger pieces are anchors);
  units='bumps':  the 4-connected regions of nodes inside detected bumps.
A region moves rigidly along its normals by the median of its stepped boundary offsets (THR < |offset| <= MAX_STEP),
if at least AGREE of those agree with the median within 3 vx, and the move is kept only if it removes at least half of
the region's boundary steps. Smallest regions go first; rounds repeat until nothing moves. No snapping afterwards:
the target comes from the neighbours' actual positions, which already lie on their sheet."""
import numpy as np
from scipy import ndimage as ndi
from .io import normals
from .fix import seam_components

NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def line_steps(P, n, ax):
    """normal offsets between grid neighbours along axis AX (NaN where invalid or > 30 vx apart in 3D)."""
    sa = [slice(None)] * 2; sb = [slice(None)] * 2; sa[ax] = slice(0, -1); sb[ax] = slice(1, None); sa, sb = tuple(sa), tuple(sb)
    d = ((P[sb] - P[sa]) * 0.5 * (n[sa] + n[sb])).sum(-1); g = np.linalg.norm(P[sb] - P[sa], axis=-1)
    d[~np.isfinite(d) | (g > 30)] = np.nan
    return d


def bump_mask(P, n, thr=4.0, kmax=40, back=3.0, flat=2.5, max_step=25.0):
    """(nodes strictly inside a bump along rows or columns, number of bumps)."""
    H, W = P.shape[:2]; M = np.zeros((H, W), bool); count = 0
    for ax in (1, 0):
        D = line_steps(P, n, ax); D = D if ax == 1 else D.T
        for i in range(D.shape[0]):
            row = D[i]; big = np.flatnonzero(np.isfinite(row) & (np.abs(row) > thr) & (np.abs(row) <= max_step))
            for p, j1 in enumerate(big):
                for j2 in big[p + 1:]:
                    if j2 - j1 > kmax: break
                    mid = row[j1 + 1:j2]
                    if len(mid) and not (np.isfinite(mid).all() and (np.abs(mid) <= flat).all()): break
                    if np.sign(row[j2]) != np.sign(row[j1]) and abs(row[j1] + row[j2]) <= back:
                        if ax == 1: M[i, j1 + 1:j2 + 1] = True
                        else: M[j1 + 1:j2 + 1, i] = True
                        count += 1; break
    return M, count


def _regions(P, n, ok, units, thr, max_piece, M):
    H, W = ok.shape
    if units == 'bumps':
        return ndi.label(M)
    lab0, _, _ = seam_components(P, n, thr); _, inv = np.unique(lab0[ok], return_inverse=True)
    lab = np.zeros((H, W), np.int64); lab[ok] = inv + 1; sz = np.bincount(lab.ravel())
    lab[sz[lab] > max_piece] = 0; u_, inv = np.unique(lab, return_inverse=True)
    lab = inv.reshape(H, W) if u_[0] == 0 else inv.reshape(H, W) + 1
    return lab, int(lab.max())


def bump_pass(P, units='pieces', max_piece=300, thr=4.0, kmax=40, back=3.0, flat=2.5, max_step=25.0, agree=0.6,
              rounds=3, log=print):
    """returns (moved copy of P, per-round report)."""
    P = P.copy(); H, W = P.shape[:2]; report = []
    bk = dict(thr=thr, kmax=kmax, back=back, flat=flat, max_step=max_step)
    for rnd in range(rounds):
        n = normals(P, smooth=2.0); ok = np.isfinite(P).all(-1) & np.isfinite(n).all(-1)
        M, nb = bump_mask(P, n, **bk); M &= ok
        lab, nr = _regions(P, n, ok, units, thr, max_piece, M)
        objs = ndi.find_objects(lab); tried = kept = nodes_moved = 0
        cnt = np.bincount(lab.ravel(), minlength=len(objs) + 1)
        for k in sorted(range(1, len(objs) + 1), key=lambda k: cnt[k]):
            sl = objs[k - 1]
            if sl is None or cnt[k] == 0: continue
            r0, r1 = max(0, sl[0].start - 1), min(H, sl[0].stop + 1); c0, c1 = max(0, sl[1].start - 1), min(W, sl[1].stop + 1)
            rr, cc = np.nonzero(lab[r0:r1, c0:c1] == k); rr += r0; cc += c0; nu = n[rr, cc]
            con = []                                                # boundary contacts: (index in region, outside node)
            for dr, dc in NB4:
                r2, c2 = rr + dr, cc + dc; v = np.flatnonzero((r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W))
                r2, c2 = r2[v], c2[v]; o = ok[r2, c2] & (lab[r2, c2] != k)
                con.append((v[o], r2[o], c2[o]))
            iv = np.concatenate([c[0] for c in con]); orr = np.concatenate([c[1] for c in con]); occ = np.concatenate([c[2] for c in con])
            if not len(iv): continue
            near = np.linalg.norm(P[orr, occ] - P[rr[iv], cc[iv]], axis=-1) <= 30
            offs = ((P[orr, occ] - P[rr[iv], cc[iv]]) * nu[iv]).sum(-1)[near]
            if len(offs) < 2: continue
            step = offs[(np.abs(offs) > thr) & (np.abs(offs) <= max_step)]
            if len(step) < 2: continue
            m = float(np.median(step))
            if (np.abs(step - m) <= 3.0).mean() < agree: continue
            tried += 1
            Q = P[rr, cc] + m * nu
            before = (np.abs(offs) > thr).sum()
            after = (np.abs(((P[orr, occ] - Q[iv]) * nu[iv]).sum(-1)) > thr).sum()
            if after <= 0.5 * before:
                P[rr, cc] = Q; kept += 1; nodes_moved += len(rr)
        log(f'{units} round {rnd + 1}: {nb} bumps, {nr} regions, {tried} tried, {kept} kept ({nodes_moved} nodes)')
        report.append(dict(round=rnd + 1, bumps=nb, regions=nr, tried=tried, kept=kept, nodes_moved=nodes_moved))
        if not kept: break
    return P, report
