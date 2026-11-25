#!/usr/bin/env bash
# Repair step for finished Paris 4 tiles (detector held under the amended rule): seam_audit --repair on the unshifted
# surface (reference frame, Paris 4 m7 prediction), then p4_score --variant with the amended rule.
# usage: tile_repair.sh TILES_ROOT CROSS_REL.npz   (skips tiles already repaired or not yet scored)
set -u
root=$1; cross=$2
PY=${PYTHON:-python3}; P4=$(cd "$(dirname "$0")" && pwd); SD=$P4/../scripts
PRED=PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0
for d in "$root"/atlas_tile_*; do
  [ -f "$d/score_rel_r48_t5.json" ] && [ ! -f "$d/score_rel_repair.json" ] || continue
  "$PY" "$SD/seam_audit.py" "$d/audit/surface.tifxyz" "$d/repair" --repair "$PRED" > "$d/repair.log" 2>&1 || continue
  "$PY" "$P4/p4_score.py" "$d/audit/surface.tifxyz" "$d/audit/seams.csv" "$cross" "$d/score_rel_repair.json" \
    --r-side 48 --tol 5 --variant "$d/repair/repaired.tifxyz" > "$d/score_rel_repair.log" 2>&1
done
