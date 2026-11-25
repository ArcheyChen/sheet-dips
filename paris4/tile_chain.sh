#!/usr/bin/env bash
# One Paris 4 tile: atlas_overlap_fix -> seam_audit -> unshift -> p4_score (rule: README section 5).
# usage: tile_chain.sh TILES_ROOT TILE SHIFT.json CROSS.npz     (paths relative to the cwd are fine)
# Writes TILES_ROOT/atlas_TILE/{overlap_fix.log, audit/, audit.log, score.json, score.log, chain.done}.
# TILE_TIMEOUT (s, default 10800) bounds overlap_fix. CROSS_REL (default CROSS minus .npz + _rel15.npz,
# used if present): reliable-cube crossings for the amended rule.
set -u
root=$1; t=$2; shift_json=$3; cross=$4
out=$root/atlas_$t
# FINISH_ONLY=1: skip the lock and overlap_fix (its outputs already exist) and run the rest
[ -n "${FINISH_ONLY:-}" ] || mkdir "$root/.lock_$t" 2>/dev/null || { echo "$t already claimed"; exit 0; }   # one run per tile per host
mkdir -p "$out"
SF=${SF_BUILD:-}; [ -n "${FINISH_ONLY:-}" ] || [ -n "$SF" ] || { echo "set SF_BUILD to the ScrollFiesta CMake build dir"; exit 2; }
PY=${PYTHON:-python3}
P4=$(cd "$(dirname "$0")" && pwd); SD=$P4/../scripts
s=$(date +%s); e2=-; e3=-; e4=-
if [ -n "${FINISH_ONLY:-}" ]; then [ -s "$out/atlas_bake.obj" ]; e1=$?
else timeout "${TILE_TIMEOUT:-10800}" "$SF/atlas_overlap_fix" "$root/$t" "$out" --rounds 7 > "$out/overlap_fix.log" 2>&1; e1=$?; fi
if [ "$e1" = 0 ]; then
  "$PY" "$SD/seam_audit.py" "$out/atlas_bake.obj" "$out/audit" > "$out/audit.log" 2>&1; e2=$?
  if [ "$e2" = 0 ]; then
    "$PY" "$P4/unshift_tifxyz.py" "$out/audit/surface.tifxyz" "$shift_json" >> "$out/audit.log" 2>&1; e3=$?
    [ "$e3" = 0 ] && { "$PY" "$P4/p4_score.py" "$out/audit/surface.tifxyz" "$out/audit/seams.csv" "$cross" \
      "$out/score.json" > "$out/score.log" 2>&1; e4=$?; }
    # amended rule (README section 5): reliable-cube reference, r_side 48, tol 5
    CROSS_REL=${CROSS_REL:-${cross%.npz}_rel15.npz}
    [ "$e3" = 0 ] && [ -f "$CROSS_REL" ] && "$PY" "$P4/p4_score.py" "$out/audit/surface.tifxyz" "$out/audit/seams.csv" \
      "$CROSS_REL" "$out/score_rel_r48_t5.json" --r-side 48 --tol 5 > "$out/score_rel.log" 2>&1
  fi
fi
echo "tile=$t overlap_fix=$e1 audit=$e2 unshift=$e3 score=$e4 sec=$(( $(date +%s) - s )) host=$(uname -n)" > "$out/chain.done"
