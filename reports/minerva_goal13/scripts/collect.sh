#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/CV
source_dir="$base/training_tools/advantage_studies_src"
here="$base/training_tools/minerva_goal13"
root="${1:?supply the run directory}"
[[ "$(git -c safe.directory="$source_dir" -C "$source_dir" rev-parse HEAD)" == 7671b454a281506a68706e7b57660742b7da85da ]]
mkdir -p "$root/episodes"
exec 8>"$root/.collect.lock"
flock -n 8
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/home/figo/minerva_smoke/libero-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$source_dir/src"
workers=4
pids=()
for ((i=0; i<workers; i++)); do
    /home/figo/minerva_smoke/venv/bin/python -u "$here/collect.py" "$root" "$i" "$workers" >> "$root/worker$i.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
