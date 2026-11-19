#!/usr/bin/env bash
set -euo pipefail
src=/mnt/d/CV/training_tools/advantage_studies_src
here=/mnt/d/CV/training_tools/minerva_goal13
out=${1:?supply evaluation directory}
mkdir -p "$out"
exec 8>"$out/.run.lock"
flock -n 8
[[ "$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)" == 7671b454a281506a68706e7b57660742b7da85da ]]
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=/home/figo/minerva_smoke/libero-config
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$src/src"
for mode in verify branch; do
    pids=()
    for ((i=0; i<4; i++)); do
        /home/figo/minerva_smoke/venv/bin/python -u "$here/branch_eval.py" "$out" "$mode" "$i" 4 >> "$out/${mode}_$i.log" 2>&1 &
        pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    [[ $status == 0 ]] || exit 1
done
