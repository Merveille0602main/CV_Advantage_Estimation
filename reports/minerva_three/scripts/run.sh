#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=${STUDY_ROOT:-/mnt/d/CV/training_runs/minerva_three/run01}
export STUDY_ROOT="$root"
export PYTHONPATH="${ADVANTAGE_SRC:-/mnt/d/CV/training_tools/advantage_studies_src/src}:${MINERVA_OLD_SCRIPTS:-/mnt/d/CV/training_tools/minerva_goal13}"
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-/home/figo/minerva_smoke/libero-config}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
sim=${MINERVA_PYTHON:-/home/figo/minerva_smoke/venv/bin/python}
learn=${ADVANTAGE_PYTHON:-/home/figo/venvs/cv-advantage/bin/python}
mode=${1:?collect, train, augment, anchors, branches, or metrics}
mkdir -p "$root/logs"
exec 9>"$root/.$mode.lock"
flock -n 9
workers=${STUDY_WORKERS:-4}
pool() {
    local script="$1" task="$2"
    shift 2
    local pids=() status=0
    for ((i=0; i<workers; i++)); do
        "$sim" -u "$here/$script" "$task" "$i" "$workers" "$@" >>"$root/logs/${mode}_${task}_$i.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}
case "$mode" in
    complete)
        flock "$root/.collect.lock" true
        bash "$here/run.sh" collect
        for phase in anchors augment train branches metrics; do
            echo "Starting $phase $(date -Iseconds)"
            bash "$here/run.sh" "$phase"
        done ;;
    collect) for task in 19 25; do pool simulator.py "$task"; done ;;
    anchors|augment|branches) for task in 13 19 25; do pool branches.py "$task" "$mode"; done ;;
    train|metrics) OMP_NUM_THREADS=1 "$learn" -u "$here/analysis.py" "$mode" ;;
    *) exit 2 ;;
esac
