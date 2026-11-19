#!/usr/bin/env bash
set -euo pipefail
base=/mnt/d/CV
src="$base/training_tools/advantage_studies_src"
root="${1:?supply the completed collection directory}"
[[ "$(git -c safe.directory="$src" -C "$src" rev-parse HEAD)" == 7671b454a281506a68706e7b57660742b7da85da ]]
export PYTHONPATH="$src/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
exec 9>"$root/.train.lock"
flock -n 9
/home/figo/venvs/cv-advantage/bin/python -u "$base/training_tools/minerva_goal13/train_validate.py" "$root" 2>&1 | tee -a "$root/train.log"
