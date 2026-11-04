#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PY=/home/figo/venvs/cv-advantage/bin/python
REPO=/home/figo/CV_Advantage_Estimation
RAW=/mnt/d/CV/data/raw/samp830
BASE=/mnt/d/CV/training_runs/samp830_lowdim_v1
RUN_ID=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
RUN="$BASE/$RUN_ID"
EXPECTED=6c00b283950e6957e42c259688aa10ad94260495
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED"
git -C "$REPO" diff --exit-code
mkdir -p "$BASE"
mkdir "$RUN"
mkdir -p "$RUN/logs" "$RUN/models" "$RUN/scores" "$RUN/smoke" "$RUN/provenance"
printf '%s\n' "$RUN" > "$BASE/LATEST.txt"
exec > >(tee "$RUN/run.log") 2>&1
printf '%s\n' "started" > "$RUN/STATUS.txt"
trap 'printf "%s\n" "failed at line $LINENO" > "$RUN/STATUS.txt"' ERR
cp /mnt/d/CV/training_tools/samp830_lowdim_v1/run_training.sh "$RUN/provenance/"
cp /home/figo/cv-advantage-setup/requirements-installed.txt "$RUN/provenance/"
cp /home/figo/cv-advantage-setup/copied-distributions.json "$RUN/provenance/"
cp /home/figo/cv-advantage-setup/original-environment-before.json "$RUN/provenance/"
git -C "$REPO" rev-parse HEAD > "$RUN/provenance/repository-commit.txt"
cd "$REPO"
printf 'RUN %s\n' "$RUN"
"$PY" -B -c 'import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device="cuda"); assert (x@x)[0,0].item()==32; print("CUDA PASS",torch.__version__,torch.cuda.get_device_name(0),flush=True)' | tee "$RUN/logs/cuda.log"
"$PY" -B -m unittest discover -s tests -v > "$RUN/logs/tests.log" 2>&1
tail -n 5 "$RUN/logs/tests.log"
for task in 0 13 22 31; do
    printf 'CONVERT task=%s\n' "$task"
    "$PY" -B -m advantage.cli convert-samp830 "$RAW" --task-index "$task" --seed 12 \
        --output "$RUN/datasets/task_$task" --work-dir "$RUN/staging/task_$task" \
        > "$RUN/logs/convert_task_$task.json" 2>&1
done
DS="$RUN/datasets/task_0"
printf '%s\n' "Task 0 short CUDA training and reload/scoring check"
"$PY" -B -m advantage.cli train "$DS" --method mc --gamma 0.99 --updates 100 --seed 0 --device cuda --output "$RUN/smoke/v.pt" > "$RUN/logs/smoke_v.log" 2>&1
"$PY" -B -m advantage.cli train-q "$DS" --gamma 0.99 --updates 100 --seed 0 --device cuda --output "$RUN/smoke/q.pt" > "$RUN/logs/smoke_q.log" 2>&1
"$PY" -B -m advantage.cli score "$DS" --checkpoint "$RUN/smoke/v.pt" --ns 1 5 10 20 --split validation --output "$RUN/smoke/td.csv" > "$RUN/logs/smoke_td.log" 2>&1
"$PY" -B -m advantage.cli score-qv "$DS" --q-checkpoint "$RUN/smoke/q.pt" --v-checkpoint "$RUN/smoke/v.pt" --split validation --output "$RUN/smoke/qv.csv" > "$RUN/logs/smoke_qv.log" 2>&1
printf '%s\n' "SMOKE PASS"
for task in 0 13 22 31; do
    DS="$RUN/datasets/task_$task"
    for seed in 0 1 2; do
        PREFIX="task_"$task"_seed"$seed
        printf 'TRAIN task=%s seed=%s V\n' "$task" "$seed"
        "$PY" -B -m advantage.cli train "$DS" --method mc --gamma 0.99 --updates 1000 --seed "$seed" --device cuda \
            --output "$RUN/models/"$PREFIX"_v.pt" | tee "$RUN/logs/"$PREFIX"_v.log"
        printf 'TRAIN task=%s seed=%s Q\n' "$task" "$seed"
        "$PY" -B -m advantage.cli train-q "$DS" --gamma 0.99 --updates 1000 --seed "$seed" --device cuda \
            --output "$RUN/models/"$PREFIX"_q.pt" | tee "$RUN/logs/"$PREFIX"_q.log"
        for split in validation test; do
            "$PY" -B -m advantage.cli score "$DS" --checkpoint "$RUN/models/"$PREFIX"_v.pt" \
                --ns 1 5 10 20 --split "$split" --output "$RUN/scores/"$PREFIX"_"$split"_td.csv" \
                > "$RUN/logs/"$PREFIX"_"$split"_td.log" 2>&1
            "$PY" -B -m advantage.cli score-qv "$DS" --q-checkpoint "$RUN/models/"$PREFIX"_q.pt" \
                --v-checkpoint "$RUN/models/"$PREFIX"_v.pt" --split "$split" \
                --output "$RUN/scores/"$PREFIX"_"$split"_qv.csv" \
                > "$RUN/logs/"$PREFIX"_"$split"_qv.log" 2>&1
        done
        printf 'DONE task=%s seed=%s\n' "$task" "$seed"
    done
done
printf '%s\n' "training_and_scoring_complete" > "$RUN/STATUS.txt"
printf 'TRAINING COMPLETE %s\n' "$RUN"
