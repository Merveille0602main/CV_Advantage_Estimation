$ErrorActionPreference = 'Stop'
wsl.exe -d Ubuntu-24.04 -u figo --cd /home/figo -- bash /mnt/d/CV/training_tools/samp830_lowdim_v1/run_training.sh
if ($LASTEXITCODE -ne 0) { throw "训练失败，查看运行目录的 run.log。" }
wsl.exe -d Ubuntu-24.04 -u figo --cd /home/figo -- /home/figo/venvs/cv-advantage/bin/python -B /mnt/d/CV/training_tools/samp830_lowdim_v1/report_training.py
if ($LASTEXITCODE -ne 0) { throw "结果核验或报告生成失败。" }
