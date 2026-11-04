# CV Advantage Estimation

机器人离线轨迹的 TD / N-step 工具、统一数据契约，以及可复现的人工轨迹实验。

当前阶段：数值语义与人工实验。真实 RLinf 数据仍在下载；没有宣称完成真实数据训练或动作质量评价。

## 安装与运行

在 WSL 的现有 ML 环境中：

```bash
conda activate ML
cd /home/merveille/CV_Advantage_Estimation
pip install -e '.[rlinf,plots]'
python -m unittest discover -s tests -v
advantage synthetic --output artifacts/synthetic_v1 --seeds 0 1 2
```

核心工具仅依赖 NumPy。RLinf 适配额外使用 PyArrow；图表额外使用 Matplotlib。
真实低维 V 学习器使用 ML 环境已有的 PyTorch，不会安装整个 RLinf 或大型策略模型。

## 两类接口，避免混淆

- `td_targets` / `n_step_targets`：生成学习 V 的回归目标。
- `td_scores` / `n_step_scores`：从目标中减去当前 V，生成回顾式动作评分。
- `discounted_returns`：反向计算完整折扣回报；截断轨迹必须显式提供尾部 bootstrap。

```python
from advantage import td_targets, n_step_scores

# ep 为一条经过校验的 Episode；value_fn 必须包含剩余时间信息。
next_values = value_fn(ep.next_obs, ep.remaining_steps - 1)
values = value_fn(ep.obs, ep.remaining_steps)
training_targets = td_targets(ep, next_values, gamma=0.99)
scores = n_step_scores(ep, values, next_values, n=5, gamma=0.99)
```

单步 N-step 与一步 TD 相同。真正终止后不 bootstrap；采集中断必须用真实的末尾 next_obs bootstrap。
每次只处理一条轨迹，任何跨度都不会越过轨迹边界。分数读取了动作后的真实未来，不能冒充执行前评分。

## 已实现的数据接口

见 [数据契约](docs/DATA_CONTRACT.md)。原生格式为无 pickle 的 NPZ，每条轨迹一文件；manifest 保存 SHA256、家族和 split。

RLinf RECAP 官方说明采用 LeRobot 格式，原始奖励可以来自独立的 `meta/returns_{tag}.parquet`。
本项目按 `episode_index` 和 `frame_index` 严格合并，而不按文件行序合并。

```bash
advantage inspect-rlinf /path/to/downloaded_dataset
# 复制并逐项核实 configs/rlinf_lerobot.example.json；模板故意不直接接受未知动作/终止语义。
advantage convert-rlinf /path/to/downloaded_dataset \
  --config configs/confirmed_dataset.json --output data/canonical_v1
advantage train data/canonical_v1 --method mc --updates 1000 --device cuda --output artifacts/shared_v_seed0.pt
advantage score data/canonical_v1 --checkpoint artifacts/shared_v_seed0.pt \
  --ns 1 5 10 20 --split test --output artifacts/shared_v_scores.csv
```

适配器已通过人工 Parquet 文件的读写、奖励合并和终止检查；真实样本到达后还需确认字段、奖励时间、动作 chunk、任务时限、参考策略及家族标识。
第一版只导入低维状态，不解码视频。只含图像的数据应先增加独立且版本固定的特征提取器，不能自动把图像展平当低维状态。
`train` 只读取 manifest 中的训练轨迹及其奖励；不读取 test、现成 advantage 标签或 oracle。

## 主研究协议：一个固定 V，比较 Advantage 构造

默认 `train --method mc` 仅用完整训练轨迹的折扣回报监督小型 MLP，训练后固定模型。
`score` 对所有跨度复用同一个 checkpoint，并在每行输出同一个 checkpoint SHA256。
输入包含当前状态与剩余时长；没有使用测试质量标签。价值训练与 Advantage 构造的跨度不再混淆。
`--method nstep --n 5` 仅作为辅助训练诊断，不属于主比较。MC 训练拒绝缺失完整回报的截断轨迹。

## 人工轨迹实验

一个有限时域的随机“推进任务”模型：动作可推进、停留或失败；首次到达目标给成功奖励；固定随机参考策略。
状态包含任务进度和剩余时间。精确动态规划可得评价用的真实 V 和动作 advantage，不向训练泄漏。

1. **辅助诊断：学习 V 对比**：同一批训练轨迹、零初始化、相同更新轮数和学习率，用 TD1、N5、N10、N20 目标分别学习表格 V。记录测试状态上的真实 V 误差和学习曲线。
2. **主比较：固定 V 评分对比**：仅用训练轨迹的完整回报拟合一个共享 V，在相同测试轨迹上改变评分跨度，报告对精确 advantage 的误差。

第一项隔离目标传播机制，不代表神经网络性能；第二项受后续随机动作影响，不是同状态反事实排序准确率。
模型使用全训练批次的同步半梯度更新，长跨度目标的计算成本更高，等轮数不等计算时间。
数据按整条独立轨迹家族划分；预留 validation，当前固定实验预算不调参。默认三个随机数据种子。

输出 `results.json`、`metrics.csv`、`learning_curves.csv`、`scores.csv`、分组清单、模型表、样例 NPZ 和 `comparison.png`。
输出目录必须为空，避免覆盖旧实验。完整生成数据保留本地；只推送精简结果和源代码。

## 下一步

下载完成后先运行字段检查，审计少量真实轨迹，再确认映射配置。
混合多个采集策略不能直接声称估计同一固定策略的 V；若用不同策略的示范，需要重新界定估计目标。
训练小型 MLP 的初步诊断不能替代独立质量标签、同状态排序或实际机器人评价。

参考：[RLinf RECAP 官方数据及流程](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)。
RLinf 自身的奖励定义与归一化可以不同；本工具读取原始 reward 并由配置确定折扣，不自动照搬归一化 advantage。


## MC-Q 与独立评价

Q 学习器的输入是当前状态、剩余时长和当前动作，监督为完整训练轨迹的折扣回报。
它与共享 V 使用同一个参考策略、奖励和折扣定义，但额外以动作作为条件。
`mc_q_minus_shared_v` 是一个前瞻式对照分数；它不读取实际未来轨迹，也不充当真实标签。

```bash
advantage train-q data/canonical_v1 --updates 1000 --device cuda --output artifacts/mc_q_seed0.pt
advantage score-qv data/canonical_v1 --q-checkpoint artifacts/mc_q_seed0.pt \
  --v-checkpoint artifacts/shared_v_seed0.pt --split test --output artifacts/qv_scores.csv
advantage synthetic-q --output artifacts/shared_v_q_v1 --seeds 0 1 2
```

`synthetic-q` 在每个种子的同一训练集上拟合 MC-V 和 MC-Q，然后冻结模型。
每个测试 episode 最多选一个 anchor；在相同状态上比较两种候选动作。
TD/N-step 读取独立评分分支，每个候选默认平均 16 次评分；MC-Q 减 V 直接预测。
所有方法都与人工环境精确动态规划的动作价值排序比较；oracle 只用于测试，不监督训练。
同时报告 Advantage 数值误差、Q/V 各自的测试误差、排序准确率与标签可辨覆盖率。
这些方法可读取的未来信息不同，结果是能力对照，不能直接解释为同信息权限下算法优劣。

见 [评价协议](docs/EVALUATION.md)。真实数据中不能把另一个已学习 Q 当真值；需独立分支续跑或明确强度较弱的代理标签。
