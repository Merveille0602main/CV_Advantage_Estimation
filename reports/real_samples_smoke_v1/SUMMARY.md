# 真实样本流程检查

结论：在 WSL Ubuntu 的 ML 环境、RTX 5060 CUDA 设备上通过。
四条真实轨迹，来自四个不同任务，共 1257 个 transition。
每个任务分别训练 MC-V 和 MC-Q，各更新 200 次，避免混合任务。
保存并重新加载模型后计算 TD1、N5、N10、N20 和 Q 减共享 V，共 6285 条有限分数。
全部四个任务的同轨迹训练损失下降；这只是优化器与数据流检查。

样本 SHA256 全部匹配，状态维度为 8、动作维度为 7，数值有限，动作范围和时间索引通过校验。
三十项回归测试全部通过，其中包括直接读取真实 Parquet 文件的端到端测试。
另修复了源任务编号与清单不一致时未报错，以及失败轨迹未到时限时仍被当成终止的问题。

## 运行

```bash
conda activate ML
cd /home/merveille/CV_Advantage_Estimation
advantage smoke-samples samples --output artifacts/real_samples_smoke_v1 --updates 200 --device cuda
```

输出目录必须为空；重跑请换用新的输出目录。无 GPU 时使用 `--device cpu`。
模型、完整分数和转换数据保留在本地 artifacts，报告见 report.json。

## 结果边界

每个任务仅有一条轨迹，不能创建有效的同任务训练/验证/测试集。
检查特意标记为 `same_episode_smoke_only`，不把相同轨迹包装成独立测试结果。
默认正式训练入口仍会拒绝空训练划分。正式训练需补齐同任务的多条完整成功与失败轨迹。
样本来自 samp830/libero_40_suboptimal_v1，不是 RLinf RECAP 发布的数据集。
源文件无逐步奖励和终止列，按样本清单重建末步成功奖励与时限终止；尚未用模拟器回放独立核验。
当前只使用低维 state，未使用图像，不宣称已得到真实动作质量或泛化准确率。


## 原运行曲线与状态计数

图表读取原先保存的 checkpoint loss 历史与 scores.csv，没有重新训练。

- [V loss 曲线](figures/v_loss.png)
- [TD 与多步 TD 分数](figures/td_nstep.png)
- [V 与同轨迹回报的拟合](figures/v_fit.png)
- [逐次 loss 数据](figures/v_loss_history.csv)
- [状态计数](figures/state_counts.json)

四个任务各一条轨迹，当前状态样本依次为 147、500、110、500，总计 1257。
每条轨迹内按原始八维状态完全相同去重后数量不变；这不代表统计独立，因为它们来自连续时间帧。
V 输入为八维状态加剩余时长，总共九维；每任务独立训练一个 V，各更新 200 次，batch 为 128。

这里使用同一条轨迹的实际 MC 回报拟合 V，再给该轨迹评分。若完全拟合，各跨度 TD 分数都会为零。
因此当前 TD 曲线中的差异反映拟合残差，不能作为动作质量真值或方法优劣的证据。
失败轨迹的奖励和 MC 回报均为零，低训练 loss 本身不能证明动作辨别能力。
