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
