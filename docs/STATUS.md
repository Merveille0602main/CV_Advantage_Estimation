# 开发状态

正式仓库与运行位置：`/home/merveille/CV_Advantage_Estimation`，WSL Ubuntu，Conda ML。
Windows 工作目录曾用于中间编写；该中间副本已清理，后续直接在 WSL 修改正式仓库。

主协议：MC 完整回报回归训练共享 V；冻结一个 checkpoint；比较 TD1/N5/N10/N20 的 Advantage 构造。
现有 TD/N-step 学习 V 的表格实验只保留为辅助诊断。
已完成三种子人工实验和15 项数值/Parquet 适配测试。真实 RLinf 数据仍待下载与字段审计。
人工任务有精确 DP 评价参考；真实数据不默认含真实 V。
