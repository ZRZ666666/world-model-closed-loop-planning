# 代码来源与许可

本仓库保留 23 个实验、环境适配和分析源文件，来源摘要见[源码清单](../archive/source_manifest.json)。公开源码与归档来源的 SHA256 一致。

## 模型与规划基础

[Temporal Straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) 提供世界模型、原生 GD/MPC 规划器、目标函数与预处理。固定提交为 2c3c7666a69a730042590d548c6731259c4183ac，权重及配置摘要见[复现说明](REPRODUCIBILITY.md)。上游继承 [DINO-WM](https://github.com/gaoyuezhou/dino_wm) 的相关实现。

官方代码许可证为 MIT，Copyright (c) 2025 gaoyuezhou。[许可证副本](../licenses/TemporalStraightening_MIT.txt)

## 环境与附加对照

[JEPA-WMs](https://github.com/facebookresearch/jepa-wms) 提供 PushT 环境和包装接口。其仓库根许可证为 CC BY-NC 4.0；所用 PushT 环境文件另有 MIT 声明，二者须按实际文件分别保留。本仓库没有分发其环境源码或模型。[仓库级许可证副本](../licenses/JEPA_WMs_repository_CC_BY_NC_4_0.txt)

附加残差对照采用 [Sandwich-Residuals v1](https://arxiv.org/abs/2609.21740v1) 的本地论文式移植，包含 2,512 个可训练参数及恒等初始化适配，不是作者官方实现。[来源说明与实现](../archive/work/wm29_20261003/sandwich_core.py)

## 发布范围

发布内容为本地实验连接、规划干预、记录与分析代码，以及保存结果。检查点、完整任务缓存、运行环境和内部协调脚本不随包提供；完整运行要求见[复现说明](REPRODUCIBILITY.md)。

本地代码尚未授予统一开源许可证。第三方许可证副本仅记录对应依赖的条款，不自动适用于整个仓库；使用时应保留原始权利与引用。
