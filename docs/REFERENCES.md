# 参考文献

1. **Wang et al. Temporal Straightening for Latent Planning.** 2026，v3 标注 ICML 2026 Camera Ready。本项目使用其预训练模型和梯度规划器，改进部署时的目标处理与执行策略。[论文](https://arxiv.org/abs/2603.12231v3) · [代码](https://github.com/agentic-learning-ai-lab/temporal-straightening)

2. **Bansal et al. LePlanner: An Iterative Amortized Controller For World Models.** 2026，v1。其到达拖延分析与本项目的期限处理相关；本项目使用运行时动作优化，没有实现其学习式控制器。[论文](https://arxiv.org/abs/2609.13845v1)

3. **Cheng et al. AdaReP: Adaptive Re-Planning under Model Mismatch for Neural World-Model Predictive Control.** 2026，v2。其反馈式重规划为计划复用的相关研究；本项目使用固定阈值，不声称复现或超过其自适应容差方法。[论文](https://arxiv.org/abs/2606.23079v2)

环境接口及附加对照的出处见[代码来源](CODE_PROVENANCE.md)。使用本仓库时，应引用实际采用的上游研究，并注明代码提交版本。
