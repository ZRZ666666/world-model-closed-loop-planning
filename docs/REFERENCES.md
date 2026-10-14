# 相关工作与引用

本页记录实验底座、最直接的机制近邻及实现依赖。版本用于固定此仓库的比较对象，不将不同论文协议下的成功率作为本项目的直接对照。

## 主要文献

### Temporal Straightening for Latent Planning

Ying Wang, Oumayma Bounou, Gaoyue Zhou, Randall Balestriero, Tim G. J. Rudner, Yann LeCun, Mengye Ren. 2026. [arXiv:2603.12231v3](https://arxiv.org/abs/2603.12231v3)，2026-08-11 修订；版本页面标注 ICML 2026 Camera Ready。

该研究通过潜在轨迹曲率正则改善表征与规划。本仓库使用其预训练模型和官方规划组件，研究运行时的闭环执行，不重新训练该世界模型。[官方代码](https://github.com/agentic-learning-ai-lab/temporal-straightening)

### LePlanner: An Iterative Amortized Controller For World Models

Saksham Bansal, Om Naphade, Chayan Aggarwal, Vrishin M. 2026. [arXiv:2609.13845v1](https://arxiv.org/abs/2609.13845v1)，2026-09-12。

论文 §3 与附录 C 讨论滚动时域重置与到达拖延，用到达与保持损失训练迭代控制器。本文显式期限作用于在线动作优化，未实现其学习式控制器；三帧历史本身也不构成区别性创新。[正文](https://arxiv.org/html/2609.13845v1)

### AdaReP: Adaptive Re-Planning under Model Mismatch for Neural World-Model Predictive Control

Yutian Cheng, Xiaojian Ma, Xianhao Wang, Min Yang, Rongpeng Su, Hangxin Liu, Xi Chen, Shuai Li, Qing Li. 2026. [arXiv:2606.23079v2](https://arxiv.org/abs/2606.23079v2)，2026-09-21 修订；版本页面注明 ICANN 2026 oral。

该工作以预测偏差和局部动力学敏感性调整重规划容差。本文 BG 的固定误差比阈值与复用上限属于相关方向的本地实现，不是对 AdaReP 的官方复现或优越性验证。

## 实现与环境依赖

| 来源 | 在本仓库中的作用 |
|---|---|
| [Temporal Straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) | 世界模型、原生动作规划器、目标函数与预处理 |
| [DINO-WM](https://github.com/gaoyuezhou/dino_wm) | 上游实现继承关系；本仓库未报告 DINO-WM 模型实验 |
| [JEPA-WMs](https://github.com/facebookresearch/jepa-wms) | PushT 环境与包装接口；不是此处的模型底座 |
| [Sandwich-Residuals v1](https://arxiv.org/abs/2609.21740v1) | BR/BRG 的本地论文式残差移植，归属说明见源码及代码来源文档 |

代码、数据和模型分别遵守原来源许可，详见 [CODE_PROVENANCE.md](CODE_PROVENANCE.md)。引用仓库时，应同时引用实际使用的上游论文与实现。

## 如何引用本仓库

GitHub 可读取根目录 [CITATION.cff](../CITATION.cff) 中的软件元数据。为使结果可追溯，请在引用中附上所使用的提交 SHA。该引用指向代码和实验记录，不代表同名论文已发表，也不替代上游文献引用。
