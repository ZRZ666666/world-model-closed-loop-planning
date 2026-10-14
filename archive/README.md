# 实现索引

本项目基于 [Temporal Straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) 的视觉世界模型，研究三项部署改进：利用真实历史观测、固定目标到达时刻，以及根据执行反馈复用计划。轻量残差适应采用 [Sandwich-Residuals](https://arxiv.org/abs/2609.21740) 的本地移植，用于检验组合效果。

## 核心代码

- [原生冻结规划基线](work/wm105_native_phase_baseline_20261004/runtime/run_frozen.py)
- [历史上下文与目标时刻改进](work/wm108_temporal_context_20261004/run_intervention.py)
- [反馈计划复用与残差组合](work/wm114_integrated_adaptation_20261005/run_case.py)
- [轻量残差模块](work/wm29_20261003/sandwich_core.py)
- [模型加载与闭环接口](work/wm13_20261002/runner/run_temporal_mpc.py)
- [环境反馈与成功判据](work/wm105_native_phase_baseline_20261004/runtime/env_worker.py)
- [时序改进结果汇总](work/wm108_temporal_context_20261004/analyze_saved.py)
- [组合方法结果汇总](work/wm114_integrated_adaptation_20261005/saved_analysis.py)

共保留 23 个历史源文件；[来源清单](source_manifest.json)记录原文件与公开副本的摘要。原目录结构用于追溯，不作为方法名称。

第三方模型、权重和完整实验数据未包含在此目录。固定版本、必要资产及复现限制见[复现说明](../docs/REPRODUCIBILITY.md)，引用与许可见[来源说明](../docs/CODE_PROVENANCE.md)。
