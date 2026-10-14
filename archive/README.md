# 源码索引

本目录收录世界模型闭环规划实验的 **23 个 Python 源文件**，按原 `work/` 相对目录结构组织。内容包括原生冻结基线、时间上下文与目标时刻干预、在线残差与计划复用，以及运行记录和结果审计。目录名保留实验编号，便于与配置、结果表和文件来源对应。

## 阅读入口

建议先读公共模型接口与 WM105 基线，再读 WM108，最后检查 WM114 的组合实验。WM108 与 WM114 的世界模型底座均来自 Temporal Straightening；JEPA-WMs 在这些实验中提供 PushT 环境接口。

| 模块 | 主要入口 | 内容 |
|---|---|---|
| 公共模型接口 | [run_temporal_mpc.py](work/wm13_20261002/runner/run_temporal_mpc.py)、[common.py](work/wm13_20261002/runner/common.py) | 固定版本加载、底座冻结、归一化常量与持续环境连接 |
| WM105：原生冻结基线 | [run_frozen.py](work/wm105_native_phase_baseline_20261004/runtime/run_frozen.py) | 固定目标与预算下的原生闭环执行 |
| WM108：上下文与目标时刻 | [run_intervention.py](work/wm108_temporal_context_20261004/run_intervention.py) | `deadline`、`history3` 及两者组合 |
| WM114：组合实验 | [run_case.py](work/wm114_integrated_adaptation_20261005/run_case.py) | 强冻结基线 B、残差 BR、计划复用 BG、组合 BRG |
| R：在线残差 | [sandwich_core.py](work/wm29_20261003/sandwich_core.py) | Sandwich-Residuals 的本地移植与反馈训练接口 |

WM108 的 deadline 改变动作规划评分位置，history3 提供已执行产生的观测和动作上下文。WM114 的 R 更新轻量残差参数；G 根据反馈决定复用原计划后缀或重新优化动作。这些入口分别承担规划与学习功能，阅读时应区分。

原始动作、动作 token 和重规划轮次采用不同计数单位。阅读执行与统计代码时应保持这些单位一致，并同时查看成功率、封顶交互数以及计算开销。实验队列保留失败案例及停止规则；跳过完整案例继续执行，与恢复半个回合的状态具有不同含义。

## 环境、调度与结果分析

| 用途 | 入口 |
|---|---|
| 目标缓存与信息边界 | [task_contract.py](work/wm105_native_phase_baseline_20261004/runtime/task_contract.py) |
| 环境工作进程与评估连接 | [env_worker.py](work/wm105_native_phase_baseline_20261004/runtime/env_worker.py)、[local_evaluator.py](work/wm105_native_phase_baseline_20261004/runtime/local_evaluator.py) |
| WM108 队列与单例审计 | [controller.py](work/wm108_temporal_context_20261004/controller.py)、[audit_case.py](work/wm108_temporal_context_20261004/audit_case.py) |
| WM108 保存结果汇总 | [analyze_saved.py](work/wm108_temporal_context_20261004/analyze_saved.py) |
| WM114 队列 | [controller.py](work/wm114_integrated_adaptation_20261005/controller.py) |
| WM114 保存结果汇总 | [saved_analysis.py](work/wm114_integrated_adaptation_20261005/saved_analysis.py) |

审计与分析文件读取协议、完成记录和保存数组，检查同目标配对、实际动作数、成功判据及成本字段。其输出依赖完整原始记录，不能仅凭公开摘要替代重算。文件名中的 `selfcheck` 表示原有验证源码，不表示发布版本已在新环境执行测试。

## 目录与来源

23 个文件分布为：WM108 6 个、WM114 4 个、WM29 1 个、WM13 公共接口 3 个、WM105 8 个、WM03 观测连接 1 个。[来源清单](source_manifest.json)逐项记录原相对路径与源文件、公开副本的 SHA256。模型、原生规划器与残差思想的继承关系见[代码来源说明](../docs/CODE_PROVENANCE.md)。

本目录保留历史实现与协议校验方式。第三方实现、模型权重、目标缓存和完整运行记录不随源码一起发布；原配置路径及输入摘要也需要在迁移时对应恢复。安装条件、指标定义、资产缺口和迁移顺序统一见[复现说明](../docs/REPRODUCIBILITY.md)。
