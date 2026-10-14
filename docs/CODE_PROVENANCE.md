# 代码来源、依赖与许可

## 本仓库整理的内容

本次从已有研究工作目录选择 23 个实验、适配和分析源文件，保留其 `work/` 相对路径，存放于 `archive/work/`。所有公开副本与原件文件哈希相同；清单见 `archive/source_manifest.json`。这说明复制内容一致，不代表代码已经在新仓库重新运行或得到新的复现结果。

“本地实现”指研究过程中编写的实验连接、记录、审计和方法移植代码。它不意味着相关算法思想全部原创，也不改变其调用的第三方代码和模型的归属。

| 归档目录 | 文件数 | 本地工作与继承关系 |
|---|---:|---|
| `wm108_temporal_context_20261004` | 6 | 时间上下文/固定目标时刻干预、串行调度、协议校验、单例审计和保存结果汇总；底座规划器仍来自 Temporal Straightening |
| `wm114_integrated_adaptation_20261005` | 4 | 四臂组合实验、计划复用逻辑、运行记录及保存结果统计；在线残差继承下述论文式实现 |
| `wm29_20261003` | 1 | `sandwich_core.py`：依据 Sandwich-Residuals 的本地实现和架构适配 |
| `wm13_20261002/runner` | 3 | 固定官方模型的加载、输入边界、持续环境接口和来源哈希核对 |
| `wm105_native_phase_baseline_20261004` | 8 | 原生冻结基线运行、环境适配、参考缓存契约、审计与保存结果分析 |
| `wm03_20261002` | 1 | `run_native_pusht.py`：本地运行连接与观测转换；后续环境工作进程引用它的观测转换函数 |

`archive/work/wm114_integrated_adaptation_20261005/prepare.py` 没有收入公开包，因为原文件兼有协作状态更新和私聊引用。没有以新代码替换它或声称具备完整配置生成入口。

## Temporal Straightening 底座

- 官方仓库：https://github.com/agentic-learning-ai-lab/temporal-straightening
- 项目主页：https://agenticlearning.ai/temporal-straightening/
- 本地加载器锁定提交：`2c3c7666a69a730042590d548c6731259c4183ac`。
- 本地模型目录标识：`pusht_aggmlpcos1e-1_resnetplus_dim8_hw14_sgTrue_lr1e-5`。
- 检查点 SHA256：`30b34eb854cead76287cb3796db22f9d61f725a658527003d65b3b9124634eaa`。
- Hydra 配置 SHA256：`307b43a7bb72ffef7e2e604fd0d5b73b78cf33ed723990fc612738c48d690d05`。
- 本地官方仓库 `LICENSE`：MIT，Copyright (c) 2025 gaoyuezhou。原文副本：`licenses/TemporalStraightening_MIT.txt`。

世界模型、原生 GD/MPC 规划器、目标函数、预处理与相关数据接口属于该官方依赖，不作为本地原创模块发表。官方实现继承 DINO-WM 的相关基础，应同时保留其引用：https://github.com/gaoyuezhou/dino_wm 。本次没有复制第三方实现和检查点到公开包。

## JEPA-WMs 与 PushT 环境

- 官方仓库：https://github.com/facebookresearch/jepa-wms
- 历史工作进程从该仓库导入 PushT 环境、gym 包装及反馈接口。
- 仓库根 `LICENSE` 为 Creative Commons Attribution-NonCommercial 4.0；文本副本：`licenses/JEPA_WMs_repository_CC_BY_NC_4_0.txt`。
- 所查阅的 `evals/simu_env_planning/envs/pusht_env/pusht_env.py` 文件头另明确声明 Copyright (c) Facebook, Inc. and its affiliates、Inspired from `gaoyuezhou/dino_wm`、Licensed under MIT License。仓库级声明与文件级声明应分别保留，不能把所有文件一概重新许可。
- 本次没有打包 JEPA-WMs 环境源码，也没有将其世界模型结果归属到 WM108/WM114。

## Sandwich-Residuals 的继承关系

- 论文：https://arxiv.org/abs/2609.21740
- 本地实现依据版本：v1，III-A 式 4–6、III-B 式 7–13；来源说明已保留在 `sandwich_core.py` 顶部。
- 本地移植包含两个共享 patch MLP、动作线性残差和本体线性残差，共 2,512 个可训练参数；与 Temporal visual8 底座相适配。
- 源码自身明确说明：这是论文式本地实现，不是作者代码，也不是对其 CEM 基准的完整复现。
- WM114 将视觉末层改为精确零初始化，以获得初始恒等映射；该修改在 `run_case.py` 中明确写出。

残差结构必须归功于已有工作。WM114 的四臂对照用于检查它与当前强冻结规划及反馈计划复用的组合效果，不能据此宣称残差结构本身原创。

## 本地代码的许可边界

这里的第三方许可证副本用于记录外部依赖条款，不自动将本地代码统一重新许可。仓库维护者为本地整理文件另行选择许可证时，应保留上述来源与继承说明，且不得把第三方代码或检查点更改为自己的原创作品。

公开包未包含数据集、检查点、二进制环境或私人通信。后续复现需要从原作者渠道取得依赖，遵守其各自许可；本次没有运行项目或验证新环境安装。
