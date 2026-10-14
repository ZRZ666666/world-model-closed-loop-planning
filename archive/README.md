# 历史实验源码归档

这里保存了世界模型闭环规划研究中已经使用过的 23 个 Python 源文件，保留原 `work/` 相对目录结构。源码整理于 2026-10-14；本次只进行了文件读取、复制和文件哈希核对，没有执行项目脚本、模型、训练、仿真或测试。

这些文件用于查阅已完成实验的具体实现。此目录不是已经验证过的“一键运行”软件包。实验结果和适用范围请以仓库中的汇报及结果表为准。

## 从哪里读起

| 实验 | 主要入口 | 作用 |
|---|---|---|
| WM105 原生冻结基线 | `work/wm105_native_phase_baseline_20261004/runtime/run_frozen.py` | 固定底座和原生规划配置的闭环执行 |
| WM108 时间上下文与目标时刻 | `work/wm108_temporal_context_20261004/run_intervention.py` | 真实历史上下文、固定初始目标时刻及两者组合 |
| WM114 残差与计划复用 | `work/wm114_integrated_adaptation_20261005/run_case.py` | 强冻结基线 B、在线残差 BR、计划复用 BG、两者结合 BRG |
| WM114 在线残差 | `work/wm29_20261003/sandwich_core.py` | 基于 Sandwich-Residuals 论文的本地移植，2,512 个可训练参数 |
| WM108 汇总 | `work/wm108_temporal_context_20261004/analyze_saved.py` | 读取原始完成记录并形成配对结果 |
| WM114 汇总 | `work/wm114_integrated_adaptation_20261005/saved_analysis.py` | 读取四臂保存记录并核对、汇总 |

WM108/WM114 的实际世界模型底座来自 Temporal Straightening。JEPA-WMs 在该实现中提供 PushT 环境依赖；不能据此把这些实验称为 JEPA-WMs 世界模型的改进结果。

## 已有运行接口

以下仅记录源代码声明的接口，不表示当前公开归档已具备执行条件：

- WM108 单例：`run_intervention.py --protocol <协议.json> --run <run_name> --output <新输出目录>`。
- WM108 队列：`controller.py --protocol <协议.json>`；`--resume` 仅用于已有完整前缀。
- WM114 单例：`run_case.py --protocol <协议.json> --run <run_name> --output <新输出目录>`。
- WM114 队列：`controller.py --protocol <协议.json> --output <新控制器目录>`；`--resume-completed` 跳过完整完成案例，不能恢复半个案例。
- WM114 历史汇总：`saved_analysis.py --protocol <协议.json> --output <新汇总目录>`。

文件名中的 `audit`、`analysis` 或 `selfcheck` 不意味着本次已运行过它们。`saved_analysis_selfcheck.py` 作为原有分析器验证源码留存，未在本次整理中执行。

## 尚需补齐的复现资产

1. 固定版本的 Temporal Straightening 官方代码、官方配置和可信模型权重；版本和 SHA256 见 `../docs/CODE_PROVENANCE.md`。
2. JEPA-WMs 环境源码及其 Python 依赖。作者两个仓库的环境说明面向不同环境，不能直接视为本地运行环境的完整锁定文件。
3. 历史目标缓存、目标生成记录、输入图像与状态数组、WM105/WM108/WM114 原始协议以及完整完成记录。这些数据未随本源码归档公开。
4. WM108 的协议构建还读取 WM107 的历史完成记录。WM105 的审计器包含读取 WM106 已有结果的分支；对应数据没有打包。
5. 原始代码中沿用特定目录结构、离线资源查找、绝对路径和源文件哈希校验。更换目录后需要明确重建路径配置与协议；不能直接修改历史协议冒充原始文件，也不能关闭校验后声称完成等价复现。

除标准库外，主要依赖包括 PyTorch、NumPy、OmegaConf、Hydra、einops、torchvision 以及 PushT 使用的 gym、pygame、pymunk、shapely、scikit-image、OpenCV 等。本归档没有生成未经运行验证的新依赖锁文件。历史 WM108 依赖回执记录的 Hydra 版本为 1.3.2；其余具体版本需要依据原环境证据补全。

`run_temporal_mpc.py` 使用 `torch.load(..., weights_only=False)` 加载原官方对象式检查点，因此需要可信的对应官方权重，不能换成未知来源检查点。

## 公开整理边界

- 未包含模型权重、数据集、缓存、虚拟环境、二进制依赖、完整轨迹、Slack/私聊记录、账户信息和进程启动回执。
- 未包含 WM114 的 `prepare.py`：它同时写入历史研究交接和协作审核状态，不适合作为公开复现入口。主要运行与统计实现已经保留。
- 未复制第三方模型、规划器或环境实现，只记录依赖出处并保留所查阅的许可证文本。
- 已检查所选源码中私人用户名路径；这 23 个文件均未包含该类路径，所以无需替换，公开副本与原件 SHA256 一致。若后续另行整理含私人路径的配置，应使用 `PUBLIC_USER_HOME` / `PUBLIC_RESEARCH_ROOT` 占位符并单独记录修改。

完整文件清单、来源相对路径和两侧哈希见 `source_manifest.json`。没有修改原始研究目录。
