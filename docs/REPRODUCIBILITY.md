# 复现说明

本项目使用 [Temporal Straightening](https://github.com/agentic-learning-ai-lab/temporal-straightening) 的冻结世界模型与规划器，PushT 环境来自 [JEPA-WMs](https://github.com/facebookresearch/jepa-wms)。[实现索引](../archive/README.md)列出基线、改进方法及分析入口。

## 固定资产

以下版本和摘要来自[归档加载配置](../archive/work/wm13_20261002/runner/common.py)：

| 资产 | 固定值 |
|---|---|
| 底座提交 | `2c3c7666a69a730042590d548c6731259c4183ac` |
| 权重 SHA256 | `30b34eb854cead76287cb3796db22f9d61f725a658527003d65b3b9124634eaa` |
| 模型配置 SHA256 | `307b43a7bb72ffef7e2e604fd0d5b73b78cf33ed723990fc612738c48d690d05` |

## 必要条件

原入口使用 CUDA、FP32、PyTorch、NumPy、OmegaConf、Hydra 和 PushT 依赖。历史记录确认 Hydra 1.3.2；完整环境锁尚未发布。权重加载使用对象反序列化，须取得对应可信官方资产。

完整重跑还缺官方依赖、权重与配置、目标缓存、原始实验协议及逐轮记录。历史协议含路径和摘要校验，迁移时需明确对应关系；公开结果表不能代替完整输入。

## 评估口径

每轮执行 5 个原始动作，最多 100 步；每次重规划优化 100 次。成功要求联合位置误差小于 20、角度误差小于 π/9。主要报告 100 步成功率与封顶交互数，失败按 100 步计；25 步表现为补充，同时报告计算成本。

各方法按同一目标配对，保留恢复与新增失败。已有窗口属于开发数据，同轨迹窗口不能当作独立样本。

发布整理仅核对文件与既有记录，未执行研究脚本或验证新环境。代码来源与许可见[来源说明](CODE_PROVENANCE.md)。
