# 保存结果来源与公开副本说明

整理日期：2026-10-14。

本目录根据既有结果文件制作公开归档。本次仅进行了文件读取、复制、路径脱敏和 SHA256 校验，没有运行实验、模型、训练、项目分析脚本，也没有重新计算统计结果。

## 范围与证据级别

- WM108：保留四臂的完整 200 行逐案例表（其中 150 项新执行、50 项复用基线）、150 行配对表、50 行因子对照表、完整失败表以及既有汇总与完成记录。50 个窗口来自 20 条已消费开发轨迹。
- WM114：保留四臂的完整 64 行逐案例表、48 行配对表、既有汇总、完成记录与历史机械核对产物。16 个窗口来自 11 条已消费开发轨迹。源目录没有独立失败 CSV，本归档不生成新的统计表；全部失败保留于 all_cases.csv、summary.json 及 report.txt 中。
- 所有四臂均保留，包含无收益、退化条件与失败案例；没有按结果筛除表格行。
- 文件名含 historical 的审计产物来自过去的检查。报告中的“本次核验”“独立重算”指原报告生成当时，不表示 2026-10-14 重新执行了这些检查。
- 本次文件 SHA256 仅用于核对复制来源与公开副本的字节内容，不验证物理轨迹、模型结果或统计结论的正确性。
- 没有包含大型原始轨迹数组、权重、大型输入哈希清单、完整 manifest、私人协作消息或自动化调度。已保存结果具备来源线索，完整重跑仍需要原始数据、依赖与模型。

## 脱敏规则

1. 原研究目录的绝对路径（包括其历史 C 盘位置、D 盘位置及 JSON 转义形式）替换为 `<RESEARCH_ROOT>/`，保留其后的相对路径。
2. WM108 历史报告内含用户目录的 Python 可执行文件路径替换为 `<PYTHON>`。
3. 数值、任务 ID、轨迹分组、成功与失败标记、方法名及既有哈希值不修改；没有重算或调整指标。
4. 原始报告和 terminal 内嵌的 SHA256 仍指向原始文件。经脱敏的公开副本哈希可能不同，应以本表两列区分，不能用旧哈希直接验证脱敏后的文件。
5. 全部原件保持原样；下表的“原始相对路径”均相对于 `<RESEARCH_ROOT>/`。

## 文件级来源

| 原始相对路径 | 公开归档路径 | 原始 SHA256 | 公开副本 SHA256 | 处理 |
|---|---|---|---|---|
| `work/wm108_temporal_context_20261004/final_v1/summary.json` | `results/wm108/summary.json` | `1d0778e079e4a9833fdc6f6fd0eace755dfd211a884da5baa6eb3ffe8207a3b6` | `1d0778e079e4a9833fdc6f6fd0eace755dfd211a884da5baa6eb3ffe8207a3b6` | 字节一致 |
| `work/wm108_temporal_context_20261004/final_v1/all200_tasks.csv` | `results/wm108/all200_tasks.csv` | `22bd4714616ecd9f18cf9754906f389298c6247a3af99862f7e28bd13d1729b4` | `22bd4714616ecd9f18cf9754906f389298c6247a3af99862f7e28bd13d1729b4` | 字节一致 |
| `work/wm108_temporal_context_20261004/final_v1/all150_paired.csv` | `results/wm108/all150_paired.csv` | `5807c119eaf08a9e78c6fcd730445f6778632f5035c2abe253d5b29f71063bc1` | `5807c119eaf08a9e78c6fcd730445f6778632f5035c2abe253d5b29f71063bc1` | 字节一致 |
| `work/wm108_temporal_context_20261004/final_v1/all50_factorial_contrasts.csv` | `results/wm108/all50_factorial_contrasts.csv` | `1d7b87300ecf1d88613f2c39bb2d6333f12cf5bf1ebae532037f7be6f2e44002` | `1d7b87300ecf1d88613f2c39bb2d6333f12cf5bf1ebae532037f7be6f2e44002` | 字节一致 |
| `work/wm108_temporal_context_20261004/final_v1/all_failed_cases.csv` | `results/wm108/all_failed_cases.csv` | `2f5bb0546befb5f3c2953e03e2e11a6f5d4ef09b0ac55966aeca5a4140281295` | `2f5bb0546befb5f3c2953e03e2e11a6f5d4ef09b0ac55966aeca5a4140281295` | 字节一致 |
| `work/wm108_temporal_context_20261004/final_v1/terminal.json` | `results/wm108/terminal.json` | `3eb3ed3f872fcf3728b962c0bf691c74571cff3b86bfb8bd2d6efadebeb543d7` | `b91c321a51d39af42118fe997acd3eb163fb5987a3a5646cc71068fb6dabb326` | 仅路径脱敏 |
| `outputs/WM108完整对照_20261004/report.txt` | `results/wm108/report.txt` | `6280d47ac2477be7f51cc46050c016cb5d835914661225d9c428e4de22429110` | `7cb492294d7c4adf28f8704dfd1dad67ef0707c2be3a87846a00d537c86bb899` | 仅路径脱敏 |
| `outputs/WM108完整对照_20261004/verification.json` | `results/wm108/historical_verification.json` | `69916847c814050c8a564c8a886d365d3377e11324fcff79d3cc2a2f7aa9d98b` | `72bcc585b6335d866854e8835c2357320678c4bd407468c217d17e137fd8f20d` | 仅路径脱敏 |
| `work/wm114_integrated_adaptation_20261005/controller_v1/final_analysis/summary.json` | `results/wm114/summary.json` | `c0e11e9328ec648a20d3fa5748d947fd17469f3cb0dc0eb14ab95f8a70bfec1b` | `c0e11e9328ec648a20d3fa5748d947fd17469f3cb0dc0eb14ab95f8a70bfec1b` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/controller_v1/final_analysis/all_cases.csv` | `results/wm114/all_cases.csv` | `542f529061d8a0afe982e4a0561bb24385ce93e6a297c9523c3435af5a163369` | `189ff777a44602994d1cbaa2deb26f61b37309d9d53aa10de38a36df09dd92dd` | 仅路径脱敏 |
| `work/wm114_integrated_adaptation_20261005/controller_v1/final_analysis/paired_vs_B.csv` | `results/wm114/paired_vs_B.csv` | `1e07bc5e2735574047ac8c7df9a712ee3fc68b0752f4f03d2b795070a2fa8b46` | `1e07bc5e2735574047ac8c7df9a712ee3fc68b0752f4f03d2b795070a2fa8b46` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/controller_v1/final_analysis/report.txt` | `results/wm114/report.txt` | `fd760f843f3f6b73b920dd61165955686567772d2125eea2a53bf7ad911006e8` | `fd760f843f3f6b73b920dd61165955686567772d2125eea2a53bf7ad911006e8` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/controller_v1/final_analysis/terminal.json` | `results/wm114/terminal.json` | `0703aa519787cd924d3cf824074b20919dda4d42d400f5d03134f23929b109ec` | `0703aa519787cd924d3cf824074b20919dda4d42d400f5d03134f23929b109ec` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/final_count_review_v1/report.txt` | `results/wm114/historical_count_review.txt` | `0728cd267db71f51144cc5f1c91ebff65fe140210e967bda81c53b9bd93ea1cb` | `0728cd267db71f51144cc5f1c91ebff65fe140210e967bda81c53b9bd93ea1cb` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/final_count_review_v1/terminal.json` | `results/wm114/historical_count_review_terminal.json` | `5d09339adc7613ab24c7b49548b0e2ec46226b8b5e24063025d6daa06f94cba1` | `5d09339adc7613ab24c7b49548b0e2ec46226b8b5e24063025d6daa06f94cba1` | 字节一致 |
| `work/wm114_integrated_adaptation_20261005/final_count_review_v1/recomputed_arms.csv` | `results/wm114/historical_recomputed_arms.csv` | `58fbafe31d07e1dea4dedcd1ff41a17a9860d15a534b879a49d20178c9260fcc` | `58fbafe31d07e1dea4dedcd1ff41a17a9860d15a534b879a49d20178c9260fcc` | 字节一致 |
