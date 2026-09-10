# Hy4-preview、Luna 与 GLM：50 份双人 Rubric 对照

结论：Hy4-preview 的七维总分与人工评分的 Spearman 为 −0.0033，接近零；四个维度在 50 份样本中全部满分。本次结果尚不足以支持用 Hy4-preview 可靠地替代人工质量排序。相对 Luna 的差异区间跨零，不应表述为已经证明 Hy4-preview 在总体上显著更差。

日期：2026-09-10。以两位人工的逐维算术平均为参照，计算七个主观维度总分的 Spearman（含并列秩）；不含确定性长度分。

人工七维总分平均：18.62 / 21。50 份样本、350 个主观维度评分。

| 裁判 | 七维均分 / 21 | 相对人工差 | 满分维度比例 | 总分 Spearman | 逐维 MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| hy4-preview | 20.18 | +1.56 | 309/350 (88.3%) | -0.0033 | 0.3314 |
| luna | 17.88 | -0.74 | 195/350 (55.7%) | 0.2362 | 0.3343 |
| glm | 19.50 | +0.88 | 275/350 (78.6%) | 0.0328 | 0.3114 |

按稿件配对重采样 2,000 次（seed=20260910）的百分位 95% 区间：Hy4 Spearman [-0.2916, 0.2694]；Hy4 − Luna 的相关系数差 [-0.5886, 0.0992]。该区间不能消除样本选择偏差或验证跨批次泛化。

Hy4 完整记录的输出 token 合计：2,623,007（包括推理 token；不含中断但未落盘的请求）。

## Hy4 分维度结果

| 维度 | 人工均分 | Hy4 均分 | 分数分布 | Spearman | MAE |
| --- | ---: | ---: | --- | ---: | ---: |
| 选题匹配度 | 2.94 | 3.00 | {3: 50} | 不可定义（无方差） | 0.0600 |
| 主题明确与信息量 | 2.70 | 3.00 | {3: 50} | 不可定义（无方差） | 0.3000 |
| 吸引力 | 2.72 | 3.00 | {3: 50} | 不可定义（无方差） | 0.2800 |
| 口播流畅度 | 2.45 | 2.58 | {2: 21, 3: 29} | 0.0980 | 0.4900 |
| 修辞与记忆点 | 2.80 | 3.00 | {3: 50} | 不可定义（无方差） | 0.2000 |
| 语言逻辑与结构 | 2.37 | 2.92 | {2: 4, 3: 46} | 0.1463 | 0.5900 |
| 合规性 | 2.64 | 2.68 | {2: 16, 3: 34} | 0.1130 | 0.4000 |

## 实验约束与复核

- 使用 Rubric v1.1、Judge v3.3.0、提示词 script-quality-grounded-groups-v3.3；reasoning_effort=high、temperature=0。原接口显式发送 top_p=1；OpenRouter 未声明支持 top_p，因此省略中性默认值，并要求路由支持发送的参数。
- 原始 50 份冻结 trace、两位人工的 run_id、SHA-256 和文案正文均已核对。人工分数不发送给模型。
- Luna、GLM 使用第一轮记录；不取重复评分平均。复算得到 Luna 0.236212、GLM 0.032764，和原报告一致。
- 本次是评分组件对照，不重新生成文案，不发起搜索，也不重新运行完整双门控流程。
- 按用户明确授权复用旧通道已完成的 25 份 hy4-preview 评分，其余 25 份使用 OpenRouter 的 tencent/hy4-preview。两通道记录的原始 fingerprint 均保留；不将旧记录伪装为 OpenRouter 结果。这是混合通道续跑，不是独立的纯 OpenRouter 50 份实验。
- 中断时尚未落盘的部分请求可能已产生计费，其用量不完整。缓存授权与文件哈希见 cache_reuse.json；原始启动与切换信息见 experiment.json。
- 50 份均为高质量基线样本，范围受限；单次评分不能证明跨批次稳定性。观察到更低满分率也不等于与人工排序更一致。
- OpenRouter 首轮 22/25 成功，1 份结构化输出校验失败、2 份接口请求失败；只对缺失的 3 份补试，成功记录直接复用。首轮记录保存在 openrouter-first-attempt。
- 默认单元测试：300 项通过。新增独立 OpenRouter 裁判配置和原生 reasoning 参数适配；未修改默认生成模型或历史评测记录。

## 复现

从仓库根目录运行。评分命令会校验并复用已完成记录；已有结果时无需再次调用模型。

```bash
uv run --no-sync python scripts/run_evaluation.py score --trace-manifest eval/experiments/hy4-preview-human-50-v1/openrouter_manifest.json --evaluators judge --judge-provider openrouter --judge-model-id tencent/hy4-preview --reasoning-effort high --concurrency 12 --output-dir eval/experiments/hy4-preview-human-50-v1/results-openrouter
uv run --no-sync python eval/experiments/hy4-preview-human-50-v1/analyze.py
```

comparison.json 保存统计与各通道 fingerprint；paired_scores.json 保存逐样本对照与原记录路径；results-full/items 与 results-openrouter/items 保存带评分依据的原始结果。
