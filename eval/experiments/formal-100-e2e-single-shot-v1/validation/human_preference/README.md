# 人工偏好测评材料

一位评审者完成同题、同目标字数的 300 对稿件评审。全量选择主编稿 261 对、直接生成稿
39 对，主编偏好率 87.0%。这是单人采用偏好，不代表专家共识或真实传播效果。

## 文件

| 文件 | 内容 |
| --- | --- |
| [human_review.xlsx](human_review.xlsx) | 原始填写工作簿，归档时仅更改文件名，字节保持不变 |
| [ratings.csv](ratings.csv) | “采用偏好”工作表第 2–301 行 A:G，列名转为英文，空单元格记为空字符串，评分及备注未改写 |
| [template.xlsx](template.xlsx) | 原始空白评分模板，包含全部配对正文 |
| [source_mapping.csv](source_mapping.csv) | 盲评编号与 A/B 流程、run_id、trace SHA-256 的对应关系 |
| [protocol.md](protocol.md) | 原始组织说明；其中“配对盲评评分模板.xlsx”对应本目录 template.xlsx |
| [manifest.json](manifest.json) | 原始材料哈希、提取规则和配对结果输入哈希 |
| [summary.json](summary.json) | 离线复算结果，独立于冻结生成轨迹 |

原始组织说明中的评审人数建议属于实验组织指南；本批实际评审人数为一人。
解盲表只用于收齐评审后的分析，开展新的盲评时不要向评审者提供。

归档时已逐条核对评分编号、工作簿正文、600 条冻结 trace 的正文及哈希。
CSV 是便于复算的原始观察记录，不是模型评分结果。

## 独立复算

在仓库根目录运行，不访问网络，不新增模型或搜索调用：

```bash
uv run --no-sync python scripts/report_human_preference.py --check
```

脚本先验证输入哈希与编号，再按 A_source / B_source 解盲，核对 run_id 与正式配对表一致，
复算全量与双方通过门控的子集（293 对），并与 summary.json 精确比较。
去掉 --check 可将结果输出到标准输出，不覆写原始材料。

无需 uv 或第三方依赖也可以运行（Python 3.12+）：

```bash
PYTHONPATH=src python3 scripts/report_human_preference.py --check
```

偏好率分母包含明确选择与“无明显偏好”，排除“无法判断”和缺失；本批全为明确选择。
bootstrap 按 100 个选题有放回抽样，每次保留同题三档稿件，使用 Python 标准库
random.Random(20260909).choices，重复 20,000 次，以线性插值计算 2.5% 和 97.5% 分位数。
主编偏好率区间为 83.3%–90.3%，固定本位评审者，仅表示题目抽样不确定性。

完整解释见[人工偏好测评结论](../../../../../docs/paired-expert-review-2026-09-06.md)。
