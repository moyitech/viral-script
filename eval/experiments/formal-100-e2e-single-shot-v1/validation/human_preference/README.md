# 人工偏好测评材料

两位评审对同一批 300 对稿件完成 600 次判断，分别选择 Agent 主编稿 261 次和 184 次，
合计 445/600（74.2%）。结果支持当前样本与评审条件下更高的人工采用偏好，
不代表事实准确性、成本或传播效果全面更好。600 次判断不是 600 个独立配对。

## 文件

| 文件 | 内容 |
| --- | --- |
| [human_review.xlsx](human_review.xlsx) | 第一位原始工作簿，归档只改名，字节不变 |
| [ratings.csv](ratings.csv) | 第一位原始评分与备注的 CSV 提取 |
| [human_review_02.xlsx](human_review_02.xlsx) | 第二位原始工作簿，归档只改名，字节不变 |
| [ratings_02.csv](ratings_02.csv) | 第二位原始评分与备注的 CSV 提取 |
| [template.xlsx](template.xlsx) | 原始空白模板及全部配对正文 |
| [source_mapping.csv](source_mapping.csv) | 盲评编号、A/B 来源、run_id 与 trace 哈希 |
| [protocol.md](protocol.md) | 原始组织说明，其中原模板文件名对应 template.xlsx |
| [manifest.json](manifest.json) | 两位输入文件对应关系、SHA-256 和提取规则，版本 2.0 |
| [summary.json](summary.json) | 两位及合计的整体采用偏好复算结果 |

CSV 提取各工作簿“采用偏好”第 2–301 行 A:G，列名转为英文，空单元格记为空字符串，
评分和备注未改写。归档时已核对两位样本编号及正文与模板一致；第一位归档时还校验了
全部 600 条冻结 trace 的正文和哈希。材料保留原始可用性与备注，但当前报告只分析整体采用偏好。
解盲表用于收齐结果后的分析，不应提供给新评审者。

## 独立复算

在仓库根目录运行，无需网络，不调用模型或搜索：

```bash
uv run --no-sync python scripts/report_human_preference.py --check
```

也可仅使用 Python 3.12+ 标准库：

```bash
PYTHONPATH=src python3 scripts/report_human_preference.py --check
```

脚本验证两位原始表、CSV 与公共输入的哈希、评审人数、配对数、编号完整性和 run_id，
按 A_source / B_source 解盲，复算每位及合计，并与 summary.json 精确比较。
每位评审内 blind_id 必须唯一，但两位对同一 blind_id 的判断分别保留，不去重、不覆盖。
输出同时记录 300 个配对、100 个选题、600 次判断，不计算评审相关性或按字数展开。
双方通过门控的 293 对作为单独的复核子集，不替代全量主结果。

偏好率分母包含明确选择和“无明显偏好”，排除“无法判断”与缺失。本批两位均为 300 次
明确选择。combined.editorial_preference_rate 是汇总计数的比例，
combined.equal_reviewer_editorial_rate 是各评审偏好率的等权均值；本批分母相同，二者一致。
若后续有效判断数不同，两种量分别保留，不能混用。当前整体汇总不进行 bootstrap 或显著性检验。

去掉 --check 会将结果输出到标准输出，不覆写原始材料。
详见[人工偏好测评结论](../../../../../docs/paired-expert-review-2026-09-06.md)。
