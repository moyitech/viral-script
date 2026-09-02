# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.9905
- 七维全部一致率：0.9367
- 归一化总分完全一致率：0.9367
- 首轮/复评平均分：0.998095 / 0.997143
- 复评平均分变化：-0.000952
- 归一化总分 MAE：0.003175
- 归一化总分 Spearman：0.27343751729640137

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.9933 | 0.4975 | 0.0067 |
| logic_structure | 0.9967 | 0.0000 | 0.0033 |
| oral_fluency | 0.9933 | -0.0033 | 0.0067 |
| rhetoric_memorability | 1.0000 | 1.0000 | 0.0000 |
| safety_compliance | 0.9500 | 0.2613 | 0.0500 |
| theme_information | 1.0000 | 1.0000 | 0.0000 |
| topic_alignment | 1.0000 | 1.0000 | 0.0000 |

## 主要分歧样本

- `T013-L280`：变化维度 engagement|safety_compliance；总分差 -0.095238。
- `T015-L450`：变化维度 safety_compliance；总分差 -0.047619。
- `T022-L280`：变化维度 safety_compliance；总分差 -0.047619。
- `T020-L700`：变化维度 oral_fluency；总分差 +0.047619。
- `T023-L700`：变化维度 safety_compliance；总分差 -0.047619。
- `T034-L280`：变化维度 logic_structure；总分差 +0.047619。
- `T038-L280`：变化维度 safety_compliance；总分差 -0.047619。
- `T037-L280`：变化维度 safety_compliance；总分差 +0.047619。
- `T029-L700`：变化维度 oral_fluency；总分差 -0.047619。
- `T046-L700`：变化维度 safety_compliance；总分差 -0.047619。
- `T054-L280`：变化维度 engagement；总分差 -0.047619。
- `T050-L450`：变化维度 safety_compliance；总分差 +0.047619。
- `T051-L280`：变化维度 safety_compliance；总分差 -0.047619。
- `T061-L450`：变化维度 safety_compliance；总分差 -0.047619。
- `T066-L280`：变化维度 safety_compliance；总分差 +0.047619。
- `T070-L280`：变化维度 safety_compliance；总分差 +0.047619。
- `T091-L280`：变化维度 safety_compliance；总分差 -0.047619。
- `T096-L450`：变化维度 safety_compliance；总分差 +0.047619。
- `T097-L450`：变化维度 safety_compliance；总分差 -0.047619。
