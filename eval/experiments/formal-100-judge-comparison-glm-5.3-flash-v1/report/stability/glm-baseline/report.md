# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.8967
- 七维全部一致率：0.4200
- 归一化总分完全一致率：0.4633
- 首轮/复评平均分：0.936032 / 0.934286
- 复评平均分变化：-0.001746
- 归一化总分 MAE：0.030000
- 归一化总分 Spearman：0.5159529696583501

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.9700 | 0.5559 | 0.0300 |
| logic_structure | 0.7800 | 0.4375 | 0.2200 |
| oral_fluency | 0.7900 | 0.4566 | 0.2100 |
| rhetoric_memorability | 0.9833 | 0.5370 | 0.0167 |
| safety_compliance | 0.8033 | 0.4871 | 0.1967 |
| theme_information | 0.9500 | 0.3743 | 0.0500 |
| topic_alignment | 1.0000 | 1.0000 | 0.0000 |

## 主要分歧样本

- `T041-L280`：变化维度 engagement|oral_fluency|theme_information；总分差 +0.142857。
- `T080-L450`：变化维度 oral_fluency|rhetoric_memorability|safety_compliance；总分差 -0.142857。
- `T018-L450`：变化维度 logic_structure|safety_compliance|theme_information；总分差 -0.047619。
- `T002-L280`：变化维度 safety_compliance|theme_information；总分差 -0.095238。
- `T022-L700`：变化维度 safety_compliance|theme_information；总分差 -0.095238。
- `T046-L450`：变化维度 engagement|logic_structure；总分差 +0.095238。
- `T059-L450`：变化维度 safety_compliance|theme_information；总分差 -0.095238。
- `T060-L280`：变化维度 engagement|safety_compliance；总分差 -0.095238。
- `T099-L450`：变化维度 logic_structure|theme_information；总分差 +0.095238。
- `T011-L700`：变化维度 engagement|logic_structure；总分差 -0.095238。
- `T021-L450`：变化维度 logic_structure|safety_compliance；总分差 -0.095238。
- `T023-L280`：变化维度 oral_fluency|safety_compliance；总分差 -0.095238。
- `T022-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.095238。
- `T031-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.095238。
- `T042-L280`：变化维度 logic_structure|oral_fluency；总分差 +0.095238。
- `T048-L450`：变化维度 oral_fluency|safety_compliance；总分差 -0.095238。
- `T055-L450`：变化维度 logic_structure|oral_fluency；总分差 -0.095238。
- `T062-L280`：变化维度 oral_fluency|safety_compliance；总分差 +0.095238。
- `T065-L280`：变化维度 logic_structure|theme_information；总分差 -0.095238。
- `T067-L280`：变化维度 logic_structure|theme_information；总分差 -0.095238。
