# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.9348
- 七维全部一致率：0.5967
- 归一化总分完全一致率：0.6100
- 首轮/复评平均分：0.856349 / 0.858413
- 复评平均分变化：+0.002063
- 归一化总分 MAE：0.020159
- 归一化总分 Spearman：0.5967618866504003

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.9967 | 0.0000 | 0.0033 |
| logic_structure | 0.8167 | 0.4423 | 0.1833 |
| oral_fluency | 0.9433 | 0.4516 | 0.0567 |
| rhetoric_memorability | 0.9333 | 0.5318 | 0.0667 |
| safety_compliance | 0.9667 | 0.4395 | 0.0333 |
| theme_information | 0.8900 | 0.5391 | 0.1100 |
| topic_alignment | 0.9967 | 0.8872 | 0.0033 |

## 主要分歧样本

- `T013-L280`：变化维度 engagement|rhetoric_memorability|topic_alignment；总分差 -0.142857。
- `T055-L700`：变化维度 logic_structure|oral_fluency|safety_compliance；总分差 -0.047619。
- `T016-L280`：变化维度 rhetoric_memorability|safety_compliance；总分差 +0.095238。
- `T012-L450`：变化维度 oral_fluency|theme_information；总分差 +0.095238。
- `T053-L450`：变化维度 rhetoric_memorability|safety_compliance；总分差 +0.095238。
- `T056-L280`：变化维度 logic_structure|theme_information；总分差 +0.095238。
- `T077-L280`：变化维度 oral_fluency|theme_information；总分差 +0.095238。
- `T081-L450`：变化维度 logic_structure|theme_information；总分差 +0.095238。
- `T014-L700`：变化维度 logic_structure|safety_compliance；总分差 +0.095238。
- `T020-L280`：变化维度 logic_structure|safety_compliance；总分差 -0.095238。
- `T014-L280`：变化维度 logic_structure|oral_fluency；总分差 +0.000000。
- `T015-L700`：变化维度 rhetoric_memorability|theme_information；总分差 +0.000000。
- `T030-L280`：变化维度 oral_fluency|theme_information；总分差 +0.000000。
- `T032-L700`：变化维度 logic_structure|oral_fluency；总分差 +0.000000。
- `T006-L280`：变化维度 rhetoric_memorability；总分差 +0.047619。
- `T002-L280`：变化维度 theme_information；总分差 -0.047619。
- `T008-L280`：变化维度 theme_information；总分差 +0.047619。
- `T009-L450`：变化维度 theme_information；总分差 -0.047619。
- `T006-L450`：变化维度 rhetoric_memorability；总分差 +0.047619。
- `T008-L450`：变化维度 theme_information；总分差 -0.047619。
