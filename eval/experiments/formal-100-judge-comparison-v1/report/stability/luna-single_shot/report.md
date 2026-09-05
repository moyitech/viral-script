# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.9148
- 七维全部一致率：0.5033
- 归一化总分完全一致率：0.5567
- 首轮/复评平均分：0.869524 / 0.870635
- 复评平均分变化：+0.001111
- 归一化总分 MAE：0.022698
- 归一化总分 Spearman：0.5917538139028377

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.9433 | 0.5338 | 0.0567 |
| logic_structure | 0.7267 | 0.4377 | 0.2733 |
| oral_fluency | 0.9600 | 0.1416 | 0.0400 |
| rhetoric_memorability | 0.9567 | 0.1120 | 0.0433 |
| safety_compliance | 0.9267 | 0.4681 | 0.0733 |
| theme_information | 0.9033 | 0.5128 | 0.0967 |
| topic_alignment | 0.9867 | 0.5935 | 0.0133 |

## 主要分歧样本

- `T018-L280`：变化维度 logic_structure|oral_fluency|safety_compliance；总分差 +0.047619。
- `T067-L280`：变化维度 engagement|theme_information|topic_alignment；总分差 -0.047619。
- `T056-L280`：变化维度 logic_structure|theme_information；总分差 +0.095238。
- `T043-L450`：变化维度 logic_structure|theme_information；总分差 +0.095238。
- `T034-L280`：变化维度 engagement|logic_structure；总分差 +0.095238。
- `T081-L700`：变化维度 logic_structure|rhetoric_memorability；总分差 +0.095238。
- `T098-L700`：变化维度 engagement|logic_structure；总分差 -0.095238。
- `T076-L700`：变化维度 rhetoric_memorability|safety_compliance；总分差 +0.095238。
- `T053-L700`：变化维度 logic_structure|theme_information；总分差 -0.095238。
- `T051-L450`：变化维度 logic_structure|rhetoric_memorability；总分差 -0.095238。
- `T068-L280`：变化维度 logic_structure|safety_compliance；总分差 -0.095238。
- `T053-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.095238。
- `T084-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.000000。
- `T092-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.000000。
- `T082-L450`：变化维度 logic_structure|safety_compliance；总分差 +0.000000。
- `T075-L280`：变化维度 rhetoric_memorability|theme_information；总分差 +0.000000。
- `T046-L700`：变化维度 logic_structure|rhetoric_memorability；总分差 +0.000000。
- `T083-L280`：变化维度 engagement|theme_information；总分差 +0.000000。
- `T033-L700`：变化维度 logic_structure|rhetoric_memorability；总分差 +0.000000。
- `T049-L280`：变化维度 engagement|logic_structure；总分差 +0.000000。
