# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.9352
- 七维全部一致率：0.6200
- 归一化总分完全一致率：0.6367
- 首轮/复评平均分：0.935079 / 0.933810
- 复评平均分变化：-0.001270
- 归一化总分 MAE：0.020000
- 归一化总分 Spearman：0.6518847265257568

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.9233 | 0.7173 | 0.0767 |
| logic_structure | 0.8967 | 0.5051 | 0.1033 |
| oral_fluency | 0.9067 | 0.6422 | 0.0933 |
| rhetoric_memorability | 0.9667 | 0.7045 | 0.0333 |
| safety_compliance | 0.8733 | 0.5294 | 0.1267 |
| theme_information | 0.9867 | 0.3274 | 0.0133 |
| topic_alignment | 0.9933 | 0.4966 | 0.0067 |

## 主要分歧样本

- `T083-L280`：变化维度 logic_structure|rhetoric_memorability|safety_compliance|theme_information；总分差 +0.190476。
- `T044-L280`：变化维度 oral_fluency|safety_compliance；总分差 -0.095238。
- `T048-L280`：变化维度 engagement|rhetoric_memorability；总分差 +0.095238。
- `T086-L280`：变化维度 engagement|rhetoric_memorability；总分差 -0.095238。
- `T085-L280`：变化维度 logic_structure|theme_information；总分差 -0.095238。
- `T089-L280`：变化维度 engagement|oral_fluency；总分差 -0.095238。
- `T098-L280`：变化维度 engagement|rhetoric_memorability；总分差 -0.095238。
- `T062-L450`：变化维度 engagement|logic_structure；总分差 +0.095238。
- `T090-L280`：变化维度 engagement|rhetoric_memorability；总分差 -0.095238。
- `T066-L450`：变化维度 logic_structure|safety_compliance；总分差 -0.095238。
- `T059-L450`：变化维度 logic_structure|oral_fluency；总分差 -0.095238。
- `T093-L280`：变化维度 engagement|oral_fluency；总分差 -0.095238。
- `T060-L700`：变化维度 oral_fluency|rhetoric_memorability；总分差 -0.095238。
- `T021-L450`：变化维度 engagement|safety_compliance；总分差 -0.095238。
- `T035-L700`：变化维度 oral_fluency|safety_compliance；总分差 -0.095238。
- `T042-L280`：变化维度 engagement|logic_structure；总分差 +0.000000。
- `T033-L700`：变化维度 oral_fluency|safety_compliance；总分差 +0.000000。
- `T052-L280`：变化维度 oral_fluency|safety_compliance；总分差 +0.000000。
- `T040-L280`：变化维度 logic_structure|oral_fluency；总分差 +0.000000。
- `T082-L280`：变化维度 logic_structure|safety_compliance；总分差 +0.000000。
