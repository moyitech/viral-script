# Hy3 Judge 重复评价内部一致性

- 冻结轨迹：300 条
- Judge 维度：7 个
- 逐维完全一致率：0.9462
- 七维全部一致率：0.6900
- 归一化总分完全一致率：0.7100
- 首轮/复评平均分：0.967937 / 0.970000
- 复评平均分变化：+0.002063
- 归一化总分 MAE：0.015079
- 归一化总分 Spearman：0.7072324597620894

## 分维度结果

| 维度 | 完全一致率 | 加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| engagement | 0.8267 | 0.6039 | 0.1733 |
| logic_structure | 0.9900 | -0.0045 | 0.0100 |
| oral_fluency | 0.9133 | 0.7284 | 0.0867 |
| rhetoric_memorability | 0.9600 | 0.7826 | 0.0400 |
| safety_compliance | 0.9433 | 0.0860 | 0.0567 |
| theme_information | 1.0000 | 1.0000 | 0.0000 |
| topic_alignment | 0.9900 | -0.0045 | 0.0167 |

## 主要分歧样本

- `T094-L280`：变化维度 engagement|rhetoric_memorability|safety_compliance；总分差 +0.142857。
- `T044-L280`：变化维度 engagement|oral_fluency|rhetoric_memorability；总分差 +0.142857。
- `T037-L280`：变化维度 engagement|oral_fluency|rhetoric_memorability；总分差 +0.142857。
- `T042-L280`：变化维度 engagement|logic_structure|safety_compliance；总分差 +0.047619。
- `T006-L280`：变化维度 engagement|oral_fluency|safety_compliance；总分差 +0.047619。
- `T090-L280`：变化维度 engagement|rhetoric_memorability|safety_compliance；总分差 -0.047619。
- `T022-L280`：变化维度 engagement|oral_fluency；总分差 +0.095238。
- `T066-L280`：变化维度 oral_fluency|topic_alignment；总分差 -0.047619。
- `T057-L280`：变化维度 oral_fluency|rhetoric_memorability；总分差 +0.000000。
- `T013-L280`：变化维度 engagement|oral_fluency；总分差 +0.000000。
- `T092-L280`：变化维度 oral_fluency|safety_compliance；总分差 +0.000000。
- `T087-L280`：变化维度 engagement|oral_fluency；总分差 +0.000000。
- `T029-L280`：变化维度 engagement|safety_compliance；总分差 +0.000000。
- `T037-L700`：变化维度 engagement|safety_compliance；总分差 +0.000000。
- `T021-L280`：变化维度 topic_alignment；总分差 +0.095238。
- `T023-L280`：变化维度 safety_compliance；总分差 +0.047619。
- `T084-L280`：变化维度 rhetoric_memorability；总分差 -0.047619。
- `T009-L280`：变化维度 engagement；总分差 +0.047619。
- `T075-L280`：变化维度 rhetoric_memorability；总分差 -0.047619。
- `T065-L280`：变化维度 engagement；总分差 -0.047619。
