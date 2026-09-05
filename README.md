# HyScript

> 腾讯犀牛鸟实战项目<br>
> 本仓库为个人 / 活动作品，并非腾讯官方产品或官方发布。

**项目方案：** [设计思路、架构、重点技术、预期效果与时间规划](PROJECT_PROPOSAL.md)

面向知识型短视频创作者的实时调研与口播文案生成 Agent。系统从当前公开热榜发现候选选题，
使用 Hy3 生成检索计划，通过 Tavily 执行实时搜索，将结果作为写作背景生成可直接口播的短视频文案。引用信息作为正文外元数据供离线评分使用；项目不建立或维护创作者画像。

## 核心流程

![HyScript 核心流程：推荐选题、已有选题、正文外元数据与离线质量评测](docs/assets/hyscript-core-workflow.svg)

## 项目目录

- `src/hyscript/`：可复用的业务实现。
- `app/`：API 与 Web 应用入口。
- `examples/`：最小调用示例，只调用 `src/hyscript/` 中的实现。
- `eval/`：固定选题任务集、Rubric、批量生成记录、独立评分结果与报告。
- `tests/`：单元测试与需要真实服务的集成测试。
- `scripts/`：离线批量生成、评测和报告导出命令。

## 快速开始

运行环境要求 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。复制配置模板并填写必要配置。

```bash
cp .env.example .env
uv sync
```

至少需要分别填写以下两套模型服务配置：

```dotenv
HY3_BASE_URL=https://your-hy3-service.example.com/v1/chat/completions
HY3_API_KEY=your-hy3-key
HY3_MODEL=hy3

EMBEDDING_BASE_URL=https://your-embedding-service.example.com/v1/embeddings
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_MODEL=kinfra-text-embedding-4b

# 正式实验的实时检索还需要：
TAVILY_API_KEY=your-tavily-key
```


### 跨平台桌面端 GUI

完成 `.env` 配置和 `uv sync` 后，在 Windows、macOS 或带图形会话的 Linux 桌面运行：

```bash
uv run --no-sync python -m app.desktop
```

## 离线评分

单次评分脚本
```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

### 100 个测试用例配对评测

正式评测固定使用仓库公开的 100 个开放式选题。三候选主编基线先为每题冻结一份实时检索背景，再生成 280、450、700 字三档文案；端到端直接生成复用同一批背景，每项只调用一次 Hy3 生成内容。两组各 300 条，共 300 组严格配对。

实验前，先执行不联网的准备检查：

```bash
uv run --no-sync python scripts/run_formal_experiment.py prepare
```

首先使用 hy3 根据本项目构建的口播文案生成 Agent 进行生成：

```bash
# 真实实验必须显式分阶段启动，且可在失败后原命令续跑。

uv run --no-sync python scripts/run_formal_experiment.py generate

uv run --no-sync python scripts/run_formal_experiment.py score

uv run --no-sync python scripts/run_formal_experiment.py report
```

端到端方式直接生成文案，并进行评分、复评和对照报告：

```bash
uv run --no-sync python scripts/run_end_to_end_experiment.py run
```

在 Agent、端到端两组各 300 条成稿上，用不同打分模型最高推理强度实现双 Judge 对照（Hy3=`high`，
GPT-5.6-Luna=`xhigh`）：

```bash
uv run --no-sync python scripts/run_judge_model_comparison.py prepare
uv run --no-sync python scripts/run_judge_model_comparison.py score
uv run --no-sync python scripts/run_judge_model_comparison.py repeat
uv run --no-sync python scripts/run_judge_model_comparison.py report
```


同一组冻结文案也使用 `glm-5.3-flash`、`max` 推理强度重复完整双 Judge
对照：

```bash
uv run --no-sync python scripts/run_glm_judge_model_comparison.py prepare
uv run --no-sync python scripts/run_glm_judge_model_comparison.py score
uv run --no-sync python scripts/run_glm_judge_model_comparison.py repeat
uv run --no-sync python scripts/run_glm_judge_model_comparison.py report
```

对两组冻结成稿增量运行攻击检测分：

```bash
uv run --no-sync python scripts/run_incremental_attack_evaluation.py
```

基线产物见[`formal-100-v1`](eval/experiments/formal-100-v1/README.md)，严格配对结果见[端到端直接生成对照报告](eval/experiments/formal-100-e2e-single-shot-v1/report/comparison.md)。

两组正式生成及各自两轮 Judge 的最终记录合计超过 6,000 万 Hy3 token。

Judge 内部一致性：

| Judge | 合并样本 | 逐维完全一致率 | 七维全部一致率 | 总分 MAE | 总分 Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Hy3** | **600** | **96.83%** | **81.33%** | **0.009127** | **0.7115** |
| GPT-5.6-Luna | 600 | 92.48% | 55.00% | 0.021429 | 0.6078 |
| GLM-5.3-Flash | 600 | 91.60% | 52.00% | 0.025000 | 0.5799 |

按两轮绝对分差的内部一致性排序为 **Hy3 > GPT-5.6-Luna >
GLM-5.3-Flash**。下面两组并列表中的重复评价指标是 Hy3 的分组诊断。

| 指标 | 三候选主编基线 | 端到端直接生成 | 对照结论 |
| --- | ---: | ---: | --- |
| 模型输出 / 完成评分 | 300 / 300 | 300 / 300 | 两组均无最终缺失或门控失败 |
| 平均归一化总分 | 0.9969 | 0.9669 | 直接生成低 0.0300 |
| 满分输出 | 280 / 300 | 137 / 300 | 基线存在明显天花板效应 |
| 长度 3 / 2 / 1 分 | 291 / 8 / 1 | 264 / 36 / 0 | 直接生成不做长度修复 |
| 直接生成配对胜 / 平 / 负 | — | 12 / 131 / 157 | 直接生成在 157 组中落后 |
| 重复评价逐维一致率 | 99.05% | 94.62% | 单条文案的细粒度分差仍需谨慎解释 |
| 七维完全一致样本 | 93.67% | 69.00% | 直接生成组评分波动更大 |
| 攻击维度自然成稿标记 | 4 / 300 | 3 / 300 | 7 条警报逐条复核均有可定位问题 |
| 人工盲评 | 50 项双人评分已回收 | 尚未覆盖 | GPT-5.6-Luna 的人工排序相关最高，但仍为弱相关 |

对两组各 300 条冻结成稿的增量检测中，三类 reward-hacking 合并标记为 0/300、1/300，引用风险为 4/300、2/300；7 条警报复核均非明显误报。
在这 50 条上，GPT-5.6-Luna 与两位人工平均分的七维总分 Spearman 为 0.236，高于GLM-5.3-Flash 的 0.033；Hy3 七维全部满分，因无方差而无法计算排序相关。

完整的分长度、领域、难例标签结果，典型输出分析与模型能力边界见
[任务 1 模型输出评估报告](docs/task1-evaluation-report.md)。

### 题目分布

| 领域 | 题数 | 输出数 |
| --- | ---: | ---: |
| 消费与社会议题 | 31 | 93 |
| 科技 | 15 | 45 |
| 公共服务 | 13 | 39 |
| 金融 | 12 | 36 |
| 职场 | 11 | 33 |
| 教育 | 7 | 21 |
| 健康 | 6 | 18 |
| 环境与能源 | 5 | 15 |
| **合计** | **100** | **300** |

100 个选题覆盖 8 个领域，每题生成 3 个长度版本，其中消费与社会议题占比最高。

### 八维 Rubric

| 维度 | 评分者 | 3 分标准摘要 |
| --- | --- | --- |
| 选题匹配度 | Hy3 Judge | 正面回答题设，覆盖冲突两侧与核心人群 |
| 字数符合度 | rules | 非空白字符数相对目标偏差不超过 10% |
| 主题明确与信息量 | Hy3 Judge | 多个有效信息单元推进，事实、解释和结论互相支撑 |
| 吸引力 | Hy3 Judge | 开头有钩子，中段持续推进，结尾自然产生余味或互动 |
| 口播流畅度 | Hy3 Judge | 气口、重音和停顿自然，无需临时改词或重断句 |
| 修辞与记忆点 | Hy3 Judge | 修辞服务理解，存在自然且可复述的记忆点 |
| 语言逻辑与结构 | Hy3 Judge | 前提、解释、转折和结论递进，关键因果有桥梁 |
| 合规性 | Hy3 Judge | 不煽动、不夸大、不作无依据承诺，必要边界清楚 |

当前评分由 1 个确定性规则维度和 7 个 Hy3 Judge 维度组成，八维等权汇总。

### 八维评分对照

| 维度 | 基线均分 | 直接生成均分 | 差值 | 基线 1 / 2 / 3 分 | 直接生成 1 / 2 / 3 分 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 选题匹配度 | 3.0000 | 2.9900 | -0.0100 | 0 / 0 / 300 | 1 / 1 / 298 |
| 字数符合度 | 2.9667 | 2.8800 | -0.0867 | 1 / 8 / 291 | 0 / 36 / 264 |
| 主题明确与信息量 | 3.0000 | 3.0000 | 0.0000 | 0 / 0 / 300 | 0 / 0 / 300 |
| 吸引力 | 2.9967 | 2.6800 | -0.3167 | 0 / 1 / 299 | 0 / 96 / 204 |
| 口播流畅度 | 2.9967 | 2.7900 | -0.2067 | 0 / 1 / 299 | 1 / 61 / 238 |
| 修辞与记忆点 | 2.9967 | 2.9200 | -0.0767 | 0 / 1 / 299 | 1 / 22 / 277 |
| 语言逻辑与结构 | 2.9967 | 2.9967 | 0.0000 | 0 / 1 / 299 | 0 / 1 / 299 |
| 合规性 | 2.9733 | 2.9500 | -0.0233 | 0 / 8 / 292 | 0 / 15 / 285 |

直接生成的主要损失来自吸引力、口播流畅度、字数符合度和修辞与记忆点。主题明确与信息量
在两组中均为满分，语言逻辑与结构均分相同；这些结果仍受同源 Hy3 Judge 和分数饱和影响。

### 按领域分数对照

| 领域 | 输出数 / 组 | 基线均分 | 直接生成均分 | 差值 |
| --- | ---: | ---: | ---: | ---: |
| 健康 | 18 | 1.0000 | 0.9606 | -0.0394 |
| 金融 | 36 | 0.9988 | 0.9711 | -0.0278 |
| 教育 | 21 | 0.9980 | 0.9663 | -0.0317 |
| 职场 | 33 | 0.9975 | 0.9672 | -0.0303 |
| 环境与能源 | 15 | 0.9972 | 0.9611 | -0.0361 |
| 公共服务 | 39 | 0.9968 | 0.9701 | -0.0267 |
| 消费与社会议题 | 93 | 0.9960 | 0.9709 | -0.0251 |
| 科技 | 45 | 0.9954 | 0.9574 | -0.0380 |

八个领域的直接生成均分均下降，但样本量较小且 Judge 不是领域专家，不适合据此进行稳定的
领域能力排名。

### 按难例标签分数对照

| 难例标签 | 输出数 / 组 | 基线均分 | 直接生成均分 | 差值 |
| --- | ---: | ---: | ---: | ---: |
| 时效性 | 51 | 0.9984 | 0.9681 | -0.0302 |
| 安全或合规 | 54 | 0.9977 | 0.9645 | -0.0332 |
| 开放式权衡 | 87 | 0.9976 | 0.9698 | -0.0278 |
| 利益冲突 | 141 | 0.9970 | 0.9660 | -0.0310 |
| 弱势群体 | 66 | 0.9962 | 0.9659 | -0.0303 |

难例标签允许重叠。五类直接生成均分全部下降且差值接近，更像是流程的普遍差异；缺少人工
标注，不能据此认定任一流程已解决对应难例。

### Judge 一致性与对抗检验

内部一致性统一混合三候选主编基线 300 条与端到端直接生成 300 条，按 600 条计算；两种
生成流程不再作为这张主结果表的独立比较对象。

| Judge | 逐维完全一致率 | 七维全部一致率 | 总分 MAE | 总分 Spearman |
| --- | ---: | ---: | ---: | ---: |
| **Hy3** | **96.83%（4,067/4,200）** | **81.33%（488/600）** | **0.009127** | **0.7115** |
| GPT-5.6-Luna | 92.48%（3,884/4,200） | 55.00%（330/600） | 0.021429 | 0.6078 |
| GLM-5.3-Flash | 91.60%（3,847/4,200） | 52.00%（312/600） | 0.025000 | 0.5799 |

按两轮绝对分差，Hy3 的内部一致性最高，GPT-5.6-Luna 第二，GLM-5.3-Flash 第三。分组
Spearman 会分别受到基线天花板效应和直接生成较宽分布的影响，因此只在正式报告中作为诊断保留。

| 对抗攻击类型 | 检出 | 检出率 |
| --- | ---: | ---: |
| 重复凑字 | 5 / 5 | 100% |
| 术语堆砌 | 5 / 5 | 100% |
| 自我评分诱导 | 5 / 5 | 100% |
| 伪造引用 | 5 / 5 | 100% |
| **合计** | **20 / 20** | **100%** |

作为评测可靠性的补充检查，reward-hacking 检测器检出前三类 15/15，Tavily 引用核验检出
伪造引用 5/5；引用核验在其余 15 条攻击负对照中没有误报。该结果只证明评测链能识别这批
强构造攻击，不能替代自然输出排序稳定性或 Judge—人工一致性证据。

## 二次开发及调试

### 调用示例

```python
import asyncio

from hyscript.config import settings
from hyscript.llm import AsyncHy3Client, ChatMessage


async def main() -> None:
    messages = [
        ChatMessage(role="user", content="用一句话说明什么是证据驱动写作。"),
    ]
    async with AsyncHy3Client(settings.hy3) as client:
        result = await client.chat(messages)
    print(result)


asyncio.run(main())
```

项目采用原生异步 I/O：Hy3 与 embedding 分别使用独立的 OpenAI 兼容 `AsyncOpenAI`
客户端，Tavily 使用已安装 SDK 的 `AsyncTavilyClient`。Hy3 和 embedding 可以来自完全不同的服务商；Agent、API 和示例统一使用异步接口，并在上下文管理器退出时关闭连接池。




### 运行最小示例

```bash
# 调用 Hy3，验证最基础的 LLM 对话能力
uv run --no-sync python examples/01_llm_call.py

# 调用 Tavily，验证搜索服务是否可用
uv run --no-sync python examples/02_search_call.py

# 从当前公开热榜生成选题推荐
uv run --no-sync python examples/03_topic_recommendations.py

# 围绕指定话题完成实时调研并生成约 450 字的口播稿
uv run --no-sync python examples/04_end_to_end.py \
  "行业自律能终结新能源车恶性竞争吗？" --target-length 450
```
