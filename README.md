# HyScript

> 腾讯犀牛鸟实战项目
>
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

### 100 个测试用例打分实验

正式实验固定使用仓库公开、由本项目构造的 100 个开放式选题；每题最终选定一份经实时网络搜索形成的背景，失败题按 attempt 重试，再生成 280、450、700 字三档文案，共 300 个评分样本。

实验前，先执行不联网的准备检查：

```bash
uv run --no-sync python scripts/run_formal_experiment.py prepare
```

真实实验必须显式分阶段启动，且可在失败后原命令续跑：

```bash
uv run --no-sync python scripts/run_formal_experiment.py generate

uv run --no-sync python scripts/run_formal_experiment.py score

uv run --no-sync python scripts/run_formal_experiment.py report
```

默认并发为 32 个任务、64 个全局 Hy3 请求、8 个 Tavily 请求和 64 个 Judge 请求。
可通过同名 `--*-concurrency` 参数降低并发。
详细产物和人工盲评说明见 [`eval/experiments/formal-100-v1/README.md`](eval/experiments/formal-100-v1/README.md)。

本次正式生成与两轮 Judge 的最终记录合计超过 4,000 万个 Hy3 token；重新联网执行还会产生
Tavily 请求，运行前请先确认服务配额和成本。外部搜索结果及 `hy3` 服务端模型可能变化，
冻结结果可审计、离线汇总可重算，但重新联网执行不保证复现相同数字。


| 指标 | 结果 | 结论 |
| --- | ---: | --- |
| 模型输出 / 完成评分 | 300 / 300 | 三个长度各 100 条，无生成或评分失败 |
| 平均归一化总分 | 0.9969 | 280 条满分，分数高度饱和 |
| 长度 3 / 2 / 1 分 | 291 / 8 / 1 | 700 字档出现 442 字严重不足样本 |
| 好/中/差严格排序率 | 100%（20/20） | 能区分本次人为构造的强退化样本 |
| 成对排序准确率 | 100%（60/60） | 未证明自然候选的细粒度排序能力 |
| 重复评价逐维一致率 | 99.05%（2080/2100） | 高度饱和样本上的表面绝对一致率高 |
| 七维完全一致样本 | 93.67%（281/300） | 合规性是波动最大的维度 |
| 对抗样本错误通过率 | 75%（15/20） | 伪造引用、术语堆砌和自我评分仍会通过 |
| 人工盲评 | 50 项模板待标注 | 当前没有 Judge—人工一致性证据 |

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

### 按领域分数分布

| 领域 | 输出数 | 平均归一化总分 | 最低分 |
| --- | ---: | ---: | ---: |
| 健康 | 18 | 1.0000 | 1.0000 |
| 金融 | 36 | 0.9988 | 0.9583 |
| 教育 | 21 | 0.9980 | 0.9583 |
| 职场 | 33 | 0.9975 | 0.9583 |
| 环境与能源 | 15 | 0.9972 | 0.9583 |
| 公共服务 | 39 | 0.9968 | 0.9167 |
| 消费与社会议题 | 93 | 0.9960 | 0.9167 |
| 科技 | 45 | 0.9954 | 0.9583 |

各领域均分差仅 0.0046，分数饱和程度高，不适合据此进行领域能力排名。

### 按难例标签分数分布

| 难例标签 | 输出数 | 平均归一化总分 | 最低分 |
| --- | ---: | ---: | ---: |
| 时效性 | 51 | 0.9984 | 0.9583 |
| 安全或合规 | 54 | 0.9977 | 0.9167 |
| 开放式权衡 | 87 | 0.9976 | 0.9167 |
| 利益冲突 | 141 | 0.9970 | 0.9583 |
| 弱势群体 | 66 | 0.9962 | 0.9583 |

难例标签允许重叠，各组仍高度接近满分，因此不能据此认定模型已经解决对应难例。

### Judge 内部一致性

| Judge 维度 | 完全一致率 | 二次加权 Kappa | MAE |
| --- | ---: | ---: | ---: |
| 选题匹配度 | 100.00% | 1.0000 | 0.0000 |
| 主题明确与信息量 | 100.00% | 1.0000 | 0.0000 |
| 修辞与记忆点 | 100.00% | 1.0000 | 0.0000 |
| 语言逻辑与结构 | 99.67% | 0.0000 | 0.0033 |
| 吸引力 | 99.33% | 0.4975 | 0.0067 |
| 口播流畅度 | 99.33% | -0.0033 | 0.0067 |
| 合规性 | 95.00% | 0.2613 | 0.0500 |
| **全部 2,100 个维度配对** | **99.05%** | — | — |

两轮评价的表面绝对一致率较高，但大量满分导致 Spearman 仅为 0.2734，排序稳定性证据较弱。



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
