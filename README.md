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

## 正式评估：双门控后进行八维评分

当前正式评估和桌面“开始质量评分”统一采用：

```text
冻结成稿 → Reward-hacking 门控 + 引用风险门控 → 通过后执行八维评分
```

任一门控未通过，记录具体原因并停止长度规则和七维 Judge，总分为空且不可判为合格；
检测请求失败属于“评估未完成”，支持续跑，不视为通过。引用门控核验显式出处声明，
必要时调用 Tavily；它不等同于对正文所有事实的全面核验。

生成与评估保持分离。桌面仅在用户主动点击后评估已冻结的 trace；所有结果通过 `run_id`、
trace SHA-256、Rubric 和两个检测器指纹关联。门控或 Judge 版本变化不会复用旧合格结论。

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

`rules,judge` 是默认完整评估，会执行两个前置门控并可能消耗 Hy3、Tavily 配额。
`--evaluators rules` 是不联网的规则组件诊断；`--evaluators judge` 用于独立复评 Judge。
组件诊断不提供完整的合格总分。

### 已有实验结果

100 个开放式选题覆盖 8 个领域，每题生成 280、450、700 字版本。三候选主编与直接生成
两组各 300 条。下表复用已经完成的检测和评分记录，按双门控协议汇总，没有重跑 600 份实验。

| 指标 | 三候选主编 | 直接生成 |
| --- | ---: | ---: |
| 冻结成稿 / 完成门控 | 300 / 300 | 300 / 300 |
| 门控拦截 | 4 | 3 |
| 通过门控、具有八维分数 | 296 | 297 |
| 通过后平均分 | 0.996903 | 0.967172 |
| Reward-hacking 拦截 | 0 | 1 |
| 引用风险拦截 | 4 | 2 |

300 个题目与长度配对中，两侧均通过门控的有 **293 对**；直接生成胜 12、平 128、负 153，
配对平均差为 **-0.029721**。其余 7 对不进入质量分差统计。两个单组均分的分母分别为 296、297，
不能把它们的直接相减当作严格配对差值。

构造攻击集 **20/20 被门控拦截**，没有进入八维评分：Reward-hacking 拦截重复、术语堆砌和
自我评分三类共 15 条；引用门控拦截伪造引用 5 条。这只证明对当前构造样本的识别能力。
自然成稿中 7 条拦截已复核，未观察到明显误报；其余成稿未有独立逐项负标，不报告严格误报率。

- [基线门控后完整结果](eval/experiments/formal-100-v1/results-gated-v1/full_results.csv)
- [直接生成门控后完整结果](eval/experiments/formal-100-e2e-single-shot-v1/results-gated-v1/full_results.csv)
- [门控后的配对汇总](eval/experiments/formal-100-e2e-single-shot-v1/results-gated-v1/comparison.json)
- [攻击集门控结果](eval/experiments/formal-100-v1/validation/attacks-gated-v1/report.md)
- [评估方法、分层统计、Judge 诊断和典型案例](docs/task1-evaluation-report.md)

八维保留选题匹配度、字数符合度、主题与信息量、吸引力、口播流畅度、修辞记忆点、
逻辑结构和合规性。每维 1–3 分，长度由规则评分，其余七维由 Judge 评分，通过门控后等权汇总。
人工盲评及 Hy3、GPT-5.6-Luna、GLM-5.3-Flash 的重复评分用于验证 Judge 组件；
分数饱和和人工排序相关性偏弱的限制仍然存在，不能把高分或门控通过率解释为整体质量保证。

### 复用结果与新实验

现有 600 份成稿、检测结果和评分已保存，日常核查无需重新调用模型或搜索。
需要重建门控后的汇总时，可执行完全不联网的记录重放：

```bash
uv run --no-sync python scripts/replay_gated_evaluation.py
```

它写入独立的 `results-gated-v1` 目录，验证输入哈希和评估器指纹，保留原始 trace 和既有评分。
原始研究、逐项检测和评分记录使用 Git LFS；CSV、清单和汇总可直接查阅。

如需开展新的实验，以下命令会按阶段调用真实服务；不需要为本次门控接入重跑：

```bash
uv run --no-sync python scripts/run_formal_experiment.py prepare
uv run --no-sync python scripts/run_formal_experiment.py generate
uv run --no-sync python scripts/run_formal_experiment.py score
uv run --no-sync python scripts/run_formal_experiment.py report
```

正式评分入口自动采用双门控，并在新目录中续跑。已有检测缓存与指纹匹配的八维分数可复用；
被门控拦截的稿件不会复制或执行八维评分。

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
