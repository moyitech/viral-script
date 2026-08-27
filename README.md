# HyScript

> 腾讯犀牛鸟实战项目

**项目方案：** [设计思路、架构、重点技术、预期效果与时间规划](PROJECT_PROPOSAL.md)

**当前进度：** 已完成 Hy3/Tavily 异步适配、统一配置、冻结轨迹评分内核和基础测试；端到端
Agent、应用入口及“任务集 → 实时生成 → 冻结轨迹”的批量生成阶段仍待实现。

面向知识型短视频创作者的实时调研与口播文案生成 Agent。系统从当前公开热榜发现候选选题，
使用 Hy3 生成检索计划，通过 Tavily 执行多轮实时搜索、整理可追溯证据，并生成可直接口播的
短视频文案。项目不建立或维护创作者画像。

## 核心流程

- 选题发现：实时热榜 → 热点聚合与检索核验 → 选题推荐 → 深入调研 → 生成口播文案。
- 已有选题：用户输入选题 → 深入调研 → 生成口播文案。

两种流程都会保留 Agent 动态生成的搜索词、检索证据和论断—来源映射。在线应用在文案生成后
结束；质量评测由独立离线脚本批量运行，不进入创作者使用流程。

## 目录约定

- `src/hyscript/`：可复用的业务实现。
- `app/`：API 与 Web 应用入口。
- `examples/`：最小调用示例，只调用 `src/hyscript/` 中的实现。
- `eval/`：计划构建的固定选题任务集、Rubric、批量生成中间文件、独立评分结果与报告。
- `tests/`：单元测试与需要真实服务的集成测试。
- `scripts/`：离线批量生成、评测和报告导出命令。

## 评测原则

评测不属于在线 Application。独立 Runner 将从固定选题任务集读取任务，批量调用自定义选题
生成链路；生成期间仍由 Agent 实时规划搜索词并访问搜索服务。Runner 先保存并冻结每条任务的
输入、配置、查询、搜索结果、证据和最终文案，再由评测器读取这些冻结产物进行规则、Judge
和人工评测。评分单独保存并通过 `run_id` 关联，不回写或修改生成中间文件。固定语料只用于
组件级诊断，不替代这种端到端真实检索评测。

## 快速开始

运行环境要求 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。复制配置模板并填写
`HY3_API_KEY`、`TAVILY_API_KEY` 等必要配置：

```bash
cp .env.example .env
uv sync
```

禁止将 API Key、Cookie 或其他密钥提交到仓库。

`hyscript.config.settings` 是唯一读取环境配置的模块。它会从
`pyproject.toml` 向上定位项目根目录并自动读取根目录 `.env`；同名的进程环境变量优先，
因此部署配置可以安全覆盖本地文件。其他模块只接收经过校验的 `settings.hy3`、
`settings.tavily` 或 `settings.runtime`，不自行读取 `.env`。

调用示例：

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

项目采用原生异步 I/O：Hy3 使用 OpenAI 兼容 SDK 的 `AsyncOpenAI`，Tavily 使用已安装
SDK 的 `AsyncTavilyClient`。项目不维护自定义 Hy3 HTTP 传输；Agent、API 和示例统一使用
异步接口，并在上下文管理器退出时关闭连接池。

Tavily 默认使用官方端点。若要改用兼容的第三方 Hub，可以在 `.env` 中设置完整搜索地址：

```dotenv
TAVILY_BASE_URL=https://your-compatible-tavily-host.example.com/search
```

配置层会自动去掉末尾 `/search` 后再交给 Tavily SDK，避免形成 `/search/search`。第三方 Hub
返回的多层 `data` 封装也会统一转换为官方响应结构。第三方 Hub 会接触 API Key、搜索词和
返回内容，应仅在信任其运营方与数据处理方式时启用，并建议使用独立、可撤销且限额的 Key。
评测的所有对比组必须使用同一端点并在轨迹中记录有效地址。

运行最小示例：

```bash
uv run --no-sync python examples/01_llm_call.py
uv run --no-sync python examples/02_search_call.py
```

这些命令会产生真实 API 调用和额度消耗；单元测试不会访问网络。

## 离线评分

评分器只读取已经冻结的生成轨迹，不调用搜索服务，也不会修改轨迹。先用仓库中的脱敏示例
运行确定性规则：

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace eval/traces/example_trace.json \
  --evaluators rules \
  --output-dir /tmp/hyscript-eval
```

需要八维 Hy3 Judge 时显式增加 `judge`：

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

规则评分不读取 `.env`，Hy3 Judge 会产生真实 API 调用和费用。每条结果分别写入
`rules.json`、`hy3_judge.json` 和 `combined.json`，并通过 `run_id` 与轨迹 SHA-256 关联。
`rules.json` 只保存长度、引用覆盖率和确定性门控，不生成八维分数；八维 0～4 分保存在
`hy3_judge.json`。`combined.json` 汇总两类结果，触发重大事实错误、伪造引用、严重合规或
reward hacking 门控后仍保留诊断分，但 `final_score` 会置空。

重复执行时，只有轨迹集合和完整评测指纹都一致才会跳过。指纹包含 Rubric、规则阈值、Judge
模型与提示词版本、推理和上下文参数、采样参数及聚合器版本；不一致时返回
`resume_conflict`，需更换输出目录或显式传入 `--overwrite`。
