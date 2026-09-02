# HyScript

> 腾讯犀牛鸟实战项目

**项目方案：** [设计思路、架构、重点技术、预期效果与时间规划](PROJECT_PROPOSAL.md)

**当前进度：** 已完成 NewsNow 热榜获取、20 条选题推荐、Hy3/Tavily 异步适配、选中选题后的
实时背景检索、口播文案生成、冻结轨迹适配、批量生成脚本、离线评分内核，以及可在
Windows、macOS、Linux 源码运行的 pywebview 桌面 GUI；Web/API 入口仍待实现。

面向知识型短视频创作者的实时调研与口播文案生成 Agent。系统从当前公开热榜发现候选选题，
使用 Hy3 生成检索计划，通过 Tavily 执行实时搜索，将结果作为写作背景生成可直接口播的短视频文案。
引用信息作为正文外元数据供离线评分使用；项目不建立或维护创作者画像。

## 核心流程

- 选题发现：NewsNow 实时热榜 → `kinfra-text-embedding-4b`（阈值 `0.72`）事件级去重 →
  选取最多 40 个事件 → 4 个 Hy3 `high` 批次并发生成（每批 5 条）→ 20 个选题 →
  用户选择 → 3 个并发初始查询（总计最多 5 次搜索）→ 整理写作背景 → 生成口播文案。
- 已有选题：用户输入选题 → 深入调研 → 生成口播文案。

两种流程都会保留 Agent 动态生成的搜索词、搜索结果和成稿实际选用的引用 ID。引用不进入口播正文；
质量评测由独立离线脚本读取冻结轨迹运行，不进入创作者的生成流程。

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
输入、配置、查询、搜索结果、引用元数据和最终文案，再由评测器读取这些冻结产物进行规则、Judge
和人工评测。评分单独保存并通过 `run_id` 关联，不回写或修改生成中间文件。固定语料只用于
组件级诊断，不替代这种端到端真实检索评测。

## 快速开始

运行环境要求 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。复制配置模板并填写
`HY3_API_KEY`、`EMBEDDING_API_KEY`、`TAVILY_API_KEY` 等必要配置。Hy3 与
embedding 分别使用自己的 `BASE_URL` 和 `API_KEY`：

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
```

禁止将 API Key、Cookie 或其他密钥提交到仓库。

### 跨平台桌面 GUI

完成 `.env` 配置和 `uv sync` 后，在 Windows、macOS 或带图形会话的 Linux 桌面运行：

```bash
uv run --no-sync python -m app.desktop
```

应用启动后会在“今天想讲什么”页面后台静默获取 20 条推荐选题，也可以直接输入自定义
选题；点击“选题推荐”后才显示准备日志或推荐列表。字数滑杆支持 100～1000 字自由选择，
并在 280、450、700 字设置磁吸档位。生成、成稿和评分分别使用独立界面，原生窗口会随
阶段调整尺寸。生成期间会实时显示 HyScript 日志并允许取消；成稿会先冻结到
`HYSCRIPT_RUNS_DIR`，随后才可由用户主动运行正式评分。评分结果默认保存在
`HYSCRIPT_EVALUATION_DIR`，相同 trace 和评测配置会复用已有结果，不重复调用 Judge。

- Windows 使用系统 WebView2；若系统未预装，请安装 Microsoft Edge WebView2 Runtime。
- macOS 使用系统 WKWebView。
- Linux 通过 `pywebview[qt]` 使用 Qt backend，并要求 `DISPLAY` 或 `WAYLAND_DISPLAY`
  图形会话。项目依赖使用平台 marker，不会在 Windows/macOS 安装 Linux Qt extra。

推荐、检索、生成和首次 Judge 评分都会访问真实服务并消耗 API 配额。默认单元测试仍全部
使用 mock/fake，不访问网络。

`hyscript.config.settings` 是唯一读取环境配置的模块。它会从
`pyproject.toml` 向上定位项目根目录并自动读取根目录 `.env`；同名的进程环境变量优先，
因此部署配置可以安全覆盖本地文件。其他模块只接收经过校验的 `settings.hy3`、
`settings.embedding`、`settings.topic_recommendation`、`settings.research`、`settings.script_generation`、
`settings.tavily`、`settings.newsnow` 或 `settings.runtime`，不自行读取 `.env`。

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

项目采用原生异步 I/O：Hy3 与 embedding 分别使用独立的 OpenAI 兼容 `AsyncOpenAI`
客户端，Tavily 使用已安装 SDK 的 `AsyncTavilyClient`。Hy3 和 embedding 可以来自完全不同的
服务商；Agent、API 和示例统一使用异步接口，并在上下文管理器退出时关闭连接池。

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
uv run --no-sync python examples/03_topic_recommendations.py
uv run --no-sync python examples/04_end_to_end.py \
  "行业自律能终结新能源车恶性竞争吗？" --target-length 450
```

这些命令会产生真实的 Hy3、embedding 或 Tavily API 调用；单元测试不会访问网络。一次默认选题推荐
包含 1 次独立 embedding 服务请求、本地余弦连通分量去重和 4 次并发的 Hy3 高推理生成请求；生成
请求禁用客户端超时且不设置 `max_tokens`。选题推荐阶段只把热榜作为当前关注信号；页面统一提示“选中后需
补充背景”，不在每条推荐中重复保存固定状态字段。用户选中后，`ResearchAgent` 生成查询并
实时搜索；搜索结果以背景材料提供给 `ScriptAgent`，不再经过 claim、标题链或 Grounding 复核。
正文以吸引力、信息价值和口播成稿质量为主要目标。实际采用的来源 ID、标题、URL 和背景摘录
作为正文外元数据保存在冻结 trace 中，供离线评分使用，不要求出现在正文。调研和写稿均使用
Hy3 `high`，禁用客户端超时且不设置 `max_tokens`。

端到端入口会统计本次 Tavily 尝试、成功和失败请求数，并汇总 Hy3 服务端返回的输入、输出、
总量、推理及缓存输入 token。控制台显示运行汇总；冻结 trace 的 `token_usage` 保存汇总值，
`lineage.llm_calls` 保存每次调用的阶段、重试次数、请求 ID 和原始 `usage`。如果某次失败请求
没有服务端 usage，统计会明确显示 `reported_usage_calls`，不会自行估算缺失 token。

## 离线评分

评分器默认直接使用项目最初的 `script_quality_v1.json` 七维 Judge 标准，并把独立的长度规则分数
与七维得分相加后归一化。评分只读取已经冻结的生成轨迹，
不调用搜索服务，也不会修改轨迹。引用元数据只在评分上下文中提供，不要求正文显示引用，也不
执行标题链、Grounding 或证据可追溯性满分门控。先用仓库中的脱敏示例
运行确定性规则：

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace eval/traces/example_trace.json \
  --evaluators rules \
  --output-dir /tmp/hyscript-eval
```

需要七维 Hy3 Judge 时显式增加 `judge`：

```bash
uv run --no-sync python scripts/run_evaluation.py score \
  --trace-dir eval/traces/runs/<batch-id> \
  --evaluators rules,judge \
  --output-dir eval/results/runs/<evaluation-id> \
  --concurrency 2
```

规则评分不读取 `.env`，Hy3 Judge 会产生真实 API 调用和费用。每条结果分别写入
`rules.json`、`hy3_judge.json` 和 `combined.json`，并通过 `run_id` 与轨迹 SHA-256 关联。
`rules.json` 保存独立的长度分数、引用覆盖率和确定性检查；原始七维 1～3 分保存在
`hy3_judge.json`。`combined.json` 将七维分数与长度分数直接相加，再按总满分归一化。
Reward hacking 仍独立检测，不作为第八个 Judge 维度。

重复执行时，只有轨迹集合和完整评测指纹都一致才会跳过。指纹包含 Rubric、规则阈值、Judge
模型与提示词版本、推理和上下文参数、采样参数及聚合器版本；不一致时返回
`resume_conflict`，需更换输出目录或显式传入 `--overwrite`。
