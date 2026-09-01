# Developer examples

These files are intentionally small. They demonstrate one boundary at a time
and must import the production implementation from `src/hyscript/`. Examples
that need credentials import the central `hyscript.config.settings` object;
they never parse `.env` themselves.

1. `01_llm_call.py`: async Hy3 connectivity through `openai.AsyncOpenAI`.
2. `02_search_call.py`: async Tavily connectivity, normalization, and metadata.
3. `03_topic_recommendations.py`: NewsNow hot lists through one 4B embedding
   deduplication call and four concurrent Hy3 five-topic generation batches.
4. `03_query_planning.py`: Agent-generated query planning scaffold.
5. `04_end_to_end.py`: selected topic → live Tavily background search → Hy3
   oral script with separate citation metadata → immutable generation trace. For example:

   ```bash
   uv run --no-sync python examples/04_end_to_end.py \
     "行业自律能终结新能源车恶性竞争吗？" --target-length 450
   ```

   该入口默认按 `HYSCRIPT_LOG_LEVEL=INFO` 显示五个阶段，以及每个并发搜索的开始、完成或失败
   状态；日志不会打印 API Key 或搜索正文。运行结束还会输出 Tavily 尝试/成功/失败次数，以及
   Hy3 输入、输出、总量、推理和缓存输入 token。以上统计和逐次 Hy3 原始 `usage` 也会写入
   冻结 trace。
