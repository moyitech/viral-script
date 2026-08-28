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
5. `04_end_to_end.py`: topic-to-script trace scaffold.
