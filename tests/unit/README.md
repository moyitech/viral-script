# Unit tests

Test deterministic prompt assembly, query parsing, normalization, trace
serialization, rule scoring, and aggregation without calling external services.

`test_tavily_provider.py` injects a fake SDK client, so it validates Tavily
configuration, limits, metadata normalization, and secret-safe errors without
using API credits.

`test_async_hy3_client.py` injects a fake `AsyncOpenAI` boundary to validate
endpoint normalization, SDK arguments, response metadata, client cleanup, and
secret-safe failures without maintaining a custom HTTP transport.

`test_async_embedding_client.py` verifies that topic embeddings use their own
OpenAI-compatible endpoint and key, restore vector order, close owned clients,
and keep provider errors secret-safe.

`test_async_tavily_provider.py` exercises the installed asynchronous Tavily SDK
boundary, including normalization of the optional third-party Hub envelope.

`test_newsnow_provider.py` and `test_topic_agent.py` use fake async HTTP, embedding,
and LLM clients. They verify partial hot-list failures, normalized ranking,
browser-compatible request headers, one embedding request, cosine connected-component
clustering, four concurrent high-reasoning generation batches, exact 20-item output,
source lineage, isolated retry behavior, and strict JSON validation.
