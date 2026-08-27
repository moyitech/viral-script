# Unit tests

Test deterministic prompt assembly, query parsing, normalization, trace
serialization, rule scoring, and aggregation without calling external services.

`test_tavily_provider.py` injects a fake SDK client, so it validates Tavily
configuration, limits, metadata normalization, and secret-safe errors without
using API credits.

`test_async_hy3_client.py` injects a fake `AsyncOpenAI` boundary to validate
endpoint normalization, SDK arguments, response metadata, client cleanup, and
secret-safe failures without maintaining a custom HTTP transport.

`test_async_tavily_provider.py` exercises the installed asynchronous Tavily SDK
boundary, including normalization of the optional third-party Hub envelope.
