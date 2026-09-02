"""Offline tests for the independent embedding-service adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from hyscript.config import EmbeddingConfig
from hyscript.llm import AsyncOpenAIEmbeddingClient, EmbeddingProviderError


class _Response:
    def __init__(self, payload) -> None:
        self.payload = payload

    def model_dump(self):
        return self.payload


class _Embeddings:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _Response(self.response)


class _Client:
    def __init__(self, embeddings: _Embeddings) -> None:
        self.embeddings = embeddings
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class AsyncOpenAIEmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.config = EmbeddingConfig(
            base_url="https://embedding.example/v1/embeddings",
            api_key="embedding-secret",
            model="embedding-model",
        )

    async def test_uses_independent_endpoint_key_and_preserves_order(self) -> None:
        embeddings = _Embeddings(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )
        owned_client = _Client(embeddings)

        with patch(
            "hyscript.llm.async_embedding_client.AsyncOpenAI",
            return_value=owned_client,
        ) as factory:
            async with AsyncOpenAIEmbeddingClient(self.config) as client:
                vectors = await client.embed(
                    ["热点甲", "热点乙"],
                    model=self.config.model,
                )

        factory.assert_called_once_with(
            api_key="embedding-secret",
            base_url="https://embedding.example/v1",
            timeout=None,
            max_retries=0,
        )
        self.assertEqual(vectors, ((1.0, 0.0), (0.0, 1.0)))
        self.assertEqual(
            embeddings.calls,
            [
                {
                    "model": "embedding-model",
                    "input": ["热点甲", "热点乙"],
                    "encoding_format": "float",
                }
            ],
        )
        self.assertTrue(owned_client.closed)

    async def test_provider_error_is_stable_and_secret_safe(self) -> None:
        client = AsyncOpenAIEmbeddingClient(
            self.config,
            client=_Client(_Embeddings(error=RuntimeError("embedding-secret"))),
        )

        with self.assertRaisesRegex(
            EmbeddingProviderError,
            "Embedding request failed",
        ) as caught:
            await client.embed(["热点"], model=self.config.model)

        self.assertNotIn("embedding-secret", str(caught.exception))

    async def test_rejects_invalid_vector_shape(self) -> None:
        client = AsyncOpenAIEmbeddingClient(
            self.config,
            client=_Client(
                _Embeddings(
                    {
                        "data": [
                            {"index": 0, "embedding": [1.0, 0.0]},
                            {"index": 1, "embedding": [1.0]},
                        ]
                    }
                )
            ),
        )

        with self.assertRaisesRegex(
            EmbeddingProviderError,
            "invalid response payload",
        ):
            await client.embed(["热点甲", "热点乙"], model=self.config.model)


if __name__ == "__main__":
    unittest.main()
