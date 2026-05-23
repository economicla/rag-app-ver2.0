"""
OpenAIEmbeddingAdapter - OpenAI embeddings API adapter.

Used for VPS/demo deployments where embeddings are generated through GPT/OpenAI
instead of a self-hosted Jina service.
"""

import logging
import time
from typing import List, Optional

import httpx

from app_refactored.core.interfaces import IEmbeddingService

logger = logging.getLogger(__name__)


class OpenAIEmbeddingAdapter(IEmbeddingService):
    """Async adapter for OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int = 2048,
        timeout: int = 120,
        max_connections: int = 10,
        embed_batch_size: int = 128,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimensions = int(dimensions)
        self.timeout = timeout
        self.embed_batch_size = max(1, int(embed_batch_size))
        self.endpoint = f"{self.base_url}/embeddings"
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
            ),
        )

    async def embed_text(self, text: str) -> List[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out: List[List[float]] = []
        t_all = time.monotonic()
        for start in range(0, len(texts), self.embed_batch_size):
            chunk = texts[start : start + self.embed_batch_size]
            out.extend(await self._embed_batch_chunk(chunk))

        elapsed_ms = (time.monotonic() - t_all) * 1000
        logger.info(
            f"✅ OpenAI embeddings: {len(out)} vectors in {elapsed_ms:.0f}ms "
            f"(sub_batch_size={self.embed_batch_size}, dimensions={self.dimensions})"
        )
        return out

    async def _embed_batch_chunk(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post(self.endpoint, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(f"❌ OpenAI embedding HTTP {response.status_code}: {response.text[:500]}")
            raise exc

        data = response.json().get("data", [])
        data.sort(key=lambda item: item.get("index", 0))
        embeddings = [item["embedding"] for item in data]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(texts)}")
        return embeddings

    async def get_dimension(self) -> int:
        return self.dimensions

    async def is_available(self) -> bool:
        try:
            embedding = await self.embed_text("ping")
            return bool(embedding)
        except Exception as exc:
            logger.warning(f"OpenAI embedding health check failed: {exc}")
            return False

    async def close(self):
        await self.client.aclose()
