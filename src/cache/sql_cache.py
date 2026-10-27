"""Optional async Redis cache for successful generated SQL responses."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any


def canonical_key(*, question: str, schema: str, model_version: str,
                  prompt_version: str, generation: dict[str, Any], scope: str = "public") -> str:
    """Return a non-reversible, collision-resistant key for all output inputs."""
    payload = {
        "question": " ".join(question.casefold().split()),
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "model_version": model_version,
        "prompt_version": prompt_version,
        "generation": generation,
        "scope": scope,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "nl2sql:v1:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SQLCache:
    def __init__(self, redis_client=None, *, ttl: int = 300, enabled: bool = True):
        self.client = redis_client
        self.ttl = max(1, ttl)
        self.enabled = enabled
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = self.misses = self.errors = 0

    @classmethod
    def from_env(cls):
        enabled = os.getenv("REDIS_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
        ttl = int(os.getenv("REDIS_CACHE_TTL_SECONDS", "300"))
        if not enabled or not os.getenv("REDIS_URL"):
            return cls(ttl=ttl, enabled=False)
        try:
            from redis.asyncio import Redis
            client = Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=1,
                                    socket_timeout=1, decode_responses=True,
                                    max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")))
            return cls(client, ttl=ttl, enabled=True)
        except ImportError:
            return cls(ttl=ttl, enabled=False)

    async def get(self, key: str):
        if not self.enabled or self.client is None:
            self.misses += 1
            return None
        try:
            value = await self.client.get(key)
            if value is None:
                self.misses += 1
                return None
            self.hits += 1
            return json.loads(value)
        except Exception:
            self.errors += 1
            self.enabled = False
            return None

    async def set(self, key: str, value: dict[str, Any]):
        if not self.enabled or self.client is None:
            return False
        try:
            await self.client.set(key, json.dumps(value, separators=(",", ":")), ex=self.ttl)
            return True
        except Exception:
            self.errors += 1
            self.enabled = False
            return False

    def lock_for(self, key: str) -> asyncio.Lock:
        # Cache key is a digest, so this bounded map contains no user data.
        if len(self._locks) > 2048:
            self._locks = {k: v for k, v in self._locks.items() if v.locked()}
        return self._locks.setdefault(key, asyncio.Lock())

    async def invalidate(self):
        """Invalidate this app's keys (e.g. after model/schema rollout)."""
        if not self.enabled or self.client is None:
            return 0
        try:
            count = 0
            async for key in self.client.scan_iter(match="nl2sql:v1:*"):
                count += await self.client.delete(key)
            return count
        except Exception:
            self.errors += 1
            self.enabled = False
            return 0

    async def close(self):
        if self.client is not None:
            close = getattr(self.client, "aclose", None) or getattr(self.client, "close", None)
            if close:
                result = close()
                if hasattr(result, "__await__"):
                    await result
