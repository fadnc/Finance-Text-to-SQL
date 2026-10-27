import asyncio
import json
import time
import unittest

from src.cache.sql_cache import SQLCache, canonical_key


class FakeRedis:
    def __init__(self): self.data = {}; self.expiry = {}; self.clock = time.monotonic
    async def get(self, key):
        if key in self.data and self.clock() >= self.expiry[key]:
            self.data.pop(key); self.expiry.pop(key)
        return self.data.get(key)
    async def set(self, key, value, ex): self.data[key] = value; self.expiry[key] = self.clock() + ex
    async def delete(self, key): return int(self.data.pop(key, None) is not None)
    async def scan_iter(self, match):
        for key in list(self.data):
            if key.startswith("nl2sql:v1:"): yield key
    async def aclose(self): pass


class BrokenRedis:
    async def get(self, key): raise ConnectionError("offline")


class CacheTests(unittest.TestCase):
    def setUp(self): self.key = canonical_key(question="  Show  total ", schema="S", model_version="m1", prompt_version="p1", generation={"n": 1})
    def test_identity_separates_schema_and_model(self):
        self.assertEqual(self.key, canonical_key(question="show total", schema="S", model_version="m1", prompt_version="p1", generation={"n": 1}))
        self.assertNotEqual(self.key, canonical_key(question="show total", schema="S2", model_version="m1", prompt_version="p1", generation={"n": 1}))
        self.assertNotEqual(self.key, canonical_key(question="show total", schema="S", model_version="m2", prompt_version="p1", generation={"n": 1}))
    def test_hit_miss_ttl_and_invalidation(self):
        async def run():
            client = FakeRedis(); cache = SQLCache(client, ttl=300)
            self.assertIsNone(await cache.get(self.key))
            await cache.set(self.key, {"sql": "SELECT 1"})
            self.assertEqual(await cache.get(self.key), {"sql": "SELECT 1"})
            self.assertGreater(client.expiry[self.key], client.clock())
            self.assertEqual(await cache.invalidate(), 1)
            self.assertIsNone(await cache.get(self.key))
        asyncio.run(run())

    def test_ttl_expiration(self):
        async def run():
            client = FakeRedis(); now = [100.0]; client.clock = lambda: now[0]
            cache = SQLCache(client, ttl=3)
            await cache.set(self.key, {"sql": "SELECT 1"})
            now[0] += 4
            self.assertIsNone(await cache.get(self.key))
        asyncio.run(run())
    def test_unavailable_cache_falls_back(self):
        async def run():
            cache = SQLCache(BrokenRedis())
            self.assertIsNone(await cache.get(self.key))
            self.assertFalse(cache.enabled)
            self.assertEqual(cache.errors, 1)
        asyncio.run(run())


if __name__ == "__main__": unittest.main()
