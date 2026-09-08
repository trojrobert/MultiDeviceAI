"""Thin client for the exo API.

exo exposes an OpenAI-compatible chat API *plus* a lower-level instance
lifecycle API (preview placements -> create instance -> await ready). This
module wraps both so the CLI can offer one-command UX.

See exo's README/`docs/api.md` for the underlying endpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from . import config


class ExoNotRunning(RuntimeError):
    """Raised when the exo API can't be reached."""


@dataclass
class Placement:
    """A candidate way to shard a model across the current cluster."""

    model_id: str
    sharding: str
    instance_meta: str
    instance: dict[str, Any]
    memory_delta_by_node: dict[str, int]
    error: str | None = None

    @property
    def nodes(self) -> list[str]:
        return list(self.memory_delta_by_node.keys())

    @property
    def total_memory(self) -> int:
        return sum(self.memory_delta_by_node.values())


class ExoClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or config.EXO_API_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.HTTP_TIMEOUT

    # -- low-level helpers -------------------------------------------------
    def _client(self, timeout: float | None = None) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout or self.timeout)

    def _get(self, path: str, **kwargs) -> Any:
        try:
            with self._client() as c:
                r = c.get(path, **kwargs)
                r.raise_for_status()
                return r.json()
        except httpx.ConnectError as e:  # exo not up
            raise ExoNotRunning(
                f"Could not reach exo at {self.base_url}. Is the cluster running "
                f"(`mdllm up`)?"
            ) from e

    # -- health / topology -------------------------------------------------
    def ping(self) -> bool:
        try:
            with self._client(timeout=3) as c:
                c.get("/models")
            return True
        except Exception:
            return False

    def state(self) -> dict[str, Any]:
        """Full deployment/topology state (nodes, instances, memory)."""
        return self._get("/state")

    # -- models ------------------------------------------------------------
    def models(self, status: str | None = None) -> Any:
        params = {"status": status} if status else None
        return self._get("/models", params=params)

    def search_models(self, query: str, limit: int = 10) -> Any:
        return self._get("/models/search", params={"query": query, "limit": limit})

    # -- instance lifecycle ------------------------------------------------
    def previews(self, model_id: str) -> list[Placement]:
        data = self._get("/instance/previews", params={"model_id": model_id})
        out: list[Placement] = []
        for p in data.get("previews", []):
            out.append(
                Placement(
                    model_id=p.get("model_id", model_id),
                    sharding=p.get("sharding", "?"),
                    instance_meta=p.get("instance_meta", "?"),
                    instance=p.get("instance", {}),
                    memory_delta_by_node=p.get("memory_delta_by_node", {}),
                    error=p.get("error"),
                )
            )
        return out

    def create_instance(self, instance: dict[str, Any]) -> dict[str, Any]:
        with self._client() as c:
            r = c.post("/instance", json={"instance": instance})
            r.raise_for_status()
            return r.json()

    def await_ready(self, model_id: str, timeout_seconds: float = 600) -> dict[str, Any]:
        """Block on exo's SSE stream until the instance is ready or times out."""
        params = {"model_id": model_id, "timeout_seconds": int(timeout_seconds)}
        with self._client(timeout=timeout_seconds + 30) as c:
            with c.stream("GET", "/instance/await", params=params) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if not payload:
                        continue
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if evt.get("type") in {"ready", "timeout", "error"}:
                        return evt
        return {"type": "closed"}

    def delete_instance(self, instance_id: str) -> None:
        with self._client() as c:
            r = c.delete(f"/instance/{instance_id}")
            r.raise_for_status()

    # -- chat --------------------------------------------------------------
    def chat_stream(
        self, model: str, messages: list[dict[str, str]], **kwargs
    ) -> Iterator[str]:
        """Yield content deltas from a streaming chat completion."""
        body = {"model": model, "messages": messages, "stream": True, **kwargs}
        with self._client(timeout=None) as c:
            with c.stream("POST", "/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            yield delta
