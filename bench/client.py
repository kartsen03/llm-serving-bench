"""Async streaming client for OpenAI-compatible /v1/chat/completions endpoints.

One closed-loop run keeps exactly `concurrency` requests in flight: each of `concurrency`
workers sends a request, waits for the full streamed answer, then sends the next one, until
the run's request list is used up. Timing uses time.perf_counter() on the client, so every
number includes HTTP, JSON and SSE overhead on both servers alike.
"""

import asyncio
import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field

import aiohttp

# Identical for both servers. Greedy decoding with every penalty off; the servers are also
# launched with neutral defaults (vLLM --generation-config vllm, Ollama Modelfile params).
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0}


@dataclass
class RequestResult:
    index: int
    prompt_id: int
    tag: str
    ok: bool = False
    error: str | None = None
    t_send: float = 0.0              # seconds since run start
    ttft: float | None = None        # send -> first non-empty content chunk
    e2e: float | None = None         # send -> end of stream
    chunk_offsets: list[float] = field(default_factory=list)  # content chunks, since send
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    text_sha1: str | None = None
    text_head: str | None = None

    @property
    def itl(self) -> float | None:
        """Mean inter-token latency: decode time spread over the tokens after the first."""
        n = self.completion_tokens
        if not self.ok or not n or n < 2 or len(self.chunk_offsets) < 2:
            return None
        return (self.chunk_offsets[-1] - self.chunk_offsets[0]) / (n - 1)

    def to_json(self) -> dict:
        d = asdict(self)
        d["itl"] = self.itl
        d["chunk_offsets"] = [round(x, 5) for x in self.chunk_offsets]
        return d


def build_messages(prompt: dict, tag: str) -> list[dict]:
    # The tag goes first so no two requests in the study share a cached prefix, on either
    # server, while prompt i of repeat r at level c is still byte-identical across servers.
    return [{"role": "system", "content": f"Session {tag}. {prompt['system']}"},
            {"role": "user", "content": prompt["user"]}]


async def one_request(session: aiohttp.ClientSession, url: str, model: str, prompt: dict,
                      tag: str, index: int, max_tokens: int, t_run0: float) -> RequestResult:
    res = RequestResult(index=index, prompt_id=prompt["id"], tag=tag)
    body = {"model": model, "messages": build_messages(prompt, tag), "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}, **SAMPLING}
    parts: list[str] = []
    t0 = time.perf_counter()
    res.t_send = t0 - t_run0
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return res
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                now = time.perf_counter()
                if chunk.get("usage"):
                    res.prompt_tokens = chunk["usage"].get("prompt_tokens")
                    res.completion_tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        res.chunk_offsets.append(now - t0)
                        parts.append(text)
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
        res.e2e = time.perf_counter() - t0
        if res.chunk_offsets:
            res.ttft = res.chunk_offsets[0]
        text = "".join(parts)
        res.text_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
        res.text_head = text[:120]
        res.ok = bool(res.chunk_offsets)
        if not res.ok:
            res.error = "stream ended without content"
    except Exception as e:  # recorded, never silently dropped
        res.error = f"{type(e).__name__}: {e}"
    return res


async def closed_loop(base_url: str, model: str, jobs: list[tuple[dict, str]],
                      concurrency: int, max_tokens: int) -> tuple[list[RequestResult], float]:
    """Send `jobs` (prompt, tag) with `concurrency` in flight. Returns results and wall time."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    results: list[RequestResult | None] = [None] * len(jobs)
    next_index = 0

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        t_run0 = time.perf_counter()

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(jobs):
                i = next_index
                next_index += 1
                prompt, tag = jobs[i]
                results[i] = await one_request(session, url, model, prompt, tag, i,
                                               max_tokens, t_run0)

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        wall = time.perf_counter() - t_run0
    return [r for r in results if r is not None], wall


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results: list[RequestResult], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft for r in ok if r.ttft is not None]
    itl = [r.itl for r in ok if r.itl is not None]
    e2e = [r.e2e for r in ok if r.e2e is not None]
    gaps = [b - a for r in ok for a, b in zip(r.chunk_offsets, r.chunk_offsets[1:])]
    out_tok = sum(r.completion_tokens or 0 for r in ok)
    chunks = sum(len(r.chunk_offsets) for r in ok)
    return {
        "n_requests": len(results),
        "n_ok": len(ok),
        "n_errors": len(results) - len(ok),
        "errors": sorted({r.error for r in results if r.error})[:5],
        "wall_s": wall,
        "output_tokens": out_tok,
        "prompt_tokens": sum(r.prompt_tokens or 0 for r in ok),
        "missing_usage": sum(1 for r in ok if r.completion_tokens is None),
        "content_chunks": chunks,
        "tokens_per_chunk": out_tok / chunks if chunks else None,
        "finish_length_frac": sum(r.finish_reason == "length" for r in ok) / len(ok) if ok else None,
        "output_tok_per_s": out_tok / wall,
        "requests_per_s": len(ok) / wall,
        "ttft_s": {"p50": pct(ttft, .5), "p90": pct(ttft, .9), "p99": pct(ttft, .99),
                   "mean": statistics.fmean(ttft) if ttft else None},
        "itl_s": {"p50": pct(itl, .5), "p90": pct(itl, .9),
                  "mean": statistics.fmean(itl) if itl else None},
        "chunk_gap_s": {"p50": pct(gaps, .5), "p90": pct(gaps, .9), "p99": pct(gaps, .99),
                        "max": max(gaps) if gaps else None},
        "e2e_s": {"p50": pct(e2e, .5), "p90": pct(e2e, .9)},
    }
