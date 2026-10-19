# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Static decode table: tokens per second per user at each batch size, the number the
plugin's PR description reports for the switch on and off.

Every request is a unique token-id prompt (seeded per point, so every arm and every
arm order gets the same ids), streamed, greedy, ``ignore_eos``, through
``/v1/completions``. A point at concurrency C sends 2 x C warm-up requests, then
5 x C timed ones, C at a time (closed loop). Per request: TTFT, decode time after the
first chunk, tok/s/user = (completion tokens - 1) / decode time, and the gaps between
chunks (one chunk per engine step). Per point: p50 / p10 tok/s/user, p50 / p10 / p90
step, TTFT, output tok/s. Run one pass forward and one ``--reverse`` per launch and
report their mean.

    python benchmark/static_table.py --base-url http://127.0.0.1:30000 \
        --isl 8192 --osl 1024 --conc 1,2,4,8,16,32,64 --out results/on-r1
"""

import argparse
import asyncio
import json
import os
import random
import time
import zlib

import aiohttp

VOCAB = (1000, 120000)


def ids(n, key):
    rng = random.Random(zlib.crc32(key.encode()))
    return [rng.randrange(*VOCAB) for _ in range(n)]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else None


async def one(sess, base, model, prompt, max_new):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.time()
    chunks, usage = [], {}
    async with sess.post(base + "/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:500]}")
        async for line in r.content:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                if c.get("text") or c.get("finish_reason"):
                    chunks.append(time.time())
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    dec = t1 - chunks[0] if chunks else None
    return {
        "ttft": chunks[0] - t0 if chunks else None,
        "completion_tokens": n,
        "tps_user": (n - 1) / dec if n > 1 and dec else None,
        "gaps": [b - a for a, b in zip(chunks, chunks[1:])],
    }


async def closed_loop(sess, base, model, prompts, osl, conc):
    sem = asyncio.Semaphore(conc)

    async def run(p):
        async with sem:
            return await one(sess, base, model, p, osl)

    t0 = time.time()
    rows = await asyncio.gather(*(run(p) for p in prompts))
    return rows, time.time() - t0


async def point(sess, a, model, conc):
    key = f"decode-{a.isl}-{a.osl}-c{conc}{a.salt}"
    await sess.post(a.base_url + "/flush_cache")
    await closed_loop(
        sess,
        a.base_url,
        model,
        [ids(a.isl, f"{key}-w{i}") for i in range(2 * conc)],
        a.osl,
        conc,
    )
    rows, wall = await closed_loop(
        sess,
        a.base_url,
        model,
        [ids(a.isl, f"{key}-t{i}") for i in range(5 * conc)],
        a.osl,
        conc,
    )
    gaps = [g for r in rows for g in r["gaps"]]
    tps = [r["tps_user"] for r in rows]
    s = {
        "conc": conc,
        "isl": a.isl,
        "osl": a.osl,
        "label": a.label,
        "n": len(rows),
        "tps_user_p50": round(pct(tps, 0.5), 1),
        "tps_user_p10": round(pct(tps, 0.1), 1),
        "step_p50_ms": round(1e3 * pct(gaps, 0.5), 3),
        "step_p10_ms": round(1e3 * pct(gaps, 0.1), 3),
        "step_p90_ms": round(1e3 * pct(gaps, 0.9), 3),
        "ttft_p50_ms": round(1e3 * pct([r["ttft"] for r in rows], 0.5), 1),
        "out_tok_s": round(sum(r["completion_tokens"] for r in rows) / wall, 1),
    }
    print("SUMMARY " + json.dumps(s), flush=True)
    with open(os.path.join(a.out, "summary.jsonl"), "a") as f:
        f.write(json.dumps(s) + "\n")


async def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--conc", default="1,2,4,8,16,32,64")
    ap.add_argument(
        "--reverse", action="store_true", help="run the points in reverse order"
    )
    ap.add_argument(
        "--salt",
        default="",
        help="appended to every prompt seed (a fresh prompt set per pass)",
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    concs = [int(x) for x in a.conc.split(",")]
    tmo = aiohttp.ClientTimeout(total=3600)
    # force_close: a pooled keep-alive socket can race the server's idle close
    async with aiohttp.ClientSession(
        timeout=tmo, connector=aiohttp.TCPConnector(limit=0, force_close=True)
    ) as sess:
        async with sess.get(a.base_url + "/v1/models") as r:
            model = (await r.json())["data"][0]["id"]
        for c in concs[::-1] if a.reverse else concs:
            await point(sess, a, model, c)


if __name__ == "__main__":
    asyncio.run(main())
