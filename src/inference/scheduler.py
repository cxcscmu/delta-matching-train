"""Phased continuous-batching scheduler.

Keeps the decode batch full: admit waiting requests into free slots (per-sequence prefill
for the MVP: prefill is <1.5% of reasoning compute), then run one decode forward over all
running requests (one token each, state/KV gathered by slot id), retire finished requests
and free their slots. Repeats until the queue drains.
"""

from __future__ import annotations

import os
import time
from collections import deque

import torch

from .context import PHASE_DECODE, PHASE_PREFILL, EngineContext
from .sampler import sample_tokens, suffix_hits

_PROGRESS = os.environ.get("ENGINE_PROGRESS", "1") != "0"


class Request:
    __slots__ = ("rid", "prompt_ids", "max_new", "slot", "gen", "finish_reason")

    def __init__(self, rid: int, prompt_ids: list[int], max_new: int):
        self.rid = rid
        self.prompt_ids = prompt_ids
        self.max_new = max_new
        self.slot = -1
        self.gen: list[int] = []
        self.finish_reason: str | None = None  # "stop" | "length" | None(running)


class Scheduler:
    def __init__(self, engine, pool, *, do_sample, temperature, top_k, top_p,
                 stop_seqs, eos_ids, generator=None):
        self.engine = engine
        self.pool = pool
        self.device = engine.device
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_seqs = stop_seqs
        self.eos_ids = set(eos_ids)
        self.generator = generator

    def _sample(self, logits_2d: torch.Tensor) -> torch.Tensor:  # (N,vocab)->(N,)
        return sample_tokens(logits_2d, self.do_sample, self.temperature,
                             self.top_k, self.top_p, self.generator)

    def _finish(self, req: Request) -> bool:
        last = req.gen[-1]
        if last in self.eos_ids:
            req.finish_reason = "stop"
        elif suffix_hits(req.gen, self.stop_seqs):
            req.finish_reason = "stop"
        elif len(req.gen) >= req.max_new:
            req.finish_reason = "length"
        else:
            return False
        return True

    def _prefill(self, req: Request) -> None:
        slot = self.pool.alloc()
        req.slot = slot
        ids = torch.tensor(req.prompt_ids, device=self.device, dtype=torch.long)[None]
        ctx = EngineContext(pool=self.pool, phase=PHASE_PREFILL, prefill_slot=slot)
        logits = self.engine.forward(ids, ctx, logits_last_only=True)  # (1,1,vocab)
        self.pool.seqlens[slot] = len(req.prompt_ids)
        tok = int(self._sample(logits[:, -1])[0])
        req.gen = [tok]

    def _decode_step(self, running: list[Request]) -> None:
        slot_ids = torch.tensor([r.slot for r in running], device=self.device, dtype=torch.int32)
        last = torch.tensor([[r.gen[-1]] for r in running], device=self.device, dtype=torch.long)  # (N,1)
        ctx = EngineContext(pool=self.pool, phase=PHASE_DECODE, slot_ids=slot_ids)
        logits = self.engine.forward(last, ctx, logits_last_only=True)  # (N,1,vocab)
        self.pool.seqlens[slot_ids] += 1
        toks = self._sample(logits[:, -1]).tolist()
        for r, t in zip(running, toks):
            r.gen.append(t)

    @torch.inference_mode()
    def run(self, requests: list[Request], claim=None) -> list[Request]:
        """Continuous-batching loop. `claim`, if given, is a zero-arg callable returning the
        next Request to admit (or None when the global pool is exhausted): used for
        cross-rank work-stealing: when the local `waiting` queue empties, free slots pull
        from `claim()` instead of idling. With claim=None this is identical to before."""
        waiting = deque(requests)
        running: list[Request] = []
        done: list[Request] = []
        claim_active = claim is not None
        total = len(requests)
        t0 = time.time()
        log_step = max(1, (total if total else 32) // 10)
        next_log = log_step
        verbose = _PROGRESS and (claim_active or total >= 16)
        while waiting or running or claim_active:
            # admit + prefill into every free slot (local queue first, then steal)
            while self.pool.num_free > 0:
                if waiting:
                    req = waiting.popleft()
                elif claim_active:
                    req = claim()
                    if req is None:
                        claim_active = False
                        break
                else:
                    break
                if req.max_new <= 0:   # nothing to generate: done without a prefill (the prefill samples a token)
                    req.finish_reason = "length"
                    done.append(req)
                    continue
                self._prefill(req)
                if self._finish(req):
                    self.pool.free(req.slot)
                    done.append(req)
                else:
                    running.append(req)
            # one decode step over all running, then retire finished
            if running:
                self._decode_step(running)
                still: list[Request] = []
                for req in running:
                    if self._finish(req):
                        self.pool.free(req.slot)
                        done.append(req)
                    else:
                        still.append(req)
                running = still
            if verbose and len(done) >= next_log:
                next_log = len(done) + log_step
                gen = sum(len(r.gen) for r in done) + sum(len(r.gen) for r in running)
                el = time.time() - t0
                tag = "steal" if claim is not None else f"{len(done)}/{total}"
                print(f"[engine] {len(done)} done ({tag}) | run={len(running)} | "
                      f"{gen} tok | {gen / max(el, 1e-9):.0f} tok/s | {el:.0f}s", flush=True)
        return done
