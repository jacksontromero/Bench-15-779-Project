#!/usr/bin/env python3
"""CUDA-event benchmark sweep for native chunk_attn (Sequence-First mode).

This benchmarks the 'seq_first' path of the native chunk_attn kernel.
It mirrors benchmark_chunk_attn_sweep.py but sets partition=2.

Example:
  python bench/scripts/benchmark_seq_first_sweep.py --csv results/chunk_attn_native/sweep_seq_first.csv
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass

import torch


H100_PEAK_TFLOPS_BF16 = 989.0
H100_PEAK_GBPS = 3350.0


@dataclass(frozen=True)
class Cfg:
    n_seqs: int
    n_heads: int
    n_chunks: int
    chunk_size: int = 64
    d_head: int = 128

    @property
    def blocks(self) -> int:
        return self.n_heads * self.n_chunks


def flops_total(cfg: Cfg) -> float:
    # Same estimate as TK harness: 4 * n_seqs * chunk_size * d_head per (head, chunk)
    return 4.0 * cfg.n_seqs * cfg.chunk_size * cfg.d_head * cfg.n_heads * cfg.n_chunks


def kv_bytes_total(cfg: Cfg) -> float:
    # Count only K and V reads (bf16/fp16 = 2 bytes)
    # bytes_per_block = (K + V) = 2 * chunk_size * d_head * 2
    return 2.0 * cfg.chunk_size * cfg.d_head * 2.0 * cfg.n_heads * cfg.n_chunks


def make_shared_inputs(cfg: Cfg, device: torch.device, dtype: torch.dtype, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Shared prefix tokens length determines number of chunks.
    prefix_len = cfg.n_chunks * cfg.chunk_size
    tokens = list(range(prefix_len))

    # K/V for the shared prefix: shape (prefix_len, n_heads, d_head)
    k = torch.randn((prefix_len, cfg.n_heads, cfg.d_head), device=device, dtype=dtype, generator=g)
    v = torch.randn((prefix_len, cfg.n_heads, cfg.d_head), device=device, dtype=dtype, generator=g)

    # Q for current token(s): shape (n_seqs, n_heads, d_head)
    q = torch.randn((cfg.n_seqs, cfg.n_heads, cfg.d_head), device=device, dtype=dtype, generator=g)

    return tokens, k, v, q


def bench_cfg(cfg: Cfg, warmup: int, iters: int, seed: int, device_idx: int, dtype: str, partition: int):
    import chunk_attn  # pylint: disable=import-error

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    device = torch.device(f"cuda:{device_idx}")
    torch.cuda.set_device(device)

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

    # Construct attention object.
    # share_prefix=True to force prefix trie sharing.
    attn = chunk_attn.Attention(
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        chunk_size=cfg.chunk_size,
        memory_mb=8192,
        share_prefix=True,
        tpp_threshold=1,
        dtype=torch_dtype,
        device=device,
    )

    tokens, k, v, q = make_shared_inputs(cfg, device=device, dtype=torch_dtype, seed=seed)

    # Add one sequence then duplicate to get n_seqs sequences sharing the same prefix.
    attn.add_seq(tokens=tokens, k=k, v=v)
    if cfg.n_seqs > 1:
        attn.duplicate(0, cfg.n_seqs - 1)

    # Ensure kernel context is ready.
    attn.refresh_kernel_context(force=True)

    # Warmup
    for _ in range(warmup):
        _ = attn.forward(q=q, partition=partition)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = attn.forward(q=q, partition=partition)
    end.record()
    end.synchronize()

    ms_total = start.elapsed_time(end)
    return ms_total / float(iters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sweep_native_seq_first.csv")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    # partition=2 forces the seq_first kernel path in chunk_attn
    ap.add_argument("--partition", type=int, default=2, help="chunk_attn forward partition arg (2=seq_first)")
    args = ap.parse_args()

    # Identical sweep config to chunk_first benchmark
    cfgs = [
        Cfg(64, 4, 4),
        Cfg(64, 4, 8),
        Cfg(64, 4, 16),
        Cfg(64, 4, 32),
        Cfg(64, 4, 64),
        Cfg(64, 16, 16),
        Cfg(64, 32, 8),
        Cfg(64, 32, 16),
        Cfg(64, 8, 64),
        Cfg(64, 64, 16),
        Cfg(64, 16, 64),
        Cfg(64, 64, 64),
        Cfg(64, 128, 64),
        Cfg(64, 64, 128),
        Cfg(64, 128, 128),
    ]

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n_seqs",
                "n_heads",
                "n_chunks",
                "blocks",
                "latency_ms",
                "tflops",
                "compute_eff_pct",
                "kv_gbps",
                "kv_mem_eff_pct",
            ]
        )

        for cfg in cfgs:
            ms = bench_cfg(
                cfg,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
                device_idx=args.device,
                dtype=args.dtype,
                partition=args.partition,
            )
            tflops = flops_total(cfg) / (ms * 1e-3) / 1e12
            kv_gbps = kv_bytes_total(cfg) / (ms * 1e-3) / 1e9
            ceff = 100.0 * tflops / H100_PEAK_TFLOPS_BF16
            meff = 100.0 * kv_gbps / H100_PEAK_GBPS

            w.writerow([cfg.n_seqs, cfg.n_heads, cfg.n_chunks, cfg.blocks, f"{ms:.6f}", f"{tflops:.4f}", f"{ceff:.4f}", f"{kv_gbps:.2f}", f"{meff:.4f}"])

            print(
                f"blocks={cfg.blocks:4d} heads={cfg.n_heads:2d} chunks={cfg.n_chunks:2d} "
                f"lat={ms:.6f} ms  tflops={tflops:.3f}  kv_gbps={kv_gbps:.1f}"
            )


if __name__ == "__main__":
    main()
