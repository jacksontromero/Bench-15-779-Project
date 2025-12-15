#!/usr/bin/env bash
set -euo pipefail

# TK (ThunderKittens) sweep runner for Sequence-First kernel.
# This script wraps run_tk_sweep.sh but targets the seq_first kernel.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Call the main TK runner with specific target/binary overrides
"$SCRIPT_DIR/run_tk_sweep.sh" \
  --kernel_dir "chunk-attention-tk/ThunderKittens/kernels/chunked_attn" \
  --bin "attn_seq_first" \
  --out "results/chunk_attn_tk/sweep_seq_first.csv" \
  -- --sweep --warmup 50 --iters 200
