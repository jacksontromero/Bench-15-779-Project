#!/usr/bin/env bash
set -euo pipefail

# Generic TK (ThunderKittens) sweep runner.
#
# This script is meant to make it easy for teammates to benchmark *their own* TK kernels:
# - it builds a kernel directory via `make`
# - runs the produced binary (expects it to print CSV to stdout)
# - saves the CSV to a chosen output path
#
# Defaults target the ChunkAttention TK kernel in this repo.
#
# Usage:
#   ./bench/scripts/run_tk_sweep.sh \
#     [--kernel_dir <path>] \
#     [--bin <name>] \
#     [--out <csv_path>] \
#     [--] <extra args passed to the binary>
#
# Examples:
#   # Default (ChunkAttention TK):
#   ./bench/scripts/run_tk_sweep.sh --out results/chunk_attn_tk/sweep.csv
#
#   # Run a single point:
#   ./bench/scripts/run_tk_sweep.sh -- --bench --n-heads 32 --n-chunks 16 --warmup 50 --iters 200
#
#   # Benchmark a different TK kernel directory/binary:
#   ./bench/scripts/run_tk_sweep.sh \
#     --kernel_dir hydragen-tk/ThunderKittens/kernels/some_kernel \
#     --bin some_kernel_bin \
#     --out results/hydragen_tk/sweep.csv \
#     -- --sweep --warmup 50 --iters 200

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

KERNEL_DIR="chunk-attention-tk/ThunderKittens/kernels/chunked_attn"
BIN_NAME="attn_chunk_first"
OUT_CSV="results/chunk_attn_tk/sweep.csv"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel_dir)
      KERNEL_DIR="${2:-}"; shift 2;;
    --bin)
      BIN_NAME="${2:-}"; shift 2;;
    --out)
      OUT_CSV="${2:-}"; shift 2;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;; 
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;; 
  esac
done

abs_path() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf "%s" "$p"
  else
    printf "%s/%s" "$REPO_ROOT" "$p"
  fi
}

KERNEL_DIR_ABS="$(abs_path "$KERNEL_DIR")"
OUT_CSV_ABS="$(abs_path "$OUT_CSV")"
BIN_ABS="$KERNEL_DIR_ABS/$BIN_NAME"

# Ensure ThunderKittens root for includes.
export THUNDERKITTENS_ROOT="${THUNDERKITTENS_ROOT:-$REPO_ROOT/chunk-attention-tk/ThunderKittens}"

mkdir -p "$(dirname "$OUT_CSV_ABS")"

echo "[TK] repo_root=$REPO_ROOT"
echo "[TK] kernel_dir=$KERNEL_DIR_ABS"
echo "[TK] binary=$BIN_ABS"
echo "[TK] out_csv=$OUT_CSV_ABS"
echo "[TK] THUNDERKITTENS_ROOT=$THUNDERKITTENS_ROOT"

# Default behavior: build all
make -C "$KERNEL_DIR_ABS" clean all

if [[ ! -x "$BIN_ABS" ]]; then
  echo "Error: expected executable not found: $BIN_ABS" >&2
  exit 1
fi

if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  EXTRA_ARGS=(--sweep --warmup 50 --iters 200)
fi

echo "[TK] running: $BIN_ABS ${EXTRA_ARGS[*]}"
"$BIN_ABS" "${EXTRA_ARGS[@]}" > "$OUT_CSV_ABS"

echo "[TK] wrote: $OUT_CSV_ABS"