#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_compare.sh <tk_csv> <native_csv> <outdir>
#
# Example:
#   ./plot_compare.sh \
#     ../../results/chunk_attn_tk/sweep.csv \
#     ../../results/chunk_attn_native/sweep_native_kernel_only.csv \
#     ../../results/chunk_attn_tk/comparison_vs_native

TK_CSV=${1:?missing tk csv path}
NATIVE_CSV=${2:?missing native csv path}
OUTDIR=${3:?missing output directory}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/plot"

mkdir -p "$OUTDIR"

# Aggregate to one point per blocks (write intermediates into OUTDIR)
python3 "$SCRIPT_DIR/aggregate_by_blocks.py" "$TK_CSV" "$OUTDIR/tk_agg.csv"
python3 "$SCRIPT_DIR/aggregate_by_blocks.py" "$NATIVE_CSV" "$OUTDIR/native_agg.csv"

gnuplot -e "tk='$OUTDIR/tk_agg.csv'; native='$OUTDIR/native_agg.csv'; outdir='${OUTDIR}'" "$PLOT_DIR/plot_compare.gp"

echo "Wrote comparison plots to: ${OUTDIR}"
