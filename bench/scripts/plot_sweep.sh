#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_sweep.sh <csv> <outdir> [--plot_script <path/to/plot_sweep.gp>]
#
# Example:
#   ./plot_sweep.sh ../../results/chunk_attn_tk/sweep.csv ../../results/chunk_attn_tk/minimal_plots

DATAFILE=${1:?missing csv path}
OUTDIR=${2:?missing output directory}

PLOT_SCRIPT=""
if [[ "${3:-}" == "--plot_script" ]]; then
  PLOT_SCRIPT="${4:-}"
fi
if [[ -z "$PLOT_SCRIPT" ]]; then
  # assume repo layout: bench/scripts/ -> bench/plot/
  PLOT_SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/plot/plot_sweep.gp"
fi

mkdir -p "$OUTDIR"

gnuplot -e "datafile='${DATAFILE}'; outdir='${OUTDIR}'" "$PLOT_SCRIPT"

echo "Wrote plots to: ${OUTDIR}"
