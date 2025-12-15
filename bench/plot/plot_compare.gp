# gnuplot script to compare TK vs native kernel-only sweeps
# Usage (via plot_compare.sh):
#   gnuplot -e "tk='tk_agg.csv'; native='native_agg.csv'; outdir='comparison_plots'" plot_compare.gp

if (!exists("tk")) tk = "sweep_kv_agg.csv"
if (!exists("native")) native = "sweep_native_kernel_only_agg.csv"
if (!exists("outdir")) outdir = "comparison_plots"

set datafile separator ","
set grid back
set key top left
set tics out
set border linewidth 1.2

# Columns:
# n_seqs,n_heads,n_chunks,blocks,latency_ms,tflops,compute_eff_pct,kv_gbps,kv_mem_eff_pct
col_blocks = 4
col_latency = 5
col_tflops = 6
col_ceff = 7
col_gbps = 8
col_meff = 9

set term pngcairo size 1500,900 font ",16"

TK_COLOR = "#1f77b4"
NATIVE_COLOR = "#d62728"

# -----------------------------------------------------------------------------
# Latency scaling (log-log)
# -----------------------------------------------------------------------------
set output sprintf("%s/latency_compare.png", outdir)
set title "Chunk Attention Chunk-First: Latency vs Grid Size (Kernel-only)"
set xlabel "Grid size (blocks = heads × chunks)"
set ylabel "Latency per launch (ms)"
set logscale x 2
set logscale y 10
set format x "%g"
set format y "%.3g"
plot \
  tk using col_blocks:col_latency with linespoints lw 2 pt 7 ps 1.1 lc rgb TK_COLOR title "TK (bf16)", \
  native using col_blocks:col_latency with linespoints lw 2 pt 5 ps 1.1 lc rgb NATIVE_COLOR title "Native (fp16)"

# -----------------------------------------------------------------------------
# Throughput scaling
# -----------------------------------------------------------------------------
unset logscale y
set output sprintf("%s/throughput_compare.png", outdir)
set title "Chunk Attention Chunk-First: Throughput vs Grid Size (Kernel-only)"
set xlabel "Blocks"
set ylabel "TFLOPS"
plot \
  tk using col_blocks:col_tflops with linespoints lw 2 pt 7 ps 1.1 lc rgb TK_COLOR title "TK (bf16)", \
  native using col_blocks:col_tflops with linespoints lw 2 pt 5 ps 1.1 lc rgb NATIVE_COLOR title "Native (fp16)"

# -----------------------------------------------------------------------------
# KV bandwidth
# -----------------------------------------------------------------------------
set output sprintf("%s/bandwidth_compare.png", outdir)
set title "Chunk Attention Chunk-First: KV Bandwidth vs Grid Size (Kernel-only)"
set xlabel "Blocks"
set ylabel "KV bandwidth (GB/s)"
plot \
  tk using col_blocks:col_gbps with linespoints lw 2 pt 7 ps 1.1 lc rgb TK_COLOR title "TK (KV GB/s)", \
  native using col_blocks:col_gbps with linespoints lw 2 pt 5 ps 1.1 lc rgb NATIVE_COLOR title "Native (KV GB/s)"

# -----------------------------------------------------------------------------
# Efficiency
# -----------------------------------------------------------------------------
set output sprintf("%s/efficiency_compare.png", outdir)
set title "Chunk Attention Chunk-First: Efficiency vs Grid Size (Kernel-only)"
set xlabel "Blocks"
set ylabel "Efficiency (% of peak)"
plot \
  tk using col_blocks:col_ceff with linespoints lw 2 pt 7 ps 1.0 lc rgb TK_COLOR title "TK compute eff (%)", \
  tk using col_blocks:col_meff with linespoints lw 2 pt 9 ps 0.9 lc rgb TK_COLOR dt 2 title "TK KV mem eff (%)", \
  native using col_blocks:col_ceff with linespoints lw 2 pt 5 ps 1.0 lc rgb NATIVE_COLOR title "Native compute eff (%)", \
  native using col_blocks:col_meff with linespoints lw 2 pt 11 ps 0.9 lc rgb NATIVE_COLOR dt 2 title "Native KV mem eff (%)"

# -----------------------------------------------------------------------------
# Combined (2-panel): latency + throughput
# -----------------------------------------------------------------------------
set output sprintf("%s/scaling_combined_compare.png", outdir)
set multiplot layout 1,2 title "Chunk Attention Chunk-First Scaling (Kernel-only)"

set title "Latency"
set xlabel "Blocks"
set ylabel "Latency (ms)"
set logscale y 10
plot \
  tk using col_blocks:col_latency with linespoints lw 2 pt 7 ps 1.0 lc rgb TK_COLOR title "TK", \
  native using col_blocks:col_latency with linespoints lw 2 pt 5 ps 1.0 lc rgb NATIVE_COLOR title "Native"

unset logscale y
set title "Throughput"
set xlabel "Blocks"
set ylabel "TFLOPS"
plot \
  tk using col_blocks:col_tflops with linespoints lw 2 pt 7 ps 1.0 lc rgb TK_COLOR title "TK", \
  native using col_blocks:col_tflops with linespoints lw 2 pt 5 ps 1.0 lc rgb NATIVE_COLOR title "Native"

unset multiplot
set output
