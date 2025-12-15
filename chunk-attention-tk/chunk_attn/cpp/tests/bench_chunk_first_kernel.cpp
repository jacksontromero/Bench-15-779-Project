#include <cassert>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include "chunk_allocator.h"
#include "chunk_info.h"
#include "kernel_cuda.h"

static int parse_int_flag(int argc, char** argv, const char* flag, int default_value) {
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], flag) == 0) return std::atoi(argv[i + 1]);
    }
    return default_value;
}

static void set_device_from_flags(int argc, char** argv) {
    int dev = parse_int_flag(argc, argv, "--device", -1);
    if (dev >= 0) {
        cudaSetDevice(dev);
        cudaDeviceSynchronize();
    }
}

static double flops_total(int n_seqs, int n_heads, int n_chunks, int chunk_size, int d_head) {
    const double blocks = double(n_heads) * double(n_chunks);
    // QK^T + (exp_scores)V, each counted as 2 flops/FMA => 4 * n_seqs * chunk_size * d_head per block.
    return blocks * 4.0 * double(n_seqs) * double(chunk_size) * double(d_head);
}

static double kv_bytes_total(int n_heads, int n_chunks, int chunk_size, int d_head) {
    const double blocks = double(n_heads) * double(n_chunks);
    // K and V reads once per block: 2 * (chunk_size*d_head) fp16 elements @2 bytes => 4*chunk_size*d_head bytes/block
    return blocks * 4.0 * double(chunk_size) * double(d_head);
}

static float time_attn_chunks_first_ms(GPT::KernelContext& ctx,
                                       torch::Tensor& query,
                                       int warmup,
                                       int iters) {
    for (int i = 0; i < warmup; i++) {
        GPT::attn_chunks_first(ctx, query);
    }
    cudaDeviceSynchronize();

    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < iters; i++) {
        GPT::attn_chunks_first(ctx, query);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= float(iters);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static void print_csv_header() {
    std::cout << "n_seqs,n_heads,n_chunks,blocks,latency_ms,tflops,compute_eff_pct,kv_gbps,kv_mem_eff_pct\n";
}

static void setup_kernel_context(GPT::KernelContext& context, std::vector<GPT::ChunkInfo>& chunk_infos, int n_seqs) {
    int n_chunks = chunk_infos.size();
    int n_shared_chunks = 0;
    for (auto& chunk_info : chunk_infos) {
        int n = chunk_info.seq_idx_end - chunk_info.seq_idx_begin;
        if (n > context.tpp_threshold) {
            n_shared_chunks++;
        }
    }

    auto host_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
    auto host_ptr_options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

    torch::Tensor begins_ends_offsets_host = torch::empty({3, n_shared_chunks}, host_options);
    int* begins_host_ptr = begins_ends_offsets_host.data_ptr<int>();
    int* ends_host_ptr = begins_host_ptr + n_shared_chunks;
    int* offsets_host_ptr = ends_host_ptr + n_shared_chunks;

    int shared_chunks_visited = 0;
    int partial_attn_dim = 0;
    for (auto& chunk_info : chunk_infos) {
        int n = chunk_info.seq_idx_end - chunk_info.seq_idx_begin;
        if (n <= context.tpp_threshold) continue;

        begins_host_ptr[shared_chunks_visited] = chunk_info.seq_idx_begin;
        ends_host_ptr[shared_chunks_visited] = chunk_info.seq_idx_end;
        offsets_host_ptr[shared_chunks_visited] = partial_attn_dim;
        partial_attn_dim += n;
        shared_chunks_visited++;
    }

    torch::Tensor keys_values_host = torch::empty({2, n_chunks}, host_ptr_options);
    void** keys_host_ptr = reinterpret_cast<void**>(keys_values_host.data_ptr());
    void** values_host_ptr = keys_host_ptr + n_chunks;

    int idx = 0;
    for (auto& chunk_info : chunk_infos) {
        keys_host_ptr[idx] = chunk_info.chunk->key_ptr();
        values_host_ptr[idx] = chunk_info.chunk->value_ptr();
        idx++;
    }

    auto device = context.options.device();
    context.begins_ends_offsets = begins_ends_offsets_host.to(device, torch::kInt32);
    context.keys_values = keys_values_host.to(device, torch::kInt64);
    
    // Resize intermediate result tensors
    context.attns = torch::empty({partial_attn_dim, context.n_heads, context.d_head}, context.options);
    context.maxs_sums = torch::empty({2, partial_attn_dim, context.n_heads}, context.options.dtype(torch::kFloat32));

    context.n_chunks = n_chunks;
    context.n_shared_chunks = n_shared_chunks;
    context.valid = true;
    context.delta_tokens = 0;
}

static void bench_one(int n_seqs, int n_heads, int n_chunks, int chunk_size, int d_head, int warmup, int iters, int device_id) {
    constexpr double H100_FP16_TFLOPS = 989.0; // used in existing result CSVs in this repo
    constexpr double H100_HBM3_PEAK_GBPS = 3350.0;

    if (device_id < 0) device_id = 0;
    auto options = torch::TensorOptions()
                      .device(torch::Device(torch::kCUDA, device_id))
                      .dtype(torch::kFloat16);

    // Build a minimal kernel context with n_chunks shared-prefix chunks.
    GPT::KernelContext ctx(n_heads, d_head, chunk_size, options);

    // Allocate chunks (KV storage) and mark each as covering [0, n_seqs).
    GPT::ChunkAllocator alloc(/*memory_mb=*/2048, chunk_size, n_heads, d_head, options);
    std::vector<GPT::ChunkInfo> chunk_infos;
    chunk_infos.reserve(n_chunks);
    for (int i = 0; i < n_chunks; i++) {
        GPT::Chunk* ch = alloc.allocate();
        ch->key().zero_();
        ch->value().zero_();
        chunk_infos.emplace_back(ch, /*seq_idx_begin=*/0, /*seq_idx_end=*/n_seqs);
    }
    
    // Use manual setup instead of linked GPT::refresh_kernel_context
    setup_kernel_context(ctx, chunk_infos, n_seqs);

    std::cout << "DEBUG: n_shared_chunks=" << ctx.n_shared_chunks << std::endl;
    assert(ctx.n_shared_chunks > 0);
    assert(ctx.keys_values.data_ptr() != nullptr);
    assert(ctx.begins_ends_offsets.data_ptr() != nullptr);
    assert(ctx.attns.sizes().size() == 3);

    // Query layout expected by kernel: [n_seqs, n_heads, d_head]
    auto query = torch::zeros({n_seqs, n_heads, d_head}, options);

    const float latency_ms = time_attn_chunks_first_ms(ctx, query, warmup, iters);
    const double latency_s = double(latency_ms) * 1e-3;

    const int blocks = n_heads * n_chunks;
    const double tflops = flops_total(n_seqs, n_heads, n_chunks, chunk_size, d_head) / latency_s / 1e12;
    const double compute_eff_pct = 100.0 * tflops / H100_FP16_TFLOPS;
    const double kv_gbps = kv_bytes_total(n_heads, n_chunks, chunk_size, d_head) / latency_s / 1e9;
    const double kv_mem_eff_pct = 100.0 * kv_gbps / H100_HBM3_PEAK_GBPS;

    std::cout << n_seqs << "," << n_heads << "," << n_chunks << "," << blocks << ","
              << std::fixed << std::setprecision(6) << latency_ms << ","
              << std::setprecision(4) << tflops << ","
              << std::setprecision(4) << compute_eff_pct << ","
              << std::setprecision(2) << kv_gbps << ","
              << std::setprecision(4) << kv_mem_eff_pct
              << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
          << "Usage:\n"
          << "  " << argv[0] << " --bench --n-heads <H> --n-chunks <C> [--warmup N --iters N --device D]\n"
          << "  " << argv[0] << " --sweep [--warmup N --iters N --device D]\n";
        return 1;
    }

    set_device_from_flags(argc, argv);
    int dev = parse_int_flag(argc, argv, "--device", 0);

    const int warmup = parse_int_flag(argc, argv, "--warmup", 50);
    const int iters = parse_int_flag(argc, argv, "--iters", 200);

    constexpr int n_seqs = 64;  // Match TK kernel (requires 64 for 4 warps × 16 rows each)
    constexpr int chunk_size = 64;
    constexpr int d_head = 128;

    print_csv_header();

    if (std::strcmp(argv[1], "--bench") == 0) {
        const int n_heads = parse_int_flag(argc, argv, "--n-heads", -1);
        const int n_chunks = parse_int_flag(argc, argv, "--n-chunks", -1);
        if (n_heads <= 0 || n_chunks <= 0) {
            std::cerr << "Error: --bench requires --n-heads <H> and --n-chunks <C>\n";
            return 2;
        }
        bench_one(n_seqs, n_heads, n_chunks, chunk_size, d_head, warmup, iters, dev);
        return 0;
    }

    if (std::strcmp(argv[1], "--sweep") == 0) {
        const struct { int h; int c; } sweep[] = {
          {4, 4}, {4, 8}, {4, 16}, {4, 32}, {4, 64},
          {16, 16}, {32, 8},
          {32, 16}, {8, 64},
          {64, 16}, {16, 64},
          {64, 64},
        };
        for (auto cfg : sweep) {
            bench_one(n_seqs, cfg.h, cfg.c, chunk_size, d_head, warmup, iters, dev);
        }
        return 0;
    }

    std::cerr << "Error: first argument must be --bench or --sweep\n";
    return 2;
}