__host__ void attn_chunks_first(KernelContext& context, torch::Tensor& query) {
    printf("DEBUG: calling attn_chunks_first\n");
    uint32_t batch_size = ((query.size(0) - 1) / 16 + 1) * 16;
    uint32_t elem_size = query.element_size();
    int chunk_size = context.chunk_size;
    float scale = 1.f / sqrtf(static_cast<float>(context.d_head));

    size_t query_shared_mem_size =
      batch_size * context.d_head * elem_size + 16 * batch_size; // 16 is for padding
    size_t keys_shared_mem_size = chunk_size * context.d_head * elem_size + 16 * chunk_size;
    size_t score_shared_mem_size = (chunk_size + 16) * batch_size * sizeof(float);
    size_t shared_mem_size = query_shared_mem_size + keys_shared_mem_size + score_shared_mem_size;

    dim3 grid(context.n_heads, context.n_shared_chunks);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#define CALL_ATTN_CHUNK_KERNEL_FUNCTION(scalar_t, batch_size, chunk_size, d_head)                  \
    do {                                                                                           \
        if (shared_mem_size >= 48 * 1024) {                                                        \
            C10_CUDA_CHECK(cudaFuncSetAttribute(                                                   \
              attn_chunk_first_kernel_v2<scalar_t, batch_size, chunk_size, d_head>,                   \
              cudaFuncAttributeMaxDynamicSharedMemorySize,                                         \
              shared_mem_size));                                                                   \
        }                                                                                          \
        attn_chunk_first_kernel_v2<scalar_t, batch_size, chunk_size, d_head>                          \
          <<<grid, 128, shared_mem_size, stream>>>(query_ptr,                                      \
                                                   keys_ptr,                                       \
                                                   values_ptr,                                     \
                                                   attns_ptr,                                      \
                                                   maxs_ptr,                                       \
                                                   sums_ptr,                                       \
                                                   offsets_ptr,                                    \
                                                   begins_ptr,                                     \
                                                   ends_ptr,                                       \
                                                   scale,                                          \
                                                   context.n_heads);                               \
    } while (0)
//    std::cout << "v2 chunk first" << std::endl;
    if (query.dtype() == at::ScalarType::Half) {
        half* query_ptr = reinterpret_cast<half*>(query.data_ptr());
        void** keys_ptr = reinterpret_cast<void**>(context.keys_values.data_ptr());
        void** values_ptr = keys_ptr + context.keys_values.stride(0);
        half* attns_ptr = reinterpret_cast<half*>(context.attns.data_ptr());
        float* maxs_ptr = context.maxs_sums.data_ptr<float>();
        float* sums_ptr = context.maxs_sums.data_ptr<float>() + context.maxs_sums.stride(0);
        int* begins_ptr = context.begins_ends_offsets.data_ptr<int>();
        int* ends_ptr = begins_ptr + context.begins_ends_offsets.stride(0);
        int* offsets_ptr = ends_ptr + context.begins_ends_offsets.stride(0);

        if (batch_size == 16 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 16, 64, 128);
        } else if (batch_size == 32 && chunk_size == 32) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 32, 32, 128);
        } else if (batch_size == 32 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 32, 64, 128);
        } else if (batch_size == 48 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 48, 64, 128);
        } else if (batch_size == 64 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 64, 128);
        } else if (batch_size == 96 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 96, 64, 128);
        } else if (batch_size == 128 && chunk_size == 64) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 128, 64, 128);
        } else if (batch_size == 64 && chunk_size == 32) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 32, 128);
        } else if (batch_size == 64 && chunk_size == 128) {
            CALL_ATTN_CHUNK_KERNEL_FUNCTION(half, 64, 128, 128);
        } else {
            LOG_ERROR("unsupported chunk_size {} or batch_size {}", chunk_size, batch_size);
            TORCH_CHECK(false, "unsupported chunk_size ", chunk_size, " or batch_size ", batch_size);
        }
    } else {
        LOG_ERROR("unsupported data type {}", query.dtype());
        TORCH_CHECK(false, "unsupported data type ", query.dtype());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

__host__ void attn_seqs_first(KernelContext& context,
                              const torch::Tensor& query,
                              torch::Tensor& output) {
    printf("DEBUG: calling attn_seqs_first\n");
    uint32_t n_seqs = query.size(0);
    if (context.d_head != 128) {
        LOG_ERROR("unsupported head dim {}", context.d_head);
        TORCH_CHECK(false, "unsupported head dim ", context.d_head);
    }

    float scale = 1.f / sqrtf(static_cast<float>(context.d_head));
    int shared_mem_size =
      4 * (context.d_head * context.chunk_size * sizeof(half) + context.chunk_size * 16);
    dim3 grid(context.n_heads, n_seqs);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();