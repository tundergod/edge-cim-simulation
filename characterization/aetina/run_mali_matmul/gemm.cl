// Phase 0.3 A5 — tiled GEMM kernels for Mali-G610 (FP32 + FP16).
// C[M,N] = A[M,K] * B[K,N]. One work-item per C element; TILE x TILE local-memory tiling.
// FP16 accumulates in float for accuracy. Used to characterize GPU matmul + attention bmm
// (native activation x activation, the offload alternative to CIM).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TILE 16

__kernel void gemm_f32(const int M, const int K, const int N,
                       __global const float* A, __global const float* B, __global float* C) {
    __local float As[TILE][TILE];
    __local float Bs[TILE][TILE];
    int lr = get_local_id(1), lc = get_local_id(0);
    int row = get_global_id(1), col = get_global_id(0);
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + lc, bk = t * TILE + lr;
        As[lr][lc] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        Bs[lr][lc] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k) acc += As[lr][k] * Bs[k][lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

__kernel void gemm_f16(const int M, const int K, const int N,
                       __global const half* A, __global const half* B, __global half* C) {
    __local half As[TILE][TILE];
    __local half Bs[TILE][TILE];
    int lr = get_local_id(1), lc = get_local_id(0);
    int row = get_global_id(1), col = get_global_id(0);
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ak = t * TILE + lc, bk = t * TILE + lr;
        As[lr][lc] = (row < M && ak < K) ? A[row * K + ak] : (half)0;
        Bs[lr][lc] = (bk < K && col < N) ? B[bk * N + col] : (half)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; ++k) acc += (float)As[lr][k] * (float)Bs[k][lc];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = (half)acc;
}
