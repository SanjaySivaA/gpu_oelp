#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include "flash_attn_kernels.h"

// ==========================================
// GLOBALS & MACROS
// ==========================================
#define d 64
#define Br_32 4
#define Bc_32 4

// Marked static so it doesn't conflict with the FP16 file during linking
static __device__ float fmax_val(float a, float b) {
    return a > b ? a : b;
}

// ==========================================
// FP32 KERNEL (Standard Cores)
// ==========================================
__global__ void flash_attention_4d(float *Q, float *K, float *V, float *O, int B, int nh, int N) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row_chunk = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = row_chunk * Br_32 + ty;

    int head_offset = (b * nh * N * d) + (h * N * d);

    __shared__ float s_Q[Br_32][d];
    __shared__ float s_K[Bc_32][d];
    __shared__ float s_V[Bc_32][d];

    float acc_o = 0.0f;
    float l = 0.0f;
    float m = -1e20f;

    if(row < N){
        s_Q[ty][tx] = Q[head_offset + row * d + tx];
    } else {
        s_Q[ty][tx] = 0.0f;
    }
    __syncthreads();

    for(int j = 0; j < N; j += Bc_32) {
        int col = j + ty;

        if(col < N){
            s_K[ty][tx] = K[head_offset + col * d + tx];
            s_V[ty][tx] = V[head_offset + col * d + tx];
        } else {
            s_K[ty][tx] = 0.0f;
            s_V[ty][tx] = 0.0f;
        }
        __syncthreads();

        for(int k_idx = 0; k_idx < Bc_32; k_idx++){
            int global_col = j + k_idx;
            if (global_col >= N) continue;
            if (global_col > row) continue; // Causal Mask

            float score = 0.0f;
            for (int dim = 0; dim < d; dim++) {
                score += s_Q[ty][dim] * s_K[k_idx][dim];
            }
            score *= (1.0f / sqrtf((float)d));

            float m_prev = m;
            m = fmax_val(m_prev, score);
            float P = expf(score - m);
            float scale = expf(m_prev - m);

            l = (l * scale) + P;
            acc_o = (acc_o * scale) + (P * s_V[k_idx][tx]);
        }
        __syncthreads();
    }

    if (row < N) {
        O[head_offset + row * d + tx] = acc_o / l;
    }
}

// C++ Wrapper
torch::Tensor forward_fp32(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);

    auto O = torch::empty_like(Q);

    dim3 threads(d, Br_32);
    dim3 grid((N + Br_32 - 1) / Br_32, nh, B);

    flash_attention_4d<<<grid, threads>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        B, nh, N
    );

    return O;
}