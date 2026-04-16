#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "flash_attn_kernels.h"

using namespace nvcuda;

// ==========================================
// GLOBALS & MACROS
// ==========================================
#define d 64
#define Br_16 16
#define Bc_16 16

static __device__ float fmax_val(float a, float b) {
    return a > b ? a : b;
}

// ==========================================
// FP16 KERNEL (Tensor Cores)
// ==========================================
__global__ void flash_attention_tensor_cores(half *Q, half *K, half *V, float *O, int B, int nh, int N) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row_chunk = blockIdx.x;
    int tid = threadIdx.x;

    int head_offset = (b * nh * N * d) + (h * N * d);
    int row_start = row_chunk * Br_16;

    __shared__ half s_Q[Br_16][d];
    __shared__ half s_K[Bc_16][d];
    __shared__ half s_V[Bc_16][d];
    __shared__ float s_S[Br_16][Bc_16];
    __shared__ half s_P[Br_16][Bc_16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_Q;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_K;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_P;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_V;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_S;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_O[d / 16];
    for(int i = 0; i < (d / 16); i++) {
        wmma::fill_fragment(frag_O[i], 0.0f);
    }

    float m_arr[Br_16];
    float l_arr[Br_16];
    for(int i = 0; i < Br_16; i++) {
        m_arr[i] = -1e20f;
        l_arr[i] = 0.0f;
    }

    for (int i = tid; i < Br_16 * d; i += 32) {
        int r = i / d;
        int c = i % d;
        s_Q[r][c] = (row_start + r < N) ? Q[head_offset + (row_start + r) * d + c] : __float2half(0.0f);
    }
    __syncthreads();

    for(int j = 0; j < N; j += Bc_16) {
        for (int i = tid; i < Bc_16 * d; i += 32) {
            int r = i / d;
            int c = i % d;
            s_K[r][c] = (j + r < N) ? K[head_offset + (j + r) * d + c] : __float2half(0.0f);
            s_V[r][c] = (j + r < N) ? V[head_offset + (j + r) * d + c] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fill_fragment(frag_S, 0.0f);

        for (int dim = 0; dim < d; dim += 16) {
            wmma::load_matrix_sync(frag_Q, &s_Q[0][dim], d);
            wmma::load_matrix_sync(frag_K, &s_K[0][dim], d);
            wmma::mma_sync(frag_S, frag_Q, frag_K, frag_S);
        }

        wmma::store_matrix_sync(&s_S[0][0], frag_S, Bc_16, wmma::mem_row_major);
        __syncthreads();

        if (tid < Br_16) {
            float row_max = -1e20f;
            for(int c = 0; c < Bc_16; c++) {
                s_S[tid][c] *= (1.0f / sqrtf((float)d));
                row_max = fmax_val(row_max, s_S[tid][c]);
            }

            float m_prev = m_arr[tid];
            m_arr[tid] = fmax_val(m_prev, row_max);
            float scale = expf(m_prev - m_arr[tid]);

            float row_sum = 0.0f;
            for(int c = 0; c < Bc_16; c++) {
                float p = expf(s_S[tid][c] - m_arr[tid]);
                s_P[tid][c] = __float2half(p);
                row_sum += p;
            }
            l_arr[tid] = (l_arr[tid] * scale) + row_sum;
        }
        __syncthreads();

        wmma::load_matrix_sync(frag_P, &s_P[0][0], Bc_16);
        for (int dim = 0; dim < d; dim += 16) {
            wmma::load_matrix_sync(frag_V, &s_V[0][dim], d);
            wmma::mma_sync(frag_O[dim/16], frag_P, frag_V, frag_O[dim/16]);
        }
        __syncthreads();
    }
}

// C++ Wrapper
torch::Tensor forward_fp16(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    int B = Q.size(0);
    int nh = Q.size(1);
    int N = Q.size(2);

    auto O = torch::empty({B, nh, N, d}, torch::TensorOptions().dtype(torch::kFloat32).device(Q.device()));

    dim3 threads(32);
    dim3 grid((N + Br_16 - 1) / Br_16, nh, B);

    flash_attention_tensor_cores<<<grid, threads>>>(
        reinterpret_cast<half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<half*>(K.data_ptr<at::Half>()),
        reinterpret_cast<half*>(V.data_ptr<at::Half>()),
        O.data_ptr<float>(),
        B, nh, N
    );

    return O;
}