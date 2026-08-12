#include <cmath>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

// half inputs are widened once and all the math below runs in float.
__device__ __forceinline__ float toFloat(float x) { return x; }
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T fromFloat(float x);
template <>
__device__ __forceinline__ float fromFloat<float>(float x) { return x; }
template <>
__device__ __forceinline__ half fromFloat<half>(float x) { return __float2half(x); }

__device__ __forceinline__ float warpReduceSum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

__device__ __forceinline__ float warpReduceMax(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset));
  return x;
}

// VEC is chosen host-side so VEC * sizeof(T) == 16, i.e. one 16-byte access.
template <typename T, int VEC>
struct alignas(sizeof(T) * VEC) Pack {
  T v[VEC];
};

// One block per row, two passes over it: sum of squares, then scale and store.
template <typename T, int VEC>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output, size_t hidden_dim,
                              float inv_hidden_dim, float eps) {
  using PackT = Pack<T, VEC>;

  const size_t row = blockIdx.x;
  const size_t packs = hidden_dim / VEC;  // divisibility checked host-side
  const PackT* in_row = reinterpret_cast<const PackT*>(input + row * hidden_dim);
  const PackT* w_row = reinterpret_cast<const PackT*>(weight);
  PackT* out_row = reinterpret_cast<PackT*>(output + row * hidden_dim);

  float sum = 0.0f;
  for (size_t i = threadIdx.x; i < packs; i += blockDim.x) {
    const PackT p = in_row[i];
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      const float x = toFloat(p.v[j]);
      sum += x * x;
    }
  }

  __shared__ float s_warp_sum[32];
  __shared__ float s_scale;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  sum = warpReduceSum(sum);
  if (lane == 0) s_warp_sum[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = (lane < num_warps) ? s_warp_sum[lane] : 0.0f;
    sum = warpReduceSum(sum);
    if (lane == 0) s_scale = rsqrtf(sum * inv_hidden_dim + eps);
  }
  __syncthreads();
  const float scale = s_scale;

  for (size_t i = threadIdx.x; i < packs; i += blockDim.x) {
    const PackT p = in_row[i];
    const PackT w = w_row[i];
    PackT out;
#pragma unroll
    for (int j = 0; j < VEC; ++j)
      out.v[j] = fromFloat<T>(toFloat(p.v[j]) * scale * toFloat(w.v[j]));
    out_row[i] = out;
  }
}

inline size_t attnSharedFloats(int block_m, int block_n, int d_stride) {
  return static_cast<size_t>(block_m) * d_stride * 2      // Q tile, O accumulator
       + static_cast<size_t>(block_n) * d_stride * 2      // K tile, V tile
       + static_cast<size_t>(block_m) * block_n           // score tile
       + static_cast<size_t>(block_m) * 2;                // row max, row sum
}

// A block owns block_m query rows of one (batch, query head) and walks the key
// sequence in tiles of block_n, so only one score tile is ever live. Tiles are
// held in shared memory as float regardless of T.
//
// Per-head-dim rows carry one float of padding (d_stride) so the score loop can
// walk K row by row without bank conflicts.
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v,
    T* __restrict__ o, int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, int group_size, float scale, bool is_causal,
    int block_m, int block_n) {
  extern __shared__ float smem[];

  const int d_stride = head_dim + 1;
  float* s_q = smem;
  float* s_k = s_q + block_m * d_stride;
  float* s_v = s_k + block_n * d_stride;
  float* s_o = s_v + block_n * d_stride;
  float* s_score = s_o + block_m * d_stride;
  float* s_max = s_score + block_m * block_n;
  float* s_sum = s_max + block_m;

  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int batch = blockIdx.z;
  const int head = blockIdx.y;
  const int kv_head = head / group_size;  // GQA
  const int m_base = blockIdx.x * block_m;

  for (int idx = tid; idx < block_m * head_dim; idx += threads) {
    const int m = idx / head_dim;
    const int d = idx - m * head_dim;
    const int row = m_base + m;
    float value = 0.0f;
    if (row < target_seq_len) {
      const size_t off =
          ((static_cast<size_t>(batch) * target_seq_len + row) * query_heads + head) *
              head_dim + d;
      value = toFloat(q[off]);
    }
    s_q[m * d_stride + d] = value;
    s_o[m * d_stride + d] = 0.0f;
  }
  for (int m = tid; m < block_m; m += threads) {
    s_max[m] = -INFINITY;
    s_sum[m] = 0.0f;
  }

  // Causal mask is top-left aligned (key j survives only for j <= i), so key
  // tiles past this block's largest query index can be skipped outright.
  int n_limit = src_seq_len;
  if (is_causal) {
    const int i_max = min(m_base + block_m - 1, target_seq_len - 1);
    n_limit = min(src_seq_len, i_max + 1);
  }

  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = threads >> 5;

  // Pass 0 collects the exact row maxima, pass 1 exponentiates against them and
  // accumulates. Deliberately not a single-pass online softmax: rescaling the
  // partial sums rounds differently from a plain softmax, and rows whose
  // weighted value sum cancels do not survive that.
  for (int pass = 0; pass < 2; ++pass) {
    for (int n_base = 0; n_base < n_limit; n_base += block_n) {
      __syncthreads();  // last iteration has finished reading K/V

      for (int idx = tid; idx < block_n * head_dim; idx += threads) {
        const int n = idx / head_dim;
        const int d = idx - n * head_dim;
        const int row = n_base + n;
        float kv_k = 0.0f, kv_v = 0.0f;
        if (row < src_seq_len) {
          const size_t off =
              ((static_cast<size_t>(batch) * src_seq_len + row) * kv_heads + kv_head) *
                  head_dim + d;
          kv_k = toFloat(k[off]);
          if (pass == 1) kv_v = toFloat(v[off]);
        }
        s_k[n * d_stride + d] = kv_k;
        if (pass == 1) s_v[n * d_stride + d] = kv_v;
      }
      __syncthreads();

      // Raw Q * K^T, masked entries set to -inf. 1/sqrt(head_dim) is left out
      // here on purpose and folded into the exponent below as one FMA.
      //
      // Each thread takes four keys of one row, a quarter of the tile apart:
      // four independent chains for the FMA pipeline, one query load shared by
      // all four, and neighbouring threads land one K row apart (no conflicts).
      const int quarter = block_n >> 2;
      for (int base = tid; base < block_m * quarter; base += threads) {
        const int m = base / quarter;
        const int n0 = base - m * quarter;
        const float* qp = s_q + m * d_stride;
        const float* kp = s_k + n0 * d_stride;
        const int step = quarter * d_stride;

        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#pragma unroll 4
        for (int d = 0; d < head_dim; ++d) {
          const float qv = qp[d];
          a0 = __fmaf_rn(qv, kp[d], a0);
          a1 = __fmaf_rn(qv, kp[step + d], a1);
          a2 = __fmaf_rn(qv, kp[2 * step + d], a2);
          a3 = __fmaf_rn(qv, kp[3 * step + d], a3);
        }

        const float dots[4] = {a0, a1, a2, a3};
        const int q_row = m_base + m;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int n = n0 + j * quarter;
          const int k_row = n_base + n;
          const bool keep = q_row < target_seq_len && k_row < src_seq_len &&
                            (!is_causal || k_row <= q_row);
          s_score[m * block_n + n] = keep ? dots[j] : -INFINITY;
        }
      }
      __syncthreads();

      if (pass == 0) {
        for (int m = warp; m < block_m; m += num_warps) {
          const float* row = s_score + m * block_n;
          float tile_max = -INFINITY;
          for (int n = lane; n < block_n; n += 32) tile_max = fmaxf(tile_max, row[n]);
          tile_max = warpReduceMax(tile_max);
          if (lane == 0) s_max[m] = fmaxf(s_max[m], tile_max);
        }
      } else {
        // Scores become probabilities in place. Scaling the max once and fusing
        // the per-element scale into the subtraction rounds the exponent
        // argument once instead of twice.
        for (int idx = tid; idx < block_m * block_n; idx += threads) {
          const float dot = s_score[idx];
          const float row_max = s_max[idx / block_n] * scale;
          s_score[idx] = (dot == -INFINITY)
                             ? 0.0f
                             : expf(__fmaf_rn(dot, scale, -row_max));
        }
        __syncthreads();

        // Strictly left to right: the denominator has to round the way an
        // untiled softmax rounds it. Masked entries add an exact 0. The chain
        // is serial, so rows get separate threads instead of sharing a warp.
        for (int m = tid; m < block_m; m += threads) {
          const float* row = s_score + m * block_n;
          float row_sum = s_sum[m];
          for (int n = 0; n < block_n; ++n) row_sum += row[n];
          s_sum[m] = row_sum;
        }

        // No barrier here, the loop below only reads probabilities. O += P * V,
        // extending the running sum instead of adding a per-tile subtotal. Four
        // accumulators again, this time four query rows sharing a V element.
        const int quarter_m = block_m >> 2;
        const int p_step = quarter_m * block_n;
        for (int base = tid; base < quarter_m * head_dim; base += threads) {
          const int m = base / head_dim;
          const int d = base - m * head_dim;
          const float* p = s_score + m * block_n;
          float* dst = s_o + m * d_stride + d;
          const int o_step = quarter_m * d_stride;

          float a0 = dst[0], a1 = dst[o_step];
          float a2 = dst[2 * o_step], a3 = dst[3 * o_step];
#pragma unroll 4
          for (int n = 0; n < block_n; ++n) {
            const float vv = s_v[n * d_stride + d];
            a0 = __fmaf_rn(p[n], vv, a0);
            a1 = __fmaf_rn(p[p_step + n], vv, a1);
            a2 = __fmaf_rn(p[2 * p_step + n], vv, a2);
            a3 = __fmaf_rn(p[3 * p_step + n], vv, a3);
          }
          dst[0] = a0;
          dst[o_step] = a1;
          dst[2 * o_step] = a2;
          dst[3 * o_step] = a3;
        }
      }
    }
  }

  __syncthreads();

  // Reciprocal first, then multiply: that is the rounding the reference uses.
  for (int idx = tid; idx < block_m * head_dim; idx += threads) {
    const int m = idx / head_dim;
    const int d = idx - m * head_dim;
    const int row = m_base + m;
    if (row >= target_seq_len) continue;
    const float denom = s_sum[m];
    const float value =
        (denom > 0.0f) ? s_o[m * d_stride + d] * (1.0f / denom) : 0.0f;
    const size_t off =
        ((static_cast<size_t>(batch) * target_seq_len + row) * query_heads + head) *
            head_dim + d;
    o[off] = fromFloat<T>(value);
  }
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  const size_t elems = rows * hidden_dim;
  if (elems == 0) return;

  // One allocation for all three buffers: the caller times the whole function.
  const size_t total = elems * 2 + hidden_dim;
  T* d_base = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_base, total * sizeof(T)));
  T* d_input = d_base;
  T* d_weight = d_input + elems;
  T* d_output = d_weight + hidden_dim;

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T),
                           cudaMemcpyHostToDevice));

  // Vectorised when hidden_dim is a multiple of the pack width, which also
  // keeps every row start 16-byte aligned. Scalar otherwise.
  constexpr int kVec = 16 / sizeof(T);
  const size_t packs = hidden_dim / ((hidden_dim % kVec == 0) ? kVec : 1);
  const size_t warps = (packs + 31) / 32;
  const int threads = static_cast<int>((warps > 32 ? 32 : warps) * 32);
  const float inv_hidden_dim = 1.0f / static_cast<float>(hidden_dim);

  if (hidden_dim % kVec == 0) {
    rmsNormKernel<T, kVec><<<rows, threads>>>(d_input, d_weight, d_output,
                                              hidden_dim, inv_hidden_dim, eps);
  } else {
    rmsNormKernel<T, 1><<<rows, threads>>>(d_input, d_weight, d_output,
                                           hidden_dim, inv_hidden_dim, eps);
  }
  RUNTIME_CHECK(cudaGetLastError());

  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, elems * sizeof(T),
                           cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_base));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const size_t q_elems = static_cast<size_t>(batch_size) * target_seq_len *
                         query_heads * head_dim;
  const size_t kv_elems = static_cast<size_t>(batch_size) * src_seq_len *
                          kv_heads * head_dim;
  if (q_elems == 0) return;

  T* d_base = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_base, (q_elems * 2 + kv_elems * 2) * sizeof(T)));
  T* d_q = d_base;
  T* d_k = d_q + q_elems;
  T* d_v = d_k + kv_elems;
  T* d_o = d_v + kv_elems;

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(T),
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(T),
                           cudaMemcpyHostToDevice));

  // Largest tiling that fits the 48 KB every device gives a block without an
  // opt-in. Every block_n here is a multiple of 4, which the score loop needs.
  constexpr size_t kSharedBudget = 48 * 1024;
  const int d_stride = head_dim + 1;
  const int candidates[][2] = {{64, 64}, {32, 64}, {32, 32},
                               {32, 16}, {16, 16}, {8, 8}, {4, 4}};
  int block_m = 0, block_n = 0;
  size_t shared_bytes = 0;
  for (const auto& c : candidates) {
    const size_t bytes = attnSharedFloats(c[0], c[1], d_stride) * sizeof(float);
    if (bytes <= kSharedBudget) {
      block_m = c[0];
      block_n = c[1];
      shared_bytes = bytes;
      break;
    }
  }
  if (block_m == 0) {
    std::cerr << "head_dim " << head_dim
              << " is too large for the shared-memory budget\n";
    exit(EXIT_FAILURE);
  }

  const dim3 grid((target_seq_len + block_m - 1) / block_m, query_heads,
                  batch_size);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  flashAttentionKernel<T><<<grid, 512, shared_bytes>>>(
      d_q, d_k, d_v, d_o, target_seq_len, src_seq_len, query_heads, kv_heads,
      head_dim, query_heads / kv_heads, scale, is_causal, block_m, block_n);
  RUNTIME_CHECK(cudaGetLastError());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, q_elems * sizeof(T),
                           cudaMemcpyDeviceToHost));
  RUNTIME_CHECK(cudaFree(d_base));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
