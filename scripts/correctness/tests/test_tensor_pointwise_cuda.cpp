// Descriptor correctness, including batch-broadcast materialization and cache
// reuse.
#include "polygeist_cublas_rt.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime_api.h>
#include <vector>
#define CU(x)                                                                  \
  do {                                                                         \
    auto e = (x);                                                              \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "%s\n", cudaGetErrorString(e));                          \
      return 90;                                                               \
    }                                                                          \
  } while (0)
int main() {
  const int64_t shapes[][3] = {{1, 3, 7}, {2, 5, 11}, {7, 1, 13}, {3, 9, 1}};
  size_t checked = 0;
  for (const auto &shape : shapes)
    for (int axis = 1; axis <= 2; ++axis) {
      PolygeistPointwiseTensorLayout layout = {};
      layout.rank = 3;
      layout.inputs = 2;
      layout.nodes = 2;
      int64_t total = 1, small = shape[axis];
      for (int d = 2; d >= 0; --d) {
        layout.dims[d] = shape[d];
        layout.output_strides[d] = total;
        total *= shape[d];
      }
      int64_t dense = 1, broadcast = 1;
      for (int d = 2; d >= 0; --d) {
        layout.input_dims[0][d] = shape[d];
        layout.input_strides[0][d] = dense;
        dense *= shape[d];
        layout.input_dims[1][d] = d == axis ? shape[d] : 1;
        layout.input_strides[1][d] = broadcast;
        broadcast *= layout.input_dims[1][d];
      }
      // (broadcast + full) * scalar: descriptor emission must commute ADD's
      // operands when cuDNN requires the full tensor first, without
      // reassociation.
      uint32_t add = (1u << 24) | (1u << 16) | (0u << 8),
               mul = (2u << 24) | (16u << 16) | (8u << 8);
      layout.words[0] = uint64_t(add) | (uint64_t(mul) << 32);
      constexpr int guard = 16;
      const float sentinel = 123456.0f;
      std::vector<float> full(total + 2 * guard, sentinel),
          vec(small + 2 * guard, sentinel), out(total + 2 * guard, sentinel),
          saved;
      float *df, *dv, *dy;
      CU(cudaMalloc((void **)&df, full.size() * 4));
      CU(cudaMalloc((void **)&dv, vec.size() * 4));
      CU(cudaMalloc((void **)&dy, out.size() * 4));
      const float *inputs[] = {df + guard, dv + guard};
      for (int repeat = 0; repeat < 2; ++repeat) {
        for (int64_t i = 0; i < total; ++i)
          full[guard + i] = float((i * 17 + repeat * 11) % 31 - 15) / 8;
        for (int64_t i = 0; i < small; ++i)
          vec[guard + i] = float((i * 7 + repeat * 3) % 19 - 9) / 4;
        std::fill(out.begin(), out.end(), sentinel);
        CU(cudaMemcpy(df, full.data(), full.size() * 4,
                      cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dv, vec.data(), vec.size() * 4, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(dy, out.data(), out.size() * 4, cudaMemcpyHostToDevice));
        float scalar[8] = {repeat ? 0.5f : -0.25f};
        polygeist_cudnn_tensor_pointwise_f32(&layout, inputs, scalar,
                                             dy + guard);
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost));
        for (int64_t i = 0; i < total; ++i) {
          int64_t coordinate = (i / layout.output_strides[axis]) % shape[axis];
          float expected =
              (vec[guard + coordinate] + full[guard + i]) * scalar[0];
          if (out[guard + i] != expected) {
            fprintf(stderr, "wrong value shape=%lld,%lld,%lld axis=%d i=%lld\n",
                    (long long)shape[0], (long long)shape[1],
                    (long long)shape[2], axis, (long long)i);
            return 1;
          }
        }
        for (int i = 0; i < guard; ++i)
          if (out[i] != sentinel || out[guard + total + i] != sentinel)
            return 2;
        saved.resize(full.size());
        CU(cudaMemcpy(saved.data(), df, full.size() * 4,
                      cudaMemcpyDeviceToHost));
        if (saved != full)
          return 3;
        saved.resize(vec.size());
        CU(cudaMemcpy(saved.data(), dv, vec.size() * 4,
                      cudaMemcpyDeviceToHost));
        if (saved != vec)
          return 4;
        checked += total;
      }
      CU(cudaFree(df));
      CU(cudaFree(dv));
      CU(cudaFree(dy));
    }
  printf("PASS tensor pointwise: %zu values; projected broadcasts, batch "
         "copies, scalar and pointer cache reuse, guards, unchanged inputs\n",
         checked);
}
