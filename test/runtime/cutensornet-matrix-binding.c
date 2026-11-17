// RUN: clang -O2 -I%polygeist_src_root/runtime %s -lm -o %t
// RUN: %t
// RUN: not %t invalid 2>&1 | FileCheck %s
// CHECK: invalid cuTensorNet matrix layout

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef POLYGEIST_TEST_CUDA
#include <cuda_runtime_api.h>
#include "polygeist_cublas_rt.h"
#define GPU_CHECK(call) do { if ((call) != cudaSuccess) abort(); } while (0)
#else
/* Independent interpreter for the existing tensor-network ABI. This test
 * checks the adapter's modes, extents, leading dimensions and accumulation,
 * without requiring a GPU on compiler CI. */
static void polygeist_cutensornet_network_f64_device(
    const int64_t *pointers, const int64_t *metadata) {
  if (metadata[0] != 1 || metadata[1] != 2) abort();
  int64_t extents[3] = {0}, strides[3][3] = {{0}};
  int present[3][3] = {{0}};
  int cursor = 6;
  for (int tensor = 0; tensor < 3; ++tensor) {
    if (metadata[3 + tensor] != 2) abort();
    for (int dim = 0; dim < 2; ++dim) {
      int64_t n = metadata[cursor++], stride = metadata[cursor++];
      int mode = metadata[cursor++];
      if (mode < 0 || mode >= 3 || (extents[mode] && extents[mode] != n))
        abort();
      extents[mode] = n;
      strides[tensor][mode] = stride;
      present[tensor][mode] = 1;
    }
  }
  const double *a = (const double *)(uintptr_t)pointers[0];
  const double *b = (const double *)(uintptr_t)pointers[1];
  double *c = (double *)(uintptr_t)pointers[2];
  for (int64_t x = 0; x < extents[0]; ++x)
    for (int64_t y = 0; y < extents[1]; ++y)
      for (int64_t z = 0; z < extents[2]; ++z) {
        int64_t coords[3] = {x, y, z}, offsets[3] = {0};
        int first = 1;
        for (int mode = 0; mode < 3; ++mode) {
          for (int tensor = 0; tensor < 3; ++tensor)
            offsets[tensor] += coords[mode] * strides[tensor][mode];
          if (!present[2][mode] && coords[mode]) first = 0;
        }
        if (first && !metadata[2]) c[offsets[2]] = 0;
        c[offsets[2]] += a[offsets[0]] * b[offsets[1]];
      }
}

#define POLYGEIST_HAS_CUTENSORNET 1
#include "polygeist_cutensornet_matrix.inc"
#endif

int main(int argc, char **argv) {
  enum { M = 3, N = 5, K = 4, LD = 8, CAP = 64 };
  double a[CAP], b[CAP], c[CAP], expected[CAP], saved_a[CAP], saved_b[CAP];
  for (int ta = 0; ta <= 1; ++ta)
    for (int tb = 0; tb <= 1; ++tb)
      for (int accumulate = 0; accumulate <= 1; ++accumulate) {
        for (int q = 0; q < CAP; ++q) {
          a[q] = 91; b[q] = -73; c[q] = expected[q] = 19;
        }
        for (int i = 0; i < M; ++i)
          for (int k = 0; k < K; ++k)
            a[ta ? i * LD + k : k * LD + i] = (i - 2 * k) * 0.25;
        for (int k = 0; k < K; ++k)
          for (int j = 0; j < N; ++j)
            b[tb ? k * LD + j : j * LD + k] = (3 * k + j - 2) * 0.5;
        for (int i = 0; i < M; ++i)
          for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k)
              sum += (i - 2 * k) * 0.25 * (3 * k + j - 2) * 0.5;
            expected[j * LD + i] = sum + (accumulate ? 19 : 0);
          }
        memcpy(saved_a, a, sizeof(a)); memcpy(saved_b, b, sizeof(b));
#ifdef POLYGEIST_TEST_CUDA
        double *da, *db, *dc;
        GPU_CHECK(cudaMalloc((void **)&da, sizeof(a)));
        GPU_CHECK(cudaMalloc((void **)&db, sizeof(b)));
        GPU_CHECK(cudaMalloc((void **)&dc, sizeof(c)));
        GPU_CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
        GPU_CHECK(cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice));
        polygeist_cutensornet_matrix_f64(M, N, K, argc > 1 ? 1 : LD,
                                        da, LD, db, LD, dc, ta, tb, accumulate);
        GPU_CHECK(cudaDeviceSynchronize());
        GPU_CHECK(cudaMemcpy(a, da, sizeof(a), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(b, db, sizeof(b), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
        GPU_CHECK(cudaFree(da)); GPU_CHECK(cudaFree(db)); GPU_CHECK(cudaFree(dc));
#else
        polygeist_cutensornet_matrix_f64(M, N, K, argc > 1 ? 1 : LD,
                                        a, LD, b, LD, c, ta, tb, accumulate);
#endif
        if (memcmp(a, saved_a, sizeof(a)) || memcmp(b, saved_b, sizeof(b))) return 1;
        for (int q = 0; q < CAP; ++q)
          if (fabs(c[q] - expected[q]) > 1e-12) return 2;
      }
  for (int trans = 0; trans <= 1; ++trans)
    for (int accumulate = 0; accumulate <= 1; ++accumulate) {
      int output = trans ? N : M, reduction = trans ? M : N;
      for (int q = 0; q < CAP; ++q) {
        a[q] = 91; b[q] = -73; c[q] = expected[q] = 19;
      }
      for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
          a[j * LD + i] = (i - 2 * j) * 0.25;
      for (int k = 0; k < reduction; ++k) b[k] = (k - 2) * 0.5;
      for (int i = 0; i < output; ++i) {
        double sum = 0;
        for (int k = 0; k < reduction; ++k)
          sum += (trans ? (k - 2 * i) : (i - 2 * k)) * 0.25 * (k - 2) * 0.5;
        expected[i] = sum + (accumulate ? 19 : 0);
      }
      memcpy(saved_a, a, sizeof(a)); memcpy(saved_b, b, sizeof(b));
#ifdef POLYGEIST_TEST_CUDA
      double *da, *db, *dc;
      GPU_CHECK(cudaMalloc((void **)&da, sizeof(a)));
      GPU_CHECK(cudaMalloc((void **)&db, sizeof(b)));
      GPU_CHECK(cudaMalloc((void **)&dc, sizeof(c)));
      GPU_CHECK(cudaMemcpy(da, a, sizeof(a), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(db, b, sizeof(b), cudaMemcpyHostToDevice));
      GPU_CHECK(cudaMemcpy(dc, c, sizeof(c), cudaMemcpyHostToDevice));
      polygeist_cutensornet_matvec_f64(M, N, LD, da, db, dc, trans, accumulate);
      GPU_CHECK(cudaDeviceSynchronize());
      GPU_CHECK(cudaMemcpy(a, da, sizeof(a), cudaMemcpyDeviceToHost));
      GPU_CHECK(cudaMemcpy(b, db, sizeof(b), cudaMemcpyDeviceToHost));
      GPU_CHECK(cudaMemcpy(c, dc, sizeof(c), cudaMemcpyDeviceToHost));
      GPU_CHECK(cudaFree(da)); GPU_CHECK(cudaFree(db)); GPU_CHECK(cudaFree(dc));
#else
      polygeist_cutensornet_matvec_f64(M, N, LD, a, b, c, trans, accumulate);
#endif
      if (memcmp(a, saved_a, sizeof(a)) || memcmp(b, saved_b, sizeof(b))) return 3;
      for (int q = 0; q < CAP; ++q)
        if (fabs(c[q] - expected[q]) > 1e-12) return 4;
    }
  puts("cuTensorNet bindings: eight matrix and four matrix-vector cases PASS");
  return 0;
}
