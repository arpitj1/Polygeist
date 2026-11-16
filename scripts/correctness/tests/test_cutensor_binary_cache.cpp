// Link with the CUDA runtime and its cuTENSOR/Thrust dependencies.
// Exercise reuse, eviction, and destruction using different shapes and buffers.
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void polygeist_cutensor_kronecker_product2d_f32(
    int32_t, int32_t, int32_t, int32_t, const float *, const float *, float *);
extern "C" void polygeist_cublas_destroy();

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(result));
    abort();
  }
}

static void run(int a, int b, int c, int d, int seed) {
  std::vector<float> x(a * b), y(c * d), out(a * b * c * d);
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = (int(i) % 11 - 5 + seed) * 0.25f;
  for (size_t i = 0; i < y.size(); ++i)
    y[i] = (int(i) % 7 - 3 - seed) * 0.5f;
  x[0] = -0.0f;
  y[0] = 1.0f;
  float *dx, *dy, *dz;
  check(cudaMalloc(reinterpret_cast<void **>(&dx), x.size() * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void **>(&dy), y.size() * sizeof(float)));
  check(cudaMalloc(reinterpret_cast<void **>(&dz), out.size() * sizeof(float)));
  check(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dy, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice));
  // Old output contents must not affect a product, including its signed zero.
  check(cudaMemset(dz, 0xff, out.size() * sizeof(float)));
  polygeist_cutensor_kronecker_product2d_f32(a, b, c, d, dx, dy, dz);
  check(cudaMemcpy(out.data(), dz, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < a; ++i)
    for (int j = 0; j < c; ++j)
      for (int k = 0; k < b; ++k)
        for (int l = 0; l < d; ++l) {
          float expected = x[i * b + k] * y[j * d + l];
          float actual = out[((i * c + j) * b + k) * d + l];
          if (memcmp(&expected, &actual, sizeof(float)) != 0) {
            fprintf(stderr, "Mismatch at shape %d,%d,%d,%d\n", a, b, c, d);
            abort();
          }
        }
  check(cudaFree(dx));
  check(cudaFree(dy));
  check(cudaFree(dz));
}

int main() {
  run(2, 3, 4, 5, 0);
  run(2, 3, 4, 5, 1); // Reuse with different contents and allocations.
  for (int i = 1; i <= 20; ++i)
    run(i, 2 + i % 3, 1 + i % 4, 3, i); // Exceed the 16-plan cache.
  run(2, 3, 4, 5, 2); // Rebuild an evicted shape.
  polygeist_cublas_destroy();
  run(2, 3, 4, 5, 3); // Reinitialize after explicit cleanup.
  polygeist_cublas_destroy();
  puts("PASS: cuTENSOR plan reuse, eviction, signed zero, and reinitialization");
}
