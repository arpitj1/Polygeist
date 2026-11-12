// Exact power-of-two products expose row-boundary, seed and buffer errors.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
extern "C" void polygeist_cub_row_product_scan_f32(
    int, int, float, const float *, float *);

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  const int shapes[][2] = {{1,1}, {3,7}, {5,513}, {2,2049}};
  size_t checked = 0;
  for (const auto &shape : shapes) {
    const int rows = shape[0], cols = shape[1];
    const size_t count = size_t(rows) * cols, bytes = (count + 6) * sizeof(float);
    std::vector<float> input(count + 6, 123.0f), result(count + 6, 123.0f);
    for (size_t i = 0; i < count; ++i) {
      float v = i % 3 == 0 ? 0.5f : i % 3 == 1 ? 1.0f : 2.0f;
      input[i + 3] = i % 2 ? -v : v;
    }
    float *dx = nullptr, *dy = nullptr;
    check(cudaMalloc(&dx, bytes));
    check(cudaMalloc(&dy, bytes));
    check(cudaMemcpy(dx, input.data(), bytes, cudaMemcpyHostToDevice));
    for (float seed : {-0.5f, 1.0f, 0.0f, -0.0f}) {
      std::fill(result.begin(), result.end(), 123.0f);
      check(cudaMemcpy(dy, result.data(), bytes, cudaMemcpyHostToDevice));
      polygeist_cub_row_product_scan_f32(rows, cols, seed, dx + 3, dy + 3);
      check(cudaMemcpy(result.data(), dy, bytes, cudaMemcpyDeviceToHost));
      for (int r = 0; r < rows; ++r) {
        float value = seed;
        for (int c = 0; c < cols; ++c) {
          const size_t i = size_t(r) * cols + c + 3;
          value *= input[i];
          if (std::memcmp(&value, &result[i], sizeof(float))) {
            std::fprintf(stderr, "row scan mismatch rows=%d cols=%d row=%d col=%d seed=%g\n",
                         rows, cols, r, c, seed);
            return 1;
          }
          ++checked;
        }
      }
      for (size_t i = 0; i < 3; ++i)
        if (result[i] != 123.0f || result[count + 3 + i] != 123.0f) return 1;
      std::vector<float> after(input.size());
      check(cudaMemcpy(after.data(), dx, bytes, cudaMemcpyDeviceToHost));
      if (std::memcmp(input.data(), after.data(), bytes)) return 1;
    }
    check(cudaFree(dx));
    check(cudaFree(dy));
  }
  std::printf("PASS row scan: %zu exact values; seeds, signed zeros, row boundaries, guards and input preservation\n", checked);
}
