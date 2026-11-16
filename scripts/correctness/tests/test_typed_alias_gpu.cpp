// Link against the generated v2 wrappers, vendor runtime, and CUDA.
#include "polygeist_v2_rt.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern "C" void polygeist_cublas_destroy();

static void check(cudaError_t error) { assert(error == cudaSuccess); }

static void partialOverlapMustFail() {
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    const rlimit limit{0, 0};
    setrlimit(RLIMIT_CORE, &limit);
    float values[18] = {};
    polygeist_v2_transform_thrust_multiplies_scalar_left_f32(
        16, 2.0f, values, 16 * sizeof(float), 1, values + 1,
        16 * sizeof(float), 1);
    _exit(1);
  }
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static void run(int n, int alias, bool device) {
  std::vector<float> x(n + 2, 12345.0f), y(n + 2, 12345.0f);
  std::vector<float> out(n + 2, 12345.0f), expected(n);
  for (int i = 0; i < n; ++i) {
    x[i + 1] = (i % 17 - 8) * 0.25f;
    y[i + 1] = (i % 13 - 6) * 0.5f;
    expected[i] = x[i + 1] + y[i + 1];
  }
  const auto originalX = x, originalY = y;
  float *dx = x.data(), *dy = y.data(), *dz = out.data();
  const size_t bytes = x.size() * sizeof(float);
  if (device) {
    check(cudaMalloc(reinterpret_cast<void **>(&dx), bytes));
    check(cudaMalloc(reinterpret_cast<void **>(&dy), bytes));
    check(cudaMalloc(reinterpret_cast<void **>(&dz), bytes));
    check(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dz, out.data(), bytes, cudaMemcpyHostToDevice));
  }
  float *destination = (alias == 1 ? dx : alias == 2 ? dy : dz) + 1;
  polygeist_v2_transform_thrust_plus_f32(
      n, dx + 1, n * sizeof(float), 1, dy + 1, n * sizeof(float), 1,
      destination, n * sizeof(float), 1);
  polygeist_v2_transform_thrust_multiplies_scalar_left_f32(
      n, -0.5f, destination, n * sizeof(float), 1, destination,
      n * sizeof(float), 1);
  if (device) {
    check(cudaMemcpy(x.data(), dx, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(out.data(), dz, bytes, cudaMemcpyDeviceToHost));
    check(cudaFree(dx));
    check(cudaFree(dy));
    check(cudaFree(dz));
  }
  const auto &actual = alias == 1 ? x : alias == 2 ? y : out;
  for (int i = 0; i < n; ++i) {
    expected[i] *= -0.5f;
    assert(memcmp(&actual[i + 1], &expected[i], sizeof(float)) == 0);
  }
  assert(x.front() == 12345.0f && x.back() == 12345.0f);
  assert(y.front() == 12345.0f && y.back() == 12345.0f);
  assert(out.front() == 12345.0f && out.back() == 12345.0f);
  if (alias != 1)
    assert(x == originalX);
  if (alias != 2)
    assert(y == originalY);
  polygeist_cublas_destroy();
}

int main() {
  partialOverlapMustFail();
  for (int n : {1, 257, 4096})
    for (int alias = 0; alias != 3; ++alias)
      for (bool device : {false, true})
        run(n, alias, device);
  puts("typed wrapper alias checks PASS: host/device, left/right/disjoint, guards, partial overlap");
}
