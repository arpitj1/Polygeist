/* Contiguous FP32 specialization of the pinned ATen CUDA log1p scalar
 * functor, expanded over the pinned Thrust transform traversal. The runtime
 * invokes actual Thrust transform with the same CUDA math function.
 * errno and floating exception flags are not observed. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
extern __attribute__((const)) float log1pf(float);
void NETLIB_SPEC(int n, float input[n], float output[n]) {
  for (int i = 0; i < n; ++i)
    output[i] = log1pf(input[i]);
}
