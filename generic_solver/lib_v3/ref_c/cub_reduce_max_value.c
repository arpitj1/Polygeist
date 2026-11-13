/* CUB ThreadReduceSequential with cuda::maximum<float>.
 * The public floating-point functor calls fmax. Retain that operation,
 * the caller's seed, and the scalar return; only the extent is dynamic.
 */
extern __attribute__((const)) float fmaxf(float, float);
float NETLIB_SPEC(int n, float seed, float input[]) {
  float value = seed;
  for (int i = 0; i < n; ++i)
    value = fmaxf(value, input[i]);
  return value;
}
