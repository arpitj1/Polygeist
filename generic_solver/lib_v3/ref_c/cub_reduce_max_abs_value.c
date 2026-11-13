/* Public CUB ThreadReduce over a transform iterator using cuda::std::fabs,
 * with cuda::maximum<float> and a caller-provided seed. Inline only the
 * iterator dereference and scalar functors; retain their evaluation order.
 */
extern __attribute__((const)) float fabsf(float);
extern __attribute__((const)) float fmaxf(float, float);
float NETLIB_SPEC(int n, float seed, float input[]) {
  float value = seed;
  for (int i = 0; i < n; ++i)
    value = fmaxf(value, fabsf(input[i]));
  return value;
}
