/* Broadcast base constructor and unchanged pow lane method from the pinned
 * public ATen Vectorized<float> implementation. Both inputs to pow are FP32.
 */
extern __attribute__((const)) float powf(float, float);
void NETLIB_SPEC(int n, float base, float exponent[n], float output[n]) {
  for (int i = 0; i < n; ++i) output[i] = powf(base, exponent[i]);
}
