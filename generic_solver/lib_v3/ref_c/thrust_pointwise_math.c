/* Contiguous FP32 specialization of the public ATen CUDA math functors
 * identified in the manifest, applied by Thrust transform. Keep each libm
 * primitive intact: expm1(x), for example, is not exp(x)-1.
 * errno and floating exception flags are not part of this array contract. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if MATH_ARITY == 1
extern __attribute__((const)) float MATH_FUNCTION(float);
void NETLIB_SPEC(int n, float input[n], float output[n]) {
  for (int i = 0; i < n; ++i)
    output[i] = MATH_FUNCTION(input[i]);
}
#elif MATH_ARITY == 2
extern __attribute__((const)) float MATH_FUNCTION(float, float);
void NETLIB_SPEC(int n, float left[n], float right[n], float output[n]) {
  for (int i = 0; i < n; ++i)
    output[i] = MATH_FUNCTION(left[i], right[i]);
}
#else
#error "MATH_ARITY must be one or two"
#endif
