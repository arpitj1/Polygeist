/* Compose the public ATen FP32 comparison and where scalar functions.
 * TensorIterator broadcasting is specialized to contiguous arrays or scalar
 * parameters. Comparison direction and branch order remain explicit.
 * The cuDNN descriptor is emitted from the imported reference, not an input
 * application. See the manifest for source hashes and specializations. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if BOUND_ARRAY
#define BOUND_PARAMETER float bound[n]
#define BOUND_VALUE bound[i]
#else
#define BOUND_PARAMETER float bound
#define BOUND_VALUE bound
#endif
void NETLIB_SPEC(int n, float input[n], float output[n], BOUND_PARAMETER
#if HAS_VALUE
                , float value[n]
#endif
#if HAS_OTHER
                , float other[n]
#endif
#if HAS_CONSTANT
                , float constant
#endif
                ) {
  for (int i = 0; i < n; ++i)
    output[i] = input[i] COMPARE BOUND_VALUE ? SELECT_TRUE : SELECT_FALSE;
}
