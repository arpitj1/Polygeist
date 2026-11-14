/* Source-derived FP32 Thrust binary transform specialization.
 * Inline sequential for_each over the zip iterator and the actual
 * cuda::std::{plus,minus,multiplies}<SCALAR> source operator. Pointer and constant iterators
 * become their exact C reads; CUDA execution is validated independently.
 * Public Apache-2.0/LLVM source files and hashes are pinned in the manifest. */
#ifndef SCALAR
#define SCALAR float
#endif
#ifndef SCALAR_OUT
#define SCALAR_OUT SCALAR
#endif
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n,
#if LEFT_SCALAR
SCALAR x,
#else
SCALAR x[],
#endif
#if RIGHT_SCALAR
SCALAR y,
#else
SCALAR y[],
#endif
SCALAR_OUT output[]) {
 for(int i=0;i<n;++i){
#if LEFT_SCALAR
  SCALAR left=x;
#else
  SCALAR left=x[i];
#endif
#if RIGHT_SCALAR
  SCALAR right=y;
#else
  SCALAR right=y[i];
#endif
#if THRUST_OPERATION == 0
  output[i] = left + right;
#elif THRUST_OPERATION == 1
  output[i] = left - right;
#elif THRUST_OPERATION == 2
  output[i] = left * right;
#elif THRUST_OPERATION == 3
  output[i] = left % right;
#else
#error "unsupported public functor"
#endif
 }
}
