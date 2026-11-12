/* FP32 specialization of ATen CUDA RenormKernel.cu's scalar lambda.
 * Expand TensorIterator as a contiguous loop; preserve the comparison,
 * branch, constants and scalar order. No floating algebra is changed. */
void NETLIB_SPEC(int n, float norm[], float maxnorm, float output[]) {
  for (int i = 0; i < n; ++i) {
    const float eps = (float)1e-7;
    const float one = (float)1.0;
    output[i] = norm[i] > maxnorm ? maxnorm / (norm[i] + eps) : one;
  }
}
