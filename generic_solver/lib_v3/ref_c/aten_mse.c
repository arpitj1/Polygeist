/* FP32 TensorIterator traversal of the pinned ATen CPU mse_kernel and
 * mse_backward_cpu_kernel. The backward specialization has unit grad_output;
 * the normalization remains an input, as in the public scalar kernel. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
#if BACKWARD
void NETLIB_SPEC(int n, float input[n], float target[n], float value,
                 float output[n]) {
  for (int i = 0; i < n; ++i)
    output[i] = value * (input[i] - target[i]) * 1.0f;
}
#else
void NETLIB_SPEC(int n, float input[n], float target[n], float output[n]) {
  for (int i = 0; i < n; ++i) {
    float difference = input[i] - target[i];
    output[i] = difference * difference;
  }
}
#endif
