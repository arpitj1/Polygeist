/* Public thrust::copy's identity transform followed by destination assignment.
 * Preserve the C++ source-to-destination scalar conversion, including FP32
 * rounding for the int32-to-float specialization.
 */
void NETLIB_SPEC(int n, SCALAR_IN input[n], SCALAR_OUT output[n]) {
  for (int i = 0; i < n; ++i) output[i] = input[i];
}
