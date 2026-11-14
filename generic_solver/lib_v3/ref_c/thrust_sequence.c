/* Public Thrust sequence<float>, specialized to contiguous storage.
 * compute_sequence_value<float> returns init + step * static_cast<float>(i).
 * For 0 <= i < n <= INT_MAX, int and size_t induction values agree exactly.
 * Preserve separate multiply/add; CUDA wrappers are built with --fmad=false.
 */
void NETLIB_SPEC(int n, float init, float step, float output[n]) {
  for (int i = 0; i < n; ++i) output[i] = init + step * (float)i;
}
