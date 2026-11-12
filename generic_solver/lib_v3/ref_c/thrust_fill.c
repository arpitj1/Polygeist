/* Pinned public Thrust fill_n -> generate_n(fill_functor<T>{value}).
 * Inline the assignment and sequential for_each_n traversal, specialize T to
 * float and output to a pointer, and express pointer increments as subscripts.
 * No arithmetic is introduced; the value's FP32 representation is retained. */
void NETLIB_SPEC(int n, float value, float output[]) {
  for (int i = 0; i < n; ++i) output[i] = value;
}
