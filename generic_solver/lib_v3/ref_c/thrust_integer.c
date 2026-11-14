/* Specializations of public Thrust transform with integer scalar functors.
 * Shifts are evaluated only on the C language's defined input domain.
 */
void NETLIB_SPEC(int n, int left[n],
#if !UNARY
                int right[n],
#endif
                int output[n]) {
  for (int i = 0; i < n; ++i)
#if UNARY
    output[i] = OP left[i];
#else
    output[i] = left[i] OP right[i];
#endif
}
