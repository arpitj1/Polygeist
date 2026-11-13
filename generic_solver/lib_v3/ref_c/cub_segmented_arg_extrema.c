/* CUB seeded ThreadReduce with ArgMax/ArgMin, projected to its key.
 * Specialize segments to nonempty contiguous FP32 rows and visit their keys
 * in ascending order. The current best key is always less than the next key,
 * so the public functor's equal-value/smaller-key arm is unreachable in this
 * sequential reference. The parallel backend retains the full tie rule.
 */
void NETLIB_SPEC(int rows, int cols, float input[rows][cols], int output[rows]) {
  for (int r = 0; r < rows; ++r) {
    float value = input[r][0];
    int key = 0;
    for (int c = 1; c < cols; ++c) {
#if ARG_MIN
      if (input[r][c] < value) {
#else
      if (input[r][c] > value) {
#endif
        value = input[r][c];
        key = c;
      }
    }
    output[r] = key;
  }
}
