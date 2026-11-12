/* Equal-length segments of CUB DeviceSegmentedReduce, each defined by the
 * seeded sequential CUB ThreadReduce and public CUDA logical operators.
 * Specialize begin/end offsets to r*cols/(r+1)*cols, retaining each segment's
 * seed and operator. Parameterize segment count and length. */
void NETLIB_SPEC(int rows, int cols, int seed, int input[rows][cols], int output[rows]) {
  for (int r = 0; r < rows; ++r) {
    int value = seed;
    for (int c = 0; c < cols; ++c) {
#if LOGICAL_AND
      value = value && input[r][c];
#else
      value = value || input[r][c];
#endif
    }
    output[r] = value;
  }
}
