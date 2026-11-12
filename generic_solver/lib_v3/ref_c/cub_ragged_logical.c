/* CUB DeviceSegmentedReduce with public ThreadReduce's seeded sequential
 * reference and CUDA logical_and<int>. Segment r begins at r*cols and ends
 * at r*cols+length[r]. The imported computation includes empty segments.
 * This is the same public reference specialization as cub_segmented_logical,
 * with independent segment endpoints instead of equal row lengths.
 * Because the seed and every result are 0/1, logical_and is exactly
 * value & (input != 0), with no floating-point assumptions. */
void NETLIB_SPEC(int rows, int cols, int input[rows][cols],
                 int length[rows], int output[rows]) {
  for (int r = 0; r < rows; ++r) {
    int value = 1;
    for (int c = 0; c < length[r]; ++c)
      value = value & (input[r][c] != 0);
    output[r] = value;
  }
}
