/* Pinned public CUB ThreadScanInclusive, instantiated with float and
 * cuda::std::multiplies<float>, independently over each contiguous row.
 * Runtime extents replace LENGTH; the unused aggregate return is discarded.
 * DeviceScan::InclusiveScanByKey uses the row number as its key and multiplies
 * each row's first input by the seed. Parallel association is checked by the
 * backend correctness tests, separately from reference-to-application proof. */
void NETLIB_SPEC(int rows, int cols, float seed,
                 float input[rows][cols], float output[rows][cols]) {
  for (int r = 0; r < rows; ++r) {
    float value = seed;
    for (int c = 0; c < cols; ++c) {
      value *= input[r][c];
      output[r][c] = value;
    }
  }
}
