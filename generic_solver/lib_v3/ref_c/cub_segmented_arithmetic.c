/* CUB seeded ThreadReduceSequential, applied to equal-length segments.
 * Specialize the public CUDA plus/bit_xor operator; retain the seed and order.
 * DeviceSegmentedReduce uses parallel association for floating-point sums. */
void NETLIB_SPEC(int rows, int cols, SCALAR seed, SCALAR input[rows][cols], SCALAR output[rows]) {
  for (int r=0; r<rows; ++r) {
    SCALAR value=seed;
    for (int c=0; c<cols; ++c) {
#if REDUCE_XOR
      value = value ^ input[r][c];
#else
      value = value + input[r][c];
#endif
    }
    output[r]=value;
  }
}
