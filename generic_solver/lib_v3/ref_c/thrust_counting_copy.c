/* Public thrust::copy from counting_iterator<int>(0) to contiguous output.
 * counting_iterator::dereference returns its current integer value; copy
 * assigns that value to the destination scalar type without arithmetic.
 * Bound n by INT_MAX, so the iterator never overflows its signed int value.
 */
void NETLIB_SPEC(int n, SCALAR_OUT output[n]) {
  for (int i = 0; i < n; ++i) output[i] = i;
}
