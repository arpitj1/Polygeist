/* Public cuda::std::plus<uint32_t> with int32 input/output storage.
 * Conversion to unsigned preserves defined modulo-2^32 arithmetic. */
void NETLIB_SPEC(int n, int x[n], int y, int output[n]) {
  for(int i=0;i<n;++i) output[i]=(int)((unsigned int)x[i]+(unsigned int)y);
}
