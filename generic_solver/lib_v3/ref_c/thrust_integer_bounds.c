/* Public cuda::minimum/maximum<int32_t> with a constant iterator. */
void NETLIB_SPEC(int n,int input[n],int bound,int output[n]) {
 for(int i=0;i<n;++i) {
#if UPPER
  output[i]=bound<input[i]?bound:input[i];
#else
  output[i]=input[i]<bound?bound:input[i];
#endif
 }
}
