/* Public counting_iterator<uint64_t>(0) transformed with divides/modulus
 * and a constant divisor iterator. The output assignment converts to int32.
 * n is positive and bounded by INT_MAX; stride is a positive uint32 value. */
void NETLIB_SPEC(int n,unsigned int stride,int output[n]) {
 for(long i=0;i<n;++i) {
#if REMAINDER
  output[i]=(int)((unsigned long)i % (unsigned long)stride);
#else
  output[i]=(int)((unsigned long)i / (unsigned long)stride);
#endif
 }
}
