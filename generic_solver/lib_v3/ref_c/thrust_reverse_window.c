/* Public reverse_copy (or transform with public negate<float>) between
 * disjoint windows of one array. reverse_iterator dereferences end-1-i.
 * Nonnegative prefix/gap and positive n prove that source and destination
 * windows do not overlap. Index arithmetic is 64-bit on supported hosts.
 */
void NETLIB_SPEC(int n,int prefix,int gap,float buffer[(long)prefix+2L*n+gap]) {
  for(int i=0;i<n;++i) {
    float value=buffer[(long)prefix+n-1-i];
#if NEGATE
    value=-value;
#endif
    buffer[(long)prefix+n+gap+i]=value;
  }
}
