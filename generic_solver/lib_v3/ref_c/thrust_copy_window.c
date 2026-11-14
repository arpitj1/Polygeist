/* Public Thrust copy or unary transform into an offset output iterator.
 * The complete output remains an operand so prefixes and tails are preserved.
 * The transform variant instantiates the public cuda::std::negate<float>.
 */
void NETLIB_SPEC(int n,int offset,float input[n],float output[(long)n+offset]) {
 for(int i=0;i<n;++i) {
#if NEGATE
  output[(long)offset+i]=-input[i];
#else
  output[(long)offset+i]=input[i];
#endif
 }
}
