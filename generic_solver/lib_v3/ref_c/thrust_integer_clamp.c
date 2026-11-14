/* Public cuda::minimum<int>(upper, cuda::maximum<int>(x, lower)),
 * followed by the int8 output conversion of a typed Thrust transform.
 * Keep both comparisons, including their equal-value operand choices.
 */
void NETLIB_SPEC(int n, int input[n],
#ifndef OUTPUT_BITS
int lower, int upper,
#endif
signed char output[n]) {
#ifdef OUTPUT_BITS
  const int lower = -(1 << (OUTPUT_BITS - 1));
  const int upper = (1 << (OUTPUT_BITS - 1)) - 1;
#endif
  for(int i=0;i<n;++i){
    int value=input[i]<lower?lower:input[i];
    output[i]=(signed char)(upper<value?upper:value);
  }
}
