/* Public cuda::std comparison functors through Thrust transform. */
void NETLIB_SPEC(int n,int x[n],
#if RIGHT_SCALAR
int y,
#else
int y[n],
#endif
SCALAR_OUT output[n]) {
 for(int i=0;i<n;++i) {
#if RIGHT_SCALAR
  int right=y;
#else
  int right=y[i];
#endif
#if EQUAL == 1
  output[i]=x[i]==right;
#elif EQUAL == 2
  output[i]=x[i]<right;
#elif EQUAL == 3
  output[i]=x[i]<=right;
#else
  output[i]=x[i]>=right;
#endif
 }
}
