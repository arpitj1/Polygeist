/* Binary Thrust transform with the CUDA/libm ldexpf scalar primitive. */
extern __attribute__((const)) float ldexpf(float,int);
void NETLIB_SPEC(int n, float input[n], int exponent[n], float output[n]) {
  for (int i=0;i<n;++i) output[i]=ldexpf(input[i],exponent[i]);
}
