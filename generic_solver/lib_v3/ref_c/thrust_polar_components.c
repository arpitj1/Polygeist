/* FP32 ATen CUDA polar_kernel_cuda scalar lambda; the two components
 * are returned through a Thrust zip output iterator in the fixed C ABI. */
extern __attribute__((const)) float sinf(float);
extern __attribute__((const)) float cosf(float);
void NETLIB_SPEC(int n,float magnitude[n],float angle[n],float real[n],float imag[n]) {
 for(int i=0;i<n;++i){real[i]=magnitude[i]*cosf(angle[i]);imag[i]=magnitude[i]*sinf(angle[i]);}
}
