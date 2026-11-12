/* One component from the pinned ATen apply_cross<float> implementation.
 * Four logical read views are formal inputs. The unchanged component scalar
 * expression is traversed over independent lanes; no geometric recognizer. */
#ifndef NETLIB_SPEC
#error "NETLIB_SPEC must name the imported function"
#endif
void NETLIB_SPEC(int n,float a[],float b[],float c[],float d[],float output[]){
 for(int i=0;i<n;++i)output[i]=a[i]*b[i] - c[i]*d[i];
}
