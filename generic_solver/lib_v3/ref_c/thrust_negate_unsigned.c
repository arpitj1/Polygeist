/* Public cuda::std::negate<uint32_t>, then int32 output assignment. */
void NETLIB_SPEC(int n,unsigned int input[n],int output[n]) {
 for(int i=0;i<n;++i) output[i]=(int)(0u-input[i]);
}
