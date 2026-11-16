/* CUB segmented sum over a Thrust transform iterator using the public
 * cuda::std::not_equal_to<float> operator and zero. Contiguous equal-length
 * segments; int32 counts fit because each positive segment is <= INT_MAX. */
#ifdef NETLIB_SINGLE_SEGMENT
#define CUB_COUNT_ROWS 1
#define CUB_COUNT_INPUT(r, c) input[c]
void NETLIB_SPEC(int cols,float input[cols],int output[1]) {
#else
#define CUB_COUNT_ROWS rows
#define CUB_COUNT_INPUT(r, c) input[r][c]
void NETLIB_SPEC(int rows,int cols,float input[rows][cols],int output[rows]) {
#endif
  for(int r=0;r<CUB_COUNT_ROWS;++r){
    int count=0;
    for(int c=0;c<cols;++c) count += CUB_COUNT_INPUT(r,c) != 0.0f;
    output[r]=count;
  }
}
#undef CUB_COUNT_ROWS
#undef CUB_COUNT_INPUT
