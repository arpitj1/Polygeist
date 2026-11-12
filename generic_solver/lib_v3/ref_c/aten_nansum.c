/* FP32 NanSumOps::reduce from public ATen/native/SharedReduceOps.h,
 * specialized as an ordered row fold. CUB ThreadReduce can retain this order. */
void NETLIB_SPEC(int rows,int cols,float input[rows][cols],float output[rows]) {
 for(int r=0;r<rows;++r){float value=0.0f;
  for(int c=0;c<cols;++c){float x=input[r][c];value=value+(x!=x?0.0f:x);}
  output[r]=value;
 }
}
