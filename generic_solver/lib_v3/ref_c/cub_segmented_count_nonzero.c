/* CUB segmented sum over a Thrust transform iterator using the public
 * cuda::std::not_equal_to<float> operator and zero. Contiguous equal-length
 * segments; int32 counts fit because each positive segment is <= INT_MAX. */
void NETLIB_SPEC(int rows,int cols,float input[rows][cols],int output[rows]) {
  for(int r=0;r<rows;++r){
    int count=0;
    for(int c=0;c<cols;++c) count += input[r][c] != 0.0f;
    output[r]=count;
  }
}
