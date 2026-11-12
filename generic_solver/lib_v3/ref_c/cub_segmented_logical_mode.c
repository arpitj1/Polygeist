/* Public CUDA logical_and<int>/logical_or<int> selected by a scalar mode,
 * reduced by seeded CUB DeviceSegmentedReduce on contiguous row segments. */
void NETLIB_SPEC(int rows,int cols,int all,int input[rows][cols],int output[rows]) {
 for(int r=0;r<rows;++r){int value=all;
  for(int c=0;c<cols;++c)value=all ? (value && input[r][c]) : (value || input[r][c]);
  output[r]=value;
 }
}
