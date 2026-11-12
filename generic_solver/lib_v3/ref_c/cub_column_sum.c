/* CUB seeded ThreadReduce with CUDA plus<int>, reading each matrix column
 * through a counting/transform iterator. Int8 is converted exactly to int32.
 * The traversal is segmented by output column; seed and extents are runtime
 * parameters. Bounds in the manifest ensure no signed overflow. */
void NETLIB_SPEC(int rows,int cols,int seed,signed char input[rows][cols],int output[cols]) {
 for(int c=0;c<cols;++c){int value=seed;
  for(int r=0;r<rows;++r)value+=(int)input[r][c];
  output[c]=value;
 }
}
