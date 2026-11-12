/* Public CUB ordered ThreadReduce with a per-row prefix length.
 * The leading stride remains the declared number of columns. */
#if LOGICAL
#define ELEMENT int
#else
#define ELEMENT float
#endif
void NETLIB_SPEC(int rows,int cols,ELEMENT input[rows][cols],int length[rows],ELEMENT output[rows]) {
 for(int r=0;r<rows;++r){
#if LOGICAL
  int value=1;
#else
  float value=0.0f;
#endif
  for(int c=0;c<length[r];++c){
#if LOGICAL
   value=value && input[r][c];
#else
   value=value+input[r][c];
#endif
  }
  output[r]=value;
 }
}
