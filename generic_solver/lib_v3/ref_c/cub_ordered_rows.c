/* Pinned public CUB ThreadReduceSequential with public CUDA plus and
 * optional multiplies/divides input transformation. The padded ABI adapter
 * skips absent items and threads the seed through ordered chunks. */
void NETLIB_SPEC(int rows,int cols,float seed,
#if TRANSFORM_KIND
float factor,
#endif
float input[rows][cols],float output[rows]) {
  for(int r=0;r<rows;++r){
    float value=seed;
    for(int c=0;c<cols;++c){
      float x=input[r][c];
#if TRANSFORM_KIND == 1
      x=factor*x;
#elif TRANSFORM_KIND == 2
      x=x/factor;
#endif
      value=value+x;
    }
    output[r]=value;
  }
}
