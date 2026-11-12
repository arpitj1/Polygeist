#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "polygeist_v2_matrix_layout.inc"
#include "polygeist_v2_memref_copy.inc"

static int check_dense_footprints(void) {
  /* Independently enumerate physical addresses. Any footprint accepted for a
   * whole-buffer CUDA transfer must have every byte represented by the view. */
  for (int64_t a=1;a<=4;++a) for (int64_t b=1;b<=4;++b)
    for (int64_t c=1;c<=4;++c)
      for (int64_t x=-12;x<=12;++x) for (int64_t y=-12;y<=12;++y)
        for (int64_t z=-12;z<=12;++z) {
          int64_t sizes[]={a,b,c}, strides[]={x,y,z};
          if (!pv2_copy_dense_footprint(3,sizes,strides)) continue;
          unsigned char seen[256]={0}; int low=128, high=128;
          for (int i=0;i<a;++i) for (int j=0;j<b;++j) for (int k=0;k<c;++k) {
            int address=128+i*x+j*y+k*z;
            seen[address]=1;
            if (address<low)low=address;
            if (address>high)high=address;
          }
          for (int i=low;i<=high;++i) if (!seen[i]) return 1;
        }
  return 0;
}

int main(int argc, char **argv) {
  int64_t sizes[3] = {3,4,2}, source_strides[3] = {9,2,1};
  int64_t dest_strides[3] = {8,1,1};
  uint32_t source[96], dest[96], expected[96];
  for (int i=0;i<96;++i) source[i]=0x80000000u + i*19;
  for (int i=0;i<96;++i) dest[i]=expected[i]=0xdeadbeefu;
  if (argc>1) {
    if (!strcmp(argv[1],"overflow")) source_strides[0]=INT64_MAX;
    else if (!strcmp(argv[1],"negative")) sizes[0]=-1;
    else if (!strcmp(argv[1],"null"))
      polygeist_v2_memref_copy(2,4,sizes,source_strides,dest_strides,NULL,dest);
    else return 9;
    polygeist_v2_memref_copy(2,4,sizes,source_strides,dest_strides,source,dest);
    return 10;
  }
  if (check_dense_footprints()) return 11;
  for(int r=0;r<3;++r) for(int c=0;c<4;++c)
    expected[3+r*8+c]=source[2+r*9+c*2];
  polygeist_v2_memref_copy(2,4,sizes,source_strides,dest_strides,source+2,dest+3);
  if(memcmp(dest,expected,sizeof(dest))) return 1;
  int64_t a[3]={2,3,4},sa[3]={12,4,1},sb[3]={1,2,6};
  memcpy(expected,dest,sizeof(dest));
  for(int i=0;i<2;++i)for(int j=0;j<3;++j)for(int k=0;k<4;++k)
    expected[5+i+2*j+6*k]=source[4+i*12+j*4+k];
  polygeist_v2_memref_copy(3,4,a,sa,sb,source+4,dest+5);
  if(memcmp(dest,expected,sizeof(dest))) return 2;
  uint32_t overlap[]={1,2,3,4,5,6,7,8};
  int64_t four=4,one=1;
  polygeist_v2_memref_copy(1,4,&four,&one,&one,overlap,overlap+2);
  uint32_t want[]={1,2,1,2,3,4,7,8};
  if(memcmp(overlap,want,sizeof(want))) return 3;
  int64_t two[2]={2,2},normal[2]={2,1},transpose[2]={1,2};
  polygeist_v2_memref_copy(2,4,two,normal,transpose,overlap,overlap);
  uint32_t transposed[]={1,1,2,2,3,4,7,8};
  if(memcmp(overlap,transposed,sizeof(overlap))) return 4;
  int64_t reverse=-1;
  polygeist_v2_memref_copy(1,4,&four,&reverse,&one,source+7,dest+9);
  for(int i=0;i<4;i++)if(dest[9+i]!=source[7-i])return 6;
  uint64_t bits=UINT64_C(0x7ff8000000000042),out=0;
  polygeist_v2_memref_copy(0,8,NULL,NULL,NULL,&bits,&out);
  if(out!=bits)return 5;
  int64_t zero=0;
  polygeist_v2_memref_copy(1,4,&zero,&one,&one,NULL,NULL);
  polygeist_v2_memref_copy(1,4,&four,&one,&one,overlap,overlap);
  puts("PASS rank0/1/2/3 copies; offsets, holes, overlap snapshots, empty views and input bits");
  return 0;
}
