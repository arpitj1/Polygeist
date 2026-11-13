#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
static size_t transfers = 0, transferred_bytes = 0;
static cudaError_t tracked_memcpy(void *out, const void *in, size_t bytes,
                                  cudaMemcpyKind kind) {
  ++transfers;
  transferred_bytes += bytes;
  return cudaMemcpy(out, in, bytes, kind);
}
#define POLYGEIST_V2_MATRIX_CUDA
#include "polygeist_v2_matrix_layout.inc"
#define cudaMemcpy tracked_memcpy
#include "polygeist_v2_memref_copy.inc"
#undef cudaMemcpy

extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
#define CU(x) do { cudaError_t error=(x); if(error!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(error)); return 90; } } while(0)

int main() {
  constexpr size_t bytes=262144;
  std::vector<unsigned char> source(bytes), output(bytes), expected(bytes), saved(bytes);
  uint32_t patterns[]={0,0x80000000,0x7f800000,0xff800000,0x7f800001,
    0x7fc00001,0xffc12345,1,0x80000001,0x7f7fffff,0x00800000,0x12345678};
  for(size_t i=0;i<bytes/4;i++) memcpy(source.data()+i*4,&patterns[i%12],4);
  unsigned char *a,*b;
  CU(cudaMalloc((void**)&a,bytes)); CU(cudaMalloc((void**)&b,bytes));
  CU(cudaMemcpy(a,source.data(),bytes,cudaMemcpyHostToDevice));
  struct Layout {int64_t rank, sizes[16], src[16], dst[16], offset;};
  Layout layouts[]={
    {6,{2,3,3,3,6,6},{192,64,8,1,8,1},{972,324,108,36,6,1},0},
    {2,{3,7},{19,2},{1,5},0}, // transposed, padded output
    {3,{4,1,7},{12,0,1},{21,0,2},0}, // ignored unit dimension
    {2,{5,7},{-8,1},{9,1},32}, // reversed source
    {2,{5,7},{0,1},{9,1},0}, // repeated source reads
    {2,{5,7},{8,1},{0,1},0}, // overlapping output uses serial fallback
    {2,{5,7},{8,1},{-9,1},0}, // negative output uses serial fallback
    {3,{4,5,7},{0,7,1},{35,1,5},0}, // dense host broadcast to permuted output
    {2,{5,7},{-7,1},{1,5},28}, // dense reversed source
    {2,{5,7},{19,2},{7,1},0}, // device holes to dense host output
    {3,{17,31,7},{217,7,1},{217,1,31},0}, // non-power-of-two divisors
    {4,{3,7,5,11},{385,55,11,1},{385,55,11,1},0}, // collapsible adjacent dims
    {2,{65,97},{97,1},{1,65},0}, // tiled transpose, both partial edge tiles
    {3,{2,65,37},{0,1,65},{2405,37,1},0}, // tiled transpose with batch broadcast
    {3,{2,65,37},{2405,1,-65},{2405,37,1},2340}, // reversed tiled source axis
    {5,{2,2,9,11,11},{2178,1089,121,11,1},{2178,1089,1,99,9},0}, // tiled copy with three batch axes
    {5,{2,2,17,31,2},{2108,1054,62,1,31},{2108,1054,1,34,17},0}, // occupied tiles with three batch axes
    {16,{1,1,1,1,1,1,1,1,1,1,1,1,1,1,7,11},
       {0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,1},
       {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,7},0}, // rank limit/unit dims
    {2,{1,1},{0,0},{0,0},0} // all unit dimensions collapse to one payload
  };
  size_t checked=0;
  for(int element: {1,2,3,4,5,8,12,16,24}) for(const auto &l:layouts)
   for(int direction=0;direction<3;++direction) for(int misalignment: {0,1,3}) {
    std::fill(output.begin(),output.end(),0xa5); expected=output;
    CU(cudaMemcpy(b,output.data(),bytes,cudaMemcpyHostToDevice));
    int64_t count=1;for(int d=0;d<l.rank;d++)count*=l.sizes[d];
    const int64_t srcbase=64+l.offset, dstbase=64;
    for(int64_t i=0;i<count;i++) {
      int64_t index=i,src=srcbase,dst=dstbase;
      for(int d=l.rank;d-->0;) {src+=(index%l.sizes[d])*l.src[d];
        dst+=(index%l.sizes[d])*l.dst[d]; index/=l.sizes[d];}
      memcpy(expected.data()+misalignment+dst*element,source.data()+misalignment+src*element,element);
    }
    const unsigned char *src = direction==1 ? source.data() : a;
    unsigned char *dst = direction==2 ? output.data() : b;
    polygeist_v2_memref_copy(l.rank,element,l.sizes,l.src,l.dst,
                           src+misalignment+srcbase*element,dst+misalignment+dstbase*element);
    if(direction!=2) CU(cudaMemcpy(output.data(),b,bytes,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(saved.data(),a,bytes,cudaMemcpyDeviceToHost));
    if(output!=expected || saved!=source) return 1;
    checked+=bytes*2;
  }
  // Broadcast only transfers the seven physical source elements, not a CUDA
  // transaction for every logical output. The return copy is one transaction.
  int64_t shape[2]={512,7}, broadcast[2]={0,1}, dense[2]={7,1};
  transfers=transferred_bytes=0;
  polygeist_v2_memref_copy(2,4,shape,broadcast,dense,source.data(),b);
  if(transfers!=1 || transferred_bytes!=7*4)return 3;
  int64_t transposed[2]={1,512};
  transfers=transferred_bytes=0;
  polygeist_v2_memref_copy(2,4,shape,dense,transposed,b,output.data());
  if(transfers!=1 || transferred_bytes!=512*7*4)return 4;
  for(int i=0;i<512;++i)for(int j=0;j<7;++j)
    if(memcmp(output.data()+(i+j*512)*4,source.data()+j*4,4))return 5;
  // Overlap must snapshot before writing, including signaling NaN bit patterns.
  int64_t count=64,one=1;
  expected=source;memmove(expected.data()+12,expected.data(),count*4);
  polygeist_v2_memref_copy(1,4,&count,&one,&one,a,a+12);
  CU(cudaMemcpy(output.data(),a,bytes,cudaMemcpyDeviceToHost));
  if(output!=expected)return 2;
  CU(cudaFree(a));CU(cudaFree(b));
  printf("PASS exact descriptor copies: %zu bytes, IEEE bits, guards and source preservation\n",checked+bytes);
}
