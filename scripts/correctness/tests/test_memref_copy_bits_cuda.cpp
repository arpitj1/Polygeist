#include <cuda_runtime_api.h>
#include <cstdint>
#include <vector>
#define POLYGEIST_V2_MATRIX_CUDA
#include "polygeist_v2_matrix_layout.inc"
#include "polygeist_v2_memref_copy.inc"

extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
#define CU(x) do { cudaError_t error=(x); if(error!=cudaSuccess) { \
  fprintf(stderr,"%s\n",cudaGetErrorString(error)); return 90; } } while(0)

int main() {
  constexpr size_t bytes=65536;
  std::vector<unsigned char> source(bytes), output(bytes), expected(bytes), saved(bytes);
  uint32_t patterns[]={0,0x80000000,0x7f800000,0xff800000,0x7f800001,
    0x7fc00001,0xffc12345,1,0x80000001,0x7f7fffff,0x00800000,0x12345678};
  for(size_t i=0;i<bytes/4;i++) memcpy(source.data()+i*4,&patterns[i%12],4);
  unsigned char *a,*b;
  CU(cudaMalloc((void**)&a,bytes)); CU(cudaMalloc((void**)&b,bytes));
  CU(cudaMemcpy(a,source.data(),bytes,cudaMemcpyHostToDevice));
  struct Layout {int64_t rank, sizes[6], src[6], dst[6], offset;};
  Layout layouts[]={
    {6,{2,3,3,3,6,6},{192,64,8,1,8,1},{972,324,108,36,6,1},0},
    {2,{3,7},{19,2},{1,5},0}, // transposed, padded output
    {3,{4,1,7},{12,0,1},{21,0,2},0}, // ignored unit dimension
    {2,{5,7},{-8,1},{9,1},32}, // reversed source
    {2,{5,7},{0,1},{9,1},0}, // repeated source reads
    {2,{5,7},{8,1},{0,1},0}, // overlapping output uses serial fallback
    {2,{5,7},{8,1},{-9,1},0} // negative output uses serial fallback
  };
  size_t checked=0;
  for(int element: {1,2,4,8,16}) for(const auto &l:layouts) {
    std::fill(output.begin(),output.end(),0xa5); expected=output;
    CU(cudaMemcpy(b,output.data(),bytes,cudaMemcpyHostToDevice));
    int64_t count=1;for(int d=0;d<l.rank;d++)count*=l.sizes[d];
    const int64_t srcbase=64+l.offset, dstbase=64;
    for(int64_t i=0;i<count;i++) {
      int64_t index=i,src=srcbase,dst=dstbase;
      for(int d=l.rank;d-->0;) {src+=(index%l.sizes[d])*l.src[d];
        dst+=(index%l.sizes[d])*l.dst[d]; index/=l.sizes[d];}
      memcpy(expected.data()+dst*element,source.data()+src*element,element);
    }
    polygeist_v2_memref_copy(l.rank,element,l.sizes,l.src,l.dst,
                           a+srcbase*element,b+dstbase*element);
    CU(cudaMemcpy(output.data(),b,bytes,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(saved.data(),a,bytes,cudaMemcpyDeviceToHost));
    if(output!=expected || saved!=source) return 1;
    checked+=bytes*2;
  }
  // Overlap must snapshot before writing, including signaling NaN bit patterns.
  int64_t count=64,one=1;
  expected=source;memmove(expected.data()+12,expected.data(),count*4);
  polygeist_v2_memref_copy(1,4,&count,&one,&one,a,a+12);
  CU(cudaMemcpy(output.data(),a,bytes,cudaMemcpyDeviceToHost));
  if(output!=expected)return 2;
  CU(cudaFree(a));CU(cudaFree(b));
  printf("PASS exact descriptor copies: %zu bytes, IEEE bits, guards and source preservation\n",checked+bytes);
}
