// Exact counts, empty bins, repeated calls, guards and input preservation.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
extern "C" void polygeist_cub_histogram_even_i32(int32_t, const int32_t *, int32_t, int32_t *);
static void check(cudaError_t s) {
  if(s!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(2); }
}
int main() {
  size_t checked=0;
  for(int bins : {1,7,257,1025}) for(int count : {1,17,8193}) {
    std::vector<int32_t> input(count+6,123456), got(bins+6,123456), expected(bins,0);
    for(int i=0;i<count;++i) { int value=(i%3==0?bins-1:(i*7)%bins);input[i+3]=value;++expected[value]; }
    int32_t *x=nullptr,*y=nullptr;
    check(cudaMalloc(&x,input.size()*sizeof(int32_t)));
    check(cudaMalloc(&y,got.size()*sizeof(int32_t)));
    check(cudaMemcpy(x,input.data(),input.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    check(cudaMemcpy(y,got.data(),got.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    for(int run=0;run<2;++run) {
      polygeist_cub_histogram_even_i32(count,x+3,bins,y+3);
      check(cudaMemcpy(got.data(),y,got.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));
      for(int i=0;i<bins;++i) { if(got[i+3]!=expected[i])return 1;++checked; }
      for(int i=0;i<3;++i) if(got[i]!=123456||got[bins+3+i]!=123456)return 1;
    }
    std::vector<int32_t> after(input.size());
    check(cudaMemcpy(after.data(),x,input.size()*sizeof(int32_t),cudaMemcpyDeviceToHost));
    if(input!=after)return 1;
    check(cudaFree(x));check(cudaFree(y));
  }
  std::printf("PASS exact histogram values=%zu; guards and input preserved\n",checked);
}
