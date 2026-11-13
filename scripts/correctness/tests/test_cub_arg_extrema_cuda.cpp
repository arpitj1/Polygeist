// Exercise the actual CUB index-result ABI, including first-occurrence ties.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
extern "C" void polygeist_cub_segmented_argmax_f32(int,int,const float*,int*);
extern "C" void polygeist_cub_segmented_argmin_f32(int,int,const float*,int*);
static void check(cudaError_t status) {
  if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status));std::exit(2); }
}
int main() {
  int checked=0;
  for(int cols:{1,2,7,64,257,2049}) {
    int rows=11;
    std::vector<float> input(rows*cols);
    for(int r=0;r<rows;++r)for(int c=0;c<cols;++c)
      input[r*cols+c]=r==0?3.f:r==1?(c%2?-0.f:0.f):float((c*37+r*11)%23-11);
    float *x=nullptr;int *y=nullptr;
    check(cudaMalloc(&x,input.size()*sizeof(float)));
    check(cudaMalloc(&y,(rows+2)*sizeof(int)));
    check(cudaMemcpy(x,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(int minimum=0;minimum<2;++minimum)for(int repeat=0;repeat<3;++repeat) {
      std::vector<int> output(rows+2,-777);
      check(cudaMemcpy(y,output.data(),output.size()*sizeof(int),cudaMemcpyHostToDevice));
      if(minimum)polygeist_cub_segmented_argmin_f32(rows,cols,x,y+1);
      else polygeist_cub_segmented_argmax_f32(rows,cols,x,y+1);
      check(cudaMemcpy(output.data(),y,output.size()*sizeof(int),cudaMemcpyDeviceToHost));
      if(output.front()!=-777||output.back()!=-777)return 3;
      for(int r=0;r<rows;++r) {
        int best=0;
        for(int c=1;c<cols;++c)
          if(minimum?input[r*cols+c]<input[r*cols+best]:input[r*cols+c]>input[r*cols+best])best=c;
        if(output[r+1]!=best) {std::printf("FAIL cols=%d row=%d min=%d got=%d expected=%d\n",cols,r,minimum,output[r+1],best);return 4;}
        ++checked;
      }
    }
    std::vector<float> after(input.size());
    check(cudaMemcpy(after.data(),x,input.size()*sizeof(float),cudaMemcpyDeviceToHost));
    if(after!=input)return 5;
    check(cudaFree(x));check(cudaFree(y));
  }
  std::printf("PASS CUB arg extrema %d indices\n",checked);
}
