// Validate the real segmented CUB backend, including ragged row offsets.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void *polygeist_cuda_graph_stream() { return nullptr; }
extern "C" void polygeist_cub_ragged_reduce_and_i32(int,int,const int*,const int*,int*);
extern "C" void polygeist_cub_segmented_reduce_and_i32(int,int,int,const int*,int*);
extern "C" void polygeist_cub_segmented_reduce_or_i32(int,int,int,const int*,int*);

static void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  constexpr int guard=123456789;
  size_t checked=0;
  const int shape[][2]={{1,0},{1,1},{3,7},{7,513},{33,2049}};
  for (const auto &s:shape) {
    int rows=s[0],cols=s[1];size_t count=size_t(rows)*cols;
    std::vector<int> input(count+6,guard),lengths(rows+6,guard),output(rows+6,guard);
    const int values[]={INT32_MIN,-7,-1,1,2,INT32_MAX,0};
    for(int row=0;row<rows;++row)for(int col=0;col<cols;++col)
      input[3+size_t(row)*cols+col]=values[(row+col)%(row%2?7:6)];
    int *x=nullptr,*len=nullptr,*y=nullptr;
    check(cudaMalloc(&x,input.size()*sizeof(int)));
    check(cudaMalloc(&len,lengths.size()*sizeof(int)));
    check(cudaMalloc(&y,output.size()*sizeof(int)));
    check(cudaMemcpy(x,input.data(),input.size()*sizeof(int),cudaMemcpyHostToDevice));
    for(int pattern=0;pattern<3;++pattern) {
      for(int row=0;row<rows;++row) {
        int mixed[]={0,cols>0?1:0,cols,cols/2,cols>0?cols-1:0};
        lengths[row+3]=pattern==0?0:pattern==1?cols:mixed[row%5];
      }
      check(cudaMemcpy(len,lengths.data(),lengths.size()*sizeof(int),cudaMemcpyHostToDevice));
      for(int operation=0;operation<3;++operation)for(int seed=0;seed<=1;++seed) {
        if(operation==0&&seed==0)continue;
        for(int repeat=0;repeat<2;++repeat) {
          std::fill(output.begin(),output.end(),guard);
          check(cudaMemcpy(y,output.data(),output.size()*sizeof(int),cudaMemcpyHostToDevice));
          if(operation==0)polygeist_cub_ragged_reduce_and_i32(rows,cols,x+3,len+3,y+3);
          else if(operation==1)polygeist_cub_segmented_reduce_and_i32(rows,cols,seed,x+3,y+3);
          else polygeist_cub_segmented_reduce_or_i32(rows,cols,seed,x+3,y+3);
          check(cudaMemcpy(output.data(),y,output.size()*sizeof(int),cudaMemcpyDeviceToHost));
          for(int row=0;row<rows;++row) {
            int expected=seed,end=operation==0?lengths[row+3]:cols;
            for(int col=0;col<end;++col) {
              int value=input[3+size_t(row)*cols+col];
              expected=operation==2?(expected||value):(expected&&value);
            }
            if(output[row+3]!=expected) {
              std::fprintf(stderr,"logical mismatch rows=%d cols=%d pattern=%d operation=%d seed=%d row=%d got=%d expected=%d\n",rows,cols,pattern,operation,seed,row,output[row+3],expected);
              return 1;
            }
            ++checked;
          }
          for(int i=0;i<3;++i)if(output[i]!=guard||output[rows+3+i]!=guard)return 1;
        }
      }
      std::vector<int> after_x(input.size()),after_len(lengths.size());
      check(cudaMemcpy(after_x.data(),x,after_x.size()*sizeof(int),cudaMemcpyDeviceToHost));
      check(cudaMemcpy(after_len.data(),len,after_len.size()*sizeof(int),cudaMemcpyDeviceToHost));
      if(after_x!=input||after_len!=lengths)return 1;
    }
    check(cudaFree(x));check(cudaFree(len));check(cudaFree(y));
  }
  std::printf("PASS segmented logical: %zu exact values; empty/mixed/full lengths, nonboolean integers, repeated calls, guards and unchanged inputs\n",checked);
}
