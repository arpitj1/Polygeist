/* An isolated typed division is emitted through actual Thrust transform.
 * Persistent scratch protects strided/overlapping views and captured graphs. */
#define POINTWISE_DIVISION_CACHE_CAP 8
struct pointwise_division_plan {
  int32_t n;
  float *left, *right, *output;
};
static struct pointwise_division_plan
    g_pointwise_division_cache[POINTWISE_DIVISION_CACHE_CAP];
static struct pointwise_division_plan *pointwise_division_find(int32_t n,
                                                             int capturing) {
  for (int i=0;i<POINTWISE_DIVISION_CACHE_CAP;++i)
    if (g_pointwise_division_cache[i].n == n) return &g_pointwise_division_cache[i];
  if (capturing) {
    fprintf(stderr,"polygeist runtime: Thrust division requires warmup before capture\n");
    abort();
  }
  for (int i=0;i<POINTWISE_DIVISION_CACHE_CAP;++i) {
    struct pointwise_division_plan *p=&g_pointwise_division_cache[i];
    if (p->n) continue;
    CUDA_CHECK(cudaMalloc((void**)&p->left,(size_t)n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&p->right,(size_t)n*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&p->output,(size_t)n*sizeof(float)));
    p->n=n;
    return p;
  }
  fprintf(stderr,"polygeist runtime: Thrust division cache full\n");
  abort();
}
static void pointwise_division_execute(
    int32_t n, uint32_t instruction, const float scalars[8],
    const float *inputs[4], const int32_t strides[4], int32_t out_stride,
    float *output, int capturing) {
  int refs[2]={(int)((instruction>>16)&255),(int)((instruction>>8)&255)};
  void *host[3]={NULL,NULL,output}, *device[3];
  int32_t inc[3]={1,1,out_stride};
  size_t bytes[3]={0};
  for (int i=0;i<2;++i) {
    if (refs[i]>=12) {
      fprintf(stderr,"polygeist runtime: isolated division has unresolved operand\n");
      abort();
    }
    if (refs[i]<4) {host[i]=(void*)inputs[refs[i]];inc[i]=strides[refs[i]];}
  }
  for (int i=0;i<3;++i) {
    if (i<2 && refs[i]>=4) continue;
    if (!host[i] || inc[i]<=0 || (size_t)(n-1)>
        (SIZE_MAX/sizeof(float)-1)/(size_t)inc[i]) {
      fprintf(stderr,"polygeist runtime: invalid division view\n");
      abort();
    }
    bytes[i]=((size_t)(n-1)*(size_t)inc[i]+1)*sizeof(float);
  }
  struct pointwise_division_plan *p=pointwise_division_find(n,capturing);
  register_host_operands_safe(host,bytes,device,3);
  size_t dense_bytes=(size_t)n*sizeof(float);
  double host_start_ms=wall_time_ms();
  timing_gpu_begin();
  for (int i=0;i<2;++i) {
    if (refs[i]>=4) continue;
    float *dst=i?p->right:p->left;
    if (inc[i]==1)
      CUDA_CHECK(cudaMemcpyAsync(dst,device[i],dense_bytes,
                                 cudaMemcpyDeviceToDevice,g_stream));
    else
      CUDA_CHECK(cudaMemcpy2DAsync(dst,sizeof(float),device[i],
                                   (size_t)inc[i]*sizeof(float),sizeof(float),n,
                                   cudaMemcpyDeviceToDevice,g_stream));
  }
  if (refs[0]<4 && refs[1]<4)
    polygeist_thrust_divide_f32(n,p->left,p->right,p->output);
  else if (refs[0]>=4 && refs[1]<4)
    polygeist_thrust_divide_scalar_left_f32(n,scalars[refs[0]-4],p->right,p->output);
  else if (refs[0]<4 && refs[1]>=4)
    polygeist_thrust_divide_scalar_right_f32(n,p->left,scalars[refs[1]-4],p->output);
  else
    polygeist_thrust_divide_scalars_f32(n,scalars[refs[0]-4],scalars[refs[1]-4],p->output);
  if (out_stride==1)
    CUDA_CHECK(cudaMemcpyAsync(device[2],p->output,dense_bytes,
                               cudaMemcpyDeviceToDevice,g_stream));
  else
    CUDA_CHECK(cudaMemcpy2DAsync(device[2],(size_t)out_stride*sizeof(float),
                                 p->output,sizeof(float),sizeof(float),n,
                                 cudaMemcpyDeviceToDevice,g_stream));
  timing_gpu_end("thrustTransformDivide_f32",1,n,1,host_start_ms);
}
