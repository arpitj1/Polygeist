/* Scalar C translation of public PyTorch Normalization.cuh,
 * batch_norm_transform_input_kernel<float,float,float,true,index_t>.
 * BSD license and copyright: third_party/pytorch/LICENSE.
 * Present weight/bias arrays and caller-supplied saved mean/inverse standard
 * deviation are selected. Independent CUDA threads are serialized; the flat
 * spatial feature accessor is exposed as contiguous H,W axes of NCHW storage.
 * The per-element arithmetic retains the original FP32 operation order.
 */
void NETLIB_SPEC(int N,int C,int H,int W,
                const float input[N][C][H][W],const float mean[C],
                const float invstd[C],const float weight[C],const float bias[C],
                float output[N][C][H][W]) {
 for(int plane=0;plane<C;++plane) {
  float gamma=weight[plane];
  float beta=bias[plane];
  float center=mean[plane];
  float inverse=invstd[plane];
  for(int batch=0;batch<N;++batch)
   for(int y=0;y<H;++y)
    for(int x=0;x<W;++x)
     output[batch][plane][y][x]=gamma*(input[batch][plane][y][x]-center)*inverse+beta;
 }
}
