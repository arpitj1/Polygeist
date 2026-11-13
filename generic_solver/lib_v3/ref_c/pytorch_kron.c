/* Scalar C translation of public PyTorch KronImpl::kron_out for rank-two,
 * contiguous FP32 inputs. See third_party/pytorch/LICENSE for BSD terms.
 * LinearAlgebra.cpp constructs self_view [M,1,N,1], other_view [1,P,1,Q],
 * and result_mul [M,P,N,Q], then calls at::mul_out. BinaryMulKernel.cu uses
 * MulFunctor<float>. The loops below serialize that broadcast iterator in
 * the contiguous result_mul order; the final unsafe_view changes no bytes.
 * The output indices apply the final rank-two view of result_mul; no
 * application-derived formulas or fixed benchmark sizes appear here.
 */
void NETLIB_SPEC(int M,int N,int P,int Q,const float self[M][N],
                const float other[P][Q],float output[M*P][N*Q]) {
 for(int i=0;i<M;++i)
  for(int j=0;j<P;++j)
   for(int k=0;k<N;++k)
    for(int l=0;l<Q;++l)
     output[i*P+j][k*Q+l]=self[i][k]*other[j][l];
}
