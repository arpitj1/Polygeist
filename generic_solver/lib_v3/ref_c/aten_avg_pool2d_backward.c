/* Scalar FP32 translation of public PyTorch cpu_avg_pool2d_backward in
 * aten/src/ATen/native/cpu/AvgPoolKernel.cpp, including the zero initialization
 * in aten/src/ATen/native/AveragePool2d.cpp. BSD: third_party/pytorch/LICENSE.
 * Specialize window=stride=(2,2), padding=0, ceil_mode=false, no divisor
 * override. The checked input=2*output+remainder relation makes clipping
 * redundant and fixes the divisor at 4. Independent channel slices are
 * serialized; spatial indexing is exposed as contiguous multidimensional C.
 * This public reference specifies the separately validated cuDNN API.
 */
void NETLIB_SPEC(int S, int O0, int O1, int R0, int R1,
                float grad_output[S][O0][O1],
                float grad_input[S][2*O0+R0][2*O1+R1]) {
  for (int s=0; s<S; ++s) {
    for (int y=0; y<2*O0+R0; ++y)
      for (int x=0; x<2*O1+R1; ++x)
        grad_input[s][y][x]=0.0f;
    for (int y=0; y<O0; ++y)
      for (int x=0; x<O1; ++x) {
        float delta=grad_output[s][y][x]/4.0f;
        for (int iy=y*2; iy<y*2+2; ++iy)
          for (int ix=x*2; ix<x*2+2; ++ix)
            grad_input[s][iy][ix]+=delta;
      }
  }
}
