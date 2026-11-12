/* Scalar FP32 counterpart of public PyTorch
 * aten/src/ATen/native/AveragePool3d.cpp:336-415,
 * avg_pool3d_backward_out_frame. BSD license: third_party/pytorch/LICENSE.
 * Specialize window=stride=(2,2,2), padding=0, ceil_mode=false,
 * count_include_pad=true, divisor_override absent. Under these conditions,
 * every visited window lies inside the input and divide_factor is exactly 8.
 * Serializing independent slices and exposing the slice and spatial array axes
 * preserves the reference scatter order. Input lengths are 2*O+R where
 * R is 0 or 1; untouched odd-tail positions retain the initial zero.
 * This is a public reference counterpart for cuDNN PoolingBackward, whose
 * implementation is unavailable. It is not copied from the ATen fixture.
 */
void NETLIB_SPEC(int S, int O0, int O1, int O2,
                int R0, int R1, int R2,
                float grad_output[S][O0][O1][O2],
                float grad_input[S][2*O0+R0][2*O1+R1][2*O2+R2]) {
  for (int s=0; s<S; ++s) {
      for (int z=0; z<2*O0+R0; ++z)
        for (int y=0; y<2*O1+R1; ++y)
          for (int x=0; x<2*O2+R2; ++x)
            grad_input[s][z][y][x]=0.0f;
      for (int t=0; t<O0; ++t)
        for (int y=0; y<O1; ++y)
          for (int x=0; x<O2; ++x) {
            float val=grad_output[s][t][y][x];
            for (int iz=t*2; iz<t*2+2; ++iz)
              for (int iy=y*2; iy<y*2+2; ++iy)
                for (int ix=x*2; ix<x*2+2; ++ix)
                  grad_input[s][iz][iy][ix]+=val/8.0f;
          }
    }
}
