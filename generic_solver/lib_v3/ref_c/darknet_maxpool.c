/* Public-domain Darknet forward_maxpool_layer, src/maxpool_layer.c.
 * Project its values output (the independent argmax output is unused), expose
 * NCHW axes, and specialize padding=0, size=stride=2, even spatial extents.
 * Each original load is then in bounds. Preserve the ordered comparison,
 * -FLT_MAX seed, and per-output reduction order. Tensor extents stay dynamic.
 * This is a reference counterpart for the closed-source cuDNN pooling API.
 */
void NETLIB_SPEC(int B, int C, int OH, int OW,
                float input[B][C][2*OH][2*OW],
                float output[B][C][OH][OW]) {
  for (int b=0; b<B; ++b)
    for (int c=0; c<C; ++c)
      for (int y=0; y<OH; ++y)
        for (int x=0; x<OW; ++x) {
          float maximum=-3.402823466e38f;
          for (int ky=0; ky<2; ++ky)
            for (int kx=0; kx<2; ++kx) {
              float value=input[b][c][2*y+ky][2*x+kx];
              maximum=value>maximum ? value : maximum;
            }
          output[b][c][y][x]=maximum;
        }
}
