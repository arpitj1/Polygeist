/* Copyright (C) 2017, Intel Corporation, all rights reserved.
 * Derived from OpenCV; Apache-2.0 license: third_party/opencv/LICENSE.
 * Third party copyrights are property of their respective owners.
 */
/* OpenCV modules/dnn/src/layers/resize_layer.cpp, ResizeLayerImpl::forward,
 * floating-point bilinear branch. Reference counterpart for the cuDNN sampler.
 * Expose the contiguous NCHW planes and specialize both scales to 1/2.
 * For nonnegative y < 2*H <= 2^23, trunc(y*0.5f) == y/2 and the
 * fractional coordinate is (y%2)*0.5f exactly. At the last input coordinate,
 * min(y0+1,H-1) == y0; elsewhere it is y0+1. Apply the same facts to x.
 * Keep the upstream difference-form interpolation expression unchanged.
 */
void NETLIB_SPEC(int B, int C, int H, int W,
                float input[B][C][H][W], float output[B][C][2*H][2*W]) {
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int y = 0; y < 2*H; ++y)
        for (int x = 0; x < 2*W; ++x) {
          int y0 = y/2, x0 = x/2;
          int y1 = y0+1 < H ? y0+1 : y0;
          int x1 = x0+1 < W ? x0+1 : x0;
          float fy = (float)(y%2)*0.5f;
          float fx = (float)(x%2)*0.5f;
          float a = input[b][c][y0][x0];
          float d = input[b][c][y1][x1];
          float v = input[b][c][y1][x0];
          float u = input[b][c][y0][x1];
          output[b][c][y][x] = a + fy*(v-a) + fx*(u-a + fy*(d-u-v+a));
        }
}
