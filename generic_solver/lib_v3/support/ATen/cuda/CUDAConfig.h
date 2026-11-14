#pragma once
// Standalone public CUDA math header configuration. Optional ATen tensor
// backends are unused by these scalar definitions and are not built here.
#define AT_CUDNN_ENABLED() 0
#define AT_CUSPARSELT_ENABLED() 0
#define AT_HIPSPARSELT_ENABLED() 0
#define AT_ROCM_ENABLED() 0
#define AT_MAGMA_ENABLED() 0
#define NVCC_FLAGS_EXTRA ""
