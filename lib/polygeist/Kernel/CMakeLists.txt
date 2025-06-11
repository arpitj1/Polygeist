add_mlir_dialect_library(MLIRPolygeistKernel
  KernelDialect.cpp
  KernelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/polygeist/Kernel

  DEPENDS
  MLIRKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
  MLIRSupport
) 