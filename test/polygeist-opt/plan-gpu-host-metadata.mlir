// RUN: polygeist-opt %s --plan-gpu-data-residency="function=copy_metadata promote-function-arguments=false" | FileCheck %s

// The first pointer is read on the host; only the second accepts GPU storage.
// CHECK-LABEL: func.func @copy_metadata
// CHECK: gpu.alloc
// CHECK-SAME: memref<32xf32>
// CHECK: memref.alloca() : memref<1xi64>
// CHECK-NOT: gpu.alloc{{.*}}memref<1xi64>
// CHECK: call @runtime_copy
func.func private @runtime_copy(!llvm.ptr, !llvm.ptr) attributes {polygeist.gpu_runtime, polygeist.gpu_data_operands=array<i64: 1>}
func.func @copy_metadata() {
  %meta = memref.alloca() : memref<1xi64>
  %data = memref.alloc() : memref<32xf32>
  %mi = memref.extract_aligned_pointer_as_index %meta : memref<1xi64> -> index
  %di = memref.extract_aligned_pointer_as_index %data : memref<32xf32> -> index
  %mint = arith.index_cast %mi : index to i64
  %dint = arith.index_cast %di : index to i64
  %mp = llvm.inttoptr %mint : i64 to !llvm.ptr
  %dp = llvm.inttoptr %dint : i64 to !llvm.ptr
  // Pointer offsets do not dereference the data or change its ownership.
  %zero = arith.constant 0 : i64
  %mp_offset = llvm.getelementptr %mp[%zero] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %dp_offset = llvm.getelementptr %dp[%zero] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  func.call @runtime_copy(%mp_offset,%dp_offset) : (!llvm.ptr,!llvm.ptr)->()
  memref.dealloc %data : memref<32xf32>
  return
}
