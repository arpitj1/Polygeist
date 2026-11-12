// RUN: polygeist-opt %s --raise-affine-to-linalg-pipeline | FileCheck %s

// CHECK: d0 floordiv 4
// CHECK: d0 mod 4
// CHECK-LABEL: func.func @blocked_permutation
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK-NOT: affine.for
// CHECK: return
module {
  func.func @blocked_permutation(%arg0: memref<?x64xf32>, %arg1: memref<?x16x4x4xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1 = arith.constant -1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    affine.for %arg2 = 0 to 64 {
      %0 = arith.cmpi slt, %arg2, %c0 : index
      %1 = arith.subi %c-1, %arg2 : index
      %2 = arith.select %0, %1, %arg2 : index
      %3 = arith.divsi %2, %c4 : index
      %4 = arith.subi %c-1, %3 : index
      %5 = arith.select %0, %4, %3 : index
      %6 = arith.remsi %arg2, %c4 : index
      %7 = arith.cmpi slt, %6, %c0 : index
      %8 = arith.addi %6, %c4 : index
      %9 = arith.select %7, %8, %6 : index
      affine.for %arg3 = 0 to 64 {
        %10 = affine.load %arg0[%arg2, %arg3] : memref<?x64xf32>
        %11 = arith.cmpi slt, %arg3, %c0 : index
        %12 = arith.subi %c-1, %arg3 : index
        %13 = arith.select %11, %12, %arg3 : index
        %14 = arith.divsi %13, %c4 : index
        %15 = arith.subi %c-1, %14 : index
        %16 = arith.select %11, %15, %14 : index
        %17 = arith.remsi %arg3, %c4 : index
        %18 = arith.cmpi slt, %17, %c0 : index
        %19 = arith.addi %17, %c4 : index
        %20 = arith.select %18, %19, %17 : index
        memref.store %10, %arg1[%5, %16, %9, %20] : memref<?x16x4x4xf32>
      }
    }
    return
  }
}
