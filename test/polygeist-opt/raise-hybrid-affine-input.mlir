// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

// A mixed affine/indirect read is a scalar payload. Non-permutation input
// indices must retain their floor/mod arithmetic when the loop is raised.
// CHECK-LABEL: func.func @floor_and_gather
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK: arith.divsi
// CHECK: memref.load
// CHECK: memref.load
// CHECK: linalg.yield
// CHECK-NOT: affine.for
// CHECK: return
func.func @floor_and_gather(%input: memref<8xf32>,
                           %indices: memref<16xi32>,
                           %output: memref<16xf32>) {
  affine.for %i = 0 to 16 {
    %a = affine.load %input[%i floordiv 2] : memref<8xf32>
    %j32 = affine.load %indices[%i] : memref<16xi32>
    %j = arith.index_cast %j32 : i32 to index
    %b = memref.load %input[%j] : memref<8xf32>
    %v = arith.addf %a, %b : f32
    affine.store %v, %output[%i] : memref<16xf32>
  }
  return
}

// The broader input-map support must not bypass a destination recurrence.
// CHECK-LABEL: func.func @destination_recurrence
// CHECK: affine.for
// CHECK: affine.load
// CHECK: affine.store
// CHECK: return
func.func @destination_recurrence(%input: memref<8xf32>,
                                 %output: memref<16xf32>, %index: index) {
  affine.for %i = 1 to 16 {
    %a = affine.load %output[%i - 1] : memref<16xf32>
    %b = memref.load %input[%index] : memref<8xf32>
    %v = arith.addf %a, %b : f32
    affine.store %v, %output[%i] : memref<16xf32>
  }
  return
}
