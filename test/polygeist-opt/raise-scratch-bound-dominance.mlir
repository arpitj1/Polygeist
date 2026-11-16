// RUN: polygeist-opt %s --raise-affine-to-linalg | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

// Expanding scratch to one slot per inner iteration must not hoist a size
// that depends on the outer induction variable outside the outer loop.
func.func @triangular_scratch(%n: index, %input: memref<?x?xf64>,
                              %output: memref<?x?xf64>) {
  %scratch = memref.alloca() : memref<f64>
  affine.for %i = 0 to %n {
    affine.for %j = 0 to #identity(%i) {
      %initial = affine.load %input[%i, %j] : memref<?x?xf64>
      affine.store %initial, %scratch[] : memref<f64>
      affine.for %k = 0 to #identity(%j) {
        %value = affine.load %input[%i, %k] : memref<?x?xf64>
        %acc = affine.load %scratch[] : memref<f64>
        %next = arith.subf %acc, %value : f64
        affine.store %next, %scratch[] : memref<f64>
      }
      %result = affine.load %scratch[] : memref<f64>
      affine.store %result, %output[%i, %j] : memref<?x?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @triangular_scratch
// CHECK: memref.alloca() : memref<f64>
// CHECK: affine.for
// CHECK: linalg.generic
// CHECK: return

// An invariant size is available at the original allocation scope, so scratch
// expansion remains enabled for rectangular iteration spaces.
func.func @rectangular_scratch(%n: index, %input: memref<?x?xf64>,
                               %output: memref<?x?xf64>) {
  %scratch = memref.alloca() : memref<f64>
  affine.for %i = 0 to %n {
    affine.for %j = 0 to %n {
      %initial = affine.load %input[%i, %j] : memref<?x?xf64>
      affine.store %initial, %scratch[] : memref<f64>
      affine.for %k = 0 to #identity(%j) {
        %value = affine.load %input[%i, %k] : memref<?x?xf64>
        %acc = affine.load %scratch[] : memref<f64>
        %next = arith.subf %acc, %value : f64
        affine.store %next, %scratch[] : memref<f64>
      }
      %result = affine.load %scratch[] : memref<f64>
      affine.store %result, %output[%i, %j] : memref<?x?xf64>
    }
  }
  return
}

// CHECK-LABEL: func.func @rectangular_scratch
// CHECK: memref.alloca({{.*}}) : memref<?xf64>
// CHECK: linalg.generic
// CHECK: return
