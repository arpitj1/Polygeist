// RUN: polygeist-opt --delinearize-memref-accesses --split-input-file %s | FileCheck %s

// A flat invariant fill covers complete rows of the statically recovered
// stride chain. Reconstructing its lexicographic indices preserves store
// order and lets every access share the same multidimensional view.
// CHECK-LABEL: func.func @flat_fill_rows
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [%{{.*}}, 5], strides: [5, 1]
// CHECK: affine.for %[[I:.*]] = 0 to 3
// CHECK: affine.for %[[J:.*]] = 0 to 5
// CHECK: affine.store %{{.*}}, %[[VIEW]][%[[I]], %[[J]]]
// CHECK: affine.load %[[VIEW]][%{{.*}}, %{{.*}}]
func.func @flat_fill_rows(%out: memref<?xf32>, %seed: f32) {
  affine.for %p = 0 to 15 {
    affine.store %seed, %out[%p] : memref<?xf32>
  }
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 5 {
      %v = affine.load %out[%i * 5 + %j] : memref<?xf32>
      %w = arith.addf %v, %seed : f32
      affine.store %w, %out[%i * 5 + %j] : memref<?xf32>
    }
  }
  return
}

// -----

// A multidimensional tile does not cover the last input row. The full fill
// must retain those untouched elements, and cannot be limited to the tile.
// CHECK-LABEL: func.func @flat_fill_tiled
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [%{{.*}}, 5, 6], strides: [30, 6, 1]
// CHECK: affine.for %[[I:.*]] = 0 to 3
// CHECK: affine.for %[[J:.*]] = 0 to 5
// CHECK: affine.for %[[K:.*]] = 0 to 6
// CHECK: affine.store %{{.*}}, %[[VIEW]][%[[I]], %[[J]], %[[K]]]
// CHECK: affine.for %{{.*}} = 0 to 3
// CHECK: affine.for %{{.*}} = 0 to 2
func.func @flat_fill_tiled(%out: memref<?xf64>, %seed: f64) {
  affine.for %p = 0 to 90 {
    affine.store %seed, %out[%p] : memref<?xf64>
  }
  affine.for %i = 0 to 3 {
    affine.for %t = 0 to 2 {
      affine.for %j = affine_map<(d0) -> (2 * d0)>(%t) to affine_map<(d0) -> (2 * d0 + 2)>(%t) {
        affine.for %k = 0 to 6 {
          %v = affine.load %out[%i * 30 + %j * 6 + %k] : memref<?xf64>
          %w = arith.addf %v, %seed : f64
          affine.store %w, %out[%i * 30 + %j * 6 + %k] : memref<?xf64>
        }
      }
    }
  }
  return
}

// -----

// An incomplete final row cannot be expanded into rectangular loops.
// CHECK-LABEL: func.func @flat_fill_partial_row
// CHECK-NOT: memref.reinterpret_cast
// CHECK: affine.for %[[P:.*]] = 0 to 14
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[P]]] : memref<?xf32>
func.func @flat_fill_partial_row(%out: memref<?xf32>, %seed: f32) {
  affine.for %p = 0 to 14 {
    affine.store %seed, %out[%p] : memref<?xf32>
  }
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 5 {
      %v = affine.load %out[%i * 5 + %j] : memref<?xf32>
      affine.store %v, %out[%i * 5 + %j] : memref<?xf32>
    }
  }
  return
}

// -----

// Induction-dependent values cannot be moved to the nested fill loop.
// CHECK-LABEL: func.func @flat_fill_indexed
// CHECK-NOT: memref.reinterpret_cast
// CHECK: affine.for %[[P:.*]] = 0 to 15
// CHECK: arith.index_cast %[[P]]
func.func @flat_fill_indexed(%out: memref<?xi32>) {
  affine.for %p = 0 to 15 {
    %value = arith.index_cast %p : index to i32
    affine.store %value, %out[%p] : memref<?xi32>
  }
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 5 {
      %v = affine.load %out[%i * 5 + %j] : memref<?xi32>
      affine.store %v, %out[%i * 5 + %j] : memref<?xi32>
    }
  }
  return
}
