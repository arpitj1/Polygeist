// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s

// Logical pointwise coordinates must collapse while retaining the storage permutation.
// CHECK-LABEL: func.func @permuted_copy
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: linalg.yield
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
  func.func @permuted_copy(%arg0: memref<?x16x32xf32>, %arg1: memref<?x4x32xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x4x32xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x16x32xf32>
    %2 = polygeist.submap(%1, %c4, %c16, %c32) {map = #map} : (tensor<?x16x32xf32>, index, index, index) -> tensor<?x?x?xf32>
    %3 = polygeist.submap(%0, %c4, %c16, %c32) {map = #map1} : (tensor<?x4x32xf32>, index, index, index) -> tensor<?x?x?xf32>
    %4 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%2 : tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?x?xf32>
    %5 = polygeist.submapInverse(%0, %4, %c4, %c16, %c32) {map = #map1} : (tensor<?x4x32xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x4x32xf32>
    %6 = bufferization.to_memref %5 : memref<?x4x32xf32>
    memref.copy %6, %arg1 : memref<?x4x32xf32> to memref<?x4x32xf32>
    return
  }


// CHECK-LABEL: func.func @broadcast_read
// CHECK-NOT: scf.for
// CHECK: tensor.collapse_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: linalg.yield
func.func @broadcast_read(%x: tensor<3xf32>, %y: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %a = polygeist.submap(%x, %c3, %c4) {map=affine_map<(i,j)->(i)>} : (tensor<3xf32>,index,index) -> tensor<?x?xf32>
  %b = polygeist.submap(%y, %c3, %c4) {map=affine_map<(i,j)->(i,j)>} : (tensor<3x4xf32>,index,index) -> tensor<?x?xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%a:tensor<?x?xf32>) outs(%b:tensor<?x?xf32>) {
  ^bb0(%v:f32,%old:f32):
    linalg.yield %v : f32
  } -> tensor<?x?xf32>
  %out = polygeist.submapInverse(%y,%r,%c3,%c4) {map=affine_map<(i,j)->(i,j)>} : (tensor<3x4xf32>,tensor<?x?xf32>,index,index) -> tensor<3x4xf32>
  return %out : tensor<3x4xf32>
}

// Fixed coordinates are slices, independent of the scalar computation. Keep
// the logical domain so a nonzero slice and transposed output collapse too.
// CHECK-LABEL: func.func @fixed_coordinate_add
// CHECK-NOT: scf.for
// CHECK: tensor.collapse_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: arith.addf
// CHECK-NOT: scf.for
// CHECK: return
func.func @fixed_coordinate_add(%x: tensor<3x5x7xf32>, %b: tensor<3x7xf32>, %y: tensor<7x5xf32>) -> tensor<7x5xf32> {
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %a = polygeist.submap(%x,%c5,%c7) {map=affine_map<(i,j)->(2,i,j)>} : (tensor<3x5x7xf32>,index,index) -> tensor<?x?xf32>
  %bias = polygeist.submap(%b,%c5,%c7) {map=affine_map<(i,j)->(2,j)>} : (tensor<3x7xf32>,index,index) -> tensor<?x?xf32>
  %out = polygeist.submap(%y,%c5,%c7) {map=affine_map<(i,j)->(j,i)>} : (tensor<7x5xf32>,index,index) -> tensor<?x?xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%a,%bias:tensor<?x?xf32>,tensor<?x?xf32>) outs(%out:tensor<?x?xf32>) {
  ^bb0(%v:f32,%bias_value:f32,%old:f32):
    %sum = arith.addf %v,%bias_value : f32
    linalg.yield %sum : f32
  } -> tensor<?x?xf32>
  %result = polygeist.submapInverse(%y,%r,%c5,%c7) {map=affine_map<(i,j)->(j,i)>} : (tensor<7x5xf32>,tensor<?x?xf32>,index,index) -> tensor<7x5xf32>
  return %result : tensor<7x5xf32>
}
