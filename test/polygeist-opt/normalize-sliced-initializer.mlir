// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s

// A tiled scalar reduction must retain its explicit scalar seed.
// CHECK-LABEL: func.func @row_conjunction
// CHECK: scf.for
// CHECK: tensor.insert %c1_i32
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction"]
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @row_conjunction(%arg0: memref<?x64xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %false = arith.constant false
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg1 : memref<?xi32>
    %1 = bufferization.to_tensor %arg0 : memref<?x64xi32>
    %2 = polygeist.submap(%0, %c32) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%2 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c1_i32 : i32
    } -> tensor<?xi32>
    %4 = polygeist.submapInverse(%0, %3, %c32) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %5 = polygeist.submap(%1, %c32, %c64) {map = #map1} : (tensor<?x64xi32>, index, index) -> tensor<?x?xi32>
    %6 = polygeist.submap(%4, %c32, %c64) {map = #map2} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%5 : tensor<?x?xi32>) outs(%6 : tensor<?x?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %10 = arith.cmpi ne, %out, %c0_i32 : i32
      %11 = arith.cmpi ne, %in, %c0_i32 : i32
      %12 = arith.select %10, %11, %false : i1
      %13 = arith.extui %12 : i1 to i32
      linalg.yield %13 : i32
    } -> tensor<?x?xi32>
    %8 = polygeist.submapInverse(%4, %7, %c32, %c64) {map = #map2} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %9 = bufferization.to_memref %8 : memref<?xi32>
    memref.copy %9, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }
}
