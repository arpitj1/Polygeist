#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_segmented_count_nonzero_f32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = polygeist.submap(%0, %2) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%4 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %6 = polygeist.submapInverse(%0, %5, %2) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %7 = polygeist.submap(%1, %2, %3) {map = #map1} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %8 = polygeist.submap(%6, %2, %3) {map = #map2} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %9 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%7 : tensor<?x?xf32>) outs(%8 : tensor<?x?xi32>) {
    ^bb0(%in: f32, %out: i32):
      %12 = arith.cmpf une, %in, %cst : f32
      %13 = arith.extui %12 : i1 to i32
      %14 = arith.addi %out, %13 : i32
      linalg.yield %14 : i32
    } -> tensor<?x?xi32>
    %10 = polygeist.submapInverse(%6, %9, %2, %3) {map = #map2} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %11 = bufferization.to_memref %10 : memref<?xi32>
    memref.copy %11, %arg3 : memref<?xi32> to memref<?xi32>
    return
  }
}

