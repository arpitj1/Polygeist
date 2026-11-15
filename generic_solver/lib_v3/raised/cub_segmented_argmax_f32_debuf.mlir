#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#map6 = affine_map<(d0) -> (d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_segmented_argmax_f32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xf32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg3 : memref<?xi32>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xf32>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = affine.apply #map()[%3]
    %5 = tensor.empty(%2) : tensor<?xi32>
    %6 = tensor.empty(%2) : tensor<?xf32>
    %7 = polygeist.submap(%5, %2) {map = #map1} : (tensor<?xi32>, index) -> tensor<?xi32>
    %8 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%7 : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?xi32>
    %9 = polygeist.submapInverse(%5, %8, %2) {map = #map1} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %10 = polygeist.submap(%1, %2) {map = #map2} : (tensor<?x?xf32>, index) -> tensor<?xf32>
    %11 = polygeist.submap(%6, %2) {map = #map1} : (tensor<?xf32>, index) -> tensor<?xf32>
    %12 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%10 : tensor<?xf32>) outs(%11 : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?xf32>
    %13 = polygeist.submapInverse(%6, %12, %2) {map = #map1} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %14 = polygeist.submap(%1, %2, %4) {map = #map3} : (tensor<?x?xf32>, index, index) -> tensor<?x?xf32>
    %15 = polygeist.submap(%9, %2, %4) {map = #map4} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %16 = polygeist.submap(%13, %2, %4) {map = #map4} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
    %17:2 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction"], library_call = ""} ins(%14 : tensor<?x?xf32>) outs(%15, %16 : tensor<?x?xi32>, tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: i32, %out_0: f32):
      %24 = linalg.index 1 : index
      %25 = affine.apply #map6(%24)
      %26 = arith.index_cast %25 : index to i32
      %27 = arith.cmpf ogt, %in, %out_0 : f32
      %28 = arith.select %27, %26, %out : i32
      %29 = arith.select %27, %in, %out_0 : f32
      linalg.yield %28, %29 : i32, f32
    } -> (tensor<?x?xi32>, tensor<?x?xf32>)
    %18 = polygeist.submapInverse(%9, %17#0, %2, %4) {map = #map4} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %19 = polygeist.submap(%18, %2) {map = #map1} : (tensor<?xi32>, index) -> tensor<?xi32>
    %20 = polygeist.submap(%0, %2) {map = #map1} : (tensor<?xi32>, index) -> tensor<?xi32>
    %21 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%19 : tensor<?xi32>) outs(%20 : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?xi32>
    %22 = polygeist.submapInverse(%0, %21, %2) {map = #map1} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %23 = bufferization.to_memref %22 : memref<?xi32>
    memref.copy %23, %arg3 : memref<?xi32> to memref<?xi32>
    return
  }
}

