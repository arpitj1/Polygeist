#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2)[s0] -> (s0, d1, d2)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (s0, d1, d2 + d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0)[s0] -> (s0)>
#map6 = affine_map<(d0)[s0, s1] -> (s0, s1, d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv1d_bias_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>, %arg7: memref<?xf32>, %arg8: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg8 : memref<?x?x?xf32>
    %1 = bufferization.to_tensor %arg7 : memref<?xf32>
    %2 = bufferization.to_tensor %arg6 : memref<?x?x?xf32>
    %3 = bufferization.to_tensor %arg5 : memref<?x?x?xf32>
    %4 = arith.index_cast %arg0 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = arith.index_cast %arg4 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = affine.for %arg9 = 0 to %4 iter_args(%arg10 = %0) -> (tensor<?x?x?xf32>) {
      %11 = affine.for %arg11 = 0 to %5 iter_args(%arg12 = %arg10) -> (tensor<?x?x?xf32>) {
        %alloca = memref.alloca(%6) : memref<?xf32>
        %12 = bufferization.to_tensor %alloca : memref<?xf32>
        %13 = polygeist.submap(%12, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
        %14 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%13 : tensor<?xf32>) {
        ^bb0(%out: f32):
          linalg.yield %cst : f32
        } -> tensor<?xf32>
        %15 = polygeist.submapInverse(%12, %14, %6) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
        %16 = polygeist.submap(%2, %arg11, %6, %7, %8) {map = #map1} : (tensor<?x?x?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
        %17 = polygeist.submap(%3, %arg9, %6, %7, %8) {map = #map2} : (tensor<?x?x?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
        %18 = polygeist.submap(%15, %6, %7, %8) {map = #map3} : (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
        %19 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "reduction", "reduction"], library_call = ""} ins(%16, %17 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %26 = arith.mulf %in, %in_0 : f32
          %27 = arith.addf %26, %out : f32
          linalg.yield %27 : f32
        } -> tensor<?x?x?xf32>
        %20 = polygeist.submapInverse(%15, %19, %6, %7, %8) {map = #map3} : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?xf32>
        %21 = polygeist.submap(%20, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
        %22 = polygeist.submap(%1, %arg11, %6) {map = #map5} : (tensor<?xf32>, index, index) -> tensor<?xf32>
        %23 = polygeist.submap(%arg12, %arg9, %arg11, %6) {map = #map6} : (tensor<?x?x?xf32>, index, index, index) -> tensor<?xf32>
        %24 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%21, %22 : tensor<?xf32>, tensor<?xf32>) outs(%23 : tensor<?xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %26 = arith.addf %in, %in_0 : f32
          linalg.yield %26 : f32
        } -> tensor<?xf32>
        %25 = polygeist.submapInverse(%arg12, %24, %arg9, %arg11, %6) {map = #map6} : (tensor<?x?x?xf32>, tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
        affine.yield %25 : tensor<?x?x?xf32>
      }
      affine.yield %11 : tensor<?x?x?xf32>
    }
    %10 = bufferization.to_memref %9 : memref<?x?x?xf32>
    memref.copy %10, %arg8 : memref<?x?x?xf32> to memref<?x?x?xf32>
    return
  }
}

