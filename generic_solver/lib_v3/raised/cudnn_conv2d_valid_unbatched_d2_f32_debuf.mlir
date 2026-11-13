#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4 * 2 + d1, d5 * 2 + d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_valid_unbatched_d2_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: memref<?x?x?xf32>, %arg9: memref<?x?x?x?xf32>, %arg10: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg10 : memref<?x?x?xf32>
    %1 = bufferization.to_tensor %arg9 : memref<?x?x?x?xf32>
    %2 = bufferization.to_tensor %arg8 : memref<?x?x?xf32>
    %3 = arith.index_cast %arg3 : i32 to index
    %4 = arith.index_cast %arg6 : i32 to index
    %5 = arith.index_cast %arg7 : i32 to index
    %6 = arith.index_cast %arg0 : i32 to index
    %7 = arith.index_cast %arg4 : i32 to index
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = polygeist.submap(%0, %3, %4, %5) {map = #map} : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %10 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?xf32>
    %11 = polygeist.submapInverse(%0, %10, %3, %4, %5) {map = #map} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %12 = polygeist.submap(%1, %3, %4, %5, %6, %7, %8) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %13 = polygeist.submap(%2, %3, %4, %5, %6, %7, %8) {map = #map2} : (tensor<?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %14 = polygeist.submap(%11, %3, %4, %5, %6, %7, %8) {map = #map3} : (tensor<?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %15 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%12, %13 : tensor<?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?xf32>) outs(%14 : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %18 = arith.mulf %in, %in_0 : f32
      %19 = arith.addf %18, %out : f32
      linalg.yield %19 : f32
    } -> tensor<?x?x?x?x?x?xf32>
    %16 = polygeist.submapInverse(%11, %15, %3, %4, %5, %6, %7, %8) {map = #map3} : (tensor<?x?x?xf32>, tensor<?x?x?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x?x?xf32>
    %17 = bufferization.to_memref %16 : memref<?x?x?xf32>
    memref.copy %17, %arg10 : memref<?x?x?xf32> to memref<?x?x?xf32>
    return
  }
}

