#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5 + d2, d6 + d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_valid_d1_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: memref<?x?x?x?xf32>, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg11 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg10 : memref<?x?x?x?xf32>
    %2 = bufferization.to_tensor %arg9 : memref<?x?x?x?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg4 : i32 to index
    %5 = arith.index_cast %arg7 : i32 to index
    %6 = arith.index_cast %arg8 : i32 to index
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = arith.index_cast %arg5 : i32 to index
    %9 = arith.index_cast %arg6 : i32 to index
    %10 = polygeist.submap(%0, %3, %4, %5, %6) {map = #map} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %11 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?x?xf32>
    %12 = polygeist.submapInverse(%0, %11, %3, %4, %5, %6) {map = #map} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %13 = polygeist.submap(%1, %3, %4, %5, %6, %7, %8, %9) {map = #map1} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %14 = polygeist.submap(%2, %3, %4, %5, %6, %7, %8, %9) {map = #map2} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %15 = polygeist.submap(%12, %3, %4, %5, %6, %7, %8, %9) {map = #map3} : (tensor<?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %16 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%13, %14 : tensor<?x?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?x?xf32>) outs(%15 : tensor<?x?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %19 = arith.mulf %in, %in_0 : f32
      %20 = arith.addf %19, %out : f32
      linalg.yield %20 : f32
    } -> tensor<?x?x?x?x?x?x?xf32>
    %17 = polygeist.submapInverse(%12, %16, %3, %4, %5, %6, %7, %8, %9) {map = #map3} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?xf32>
    %18 = bufferization.to_memref %17 : memref<?x?x?x?xf32>
    memref.copy %18, %arg11 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}

