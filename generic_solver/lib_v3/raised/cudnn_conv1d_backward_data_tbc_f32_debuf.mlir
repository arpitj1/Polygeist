#map = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map3 = affine_map<(d0, d1, d2, d3)[s0] -> (s0, d0, d1)>
#map4 = affine_map<(d0, d1, d2, d3)[s0] -> (d2 + s0, d0, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv1d_backward_data_tbc_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>, %arg7: memref<?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg7 : memref<?x?x?xf32>
    %1 = bufferization.to_tensor %arg6 : memref<?x?x?xf32>
    %2 = bufferization.to_tensor %arg5 : memref<?x?x?xf32>
    %3 = arith.addi %arg0, %arg4 : i32
    %4 = arith.index_cast %3 : i32 to index
    %5 = arith.index_cast %arg1 : i32 to index
    %6 = arith.index_cast %arg2 : i32 to index
    %7 = affine.apply #map()[%4]
    %8 = polygeist.submap(%0, %7, %5, %6) {map = #map1} : (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %9 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?xf32>
    %10 = polygeist.submapInverse(%0, %9, %7, %5, %6) {map = #map1} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %11 = arith.index_cast %arg0 : i32 to index
    %12 = arith.index_cast %arg3 : i32 to index
    %13 = arith.index_cast %arg4 : i32 to index
    %14 = affine.for %arg8 = 0 to %11 iter_args(%arg9 = %10) -> (tensor<?x?x?xf32>) {
      %16 = polygeist.submap(%1, %5, %12, %13, %6) {map = #map2} : (tensor<?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
      %17 = polygeist.submap(%2, %arg8, %5, %12, %13, %6) {map = #map3} : (tensor<?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?x?xf32>
      %18 = polygeist.submap(%arg9, %arg8, %5, %12, %13, %6) {map = #map4} : (tensor<?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?x?xf32>
      %19 = linalg.generic {doc = "", indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "reduction", "parallel", "parallel"], library_call = ""} ins(%16, %17 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%18 : tensor<?x?x?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %21 = arith.mulf %in, %in_0 : f32
        %22 = arith.addf %out, %21 : f32
        linalg.yield %22 : f32
      } -> tensor<?x?x?x?xf32>
      %20 = polygeist.submapInverse(%arg9, %19, %arg8, %5, %12, %13, %6) {map = #map4} : (tensor<?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x?x?xf32>
      affine.yield %20 : tensor<?x?x?xf32>
    }
    %15 = bufferization.to_memref %14 : memref<?x?x?xf32>
    memref.copy %15, %arg7 : memref<?x?x?xf32> to memref<?x?x?xf32>
    return
  }
}

