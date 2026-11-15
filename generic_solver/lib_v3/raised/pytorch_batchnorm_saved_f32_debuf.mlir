#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @pytorch_batchnorm_saved_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: memref<?x?x?x?xf32>, %arg5: memref<?xf32>, %arg6: memref<?xf32>, %arg7: memref<?xf32>, %arg8: memref<?xf32>, %arg9: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = bufferization.to_tensor %arg9 : memref<?x?x?x?xf32>
    %1 = bufferization.to_tensor %arg8 : memref<?xf32>
    %2 = bufferization.to_tensor %arg7 : memref<?xf32>
    %3 = bufferization.to_tensor %arg6 : memref<?xf32>
    %4 = bufferization.to_tensor %arg5 : memref<?xf32>
    %5 = bufferization.to_tensor %arg4 : memref<?x?x?x?xf32>
    %6 = arith.index_cast %arg1 : i32 to index
    %7 = arith.index_cast %arg0 : i32 to index
    %8 = arith.index_cast %arg2 : i32 to index
    %9 = arith.index_cast %arg3 : i32 to index
    %10 = polygeist.submap(%2, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %11 = polygeist.submap(%10, %6, %7, %8, %9) {map = #map1} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %12 = polygeist.submap(%1, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %13 = polygeist.submap(%12, %6, %7, %8, %9) {map = #map1} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %14 = polygeist.submap(%4, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %15 = polygeist.submap(%14, %6, %7, %8, %9) {map = #map1} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %16 = polygeist.submap(%3, %6) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %17 = polygeist.submap(%16, %6, %7, %8, %9) {map = #map1} : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %18 = polygeist.submap(%5, %6, %7, %8, %9) {map = #map2} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %19 = polygeist.submap(%0, %6, %7, %8, %9) {map = #map2} : (tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %20 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map3, #map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%11, %13, %15, %17, %18 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%19 : tensor<?x?x?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %in_3: f32, %out: f32):
      %23 = arith.subf %in_3, %in_1 : f32
      %24 = arith.mulf %in, %23 : f32
      %25 = arith.mulf %24, %in_2 : f32
      %26 = arith.addf %25, %in_0 : f32
      linalg.yield %26 : f32
    } -> tensor<?x?x?x?xf32>
    %21 = polygeist.submapInverse(%0, %20, %6, %7, %8, %9) {map = #map2} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %22 = bufferization.to_memref %21 : memref<?x?x?x?xf32>
    memref.copy %22, %arg9 : memref<?x?x?x?xf32> to memref<?x?x?x?xf32>
    return
  }
}

