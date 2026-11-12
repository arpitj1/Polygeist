#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cutlass_gemm_i8_i32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xi8>, %arg4: memref<?x?xi8>, %arg5: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg5 : memref<?x?xi32>
    %1 = bufferization.to_tensor %arg4 : memref<?x?xi8>
    %2 = bufferization.to_tensor %arg3 : memref<?x?xi8>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg2 : i32 to index
    %6 = polygeist.submap(%0, %3, %4) {map = #map} : (tensor<?x?xi32>, index, index) -> tensor<?x?xi32>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel"], library_call = ""} outs(%6 : tensor<?x?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    } -> tensor<?x?xi32>
    %8 = polygeist.submapInverse(%0, %7, %3, %4) {map = #map} : (tensor<?x?xi32>, tensor<?x?xi32>, index, index) -> tensor<?x?xi32>
    %9 = polygeist.submap(%2, %3, %4, %5) {map = #map1} : (tensor<?x?xi8>, index, index, index) -> tensor<?x?x?xi8>
    %10 = polygeist.submap(%1, %3, %4, %5) {map = #map2} : (tensor<?x?xi8>, index, index, index) -> tensor<?x?x?xi8>
    %11 = polygeist.submap(%8, %3, %4, %5) {map = #map3} : (tensor<?x?xi32>, index, index, index) -> tensor<?x?x?xi32>
    %12 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "reduction"], library_call = ""} ins(%9, %10 : tensor<?x?x?xi8>, tensor<?x?x?xi8>) outs(%11 : tensor<?x?x?xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %15 = arith.extsi %in : i8 to i32
      %16 = arith.extsi %in_0 : i8 to i32
      %17 = arith.muli %15, %16 : i32
      %18 = arith.addi %out, %17 : i32
      linalg.yield %18 : i32
    } -> tensor<?x?x?xi32>
    %13 = polygeist.submapInverse(%8, %12, %3, %4, %5) {map = #map3} : (tensor<?x?xi32>, tensor<?x?x?xi32>, index, index, index) -> tensor<?x?xi32>
    %14 = bufferization.to_memref %13 : memref<?x?xi32>
    memref.copy %14, %arg5 : memref<?x?xi32> to memref<?x?xi32>
    return
  }
}

