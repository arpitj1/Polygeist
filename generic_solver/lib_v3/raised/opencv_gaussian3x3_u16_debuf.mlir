#map = affine_map<()[s0] -> (s0 - 2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (d1 + s0, d2 + d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0)[s0] -> (s0 + 1, d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_gaussian3x3_u16(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi16>, %arg3: memref<?x?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg3 : memref<?x?xi16>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xi16>
    %2 = arith.index_cast %arg0 : i32 to index
    %3 = arith.index_cast %arg1 : i32 to index
    %4 = tensor.empty() : tensor<3xi32>
    %inserted = tensor.insert %c1_i32 into %4[%c0] : tensor<3xi32>
    %inserted_0 = tensor.insert %c2_i32 into %inserted[%c1] : tensor<3xi32>
    %inserted_1 = tensor.insert %c1_i32 into %inserted_0[%c2] : tensor<3xi32>
    %5 = affine.apply #map()[%3]
    %6 = affine.apply #map()[%3]
    %7 = affine.apply #map()[%3]
    %8 = affine.apply #map()[%3]
    %9 = affine.for %arg4 = 0 to #map()[%2] iter_args(%arg5 = %0) -> (tensor<?x?xi16>) {
      %alloca = memref.alloca(%5) : memref<?xi32>
      %11 = bufferization.to_tensor %alloca : memref<?xi32>
      %12 = polygeist.submap(%11, %6) {map = #map1} : (tensor<?xi32>, index) -> tensor<?xi32>
      %13 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%12 : tensor<?xi32>) {
      ^bb0(%out: i32):
        linalg.yield %c0_i32 : i32
      } -> tensor<?xi32>
      %14 = polygeist.submapInverse(%11, %13, %6) {map = #map1} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
      %15 = polygeist.submap(%inserted_1, %7, %c3, %c3) {map = #map2} : (tensor<3xi32>, index, index, index) -> tensor<?x?x?xi32>
      %16 = polygeist.submap(%inserted_1, %7, %c3, %c3) {map = #map3} : (tensor<3xi32>, index, index, index) -> tensor<?x?x?xi32>
      %17 = polygeist.submap(%1, %arg4, %7, %c3, %c3) {map = #map4} : (tensor<?x?xi16>, index, index, index, index) -> tensor<?x?x?xi16>
      %18 = polygeist.submap(%14, %7, %c3, %c3) {map = #map5} : (tensor<?xi32>, index, index, index) -> tensor<?x?x?xi32>
      %19 = linalg.generic {doc = "", indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction", "reduction"], library_call = ""} ins(%15, %16, %17 : tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<?x?x?xi16>) outs(%18 : tensor<?x?x?xi32>) {
      ^bb0(%in: i32, %in_2: i32, %in_3: i16, %out: i32):
        %25 = arith.muli %in, %in_2 : i32
        %26 = arith.extui %in_3 : i16 to i32
        %27 = arith.muli %25, %26 : i32
        %28 = arith.addi %out, %27 : i32
        linalg.yield %28 : i32
      } -> tensor<?x?x?xi32>
      %20 = polygeist.submapInverse(%14, %19, %7, %c3, %c3) {map = #map5} : (tensor<?xi32>, tensor<?x?x?xi32>, index, index, index) -> tensor<?xi32>
      %21 = polygeist.submap(%20, %8) {map = #map1} : (tensor<?xi32>, index) -> tensor<?xi32>
      %22 = polygeist.submap(%arg5, %arg4, %8) {map = #map7} : (tensor<?x?xi16>, index, index) -> tensor<?xi16>
      %23 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel"], library_call = ""} ins(%21 : tensor<?xi32>) outs(%22 : tensor<?xi16>) {
      ^bb0(%in: i32, %out: i16):
        %25 = arith.addi %in, %c8_i32 : i32
        %26 = arith.shrsi %25, %c4_i32 : i32
        %27 = arith.trunci %26 : i32 to i16
        linalg.yield %27 : i16
      } -> tensor<?xi16>
      %24 = polygeist.submapInverse(%arg5, %23, %arg4, %8) {map = #map7} : (tensor<?x?xi16>, tensor<?xi16>, index, index) -> tensor<?x?xi16>
      affine.yield %24 : tensor<?x?xi16>
    }
    %10 = bufferization.to_memref %9 : memref<?x?xi16>
    memref.copy %10, %arg3 : memref<?x?xi16> to memref<?x?xi16>
    return
  }
}

