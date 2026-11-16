#map = affine_map<()[s0] -> (s0 - 2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1 + 2)>
#map3 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map4 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map5 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_gaussian3x3_u8(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?x?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10_i32 = arith.constant 10 : i32
    %c512_i32 = arith.constant 512 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = bufferization.to_tensor %arg3 : memref<?x?xi8>
    %1 = bufferization.to_tensor %arg2 : memref<?x?xi8>
    %2 = arith.index_cast %arg1 : i32 to index
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = tensor.empty(%3, %2) : tensor<?x?xi16>
    %5 = affine.apply #map()[%2]
    %6 = polygeist.submap(%1, %3, %5) {map = #map1} : (tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %7 = polygeist.submap(%1, %3, %5) {map = #map2} : (tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %8 = polygeist.submap(%1, %3, %5) {map = #map3} : (tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %9 = polygeist.submap(%4, %3, %5) {map = #map3} : (tensor<?x?xi16>, index, index) -> tensor<?x?xi16>
    %10 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%6, %7, %8 : tensor<?x?xi8>, tensor<?x?xi8>, tensor<?x?xi8>) outs(%9 : tensor<?x?xi16>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i16):
      %21 = arith.extui %in : i8 to i32
      %22 = arith.shli %21, %c8_i32 : i32
      %23 = arith.trunci %22 : i32 to i16
      %24 = arith.extui %23 : i16 to i32
      %25 = arith.shrsi %24, %c2_i32 : i32
      %26 = arith.trunci %25 : i32 to i16
      %27 = arith.extui %in_0 : i8 to i32
      %28 = arith.shli %27, %c8_i32 : i32
      %29 = arith.trunci %28 : i32 to i16
      %30 = arith.extui %29 : i16 to i32
      %31 = arith.shrsi %30, %c2_i32 : i32
      %32 = arith.trunci %31 : i32 to i16
      %33 = arith.extui %in_1 : i8 to i32
      %34 = arith.shli %33, %c8_i32 : i32
      %35 = arith.trunci %34 : i32 to i16
      %36 = arith.extui %35 : i16 to i32
      %37 = arith.shrsi %36, %c1_i32 : i32
      %38 = arith.trunci %37 : i32 to i16
      %39 = arith.extui %26 : i16 to i32
      %40 = arith.extui %32 : i16 to i32
      %41 = arith.addi %39, %40 : i32
      %42 = arith.trunci %41 : i32 to i16
      %43 = arith.extui %42 : i16 to i32
      %44 = arith.extui %38 : i16 to i32
      %45 = arith.addi %43, %44 : i32
      %46 = arith.trunci %45 : i32 to i16
      linalg.yield %46 : i16
    } -> tensor<?x?xi16>
    %11 = polygeist.submapInverse(%4, %10, %3, %5) {map = #map3} : (tensor<?x?xi16>, tensor<?x?xi16>, index, index) -> tensor<?x?xi16>
    %12 = affine.apply #map()[%2]
    %13 = affine.apply #map()[%3]
    %14 = polygeist.submap(%11, %13, %12) {map = #map3} : (tensor<?x?xi16>, index, index) -> tensor<?x?xi16>
    %15 = polygeist.submap(%11, %13, %12) {map = #map4} : (tensor<?x?xi16>, index, index) -> tensor<?x?xi16>
    %16 = polygeist.submap(%11, %13, %12) {map = #map5} : (tensor<?x?xi16>, index, index) -> tensor<?x?xi16>
    %17 = polygeist.submap(%0, %13, %12) {map = #map5} : (tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %18 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%14, %15, %16 : tensor<?x?xi16>, tensor<?x?xi16>, tensor<?x?xi16>) outs(%17 : tensor<?x?xi8>) {
    ^bb0(%in: i16, %in_0: i16, %in_1: i16, %out: i8):
      %21 = arith.extui %in : i16 to i32
      %22 = arith.extui %in_0 : i16 to i32
      %23 = arith.addi %21, %22 : i32
      %24 = arith.extui %in_1 : i16 to i32
      %25 = arith.shli %24, %c1_i32 : i32
      %26 = arith.addi %23, %25 : i32
      %27 = arith.addi %26, %c512_i32 : i32
      %28 = arith.shrsi %27, %c10_i32 : i32
      %29 = arith.trunci %28 : i32 to i8
      linalg.yield %29 : i8
    } -> tensor<?x?xi8>
    %19 = polygeist.submapInverse(%0, %18, %13, %12) {map = #map5} : (tensor<?x?xi8>, tensor<?x?xi8>, index, index) -> tensor<?x?xi8>
    %20 = bufferization.to_memref %19 : memref<?x?xi8>
    memref.copy %20, %arg3 : memref<?x?xi8> to memref<?x?xi8>
    return
  }
}

