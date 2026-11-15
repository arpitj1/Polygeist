#map = affine_map<()[s0] -> (s0 - 2)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1 + 2)>
#map3 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map4 = affine_map<(d0, d1) -> (d0 + 2, d1 + 1)>
#map5 = affine_map<(d0, d1) -> (d0 + 1, d1 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_gaussian3x3_u8(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?x?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c512_i32 = arith.constant 512 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca(%1, %0) : memref<?x?xi16>
    %2 = affine.apply #map()[%0]
    %3 = polygeist.submap(%arg2, %1, %2) {map = #map1} : (memref<?x?xi8>, index, index) -> memref<?x?xi8>
    %4 = polygeist.submap(%arg2, %1, %2) {map = #map2} : (memref<?x?xi8>, index, index) -> memref<?x?xi8>
    %5 = polygeist.submap(%arg2, %1, %2) {map = #map3} : (memref<?x?xi8>, index, index) -> memref<?x?xi8>
    %6 = polygeist.submap(%alloca, %1, %2) {map = #map3} : (memref<?x?xi16>, index, index) -> memref<?x?xi16>
    linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3, %4, %5 : memref<?x?xi8>, memref<?x?xi8>, memref<?x?xi8>) outs(%6 : memref<?x?xi16>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i16):
      %13 = arith.extui %in : i8 to i32
      %14 = arith.shli %13, %c8_i32 : i32
      %15 = arith.trunci %14 : i32 to i16
      %16 = arith.extui %15 : i16 to i32
      %17 = arith.shrsi %16, %c2_i32 : i32
      %18 = arith.trunci %17 : i32 to i16
      %19 = arith.extui %in_0 : i8 to i32
      %20 = arith.shli %19, %c8_i32 : i32
      %21 = arith.trunci %20 : i32 to i16
      %22 = arith.extui %21 : i16 to i32
      %23 = arith.shrsi %22, %c2_i32 : i32
      %24 = arith.trunci %23 : i32 to i16
      %25 = arith.extui %in_1 : i8 to i32
      %26 = arith.shli %25, %c8_i32 : i32
      %27 = arith.trunci %26 : i32 to i16
      %28 = arith.extui %27 : i16 to i32
      %29 = arith.shrsi %28, %c1_i32 : i32
      %30 = arith.trunci %29 : i32 to i16
      %31 = arith.extui %18 : i16 to i32
      %32 = arith.extui %24 : i16 to i32
      %33 = arith.addi %31, %32 : i32
      %34 = arith.trunci %33 : i32 to i16
      %35 = arith.extui %34 : i16 to i32
      %36 = arith.extui %30 : i16 to i32
      %37 = arith.addi %35, %36 : i32
      %38 = arith.trunci %37 : i32 to i16
      linalg.yield %38 : i16
    }
    %7 = affine.apply #map()[%0]
    %8 = affine.apply #map()[%1]
    %9 = polygeist.submap(%alloca, %8, %7) {map = #map3} : (memref<?x?xi16>, index, index) -> memref<?x?xi16>
    %10 = polygeist.submap(%alloca, %8, %7) {map = #map4} : (memref<?x?xi16>, index, index) -> memref<?x?xi16>
    %11 = polygeist.submap(%alloca, %8, %7) {map = #map5} : (memref<?x?xi16>, index, index) -> memref<?x?xi16>
    %12 = polygeist.submap(%arg3, %8, %7) {map = #map5} : (memref<?x?xi8>, index, index) -> memref<?x?xi8>
    linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%9, %10, %11 : memref<?x?xi16>, memref<?x?xi16>, memref<?x?xi16>) outs(%12 : memref<?x?xi8>) {
    ^bb0(%in: i16, %in_0: i16, %in_1: i16, %out: i8):
      %13 = arith.extui %in : i16 to i32
      %14 = arith.extui %in_0 : i16 to i32
      %15 = arith.addi %13, %14 : i32
      %16 = arith.extui %in_1 : i16 to i32
      %17 = arith.shli %16, %c1_i32 : i32
      %18 = arith.addi %15, %17 : i32
      %19 = arith.addi %18, %c512_i32 : i32
      %20 = arith.shrsi %19, %c10_i32 : i32
      %21 = arith.trunci %20 : i32 to i8
      linalg.yield %21 : i8
    }
    return
  }
}

