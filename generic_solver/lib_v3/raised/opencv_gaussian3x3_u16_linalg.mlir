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
    %c3 = arith.constant 3 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %alloca = memref.alloca() : memref<3xi32>
    affine.store %c1_i32, %alloca[0] : memref<3xi32>
    affine.store %c2_i32, %alloca[1] : memref<3xi32>
    affine.store %c1_i32, %alloca[2] : memref<3xi32>
    %2 = affine.apply #map()[%1]
    %3 = affine.apply #map()[%1]
    %4 = affine.apply #map()[%1]
    %5 = affine.apply #map()[%1]
    affine.for %arg4 = 0 to #map()[%0] {
      %alloca_0 = memref.alloca(%2) : memref<?xi32>
      %6 = polygeist.submap(%alloca_0, %3) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : memref<?xi32>) {
      ^bb0(%out: i32):
        linalg.yield %c0_i32 : i32
      }
      %7 = polygeist.submap(%alloca, %4, %c3, %c3) {map = #map2} : (memref<3xi32>, index, index, index) -> memref<?x?x?xi32>
      %8 = polygeist.submap(%alloca, %4, %c3, %c3) {map = #map3} : (memref<3xi32>, index, index, index) -> memref<?x?x?xi32>
      %9 = polygeist.submap(%arg2, %arg4, %4, %c3, %c3) {map = #map4} : (memref<?x?xi16>, index, index, index, index) -> memref<?x?x?xi16>
      %10 = polygeist.submap(%alloca_0, %4, %c3, %c3) {map = #map5} : (memref<?xi32>, index, index, index) -> memref<?x?x?xi32>
      linalg.generic {indexing_maps = [#map6, #map6, #map6, #map6], iterator_types = ["parallel", "reduction", "reduction"]} ins(%7, %8, %9 : memref<?x?x?xi32>, memref<?x?x?xi32>, memref<?x?x?xi16>) outs(%10 : memref<?x?x?xi32>) {
      ^bb0(%in: i32, %in_1: i32, %in_2: i16, %out: i32):
        %13 = arith.muli %in, %in_1 : i32
        %14 = arith.extui %in_2 : i16 to i32
        %15 = arith.muli %13, %14 : i32
        %16 = arith.addi %out, %15 : i32
        linalg.yield %16 : i32
      }
      %11 = polygeist.submap(%alloca_0, %5) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
      %12 = polygeist.submap(%arg3, %arg4, %5) {map = #map7} : (memref<?x?xi16>, index, index) -> memref<?xi16>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%11 : memref<?xi32>) outs(%12 : memref<?xi16>) {
      ^bb0(%in: i32, %out: i16):
        %13 = arith.addi %in, %c8_i32 : i32
        %14 = arith.shrsi %13, %c4_i32 : i32
        %15 = arith.trunci %14 : i32 to i16
        linalg.yield %15 : i16
      }
    } {polygeist.was_parallel}
    return
  }
}

