#map = affine_map<()[s0] -> (s0 - 2)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1, d2)[s0] -> (d1 + s0, d2 + d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map5 = affine_map<(d0)[s0] -> (s0 + 1, d0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_box3x3_u16(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi16>, %arg3: memref<?x?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c9_i32 = arith.constant 9 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = affine.apply #map()[%1]
    %3 = affine.apply #map()[%1]
    %4 = affine.apply #map()[%1]
    %5 = affine.apply #map()[%1]
    affine.for %arg4 = 0 to #map()[%0] {
      %alloca = memref.alloca(%2) : memref<?xi32>
      %6 = polygeist.submap(%alloca, %3) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
      linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel"]} outs(%6 : memref<?xi32>) {
      ^bb0(%out: i32):
        linalg.yield %c0_i32 : i32
      }
      %7 = polygeist.submap(%arg2, %arg4, %4, %c3, %c3) {map = #map2} : (memref<?x?xi16>, index, index, index, index) -> memref<?x?x?xi16>
      %8 = polygeist.submap(%alloca, %4, %c3, %c3) {map = #map3} : (memref<?xi32>, index, index, index) -> memref<?x?x?xi32>
      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "reduction", "reduction"]} ins(%7 : memref<?x?x?xi16>) outs(%8 : memref<?x?x?xi32>) {
      ^bb0(%in: i16, %out: i32):
        %11 = arith.extui %in : i16 to i32
        %12 = arith.addi %out, %11 : i32
        linalg.yield %12 : i32
      }
      %9 = polygeist.submap(%alloca, %5) {map = #map1} : (memref<?xi32>, index) -> memref<?xi32>
      %10 = polygeist.submap(%arg3, %arg4, %5) {map = #map5} : (memref<?x?xi16>, index, index) -> memref<?xi16>
      linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%9 : memref<?xi32>) outs(%10 : memref<?xi16>) {
      ^bb0(%in: i32, %out: i16):
        %11 = arith.addi %in, %c4_i32 : i32
        %12 = arith.divsi %11, %c9_i32 : i32
        %13 = arith.trunci %12 : i32 to i16
        linalg.yield %13 : i16
      }
    } {polygeist.was_parallel}
    return
  }
}

