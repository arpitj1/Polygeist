#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cutlass_gemm_i8_i32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xi8>, %arg4: memref<?x?xi8>, %arg5: memref<?x?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.index_cast %arg2 : i32 to index
    %3 = polygeist.submap(%arg5, %0, %1) {map = #map} : (memref<?x?xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%3 : memref<?x?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %4 = polygeist.submap(%arg3, %0, %1, %2) {map = #map1} : (memref<?x?xi8>, index, index, index) -> memref<?x?x?xi8>
    %5 = polygeist.submap(%arg4, %0, %1, %2) {map = #map2} : (memref<?x?xi8>, index, index, index) -> memref<?x?x?xi8>
    %6 = polygeist.submap(%arg5, %0, %1, %2) {map = #map3} : (memref<?x?xi32>, index, index, index) -> memref<?x?x?xi32>
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %5 : memref<?x?x?xi8>, memref<?x?x?xi8>) outs(%6 : memref<?x?x?xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %7 = arith.extsi %in : i8 to i32
      %8 = arith.extsi %in_0 : i8 to i32
      %9 = arith.muli %7, %8 : i32
      %10 = arith.addi %out, %9 : i32
      linalg.yield %10 : i32
    }
    return
  }
}

