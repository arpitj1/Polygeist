#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_segmented_logical_mode_i32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xi32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.cmpi ne, %arg2, %c0_i32 : i32
    %3 = polygeist.submap(%arg4, %0) {map = #map} : (memref<?xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%3 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %arg2 : i32
    }
    %4 = polygeist.submap(%arg3, %0, %1) {map = #map1} : (memref<?x?xi32>, index, index) -> memref<?x?xi32>
    %5 = polygeist.submap(%arg4, %0, %1) {map = #map2} : (memref<?xi32>, index, index) -> memref<?x?xi32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%4 : memref<?x?xi32>) outs(%5 : memref<?x?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %6 = arith.cmpi ne, %out, %c0_i32 : i32
      %7 = arith.cmpi ne, %in, %c0_i32 : i32
      %8 = arith.select %6, %7, %false : i1
      %9 = arith.extui %8 : i1 to i32
      %10 = arith.cmpi ne, %out, %c0_i32 : i32
      %11 = arith.cmpi ne, %in, %c0_i32 : i32
      %12 = arith.select %10, %true, %11 : i1
      %13 = arith.extui %12 : i1 to i32
      %14 = arith.select %2, %9, %13 : i32
      linalg.yield %14 : i32
    }
    return
  }
}

