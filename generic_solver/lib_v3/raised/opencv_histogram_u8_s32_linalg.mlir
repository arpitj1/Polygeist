#map = affine_map<(d0) -> (d0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_histogram_u8_s32(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c256 = arith.constant 256 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<256xi32>
    %0 = polygeist.submap(%alloca, %c256) {map = #map} : (memref<256xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %1 = arith.index_cast %arg0 : i32 to index
    %2 = arith.index_cast %arg1 : i32 to index
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 0 to %2 {
        %5 = affine.load %arg2[%arg4, %arg5] : memref<?x?xi8>
        %6 = arith.extui %5 : i8 to i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %alloca[%7] : memref<256xi32>
        %9 = arith.addi %8, %c1_i32 : i32
        memref.store %9, %alloca[%7] : memref<256xi32>
      }
    }
    %3 = polygeist.submap(%alloca, %c256) {map = #map} : (memref<256xi32>, index) -> memref<?xi32>
    %4 = polygeist.submap(%arg3, %c256) {map = #map} : (memref<?xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%3 : memref<?xi32>) outs(%4 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    return
  }
}

