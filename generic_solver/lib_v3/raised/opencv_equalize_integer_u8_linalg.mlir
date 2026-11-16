#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_equalize_integer_u8(%arg0: i32, %arg1: i32, %arg2: memref<?xi8>, %arg3: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c256 = arith.constant 256 : index
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<256xi32>
    %alloca_0 = memref.alloca() : memref<256xi32>
    %0 = polygeist.submap(%alloca_0, %c256) {map = #map} : (memref<256xi32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : memref<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
    }
    %1 = arith.muli %arg0, %arg1 : i32
    %2 = arith.index_cast %1 : i32 to index
    affine.for %arg4 = 0 to %2 {
      %11 = affine.load %arg2[%arg4] : memref<?xi8>
      %12 = arith.index_castui %11 : i8 to index
      %13 = memref.load %alloca_0[%12] : memref<256xi32>
      %14 = arith.addi %13, %c1_i32 : i32
      memref.store %14, %alloca_0[%12] : memref<256xi32>
    }
    %3 = affine.load %alloca_0[0] : memref<256xi32>
    affine.store %3, %alloca[0] : memref<256xi32>
    affine.for %arg4 = 0 to 255 {
      %11 = affine.load %alloca[%arg4] : memref<256xi32>
      %12 = affine.load %alloca_0[%arg4 + 1] : memref<256xi32>
      %13 = arith.addi %11, %12 : i32
      affine.store %13, %alloca[%arg4 + 1] : memref<256xi32>
    }
    %alloca_1 = memref.alloca() : memref<i32>
    affine.store %c0_i32, %alloca_1[] : memref<i32>
    %4 = polygeist.submap(%alloca, %c256) {map = #map} : (memref<256xi32>, index) -> memref<?xi32>
    %5 = polygeist.submap(%alloca_1, %c256) {map = #map1} : (memref<i32>, index) -> memref<?xi32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["reduction"]} ins(%4 : memref<?xi32>) outs(%5 : memref<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %11 = arith.cmpi eq, %out, %c0_i32 : i32
      %12 = arith.cmpi ne, %in, %c0_i32 : i32
      %13 = arith.select %12, %in, %out : i32
      %14 = arith.select %11, %13, %out : i32
      linalg.yield %14 : i32
    }
    %6 = affine.load %alloca_1[] : memref<i32>
    %7 = arith.cmpi sgt, %1, %6 : i32
    %8 = arith.subi %1, %6 : i32
    %9 = arith.select %7, %8, %c1_i32 : i32
    %10 = polygeist.submap(%arg3, %2) {map = #map} : (memref<?xi8>, index) -> memref<?xi8>
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%10 : memref<?xi8>) {
    ^bb0(%out: i8):
      %11 = linalg.index 0 : index
      %12 = memref.load %arg2[%11] : memref<?xi8>
      %13 = arith.index_castui %12 : i8 to index
      %14 = memref.load %alloca[%13] : memref<256xi32>
      %15 = arith.subi %14, %6 : i32
      %16 = arith.muli %15, %c255_i32 : i32
      %17 = arith.divsi %16, %9 : i32
      %18 = arith.cmpi slt, %17, %c0_i32 : i32
      %19 = arith.cmpi sgt, %17, %c255_i32 : i32
      %20 = arith.select %19, %c255_i32, %17 : i32
      %21 = arith.select %18, %c0_i32, %20 : i32
      %22 = arith.trunci %21 : i32 to i8
      linalg.yield %22 : i8
    }
    return
  }
}

