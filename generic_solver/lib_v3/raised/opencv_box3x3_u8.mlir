#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_box3x3_u8(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?x?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4_i32 = arith.constant 4 : i32
    %c932068_i32 = arith.constant 932068 : i32
    %c23_i32 = arith.constant 23 : i32
    %0 = arith.index_cast %arg1 : i32 to index
    %1 = arith.index_cast %arg0 : i32 to index
    %alloca = memref.alloca(%1, %0) : memref<?x?xi16>
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 1 to #map()[%0] {
        %2 = affine.load %arg2[%arg4, %arg5 - 1] : memref<?x?xi8>
        %3 = arith.extui %2 : i8 to i32
        %4 = affine.load %arg2[%arg4, %arg5] : memref<?x?xi8>
        %5 = arith.extui %4 : i8 to i32
        %6 = arith.addi %3, %5 : i32
        %7 = affine.load %arg2[%arg4, %arg5 + 1] : memref<?x?xi8>
        %8 = arith.extui %7 : i8 to i32
        %9 = arith.addi %6, %8 : i32
        %10 = arith.trunci %9 : i32 to i16
        affine.store %10, %alloca[%arg4, %arg5] : memref<?x?xi16>
      }
    }
    affine.for %arg4 = 1 to #map()[%1] {
      affine.for %arg5 = 1 to #map()[%0] {
        %2 = affine.load %alloca[%arg4 - 1, %arg5] : memref<?x?xi16>
        %3 = arith.extui %2 : i16 to i32
        %4 = affine.load %alloca[%arg4, %arg5] : memref<?x?xi16>
        %5 = arith.extui %4 : i16 to i32
        %6 = arith.addi %3, %5 : i32
        %7 = affine.load %alloca[%arg4 + 1, %arg5] : memref<?x?xi16>
        %8 = arith.extui %7 : i16 to i32
        %9 = arith.addi %6, %8 : i32
        %10 = arith.addi %9, %c4_i32 : i32
        %11 = arith.muli %10, %c932068_i32 : i32
        %12 = arith.shrsi %11, %c23_i32 : i32
        %13 = arith.trunci %12 : i32 to i8
        affine.store %13, %arg3[%arg4, %arg5] : memref<?x?xi8>
      }
    }
    return
  }
}
