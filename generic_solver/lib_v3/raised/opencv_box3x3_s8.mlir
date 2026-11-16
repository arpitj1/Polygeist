#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_box3x3_s8(%arg0: i32, %arg1: i32, %arg2: memref<?x?xi8>, %arg3: memref<?x?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c9_i32 = arith.constant 9 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%1] {
        %2 = affine.for %arg6 = -1 to 2 iter_args(%arg7 = %c0_i32) -> (i32) {
          %6 = affine.for %arg8 = -1 to 2 iter_args(%arg9 = %arg7) -> (i32) {
            %7 = affine.load %arg2[%arg4 + %arg6, %arg5 + %arg8] : memref<?x?xi8>
            %8 = arith.extsi %7 : i8 to i32
            %9 = arith.addi %arg9, %8 : i32
            affine.yield %9 : i32
          }
          affine.yield %6 : i32
        }
        %3 = arith.addi %2, %c4_i32 : i32
        %4 = arith.divsi %3, %c9_i32 : i32
        %5 = arith.trunci %4 : i32 to i8
        affine.store %5, %arg3[%arg4, %arg5] : memref<?x?xi8>
      }
    }
    return
  }
}
