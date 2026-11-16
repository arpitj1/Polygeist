module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cub_segmented_logical_mode_i32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x?xi32>, %arg4: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %2 = arith.cmpi ne, %arg2, %c0_i32 : i32
    affine.for %arg5 = 0 to %0 {
      %3 = affine.for %arg6 = 0 to %1 iter_args(%arg7 = %arg2) -> (i32) {
        %4 = scf.if %2 -> (i32) {
          %5 = arith.cmpi ne, %arg7, %c0_i32 : i32
          %6 = scf.if %5 -> (i1) {
            %8 = affine.load %arg3[%arg5, %arg6] : memref<?x?xi32>
            %9 = arith.cmpi ne, %8, %c0_i32 : i32
            scf.yield %9 : i1
          } else {
            scf.yield %false : i1
          }
          %7 = arith.extui %6 : i1 to i32
          scf.yield %7 : i32
        } else {
          %5 = arith.cmpi ne, %arg7, %c0_i32 : i32
          %6 = scf.if %5 -> (i1) {
            scf.yield %true : i1
          } else {
            %8 = affine.load %arg3[%arg5, %arg6] : memref<?x?xi32>
            %9 = arith.cmpi ne, %8, %c0_i32 : i32
            scf.yield %9 : i1
          }
          %7 = arith.extui %6 : i1 to i32
          scf.yield %7 : i32
        }
        affine.yield %4 : i32
      }
      affine.store %3, %arg4[%arg5] : memref<?xi32>
    }
    return
  }
}
