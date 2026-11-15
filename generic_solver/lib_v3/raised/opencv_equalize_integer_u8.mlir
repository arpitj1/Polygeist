module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @opencv_equalize_integer_u8(%arg0: i32, %arg1: i32, %arg2: memref<?xi8>, %arg3: memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c255_i32 = arith.constant 255 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<256xi32>
    %alloca_0 = memref.alloca() : memref<256xi32>
    affine.for %arg4 = 0 to 256 {
      affine.store %c0_i32, %alloca_0[%arg4] : memref<256xi32>
    }
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.index_cast %0 : i32 to index
    affine.for %arg4 = 0 to %1 {
      %6 = affine.load %arg2[%arg4] : memref<?xi8>
      %7 = arith.index_castui %6 : i8 to index
      %8 = memref.load %alloca_0[%7] : memref<256xi32>
      %9 = arith.addi %8, %c1_i32 : i32
      memref.store %9, %alloca_0[%7] : memref<256xi32>
    }
    %2 = affine.load %alloca_0[0] : memref<256xi32>
    affine.store %2, %alloca[0] : memref<256xi32>
    affine.for %arg4 = 1 to 256 {
      %6 = affine.load %alloca[%arg4 - 1] : memref<256xi32>
      %7 = affine.load %alloca_0[%arg4] : memref<256xi32>
      %8 = arith.addi %6, %7 : i32
      affine.store %8, %alloca[%arg4] : memref<256xi32>
    }
    %3 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %c0_i32) -> (i32) {
      %6 = arith.cmpi eq, %arg5, %c0_i32 : i32
      %7 = scf.if %6 -> (i32) {
        %8 = affine.load %alloca[%arg4] : memref<256xi32>
        %9 = arith.cmpi ne, %8, %c0_i32 : i32
        %10 = arith.select %9, %8, %arg5 : i32
        scf.yield %10 : i32
      } else {
        scf.yield %arg5 : i32
      }
      affine.yield %7 : i32
    }
    %4 = arith.cmpi sgt, %0, %3 : i32
    %5 = scf.if %4 -> (i32) {
      %6 = arith.subi %0, %3 : i32
      scf.yield %6 : i32
    } else {
      scf.yield %c1_i32 : i32
    }
    affine.for %arg4 = 0 to %1 {
      %6 = affine.load %arg2[%arg4] : memref<?xi8>
      %7 = arith.index_castui %6 : i8 to index
      %8 = memref.load %alloca[%7] : memref<256xi32>
      %9 = arith.subi %8, %3 : i32
      %10 = arith.muli %9, %c255_i32 : i32
      %11 = arith.divsi %10, %5 : i32
      %12 = arith.cmpi slt, %11, %c0_i32 : i32
      %13 = scf.if %12 -> (i32) {
        scf.yield %c0_i32 : i32
      } else {
        %15 = arith.cmpi sgt, %11, %c255_i32 : i32
        %16 = arith.select %15, %c255_i32, %11 : i32
        scf.yield %16 : i32
      }
      %14 = arith.trunci %13 : i32 to i8
      affine.store %14, %arg3[%arg4] : memref<?xi8>
    }
    return
  }
}
