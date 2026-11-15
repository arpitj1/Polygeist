module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_he_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant 0x7FC00000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.index_cast %4 : i64 to index
      %6 = arith.cmpi slt, %4, %c0_i64 : i64
      %7 = arith.cmpi sge, %4, %c0_i64 : i64
      %8 = arith.select %6, %cst_0, %1 : f32
      %9:2 = scf.if %7 -> (i1, f32) {
        %16 = arith.cmpi eq, %4, %c0_i64 : i64
        %17 = arith.cmpi ne, %4, %c0_i64 : i64
        %18 = arith.select %16, %cst_1, %8 : f32
        scf.yield %17, %18 : i1, f32
      } else {
        scf.yield %false, %8 : i1, f32
      }
      %10 = arith.cmpi eq, %4, %c1_i64 : i64
      %11 = arith.cmpi ne, %4, %c1_i64 : i64
      %12 = arith.select %10, %2, %9#1 : f32
      %13 = arith.cmpi sgt, %4, %c128_i64 : i64
      %14 = arith.cmpi sle, %4, %c128_i64 : i64
      %15 = scf.if %9#0 -> (f32) {
        %16 = scf.if %11 -> (f32) {
          %17 = scf.if %14 -> (f32) {
            %18:3 = scf.for %arg5 = %c1 to %5 step %c1 iter_args(%arg6 = %2, %arg7 = %cst_1, %arg8 = %1) -> (f32, f32, f32) {
              %19 = arith.index_cast %arg5 : index to i64
              %20 = arith.mulf %2, %arg6 : f32
              %21 = arith.sitofp %19 : i64 to f32
              %22 = arith.mulf %21, %arg7 : f32
              %23 = arith.subf %20, %22 : f32
              scf.yield %23, %arg6, %23 : f32, f32, f32
            }
            scf.yield %18#2 : f32
          } else {
            %18 = arith.select %13, %cst, %12 : f32
            scf.yield %18 : f32
          }
          scf.yield %17 : f32
        } else {
          scf.yield %12 : f32
        }
        scf.yield %16 : f32
      } else {
        scf.yield %9#1 : f32
      }
      affine.store %15, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
}

