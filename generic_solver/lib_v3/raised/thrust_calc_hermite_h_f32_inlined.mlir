module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_h_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0x7F800000 : f32
    %c-1_i64 = arith.constant -1 : i64
    %c128_i64 = arith.constant 128 : i64
    %cst_0 = arith.constant 0x7FC00000 : f32
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %true = arith.constant true
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = math.absf %3 : f32
      %5 = arith.cmpf oeq, %4, %cst : f32
      %6:8 = scf.if %5 -> (i64, i1, i1, i1, i1, i1, i1, i1) {
        scf.yield %c-1_i64, %true, %false, %false, %true, %false, %false, %true : i64, i1, i1, i1, i1, i1, i1, i1
      } else {
        %16 = arith.cmpf uno, %3, %3 : f32
        %17:8 = scf.if %16 -> (i64, i1, i1, i1, i1, i1, i1, i1) {
          scf.yield %c-1_i64, %true, %false, %false, %true, %false, %false, %true : i64, i1, i1, i1, i1, i1, i1, i1
        } else {
          %18 = arith.fptosi %3 : f32 to i64
          %19 = arith.cmpi slt, %18, %c0_i64 : i64
          %20 = arith.cmpi sge, %18, %c0_i64 : i64
          %21 = arith.cmpi eq, %18, %c0_i64 : i64
          %22 = arith.cmpi ne, %18, %c0_i64 : i64
          %23 = arith.cmpi eq, %18, %c1_i64 : i64
          %24 = arith.cmpi sgt, %18, %c128_i64 : i64
          %25 = arith.cmpi sle, %18, %c128_i64 : i64
          scf.yield %18, %19, %20, %21, %22, %23, %24, %25 : i64, i1, i1, i1, i1, i1, i1, i1
        }
        scf.yield %17#0, %17#1, %17#2, %17#3, %17#4, %17#5, %17#6, %17#7 : i64, i1, i1, i1, i1, i1, i1, i1
      }
      %7 = arith.index_cast %6#0 : i64 to index
      %8 = arith.select %6#1, %cst_1, %1 : f32
      %9 = arith.andi %6#2, %6#4 : i1
      %10 = scf.if %6#2 -> (f32) {
        %16 = arith.select %6#3, %cst_2, %8 : f32
        scf.yield %16 : f32
      } else {
        scf.yield %8 : f32
      }
      %11 = arith.andi %9, %6#5 : i1
      %12 = arith.xori %11, %true : i1
      %13 = arith.andi %12, %9 : i1
      %14 = scf.if %11 -> (f32) {
        %16 = arith.addf %2, %2 : f32
        scf.yield %16 : f32
      } else {
        scf.yield %10 : f32
      }
      %15 = scf.if %13 -> (f32) {
        %16 = scf.if %6#7 -> (f32) {
          %17 = arith.addf %2, %2 : f32
          %18 = arith.muli %7, %c2 : index
          %19:3 = scf.for %arg5 = %c2 to %18 step %c2 iter_args(%arg6 = %17, %arg7 = %cst_2, %arg8 = %cst_1) -> (f32, f32, f32) {
            %20 = arith.index_cast %arg5 : index to i64
            %21 = arith.mulf %17, %arg6 : f32
            %22 = arith.sitofp %20 : i64 to f32
            %23 = arith.mulf %22, %arg7 : f32
            %24 = arith.subf %21, %23 : f32
            scf.yield %24, %arg6, %24 : f32, f32, f32
          }
          scf.yield %19#2 : f32
        } else {
          %17 = arith.select %6#6, %cst_0, %14 : f32
          scf.yield %17 : f32
        }
        scf.yield %16 : f32
      } else {
        scf.yield %14 : f32
      }
      affine.store %15, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
}

