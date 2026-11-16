module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_u_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c-1_i64 = arith.constant -1 : i64
    %c0_i8 = arith.constant 0 : i8
    %c8_i64 = arith.constant 8 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_0, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.andi %6, %9 : i1
      %12 = arith.andi %6, %10 : i1
      %13 = arith.cmpf ogt, %2, %cst_0 : f32
      %14 = arith.remsi %4, %c2_i64 : i64
      %15 = arith.cmpi eq, %14, %c0_i64 : i64
      %16 = arith.select %13, %true, %15 : i1
      %17 = arith.addi %4, %c1_i64 : i64
      %18 = arith.sitofp %17 : i64 to f32
      %19 = arith.subi %c-1_i64, %4 : i64
      %20 = arith.sitofp %19 : i64 to f32
      %21 = arith.select %16, %18, %20 : f32
      %22 = arith.select %11, %21, %7 : f32
      %23 = arith.cmpi sgt, %4, %c8_i64 : i64
      %24 = arith.andi %12, %23 : i1
      %25 = arith.cmpf olt, %8, %cst : f32
      %26 = arith.xori %25, %true : i1
      %27 = arith.andi %26, %12 : i1
      %28 = arith.andi %25, %12 : i1
      %29 = arith.andi %24, %28 : i1
      %30 = arith.xori %24, %true : i1
      %31 = arith.andi %24, %27 : i1
      %32 = arith.andi %30, %12 : i1
      %33 = arith.ori %31, %32 : i1
      %34 = scf.if %29 -> (f32) {
        %45 = func.call @acosf(%2) : (f32) -> f32
        %46 = math.sin %45 : f32
        %47 = arith.cmpf une, %46, %cst_0 : f32
        %48 = scf.if %47 -> (f32) {
          %49 = arith.addi %4, %c1_i64 : i64
          %50 = arith.sitofp %49 : i64 to f32
          %51 = func.call @acosf(%2) : (f32) -> f32
          %52 = arith.mulf %50, %51 : f32
          %53 = math.sin %52 : f32
          %54 = func.call @acosf(%2) : (f32) -> f32
          %55 = math.sin %54 : f32
          %56 = arith.divf %53, %55 : f32
          scf.yield %56 : f32
        } else {
          %49 = arith.addi %4, %c1_i64 : i64
          %50 = arith.sitofp %49 : i64 to f32
          %51 = func.call @acosf(%2) : (f32) -> f32
          %52 = arith.mulf %50, %51 : f32
          %53 = math.cos %52 : f32
          %54 = arith.mulf %50, %53 : f32
          %55 = arith.divf %54, %2 : f32
          scf.yield %55 : f32
        }
        scf.yield %48 : f32
      } else {
        scf.yield %22 : f32
      }
      %35 = arith.cmpi eq, %4, %c0_i64 : i64
      %36 = arith.cmpi ne, %4, %c0_i64 : i64
      %37 = arith.select %35, %cst, %34 : f32
      %38 = arith.select %33, %36, %false : i1
      %39 = arith.select %33, %37, %34 : f32
      %40 = arith.cmpi eq, %4, %c1_i64 : i64
      %41 = arith.andi %38, %40 : i1
      %42 = arith.xori %41, %true : i1
      %43 = arith.andi %42, %38 : i1
      %44 = scf.if %43 -> (f32) {
        %45 = arith.addf %2, %2 : f32
        %46:4 = scf.while (%arg5 = %c2_i64, %arg6 = %45, %arg7 = %45, %arg8 = %cst) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %47 = arith.cmpi sle, %arg5, %4 : i64
          %48 = arith.cmpf uno, %arg7, %arg7 : f32
          %49 = arith.extui %48 : i1 to i8
          %50 = arith.cmpi eq, %49, %c0_i8 : i8
          %51 = arith.andi %47, %50 : i1
          scf.condition(%51) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %47 = arith.cmpf uno, %arg7, %arg7 : f32
          %48 = arith.extui %47 : i1 to i8
          %49 = arith.cmpi eq, %48, %c0_i8 : i8
          %50 = arith.select %49, %arg7, %arg8 : f32
          %51 = arith.cmpi sle, %arg6, %4 : i64
          %52 = arith.select %51, %50, %arg8 : f32
          %53 = arith.mulf %45, %arg7 : f32
          %54 = arith.subf %53, %arg8 : f32
          %55 = arith.addi %arg6, %c1_i64 : i64
          scf.yield %55, %54, %54, %52 : i64, f32, f32, f32
        }
        scf.yield %46#0 : f32
      } else {
        %45 = arith.addf %2, %2 : f32
        %46 = arith.select %41, %45, %39 : f32
        scf.yield %46 : f32
      }
      affine.store %44, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

