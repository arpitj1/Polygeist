module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_u_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c0_i8 = arith.constant 0 : i8
    %c-1_i64 = arith.constant -1 : i64
    %true = arith.constant true
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_1 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = arith.fptosi %extracted_1 : f32 to i64
      %8 = arith.cmpi slt, %7, %c0_i64 : i64
      %9 = arith.cmpi sge, %7, %c0_i64 : i64
      %10 = arith.select %8, %cst, %4 : f32
      %11 = math.absf %extracted : f32
      %12 = arith.cmpf oeq, %11, %cst_0 : f32
      %13 = arith.xori %12, %true : i1
      %14 = arith.andi %9, %12 : i1
      %15 = arith.andi %9, %13 : i1
      %16 = arith.cmpf ogt, %extracted, %cst : f32
      %17 = arith.remsi %7, %c2_i64 : i64
      %18 = arith.cmpi eq, %17, %c0_i64 : i64
      %19 = arith.xori %16, %true : i1
      %20 = arith.andi %19, %18 : i1
      %21 = arith.ori %16, %20 : i1
      %22 = arith.addi %7, %c1_i64 : i64
      %23 = arith.sitofp %22 : i64 to f32
      %24 = arith.subi %c-1_i64, %7 : i64
      %25 = arith.sitofp %24 : i64 to f32
      %26 = arith.select %21, %23, %25 : f32
      %27 = arith.select %14, %26, %10 : f32
      %28 = arith.cmpi sgt, %7, %c8_i64 : i64
      %29 = arith.andi %15, %28 : i1
      %30 = arith.cmpf olt, %11, %cst_0 : f32
      %31 = arith.xori %30, %true : i1
      %32 = arith.andi %31, %15 : i1
      %33 = arith.andi %30, %15 : i1
      %34 = arith.andi %29, %33 : i1
      %35 = arith.xori %29, %true : i1
      %36 = arith.andi %29, %32 : i1
      %37 = arith.andi %35, %15 : i1
      %38 = arith.ori %36, %37 : i1
      %39 = scf.if %34 -> (f32) {
        %50 = func.call @acosf(%extracted) : (f32) -> f32
        %51 = math.sin %50 : f32
        %52 = arith.cmpf une, %51, %cst : f32
        %53 = scf.if %52 -> (f32) {
          %54 = arith.addi %7, %c1_i64 : i64
          %55 = arith.sitofp %54 : i64 to f32
          %56 = func.call @acosf(%extracted) : (f32) -> f32
          %57 = arith.mulf %55, %56 : f32
          %58 = math.sin %57 : f32
          %59 = func.call @acosf(%extracted) : (f32) -> f32
          %60 = math.sin %59 : f32
          %61 = arith.divf %58, %60 : f32
          scf.yield %61 : f32
        } else {
          %54 = arith.addi %7, %c1_i64 : i64
          %55 = arith.sitofp %54 : i64 to f32
          %56 = func.call @acosf(%extracted) : (f32) -> f32
          %57 = arith.mulf %55, %56 : f32
          %58 = math.cos %57 : f32
          %59 = arith.mulf %55, %58 : f32
          %60 = arith.divf %59, %extracted : f32
          scf.yield %60 : f32
        }
        scf.yield %53 : f32
      } else {
        scf.yield %27 : f32
      }
      %40 = arith.cmpi eq, %7, %c0_i64 : i64
      %41 = arith.cmpi ne, %7, %c0_i64 : i64
      %42 = arith.select %40, %cst_0, %39 : f32
      %43 = arith.andi %38, %41 : i1
      %44 = arith.select %38, %42, %39 : f32
      %45 = arith.cmpi eq, %7, %c1_i64 : i64
      %46 = arith.andi %43, %45 : i1
      %47 = arith.xori %46, %true : i1
      %48 = arith.andi %47, %43 : i1
      %49 = scf.if %48 -> (f32) {
        %50 = arith.addf %extracted, %extracted : f32
        %51:4 = scf.while (%arg6 = %c2_i64, %arg7 = %50, %arg8 = %50, %arg9 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %52 = arith.cmpi sle, %arg6, %7 : i64
          %53 = arith.cmpf uno, %arg8, %arg8 : f32
          %54 = arith.extui %53 : i1 to i8
          %55 = arith.cmpi eq, %54, %c0_i8 : i8
          %56 = arith.andi %52, %55 : i1
          scf.condition(%56) %arg7, %arg6, %arg8, %arg9 : f32, i64, f32, f32
        } do {
        ^bb0(%arg6: f32, %arg7: i64, %arg8: f32, %arg9: f32):
          %52 = arith.cmpf uno, %arg8, %arg8 : f32
          %53 = arith.extui %52 : i1 to i8
          %54 = arith.cmpi eq, %53, %c0_i8 : i8
          %55 = arith.select %54, %arg8, %arg9 : f32
          %56 = arith.cmpi sle, %arg7, %7 : i64
          %57 = arith.select %56, %55, %arg9 : f32
          %58 = arith.mulf %50, %arg8 : f32
          %59 = arith.subf %58, %arg9 : f32
          %60 = arith.addi %arg7, %c1_i64 : i64
          scf.yield %60, %59, %59, %57 : i64, f32, f32, f32
        }
        scf.yield %51#0 : f32
      } else {
        %50 = arith.addf %extracted, %extracted : f32
        %51 = arith.select %46, %50, %44 : f32
        scf.yield %51 : f32
      }
      %inserted = tensor.insert %49 into %arg5[%arg4] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @acosf(f32) -> f32
}

