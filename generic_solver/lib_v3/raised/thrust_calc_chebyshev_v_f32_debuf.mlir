module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_v_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c0_i8 = arith.constant 0 : i8
    %c-1_i64 = arith.constant -1 : i64
    %false = arith.constant false
    %true = arith.constant true
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_3 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = arith.fptosi %extracted_3 : f32 to i64
      %8 = arith.cmpi slt, %7, %c0_i64 : i64
      %9 = arith.cmpi sge, %7, %c0_i64 : i64
      %10 = arith.select %8, %cst, %4 : f32
      %11 = math.absf %extracted : f32
      %12 = arith.cmpf oeq, %11, %cst_0 : f32
      %13 = arith.xori %12, %true : i1
      %14 = arith.cmpf ogt, %extracted, %cst : f32
      %15 = arith.xori %14, %true : i1
      %16 = arith.remsi %7, %c2_i64 : i64
      %17 = arith.cmpi eq, %16, %c0_i64 : i64
      %18 = arith.andi %15, %17 : i1
      %19 = arith.xori %18, %true : i1
      %20 = arith.andi %19, %15 : i1
      %21 = arith.andi %9, %12 : i1
      %22 = arith.andi %9, %13 : i1
      %23 = arith.addi %7, %7 : i64
      %24 = arith.subi %c-1_i64, %23 : i64
      %25 = arith.sitofp %24 : i64 to f32
      %26 = arith.addi %7, %7 : i64
      %27 = arith.addi %26, %c1_i64 : i64
      %28 = arith.sitofp %27 : i64 to f32
      %29 = arith.select %18, %28, %cst_0 : f32
      %30 = arith.select %20, %25, %29 : f32
      %31 = arith.select %21, %30, %10 : f32
      %32 = arith.cmpi sgt, %7, %c8_i64 : i64
      %33 = arith.andi %22, %32 : i1
      %34 = arith.cmpf olt, %11, %cst_0 : f32
      %35 = arith.xori %34, %true : i1
      %36 = arith.andi %35, %22 : i1
      %37 = arith.andi %33, %34 : i1
      %38 = arith.xori %33, %true : i1
      %39 = arith.andi %33, %36 : i1
      %40 = arith.andi %38, %22 : i1
      %41 = arith.ori %39, %40 : i1
      %42 = scf.if %37 -> (f32) {
        %53:2 = scf.if %22 -> (i1, f32) {
          %65 = func.call @acosf(%extracted) : (f32) -> f32
          %66 = arith.divf %65, %cst_1 : f32
          %67 = math.sin %66 : f32
          %68 = arith.cmpf une, %67, %cst_0 : f32
          %69 = arith.xori %68, %true : i1
          %70 = scf.if %68 -> (f32) {
            %71 = arith.sitofp %7 : i64 to f32
            %72 = arith.addf %71, %cst_2 : f32
            %73 = func.call @acosf(%extracted) : (f32) -> f32
            %74 = arith.mulf %72, %73 : f32
            %75 = math.cos %74 : f32
            %76 = func.call @acosf(%extracted) : (f32) -> f32
            %77 = arith.divf %76, %cst_1 : f32
            %78 = math.cos %77 : f32
            %79 = arith.divf %75, %78 : f32
            scf.yield %79 : f32
          } else {
            scf.yield %31 : f32
          }
          scf.yield %69, %70 : i1, f32
        } else {
          scf.yield %false, %31 : i1, f32
        }
        %54 = arith.andi %53#0, %17 : i1
        %55 = arith.xori %54, %true : i1
        %56 = arith.andi %55, %53#0 : i1
        %57 = arith.addi %7, %7 : i64
        %58 = arith.subi %c-1_i64, %57 : i64
        %59 = arith.sitofp %58 : i64 to f32
        %60 = arith.addi %7, %7 : i64
        %61 = arith.addi %60, %c1_i64 : i64
        %62 = arith.sitofp %61 : i64 to f32
        %63 = arith.select %54, %62, %53#1 : f32
        %64 = arith.select %56, %59, %63 : f32
        scf.yield %64 : f32
      } else {
        scf.yield %31 : f32
      }
      %43 = arith.cmpi eq, %7, %c0_i64 : i64
      %44 = arith.cmpi ne, %7, %c0_i64 : i64
      %45 = arith.select %43, %cst_0, %42 : f32
      %46 = arith.andi %41, %44 : i1
      %47 = arith.select %41, %45, %42 : f32
      %48 = arith.cmpi eq, %7, %c1_i64 : i64
      %49 = arith.andi %46, %48 : i1
      %50 = arith.xori %49, %true : i1
      %51 = arith.andi %50, %46 : i1
      %52 = scf.if %51 -> (f32) {
        %53 = arith.addf %extracted, %extracted : f32
        %54 = arith.subf %53, %cst_0 : f32
        %55:4 = scf.while (%arg6 = %c2_i64, %arg7 = %54, %arg8 = %54, %arg9 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %56 = arith.cmpi sle, %arg6, %7 : i64
          %57 = arith.cmpf uno, %arg8, %arg8 : f32
          %58 = arith.extui %57 : i1 to i8
          %59 = arith.cmpi eq, %58, %c0_i8 : i8
          %60 = arith.andi %56, %59 : i1
          scf.condition(%60) %arg7, %arg6, %arg8, %arg9 : f32, i64, f32, f32
        } do {
        ^bb0(%arg6: f32, %arg7: i64, %arg8: f32, %arg9: f32):
          %56 = arith.cmpf uno, %arg8, %arg8 : f32
          %57 = arith.extui %56 : i1 to i8
          %58 = arith.cmpi eq, %57, %c0_i8 : i8
          %59 = arith.select %58, %arg8, %arg9 : f32
          %60 = arith.cmpi sle, %arg7, %7 : i64
          %61 = arith.select %60, %59, %arg9 : f32
          %62 = arith.mulf %53, %arg8 : f32
          %63 = arith.subf %62, %arg9 : f32
          %64 = arith.addi %arg7, %c1_i64 : i64
          scf.yield %64, %63, %63, %61 : i64, f32, f32, f32
        }
        scf.yield %55#0 : f32
      } else {
        %53 = arith.addf %extracted, %extracted : f32
        %54 = arith.subf %53, %cst_0 : f32
        %55 = arith.select %49, %54, %47 : f32
        scf.yield %55 : f32
      }
      %inserted = tensor.insert %52 into %arg5[%arg4] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @acosf(f32) -> f32
}

