module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_w_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
    %c8_i64 = arith.constant 8 : i64
    %cst_2 = arith.constant 2.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %c0_i8 = arith.constant 0 : i8
    %false = arith.constant false
    %true = arith.constant true
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_4 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = arith.fptosi %extracted_4 : f32 to i64
      %8 = arith.cmpi slt, %7, %c0_i64 : i64
      %9 = arith.cmpi sge, %7, %c0_i64 : i64
      %10 = arith.select %8, %cst, %4 : f32
      %11 = math.absf %extracted : f32
      %12 = arith.cmpf oeq, %11, %cst_0 : f32
      %13 = arith.xori %12, %true : i1
      %14 = arith.andi %9, %12 : i1
      %15 = arith.andi %9, %13 : i1
      %16 = arith.cmpf ogt, %extracted, %cst : f32
      %17 = arith.addi %7, %7 : i64
      %18 = arith.addi %17, %c1_i64 : i64
      %19 = arith.sitofp %18 : i64 to f32
      %20 = arith.remsi %7, %c2_i64 : i64
      %21 = arith.cmpi eq, %20, %c0_i64 : i64
      %22 = arith.cmpi ne, %20, %c0_i64 : i64
      %23 = arith.select %21, %cst_0, %10 : f32
      %24 = arith.xori %16, %true : i1
      %25 = arith.andi %24, %22 : i1
      %26 = arith.select %16, %19, %23 : f32
      %27 = arith.select %25, %cst_1, %26 : f32
      %28 = arith.select %14, %27, %10 : f32
      %29 = arith.cmpi sgt, %7, %c8_i64 : i64
      %30 = arith.andi %15, %29 : i1
      %31 = arith.cmpf olt, %11, %cst_0 : f32
      %32 = arith.xori %31, %true : i1
      %33 = arith.andi %32, %15 : i1
      %34 = arith.andi %30, %31 : i1
      %35 = arith.xori %30, %true : i1
      %36 = arith.andi %30, %33 : i1
      %37 = arith.andi %35, %15 : i1
      %38 = arith.ori %36, %37 : i1
      %39 = scf.if %34 -> (f32) {
        %50:2 = scf.if %15 -> (i1, f32) {
          %66 = func.call @acosf(%extracted) : (f32) -> f32
          %67 = arith.divf %66, %cst_2 : f32
          %68 = math.cos %67 : f32
          %69 = arith.cmpf une, %68, %cst_0 : f32
          %70 = arith.xori %69, %true : i1
          %71 = scf.if %69 -> (f32) {
            %72 = arith.sitofp %7 : i64 to f32
            %73 = arith.addf %72, %cst_3 : f32
            %74 = func.call @acosf(%extracted) : (f32) -> f32
            %75 = arith.mulf %73, %74 : f32
            %76 = math.sin %75 : f32
            %77 = func.call @acosf(%extracted) : (f32) -> f32
            %78 = arith.divf %77, %cst_2 : f32
            %79 = math.sin %78 : f32
            %80 = arith.divf %76, %79 : f32
            scf.yield %80 : f32
          } else {
            scf.yield %28 : f32
          }
          scf.yield %70, %71 : i1, f32
        } else {
          scf.yield %false, %28 : i1, f32
        }
        %51 = arith.cmpf ogt, %extracted, %cst : f32
        %52 = arith.andi %50#0, %51 : i1
        %53 = arith.xori %52, %true : i1
        %54 = arith.andi %53, %50#0 : i1
        %55 = arith.addi %7, %7 : i64
        %56 = arith.addi %55, %c1_i64 : i64
        %57 = arith.sitofp %56 : i64 to f32
        %58 = arith.select %52, %57, %50#1 : f32
        %59 = arith.remsi %7, %c2_i64 : i64
        %60 = arith.cmpi eq, %59, %c0_i64 : i64
        %61 = arith.cmpi ne, %59, %c0_i64 : i64
        %62 = arith.andi %54, %61 : i1
        %63 = arith.select %60, %cst_0, %50#1 : f32
        %64 = arith.select %54, %63, %58 : f32
        %65 = arith.select %62, %cst_1, %64 : f32
        scf.yield %65 : f32
      } else {
        scf.yield %28 : f32
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
        %51 = arith.addf %50, %cst_0 : f32
        %52:4 = scf.while (%arg6 = %c2_i64, %arg7 = %51, %arg8 = %51, %arg9 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %53 = arith.cmpi sle, %arg6, %7 : i64
          %54 = arith.cmpf uno, %arg8, %arg8 : f32
          %55 = arith.extui %54 : i1 to i8
          %56 = arith.cmpi eq, %55, %c0_i8 : i8
          %57 = arith.andi %53, %56 : i1
          scf.condition(%57) %arg7, %arg6, %arg8, %arg9 : f32, i64, f32, f32
        } do {
        ^bb0(%arg6: f32, %arg7: i64, %arg8: f32, %arg9: f32):
          %53 = arith.cmpf uno, %arg8, %arg8 : f32
          %54 = arith.extui %53 : i1 to i8
          %55 = arith.cmpi eq, %54, %c0_i8 : i8
          %56 = arith.select %55, %arg8, %arg9 : f32
          %57 = arith.cmpi sle, %arg7, %7 : i64
          %58 = arith.select %57, %56, %arg9 : f32
          %59 = arith.mulf %50, %arg8 : f32
          %60 = arith.subf %59, %arg9 : f32
          %61 = arith.addi %arg7, %c1_i64 : i64
          scf.yield %61, %60, %60, %58 : i64, f32, f32, f32
        }
        scf.yield %52#0 : f32
      } else {
        %50 = arith.addf %extracted, %extracted : f32
        %51 = arith.addf %50, %cst_0 : f32
        %52 = arith.select %46, %51, %44 : f32
        scf.yield %52 : f32
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

