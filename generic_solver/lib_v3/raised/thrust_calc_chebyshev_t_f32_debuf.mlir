module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_t_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
    %c6_i64 = arith.constant 6 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %0 = bufferization.to_tensor %arg3 : memref<?xf32>
    %1 = bufferization.to_tensor %arg2 : memref<?xf32>
    %2 = bufferization.to_tensor %arg1 : memref<?xf32>
    %3 = arith.index_cast %arg0 : i32 to index
    %4 = llvm.mlir.undef : f32
    %5 = affine.for %arg4 = 0 to %3 iter_args(%arg5 = %0) -> (tensor<?xf32>) {
      %extracted = tensor.extract %2[%arg4] : tensor<?xf32>
      %extracted_2 = tensor.extract %1[%arg4] : tensor<?xf32>
      %7 = arith.fptosi %extracted_2 : f32 to i64
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
      %22 = arith.xori %21, %true : i1
      %23 = arith.select %22, %cst_1, %cst_0 : f32
      %24 = arith.select %14, %23, %10 : f32
      %25 = arith.cmpi sgt, %7, %c6_i64 : i64
      %26 = arith.andi %15, %25 : i1
      %27 = arith.cmpf olt, %11, %cst_0 : f32
      %28 = arith.andi %27, %15 : i1
      %29 = arith.xori %28, %true : i1
      %30 = arith.andi %29, %15 : i1
      %31 = arith.andi %26, %28 : i1
      %32 = arith.xori %26, %true : i1
      %33 = arith.andi %26, %30 : i1
      %34 = arith.andi %32, %15 : i1
      %35 = arith.ori %33, %34 : i1
      %36 = scf.if %31 -> (f32) {
        %46 = arith.sitofp %7 : i64 to f32
        %47 = func.call @acosf(%extracted) : (f32) -> f32
        %48 = arith.mulf %46, %47 : f32
        %49 = math.cos %48 : f32
        scf.yield %49 : f32
      } else {
        scf.yield %24 : f32
      }
      %37 = arith.cmpi eq, %7, %c0_i64 : i64
      %38 = arith.cmpi ne, %7, %c0_i64 : i64
      %39 = arith.select %37, %cst_0, %36 : f32
      %40 = arith.andi %35, %38 : i1
      %41 = arith.select %35, %39, %36 : f32
      %42 = arith.cmpi eq, %7, %c1_i64 : i64
      %43 = arith.cmpi ne, %7, %c1_i64 : i64
      %44 = arith.select %42, %extracted, %41 : f32
      %45 = scf.if %40 -> (f32) {
        %46 = scf.if %43 -> (f32) {
          %47 = arith.addf %extracted, %extracted : f32
          %48:4 = scf.while (%arg6 = %c2_i64, %arg7 = %extracted, %arg8 = %extracted, %arg9 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %49 = arith.cmpi sle, %arg6, %7 : i64
            %50 = arith.cmpf uno, %arg8, %arg8 : f32
            %51 = arith.extui %50 : i1 to i8
            %52 = arith.cmpi eq, %51, %c0_i8 : i8
            %53 = arith.andi %49, %52 : i1
            scf.condition(%53) %arg7, %arg6, %arg8, %arg9 : f32, i64, f32, f32
          } do {
          ^bb0(%arg6: f32, %arg7: i64, %arg8: f32, %arg9: f32):
            %49 = arith.cmpf uno, %arg8, %arg8 : f32
            %50 = arith.extui %49 : i1 to i8
            %51 = arith.cmpi eq, %50, %c0_i8 : i8
            %52 = arith.select %51, %arg8, %arg9 : f32
            %53 = arith.cmpi sle, %arg7, %7 : i64
            %54 = arith.select %53, %52, %arg9 : f32
            %55 = arith.mulf %47, %arg8 : f32
            %56 = arith.subf %55, %arg9 : f32
            %57 = arith.addi %arg7, %c1_i64 : i64
            scf.yield %57, %56, %56, %54 : i64, f32, f32, f32
          }
          scf.yield %48#0 : f32
        } else {
          scf.yield %44 : f32
        }
        scf.yield %46 : f32
      } else {
        scf.yield %41 : f32
      }
      %inserted = tensor.insert %45 into %arg5[%arg4] : tensor<?xf32>
      affine.yield %inserted : tensor<?xf32>
    }
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg3 : memref<?xf32> to memref<?xf32>
    return
  }
  func.func private @acosf(f32) -> f32
}

