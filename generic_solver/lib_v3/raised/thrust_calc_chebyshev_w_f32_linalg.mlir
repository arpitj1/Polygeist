module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_w_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c8_i64 = arith.constant 8 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_3, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst_2 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.andi %6, %9 : i1
      %12 = arith.andi %6, %10 : i1
      %13 = arith.cmpf ogt, %2, %cst_3 : f32
      %14 = arith.addi %4, %4 : i64
      %15 = arith.addi %14, %c1_i64 : i64
      %16 = arith.sitofp %15 : i64 to f32
      %17 = arith.remsi %4, %c2_i64 : i64
      %18 = arith.cmpi eq, %17, %c0_i64 : i64
      %19 = arith.cmpi ne, %17, %c0_i64 : i64
      %20 = arith.select %18, %cst_2, %7 : f32
      %21 = arith.select %13, %false, %19 : i1
      %22 = arith.select %13, %16, %20 : f32
      %23 = arith.select %21, %cst_1, %22 : f32
      %24 = arith.select %11, %23, %7 : f32
      %25 = arith.cmpi sgt, %4, %c8_i64 : i64
      %26 = arith.andi %12, %25 : i1
      %27 = arith.cmpf olt, %8, %cst_2 : f32
      %28 = arith.xori %27, %true : i1
      %29 = arith.andi %28, %12 : i1
      %30 = arith.andi %26, %27 : i1
      %31 = arith.xori %26, %true : i1
      %32 = arith.andi %26, %29 : i1
      %33 = arith.andi %31, %12 : i1
      %34 = arith.ori %32, %33 : i1
      %35 = scf.if %30 -> (f32) {
        %46:2 = scf.if %12 -> (i1, f32) {
          %62 = func.call @acosf(%2) : (f32) -> f32
          %63 = arith.divf %62, %cst_0 : f32
          %64 = math.cos %63 : f32
          %65 = arith.cmpf une, %64, %cst_2 : f32
          %66 = arith.xori %65, %true : i1
          %67 = scf.if %65 -> (f32) {
            %68 = arith.sitofp %4 : i64 to f32
            %69 = arith.addf %68, %cst : f32
            %70 = func.call @acosf(%2) : (f32) -> f32
            %71 = arith.mulf %69, %70 : f32
            %72 = math.sin %71 : f32
            %73 = func.call @acosf(%2) : (f32) -> f32
            %74 = arith.divf %73, %cst_0 : f32
            %75 = math.sin %74 : f32
            %76 = arith.divf %72, %75 : f32
            scf.yield %76 : f32
          } else {
            scf.yield %24 : f32
          }
          scf.yield %66, %67 : i1, f32
        } else {
          scf.yield %false, %24 : i1, f32
        }
        %47 = arith.cmpf ogt, %2, %cst_3 : f32
        %48 = arith.andi %46#0, %47 : i1
        %49 = arith.xori %48, %true : i1
        %50 = arith.andi %49, %46#0 : i1
        %51 = arith.addi %4, %4 : i64
        %52 = arith.addi %51, %c1_i64 : i64
        %53 = arith.sitofp %52 : i64 to f32
        %54 = arith.select %48, %53, %46#1 : f32
        %55 = arith.remsi %4, %c2_i64 : i64
        %56 = arith.cmpi eq, %55, %c0_i64 : i64
        %57 = arith.cmpi ne, %55, %c0_i64 : i64
        %58 = arith.select %56, %cst_2, %54 : f32
        %59 = arith.select %50, %57, %false : i1
        %60 = arith.select %50, %58, %54 : f32
        %61 = arith.select %59, %cst_1, %60 : f32
        scf.yield %61 : f32
      } else {
        scf.yield %24 : f32
      }
      %36 = arith.cmpi eq, %4, %c0_i64 : i64
      %37 = arith.cmpi ne, %4, %c0_i64 : i64
      %38 = arith.select %36, %cst_2, %35 : f32
      %39 = arith.select %34, %37, %false : i1
      %40 = arith.select %34, %38, %35 : f32
      %41 = arith.cmpi eq, %4, %c1_i64 : i64
      %42 = arith.andi %39, %41 : i1
      %43 = arith.xori %42, %true : i1
      %44 = arith.andi %43, %39 : i1
      %45 = scf.if %44 -> (f32) {
        %46 = arith.addf %2, %2 : f32
        %47 = arith.addf %46, %cst_2 : f32
        %48:4 = scf.while (%arg5 = %c2_i64, %arg6 = %47, %arg7 = %47, %arg8 = %cst_2) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %49 = arith.cmpi sle, %arg5, %4 : i64
          %50 = arith.cmpf uno, %arg7, %arg7 : f32
          %51 = arith.extui %50 : i1 to i8
          %52 = arith.cmpi eq, %51, %c0_i8 : i8
          %53 = arith.andi %49, %52 : i1
          scf.condition(%53) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %49 = arith.cmpf uno, %arg7, %arg7 : f32
          %50 = arith.extui %49 : i1 to i8
          %51 = arith.cmpi eq, %50, %c0_i8 : i8
          %52 = arith.select %51, %arg7, %arg8 : f32
          %53 = arith.cmpi sle, %arg6, %4 : i64
          %54 = arith.select %53, %52, %arg8 : f32
          %55 = arith.mulf %46, %arg7 : f32
          %56 = arith.subf %55, %arg8 : f32
          %57 = arith.addi %arg6, %c1_i64 : i64
          scf.yield %57, %56, %56, %54 : i64, f32, f32, f32
        }
        scf.yield %48#0 : f32
      } else {
        %46 = arith.addf %2, %2 : f32
        %47 = arith.addf %46, %cst_2 : f32
        %48 = arith.select %42, %47, %40 : f32
        scf.yield %48 : f32
      }
      affine.store %45, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

