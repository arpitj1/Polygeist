module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_chebyshev_v_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %true = arith.constant true
    %false = arith.constant false
    %c-1_i64 = arith.constant -1 : i64
    %c0_i8 = arith.constant 0 : i8
    %cst = arith.constant 5.000000e-01 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %c8_i64 = arith.constant 8 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_2, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst_1 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.cmpf ogt, %2, %cst_2 : f32
      %12 = arith.xori %11, %true : i1
      %13 = arith.select %11, %cst_1, %7 : f32
      %14 = arith.remsi %4, %c2_i64 : i64
      %15 = arith.cmpi eq, %14, %c0_i64 : i64
      %16 = arith.andi %12, %15 : i1
      %17 = arith.xori %16, %true : i1
      %18 = arith.andi %17, %12 : i1
      %19 = arith.andi %6, %9 : i1
      %20 = arith.andi %6, %10 : i1
      %21 = arith.addi %4, %4 : i64
      %22 = arith.subi %c-1_i64, %21 : i64
      %23 = arith.sitofp %22 : i64 to f32
      %24 = arith.addi %4, %4 : i64
      %25 = arith.addi %24, %c1_i64 : i64
      %26 = arith.sitofp %25 : i64 to f32
      %27 = arith.select %16, %26, %13 : f32
      %28 = arith.select %18, %23, %27 : f32
      %29 = arith.select %19, %28, %7 : f32
      %30 = arith.cmpi sgt, %4, %c8_i64 : i64
      %31 = arith.andi %20, %30 : i1
      %32 = arith.cmpf olt, %8, %cst_1 : f32
      %33 = arith.xori %32, %true : i1
      %34 = arith.andi %33, %20 : i1
      %35 = arith.andi %31, %32 : i1
      %36 = arith.xori %31, %true : i1
      %37 = arith.andi %31, %34 : i1
      %38 = arith.andi %36, %20 : i1
      %39 = arith.ori %37, %38 : i1
      %40 = scf.if %35 -> (f32) {
        %51:2 = scf.if %20 -> (i1, f32) {
          %63 = func.call @acosf(%2) : (f32) -> f32
          %64 = arith.divf %63, %cst_0 : f32
          %65 = math.sin %64 : f32
          %66 = arith.cmpf une, %65, %cst_1 : f32
          %67 = arith.xori %66, %true : i1
          %68 = scf.if %66 -> (f32) {
            %69 = arith.sitofp %4 : i64 to f32
            %70 = arith.addf %69, %cst : f32
            %71 = func.call @acosf(%2) : (f32) -> f32
            %72 = arith.mulf %70, %71 : f32
            %73 = math.cos %72 : f32
            %74 = func.call @acosf(%2) : (f32) -> f32
            %75 = arith.divf %74, %cst_0 : f32
            %76 = math.cos %75 : f32
            %77 = arith.divf %73, %76 : f32
            scf.yield %77 : f32
          } else {
            scf.yield %29 : f32
          }
          scf.yield %67, %68 : i1, f32
        } else {
          scf.yield %false, %29 : i1, f32
        }
        %52 = arith.andi %51#0, %15 : i1
        %53 = arith.xori %52, %true : i1
        %54 = arith.andi %53, %51#0 : i1
        %55 = arith.addi %4, %4 : i64
        %56 = arith.subi %c-1_i64, %55 : i64
        %57 = arith.sitofp %56 : i64 to f32
        %58 = arith.addi %4, %4 : i64
        %59 = arith.addi %58, %c1_i64 : i64
        %60 = arith.sitofp %59 : i64 to f32
        %61 = arith.select %52, %60, %51#1 : f32
        %62 = arith.select %54, %57, %61 : f32
        scf.yield %62 : f32
      } else {
        scf.yield %29 : f32
      }
      %41 = arith.cmpi eq, %4, %c0_i64 : i64
      %42 = arith.cmpi ne, %4, %c0_i64 : i64
      %43 = arith.select %41, %cst_1, %40 : f32
      %44 = arith.select %39, %42, %false : i1
      %45 = arith.select %39, %43, %40 : f32
      %46 = arith.cmpi eq, %4, %c1_i64 : i64
      %47 = arith.andi %44, %46 : i1
      %48 = arith.xori %47, %true : i1
      %49 = arith.andi %48, %44 : i1
      %50 = scf.if %49 -> (f32) {
        %51 = arith.addf %2, %2 : f32
        %52 = arith.subf %51, %cst_1 : f32
        %53:4 = scf.while (%arg5 = %c2_i64, %arg6 = %52, %arg7 = %52, %arg8 = %cst_1) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %54 = arith.cmpi sle, %arg5, %4 : i64
          %55 = arith.cmpf uno, %arg7, %arg7 : f32
          %56 = arith.extui %55 : i1 to i8
          %57 = arith.cmpi eq, %56, %c0_i8 : i8
          %58 = arith.andi %54, %57 : i1
          scf.condition(%58) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %54 = arith.cmpf uno, %arg7, %arg7 : f32
          %55 = arith.extui %54 : i1 to i8
          %56 = arith.cmpi eq, %55, %c0_i8 : i8
          %57 = arith.select %56, %arg7, %arg8 : f32
          %58 = arith.cmpi sle, %arg6, %4 : i64
          %59 = arith.select %58, %57, %arg8 : f32
          %60 = arith.mulf %51, %arg7 : f32
          %61 = arith.subf %60, %arg8 : f32
          %62 = arith.addi %arg6, %c1_i64 : i64
          scf.yield %62, %61, %61, %59 : i64, f32, f32, f32
        }
        scf.yield %53#0 : f32
      } else {
        %51 = arith.addf %2, %2 : f32
        %52 = arith.subf %51, %cst_1 : f32
        %53 = arith.select %47, %52, %45 : f32
        scf.yield %53 : f32
      }
      affine.store %50, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

