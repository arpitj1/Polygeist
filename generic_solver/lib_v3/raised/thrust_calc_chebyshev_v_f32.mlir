module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      %2 = memref.load %arg1[%arg4] : memref<?xf32>
      %3 = memref.load %arg2[%arg4] : memref<?xf32>
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
      %21 = scf.if %19 -> (f32) {
        %39 = scf.if %18 -> (f32) {
          %40 = arith.addi %4, %4 : i64
          %41 = arith.subi %c-1_i64, %40 : i64
          %42 = arith.sitofp %41 : i64 to f32
          scf.yield %42 : f32
        } else {
          %40 = scf.if %16 -> (f32) {
            %41 = arith.addi %4, %4 : i64
            %42 = arith.addi %41, %c1_i64 : i64
            %43 = arith.sitofp %42 : i64 to f32
            scf.yield %43 : f32
          } else {
            scf.yield %13 : f32
          }
          scf.yield %40 : f32
        }
        scf.yield %39 : f32
      } else {
        scf.yield %7 : f32
      }
      %22 = arith.cmpi sgt, %4, %c8_i64 : i64
      %23 = arith.andi %20, %22 : i1
      %24 = arith.cmpf olt, %8, %cst_1 : f32
      %25 = arith.xori %24, %true : i1
      %26 = arith.andi %25, %20 : i1
      %27 = arith.andi %23, %24 : i1
      %28 = arith.xori %23, %true : i1
      %29 = arith.andi %23, %26 : i1
      %30 = arith.andi %28, %20 : i1
      %31 = arith.ori %29, %30 : i1
      %32 = scf.if %27 -> (f32) {
        %39:2 = scf.if %20 -> (i1, f32) {
          %44 = func.call @acosf(%2) : (f32) -> f32
          %45 = arith.divf %44, %cst_0 : f32
          %46 = math.sin %45 : f32
          %47 = arith.cmpf une, %46, %cst_1 : f32
          %48 = arith.xori %47, %true : i1
          %49 = scf.if %47 -> (f32) {
            %50 = arith.sitofp %4 : i64 to f32
            %51 = arith.addf %50, %cst : f32
            %52 = func.call @acosf(%2) : (f32) -> f32
            %53 = arith.mulf %51, %52 : f32
            %54 = math.cos %53 : f32
            %55 = func.call @acosf(%2) : (f32) -> f32
            %56 = arith.divf %55, %cst_0 : f32
            %57 = math.cos %56 : f32
            %58 = arith.divf %54, %57 : f32
            scf.yield %58 : f32
          } else {
            scf.yield %21 : f32
          }
          scf.yield %48, %49 : i1, f32
        } else {
          scf.yield %false, %21 : i1, f32
        }
        %40 = arith.andi %39#0, %15 : i1
        %41 = arith.xori %40, %true : i1
        %42 = arith.andi %41, %39#0 : i1
        %43 = scf.if %42 -> (f32) {
          %44 = arith.addi %4, %4 : i64
          %45 = arith.subi %c-1_i64, %44 : i64
          %46 = arith.sitofp %45 : i64 to f32
          scf.yield %46 : f32
        } else {
          %44 = scf.if %40 -> (f32) {
            %45 = arith.addi %4, %4 : i64
            %46 = arith.addi %45, %c1_i64 : i64
            %47 = arith.sitofp %46 : i64 to f32
            scf.yield %47 : f32
          } else {
            scf.yield %39#1 : f32
          }
          scf.yield %44 : f32
        }
        scf.yield %43 : f32
      } else {
        scf.yield %21 : f32
      }
      %33:2 = scf.if %31 -> (i1, f32) {
        %39 = arith.cmpi eq, %4, %c0_i64 : i64
        %40 = arith.cmpi ne, %4, %c0_i64 : i64
        %41 = arith.select %39, %cst_1, %32 : f32
        scf.yield %40, %41 : i1, f32
      } else {
        scf.yield %false, %32 : i1, f32
      }
      %34 = arith.cmpi eq, %4, %c1_i64 : i64
      %35 = arith.andi %33#0, %34 : i1
      %36 = arith.xori %35, %true : i1
      %37 = arith.andi %36, %33#0 : i1
      %38 = scf.if %37 -> (f32) {
        %39 = arith.addf %2, %2 : f32
        %40 = arith.subf %39, %cst_1 : f32
        %41:4 = scf.while (%arg5 = %c2_i64, %arg6 = %40, %arg7 = %40, %arg8 = %cst_1) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %42 = arith.cmpi sle, %arg5, %4 : i64
          %43 = arith.cmpf uno, %arg7, %arg7 : f32
          %44 = arith.extui %43 : i1 to i8
          %45 = arith.cmpi eq, %44, %c0_i8 : i8
          %46 = arith.andi %42, %45 : i1
          scf.condition(%46) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %42 = arith.cmpf uno, %arg7, %arg7 : f32
          %43 = arith.extui %42 : i1 to i8
          %44 = arith.cmpi eq, %43, %c0_i8 : i8
          %45 = arith.select %44, %arg7, %arg8 : f32
          %46 = arith.cmpi sle, %arg6, %4 : i64
          %47 = arith.select %46, %45, %arg8 : f32
          %48 = arith.mulf %39, %arg7 : f32
          %49 = arith.subf %48, %arg8 : f32
          %50 = arith.addi %arg6, %c1_i64 : i64
          scf.yield %50, %49, %49, %47 : i64, f32, f32, f32
        }
        scf.yield %41#0 : f32
      } else {
        %39 = scf.if %35 -> (f32) {
          %40 = arith.addf %2, %2 : f32
          %41 = arith.subf %40, %cst_1 : f32
          scf.yield %41 : f32
        } else {
          scf.yield %33#1 : f32
        }
        scf.yield %39 : f32
      }
      memref.store %38, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func private @acosf(f32) -> f32
}

