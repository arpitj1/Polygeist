module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_laguerre_l_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
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
      %9 = arith.cmpf oeq, %8, %cst_0 : f32
      %10 = arith.select %9, %cst, %7 : f32
      %11:2 = scf.if %6 -> (i1, f32) {
        %17:2 = scf.if %9 -> (i1, f32) {
          scf.yield %false, %10 : i1, f32
        } else {
          %18 = arith.cmpi eq, %4, %c0_i64 : i64
          %19 = arith.cmpi ne, %4, %c0_i64 : i64
          %20 = arith.select %18, %cst, %10 : f32
          scf.yield %19, %20 : i1, f32
        }
        scf.yield %17#0, %17#1 : i1, f32
      } else {
        scf.yield %false, %7 : i1, f32
      }
      %12 = arith.cmpi eq, %4, %c1_i64 : i64
      %13 = arith.andi %11#0, %12 : i1
      %14 = arith.xori %13, %true : i1
      %15 = arith.andi %14, %11#0 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.subf %cst, %2 : f32
        %18:4 = scf.while (%arg5 = %c1_i64, %arg6 = %17, %arg7 = %17, %arg8 = %cst) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %19 = arith.cmpi slt, %arg5, %4 : i64
          %20 = arith.cmpf uno, %arg7, %arg7 : f32
          %21 = arith.extui %20 : i1 to i8
          %22 = arith.cmpi eq, %21, %c0_i8 : i8
          %23 = arith.andi %19, %22 : i1
          scf.condition(%23) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %19 = arith.cmpf uno, %arg7, %arg7 : f32
          %20 = arith.extui %19 : i1 to i8
          %21 = arith.cmpi eq, %20, %c0_i8 : i8
          %22 = arith.select %21, %arg7, %arg8 : f32
          %23 = arith.cmpi slt, %arg6, %4 : i64
          %24 = arith.select %23, %22, %arg8 : f32
          %25 = arith.addi %arg6, %arg6 : i64
          %26 = arith.sitofp %25 : i64 to f32
          %27 = arith.addf %26, %17 : f32
          %28 = arith.mulf %27, %arg7 : f32
          %29 = arith.sitofp %arg6 : i64 to f32
          %30 = arith.mulf %29, %arg8 : f32
          %31 = arith.subf %28, %30 : f32
          %32 = arith.addi %arg6, %c1_i64 : i64
          %33 = arith.sitofp %32 : i64 to f32
          %34 = arith.divf %31, %33 : f32
          scf.yield %32, %34, %34, %24 : i64, f32, f32, f32
        }
        scf.yield %18#0 : f32
      } else {
        %17 = scf.if %13 -> (f32) {
          %18 = arith.subf %cst, %2 : f32
          scf.yield %18 : f32
        } else {
          scf.yield %11#1 : f32
        }
        scf.yield %17 : f32
      }
      affine.store %16, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @calc_laguerre_lf(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %false = arith.constant false
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst : f32
    %7 = arith.select %6, %cst_0, %4 : f32
    %8:2 = scf.if %3 -> (i1, f32) {
      %14:2 = scf.if %6 -> (i1, f32) {
        scf.yield %false, %7 : i1, f32
      } else {
        %15 = arith.cmpi eq, %0, %c0_i64 : i64
        %16 = arith.cmpi ne, %0, %c0_i64 : i64
        %17 = arith.select %15, %cst_0, %7 : f32
        scf.yield %16, %17 : i1, f32
      }
      scf.yield %14#0, %14#1 : i1, f32
    } else {
      scf.yield %false, %4 : i1, f32
    }
    %9 = arith.cmpi eq, %0, %c1_i64 : i64
    %10 = arith.andi %8#0, %9 : i1
    %11 = arith.xori %10, %true : i1
    %12 = arith.andi %11, %8#0 : i1
    %13 = scf.if %12 -> (f32) {
      %14 = arith.subf %cst_0, %arg0 : f32
      %15:4 = scf.while (%arg2 = %c1_i64, %arg3 = %14, %arg4 = %14, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %16 = arith.cmpi slt, %arg2, %0 : i64
        %17 = arith.cmpf uno, %arg4, %arg4 : f32
        %18 = arith.extui %17 : i1 to i8
        %19 = arith.cmpi eq, %18, %c0_i8 : i8
        %20 = arith.andi %16, %19 : i1
        scf.condition(%20) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %16 = arith.cmpf uno, %arg4, %arg4 : f32
        %17 = arith.extui %16 : i1 to i8
        %18 = arith.cmpi eq, %17, %c0_i8 : i8
        %19 = arith.select %18, %arg4, %arg5 : f32
        %20 = arith.cmpi slt, %arg3, %0 : i64
        %21 = arith.select %20, %19, %arg5 : f32
        %22 = arith.addi %arg3, %arg3 : i64
        %23 = arith.sitofp %22 : i64 to f32
        %24 = arith.addf %23, %14 : f32
        %25 = arith.mulf %24, %arg4 : f32
        %26 = arith.sitofp %arg3 : i64 to f32
        %27 = arith.mulf %26, %arg5 : f32
        %28 = arith.subf %25, %27 : f32
        %29 = arith.addi %arg3, %c1_i64 : i64
        %30 = arith.sitofp %29 : i64 to f32
        %31 = arith.divf %28, %30 : f32
        scf.yield %29, %31, %31, %21 : i64, f32, f32, f32
      }
      scf.yield %15#0 : f32
    } else {
      %14 = scf.if %10 -> (f32) {
        %15 = arith.subf %cst_0, %arg0 : f32
        scf.yield %15 : f32
      } else {
        scf.yield %8#1 : f32
      }
      scf.yield %14 : f32
    }
    return %13 : f32
  }
  func.func @_Z29laguerre_polynomial_l_forwardIfLb0EET_S0_S0_(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %c0_i8 = arith.constant 0 : i8
    %true = arith.constant true
    %false = arith.constant false
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst : f32
    %7 = arith.select %6, %cst_0, %4 : f32
    %8:2 = scf.if %3 -> (i1, f32) {
      %14:2 = scf.if %6 -> (i1, f32) {
        scf.yield %false, %7 : i1, f32
      } else {
        %15 = arith.cmpi eq, %0, %c0_i64 : i64
        %16 = arith.cmpi ne, %0, %c0_i64 : i64
        %17 = arith.select %15, %cst_0, %7 : f32
        scf.yield %16, %17 : i1, f32
      }
      scf.yield %14#0, %14#1 : i1, f32
    } else {
      scf.yield %false, %4 : i1, f32
    }
    %9 = arith.cmpi eq, %0, %c1_i64 : i64
    %10 = arith.andi %8#0, %9 : i1
    %11 = arith.xori %10, %true : i1
    %12 = arith.andi %11, %8#0 : i1
    %13 = scf.if %12 -> (f32) {
      %14 = arith.subf %cst_0, %arg0 : f32
      %15:4 = scf.while (%arg2 = %c1_i64, %arg3 = %14, %arg4 = %14, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %16 = arith.cmpi slt, %arg2, %0 : i64
        %17 = arith.cmpf uno, %arg4, %arg4 : f32
        %18 = arith.extui %17 : i1 to i8
        %19 = arith.cmpi eq, %18, %c0_i8 : i8
        %20 = arith.andi %16, %19 : i1
        scf.condition(%20) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %16 = arith.cmpf uno, %arg4, %arg4 : f32
        %17 = arith.extui %16 : i1 to i8
        %18 = arith.cmpi eq, %17, %c0_i8 : i8
        %19 = arith.select %18, %arg4, %arg5 : f32
        %20 = arith.cmpi slt, %arg3, %0 : i64
        %21 = arith.select %20, %19, %arg5 : f32
        %22 = arith.addi %arg3, %arg3 : i64
        %23 = arith.sitofp %22 : i64 to f32
        %24 = arith.addf %23, %14 : f32
        %25 = arith.mulf %24, %arg4 : f32
        %26 = arith.sitofp %arg3 : i64 to f32
        %27 = arith.mulf %26, %arg5 : f32
        %28 = arith.subf %25, %27 : f32
        %29 = arith.addi %arg3, %c1_i64 : i64
        %30 = arith.sitofp %29 : i64 to f32
        %31 = arith.divf %28, %30 : f32
        scf.yield %29, %31, %31, %21 : i64, f32, f32, f32
      }
      scf.yield %15#0 : f32
    } else {
      %14 = scf.if %10 -> (f32) {
        %15 = arith.subf %cst_0, %arg0 : f32
        scf.yield %15 : f32
      } else {
        scf.yield %8#1 : f32
      }
      scf.yield %14 : f32
    }
    return %13 : f32
  }
  func.func @_Z29laguerre_polynomial_l_forwardIfET_S0_l(%arg0: f32, %arg1: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %2 = arith.cmpi sge, %arg1, %c0_i64 : i64
    %3 = arith.select %1, %cst_0, %0 : f32
    %4 = math.absf %arg0 : f32
    %5 = arith.cmpf oeq, %4, %cst_0 : f32
    %6 = arith.select %5, %cst, %3 : f32
    %7:2 = scf.if %2 -> (i1, f32) {
      %13:2 = scf.if %5 -> (i1, f32) {
        scf.yield %false, %6 : i1, f32
      } else {
        %14 = arith.cmpi eq, %arg1, %c0_i64 : i64
        %15 = arith.cmpi ne, %arg1, %c0_i64 : i64
        %16 = arith.select %14, %cst, %6 : f32
        scf.yield %15, %16 : i1, f32
      }
      scf.yield %13#0, %13#1 : i1, f32
    } else {
      scf.yield %false, %3 : i1, f32
    }
    %8 = arith.cmpi eq, %arg1, %c1_i64 : i64
    %9 = arith.andi %7#0, %8 : i1
    %10 = arith.xori %9, %true : i1
    %11 = arith.andi %10, %7#0 : i1
    %12 = scf.if %11 -> (f32) {
      %13 = arith.subf %cst, %arg0 : f32
      %14:4 = scf.while (%arg2 = %c1_i64, %arg3 = %13, %arg4 = %13, %arg5 = %cst) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %15 = arith.cmpi slt, %arg2, %arg1 : i64
        %16 = arith.cmpf uno, %arg4, %arg4 : f32
        %17 = arith.extui %16 : i1 to i8
        %18 = arith.cmpi eq, %17, %c0_i8 : i8
        %19 = arith.andi %15, %18 : i1
        scf.condition(%19) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %15 = arith.cmpf uno, %arg4, %arg4 : f32
        %16 = arith.extui %15 : i1 to i8
        %17 = arith.cmpi eq, %16, %c0_i8 : i8
        %18 = arith.select %17, %arg4, %arg5 : f32
        %19 = arith.cmpi slt, %arg3, %arg1 : i64
        %20 = arith.select %19, %18, %arg5 : f32
        %21 = arith.addi %arg3, %arg3 : i64
        %22 = arith.sitofp %21 : i64 to f32
        %23 = arith.addf %22, %13 : f32
        %24 = arith.mulf %23, %arg4 : f32
        %25 = arith.sitofp %arg3 : i64 to f32
        %26 = arith.mulf %25, %arg5 : f32
        %27 = arith.subf %24, %26 : f32
        %28 = arith.addi %arg3, %c1_i64 : i64
        %29 = arith.sitofp %28 : i64 to f32
        %30 = arith.divf %27, %29 : f32
        scf.yield %28, %30, %30, %20 : i64, f32, f32, f32
      }
      scf.yield %14#0 : f32
    } else {
      %13 = scf.if %9 -> (f32) {
        %14 = arith.subf %cst, %arg0 : f32
        scf.yield %14 : f32
      } else {
        scf.yield %7#1 : f32
      }
      scf.yield %13 : f32
    }
    return %12 : f32
  }
  func.func @_ZSt3absf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.absf %arg0 : f32
    return %0 : f32
  }
  func.func @_ZSt5isnanf(%arg0: f32) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = arith.cmpf uno, %arg0, %arg0 : f32
    %1 = arith.extui %0 : i1 to i8
    return %1 : i8
  }
}
