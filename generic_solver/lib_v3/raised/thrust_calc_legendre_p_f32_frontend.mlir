module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_legendre_p_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant -1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.cmpi slt, %4, %c0_i64 : i64
      %6 = arith.cmpi sge, %4, %c0_i64 : i64
      %7 = arith.select %5, %cst_1, %1 : f32
      %8 = math.absf %2 : f32
      %9 = arith.cmpf oeq, %8, %cst_0 : f32
      %10:2 = scf.if %6 -> (i1, f32) {
        %15:2 = scf.if %9 -> (i1, f32) {
          %16 = arith.cmpf ogt, %2, %cst_1 : f32
          %17 = scf.if %16 -> (i1) {
            scf.yield %true : i1
          } else {
            %20 = arith.remsi %4, %c2_i64 : i64
            %21 = arith.cmpi eq, %20, %c0_i64 : i64
            scf.yield %21 : i1
          }
          %18 = arith.xori %17, %true : i1
          %19 = arith.select %18, %cst, %cst_0 : f32
          scf.yield %false, %19 : i1, f32
        } else {
          %16 = arith.cmpi eq, %4, %c0_i64 : i64
          %17 = arith.cmpi ne, %4, %c0_i64 : i64
          %18 = arith.select %16, %cst_0, %7 : f32
          scf.yield %17, %18 : i1, f32
        }
        scf.yield %15#0, %15#1 : i1, f32
      } else {
        scf.yield %false, %7 : i1, f32
      }
      %11 = arith.cmpi eq, %4, %c1_i64 : i64
      %12 = arith.cmpi ne, %4, %c1_i64 : i64
      %13 = arith.select %11, %2, %10#1 : f32
      %14 = scf.if %10#0 -> (f32) {
        %15 = scf.if %12 -> (f32) {
          %16:4 = scf.while (%arg5 = %c1_i64, %arg6 = %2, %arg7 = %2, %arg8 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
            %17 = arith.cmpi slt, %arg5, %4 : i64
            %18 = arith.cmpf uno, %arg7, %arg7 : f32
            %19 = arith.extui %18 : i1 to i8
            %20 = arith.cmpi eq, %19, %c0_i8 : i8
            %21 = arith.andi %17, %20 : i1
            scf.condition(%21) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
          } do {
          ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
            %17 = arith.cmpf uno, %arg7, %arg7 : f32
            %18 = arith.extui %17 : i1 to i8
            %19 = arith.cmpi eq, %18, %c0_i8 : i8
            %20 = arith.select %19, %arg7, %arg8 : f32
            %21 = arith.cmpi slt, %arg6, %4 : i64
            %22 = arith.select %21, %20, %arg8 : f32
            %23 = arith.addi %arg6, %arg6 : i64
            %24 = arith.addi %23, %c1_i64 : i64
            %25 = arith.sitofp %24 : i64 to f32
            %26 = arith.mulf %25, %2 : f32
            %27 = arith.mulf %26, %arg7 : f32
            %28 = arith.sitofp %arg6 : i64 to f32
            %29 = arith.mulf %28, %arg8 : f32
            %30 = arith.subf %27, %29 : f32
            %31 = arith.addi %arg6, %c1_i64 : i64
            %32 = arith.sitofp %31 : i64 to f32
            %33 = arith.divf %30, %32 : f32
            scf.yield %31, %33, %33, %22 : i64, f32, f32, f32
          }
          scf.yield %16#0 : f32
        } else {
          scf.yield %13 : f32
        }
        scf.yield %15 : f32
      } else {
        scf.yield %10#1 : f32
      }
      affine.store %14, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @calc_legendre_pf(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
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
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7:2 = scf.if %3 -> (i1, f32) {
      %12:2 = scf.if %6 -> (i1, f32) {
        %13 = arith.cmpf ogt, %arg0, %cst : f32
        %14 = scf.if %13 -> (i1) {
          scf.yield %true : i1
        } else {
          %17 = arith.remsi %0, %c2_i64 : i64
          %18 = arith.cmpi eq, %17, %c0_i64 : i64
          scf.yield %18 : i1
        }
        %15 = arith.xori %14, %true : i1
        %16 = arith.select %15, %cst_1, %cst_0 : f32
        scf.yield %false, %16 : i1, f32
      } else {
        %13 = arith.cmpi eq, %0, %c0_i64 : i64
        %14 = arith.cmpi ne, %0, %c0_i64 : i64
        %15 = arith.select %13, %cst_0, %4 : f32
        scf.yield %14, %15 : i1, f32
      }
      scf.yield %12#0, %12#1 : i1, f32
    } else {
      scf.yield %false, %4 : i1, f32
    }
    %8 = arith.cmpi eq, %0, %c1_i64 : i64
    %9 = arith.cmpi ne, %0, %c1_i64 : i64
    %10 = arith.select %8, %arg0, %7#1 : f32
    %11 = scf.if %7#0 -> (f32) {
      %12 = scf.if %9 -> (f32) {
        %13:4 = scf.while (%arg2 = %c1_i64, %arg3 = %arg0, %arg4 = %arg0, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %14 = arith.cmpi slt, %arg2, %0 : i64
          %15 = arith.cmpf uno, %arg4, %arg4 : f32
          %16 = arith.extui %15 : i1 to i8
          %17 = arith.cmpi eq, %16, %c0_i8 : i8
          %18 = arith.andi %14, %17 : i1
          scf.condition(%18) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
        } do {
        ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
          %14 = arith.cmpf uno, %arg4, %arg4 : f32
          %15 = arith.extui %14 : i1 to i8
          %16 = arith.cmpi eq, %15, %c0_i8 : i8
          %17 = arith.select %16, %arg4, %arg5 : f32
          %18 = arith.cmpi slt, %arg3, %0 : i64
          %19 = arith.select %18, %17, %arg5 : f32
          %20 = arith.addi %arg3, %arg3 : i64
          %21 = arith.addi %20, %c1_i64 : i64
          %22 = arith.sitofp %21 : i64 to f32
          %23 = arith.mulf %22, %arg0 : f32
          %24 = arith.mulf %23, %arg4 : f32
          %25 = arith.sitofp %arg3 : i64 to f32
          %26 = arith.mulf %25, %arg5 : f32
          %27 = arith.subf %24, %26 : f32
          %28 = arith.addi %arg3, %c1_i64 : i64
          %29 = arith.sitofp %28 : i64 to f32
          %30 = arith.divf %27, %29 : f32
          scf.yield %28, %30, %30, %19 : i64, f32, f32, f32
        }
        scf.yield %13#0 : f32
      } else {
        scf.yield %10 : f32
      }
      scf.yield %12 : f32
    } else {
      scf.yield %7#1 : f32
    }
    return %11 : f32
  }
  func.func @_Z29legendre_polynomial_p_forwardIfLb0EET_S0_S0_(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_1 = arith.constant -1.000000e+00 : f32
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
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7:2 = scf.if %3 -> (i1, f32) {
      %12:2 = scf.if %6 -> (i1, f32) {
        %13 = arith.cmpf ogt, %arg0, %cst : f32
        %14 = scf.if %13 -> (i1) {
          scf.yield %true : i1
        } else {
          %17 = arith.remsi %0, %c2_i64 : i64
          %18 = arith.cmpi eq, %17, %c0_i64 : i64
          scf.yield %18 : i1
        }
        %15 = arith.xori %14, %true : i1
        %16 = arith.select %15, %cst_1, %cst_0 : f32
        scf.yield %false, %16 : i1, f32
      } else {
        %13 = arith.cmpi eq, %0, %c0_i64 : i64
        %14 = arith.cmpi ne, %0, %c0_i64 : i64
        %15 = arith.select %13, %cst_0, %4 : f32
        scf.yield %14, %15 : i1, f32
      }
      scf.yield %12#0, %12#1 : i1, f32
    } else {
      scf.yield %false, %4 : i1, f32
    }
    %8 = arith.cmpi eq, %0, %c1_i64 : i64
    %9 = arith.cmpi ne, %0, %c1_i64 : i64
    %10 = arith.select %8, %arg0, %7#1 : f32
    %11 = scf.if %7#0 -> (f32) {
      %12 = scf.if %9 -> (f32) {
        %13:4 = scf.while (%arg2 = %c1_i64, %arg3 = %arg0, %arg4 = %arg0, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %14 = arith.cmpi slt, %arg2, %0 : i64
          %15 = arith.cmpf uno, %arg4, %arg4 : f32
          %16 = arith.extui %15 : i1 to i8
          %17 = arith.cmpi eq, %16, %c0_i8 : i8
          %18 = arith.andi %14, %17 : i1
          scf.condition(%18) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
        } do {
        ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
          %14 = arith.cmpf uno, %arg4, %arg4 : f32
          %15 = arith.extui %14 : i1 to i8
          %16 = arith.cmpi eq, %15, %c0_i8 : i8
          %17 = arith.select %16, %arg4, %arg5 : f32
          %18 = arith.cmpi slt, %arg3, %0 : i64
          %19 = arith.select %18, %17, %arg5 : f32
          %20 = arith.addi %arg3, %arg3 : i64
          %21 = arith.addi %20, %c1_i64 : i64
          %22 = arith.sitofp %21 : i64 to f32
          %23 = arith.mulf %22, %arg0 : f32
          %24 = arith.mulf %23, %arg4 : f32
          %25 = arith.sitofp %arg3 : i64 to f32
          %26 = arith.mulf %25, %arg5 : f32
          %27 = arith.subf %24, %26 : f32
          %28 = arith.addi %arg3, %c1_i64 : i64
          %29 = arith.sitofp %28 : i64 to f32
          %30 = arith.divf %27, %29 : f32
          scf.yield %28, %30, %30, %19 : i64, f32, f32, f32
        }
        scf.yield %13#0 : f32
      } else {
        scf.yield %10 : f32
      }
      scf.yield %12 : f32
    } else {
      scf.yield %7#1 : f32
    }
    return %11 : f32
  }
  func.func @_Z29legendre_polynomial_p_forwardIfET_S0_l(%arg0: f32, %arg1: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %false = arith.constant false
    %true = arith.constant true
    %c0_i8 = arith.constant 0 : i8
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant -1.000000e+00 : f32
    %c2_i64 = arith.constant 2 : i64
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %2 = arith.cmpi sge, %arg1, %c0_i64 : i64
    %3 = arith.select %1, %cst_1, %0 : f32
    %4 = math.absf %arg0 : f32
    %5 = arith.cmpf oeq, %4, %cst_0 : f32
    %6:2 = scf.if %2 -> (i1, f32) {
      %11:2 = scf.if %5 -> (i1, f32) {
        %12 = arith.cmpf ogt, %arg0, %cst_1 : f32
        %13 = scf.if %12 -> (i1) {
          scf.yield %true : i1
        } else {
          %16 = arith.remsi %arg1, %c2_i64 : i64
          %17 = arith.cmpi eq, %16, %c0_i64 : i64
          scf.yield %17 : i1
        }
        %14 = arith.xori %13, %true : i1
        %15 = arith.select %14, %cst, %cst_0 : f32
        scf.yield %false, %15 : i1, f32
      } else {
        %12 = arith.cmpi eq, %arg1, %c0_i64 : i64
        %13 = arith.cmpi ne, %arg1, %c0_i64 : i64
        %14 = arith.select %12, %cst_0, %3 : f32
        scf.yield %13, %14 : i1, f32
      }
      scf.yield %11#0, %11#1 : i1, f32
    } else {
      scf.yield %false, %3 : i1, f32
    }
    %7 = arith.cmpi eq, %arg1, %c1_i64 : i64
    %8 = arith.cmpi ne, %arg1, %c1_i64 : i64
    %9 = arith.select %7, %arg0, %6#1 : f32
    %10 = scf.if %6#0 -> (f32) {
      %11 = scf.if %8 -> (f32) {
        %12:4 = scf.while (%arg2 = %c1_i64, %arg3 = %arg0, %arg4 = %arg0, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %13 = arith.cmpi slt, %arg2, %arg1 : i64
          %14 = arith.cmpf uno, %arg4, %arg4 : f32
          %15 = arith.extui %14 : i1 to i8
          %16 = arith.cmpi eq, %15, %c0_i8 : i8
          %17 = arith.andi %13, %16 : i1
          scf.condition(%17) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
        } do {
        ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
          %13 = arith.cmpf uno, %arg4, %arg4 : f32
          %14 = arith.extui %13 : i1 to i8
          %15 = arith.cmpi eq, %14, %c0_i8 : i8
          %16 = arith.select %15, %arg4, %arg5 : f32
          %17 = arith.cmpi slt, %arg3, %arg1 : i64
          %18 = arith.select %17, %16, %arg5 : f32
          %19 = arith.addi %arg3, %arg3 : i64
          %20 = arith.addi %19, %c1_i64 : i64
          %21 = arith.sitofp %20 : i64 to f32
          %22 = arith.mulf %21, %arg0 : f32
          %23 = arith.mulf %22, %arg4 : f32
          %24 = arith.sitofp %arg3 : i64 to f32
          %25 = arith.mulf %24, %arg5 : f32
          %26 = arith.subf %23, %25 : f32
          %27 = arith.addi %arg3, %c1_i64 : i64
          %28 = arith.sitofp %27 : i64 to f32
          %29 = arith.divf %26, %28 : f32
          scf.yield %27, %29, %29, %18 : i64, f32, f32, f32
        }
        scf.yield %12#0 : f32
      } else {
        scf.yield %9 : f32
      }
      scf.yield %11 : f32
    } else {
      scf.yield %6#1 : f32
    }
    return %10 : f32
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
