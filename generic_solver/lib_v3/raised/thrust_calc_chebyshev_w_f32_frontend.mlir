module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
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
      %13 = scf.if %11 -> (f32) {
        %31 = arith.cmpf ogt, %2, %cst_3 : f32
        %32:2 = scf.if %31 -> (i1, f32) {
          %34 = arith.addi %4, %4 : i64
          %35 = arith.addi %34, %c1_i64 : i64
          %36 = arith.sitofp %35 : i64 to f32
          scf.yield %false, %36 : i1, f32
        } else {
          %34 = arith.remsi %4, %c2_i64 : i64
          %35 = arith.cmpi eq, %34, %c0_i64 : i64
          %36 = arith.cmpi ne, %34, %c0_i64 : i64
          %37 = arith.select %35, %cst_2, %7 : f32
          scf.yield %36, %37 : i1, f32
        }
        %33 = arith.select %32#0, %cst_1, %32#1 : f32
        scf.yield %33 : f32
      } else {
        scf.yield %7 : f32
      }
      %14 = arith.cmpi sgt, %4, %c8_i64 : i64
      %15 = arith.andi %12, %14 : i1
      %16 = arith.cmpf olt, %8, %cst_2 : f32
      %17 = arith.xori %16, %true : i1
      %18 = arith.andi %17, %12 : i1
      %19 = arith.andi %15, %16 : i1
      %20 = arith.xori %15, %true : i1
      %21 = arith.andi %15, %18 : i1
      %22 = arith.andi %20, %12 : i1
      %23 = arith.ori %21, %22 : i1
      %24 = scf.if %19 -> (f32) {
        %31:2 = scf.if %12 -> (i1, f32) {
          %39 = func.call @acosf(%2) : (f32) -> f32
          %40 = arith.divf %39, %cst_0 : f32
          %41 = math.cos %40 : f32
          %42 = arith.cmpf une, %41, %cst_2 : f32
          %43 = arith.xori %42, %true : i1
          %44 = scf.if %42 -> (f32) {
            %45 = arith.sitofp %4 : i64 to f32
            %46 = arith.addf %45, %cst : f32
            %47 = func.call @acosf(%2) : (f32) -> f32
            %48 = arith.mulf %46, %47 : f32
            %49 = math.sin %48 : f32
            %50 = func.call @acosf(%2) : (f32) -> f32
            %51 = arith.divf %50, %cst_0 : f32
            %52 = math.sin %51 : f32
            %53 = arith.divf %49, %52 : f32
            scf.yield %53 : f32
          } else {
            scf.yield %13 : f32
          }
          scf.yield %43, %44 : i1, f32
        } else {
          scf.yield %false, %13 : i1, f32
        }
        %32 = arith.cmpf ogt, %2, %cst_3 : f32
        %33 = arith.andi %31#0, %32 : i1
        %34 = arith.xori %33, %true : i1
        %35 = arith.andi %34, %31#0 : i1
        %36 = scf.if %33 -> (f32) {
          %39 = arith.addi %4, %4 : i64
          %40 = arith.addi %39, %c1_i64 : i64
          %41 = arith.sitofp %40 : i64 to f32
          scf.yield %41 : f32
        } else {
          scf.yield %31#1 : f32
        }
        %37:2 = scf.if %35 -> (i1, f32) {
          %39 = arith.remsi %4, %c2_i64 : i64
          %40 = arith.cmpi eq, %39, %c0_i64 : i64
          %41 = arith.cmpi ne, %39, %c0_i64 : i64
          %42 = arith.select %40, %cst_2, %36 : f32
          scf.yield %41, %42 : i1, f32
        } else {
          scf.yield %false, %36 : i1, f32
        }
        %38 = arith.select %37#0, %cst_1, %37#1 : f32
        scf.yield %38 : f32
      } else {
        scf.yield %13 : f32
      }
      %25:2 = scf.if %23 -> (i1, f32) {
        %31 = arith.cmpi eq, %4, %c0_i64 : i64
        %32 = arith.cmpi ne, %4, %c0_i64 : i64
        %33 = arith.select %31, %cst_2, %24 : f32
        scf.yield %32, %33 : i1, f32
      } else {
        scf.yield %false, %24 : i1, f32
      }
      %26 = arith.cmpi eq, %4, %c1_i64 : i64
      %27 = arith.andi %25#0, %26 : i1
      %28 = arith.xori %27, %true : i1
      %29 = arith.andi %28, %25#0 : i1
      %30 = scf.if %29 -> (f32) {
        %31 = arith.addf %2, %2 : f32
        %32 = arith.addf %31, %cst_2 : f32
        %33:4 = scf.while (%arg5 = %c2_i64, %arg6 = %32, %arg7 = %32, %arg8 = %cst_2) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
          %34 = arith.cmpi sle, %arg5, %4 : i64
          %35 = arith.cmpf uno, %arg7, %arg7 : f32
          %36 = arith.extui %35 : i1 to i8
          %37 = arith.cmpi eq, %36, %c0_i8 : i8
          %38 = arith.andi %34, %37 : i1
          scf.condition(%38) %arg6, %arg5, %arg7, %arg8 : f32, i64, f32, f32
        } do {
        ^bb0(%arg5: f32, %arg6: i64, %arg7: f32, %arg8: f32):
          %34 = arith.cmpf uno, %arg7, %arg7 : f32
          %35 = arith.extui %34 : i1 to i8
          %36 = arith.cmpi eq, %35, %c0_i8 : i8
          %37 = arith.select %36, %arg7, %arg8 : f32
          %38 = arith.cmpi sle, %arg6, %4 : i64
          %39 = arith.select %38, %37, %arg8 : f32
          %40 = arith.mulf %31, %arg7 : f32
          %41 = arith.subf %40, %arg8 : f32
          %42 = arith.addi %arg6, %c1_i64 : i64
          scf.yield %42, %41, %41, %39 : i64, f32, f32, f32
        }
        scf.yield %33#0 : f32
      } else {
        %31 = scf.if %27 -> (f32) {
          %32 = arith.addf %2, %2 : f32
          %33 = arith.addf %32, %cst_2 : f32
          scf.yield %33 : f32
        } else {
          scf.yield %25#1 : f32
        }
        scf.yield %31 : f32
      }
      affine.store %30, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @calc_chebyshev_wf(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7 = arith.xori %6, %true : i1
    %8 = arith.andi %3, %6 : i1
    %9 = arith.andi %3, %7 : i1
    %10 = scf.if %8 -> (f32) {
      %28 = arith.cmpf ogt, %arg0, %cst : f32
      %29:2 = scf.if %28 -> (i1, f32) {
        %31 = arith.addi %0, %0 : i64
        %32 = arith.addi %31, %c1_i64 : i64
        %33 = arith.sitofp %32 : i64 to f32
        scf.yield %false, %33 : i1, f32
      } else {
        %31 = arith.remsi %0, %c2_i64 : i64
        %32 = arith.cmpi eq, %31, %c0_i64 : i64
        %33 = arith.cmpi ne, %31, %c0_i64 : i64
        %34 = arith.select %32, %cst_0, %4 : f32
        scf.yield %33, %34 : i1, f32
      }
      %30 = arith.select %29#0, %cst_1, %29#1 : f32
      scf.yield %30 : f32
    } else {
      scf.yield %4 : f32
    }
    %11 = arith.cmpi sgt, %0, %c8_i64 : i64
    %12 = arith.andi %9, %11 : i1
    %13 = arith.cmpf olt, %5, %cst_0 : f32
    %14 = arith.xori %13, %true : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %12, %13 : i1
    %17 = arith.xori %12, %true : i1
    %18 = arith.andi %12, %15 : i1
    %19 = arith.andi %17, %9 : i1
    %20 = arith.ori %18, %19 : i1
    %21 = scf.if %16 -> (f32) {
      %28:2 = scf.if %9 -> (i1, f32) {
        %36 = func.call @acosf(%arg0) : (f32) -> f32
        %37 = arith.divf %36, %cst_2 : f32
        %38 = math.cos %37 : f32
        %39 = arith.cmpf une, %38, %cst_0 : f32
        %40 = arith.xori %39, %true : i1
        %41 = scf.if %39 -> (f32) {
          %42 = arith.sitofp %0 : i64 to f32
          %43 = arith.addf %42, %cst_3 : f32
          %44 = func.call @acosf(%arg0) : (f32) -> f32
          %45 = arith.mulf %43, %44 : f32
          %46 = math.sin %45 : f32
          %47 = func.call @acosf(%arg0) : (f32) -> f32
          %48 = arith.divf %47, %cst_2 : f32
          %49 = math.sin %48 : f32
          %50 = arith.divf %46, %49 : f32
          scf.yield %50 : f32
        } else {
          scf.yield %10 : f32
        }
        scf.yield %40, %41 : i1, f32
      } else {
        scf.yield %false, %10 : i1, f32
      }
      %29 = arith.cmpf ogt, %arg0, %cst : f32
      %30 = arith.andi %28#0, %29 : i1
      %31 = arith.xori %30, %true : i1
      %32 = arith.andi %31, %28#0 : i1
      %33 = scf.if %30 -> (f32) {
        %36 = arith.addi %0, %0 : i64
        %37 = arith.addi %36, %c1_i64 : i64
        %38 = arith.sitofp %37 : i64 to f32
        scf.yield %38 : f32
      } else {
        scf.yield %28#1 : f32
      }
      %34:2 = scf.if %32 -> (i1, f32) {
        %36 = arith.remsi %0, %c2_i64 : i64
        %37 = arith.cmpi eq, %36, %c0_i64 : i64
        %38 = arith.cmpi ne, %36, %c0_i64 : i64
        %39 = arith.select %37, %cst_0, %33 : f32
        scf.yield %38, %39 : i1, f32
      } else {
        scf.yield %false, %33 : i1, f32
      }
      %35 = arith.select %34#0, %cst_1, %34#1 : f32
      scf.yield %35 : f32
    } else {
      scf.yield %10 : f32
    }
    %22:2 = scf.if %20 -> (i1, f32) {
      %28 = arith.cmpi eq, %0, %c0_i64 : i64
      %29 = arith.cmpi ne, %0, %c0_i64 : i64
      %30 = arith.select %28, %cst_0, %21 : f32
      scf.yield %29, %30 : i1, f32
    } else {
      scf.yield %false, %21 : i1, f32
    }
    %23 = arith.cmpi eq, %0, %c1_i64 : i64
    %24 = arith.andi %22#0, %23 : i1
    %25 = arith.xori %24, %true : i1
    %26 = arith.andi %25, %22#0 : i1
    %27 = scf.if %26 -> (f32) {
      %28 = arith.addf %arg0, %arg0 : f32
      %29 = arith.addf %28, %cst_0 : f32
      %30:4 = scf.while (%arg2 = %c2_i64, %arg3 = %29, %arg4 = %29, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %31 = arith.cmpi sle, %arg2, %0 : i64
        %32 = arith.cmpf uno, %arg4, %arg4 : f32
        %33 = arith.extui %32 : i1 to i8
        %34 = arith.cmpi eq, %33, %c0_i8 : i8
        %35 = arith.andi %31, %34 : i1
        scf.condition(%35) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %31 = arith.cmpf uno, %arg4, %arg4 : f32
        %32 = arith.extui %31 : i1 to i8
        %33 = arith.cmpi eq, %32, %c0_i8 : i8
        %34 = arith.select %33, %arg4, %arg5 : f32
        %35 = arith.cmpi sle, %arg3, %0 : i64
        %36 = arith.select %35, %34, %arg5 : f32
        %37 = arith.mulf %28, %arg4 : f32
        %38 = arith.subf %37, %arg5 : f32
        %39 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %39, %38, %38, %36 : i64, f32, f32, f32
      }
      scf.yield %30#0 : f32
    } else {
      %28 = scf.if %24 -> (f32) {
        %29 = arith.addf %arg0, %arg0 : f32
        %30 = arith.addf %29, %cst_0 : f32
        scf.yield %30 : f32
      } else {
        scf.yield %22#1 : f32
      }
      scf.yield %28 : f32
    }
    return %27 : f32
  }
  func.func @_Z30chebyshev_polynomial_w_forwardIfLb0EET_S0_S0_(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
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
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7 = arith.xori %6, %true : i1
    %8 = arith.andi %3, %6 : i1
    %9 = arith.andi %3, %7 : i1
    %10 = scf.if %8 -> (f32) {
      %28 = arith.cmpf ogt, %arg0, %cst : f32
      %29:2 = scf.if %28 -> (i1, f32) {
        %31 = arith.addi %0, %0 : i64
        %32 = arith.addi %31, %c1_i64 : i64
        %33 = arith.sitofp %32 : i64 to f32
        scf.yield %false, %33 : i1, f32
      } else {
        %31 = arith.remsi %0, %c2_i64 : i64
        %32 = arith.cmpi eq, %31, %c0_i64 : i64
        %33 = arith.cmpi ne, %31, %c0_i64 : i64
        %34 = arith.select %32, %cst_0, %4 : f32
        scf.yield %33, %34 : i1, f32
      }
      %30 = arith.select %29#0, %cst_1, %29#1 : f32
      scf.yield %30 : f32
    } else {
      scf.yield %4 : f32
    }
    %11 = arith.cmpi sgt, %0, %c8_i64 : i64
    %12 = arith.andi %9, %11 : i1
    %13 = arith.cmpf olt, %5, %cst_0 : f32
    %14 = arith.xori %13, %true : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %12, %13 : i1
    %17 = arith.xori %12, %true : i1
    %18 = arith.andi %12, %15 : i1
    %19 = arith.andi %17, %9 : i1
    %20 = arith.ori %18, %19 : i1
    %21 = scf.if %16 -> (f32) {
      %28:2 = scf.if %9 -> (i1, f32) {
        %36 = func.call @acosf(%arg0) : (f32) -> f32
        %37 = arith.divf %36, %cst_2 : f32
        %38 = math.cos %37 : f32
        %39 = arith.cmpf une, %38, %cst_0 : f32
        %40 = arith.xori %39, %true : i1
        %41 = scf.if %39 -> (f32) {
          %42 = arith.sitofp %0 : i64 to f32
          %43 = arith.addf %42, %cst_3 : f32
          %44 = func.call @acosf(%arg0) : (f32) -> f32
          %45 = arith.mulf %43, %44 : f32
          %46 = math.sin %45 : f32
          %47 = func.call @acosf(%arg0) : (f32) -> f32
          %48 = arith.divf %47, %cst_2 : f32
          %49 = math.sin %48 : f32
          %50 = arith.divf %46, %49 : f32
          scf.yield %50 : f32
        } else {
          scf.yield %10 : f32
        }
        scf.yield %40, %41 : i1, f32
      } else {
        scf.yield %false, %10 : i1, f32
      }
      %29 = arith.cmpf ogt, %arg0, %cst : f32
      %30 = arith.andi %28#0, %29 : i1
      %31 = arith.xori %30, %true : i1
      %32 = arith.andi %31, %28#0 : i1
      %33 = scf.if %30 -> (f32) {
        %36 = arith.addi %0, %0 : i64
        %37 = arith.addi %36, %c1_i64 : i64
        %38 = arith.sitofp %37 : i64 to f32
        scf.yield %38 : f32
      } else {
        scf.yield %28#1 : f32
      }
      %34:2 = scf.if %32 -> (i1, f32) {
        %36 = arith.remsi %0, %c2_i64 : i64
        %37 = arith.cmpi eq, %36, %c0_i64 : i64
        %38 = arith.cmpi ne, %36, %c0_i64 : i64
        %39 = arith.select %37, %cst_0, %33 : f32
        scf.yield %38, %39 : i1, f32
      } else {
        scf.yield %false, %33 : i1, f32
      }
      %35 = arith.select %34#0, %cst_1, %34#1 : f32
      scf.yield %35 : f32
    } else {
      scf.yield %10 : f32
    }
    %22:2 = scf.if %20 -> (i1, f32) {
      %28 = arith.cmpi eq, %0, %c0_i64 : i64
      %29 = arith.cmpi ne, %0, %c0_i64 : i64
      %30 = arith.select %28, %cst_0, %21 : f32
      scf.yield %29, %30 : i1, f32
    } else {
      scf.yield %false, %21 : i1, f32
    }
    %23 = arith.cmpi eq, %0, %c1_i64 : i64
    %24 = arith.andi %22#0, %23 : i1
    %25 = arith.xori %24, %true : i1
    %26 = arith.andi %25, %22#0 : i1
    %27 = scf.if %26 -> (f32) {
      %28 = arith.addf %arg0, %arg0 : f32
      %29 = arith.addf %28, %cst_0 : f32
      %30:4 = scf.while (%arg2 = %c2_i64, %arg3 = %29, %arg4 = %29, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %31 = arith.cmpi sle, %arg2, %0 : i64
        %32 = arith.cmpf uno, %arg4, %arg4 : f32
        %33 = arith.extui %32 : i1 to i8
        %34 = arith.cmpi eq, %33, %c0_i8 : i8
        %35 = arith.andi %31, %34 : i1
        scf.condition(%35) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %31 = arith.cmpf uno, %arg4, %arg4 : f32
        %32 = arith.extui %31 : i1 to i8
        %33 = arith.cmpi eq, %32, %c0_i8 : i8
        %34 = arith.select %33, %arg4, %arg5 : f32
        %35 = arith.cmpi sle, %arg3, %0 : i64
        %36 = arith.select %35, %34, %arg5 : f32
        %37 = arith.mulf %28, %arg4 : f32
        %38 = arith.subf %37, %arg5 : f32
        %39 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %39, %38, %38, %36 : i64, f32, f32, f32
      }
      scf.yield %30#0 : f32
    } else {
      %28 = scf.if %24 -> (f32) {
        %29 = arith.addf %arg0, %arg0 : f32
        %30 = arith.addf %29, %cst_0 : f32
        scf.yield %30 : f32
      } else {
        scf.yield %22#1 : f32
      }
      scf.yield %28 : f32
    }
    return %27 : f32
  }
  func.func @_Z30chebyshev_polynomial_w_forwardIfET_S0_l(%arg0: f32, %arg1: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
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
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %2 = arith.cmpi sge, %arg1, %c0_i64 : i64
    %3 = arith.select %1, %cst_3, %0 : f32
    %4 = math.absf %arg0 : f32
    %5 = arith.cmpf oeq, %4, %cst_2 : f32
    %6 = arith.xori %5, %true : i1
    %7 = arith.andi %2, %5 : i1
    %8 = arith.andi %2, %6 : i1
    %9 = scf.if %7 -> (f32) {
      %27 = arith.cmpf ogt, %arg0, %cst_3 : f32
      %28:2 = scf.if %27 -> (i1, f32) {
        %30 = arith.addi %arg1, %arg1 : i64
        %31 = arith.addi %30, %c1_i64 : i64
        %32 = arith.sitofp %31 : i64 to f32
        scf.yield %false, %32 : i1, f32
      } else {
        %30 = arith.remsi %arg1, %c2_i64 : i64
        %31 = arith.cmpi eq, %30, %c0_i64 : i64
        %32 = arith.cmpi ne, %30, %c0_i64 : i64
        %33 = arith.select %31, %cst_2, %3 : f32
        scf.yield %32, %33 : i1, f32
      }
      %29 = arith.select %28#0, %cst_1, %28#1 : f32
      scf.yield %29 : f32
    } else {
      scf.yield %3 : f32
    }
    %10 = arith.cmpi sgt, %arg1, %c8_i64 : i64
    %11 = arith.andi %8, %10 : i1
    %12 = arith.cmpf olt, %4, %cst_2 : f32
    %13 = arith.xori %12, %true : i1
    %14 = arith.andi %13, %8 : i1
    %15 = arith.andi %11, %12 : i1
    %16 = arith.xori %11, %true : i1
    %17 = arith.andi %11, %14 : i1
    %18 = arith.andi %16, %8 : i1
    %19 = arith.ori %17, %18 : i1
    %20 = scf.if %15 -> (f32) {
      %27:2 = scf.if %8 -> (i1, f32) {
        %35 = func.call @acosf(%arg0) : (f32) -> f32
        %36 = arith.divf %35, %cst_0 : f32
        %37 = math.cos %36 : f32
        %38 = arith.cmpf une, %37, %cst_2 : f32
        %39 = arith.xori %38, %true : i1
        %40 = scf.if %38 -> (f32) {
          %41 = arith.sitofp %arg1 : i64 to f32
          %42 = arith.addf %41, %cst : f32
          %43 = func.call @acosf(%arg0) : (f32) -> f32
          %44 = arith.mulf %42, %43 : f32
          %45 = math.sin %44 : f32
          %46 = func.call @acosf(%arg0) : (f32) -> f32
          %47 = arith.divf %46, %cst_0 : f32
          %48 = math.sin %47 : f32
          %49 = arith.divf %45, %48 : f32
          scf.yield %49 : f32
        } else {
          scf.yield %9 : f32
        }
        scf.yield %39, %40 : i1, f32
      } else {
        scf.yield %false, %9 : i1, f32
      }
      %28 = arith.cmpf ogt, %arg0, %cst_3 : f32
      %29 = arith.andi %27#0, %28 : i1
      %30 = arith.xori %29, %true : i1
      %31 = arith.andi %30, %27#0 : i1
      %32 = scf.if %29 -> (f32) {
        %35 = arith.addi %arg1, %arg1 : i64
        %36 = arith.addi %35, %c1_i64 : i64
        %37 = arith.sitofp %36 : i64 to f32
        scf.yield %37 : f32
      } else {
        scf.yield %27#1 : f32
      }
      %33:2 = scf.if %31 -> (i1, f32) {
        %35 = arith.remsi %arg1, %c2_i64 : i64
        %36 = arith.cmpi eq, %35, %c0_i64 : i64
        %37 = arith.cmpi ne, %35, %c0_i64 : i64
        %38 = arith.select %36, %cst_2, %32 : f32
        scf.yield %37, %38 : i1, f32
      } else {
        scf.yield %false, %32 : i1, f32
      }
      %34 = arith.select %33#0, %cst_1, %33#1 : f32
      scf.yield %34 : f32
    } else {
      scf.yield %9 : f32
    }
    %21:2 = scf.if %19 -> (i1, f32) {
      %27 = arith.cmpi eq, %arg1, %c0_i64 : i64
      %28 = arith.cmpi ne, %arg1, %c0_i64 : i64
      %29 = arith.select %27, %cst_2, %20 : f32
      scf.yield %28, %29 : i1, f32
    } else {
      scf.yield %false, %20 : i1, f32
    }
    %22 = arith.cmpi eq, %arg1, %c1_i64 : i64
    %23 = arith.andi %21#0, %22 : i1
    %24 = arith.xori %23, %true : i1
    %25 = arith.andi %24, %21#0 : i1
    %26 = scf.if %25 -> (f32) {
      %27 = arith.addf %arg0, %arg0 : f32
      %28 = arith.addf %27, %cst_2 : f32
      %29:4 = scf.while (%arg2 = %c2_i64, %arg3 = %28, %arg4 = %28, %arg5 = %cst_2) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %30 = arith.cmpi sle, %arg2, %arg1 : i64
        %31 = arith.cmpf uno, %arg4, %arg4 : f32
        %32 = arith.extui %31 : i1 to i8
        %33 = arith.cmpi eq, %32, %c0_i8 : i8
        %34 = arith.andi %30, %33 : i1
        scf.condition(%34) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %30 = arith.cmpf uno, %arg4, %arg4 : f32
        %31 = arith.extui %30 : i1 to i8
        %32 = arith.cmpi eq, %31, %c0_i8 : i8
        %33 = arith.select %32, %arg4, %arg5 : f32
        %34 = arith.cmpi sle, %arg3, %arg1 : i64
        %35 = arith.select %34, %33, %arg5 : f32
        %36 = arith.mulf %27, %arg4 : f32
        %37 = arith.subf %36, %arg5 : f32
        %38 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %38, %37, %37, %35 : i64, f32, f32, f32
      }
      scf.yield %29#0 : f32
    } else {
      %27 = scf.if %23 -> (f32) {
        %28 = arith.addf %arg0, %arg0 : f32
        %29 = arith.addf %28, %cst_2 : f32
        scf.yield %29 : f32
      } else {
        scf.yield %21#1 : f32
      }
      scf.yield %27 : f32
    }
    return %26 : f32
  }
  func.func @_ZSt3absf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.absf %arg0 : f32
    return %0 : f32
  }
  func.func @_ZSt4acosf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = call @acosf(%arg0) : (f32) -> f32
    return %0 : f32
  }
  func.func @_ZSt5isnanf(%arg0: f32) -> i8 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = arith.cmpf uno, %arg0, %arg0 : f32
    %1 = arith.extui %0 : i1 to i8
    return %1 : i8
  }
  func.func private @acosf(f32) -> f32
}
