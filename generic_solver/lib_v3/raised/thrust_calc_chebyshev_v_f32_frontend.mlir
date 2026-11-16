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
      affine.store %38, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @calc_chebyshev_vf(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7 = arith.xori %6, %true : i1
    %8 = arith.cmpf ogt, %arg0, %cst : f32
    %9 = arith.xori %8, %true : i1
    %10 = arith.select %8, %cst_0, %4 : f32
    %11 = arith.remsi %0, %c2_i64 : i64
    %12 = arith.cmpi eq, %11, %c0_i64 : i64
    %13 = arith.andi %9, %12 : i1
    %14 = arith.xori %13, %true : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %3, %6 : i1
    %17 = arith.andi %3, %7 : i1
    %18 = scf.if %16 -> (f32) {
      %36 = scf.if %15 -> (f32) {
        %37 = arith.addi %0, %0 : i64
        %38 = arith.subi %c-1_i64, %37 : i64
        %39 = arith.sitofp %38 : i64 to f32
        scf.yield %39 : f32
      } else {
        %37 = scf.if %13 -> (f32) {
          %38 = arith.addi %0, %0 : i64
          %39 = arith.addi %38, %c1_i64 : i64
          %40 = arith.sitofp %39 : i64 to f32
          scf.yield %40 : f32
        } else {
          scf.yield %10 : f32
        }
        scf.yield %37 : f32
      }
      scf.yield %36 : f32
    } else {
      scf.yield %4 : f32
    }
    %19 = arith.cmpi sgt, %0, %c8_i64 : i64
    %20 = arith.andi %17, %19 : i1
    %21 = arith.cmpf olt, %5, %cst_0 : f32
    %22 = arith.xori %21, %true : i1
    %23 = arith.andi %22, %17 : i1
    %24 = arith.andi %20, %21 : i1
    %25 = arith.xori %20, %true : i1
    %26 = arith.andi %20, %23 : i1
    %27 = arith.andi %25, %17 : i1
    %28 = arith.ori %26, %27 : i1
    %29 = scf.if %24 -> (f32) {
      %36:2 = scf.if %17 -> (i1, f32) {
        %41 = func.call @acosf(%arg0) : (f32) -> f32
        %42 = arith.divf %41, %cst_1 : f32
        %43 = math.sin %42 : f32
        %44 = arith.cmpf une, %43, %cst_0 : f32
        %45 = arith.xori %44, %true : i1
        %46 = scf.if %44 -> (f32) {
          %47 = arith.sitofp %0 : i64 to f32
          %48 = arith.addf %47, %cst_2 : f32
          %49 = func.call @acosf(%arg0) : (f32) -> f32
          %50 = arith.mulf %48, %49 : f32
          %51 = math.cos %50 : f32
          %52 = func.call @acosf(%arg0) : (f32) -> f32
          %53 = arith.divf %52, %cst_1 : f32
          %54 = math.cos %53 : f32
          %55 = arith.divf %51, %54 : f32
          scf.yield %55 : f32
        } else {
          scf.yield %18 : f32
        }
        scf.yield %45, %46 : i1, f32
      } else {
        scf.yield %false, %18 : i1, f32
      }
      %37 = arith.andi %36#0, %12 : i1
      %38 = arith.xori %37, %true : i1
      %39 = arith.andi %38, %36#0 : i1
      %40 = scf.if %39 -> (f32) {
        %41 = arith.addi %0, %0 : i64
        %42 = arith.subi %c-1_i64, %41 : i64
        %43 = arith.sitofp %42 : i64 to f32
        scf.yield %43 : f32
      } else {
        %41 = scf.if %37 -> (f32) {
          %42 = arith.addi %0, %0 : i64
          %43 = arith.addi %42, %c1_i64 : i64
          %44 = arith.sitofp %43 : i64 to f32
          scf.yield %44 : f32
        } else {
          scf.yield %36#1 : f32
        }
        scf.yield %41 : f32
      }
      scf.yield %40 : f32
    } else {
      scf.yield %18 : f32
    }
    %30:2 = scf.if %28 -> (i1, f32) {
      %36 = arith.cmpi eq, %0, %c0_i64 : i64
      %37 = arith.cmpi ne, %0, %c0_i64 : i64
      %38 = arith.select %36, %cst_0, %29 : f32
      scf.yield %37, %38 : i1, f32
    } else {
      scf.yield %false, %29 : i1, f32
    }
    %31 = arith.cmpi eq, %0, %c1_i64 : i64
    %32 = arith.andi %30#0, %31 : i1
    %33 = arith.xori %32, %true : i1
    %34 = arith.andi %33, %30#0 : i1
    %35 = scf.if %34 -> (f32) {
      %36 = arith.addf %arg0, %arg0 : f32
      %37 = arith.subf %36, %cst_0 : f32
      %38:4 = scf.while (%arg2 = %c2_i64, %arg3 = %37, %arg4 = %37, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %39 = arith.cmpi sle, %arg2, %0 : i64
        %40 = arith.cmpf uno, %arg4, %arg4 : f32
        %41 = arith.extui %40 : i1 to i8
        %42 = arith.cmpi eq, %41, %c0_i8 : i8
        %43 = arith.andi %39, %42 : i1
        scf.condition(%43) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %39 = arith.cmpf uno, %arg4, %arg4 : f32
        %40 = arith.extui %39 : i1 to i8
        %41 = arith.cmpi eq, %40, %c0_i8 : i8
        %42 = arith.select %41, %arg4, %arg5 : f32
        %43 = arith.cmpi sle, %arg3, %0 : i64
        %44 = arith.select %43, %42, %arg5 : f32
        %45 = arith.mulf %36, %arg4 : f32
        %46 = arith.subf %45, %arg5 : f32
        %47 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %47, %46, %46, %44 : i64, f32, f32, f32
      }
      scf.yield %38#0 : f32
    } else {
      %36 = scf.if %32 -> (f32) {
        %37 = arith.addf %arg0, %arg0 : f32
        %38 = arith.subf %37, %cst_0 : f32
        scf.yield %38 : f32
      } else {
        scf.yield %30#1 : f32
      }
      scf.yield %36 : f32
    }
    return %35 : f32
  }
  func.func @_Z30chebyshev_polynomial_v_forwardIfLb0EET_S0_S0_(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
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
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %0, %c0_i64 : i64
    %3 = arith.cmpi sge, %0, %c0_i64 : i64
    %4 = arith.select %2, %cst, %1 : f32
    %5 = math.absf %arg0 : f32
    %6 = arith.cmpf oeq, %5, %cst_0 : f32
    %7 = arith.xori %6, %true : i1
    %8 = arith.cmpf ogt, %arg0, %cst : f32
    %9 = arith.xori %8, %true : i1
    %10 = arith.select %8, %cst_0, %4 : f32
    %11 = arith.remsi %0, %c2_i64 : i64
    %12 = arith.cmpi eq, %11, %c0_i64 : i64
    %13 = arith.andi %9, %12 : i1
    %14 = arith.xori %13, %true : i1
    %15 = arith.andi %14, %9 : i1
    %16 = arith.andi %3, %6 : i1
    %17 = arith.andi %3, %7 : i1
    %18 = scf.if %16 -> (f32) {
      %36 = scf.if %15 -> (f32) {
        %37 = arith.addi %0, %0 : i64
        %38 = arith.subi %c-1_i64, %37 : i64
        %39 = arith.sitofp %38 : i64 to f32
        scf.yield %39 : f32
      } else {
        %37 = scf.if %13 -> (f32) {
          %38 = arith.addi %0, %0 : i64
          %39 = arith.addi %38, %c1_i64 : i64
          %40 = arith.sitofp %39 : i64 to f32
          scf.yield %40 : f32
        } else {
          scf.yield %10 : f32
        }
        scf.yield %37 : f32
      }
      scf.yield %36 : f32
    } else {
      scf.yield %4 : f32
    }
    %19 = arith.cmpi sgt, %0, %c8_i64 : i64
    %20 = arith.andi %17, %19 : i1
    %21 = arith.cmpf olt, %5, %cst_0 : f32
    %22 = arith.xori %21, %true : i1
    %23 = arith.andi %22, %17 : i1
    %24 = arith.andi %20, %21 : i1
    %25 = arith.xori %20, %true : i1
    %26 = arith.andi %20, %23 : i1
    %27 = arith.andi %25, %17 : i1
    %28 = arith.ori %26, %27 : i1
    %29 = scf.if %24 -> (f32) {
      %36:2 = scf.if %17 -> (i1, f32) {
        %41 = func.call @acosf(%arg0) : (f32) -> f32
        %42 = arith.divf %41, %cst_1 : f32
        %43 = math.sin %42 : f32
        %44 = arith.cmpf une, %43, %cst_0 : f32
        %45 = arith.xori %44, %true : i1
        %46 = scf.if %44 -> (f32) {
          %47 = arith.sitofp %0 : i64 to f32
          %48 = arith.addf %47, %cst_2 : f32
          %49 = func.call @acosf(%arg0) : (f32) -> f32
          %50 = arith.mulf %48, %49 : f32
          %51 = math.cos %50 : f32
          %52 = func.call @acosf(%arg0) : (f32) -> f32
          %53 = arith.divf %52, %cst_1 : f32
          %54 = math.cos %53 : f32
          %55 = arith.divf %51, %54 : f32
          scf.yield %55 : f32
        } else {
          scf.yield %18 : f32
        }
        scf.yield %45, %46 : i1, f32
      } else {
        scf.yield %false, %18 : i1, f32
      }
      %37 = arith.andi %36#0, %12 : i1
      %38 = arith.xori %37, %true : i1
      %39 = arith.andi %38, %36#0 : i1
      %40 = scf.if %39 -> (f32) {
        %41 = arith.addi %0, %0 : i64
        %42 = arith.subi %c-1_i64, %41 : i64
        %43 = arith.sitofp %42 : i64 to f32
        scf.yield %43 : f32
      } else {
        %41 = scf.if %37 -> (f32) {
          %42 = arith.addi %0, %0 : i64
          %43 = arith.addi %42, %c1_i64 : i64
          %44 = arith.sitofp %43 : i64 to f32
          scf.yield %44 : f32
        } else {
          scf.yield %36#1 : f32
        }
        scf.yield %41 : f32
      }
      scf.yield %40 : f32
    } else {
      scf.yield %18 : f32
    }
    %30:2 = scf.if %28 -> (i1, f32) {
      %36 = arith.cmpi eq, %0, %c0_i64 : i64
      %37 = arith.cmpi ne, %0, %c0_i64 : i64
      %38 = arith.select %36, %cst_0, %29 : f32
      scf.yield %37, %38 : i1, f32
    } else {
      scf.yield %false, %29 : i1, f32
    }
    %31 = arith.cmpi eq, %0, %c1_i64 : i64
    %32 = arith.andi %30#0, %31 : i1
    %33 = arith.xori %32, %true : i1
    %34 = arith.andi %33, %30#0 : i1
    %35 = scf.if %34 -> (f32) {
      %36 = arith.addf %arg0, %arg0 : f32
      %37 = arith.subf %36, %cst_0 : f32
      %38:4 = scf.while (%arg2 = %c2_i64, %arg3 = %37, %arg4 = %37, %arg5 = %cst_0) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %39 = arith.cmpi sle, %arg2, %0 : i64
        %40 = arith.cmpf uno, %arg4, %arg4 : f32
        %41 = arith.extui %40 : i1 to i8
        %42 = arith.cmpi eq, %41, %c0_i8 : i8
        %43 = arith.andi %39, %42 : i1
        scf.condition(%43) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %39 = arith.cmpf uno, %arg4, %arg4 : f32
        %40 = arith.extui %39 : i1 to i8
        %41 = arith.cmpi eq, %40, %c0_i8 : i8
        %42 = arith.select %41, %arg4, %arg5 : f32
        %43 = arith.cmpi sle, %arg3, %0 : i64
        %44 = arith.select %43, %42, %arg5 : f32
        %45 = arith.mulf %36, %arg4 : f32
        %46 = arith.subf %45, %arg5 : f32
        %47 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %47, %46, %46, %44 : i64, f32, f32, f32
      }
      scf.yield %38#0 : f32
    } else {
      %36 = scf.if %32 -> (f32) {
        %37 = arith.addf %arg0, %arg0 : f32
        %38 = arith.subf %37, %cst_0 : f32
        scf.yield %38 : f32
      } else {
        scf.yield %30#1 : f32
      }
      scf.yield %36 : f32
    }
    return %35 : f32
  }
  func.func @_Z30chebyshev_polynomial_v_forwardIfET_S0_l(%arg0: f32, %arg1: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
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
    %0 = llvm.mlir.undef : f32
    %1 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %2 = arith.cmpi sge, %arg1, %c0_i64 : i64
    %3 = arith.select %1, %cst_2, %0 : f32
    %4 = math.absf %arg0 : f32
    %5 = arith.cmpf oeq, %4, %cst_1 : f32
    %6 = arith.xori %5, %true : i1
    %7 = arith.cmpf ogt, %arg0, %cst_2 : f32
    %8 = arith.xori %7, %true : i1
    %9 = arith.select %7, %cst_1, %3 : f32
    %10 = arith.remsi %arg1, %c2_i64 : i64
    %11 = arith.cmpi eq, %10, %c0_i64 : i64
    %12 = arith.andi %8, %11 : i1
    %13 = arith.xori %12, %true : i1
    %14 = arith.andi %13, %8 : i1
    %15 = arith.andi %2, %5 : i1
    %16 = arith.andi %2, %6 : i1
    %17 = scf.if %15 -> (f32) {
      %35 = scf.if %14 -> (f32) {
        %36 = arith.addi %arg1, %arg1 : i64
        %37 = arith.subi %c-1_i64, %36 : i64
        %38 = arith.sitofp %37 : i64 to f32
        scf.yield %38 : f32
      } else {
        %36 = scf.if %12 -> (f32) {
          %37 = arith.addi %arg1, %arg1 : i64
          %38 = arith.addi %37, %c1_i64 : i64
          %39 = arith.sitofp %38 : i64 to f32
          scf.yield %39 : f32
        } else {
          scf.yield %9 : f32
        }
        scf.yield %36 : f32
      }
      scf.yield %35 : f32
    } else {
      scf.yield %3 : f32
    }
    %18 = arith.cmpi sgt, %arg1, %c8_i64 : i64
    %19 = arith.andi %16, %18 : i1
    %20 = arith.cmpf olt, %4, %cst_1 : f32
    %21 = arith.xori %20, %true : i1
    %22 = arith.andi %21, %16 : i1
    %23 = arith.andi %19, %20 : i1
    %24 = arith.xori %19, %true : i1
    %25 = arith.andi %19, %22 : i1
    %26 = arith.andi %24, %16 : i1
    %27 = arith.ori %25, %26 : i1
    %28 = scf.if %23 -> (f32) {
      %35:2 = scf.if %16 -> (i1, f32) {
        %40 = func.call @acosf(%arg0) : (f32) -> f32
        %41 = arith.divf %40, %cst_0 : f32
        %42 = math.sin %41 : f32
        %43 = arith.cmpf une, %42, %cst_1 : f32
        %44 = arith.xori %43, %true : i1
        %45 = scf.if %43 -> (f32) {
          %46 = arith.sitofp %arg1 : i64 to f32
          %47 = arith.addf %46, %cst : f32
          %48 = func.call @acosf(%arg0) : (f32) -> f32
          %49 = arith.mulf %47, %48 : f32
          %50 = math.cos %49 : f32
          %51 = func.call @acosf(%arg0) : (f32) -> f32
          %52 = arith.divf %51, %cst_0 : f32
          %53 = math.cos %52 : f32
          %54 = arith.divf %50, %53 : f32
          scf.yield %54 : f32
        } else {
          scf.yield %17 : f32
        }
        scf.yield %44, %45 : i1, f32
      } else {
        scf.yield %false, %17 : i1, f32
      }
      %36 = arith.andi %35#0, %11 : i1
      %37 = arith.xori %36, %true : i1
      %38 = arith.andi %37, %35#0 : i1
      %39 = scf.if %38 -> (f32) {
        %40 = arith.addi %arg1, %arg1 : i64
        %41 = arith.subi %c-1_i64, %40 : i64
        %42 = arith.sitofp %41 : i64 to f32
        scf.yield %42 : f32
      } else {
        %40 = scf.if %36 -> (f32) {
          %41 = arith.addi %arg1, %arg1 : i64
          %42 = arith.addi %41, %c1_i64 : i64
          %43 = arith.sitofp %42 : i64 to f32
          scf.yield %43 : f32
        } else {
          scf.yield %35#1 : f32
        }
        scf.yield %40 : f32
      }
      scf.yield %39 : f32
    } else {
      scf.yield %17 : f32
    }
    %29:2 = scf.if %27 -> (i1, f32) {
      %35 = arith.cmpi eq, %arg1, %c0_i64 : i64
      %36 = arith.cmpi ne, %arg1, %c0_i64 : i64
      %37 = arith.select %35, %cst_1, %28 : f32
      scf.yield %36, %37 : i1, f32
    } else {
      scf.yield %false, %28 : i1, f32
    }
    %30 = arith.cmpi eq, %arg1, %c1_i64 : i64
    %31 = arith.andi %29#0, %30 : i1
    %32 = arith.xori %31, %true : i1
    %33 = arith.andi %32, %29#0 : i1
    %34 = scf.if %33 -> (f32) {
      %35 = arith.addf %arg0, %arg0 : f32
      %36 = arith.subf %35, %cst_1 : f32
      %37:4 = scf.while (%arg2 = %c2_i64, %arg3 = %36, %arg4 = %36, %arg5 = %cst_1) : (i64, f32, f32, f32) -> (f32, i64, f32, f32) {
        %38 = arith.cmpi sle, %arg2, %arg1 : i64
        %39 = arith.cmpf uno, %arg4, %arg4 : f32
        %40 = arith.extui %39 : i1 to i8
        %41 = arith.cmpi eq, %40, %c0_i8 : i8
        %42 = arith.andi %38, %41 : i1
        scf.condition(%42) %arg3, %arg2, %arg4, %arg5 : f32, i64, f32, f32
      } do {
      ^bb0(%arg2: f32, %arg3: i64, %arg4: f32, %arg5: f32):
        %38 = arith.cmpf uno, %arg4, %arg4 : f32
        %39 = arith.extui %38 : i1 to i8
        %40 = arith.cmpi eq, %39, %c0_i8 : i8
        %41 = arith.select %40, %arg4, %arg5 : f32
        %42 = arith.cmpi sle, %arg3, %arg1 : i64
        %43 = arith.select %42, %41, %arg5 : f32
        %44 = arith.mulf %35, %arg4 : f32
        %45 = arith.subf %44, %arg5 : f32
        %46 = arith.addi %arg3, %c1_i64 : i64
        scf.yield %46, %45, %45, %43 : i64, f32, f32, f32
      }
      scf.yield %37#0 : f32
    } else {
      %35 = scf.if %31 -> (f32) {
        %36 = arith.addf %arg0, %arg0 : f32
        %37 = arith.subf %36, %cst_1 : f32
        scf.yield %37 : f32
      } else {
        scf.yield %29#1 : f32
      }
      scf.yield %35 : f32
    }
    return %34 : f32
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
