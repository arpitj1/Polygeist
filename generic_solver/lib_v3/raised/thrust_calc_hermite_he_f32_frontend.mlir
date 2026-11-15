module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @thrust_calc_hermite_he_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant 0x7FC00000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    affine.for %arg4 = 0 to %0 {
      %2 = affine.load %arg1[%arg4] : memref<?xf32>
      %3 = affine.load %arg2[%arg4] : memref<?xf32>
      %4 = arith.fptosi %3 : f32 to i64
      %5 = arith.index_cast %4 : i64 to index
      %6 = arith.cmpi slt, %4, %c0_i64 : i64
      %7 = arith.cmpi sge, %4, %c0_i64 : i64
      %8 = arith.select %6, %cst_0, %1 : f32
      %9:2 = scf.if %7 -> (i1, f32) {
        %16 = arith.cmpi eq, %4, %c0_i64 : i64
        %17 = arith.cmpi ne, %4, %c0_i64 : i64
        %18 = arith.select %16, %cst_1, %8 : f32
        scf.yield %17, %18 : i1, f32
      } else {
        scf.yield %false, %8 : i1, f32
      }
      %10 = arith.cmpi eq, %4, %c1_i64 : i64
      %11 = arith.cmpi ne, %4, %c1_i64 : i64
      %12 = arith.select %10, %2, %9#1 : f32
      %13 = arith.cmpi sgt, %4, %c128_i64 : i64
      %14 = arith.cmpi sle, %4, %c128_i64 : i64
      %15 = scf.if %9#0 -> (f32) {
        %16 = scf.if %11 -> (f32) {
          %17 = scf.if %14 -> (f32) {
            %18:3 = scf.for %arg5 = %c1 to %5 step %c1 iter_args(%arg6 = %2, %arg7 = %cst_1, %arg8 = %1) -> (f32, f32, f32) {
              %19 = arith.index_cast %arg5 : index to i64
              %20 = arith.mulf %2, %arg6 : f32
              %21 = arith.sitofp %19 : i64 to f32
              %22 = arith.mulf %21, %arg7 : f32
              %23 = arith.subf %20, %22 : f32
              scf.yield %23, %arg6, %23 : f32, f32, f32
            }
            scf.yield %18#2 : f32
          } else {
            %18 = arith.select %13, %cst, %12 : f32
            scf.yield %18 : f32
          }
          scf.yield %17 : f32
        } else {
          scf.yield %12 : f32
        }
        scf.yield %16 : f32
      } else {
        scf.yield %9#1 : f32
      }
      affine.store %15, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
  func.func @calc_hermite_hef(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0x7FC00000 : f32
    %c128_i64 = arith.constant 128 : i64
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = llvm.mlir.undef : f32
    %3 = arith.cmpi slt, %0, %c0_i64 : i64
    %4 = arith.cmpi sge, %0, %c0_i64 : i64
    %5 = arith.select %3, %cst_0, %2 : f32
    %6:2 = scf.if %4 -> (i1, f32) {
      %13 = arith.cmpi eq, %0, %c0_i64 : i64
      %14 = arith.cmpi ne, %0, %c0_i64 : i64
      %15 = arith.select %13, %cst, %5 : f32
      scf.yield %14, %15 : i1, f32
    } else {
      scf.yield %false, %5 : i1, f32
    }
    %7 = arith.cmpi eq, %0, %c1_i64 : i64
    %8 = arith.cmpi ne, %0, %c1_i64 : i64
    %9 = arith.select %7, %arg0, %6#1 : f32
    %10 = arith.cmpi sgt, %0, %c128_i64 : i64
    %11 = arith.cmpi sle, %0, %c128_i64 : i64
    %12 = scf.if %6#0 -> (f32) {
      %13 = scf.if %8 -> (f32) {
        %14 = scf.if %11 -> (f32) {
          %15:3 = affine.for %arg2 = 1 to %1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %2) -> (f32, f32, f32) {
            %16 = arith.index_cast %arg2 : index to i64
            %17 = arith.mulf %arg0, %arg3 : f32
            %18 = arith.sitofp %16 : i64 to f32
            %19 = arith.mulf %18, %arg4 : f32
            %20 = arith.subf %17, %19 : f32
            affine.yield %20, %arg3, %20 : f32, f32, f32
          }
          scf.yield %15#2 : f32
        } else {
          %15 = arith.select %10, %cst_1, %9 : f32
          scf.yield %15 : f32
        }
        scf.yield %14 : f32
      } else {
        scf.yield %9 : f32
      }
      scf.yield %13 : f32
    } else {
      scf.yield %6#1 : f32
    }
    return %12 : f32
  }
  func.func @_Z29hermite_polynomial_he_forwardIfLb0EET_S0_S0_(%arg0: f32, %arg1: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %false = arith.constant false
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_1 = arith.constant 0x7FC00000 : f32
    %c128_i64 = arith.constant 128 : i64
    %0 = arith.fptosi %arg1 : f32 to i64
    %1 = arith.index_cast %0 : i64 to index
    %2 = llvm.mlir.undef : f32
    %3 = arith.cmpi slt, %0, %c0_i64 : i64
    %4 = arith.cmpi sge, %0, %c0_i64 : i64
    %5 = arith.select %3, %cst_0, %2 : f32
    %6:2 = scf.if %4 -> (i1, f32) {
      %13 = arith.cmpi eq, %0, %c0_i64 : i64
      %14 = arith.cmpi ne, %0, %c0_i64 : i64
      %15 = arith.select %13, %cst, %5 : f32
      scf.yield %14, %15 : i1, f32
    } else {
      scf.yield %false, %5 : i1, f32
    }
    %7 = arith.cmpi eq, %0, %c1_i64 : i64
    %8 = arith.cmpi ne, %0, %c1_i64 : i64
    %9 = arith.select %7, %arg0, %6#1 : f32
    %10 = arith.cmpi sgt, %0, %c128_i64 : i64
    %11 = arith.cmpi sle, %0, %c128_i64 : i64
    %12 = scf.if %6#0 -> (f32) {
      %13 = scf.if %8 -> (f32) {
        %14 = scf.if %11 -> (f32) {
          %15:3 = affine.for %arg2 = 1 to %1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %2) -> (f32, f32, f32) {
            %16 = arith.index_cast %arg2 : index to i64
            %17 = arith.mulf %arg0, %arg3 : f32
            %18 = arith.sitofp %16 : i64 to f32
            %19 = arith.mulf %18, %arg4 : f32
            %20 = arith.subf %17, %19 : f32
            affine.yield %20, %arg3, %20 : f32, f32, f32
          }
          scf.yield %15#2 : f32
        } else {
          %15 = arith.select %10, %cst_1, %9 : f32
          scf.yield %15 : f32
        }
        scf.yield %14 : f32
      } else {
        scf.yield %9 : f32
      }
      scf.yield %13 : f32
    } else {
      scf.yield %6#1 : f32
    }
    return %12 : f32
  }
  func.func @_Z29hermite_polynomial_he_forwardIfET_S0_l(%arg0: f32, %arg1: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c128_i64 = arith.constant 128 : i64
    %cst = arith.constant 0x7FC00000 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %0 = arith.index_cast %arg1 : i64 to index
    %1 = llvm.mlir.undef : f32
    %2 = arith.cmpi slt, %arg1, %c0_i64 : i64
    %3 = arith.cmpi sge, %arg1, %c0_i64 : i64
    %4 = arith.select %2, %cst_0, %1 : f32
    %5:2 = scf.if %3 -> (i1, f32) {
      %12 = arith.cmpi eq, %arg1, %c0_i64 : i64
      %13 = arith.cmpi ne, %arg1, %c0_i64 : i64
      %14 = arith.select %12, %cst_1, %4 : f32
      scf.yield %13, %14 : i1, f32
    } else {
      scf.yield %false, %4 : i1, f32
    }
    %6 = arith.cmpi eq, %arg1, %c1_i64 : i64
    %7 = arith.cmpi ne, %arg1, %c1_i64 : i64
    %8 = arith.select %6, %arg0, %5#1 : f32
    %9 = arith.cmpi sgt, %arg1, %c128_i64 : i64
    %10 = arith.cmpi sle, %arg1, %c128_i64 : i64
    %11 = scf.if %5#0 -> (f32) {
      %12 = scf.if %7 -> (f32) {
        %13 = scf.if %10 -> (f32) {
          %14:3 = affine.for %arg2 = 1 to %0 iter_args(%arg3 = %arg0, %arg4 = %cst_1, %arg5 = %1) -> (f32, f32, f32) {
            %15 = arith.index_cast %arg2 : index to i64
            %16 = arith.mulf %arg0, %arg3 : f32
            %17 = arith.sitofp %15 : i64 to f32
            %18 = arith.mulf %17, %arg4 : f32
            %19 = arith.subf %16, %18 : f32
            affine.yield %19, %arg3, %19 : f32, f32, f32
          }
          scf.yield %14#2 : f32
        } else {
          %14 = arith.select %9, %cst, %8 : f32
          scf.yield %14 : f32
        }
        scf.yield %13 : f32
      } else {
        scf.yield %8 : f32
      }
      scf.yield %12 : f32
    } else {
      scf.yield %5#1 : f32
    }
    return %11 : f32
  }
  func.func @_Z17getHermitianLimitIfEDav() -> i32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %c128_i32 = arith.constant 128 : i32
    return %c128_i32 : i32
  }
  func.func @_ZNSt14numeric_limitsIfE9quiet_NaNEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7FC00000 : f32
    return %cst : f32
  }
}
