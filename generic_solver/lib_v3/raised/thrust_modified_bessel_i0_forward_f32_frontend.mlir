module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32> = dense<[-7.23318078E-18, -4.83050441E-18, 4.46562156E-17, 3.46122279E-17, -2.82762388E-16, -3.42548568E-16, 1.77256012E-15, 3.81168087E-15, -9.55484674E-15, -4.15056918E-14, 1.54008615E-14, 3.85277829E-13, 7.18012455E-13, -1.79417852E-12, -1.32158121E-11, -3.14991644E-11, 1.18891468E-11, 4.94060237E-10, 3.39623196E-9, 2.26666899E-8, 2.04891862E-7, 2.89137051E-6, 6.88975852E-5, 0.00336911646, 0.804490387]>
  memref.global constant @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32> = dense<[-4.41534163E-18, 3.33079461E-17, -2.431280e-16, 1.71539133E-15, -1.16853328E-14, 7.67618526E-14, -4.85644673E-13, 2.95505265E-12, -1.72682632E-11, 9.67580876E-11, -5.18979582E-10, 2.65982369E-9, -1.30002498E-8, 6.04699508E-8, -2.67079372E-7, 1.11738757E-6, -4.41673819E-6, 1.64484482E-5, -5.75419508E-5, 1.88502891E-4, -5.76375576E-4, 0.00163947558, -4.324310e-03, 0.0105464607, -0.0237374157, 0.0493052825, -9.490110e-02, 0.171620905, -0.304682672, 0.676795303]>
  func.func @thrust_modified_bessel_i0_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 3.200000e+01 : f32
    %cst_4 = arith.constant -7.23318078E-18 : f32
    %cst_5 = arith.constant -4.41534163E-18 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %2 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
    affine.for %arg3 = 0 to %0 {
      %3 = affine.load %arg1[%arg3] : memref<?xf32>
      %4 = math.absf %3 : f32
      %5 = arith.cmpf ole, %4, %cst_0 : f32
      %6 = scf.if %5 -> (f32) {
        %7 = arith.divf %4, %cst_1 : f32
        %8 = arith.subf %7, %cst_1 : f32
        %9:3 = affine.for %arg4 = 1 to 30 iter_args(%arg5 = %cst_5, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
          %14 = arith.index_cast %arg4 : index to i8
          %15 = arith.mulf %8, %arg5 : f32
          %16 = arith.subf %15, %arg6 : f32
          %17 = arith.index_castui %14 : i8 to index
          %18 = memref.load %2[%17] : memref<30xf32>
          %19 = arith.addf %16, %18 : f32
          affine.yield %19, %arg5, %arg6 : f32, f32, f32
        }
        %10 = math.exp %4 : f32
        %11 = arith.subf %9#0, %9#2 : f32
        %12 = arith.mulf %11, %cst_2 : f32
        %13 = arith.mulf %10, %12 : f32
        scf.yield %13 : f32
      } else {
        %7 = arith.divf %cst_3, %4 : f32
        %8 = arith.subf %7, %cst_1 : f32
        %9:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
          %16 = arith.index_cast %arg4 : index to i8
          %17 = arith.mulf %8, %arg5 : f32
          %18 = arith.subf %17, %arg6 : f32
          %19 = arith.index_castui %16 : i8 to index
          %20 = memref.load %1[%19] : memref<25xf32>
          %21 = arith.addf %18, %20 : f32
          affine.yield %21, %arg5, %arg6 : f32, f32, f32
        }
        %10 = math.exp %4 : f32
        %11 = arith.subf %9#0, %9#2 : f32
        %12 = arith.mulf %11, %cst_2 : f32
        %13 = arith.mulf %10, %12 : f32
        %14 = math.sqrt %4 : f32
        %15 = arith.divf %13, %14 : f32
        scf.yield %15 : f32
      }
      affine.store %6, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @modified_bessel_i0_forwardf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant -4.41534163E-18 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant 3.200000e+01 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %1 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
    %2 = math.absf %arg0 : f32
    %3 = arith.cmpf ole, %2, %cst_4 : f32
    %4 = scf.if %3 -> (f32) {
      %5 = arith.divf %2, %cst_3 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 30 iter_args(%arg2 = %cst, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %12 = arith.index_cast %arg1 : index to i8
        %13 = arith.mulf %6, %arg2 : f32
        %14 = arith.subf %13, %arg3 : f32
        %15 = arith.index_castui %12 : i8 to index
        %16 = memref.load %1[%15] : memref<30xf32>
        %17 = arith.addf %14, %16 : f32
        affine.yield %17, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      scf.yield %11 : f32
    } else {
      %5 = arith.divf %cst_1, %2 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_0, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %14 = arith.index_cast %arg1 : index to i8
        %15 = arith.mulf %6, %arg2 : f32
        %16 = arith.subf %15, %arg3 : f32
        %17 = arith.index_castui %14 : i8 to index
        %18 = memref.load %0[%17] : memref<25xf32>
        %19 = arith.addf %16, %18 : f32
        affine.yield %19, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      %12 = math.sqrt %2 : f32
      %13 = arith.divf %11, %12 : f32
      scf.yield %13 : f32
    }
    return %4 : f32
  }
  func.func @_Z26modified_bessel_i0_forwardIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant -4.41534163E-18 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant 3.200000e+01 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 2.000000e+00 : f32
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1B" : memref<25xf32>
    %1 = memref.get_global @"_Z26modified_bessel_i0_forwardIfET_S0_@static@_ZZ26modified_bessel_i0_forwardIfET_S0_E1A" : memref<30xf32>
    %2 = math.absf %arg0 : f32
    %3 = arith.cmpf ole, %2, %cst_4 : f32
    %4 = scf.if %3 -> (f32) {
      %5 = arith.divf %2, %cst_3 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 30 iter_args(%arg2 = %cst, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %12 = arith.index_cast %arg1 : index to i8
        %13 = arith.mulf %6, %arg2 : f32
        %14 = arith.subf %13, %arg3 : f32
        %15 = arith.index_castui %12 : i8 to index
        %16 = memref.load %1[%15] : memref<30xf32>
        %17 = arith.addf %14, %16 : f32
        affine.yield %17, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      scf.yield %11 : f32
    } else {
      %5 = arith.divf %cst_1, %2 : f32
      %6 = arith.subf %5, %cst_3 : f32
      %7:3 = affine.for %arg1 = 1 to 25 iter_args(%arg2 = %cst_0, %arg3 = %cst_5, %arg4 = %cst_5) -> (f32, f32, f32) {
        %14 = arith.index_cast %arg1 : index to i8
        %15 = arith.mulf %6, %arg2 : f32
        %16 = arith.subf %15, %arg3 : f32
        %17 = arith.index_castui %14 : i8 to index
        %18 = memref.load %0[%17] : memref<25xf32>
        %19 = arith.addf %16, %18 : f32
        affine.yield %19, %arg2, %arg3 : f32, f32, f32
      }
      %8 = math.exp %2 : f32
      %9 = arith.subf %7#0, %7#2 : f32
      %10 = arith.mulf %9, %cst_2 : f32
      %11 = arith.mulf %8, %10 : f32
      %12 = math.sqrt %2 : f32
      %13 = arith.divf %11, %12 : f32
      scf.yield %13 : f32
    }
    return %4 : f32
  }
  func.func @_ZSt3absf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.absf %arg0 : f32
    return %0 : f32
  }
  func.func @_ZSt4sqrtf(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %0 = math.sqrt %arg0 : f32
    return %0 : f32
  }
}
