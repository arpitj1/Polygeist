module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1B" : memref<25xf32> = dense<[7.51729664E-18, 4.41434819E-18, -4.65030551E-17, -3.20952596E-17, 2.96262902E-16, 3.30820232E-16, -1.8803547E-15, -3.81440324E-15, 1.04202772E-14, 4.272440e-14, -2.10154177E-14, -4.08355099E-13, -7.19855165E-13, 2.03562861E-12, 1.41258073E-11, 3.25260374E-11, -1.89749588E-11, -5.58974367E-10, -3.83538046E-9, -2.63146891E-8, -2.51223611E-7, -3.88256467E-6, -1.10588939E-4, -0.00976109784, 0.778576254]>
  memref.global constant @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1A" : memref<29xf32> = dense<[2.77791403E-18, -2.1114212E-17, 1.55363193E-16, -1.10559695E-15, 7.60068413E-15, -5.04218554E-14, 3.2237933E-13, -1.9839744E-12, 1.17361867E-11, -6.663490e-11, 3.62559038E-10, -1.88724969E-9, 9.38153732E-9, -4.44505908E-8, 2.00329481E-7, -8.568720e-07, 3.47025139E-6, -1.32731639E-5, 4.78156508E-5, -1.61760821E-4, 5.122860e-04, -0.00151357241, 0.0041564228, -0.0105640851, 0.0247264486, -0.0529459827, 0.102643661, -0.176416516, 0.252587199]>
  func.func @thrust_modified_bessel_i1_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %cst_3 = arith.constant 3.200000e+01 : f32
    %cst_4 = arith.constant 7.51729664E-18 : f32
    %cst_5 = arith.constant 2.77791403E-18 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1B" : memref<25xf32>
    %2 = memref.get_global @"_Z26modified_bessel_i1_forwardIfET_S0_@static@_ZZ26modified_bessel_i1_forwardIfET_S0_E1A" : memref<29xf32>
    affine.for %arg3 = 0 to %0 {
      %3 = affine.load %arg1[%arg3] : memref<?xf32>
      %4 = math.absf %3 : f32
      %5 = arith.cmpf ole, %4, %cst_0 : f32
      %6 = scf.if %5 -> (f32) {
        %7 = arith.divf %4, %cst_1 : f32
        %8 = arith.subf %7, %cst_1 : f32
        %9:3 = affine.for %arg4 = 1 to 29 iter_args(%arg5 = %cst_5, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
          %12 = arith.index_cast %arg4 : index to i8
          %13 = arith.mulf %8, %arg5 : f32
          %14 = arith.subf %13, %arg6 : f32
          %15 = arith.index_castui %12 : i8 to index
          %16 = memref.load %2[%15] : memref<29xf32>
          %17 = arith.addf %14, %16 : f32
          affine.yield %17, %arg5, %arg6 : f32, f32, f32
        }
        %10 = arith.cmpf olt, %3, %cst : f32
        %11 = scf.if %10 -> (f32) {
          %12 = arith.subf %9#0, %9#2 : f32
          %13 = arith.mulf %12, %cst_2 : f32
          %14 = arith.mulf %13, %4 : f32
          %15 = math.exp %4 : f32
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.negf %16 : f32
          scf.yield %17 : f32
        } else {
          %12 = arith.subf %9#0, %9#2 : f32
          %13 = arith.mulf %12, %cst_2 : f32
          %14 = arith.mulf %13, %4 : f32
          %15 = math.exp %4 : f32
          %16 = arith.mulf %14, %15 : f32
          scf.yield %16 : f32
        }
        scf.yield %11 : f32
      } else {
        %7 = arith.divf %cst_3, %4 : f32
        %8 = arith.subf %7, %cst_1 : f32
        %9:3 = affine.for %arg4 = 1 to 25 iter_args(%arg5 = %cst_4, %arg6 = %cst, %arg7 = %cst) -> (f32, f32, f32) {
          %12 = arith.index_cast %arg4 : index to i8
          %13 = arith.mulf %8, %arg5 : f32
          %14 = arith.subf %13, %arg6 : f32
          %15 = arith.index_castui %12 : i8 to index
          %16 = memref.load %1[%15] : memref<25xf32>
          %17 = arith.addf %14, %16 : f32
          affine.yield %17, %arg5, %arg6 : f32, f32, f32
        }
        %10 = arith.cmpf olt, %3, %cst : f32
        %11 = scf.if %10 -> (f32) {
          %12 = math.exp %4 : f32
          %13 = arith.subf %9#0, %9#2 : f32
          %14 = arith.mulf %13, %cst_2 : f32
          %15 = arith.mulf %12, %14 : f32
          %16 = math.sqrt %4 : f32
          %17 = arith.divf %15, %16 : f32
          %18 = arith.negf %17 : f32
          scf.yield %18 : f32
        } else {
          %12 = math.exp %4 : f32
          %13 = arith.subf %9#0, %9#2 : f32
          %14 = arith.mulf %13, %cst_2 : f32
          %15 = arith.mulf %12, %14 : f32
          %16 = math.sqrt %4 : f32
          %17 = arith.divf %15, %16 : f32
          scf.yield %17 : f32
        }
        scf.yield %11 : f32
      }
      affine.store %6, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

