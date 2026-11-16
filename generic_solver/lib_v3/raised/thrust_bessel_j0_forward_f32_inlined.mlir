module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RQ" : memref<8xf32> = dense<[499.563141, 173785.406, 0x4C38C9A1, 1.11855534E+10, 2.11277526E+12, 3.10518217E+14, 3.18121957E+16, 1.71086291E+18]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RP" : memref<4xf32> = dense<[-4.794432E+9, 1.95617489E+12, -2.49248344E+14, 9.70862217E+15]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[64.3178253, 856.430054, 3882.40186, 7240.46777, 5930.72705, 2062.09326, 242.005737]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[-0.0113663841, -1.28252721, -19.5539551, -93.2060165, -177.681168, -147.077499, -51.4105339, -6.05014372]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[9.24408785E-4, 0.0856288447, 1.2535274, 5.47097731, 8.76190853, 5.30605268, 1.000000e+00]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.96936743E-4, 0.0828352422, 1.23953366, 5.447250e+00, 8.74716472, 5.3032403, 1.000000e+00]>
  func.func @thrust_bessel_j0_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 5.000000e+00 : f32
    %cst_1 = arith.constant 9.99999974E-6 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant 4.000000e+00 : f32
    %cst_4 = arith.constant 5.78318596 : f32
    %cst_5 = arith.constant 30.471262 : f32
    %cst_6 = arith.constant 2.500000e+01 : f32
    %cst_7 = arith.constant 0.785398185 : f32
    %cst_8 = arith.constant 0.797884583 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RQ" : memref<8xf32>
    %2 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RP" : memref<4xf32>
    %3 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QQ" : memref<7xf32>
    %4 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QP" : memref<8xf32>
    %5 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PQ" : memref<7xf32>
    %6 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PP" : memref<7xf32>
    affine.for %arg3 = 0 to %0 {
      %7 = affine.load %arg1[%arg3] : memref<?xf32>
      %8 = arith.cmpf olt, %7, %cst : f32
      %9 = scf.if %8 -> (f32) {
        %12 = arith.negf %7 : f32
        scf.yield %12 : f32
      } else {
        scf.yield %7 : f32
      }
      %10 = arith.cmpf ole, %9, %cst_0 : f32
      %11 = scf.if %10 -> (f32) {
        %12 = arith.cmpf olt, %9, %cst_1 : f32
        %13 = scf.if %12 -> (f32) {
          %14 = arith.mulf %9, %9 : f32
          %15 = arith.divf %14, %cst_3 : f32
          %16 = arith.subf %cst_2, %15 : f32
          scf.yield %16 : f32
        } else {
          %14 = arith.mulf %9, %9 : f32
          %15 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (f32) {
            %22 = arith.index_cast %arg4 : index to i8
            %23 = arith.mulf %arg5, %14 : f32
            %24 = arith.index_castui %22 : i8 to index
            %25 = memref.load %2[%24] : memref<4xf32>
            %26 = arith.addf %23, %25 : f32
            affine.yield %26 : f32
          }
          %16 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
            %22 = arith.index_cast %arg4 : index to i8
            %23 = arith.mulf %arg5, %14 : f32
            %24 = arith.index_castui %22 : i8 to index
            %25 = memref.load %1[%24] : memref<8xf32>
            %26 = arith.addf %23, %25 : f32
            affine.yield %26 : f32
          }
          %17 = arith.subf %14, %cst_4 : f32
          %18 = arith.subf %14, %cst_5 : f32
          %19 = arith.mulf %17, %18 : f32
          %20 = arith.mulf %19, %15 : f32
          %21 = arith.divf %20, %16 : f32
          scf.yield %21 : f32
        }
        scf.yield %13 : f32
      } else {
        %12 = arith.mulf %9, %9 : f32
        %13 = arith.divf %cst_6, %12 : f32
        %14 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %31 = arith.index_cast %arg4 : index to i8
          %32 = arith.mulf %arg5, %13 : f32
          %33 = arith.index_castui %31 : i8 to index
          %34 = memref.load %6[%33] : memref<7xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35 : f32
        }
        %15 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %31 = arith.index_cast %arg4 : index to i8
          %32 = arith.mulf %arg5, %13 : f32
          %33 = arith.index_castui %31 : i8 to index
          %34 = memref.load %5[%33] : memref<7xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35 : f32
        }
        %16 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst) -> (f32) {
          %31 = arith.index_cast %arg4 : index to i8
          %32 = arith.mulf %arg5, %13 : f32
          %33 = arith.index_castui %31 : i8 to index
          %34 = memref.load %4[%33] : memref<8xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35 : f32
        }
        %17 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst) -> (f32) {
          %31 = arith.index_cast %arg4 : index to i8
          %32 = arith.mulf %arg5, %13 : f32
          %33 = arith.index_castui %31 : i8 to index
          %34 = memref.load %3[%33] : memref<7xf32>
          %35 = arith.addf %32, %34 : f32
          affine.yield %35 : f32
        }
        %18 = arith.divf %14, %15 : f32
        %19 = arith.subf %9, %cst_7 : f32
        %20 = math.cos %19 : f32
        %21 = arith.mulf %18, %20 : f32
        %22 = arith.divf %cst_0, %9 : f32
        %23 = arith.divf %16, %17 : f32
        %24 = arith.mulf %22, %23 : f32
        %25 = math.sin %19 : f32
        %26 = arith.mulf %24, %25 : f32
        %27 = arith.subf %21, %26 : f32
        %28 = arith.mulf %27, %cst_8 : f32
        %29 = math.sqrt %9 : f32
        %30 = arith.divf %28, %29 : f32
        scf.yield %30 : f32
      }
      affine.store %11, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

