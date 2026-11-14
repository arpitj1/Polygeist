module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RQ" : memref<8xf32> = dense<[499.563141, 173785.406, 0x4C38C9A1, 1.11855534E+10, 2.11277526E+12, 3.10518217E+14, 3.18121957E+16, 1.71086291E+18]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RP" : memref<4xf32> = dense<[-4.794432E+9, 1.95617489E+12, -2.49248344E+14, 9.70862217E+15]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[64.3178253, 856.430054, 3882.40186, 7240.46777, 5930.72705, 2062.09326, 242.005737]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[-0.0113663841, -1.28252721, -19.5539551, -93.2060165, -177.681168, -147.077499, -51.4105339, -6.05014372]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[9.24408785E-4, 0.0856288447, 1.2535274, 5.47097731, 8.76190853, 5.30605268, 1.000000e+00]>
  memref.global constant @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.96936743E-4, 0.0828352422, 1.23953366, 5.447250e+00, 8.74716472, 5.3032403, 1.000000e+00]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2YQ" : memref<7xf32> = dense<[1041.28357, 626107.313, 0x4D803B1B, 8.64002457E+10, 2.02979609E+13, 3.17157753E+15, 2.50596258E+17]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2YP" : memref<8xf32> = dense<[15592.4365, 0xCB5FC0FA, 5.43526502E+9, -9.82136061E+11, 8.75906402E+13, -3.46628315E+15, 4.4273326E+16, -1.84950805E+16]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2QQ" : memref<7xf32> = dense<[64.3178253, 856.430054, 3882.40186, 7240.46777, 5930.72705, 2062.09326, 242.005737]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2QP" : memref<8xf32> = dense<[-0.0113663841, -1.28252721, -19.5539551, -93.2060165, -177.681168, -147.077499, -51.4105339, -6.05014372]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2PQ" : memref<7xf32> = dense<[9.24408785E-4, 0.0856288447, 1.2535274, 5.47097731, 8.76190853, 5.30605268, 1.000000e+00]>
  memref.global constant @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2PP" : memref<7xf32> = dense<[7.96936743E-4, 0.0828352422, 1.23953366, 5.447250e+00, 8.74716472, 5.3032403, 1.000000e+00]>
  func.func @thrust_bessel_y0_forward_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 0.636619746 : f32
    %cst_2 = arith.constant 2.500000e+01 : f32
    %cst_3 = arith.constant 0.785398185 : f32
    %cst_4 = arith.constant 0.797884583 : f32
    %true = arith.constant true
    %cst_5 = arith.constant 0x7FC00000 : f32
    %cst_6 = arith.constant 30.471262 : f32
    %cst_7 = arith.constant 5.78318596 : f32
    %cst_8 = arith.constant 4.000000e+00 : f32
    %cst_9 = arith.constant 1.000000e+00 : f32
    %cst_10 = arith.constant 9.99999974E-6 : f32
    %cst_11 = arith.constant 0xFF800000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2YQ" : memref<7xf32>
    %3 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2YP" : memref<8xf32>
    %4 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2QQ" : memref<7xf32>
    %5 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2QP" : memref<8xf32>
    %6 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2PQ" : memref<7xf32>
    %7 = memref.get_global @"_Z17bessel_y0_forwardIfET_S0_@static@_ZZ17bessel_y0_forwardIfET_S0_E2PP" : memref<7xf32>
    %8 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RQ" : memref<8xf32>
    %9 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2RP" : memref<4xf32>
    %10 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QQ" : memref<7xf32>
    %11 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2QP" : memref<8xf32>
    %12 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PQ" : memref<7xf32>
    %13 = memref.get_global @"_Z17bessel_j0_forwardIfET_S0_@static@_ZZ17bessel_j0_forwardIfET_S0_E2PP" : memref<7xf32>
    affine.for %arg3 = 0 to %0 {
      %14 = affine.load %arg1[%arg3] : memref<?xf32>
      %15 = arith.cmpf ole, %14, %cst : f32
      %16 = scf.if %15 -> (f32) {
        %17 = arith.cmpf oeq, %14, %cst_0 : f32
        %18 = arith.xori %17, %true : i1
        %19 = arith.select %17, %cst_11, %1 : f32
        %20 = arith.cmpf olt, %14, %cst_0 : f32
        %21 = arith.andi %20, %18 : i1
        %22 = arith.xori %21, %true : i1
        %23 = arith.andi %22, %18 : i1
        %24 = scf.if %23 -> (f32) {
          %25 = arith.mulf %14, %14 : f32
          %26 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
            %36 = arith.index_cast %arg4 : index to i8
            %37 = arith.mulf %arg5, %25 : f32
            %38 = arith.index_castui %36 : i8 to index
            %39 = memref.load %3[%38] : memref<8xf32>
            %40 = arith.addf %37, %39 : f32
            affine.yield %40 : f32
          }
          %27 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
            %36 = arith.index_cast %arg4 : index to i8
            %37 = arith.mulf %arg5, %25 : f32
            %38 = arith.index_castui %36 : i8 to index
            %39 = memref.load %2[%38] : memref<7xf32>
            %40 = arith.addf %37, %39 : f32
            affine.yield %40 : f32
          }
          %28 = arith.divf %26, %27 : f32
          %29 = math.log %14 : f32
          %30 = arith.mulf %29, %cst_1 : f32
          %31 = scf.if %20 -> (f32) {
            %36 = arith.negf %14 : f32
            scf.yield %36 : f32
          } else {
            scf.yield %14 : f32
          }
          %32 = arith.cmpf ole, %31, %cst : f32
          %33 = scf.if %32 -> (f32) {
            %36 = arith.cmpf olt, %31, %cst_10 : f32
            %37 = scf.if %36 -> (f32) {
              %38 = arith.mulf %31, %31 : f32
              %39 = arith.divf %38, %cst_8 : f32
              %40 = arith.subf %cst_9, %39 : f32
              scf.yield %40 : f32
            } else {
              %38 = arith.mulf %31, %31 : f32
              %39 = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst_0) -> (f32) {
                %46 = arith.index_cast %arg4 : index to i8
                %47 = arith.mulf %arg5, %38 : f32
                %48 = arith.index_castui %46 : i8 to index
                %49 = memref.load %9[%48] : memref<4xf32>
                %50 = arith.addf %47, %49 : f32
                affine.yield %50 : f32
              }
              %40 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
                %46 = arith.index_cast %arg4 : index to i8
                %47 = arith.mulf %arg5, %38 : f32
                %48 = arith.index_castui %46 : i8 to index
                %49 = memref.load %8[%48] : memref<8xf32>
                %50 = arith.addf %47, %49 : f32
                affine.yield %50 : f32
              }
              %41 = arith.subf %38, %cst_7 : f32
              %42 = arith.subf %38, %cst_6 : f32
              %43 = arith.mulf %41, %42 : f32
              %44 = arith.mulf %43, %39 : f32
              %45 = arith.divf %44, %40 : f32
              scf.yield %45 : f32
            }
            scf.yield %37 : f32
          } else {
            %36 = arith.mulf %31, %31 : f32
            %37 = arith.divf %cst_2, %36 : f32
            %38 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
              %55 = arith.index_cast %arg4 : index to i8
              %56 = arith.mulf %arg5, %37 : f32
              %57 = arith.index_castui %55 : i8 to index
              %58 = memref.load %13[%57] : memref<7xf32>
              %59 = arith.addf %56, %58 : f32
              affine.yield %59 : f32
            }
            %39 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
              %55 = arith.index_cast %arg4 : index to i8
              %56 = arith.mulf %arg5, %37 : f32
              %57 = arith.index_castui %55 : i8 to index
              %58 = memref.load %12[%57] : memref<7xf32>
              %59 = arith.addf %56, %58 : f32
              affine.yield %59 : f32
            }
            %40 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
              %55 = arith.index_cast %arg4 : index to i8
              %56 = arith.mulf %arg5, %37 : f32
              %57 = arith.index_castui %55 : i8 to index
              %58 = memref.load %11[%57] : memref<8xf32>
              %59 = arith.addf %56, %58 : f32
              affine.yield %59 : f32
            }
            %41 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
              %55 = arith.index_cast %arg4 : index to i8
              %56 = arith.mulf %arg5, %37 : f32
              %57 = arith.index_castui %55 : i8 to index
              %58 = memref.load %10[%57] : memref<7xf32>
              %59 = arith.addf %56, %58 : f32
              affine.yield %59 : f32
            }
            %42 = arith.divf %38, %39 : f32
            %43 = arith.subf %31, %cst_3 : f32
            %44 = math.cos %43 : f32
            %45 = arith.mulf %42, %44 : f32
            %46 = arith.divf %cst, %31 : f32
            %47 = arith.divf %40, %41 : f32
            %48 = arith.mulf %46, %47 : f32
            %49 = math.sin %43 : f32
            %50 = arith.mulf %48, %49 : f32
            %51 = arith.subf %45, %50 : f32
            %52 = arith.mulf %51, %cst_4 : f32
            %53 = math.sqrt %31 : f32
            %54 = arith.divf %52, %53 : f32
            scf.yield %54 : f32
          }
          %34 = arith.mulf %30, %33 : f32
          %35 = arith.addf %28, %34 : f32
          scf.yield %35 : f32
        } else {
          %25 = arith.select %21, %cst_5, %19 : f32
          scf.yield %25 : f32
        }
        scf.yield %24 : f32
      } else {
        %17 = arith.mulf %14, %14 : f32
        %18 = arith.divf %cst_2, %17 : f32
        %19 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %36 = arith.index_cast %arg4 : index to i8
          %37 = arith.mulf %arg5, %18 : f32
          %38 = arith.index_castui %36 : i8 to index
          %39 = memref.load %7[%38] : memref<7xf32>
          %40 = arith.addf %37, %39 : f32
          affine.yield %40 : f32
        }
        %20 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %36 = arith.index_cast %arg4 : index to i8
          %37 = arith.mulf %arg5, %18 : f32
          %38 = arith.index_castui %36 : i8 to index
          %39 = memref.load %6[%38] : memref<7xf32>
          %40 = arith.addf %37, %39 : f32
          affine.yield %40 : f32
        }
        %21 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %cst_0) -> (f32) {
          %36 = arith.index_cast %arg4 : index to i8
          %37 = arith.mulf %arg5, %18 : f32
          %38 = arith.index_castui %36 : i8 to index
          %39 = memref.load %5[%38] : memref<8xf32>
          %40 = arith.addf %37, %39 : f32
          affine.yield %40 : f32
        }
        %22 = affine.for %arg4 = 0 to 7 iter_args(%arg5 = %cst_0) -> (f32) {
          %36 = arith.index_cast %arg4 : index to i8
          %37 = arith.mulf %arg5, %18 : f32
          %38 = arith.index_castui %36 : i8 to index
          %39 = memref.load %4[%38] : memref<7xf32>
          %40 = arith.addf %37, %39 : f32
          affine.yield %40 : f32
        }
        %23 = arith.divf %19, %20 : f32
        %24 = arith.subf %14, %cst_3 : f32
        %25 = math.sin %24 : f32
        %26 = arith.mulf %23, %25 : f32
        %27 = arith.divf %cst, %14 : f32
        %28 = arith.divf %21, %22 : f32
        %29 = arith.mulf %27, %28 : f32
        %30 = math.cos %24 : f32
        %31 = arith.mulf %29, %30 : f32
        %32 = arith.addf %26, %31 : f32
        %33 = arith.mulf %32, %cst_4 : f32
        %34 = math.sqrt %14 : f32
        %35 = arith.divf %33, %34 : f32
        scf.yield %35 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

