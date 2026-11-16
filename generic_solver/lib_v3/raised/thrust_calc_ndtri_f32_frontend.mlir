#map = affine_map<()[s0] -> (s0 + 1)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q2" : memref<9xf32> = dense<[1.000000e+00, 6.02427053, 3.67983556, 1.37702096, 2.162370e-01, 1.342040e-02, 3.28014459E-4, 2.89247873E-6, 6.79019418E-9]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P2" : memref<9xf32> = dense<[3.23774886, 6.91522884, 3.93881035, 1.33303463, 0.201485395, 0.0123716639, 3.01581546E-4, 2.65806966E-6, 6.23974561E-9]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q1" : memref<9xf32> = dense<[1.000000e+00, 15.7799883, 45.3907623, 41.3172035, 15.0425386, 2.5046494, -0.142182916, -0.0380806401, -9.33259492E-4]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P1" : memref<9xf32> = dense<[4.05544901, 31.5251102, 57.1628189, 44.0805092, 14.6849566, 2.18663311, -0.140256077, -0.035042461, -8.57456762E-4]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q0" : memref<9xf32> = dense<[1.000000e+00, 1.95448864, 4.67627907, 86.3602447, -225.462692, 200.260208, -82.0372238, 15.9056225, -1.18331623]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P0" : memref<5xf32> = dense<[-59.9633484, 98.0010757, -56.6762848, 13.9312611, -1.23916578]>
  func.func @thrust_calc_ndtri_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2.50662827 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0.135335281 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant -2.000000e+00 : f64
    %true = arith.constant true
    %cst_6 = arith.constant 0.864664733 : f32
    %cst_7 = arith.constant 0x7F800000 : f32
    %cst_8 = arith.constant 0x7FC00000 : f32
    %cst_9 = arith.constant 0xFF800000 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q2" : memref<9xf32>
    %3 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P2" : memref<9xf32>
    %4 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q1" : memref<9xf32>
    %5 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P1" : memref<9xf32>
    %6 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q0" : memref<9xf32>
    %7 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P0" : memref<5xf32>
    affine.for %arg3 = 0 to %0 {
      %8 = affine.load %arg1[%arg3] : memref<?xf32>
      %9 = arith.cmpf oeq, %8, %cst_1 : f32
      %10 = arith.xori %9, %true : i1
      %11 = arith.select %9, %cst_9, %1 : f32
      %12 = arith.cmpf oeq, %8, %cst_0 : f32
      %13 = arith.andi %10, %12 : i1
      %14 = arith.xori %13, %true : i1
      %15 = arith.andi %14, %10 : i1
      %16 = scf.if %15 -> (f32) {
        %17 = arith.cmpf olt, %8, %cst_1 : f32
        %18 = scf.if %17 -> (i1) {
          scf.yield %true : i1
        } else {
          %20 = arith.cmpf ogt, %8, %cst_0 : f32
          scf.yield %20 : i1
        }
        %19 = scf.if %18 -> (f32) {
          scf.yield %cst_8 : f32
        } else {
          %20 = arith.cmpf ogt, %8, %cst_6 : f32
          %21 = scf.if %20 -> (f32) {
            %24 = arith.subf %cst_0, %8 : f32
            scf.yield %24 : f32
          } else {
            scf.yield %8 : f32
          }
          %22 = arith.cmpf ogt, %21, %cst_2 : f32
          %23 = scf.if %22 -> (f32) {
            %24 = arith.subf %21, %cst_3 : f32
            %25 = arith.mulf %24, %24 : f32
            %26 = affine.for %arg4 = 0 to 5 iter_args(%arg5 = %cst_1) -> (f32) {
              %33 = arith.index_cast %arg4 : index to i64
              %34 = arith.mulf %arg5, %25 : f32
              %35 = arith.index_castui %33 : i64 to index
              %36 = memref.load %7[%35] : memref<5xf32>
              %37 = arith.addf %34, %36 : f32
              affine.yield %37 : f32
            }
            %27 = arith.mulf %25, %26 : f32
            %28 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst_1) -> (f32) {
              %33 = arith.index_cast %arg4 : index to i64
              %34 = arith.mulf %arg5, %25 : f32
              %35 = arith.index_castui %33 : i64 to index
              %36 = memref.load %6[%35] : memref<9xf32>
              %37 = arith.addf %34, %36 : f32
              affine.yield %37 : f32
            }
            %29 = arith.divf %27, %28 : f32
            %30 = arith.mulf %24, %29 : f32
            %31 = arith.addf %24, %30 : f32
            %32 = arith.mulf %31, %cst : f32
            scf.yield %32 : f32
          } else {
            %24 = arith.extf %21 : f32 to f64
            %25 = math.log %24 : f64
            %26 = arith.mulf %25, %cst_5 : f64
            %27 = math.sqrt %26 : f64
            %28 = arith.truncf %27 : f64 to f32
            %29 = arith.extf %28 : f32 to f64
            %30 = math.log %29 : f64
            %31 = arith.divf %30, %29 : f64
            %32 = arith.subf %29, %31 : f64
            %33 = arith.truncf %32 : f64 to f32
            %34 = arith.divf %cst_0, %28 : f32
            %35 = arith.cmpf olt, %28, %cst_4 : f32
            %36 = scf.if %35 -> (f32) {
              %39 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst_1) -> (f32) {
                %43 = arith.index_cast %arg4 : index to i64
                %44 = arith.mulf %arg5, %34 : f32
                %45 = arith.index_castui %43 : i64 to index
                %46 = memref.load %5[%45] : memref<9xf32>
                %47 = arith.addf %44, %46 : f32
                affine.yield %47 : f32
              }
              %40 = arith.mulf %34, %39 : f32
              %41 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst_1) -> (f32) {
                %43 = arith.index_cast %arg4 : index to i64
                %44 = arith.mulf %arg5, %34 : f32
                %45 = arith.index_castui %43 : i64 to index
                %46 = memref.load %4[%45] : memref<9xf32>
                %47 = arith.addf %44, %46 : f32
                affine.yield %47 : f32
              }
              %42 = arith.divf %40, %41 : f32
              scf.yield %42 : f32
            } else {
              %39 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst_1) -> (f32) {
                %43 = arith.index_cast %arg4 : index to i64
                %44 = arith.mulf %arg5, %34 : f32
                %45 = arith.index_castui %43 : i64 to index
                %46 = memref.load %3[%45] : memref<9xf32>
                %47 = arith.addf %44, %46 : f32
                affine.yield %47 : f32
              }
              %40 = arith.mulf %34, %39 : f32
              %41 = affine.for %arg4 = 0 to 9 iter_args(%arg5 = %cst_1) -> (f32) {
                %43 = arith.index_cast %arg4 : index to i64
                %44 = arith.mulf %arg5, %34 : f32
                %45 = arith.index_castui %43 : i64 to index
                %46 = memref.load %2[%45] : memref<9xf32>
                %47 = arith.addf %44, %46 : f32
                affine.yield %47 : f32
              }
              %42 = arith.divf %40, %41 : f32
              scf.yield %42 : f32
            }
            %37 = arith.subf %33, %36 : f32
            %38 = scf.if %20 -> (f32) {
              scf.yield %37 : f32
            } else {
              %39 = arith.negf %37 : f32
              scf.yield %39 : f32
            }
            scf.yield %38 : f32
          }
          scf.yield %23 : f32
        }
        scf.yield %19 : f32
      } else {
        %17 = arith.select %13, %cst_7, %11 : f32
        scf.yield %17 : f32
      }
      affine.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
  func.func @calc_ndtrif(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0x7FC00000 : f32
    %cst_1 = arith.constant 0x7F800000 : f32
    %cst_2 = arith.constant 0.864664733 : f32
    %true = arith.constant true
    %cst_3 = arith.constant -2.000000e+00 : f64
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 0.135335281 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 1.000000e+00 : f32
    %cst_9 = arith.constant 2.50662827 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q2" : memref<9xf32>
    %2 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P2" : memref<9xf32>
    %3 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q1" : memref<9xf32>
    %4 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P1" : memref<9xf32>
    %5 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q0" : memref<9xf32>
    %6 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P0" : memref<5xf32>
    %7 = arith.cmpf oeq, %arg0, %cst_7 : f32
    %8 = arith.xori %7, %true : i1
    %9 = arith.select %7, %cst, %0 : f32
    %10 = arith.cmpf oeq, %arg0, %cst_8 : f32
    %11 = arith.andi %8, %10 : i1
    %12 = arith.xori %11, %true : i1
    %13 = arith.andi %12, %8 : i1
    %14 = scf.if %13 -> (f32) {
      %15 = arith.cmpf olt, %arg0, %cst_7 : f32
      %16 = scf.if %15 -> (i1) {
        scf.yield %true : i1
      } else {
        %18 = arith.cmpf ogt, %arg0, %cst_8 : f32
        scf.yield %18 : i1
      }
      %17 = scf.if %16 -> (f32) {
        scf.yield %cst_0 : f32
      } else {
        %18 = arith.cmpf ogt, %arg0, %cst_2 : f32
        %19 = scf.if %18 -> (f32) {
          %22 = arith.subf %cst_8, %arg0 : f32
          scf.yield %22 : f32
        } else {
          scf.yield %arg0 : f32
        }
        %20 = arith.cmpf ogt, %19, %cst_6 : f32
        %21 = scf.if %20 -> (f32) {
          %22 = arith.subf %19, %cst_5 : f32
          %23 = arith.mulf %22, %22 : f32
          %24 = affine.for %arg1 = 0 to 5 iter_args(%arg2 = %cst_7) -> (f32) {
            %31 = arith.index_cast %arg1 : index to i64
            %32 = arith.mulf %arg2, %23 : f32
            %33 = arith.index_castui %31 : i64 to index
            %34 = memref.load %6[%33] : memref<5xf32>
            %35 = arith.addf %32, %34 : f32
            affine.yield %35 : f32
          }
          %25 = arith.mulf %23, %24 : f32
          %26 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
            %31 = arith.index_cast %arg1 : index to i64
            %32 = arith.mulf %arg2, %23 : f32
            %33 = arith.index_castui %31 : i64 to index
            %34 = memref.load %5[%33] : memref<9xf32>
            %35 = arith.addf %32, %34 : f32
            affine.yield %35 : f32
          }
          %27 = arith.divf %25, %26 : f32
          %28 = arith.mulf %22, %27 : f32
          %29 = arith.addf %22, %28 : f32
          %30 = arith.mulf %29, %cst_9 : f32
          scf.yield %30 : f32
        } else {
          %22 = arith.extf %19 : f32 to f64
          %23 = math.log %22 : f64
          %24 = arith.mulf %23, %cst_3 : f64
          %25 = math.sqrt %24 : f64
          %26 = arith.truncf %25 : f64 to f32
          %27 = arith.extf %26 : f32 to f64
          %28 = math.log %27 : f64
          %29 = arith.divf %28, %27 : f64
          %30 = arith.subf %27, %29 : f64
          %31 = arith.truncf %30 : f64 to f32
          %32 = arith.divf %cst_8, %26 : f32
          %33 = arith.cmpf olt, %26, %cst_4 : f32
          %34 = scf.if %33 -> (f32) {
            %37 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %4[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %38 = arith.mulf %32, %37 : f32
            %39 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %3[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %40 = arith.divf %38, %39 : f32
            scf.yield %40 : f32
          } else {
            %37 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %2[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %38 = arith.mulf %32, %37 : f32
            %39 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %1[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %40 = arith.divf %38, %39 : f32
            scf.yield %40 : f32
          }
          %35 = arith.subf %31, %34 : f32
          %36 = scf.if %18 -> (f32) {
            scf.yield %35 : f32
          } else {
            %37 = arith.negf %35 : f32
            scf.yield %37 : f32
          }
          scf.yield %36 : f32
        }
        scf.yield %21 : f32
      }
      scf.yield %17 : f32
    } else {
      %15 = arith.select %11, %cst_1, %9 : f32
      scf.yield %15 : f32
    }
    return %14 : f32
  }
  func.func @_Z10calc_ndtriIfET_S0_(%arg0: f32) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0x7FC00000 : f32
    %cst_1 = arith.constant 0x7F800000 : f32
    %cst_2 = arith.constant 0.864664733 : f32
    %true = arith.constant true
    %cst_3 = arith.constant -2.000000e+00 : f64
    %cst_4 = arith.constant 8.000000e+00 : f32
    %cst_5 = arith.constant 5.000000e-01 : f32
    %cst_6 = arith.constant 0.135335281 : f32
    %cst_7 = arith.constant 0.000000e+00 : f32
    %cst_8 = arith.constant 1.000000e+00 : f32
    %cst_9 = arith.constant 2.50662827 : f32
    %0 = llvm.mlir.undef : f32
    %1 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q2" : memref<9xf32>
    %2 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P2" : memref<9xf32>
    %3 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q1" : memref<9xf32>
    %4 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P1" : memref<9xf32>
    %5 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q0" : memref<9xf32>
    %6 = memref.get_global @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P0" : memref<5xf32>
    %7 = arith.cmpf oeq, %arg0, %cst_7 : f32
    %8 = arith.xori %7, %true : i1
    %9 = arith.select %7, %cst, %0 : f32
    %10 = arith.cmpf oeq, %arg0, %cst_8 : f32
    %11 = arith.andi %8, %10 : i1
    %12 = arith.xori %11, %true : i1
    %13 = arith.andi %12, %8 : i1
    %14 = scf.if %13 -> (f32) {
      %15 = arith.cmpf olt, %arg0, %cst_7 : f32
      %16 = scf.if %15 -> (i1) {
        scf.yield %true : i1
      } else {
        %18 = arith.cmpf ogt, %arg0, %cst_8 : f32
        scf.yield %18 : i1
      }
      %17 = scf.if %16 -> (f32) {
        scf.yield %cst_0 : f32
      } else {
        %18 = arith.cmpf ogt, %arg0, %cst_2 : f32
        %19 = scf.if %18 -> (f32) {
          %22 = arith.subf %cst_8, %arg0 : f32
          scf.yield %22 : f32
        } else {
          scf.yield %arg0 : f32
        }
        %20 = arith.cmpf ogt, %19, %cst_6 : f32
        %21 = scf.if %20 -> (f32) {
          %22 = arith.subf %19, %cst_5 : f32
          %23 = arith.mulf %22, %22 : f32
          %24 = affine.for %arg1 = 0 to 5 iter_args(%arg2 = %cst_7) -> (f32) {
            %31 = arith.index_cast %arg1 : index to i64
            %32 = arith.mulf %arg2, %23 : f32
            %33 = arith.index_castui %31 : i64 to index
            %34 = memref.load %6[%33] : memref<5xf32>
            %35 = arith.addf %32, %34 : f32
            affine.yield %35 : f32
          }
          %25 = arith.mulf %23, %24 : f32
          %26 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
            %31 = arith.index_cast %arg1 : index to i64
            %32 = arith.mulf %arg2, %23 : f32
            %33 = arith.index_castui %31 : i64 to index
            %34 = memref.load %5[%33] : memref<9xf32>
            %35 = arith.addf %32, %34 : f32
            affine.yield %35 : f32
          }
          %27 = arith.divf %25, %26 : f32
          %28 = arith.mulf %22, %27 : f32
          %29 = arith.addf %22, %28 : f32
          %30 = arith.mulf %29, %cst_9 : f32
          scf.yield %30 : f32
        } else {
          %22 = arith.extf %19 : f32 to f64
          %23 = math.log %22 : f64
          %24 = arith.mulf %23, %cst_3 : f64
          %25 = math.sqrt %24 : f64
          %26 = arith.truncf %25 : f64 to f32
          %27 = arith.extf %26 : f32 to f64
          %28 = math.log %27 : f64
          %29 = arith.divf %28, %27 : f64
          %30 = arith.subf %27, %29 : f64
          %31 = arith.truncf %30 : f64 to f32
          %32 = arith.divf %cst_8, %26 : f32
          %33 = arith.cmpf olt, %26, %cst_4 : f32
          %34 = scf.if %33 -> (f32) {
            %37 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %4[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %38 = arith.mulf %32, %37 : f32
            %39 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %3[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %40 = arith.divf %38, %39 : f32
            scf.yield %40 : f32
          } else {
            %37 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %2[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %38 = arith.mulf %32, %37 : f32
            %39 = affine.for %arg1 = 0 to 9 iter_args(%arg2 = %cst_7) -> (f32) {
              %41 = arith.index_cast %arg1 : index to i64
              %42 = arith.mulf %arg2, %32 : f32
              %43 = arith.index_castui %41 : i64 to index
              %44 = memref.load %1[%43] : memref<9xf32>
              %45 = arith.addf %42, %44 : f32
              affine.yield %45 : f32
            }
            %40 = arith.divf %38, %39 : f32
            scf.yield %40 : f32
          }
          %35 = arith.subf %31, %34 : f32
          %36 = scf.if %18 -> (f32) {
            scf.yield %35 : f32
          } else {
            %37 = arith.negf %35 : f32
            scf.yield %37 : f32
          }
          scf.yield %36 : f32
        }
        scf.yield %21 : f32
      }
      scf.yield %17 : f32
    } else {
      %15 = arith.select %11, %cst_1, %9 : f32
      scf.yield %15 : f32
    }
    return %14 : f32
  }
  func.func @_ZNSt14numeric_limitsIfE8infinityEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7F800000 : f32
    return %cst : f32
  }
  func.func @_ZNSt14numeric_limitsIfE9quiet_NaNEv() -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0x7FC00000 : f32
    return %cst : f32
  }
  func.func @_Z6polevlIfET_S0_PKS0_m(%arg0: f32, %arg1: memref<?xf32>, %arg2: i64) -> f32 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg2 : i64 to index
    %1 = affine.for %arg3 = 0 to #map()[%0] iter_args(%arg4 = %cst) -> (f32) {
      %2 = arith.index_cast %arg3 : index to i64
      %3 = arith.mulf %arg4, %arg0 : f32
      %4 = arith.index_castui %2 : i64 to index
      %5 = memref.load %arg1[%4] : memref<?xf32>
      %6 = arith.addf %3, %5 : f32
      affine.yield %6 : f32
    }
    return %1 : f32
  }
}
