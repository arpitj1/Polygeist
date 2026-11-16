module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q2" : memref<9xf32> = dense<[1.000000e+00, 6.02427053, 3.67983556, 1.37702096, 2.162370e-01, 1.342040e-02, 3.28014459E-4, 2.89247873E-6, 6.79019418E-9]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P2" : memref<9xf32> = dense<[3.23774886, 6.91522884, 3.93881035, 1.33303463, 0.201485395, 0.0123716639, 3.01581546E-4, 2.65806966E-6, 6.23974561E-9]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q1" : memref<9xf32> = dense<[1.000000e+00, 15.7799883, 45.3907623, 41.3172035, 15.0425386, 2.5046494, -0.142182916, -0.0380806401, -9.33259492E-4]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P1" : memref<9xf32> = dense<[4.05544901, 31.5251102, 57.1628189, 44.0805092, 14.6849566, 2.18663311, -0.140256077, -0.035042461, -8.57456762E-4]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2Q0" : memref<9xf32> = dense<[1.000000e+00, 1.95448864, 4.67627907, 86.3602447, -225.462692, 200.260208, -82.0372238, 15.9056225, -1.18331623]>
  memref.global constant @"_Z10calc_ndtriIfET_S0_@static@_ZZ10calc_ndtriIfET_S0_E2P0" : memref<5xf32> = dense<[-59.9633484, 98.0010757, -56.6762848, 13.9312611, -1.23916578]>
  func.func @thrust_calc_ndtri_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
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
      %8 = memref.load %arg1[%arg3] : memref<?xf32>
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
            %26 = arith.mulf %25, %cst_1 : f32
            %27 = memref.load %7[%c0] : memref<5xf32>
            %28 = arith.addf %26, %27 : f32
            %29 = arith.mulf %28, %25 : f32
            %30 = memref.load %7[%c1] : memref<5xf32>
            %31 = arith.addf %29, %30 : f32
            %32 = arith.mulf %31, %25 : f32
            %33 = memref.load %7[%c2] : memref<5xf32>
            %34 = arith.addf %32, %33 : f32
            %35 = arith.mulf %34, %25 : f32
            %36 = memref.load %7[%c3] : memref<5xf32>
            %37 = arith.addf %35, %36 : f32
            %38 = arith.mulf %37, %25 : f32
            %39 = memref.load %7[%c4] : memref<5xf32>
            %40 = arith.addf %38, %39 : f32
            %41 = arith.mulf %25, %40 : f32
            %42 = memref.load %6[%c0] : memref<9xf32>
            %43 = arith.addf %26, %42 : f32
            %44 = arith.mulf %43, %25 : f32
            %45 = memref.load %6[%c1] : memref<9xf32>
            %46 = arith.addf %44, %45 : f32
            %47 = arith.mulf %46, %25 : f32
            %48 = memref.load %6[%c2] : memref<9xf32>
            %49 = arith.addf %47, %48 : f32
            %50 = arith.mulf %49, %25 : f32
            %51 = memref.load %6[%c3] : memref<9xf32>
            %52 = arith.addf %50, %51 : f32
            %53 = arith.mulf %52, %25 : f32
            %54 = memref.load %6[%c4] : memref<9xf32>
            %55 = arith.addf %53, %54 : f32
            %56 = arith.mulf %55, %25 : f32
            %57 = memref.load %6[%c5] : memref<9xf32>
            %58 = arith.addf %56, %57 : f32
            %59 = arith.mulf %58, %25 : f32
            %60 = memref.load %6[%c6] : memref<9xf32>
            %61 = arith.addf %59, %60 : f32
            %62 = arith.mulf %61, %25 : f32
            %63 = memref.load %6[%c7] : memref<9xf32>
            %64 = arith.addf %62, %63 : f32
            %65 = arith.mulf %64, %25 : f32
            %66 = memref.load %6[%c8] : memref<9xf32>
            %67 = arith.addf %65, %66 : f32
            %68 = arith.divf %41, %67 : f32
            %69 = arith.mulf %24, %68 : f32
            %70 = arith.addf %24, %69 : f32
            %71 = arith.mulf %70, %cst : f32
            scf.yield %71 : f32
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
              %39 = arith.mulf %34, %cst_1 : f32
              %40 = memref.load %5[%c0] : memref<9xf32>
              %41 = arith.addf %39, %40 : f32
              %42 = arith.mulf %41, %34 : f32
              %43 = memref.load %5[%c1] : memref<9xf32>
              %44 = arith.addf %42, %43 : f32
              %45 = arith.mulf %44, %34 : f32
              %46 = memref.load %5[%c2] : memref<9xf32>
              %47 = arith.addf %45, %46 : f32
              %48 = arith.mulf %47, %34 : f32
              %49 = memref.load %5[%c3] : memref<9xf32>
              %50 = arith.addf %48, %49 : f32
              %51 = arith.mulf %50, %34 : f32
              %52 = memref.load %5[%c4] : memref<9xf32>
              %53 = arith.addf %51, %52 : f32
              %54 = arith.mulf %53, %34 : f32
              %55 = memref.load %5[%c5] : memref<9xf32>
              %56 = arith.addf %54, %55 : f32
              %57 = arith.mulf %56, %34 : f32
              %58 = memref.load %5[%c6] : memref<9xf32>
              %59 = arith.addf %57, %58 : f32
              %60 = arith.mulf %59, %34 : f32
              %61 = memref.load %5[%c7] : memref<9xf32>
              %62 = arith.addf %60, %61 : f32
              %63 = arith.mulf %62, %34 : f32
              %64 = memref.load %5[%c8] : memref<9xf32>
              %65 = arith.addf %63, %64 : f32
              %66 = arith.mulf %34, %65 : f32
              %67 = memref.load %4[%c0] : memref<9xf32>
              %68 = arith.addf %39, %67 : f32
              %69 = arith.mulf %68, %34 : f32
              %70 = memref.load %4[%c1] : memref<9xf32>
              %71 = arith.addf %69, %70 : f32
              %72 = arith.mulf %71, %34 : f32
              %73 = memref.load %4[%c2] : memref<9xf32>
              %74 = arith.addf %72, %73 : f32
              %75 = arith.mulf %74, %34 : f32
              %76 = memref.load %4[%c3] : memref<9xf32>
              %77 = arith.addf %75, %76 : f32
              %78 = arith.mulf %77, %34 : f32
              %79 = memref.load %4[%c4] : memref<9xf32>
              %80 = arith.addf %78, %79 : f32
              %81 = arith.mulf %80, %34 : f32
              %82 = memref.load %4[%c5] : memref<9xf32>
              %83 = arith.addf %81, %82 : f32
              %84 = arith.mulf %83, %34 : f32
              %85 = memref.load %4[%c6] : memref<9xf32>
              %86 = arith.addf %84, %85 : f32
              %87 = arith.mulf %86, %34 : f32
              %88 = memref.load %4[%c7] : memref<9xf32>
              %89 = arith.addf %87, %88 : f32
              %90 = arith.mulf %89, %34 : f32
              %91 = memref.load %4[%c8] : memref<9xf32>
              %92 = arith.addf %90, %91 : f32
              %93 = arith.divf %66, %92 : f32
              scf.yield %93 : f32
            } else {
              %39 = arith.mulf %34, %cst_1 : f32
              %40 = memref.load %3[%c0] : memref<9xf32>
              %41 = arith.addf %39, %40 : f32
              %42 = arith.mulf %41, %34 : f32
              %43 = memref.load %3[%c1] : memref<9xf32>
              %44 = arith.addf %42, %43 : f32
              %45 = arith.mulf %44, %34 : f32
              %46 = memref.load %3[%c2] : memref<9xf32>
              %47 = arith.addf %45, %46 : f32
              %48 = arith.mulf %47, %34 : f32
              %49 = memref.load %3[%c3] : memref<9xf32>
              %50 = arith.addf %48, %49 : f32
              %51 = arith.mulf %50, %34 : f32
              %52 = memref.load %3[%c4] : memref<9xf32>
              %53 = arith.addf %51, %52 : f32
              %54 = arith.mulf %53, %34 : f32
              %55 = memref.load %3[%c5] : memref<9xf32>
              %56 = arith.addf %54, %55 : f32
              %57 = arith.mulf %56, %34 : f32
              %58 = memref.load %3[%c6] : memref<9xf32>
              %59 = arith.addf %57, %58 : f32
              %60 = arith.mulf %59, %34 : f32
              %61 = memref.load %3[%c7] : memref<9xf32>
              %62 = arith.addf %60, %61 : f32
              %63 = arith.mulf %62, %34 : f32
              %64 = memref.load %3[%c8] : memref<9xf32>
              %65 = arith.addf %63, %64 : f32
              %66 = arith.mulf %34, %65 : f32
              %67 = memref.load %2[%c0] : memref<9xf32>
              %68 = arith.addf %39, %67 : f32
              %69 = arith.mulf %68, %34 : f32
              %70 = memref.load %2[%c1] : memref<9xf32>
              %71 = arith.addf %69, %70 : f32
              %72 = arith.mulf %71, %34 : f32
              %73 = memref.load %2[%c2] : memref<9xf32>
              %74 = arith.addf %72, %73 : f32
              %75 = arith.mulf %74, %34 : f32
              %76 = memref.load %2[%c3] : memref<9xf32>
              %77 = arith.addf %75, %76 : f32
              %78 = arith.mulf %77, %34 : f32
              %79 = memref.load %2[%c4] : memref<9xf32>
              %80 = arith.addf %78, %79 : f32
              %81 = arith.mulf %80, %34 : f32
              %82 = memref.load %2[%c5] : memref<9xf32>
              %83 = arith.addf %81, %82 : f32
              %84 = arith.mulf %83, %34 : f32
              %85 = memref.load %2[%c6] : memref<9xf32>
              %86 = arith.addf %84, %85 : f32
              %87 = arith.mulf %86, %34 : f32
              %88 = memref.load %2[%c7] : memref<9xf32>
              %89 = arith.addf %87, %88 : f32
              %90 = arith.mulf %89, %34 : f32
              %91 = memref.load %2[%c8] : memref<9xf32>
              %92 = arith.addf %90, %91 : f32
              %93 = arith.divf %66, %92 : f32
              scf.yield %93 : f32
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
      memref.store %16, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

