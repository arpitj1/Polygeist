module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32> = dense<[9.30892944, 19.8352928, 15.5646629, 5.47686052, 0.954293608, 0.0864580795, 0.00412656507, 1.01259087E-4, 1.17166735E-6, 4.91834573E-9]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32> = dense<[0.0197339933, 0.391103029, 1.065799, 0.939169228, 0.351465642, 0.0633888915, 0.00585804135, 2.82851601E-4, 6.98793655E-6, 8.11789249E-8, 3.41551787E-10]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32> = dense<[13.3560419, 32.6825027, 26.7367039, 9.1870737, 1.47529149, 0.115687177, 0.00440291641, 7.54720313E-5, 4.51850099E-7]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32> = dense<[-0.131696329, -0.626456558, -0.693158031, -0.279779971, -0.0491900146, -0.00406265911, -1.59276489E-4, -2.7764911E-6, -1.67787704E-8]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32> = dense<[0.567594528, 14.7562561, 8.451390e+01, 177.318085, 164.234695, 71.4778366, 14.0959139, 1.000000e+00]>
  memref.global constant @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32> = dense<[0.346538097, 12.0075951, 7.627960e+01, 168.089218, 159.756393, 70.536087, 14.0264692, 1.000000e+00]>
  func.func @thrust_calc_airy_ai_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c10 = arith.constant 10 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_0 = arith.constant 1.038920e+02 : f32
    %cst_1 = arith.constant -2.090000e+00 : f32
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant -2.000000e+00 : f32
    %cst_4 = arith.constant 3.000000e+00 : f32
    %cst_5 = arith.constant 0.564189613 : f32
    %cst_6 = arith.constant 2.090000e+00 : f32
    %c5_i32 = arith.constant 5 : i32
    %cst_7 = arith.constant 2.000000e+00 : f32
    %cst_8 = arith.constant 8.32033538 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant 0.355028063 : f32
    %cst_10 = arith.constant 0.258819401 : f32
    %true = arith.constant true
    %false = arith.constant false
    %cst_11 = arith.constant 0x7F800000 : f32
    %cst_12 = arith.constant 0x7FC00000 : f32
    %cst_13 = arith.constant 1.1920929E-7 : f32
    %cst_14 = arith.constant 0.785398185 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = llvm.mlir.undef : f32
    %2 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGD" : memref<10xf32>
    %3 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AGN" : memref<11xf32>
    %4 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFD" : memref<9xf32>
    %5 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E3AFN" : memref<9xf32>
    %6 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AD" : memref<8xf32>
    %7 = memref.get_global @"_Z15airy_ai_forwardIfET_S0_@static@_ZZ15airy_ai_forwardIfET_S0_E2AN" : memref<8xf32>
    affine.for %arg3 = 0 to %0 {
      %8 = memref.load %arg1[%arg3] : memref<?xf32>
      %9 = math.absf %8 : f32
      %10 = arith.cmpf oeq, %9, %cst_11 : f32
      %11 = arith.extui %10 : i1 to i8
      %12 = arith.cmpi eq, %11, %c0_i8 : i8
      %13 = arith.select %10, %cst_12, %1 : f32
      %14:2 = scf.if %12 -> (i1, f32) {
        %27 = arith.cmpf ogt, %8, %cst_0 : f32
        %28 = arith.xori %27, %true : i1
        %29 = arith.select %27, %cst, %13 : f32
        scf.yield %28, %29 : i1, f32
      } else {
        scf.yield %false, %13 : i1, f32
      }
      %15 = arith.cmpf olt, %8, %cst_1 : f32
      %16 = arith.andi %15, %14#0 : i1
      %17 = arith.xori %15, %true : i1
      %18 = arith.andi %17, %14#0 : i1
      %19 = scf.if %16 -> (f32) {
        %27 = arith.mulf %8, %cst_3 : f32
        %28 = arith.negf %8 : f32
        %29 = math.sqrt %28 : f32
        %30 = arith.mulf %27, %29 : f32
        %31 = arith.divf %30, %cst_4 : f32
        %32 = arith.divf %cst_2, %31 : f32
        %33 = arith.mulf %32, %32 : f32
        %34 = arith.mulf %33, %cst : f32
        %35 = memref.load %5[%c0] : memref<9xf32>
        %36 = arith.addf %34, %35 : f32
        %37 = arith.mulf %36, %33 : f32
        %38 = memref.load %5[%c1] : memref<9xf32>
        %39 = arith.addf %37, %38 : f32
        %40 = arith.mulf %39, %33 : f32
        %41 = memref.load %5[%c2] : memref<9xf32>
        %42 = arith.addf %40, %41 : f32
        %43 = arith.mulf %42, %33 : f32
        %44 = memref.load %5[%c3] : memref<9xf32>
        %45 = arith.addf %43, %44 : f32
        %46 = arith.mulf %45, %33 : f32
        %47 = memref.load %5[%c4] : memref<9xf32>
        %48 = arith.addf %46, %47 : f32
        %49 = arith.mulf %48, %33 : f32
        %50 = memref.load %5[%c5] : memref<9xf32>
        %51 = arith.addf %49, %50 : f32
        %52 = arith.mulf %51, %33 : f32
        %53 = memref.load %5[%c6] : memref<9xf32>
        %54 = arith.addf %52, %53 : f32
        %55 = arith.mulf %54, %33 : f32
        %56 = memref.load %5[%c7] : memref<9xf32>
        %57 = arith.addf %55, %56 : f32
        %58 = arith.mulf %57, %33 : f32
        %59 = memref.load %5[%c8] : memref<9xf32>
        %60 = arith.addf %58, %59 : f32
        %61 = memref.load %4[%c0] : memref<9xf32>
        %62 = arith.addf %34, %61 : f32
        %63 = arith.mulf %62, %33 : f32
        %64 = memref.load %4[%c1] : memref<9xf32>
        %65 = arith.addf %63, %64 : f32
        %66 = arith.mulf %65, %33 : f32
        %67 = memref.load %4[%c2] : memref<9xf32>
        %68 = arith.addf %66, %67 : f32
        %69 = arith.mulf %68, %33 : f32
        %70 = memref.load %4[%c3] : memref<9xf32>
        %71 = arith.addf %69, %70 : f32
        %72 = arith.mulf %71, %33 : f32
        %73 = memref.load %4[%c4] : memref<9xf32>
        %74 = arith.addf %72, %73 : f32
        %75 = arith.mulf %74, %33 : f32
        %76 = memref.load %4[%c5] : memref<9xf32>
        %77 = arith.addf %75, %76 : f32
        %78 = arith.mulf %77, %33 : f32
        %79 = memref.load %4[%c6] : memref<9xf32>
        %80 = arith.addf %78, %79 : f32
        %81 = arith.mulf %80, %33 : f32
        %82 = memref.load %4[%c7] : memref<9xf32>
        %83 = arith.addf %81, %82 : f32
        %84 = arith.mulf %83, %33 : f32
        %85 = memref.load %4[%c8] : memref<9xf32>
        %86 = arith.addf %84, %85 : f32
        %87 = memref.load %3[%c0] : memref<11xf32>
        %88 = arith.addf %34, %87 : f32
        %89 = arith.mulf %88, %33 : f32
        %90 = memref.load %3[%c1] : memref<11xf32>
        %91 = arith.addf %89, %90 : f32
        %92 = arith.mulf %91, %33 : f32
        %93 = memref.load %3[%c2] : memref<11xf32>
        %94 = arith.addf %92, %93 : f32
        %95 = arith.mulf %94, %33 : f32
        %96 = memref.load %3[%c3] : memref<11xf32>
        %97 = arith.addf %95, %96 : f32
        %98 = arith.mulf %97, %33 : f32
        %99 = memref.load %3[%c4] : memref<11xf32>
        %100 = arith.addf %98, %99 : f32
        %101 = arith.mulf %100, %33 : f32
        %102 = memref.load %3[%c5] : memref<11xf32>
        %103 = arith.addf %101, %102 : f32
        %104 = arith.mulf %103, %33 : f32
        %105 = memref.load %3[%c6] : memref<11xf32>
        %106 = arith.addf %104, %105 : f32
        %107 = arith.mulf %106, %33 : f32
        %108 = memref.load %3[%c7] : memref<11xf32>
        %109 = arith.addf %107, %108 : f32
        %110 = arith.mulf %109, %33 : f32
        %111 = memref.load %3[%c8] : memref<11xf32>
        %112 = arith.addf %110, %111 : f32
        %113 = arith.mulf %112, %33 : f32
        %114 = memref.load %3[%c9] : memref<11xf32>
        %115 = arith.addf %113, %114 : f32
        %116 = arith.mulf %115, %33 : f32
        %117 = memref.load %3[%c10] : memref<11xf32>
        %118 = arith.addf %116, %117 : f32
        %119 = memref.load %2[%c0] : memref<10xf32>
        %120 = arith.addf %34, %119 : f32
        %121 = arith.mulf %120, %33 : f32
        %122 = memref.load %2[%c1] : memref<10xf32>
        %123 = arith.addf %121, %122 : f32
        %124 = arith.mulf %123, %33 : f32
        %125 = memref.load %2[%c2] : memref<10xf32>
        %126 = arith.addf %124, %125 : f32
        %127 = arith.mulf %126, %33 : f32
        %128 = memref.load %2[%c3] : memref<10xf32>
        %129 = arith.addf %127, %128 : f32
        %130 = arith.mulf %129, %33 : f32
        %131 = memref.load %2[%c4] : memref<10xf32>
        %132 = arith.addf %130, %131 : f32
        %133 = arith.mulf %132, %33 : f32
        %134 = memref.load %2[%c5] : memref<10xf32>
        %135 = arith.addf %133, %134 : f32
        %136 = arith.mulf %135, %33 : f32
        %137 = memref.load %2[%c6] : memref<10xf32>
        %138 = arith.addf %136, %137 : f32
        %139 = arith.mulf %138, %33 : f32
        %140 = memref.load %2[%c7] : memref<10xf32>
        %141 = arith.addf %139, %140 : f32
        %142 = arith.mulf %141, %33 : f32
        %143 = memref.load %2[%c8] : memref<10xf32>
        %144 = arith.addf %142, %143 : f32
        %145 = arith.mulf %144, %33 : f32
        %146 = memref.load %2[%c9] : memref<10xf32>
        %147 = arith.addf %145, %146 : f32
        %148 = arith.addf %31, %cst_14 : f32
        %149 = math.sqrt %29 : f32
        %150 = arith.divf %cst_5, %149 : f32
        %151 = math.sin %148 : f32
        %152 = arith.mulf %33, %60 : f32
        %153 = arith.divf %152, %86 : f32
        %154 = arith.addf %153, %cst_2 : f32
        %155 = arith.mulf %151, %154 : f32
        %156 = math.cos %148 : f32
        %157 = arith.mulf %32, %118 : f32
        %158 = arith.divf %157, %147 : f32
        %159 = arith.mulf %156, %158 : f32
        %160 = arith.subf %155, %159 : f32
        %161 = arith.mulf %150, %160 : f32
        scf.yield %161 : f32
      } else {
        scf.yield %14#1 : f32
      }
      %20 = arith.cmpf oge, %8, %cst_6 : f32
      %21 = arith.andi %18, %20 : i1
      %22 = arith.select %20, %c5_i32, %c0_i32 : i32
      %23 = arith.select %18, %22, %c0_i32 : i32
      %24:3 = scf.if %21 -> (f32, i1, f32) {
        %27:3 = scf.if %18 -> (f32, i1, f32) {
          %28 = arith.mulf %8, %cst_7 : f32
          %29 = math.sqrt %8 : f32
          %30 = arith.mulf %28, %29 : f32
          %31 = arith.divf %30, %cst_4 : f32
          %32 = arith.divf %cst_2, %31 : f32
          %33 = arith.mulf %32, %cst : f32
          %34 = memref.load %7[%c0] : memref<8xf32>
          %35 = arith.addf %33, %34 : f32
          %36 = arith.mulf %35, %32 : f32
          %37 = memref.load %7[%c1] : memref<8xf32>
          %38 = arith.addf %36, %37 : f32
          %39 = arith.mulf %38, %32 : f32
          %40 = memref.load %7[%c2] : memref<8xf32>
          %41 = arith.addf %39, %40 : f32
          %42 = arith.mulf %41, %32 : f32
          %43 = memref.load %7[%c3] : memref<8xf32>
          %44 = arith.addf %42, %43 : f32
          %45 = arith.mulf %44, %32 : f32
          %46 = memref.load %7[%c4] : memref<8xf32>
          %47 = arith.addf %45, %46 : f32
          %48 = arith.mulf %47, %32 : f32
          %49 = memref.load %7[%c5] : memref<8xf32>
          %50 = arith.addf %48, %49 : f32
          %51 = arith.mulf %50, %32 : f32
          %52 = memref.load %7[%c6] : memref<8xf32>
          %53 = arith.addf %51, %52 : f32
          %54 = arith.mulf %53, %32 : f32
          %55 = memref.load %7[%c7] : memref<8xf32>
          %56 = arith.addf %54, %55 : f32
          %57 = memref.load %6[%c0] : memref<8xf32>
          %58 = arith.addf %33, %57 : f32
          %59 = arith.mulf %58, %32 : f32
          %60 = memref.load %6[%c1] : memref<8xf32>
          %61 = arith.addf %59, %60 : f32
          %62 = arith.mulf %61, %32 : f32
          %63 = memref.load %6[%c2] : memref<8xf32>
          %64 = arith.addf %62, %63 : f32
          %65 = arith.mulf %64, %32 : f32
          %66 = memref.load %6[%c3] : memref<8xf32>
          %67 = arith.addf %65, %66 : f32
          %68 = arith.mulf %67, %32 : f32
          %69 = memref.load %6[%c4] : memref<8xf32>
          %70 = arith.addf %68, %69 : f32
          %71 = arith.mulf %70, %32 : f32
          %72 = memref.load %6[%c5] : memref<8xf32>
          %73 = arith.addf %71, %72 : f32
          %74 = arith.mulf %73, %32 : f32
          %75 = memref.load %6[%c6] : memref<8xf32>
          %76 = arith.addf %74, %75 : f32
          %77 = arith.mulf %76, %32 : f32
          %78 = memref.load %6[%c7] : memref<8xf32>
          %79 = arith.addf %77, %78 : f32
          %80 = arith.divf %56, %79 : f32
          %81 = arith.mulf %80, %cst_5 : f32
          %82 = math.sqrt %29 : f32
          %83 = arith.mulf %82, %cst_7 : f32
          %84 = math.exp %31 : f32
          %85 = arith.mulf %83, %84 : f32
          %86 = arith.divf %81, %85 : f32
          %87 = arith.cmpf ogt, %8, %cst_8 : f32
          %88 = arith.xori %87, %true : i1
          %89 = arith.select %87, %86, %19 : f32
          scf.yield %86, %88, %89 : f32, i1, f32
        } else {
          scf.yield %cst, %false, %19 : f32, i1, f32
        }
        scf.yield %27#0, %27#1, %27#2 : f32, i1, f32
      } else {
        scf.yield %cst, %18, %19 : f32, i1, f32
      }
      %25:2 = scf.if %24#1 -> (i1, f32) {
        %27 = arith.mulf %8, %8 : f32
        %28 = arith.mulf %27, %8 : f32
        %29:5 = scf.while (%arg4 = %cst_2, %arg5 = %8, %arg6 = %cst_2, %arg7 = %cst_2, %arg8 = %8, %arg9 = %cst_2) : (f32, f32, f32, f32, f32, f32) -> (f32, f32, f32, f32, f32) {
          %34 = arith.cmpf ogt, %arg4, %cst_13 : f32
          scf.condition(%34) %arg8, %arg9, %arg5, %arg6, %arg7 : f32, f32, f32, f32, f32
        } do {
        ^bb0(%arg4: f32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
          %34 = arith.mulf %arg7, %28 : f32
          %35 = arith.addf %arg8, %cst_2 : f32
          %36 = arith.divf %34, %35 : f32
          %37 = arith.mulf %arg6, %28 : f32
          %38 = arith.addf %35, %cst_2 : f32
          %39 = arith.divf %37, %38 : f32
          %40 = arith.divf %36, %38 : f32
          %41 = arith.addf %arg5, %40 : f32
          %42 = arith.addf %38, %cst_2 : f32
          %43 = arith.divf %39, %42 : f32
          %44 = arith.addf %arg4, %43 : f32
          %45 = arith.divf %40, %41 : f32
          %46 = math.absf %45 : f32
          scf.yield %46, %43, %40, %42, %44, %41 : f32, f32, f32, f32, f32, f32
        }
        %30 = arith.andi %23, %c1_i32 : i32
        %31 = arith.cmpi eq, %30, %c0_i32 : i32
        %32 = arith.cmpi ne, %30, %c0_i32 : i32
        %33 = scf.if %31 -> (f32) {
          %34 = arith.mulf %29#1, %cst_9 : f32
          %35 = arith.mulf %29#0, %cst_10 : f32
          %36 = arith.subf %34, %35 : f32
          scf.yield %36 : f32
        } else {
          scf.yield %24#2 : f32
        }
        scf.yield %32, %33 : i1, f32
      } else {
        scf.yield %false, %24#2 : i1, f32
      }
      %26 = arith.select %25#0, %24#0, %25#1 : f32
      memref.store %26, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

