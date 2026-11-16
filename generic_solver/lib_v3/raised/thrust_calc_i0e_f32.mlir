module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  memref.global "private" constant @"_ZN12_GLOBAL__N_18calc_i0eIfEET_S1_@static@_ZZN12_GLOBAL__N_18calc_i0eIfEET_S1_E12coefficients_0" : memref<25xf32> = dense<[-7.23318078E-18, -4.83050441E-18, 4.46562156E-17, 3.46122279E-17, -2.82762388E-16, -3.42548568E-16, 1.77256012E-15, 3.81168087E-15, -9.55484674E-15, -4.15056918E-14, 1.54008615E-14, 3.85277829E-13, 7.18012455E-13, -1.79417852E-12, -1.32158121E-11, -3.14991644E-11, 1.18891468E-11, 4.94060237E-10, 3.39623196E-9, 2.26666899E-8, 2.04891862E-7, 2.89137051E-6, 6.88975852E-5, 0.00336911646, 0.804490387]>
  memref.global "private" constant @"_ZN12_GLOBAL__N_18calc_i0eIfEET_S1_@static@_ZZN12_GLOBAL__N_18calc_i0eIfEET_S1_E12coefficients" : memref<30xf32> = dense<[-4.41534163E-18, 3.33079461E-17, -2.431280e-16, 1.71539133E-15, -1.16853328E-14, 7.67618526E-14, -4.85644673E-13, 2.95505265E-12, -1.72682632E-11, 9.67580876E-11, -5.18979582E-10, 2.65982369E-9, -1.30002498E-8, 6.04699508E-8, -2.67079372E-7, 1.11738757E-6, -4.41673819E-6, 1.64484482E-5, -5.75419508E-5, 1.88502891E-4, -5.76375576E-4, 0.00163947558, -4.324310e-03, 0.0105464607, -0.0237374157, 0.0493052825, -9.490110e-02, 0.171620905, -0.304682672, 0.676795303]>
  func.func @thrust_calc_i0e_f32(%arg0: i32, %arg1: memref<?xf32>, %arg2: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant -4.41534163E-18 : f32
    %cst_0 = arith.constant -7.23318078E-18 : f32
    %cst_1 = arith.constant 8.000000e+00 : f32
    %cst_2 = arith.constant 2.000000e+00 : f32
    %c30_i32 = arith.constant 30 : i32
    %cst_3 = arith.constant 3.200000e+01 : f32
    %c25_i32 = arith.constant 25 : i32
    %cst_4 = arith.constant 5.000000e-01 : f32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = memref.get_global @"_ZN12_GLOBAL__N_18calc_i0eIfEET_S1_@static@_ZZN12_GLOBAL__N_18calc_i0eIfEET_S1_E12coefficients_0" : memref<25xf32>
    %2 = memref.get_global @"_ZN12_GLOBAL__N_18calc_i0eIfEET_S1_@static@_ZZN12_GLOBAL__N_18calc_i0eIfEET_S1_E12coefficients" : memref<30xf32>
    affine.for %arg3 = 0 to %0 {
      %3 = memref.load %arg1[%arg3] : memref<?xf32>
      %4 = math.absf %3 : f32
      %5 = arith.cmpf ole, %4, %cst_1 : f32
      %6 = scf.if %5 -> (f32) {
        %7 = arith.divf %4, %cst_2 : f32
        %8 = arith.subf %7, %cst_2 : f32
        %9:5 = scf.while (%arg4 = %c1_i32, %arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst) : (i32, f32, f32, f32) -> (i32, f32, f32, f32, f32) {
          %12 = arith.cmpi slt, %arg4, %c30_i32 : i32
          %13:2 = scf.if %12 -> (i32, f32) {
            %14 = arith.mulf %8, %arg7 : f32
            %15 = arith.subf %14, %arg6 : f32
            %16 = arith.index_cast %arg4 : i32 to index
            %17 = memref.load %2[%16] : memref<30xf32>
            %18 = arith.addf %15, %17 : f32
            %19 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %19, %18 : i32, f32
          } else {
            scf.yield %arg4, %arg7 : i32, f32
          }
          scf.condition(%12) %13#0, %arg5, %arg6, %arg7, %13#1 : i32, f32, f32, f32, f32
        } do {
        ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
          scf.yield %arg4, %arg6, %arg7, %arg8 : i32, f32, f32, f32
        }
        %10 = arith.subf %9#4, %9#1 : f32
        %11 = arith.mulf %10, %cst_4 : f32
        scf.yield %11 : f32
      } else {
        %7 = arith.divf %cst_3, %4 : f32
        %8 = arith.subf %7, %cst_2 : f32
        %9:5 = scf.while (%arg4 = %c1_i32, %arg5 = %cst_5, %arg6 = %cst_5, %arg7 = %cst_0) : (i32, f32, f32, f32) -> (i32, f32, f32, f32, f32) {
          %14 = arith.cmpi slt, %arg4, %c25_i32 : i32
          %15:2 = scf.if %14 -> (i32, f32) {
            %16 = arith.mulf %8, %arg7 : f32
            %17 = arith.subf %16, %arg6 : f32
            %18 = arith.index_cast %arg4 : i32 to index
            %19 = memref.load %1[%18] : memref<25xf32>
            %20 = arith.addf %17, %19 : f32
            %21 = arith.addi %arg4, %c1_i32 : i32
            scf.yield %21, %20 : i32, f32
          } else {
            scf.yield %arg4, %arg7 : i32, f32
          }
          scf.condition(%14) %15#0, %arg5, %arg6, %arg7, %15#1 : i32, f32, f32, f32, f32
        } do {
        ^bb0(%arg4: i32, %arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
          scf.yield %arg4, %arg6, %arg7, %arg8 : i32, f32, f32, f32
        }
        %10 = arith.subf %9#4, %9#1 : f32
        %11 = arith.mulf %10, %cst_4 : f32
        %12 = math.sqrt %4 : f32
        %13 = arith.divf %11, %12 : f32
        scf.yield %13 : f32
      }
      memref.store %6, %arg2[%arg3] : memref<?xf32>
    }
    return
  }
}

