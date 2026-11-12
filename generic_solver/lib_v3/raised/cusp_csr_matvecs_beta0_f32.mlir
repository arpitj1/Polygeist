module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cusp_csr_matvecs_beta0_f32(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: i32, %arg4: memref<?xi32>, %arg5: i32, %arg6: memref<?xf32>, %arg7: i32, %arg8: i32, %arg9: memref<?x?xf32>, %arg10: i32, %arg11: i32, %arg12: memref<?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg8 : i32 to index
    affine.for %arg13 = 0 to %0 {
      affine.for %arg14 = 0 to %1 {
        affine.store %cst, %arg12[%arg13, %arg14] : memref<?x?xf32>
      }
    }
    affine.for %arg13 = 0 to %0 {
      %2 = affine.load %arg2[%arg13] : memref<?xi32>
      %3 = scf.while (%arg14 = %2) : (i32) -> i32 {
        %4 = affine.load %arg2[%arg13 + 1] : memref<?xi32>
        %5 = arith.cmpi slt, %arg14, %4 : i32
        scf.condition(%5) %arg14 : i32
      } do {
      ^bb0(%arg14: i32):
        %4 = arith.index_cast %arg14 : i32 to index
        %5 = memref.load %arg4[%4] : memref<?xi32>
        %6 = memref.load %arg6[%4] : memref<?xf32>
        %7 = arith.index_cast %5 : i32 to index
        affine.for %arg15 = 0 to %1 {
          %9 = memref.load %arg9[%7, %arg15] : memref<?x?xf32>
          %10 = arith.mulf %6, %9 : f32
          %11 = affine.load %arg12[%arg13, %arg15] : memref<?x?xf32>
          %12 = arith.addf %11, %10 : f32
          affine.store %12, %arg12[%arg13, %arg15] : memref<?x?xf32>
        }
        %8 = arith.addi %arg14, %c1_i32 : i32
        scf.yield %8 : i32
      }
    }
    return
  }
}
