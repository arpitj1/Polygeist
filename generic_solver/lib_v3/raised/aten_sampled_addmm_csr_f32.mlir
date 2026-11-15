module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @aten_sampled_addmm_csr_f32(%arg0: i32, %arg1: i32, %arg2: memref<?xi32>, %arg3: i32, %arg4: memref<?xi32>, %arg5: i32, %arg6: memref<?xf32>, %arg7: i32, %arg8: i32, %arg9: memref<?x?xf32>, %arg10: i32, %arg11: i32, %arg12: memref<?x?xf32>, %arg13: f32, %arg14: f32, %arg15: i32, %arg16: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg8 : i32 to index
    affine.for %arg17 = 0 to %0 {
      %2 = affine.load %arg2[%arg17] : memref<?xi32>
      %3 = affine.load %arg2[%arg17 + 1] : memref<?xi32>
      %4 = arith.index_cast %3 : i32 to index
      %5 = arith.index_cast %2 : i32 to index
      scf.for %arg18 = %5 to %4 step %c1 {
        %6 = memref.load %arg4[%arg18] : memref<?xi32>
        %7 = memref.load %arg6[%arg18] : memref<?xf32>
        %8 = arith.index_cast %6 : i32 to index
        %9 = affine.for %arg19 = 0 to %1 iter_args(%arg20 = %cst) -> (f32) {
          %13 = affine.load %arg9[%arg17, %arg19] : memref<?x?xf32>
          %14 = memref.load %arg12[%arg19, %8] : memref<?x?xf32>
          %15 = arith.mulf %13, %14 : f32
          %16 = arith.addf %arg20, %15 : f32
          affine.yield %16 : f32
        }
        %10 = arith.mulf %arg13, %9 : f32
        %11 = arith.mulf %arg14, %7 : f32
        %12 = arith.addf %10, %11 : f32
        memref.store %12, %arg16[%arg18] : memref<?xf32>
      }
    }
    return
  }
}
