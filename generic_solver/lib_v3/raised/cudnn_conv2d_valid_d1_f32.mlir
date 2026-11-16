module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @cudnn_conv2d_valid_d1_f32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: memref<?x?x?x?xf32>, %arg10: memref<?x?x?x?xf32>, %arg11: memref<?x?x?x?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg4 : i32 to index
    %2 = arith.index_cast %arg7 : i32 to index
    %3 = arith.index_cast %arg8 : i32 to index
    %4 = arith.index_cast %arg1 : i32 to index
    %5 = arith.index_cast %arg5 : i32 to index
    %6 = arith.index_cast %arg6 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %1 {
        affine.for %arg14 = 0 to %2 {
          affine.for %arg15 = 0 to %3 {
            %7 = affine.for %arg16 = 0 to %4 iter_args(%arg17 = %cst) -> (f32) {
              %8 = affine.for %arg18 = 0 to %5 iter_args(%arg19 = %arg17) -> (f32) {
                %9 = affine.for %arg20 = 0 to %6 iter_args(%arg21 = %arg19) -> (f32) {
                  %10 = affine.load %arg10[%arg13, %arg16, %arg18, %arg20] : memref<?x?x?x?xf32>
                  %11 = affine.load %arg9[%arg12, %arg16, %arg14 + %arg18, %arg15 + %arg20] : memref<?x?x?x?xf32>
                  %12 = arith.mulf %10, %11 : f32
                  %13 = arith.addf %12, %arg21 : f32
                  affine.yield %13 : f32
                }
                affine.yield %9 : f32
              }
              affine.yield %8 : f32
            }
            affine.store %7, %arg11[%arg12, %arg13, %arg14, %arg15] : memref<?x?x?x?xf32>
          }
        }
      }
    }
    return
  }
}
